"""Integration smoke test for shadow-ensemble step 3a.

Verifies that the shadow ensemble's training loss (added to
dreamerv3/agent.py's Agent.loss(), right after losses['con']) works
end-to-end when driven through the real dreamerv3.agent.Agent class and
the real dreamerv3/configs.yaml config (debug + size1m presets), rather
than through hand-built RSSM/ShadowModel instances like
test_shadow_rssm.py and test_shadow_ensemble.py do.

This bypasses embodied.jax.Agent's device/mesh/sharding wrapper (which
assumes a full multi-device training setup) and drives the inner
dreamerv3.agent.Agent model directly through nj.init/nj.pure, the same
pattern used by the other standalone tests in this repo.
"""

import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from embodied.envs.dummy import Dummy

BATCH = 3
LENGTH = 6

print("JAX devices:", jax.devices())


# ============================================================
# Config: defaults + debug + size1m, same presets main.py would use
# for a fast CPU run, with replay_context disabled to keep the
# synthetic batch construction below simple.
# ============================================================

root = pathlib.Path(__file__).parent / 'dreamerv3'
raw = yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())
config = elements.Config(raw['defaults'])
config = config.update(raw['debug'])
config = config.update(raw['size1m'])
config = config.update({'replay_context': 0})


# ============================================================
# Spaces (from the repo's own dummy test env)
# ============================================================

env = Dummy('disc', size=(64, 64), length=20)
obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
env.close()


# ============================================================
# Build the inner model directly (same object dreamerv3.agent.Agent's
# metaclass-style __new__ would build as `model`), skipping
# embodied.jax.Agent's __new__/__init__, which sets up multi-device
# meshes/sharding that this CPU smoke test doesn't need.
# ============================================================

model = object.__new__(Agent)
Agent.__init__(model, obs_space, act_space, config.agent)

print("Modules:", [m.name for m in model.modules])
assert any(m.name.startswith('shadow') for m in model.modules), (
    "Shadow ensemble modules were not registered in self.modules -- "
    "their parameters would never be trained by the optimizer.")


# ============================================================
# Synthetic batch
# ============================================================

def sample_space(space, rng):
  # Space.sample() clips unbounded float dimensions to +-float32 max,
  # which is unrealistic for e.g. an unbounded continuous action space
  # and blows up downstream MLPs. Use a bounded [-1, 1] range instead,
  # matching what a real (tanh-squashed) policy output looks like.
  if np.issubdtype(space.dtype, np.floating) and not np.isfinite(
      space.low).all():
    return rng.uniform(-1.0, 1.0, space.shape).astype(space.dtype)
  return space.sample()


def make_batch(space_dict, batch, length, seed=0):
  rng = np.random.RandomState(seed)
  out = {}
  for key, space in space_dict.items():
    arr = np.stack([
        np.stack([sample_space(space, rng) for _ in range(length)])
        for _ in range(batch)])
    out[key] = jnp.asarray(arr)
  return out


obs = make_batch(obs_space, BATCH, LENGTH)
obs['is_first'] = jnp.zeros((BATCH, LENGTH), bool).at[:, 0].set(True)
obs['is_last'] = jnp.zeros((BATCH, LENGTH), bool)
obs['is_terminal'] = jnp.zeros((BATCH, LENGTH), bool)

prevact = make_batch(act_space, BATCH, LENGTH)

carry = model.init_train(BATCH)[:3]  # (enc_carry, dyn_carry, dec_carry)


# ============================================================
# Forward / loss, through nj.init + nj.pure (module captured via
# closure, per the pattern established in test_shadow_rssm.py /
# test_shadow_ensemble.py).
# ============================================================

def make_loss_fn(model):
  def loss_fn(carry, obs, prevact):
    return model.loss(carry, obs, prevact, training=True)
  return loss_fn


loss_fn = make_loss_fn(model)

params = nj.init(loss_fn)({}, carry, obs, prevact, seed=42)
print("Initialized parameter entries:", len(params))


def pure_loss(params, carry, obs, prevact):
  _, (loss, aux) = nj.pure(loss_fn)(params, carry, obs, prevact, seed=123)
  return loss, aux


(loss, (new_carry, entries, outs, metrics)), grads = jax.value_and_grad(
    pure_loss, has_aux=True)(params, carry, obs, prevact)


# ============================================================
# Shadow loss checks
# ============================================================

losses = outs['losses']
print()
print("Loss keys:", sorted(losses.keys()))
assert 'shadow' in losses, (
    "losses['shadow'] is missing -- the shadow ensemble loss was not "
    "wired into Agent.loss().")

shadow_loss = losses['shadow']
print("losses['shadow'] shape:", shadow_loss.shape)
print("losses['shadow'] mean:", float(shadow_loss.mean()))
print("loss/shadow (scaled into total loss):", float(metrics['loss/shadow']))

assert shadow_loss.shape == (BATCH, LENGTH), shadow_loss.shape
assert jnp.isfinite(shadow_loss).all()
assert set(losses.keys()) == set(model.scales.keys()), (
    sorted(losses.keys()), sorted(model.scales.keys()))

print()
print("Total loss:", float(loss))
assert jnp.isfinite(loss)


# ============================================================
# Gradient checks (combined loss): every shadow model must receive a
# nonzero gradient from the scaled total loss.
# ============================================================

all_leaves = jax.tree.leaves(grads)
finite_grads = all(bool(jnp.all(jnp.isfinite(x))) for x in all_leaves)
print()
print("Combined-loss gradients finite:", finite_grads)
assert finite_grads

for i in range(config.agent.shadow.n_models):
  keys = [k for k in grads if k.startswith(f'shadow{i}/')]
  assert keys, f"No parameters found for shadow{i}"
  norm = jnp.sqrt(sum(
      jnp.sum(grads[k].astype(jnp.float32) ** 2) for k in keys))
  print(f"Shadow {i} gradient norm (combined loss):", float(norm))
  assert float(norm) > 0.0, f"Shadow {i} received no gradient!"


# ============================================================
# Stop-gradient boundary check, reproduced inside the real Agent.loss()
# computation graph (not just the standalone shadow-only scripts):
# differentiating ONLY losses['shadow'] must not touch dyn/enc/dec
# (the world model) at all, since every shadow input is stop_gradient'd
# in agent.py.
# ============================================================

def shadow_only_loss(params, carry, obs, prevact):
  _, (_, aux) = nj.pure(loss_fn)(params, carry, obs, prevact, seed=123)
  return aux[2]['losses']['shadow'].mean()


shadow_grads = jax.grad(shadow_only_loss)(params, carry, obs, prevact)

print()
print("Stop-gradient boundary check (shadow-only loss):")
for prefix in ('dyn/', 'enc/', 'dec/'):
  keys = [k for k in shadow_grads if k.startswith(prefix)]
  norm = jnp.sqrt(sum(
      jnp.sum(shadow_grads[k].astype(jnp.float32) ** 2) for k in keys))
  print(f"  {prefix} gradient norm (should be 0):", float(norm))
  assert float(norm) == 0.0, (
      f"Gradient from losses['shadow'] leaked into {prefix} parameters! "
      "stop_gradient boundary in agent.py is broken.")

for i in range(config.agent.shadow.n_models):
  keys = [k for k in shadow_grads if k.startswith(f'shadow{i}/')]
  norm = jnp.sqrt(sum(
      jnp.sum(shadow_grads[k].astype(jnp.float32) ** 2) for k in keys))
  print(f"  shadow{i}/ gradient norm (should be > 0):", float(norm))
  assert float(norm) > 0.0, f"Shadow {i} received no gradient from its own loss!"

print()
print("Shadow ensemble Agent.loss() integration PASS")
