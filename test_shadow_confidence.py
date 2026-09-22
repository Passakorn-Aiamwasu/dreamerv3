"""Integration smoke test for shadow-ensemble step 3b.

Verifies dreamerv3.agent.Agent._shadow_confidence(), which computes the
shadow ensemble's per-imagined-step disagreement U_t and turns it into a
confidence weight C_t (design decisions #4/#5/#6), queried on the real
imagined trajectory the same way Agent.loss()'s imagination block does
(imgfeat/imgact built via self.dyn.imagine(), vectorized over the whole
rollout instead of stepping through RSSM.imagine() itself -- the
architecture agreed for step 3b).

This does NOT yet check that C_t is mixed into con (that is step 3c);
Agent.loss() currently only logs shadow_uncertainty/shadow_confidence as
metrics.
"""

import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent, concat, sample, sg
from embodied.envs.dummy import Dummy

BATCH = 3
LENGTH = 6

print("JAX devices:", jax.devices())


# ============================================================
# Config / spaces / model (same setup as test_shadow_agent_loss.py)
# ============================================================

root = pathlib.Path(__file__).parent / 'dreamerv3'
raw = yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())
config = elements.Config(raw['defaults'])
config = config.update(raw['debug'])
config = config.update(raw['size1m'])
config = config.update({'replay_context': 0})

env = Dummy('disc', size=(64, 64), length=20)
obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
env.close()

model = object.__new__(Agent)
Agent.__init__(model, obs_space, act_space, config.agent)
print("Shadow config:", dict(config.agent.shadow))


def sample_space(space, rng):
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
carry = model.init_train(BATCH)[:3]


# ============================================================
# Reproduce Agent.loss()'s imagination block up through imgfeat/imgact,
# then call _shadow_confidence() directly on the result.
# ============================================================

def make_fn(model, training):
  # `training` must be a concrete Python bool baked into the traced
  # function via closure, not a runtime argument -- it gates Python-level
  # `if training:` branches throughout agent.py/rssm.py, same as every
  # other call site in this codebase (e.g. Agent.loss()'s own `training`
  # parameter is always passed as a literal True/False, never an array).
  def fn(carry, obs, prevact):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    enc_carry, enc_entries, tokens = model.enc(enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = model.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)

    K = min(model.config.imag_last or T, T)
    H = model.config.imag_length
    starts = model.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(model.pol(model.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = model.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)

    conf, u, u_tilde = model._shadow_confidence(imgfeat, imgact, training)
    return imgfeat, imgact, conf, u, u_tilde
  return fn


fn_train = make_fn(model, True)
fn_eval = make_fn(model, False)

params = nj.init(fn_train)({}, carry, obs, prevact, seed=42)
print("Initialized parameter entries:", len(params))
assert 'shadow_u_ema/value' in params, (
    "shadow_u_ema variable was not created -- _shadow_confidence() was "
    "never called, or nj.Variable naming changed.")
print("Initial shadow_u_ema:", float(params['shadow_u_ema/value']))


def run(params, training, seed):
  fn = fn_train if training else fn_eval
  new_params, (imgfeat, imgact, conf, u, u_tilde) = nj.pure(fn)(
      params, carry, obs, prevact, seed=seed)
  return new_params, imgfeat, imgact, conf, u, u_tilde


# ============================================================
# Shape / range / finiteness checks
# ============================================================

H = config.agent.imag_length
K = min(config.agent.imag_last or LENGTH, LENGTH)
expected_shape = (BATCH * K, H + 1)

params, imgfeat, imgact, conf, u, u_tilde = run(params, True, 123)

print()
print("imgfeat/imgact time steps:", H + 1)
print("shadow_confidence shape:", conf.shape)
print("shadow_uncertainty shape:", u.shape)
assert conf.shape == expected_shape, conf.shape
assert u.shape == expected_shape, u.shape
assert u_tilde.shape == expected_shape, u_tilde.shape

print("shadow_uncertainty (U_t):", float(u.mean()), "min", float(u.min()), "max", float(u.max()))
print(
    "shadow_uncertainty_tilde (U_t/EMA):", float(u_tilde.mean()),
    "p50", float(jnp.percentile(u_tilde, 50)),
    "p75", float(jnp.percentile(u_tilde, 75)),
    "p90", float(jnp.percentile(u_tilde, 90)))
print("shadow_confidence:", float(conf.mean()), "min", float(conf.min()), "max", float(conf.max()))

assert jnp.isfinite(u).all()
assert jnp.isfinite(u_tilde).all()
assert jnp.isfinite(conf).all()
assert float(u.min()) >= 0.0, "KL-based disagreement must be non-negative"
assert float(u_tilde.min()) >= 0.0, "normalized disagreement must be non-negative"
assert float(conf.min()) > 0.0, "confidence weight must be strictly positive"
assert float(conf.max()) <= 1.0 + 1e-6, "confidence weight must be <= 1"


# ============================================================
# EMA update check: shadow_u_ema must move after a training=True call,
# and stay frozen after a training=False call.
# ============================================================

ema0 = float(params['shadow_u_ema/value'])
params_after_train, *_ = run(params, True, 124)
ema1 = float(params_after_train['shadow_u_ema/value'])
print()
print("shadow_u_ema before:", ema0, "after training=True call:", ema1)
assert ema1 != ema0, "shadow_u_ema did not update on a training=True call"

params_after_eval, *_ = run(params_after_train, False, 125)
ema2 = float(params_after_eval['shadow_u_ema/value'])
print("shadow_u_ema after training=False call:", ema2)
assert ema2 == ema1, (
    "shadow_u_ema changed on a training=False call -- EMA should only "
    "update during training")


# ============================================================
# Stop-gradient check: nothing computed from shadow_confidence/
# shadow_uncertainty may carry a gradient into any trainable parameter
# (design decision #5 -- confidence weight must always be detached, so
# the policy can never learn to make the ensemble agree with itself).
# ============================================================

def conf_sum_loss(params, carry, obs, prevact):
  _, (_, _, conf, u, u_tilde) = nj.pure(fn_train)(
      params, carry, obs, prevact, seed=123)
  return conf.sum() + u.sum() + u_tilde.sum()


conf_grads = jax.grad(conf_sum_loss)(params, carry, obs, prevact)
all_leaves = jax.tree.leaves(conf_grads)
total_norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in all_leaves))
print()
print("Gradient L2 norm of sum(confidence) + sum(uncertainty) w.r.t. all "
      "params (should be 0):", float(total_norm))
assert float(total_norm) == 0.0, (
    "Gradient leaked from shadow_confidence/shadow_uncertainty into "
    "trainable parameters -- confidence weight is not fully detached!")

print()
print("Shadow confidence (step 3b) integration PASS")
