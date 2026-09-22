"""Shadow-ensemble ablation (next-step 5): smoke-scale comparison.

Trains the real dreamerv3.agent.Agent for a number of optimizer steps on a
fixed synthetic batch, once with agent.shadow.enabled=True and once with it
False (config.shadow.enabled, added alongside this script), so the two runs
differ only in whether the shadow ensemble trains and discounts `con` --
everything else (seeds, data, model size, optimizer) is identical.

This is NOT a claim about task performance -- the "environment" here is
random noise, so there's nothing to actually solve. It is a mechanical
check: does the shadow-enabled run behave sanely (uncertainty/confidence
finite and moving, other losses still optimize) relative to the exact same
setup with the ensemble compiled out entirely.

Scale is chosen automatically from the detected JAX backend (see
`PLATFORM`/`SCALE` below): a real GPU gets a much larger, closer-to-production
config (size12m, full batch_size/batch_length from configs.yaml's defaults);
CPU-only falls back to the tiny debug+size1m config this script originally
shipped with, since a size12m model run step-by-step, un-jitted-by-default
Python loop on CPU is impractically slow. Nothing here hardcodes CPU --
jax.jit picks up whatever backend the installed jaxlib was built for and
the machine actually has (see requirements.txt's jax[cuda12] for the GPU
wheel), so this same script is what you'd run on a GPU machine, not a
separate version of it.
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

PLATFORM = jax.devices()[0].platform  # 'cpu', 'gpu', or 'tpu'
print("JAX devices:", jax.devices())

if PLATFORM == 'cpu':
  print("No GPU detected -- falling back to the tiny debug+size1m scale.")
  SCALE = 'debug'
  BATCH, LENGTH, STEPS = 4, 8, 60
else:
  print(f"{PLATFORM.upper()} detected -- using the larger size12m scale.")
  SCALE = 'size12m'
  BATCH, LENGTH, STEPS = 16, 64, 200


# ============================================================
# Config / spaces / synthetic batch (built once, shared across both runs
# and every step, so the only difference between conditions is the
# shadow.enabled flag).
# ============================================================

root = pathlib.Path(__file__).parent / 'dreamerv3'
raw = yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())
base_config = elements.Config(raw['defaults'])
if SCALE == 'debug':
  base_config = base_config.update(raw['debug'])
  base_config = base_config.update(raw['size1m'])
else:
  base_config = base_config.update(raw[SCALE])
base_config = base_config.update({'replay_context': 0})

env = Dummy('disc', size=(64, 64), length=20)
obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
env.close()


def sample_space(space, rng):
  if np.issubdtype(space.dtype, np.floating) and not np.isfinite(
      space.low).all():
    return rng.uniform(-1.0, 1.0, space.shape).astype(space.dtype)
  return space.sample()


def make_batch(space_dict, batch, length, seed):
  rng = np.random.RandomState(seed)
  out = {}
  for key, space in space_dict.items():
    arr = np.stack([
        np.stack([sample_space(space, rng) for _ in range(length)])
        for _ in range(batch)])
    out[key] = jnp.asarray(arr)
  return out


obs = make_batch(obs_space, BATCH, LENGTH, seed=0)
obs['is_first'] = jnp.zeros((BATCH, LENGTH), bool).at[:, 0].set(True)
obs['is_last'] = jnp.zeros((BATCH, LENGTH), bool)
obs['is_terminal'] = jnp.zeros((BATCH, LENGTH), bool)
prevact = make_batch(act_space, BATCH, LENGTH, seed=1)


# ============================================================
# Run one condition (shadow enabled or disabled) for STEPS optimizer steps
# on the same fixed batch, recording per-step loss/* and shadow_* metrics.
# ============================================================

def run_condition(shadow_enabled, seed):
  config = base_config.update({'agent': {'shadow': {'enabled': shadow_enabled}}})
  model = object.__new__(Agent)
  Agent.__init__(model, obs_space, act_space, config.agent)
  carry = model.init_train(BATCH)[:3]

  def train_step(carry, obs, prevact):
    return model.opt(model.loss, carry, obs, prevact, training=True, has_aux=True)

  params = nj.init(train_step)({}, carry, obs, prevact, seed=seed)

  # The production training loop (embodied/jax/transform.py) always runs
  # under jax.jit; our step function must too, or JAX falls back to
  # compiling every primitive op eagerly on each Python-level call, which
  # is both extremely slow and -- observed directly in this sandboxed
  # environment -- can exhaust memory after just one or two steps. `seed`
  # must be passed as a traced array (not a Python int) so it varies
  # without forcing recompilation.
  pure_step = nj.pure(train_step)
  jit_step = jax.jit(lambda params, carry, obs, prevact, seed: pure_step(
      params, carry, obs, prevact, seed=seed))

  history = []
  for step in range(STEPS):
    seed_arr = jnp.array([seed + 1 + step, seed + 1 + step], jnp.uint32)
    params, (metrics, (carry, entries, outs, mets)) = jit_step(
        params, carry, obs, prevact, seed_arr)
    metrics.update(mets)
    row = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
    history.append(row)
    if step == 0 or step == STEPS - 1:
      print(f"  [{'shadow' if shadow_enabled else 'no_shadow'}] "
            f"step {step}: opt/loss={row.get('opt/loss', float('nan')):.4f}")
  return history


print()
print(f"Running {STEPS} optimizer steps WITH shadow ensemble...")
with_shadow = run_condition(True, seed=42)

print()
print(f"Running {STEPS} optimizer steps WITHOUT shadow ensemble...")
without_shadow = run_condition(False, seed=42)


# ============================================================
# Summary
# ============================================================

shared_keys = ['loss/dyn', 'loss/rep', 'loss/rew', 'loss/con',
               'loss/policy', 'loss/value', 'opt/loss']

print()
print("=" * 70)
print(f"{'metric':<16} {'with[0]':>10} {'with[-1]':>10} "
      f"{'no[0]':>10} {'no[-1]':>10}")
print("=" * 70)
for key in shared_keys:
  w0, w1 = with_shadow[0].get(key), with_shadow[-1].get(key)
  n0, n1 = without_shadow[0].get(key), without_shadow[-1].get(key)
  if w0 is None or n0 is None:
    continue
  print(f"{key:<16} {w0:>10.4f} {w1:>10.4f} {n0:>10.4f} {n1:>10.4f}")

print()
print("Shadow-only metrics (with-shadow run):")
for key in ['loss/shadow', 'shadow_uncertainty', 'shadow_uncertainty_tilde',
            'shadow_uncertainty_tilde_p50', 'shadow_uncertainty_tilde_p75',
            'shadow_uncertainty_tilde_p90', 'shadow_confidence',
            'shadow_confidence_min']:
  vals = [row[key] for row in with_shadow if key in row]
  if vals:
    print(f"  {key:<28} first={vals[0]:.4f} last={vals[-1]:.4f}")

assert 'loss/shadow' not in without_shadow[0], (
    "shadow.enabled=False run still logged loss/shadow -- ablation toggle "
    "is not actually disabling the shadow loss.")
assert 'shadow_confidence' not in without_shadow[0], (
    "shadow.enabled=False run still logged shadow_confidence -- ablation "
    "toggle is not actually disabling the confidence weighting.")
for key in shared_keys:
  assert all(np.isfinite(row[key]) for row in with_shadow if key in row)
  assert all(np.isfinite(row[key]) for row in without_shadow if key in row)

print()
print("Ablation smoke run PASS")


# ============================================================
# Save raw history for plotting / re-analysis.
# ============================================================

import json

out_dir = pathlib.Path(__file__).parent / 'ablation_results'
out_dir.mkdir(exist_ok=True)
(out_dir / 'with_shadow.json').write_text(json.dumps(with_shadow))
(out_dir / 'without_shadow.json').write_text(json.dumps(without_shadow))
print(f"\nSaved raw per-step histories to {out_dir}/")
