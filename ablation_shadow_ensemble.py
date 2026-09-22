"""Shadow-ensemble ablation (next-steps 5 and 4): smoke-scale comparison.

Trains the real dreamerv3.agent.Agent for a number of optimizer steps on a
fixed synthetic batch, under three conditions that differ only in
config.shadow.enabled / config.shadow.confidence_mode -- everything else
(seeds, data, model size, optimizer) is identical:

  - no_shadow:    shadow.enabled=False (baseline, ensemble compiled out)
  - per_step:     shadow.enabled=True, confidence_mode='per_step' (default
                  -- C_t multiplied into `con` every imagined step, so it
                  compounds through imag_loss()'s cumprod over the horizon)
  - trajectory:   shadow.enabled=True, confidence_mode='trajectory' (next-
                  step 4 -- con is left untouched; the trajectory's mean
                  C_t instead scales the whole policy/value loss once)

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

CONDITIONS = {
    'no_shadow': {'enabled': False},
    'per_step': {'enabled': True, 'confidence_mode': 'per_step'},
    'trajectory': {'enabled': True, 'confidence_mode': 'trajectory'},
}


def run_condition(name, shadow_overrides, seed):
  config = base_config.update({'agent': {'shadow': shadow_overrides}})
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
      print(f"  [{name}] step {step}: opt/loss={row.get('opt/loss', float('nan')):.4f}")
  return history


results = {}
for name, overrides in CONDITIONS.items():
  print()
  print(f"Running {STEPS} optimizer steps -- condition '{name}' ({overrides})...")
  results[name] = run_condition(name, overrides, seed=42)


# ============================================================
# Summary
# ============================================================

shared_keys = ['loss/dyn', 'loss/rep', 'loss/rew', 'loss/con',
               'loss/policy', 'loss/value', 'opt/loss']

print()
print("=" * 88)
header = f"{'metric':<16}"
for name in CONDITIONS:
  header += f"{name + '[0]':>18}{name + '[-1]':>18}"
print(header)
print("=" * 88)
for key in shared_keys:
  row = f"{key:<16}"
  missing = False
  for name in CONDITIONS:
    v0, v1 = results[name][0].get(key), results[name][-1].get(key)
    if v0 is None:
      missing = True
      break
    row += f"{v0:>18.4f}{v1:>18.4f}"
  if not missing:
    print(row)

for name in ('per_step', 'trajectory'):
  print()
  print(f"Shadow-only metrics ('{name}' run):")
  for key in ['loss/shadow', 'shadow_uncertainty', 'shadow_uncertainty_tilde',
              'shadow_uncertainty_tilde_p50', 'shadow_uncertainty_tilde_p75',
              'shadow_uncertainty_tilde_p90', 'shadow_confidence',
              'shadow_confidence_min', 'shadow_confidence_traj']:
    vals = [row[key] for row in results[name] if key in row]
    if vals:
      print(f"  {key:<28} first={vals[0]:.4f} last={vals[-1]:.4f}")

assert 'loss/shadow' not in results['no_shadow'][0], (
    "shadow.enabled=False run still logged loss/shadow -- ablation toggle "
    "is not actually disabling the shadow loss.")
assert 'shadow_confidence' not in results['no_shadow'][0], (
    "shadow.enabled=False run still logged shadow_confidence -- ablation "
    "toggle is not actually disabling the confidence weighting.")
assert 'shadow_confidence_traj' not in results['per_step'][0], (
    "'per_step' run logged shadow_confidence_traj -- that metric is "
    "'trajectory'-mode only.")
assert 'shadow_confidence_traj' in results['trajectory'][0], (
    "'trajectory' run did not log shadow_confidence_traj -- confidence_mode "
    "did not take effect.")
for name in CONDITIONS:
  for key in shared_keys:
    assert all(np.isfinite(row[key]) for row in results[name] if key in row)

print()
print("Ablation smoke run PASS")


# ============================================================
# Save raw history for plotting / re-analysis.
# ============================================================

import json

out_dir = pathlib.Path(__file__).parent / 'ablation_results'
out_dir.mkdir(exist_ok=True)
for name, history in results.items():
  (out_dir / f'{name}.json').write_text(json.dumps(history))
print(f"\nSaved raw per-step histories to {out_dir}/")
