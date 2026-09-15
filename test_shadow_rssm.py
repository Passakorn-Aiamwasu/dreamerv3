import jax
import jax.numpy as jnp
import ninjax as nj
import elements
import embodied.jax
import embodied.jax.nets as nn

from dreamerv3.rssm import RSSM
from dreamerv3.shadow import ShadowModel


BATCH = 2
TIME = 6

STOCH = 32
CLASSES = 4

RSSM_DETER = 512
RSSM_HIDDEN = 64

print("JAX devices:", jax.devices())


act_space = {
    'action': elements.Space(
        jnp.float32,
        (1,),
        low=-1.0,
        high=1.0,
    )
}

TOKEN_DIM = 32

tokens = jax.random.normal(
    jax.random.PRNGKey(0),
    (BATCH, TIME, TOKEN_DIM),
    dtype=jnp.bfloat16,
)

actions = {
    'action': jax.random.uniform(
        jax.random.PRNGKey(1),
        (BATCH, TIME, 1),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.bfloat16,
    )
}

reset = jnp.zeros((BATCH, TIME), dtype=bool)
reset = reset.at[:, 0].set(True)


rssm = RSSM(
    act_space,
    deter=RSSM_DETER,
    hidden=RSSM_HIDDEN,
    stoch=STOCH,
    classes=CLASSES,
    unimix=0.01,
    name='rssm',
)

carry = rssm.initial(BATCH)

shadow = ShadowModel(
    act_space,
    units=64,
    layers=2,
    classes=CLASSES,
    stoch=STOCH,
    unimix=0.01,
    name='shadow',
)


# ============================================================
# IMPORTANT:
# rssm / shadow must be captured via closure, NOT passed as
# positional arguments to nj.init/nj.pure. Those functions trace
# every positional argument as a JAX pytree of arrays, and a
# nj.Module instance is not an array (it has no .dtype) -- hence
# the earlier "Cannot interpret value ... as an abstract array"
# error. Closures keep the modules out of the traced arg list.
# ============================================================

def make_forward(rssm, shadow):
    def forward(carry, tokens, actions, reset):
        carry, entries, feat = rssm.observe(
            carry, tokens, actions, reset, training=True,
        )
        z_t = feat['stoch'][:, :-1]
        a_t = {k: v[:, :-1] for k, v in actions.items()}
        logits = shadow(z_t, a_t)
        return carry, entries, feat, logits
    return forward


forward = make_forward(rssm, shadow)

params = nj.init(forward)(
    {}, carry, tokens, actions, reset, seed=42,
)

print("Initialized parameter entries:", len(params))


def make_loss_fn(rssm, shadow):
    def loss_fn(carry, tokens, actions, reset):
        carry, entries, feat = rssm.observe(
            carry, tokens, actions, reset, training=True,
        )
        z = feat['stoch']

        # z[:, :-1] = z_t, z[:, 1:] = sampled z_{t+1}, actions[:-1] = a_t
        z_t = jax.lax.stop_gradient(z[:, :-1])
        target_z = jax.lax.stop_gradient(z[:, 1:])
        a_t = {
            k: jax.lax.stop_gradient(v[:, :-1])
            for k, v in actions.items()
        }

        logits = shadow(z_t, a_t)
        dist = shadow.dist(logits)
        logp = dist.logp(target_z)
        loss = -logp.mean()

        return loss, {'z': z, 'logits': logits, 'logp': logp}
    return loss_fn


loss_fn = make_loss_fn(rssm, shadow)


def pure_loss(params, carry, tokens, actions, reset):
    _, (loss, metrics) = nj.pure(loss_fn)(
        params, carry, tokens, actions, reset, seed=123,
    )
    return loss, metrics


(loss, metrics), grads = jax.value_and_grad(
    pure_loss, has_aux=True,
)(params, carry, tokens, actions, reset)


z = metrics['z']
logits = metrics['logits']

print("RSSM sampled z shape:", z.shape)
print("RSSM sampled z dtype:", z.dtype)
print("Shadow input z_t shape:", z[:, :-1].shape)
print("Shadow target z_(t+1) shape:", z[:, 1:].shape)
print("Shadow logits shape:", logits.shape)
print("Loss:", float(loss))
print("Loss dtype:", loss.dtype)

leaves = jax.tree.leaves(grads)

finite_grads = all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
grad_norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in leaves))

print("Gradient leaves:", len(leaves))
print("Gradients finite:", finite_grads)
print("Gradient L2 norm:", float(grad_norm))

rssm_keys = [k for k in grads if 'rssm' in k]
shadow_keys = [k for k in grads if 'shadow' in k]

print("RSSM param keys:", len(rssm_keys))
print("Shadow param keys:", len(shadow_keys))

rssm_grad_norm = jnp.sqrt(sum(
    jnp.sum(grads[k].astype(jnp.float32) ** 2) for k in rssm_keys
)) if rssm_keys else jnp.array(0.0)

shadow_grad_norm = jnp.sqrt(sum(
    jnp.sum(grads[k].astype(jnp.float32) ** 2) for k in shadow_keys
)) if shadow_keys else jnp.array(0.0)

print("RSSM gradient norm (should be 0):", float(rssm_grad_norm))
print("Shadow gradient norm (should be > 0):", float(shadow_grad_norm))

assert z.shape == (BATCH, TIME, STOCH, CLASSES), z.shape
assert logits.shape == (BATCH, TIME - 1, STOCH, CLASSES), logits.shape
assert jnp.isfinite(loss)
assert finite_grads
assert float(grad_norm) > 0.0
assert float(rssm_grad_norm) == 0.0, "stop_gradient boundary broken -- gradient leaked into RSSM!"
assert float(shadow_grad_norm) > 0.0, "shadow model received no gradient at all"

print()
print("RSSM -> ShadowModel integration PASS")
