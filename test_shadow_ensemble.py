import jax
import jax.numpy as jnp
import ninjax as nj
import elements
import embodied.jax
import embodied.jax.nets as nn

from dreamerv3.rssm import RSSM
from dreamerv3.shadow import ShadowModel


# ============================================================
# Configuration
# ============================================================

BATCH = 2
TIME = 6

STOCH = 32
CLASSES = 4

RSSM_DETER = 512
RSSM_HIDDEN = 64

NUM_SHADOWS = 3
UNIMIX = 0.01

EPS = 1e-8

print("JAX devices:", jax.devices())


# ============================================================
# Spaces
# ============================================================

act_space = {
    'action': elements.Space(
        jnp.float32,
        (1,),
        low=-1.0,
        high=1.0,
    )
}


# ============================================================
# Dummy encoded observations / actions
# ============================================================

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

reset = jnp.zeros(
    (BATCH, TIME),
    dtype=bool,
)

reset = reset.at[:, 0].set(True)


# ============================================================
# Models
# ============================================================

rssm = RSSM(
    act_space,
    deter=RSSM_DETER,
    hidden=RSSM_HIDDEN,
    stoch=STOCH,
    classes=CLASSES,
    unimix=UNIMIX,
    name='rssm',
)


shadows = [
    ShadowModel(
        act_space,
        units=64,
        layers=2,
        classes=CLASSES,
        stoch=STOCH,
        unimix=UNIMIX,
        name=f'shadow{i}',
    )
    for i in range(NUM_SHADOWS)
]


carry = rssm.initial(BATCH)


# ============================================================
# Forward
#
# Modules are captured through closure.
# Do NOT pass nj.Module objects as traced positional arguments.
# ============================================================

def make_forward(rssm, shadows):

    def forward(carry, tokens, actions, reset):

        carry, entries, feat = rssm.observe(
            carry,
            tokens,
            actions,
            reset,
            training=True,
        )

        z = feat['stoch']

        z_t = z[:, :-1]

        a_t = {
            k: v[:, :-1]
            for k, v in actions.items()
        }

        logits = [
            shadow(z_t, a_t)
            for shadow in shadows
        ]

        return carry, entries, feat, logits

    return forward


forward = make_forward(rssm, shadows)


# ============================================================
# Initialize ALL models
# ============================================================

params = nj.init(forward)(
    {},
    carry,
    tokens,
    actions,
    reset,
    seed=42,
)

print("Initialized parameter entries:", len(params))
print("All param keys:", list(params.keys()))


# ============================================================
# Loss
#
# IMPORTANT:
# Every Shadow receives the SAME sampled target_z (shared
# sampling, as agreed: cheaper, and keeps disagreement free of
# per-model sampling noise so it reflects genuine ensemble
# diversity from init/bootstrap only).
#
# RSSM -> sampled z
#          |
#          +---- Shadow 0
#          +---- Shadow 1
#          +---- Shadow 2
#
# ============================================================

def make_loss_fn(rssm, shadows):

    def loss_fn(carry, tokens, actions, reset):

        carry, entries, feat = rssm.observe(
            carry,
            tokens,
            actions,
            reset,
            training=True,
        )

        z = feat['stoch']

        # ----------------------------------------------------
        # Shared sampled target
        # ----------------------------------------------------

        z_t = jax.lax.stop_gradient(
            z[:, :-1]
        )

        target_z = jax.lax.stop_gradient(
            z[:, 1:]
        )

        a_t = {
            k: jax.lax.stop_gradient(v[:, :-1])
            for k, v in actions.items()
        }

        # ----------------------------------------------------
        # Each Shadow predicts the SAME target
        # ----------------------------------------------------

        logits = [
            shadow(z_t, a_t)
            for shadow in shadows
        ]

        dists = [
            shadow.dist(logit)
            for shadow, logit in zip(shadows, logits)
        ]

        logps = [
            dist.logp(target_z)
            for dist in dists
        ]

        # ----------------------------------------------------
        # Individual losses (cross-entropy with sampled target,
        # as agreed -- unbiased MC estimator of the full KL)
        # ----------------------------------------------------

        losses = [
            -logp.mean()
            for logp in logps
        ]

        total_loss = jnp.mean(
            jnp.stack(losses)
        )

        # ----------------------------------------------------
        # Per-class probabilities for the disagreement metric.
        #
        # dist is a OneHot output; dist.dist is the underlying
        # Categorical, whose .logits are already the unimix-mixed,
        # f32-cast logits computed by Categorical.__init__ (see
        # embodied/jax/outs.py). Reading dist.dist.logits directly
        # reuses that computation instead of reimplementing unimix
        # mixing here, so there is no risk of the two drifting
        # apart.
        #
        # Shape:
        #   [Shadow, B, T-1, STOCH, CLASSES]
        # ----------------------------------------------------

        probs = jnp.stack([
            jax.nn.softmax(dist.dist.logits, axis=-1)
            for dist in dists
        ])

        # ----------------------------------------------------
        # Self-check: does log-softmax over dist.dist.logits match
        # what OneHot.logp() actually returns? Compare log-prob
        # derived from `probs` above against the official
        # dist.logp() output, for shadow 0 only (cheap,
        # representative check).
        #
        # If this diverges, it means dist.dist.logits no longer
        # reflects OneHot's internal mixing (e.g. after a change to
        # dreamerv3/shadow.py or embodied/jax/outs.py), and the
        # disagreement metric below would be computed against the
        # wrong distribution.
        # ----------------------------------------------------

        manual_logp_0 = (
                target_z * jax.nn.log_softmax(dists[0].dist.logits, axis=-1)
        ).sum(axis=-1)

        logp_consistency_error = jnp.abs(manual_logp_0 - logps[0]).max()

        # ----------------------------------------------------
        # Ensemble mean prediction (mixture distribution)
        # ----------------------------------------------------

        mean_probs = probs.mean(axis=0)

        # ----------------------------------------------------
        # Disagreement -- AGREED SPEC:
        #
        #   U_t = (1/N) * sum_k KL[ q_shadow^(k) || q_bar ]
        #
        # KL of each shadow's distribution against the ensemble
        # mean distribution -- appropriate for categorical
        # outputs, unlike squared-difference variance which does
        # not respect the geometry of the probability simplex.
        #
        # Shape before reduction: [Shadow, B, T-1, STOCH]
        # ----------------------------------------------------

        kl_per_shadow = (
            probs * jnp.log((probs + EPS) / (mean_probs[None, ...] + EPS))
        ).sum(axis=-1)  # sum over CLASSES -> (Shadow, B, T-1, STOCH)

        disagreement_per_step = kl_per_shadow.mean(axis=0)  # mean over N -> (B, T-1, STOCH)
        disagreement = disagreement_per_step.mean()          # scalar for logging

        return total_loss, {
            'z': z,
            'logits': logits,
            'losses': jnp.stack(losses),
            'probs': probs,
            'mean_probs': mean_probs,
            'disagreement': disagreement,
            'disagreement_per_step': disagreement_per_step,
            'logp_consistency_error': logp_consistency_error,
        }

    return loss_fn


loss_fn = make_loss_fn(
    rssm,
    shadows,
)


# ============================================================
# Pure loss
# ============================================================

def pure_loss(
    params,
    carry,
    tokens,
    actions,
    reset,
):

    _, (loss, metrics) = nj.pure(loss_fn)(
        params,
        carry,
        tokens,
        actions,
        reset,
        seed=123,
    )

    return loss, metrics


# ============================================================
# Gradient
# ============================================================

(loss, metrics), grads = jax.value_and_grad(
    pure_loss,
    has_aux=True,
)(
    params,
    carry,
    tokens,
    actions,
    reset,
)


# ============================================================
# Basic outputs
# ============================================================

z = metrics['z']

print()
print("RSSM sampled z shape:", z.shape)
print("RSSM sampled z dtype:", z.dtype)

print(
    "Shared z_t shape:",
    z[:, :-1].shape,
)

print(
    "Shared target z_(t+1) shape:",
    z[:, 1:].shape,
)

print()


# ============================================================
# Consistency check: log-softmax(dist.dist.logits) vs OneHot.logp()
# ============================================================

print(
    "dist.dist.logits-vs-official logp max abs diff (shadow0):",
    float(metrics['logp_consistency_error']),
)


# ============================================================
# Individual Shadow losses
# ============================================================

print()
print("Shadow losses:")

for i, value in enumerate(metrics['losses']):
    print(
        f"  Shadow {i}: {float(value)}"
    )


print()
print(
    "Ensemble mean loss:",
    float(loss),
)

print(
    "Ensemble disagreement (KL-based):",
    float(metrics['disagreement']),
)

print(
    "Disagreement per-step shape:",
    metrics['disagreement_per_step'].shape,
)


# ============================================================
# Shapes
# ============================================================

for i, logits in enumerate(metrics['logits']):

    print(
        f"Shadow {i} logits shape:",
        logits.shape,
    )


# ============================================================
# Gradient statistics
# ============================================================

print()
print("Gradient statistics:")

all_leaves = jax.tree.leaves(grads)

finite_grads = all(
    bool(jnp.all(jnp.isfinite(x)))
    for x in all_leaves
)

total_grad_norm = jnp.sqrt(
    sum(
        jnp.sum(
            x.astype(jnp.float32) ** 2
        )
        for x in all_leaves
    )
)

print(
    "Total gradient leaves:",
    len(all_leaves),
)

print(
    "Gradients finite:",
    finite_grads,
)

print(
    "Total gradient L2 norm:",
    float(total_grad_norm),
)


# ============================================================
# Per-model gradient norms
# ============================================================

rssm_keys = [
    k for k in grads
    if 'rssm' in k
]

print()
print(
    "RSSM parameter keys:",
    len(rssm_keys),
)

rssm_grad_norm = jnp.sqrt(
    sum(
        jnp.sum(
            grads[k].astype(jnp.float32) ** 2
        )
        for k in rssm_keys
    )
) if rssm_keys else jnp.array(0.0)

print(
    "RSSM gradient norm (should be 0):",
    float(rssm_grad_norm),
)


shadow_grad_norms = []

for i in range(NUM_SHADOWS):

    keys = [
        k for k in grads
        if f'shadow{i}' in k
    ]

    norm = jnp.sqrt(
        sum(
            jnp.sum(
                grads[k].astype(jnp.float32) ** 2
            )
            for k in keys
        )
    ) if keys else jnp.array(0.0)

    shadow_grad_norms.append(norm)

    print(
        f"Shadow {i} parameter keys:",
        len(keys),
    )

    print(
        f"Shadow {i} gradient norm:",
        float(norm),
    )


# ============================================================
# Assertions
# ============================================================

assert z.shape == (
    BATCH,
    TIME,
    STOCH,
    CLASSES,
), z.shape


for logits in metrics['logits']:

    assert logits.shape == (
        BATCH,
        TIME - 1,
        STOCH,
        CLASSES,
    ), logits.shape


assert metrics['disagreement_per_step'].shape == (
    BATCH,
    TIME - 1,
    STOCH,
), metrics['disagreement_per_step'].shape


# --------------------------------------------------
# This is the critical check: if dist.dist.logits stops matching
# OneHot's real internal mixing, this will be large and the
# disagreement metric above cannot be trusted.
# --------------------------------------------------

assert float(metrics['logp_consistency_error']) < 1e-3, (
    "log-softmax(dist.dist.logits) does not match OneHot.logp()'s "
    "internal probability computation (max abs diff = "
    f"{float(metrics['logp_consistency_error'])}). "
    "Inspect dreamerv3/shadow.py's dist()/OneHot usage and "
    "embodied/jax/outs.py's Categorical/OneHot implementation "
    "before trusting the disagreement metric."
)


assert jnp.isfinite(loss)

assert finite_grads

assert float(total_grad_norm) > 0.0

assert rssm_keys, (
    "No RSSM parameter keys matched -- check naming convention "
    "against 'All param keys' printed above."
)

assert all(
    any(f'shadow{i}' in k for k in grads)
    for i in range(NUM_SHADOWS)
), (
    "No Shadow parameter keys matched for one or more shadows -- "
    "check naming convention against 'All param keys' printed above."
)

assert float(rssm_grad_norm) == 0.0, (
    "Gradient leaked into RSSM!"
)


for i, norm in enumerate(shadow_grad_norms):

    assert float(norm) > 0.0, (
        f"Shadow {i} received no gradient!"
    )


assert jnp.isfinite(
    metrics['disagreement']
)

assert float(metrics['disagreement']) >= 0.0, (
    "KL-based disagreement must be non-negative"
)


print()
print("Shadow Ensemble integration PASS")
