import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj


class ShadowModel(nj.Module):

  units: int = 512
  layers: int = 2
  classes: int = 4
  stoch: int = 32
  unimix: float = 0.01

  def __init__(self, act_space, **kw):
    super().__init__(**kw)
    self.act_space = act_space

  def __call__(self, z_t, a_t):
    # z_t: [..., stoch, classes]
    # Flatten the categorical latent dimensions.
    z_flat = z_t.reshape((*z_t.shape[:-2], -1))

    # Convert the action dictionary into one flat vector.
    a_flat = nn.DictConcat(self.act_space, 1)(a_t)

    # Combine latent state and action.
    x = jnp.concatenate([z_flat, a_flat], -1)

    # MLP.
    for i in range(self.layers):
      x = self.sub(f'l{i}', nn.Linear, self.units)(x)
      x = nn.act('silu')(x)

    # Predict categorical logits for every stochastic variable.
    logits = self.sub(
        'out', nn.Linear, self.stoch * self.classes)(x)

    return logits.reshape(
        (*x.shape[:-1], self.stoch, self.classes))

  def dist(self, logits):
    return embodied.jax.outs.OneHot(logits, self.unimix)
