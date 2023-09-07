import jax
from flax import linen as nn


class GATLayer(nn.Module):
    in_features: int
    out_features: int

    @nn.compact
    def __call__(self, h, adj):
        wh = nn.Dense(features=self.out_features)(h)
        wh1 = nn.Dense(features=1)(wh)
        wh2 = nn.Dense(features=1)(wh)
        e = nn.leaky_relu(wh1 + wh2.T)

        zero_vec = -10e10 * jax.numpy.ones_like(e)
        attention = jax.numpy.where(adj > 0, e, zero_vec)
        attention = nn.softmax(attention)

        h_new = jax.numpy.matmul(attention, wh)

        return nn.elu(h_new)
