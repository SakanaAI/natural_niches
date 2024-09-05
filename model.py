import jax
import jax.numpy as jnp


hidden_dim = 256  # @param
input_dim = 64
output_dim = 10
num_params = (1 + input_dim) * hidden_dim + (1 + hidden_dim) * output_dim


@jax.jit
def mlp(params, data):
    """A 2-layer feedforward network."""
    ss = 0
    ee = hidden_dim * input_dim
    w = params[ss:ee].reshape([input_dim, hidden_dim])
    ss = ee
    ee = ss + hidden_dim
    b = params[ss:ee]
    x = jax.nn.relu(jnp.matmul(data, w) + b[None, ...])

    ss = ee
    ee = ss + output_dim * hidden_dim
    w = params[ss:ee].reshape([hidden_dim, output_dim])
    ss = ee
    ee = ss + output_dim
    b = params[ss:ee]
    x = jnp.matmul(x, w) + b[None, ...]

    return x

@jax.jit
def get_acc(logits, labels):
    predicted_labels = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_labels == labels)
