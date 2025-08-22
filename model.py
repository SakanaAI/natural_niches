import jax
import jax.numpy as jnp
from jax.nn import log_softmax
import optax


hidden_dim = 256
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


def get_loss(params, data, labels):
    logits = mlp(params, data)
    log_probs = log_softmax(logits)
    true_log_probs = jnp.take_along_axis(log_probs, labels[:, None], axis=1)
    return -jnp.mean(true_log_probs)


get_grad = jax.grad(get_loss)

@jax.jit
def train(params, x, y, lr=0.003, num_epoches=10):
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    for _ in range(num_epoches):
        grad = get_grad(params, x, y)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    return params
