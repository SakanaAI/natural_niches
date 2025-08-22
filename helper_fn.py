import jax
import jax.numpy as jnp

from model import train, num_params
from data import load_data


def slerp(val: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover_without_splitpoint(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    w = jax.random.uniform(rand_key)
    return slerp(w, parents[0], parents[1])


def slerp_w_splitpoint(
    val: jnp.ndarray, split_point: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    val = jnp.ones_like(x) * val
    mask = jnp.arange(len(val)) < split_point
    val = jnp.where(mask, val, 1 - val)

    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    k1, k2 = jax.random.split(rand_key)
    split_point = jax.random.randint(k1, shape=(), minval=0, maxval=parents[0].shape[0])
    w = jax.random.uniform(k2)
    return slerp_w_splitpoint(w, split_point, parents[0], parents[1])


@jax.jit
def mutate(params: jnp.ndarray, rand_key: jnp.ndarray, std: float = 0.01):
    noise = jax.random.normal(rand_key, shape=params.shape) * std
    return params + noise


def get_pre_trained_models() -> tuple[jnp.ndarray, jnp.ndarray]:
    (x_train, y_train), _ = load_data()
    mask = y_train < 5
    d0to4_images, d0to4_labels = x_train[mask], y_train[mask]
    d5to9_images, d5to9_labels = x_train[~mask], y_train[~mask]
    seed_key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(seed_key, 2)
    model_1 = jax.random.normal(key1, (num_params,)) * 0.01
    model_2 = jax.random.normal(key2, (num_params,)) * 0.01
    model_1 = train(model_1, d0to4_images, d0to4_labels)
    model_2 = train(model_2, d5to9_images, d5to9_labels)
    return model_1, model_2
