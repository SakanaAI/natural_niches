import jax
import jax.numpy as jnp
from collections import defaultdict

from tqdm import tqdm

from helper_fn import crossover, mutate, get_pre_trained_models
from model import mlp, get_acc, num_params
from data import load_data


#@jax.jit
def update_archive(
    behavior_descriptors: tuple[float, float],
    quality: float,
    param: jnp.ndarray,
    archive: jnp.ndarray,
    qualities: jnp.ndarray,
    occupied: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Calculate BCs and Q.
    odd_acc, even_acc = behavior_descriptors
    ix0 = jnp.floor(odd_acc * 10).astype(jnp.int32)
    ix1 = jnp.floor(even_acc * 10).astype(jnp.int32)

    # Update states.
    update_mask = jnp.logical_or(~occupied[ix0, ix1], quality > qualities[ix0, ix1])
    occupied = occupied.at[ix0, ix1].set(
        jax.lax.select(update_mask, True, occupied[ix0, ix1])
    )
    qualities = qualities.at[ix0, ix1].set(
        jax.lax.select(update_mask, quality, qualities[ix0, ix1])
    )
    archive = archive.at[ix0, ix1].set(
        jax.lax.select(update_mask, param, archive[ix0, ix1])
    )

    return archive, qualities, occupied


def sample_parents(
    archive: jnp.ndarray,
    occupied: jnp.ndarray,
    qualities: jnp.ndarray,
    rand_key: jnp.ndarray,
) -> jnp.ndarray:
    indices = jnp.argwhere(occupied)
    flat_qualities = qualities[occupied]
    probs = flat_qualities / jnp.sum(flat_qualities)
    sampled_indices = jax.random.choice(
        rand_key,
        flat_qualities.size,
        shape=(2,),
        p=probs,
        replace=jnp.sum(flat_qualities) < 2,
    )
    sampled_positions = indices[sampled_indices]
    return archive[sampled_positions[:, 0], sampled_positions[:, 1]]


def run_map_elites(
    runs: int, total_forward_passes: int, store_train_results: bool, use_pre_trained: bool
) -> list:
    (x_train, y_train), (x_test, y_test) = load_data()
    mask = y_train % 2 == 1
    odd_num_images, odd_num_labels = x_train[mask], y_train[mask]
    mask = y_train % 2 == 0
    even_num_images, even_num_labels = x_train[mask], y_train[mask]
    odd_len = len(odd_num_images)
    even_len = len(even_num_images)
    
    if use_pre_trained:
        model_1, model_2 = get_pre_trained_models()

    results = []
    for run in tqdm(range(runs), desc="Runes"):
        results.append(defaultdict(list))
        result = results[-1]
        seed = 42 + run
        key = jax.random.PRNGKey(seed=seed)

        # initialization
        archive = jnp.zeros([10, 10, num_params])
        qualities = jnp.zeros([10, 10])
        occupied = jnp.zeros([10, 10], dtype=jnp.bool)

        if not use_pre_trained:
            # random initialise two models and place them in the archive
            key, key1, key2 = jax.random.split(key, 3)
            model_1 = jax.random.normal(key1, (num_params,)) * 0.01
            model_2 = jax.random.normal(key2, (num_params,)) * 0.01

        for model in (model_1, model_2):
            # behaviour descriptors
            bd = get_acc(mlp(model, odd_num_images), odd_num_labels), get_acc(
                mlp(model, even_num_images), even_num_labels
            )
            quality = (bd[0] * odd_len + bd[1] * even_len) / (odd_len + even_len)

            archive, qualities, occupied = update_archive(
                bd, quality, model, archive, qualities, occupied
            )

        for step in tqdm(range(total_forward_passes), desc="Forward passes"):
            k1, k2, k3, key = jax.random.split(key, 4)
            parents = sample_parents(archive, occupied, qualities, k1)
            child = crossover(parents, k2)
            if not use_pre_trained:  # mutate only when starting from scratch
                child = mutate(child, k3)
            bd = get_acc(mlp(child, odd_num_images), odd_num_labels), get_acc(
                mlp(child, even_num_images), even_num_labels
            )
            quality = (bd[0] * odd_len + bd[1] * even_len) / (odd_len + even_len)
            archive, qualities, occupied = update_archive(
                bd, quality, child, archive, qualities, occupied
            )

            # log
            best_quality = jnp.max(qualities)
            result["evals"].append(step)
            if store_train_results:
                result["train_values"].append(best_quality)

            # test
            i, j = jnp.where(qualities == best_quality)
            acc = get_acc(mlp(archive[i[0], j[0], :], x_test), y_test)
            result["test_values"].append(acc)
            if step % 1000 == 0:
                print(f"Run: {run}, Step: {step}, Test Acc: {acc}, Train Acc: {best_quality}")
    return results


if __name__ == "__main__":
    run_map_elites(1, 10000, True)