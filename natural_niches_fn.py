import jax
import jax.numpy as jnp
from collections import defaultdict

from tqdm import tqdm

from model import num_params
from data import load_data
from model import mlp, get_acc
from helper_fn import crossover, crossover_without_splitpoint, mutate


def sample_parents(
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    rand_key: jnp.ndarray,
    alpha: float,
    use_matchmaker: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1, k2 = jax.random.split(rand_key)
    z = scores.sum(axis=0)
    z = jnp.where(z, z, 1) ** alpha
    fitness_matrix = scores / z[None, :]
    fitness = jnp.sum(fitness_matrix, axis=1)
    probs = fitness / jnp.sum(fitness)
    # first parent
    if use_matchmaker:
        parent_1_idx = jax.random.choice(k1, probs.size, shape=(1,), p=probs)[0]
        # second parent
        match_score = jnp.maximum(0, fitness_matrix - fitness_matrix[parent_1_idx, :]).sum(axis=1)
        probs = match_score / jnp.sum(match_score)
        parent_2_idx = jax.random.choice(k2, probs.size, shape=(1,), p=probs)[0]
    else:
        parent_2_idx, parent_1_idx = jax.random.choice(k1, probs.size, shape=(2,), p=probs)
    return archive[parent_1_idx], archive[parent_2_idx]


@jax.jit
def update_archive(score: jnp.ndarray, param: jnp.ndarray, archive: jnp.ndarray, 
                   scores: jnp.ndarray, alpha: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    ext_scores = jnp.concatenate([scores, score[None, ...]], axis=0) # (pop_size + 1, num_datapoints)

    z = jnp.sum(ext_scores, axis=0) ** alpha # (num_datapoints,)
    # avoid div by zero
    z = jnp.where(z, z, 1)

    
    ext_scores /= z[None, :] 
    fitness = jnp.sum(ext_scores, axis=1)  # (pop_size + 1,)

    # get worst performing
    worst_ix = jnp.argmin(fitness)
    update_mask = worst_ix < scores.shape[0]

    scores = scores.at[worst_ix].set(
        jax.lax.select(update_mask, score, scores[worst_ix])
    )
    archive = archive.at[worst_ix].set(jax.lax.select(update_mask, param, archive[worst_ix]))

    return archive, scores


def run_natural_niches(runs: int, pop_size: int, total_forward_passes: int, store_train_results: bool,
                       use_matchmaker: bool, use_crossover: bool, use_splitpoint: bool, alpha: float = 1.0) -> list:
    (x_train, y_train), (x_test, y_test) = load_data()
    results = []
            
    for run in tqdm(range(runs), desc="Runs"):
        results.append(defaultdict(list))
        result = results[-1]
        seed = 42 + run
        key = jax.random.PRNGKey(seed)
        
        # initialization
        archive = jnp.zeros([pop_size, num_params])
        scores = jnp.zeros([pop_size, len(x_train)], dtype=jnp.bool)
        
        # random initialise two models and place them in the archive
        key, key1, key2 = jax.random.split(key, 3)
        model_1 = jax.random.normal(key1, (num_params,)) * 0.01
        model_2 = jax.random.normal(key2, (num_params,)) * 0.01
        
        for model in (model_1, model_2):
            logits = mlp(model, x_train)
            score = jnp.argmax(logits, axis=1) == y_train
            archive, scores = update_archive(score, model, archive, scores, alpha)
        
        for i in tqdm(range(total_forward_passes), desc="Forward passes"):
            k1, k2, k3, key = jax.random.split(key, 4)
            parents = sample_parents(archive, scores, k1, alpha, use_matchmaker)
            if use_crossover:
                if use_splitpoint:
                    child = crossover(parents, k2)
                else:
                    child = crossover_without_splitpoint(parents, k2)
            else:
                child = parents[0]
            child = mutate(child, k3)
            logits = mlp(child, x_train)
            score = jnp.argmax(logits, axis=1) == y_train
            archive, scores = update_archive(score, child, archive, scores, alpha)
            
            # log results
            result["evals"].append(i)
            train_acc = scores.mean(axis=1)
            if store_train_results:
                # best train acc
                result["train_values"].append(train_acc.max())
            
            # test acc
            best_individual = jnp.argmax(train_acc)
            logits = mlp(archive[best_individual], x_test)
            acc = get_acc(logits, y_test)
            result["test_values"].append(acc)
            if i % 1000 == 0:
                print(f"Run {run}, Forward pass {i}, Test acc: {acc}")
    return results