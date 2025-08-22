from collections import defaultdict
import os

# From stackoverflow: https://stackoverflow.com/questions/77908236/jaxlib-xla-extension-xlaruntimeerror-internal-failed-to-execute-xla-runtime-ex
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import cma
from tqdm import tqdm
import jax.numpy as jnp
import jax
import numpy as np

from data import load_data
from model import mlp, get_acc, num_params


v_mlp = jax.vmap(mlp, in_axes=(0, None))
v_get_acc = jax.vmap(get_acc, in_axes=(0, None))


def run_cma_es(
    runs: int, pop_size: int, total_forward_passes: int, store_train_results: bool
) -> list:
    (x_train, y_train), (x_test, y_test) = load_data()
    results = []
    assert (
        total_forward_passes % pop_size == 0
    ), "Total forward passes must be divisible by pop size"

    for run in tqdm(range(runs), desc="Runs"):
        results.append(defaultdict(list))
        result = results[-1]
        seed = 42 + run
        key = jax.random.PRNGKey(seed=seed)
        rand_params = jax.random.normal(key, shape=(num_params,)) * 0.01

        solver = cma.CMAEvolutionStrategy(
            x0=rand_params,
            sigma0=0.01,
            inopts={
                "popsize": pop_size,
                "seed": seed,
                "randn": np.random.randn,
            },
        )

        for i in tqdm(range(total_forward_passes // pop_size), desc="Generations"):
            solutions = solver.ask()
            batch_logits = v_mlp(jnp.array(solutions), x_train)
            acc_batch = v_get_acc(batch_logits, y_train)
            neg_fitness = -acc_batch
            solver.tell(solutions, neg_fitness.tolist())
            result["evals"] += [i * pop_size]
            if store_train_results:
                result["train_values"] += [-solver.result.fbest]

            # test
            logits = mlp(jnp.array(solver.result.xbest), x_test)
            acc = get_acc(logits, y_test)
            result["test_values"] += [acc]
            print(f"Run {run} Iteration {i} Best Acc: {-solver.result.fbest:}")
    return results
