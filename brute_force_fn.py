import jax.numpy as jnp

from helper_fn import slerp, get_pre_trained_models
from data import load_data
from model import mlp, get_acc


def run_brute_force():
    (x_train, y_train), (x_test, y_test) = load_data()
    model_1, model_2 = get_pre_trained_models()

    best_train = -jnp.inf
    best_w = None
    for w in jnp.linspace(0, 1, 10000):
        model = slerp(w, model_1, model_2)
        logits = mlp(model, x_train)
        acc = get_acc(logits, y_train)
        if acc > best_train:
            best_train = acc
            best_w = w

    model = slerp(best_w, model_1, model_2)
    logits = mlp(model, x_test)
    acc = get_acc(logits, y_test)
    return {"best_w": best_w, "test_acc": acc}
