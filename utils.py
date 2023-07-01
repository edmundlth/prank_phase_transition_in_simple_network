import numpy as np
from approx_prank import bound
import seaborn as sns
import jax.numpy as jnp

def to_freq(array):
    tally = {}
    for elem in array:
        if elem not in tally:
            tally[elem] = 1
        else:
            tally[elem] += 1
    length = len(array)
    return {elem: count / length for elem, count in tally.items()}


def prior_prank_freq(h, prior_std=10.0, eps=0.01, num_samples=1000):
    prank_rec = []
    for i in range(num_samples):
        param = dict(
            a=np.random.randn(h) * prior_std,
            b=np.random.randn(h) * prior_std,
            c=np.random.randn(h) * prior_std,
            d=np.random.randn(1) * prior_std,
        )
        prank_rec.append(bound(eps, param))
    return to_freq(prank_rec)


def bar_plot(tally_dict, ax, color="skyblue"):
    keys = sorted(tally_dict.keys())
    sns.barplot(
        x=keys, 
        y=[tally_dict[key] for key in keys], 
        color=color,
        ax=ax
    )
    return

def param_tree_to_dict(param_tree):
    return {
        "a": jnp.squeeze(param_tree['mlp/~/linear_1']['w']), 
        "b": jnp.squeeze(param_tree['mlp/~/linear_0']['w']), 
        "c": jnp.squeeze(param_tree['mlp/~/linear_0']['b']), 
        "d": jnp.squeeze(param_tree['mlp/~/linear_1']['b'])
    }
