import jax
import jax.numpy as jnp
import jax.tree_util as jtree

import haiku as hk
import numpyro

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import numpy as np

import functools
import os
import argparse
import json

from mlp_haiku import (
    build_forward_fn,
    build_log_likelihood_fn,
    generate_input_data,
    generate_output_data,
    build_model,
    run_mcmc,
    MCMCConfig,
    ACTIVATION_FUNC_SWITCH
)
from approx_prank import bound
from utils import (
    prior_prank_freq, 
    bar_plot, 
    param_tree_to_dict, 
    to_freq
)


import logging

logger = logging.getLogger("__main__")  # TODO: better config

def main(args):
    rngseed = args.rngseed
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))
    if args.layer_sizes is None:
        true_param_array = [jnp.array([list(args.b0)]), jnp.array([list(args.a0)])]
        layer_sizes = [len(args.b0), len(args.a0)]
    else:
        true_param_array = None
        layer_sizes = args.layer_sizes

    input_dim = args.input_dim
    XMIN, XMAX = args.x_window

    activation_fn = ACTIVATION_FUNC_SWITCH[args.activation_fn_name.lower()]
    forward = build_forward_fn(
        layer_sizes=layer_sizes,
        activation_fn=activation_fn,
        initialisation_mean=args.prior_mean,
        initialisation_std=args.prior_std,
    )
    forward = hk.transform(forward)
    # log_likelihood_fn = functools.partial(
    #     build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
    # )

    X = generate_input_data(
        args.num_training_data,
        input_dim=input_dim,
        rng_key=next(rngkeyseq),
        xmin=XMIN,
        xmax=XMAX,
    )
    init_param = forward.init(next(rngkeyseq), X)
    init_param_flat, treedef = jtree.tree_flatten(init_param)
    if true_param_array is not None:
        true_param = treedef.unflatten(true_param_array)
    else:
        true_param = init_param
    Y = generate_output_data(
        forward, true_param, X, next(rngkeyseq), sigma=args.sigma_obs
    )
    param_center = true_param
    # print(json.dumps(jtree.tree_map(lambda x: x.tolist(), true_param), indent=2))
    print(param_tree_to_dict(true_param))
    # print([elem.tolist() for elem in jtree.tree_leaves(true_param)])
    # param_prior_sampler = functools.partial(
    #     const_factorised_normal_prior, param_shapes, treedef, args.prior_mean, args.prior_std
    # )
    model = functools.partial(
        build_model,
        forward.apply,
        param_center=param_center,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        sigma=args.sigma_obs,
    )
    mcmc_config = MCMCConfig(
        args.num_posterior_samples, args.num_warmup, args.num_chains, args.thinning
    )

    mcmc = run_mcmc(
        model,
        [X, Y],
        next(rngkeyseq),
        mcmc_config=mcmc_config,
        init_params=None,
        itemp=1.0,
        progress_bar=(not args.quiet),
    )
    posterior_samples = mcmc.get_samples()
    num_mcmc_samples = len(posterior_samples[list(posterior_samples.keys())[0]])

    input_dim = args.input_dim
    num_hidden_nodes = args.layer_sizes[0]
    output_dim = args.layer_sizes[-1]
    key_mappings = {"0": "c", "1": "b", "2": "d", "3": "a"}
    posterior_samples = {
        key_mappings[key]: jnp.squeeze(val) for key, val in posterior_samples.items()
    }
    prank_rec = []
    for i in range(num_mcmc_samples):
        param = {key: val[i] for key, val in posterior_samples.items()}
        prank = bound(0.1, param)
        prank_rec.append(prank)

    for key, val in posterior_samples.items():
        print(key, val.shape)
        
    fig = plt.figure(constrained_layout=True, figsize=(8, 8))

    # Define the grid
    gs = gridspec.GridSpec(4, 4, figure=fig)

    ax = fig.add_subplot(gs[:2, :2])
    prank_freqs = to_freq(prank_rec)
    print(prank_freqs)
    bar_plot(prank_freqs, ax=ax)
    ax.set_xlabel("p-rank")
    ax.set_title("posterior samples pranks")

    ax = fig.add_subplot(gs[:2, 2:])
    prior_prank_freq_dict = prior_prank_freq(
        h=num_hidden_nodes,
        prior_std=args.prior_std,
        eps=args.prank_eps,
        num_samples=10000,
    )
    bar_plot(prior_prank_freq_dict, ax=ax)
    ax.set_xlabel("p-rank")
    ax.set_title("prior samples pranks")

    # plot true function and data.
    ax = fig.add_subplot(gs[2:, :2])
    x = np.linspace(XMIN, XMAX).reshape(-1, 1)
    ax.plot(x, forward.apply(true_param, None, x), "r--")
    ax.plot(X, Y, "kx")
    true_prank = bound(args.prank_eps, param_tree_to_dict(true_param))
    ax.set_title(f"True network and data. prank={true_prank}")

    ax = fig.add_subplot(gs[2:, 2:])
    ax.plot(posterior_samples["b"], posterior_samples["c"], "k.", alpha=0.5)
    ax.set_title("In-weights and biases")
    ax.set_xlabel("b")
    ax.set_ylabel("c")

    fig.suptitle(
        f"p-rank histogram: "
        f"$n=${args.num_training_data}, "
        f"prior $\sigma$={args.prior_std}, "
        f"obs $\sigma=${args.sigma_obs}, "
        f"$\epsilon=${args.prank_eps}, "
        f"layers={[args.input_dim] + args.layer_sizes}, "
        f"num_mcmc={num_mcmc_samples}, "
        f"seed={args.rngseed}"
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = f"prankmcmc_n{args.num_training_data}_sigmap{args.prior_std}_sigmaobs{args.sigma_obs}_h{num_hidden_nodes}_seed{args.rngseed}.png"
        filepath = os.path.join(args.output_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        print(f"File saved by: {filepath}")

    if args.show_plot:
        plt.show()
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="p-rank analysis of MCMC samples of 1 hidden layer tanh model."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="a directory for storing output files.",
    )
    parser.add_argument("--num_posterior_samples", nargs="?", default=1000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=4, type=int)
    parser.add_argument("--sigma_obs", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_std", nargs="?", default=10.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)

    parser.add_argument("--num_training_data", nargs="?", default=132, type=int)
    # parser.add_argument("--a0", nargs="+", default=None, type=float)
    # parser.add_argument("--b0", nargs="+", default=None, type=float)
    parser.add_argument(
        "--prank_eps",
        nargs="?",
        default=0.01,
        type=float,
        help="Epsilon to be used in the p-rank algorithm",
    )
    parser.add_argument(
        "--x_window",
        nargs="+",
        type=int,
        default=[-20, 20],
        help="xmin and xmax",
    )
    parser.add_argument(
        "--input_dim",
        nargs="?",
        default=1,
        type=int,
        help="Dimension of the input data X.",
    )
    parser.add_argument(
        "--layer_sizes",
        nargs="+",
        default=[5, 1],
        type=int,
        help="A optional list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. If specified, --a0 and --b0 are ignored. ",
    )

    parser.add_argument("--activation_fn_name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')

    parser.add_argument("--plot_posterior_samples", action="store_true", default=False)
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument(
        "--show_plot", action="store_true", default=False, help="plt.show() if true."
    )
    parser.add_argument("--rngseed", nargs="?", default=42, type=int)

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
