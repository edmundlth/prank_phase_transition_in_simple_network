import argparse

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import scipy
import functools

import haiku as hk
import numpyro
import matplotlib.pyplot as plt

from mlp_haiku import (
    build_forward_fn,
    build_log_likelihood_fn,
    generate_input_data,
    generate_output_data,
    build_model,
    run_mcmc,
    MCMCConfig
)
import logging
logger = logging.getLogger("__main__")  # TODO: better config

ACTIVATION_FUNC_SWITCH = {
    "tanh": jax.nn.tanh, 
    "id": lambda x: x, 
    "relu": jax.nn.relu, 
    "gelu": jax.nn.gelu, 
    "swish": jax.nn.swish, 
}

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

    activation_fn = ACTIVATION_FUNC_SWITCH[args.activation_fn_name.lower()]
    forward = build_forward_fn(
        layer_sizes=layer_sizes,
        activation_fn=activation_fn,
        initialisation_mean=args.prior_mean,
        initialisation_std=args.prior_std,
    )
    forward = hk.transform(forward)
    log_likelihood_fn = functools.partial(
        build_log_likelihood_fn, forward.apply, sigma=args.sigma_obs
    )

    X = generate_input_data(
        args.num_training_data, input_dim=input_dim, rng_key=next(rngkeyseq)
    )
    init_param = forward.init(next(rngkeyseq), X)
    init_param_flat, treedef = jtree.tree_flatten(init_param)
    # param_shapes = [p.shape for p in init_param_flat]
    if true_param_array is not None:
        true_param = treedef.unflatten(true_param_array)
    else:
        true_param = init_param
    Y = generate_output_data(
        forward, true_param, X, next(rngkeyseq), sigma=args.sigma_obs
    )
    param_center = true_param

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
        progress_bar=(not args.quiet)
    )

    posterior_samples = mcmc.get_samples()
    print(posterior_samples.keys())
    print(posterior_samples['0'])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLCT estimation of MLP models.")
    parser.add_argument("--num_posterior_samples", nargs="?", default=1000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=4, type=int)
    parser.add_argument("--sigma_obs", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)
    
    parser.add_argument("--num_training_data", nargs="?", default=1032, type=int)
    parser.add_argument("--a0", nargs="+", default=[0.5], type=float)
    parser.add_argument("--b0", nargs="+", default=[0.9], type=float)
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
        default=None,
        type=int,
        help="A optional list of positive integers specifying MLP layers sizes from the first non-input layer up to and including the output layer. If specified, --a0 and --b0 are ignored. ",
    )
    
    parser.add_argument("--activation_fn_name", nargs="?", default="tanh", type=str)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="a directory for storing output files.",
    )
    parser.add_argument("--plot_posterior_samples", action="store_true", default=False)
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Lower verbosity level."
    )
    parser.add_argument("--rngseed", nargs="?", default=42, type=int)

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
