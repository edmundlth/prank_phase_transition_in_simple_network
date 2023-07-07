import jax
import jax.numpy as jnp
import jax.tree_util as jtree

import haiku as hk
import optax

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import scipy
import numpy as np
import argparse
import os

from approx_prank import bound

from mlp_haiku import (
    build_forward_fn,
    build_log_likelihood_fn,
    generate_input_data,
    generate_output_data,
    ACTIVATION_FUNC_SWITCH
)
from utils import (
    param_tree_to_dict
)

def minibatch_generator(X, Y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], Y[excerpt]


def main(args):
    rngseed = args.rngseed
    np.random.seed(args.rngseed + 5)
    rngkeyseq = hk.PRNGSequence(jax.random.PRNGKey(rngseed))
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
    X_test = generate_input_data(
        5000,
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
    Y_test = generate_output_data(
        forward, true_param, X_test, next(rngkeyseq), sigma=args.sigma_obs
    )

    ######################
    # Run SGD
    ######################
    loss_fn = lambda params, x, y: optax.l2_loss(forward.apply(params, None, x), y).mean()
    # loss_fn = lambda params, x, y: optax.huber_loss(forward.apply(params, None, x), y).mean()
    loss_fn = jax.jit(loss_fn)
    if args.optim_name.lower() == "sgd":
        optimizer = optax.sgd(args.learning_rate)
    elif args.optim_name.lower() == "adam":
        optimizer = optax.adam(args.learning_rate)
        # init_value = args.learning_rate
        # decay_steps = 1000
        # def schedule_fn(step):
        #     lr = init_value * 0.5 * (1 + jnp.cos(jnp.pi * (step % decay_steps) / decay_steps))
        #     return lr

        # optimizer = optax.chain(
        #     optax.adam(args.learning_rate), 
        #     optax.scale_by_schedule(schedule_fn)
        # )
    else:
        raise NotImplementedError(f"{args.optim_name} not implemented")

    params = jtree.tree_map(
        lambda x: jax.random.uniform(next(rngkeyseq), shape=x.shape, minval=-10.0, maxval=10.0), 
        init_param
    )
    print(params)
    opt_state = optimizer.init(params)
    rec = []
    num_snapshots = 5
    snapshots = {
        t: None 
        for t in range(args.num_epoch // num_snapshots, args.num_epoch, args.num_epoch // num_snapshots)
    }
    batch_size = int(args.batch_size) if args.batch_size >= 1 else int(args.num_training_data * args.batch_size)
    for t in range(args.num_epoch):
        for X_batch, Y_batch in minibatch_generator(X, Y, batch_size=batch_size):
            gradients = jax.grad(loss_fn)(params, X_batch, Y_batch)
            updates, opt_state = optimizer.update(gradients, opt_state)
            params = optax.apply_updates(params, updates)
        prank = bound(args.prank_eps, param_tree_to_dict(params))
        prank_small = bound(args.prank_eps / 2, param_tree_to_dict(params))
        prank_large = bound(args.prank_eps * 2, param_tree_to_dict(params))
        train_loss = loss_fn(params, X, Y)
        test_loss = loss_fn(params, X_test, Y_test)
        # steps = opt_state[0][0].count
        print(f"Epoch {t:4d}: prank={prank_large: 2d}, {prank:2d}, {prank_small:2d}, train:{train_loss:.4f}, test:{test_loss:.4f}")
        rec.append((t, prank, prank_small, prank_large, train_loss, test_loss))

        if t in snapshots:
            snapshots[t] = {
                "params": params, 
                "prank": prank, 
                "prank_small": prank_small, 
                "train_loss": train_loss, 
                "test_loss": test_loss
            }
    
    ######################
    # Plots
    ######################
    fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    gs = gridspec.GridSpec(5, 5, figure=fig)

    x = np.linspace(XMIN, XMAX).reshape(-1, 1)
    for i, t in enumerate(sorted(snapshots.keys())):    
        ax = fig.add_subplot(gs[:2, i:i + 1])
        param_snapshot = snapshots[t]["params"]
        ax.plot(x, forward.apply(true_param, None, x), "r--", alpha=0.5)
        ax.plot(X, Y, "kx", alpha=0.3)
        ax.plot(x, forward.apply(param_snapshot, None, x), "b--")
        snapshot_prank = bound(args.prank_eps, param_tree_to_dict(param_snapshot))
        ax.set_title(f"prank={snapshot_prank}")


    ax = fig.add_subplot(gs[2:, :])
    rec = np.array(rec)[args.num_epoch // 20:]
    ax.plot(rec[:, 0], rec[:, 1], "kx--", label=f"prank_eps={args.prank_eps}", alpha=0.5, markersize=3)
    ax.plot(rec[:, 0], rec[:, 2], "rx--", label=f"prank_eps={args.prank_eps / 2}", alpha=0.5, markersize=3)
    ax.plot(rec[:, 0], rec[:, 3], "bx--", label=f"prank_eps={args.prank_eps * 2}", alpha=0.5, markersize=3)
    ax.legend()
    

    ax = ax.twinx()
    ax.plot(rec[:, 0], rec[:, 4], color="orange", alpha=0.8, label="train loss")
    ax.plot(rec[:, 0], rec[:, 5], color="skyblue", alpha=0.8, label="test loss")
    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax.vlines(sorted(snapshots.keys()), ymin=ymin, ymax=ymax)
    ax.set_yscale('log')

    true_prank = bound(args.prank_eps, param_tree_to_dict(true_param))
    fig.suptitle(
        f"p-rank histogram: "
        f"$n=${args.num_training_data}, "
        f"true prank={true_prank}, "
        f"prior $\sigma$={args.prior_std}, "
        f"obs $\sigma=${args.sigma_obs}, "
        f"$\epsilon=${args.prank_eps}, "
        f"layers={[args.input_dim] + args.layer_sizes}, "
        f"num_epoch={args.num_epoch}, "
        f"batch size={args.batch_size}, "
        f"optim={args.optim_name}, "
        f"lr={args.learning_rate}, "
        f"seed={args.rngseed}"
    )

    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = (
            f"pranksgd_n{args.num_training_data}_"
            f"sigmap{args.prior_std}_"
            f"sigmaobs{args.sigma_obs}_"
            f"h{args.layer_sizes[0]}_"
            f"tprank{true_prank}_"
            f"nepoch{args.num_epoch}_"
            f"batch{args.batch_size}_"
            f"optim{args.optim_name}_"
            f"lr{args.learning_rate}_"
            f"seed{args.rngseed}.png"
        )
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
    parser.add_argument("--num_epoch", nargs="?", default=100, type=int)
    parser.add_argument(
        "--prank_eps",
        nargs="?",
        default=0.1,
        type=float,
        help="Epsilon to be used in the p-rank algorithm",
    )
    parser.add_argument("--num_training_data", nargs="?", default=132, type=int)
    parser.add_argument("--batch_size", nargs="?", default=0.1, type=float)
    parser.add_argument("--sigma_obs", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_std", nargs="?", default=10.0, type=float)
    parser.add_argument("--prior_mean", nargs="?", default=0.0, type=float)
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
    parser.add_argument(
        "--show_plot", action="store_true", default=False, help="plt.show() if true."
    )
    
    parser.add_argument("--rngseed", nargs="?", default=42, type=int)
    parser.add_argument("--activation_fn_name", nargs="?", default="tanh", type=str)
    parser.add_argument("--optim_name", nargs="?", default="sgd", type=str, help="sgd | adam")
    parser.add_argument("--learning_rate", nargs="?", default=0.001, type=float)
    
    args = parser.parse_args()

    main(args)




