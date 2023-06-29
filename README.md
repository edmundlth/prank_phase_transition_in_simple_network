# prank_phase_transition_in_simple_network
Use a notion of "proximal rank" of a 1 hidden layer neural network to investigate its phases and phase transition.

# Usage 
Getting started: 
```
$ python mcmc.py --input_dim 1 --layer_sizes 30 1 --num_posterior_samples 5000 --num_chains 6 --num_training_data 100 --output_dir ./outputs/ --show_plot --rngseed 42
```


Other parameters (see `$ python mcmc.py --help`)
```
$ python mcmc.py --help
usage: mcmc.py [-h] [--output_dir OUTPUT_DIR]
               [--num_posterior_samples [NUM_POSTERIOR_SAMPLES]] [--thinning [THINNING]]
               [--num_warmup [NUM_WARMUP]] [--num_chains [NUM_CHAINS]]
               [--sigma_obs [SIGMA_OBS]] [--prior_std [PRIOR_STD]]
               [--prior_mean [PRIOR_MEAN]] [--num_training_data [NUM_TRAINING_DATA]]
               [--prank_eps [PRANK_EPS]] [--x_window X_WINDOW [X_WINDOW ...]]
               [--input_dim [INPUT_DIM]] [--layer_sizes LAYER_SIZES [LAYER_SIZES ...]]
               [--activation_fn_name [ACTIVATION_FN_NAME]] [--device DEVICE]
               [--plot_posterior_samples] [--quiet] [--show_plot] [--rngseed [RNGSEED]]
```


# To Do's 
  - [ ] Write explanation of the project in README. 
  - [x] Use bar charts instead of histogram?
  - [x] As sanity check also plot the prank frequency of samples from prior to compare. 
  - [ ] tSNE using metric derived from canonicalize.py.
  - [ ] Visualise random samples as point cloud. 
  - [ ] Run for a range of `n` and for different `rngseed`.
  - [ ] Plot changes in p-rank frequency in `n`. 
  - [ ] Run experiments with true network with smaller rank.
  - [ ] Run experiments with true network with larger rank.
  

