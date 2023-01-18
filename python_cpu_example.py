# Simple python script to run on CPU cluster. ChatGPT conversion of the R code.

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# https://joblib.readthedocs.io/en/latest/parallel.html

# Define the target distribution (i.e. the distribution you want to sample from)

def target(x):
    # Example: target distribution is a mixture of Gaussians
    return np.exp(-(x + 3) ** 2 / 2) + np.exp(-(x - 3) ** 2 / 2)


# Define the proposal distribution

def proposal(x):
    # Example: proposal distribution is a normal distribution with mean equal to the current value of x,
    # and a fixed standard deviation of 0.5
    return np.random.normal(x, 0.5)


# Run the Metropolis-Hastings algorithm
def metropolis_hastings(num_iters, x_init=0):
    x = x_init
    samples = np.empty(num_iters)
    for i in range(num_iters):
        # Generate a proposal
        x_proposed = proposal(x)
        # Calculate the acceptance probability
        alpha = min(1, (target(x_proposed) * proposal(x)) / (target(x) * proposal(x_proposed)))
        # Generate a uniform random number
        u = np.random.uniform()
        # Accept or reject the proposal
        if u < alpha:
            x = x_proposed
        # Store the sample
        samples[i] = x
    return samples


n_iterations = 1_000_000
n_cores = 16

results = Parallel(n_jobs=n_cores)(delayed(metropolis_hastings)(num_iters=n_iterations) for i in range(n_cores))

plotting_dir = '~/Documents/ClusterTesting/Plots/'

for i in range(n_cores):
    data = results[i]
    plt.hist(data, bins=50, density=True)
    plt.title(f"Histogram of draws from Chain {i + 1}")
    plt.savefig(f"{plotting_dir}python_plot_{i + 1}.png")
