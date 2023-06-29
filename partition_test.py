import itertools
from math import sqrt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


from approx_prank import approx_partition


def generate_partition(num_points, epsilon, center):
    box_radius = (1 / sqrt(2)) * epsilon
    # uniform distribution on [-box_radius, box_radius] ^2 re-centered to center
    return stats.uniform(loc=-box_radius, scale=2 * box_radius).rvs(size=(num_points, 2)) + center



def generate_partition_centers(num_partitions, epsilon, max_tries=1000, center_var_per_epsilon=10):

    centers = []
  
    for _ in range(max_tries):

        if len(centers) == num_partitions:
            break 

        candidate = stats.norm(loc=0, scale=center_var_per_epsilon*epsilon).rvs(2)
        
        if all( sqrt(sum((c - candidate) ** 2)) > 2 * epsilon for c in centers):
            centers.append(candidate)

    return np.array(centers)


def generate_partitions(num_partitions, epsilon):
    
    partition_sizes = stats.poisson(1).rvs(num_partitions)
    partitions = []

    for size, center in zip(partition_sizes, generate_partition_centers(num_partitions, epsilon)):
        partitions.append(generate_partition(size, epsilon, center))

    return partitions


def check_test(candidate_partitions, true_partitions):
    candidate_partitions = set(list(p) for p in candidate_partitions)
    true_partitions = set(list(p) for p in true_partitions)
    return candidate_partitions == true_partitions


if __name__ == "__main__":
    

    num_tests = 3
    epsilon = 0.25

    for _ in range(num_tests):

        num_partitions = stats.poisson(5).rvs(1)

        partitions = generate_partitions(num_partitions, epsilon)

        points = np.array(list(itertools.chain(partitions)))

        candidate_partition_mask = approx_partition(epsilon, points)
        candidate_partitions = points[candidate_partition_mask]

        print(check_test(candidate_partitions, partitions))


    

