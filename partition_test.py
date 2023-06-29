import itertools
from math import sqrt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


from approx_prank import approx_partition

def generate_partition(num_points, epsilon, center):
    """ Generates a set of points, centered at the point `center`, such that 
        any two points are at most `epsilon` apart.
    """
    box_diameter = (1 / sqrt(2)) * (epsilon / 2)
    # uniform distribution on [-box_radius, box_radius] ^2 re-centered to center
    return stats.uniform(loc=-box_diameter, scale=2 * box_diameter).rvs(size=(num_points, 2)) + center


def generate_partition_centers(num_partitions, epsilon, max_tries=1000, center_var_per_epsilon=10):
    """ Generates a set of points, intended to the the centers of partitions, 
        whose points are at least 2*epsilon apart.
    """

    centers = []
  
    for _ in range(max_tries):

        if len(centers) == num_partitions:
            break 

        candidate = stats.norm(loc=0, scale=center_var_per_epsilon*epsilon).rvs(2)
        
        if all( sqrt(sum((c - candidate) ** 2)) > 2 * epsilon for c in centers):
            centers.append(candidate)

    return np.array(centers)


def generate_partitions(num_partitions, epsilon):
    """ Generates num_partitions partitions whose elements can be covered by a square 
        of diagonal length `epsilon`
    """
    partition_sizes = stats.poisson(1).rvs(num_partitions) + 1
    partitions = []

    for size, center in zip(partition_sizes, generate_partition_centers(num_partitions, epsilon)):
        partitions.append(generate_partition(size, epsilon, center))

    return partitions


def check_test(candidate_partitions, true_partitions):
    """ Compares two partitions"""
    candidate_partitions = set(tuple(tuple(p) for p in partition) for partition in candidate_partitions)
    true_partitions = set(tuple(tuple(p) for p in partition) for partition in true_partitions)
    return candidate_partitions == true_partitions


def plot_partitions(partition_points):
    """ Plots points, coloured by partition"""
    fig = plt.figure()
    ax = fig.subplots()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for c, points in zip(colors, partition_points):
        ax.scatter(points[:, 0], points[:, 1], color=c, marker='o', s=10)
        
    ax.set_title('Points in Epsilon-Sized Squares')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()
    plt.close()


if __name__ == "__main__":
    

    num_tests = 100
    epsilon = 0.25
    errors = []

    for _ in range(num_tests):

        num_partitions = min(6, stats.poisson(5).rvs(1) + 1)
        partitions = generate_partitions(num_partitions, epsilon)

        points = np.array([p for partition in partitions for p in partition])

        candidate_partition_mask = approx_partition(epsilon, points)
        candidate_partitions = [points[cpm] for cpm in candidate_partition_mask]

        if not check_test(candidate_partitions, partitions):
            errors.append((partitions, candidate_partitions))

    print(f"Completed {num_tests} tests with {len(errors)} errors")

    for true, pred in errors[:3]:

        print(f"True partition:")
        for i, p in enumerate(true):
            print(f"{i}: {p}")

        print("Predicted partition:")
        for i, p in enumerate(pred):
            print(f"{i}: {p}")

        plot_partitions(pred)

        print("\n\n-------------------------------")
    

