import numpy as np

def approx_partition(epsilon, u):
    """
    inputs:

    * epsilon: positive float
    * u: a numpy array of shape (h, 2) (basically a list of 2d points)

    outputs:

    * Pi: a partition of range(0, h) represented as an array of masks!
      * there are k rows
      * each row is an array of length h
      * each element of the row is True iff that point of u is in that
        partition group
      The partition is such that all of the points indexed in each group
      are within L_infty distance 2*epsilon of the other points in the group
      (they are actually all within epsilon of the lowest-index point in the
      group)
    """
    Pi = []
    v = []
    for i in range(len(u)):
        for j in range(len(v)):
            if uniform_distance(u[i], v[j]) <= epsilon:
                Pi[j][i] = True
                break
        else: # if no break
            v.append(u[i])
            # one-hot encode i (actually, 'true-hot')
            mask = np.zeros(len(u), dtype=bool)
            mask[i] = True
            Pi.append(mask)
    return np.array(Pi)

def uniform_distance(u, v):
    """
    inputs:

    * u: numpy array (dimension p)
    * v: numpy array (dimension p, same as u)
    
    returns:

    * uniform distance between u and v, that is:
        
        |u - v|_infty = max_i( abs( u[i] - v[i] ) )

    """
    return np.abs(u - v).max()

def bound(epsilon, w, return_partition=False):
    """
    Inputs:

    * epsilon (non-negative float): radius for the bound algorithm
    * w: parameter of the form: {
          'a': np.array(length h),
          'b': np.array(length h),
          'c': np.array(length h),
          'd': scalar,
        }
    * return_partition (bool, default: False): if true, return some metadata
      implicitly conveying the nearby smaller parameter found

    Returns:

    * bound (int): the bound (see the algorithm for its properties)
    * partition: (1d int array, only provided if return_partition is true)
      TODO: document in more detail, but it's basically:
      * 0 in the array indicates that the unit is eliminated in step 1
        (near-zero b value)
      * >0 in the array indicates that the unit is clustered with all other
        units with the same index in the array, and they are not eliminated
        in stage 3
      * <0 in the array indicates that the unit is clustered with all other
        units with the same index in the array, and they ARE eliminated in
        stage 3
    """
    a = w['a']
    b = w['b']
    c = w['c']
    h = len(a)
    sgn = np.sign(b)

    I = (sgn * b > epsilon)
    
    Pi = approx_partition(
        epsilon,
        np.column_stack([sgn*b, sgn*c])[I],
    )

    alphas = ((sgn * a)[I] * Pi).sum(axis=-1)
    K = (np.abs(alphas) > Pi.sum(axis=-1) * epsilon)
    
    bound = np.sum(K)

    if not return_partition:
        return bound
    else:
        sub_partition = np.zeros(np.sum(I), dtype=int)
        for j in range(len(Pi)):
            sub_partition[Pi[j]] = (j+1) * (2 * K[j] - 1)
        full_partition = np.zeros(h, dtype=int)
        full_partition[I] = sub_partition
        return bound, full_partition



if __name__ == "__main__":
    w = {
        'a': [1.1, 0.9, 0.2, 1.3, 1],
        'b': [1.1, 1.2, 0.6, 1.3, 1],
        'c': [0, 0, 0, 0, 0],
        'd': 0,
    }

    # w = {
    #     'a': [-0.8704732, -0.09269352, 0.3384626], 
    #     'b': [-0.952935, -0.32022476,  0.9650079], 
    #     'c': [-0.4964598,  1.8577402, -1.4160455],
    #     'd': -1.2973489
    # }

    print(bound(1, w), 1)
    print(bound(0.1, w), 4)
    
    print(bound(1, w, return_partition=True))
    print(bound(0.1, w, return_partition=True))


