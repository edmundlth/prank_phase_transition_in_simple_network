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

def bound(epsilon, w):
    """
    * u: {
          'a': np.array(length h),
          'b': np.array(length h),
          'c': np.array(length h),
          'd': scalar,
        }
    """
    a = w['a']
    b = w['b']
    c = w['c']
    h = len(a)
    sgn = np.sign(b)

    # I = [i for i in range(h) if np.abs(b[i]) > epsilon]
    I = (sgn * b > epsilon)
    
    # [[np.sign(b[i])*b[i], np.sign(b[i])*c[i]] for i in I],
    Pi = approx_partition(
        epsilon,
        np.column_stack([sgn*b, sgn*c])[I],
    )

    # alpha = [sum(np.sign(w['b'][i])*w['a'][i] for i in Pi_j) for Pi_j in Pi]
    alphas = ((sgn * a)[I] * Pi).sum(axis=-1)

    # return len([j for j in range(len(Pi)) if abs(alphas[j]) > epsilon * len(Pi[j])])
    return np.sum(np.abs(alphas) > Pi.sum(axis=-1) * epsilon)



if __name__ == "__main__":
    # w = {
    #     'a': [1.1, 0.9, 0.2, 1.3, 1],
    #     'b': [1.1, 1.2, 0.6, 1.3, 1],
    #     'c': [0, 0, 0, 0, 0],
    #     'd': 0,
    # }

    w = {
        'a': [-0.8704732, -0.09269352, 0.3384626], 
        'b': [-0.952935, -0.32022476,  0.9650079], 
        'c': [-0.4964598,  1.8577402, -1.4160455],
        'd': -1.2973489
    }

    print(bound(1, w), 1)
    print(bound(0.1, w), 4)


