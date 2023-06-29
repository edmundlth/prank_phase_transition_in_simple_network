import numpy as np

def canonicalize(param_dict):
    """ Given the parameters of a one-layer tanh neural network, computes the 
        canonical form of the parameters as defined in Algorithm 4.1 in 
        M. Farrugia-Roberts (2023) “Functional Equivalence and Path Connectivity 
        of Reducible Hyperbolic Tangent Networks.”, http://arxiv.org/abs/2305.05089. 
        
        Args:
            param_dict: a dictionary of parameters for the neural network of the form
                {
                    'a', np.array([a1, a2, ..., an]),
                    'b', np.array([b1, b2, ..., bn]),
                    'c', np.array([c1, c2, ..., cn]),
                    'd': d
                }
                This corresponds to the neural network
                f = lambda x : sum(ai * tanh(bi * x + ci) for i in range(n)) + d

        Returns:
            A dictionary of canonicalized parameters of the same form. All units 
            for which all parameters are zero (i.e. the units which don't contribute
            to the network) are moved to the *front* of the parameter list.       
    """

    num_units = len(param_dict['a'])
    zero_units_mask = np.zeros(num_units, dtype=bool)       # keeps track of zero-d units of the network
    
    # first identify the bi which are zero, and ensure all non-zero bi are positive

    b_sgn = np.sign(param_dict['b'])        # if an entry is 0, the sign is 0

    a_param = b_sgn * param_dict['a'] 
    b_param = b_sgn * param_dict['b']
    c_param = b_sgn * param_dict['c']  

    zero_indices = np.nonzero(b_sgn == 0)
    zero_units_mask[zero_indices] = True

    d = param_dict['d']
    d += np.sum(np.tanh(c_param[zero_indices]) * a_param[zero_indices])
    
    # merge with same bi , bi, sum the ai's 
    sorted_i = np.lexsort((c_param, b_param))

    a_param = a_param[sorted_i]
    b_param = b_param[sorted_i]
    c_param = c_param[sorted_i]
    zero_units_mask = zero_units_mask[sorted_i]

    i = 0
    while i < num_units:

        if zero_units_mask[i]: 
            i += 1
            continue

        j = i
        while (j+1 < num_units) and (b_param[j] == b_param[j+1]) and (c_param[j] == c_param[j+1]):
            j += 1

        if i == j:
            # nothing to merge
            i += 1
            continue
            
        # now we want to merge units in the slice [i:j+1]

        # want to zero the units in the slice [i+1:j+1]
        zero_indices = np.arange(i+1, j+1)
        
        # add the all a-weights to the a-weight of the ith unit
        a_param[i] += np.sum(a_param[zero_indices])

        a_param[zero_indices] = 0
        b_param[zero_indices] = 0
        c_param[zero_indices] = 0
        zero_units_mask[zero_indices] = True

        i = j + 1
    
    # eliminate units with zero a_i 
    zero_indices = np.nonzero(a_param == 0) 
    zero_units_mask[zero_indices] = True
    b_param[zero_indices] = 0
    c_param[zero_indices] = 0

    return {
        'a': np.hstack((a_param[zero_units_mask], a_param[~zero_units_mask])), 
        'b': np.hstack((b_param[zero_units_mask], b_param[~zero_units_mask])), 
        'c': np.hstack((c_param[zero_units_mask], c_param[~zero_units_mask])),
        'd': d
    }
 
if __name__ == '__main__':

    ## TESTING PERMUTATIONS AND NEGATIONS on an irreducible parameter
    def permute(permutation, w):
        return {
            'a': w['a'][permutation],
            'b': w['b'][permutation],
            'c': w['c'][permutation],
            'd': w['d'],
        }
    def negate(sign_vector, w):
        return {
            'a': w['a'] * sign_vector,
            'b': w['b'] * sign_vector,
            'c': w['c'] * sign_vector,
            'd': w['d'],
        }
    h = 5
    w = {
            'a': np.array([1,2,3,4,5]),
            'b': np.array([1,2,3,4,5]),
            'c': np.array([1,2,3,4,5]),
            'd': 0.0,
        }
    permutations = [
        np.array([0,1,3,2,4]),
        np.array([1,2,3,4,0]),
        np.array([4,3,2,1,0]),
        np.array([0,3,2,1,4]),
        np.array([0,2,1,4,3]),
    ]
    sign_vectors = [
        np.array([-1,-1,-1,-1,-1]),
        np.array([-1, 1, 1,-1, 1]),
        np.array([ 1, 1,-1, 1,-1]),
        np.array([-1, 1, 1,-1, 1]),
        np.array([ 1,-1,-1, 1,-1]),
    ]
    def equal(u, w):
        a = np.all(u['a'] == w['a'])
        b = np.all(u['b'] == w['b'])
        c = np.all(u['c'] == w['c'])
        d = u['d'] == w['d']
        return (a and b and c and d)

    for p in permutations:
        for s in sign_vectors:
            w1 = negate(s, permute(p, w))
            u = canonicalize(w1)
            print(equal(u, w))

    ## TESTING REDUCIBILITY

    ws = [
        {
            'a': np.array([1,2,0]),
            'b': np.array([1,2,0]),
            'c': np.array([1,2,0]),
            'd': 0.0,
        },
        {
            'a': np.array([1,2,0]),
            'b': np.array([1,2,1]),
            'c': np.array([1,2,0]),
            'd': 0.0,
        },
        {
            'a': np.array([1,2,3]),
            'b': np.array([1,2,0]),
            'c': np.array([1,2,0]),
            'd': 0.0,
        },
        {
            'a': np.array([1,1,1]),
            'b': np.array([1,2,2]),
            'c': np.array([1,2,2]),
            'd': 0.0,
        },
        {
            'a': np.array([1,-1,1]),
            'b': np.array([1,-2,2]),
            'c': np.array([1,-2,2]),
            'd': 0.0,
        },
        {
            'a': np.array([1,1,1]),
            'b': np.array([2,1,2]),
            'c': np.array([2,1,2]),
            'd': 0.0,
        },
        {
            'a': np.array([-1,1,1]),
            'b': np.array([-2,2,1]),
            'c': np.array([-2,2,1]),
            'd': 0.0,
        },
    ]
    w0 = canonicalize(ws[0])
    for w in ws[1:]:
        print(equal(w0, canonicalize(w)))

    ## TESTING REDUCIBILITY

    ws = [
        {
            'a': np.array([1,2,3,0,0,0]),
            'b': np.array([1,2,3,0,0,0]),
            'c': np.array([1,2,3,0,0,0]),
            'd': 0.0,
        },
        {
            'a': np.array([1, 2, 3, 1,-1, 0]),
            'b': np.array([1, 2, 3, 0, 0, 0]),
            'c': np.array([1, 2, 3,-4,+4, 0]),
            'd': 0.0,
        },
        {
            'a': np.array([0, 0,-1, 2, 3,+2]),
            'b': np.array([1, 2, 1, 2, 3, 1]),
            'c': np.array([1, 2, 1, 2, 3, 1]),
            'd': 0.0,
        },
    ]
    w0 = canonicalize(ws[0])
    for w in ws[1:]:
        print(equal(w0, canonicalize(w)))

