# test that is from `Factor Tutorial.m`

import numpy as np
from Factor import Factor, product, product_n, marginalization, reduction, normalization
from Network import Network

if __name__ == '__main__':
    # factor 1 contains P(X_1)
    var1 = np.array([1])
    val1 = np.array([0.11, 0.89])
    factor1 = Factor(var1, val1)

    # factor 2 contains P(X_2 | X_1)
    # note that the input is the inverse of it in MATLAB
    var2 = np.array([1, 2])
    val2 = np.array([0.59, 0.41, 0.22, 0.78]).reshape((2, 2))
    factor2 = Factor(var2, val2)

    # factor 3 contains P(X_3 | X_2)
    var3 = np.array([2, 3])
    val3 = np.array([0.39, 0.61, 0.06, 0.94]).reshape((2, 2))
    factor3 = Factor(var3, val3)

    factors = [factor1, factor2, factor3]

    # 1. Factor Product
    print('-' * 80, '\n1. Factor Product')
    print(product_n([factor1, factor2]))

    # 2. Factor Marginalization
    print('-' * 80, '\n2. Factor Marginalization')
    print(marginalization(factor2, [2]))

    # 3. Observe Evidence
    print('-' * 80, '\n3. Observe Evidence')
    for factor in factors:
        print(reduction(factor, {2: 0, 3: 1}))

    # 4. Compute Joint Distribution
    network = Network(factors)
    print('-' * 80, '\n4. Compute Joint Distribution')
    print(network.joint_distribution())

    # 5. Compute Marginal
    print('-' * 80, '\n5. Compute Marginal')
    print(network.marginal([2, 3], [], {1: 1}))
