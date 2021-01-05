import numpy as np
from Factor import Factor, product_n, marginalization, reduction, normalization


class Network(object):
    def __init__(self, factors=[]):
        self.factors = factors

    def add_factor(self, factor: Factor):
        self.factors.append(factor)

    def joint_distribution(self):
        return product_n(self.factors)

    def marginal(self, query, induce, evidence):
        reduced_factors = [reduction(factor, evidence) for factor in self.factors]
        joint = product_n(reduced_factors)
        unnormalized = marginalization(joint, induce)
        normalized = normalization(unnormalized)
        return normalized


if __name__ == '__main__':
    # 1. factors
    var1 = np.array([1, 2])
    val1 = np.arange(6).reshape((2, 3))
    factor1 = Factor(var1, val1)
    var2 = np.array([3, 2])
    val2 = np.arange(6).reshape((2, 3))
    factor2 = Factor(var2, val2)
    var3 = np.array([4, 1, 3])
    val3 = np.arange(8).reshape((2, 2, 2))
    factor3 = Factor(var3, val3)

    # 2. two factor networks
    print('-' * 80, '\n2-factor network')
    network = Network([factor1, factor2])

    print('joint distribution:')
    print(network.joint_distribution())
    print('marginal P(X1 | X3 = 1):')
    print(network.marginal([1], [2], {3: 1}))

    # 3. two factor networks
    print('-' * 80, '\n3-factor network')
    network.add_factor(factor3)

    print('joint distribution:')
    print(network.joint_distribution())
    print('marginal P(X1 | X3 = 1, X4 = 0):')
    print(network.marginal([1], [2], {3: 1, 4: 0}))
