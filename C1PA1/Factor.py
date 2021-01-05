import numpy as np
from itertools import product as iproduct


class Factor(object):
    def __init__(self, var=None, val=None):
        assert len(var) == len(val.shape), 'Number of variables do not match in var and val'
        # small variable first
        indices = np.argsort(var)
        self.var = var[indices]
        self.val = np.transpose(val, indices)
        self.card = np.array(self.val.shape)

    def __str__(self):
        message = ''
        # header
        for var in self.var:
            message = message + 'v' + str(var) + '\t'
        message = message + 'value\n'

        # iterate each
        cards = [range(i) for i in self.card]
        for comb in iproduct(*cards):
            for item in comb:
                message = message + str(item) + '\t'
            message = message + str(self.val[comb]) + '\n'
        return message

    def marginalization(self):
        pass

    def reduction(self):
        pass

    def normalization(self):
        self.val = self.val / np.sum(self.val)


def merge_var(factor1: Factor, factor2: Factor):
    var1, var2 = factor1.var, factor2.var
    card1, card2 = factor1.card, factor2.card
    i, j = 0, 0
    var = []
    shape1, shape2 = [], []
    while i < len(var1) or j < len(var2):
        if i >= len(var1):
            next_var_from = '2'
        elif j >= len(var2):
            next_var_from = '1'
        elif var1[i] > var2[j]:
            next_var_from = '2'
        elif var1[i] < var2[j]:
            next_var_from = '1'
        else:  # var1[i] == var2[j]
            assert card1[i] == card2[j], 'cardinality of val %d does not match' % var1[i]
            next_var_from = 'both'

        if next_var_from == '1':
            var.append(var1[i])
            shape1.append(card1[i])
            shape2.append(1)
            i = i + 1
        elif next_var_from == '2':
            var.append(var2[j])
            shape1.append(1)
            shape2.append(card2[j])
            j = j + 1
        else:
            var.append(var1[i])
            shape1.append(card1[i])
            shape2.append(card2[j])
            i, j = i + 1, j + 1

    return var, shape1, shape2


def product(factor1: Factor, factor2: Factor) -> Factor:
    var, shape1, shape2 = merge_var(factor1, factor2)
    var = np.array(var)
    val1, val2 = np.reshape(factor1.val, shape1), np.reshape(factor2.val, shape2)
    val = np.multiply(val1, val2)
    return Factor(var, val)


def product_n(factors: list) -> Factor:
    assert len(factors) > 0, 'No factor to compute joint distribution!'
    joint = factors[0]
    for factor in factors[1:]:
        joint = product(joint, factor)
    return joint


def marginalization(factor: Factor, var_list) -> Factor:
    index_remain = []
    index_margin = []

    for i, v in enumerate(factor.var):
        if v in var_list:
            index_margin.append(i)
        else:
            index_remain.append(i)

    var = factor.var[index_remain]
    index = index_margin + index_remain
    val = np.transpose(factor.val, index)
    shape = factor.card[index_remain]
    shape = (-1,) + tuple(shape)
    val = np.reshape(val, shape)
    val = np.sum(val, axis=0)
    return Factor(var, val)


def reduction(factor: Factor, var_dict) -> Factor:
    index_remain = []
    index_reduce = []
    value_reduce = []

    for i, v in enumerate(factor.var):
        if v in var_dict:
            index_reduce.append(i)
            value_reduce.append(var_dict[v])
        else:
            index_remain.append(i)

    var = factor.var[index_remain]
    index = index_reduce + index_remain
    val = np.transpose(factor.val, index)
    value_reduce = tuple(value_reduce)  # use slice
    val = val[value_reduce]
    return Factor(var, val)


def normalization(factor: Factor) -> Factor:
    var = factor.var
    val = factor.val
    val = val / np.sum(val)
    return Factor(var, val)


if __name__ == '__main__':
    # 1. factor
    var1 = np.array([1, 2])
    val1 = np.arange(6).reshape((2, 3))
    factor1 = Factor(var1, val1)
    print('-' * 80, '\n1. factor')
    print(factor1)

    # 2. factor product
    var2 = np.array([3, 2])
    val2 = np.arange(6).reshape((2, 3))
    factor2 = Factor(var2, val2)
    factor3 = product(factor1, factor2)
    print('-' * 80, '\n2. factor product')
    print(factor3)

    # 3. factor marginalization
    factor4 = marginalization(factor3, [1, 3])
    print('-' * 80, '\n3. factor marginalization')
    print(factor4)

    # 4. factor reduction
    factor5 = reduction(factor3, {1: 0, 3: 1})
    print('-' * 80, '\n4. factor reduction')
    print(factor5)

    # 5. factor normalization
    factor6 = normalization(factor1)
    print('-' * 80, '\n5. factor normalization')
    print(factor6)
