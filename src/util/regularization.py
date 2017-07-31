import numpy as np


class Regularization:
    def __init__(self, alpha=0.01, beta=0.001):
        self.alpha = alpha
        self.beta = beta
        self.init_hook()

    def init_hook(self):
        pass

    def __call__(self, weights):
        return np.zeros(weights.shape)


class L1(Regularization):
    def __call__(self, weights):
        return self.alpha * np.ones(weights.shape)


class L2(Regularization):
    def __call__(self, weights):
        return self.beta * weights


class Elastic(Regularization):
    def init_hook(self):
        self.L1 = L1(self.alpha)
        self.L2 = L1(self.beta)

    def __call__(self, weights):
        return self.L1(weights) + self.L2(weights)


class ConstantLearnRate(Regularization):
    def __call__(self, current, previous):
        return self.alpha


class CosineLearnRate(Regularization):
    def __call__(self, current, previous):
        a = np.sum(previous * current)
        p2 = np.sum(previous * previous)
        c2 = np.sum(current * current)

        if c2 * p2 > 0:
            cos = a / np.sqrt(c2 * p2)
            return self.alpha * (cos+1)/2

        return self.alpha


class NoisedRate(Regularization):
    '''
    We add noise from N(lambda, 2 * lamda) according to this:
    http://www.cs.cmu.edu/~fanyang1/dl-noise.pdf
    '''
    def __call__(self, deltas):
        return np.random.normal(self.alpha, 2 * self.alpha, deltas.shape) * deltas


class RegularizationFactory:
    methods = {
        "none": Regularization,
        "L1": L1,
        "L2": L2,
        "elastic": Elastic,
        "noise": NoisedRate,
        "const_rate": ConstantLearnRate,
        "cos_rate": CosineLearnRate
    }

    @staticmethod
    def build(name, alpha=0.01, beta=0.001):
        if name not in RegularizationFactory.methods:
            raise ValueError('There is no predefined regularization named: ' + name)

        return RegularizationFactory.methods[name](alpha, beta)