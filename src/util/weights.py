import numpy as np
import time
import math


class Weights:
    def layer(self, in_size, out_size, unbaised=False):
        pass

    def single(self, size, unbaised=False):
        pass


class Zeros(Weights):
    '''
    Initialize the weight vector with zeros
    '''
    def layer(self, in_size, out_size, unbaised=False):
        return np.zeros((in_size, out_size))

    def single(self, size, unbaised=False):
        return np.zeros(size)


class Uniform(Weights):
    '''
    Initialize the weight vector with values from a Uniform distribution U[-0.1,0.1]
    '''
    def single(self, size, unbaised=False):
        rns = np.random.RandomState(int(time.time()))
        result = rns.uniform(size=size) / 5.0 - 0.1
        if unbaised:
            result[0] = 0.0
        return result

    def layer(self, in_size, out_size, unbaised=False):
        rns = np.random.RandomState(int(time.time()))
        result = rns.uniform(size=(in_size, out_size)) / 5.0 - 0.1
        if unbaised:
            for k in range(in_size):
                result[k][0] = 0.0
        return result


class Xavier(Weights):
    '''
    Initialize the weight vector from a Normal distribution according ot Xavier:
    http://philipperemy.github.io/xavier-initialization/
    '''
    def single(self, size, unbaised=False):
        rns = np.random.RandomState(int(time.time()))
        result = rns.normal(0, 1.0/math.sqrt(size), size)
        if unbaised:
            result[0] = 0.0
        return result

    def layer(self, in_size, out_size, unbaised=False):
        rns = np.random.RandomState(int(time.time()))
        result = rns.normal(0, 2.0 / math.sqrt(in_size + out_size), (in_size, out_size))
        if unbaised:
            for k in range(in_size):
                result[k][0] = 0.0
        return result


class WeightFactory:
    metods = {
        "zeros": Zeros,
        "uniform": Uniform,
        "xavier": Xavier
    }

    @staticmethod
    def build(name):
        if name not in WeightFactory.metods:
            raise ValueError('There is no predefined weight builder named: ' + name)
        return WeightFactory.metods[name]()