"""
Activation functions which can be used within neurons.
"""

import numpy as np


class Activation:
    """
    Containing various activation functions and their derivatives
    """
    name = "abstract"

    def calc(self, netOutput):
        pass

    def derivative(self, netOutput):
        pass


class SignActivation(Activation):
    def __init__(self, threshold=0.0):
        self.name = "sign"
        self.threshold = threshold

    def calc(self, netOutput):
        result = np.zeros(netOutput.shape)
        for k, n in enumerate(netOutput):
            result[k] = 1.0 if netOutput >= 0.0 else -1.0
        return result

    def derivative(self, netOutput):
        return np.zeros(netOutput.size)


class SigmoidActivation(Activation):
    name = "sigmoid"

    def calc(self, netOutput):
        return 1 / (1 + np.exp(-netOutput))

    def derivative(self, netOutput):
        return netOutput * (1.0 - netOutput)


class TanhActivation(Activation):
    name = "tanh"

    def calc(self, netOutput):
        ex = np.exp(1.0 * netOutput)
        exn = np.exp(-1.0 * netOutput)
        return np.divide(ex - exn, ex + exn)

    def derivative(self, netOutput):
        return (1 - self.calc(netOutput) ** 2)


class SinActivation(Activation):
    name = "sin"

    def calc(self, netOutput):
        return np.sin(netOutput / (2*np.pi))

    def derivative(self, netOutput):
        return (0.5 * np.pi) * np.cos( netOutput / (2*np.pi))


class ReluActivation(Activation):
    name = "relu"

    def calc(self, netOutput):
        return np.asarray([max(0.0, i) for i in netOutput])

    def derivative(self, netOutput):
        if netOutput > 0:
            return np.ones(netOutput.size)
        return np.zeros(netOutput.size)


class LinearActivation(Activation):
    name = "linear"

    def calc(self, netOutput):
        return netOutput

    def derivative(self, netOutput):
        return np.ones(netOutput.size)


class SoftmaxActivation(Activation):
    name = "softmax"

    def calc(self, netOutput):
        exps = np.exp(netOutput - netOutput.max())
        return exps / np.sum(exps, axis=0)

    def derivative(self, netOutput):
        return netOutput * (1- netOutput)


class ActivationFactory:
    methods = {
        "sigmoid": SigmoidActivation,
        "softmax": SoftmaxActivation,
        "tanh": TanhActivation,
        "sin": SinActivation,
        "relu": ReluActivation,
        "linear": LinearActivation,
        "sign": SignActivation
    }

    @staticmethod
    def build(name):
        """
        Returns the activation function corresponding to the given string
        """
        if name not in ActivationFactory.methods:
            raise ValueError('Unknown activation function: ' + name)
        return ActivationFactory.methods[name]()