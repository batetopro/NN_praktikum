"""
Loss functions.
"""

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np


class Error:
    """
    Abstract class of an Error
    """
    name = "abstract"
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass
        
    @abstractmethod
    def calculateDerivative(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    name = "absolute"

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return np.sum(np.abs(target - output))
        
    def calculateDerivative(self, target, output):
        return -1.0 * np.ones(target.size)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    name = "different"

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        # return np.sum(target - output)

        result = 0
        for t, o in zip(target, output):
            result += int(t != o)
        return result

    def calculateDerivative(self, target, output):
        return -1.0 * np.ones(target.size)


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    name = "mse"

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        n = np.asarray(target).size
        return (1.0/n) * np.sum((target - output)**2)
    
    def calculateDerivative(self, target, output):
        # MSEPrime = -n/2*(target - output)
        return (2.0/target.size) * (output - target)


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    name = "sse"

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        return 0.5*np.sum((target - output)**2)
        
    def calculateDerivative(self, target, output):
        # SSEPrime = -(target - output)
        return output - target


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    name = "bce"

    def calculateError(self, target, output):
        return np.sum(target*np.log(output) + (1-target)*np.log(1-output))

    def calculateDerivative(self, target, output):
        # BCEPrime = -target/output + (1-target)/(1-output)
        return output - target
 

class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    name = "crossentropy"

    def calculateError(self, target, output):
        return -np.sum(target*np.log(output))
        
    def calculateDerivative(self, target, output):
        return output - target


class ErrorFactory:
    methods = {
        "bce": BinaryCrossEntropyError,
        "crossentropy": CrossEntropyError,

        "sse": SumSquaredError,
        "mse": MeanSquaredError,

        "different": DifferentError,
        "absolute": AbsoluteError
    }

    @staticmethod
    def build(loss):
        if loss not in ErrorFactory.methods:
            raise ValueError('There is no predefined loss function named: ' + loss)
        return ErrorFactory.methods[loss]()
