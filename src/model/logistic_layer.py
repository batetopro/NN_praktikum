import random

import numpy as np

from util.activation_functions import ActivationFactory
from util.regularization import RegularizationFactory
from util.weights import WeightFactory
from util.loss_functions import ErrorFactory

defaults = {
    "is_output": False,
    "is_input": False,
    "activation": "softmax",
    "weights": "xavier",
    "learning_rate": 0.01,
    "learning_rate_update": "const",
    "regularization": "none",
    "noise": False,              # Add noise to learning rate
    "alpha": 0.01,               # L1 rate
    "beta": 0.001,               # L2 rate
    "nesterov": True,            # Nesterov's moment
    "momentum": 0.8,
    "dropout": 0.0,              # Not implemented
}

output = {
    "loss": "crossentropy",
    "activation": "softmax",
    "dropout": 0.0,
}

hidden = {
    "loss": "crossentropy",
    "activation": "softmax",
    "dropout": 0.5,
}


class LogisticLayer():
    def __init__(self, nIn, nOut, **kwargs):
        '''
        A layer of neural network

        :param nIn: int: number of units from the previous layer (or input data)
        :param nOut: int: number of units of the current layer (or output)
        :param kwargs: see default values above
        '''
        self.nIn = nIn
        self.nOut = nOut

        if "is_output" not in kwargs:
            kwargs["is_output"] = False

        if "is_input" not in kwargs:
            kwargs["is_input"] = False

        if kwargs["is_input"] or kwargs["is_output"]:
            kwargs["dropout"] = 0.0

        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]

        if kwargs["is_output"]:
            d = output
        else:
            d = hidden

        for k in d:
            if k not in kwargs:
                kwargs[k] = d[k]

        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_update = RegularizationFactory.build(kwargs["learning_rate_update"] + "_rate", self.learning_rate)

        self.activation = ActivationFactory.build(kwargs["activation"])

        # http://colinraffel.com/wiki/neural_network_hyperparameters
        # there was some discussion that the input layer should not have B weights
        self.weights = WeightFactory.build(kwargs["weights"]).layer(nIn + 1, nOut, kwargs["is_input"])
        self.loss = ErrorFactory.build(kwargs["loss"])
        self.is_output = kwargs["is_output"]

        self.regularization = RegularizationFactory.build(
            kwargs["regularization"],
            kwargs["alpha"],
            kwargs["beta"]
        )

        self.noise = kwargs["noise"]

        self.deltas = np.zeros(self.weights.shape)
        self.velocity = np.zeros(self.weights.shape)

        self.signal = None
        self.child = None
        self.inp = None
        self.out = None

        self.nesterov = kwargs["nesterov"]
        self.momentum = kwargs["momentum"]


    def set_child(self, child):
        self.child = child

    def forward(self, inp):
        '''
        Compute forward step over the input using its weights
        :param inp: ndarray: a numpy array (nIn, 1) containing the input of the layer
        :return: ndarray: a numpy array (nOut,1) containing the output of the layer
        '''
        self.inp = inp
        self.out = self.activation.calc(np.dot(inp, self.weights[1:]) + self.weights[0])
        return self.out

    def back_propagate(self, target):
        '''
        Compute the derivatives (backward)

        This implementation uses the second section of:
        https://www.ics.uci.edu/~pjsadows/notes.pdf

        :param target: ndarray: the layer desired target from the layer (Only for the bottom layer)
        :return: ndarray: a numpy array containing the partial derivatives on this layer
        '''

        deltas = np.zeros(self.weights.shape)
        self.signal = np.zeros(self.nIn)
        if self.is_output:
            dado = self.loss.calculateDerivative(target, self.out)
            for k in range(self.nIn + 1):
                if k == 0:
                    deltas[0] = dado
                else:
                    deltas[k] = self.inp[k-1] * dado
                    self.signal[k-1] = sum([dado[i] * self.weights[k][i] for i in range(self.nOut)])
        else:
            dado = self.activation.derivative(self.out)
            signal = self.child.signal

            for k in range(self.nIn + 1):
                if k == 0:
                    deltas[0] = dado
                else:
                    deltas[k] = self.inp[k - 1] * dado * signal
                    self.signal[k-1] = sum([signal[i] * dado[i] * self.weights[k][i] for i in range(self.nOut)])

        dlr = self.learning_rate_update(deltas, self.deltas)

        if self.noise:
            '''
            We add noise from N(lambda, 2 * lamda) according to this:
            http://www.cs.cmu.edu/~fanyang1/dl-noise.pdf
            '''
            deltas *= np.random.normal(dlr, 2 * dlr, deltas.shape)
        else:
            deltas *= dlr

        reg = dlr * self.regularization(self.weights)
        deltas += reg

        self.velocity = self.momentum * self.deltas
        self.deltas = deltas

    def update(self):
        self.weights += self.velocity
        self.weights -= self.deltas