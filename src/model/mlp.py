
import numpy as np

import util.loss_functions
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

import util.output
from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    '''
    A multilayer perceptron used for classification
    '''
    def _validate(self, layers):
        if layers.__len__() < 1:
            raise Exception("NN needs at least one layer")

        for k, l in enumerate(layers):
            if "out" not in l:
                raise Exception("Every layer should have a set number of out nodes")

            if k == 0:
                if "in" not in l:
                    raise Exception("Input layer must have a set number of in nodes")
                l["is_input"] = True
            else:
                l["in"] = layers[k-1]["out"]

            l["is_output"] = k == layers.__len__() - 1

        return layers

    def __init__(self, layers=(), epochs=50):
        '''
        Multi-layer perceptron algorithm
        :param layers: list: NN architecture
        :param epochs: positive int
        '''
        self.epochs = epochs
        self.performances = []
        self.layers = []
        for k, args in enumerate(self._validate(layers)):
            L = LogisticLayer(args["in"], args["out"], **args)
            self.layers.append(L)

        for k in range(self.layers.__len__()-1):
            self.layers[k].set_child(self.layers[k+1])

    def forward(self, inp):
        '''
        Do feed forward through the layers of the network

        :param inp: nparray containing the input of the layer
        :return: nparray from the output
        '''
        current = inp
        for L in self.layers:
            current = L.forward(current)
            if L.is_output:
                return current


    ctr = 0
    def back_propagate(self, label):
        d = self.layers.__len__()
        for k in range(d):
            self.layers[d-k-1].back_propagate(label)

        #if self.ctr == 10:     exit()
        self.ctr += 1

        for l in self.layers:
            l.update()

    def train(self, training_set, validation_set):
        '''
        Train the Multi-layer Perceptrons
        :param training_set: training DataSet
        :param validation_set: validation DataSet
        :return: void
        '''
        self.performances = []

        for epoch in range(self.epochs):
            util.output.output(
                "Training epoch {0}/{1}..".format(epoch + 1, self.epochs),
                util.output.DEBUG
            )

            hits = 0
            for inp, label in zip(training_set.input, training_set.label):
                out = self.forward(inp)
                self.back_propagate(label)
                if label.argmax() == out.argmax():
                    hits += 1

            util.output.output("Hits: %d" % hits, util.output.DEBUG)
            # self.performances.append(float(hits) / trainingSet.input.__len__())

            accuracy = accuracy_score(validation_set.label, self.evaluate(validation_set))
            self.performances.append(accuracy)

            util.output.output(
                "Accuracy on validation: {0:.2f}%".format(accuracy * 100),
                util.output.DEBUG
            )

        accuracy = accuracy_score(validation_set.label, self.evaluate(validation_set))
        util.output.output(
            "Trained accuracy: {0:.2f}%".format(accuracy * 100),
            util.output.WARNING
        )

    def classify(self, test_instance):
        '''
        Classify an instance given the model of the classifier
        :param test_instance:
        :return: class
        '''
        return self.forward(test_instance).argmax()

    def evaluate(self, test):
        '''
        Evaluate a whole dataset.
        :param test: the dataset to be classified
        :return: list: List of classified decisions for the dataset's entries.
        '''
        return list(map(self.classify, test))