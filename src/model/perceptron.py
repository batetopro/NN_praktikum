import random

import numpy as np
from sklearn.metrics import accuracy_score

from util.activation_functions import ActivationFactory
from util.loss_functions import ErrorFactory
from util.weights import WeightFactory

from model.classifier import Classifier

import util.output


class Perceptron(Classifier):
    def __init__(self, target, learningRate=0.01, epochs=50, loss="different", activation="sign",  weights="xavier"):
        '''
        Perceptron
        :param target: string - target of single perceptron classification
        :param error: string - error method
        :param activation: string - update method
        :param learningRate: float - learning rate
        :param epochs: int - epochs
        :param loss: string - Loss function name
        :param activation: string - activation function name
        :param weights: string - weights function name
        '''

        self.target = target
        self.learningRate = learningRate
        self.epochs = epochs
        self.loss = ErrorFactory.build(loss)
        self.activation = ActivationFactory.build(activation)
        self.weights = WeightFactory.build(weights)
        self.weight = []


    def train(self, trainingSet, validationSet):
        '''
        Train the perceptron with the perceptron learning algorithm.
        :param trainingSet:
        :param validationSet:
        :return: void
        '''

        self.weight = self.weights.single(trainingSet.input.shape[1] + 1, unbaised=False)

        iteration = 0
        while True:
            totalError = 0
            for input, label in zip(trainingSet.input, trainingSet.label):
                output = self.fire(input, label)

                error = self.loss.calculateError(label, output)
                if np.array_equal(self.target, output):
                   error = -error

                self.updateWeights(input, error)
                totalError += int(error != 0.0)

            iteration += 1

            util.output.output("Epoch: %i; Error: %f" % (iteration, totalError), util.output.DEBUG)

            if totalError == 0 or iteration >= self.epochs:
                util.output.output("Trained: %i; Error: %f" % (iteration, totalError), util.output.INFO)
                break


    def classify(self, testInstance):
        '''
        Classify a single instance.
        :param testInstance:
        :return:
        '''
        output = np.dot(testInstance, self.weight[1:]) + self.weight[0]
        activation = self.activation.calc(np.ones(1) * output)
        if activation[0] > 0:
            return self.target.argmax()
        return -1

    def evaluate(self, test):
        '''
        Evaluate a whole data set.
        :param test:
        :return:
        '''
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        self.weight[0] += self.learningRate*error
        self.weight[1:] += self.learningRate*error*input

    def fire(self, input, target):
        '''
        Fire the output of the perceptron corresponding to the input.
        :param input:
        :return:
        '''

        output = np.dot(input, self.weight[1:]) + self.weight[0]
        activation = self.activation.calc(np.ones(1) * output)
        if activation[0] > 0:
            return self.target

        if np.array_str(target) == np.array_str(self.target):
            return (1 - self.target) / (self.target.size - 1)

        return target

    def __str__(self):
        return "P: " + self.target


class MulticlassPerceptron(Classifier):
    def encode(self, x):
        result = np.zeros(self.classes)
        result[x] = 1
        return result

    def __init__(self, classes=10, learningRate=0.01, epochs=50, loss="different", activation="sign", weights="xavier"):
        self.classes = classes
        self.P = [
            Perceptron(
                target=self.encode(k),
                learningRate=learningRate,
                epochs=epochs,
                weights=weights,
                loss=loss,
                activation=activation
            )
            for k in range(self.classes)
        ]

    def train(self, trainingSet, validationSet):
        '''
        Train the perceptron with the perceptron learning algorithm.
        :param trainingSet:
        :param validationSet:
        :return: void
        '''
        for p in self.P:
            util.output.output("Train binary perceptron: " + np.array2string(p.target), util.output.INFO)
            p.train(trainingSet, validationSet)

        accuracy = accuracy_score(validationSet.label, self.evaluate(validationSet))
        util.output.output("Validation accuracy: %f" % (accuracy), util.output.INFO)

    def classify(self, testInstance):
        '''
        Classify a single instance.
        :param testInstance:
        :return:
        '''
        result = []
        for p in self.P:
            output = p.classify(testInstance)
            if output >= 0:
                result.append(output)

        if result.__len__() > 0:
            return random.choice(result)
        else:
            return -1

    def evaluate(self, test):
        '''
        Evaluate a whole data set.
        :param test:
        :return:
        '''
        result = []
        for X, y in zip(test.input, test.label):
            c = self.classify(X)
            result.append(c)
        return result
