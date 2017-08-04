#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.stupid_recognizer import StupidRecognizer
from model.perceptron import MulticlassPerceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
import report.data_plot


import util.output
import util.activation_functions


def main():
    evaluator = Evaluator()
    sets = {
        "mnist": ["../data/mnist_seven.csv", 10, 3000, 1000, 1000, True],
        "iris": ["../data/iris.csv", 3, 3000, 1000, 1000, True],
        "four": ["../data/four.csv", 4, 800, 200, 200, True],
        "eight": ["../data/eight.csv", 8, 3600, 1000, 1000, True],
    }

    data = sets["mnist"]
    N = data[1]
    data = MNISTSeven(data[0], data[1], data[2], data[3], data[4], data[5])
    I = data.trainingSet.input.shape[1]

    # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
    HL = int((2 / 3) * (I + 1)) + N
    util.output.output("Training ...", util.output.WARNING)

    layers = [
        {
            "in": data.trainingSet.input[0].size,
            "out": HL,
            "regularization": "elastic",
            "alpha": 0.1,
            "beta": 0.001,
            "weights": "xavier",
            "momentum": 0.5,
            "noise": True,
        },
        {
            "out": N,
            "noise": True,
            "momentum": 0.5,
            "weights": "xavier",
        }
    ]

    MLP = MultilayerPerceptron(layers)
    MLP.train(data.trainingSet, data.validationSet)
    util.output.output("Multilayer perceptron trained", util.output.INFO)

    evaluator.printAccuracy(data.testSet, MLP.evaluate(data.testSet))
    plot = PerformancePlot("Multilayer perceptron validation")
    plot.draw_performance_epoch(MLP.performances, MLP.epochs)

    '''
    util.output.output("Logistic regression", util.output.INFO)
    LR = LogisticRegression(noise = False, momentum=0.8)
    LR.train(data.trainingSet, data.validationSet)
    util.output.output("Logistic regression trained", util.output.INFO)
    evaluator.printAccuracy(data.testSet, LR.evaluate(data.testSet))

    plot = PerformancePlot("Logistic Regression validation")
    plot.draw_performance_epoch(LR.performances, LR.epochs)
    '''
    '''
    for a in util.activation_functions.ActivationFactory.methods:
        util.output.output("Multi class perceptron", util.output.INFO)
        print(a)
        P = MulticlassPerceptron(classes=N, activation=a)
        P.train(data.trainingSet, data.validationSet)
        result = P.evaluate(data.testSet)
        util.output.output("Multi class perceptron trained", util.output.INFO)
        evaluator.printAccuracy(data.testSet, result)
    '''

    '''
    util.output.output("Stupid recognizer", util.output.INFO)
    stupid = StupidRecognizer()
    stupid.train(data.trainingSet, data.validationSet)
    result = stupid.evaluate(data.testSet)
    util.output.output("Stupid recognizer trained", util.output.INFO)
    evaluator.printAccuracy(data.testSet, result)
    '''


if __name__ == '__main__':
    util.output.CURRENT = util.output.DEBUG
    main()
