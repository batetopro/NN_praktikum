import numpy as np
from sklearn.metrics import accuracy_score

from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

from util.loss_functions import ErrorFactory
import util.output


class LogisticRegression(Classifier):
    """
    Multinomial Logistic Regression

    Attributes
    ----------
    performances: array of floats
    layer: LogisticLayer
    """

    def __init__(self, epochs=50, **kwargs):
        '''
        :param classes: positive int
        :param epochs: positive int
        :param kwargs: see the logistic layer constructor
        '''
        self.epochs = epochs

        kwargs["is_output"] = True
        self.kwargs = kwargs

        self.layer = None
        self.performances = []

    def train(self, training_set, validation_set):
        '''
        Train the Logistic Regression
        :param training_set: training DataSet
        :param validation_set: validation DataSet
        :return: void
        '''

        self.performances = []
        self.layer = LogisticLayer(
            training_set.input.shape[1],
            training_set.label[0].size,
            **self.kwargs
        )

        for epoch in range(self.epochs):
            util.output.output(
                "Training epoch {0}/{1}..".format(epoch + 1, self.epochs),
                util.output.DEBUG
            )

            self._train_one_epoch(training_set)

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

    def _train_one_epoch(self, training_set):
        '''
        Train one epoch, seeing all input instances
        :param training_set: training instances
        :return: void
        '''
        hits = 0
        for inp, label in zip(training_set.input, training_set.label):
            out = self.layer.forward(inp)
            self.layer.back_propagate(label)
            self.layer.update()
            if label.argmax() == out.argmax():
                hits += 1

        util.output.output("Hits: %d" % hits, util.output.DEBUG)
        # self.performances.append(float(hits) / trainingSet.input.__len__())

    def classify(self, test_instance):
        '''
        Classify a single instance.
        :param test_instance: list of floats
        :return: int: the predicted class
        '''
        return self.layer.forward(test_instance).argmax()

    def evaluate(self, test):
        '''
        Evaluate a whole dataset.
        :param test: the dataset to be classified
        :return: List of classified decisions for the dataset's entries.
        '''
        return list(map(self.classify, test))
