# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import util.output


class Evaluator:
    '''
    Print performance of a classification model over a dataset
    '''

    def printTestLabel(self, testSet):
        # print all test labels
        for label in testSet.label:
            util.output.output(
                label,
                util.output.WARNING
            )

    def printResultedLabel(self, pred):
        # print all test labels
        for result in pred:
            util.output.output(
                result,
                util.output.WARNING
            )

    def printComparison(self, testSet, pred):
        for label, result in zip(testSet.label, pred):
            util.output.output(
                "Label: %r. Prediction: %r" % (bool(label), bool(result)),
                util.output.WARNING
            )

    def printClassificationResult(self, testSet, pred, targetNames):
        util.output.output(
            classification_report(testSet.label, pred, target_names=targetNames),
            util.output.WARNING
        )

    def printConfusionMatrix(self, testSet, pred):
        util.output.output(
            confusion_matrix(testSet.label, pred),
            util.output.WARNING
        )

    def printAccuracy(self, testSet, pred):
        util.output.output(
            "Accuracy of the recognizer: %.2f%%" %
                (accuracy_score(testSet.label, pred)*100),
            util.output.WARNING
        )
