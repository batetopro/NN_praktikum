# -*- coding: utf-8 -*-

import random
from model.classifier import Classifier

__author__ = "ABC XYZ"  # Adjust this when you copy the file
__email__ = "ABC.XYZ@student.kit.edu"  # Adjust this when you copy the file


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.
    """

    def __init__(self, byChance=0.5):
        self.byChance = byChance

    def train(self, trainingSet, validationSet):
        # Do nothing
        pass

    def classify(self, testInstance):
        x = random.randint(0,9)
        return "0" * (10 - x - 1) + "1" + "0" * x

    def evaluate(self, test):
        return list(map(self.classify, test.input))
