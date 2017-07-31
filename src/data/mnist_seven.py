# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet

import util.output


class MNISTSeven(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    dataPath : string
        Path to a CSV file with delimiter ',' and unint8 values.
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.
    oneHot: bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
        Set it to False for full MNIST task
    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """
    def __init__(self, dataPath,
                        classes = 10,
                        numTrain=3000,
                        numValid=1000,
                        numTest=1000,
                        oneHot=True):

        self.trainingSet = []
        self.validationSet = []
        self.testSet = []

        self.load(dataPath, classes, numTrain, numValid, numTest, oneHot)

    def load(self, dataPath, classes, numTrain, numValid, numTest, oneHot):
        '''
        Load the data.
        :param dataPath:
        :param numTrain:
        :param numValid:
        :param numTest:
        :param oneHot:
        :return:
        '''

        util.output.output("Loading data from " + dataPath + " ...", util.output.DEBUG)

        data = np.genfromtxt(dataPath, delimiter=",", dtype="uint8")

        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:numTrain+numValid], data[numTrain+numValid:]

        shuffle(train)

        train, valid = train[:numTrain], train[numTrain:]

        self.trainingSet = DataSet(train, classes, oneHot)
        self.validationSet = DataSet(valid, classes, False)
        self.testSet = DataSet(test, classes, False)

        util.output.output("Data loaded.", util.output.DEBUG)
        util.output.output("=========================", util.output.DEBUG)
