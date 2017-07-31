import numpy as np

class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data, classes=10, oneHot=True):
        # The label of the digits is always the first fields
        # Doing normalization

        self.input = (1.0 * data[:, 1:])/255 - 0.5
        self.label = data[:, 0]
        self.oneHot = oneHot

        if oneHot:
            def encode(x):
                result = np.zeros(classes)
                result[x] = 1
                return result
            self.label = list(map(encode, self.label))

    def __iter__(self):
        return self.input.__iter__()
