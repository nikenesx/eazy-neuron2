import numpy as np


def sigmoid(x):
    return np.divide(np.int8(1), (np.int8(1) + np.exp(-x)))


def relu(x):
    return x if x > 0 else 0


ACTIVATION_FUNCTIONS_DICT = {
    'sigmoid': sigmoid,
    'relu': relu
}
