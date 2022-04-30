import numpy as np


def sigmoid(x):
    return np.divide(np.int8(1), (np.int8(1) + np.exp(-x)))


def sigmoid_diff(x):
    return sigmoid(x) * (np.int8(1) - sigmoid(x))


def relu(x):
    return x if x > 0 else 0


def relu_diff(x):
    return np.int8(1) if x > 0 else np.int8(0)


ACTIVATION_FUNCTIONS_DICT = {
    'sigmoid': sigmoid,
    'relu': relu,
}

ACTIVATION_FUNCTIONS_DIFF = {
    'sigmoid': sigmoid_diff,
    'relu': relu_diff,
}
