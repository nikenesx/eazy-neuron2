from optparse import Option

import numpy as np

from activation_functions import ACTIVATION_FUNCTIONS_DICT


class Neuron:
    def __int__(self, weights: Option[np.array] = None, activation_func: Option[str] = None):
        self.weights = weights
        self.activation_function = activation_func
        self.local_gradient = 0
        self.sum_local_gradients = 0
        self.input_value = 0
        self.output_value = 0

    def set_output(self, inputs: np.array):
        self.input_value = np.dot(self.weights, inputs)
        self.output_value = ACTIVATION_FUNCTIONS_DICT[self.activation_function](self.input_value)
