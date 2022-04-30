from typing import Optional

import numpy as np


LAMBDA = np.float16(0.01)


class Neuron:
    def __init__(self, weights: Optional[np.array] = None, activation_func: Optional[str] = None):
        self.weights = weights
        self.activation_function = activation_func
        self.activation_function_diff = None
        self.local_gradient = 0
        self.sum_local_gradients = 0
        self.input_value = 0
        self.output_value = 0

    def set_output(self, inputs: np.array):
        self.input_value = np.dot(self.weights, inputs)
        self.output_value = self.activation_function(self.input_value)

    def correct_weights(self, previous_layer_outputs: np.array):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - LAMBDA * self.sum_local_gradients * previous_layer_outputs[i]


class Bias:
    def __init__(self):
        self.output_value = np.int8(1)
        self.input_value = np.int8(1)
        self.weights = None
        self.local_gradient = 0
        self.sum_local_gradients = 0

    def set_output(self, inputs: np.array):
        pass

    def correct_weights(self, previous_layer_outputs: np.array):
        pass
