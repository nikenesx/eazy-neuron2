import numpy as np

from exceptions import UnknownActivationFunctionError
from neuron import Neuron, Bias
from activation_functions import ACTIVATION_FUNCTIONS_DICT


class Layer:
    def __init__(self, neurons_count: int):
        self.neurons_count = neurons_count
        neurons = [Neuron() for _ in range(neurons_count)]
        neurons.append(Bias())
        self.neurons = tuple(neurons)

    def get_neurons_outputs(self):
        return np.array([neuron.output_value for neuron in self.neurons], dtype=np.float64)


class InputLayer(Layer):
    pass


class HiddenLayer(Layer):
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError


class OutputLayer(Layer):
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError
