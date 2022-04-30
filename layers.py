import numpy as np

from exceptions import UnknownActivationFunctionError
from neuron import Neuron, Bias
from activation_functions import ACTIVATION_FUNCTIONS_DICT, ACTIVATION_FUNCTIONS_DIFF


class Layer:
    def __init__(self, neurons_count: int):
        self.neurons_count = neurons_count
        neurons = [Neuron() for _ in range(neurons_count)]
        neurons.append(Bias())
        self.neurons = tuple(neurons)

    def get_neurons_outputs(self):
        return np.array([neuron.output_value for neuron in self.neurons])

    def correct_neurons_weights(self, previous_layer_outputs: np.array):
        for neuron in self.neurons:
            neuron.correct_weights(previous_layer_outputs)


class InputLayer(Layer):
    pass


class HiddenLayer(Layer):
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError

        self.activation_function_diff = ACTIVATION_FUNCTIONS_DIFF.get(activation_function_name)

    def set_local_gradients(self, gradients: int):
        pass


class OutputLayer(Layer):
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError

        self.activation_function_diff = ACTIVATION_FUNCTIONS_DIFF.get(activation_function_name)

    def set_local_gradients(self, error: int):
        for neuron in self.neurons:
            if isinstance(neuron, Bias):
                break
            neuron.local_gradient = neuron.activation_function_diff(neuron.input_value) * error
            neuron.sum_local_gradients += neuron.local_gradient

