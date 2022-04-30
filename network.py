import numpy as np

from exceptions import FirstLayerError, LastLayerError, HiddenLayerError, IncorrectInputError
from layers import InputLayer, HiddenLayer, OutputLayer


class Network:
    def __init__(self, layers: tuple):
        self._check_layers(layers)
        self.layers = layers
        self.lambda_s = np.float16(0.01)

    def configure(self):
        for index in range(1, len(self.layers)):
            weights_count = self.layers[index - 1].neurons_count

            for neuron in self.layers[index].neurons:
                weights = [np.random.random() for _ in range(weights_count + 1)]
                neuron.weights = np.array(weights, dtype=np.float16)
                neuron.activation_function = self.layers[index].activation_function

    def run(self, input_vector: np.array):
        # check input vector len and first layer neurons count
        if len(input_vector) != self.layers[0].neurons_count:
            raise IncorrectInputError

        # set first layer inputs
        for index in range(self.layers[0].neurons_count):
            self.layers[0].neurons[index].output_value = input_vector[index]

        for index in range(1, len(self.layers)):
            for neuron in self.layers[index].neurons:
                neuron.set_output(self.layers[index - 1].get_neurons_outputs())


    @staticmethod
    def _check_layers(layers):
        """
        Проверяем, что первый слой это объект типа InputLayer, последний слой объект типа OutputLayer и скрытые слои
        объекты типа HiddenLayer.
        """
        if len(layers) < 2:
            raise '!!!!'

        if not isinstance(layers[0], InputLayer):
            raise FirstLayerError

        if not isinstance(layers[-1], OutputLayer):
            raise LastLayerError

        for layer in layers[1:-1]:
            if not isinstance(layer, HiddenLayer):
                raise HiddenLayerError