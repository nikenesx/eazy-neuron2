import numpy as np

from exceptions import FirstLayerError, LastLayerError, HiddenLayerError, IncorrectInputError
from layers import InputLayer, HiddenLayer, OutputLayer


class Network:
    def __init__(self, layers: tuple):
        self._check_layers(layers)
        self.layers = layers

    def configure(self):
        for index in range(1, len(self.layers)):
            weights_count = self.layers[index - 1].neurons_count

            for neuron in self.layers[index].neurons:
                weights = [np.random.random() for _ in range(weights_count + 1)]
                neuron.weights = np.array(weights, dtype=np.float16)
                neuron.activation_function = self.layers[index].activation_function
                neuron.activation_function_diff = self.layers[index].activation_function_diff

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

        return self.layers[-1].get_neurons_outputs()

    def train(self, input_vectors: np.array, result_vectors: np.array, batch_size: int, epochs: int, validation: float):
        input_vectors_len = len(input_vectors)
        validation_len = int(input_vectors_len * validation)

        for epoch in range(epochs):
            # np.random.shuffle(input_vectors)

            vec_i = 0
            categorical_sum = 0
            for index in range(validation_len, len(input_vectors)):
                vec_i += 1
                vector = input_vectors[index]

                y = self.run(vector)
                y = y[:-1]
                d = result_vectors[index]
                e = y - d
                error = e[0]
                if index == len(input_vectors) - 1:
                    for ii in range(len(y)):
                        print(d)
                        print(y[ii])
                        categorical_sum += d * np.log(y[ii]) + (1 - d) * np.log(1 - y[ii])


                self.layers[-1].set_local_gradients(error)

                for i in range(1, len(self.layers) - 1):
                    for j in range(len(self.layers[i].neurons)):
                        next_layer_gradients_sum = sum(
                            [neuron.sum_local_gradients * neuron.weights[j] for neuron in self.layers[i + 1].neurons]
                        )
                        input_value = self.layers[i].neurons[j].input_value
                        gradient = next_layer_gradients_sum * self.layers[i].activation_function_diff(input_value)
                        self.layers[i].neurons[j].local_gradient = gradient
                        self.layers[i].neurons[j].sum_local_gradients += gradient

                if index % batch_size == 0 or index == len(input_vectors) - 1:
                    for layer_i in range(1, len(self.layers)):
                        previous_neurons_outputs = self.layers[layer_i - 1].get_neurons_outputs()
                        self.layers[layer_i].correct_neurons_weights(previous_neurons_outputs)

                    for layer in self.layers:
                        for neuron in layer.neurons:
                            neuron.sum_local_gradients = 0

                print(
                    '\repoch: {0}, completed: {1}/{2}, validation vectors count: {3}, loss: {4}'.format(
                        epoch, vec_i,
                        len(input_vectors) - validation_len,
                        validation_len,
                        categorical_sum * (-1) / (len(input_vectors) - validation_len)
                    ),
                    end='',
                )
            print()

    @staticmethod
    def _check_layers(layers):
        """
        ??????????????????, ?????? ???????????? ???????? ?????? ???????????? ???????? InputLayer, ?????????????????? ???????? ???????????? ???????? OutputLayer ?? ?????????????? ????????
        ?????????????? ???????? HiddenLayer.
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