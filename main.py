# Обязательно учесть при аписании сети:
# - Критейрий качества обучения сети
# - Выборка валидации для предотвращения переобучения сети
# - Вывод графиков
# - bias с весом
# - оптимизация adam

from layers import InputLayer, HiddenLayer, OutputLayer
from network import Network
import numpy as np


def main():
    network_layers = tuple([
        InputLayer(neurons_count=2),
        HiddenLayer(neurons_count=3, activation_function_name='relu'),
        OutputLayer(neurons_count=1, activation_function_name='relu'),
    ])
    network = Network(network_layers)

    network.configure()

    input_vectors = np.array([
        [1, 2],
        [1, 1],
        [1, 1],
    ])
    result_vectors = np.array([
        [2],
        [1],
        [1],
    ])

    network.train(input_vectors=input_vectors, result_vectors=input_vectors, batch_size=1, epochs=10, validation=0.7)


if __name__ == '__main__':
    main()
