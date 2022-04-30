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
    dataset = np.loadtxt("data.csv", delimiter=",")
    input_vectors = dataset[:, 0:8]
    result_vector = dataset[:, 8]

    network = Network(tuple([
        InputLayer(neurons_count=8),
        HiddenLayer(neurons_count=12, activation_function_name='relu'),
        OutputLayer(neurons_count=1, activation_function_name='sigmoid'),
    ]))

    network.configure()
    network.train(input_vectors=input_vectors, result_vectors=result_vector, batch_size=1, epochs=10, validation=0.2)


if __name__ == '__main__':
    main()
