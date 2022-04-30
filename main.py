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
        OutputLayer(neurons_count=1, activation_function_name='relu'),
    ])
    network = Network(network_layers)

    network.configure()
    network.run(np.array([1, 1]))

    for i in range(len(network.layers)):
        print(f'Слой: {i + 1}, веса:')
        for n in network.layers[i].neurons:
            print(n.weights)
            print(n.output_value)
            print('----')




if __name__ == '__main__':
    main()
