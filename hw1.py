import numpy as np
import matplotlib.pyplot as plt


def neuron(input, weight, bias, func):
    return func(np.dot(input, weight) + bias)


def neuron_network(weight, bias, func, num):
    n = np.array([0.0]*num)
    for i in range(num):
        neuron1 = neuron(np.array(i), np.array(weight[0]), np.array(bias[0]), func[0])
        neuron2 = neuron(np.array(i), np.array(weight[1]), np.array(bias[1]), func[1])
        neuron3 = neuron(np.array([neuron1, neuron2]), np.array(weight[2:]), np.array(bias[2]), func[2])
        print(neuron1, neuron2, neuron3)
        n[i] = neuron3
    plt.plot(np.arange(num), n)  
    plt.ylabel('some numbers')
    plt.show()


#cycle
#neuron_network([10, 2, -1, -2], [1, 1, 1], [np.sinc, np.sin, np.square], 100)

#convergence
# neuron_network([-1, -2, -3, 1], [1, 3, 1], [np.sinh, np.sinh, np.sinc], 50)

#convergence
# neuron_network([1, 2, 3, 11], [-6, -2, 1], [np.square, np.sqrt, np.sinc], 100)
# neuron_network([1, 2, 3, 11], [-6, -2, 1], [np.sin, np.sinc, np.square], 100)
# neuron_network([1, 2, 3, 11], [-6, -2, 1], [np.sin, lambda x: np.tan(x**2), lambda x: x], 100)


neuron_network([1, 2, 3, 11], [-6, -2, 1], [np.sin, lambda x: (x), lambda x: np.sqrt(x)], 100)
