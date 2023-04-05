import numpy as np


class NeuralNetwork:
    def __init__(self, num_inputs: np.array, alpha: float, num_layers: int, num_layers_neuron: np.array) -> None:
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.weights = []
        self.err_func_deriv, self.act_func = lambda x: x, np.tanh
        self.act_func_deriv = lambda x: 1 - np.tanh(x) * np.tanh(x)
        for i in range(num_layers):
            self.weights.append(np.random.rand(num_layers_neuron[i], (num_inputs + 1) if i == 0 else (num_layers_neuron[i-1] + 1)))
        self.alfa = alpha

    def result(self, input: np.array) -> np.array:
        y = input
        for i in range(self.num_layers):
            y = np.append(np.array([np.ones(input.shape[1])]), np.array(list(map(self.act_func, self.weights[i] @ y))), axis=0)
        
        return np.delete(y, 0, 0)
    
    def partial_result(self, input: np.array):
        y = input
        Y, H = [], []
        for i in range(self.num_layers):
            if i != 0:
                H.append(self.weights[i] @ np.append(np.array([np.ones(input.shape[1])]),Y[i-1], axis = 0))
            else:
                H.append(self.weights[i] @ y)
            Y.append(self.act_func(H[i]))

        return Y, H
    
    def update_weights(self, sigma:np.array, input: np.array) -> None:
        Y, H = self.partial_result(input)
        delta = [0] * self.num_layers
        dEdW = [0] * self.num_layers
        for i in range(self.num_layers):
            delta[self.num_layers - i - 1] = ((Y[self.num_layers - i - 1] - sigma) if i == 0 
                                              else (np.delete(self.weights[self.num_layers - i].T, 0, 0) @ 
                                            delta[self.num_layers - i])) * self.act_func_deriv(H[self.num_layers - i - 1])
            ones = np.array([np.ones(input.shape[1])]).T
            dEdW[self.num_layers - i - 1] = delta[self.num_layers - i - 1] @ (np.append(ones, Y[self.num_layers - i - 2].T, 1) 
                                                                              if i != self.num_layers - 1 else input.T)
        self.weights = [self.weights[i] - self.alfa * dEdW[i] for i in range(self.num_layers)]
        

    def train(self, input: np.array, sigma: np.array, betta:float) -> None:
        while (self.result(input) - sigma >= betta).any():
            self.update_weights(sigma, input)
            print(self.result(input))


num_inputs = 3
num_layers_neuron = [1, 2, 3]
i = 2
X = np.array([[5, 6], [1, 2]]).T
X = np.append([[1, 1]], X, axis = 0)
print(X)
NN = NeuralNetwork(2, 0.1, 3, np.array([3, 2, 1]))

print(NN.weights)
print(NN.result(X))
sigma = np.array([[-0.5, 0.5]])
NN.train(X, sigma, 0.1)
print(NN.weights)
print(NN.result(X))