import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, num_inputs: np.array, alpha: float) -> None:
        self.num_inputs = num_inputs
        self.weights = np.random.rand(num_inputs + 1)
        self.alfa = alpha

    def activation_func(self, input: float) -> int:
        return 1 if input > 0 else -1

    def result(self, input: np.array) -> np.array:
        return np.array(list(map(self.activation_func, input @ self.weights)))
    
    def update_weights(self, sigma:np.array, input: np.array) -> None:
        self.weights = self.weights + self.alfa * ((input.T) @ -(self.result(input) - sigma))

    def train(self, input: np.array, sigma: np.array) -> None:
        while (self.result(input) != sigma).any():
            self.update_weights(sigma, input)


num_inputs = 10
dimension_inputs = 2
X = np.random.rand(dimension_inputs + 1, num_inputs) * -2
X[0] = 1
X = X.T
X = np.array([[1.0, 0, 1],
             [1, 0, 2], 
             [1, 1, 0], 
             [1, 1, 1],
             [1, 2, 1],
             [1, 2, 2],
             [1, 11, 11], 
             [1, 12, 12],
             [1, 11, 12],
             [1, 12, 11]])
NN = NeuralNetwork((X)[0].size - 1, 0.001)
sigma = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
print("Before train:    ", NN.result(X)) 
NN.train(X, sigma)
print("New weights:     ", NN.weights)
print("After train:     ", NN.result(X))


a, b, k = np.random.rand(1), np.random.rand(1), -np.random.rand(1)
for i in range(num_inputs):
    q = np.random.rand(dimension_inputs + 1)
    q[0] = 1
    X[i] = q
    sigma[i] = 1 if q @ np.array([k, a, b]) > 0 else -1

print("\n", sigma, a, b, k)
print("Before train:    ", NN.result(X)) 
NN.train(X, sigma)
print("New weights:     ", NN.weights)
print("After train:     ", NN.result(X))

rows1 = np.where(sigma == 1)
rows2 = np.where(sigma == -1)
X1 = X[rows1, :]
X2 = X[rows2, :]
x = np.array([0, 0.4, 0.7, 1.0])
y = -(NN.weights[1]) * x - NN.weights[0]
y = y / NN.weights[2]
plt.plot(x, y)
plt.plot(X1.T[1], X1.T[2], 'ro', X2.T[1], X2.T[2], 'b^')
plt.show()
