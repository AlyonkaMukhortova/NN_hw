import numpy as np

class Firstlayer:
	def __init__(self, ideal_x: np.array):
		self.weights = ideal_x/2
		
	def predict(self, x):
		return self.weights @ x
		
class Secondlayer:
	def __init__(self, epsilon, num_inputs, maxr):
		self.weights = np.empty(shape=(num_inputs, num_inputs))
		self.weights.fill(-epsilon)
		np.fill_diagonal(self.weights,1)
		self.maxr=maxr
		
	def predict(self, y):
		for i in range(self.maxr):
			y = self.weights @ y
			y[y < 0] = 0
			if np.count_nonzero(y>0) == 1:
				return np.nonzero(y>0)[0][0]	
		return -1
		
class NN:
		def __init__(self, ideal_x, epsilon, maxr):
			self.layer1 = Firstlayer(ideal_x)
			self.layer2 = Secondlayer(epsilon, len(ideal_x), maxr)
			
		def predict(self, input):
			return self.layer2.predict(self.layer1.predict(input))
		
		
ideal_x = np.array([[1, 1, 1, 1,
1, -1, -1, 1,
1, -1, -1, 1,
1, -1, -1, 1,
 1, 1, 1, 1],

[-1, -1, -1, 1,
-1, -1, 1, 1,
-1, 1, -1, 1,
-1, -1, -1, 1,
-1, -1, -1, 1],

[1, 1, 1, 1,
-1, -1, -1, 1,
-1, -1, 1, -1,
-1, 1, -1, 1,
1, 1, 1, 1],

[1, 1, 1, 1,
-1, -1, -1, 1,
-1, -1, 1, -1,
-1, -1, -1, 1,
1, 1, 1, 1],

[1, -1, -1, 1,
1, -1, -1, 1,
1, 1, 1, 1,
-1, -1, -1, 1,
-1, -1, -1, 1]])

my_pic = np.array([1, 1, 1, 1,
-1, -1, -1, 1,
1, 1, -1, -1,
-1, 1, 1, -1,
 1, 1, -1, 1])
 
NeurNet = NN(ideal_x, 0.01, 1000)
 
print(NeurNet.predict(my_pic))