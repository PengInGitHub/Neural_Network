import numpy as np
# numpy is used for math computations
# np.exp(x): e^x
# np.dot(A,B): elementwise product: sum(A1*B1, A2*B2, ... An*Bn) 
# np.array(x): map x to np.array
def sigmoid(x):
	# activation func
	# f(x) = 1 / (1+(e^(-x)))
	return 1 / (1+np.exp(-x))

class Neuron:
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias

	def feedforward(self, inputs):
		total = np.dot(self.weights, inputs) + bias
		return sigmoid(total)

weights = np.array([0,1])
bias = 4
my_neuron = Neuron(weights, bias)
x = np.array([2,3])
print(my_neuron.feedforward(x))