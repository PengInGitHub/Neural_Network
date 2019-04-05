import numpy as np
# numpy is used for math computations
# np.exp(x): e^x
# np.dot(A,B): elementwise product: sum(A1*B1, A2*B2, ... An*Bn) 
# np.array(x): map x to np.array

#########################################
#   Step 1: Build Basic Module: Neuron   #
#########################################


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
#print(my_neuron.feedforward(x))


#########################################
#     Step 2: Build Neural Networks     #
#########################################

# Neural Networks are the connections of Neurons
# Basic Structure: Input Layer -> Hidden Layer -> Output Layer
# feedforwad: the process to pass the input forward to get the output
class OurNeuralNetworks:
	"""
	A Neural Network with following structures:
		- 2 inputs
		- 1 hidden layer with 2 neurons(h1, h2)
		- 1 output layer with 1 neuron (o1)
	Each neuron has same weights and bias
		- w = [0, 1]
		- b = 0
	"""
	def __init__(self):

		weights = np.array([0, 1])
		bias = 0

		self.h1 = Neuron(weights, bias)
		self.h2 = Neuron(weights, bias)
		self.o1 = Neuron(weights, bias)


	def feedforward(self, x):
		out_h1 = self.h1.feedforward(x)
		out_h2 = self.h2.feedforward(x)

		# the inputs for o1 are the outputs of h1 and h2
		out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

		return out_o1

n = OurNeuralNetworks()
x = np.array([2, 3])
#print(n.feedforward(x))


#########################################
#     Step 3: Train Neural Networks     #
#########################################
# training is optimization
# task: prediction of gender based on the data of weight and height

# evaluation: loss func: MSE: 1/n * sum((Y_true - Y_pred)^2)
# training: minimize the loss func

def mse(pred, true):
	return ((pred-true)**2).mean()

pred = np.array([0,0,0,0])
true = np.array([1,0,1,0])

print(mse(pred,true))

# optimization
# pred is calculated by the NN, which is decided by values of weight and bias
# 2 inputs, 2 hidden layers with 2 neurons, 1 ouput
# 9 parameters determin the pred: L(w1,w2,w3,w4,w5,w6,b1,b2,b3)

# optimize value of w1 to reduce Loss
# chain rule

# take Alice's instance as example
# ∂L / ∂w_1 = (∂L / ∂Pred) * (∂Pred / ∂w_1)
# L = (1-Pred)^2
# ∂L / ∂Pred = ∂(1-Pred)^2 / ∂Pred = -2*(1-Pred)
# Pred = o1 = f(w5*h1 + w6*h2 + b3)
# only h1 contains 



















