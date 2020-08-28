import numpy as np
from neural_network import NeuralNetwork

class SimpleNeuralNetwork(NeuralNetwork):

	def __init__(self, lr, eps, batch_size, scale):
		######################################################
		# Parameters whose value are used to control the 
		# learning process
		######################################################
		self.scale = scale
		self.lr = lr
		self.eps = eps
		self.batch_size = batch_size

	def normalize(self, data):
		return data / self.scale

	def sigmoid(self, z):
		value = 1. / (1. + np.exp(-z))

		return value

	def forward(self, data, parameters):
	    temp = {}
	    temp["Z1"] = np.matmul(parameters["W1"], data) + parameters["b1"]
	    temp["A1"] = self.sigmoid(temp["Z1"])
	    temp["Z2"] = np.matmul(parameters["W2"], temp["A1"]) + parameters["b2"]
	    ###########################################################
	    # Vectorizing by stacking examples side-by-side, so 
	    # that our input matrix data has an example in each column.
	    ###########################################################
	    temp["A2"] = np.exp(temp["Z2"]) / np.sum(np.exp(temp["Z2"]), axis=0)

	    return temp

	def backward(self, data, target, parameters, temp, m_batch):
	    dZ2 = temp["A2"] - target
	    dW2 = (1./m_batch) * np.matmul(dZ2, temp["A1"].T)
	    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

	    dA1 = np.matmul(parameters["W2"].T, dZ2)
	    dZ1 = dA1 * self.sigmoid(temp["Z1"]) * (1 - self.sigmoid(temp["Z1"]))
	    dW1 = (1./m_batch) * np.matmul(dZ1, data.T)
	    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)
	    gradient = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

	    return gradient

	# Using cross entropy cost function
	def cost(self, target, Y_h):
	    summation = np.sum(np.multiply(target, np.log(Y_h)))
	    temp = target.shape[1]
	    loss_value = -(1./temp) * summation

	    return loss_value 
	
	def train(self, X_train, X_test, Y_train, Y_test, epochs, batches, nrows):
		n_x = X_train.shape[0]
		n_h = 64
		######################################################
		# Parameters, temp(and temp_values) and gradient 
		# are for conveniently passing information back and 
		# forth between the forward and backward passes.
		######################################################

		#########################################################
		# Smarten up the initialization of our network’s weights 
		# by using np.random.randn(n_h, n_x) * np.sqrt(1. / n_x).
		#########################################################
		parameters = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
				"b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
				"W2": np.random.randn(nrows, n_h) * np.sqrt(1. / n_h),
				"b2": np.zeros((nrows, 1)) * np.sqrt(1. / n_h) } 

		temp_dW1 = np.zeros(parameters["W1"].shape)
		temp_db1 = np.zeros(parameters["b1"].shape)
		temp_dW2 = np.zeros(parameters["W2"].shape)
		temp_db2 = np.zeros(parameters["b2"].shape)

		train_cost_list = []
		test_cost_list = []
		# Training
		for i in range(epochs):

			permutation = np.random.permutation(X_train.shape[1])
			new_X_train = X_train[:, permutation]
			Y_train_shuffled = Y_train[:, permutation]

			for j in range(batches):

				begin = j * self.batch_size
				end = min(begin + self.batch_size, X_train.shape[1] - 1)
				X = new_X_train[:, begin:end]
				Y = Y_train_shuffled[:, begin:end]
				m_batch = end - begin

				temp_values = self.forward(X, parameters)
				gradient = self.backward(X, Y, parameters, temp_values, m_batch)
				######################################################
				# Improve the code quality by switching from batch 
				# gradient descent  to mini-batch gradient descent.
				######################################################
				temp_dW1 = (self.eps * temp_dW1 + (1. - self.eps) * gradient["dW1"])
				temp_db1 = (self.eps * temp_db1 + (1. - self.eps) * gradient["db1"])
				temp_dW2 = (self.eps * temp_dW2 + (1. - self.eps) * gradient["dW2"])
				temp_db2 = (self.eps * temp_db2 + (1. - self.eps) * gradient["db2"])
				######################################################
				# Improve the code quality by adding  momentum to our 
				# descent algorithm.
				######################################################
				parameters["W1"] = parameters["W1"] - self.lr * temp_dW1
				parameters["b1"] = parameters["b1"] - self.lr * temp_db1
				parameters["W2"] = parameters["W2"] - self.lr * temp_dW2
				parameters["b2"] = parameters["b2"] - self.lr * temp_db2

			temp_values = self.forward(X_train, parameters)
			# temp_values["A2"] ==> Y hat
			train_cost = self.cost(Y_train, temp_values["A2"])
			temp_values = self.forward(X_test, parameters)
			test_cost = self.cost(Y_test, temp_values["A2"])
			train_cost_list.append(train_cost)
			test_cost_list.append(test_cost)

		return parameters, train_cost_list , test_cost_list