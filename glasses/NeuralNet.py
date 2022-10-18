#how to do ANN

from tokenize import Double
import numpy
import scipy.special # for sigmoid
import matplotlib.pyplot

class NeuronalNet:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# input layer
		self.i_nodes = input_nodes
		# hidden layer
		self.h_nodes = hidden_nodes
		# output layer
		self.o_nodes = output_nodes
		self.learning_rate = learning_rate

		# weight matrices between layers
		self.w_input_hidden = numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5
		self.w_hidden_output = numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5

		self.activation_function = lambda x : scipy.special.expit(x)
		pass

	def train(self, input_list, target_list):
		# 1. modifying to 2-dim array for further calculations
		inputs = numpy.array(input_list, ndmin=2).T
		targets = numpy.array(target_list, ndmin=2).T
		
		# 2. send to next layer (matrix)
		hidden_inputs = numpy.dot(self.w_input_hidden, inputs)

		# 3. activation function (nodes)
		hidden_outputs = self.activation_function(hidden_inputs)

		# 4. send to next layer(matrix)
		final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)

		# 5. activation function (nodes)
		final_outputs = self.activation_function(final_inputs)

		# 6. find error
		output_errors = targets - final_outputs

		# 7. Error into hidden_output
		hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

		# 8. weight correction
		self.w_hidden_output += self.learning_rate * numpy.dot(output_errors * final_outputs * (1 - final_outputs), numpy.transpose(hidden_outputs))
		self.w_input_hidden += self.learning_rate * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), numpy.transpose(inputs))

		pass

	def query(self, input_list):
		#modify the input list to 2D array and transpose
		inputs = numpy.array(input_list, ndmin=2).T

		#Input -> Hidden
		#W_{ih} * I = X
		h_inputs = numpy.dot(self.w_input_hidden, inputs)

		#durch den Knoten durch O_H
		h_outputs = self.activation_function(h_inputs)

		#hidden -> output X_O
		final_inputs = numpy.dot(self.w_hidden_output, h_outputs)

		#durch die Knoten durch O_O
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
		pass

	def saveToFile(self, filename):
		#open file in binary write mode
		file = open(filename, 'bw')

		numpy.savez(file,
					input_nodes = self.i_nodes,
					hidden_nodes = self.h_nodes,
					output_nodes = self.o_nodes,
					input_hidden_weights = self.w_input_hidden,
					hidden_output_weights = self.w_hidden_output
		)
		pass

	def loadFromFile(self, filename):
		data = numpy.load(filename)

		self.i_nodes = data["input_nodes"]
		self.h_nodes = data["hidden_nodes"]
		self.o_nodes = data["output_nodes"]

		self.w_input_hidden = data["input_hidden_weights"]
		self.w_hidden_output = data["hidden_output_weights"]
		pass

	def debugNet(self):
		print("w_input_hidden")
		print(self.w_input_hidden)

		print("w_hidden_output")
		print(self.w_hidden_output)
		pass
