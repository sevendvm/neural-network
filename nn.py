import random
from functools import reduce
from math import exp


class NeuralNetwork:
	def __init__(self, inputs_count, hidden_count, outputs_count, activation_function, learning_rate):
		self.inputs_count = inputs_count
		self.hidden_count = hidden_count
		self.outputs_count = outputs_count
		
		self.activation_function = activation_function
		
		self.learning_rate = learning_rate
		
		self.weights_input_hidden = [[random.random() * 2 - 1 for j in range(self.inputs_count)] for i in
									 range(self.hidden_count)]
		
		self.weights_hidden_outputs = [[random.random() * 2 - 1 for j in range(self.hidden_count)] for i in
									   range(self.outputs_count)]
		
		self.input_bias = [random.random() for i in range(self.hidden_count)]
		self.hidden_bias = [random.random() for i in range(self.outputs_count)]
		
		self.input = [0 for i in range(self.inputs_count)]
		self.hidden = [0 for i in range(self.hidden_count)]
		self.output = [0 for i in range(self.outputs_count)]
	
	def feed_forward(self):
		# evaluate hidden layer
		for j in range(self.hidden_count):
			summ = 0
			for i in range(self.inputs_count):
				summ += self.input[i] * self.weights_input_hidden[j][i]
			self.hidden[j] = self.activation_function(summ + self.input_bias[j])
		
		# evaluate output
		for j in range(self.outputs_count):
			summ = 0
			for i in range(self.hidden_count):
				summ += self.hidden[i] * self.weights_hidden_outputs[j][i]
			self.output[j] = self.activation_function(summ + self.hidden_bias[j])
	
	def guess(self, inputs):
		self.input = [float(inputs[i]) for i in range(self.inputs_count)]
		self.feed_forward()
		return self.output
	
	def train(self, inputs, targets):
		# given an input
		# guess an answer
		guess = self.guess(inputs)
		
		# evaluate an error
		error_vector = [(float(targets[i]) - guess[i]) ** 2 / 2 for i in range(self.outputs_count)]
		
		# calculate a total error, which eventually
		# could have been a fitness measurement for
		# genetic learning algorithm
		total_error = reduce(lambda a, b: a + b, error_vector)
		
		# back propagate an error to correct weights
		if total_error > 0:
			# for each weight calculate the following expression:
			# dE/dw = dE/dout * dout/dnet * dnet/dw
			#  [[0 for j in range(self.hidden_count)] for i in range(self.outputs_count)]
			correction = [[0] * self.hidden_count] * self.outputs_count
			for j in range(self.outputs_count):
				for i in range(self.hidden_count):
					correction[j][i] = -(targets[j] - self.output[j]) * \
										self.output[j] * (1 - self.output[j]) * \
										self.hidden[i]
			
			# print("hidden to outputs correction")
			# print(correction)
			
			# perform an actual correction
			for i in range(self.outputs_count):
				for j in range(self.hidden_count):
					self.weights_hidden_outputs[i][j] = self.weights_hidden_outputs[i][j] - \
														self.learning_rate * correction[i][j]
			
			# calculate the correction matrix for inputs to hidden layer
			# correction = [[0 for i in range(self.inputs_count)] for j in range(self.hidden_count)]
			correction = [[0] * self.inputs_count] * self.hidden_count
			for j in range(self.hidden_count):
				for i in range(self.inputs_count):
					# TODO: implement a correction calculation
					correction[j][i] = 0  # -(targets[j] - self.output[j]) * \
										# self.output[j] * (1 - self.output[j]) * \
										# self.hidden[i]
			
			# perform an actual correction
			for i in range(self.hidden_count):
				for j in range(self.inputs_count):
					self.weights_input_hidden[i][j] = self.weights_input_hidden[i][j] - \
														self.learning_rate * correction[i][j]
		
		return error_vector, total_error


def sigmoid(x):
	return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
	return x * (1 - x)


def main():
	nn = NeuralNetwork(7, 12, 10, activation_function=sigmoid, learning_rate=1)
	# nn = NeuralNetwork(2, 3, 2)
	
	labeled_data = {0: [[1, 1, 1, 1, 1, 1, 0],  # stands for 0 in 7-segment display
						[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # zero output should be active
					1: [[0, 0, 1, 1, 0, 0, 0],
						[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
					2: [[0, 1, 1, 0, 1, 1, 1],
						[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
					3: [[0, 1, 1, 1, 1, 0, 1],
						[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
					4: [[1, 0, 1, 1, 0, 0, 1],
						[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
					5: [[1, 1, 0, 1, 1, 0, 1],
						[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
					6: [[1, 1, 0, 1, 1, 1, 1],
						[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
					7: [[1, 1, 1, 1, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
					8: [[1, 1, 1, 1, 1, 1, 1],
						[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
					9: [[1, 1, 1, 1, 1, 0, 1],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]}
	
	print(nn.train(labeled_data[0][0], labeled_data[0][1]))
	print(nn.input)
	print(nn.output)


if __name__ == '__main__':
	main()
