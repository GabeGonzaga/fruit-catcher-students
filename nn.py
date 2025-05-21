import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        total = 0
        prev_size = self.input_size
        for layer_size in self.hidden_architecture:
            total += (prev_size + 1) * layer_size  # pesos + bias
            prev_size = layer_size
        total += prev_size + 1  # camada de saída: pesos + bias
        return total

    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w+1:]

    def forward(self, x):
        a = np.array(x)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            a = self.hidden_activation(z)
        output = np.dot(a, self.output_weights) + self.output_bias
        return self.output_activation(output)

def create_network_architecture(input_size, with_hidden_layer=False):
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # sigmoid
    output_fn = np.tanh  # output in [-1, 1]

    if with_hidden_layer:
        hidden_arch = (5,)  # 1 hidden layer with 5 neurons
    else:
        hidden_arch = ()  # perceptron (no hidden layers)

    return NeuralNetwork(input_size, hidden_arch, hidden_fn, output_fn)
