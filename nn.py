import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, output_size,
                 hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        """
        Calcula o número total de pesos e biases na rede, incluindo:
        - Pesos e biases de cada camada oculta
        - Pesos e biases da camada de saída
        """
        total = 0
        prev_size = self.input_size

        for layer_size in self.hidden_architecture:
            total += (prev_size + 1) * layer_size  # pesos + bias
            prev_size = layer_size

        # Camada de saída: pesos + biases
        total += (prev_size + 1) * self.output_size
        return total

    def load_weights(self, weights):
        """
        Carrega os pesos a partir de uma lista/array linear, separando-os por camadas.
        """
        w = np.array(weights)
        expected = self.compute_num_weights()
        assert len(w) == expected, f"Esperado {expected} pesos, recebido {len(w)}"

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size

        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            layer_w = w[start_w:end_w].reshape((input_size + 1, n))
            self.hidden_biases.append(layer_w[0, :])
            self.hidden_weights.append(layer_w[1:, :])
            start_w = end_w
            input_size = n

        # Camada de saída
        end_w = start_w + (input_size + 1) * self.output_size
        output_w = w[start_w:end_w].reshape((input_size + 1, self.output_size))
        self.output_bias = output_w[0, :]
        self.output_weights = output_w[1:, :]

    def forward(self, x):
        a = np.array(x)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            a = self.hidden_activation(z)

        z_out = np.dot(a, self.output_weights) + self.output_bias
        out = self.output_activation(z_out).item()  # escalar

        return -1 if out < 0 else 1  # movimento à esquerda ou direita

def create_network_architecture(input_size, output_size=1,
                                with_hidden_layer=True, hidden_neurons=5):
    """
    Cria uma instância da rede com uma camada escondida por omissão.
    """
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid
    output_fn = np.tanh  # [-1, 1]

    hidden_arch = (hidden_neurons,) if with_hidden_layer else ()

    return NeuralNetwork(
        input_size=input_size,
        hidden_architecture=hidden_arch,
        output_size=output_size,
        hidden_activation=hidden_fn,
        output_activation=output_fn
    )