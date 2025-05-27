import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sign(x):
    return np.sign(x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, output_size,
                 hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        # build arrays of in-/out-layer sizes, then sum all (in+1)*out
        ins = [self.input_size] + list(self.hidden_architecture)
        outs = list(self.hidden_architecture) + [self.output_size]
        return sum((i + 1) * o for i, o in zip(ins, outs))

    def load_weights(self, weights):
        w = np.array(weights, dtype=float)
        assert w.size == self.compute_num_weights(), \
               f"Esperado {self.compute_num_weights()} pesos, recebido {w.size}"

        # prepare shapes for hidden layers + output layer
        ins = [self.input_size] + list(self.hidden_architecture)
        outs = list(self.hidden_architecture) + [self.output_size]
        shapes = [(i + 1, o) for i, o in zip(ins, outs)]

        # split & reshape
        counts = [r * c for r, c in shapes]
        parts = np.split(w, np.cumsum(counts)[:-1])
        layers = [p.reshape(shape) for p, shape in zip(parts, shapes)]

        # unpack hidden vs output
        *hid, out = layers
        self.hidden_biases  = [L[0, :]   for L in hid]
        self.hidden_weights = [L[1:, :]  for L in hid]
        self.output_bias    = out[0, :]
        self.output_weights = out[1:, :]

    def forward(self, x):
        a = np.array(x, dtype=float)
        # hidden-layer propagation
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            a = self.hidden_activation(a.dot(W) + b)
        # output
        out = self.output_activation(a.dot(self.output_weights) + self.output_bias).item()
        # direct sign → decision
        return int(np.sign(out))


def create_network_architecture(
    input_size,
    output_size=1,
    with_hidden_layer=True,
    hidden_neurons=5,
    hidden_activation_type='relu'
):
    if hidden_activation_type.lower() == 'relu':
        hidden_fn = relu
    elif hidden_activation_type.lower() == 'sigmoid':
        hidden_fn = lambda x: 1 / (1 + np.exp(-x))
    elif hidden_activation_type.lower() == 'sign':
        hidden_fn = sign
    else:
        raise ValueError(f"Tipo de ativação desconhecido: {hidden_activation_type}")

    output_fn = np.tanh
    hidden_arch = (hidden_neurons,) if with_hidden_layer else ()

    return NeuralNetwork(
        input_size=input_size,
        hidden_architecture=(16, 8) + hidden_arch,
        output_size=output_size,
        hidden_activation=hidden_fn,
        output_activation=output_fn
    )