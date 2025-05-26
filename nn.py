import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, input_size, hidden_architecture, output_size):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.output_size = output_size
        # Build a simple Sequential model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))
        for units in hidden_architecture:
            self.model.add(tf.keras.layers.Dense(units, activation='sigmoid'))
        # Output layer without activation (we’ll apply softmax ourselves)
        self.model.add(tf.keras.layers.Dense(output_size, activation=None))

    def compute_num_weights(self):
        # Sum total trainable parameters
        return int(sum(tf.size(var).numpy() for var in self.model.trainable_variables))

    def load_weights(self, weights):
        # Load a flat list of weights into the model variables
        w = tf.convert_to_tensor(weights, dtype=tf.float32)
        expected = self.compute_num_weights()
        # Strip off fitness+generation if present
        if w.shape[0] == expected + 2:
            w = w[2:]
        assert w.shape[0] == expected, f"Expected {expected} weights, got {w.shape[0]}"

        start = 0
        for var in self.model.trainable_variables:
            shape = var.shape
            size = tf.reduce_prod(shape)
            new_vals = tf.reshape(w[start:start + size], shape)
            var.assign(new_vals)
            start += size

    def forward(self, x):
        # Run a forward pass and sample action via softmax
        x_input = tf.convert_to_tensor([x], dtype=tf.float32)
        logits = self.model(x_input)[0]
        probs = tf.nn.softmax(logits).numpy()
        idx = np.random.choice(len(probs), p=probs)
        return [-1, 0, 1][idx]


def create_network_architecture(input_size,
                                hidden_architecture=(50,),
                                output_size=3):
    """
    Factory to create a TensorFlow-based network with:
    - `hidden_architecture`: tuple of hidden layer sizes
    - `output_size`: number of actions (3)
    """
    return NeuralNetwork(input_size, hidden_architecture, output_size)
