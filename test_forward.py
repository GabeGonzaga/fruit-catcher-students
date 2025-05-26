# test_forward.py

from nn import create_network_architecture

# Cria rede com hidden layer
nn = create_network_architecture(10, with_hidden_layer=True, hidden_neurons=5)

# Carrega raw e extrai só os pesos que a rede espera
with open("best_individual.txt") as f:
    raw = f.read().strip().split(",")

raw_floats = list(map(float, raw))
expected = nn.compute_num_weights()

if len(raw_floats) >= expected + 2:
    weights = raw_floats[2 : 2 + expected]
else:
    weights = raw_floats[:expected]

# Verificação opcional
assert len(weights) == expected, \
    f"Esperado {expected} pesos, recebido {len(weights)}"

nn.load_weights(weights)

# Estados de teste
example_states = [
    [0.0, 0.1, 0.9, 1,   0.2, 0.8, 1,   0.3, 0.7, 1],   # 3 frutas
    [0.0, 0.8, 0.2, -1,  0.1, 0.5, 1,   0.4, 0.6, 1],  # 1 bomba + 2 frutas
    [0.0, 0.5, 0.3, -1,  0.5, 0.5, -1,  0.5, 0.7, -1], # 3 bombas
    [0.0, 0.1, 0.9, 1,   0.2, 0.8, 1,   0.3, 0.7, -1],  # 2 frutas + 1 bomba
]

print("==== Teste à rede com hidden layer ====\n")
for i, state in enumerate(example_states):
    action = nn.forward(state)
    print(f"State {i+1}: {state}")
    print(f" -> Action: {action}\n")