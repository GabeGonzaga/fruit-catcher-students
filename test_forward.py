from nn import create_network_architecture

# Estado de jogo simulado (10 valores):
# [basket_y, fruit1_x, fruit1_y, is_fruit1, fruit2_x, fruit2_y, is_fruit2, fruit3_x, fruit3_y, is_fruit3]
example_states = [
    [0.0, 0.1, 0.9, 1, 0.2, 0.8, 1, 0.3, 0.7, 1],   # 3 frutas
    [0.0, 0.8, 0.2, -1, 0.1, 0.5, 1, 0.4, 0.6, 1],  # 1 bomba + 2 frutas
    [0.0, 0.5, 0.3, -1, 0.5, 0.5, -1, 0.5, 0.7, -1],# 3 bombas
    [0.0, 0.1, 0.9, 1, 0.2, 0.8, 1, 0.3, 0.7, -1],  # 2 frutas + 1 bomba
]

# Cria rede com hidden layer
nn = create_network_architecture(10, with_hidden_layer=True, hidden_neurons=5)

# Carrega os 61 pesos diretamente (ignora fitness e geração)
with open("best_individual.txt") as f:
    weights = list(map(float, f.read().strip().split(",")))

# Verificação opcional
assert len(weights) == nn.compute_num_weights(), f"Esperado {nn.compute_num_weights()} pesos, recebido {len(weights)}"

nn.load_weights(weights)

# Testar os estados
print("==== Teste à rede com hidden layer ====\n")
for i, state in enumerate(example_states):
    action = nn.forward(state)
    print(f"State {i+1}: {state}")
    print(f" -> Action: {action}\n")