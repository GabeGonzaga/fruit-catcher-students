import random
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

nn.load_weights(weights)

# Geração de 1000 estados aleatórios válidos
actions_count = { -1: 0, 0: 0, 1: 0 }
for _ in range(1000):
    state = [
        random.uniform(0, 1),  # basket_y
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 1
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 2
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 3
    ]
    action = nn.forward(state)
    actions_count[action] += 1

# Resultados
print("==== Estatísticas de Ação ====")
print(f"Total para ESQUERDA (-1): {actions_count[-1]} ({actions_count[-1]/10:.1f}% do total)")
print(f"Total para PARADO   (0) : {actions_count[0]} ({actions_count[0]/10:.1f}% do total)")
print(f"Total para DIREITA  (+1): {actions_count[1]} ({actions_count[1]/10:.1f}% do total)")