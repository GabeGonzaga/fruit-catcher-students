import random
from nn import create_network_architecture

# Cria rede com hidden layer
nn = create_network_architecture(10, with_hidden_layer=True, hidden_neurons=5)

# Carrega pesos diretamente do ficheiro
with open("best_individual.txt") as f:
    weights = list(map(float, f.read().strip().split(",")))

nn.load_weights(weights)

# Geração de 1000 estados aleatórios válidos
left = 0
right = 0
for _ in range(1000):
    # Cada valor dentro de faixa plausível do jogo
    state = [
        random.uniform(0, 1),  # basket_y
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 1
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 2
        random.uniform(0, 1), random.uniform(0, 1), random.choice([-1, 1]),  # fruit 3
    ]
    action = nn.forward(state)
    if action == -1:
        left += 1
    else:
        right += 1

# Resultados
print("==== Estatísticas de Ação ====")
print(f"Total para a ESQUERDA (-1): {left} ({left/10:.1f}%)")
print(f"Total para a DIREITA  (+1): {right} ({right/10:.1f}%)")