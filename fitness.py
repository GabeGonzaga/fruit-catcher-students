import numpy as np
from nn import create_network_architecture
from game import FruitCatcherEnv  # ajusta ao teu módulo / nome de classe

def fitness_function(individual, seed=None):
    # 1. Instancia a rede TF
    net = create_network_architecture(
        input_size=10,
        hidden_architecture=(50,),
        output_size=3
    )
    net.load_weights(individual)

    # 2. Cria o ambiente
    env = FruitCatcherEnv(seed=seed)
    total_shaped = 0.0
    episodes = 5

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # 3. Escolhe ação com o forward TF
            action = net.forward(state)
            next_state, _, done, info = env.step(action)

            # 4. Reward-shaping
            if info.get("caught_bomb", False):
                shaped = -50.0
            elif info.get("caught_fruit", False):
                shaped = +1.0
            else:
                shaped = -0.1

            total_shaped += shaped
            state = next_state

    return total_shaped / episodes