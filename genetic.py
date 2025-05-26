import random
import inspect
import os

# == Fitness functions ==

def fitness_with_edge_penalty(net, weights, sim, alpha=5.0):
    """
    Fitness que penaliza o tempo que a cesta passa nos cantos.
    net: instância de NeuralNetwork
    weights: lista de pesos a atribuir à rede
    sim: objeto de simulação com métodos:
         - reset(): reinicia o ambiente
         - step(decision): aplica decisão e retorna (caught, basket_x)
         - total_steps: número total de passos da simulação
    alpha: penalidade por tempo nos cantos
    """
    net.load_weights(weights)
    sim.reset()
    total_caught = 0
    edge_count = 0
    for _ in range(sim.total_steps):
        state = sim.state
        decision = net.forward(state)
        caught, basket_x = sim.step(decision)
        total_caught += caught
        if basket_x < 0.33 or basket_x > 0.67:
            edge_count += 1
    penalty = alpha * (edge_count / sim.total_steps)
    return total_caught - penalty


def fitness_with_l2(net, weights, sim, lambda_l2=0.001):
    """
    Fitness que inclui regularização L2 nos pesos.
    lambda_l2: coeficiente de regularização
    """
    net.load_weights(weights)
    sim.reset()
    total_caught = 0
    for _ in range(sim.total_steps):
        state = sim.state
        decision = net.forward(state)
        caught, _ = sim.step(decision)
        total_caught += caught
    l2 = sum(w*w for w in weights)
    return total_caught - lambda_l2 * l2


# == Genetic algorithm core ==

def create_individual(individual_size):
    """
    Create an individual as a list of uniform random weights in [-1, 1].
    """
    return [random.uniform(-1, 1) for _ in range(individual_size)]


def save_best_individual(individual, fitness, generation, file_path="best_individual.txt"):
    """
    Save the best individual's fitness, generation, and weights to disk.
    """
    with open(file_path, "w") as f:
        f.write(",".join([str(fitness), str(generation)] + [str(w) for w in individual]))


def load_best_individual(file_path, individual_size):
    """
    Load the saved best individual's weights, fitness, and generation.
    Pads with zeros if fewer weights are stored.
    """
    with open(file_path) as f:
        items = f.read().strip().split(",")
    fitness = float(items[0])
    generation = int(items[1])
    weights = [float(x) for x in items[2:]]
    if len(weights) < individual_size:
        weights += [0.0] * (individual_size - len(weights))
    return weights[:individual_size], fitness, generation


def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.2,
    base_mutation_rate=0.1,
    base_mutation_strength=0.5,
    stagnation_threshold=15,
    file_path="best_individual.txt",
    verbose=True
):
    """
    Evolutionary optimization of neural network weights.

    - Resume from saved best individual if available.
    - Single fitness evaluation per individual.
    - Elitism selection, uniform crossover, Gaussian mutation.
    - Adaptive mutation rates based on stagnation.
    """
    # Determine if fitness_function accepts a seed parameter
    sig = inspect.signature(fitness_function)
    accepts_seed = 'seed' in sig.parameters or len(sig.parameters) == 2

    # Try loading previous best
    if os.path.exists(file_path):
        try:
            best_ind, best_fit, last_gen = load_best_individual(file_path, individual_size)
        except Exception:
            best_ind, best_fit, last_gen = None, float('-inf'), -1
    else:
        best_ind, best_fit, last_gen = None, float('-inf'), -1

    start_gen = last_gen + 1

    # Initialize population, injecting previous best if exists
    if best_ind is not None:
        population = [best_ind[:]] + [create_individual(individual_size) for _ in range(population_size - 1)]
    else:
        population = [create_individual(individual_size) for _ in range(population_size)]

    stagnation_count = 0

    for gen in range(start_gen, start_gen + generations):
        scored = []

        # Evaluate fitness once per individual
        for idx, ind in enumerate(population):
            if accepts_seed:
                fitness_value = fitness_function(ind, seed=idx)
            else:
                fitness_value = fitness_function(ind)
            scored.append((ind, fitness_value))

        # Sort by fitness descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ind, top_fit = scored[0]

        # Update global best if improved
        if top_fit > best_fit:
            best_fit = top_fit
            best_ind = top_ind[:]
            save_best_individual(best_ind, best_fit, gen, file_path)
            stagnation_count = 0
            if verbose:
                print(f"Gen {gen}: New best fitness = {best_fit:.4f}")
        else:
            stagnation_count += 1

        # Verbose generation summary
        avg_fit = sum(f for _, f in scored) / len(scored)
        if verbose:
            print(f"Gen {gen}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, Stag={stagnation_count}")

        # Stop early if target fitness reached
        if best_fit >= target_fitness:
            break

        # Adaptive mutation parameters
        mut_rate = base_mutation_rate * (1 + stagnation_count / 10)
        mut_strength = base_mutation_strength * (1 + stagnation_count / 10)

        # Build next generation
        elite_n = max(1, int(elite_rate * population_size))
        elite = [ind for ind, _ in scored[:elite_n]]
        new_pop = [best_ind[:]]

        for _ in range(population_size - 1):
            if random.random() < 0.25:
                # Random injection to maintain diversity
                new_pop.append(create_individual(individual_size))
            else:
                # Uniform crossover between two elites
                p1, p2 = random.sample(elite, 2)
                child = [random.choice([a, b]) for a, b in zip(p1, p2)]
                # Gaussian mutation
                child = [g + random.gauss(0, mut_strength) if random.random() < mut_rate else g for g in child]
                new_pop.append(child)

        population = new_pop

    return best_ind, best_fit