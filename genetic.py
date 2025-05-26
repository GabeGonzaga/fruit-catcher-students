# genetic.py

import random
import inspect
import os

# Limites para mutação adaptativa
MAX_MUTATION_RATE = 1.0
MAX_MUTATION_STRENGTH = 2.0

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
        f.write(
            ",".join(
                [str(fitness), str(generation)] + [str(w) for w in individual]
            )
        )

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
    elite_rate=0.1,  # reduzi de 0.2 para 0.1
    base_mutation_rate=0.5,
    base_mutation_strength=1.0,
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
    # Detecta se fitness_function aceita seed para reprodutibilidade
    accepts_seed = len(inspect.signature(fitness_function).parameters) == 2

    # Tenta retomar do melhor salvo
    if os.path.exists(file_path):
        try:
            best_ind, best_fit, last_gen = load_best_individual(file_path, individual_size)
        except Exception:
            best_ind, best_fit, last_gen = None, float('-inf'), -1
    else:
        best_ind, best_fit, last_gen = None, float('-inf'), -1

    start_gen = last_gen + 1

    # População inicial (injetando melhor prévio se houver)
    if best_ind is not None:
        population = [best_ind[:]] + [create_individual(individual_size) for _ in range(population_size - 1)]
    else:
        population = [create_individual(individual_size) for _ in range(population_size)]

    stagnation_count = 0

    for gen in range(start_gen, start_gen + generations):
        scored = []

        # Avalia fitness uma vez por indivíduo
        for idx, ind in enumerate(population):
            fitness = fitness_function(ind, seed=idx) if accepts_seed else fitness_function(ind)
            scored.append((ind, fitness))

        # Ordena por fitness decrescente
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ind, top_fit = scored[0]

        # Atualiza melhor global
        if top_fit > best_fit:
            best_fit = top_fit
            best_ind = top_ind[:]
            save_best_individual(best_ind, best_fit, gen, file_path)
            stagnation_count = 0
            if verbose:
                print(f"Gen {gen}: New best fitness = {best_fit:.4f}")
        else:
            stagnation_count += 1

        # Estatísticas da geração
        avg_fit = sum(f for _, f in scored) / len(scored)
        if verbose:
            print(f"Gen {gen}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, Stag={stagnation_count}")

        # Parada antecipada
        if best_fit >= target_fitness:
            break

        # Ajuste adaptativo e clamp nos parâmetros de mutação
        mut_rate = base_mutation_rate * (1 + stagnation_count / 10)
        mut_rate = min(mut_rate, MAX_MUTATION_RATE)

        mut_strength = base_mutation_strength * (1 + stagnation_count / 10)
        mut_strength = min(mut_strength, MAX_MUTATION_STRENGTH)

        # Taxa de injeção aleatória crescente com estagnação
        rand_inject = min(0.5, 0.1 + stagnation_count / 50)

        # Constrói próxima geração com elitismo e crossover
        elite_n = max(1, int(elite_rate * population_size))
        elite = [ind for ind, _ in scored[:elite_n]]
        new_pop = [best_ind[:]]

        for _ in range(population_size - 1):
            if random.random() < rand_inject:
                new_pop.append(create_individual(individual_size))
            else:
                # Uniform crossover entre dois elites
                p1, p2 = random.sample(elite, 2)
                child = [random.choice([a, b]) for a, b in zip(p1, p2)]
                # Mutação gaussiana
                child = [
                    g + random.gauss(0, mut_strength) if random.random() < mut_rate else g
                    for g in child
                ]
                new_pop.append(child)

        population = new_pop

    return best_ind, best_fit