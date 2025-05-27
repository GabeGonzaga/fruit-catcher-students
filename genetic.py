import random
import numpy as np

def create_individual(individual_size):
    # Creates a new individual with random weights in [-1, 1]
    return np.random.uniform(-1, 1, size=individual_size).tolist()

def generate_population(individual_size, population_size):
    # Generates initial population
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.15,
    mutation_rate=0.1
):
    """
    Evolves neural network weights using genetic algorithm
    
    Args:
        individual_size: Number of weights in neural network
        population_size: Size of population
        fitness_function: Function to evaluate individuals (accepts individual, seed)
        target_fitness: Early stopping threshold
        generations: Max generations to run (stopping threshold)
        elite_rate: Fraction of top individuals to preserve
        mutation_rate: Probability of mutating each gene
        
    Returns:
        (best_individual, best_fitness): Tuple with best solution found
    """
    population = generate_population(individual_size, population_size)
    best_individual, best_fitness = None, -np.inf
    elite_size = int(population_size * elite_rate)

    for gen in range(generations):
        # evaluate all
        evaluated = [
            (ind, fitness_function(ind, seed))
            for seed, ind in enumerate(population)
        ]

        # track best & early stop
        top_ind, top_fit = max(evaluated, key=lambda x: x[1])
        if top_fit > best_fitness:
            best_individual, best_fitness = top_ind.copy(), top_fit
            if best_fitness >= target_fitness:
                return best_individual, best_fitness

        # sort descending
        evaluated.sort(key=lambda x: x[1], reverse=True)
        elites = [ind for ind, _ in evaluated[:elite_size]]

        # tournament selection for the rest
        parents = elites + [
            max(random.sample(evaluated, 3), key=lambda x: x[1])[0]
            for _ in range(population_size - elite_size)
        ]

        # uniform crossover among elites
        offspring = [
            [
                g1 if random.random() < 0.5 else g2
                for g1, g2 in zip(*random.sample(elites, 2))
            ]
            for _ in range(population_size - elite_size)
        ]

        # gaussian mutation + clip
        offspring = [
            [
                float(np.clip(g + random.gauss(0, 0.1), -1, 1))
                if random.random() < mutation_rate else g
                for g in child
            ]
            for child in offspring
        ]

        population = elites + offspring

        if gen % 10 == 0:
            print(f"Gen {gen}: Best {best_fitness:.2f}, Mutation {mutation_rate:.3f}")

    return [float(w) for w in best_individual], best_fitness