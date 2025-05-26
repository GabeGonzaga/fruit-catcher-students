import random
import numpy as np

def create_individual(individual_size):
    #Creates a new individual with random weights in [-1, 1]
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    #Generates initial population
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.15, mutation_rate=0.1):
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
    # Initialize population and tracking variables
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = -float('inf')
    elite_size = int(population_size * elite_rate)
    prev_avg_fitness = None
    for generation in range(generations):
        # Evaluate fitness for all individuals (using their index as seed)
        evaluated_pop = []
        for seed, individual in enumerate(population):
            fitness = fitness_function(individual, seed)
            evaluated_pop.append((individual, fitness))
            
            # Track best individual
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                
                # Early stopping if target reached
                if best_fitness >= target_fitness:
                    return (best_individual, best_fitness)
        
        # Sort by fitness (descending)
        evaluated_pop.sort(key=lambda x: x[1], reverse=True)
        
        # Selection: Elitism + Tournament selection
        elites = [ind for ind, fit in evaluated_pop[:elite_size]]
        parents = elites.copy()
        
        # Fill remaining parents through tournament selection
        while len(parents) < population_size:
            # Tournament size of 3
            candidates = random.sample(evaluated_pop, 3)
            winner = max(candidates, key=lambda x: x[1])[0]
            parents.append(winner)
        
        # Crossover: Uniform crossover for remaining spots
        offspring = []
        for i in range(len(parents) - elite_size):
            parent1, parent2 = random.sample(elites, 2)  # Select from elites
            child = []
            for g1, g2 in zip(parent1, parent2):
                child.append(g1 if random.random() < 0.5 else g2)
            offspring.append(child)
            
       
        
        # Mutation: Gaussian noise
        for child in offspring:
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] += random.gauss(0, 0.1)
                    child[i] = np.clip(child[i], -1, 1)  # Keep in [-1, 1] range
        
        # Create new population
        population = elites + offspring
        
        # Progress tracking
        if generation % 10 == 0:
            # avgfitness = sum(fit for , fit in evaluated_pop) / population_size
            print(f"Gen {generation}: Best {best_fitness:.2f}, Mutation {mutation_rate:.3f}")
    
    return ([float(w) for w in best_individual], best_fitness)