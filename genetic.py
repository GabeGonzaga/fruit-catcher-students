import random
import inspect
import os


def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]


def save_best_individual(individual, fitness, file_path="best_individual.txt"):
    with open(file_path, "w") as f:
        f.write(",".join([str(fitness)] + [str(w) for w in individual]))


def load_best_individual(file_path, individual_size):
    with open(file_path) as f:
        items = f.read().strip().split(",")
    fitness = float(items[0])
    weights = [float(x) for x in items[1:]]
    if len(weights) < individual_size:
        weights += [0.0] * (individual_size - len(weights))
    return weights[:individual_size], fitness


def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.2,
    base_mutation_rate=0.1,
    base_mutation_strength=0.5,
    eval_episodes=3,
    stagnation_threshold=15,
    file_path="best_individual.txt",
    verbose=True
):
    """
    Improved GA:
     - Average fitness over multiple episodes
     - Adaptive mutation on stagnation
     - Periodic random injection
    """
    accepts_seed = len(inspect.signature(fitness_function).parameters) == 2
    # Try load best
    if os.path.exists(file_path):
        try:
            best_ind, best_fit = load_best_individual(file_path, individual_size)
        except:
            best_ind, best_fit = None, float('-inf')
    else:
        best_ind, best_fit = None, float('-inf')

    # Initialize population
    if best_ind is not None:
        population = [best_ind[:]] + [create_individual(individual_size) for _ in range(population_size-1)]
    else:
        population = [create_individual(individual_size) for _ in range(population_size)]

    history = []
    stagnation_count = 0

    for gen in range(generations):
        scored = []
        # Evaluate each individual: average over episodes
        for idx, ind in enumerate(population):
            total = 0.0
            for ep in range(eval_episodes):
                seed = ep if accepts_seed else None
                if accepts_seed:
                    total += fitness_function(ind, seed=seed)
                else:
                    total += fitness_function(ind)
            fitness = total / eval_episodes
            scored.append((ind, fitness))

        # Sort and select
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ind, top_fit = scored[0]

        # Update global best
        if top_fit > best_fit:
            best_fit = top_fit
            best_ind = top_ind[:]
            save_best_individual(best_ind, best_fit, file_path)
            stagnation_count = 0
            if verbose:
                print(f"Generation {gen}: New best fitness = {best_fit:.4f}")
        else:
            stagnation_count += 1

        history.append(best_fit)
        avg_fit = sum(f for _, f in scored) / len(scored)
        if verbose:
            print(f"G{gen}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, Stag={stagnation_count}")

        if best_fit >= target_fitness:
            break

        # Adaptive mutation
        mut_rate = base_mutation_rate * (1 + stagnation_count/10)
        mut_strength = base_mutation_strength * (1 + stagnation_count/10)

        # Build next generation
        elite_n = max(1, int(elite_rate * population_size))
        elite = [ind for ind, _ in scored[:elite_n]]
        new_pop = [best_ind[:]]
        for _ in range(population_size-1):
            # random injection with probability
            if random.random() < 0.1:
                new_pop.append(create_individual(individual_size))
            else:
                p1, p2 = random.sample(elite, 2)
                # uniform crossover
                child = [random.choice([a, b]) for a, b in zip(p1, p2)]
                # gaussian mutation
                child = [gene + random.gauss(0, mut_strength) if random.random() < mut_rate else gene for gene in child]
                new_pop.append(child)
        population = new_pop

    return best_ind, best_fit