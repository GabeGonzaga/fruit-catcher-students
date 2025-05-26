import random
import inspect
import os

# == Fitness functions ==

def fitness_with_edge_penalty(net, weights, sim, alpha=5.0):
    net.load_weights(weights)
    sim.reset()
    total_caught, edge_count = 0, 0
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

# == Selection methods ==

def tournament_selection(scored, k=3):
    contenders = random.sample(scored, k)
    return max(contenders, key=lambda x: x[1])[0]


def rank_selection(scored):
    N = len(scored)
    total_rank = N*(N+1)//2
    pick = random.uniform(0, total_rank)
    cumulative = 0
    for rank, (ind, _) in enumerate(scored, start=1):
        cumulative += (N - rank + 1)
        if pick <= cumulative:
            return ind
    return scored[-1][0]

# == Genetic algorithm core ==

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]


def save_best_individual(individual, fitness, generation, file_path="best_individual.txt"):
    with open(file_path, "w") as f:
        f.write(",".join([str(fitness), str(generation)] + [str(w) for w in individual]))


def load_best_individual(file_path, individual_size):
    with open(file_path) as f:
        items = f.read().strip().split(",")
    fitness, generation = float(items[0]), int(items[1])
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
    selection_method='elitism',  # 'elitism', 'tournament', 'rank', 'best_children'
    tournament_k=3,
    elite_rate=0.2,
    base_mutation_rate=0.1,
    base_mutation_strength=0.5,
    stagnation_threshold=150,
    file_path="best_individual.txt",
    verbose=True
):
    """
    GA com várias opções de seleção:
    - elitism (default): híbrido elitismo + injeção
    - tournament: torneio de tamanho k
    - rank: rank-based
    - best_children: mantém o 'best' e gera filhos apenas por mutação do best
    """
    sig = inspect.signature(fitness_function)
    accepts_seed = 'seed' in sig.parameters or len(sig.parameters) == 2

    if os.path.exists(file_path):
        try:
            best_ind, best_fit, last_gen = load_best_individual(file_path, individual_size)
        except:
            best_ind, best_fit, last_gen = None, float('-inf'), -1
    else:
        best_ind, best_fit, last_gen = None, float('-inf'), -1

    start_gen = last_gen + 1

    if best_ind is not None:
        population = [best_ind[:]] + [create_individual(individual_size) for _ in range(population_size-1)]
    else:
        population = [create_individual(individual_size) for _ in range(population_size)]

    stagnation = 0
    for gen in range(start_gen, start_gen + generations):
        scored = []
        for idx, ind in enumerate(population):
            fit = fitness_function(ind, seed=idx) if accepts_seed else fitness_function(ind)
            scored.append((ind, fit))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ind, top_fit = scored[0]

        if top_fit > best_fit:
            best_fit, best_ind = top_fit, top_ind[:]
            save_best_individual(best_ind, best_fit, gen, file_path)
            stagnation = 0
            if verbose:
                print(f"Gen {gen}: New best fitness = {best_fit:.4f}")
        else:
            stagnation += 1

        avg_fit = sum(f for _, f in scored) / population_size
        if verbose:
            print(f"Gen {gen}: Best={best_fit:.4f}, Avg={avg_fit:.4f}, Stag={stagnation}")
        if best_fit >= target_fitness:
            break

        if stagnation >= stagnation_threshold:
            if verbose:
                print(f"Stagnation {stagnation} ≥ {stagnation_threshold}, reinicializando...")
            population = [best_ind[:]] + [create_individual(individual_size) for _ in range(population_size-1)]
            stagnation = 0

        mut_rate = base_mutation_rate * (1 + stagnation/10)
        mut_strength = base_mutation_strength * (1 + stagnation/10)

        new_pop = [best_ind[:]]
        for _ in range(population_size - 1):
            # melhor e filhos do melhor
            if selection_method == 'best_children':
                child = [g + random.gauss(0, mut_strength) if random.random() < mut_rate else g
                         for g in best_ind]
                new_pop.append(child)
            elif random.random() < 0.1:
                new_pop.append(create_individual(individual_size))
            else:
                if selection_method == 'tournament':
                    p1 = tournament_selection(scored, tournament_k)
                    p2 = tournament_selection(scored, tournament_k)
                elif selection_method == 'rank':
                    p1 = rank_selection(scored)
                    p2 = rank_selection(scored)
                else:
                    elites = [ind for ind, _ in scored[:max(1, int(elite_rate*population_size))]]
                    p1, p2 = random.sample(elites, 2)
                child = [random.choice([a, b]) for a, b in zip(p1, p2)]
                child = [g + random.gauss(0, mut_strength) if random.random() < mut_rate else g
                         for g in child]
                new_pop.append(child)
        population = new_pop

    return best_ind, best_fit