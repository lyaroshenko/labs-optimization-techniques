"""
LAB 4 – Genetic Algorithm for the 0/1 Knapsack Problem
"""

import random
import sys

POPULATION_SIZE = 100 # how many solutions we keep at once
NUM_GENERATIONS = 500 # how many rounds of evolution we do
CROSSOVER_RATE  = 0.85 # 85% chance two parents produce children
MUTATION_RATE = 0.02 # 2% chance any single gene randomly flips
TOURNAMENT_K = 3 # how many solutions compete in each selection
ELITE_COUNT = 2 # how many best solutions survive unchanged


#  STEP 1 — CREATE A RANDOM SOLUTION
def create_random_chromosome(num_items):
      #1 = "put this item in the knapsack"
      #0 = "leave this item out"
    chromosome = []
    for i in range(num_items):
        gene = random.randint(0, 1)
        chromosome.append(gene)

    return chromosome
#  STEP 2 — SCORE A SOLUTION (FITNESS)
def calculate_fitness(chromosome, items, capacity):
    #If the total weight is over the limit → fitness is 0 (illegal solution).
    #items is a list of (weight, benefit) pairs, e.g. [(4, 4), (7, 6), (5, 3)]
    total_weight = 0
    total_benefit = 0

    for i in range(len(chromosome)):
        if chromosome[i] == 1: # this item is selected
            total_weight += items[i][0] # add its weight
            total_benefit += items[i][1] # add its benefit

    if total_weight > capacity:
        return 0 # over the weight limit → worthless

    return total_benefit
#  STEP 3 — FIX AN OVERWEIGHT SOLUTION (REPAIR)
def repair_chromosome(chromosome, items, capacity):
    """
    sometimes after crossover or mutation a chromosome becomes overweight.
    instead of throwing it away we fix it by removing items one by one
    starting from the item with the WORST value-per-kilo ratio.
    if bag is 3kg over limit, drop the least efficient item first.
    """
    # work on a copy so we don't change the original
    fixed = chromosome[:]
    # calculate current total weight
    total_weight = 0
    for i in range(len(fixed)):
        if fixed[i] == 1:
            total_weight += items[i][0]

    # if already fine, do nothing
    if total_weight <= capacity:
        return fixed

    # build a list of (ratio, index) for every selected item
    selected_items = []
    for i in range(len(fixed)):
        if fixed[i] == 1:
            weight  = items[i][0]
            benefit = items[i][1]
            # avoid dividing by zero
            if weight > 0:
                ratio = benefit / weight
            else:
                ratio = 999999  # very high ratio = never drop this item
            selected_items.append((ratio, i))

    selected_items.sort()   # sorts by ratio ascending = worst first

    # drop items until we are within the weight limit
    for ratio, index in selected_items:
        if total_weight <= capacity:
            break              # we're now within the limit, stop dropping

        fixed[index] = 0                  # remove this item
        total_weight -= items[index][0]   # subtract its weight

    return fixed
#  STEP 4 — SELECTION (PICK PARENTS)
def tournament_selection(population, fitness_scores):
    """
    Tournament selection:
      1. Pick 3 random chromosomes from the population
      2. The one with the highest fitness score wins
      3. Return a copy of the winner — it becomes a parent

    Better solutions win more often, but weaker ones still have a chance.
    This keeps variety in the population.
    """
    # pick 3 random positions in the population
    candidates = random.sample(range(len(population)), TOURNAMENT_K)
    # find which of those 3 has the best fitness
    best_index = candidates[0]
    for index in candidates:
        if fitness_scores[index] > fitness_scores[best_index]:
            best_index = index
    # return a copy of the winner
    return population[best_index][:]
#  STEP 5 — CROSSOVER (BREED TWO PARENTS)
def two_point_crossover(parent1, parent2):
    """
    Two-point crossover:
      1. Pick 2 random cut points
      2. Swap the middle segment between the two parents
      3. This creates 2 children that each inherit from both parents
    """
    n = len(parent1)
    # need at least 2 items to have two cut points
    if n < 2:
        return parent1[:], parent2[:]

    # pick 2 different random positions and sort them
    point1 = random.randint(0, n - 1)
    point2 = random.randint(0, n - 1)

    # make sure point1 < point2
    if point1 > point2:
        point1, point2 = point2, point1

    # build child1: start of parent1 + middle of parent2 + end of parent1
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]

    # build child2: start of parent2 + middle of parent1 + end of parent2
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2
#  STEP 6 — MUTATION (RANDOM GENE FLIPS)
def mutate(chromosome):
    """
    Go through every gene in the chromosome.
    Each gene has a small chance (2%) of being flipped:
      0 → 1  (start including this item)
      1 → 0  (stop including this item)
    """
    mutated = chromosome[:]   # work on a copy

    for i in range(len(mutated)):
        chance = random.random()   # random number between 0.0 and 1.0

        if chance < MUTATION_RATE:
            # flip the gene
            if mutated[i] == 0:
                mutated[i] = 1
            else:
                mutated[i] = 0
    return mutated

#  THE MAIN GA LOOP

def run_genetic_algorithm(items, capacity):
    """
    Returns the best chromosome (list of 0s and 1s) found.
    """
    num_items = len(items)
    # Make random chromosomes, repair any that are overweight
    population = []
    for i in range(POPULATION_SIZE):
        chromosome = create_random_chromosome(num_items)
        chromosome = repair_chromosome(chromosome, items, capacity)
        population.append(chromosome)

    # Keep track of the best solution ever seen
    best_chromosome = population[0][:]
    best_fitness = calculate_fitness(population[0], items, capacity)

    # check all starting chromosomes for the best
    for chromosome in population:
        score = calculate_fitness(chromosome, items, capacity)
        if score > best_fitness:
            best_fitness = score
            best_chromosome = chromosome[:]

    # ── RUN 500 GENERATIONS ────────────────────────────
    for generation in range(NUM_GENERATIONS):

        # SCORE every chromosome in the current population
        fitness_scores = []
        for chromosome in population:
            score = calculate_fitness(chromosome, items, capacity)
            fitness_scores.append(score)

        # ELITISM: find the top 2 chromosomes and carry them straight
        # into the next generation unchanged
        elite_chromosomes = []
        scores_with_index = []

        for i in range(len(population)):
            scores_with_index.append((fitness_scores[i], i))

        # sort by score descending (best first)
        scores_with_index.sort(reverse=True)

        for rank in range(ELITE_COUNT):
            score, index = scores_with_index[rank]
            elite_chromosomes.append(population[index][:])

        # start the new population with the elites
        new_population = elite_chromosomes[:]

        # FILL the rest of the new population
        while len(new_population) < POPULATION_SIZE:
            # SELECTION
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            # CROSSOVER
            if random.random() < CROSSOVER_RATE:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            # MUTATION
            child1 = mutate(child1)
            child2 = mutate(child2)
            # REPAIR
            child1 = repair_chromosome(child1, items, capacity)
            child2 = repair_chromosome(child2, items, capacity)
            # add children to new population
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        # REPLACE old population with new one
        population = new_population
        # check if this generation produced a new all-time best
        for chromosome in population:
            score = calculate_fitness(chromosome, items, capacity)
            if score > best_fitness:
                best_fitness    = score
                best_chromosome = chromosome[:]
    return best_chromosome


#  INPUT / OUTPUT

def solve():
    """
    Reads input.txt in the required format, runs the GA, prints results.
    """
    all_text = sys.stdin.read()
    numbers  = all_text.split() # split by spaces/newlines into a flat list
    position = [0] # use a list so the inner function can update it

    def read_next_number():
        value = int(numbers[position[0]])
        position[0] += 1
        return value

    num_cases = read_next_number()

    for case_number in range(1, num_cases + 1):

        num_items = read_next_number()
        capacity  = read_next_number()

        items = []
        for i in range(num_items):
            weight  = read_next_number()
            benefit = read_next_number()
            items.append((weight, benefit))

        # run the genetic algorithm
        best = run_genetic_algorithm(items, capacity)

        # decode the chromosome back into actual items
        selected_items = []
        for i in range(num_items):
            if best[i] == 1:
                selected_items.append(items[i])

        total_benefit = 0
        for weight, benefit in selected_items:
            total_benefit += benefit

        # print results
        print(f"Case {case_number}: {total_benefit}")
        print(len(selected_items))
        for weight, benefit in selected_items:
            print(weight, benefit)

#  DEMO

def demo():
    """
    run and test without any input.txt file
    """
    test_cases = [
        (50,  [(4, 4), (7, 6), (5, 3)]),# capacity=10, 3 items
        (15,  [(2, 3), (3, 4), (4, 5), (5, 6), (9, 10)]), # capacity=15, 5 items
    ]
    random.seed(42)

    for case_number, (capacity, items) in enumerate(test_cases, 1):
        best = run_genetic_algorithm(items, capacity)
        # decode result
        selected_items = []
        for i in range(len(items)):
            if best[i] == 1:
                selected_items.append(items[i])

        total_weight  = sum(w for w, b in selected_items)
        total_benefit = sum(b for w, b in selected_items)

        print(f"=== Case {case_number} ===")
        print(f"  All items : {items}")
        print(f"  Capacity  : {capacity}")
        print(f"  Chromosome: {best}")
        print(f"  Selected  : {selected_items}")
        print(f"  Weight    : {total_weight} / {capacity}")
        print(f"  Benefit   : {total_benefit}")
        print()
#  ENTRY POINT

if __name__ == "__main__":
    if sys.stdin.isatty():
        # Running normally in terminal → show demo
        demo()
    else:
        # Input is being piped in → run judge mode
        solve()