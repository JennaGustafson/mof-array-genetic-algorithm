
import random
from math import log
import numpy as np

def define_one_item(item):
    # import probability data for mof item
    wild_card1 = random.random()*0.1
    wild_card2 = 1 - (wild_card1 + 0.2 + 0.1 + 0.5)
    probability = [wild_card2, 0.2, 0.1, 0.5, wild_card1]
    return(probability)

def define_item_set(mofs_list):
#     assign probability sets for each mof
    item_set = None
    for item in mofs_list:
        if item_set is not None:
            temp_item_set = item_set.copy()
            temp_item_set.update({item: define_one_item(item)})
            item_set = temp_item_set
        else:
            item_set = {item: define_one_item(item)}
    return(item_set)

def define_mof_item_set(mofs_list):
    item_set = {'H1': [.05,.05,.05,.1,.2,.4,.1,.05,0,0], 'I1': [.05,.05,.05,.1,.1,.5,.05,.1,0,0], 'Z8': [.05,.05,.1,.15,.15,.5,0,0,0,0],
    'U66': [.05,.05,.1,.1,.05,.2,.3,.1,.05,0], 'N125': [.1,.2,.05,.1,.05,.05,.1,.2,.1,.05]}
    return(item_set)

# creates one array in the first population
def create_one_individual(mofs_list):
    individual = []
    for mof_name in mofs_list:
        individual.append(random.choice([True,False]))
    return(individual)

def create_first_population(mofs_list, population_size):
    population = []
    for i in range(population_size):
        population.append({'geneIndex': create_one_individual(mofs_list)})
    return(population)

def element_kld(item_set, selected_items):
    compound_probability = None
    for item in selected_items:
        if compound_probability is not None:
            compound_probability = [x*y for x,y in zip(compound_probability, item_set[item])]
        else:
            compound_probability = item_set[item]
    normalize_factor = sum(compound_probability)
    normalized_probability = [prob / normalize_factor for prob in compound_probability]
    reference_prob = 1 / len(normalized_probability)
    element_kld_value = sum([float(n_prob)*log((n_prob/reference_prob),2) for n_prob in normalized_probability if n_prob != 0])
    return(element_kld_value)

# function to calculate fitness of one item
def calculate_item_fitness(mofs_list, item_set, chosen_element):
    selected_items = []
    for i in range(len(chosen_element['geneIndex'])):
        if int(chosen_element['geneIndex'][i]) == 1:
            selected_items.append(mofs_list[i])
        else:
            None
    if selected_items == []:
        element_fitness = 0
    else:
        element_fitness = element_kld(item_set, selected_items)
    return(element_fitness)

# function to calculate fitness of all items
def calculate_fitness(mofs_list, item_set, first_population_):
    fitness = []
    for current_element in first_population_:
        element_fitness = calculate_item_fitness(mofs_list, item_set, current_element)
        temp_element = current_element.copy()
        temp_element.update({'fitness': element_fitness})
        fitness.append(temp_element)
    return(fitness)

def first_population(mofs_list, item_set, population_size):
    first_population_set = create_first_population(mofs_list, population_size)
    fitness_final = calculate_fitness(mofs_list, item_set, first_population_set)
    return(first_population_set, fitness_final)

# sort and pick top performing individual
def sort_population(population_fitness):
    ordered_population = sorted(population_fitness, key=lambda k: k['fitness'], reverse=True)
    return(ordered_population)

# function to select parent items
def select_breeders(population_sorted, population_size):
    result = []
    best_individuals = population_size // 5
    lucky_few = population_size // 5
    for i in range(best_individuals):
        result.append(population_sorted[i])
    for i in range(lucky_few):
        result.append(random.choice(population_sorted[best_individuals:population_size+1]))
    random.shuffle(result)
    return(result)

# function to perform crossover
def create_child(individual1, individual2):
    result = []
    for i in range(len(individual1['geneIndex'])):
        if (100 * random.random() < 50):
            result.append(individual1['geneIndex'][i])
        else:
            result.append(individual2['geneIndex'][i])
    return(result)

def create_children(breeders, number_of_child):
    result = []
    for i in range(len(breeders) // 2):
        for j in range(number_of_child):
            result.append({'geneIndex' : create_child(breeders[i], breeders[len(breeders) - 1 - i])})
    return(result)

# function to perform mutation
def mutate_one_individual(individual, mutationRate):
    for geneIndex in range(len(individual['geneIndex'])):
        if (100 * random.random() < mutationRate):
            individual['geneIndex'][geneIndex] = not individual['geneIndex'][geneIndex]
    return(individual)

def mutate_population(current_population, mutationRate):
    for individual in current_population:
        individual = mutate_one_individual(individual, mutationRate)
    return(current_population)

def run_genetic_algorithm(mofs_list, population_size, generations, mutation_rate):
    number_of_child = 5
    item_set = define_mof_item_set(mofs_list)
    first_population_items, first_population_fitness = first_population(mofs_list, item_set, population_size)
#     choose the best kld value and item with this value, save
    ordered_first_population = sort_population(first_population_fitness)
    sorted_population = ordered_first_population
    best_array = sorted_population[0]['geneIndex']
    best_fitness = sorted_population[0]['fitness']
    print(best_array, best_fitness)
    for gen in range(generations):
        parents = select_breeders(sorted_population, population_size)
        population = create_children(parents, number_of_child)
        population = mutate_population(population, mutation_rate)
        population_fitness = calculate_fitness(mofs_list, item_set, population)
        sorted_population = sort_population(population_fitness)
        if sorted_population[0]['fitness'] > best_fitness:
            best_array = sorted_population[0]['geneIndex']
            best_fitness = sorted_population[0]['fitness']
        else:
            None
        print(sorted_population)
        print(best_array, best_fitness)

    return([item_set, best_array, best_fitness])
