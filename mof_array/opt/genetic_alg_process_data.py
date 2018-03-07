"""This code imports mass adsorption data and "experimental" adsorption data
from simulated output, for multiple MOFs and gas mixtures, and calculates the
probability that specific gases are present in specific mole fractions
(in each experimental case) based on the total mass adsorbed for each MOF.
Additionally, a genetic algorithm determines the best mof array based on
performance.
"""
import os
from math import isnan, log
import csv
import random as ran
from random import random
from itertools import combinations
from functools import reduce
import operator

import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as ss
from scipy.spatial import Delaunay
import scipy.interpolate as si
from scipy.interpolate import spline
from datetime import datetime

# Function imports csv file as a dictionary

def read_output_data(filename):
    with open(filename,newline='') as csvfile:
        output_data = csv.DictReader(csvfile, delimiter="\t")
        return list(output_data)

def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)

def write_output_data(filename, data):
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for line in data:
            writer.writerow([line])
    return(writer)

def import_experimental_results(mofs_list, experimental_mass_import, mof_densities, gases):
    """Imports the experimental data and puts it in dictionary format

    Keyword arguments:
    mofs_list -- list of MOF structures simulated
    experimental_mass_import -- dictionary formatted experimental results for each mof
    mof_densities -- dictionary of densities
    gases -- list of gases in simulated mixtures
    """
    experimental_results = []
    experimental_mass_mofs = []
    experimental_mofs = []

    for mof in mofs_list:

        # Calculates masses in terms of mg/(cm3 of framework)
        masses = [float(mof_densities[row['MOF']]) * float(row['Mass']) for row in
                    experimental_mass_import if row['MOF'] == mof]
        if len(masses) is not 0:

            experimental_mofs.append(str(mof))
            # Saves composition values as a list, necessary type for the Delaunay input argument
            comps = []
            for row in experimental_mass_import:
                if row['MOF'] == mof:
                    comps.extend([[float(row[gas]) for gas in gases]])

            # List of experimental masses for current mof
            experimental_mass_temp = [ row for row in experimental_mass_import if row['MOF'] == mof]

            for index in range(len(masses)):
                temp_dict = experimental_mass_temp[index].copy()
                temp_dict.update({ 'Mass_mg/cm3' : masses[index] })
                experimental_results.extend([temp_dict])

            # Dictionary format of all the experimental data
            temp_list = {'MOF' : mof, 'Mass' :[row['Mass_mg/cm3'] for row in experimental_results if row['MOF'] == mof]}
            experimental_mass_mofs.append(temp_list)

        else:
            None

    return(experimental_results, experimental_mass_mofs, experimental_mofs)

def import_simulated_data(mofs_list, all_results, mof_densities, gases):
    """Imports simulated data and puts it in dictionary format
    If desired, interpolation is performed and may be used to create a denser data set

    Keyword arguments:
    mofs_list -- names of all mofs
    all_results -- imported csv of outputs as dictionary
    mof_densities -- dictionary of densities for each mof
    gases -- list of all gases
    """
    simulated_results = []
    for mof in mofs_list:
        # Calculates masses in terms of mg/(cm3 of framework)
        masses = [float(mof_densities[row['MOF']]) * float(row['Mass']) for row in
                    all_results if row['MOF'] == mof]

        # Saves composition values as a list, necessary type for the Delaunay input argument
        comps = []
        for row in all_results:
            if row['MOF'] == mof:
                comps.extend([[float(row[gas]) for gas in gases]])

        # Update dictionary with simulated data in terms of mg/cm3
        all_results_temp = [ row for row in all_results if row['MOF'] == mof]
        for index in range(len(masses)):
            temp_dict = all_results_temp[index].copy()
            temp_dict.update({ 'Mass_mg/cm3' : masses[index] })
            simulated_results.extend([temp_dict])

    return(simulated_results)

def add_random_gas(comps, num_mixtures):
    """ Adds random gas mixtures to the original data, between min and max of original mole fractions

    Keyword arguments:
    comps -- all simulated gas compositions
    num_mixtures -- specify integer number of mixtures to add
    """
    while (len(comps) < 78 + num_mixtures):
        random_gas = ([0.5 * round(random(), 3), 0.5 * round(random(), 3), 0.2 * round(random(), 3)])
        predicted_mass = interp_dat(random_gas)
        if sum(random_gas) <= 1 and not isnan(predicted_mass):
            comps.append(random_gas)
            masses.extend(predicted_mass)

def calculate_pmf(experimental_mass_results, import_data_results, mofs_list, stdev, mrange):
    """Calculates probability mass function of each data point

    Keyword arguments:
    experimental_mass_results -- list of dictionaries, masses from experiment
    import_data_results -- dictionary, results of import_simulated_data method
    mofs_list -- names of all MOFs
    stdev -- standard deviation for the normal distribution
    mrange -- range for which the difference between cdfs is calculated
    """
    pmf_results = []
    for mof in mofs_list:

        mof_temp_dict = []
        # Combine mole fractions, mass values, and pmfs into a numpy array for the dictionary creation.
        all_results_temp = [row for row in import_data_results if row['MOF'] == mof]
        all_masses = [row['Mass_mg/cm3'] for row in all_results_temp]

        #Loop through all of the experimental masses for each MOF, read in and save comps
        experimental_mass_data = [data_row['Mass_mg/cm3'] for data_row in experimental_mass_results
                                    if data_row['MOF'] == mof]

        for mof_mass in experimental_mass_data:
            # Sets up the truncated distribution parameters
            myclip_a, myclip_b = 0, float(max(all_masses)) * (1 + mrange)
            my_mean, my_std = float(mof_mass), float(stdev)
            a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

            new_temp_dict = []
            # Calculates all pmfs based on the experimental mass and truncated normal probability distribution.
            probs = []
            for mass in all_masses:
                probs_upper = ss.truncnorm.cdf(float(mass) * (1 + mrange), a, b, loc = my_mean, scale = my_std)
                probs_lower = ss.truncnorm.cdf(float(mass) * (1 - mrange), a, b, loc = my_mean, scale = my_std)
                probs.append(probs_upper - probs_lower)

            sum_probs = sum(probs)
            norm_probs = [(i / sum_probs) for i in probs]

            # Update dictionary with pmf for each MOF, key specified by experimental mass
            if mof_temp_dict == []:
                for index in range(len(norm_probs)):
                    mof_temp_dict = all_results_temp[index].copy()
                    mof_temp_dict.update({ 'PMF' : norm_probs[index] })
                    new_temp_dict.extend([mof_temp_dict])
                mass_temp_dict = new_temp_dict
            else:
                for index in range(len(norm_probs)):
                    mof_temp_dict = mass_temp_dict[index].copy()
                    mof_temp_dict.update({ 'PMF' : norm_probs[index] })
                    new_temp_dict.extend([mof_temp_dict])
                mass_temp_dict = new_temp_dict

        pmf_results.extend(mass_temp_dict)

    return(pmf_results)

def compound_probability(mof_array, calculate_pmf_results):
    """Combines and normalizes pmfs for a mof array and gas combination, used in method 'array_pmf'

    Keyword arguments:
    mof_array -- list of mofs in array
    calculate_pmf_results -- list of dictionaries including mof, mixture, probability
    """
    compound_pmfs = None
    for mof in mof_array:
        # Creates list of pmf values for a MOF
        mof_pmf = [ row['PMF'] for row in calculate_pmf_results if row['MOF'] == mof ]

        # Joint prob, multiply pmf values elementwise
        if compound_pmfs is not None:
            compound_pmfs = [x*y for x,y in zip(compound_pmfs, mof_pmf)]
        else:
            compound_pmfs = mof_pmf

    # Normalize joint probability, sum of all points is 1
    normalize_factor = sum(compound_pmfs)
    normalized_compound_pmfs = [ number / normalize_factor for number in compound_pmfs ]
    return(normalized_compound_pmfs)

def get_selected_mofs(mof_names, current_mof_array):
    selected_items = []
    for i in range(len(current_mof_array)):
        if int(current_mof_array[i]) == 1:
            selected_items.append(mof_names[i])
        else:
            None
    return(selected_items)

def array_pmf(gas_names, mof_names, calculate_pmf_results, current_population):
    """Sets up all combinations of MOF arrays, uses function 'compound_probability' to get pmf values
    for every array/gas/experiment combination

    Keyword arguments:
    gas_names -- list of gases
    mof_names -- list of all mofs
    calculate_pmf_results -- list of dictionaries including mof, mixture, probability
    """
    mof_array_list = []
    array_pmf = []

    # save list of dictionaries for 1 MOF to have list of all gas mole fractions
    array_temp_dict = [row for row in calculate_pmf_results if row['MOF'] == mof_names[0]]

        # Nested loops take all combinations of array/gas/experiment
    # for mof_array in current_population:
    for count in range(len(current_population)):
    # Calls outside function to calculate joint probability
        selected_mofs = get_selected_mofs(mof_names, current_population[count]['geneIndex'])
        if selected_mofs == []:
            selected_mofs = [ran.choice(mof_names)]
        mof_array_list.append(selected_mofs)

        normalized_compound_pmfs = compound_probability(selected_mofs, calculate_pmf_results)

        if count == 0:
            # First array, set up dict with keys for each array and gas, specifying pmfs and comps
            for index in range(len(array_temp_dict)):
                array_dict = {'%s' % ' '.join(selected_mofs) : normalized_compound_pmfs[index]}
                for gas in gas_names:
                    array_dict.update({ '%s' % gas : float(array_temp_dict[index][gas])})
                array_pmf.extend([array_dict])
        else:
            # Update dictionary with pmf list for each array
            for index in range(len(array_temp_dict)):
                array_dict = array_pmf[index].copy()
                array_dict.update({'%s' % ' '.join(selected_mofs) : normalized_compound_pmfs[index]})
                array_pmf[index] = array_dict

    return(array_pmf, mof_array_list)
    # return(normalized_compound_pmfs)

def create_bins(mofs_list, calculate_pmf_results, gases, num_bins):
    """Creates bins for all gases, ranging from the lowest to highest mole fractions for each.

    Keyword arguments:
    mofs_list -- list of mofs used in analysis
    calculate_pmf_results -- list of dictionaries including mof, mixture, probability
    gases -- list of present gases
    num_bins -- number of bins specified by user in config file
    """
    # Creates numpy array of all compositions, needed to calculate min/max of each gas's mole frac.
    mof = mofs_list[0]
    temp_one_mof_results = [row for row in calculate_pmf_results if row['MOF'] == mof]
    comps_array = np.array([[float(row[gas]) for gas in gases] for row in temp_one_mof_results])

    bin_range = np.column_stack([np.linspace(min(comps_array[:,index]), max(comps_array[:,index]) +
        (max(comps_array[:,index])-min(comps_array[:,index]))/num_bins, num_bins + 1) for index in range(len(gases))])

    bins = [ { gases[index] : row[index] for index in range(len(gases)) } for row in bin_range]

    return(bins)

def bin_compositions(gases, list_of_arrays, create_bins_results, array_pmf_results, experimental_mass_mofs):
    """Sorts pmfs into bins created by create_bins function.

    Keyword arguments:
    gases -- list of gases specified as user input
    list_of_arrays -- list of all array combinations
    create_bins_results -- dictionary containing bins for each gas
    array_pmf_results -- list of dictionaries, arrays, joint pmfs
    experimental_mass_mofs -- ordered list of dictionaries with each experimental mof/mass
    """

    binned_probability = []
    for gas_name in gases:
        # Assigns pmf to bin value (dictionary) by checking whether mole frac is
        # between the current and next bin value.
        for row in array_pmf_results:
             for i in range(1, len(create_bins_results)):
                if ( float(row[gas_name]) >= create_bins_results[i - 1][gas_name] and
                     float(row[gas_name]) < create_bins_results[i][gas_name]
                   ):
                    row.update({'%s bin' % gas_name: create_bins_results[i - 1][gas_name]})

        # Loops through all of the bins and takes sum over all pmfs in that bin.
        binned_probability_temporary = []
        for b in create_bins_results[0:len(create_bins_results)-1]:
            temp_array_pmf = {'%s' % ' '.join(array): [] for array in list_of_arrays}
            for line in array_pmf_results:
                # Checks that the gas' mole frac matches the current bin
                if b[gas_name] == line['%s bin' % gas_name]:
                    # For each array, assigns the pmfs to their corresponding key
                    for array in list_of_arrays:
                        temp_pmf_list = temp_array_pmf['%s' % ' '.join(array)]
                        temp_pmf_list.append(line['%s' % ' '.join(array)])
                        temp_array_pmf['%s' % ' '.join(array)] = temp_pmf_list

            # Updates pmfs for each array for current bin, summing over all pmfs
            bin_temporary = {'%s bin' % gas_name : b[gas_name]}
            for array in list_of_arrays:
                if temp_array_pmf['%s' % ' '.join(array)] == []:
                    bin_temporary.update({'%s' % ' '.join(array): 0})
                else:
                    bin_temporary.update({'%s' % ' '.join(array) : sum(temp_array_pmf['%s' % ' '.join(array)])})

            binned_probability_temporary.append(bin_temporary)

        # Creates list of binned probabilities, already normalized
        binned_probability.extend(binned_probability_temporary)

    return(binned_probability)

def information_gain(gas_names, list_of_arrays, bin_compositions_results, create_bins_results):
    """Calculates the Kullback-Liebler Divergence of a MOF array with each gas component.

    Keyword arguments:
    gas_names -- list of gases specified by user
    list_of_arrays -- list of all array combinations
    bin_compositions_results -- list of dictionaries, mof array, gas, pmfs
    create_bins_results -- dictionary result from create_bins
    """
    array_gas_info_gain = []
    reference_prob = 1/len(create_bins_results)

    # For each array, take list of dictionaries with results
    for array in list_of_arrays:
        array_gas_temp = {'mof array' : array}
        for gas in gas_names:
                pmfs_per_array = [row['%s' % ' '.join(array)] for row in bin_compositions_results if '%s bin' % gas in row.keys()]
                # For each array/gas combination, calculate the kld
                kl_divergence = sum([float(pmf)*log(float(pmf)/reference_prob,2) for pmf in pmfs_per_array if pmf != 0])
                # Result is list of dicts, dropping the pmf values
                array_gas_temp.update({'%s KLD' % gas : round(kl_divergence,4)})
        array_gas_info_gain.append(array_gas_temp)

    return(array_gas_info_gain)

def choose_best_arrays(gas_names, number_mofs, information_gain_results):
    """Choose the best MOF arrays by selecting the top KL scores for each gas

    Keyword arguments:
    gas_names -- list of gases
    number_mofs -- minimum and maximum number of mofs in an array, usr specified in config file
    information_gain_results -- list of dictionaries including, mof array, gas, and corresponding kld
    """
    # Combine KLD values for each array,taking the product over all gases
    # for mixtures having more than two components
    ranked_by_product = []
    if len(gas_names) > 2:
        for each_array in information_gain_results:
            product_temp = reduce(operator.mul, [each_array['%s KLD' % gas] for gas in gas_names], 1)
            each_array_temp = each_array.copy()
            each_array_temp.update({'joint_KLD' : product_temp, 'num_MOFs' : len(each_array['mof array'])})
            ranked_by_product.append(each_array_temp)
    else:
        for each_array in information_gain_results:
            each_array_temp = each_array.copy()
            each_array_temp.update({'joint_KLD' : each_array['%s KLD' % gas_names[0]], 'num_MOFs' : len(each_array['mof array'])})
            ranked_by_product.append(each_array_temp)

    # Sort results from highest to lowest KLD values
    best_ranked_by_product = sorted(ranked_by_product, key=lambda k: k['num_MOFs'], reverse=True)
    best_ranked_by_product = sorted(ranked_by_product, key=lambda k: k['joint_KLD'], reverse=True)
    # Sort results from lowest to highest KLD values
    worst_ranked_by_product = sorted(ranked_by_product, key=lambda k: k['num_MOFs'])
    worst_ranked_by_product = sorted(ranked_by_product, key=lambda k: k['joint_KLD'])

    arrays_to_pick_from = [best_ranked_by_product, worst_ranked_by_product]
    top_and_bottom_arrays = []

    # Saves top and bottom two arrays of each array size
    for ranked_list in arrays_to_pick_from:
        for num_mofs in range(min(number_mofs),max(number_mofs)+1):
            index = 0
            for each_array in ranked_list:
                if index < 2 and len(each_array['mof array']) == num_mofs:
                    top_and_bottom_arrays.append(each_array)
                    index +=1

    top_and_bottom_by_gas = []
    # Sort by the performance of arrays per gas and sav top and bottom two at each size
    for gas in gas_names:
        best_per_gas = sorted(information_gain_results, key=lambda k: k['%s KLD' % gas], reverse=True)
        worst_per_gas = sorted(information_gain_results, key=lambda k: k['%s KLD' % gas])
        best_and_worst_gas = [best_per_gas, worst_per_gas]
        for num_mofs in range(min(number_mofs),max(number_mofs)+1):
            for ranked_list in best_and_worst_gas:
                index = 0
                for each_array in ranked_list:
                    if index == 0 and len(each_array['mof array']) == num_mofs:
                        top_and_bottom_by_gas.append(each_array)
                        index +=1

    return(top_and_bottom_arrays, top_and_bottom_by_gas, best_ranked_by_product)

# start genetic algorithm

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

# creates one array in the first population
def create_one_individual(mofs_list):
    individual = []
    for mof_name in mofs_list:
        individual.append(ran.choice([True,False]))
    return(individual)

def create_first_population(mofs_list, population_size):
    population = []
    for i in range(population_size):
        population.append({'geneIndex': create_one_individual(mofs_list)})
    return(population)

# def element_kld(item_set, selected_items):
#     compound_probability = None
#     for item in selected_items:
#         if compound_probability is not None:
#             compound_probability = [x*y for x,y in zip(compound_probability, item_set[item])]
#         else:
#             compound_probability = item_set[item]
#     normalize_factor = sum(compound_probability)
#     normalized_probability = [prob / normalize_factor for prob in compound_probability]
#     reference_prob = 1 / len(normalized_probability)
#     element_kld_value = sum([float(n_prob)*log((n_prob/reference_prob),2) for n_prob in normalized_probability if n_prob != 0])
#     return(element_kld_value)

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
def calculate_fitness(mofs_list, item_set, population_):
    fitness = []
    for current_element in population_:
        # element_fitness = calculate_item_fitness(mofs_list, item_set, current_element)
        element_fitness = calculate_item_fitness(mofs_list, item_set, current_element)
        temp_element = current_element.copy()
        temp_element.update({'fitness': element_fitness})
        fitness.append(temp_element)
    return(fitness)

def first_population(mofs_list, population_size):
    first_population_set = create_first_population(mofs_list, population_size)
    # fitness_final = calculate_fitness(mofs_list, item_set, first_population_set)
    return(first_population_set)

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

def run_genetic_algorithm(mof_names, gas_names, calculate_pmf_results, population_size, number_bins, generations, mutation_rate):
    number_of_child = 5
    create_bins_results = create_bins(mof_names, calculate_pmf_results, gas_names, number_bins)
    first_population_items = first_population(mof_names, population_size)
    first_population_joint_pmf, list_of_arrays = array_pmf(gas_names, mof_names, calculate_pmf_results, first_population_items)
    bin_compositions_results = bin_compositions(gas_names, list_of_arrays, create_bins_results, first_population_joint_pmf, mof_names)
    kl_divergence = information_gain(gas_names, list_of_arrays, bin_compositions_results, create_bins_results)

    # ordered_first_population = sort_population(first_population_fitness)
    # sorted_population = ordered_first_population
    # best_array = sorted_population[0]['geneIndex']
    # best_fitness = sorted_population[0]['fitness']
    # print(best_array, best_fitness)
    # for gen in range(generations):
    #     parents = select_breeders(sorted_population, population_size)
    #     population = create_children(parents, number_of_child)
    #     population = mutate_population(population, mutation_rate)
    #     population_fitness = calculate_fitness(mofs_list, item_set, population)
    #     sorted_population = sort_population(population_fitness)
    #     if sorted_population[0]['fitness'] > best_fitness:
    #         best_array = sorted_population[0]['geneIndex']
    #         best_fitness = sorted_population[0]['fitness']
    #     else:
    #         None
    #     print(sorted_population)
    #     print(best_array, best_fitness)

    # return([item_set, best_array, best_fitness])
    # return(first_population_joint_pmf)
    return(kl_divergence)
