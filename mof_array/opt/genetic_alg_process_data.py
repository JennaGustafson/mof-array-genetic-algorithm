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
    """Creates list of mof names in an array from the current population

    Keyword arguments:
    mof_names -- list of names of all possible mofs in study
    current_mof_array -- list of indices for current array
    """
    selected_items = []
    for mof_index in current_mof_array:
        # Appends the mof name corresponding to the current index value
        selected_items.append(mof_names[mof_index])
    return(selected_items)

def array_pmf(gas_names, mof_names, calculate_pmf_results, current_population):
    """Sets up all combinations of MOF arrays, uses function 'compound_probability' to get pmf values
    for every array/gas/experiment combination

    Keyword arguments:
    gas_names -- list of gases
    mof_names -- list of all mofs
    calculate_pmf_results -- list of dictionaries including mof, mixture, probability
    current_population -- list of dictionaries with each array in current generation
    """
    mof_array_list = []
    array_pmf = []

    # save list of dictionaries for 1 MOF to have list of all gas mole fractions
    array_temp_dict = [row for row in calculate_pmf_results if row['MOF'] == mof_names[0]]

    # for each array's index in the current generation, calculate joint probability
    for count in range(len(current_population)):
        # Calls function to set up list of MOFs in current array
        selected_mofs = get_selected_mofs(mof_names, current_population[count]['geneIndex'])
        # Save each array's index and list of names
        mof_array_list.append({'number': count, 'mof names' : selected_mofs, 'geneIndex' : current_population[count]['geneIndex']})

        # Calls outside function to calculate joint probability
        normalized_compound_pmfs = compound_probability(selected_mofs, calculate_pmf_results)

        if count == 0:
            # First array, set up dict with keys for each array and gas, specifying pmfs and comps
            for index in range(len(array_temp_dict)):
                array_dict = {'%s' % count : normalized_compound_pmfs[index]}
                for gas in gas_names:
                    array_dict.update({ '%s' % gas : float(array_temp_dict[index][gas])})
                array_pmf.extend([array_dict])
        else:
            # Update dictionary with pmf list for each array
            for index in range(len(array_temp_dict)):
                array_dict = array_pmf[index].copy()
                array_dict.update({'%s' % count : normalized_compound_pmfs[index]})
                array_pmf[index] = array_dict

    return(array_pmf, mof_array_list)

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

def bin_compositions(gases, list_of_arrays, create_bins_results, array_pmf_results):
    """Sorts pmfs into bins created by create_bins function.

    Keyword arguments:
    gases -- list of gases specified as user input
    list_of_arrays -- list of all array combinations
    create_bins_results -- dictionary containing bins for each gas
    array_pmf_results -- list of dictionaries, arrays, joint pmfs
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
            temp_array_pmf = {'%s' % array: [] for array in range(len(list_of_arrays))}
            for line in array_pmf_results:
                # Checks that the gas' mole frac matches the current bin
                if b[gas_name] == line['%s bin' % gas_name]:
                    # For each array, assigns the pmfs to their corresponding key
                    for count in range(len(list_of_arrays)):
                        temp_pmf_list = temp_array_pmf['%s' % count]
                        temp_pmf_list.append(line['%s' % count])
                        temp_array_pmf['%s' % count] = temp_pmf_list

            # Updates pmfs for each array for current bin, summing over all pmfs
            bin_temporary = {'%s bin' % gas_name : b[gas_name]}
            for count in range(len(list_of_arrays)):
                if temp_array_pmf['%s' % count] == []:
                    bin_temporary.update({'%s' % count: 0})
                else:
                    bin_temporary.update({'%s' % count : sum(temp_array_pmf['%s' % count])})

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

    # For each array in generation, take list of dictionaries with results
    for count in range(len(list_of_arrays)):
        array_gas_temp = list_of_arrays[count].copy()
        for gas in gas_names:
                pmfs_per_array = [row['%s' % count] for row in bin_compositions_results if '%s bin' % gas in row.keys()]
                # For each array/gas combination, calculate the kld
                kl_divergence = sum([float(pmf)*log(float(pmf)/reference_prob,2) for pmf in pmfs_per_array if pmf != 0])
                # Result is list of dicts, dropping the pmf values
                array_gas_temp.update({'%s KLD' % gas : round(kl_divergence,4)})
        array_gas_info_gain.append(array_gas_temp)

    return(array_gas_info_gain)

def choose_best_arrays(gas_names, information_gain_results):
    """Choose the best MOF arrays by selecting the top KL scores for each gas

    Keyword arguments:
    gas_names -- list of gases
    information_gain_results -- list of dictionaries including, mof array index, gas, and corresponding kld
    """
    # Combine KLD values for each array,taking the product over all gases
    ranked_by_product = []
    each_array_temp = []
    for each_array in information_gain_results:
            product_temp = reduce(operator.mul, [each_array['%s KLD' % gas] for gas in gas_names], 1)
            each_array_temp = each_array.copy()
            each_array_temp.update({'joint_KLD' : product_temp, 'num_MOFs' : len(each_array['mof names'])})
            ranked_by_product.append(each_array_temp)

    # Sort results from highest to lowest joint KLD values
    best_ranked_by_product = sorted(ranked_by_product, key=lambda k: k['joint_KLD'], reverse=True)

    return(best_ranked_by_product)

def create_one_individual(array_size, mofs_list):
    """Sets up a 'random' array for one element in the first population

    Keyword arguments:
    array_size -- number of mofs that the array must contain
    mofs_list -- list of possible mofs to choose from
    """
    individual = []
    # Selects a random number from 0 to the maximum possible mof index
    new_element = ran.choice(list(range(len(mofs_list))))
    # Loop runs if the array size is less than desired for array
    while len(individual) < array_size:
        # Checks that the index chosen is one in the range of possible mofs
        if new_element <= len(mofs_list) - 1:
            element = new_element
        else:
            # Takes the remainder if the number is greater than the number of possible mofs
            element = new_element % len(mofs_list)
        individual.append(element)
        # Add new random number to previous number to avoid selecting the same as previous
        new_element += ran.choice(list(range(len(mofs_list))))
        # Takes only unique elements in list to avoid duplicate mofs
        individual = list(set(individual))
    return(sorted(individual))

def create_first_population(array_size, mofs_list, population_size):
    """Calls function to create 'random' arrays for first population

    Keyword arguments:
    array_size -- number of mofs that the array must contain
    mofs_list -- list of possible mofs to choose from
    population_size -- number of arrays in one generation
    """
    population = []
    for i in range(population_size):
        population.append({'geneIndex': create_one_individual(array_size, mofs_list)})
    return(population)

def sort_population(gas_names, population_fitness):
    """Calls function to sort arrays by joint KLD

    Keyword arguments:
    gas_names -- list of gases
    population_fitness -- list of dictionaries including, mof array index, gas, and corresponding kld
    """
    ordered_population = choose_best_arrays(gas_names, population_fitness)
    return(ordered_population)

def select_breeders(population_sorted, population_size):
    """Selects top performing arrays and random others to be parents

    Keyword arguments:
    population_sorted -- ordered list of arrays
    population_size -- number of arrays in population set
    """
    result = []
    # Take top 20% of arrays and random 20% of arrays
    best_individuals = population_size // 5
    lucky_few = population_size // 5
    for i in range(best_individuals):
        result.append({'geneIndex': population_sorted[i]['geneIndex']})
    for i in range(lucky_few):
        lucky_one = ran.choice(population_sorted[best_individuals:population_size+1])
        result.append({'geneIndex' : lucky_one['geneIndex']})
    ran.shuffle(result)
    return(result)

def create_child(individual1, individual2, mofs_list):
    """Performs crossover with two selected parents

    Keyword arguments:
    individual1 -- parent array
    individual2 -- parent array
    mofs_list -- list of possible mofs to choose from
    """
    result = []
    # Selects each array element from a random parent
    for i in range(len(individual1['geneIndex'])):
        if (100 * ran.random() < 50):
            result.append(individual1['geneIndex'][i])
        else:
            result.append(individual2['geneIndex'][i])
    # Takes unique elemens of array
    result = list(set(result))
    # Adds elements to the array until there are no duplicates
    while len(result) < len(individual1['geneIndex']):
        result.append(ran.choice(list(range(len(mofs_list)))))
        result = list(set(result))
    return(result)

def create_children(breeders, number_of_child, mofs_list):
    """Creates children by doing crossover with all parents

    Keyword arguments:
    breeders -- list of parent arrays
    number_of_child -- how many children to create from each parent
    mofs_list -- list of mofs to choose from
    """
    result = []
    for i in range(len(breeders) // 2):
        for j in range(number_of_child):
            result.append({'geneIndex' : create_child(breeders[i], breeders[len(breeders) - 1 - i], mofs_list)})
    return(result)

def mutate_one_individual(individual, mutationRate, mofs_list):
    """Performs a mutation on one array element based on the given rate

    Keyword arguments:
    individual -- current array in population
    mutationRate -- how often mutation should occur
    mofs_list -- list of mofs to choose from
    """
    # Mutates an element in the array according to the rate
    for geneIndex in range(len(individual['geneIndex'])):
        if (100 * ran.random() < mutationRate):
            # Chooses a random mof to replace element
            individual['geneIndex'][geneIndex] = ran.choice(list(range(len(mofs_list))))
    # Takes all unique elements of array
    reduced_individual = list(set(individual['geneIndex']))
    # Adds new mofs to array until there are no duplicates
    while len(reduced_individual) < len(individual['geneIndex']):
        reduced_individual.append(ran.choice(list(range(len(mofs_list)))))
        reduced_individual = list(set(reduced_individual))
    individual['geneIndex'] = sorted(reduced_individual)
    return(individual)

def mutate_population(current_population, mutationRate, mofs_list):
    """Sets up each array for possible mutation

    Keyword arguments:
    current_population -- list of arrays in current generation
    mutationRate -- how often mutation should occur
    mofs_list -- list of mofs to choose from
    """
    for individual in current_population:
        individual = mutate_one_individual(individual, mutationRate, mofs_list)
    return(current_population)

def run_genetic_algorithm(array_size, mof_names, gas_names, calculate_pmf_results, population_size, number_bins, generations, mutation_rate):
    """Optimizes mof array of a specified size to have the maximum KLD

    Keyword arguments:
    array_size -- number of mofs in desired array
    mof_names -- list of mofs to choose from
    gas_names -- list of gases
    calculate_pmf_results -- probabilities for all mof/gas mixture combinations
    population_size -- number of arrays in each generation
    number_bins -- number of bins specified by user in config file
    generations -- numer of times to run the optimization
    mutation_rate -- how often mutation should occur
    """
    number_of_child = 5
    create_bins_results = create_bins(mof_names, calculate_pmf_results, gas_names, number_bins)

    # Set up first population of test arrays and calculate KLD values
    first_population_items = create_first_population(array_size, mof_names, population_size)
    first_population_joint_pmf, list_of_arrays = array_pmf(gas_names, mof_names, calculate_pmf_results, first_population_items)
    bin_compositions_results = bin_compositions(gas_names, list_of_arrays, create_bins_results, first_population_joint_pmf)
    kl_divergence = information_gain(gas_names, list_of_arrays, bin_compositions_results, create_bins_results)

    # Order according to highest KLD value, saving top performing array
    sorted_population = sort_population(gas_names, kl_divergence)
    best_array = sorted_population[0]['geneIndex']
    best_mofs = sorted_population[0]['mof names']
    best_fitness = sorted_population[0]['joint_KLD']

    # Select first set of parents to create next generation of arrays
    parents = select_breeders(sorted_population, population_size)

    for gen in range(generations):
        # Perform crossover and mutation to create next generation of arrays
        population = create_children(parents, number_of_child, mof_names)
        population = mutate_population(population, mutation_rate, mof_names)
        # Calculate probabilities and KLD values for new arrays
        population_joint_pmf, list_of_arrays = array_pmf(gas_names, mof_names, calculate_pmf_results, population)
        bin_compositions_results = bin_compositions(gas_names, list_of_arrays, create_bins_results, population_joint_pmf)
        kl_divergence = information_gain(gas_names, list_of_arrays, bin_compositions_results, create_bins_results)
        # Sort by KLD and replace top performing array if there is one with a higher KLD
        sorted_population = sort_population(gas_names, kl_divergence)
        if sorted_population[0]['joint_KLD'] > best_fitness:
            best_array = sorted_population[0]['geneIndex']
            best_mofs = sorted_population[0]['mof names']
            best_fitness = sorted_population[0]['joint_KLD']
        else:
            None
        # Select parents for next generation of arrays
        parents = select_breeders(sorted_population, population_size)
        print(best_fitness)

    return([best_array, best_mofs, best_fitness])
