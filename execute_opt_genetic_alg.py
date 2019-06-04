#!/usr/bin/env python
import sys
from datetime import datetime
from mof_array.opt.genetic_alg_process_data import (read_output_data,
                                            yaml_loader,
                                            write_output_data,
                                            import_experimental_results,
                                            import_simulated_data,
                                            calculate_pmf,
                                            create_bins,
                                            bin_compositions,
                                            array_pmf,
                                            information_gain,
                                            choose_best_arrays,
                                            run_genetic_algorithm)
all_results_import = read_output_data(sys.argv[1])
experimental_mass_import = read_output_data(sys.argv[2])

filepath = 'settings/process_config.yaml'
data = yaml_loader(filepath)

mof_array = data['mof_array']
mof_densities_import = {}
mof_experimental_mass = {}

for mof in mof_array:
    mof_densities_import.copy()
    mof_densities_import.update({ mof : data['mofs'][mof]['density']})

num_mixtures = data['num_mixtures']
stdev = data['stdev']
mrange = data['mrange']
gases = data['gases']
number_mofs = data['number_mofs']
number_bins = data['number_bins']

array_size = data['array_size']
population_size = data['population_size']
generations = data['generations']
mutation_rate = data['mutation_rate']

experimental_mass_results, experimental_mass_mofs, experimental_mofs = import_experimental_results(mof_array, experimental_mass_import, mof_densities_import, gases)
import_data_results = import_simulated_data(experimental_mofs, all_results_import, mof_densities_import, gases)
calculate_pmf_results = calculate_pmf(experimental_mass_results, import_data_results, experimental_mofs, stdev, mrange)
results, save_results = run_genetic_algorithm(array_size, mof_array, gases, calculate_pmf_results, population_size, number_bins, generations, mutation_rate)
print(results)
