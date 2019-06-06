#!/usr/bin/env python

from generic_ga_code import run_genetic_algorithm

mofs_list = ['H1', 'I1', 'Z8', 'U66', 'N125']
population_size = 10
generations = 5
mutation_rate = 0.5

results = run_genetic_algorithm(mofs_list, population_size, generations, mutation_rate)
print(results)
