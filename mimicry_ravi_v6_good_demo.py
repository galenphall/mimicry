import numpy as np
from tqdm import tqdm
import datetime 
import zarr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, poisson
from scipy.special import expit, softmax
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def similarity(detectors, signals, phenotype_type='vector'):
    """
    Calculate the similarities between lists of signals and detectors. Signals and detectors can either all be vectors
    or bitstrings.
    """
    match phenotype_type:
        case 'vector':
            # return cosine_similarity(detectors, signals)
            return euclidean_distances(detectors, signals, squared=True)
        case 'bitstring':
            d = signals.shape[1]
            # Calculate matrix of hamming distances
            detectors_expanded = detectors[:, np.newaxis, :]
            signals_expanded = signals[np.newaxis, :, :]
            hamming_distances = np.sum(detectors_expanded != signals_expanded, axis=2)
            return 1 - hamming_distances/d  # Matrix of fractions of bits that are the same
        case _:
            raise NotImplementedError
        

def calculate_detection_matrix(detectors, signals, risk_tols, phenotype_type='vector'):
    dissimilarity_matrix = similarity(detectors, signals, phenotype_type=phenotype_type)
    predation_matrix = 1 - np.exp(-dissimilarity_matrix / risk_tols[:, np.newaxis])
    return predation_matrix


def calculate_predation_matrix(detectors, signals, risk_tols, handling_time=1/3, attack_rate=1, phenotype_type='vector'):
    num_prey = signals.shape[0]
    detection_matrix = calculate_detection_matrix(detectors, signals, risk_tols, phenotype_type)
    effective_prey_populations = detection_matrix.sum(1)  # expected number of prey that predator detects
    intake_rates = attack_rate * effective_prey_populations / (1 + attack_rate * handling_time * effective_prey_populations)  # type II functional response
    intake_fracs = intake_rates / num_prey
    return intake_fracs[:, np.newaxis] * detection_matrix

# def calculate_predation_matrix(detectors, signals, risk_tols, phenotype_type='vector'):
#     similarity_matrix = similarity(detectors, signals, phenotype_type=phenotype_type)
#     predation_matrix = expit(similarity_matrix / risk_tols[:, np.newaxis])
#     return predation_matrix / 20


# OLD VERSION WITH INTER MIMIC-MODEL COMPETITION
# def num_offspring_of_predators_and_prey(predation_matrix, venom_levels,
#                                         prey_base_reproduction_rate=2, prey_competition_coeff=0.01,
#                                         predator_conversion_rate=1, predator_competition_coeff=0.01):
    
#     num_predators, num_prey = predation_matrix.shape
#     eat_matrix = np.random.random_sample(size=(num_predators, num_prey)) < predation_matrix  # X_ij
#     venom_matrix = np.logical_and(eat_matrix, (np.random.random_sample(size=(num_predators, num_prey)) < venom_levels[np.newaxis, :])) # X_ij * V_ij

#     prey_survival = np.all(~eat_matrix, axis=0)  # ~ is element-wise negation (so ~eat_matrix is the didn't-get-eaten matrix)
#     prey_reproduction_rate = np.maximum(1 + prey_base_reproduction_rate - prey_competition_coeff * num_prey, 0)  # logistic growth in discrete time
#     prey_offspring_numbers = prey_survival * poisson.rvs(prey_reproduction_rate, size=num_prey)
#     # TODO: redo above so there's not competition between mimic and model

#     predator_survival = np.all(~venom_matrix, axis=1)
#     predator_eats_count = np.sum(eat_matrix, axis=1)
#     predator_reproduction_rates = np.maximum(predator_conversion_rate * predator_eats_count - predator_competition_coeff * num_predators, 0)
#     predator_offspring_numbers = predator_survival * poisson.rvs(predator_reproduction_rates)

#     return predator_offspring_numbers, prey_offspring_numbers


def num_offspring_of_predators_and_prey(predation_matrix, venom_levels, num_venomous,
                                        venomous_base_reproduction_rate=2, venomous_competition_coeff=0.01,
                                        mimic_base_reproduction_rate=2, mimic_competition_coeff=0.01,
                                        predator_conversion_rate=1, predator_competition_coeff=0.01):
    
    num_predators, num_prey = predation_matrix.shape
    num_mimics = num_prey - num_venomous
    eat_matrix = np.random.random_sample(size=(num_predators, num_prey)) < predation_matrix  # X_ij
    venom_matrix = np.logical_and(eat_matrix, (np.random.random_sample(size=(num_predators, num_prey)) < venom_levels[np.newaxis, :])) # X_ij * V_ij

    prey_survival = np.all(~eat_matrix, axis=0)  # ~ is element-wise negation (so ~eat_matrix is the didn't-get-eaten matrix)
    venomous_reproduction_rate = np.maximum(1 + venomous_base_reproduction_rate - venomous_competition_coeff * num_venomous, 0)  # logistic growth in discrete time
    venomous_offspring_numbers = prey_survival[:num_venomous] * poisson.rvs(venomous_reproduction_rate, size=num_venomous)
    mimic_reproduction_rate = np.maximum(1 + mimic_base_reproduction_rate - mimic_competition_coeff * num_mimics, 0)
    mimic_offspring_numbers = prey_survival[num_venomous:] * poisson.rvs(mimic_reproduction_rate, size=num_mimics)
    prey_offspring_numbers = np.hstack((venomous_offspring_numbers, mimic_offspring_numbers))

    # hacky pop. stabilization
    increments = np.random.choice(np.where(prey_survival[:num_venomous])[0], size=num_venomous, replace=True)
    venomous_offspring_numbers = np.bincount(increments, minlength=num_venomous).astype(np.int64)

    increments = np.random.choice(np.where(prey_survival[num_venomous:])[0], size=num_mimics, replace=True)
    mimic_offspring_numbers = np.bincount(increments, minlength=num_mimics).astype(np.int64)

    prey_offspring_numbers = np.hstack((venomous_offspring_numbers, mimic_offspring_numbers))
    # end hacky pop. stabilization


    predator_survival = np.all(~venom_matrix, axis=1)
    predator_eats_count = np.sum(eat_matrix, axis=1)
    predator_reproduction_rates = np.maximum(predator_conversion_rate * predator_eats_count - predator_competition_coeff * num_predators, 0)
    predator_offspring_numbers = predator_survival * poisson.rvs(predator_reproduction_rates)

    # hacky pop. stabilization
    increments = np.random.choice(np.where(predator_survival)[0], size=num_predators, replace=True)
    predator_offspring_numbers = np.bincount(increments, minlength=num_predators).astype(np.int64)
    # end hacky pop. stabilization

    return predator_offspring_numbers, prey_offspring_numbers




# def fitness_func(eats_count):
#     with np.errstate(divide='ignore'):
#         return np.log(eats_count)


# def fitness_of_predators_and_prey(predation_matrix, venom_levels):
#     num_predators, num_prey = predation_matrix.shape
#     eat_matrix = np.random.random_sample(size=(num_predators, num_prey)) < predation_matrix  # X_ij

#     prey_survival = np.all(~eat_matrix, axis=0)  # ~ is element-wise negation (so ~eat_matrix is the didn't-get-eaten matrix)
#     prey_fitness = np.zeros(num_prey)
#     prey_fitness[~prey_survival] = -np.inf

#     predator_eats_count = np.sum(eat_matrix, axis=1)
#     venom_matrix = np.logical_and(eat_matrix, (np.random.random_sample(size=(num_predators, num_prey)) < venom_levels[np.newaxis, :])) # X_ij * V_ij
#     predator_survival = np.all(~venom_matrix, axis=1)
#     predator_fitness = fitness_func(predator_eats_count)
#     predator_fitness[~predator_survival] = -np.inf

#     return predator_fitness, prey_fitness
#     # return predator_fitness, prey_fitness, eat_matrix, venom_matrix


# def sample_population(fitness):
#     num_nextgen = fitness.shape[0]  # todo: change population levels
#     selection_probabilities = softmax(fitness)
#     return np.random.choice(fitness.shape[0], size=num_nextgen, p=selection_probabilities)


def mutate(phenotypes, mutation_rate=0.01, random_dist=np.random.standard_normal):  # mutation rate is per attribute/component of the phenotype
    
    # common, small mutations
    
    if len(phenotypes.shape) == 2:
        num_individuals, d = phenotypes.shape
        noise = np.random.multivariate_normal(mean=np.zeros(d), cov=mutation_rate*np.eye(d), size=num_individuals)
    else:
        num_individuals = phenotypes.shape[0]
        noise = np.random.normal(scale=mutation_rate, size=num_individuals)

    return phenotypes + noise

    # rare, large mutations below
    flat_phenotypes = phenotypes.flatten()
    mutation_sites = np.random.random_sample(size=flat_phenotypes.shape) < mutation_rate
    flat_phenotypes[mutation_sites] = random_dist(size=np.sum(mutation_sites))
    return flat_phenotypes.reshape(phenotypes.shape)





# def sample_predators(predation_matrix, venom_levels):
#     """
#     Sample the predator population based on the predation matrix and venom levels.
#     """
#     num_predators, num_prey = predation_matrix.shape
#     death_probabilities = np.sum(predation_matrix * venom_levels, axis=1)
#     fitnesses = 1 - death_probabilities
#     P = fitnesses / np.sum(fitnesses)                                           # Normalize the probabilities
#     parents = np.random.choice(num_predators, size=num_predators, p=P)
#     return parents


# def sample_prey(predation_matrix):
#     """
#     Sample the prey population based on the predation matrix.
#     """
#     num_predators, num_prey = predation_matrix.shape
#     fitnesses = np.prod(1 - predation_matrix, axis=0)
#     P = fitnesses / np.sum(fitnesses)                                           # Normalize the probabilities
#     parents = np.random.choice(num_prey, size=num_prey, p=P)
#     return parents


# def phenotype_crossover(phenotypes, parents, phenotype_type='vector'):

#     # phenotypes can be signals or detectors
#     assert len(parents) % 2 == 0, 'Crossover not implemented yet for odd numbers of parents'
#     parent_phenotypes = phenotypes[parents]
#     child_phenotypes = np.zeros_like(parent_phenotypes)
#     match phenotype_type:
#         case 'vector':
#             interpolation_values = np.random.rand(parent_phenotypes.shape[0] // 2)[:, np.newaxis]
#             child_phenotypes[::2]  = interpolation_values * parent_phenotypes[::2] \
#                                     + (1 - interpolation_values) * parent_phenotypes[1::2]
#             child_phenotypes[1::2] = (1 - interpolation_values) * parent_phenotypes[::2] \
#                                     + interpolation_values * parent_phenotypes[1::2]
#             return child_phenotypes
#         case 'bitstring':
#             raise NotImplementedError
#         case _:
#             raise NotImplementedError
        

# def phenotype_mutate(phenotypes, mutation_rate=0.01, phenotype_type='vector'):
#     """
#     Apply random mutations to the bit strings or numerical values.
#     """
#     num_individuals, d = phenotypes.shape

#     match phenotype_type:
#         case 'vector':
#             # noise = np.random.normal(scale=mutation_rate, size=(num_individuals, d))  # scale mutation rate with d?
#             noise = np.random.multivariate_normal(mean=np.zeros(d), cov=mutation_rate*np.eye(d), size=num_individuals)
#             return phenotypes + noise
#         case 'bitstring':
#             # For bit strings, flip bits with some probability
#             raise NotImplementedError
#         case _:
#             raise NotImplementedError


def update(detectors, signals, risk_tols, venom_levels, num_venomous, 
           mutation_rate=0.01, handling_time=1/3, attack_rate=1, 
           venomous_base_reproduction_rate=2, venomous_competition_coeff=0.01,
           mimic_base_reproduction_rate=2, mimic_competition_coeff=0.01,
           predator_conversion_rate=1, predator_competition_coeff=0.01,
           phenotype_type='vector'):
    
    assert detectors.shape[0] > 0, 'Predator population crash'
    assert num_venomous > 0, 'Venomous population crash'
    assert signals.shape[0] - num_venomous > 0, 'Mimic population crash'

    assert np.all(venom_levels[num_venomous:] == 0), 'A mimic has a non-zero venom level'

    predation_matrix = calculate_predation_matrix(detectors, signals, risk_tols, 
                                                  handling_time=handling_time, attack_rate=attack_rate, 
                                                  phenotype_type=phenotype_type)

    predator_offspring_numbers, prey_offspring_numbers = num_offspring_of_predators_and_prey(predation_matrix, venom_levels, num_venomous,
            venomous_base_reproduction_rate=venomous_base_reproduction_rate, venomous_competition_coeff=venomous_competition_coeff,
            mimic_base_reproduction_rate=mimic_base_reproduction_rate, mimic_competition_coeff=mimic_competition_coeff,
            predator_conversion_rate=predator_conversion_rate, predator_competition_coeff=predator_competition_coeff)

    offspring_num_venomous = np.sum(prey_offspring_numbers[:num_venomous])
    offspring_detectors     = np.repeat(detectors, predator_offspring_numbers, axis=0)  # np.repeat is crazy
    offspring_signals       = np.repeat(signals, prey_offspring_numbers, axis=0)
    offspring_risk_tols     = np.repeat(risk_tols, predator_offspring_numbers, axis=0)
    offspring_venom_levels  = np.repeat(venom_levels, prey_offspring_numbers, axis=0)


    # apply mutations
    offspring_detectors = mutate(offspring_detectors, mutation_rate=mutation_rate)
    offspring_signals = mutate(offspring_signals, mutation_rate=mutation_rate)
    offspring_risk_tols = mutate(offspring_risk_tols, mutation_rate=mutation_rate)
    offspring_venom_levels[:offspring_num_venomous] = np.maximum(mutate(offspring_venom_levels[:offspring_num_venomous], mutation_rate=mutation_rate), 0)

    print(f'predators left: {offspring_detectors.shape[0]}')
    print(f'venomous left: {offspring_num_venomous}')
    print(f'mimics left: {offspring_signals.shape[0] - offspring_num_venomous}')

    return offspring_detectors, offspring_signals, offspring_risk_tols, offspring_venom_levels, offspring_num_venomous






    predator_fitness, prey_fitness = fitness_of_predators_and_prey(predation_matrix, venom_levels)

    predator_nextgen = sample_population(predator_fitness)
    venomous_nextgen = sample_population(prey_fitness[:num_venomous])
    mimic_nextgen = sample_population(prey_fitness[num_venomous:])

    nextgen_detectors = detectors[predator_nextgen]
    venomous_nextgen_signals = venomous_signals[venomous_nextgen]
    mimic_nextgen_signals = mimic_signals[mimic_nextgen]
    nextgen_risk_tols = risk_tols[predator_nextgen]
    venomous_nextgen_venom_levels = venom_levels[venomous_nextgen]

    nextgen_detectors = mutate(nextgen_detectors, mutation_rate=mutation_rate, random_dist=np.random.standard_normal)
    venomous_nextgen_signals = mutate(venomous_nextgen_signals, mutation_rate=mutation_rate, random_dist=np.random.standard_normal)
    mimic_nextgen_signals = mutate(mimic_nextgen_signals, mutation_rate=mutation_rate, random_dist=np.random.standard_normal)
    nextgen_risk_tols = mutate(nextgen_risk_tols, mutation_rate=mutation_rate, random_dist=np.random.standard_normal)
    venomous_nextgen_venom_levels = mutate(venomous_nextgen_venom_levels, mutation_rate=mutation_rate, random_dist=np.random.random_sample)

    nextgen_signals = np.vstack((venomous_nextgen_signals, mimic_nextgen_signals))
    nextgen_venom_levels = np.zeros_like(venom_levels)
    nextgen_venom_levels[:num_venomous] = venomous_nextgen_venom_levels

    return nextgen_detectors, nextgen_signals, nextgen_risk_tols, nextgen_venom_levels