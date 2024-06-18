import numpy as np
from tqdm import tqdm
import datetime 
import zarr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


def similarity(detectors, signals, phenotype_type='vector'):
    """
    Calculate the similarities between lists of signals and detectors. Signals and detectors can either all be vectors
    or bitstrings.
    """
    match phenotype_type:
        case 'vector':
            return cosine_similarity(detectors, signals)  # Matrix of cosine similarities
        case 'bitstring':
            d = signals.shape[1]
            # Calculate matrix of hamming distances
            detectors_expanded = detectors[:, np.newaxis, :]
            signals_expanded = signals[np.newaxis, :, :]
            hamming_distances = np.sum(detectors_expanded != signals_expanded, axis=2)
            return 1 - hamming_distances/d  # Matrix of fractions of bits that are the same
        case _:
            raise NotImplementedError
        

def calculate_predation_matrix(detectors, signals, risk_tols, phenotype_type='vector'):
    similarity_matrix = similarity(detectors, signals, phenotype_type=phenotype_type)
    predation_matrix = softmax(similarity_matrix / risk_tols[:, np.newaxis], axis=1)
    return predation_matrix


def sample_predators(predation_matrix, venom_levels):
    """
    Sample the predator population based on the predation matrix and venom levels.
    """
    num_predators, num_prey = predation_matrix.shape
    death_probabilities = np.sum(predation_matrix * venom_levels, axis=1)
    fitnesses = 1 - death_probabilities
    P = fitnesses / np.sum(fitnesses)                                           # Normalize the probabilities
    parents = np.random.choice(num_predators, size=num_predators, p=P)
    return parents


def sample_prey(predation_matrix):
    """
    Sample the prey population based on the predation matrix.
    """
    num_predators, num_prey = predation_matrix.shape
    fitnesses = np.prod(1 - predation_matrix, axis=0)
    P = fitnesses / np.sum(fitnesses)                                           # Normalize the probabilities
    parents = np.random.choice(num_prey, size=num_prey, p=P)
    return parents


def phenotype_crossover(phenotypes, parents, phenotype_type='vector'):

    # phenotypes can be signals or detectors
    assert len(parents) % 2 == 0, 'Crossover not implemented yet for odd numbers of parents'
    parent_phenotypes = phenotypes[parents]
    child_phenotypes = np.zeros_like(parent_phenotypes)
    match phenotype_type:
        case 'vector':
            interpolation_values = np.random.rand(parent_phenotypes.shape[0] // 2)[:, np.newaxis]
            child_phenotypes[::2]  = interpolation_values * parent_phenotypes[::2] \
                                    + (1 - interpolation_values) * parent_phenotypes[1::2]
            child_phenotypes[1::2] = (1 - interpolation_values) * parent_phenotypes[::2] \
                                    + interpolation_values * parent_phenotypes[1::2]
            return child_phenotypes
        case 'bitstring':
            raise NotImplementedError
        case _:
            raise NotImplementedError
        

def phenotype_mutate(phenotypes, mutation_rate=0.01, phenotype_type='vector'):
    """
    Apply random mutations to the bit strings or numerical values.
    """
    num_individuals, d = phenotypes.shape

    match phenotype_type:
        case 'vector':
            # noise = np.random.normal(scale=mutation_rate, size=(num_individuals, d))  # scale mutation rate with d?
            noise = np.random.multivariate_normal(mean=np.zeros(d), cov=mutation_rate*np.eye(d), size=num_individuals)
            return phenotypes + noise
        case 'bitstring':
            # For bit strings, flip bits with some probability
            raise NotImplementedError
        case _:
            raise NotImplementedError


def update(detectors, signals, risk_tols, venom_levels, num_venomous, mutation_rate=0.01, phenotype_type='vector'):


    assert np.all(venom_levels[num_venomous:] == 0), 'A mimic has a non-zero venom level'

    venomous_signals, mimic_signals = np.split(signals, [num_venomous])

    predation_matrix = calculate_predation_matrix(detectors, signals, risk_tols, phenotype_type=phenotype_type)

    predator_parents = sample_predators(predation_matrix, venom_levels)
    venomous_parents = sample_prey(predation_matrix[:, :num_venomous])
    mimic_parents    = sample_prey(predation_matrix[:, num_venomous:])

    predator_childrens_detectors = phenotype_crossover(detectors, predator_parents, phenotype_type=phenotype_type)
    venomous_childrens_signals = phenotype_crossover(venomous_signals, venomous_parents, phenotype_type=phenotype_type)
    mimic_childrens_signals    = phenotype_crossover(mimic_signals, mimic_parents, phenotype_type=phenotype_type)

    predator_childrens_detectors = phenotype_mutate(predator_childrens_detectors, mutation_rate=mutation_rate, phenotype_type=phenotype_type)
    venomous_childrens_signals = phenotype_mutate(venomous_childrens_signals, mutation_rate=mutation_rate, phenotype_type=phenotype_type)
    mimic_childrens_signals = phenotype_mutate(mimic_childrens_signals, mutation_rate=mutation_rate, phenotype_type=phenotype_type)

    prey_childrens_signals = np.vstack((venomous_childrens_signals, mimic_childrens_signals))

    # TODO: make risk tolerances update via crossover and mutation
    predator_childrens_risk_tols = risk_tols
    # TODO: make venom levels update via crossover and mutation
    prey_childrens_venoms = venom_levels

    return predator_childrens_detectors, prey_childrens_signals, predator_childrens_risk_tols, prey_childrens_venoms