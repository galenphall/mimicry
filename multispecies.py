import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, poisson
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

def generate(species_params):
    """
    Generate initial populations of predators and prey for multiple species.

    Parameters:
    species_params (dict): A dictionary containing parameters for each species
        {
            'predators': [{'num': int, 'detector_mean': array, 'detector_cov': array, 'risk_tol_mean': float}, ...],
            'prey': [{'num': int, 'signal_mean': array, 'signal_cov': array, 'venom_level': float}, ...]
        }

    Returns:
    tuple: (detectors, signals, risk_tols, venom_levels, species_indices)
        detectors (ndarray): Array of predator detectors
        signals (ndarray): Array of prey signals
        risk_tols (ndarray): Array of predator risk tolerances
        venom_levels (ndarray): Array of venom levels for all prey
        species_indices (dict): Dictionary mapping species to their indices in the arrays
    """
    predator_detectors = []
    predator_risk_tols = []
    prey_signals = []
    prey_venom_levels = []
    species_indices = {'predators': {}, 'prey': {}}
    
    start_idx = 0
    for i, pred in enumerate(species_params['predators']):
        detectors = np.random.multivariate_normal(mean=pred['detector_mean'],
                                                  cov=pred['detector_cov'],
                                                  size=pred['num'])
        risk_tols = np.random.exponential(scale=pred['risk_tol_mean'], size=pred['num'])
        predator_detectors.append(detectors)
        predator_risk_tols.append(risk_tols)
        species_indices['predators'][i] = (start_idx, start_idx + pred['num'])
        start_idx += pred['num']
    
    start_idx = 0
    for i, prey in enumerate(species_params['prey']):
        signals = np.random.multivariate_normal(mean=prey['signal_mean'],
                                                cov=prey['signal_cov'],
                                                size=prey['num'])
        venom_levels = np.full(prey['num'], prey['venom_level'])
        prey_signals.append(signals)
        prey_venom_levels.append(venom_levels)
        species_indices['prey'][i] = (start_idx, start_idx + prey['num'])
        start_idx += prey['num']
    
    return (np.vstack(predator_detectors), np.vstack(prey_signals),
            np.concatenate(predator_risk_tols), np.concatenate(prey_venom_levels),
            species_indices)

def similarity(detectors, signals, phenotype_type='vector', periodic_boundary=None):
    match phenotype_type:
        case 'vector':
            if periodic_boundary is None:
                dist = np.linalg.norm(detectors[:, np.newaxis] - signals, axis=2)
            else:
                dist = pairwise_periodic_distances_optimized(detectors, signals, periodic_boundary)
            return - dist**2
        case 'bitstring':
            d = signals.shape[1]
            hamming_distances = np.sum(detectors[:, np.newaxis] != signals, axis=2)
            return 1 - hamming_distances/d
        case _:
            raise NotImplementedError

def periodic_distance(x1, y1, x2, y2, b):
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.minimum(dx, 2*b - dx)
    dy = np.minimum(dy, 2*b - dy)
    return np.sqrt(dx**2 + dy**2)

def pairwise_periodic_distances(v1, v2, b):
    distances = np.zeros((len(v1), len(v2)))
    for i, (x1, y1) in enumerate(v1):
        for j, (x2, y2) in enumerate(v2):
            distances[i, j] = periodic_distance(x1, y1, x2, y2, b)
    return distances

def pairwise_periodic_distances_optimized(v1, v2, b):
    # Reshape v1 and v2 to allow broadcasting
    v1 = np.array(v1)[:, np.newaxis, :]
    v2 = np.array(v2)[np.newaxis, :, :]
    
    # Calculate differences
    diff = np.abs(v1 - v2)
    
    # Apply periodic boundary conditions
    adjusted_diff = np.minimum(diff, 2*b - diff)
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum(adjusted_diff**2, axis=2))
    
    return distances

def phenotype_mutate(phenotypes, mutation_rate=0.01, phenotype_type='vector'):
    match phenotype_type:
        case 'vector':
            return phenotypes + np.random.normal(scale=mutation_rate, size=phenotypes.shape)
        case 'bitstring':
            raise NotImplementedError
        case _:
            raise NotImplementedError

def impose_periodic_boundary(vectors, boundary=5):
    """
    Imposes periodic boundary conditions on an array of 2D vectors.
    
    Parameters:
    vectors (np.array): Array of shape (n, 2) containing 2D vectors.
    boundary (float): The boundary value for both x and y dimensions. Default is 5.
    
    Returns:
    np.array: Array of shape (n, 2) with periodic boundary conditions applied.
    """
    if boundary is None:
        return vectors
    
    # Ensure the input is a numpy array
    vectors = np.array(vectors)
    
    # Apply periodic boundary conditions
    vectors = np.mod(vectors + boundary, 2 * boundary) - boundary
    
    return vectors

def calculate_preference_matrix(detectors, signals, risk_tols, phenotype_type='vector', periodic_boundary=None):
    """
    Calculate preference matrix based on similarity and risk tolerance.

    Parameters:
    detectors (ndarray): Array of predator detectors
    signals (ndarray): Array of prey signals
    risk_tols (ndarray): Array of predator risk tolerances
    phenotype_type (str): Type of phenotype representation ('vector' or 'bitstring')
    periodic_boundary (float): Boundary for periodic distance calculation (if None, use Euclidean distance)

    Returns:
    ndarray: Preference matrix
    """
    similarity_matrix = similarity(detectors, signals, phenotype_type=phenotype_type, periodic_boundary=periodic_boundary) 
    return 1 - np.exp(similarity_matrix / risk_tols[:, np.newaxis])

def calculate_predation_matrix(detectors, signals, risk_tols, handling_times, attack_freqs, phenotype_type='vector', periodic_boundary=None):
    """
    Calculate predation matrix based on preference and predator characteristics for multiple species.

    Parameters:
    detectors (ndarray): Array of predator detectors
    signals (ndarray): Array of prey signals
    risk_tols (ndarray): Array of predator risk tolerances
    handling_times (ndarray): Array of handling times for each predator
    attack_freqs (ndarray): Array of attack frequencies for each predator
    phenotype_type (str): Type of phenotype representation ('vector' or 'bitstring')
    periodic_boundary (float): Boundary for periodic distance calculation (if None, use Euclidean distance)

    Returns:
    ndarray: Predation matrix
    """
    preference_matrix = calculate_preference_matrix(detectors, signals, risk_tols, phenotype_type, periodic_boundary)
    n_prey = preference_matrix.shape[1]
    n_effective_prey = preference_matrix.sum(1)
    intake_rates = attack_freqs / (1 + attack_freqs * handling_times * n_effective_prey)
    return intake_rates[:, None] * preference_matrix

def sample_predators(predation_matrix, venom_levels, pred_conversion_ratios, death_rates, species_indices):
    """
    Sample predators based on predation success and venom effects for multiple species.

    Parameters:
    predation_matrix (ndarray): Matrix of predation rates
    venom_levels (ndarray): Array of venom levels for prey
    pred_conversion_ratios (ndarray): Array of conversion ratios for each predator species
    species_indices (dict): Dictionary mapping species to their indices in the arrays

    Returns:
    ndarray: Indices of sampled predators
    """
    fitnesses = (predation_matrix * (1 - venom_levels) * pred_conversion_ratios[:, None] - predation_matrix * venom_levels).sum(1)
    fitnesses += 1 - death_rates[:, None]
    means = np.maximum(fitnesses, 0)
    counts = np.random.poisson(means)
    return np.repeat(np.arange(len(counts)), counts)

def sample_prey(predation_matrix, popcaps, venom_levels, species_indices):
    """
    Sample prey based on predation pressure and population cap for multiple species.

    Parameters:
    predation_matrix (ndarray): Matrix of predation rates
    popcaps (ndarray): Array of population caps for each prey species
    venom_levels (ndarray): Array of venom levels for prey
    species_indices (dict): Dictionary mapping species to their indices in the arrays

    Returns:
    ndarray: Indices of sampled prey
    """
    species_sizes = np.array([indices[1] - indices[0] for indices in species_indices['prey'].values()])
    fitnesses = np.zeros(len(venom_levels))
    for i, (start, end) in species_indices['prey'].items():
        fitnesses[start:end] = popcaps[i] / species_sizes[i] * np.prod(1 - predation_matrix[:, start:end], 0)
    
    means = np.maximum(fitnesses, 0)
    counts = np.random.poisson(means)
    return np.repeat(np.arange(len(counts)), counts)

def update(detectors, signals, risk_tols, venom_levels, species_indices,
           handling_times, attack_freqs, predator_conversion_ratios, predator_death_rates, prey_popcaps,
           mutation_rates, phenotype_type='vector', periodic_boundary=None):
    """
    Update the population of predators and prey for one generation with multiple species.

    Parameters:
    detectors (ndarray): Array of predator detectors
    signals (ndarray): Array of prey signals
    risk_tols (ndarray): Array of predator risk tolerances
    venom_levels (ndarray): Array of venom levels for prey
    species_indices (dict): Dictionary mapping species to their indices in the arrays
    handling_times (ndarray): Array of handling times for each predator species
    attack_freqs (ndarray): Array of attack frequencies for each predator species
    predator_conversion_ratios (ndarray): Array of conversion ratios for each predator species
    predator_death_rates (ndarray): Array of death rates for each predator species
    prey_popcaps (ndarray): Array of population caps for each prey species
    mutation_rates (dict): Dictionary of mutation rates for each species type ('predators' and 'prey')
    phenotype_type (str): Type of phenotype representation ('vector' or 'bitstring')
    periodic_boundary (float): Boundary for periodic distance calculation (if None, use Euclidean distance)

    Returns:
    tuple: (new_detectors, new_signals, new_risk_tols, new_venom_levels, new_species_indices)
        new_detectors (ndarray): Updated array of predator detectors
        new_signals (ndarray): Updated array of prey signals
        new_risk_tols (ndarray): Updated array of predator risk tolerances
        new_venom_levels (ndarray): Updated array of venom levels for prey
        new_species_indices (dict): Updated dictionary mapping species to their indices in the arrays
    """
    predation_matrix = calculate_predation_matrix(detectors, signals, risk_tols, 
                                                  handling_times, attack_freqs, 
                                                  phenotype_type=phenotype_type, 
                                                  periodic_boundary=periodic_boundary)

    predator_children = sample_predators(predation_matrix, venom_levels, predator_conversion_ratios, predator_death_rates, species_indices)
    prey_children = sample_prey(predation_matrix, prey_popcaps, venom_levels, species_indices)
    
    predator_children_detectors = detectors[predator_children]
    prey_children_signals = signals[prey_children]
    
    predator_childrens_detectors = phenotype_mutate(predator_children_detectors, mutation_rate=mutation_rates['predators'], phenotype_type=phenotype_type)
    prey_childrens_signals = phenotype_mutate(prey_children_signals, mutation_rate=mutation_rates['prey'], phenotype_type=phenotype_type)

    predator_childrens_detectors = impose_periodic_boundary(predator_childrens_detectors, periodic_boundary)
    prey_childrens_signals = impose_periodic_boundary(prey_childrens_signals, periodic_boundary)
    
    predator_childrens_risk_tols = phenotype_mutate(risk_tols[predator_children], mutation_rate=mutation_rates['predators'], phenotype_type=phenotype_type)
    predator_childrens_risk_tols = abs(predator_childrens_risk_tols)

    prey_childrens_venoms = venom_levels[prey_children]

    # Update species_indices
    new_species_indices = {'predators': {}, 'prey': {}}
    pred_start = 0
    for i, (start, end) in species_indices['predators'].items():
        new_end = pred_start + np.sum((predator_children >= start) & (predator_children < end))
        new_species_indices['predators'][i] = (pred_start, new_end)
        pred_start = new_end

    prey_start = 0
    for i, (start, end) in species_indices['prey'].items():
        new_end = prey_start + np.sum((prey_children >= start) & (prey_children < end))
        new_species_indices['prey'][i] = (prey_start, new_end)
        prey_start = new_end

    return (predator_childrens_detectors, prey_childrens_signals, 
            predator_childrens_risk_tols, prey_childrens_venoms, 
            new_species_indices)