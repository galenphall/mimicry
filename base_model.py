import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

def generate(num_predators, num_venomous_prey, num_mimics, dim=2, venom_const=0.5, risk_tol_scale=0.1):
    """
    Generate initial populations of predators, venomous prey, and mimics.
    
    Args:
    num_predators (int): Number of predators
    num_venomous_prey (int): Number of venomous prey
    num_mimics (int): Number of mimics
    dim (int): Dimensionality of the phenotype space
    venom_const (float): Constant venom level for venomous prey
    risk_tol_scale (float): Scale parameter for predators' risk tolerance
    
    Returns:
    tuple: Detectors, signals, risk tolerances, and venom levels
    """
    
    # Generate random initial positions for predators, venomous prey, and mimics
    detectors = np.random.uniform(-2, 2, size=(num_predators, dim))
    venomous_signals = np.random.uniform(-2, 2, size=(num_venomous_prey, dim))
    mimic_signals = np.random.uniform(-2, 2, size=(num_mimics, dim))
    
    # Combine venomous and mimic signals
    signals = np.vstack((venomous_signals, mimic_signals))
    
    # Generate risk tolerances for predators
    risk_tols = np.random.exponential(risk_tol_scale=0.1, size=num_predators)
    
    # Set venom levels: constant for venomous prey, zero for mimics
    venom_levels = np.concatenate((np.zeros(num_venomous_prey) + venom_const, np.zeros(num_mimics)))

    return detectors, signals, risk_tols, venom_levels

def similarity(detectors, signals, phenotype_type='vector', periodic_boundary=None):
    """
    Calculate similarity between detectors and signals.
    
    Args:
    detectors (np.array): Predator detectors
    signals (np.array): Prey signals
    phenotype_type (str): Type of phenotype representation
    periodic_boundary (float): Boundary for periodic conditions
    
    Returns:
    np.array: Similarity matrix
    """
    if phenotype_type == 'vector':
        if periodic_boundary is None:
            dist = np.linalg.norm(detectors[:, np.newaxis] - signals, axis=2)
        else:
            dist = pairwise_periodic_distances_optimized(detectors, signals, periodic_boundary)
        return - dist**2
    elif phenotype_type == 'bitstring':
        d = signals.shape[1]
        hamming_distances = np.sum(detectors[:, np.newaxis] != signals, axis=2)
        return 1 - hamming_distances/d
    else:
        raise NotImplementedError

def pairwise_periodic_distances_optimized(v1, v2, b):
    """
    Calculate pairwise distances with periodic boundary conditions.
    
    Args:
    v1 (np.array): First set of vectors
    v2 (np.array): Second set of vectors
    b (float): Boundary value (boundaries at [x, y] = ±b)
    
    Returns:
    np.array: Pairwise distances
    """
    v1 = np.array(v1)[:, np.newaxis, :]
    v2 = np.array(v2)[np.newaxis, :, :]
    
    diff = np.abs(v1 - v2)
    adjusted_diff = np.minimum(diff, 2*b - diff)
    
    distances = np.sqrt(np.sum(adjusted_diff**2, axis=2))
    
    return distances

def calculate_preference_matrix(detectors, signals, risk_tols, phenotype_type='vector', periodic_boundary=None):
    """
    Calculate preference matrix based on similarity and risk tolerance.
    
    Args:
    detectors (np.array): Predator detectors
    signals (np.array): Prey signals
    risk_tols (np.array): Predator risk tolerances
    phenotype_type (str): Type of phenotype representation
    periodic_boundary (float): Boundary for periodic conditions
    
    Returns:
    np.array: Preference matrix
    """
    similarity_matrix = similarity(detectors, signals, phenotype_type=phenotype_type, periodic_boundary=periodic_boundary) 
    return 1 - np.exp(similarity_matrix / risk_tols[:, np.newaxis]**2)

def calculate_predation_matrix(detectors, signals, risk_tols, handling_time, 
                               attack_freq, R, phenotype_type='vector', 
                               periodic_boundary=None):
    """
    Calculate predation matrix based on preferences and other factors.
    
    Args:
    detectors (np.array): Predator detectors
    signals (np.array): Prey signals
    risk_tols (np.array): Predator risk tolerances
    handling_time (float): Time taken to handle prey
    attack_freq (float): Frequency of attacks
    R (float): Resource availability
    phenotype_type (str): Type of phenotype representation
    periodic_boundary (float): Boundary for periodic conditions
    
    Returns:
    tuple: Predation matrix and effective number of prey
    """
    preference_matrix = calculate_preference_matrix(detectors, signals, risk_tols, phenotype_type, periodic_boundary)
    n_predators, n_prey = preference_matrix.shape
    n_effective_prey = preference_matrix.sum(1) + R
    intake_rates = attack_freq / (1 + n_predators + attack_freq * handling_time * n_effective_prey)
    return intake_rates[:, None] * preference_matrix, n_effective_prey

def sample_predators(predation_matrix, venom_levels, pred_conversion_ratio, attack_rate, handling_time, R, n_effective_prey, death_rate=0.5):
    """
    Sample the next generation of predators based on their fitness.
    
    Args:
    predation_matrix (np.array): Matrix of predation rates
    venom_levels (np.array): Venom levels of prey
    pred_conversion_ratio (float): Conversion ratio for predators
    attack_rate (float): Rate of attacks
    handling_time (float): Time taken to handle prey
    R (float): Resource availability
    n_effective_prey (np.array): Effective number of prey
    death_rate (float): Natural death rate of predators
    
    Returns:
    np.array: Indices of predators in the next generation
    """
    num_predators = predation_matrix.shape[0]
    fitnesses = (predation_matrix * (1 - venom_levels) * pred_conversion_ratio - predation_matrix * venom_levels).sum(1)
    fitnesses += 1 - death_rate 
    fitnesses += attack_rate * R / (1 + num_predators + attack_rate * handling_time * n_effective_prey) * pred_conversion_ratio
    means = fitnesses
    means[means < 0] = 0
    counts = np.random.poisson(means)
    return np.repeat(np.arange(num_predators), counts)

def sample_prey(predation_matrix, popcap, venom_levels, r=0.6):
    """
    Sample the next generation of prey based on their fitness.
    
    Args:
    predation_matrix (np.array): Matrix of predation rates
    popcap (float): Population capacity for prey
    venom_levels (np.array): Venom levels of prey
    r (float): Intrinsic growth rate of prey
    
    Returns:
    np.array: Indices of prey in the next generation
    """
    nv = (venom_levels > 0).sum()
    nm = (venom_levels == 0).sum()
    num_prey = (venom_levels > 0) * nv + (venom_levels == 0) * nm
    fitnesses = r * (1 - num_prey / popcap) - predation_matrix.sum(0)
    means = fitnesses.copy()
    means[means < 0] = 0
    counts = np.random.poisson(means)
    return np.repeat(np.arange(nv + nm), counts)

def phenotype_mutate(phenotypes, mutation_rate=0.01, phenotype_type='vector'):
    """
    Mutate phenotypes.
    
    Args:
    phenotypes (np.array): Current phenotypes
    mutation_rate (float): Rate of mutation
    phenotype_type (str): Type of phenotype representation
    
    Returns:
    np.array: Mutated phenotypes
    """
    if phenotype_type == 'vector':
        return phenotypes + np.random.normal(scale=mutation_rate, size=phenotypes.shape)
    elif phenotype_type == 'bitstring':
        raise NotImplementedError
    else:
        raise NotImplementedError

def impose_periodic_boundary(vectors, boundary=5):
    """
    Impose periodic boundary conditions on vectors.
    
    Args:
    vectors (np.array): Array of vectors
    boundary (float): Boundary value
    
    Returns:
    np.array: Vectors with periodic boundary conditions applied
    """
    if boundary is None:
        return vectors
    
    vectors = np.array(vectors)
    vectors = np.mod(vectors + boundary, 2 * boundary) - boundary
    
    return vectors

def update(detectors, signals, risk_tols, venom_levels, num_venomous, R, r_R, k_R,
           r_prey, handling_time, attack_rate, predator_conversion_ratio, prey_popcap,
           mutation_rate, phenotype_type='vector', periodic_boundary=None, mutate_venom=False, mutate_risk=False):
    """
    Update the simulation for one generation.
    
    Args:
    detectors (np.array): Predator detectors
    signals (np.array): Prey signals
    risk_tols (np.array): Predator risk tolerances
    venom_levels (np.array): Venom levels of prey
    num_venomous (int): Number of venomous prey
    R (float): Resource availability
    r_R (float): Growth rate of resource
    k_R (float): Carrying capacity of resource
    r_prey (float): Intrinsic growth rate of prey
    handling_time (float): Time taken to handle prey
    attack_rate (float): Rate of attacks
    predator_conversion_ratio (float): Conversion ratio for predators
    prey_popcap (float): Population capacity for prey
    mutation_rate (float): Rate of mutation
    phenotype_type (str): Type of phenotype representation
    periodic_boundary (float): Boundary for periodic conditions
    mutate_venom (bool): Whether to mutate venom levels
    mutate_risk (bool): Whether to mutate risk tolerances
    
    Returns:
    tuple: Updated populations and parameters for the next generation
    """
    
    # Calculate predation matrix
    predation_matrix, n_effective_prey = calculate_predation_matrix(
        detectors, signals, risk_tols,                  
        handling_time,                  
        attack_rate,       
        R,
        phenotype_type=phenotype_type, 
        periodic_boundary=periodic_boundary
    )

    # Sample next generation of predators and prey
    predator_children = sample_predators(
        predation_matrix, 
        venom_levels, 
        predator_conversion_ratio, 
        attack_rate, 
        handling_time, 
        R,
        n_effective_prey
    )
    
    prey_children = sample_prey(
        predation_matrix, 
        prey_popcap, 
        venom_levels,
        r_prey
    )

    # Update resource availability
    num_predators = predation_matrix.shape[0]
    delta_R = (attack_rate * R / (1 + num_predators + attack_rate * handling_time * n_effective_prey)).sum()
    R += r_R * R * (1 - R / k_R) - delta_R
    R = max(R, 0)

    # Get phenotypes of children
    predator_childrens_detectors = detectors[predator_children]
    prey_childrens_signals = signals[prey_children]
    
    # Mutate phenotypes
    predator_childrens_detectors = phenotype_mutate(
        predator_childrens_detectors, 
        mutation_rate=mutation_rate, 
        phenotype_type=phenotype_type)
    
    prey_childrens_signals = phenotype_mutate(
        prey_childrens_signals, 
        mutation_rate=mutation_rate, 
        phenotype_type=phenotype_type)

    # Apply periodic boundary conditions
    predator_childrens_detectors = impose_periodic_boundary(predator_childrens_detectors, periodic_boundary)
    prey_childrens_signals = impose_periodic_boundary(prey_childrens_signals, periodic_boundary)

    # Optionally mutate risk tolerances and venom levels
    if mutate_risk:
        predator_childrens_risk_tols = phenotype_mutate(
            risk_tols[predator_children], 
            mutation_rate=mutation_rate, 
            phenotype_type=phenotype_type)
        predator_childrens_risk_tols = abs(predator_childrens_risk_tols)
    else:
        predator_childrens_risk_tols = risk_tols[predator_children]

    if mutate_venom:
        prey_childrens_venoms = phenotype_mutate(
            venom_levels[prey_children], 
            mutation_rate=mutation_rate, 
            phenotype_type=phenotype_type)
        
        prey_childrens_venoms[prey_childrens_venoms > 0.9999] = 0.9999
        prey_childrens_venoms[prey_childrens_venoms < 0.0001] = 0.0001
        prey_childrens_venoms[venom_levels[prey_children] == 0] = 0
    else:
        prey_childrens_venoms = venom_levels[prey_children]

    new_num_venomous = (prey_childrens_venoms > 0).sum()

    return predator_childrens_detectors, prey_childrens_signals, predator_childrens_risk_tols, prey_childrens_venoms, new_num_venomous, R


""" Example Initialization
# Initialize population parameters
scale = 100
num_predators = 1 * scale
num_venomous_prey = 1 * scale
num_mimics = 2 * scale
d = 2
venomosity = 0.4
R = 5 * scale
r_R = 1
k_R = 5 * scale

r_prey = 2

num_generations = 5000
ht = 0.3
af = 0.3
mutation_rate = 0.01
predator_conversion_ratio = 0.85
prey_popcap = 10 * scale
periodic_boundary = 2

np.random.seed(1)
"""

""" Example run
detectors, signals, risk_tols, venom_levels = generate(num_predators, num_venomous_prey, num_mimics, d, venomosity)

detectors_history    = []
signals_history      = []
risk_tols_history    = []
venom_levels_history = []
nv_history = []
R_history = []

detectors_history.append(detectors)
signals_history.append(signals)
risk_tols_history.append(risk_tols)
venom_levels_history.append(venom_levels)
nv_history.append(num_venomous_prey)

for t in tqdm(range(num_generations)):
    
    d, s, r, v, nv, R = update(
        detectors_history[t], 
        signals_history[t], 
        risk_tols_history[t], 
        venom_levels_history[t], 
        nv_history[t], 
        R, r_R, k_R,
        r_prey,
        handling_time = ht,
        attack_rate=af,
        predator_conversion_ratio=predator_conversion_ratio,
        prey_popcap=prey_popcap,
        mutation_rate=mutation_rate, 
        phenotype_type='vector',
        periodic_boundary=periodic_boundary,
    )
    
    detectors_history.append(d)
    signals_history.append(s)
    risk_tols_history.append(r)
    venom_levels_history.append(v)
    nv_history.append(nv)
    R_history.append(R)
"""