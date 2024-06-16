"""
Code to simulate mimicry in a simple three-species system using a genetic algorithm.
There are three populations:
    - Predators (P)
    Predators each have a detection bit string `d` and a risk tolerance `r` in [0, 1]
    - Venomous prey (V)
    Venomous prey each have a signal bit string `s` and a venom level `v` in [0, 1]
    - Mimic prey (M)
    Mimics each have a signal bit string `s`

In each generation, a reproduction pool is selected from each population {P, V, M}
according to the probability of a given individual's survival. These probabilities are 
calculated using the predation matrix A, where A_{ij} gives the probability that 
predator i eats prey j, where j is in V or M. It is calculated as 

    A_{ij} = exp{-S_{ij}/r_i} / \sum_k exp{-S_{ij}/r_i}

Where S_ij is the similarity between d_i and s_j, and r_i is the risk tolerance of predator i.

Prey are sampled proportionally to the probability they are not eaten, i.e.

    P_j \propto \prod_i (1 - A_{ij})

Predators are sampled proportionally to the probability they die from eating a venomous prey:

    1 - P_i \propto \sum_j A_{ij} v_j

After sampling, the next generation is created using the standard crossover and random mutations.

TODO: Check the predation matrix calculation for a sign flip. [Galen]
TODO: Test with no mimics and fixed venomosity and risk tolerance -- the venomous phenotypes should converge.
TODO: Add in option to hardcode v, r
TODO: Add in code to generate similar starting bitstrings for each population
TODO: Bayesian updating [Sagar]
TODO: Implement genotype bitstrings and mapping (random?) to phenotypes [Ravi]
"""
import numpy as np
from tqdm import tqdm
import datetime 
import zarr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def update(
    s, d, r, v, num_venomous_prey, num_mimics, 
    rates={'s': 0.01, 'r': 0.01, 'v': 0.01, 'd': 0.01},
    hardcode={'s': None, 'r': None, 'v': None, 'd': None}):
    """
    Update the populations of predators, venomous prey, and mimics.
    """
    # Split s into venomous prey signals and mimic signals
    s_v = s[:num_venomous_prey]

    # Calculate the similarity between the detection and signal bit strings
    S_v = similarity(s_v, d)

    # Calculate the predation matrix for venomous prey and mimics
    A_v = np.exp(-S_v / r[:, None])
        
    # Normalize the predation matrices
    A_v /= np.sum(A_v, axis=1, keepdims=True)

    # Check for NaN values in A_v and A_m
    if np.any(np.isnan(A_v)):
        print("NaN values detected in A_v:", A_v)
        A_v = np.nan_to_num(A_v)

    # Sample the prey populations
    prey_v = sample_prey(A_v)

    # If there are mimics, calculate the predation matrix for mimics and 
    # sample the mimic prey population
    if num_mimics > 0:
        s_m = s[num_venomous_prey:]
        S_m = similarity(s_m, d)
        A_m = np.exp(-S_m / r[:, None])
        A_m /= np.sum(A_m, axis=1, keepdims=True)
        if np.any(np.isnan(A_m)):
            print("NaN values detected in A_m:", A_m)
            A_m = np.nan_to_num(A_m)
        prey_m = sample_prey(A_m)
        new_s_m = crossover_bitstring(s_m, prey_m)
        new_s_m = mutate(new_s_m)

    # Sample the predator population
    predators = sample_predators(A_v, v)

    # Create the next generation for venomous prey and mimics
    new_s_v = crossover_bitstring(s_v, prey_v)
    new_s_v = mutate(new_s_v)
    
    # Combine the new signals back into one array
    if num_mimics > 0:
        new_s = np.vstack((new_s_v, new_s_m))
    else:
        new_s = new_s_v

    # Create the next generation d, r, v values
    # Crossover and mutate the detection bit strings
    new_d = crossover_bitstring(d, predators)
    new_d = mutate(new_d)

    # Crossover and mutate the risk tolerance values
    new_r = crossover_realvalued(r, predators)
    new_r = mutate(new_r)

    # Crossover and mutate the venom levels
    new_v = crossover_realvalued(v, prey_v)
    new_v = mutate(new_v)

    if hardcode['v']:
        new_v = np.fill_like(new_v, hardcode['v'])

    if hardcode['r']:
        new_r = np.fill_like(new_r, hardcode['r'])

    return new_s, new_d, new_r, new_v


def similarity(s, d):
    """
    Calculate the mutual information between two bit strings or two vectors.
    Parameters:
    s: np.ndarray
        The first bit string or vector (or matrix of bit strings or vectors)
    d: np.ndarray
        The second bit string or vector (or matrix of bit strings or vectors)
    """
    if s.shape[1] != d.shape[1]:
        raise ValueError("Bit strings must have the same length.")

    assert s.ndim == d.ndim, "s and d must have the same number of dimensions."
    
    # For vectors, calculate cosine similarity
    if s.dtype == np.float64:
        return np.dot(s, d.T) / (np.linalg.norm(s) * np.linalg.norm(d, axis=1))
    # For bit strings, calculate the hamming distance and normalize
    elif s.dtype == np.int64:
        return np.sum(s == d, axis=1) / s.shape[1]
    else:
        raise ValueError("Data type not supported.")

def sample_prey(A):
    """
    Sample the prey population based on the predation matrix.
    """
    P = np.prod(1 - A, axis=0)
    P /= P.sum()  # Normalize the probabilities
    return np.random.choice(len(P), size=len(P), p=P)

def sample_predators(A, v):
    """
    Sample the predator population based on the predation matrix and venom levels.
    """
    P = np.sum(A * v, axis=1)
    P /= P.sum()  # Normalize the probabilities
    return np.random.choice(len(P), size=len(P), p=P)

def crossover_bitstring(s, prey):
    """
    Perform random crossover on the signal bit strings of the prey population
    using the Genetic Algorithm.
    """
    num_offspring = len(prey)
    new_s = np.zeros_like(s)

    for i in range(0, num_offspring, 2):
        # Select parents
        parent1 = s[prey[i]]
        parent2 = s[prey[i + 1 if i + 1 < num_offspring else 0]]

        # Choose crossover point
        crossover_point = np.random.randint(1, len(parent1))

        # Create offspring
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        # Add offspring to the new population
        new_s[i] = offspring1
        if i + 1 < num_offspring:
            new_s[i + 1] = offspring2

    return new_s


def crossover_realvalued(x, prey):
    """
    Perform random crossover on the real-valued vectors of the prey population
    by picking a random value r and setting the offspring of a, b to be the sum
    r * a + (1 - r) * b and (1 - r) * a + r * b, respectively.
    """
    num_offspring = len(prey)
    new_x = np.zeros_like(x)

    for i in range(0, num_offspring, 2):
        # Select parents
        parent1 = x[prey[i]]
        parent2 = x[prey[i + 1 if i + 1 < num_offspring else 0]]

        # Choose crossover point
        r = np.random.rand()

        # Create offspring
        offspring1 = r * parent1 + (1 - r) * parent2
        offspring2 = (1 - r) * parent1 + r * parent2

        # Add offspring to the new population
        new_x[i] = offspring1
        if i + 1 < num_offspring:
            new_x[i + 1] = offspring2

    return new_x



def mutate(x, mutation_rate=0.01):
    """
    Apply random mutations to the bit strings or numerical values.
    """
    if x.dtype == np.float64:
        # For numerical values, apply small random changes to the logit-transformed values
        # Then transform back to the original scale
        logit_x = np.log(x / (1 - x))
        mutations = np.random.normal(0, mutation_rate, x.shape)
        logit_x += mutations
        x = 1 / (1 + np.exp(-logit_x))
    else:
        # For bit strings, flip bits with some probability
        mutations = np.random.rand(*x.shape) < mutation_rate
        x = np.where(mutations, 1 - x, x)  # Flip bits
    return x

# Initialize population parameters
num_predators = 100
num_venomous_prey = 100
num_mimics = 0
bit_string_length = 5

# Randomly initialize the populations
s = np.random.randint(2, size=(num_venomous_prey + num_mimics, bit_string_length))
d = np.random.randint(2, size=(num_predators, bit_string_length))
r = np.random.rand(num_predators) / 8
v = np.random.rand(num_venomous_prey) / 4

# Run the algorithm for many generations
num_generations = 10000

# Keep track of transient data for analysis
transient_data = {
    's': [],
    'd': [],
    'r': [],
    'v': [],
}

for generation in tqdm(range(num_generations)):
    # Save transient data
    transient_data['s'].append(s.copy())
    transient_data['d'].append(d.copy())
    transient_data['r'].append(r.copy())
    transient_data['v'].append(v.copy())

    s, d, r, v = update(s, d, r, v, num_venomous_prey, num_mimics)

# Final population values
print("Final signals (s):", s)
print("Final detections (d):", d)
print("Final risk tolerances (r):", r)
print("Final venom levels (v):", v)

# Save the final populations to a zarr file
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"data/mimicry_{date}.zarr"
zarr.save(filename, **transient_data)



    