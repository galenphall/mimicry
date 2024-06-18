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
import sys

def update(
    s, d, r, v, num_venomous_prey, num_mimics, 
    rates={'s': 0.01, 'r': 0.01, 'v': 0.01, 'd': 0.01},
    hardcode={}):
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
    new_s_v = mutate01(new_s_v)
    
    # Combine the new signals back into one array
    if num_mimics > 0:
        new_s = np.vstack((new_s_v, new_s_m))
    else:
        new_s = new_s_v

    # Create the next generation d, r, v values
    # Crossover and mutate the detection bit strings
    new_d = crossover_bitstring(d, predators)
    new_d = mutate01(new_d)

    if 'v' in hardcode:
        new_v = hardcode['v']
    else:
        new_v = crossover_realvalued(v, prey_v)
        new_v = mutate01(new_v, 0.1)

    if 'r' in hardcode:
        new_r = hardcode['r']
    else:
        new_r = crossover_realvalued(r, predators)
        new_r = mutate0inf(new_r)


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
    P = np.sum(A * v, axis=1) # Probability predator i dies.
    P = 1 - P # Probability predator i survives.
    P /= P.sum()  # Normalize the probabilities.
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



def mutate01(x, mutation_rate=0.01):
    """
    Apply random mutations to the bit strings or numerical values in the [0, 1] range.
    """
    if x.dtype == np.float64:
        # For numerical values, apply small random changes to the logit-transformed values
        # Then transform back to the original scale
        
        # if x is 0 or 1, the logit transform is undefined, so we add a small value to avoid this
        x = np.clip(x, 1e-6, 1 - 1e-6)
        logit_x = np.log(x / (1 - x)) 
        mutations = np.random.normal(0, mutation_rate, x.shape)
        logit_x += mutations
        x = 1 / (1 + np.exp(-logit_x))
    else:
        # For bit strings, flip bits with some probability
        mutations = np.random.rand(*x.shape) < mutation_rate
        x = np.where(mutations, 1 - x, x)  # Flip bits
    return x

def mutate0inf(x, mutation_rate=0.01):
    """
    Apply random mutations to the real-valued vectors in the [0, inf] range.
    """
    mutations = np.random.normal(0, mutation_rate, x.shape)
    x += mutations
    return x

def save_data(data, path, **kwargs):
    """
    Todo: save data to some appropriate online store.
    """
    pass

def detector_cross_entropy(d, s, r, v):
    """
    Calculate the cross entropy of the venomosity and signals, given the detection bit strings.
    """
    # First calculate M, the predator's estimated probability a given signal bit string is venomous
    S = similarity(s, d)
    P = np.exp(-S / r[:, None])
    P /= np.sum(A, axis=1, keepdims=True)
    M = 1 - P

    # Now calculate the cross entropy with the probability 
    return -np.sum(np.log(A) * v, axis=1)


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) > 0:
        num_predators = int(args[0])
        num_venomous_prey = int(args[1])
        num_mimics = int(args[2])
        bit_string_length = int(args[3])
        generations = int(args[4])

    # Initialize the populations

    # identical s for all populations
    s = np.ones((num_venomous_prey + num_mimics, bit_string_length)).astype(np.int64)
    # s = np.random.randint(2, size=(num_venomous_prey + num_mimics, bit_string_length))

    # identical d for all populations
    d = np.ones((num_predators, bit_string_length)).astype(np.int64)

    r = np.ones(num_predators) * 0.9
    v = np.ones(num_venomous_prey) * 0.1

    # Keep track of transient data for analysis
    transient_data = {
        's': [],
        'd': [],
        'r': [],
        'v': [],
    }

    for generation in tqdm(range(generations)):
        # Save transient data; convert to 16-bit integers to save space
        s_copy = s.copy().astype(np.int16)
        d_copy = d.copy().astype(np.int16)
        r_copy = r.copy().astype(np.float16)
        v_copy = v.copy().astype(np.float16)

        transient_data['s'].append(s_copy)
        transient_data['d'].append(d_copy)
        transient_data['r'].append(r_copy)
        transient_data['v'].append(v_copy)

        s, d, r, v = update(s, d, r, v, num_venomous_prey, num_mimics)

    # Final population values
    print("Final signals (s):", s)
    print("Final detections (d):", d)
    print("Final risk tolerances (r):", r)
    print("Final venom levels (v):", v)

    # Save the final populations to a zarr file
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"mimicry_{date}.zarr"
    zarr.save(filename, **transient_data)



    
