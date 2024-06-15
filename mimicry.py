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

    A_{ij} = exp{-I(d_i, s_j)/r_i} / \sum_k exp{-I(d_i, s_k)/r_i}

Where I is the mutual information between two bit strings, and r_i is the risk tolerance of predator i.

Prey are sampled proportionally to the probability they are not eaten, i.e.

    P_j \propto \prod_i (1 - A_{ij})

Predators are sampled proportionally to the probability they die from eating a venomous prey:

    P_i \propto \sum_j A_{ij} v_j

After sampling, the next generation is created using the standard crossover and random mutations.

TODO: Save transient data to a zarr file for analysis and visualization.
"""
import numpy as np
from tqdm import tqdm
import datetime 
import zarr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def update(s, d, r, v, num_venomous_prey, num_mimics):
    """
    Update the populations of predators, venomous prey, and mimics.
    """
    # Split s into venomous prey signals and mimic signals
    s_v = s[:num_venomous_prey]
    s_m = s[num_venomous_prey:]

    # Calculate the mutual information between the detection and signal bit strings
    I_v = mutual_information(s_v, d)
    I_m = mutual_information(s_m, d)

    # Calculate the predation matrix for venomous prey and mimics
    A_v = np.exp(-I_v / r[:, None])
    A_m = np.exp(-I_m / r[:, None])

    # Normalize the predation matrices
    A_v /= np.sum(A_v, axis=1, keepdims=True)
    A_m /= np.sum(A_m, axis=1, keepdims=True)

    # Check for NaN values in A_v and A_m
    if np.any(np.isnan(A_v)):
        print("NaN values detected in A_v:", A_v)
        A_v = np.nan_to_num(A_v)
    if np.any(np.isnan(A_m)):
        print("NaN values detected in A_m:", A_m)
        A_m = np.nan_to_num(A_m)

    # Sample the prey populations
    prey_v = sample_prey(A_v)
    prey_m = sample_prey(A_m)

    # Sample the predator population
    predators = sample_predators(A_v, v)

    # Create the next generation for venomous prey and mimics
    new_s_v = crossover(s_v, prey_v)
    new_s_v = mutate(new_s_v)
    new_s_m = crossover(s_m, prey_m)
    new_s_m = mutate(new_s_m)

    # Combine the new signals back into one array
    new_s = np.vstack((new_s_v, new_s_m))

    # Mutate the selected populations for d, r, and v
    new_d = mutate(d[predators])
    new_r = mutate(r[predators])
    new_v = mutate(v[prey_v])

    return new_s, new_d, new_r, new_v


def mutual_information(s, d):
    """
    Calculate the mutual information between two bit strings.
    """
    # Assuming s and d are numpy arrays of the same shape
    return np.sum(s == d, axis=1) / s.shape[1]

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

def crossover(s, prey):
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
num_mimics = 100
bit_string_length = 10

# Randomly initialize the populations
s = np.random.randint(2, size=(num_venomous_prey + num_mimics, bit_string_length))
d = np.random.randint(2, size=(num_predators, bit_string_length))
r = np.random.rand(num_predators)
v = np.random.rand(num_venomous_prey)

# Run the algorithm for many generations
num_generations = 10000

# Keep track of transient data for analysis
transient_data = {
    's': [],
    'd': [],
    'r': [],
    'v': []
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

# Plot correlations between venom levels and signals, detections
def compute_correlations(s, v):
    """
    Compute the correlation between each bit in the signals and the venom levels.
    """
    num_bits = s.shape[1]
    correlations = np.zeros(num_bits)
    p_values = np.zeros(num_bits)

    for i in range(num_bits):
        # Calculate Pearson correlation coefficient and p-value
        corr, p_value = pearsonr(s[:, i], v)
        correlations[i] = corr
        p_values[i] = p_value

    return correlations, p_values

def plot_correlations(correlations, p_values, significance_level=0.05):
    """
    Plot the correlation coefficients for each bit in the signals with the venom levels.
    """
    num_bits = len(correlations)
    significant = p_values < significance_level

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_bits), correlations, color='b', alpha=0.7, label='Correlation')
    plt.scatter(np.where(significant), correlations[significant], color='r', label='Significant', zorder=5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Bit Index')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation between Signal Bits and Venom Levels')
    plt.legend()
    plt.show()

# Assuming s and v are the final evolved signals and venom levels
s_v = s[:num_venomous_prey]  # Venomous prey signals

# Compute correlations
correlations, p_values = compute_correlations(s_v, v)

# Plot the correlations
plot_correlations(correlations, p_values)



    
