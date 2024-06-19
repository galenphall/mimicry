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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple


def bin2vec(bin: np.ndarray) -> np.ndarray:

    """
    Converts boolean array to unit vector with a 1 in the column indexed by the integer value of the binary vector

    :param bin: binary vector
    :return: unit vector
    """
    int_arr = bin.astype(int)

    bin_str = ''.join(int_arr.astype(str))

    idx = int(bin_str, 2)

    vec = np.zeros(2**len(bin), dtype=int)

    vec[idx] = 1

    return vec

class MimicryModel:

    def __init__(self,
                 num_predators: int,
                 num_mimics: int,
                 num_venomous:int,
                 init_r: np.ndarray[float],
                 init_sm: np.ndarray[bool],
                 init_sv: np.ndarray[bool],
                 init_d: np.ndarray[float]
                 ) -> None:

        """
        Initialize mimicry model

        :param num_predators: Initial number of predators in the model
        :param num_mimics: Initial number of mimicking prey in the model
        :param num_venomous: Initial number of venomous prey in the model
        :param init_r: Array of predator risk tolerances
        :param init_sm: num_mimics x signal length array of mimicking prey phenotype vectors
        :param init_sv: num_venomous x signal length array of venomous prey phenotype vectors
        :param init_d: 2^(signal length) predator's knowledge of distribution of predators of each phenotype

        TODO: venomosity
        """

        self.num_predators = num_predators
        self.num_mimics = num_mimics
        self.num_venomous = num_venomous
        self.r = init_r
        self.sm = init_sm
        self.sv = init_sv
        self.d = init_d

        self.time = 0
        self.history = [(self.sm, self.sv, self.d, self.r)]

    def select_prey(self) -> np.ndarray[bool]:

        """
        Determines which predator and prey are killed and updates d, sm, sv accordingly.

        :return: Boolean corresponding to which predators were fed
        """

        # concatenate prey types and randomly select num_predators to be at risk of being eaten
        all_prey = np.concatenate([self.sm, self.sv])

        flags = np.array([False] * self.sm.shape[0] + [True] * self.sv.shape[0],
                         dtype=bool)  # Boolean flag for venomous

        # indices of selected prey
        idx = np.random.randint(all_prey.shape[0], size=self.d.shape[0])

        sampled_prey = all_prey[idx, :]

        # num_predator x 2^(signal length) array. Each row is the unit vector representation of a prey's signal
        selected_signals = np.apply_along_axis(bin2vec, arr=sampled_prey, axis=1)

        probs = self.d * selected_signals

        # select the one non-zero element referring to the probability the selected signal is venomous
        selected_probs = np.max(probs, axis=1)

        # eat if the probability is below the risk tolerance
        eating_preds = selected_probs < self.r
        eaten_prey = idx[eating_preds] # surviving prey are the ones that were not eaten

        poisoned_preds = eating_preds * flags[idx] # which of the predators ate venomous snakes
        fed_preds = eating_preds * ~flags[idx] # which of the predators successfully ate mimicking prey
        fecund_preds = fed_preds[~poisoned_preds]

        # disaggregate eaten_prey between mimic and venomous
        eaten_mimics = np.array([i for i in eaten_prey if i < self.sm.shape[0]])
        eaten_venomous = np.array([i-self.sm.shape[0] for i in eaten_prey if i >= self.sm.shape[0]])

        # masks to remove eaten prey
        mimic_mask = np.array([i not in eaten_mimics for i in range(len(self.sm))])
        venom_mask = np.array([i not in eaten_venomous for i in range(len(self.sv))])


        # drop all that have been eaten or poisoned
        self.d = self.d[~poisoned_preds, :]
        self.r = self.r[~poisoned_preds]

        if len(mimic_mask) > 0:
            self.sm = self.sm[mimic_mask, :]

        if len(venom_mask) > 0:
            self.sv = self.sv[venom_mask, :]

        # same with venomosities

        return fecund_preds

    def birth_predators(self,
                        fecund_preds: np.ndarray[bool],
                        var_p: float) -> None:

        """
        Birth new predators from those who have been fed with some error in the passing down of risk tolerance.
        We assume that kids follow their parents' signal detection knowledge

        :param fecund_preds: boolean array of predators that are viable to have offspring
        :param var_p: Variance in the normal distribution of risk tolerance.
        """

        parent_signals = self.d[fecund_preds]
        parent_risks = self.r[fecund_preds]

        # intializing inherited signal detection and risk tolerance from parents
        offspring_signals = list()
        offspring_risks = list()

        num_pairs = parent_risks.shape[0] // 2

        if num_pairs > 0:
            for pair in range(num_pairs):
                mate1_idx, mate2_idx = np.random.randint(parent_signals.shape[0], size=2)
                mate1 = self.d[mate1_idx]
                mate2 = self.d[mate2_idx]
                pure_inheritance = np.mean([mate1, mate2], axis=0)
                offspring_signals.append(pure_inheritance)

                mate1_r = self.r[mate1_idx]
                mate2_r = self.r[mate2_idx]
                offspring_risks.append(np.random.normal(np.mean([mate1_r, mate2_r]), var_p))

            self.d = np.concatenate([self.d, np.array(offspring_signals)])
            self.r = np.concatenate([self.r, np.array(offspring_risks)])

        else:
            pass

    def birth_mimics(self,
                     dom_m: float,
                     mut_v: float) -> None:

        """
        Modeling birth and inheritance of phenotype vectors in mimics, updating self.sm accordingly.

        TODO: more than 1 child per pair

        :param dom_m: Probability of dominant trait in mixed setting.
        :param mut_v: Probability of random recessive trait.
        """
        offspring_signals = list()

        num_pairs = self.sm.shape[0] // 2

        if num_pairs > 0:
            noise = np.random.rand(num_pairs, self.sm.shape[1])
            dom = np.random.rand(num_pairs, self.sm.shape[1])

            for pair in range(num_pairs):
                mate1_idx, mate2_idx = np.random.randint(self.sm.shape[0], size=2)
                mate1 = self.sm[mate1_idx]
                mate2 = self.sm[mate2_idx]
                pure_inheritance = mate1 * mate2

                # flip some 0s to 1s (Assumption that 0 is recessive trait)
                dom_flips = dom[pair, :] < dom_m
                dom_inheritance = pure_inheritance.astype(bool) + dom_flips.astype(bool)

                # randomly flip some 1s to 0s
                noisy_flips = noise[pair, :] >= mut_v
                noisy_inheritance = dom_inheritance.astype(bool) * noisy_flips.astype(bool)

                offspring_signals.append(noisy_inheritance)

            self.sm = np.concatenate([self.sm, np.array(offspring_signals)])

        else:
            pass

    def birth_venomous(self,
                       dom_v: float,
                       mut_v) -> None:

        """
        Modeling birth and inheritance of phenotype vectors in venomous, updating self.sv accordingly.

        :param dom_v: Probability of dominant trait in mixed setting:
        :param mut_v: Probability of random recessive trait.
        """
        offspring_signals = list()

        num_pairs = self.sv.shape[0] // 2

        if num_pairs > 0:
            dom = np.random.rand(num_pairs, self.sv.shape[1])
            noise = np.random.rand(num_pairs, self.sv.shape[1])

            for pair in range(num_pairs):
                mate1_idx, mate2_idx = np.random.randint(self.sv.shape[0], size=2)
                mate1 = self.sv[mate1_idx]
                mate2 = self.sv[mate2_idx]
                pure_inheritance = mate1 * mate2

                # flip some 0s to 1s (Assumption that 0 is recessive trait)
                dom_flips = dom[pair, :] < dom_v
                dom_inheritance = pure_inheritance.astype(bool) + dom_flips.astype(bool)

                # randomly flip some 1s to 0s
                noisy_flips = noise[pair, :] >= mut_v
                noisy_inheritance = dom_inheritance.astype(bool) * noisy_flips.astype(bool)

                offspring_signals.append(noisy_inheritance)

            self.sv = np.concatenate([self.sv, np.array(offspring_signals)])

        else:
            pass

    def update_memory(self) -> None:

        """
        Update memory by including current time step's values.
        """

        self.history.append((self.sm, self.sv, self.d, self.r))

    def update_predators(self,
                         eps: float = 1e-10) -> None:

        """
        Updates self.d based on knowledge of the mimic and venomous signal populations.
        Currently assuming all predators have the same knowledge.
        """

        sm, sv, _, _ = self.history[-1]

        sm_unit_vecs = np.apply_along_axis(bin2vec, arr=sm, axis=1)
        sv_unit_vecs = np.apply_along_axis(bin2vec, arr=sv, axis=1)

        sm_idxs = np.argmax(sm_unit_vecs, axis=1)
        sv_idxs = np.argmax(sv_unit_vecs, axis=1)

        sm_unique, sm_counts = np.unique(sm_idxs, return_counts=True)
        sv_unique, sv_counts = np.unique(sv_idxs, return_counts=True)

        sm_arr = np.zeros(2**sm.shape[1])
        for u,c in zip(sm_unique, sm_counts):
            sm_arr[u] = c

        sv_arr = np.zeros(2 ** sv.shape[1])
        for u, c in zip(sv_unique, sv_counts):
            sv_arr[u] = c

        #together = np.vstack((sm_arr, sv_arr))
        #prob_v = (together[1,:] + eps)/np.sum(together+eps, axis=0)

        prob_v = sv_arr/(sm_arr + sv_arr  + np.array([eps]*sm_arr.shape[0]))

        self.d = np.array([prob_v]*self.d.shape[0])

    def random_death(self,
                     zp: float,
                     zm: float,
                     zv: float) -> None:

        """
        Randomly remove a fraction of each population.

        :param zp: Likelihood of random death in predators
        :param zm: Likelihood of random death in mimicking prey
        :param zv: Likelihood of random death in venomous prey
        """

        pred_mask = np.random.rand(self.d.shape[0]) >= zp
        mim_mask = np.random.rand(self.sm.shape[0]) >= zm
        ven_mask = np.random.rand(self.sv.shape[0]) >= zv

        self.d = self.d[pred_mask, :]
        self.r = self.r[pred_mask]

        self.sm = self.sm[mim_mask, :]
        self.sv = self.sv[ven_mask, :]

    def run(self,
            T: int,
            var_p: float,
            dom_m: int,
            mut_m: int,
            dom_v: float,
            mut_v: float,
            zp: float,
            zm: float,
            zv: float
            ) -> None:

        for _ in tqdm(range(T)):
            self.update_memory()
            fecund_predators = self.select_prey()
            self.birth_predators(fecund_predators, var_p)
            self.birth_mimics(dom_m, mut_m)
            self.birth_venomous(dom_v, mut_v)
            self.update_predators()
            self.random_death(zp,zm,zv)
























