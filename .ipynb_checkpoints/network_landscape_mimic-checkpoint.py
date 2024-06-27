import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import zarr


def update(Gp, Gv, s, d, r, v, nv, prey_mutation_rate=0.001, predator_mutation_rate=0.001):
    """
    Parameters
    ----------
    Gp : networkx.Graph
        Network of nodes representing phenotypes, embedded in 2d space.
    Gv : networkx.Graph
        Network of nodes representing venomosity, with associated venomosity values.
    s : np.array
        list of prey node positions in Gp.
    d : np.array (n_predators, 2)
        predator detection vectors in 2d space.
    r : np.array
        predator risk tolerances
    v : np.array
        prey venomosity nodes.
    nv : int
        number of venomous prey, as opposed to mimics.
    prey_mutation_rate : float
        rate of mutation in the prey population.
    predator_mutation_rate : float
        rate of mutation in the predator population.
    """
    n_predators = len(r)
    nm = len(s) - nv

    # Calculate predation matrix
    s_vec = np.array([Gp.nodes[s_i]['pos'] for s_i in s])
    sim = 1 / cdist(d, s_vec)
    P = np.exp(- sim / r[:, None])
    P = P / P.sum(axis=1)[:, None]

    # Select predators using fitness = P(i survives) = 1 - sum_j v_j P(i -> j)
    v_vals = np.array([Gv.nodes[v_i]['v'] for v_i in v]) # Get venomosity values
    v_vals = np.concatenate([v_vals, np.zeros(nm)]) # Add zeros for mimics
    Fp = np.zeros(n_predators)
    for i in range(n_predators):
        Fp[i] = 1 - np.sum(v_vals * P[i])
    Fp = Fp / Fp.sum()
    pred_idx = np.random.choice(n_predators, size=int(n_predators/10), p=Fp, replace=True)

    # Make new predator population asexually
    r_new = []
    d_new = []
    while len(r_new) < n_predators:
        p = np.random.choice(pred_idx, size=1)[0]
        r_i = r[p] + np.random.normal(0, predator_mutation_rate)
        r_i = np.round(r_i, 2)

        # Ensure r_i is positive
        if r_i < 0:
            r_i = -r_i

        r_new.append(r_i)

        # Calculate new d values
        d_i = d[p] + np.random.normal(0, predator_mutation_rate, size=2)
        d_i = np.round(d_i, 2)
        d_new.append(d_i)


    # Select prey using fitness = \prod_i (1 - P(i -> j))
    F_prey = np.prod(1 - P, axis=0)
    Fv = F_prey[:nv] / F_prey[:nv].sum()
    Fm = F_prey[nv:] / F_prey[nv:].sum()
    venomous_prey_idx = np.random.choice(nv, size=int(nv/10), p=Fv, replace=True)
    mimic_prey_idx = np.random.choice(nm, size=int(nm/10), p=Fm, replace=True)

    s_new = []
    v_new = []

    # Make new venomous prey population
    while len(s_new) < nv:
        p = np.random.choice(venomous_prey_idx, size=1)[0]
        # The venomisity value is associated with the prey's node in Gv.
        # So inheritance is done by selecting the venomosity value of one of the parents, 
        # and mutating it.
        v_i = v[p]
        # Now we mutate by (possibly) moving to a neighboring node in venomosity space.
        if np.random.rand() < prey_mutation_rate:
            neighbors = list(Gv.neighbors(v_i))
            v_i = np.random.choice(neighbors)

        v_new.append(v_i)

        # Get new venomous prey phenotypes using graph Gp
        s_i = s[p]
        # Now we mutate by (possibly) moving to a neighboring node in phenotype space.
        if np.random.rand() < prey_mutation_rate:
            neighbors = list(Gp.neighbors(s_i))
            s_i = np.random.choice(neighbors)

        s_new.append(s_i)

    # Make new mimic prey population
    while len(s_new) < nv + nm:
        p = np.random.choice(mimic_prey_idx, size=1)[0]
        # Mimics don't have venomosity values, so we just inherit the phenotype.
        s_i = s[p]
        # Now we mutate by (possibly) moving to a neighboring node in phenotype space.
        if np.random.rand() < prey_mutation_rate:
            neighbors = list(Gp.neighbors(s_i))
            s_i = np.random.choice(neighbors)

        s_new.append(s_i)

    return np.array(s_new), np.array(d_new), np.array(r_new), np.array(v_new)


def construct_Gp(n, k, domain=[-1, 1]):
    """
    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    k : int
        Number of nearest neighbors to connect to.

    Returns
    -------
    Gp : networkx.Graph
        Graph of nodes representing phenotypes, embedded in 2d space.
    """
    Gp = nx.Graph()
    positions = np.random.uniform(domain[0], domain[1], size=(n, 2))
    for i, pos in enumerate(positions):
        Gp.add_node(i, pos=pos)
    
    # add edges
    for i, pos in enumerate(positions):
        dists = np.linalg.norm(positions - pos, axis=1)
        nearest = np.argsort(dists)[1:k+1]
        for n_i in nearest:
            Gp.add_edge(i, n_i)

    return Gp

def construct_Gv(n, k):
    """
    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    k : int
        Number of nearest neighbors to connect to.

    Returns
    -------
    Gv : networkx.Graph
        Graph of nodes representing venomosity, with associated venomosity values.
    """
    # Represent venomosity using a 2d gaussian distribution.
    # Then scatter nodes randomly in this space, and their venomosity 
    # is the value of the gaussian at their position.
    # Finally, connect each node to its k nearest neighbors.

    v = lambda x, y: np.exp(- (x**2 + y**2) / 2)
    Gv = nx.Graph()
    positions = np.random.uniform(-1, 1, size=(n, 2))
    for i, pos in enumerate(positions):
        Gv.add_node(i, v=v(*pos), pos=pos)

    # add edges
    for i, pos in enumerate(positions):
        dists = np.linalg.norm(positions - pos, axis=1)
        nearest = np.argsort(dists)[1:k+1]
        for n_i in nearest:
            Gv.add_edge(i, n_i)
    
    return Gv

def main():
    nv = 1000
    nm = 1000
    n_predators = 100

    # Generate Gp and Gv
    Gp = construct_Gp(1000, 10)
    Gv = construct_Gv(1000, 10)

    # Generate starting s, d, r, v
    # s is a list of prey node positions in Gp: the 
    # positions of venomous prey are the first nv elements and should
    # be initialized to be close to each other; the positions of mimic prey
    # are the remaining nm elements and should be initialized to be close to each other.
    
    # s = []
    # # create venomous prey s using short random walks (length 5 ± 2) near a starting point
    # start = np.random.choice(Gp.nodes)
    # while len(s) < nv:
    #     walk = [start]
    #     for _ in range(5 + int(np.random.normal(0, 2))):
    #         walk.append(np.random.choice(list(Gp.neighbors(walk[-1]))))
    #     s.extend(walk)

    # # create mimic prey s using short random walks (length 5 ± 2) near a starting point
    # start = np.random.choice(Gp.nodes)
    # while len(s) < nv + nm:
    #     walk = [start]
    #     for _ in range(5 + int(np.random.normal(0, 2))):
    #         walk.append(np.random.choice(list(Gp.neighbors(walk[-1]))))
    #     s.extend(walk)

    # s = s[:nv+nm]

    # Make totally random starting vectors
    s = np.random.choice(Gp.nodes, nv+nm)

    # create predator detection vectors d
    d = np.random.uniform(-1, 1, (n_predators, 2))

    # create predator risk tolerances r by sampling from an exponential distribution
    r = np.random.exponential(1, n_predators)

    # create prey venomosity starting nodes v by sampling from Gv
    v = np.random.choice(Gv.nodes, nv)

    # Create dict to save data
    data = {'s': [], 'd': [], 'r': [], 'v': []}

    # Run the simulation
    for _ in range(100):
        data['s'].append(s)
        data['d'].append(d)
        data['r'].append(r)
        data['v'].append(v)
        s, d, r, v = update(Gp, Gv, s, d, r, v, nv, 
                            prey_mutation_rate=0.0001, predator_mutation_rate=0.00001)
        

    return data, Gp, Gv

if __name__ == '__main__':
    data, Gp, Gv = main()
    for k in data:
        try:
            data[k] = np.array(data[k])
        except ValueError:
            for v in data[k]:
                print(len(v))
    zarr.save('data.zarr', **data)

    # Save the graphs using gml
    # First convert the numpy arrays stored in Gp and Gv to lists
    for node in Gp.nodes:
        Gp.nodes[node]['pos'] = Gp.nodes[node]['pos'].tolist()
    for node in Gv.nodes:
        Gv.nodes[node]['pos'] = Gv.nodes[node]['pos'].tolist()
        
    nx.write_gml(Gp, 'Gp.gml') 
    nx.write_gml(Gv, 'Gv.gml')

    # Draw Gp and Gv
    import matplotlib.pyplot as plt
    pos = nx.get_node_attributes(Gp, 'pos')
    nx.draw(Gp, pos, node_size=10)
    plt.savefig('Gp.png')
    plt.clf()
    v = nx.get_node_attributes(Gv, 'v')
    pos = nx.get_node_attributes(Gv, 'pos')
    nx.draw(Gv, pos, node_size=10, node_color=list(v.values()), cmap='viridis')
    plt.savefig('Gv.png')
    plt.clf()





    


            
