
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx
from ast import literal_eval as make_tuple


def load_data(name):
    if name == 'cora':
        dataset = Planetoid(root='data/', name='cora', pre_transform=ToUndirected())
    else:
        return None, None
    data = dataset.data
    G = to_networkx(data, node_attrs=['y'], to_undirected=True)

    return dataset, data, G

def contains_isolates(G):
    """Determine if a graph contains a node with degree 0."""
    if nx.number_of_isolates(G) > 0:
        return True
    else:
        return False

def sample_nodes(G):
    """Uniformly sample (without replacement) `num_nodes` nodes from `G`."""
    nodes = max(nx.connected_components(G), key=len)
    return list(np.random.choice(list(nodes), 2, replace=False))

def sample_connected_nodes(G):
    """Uniformly sample (without replacement) `num_nodes` nodes from `G`."""
    nodes = G.nodes()
    cont = True 
    while cont:
        u = np.random.choice(list(nodes),replace=False)
        connected_nodes = list(nx.node_connected_component(G, u))
        if len(connected_nodes) > 2:
            v = np.random.choice(connected_nodes,replace=False)
            if v != u:
                cont = False
    return list([u, v])

def delete_edge(G, component):
    """Uniformly sample (without replacement) `num_edges` edges from `G`."""
    Gp = G.copy()
    if component:
        nodes = max(nx.connected_components(G), key=len)
        i = np.random.choice(list(nodes), replace=False)
    else:
        i = np.random.randint(0, Gp.number_of_nodes())
    
    if Gp.degree[i] > 1:
        edges = list(Gp.edges(i))
        n = np.random.randint(0, len(edges))
        if Gp.degree(edges[n][1]) > 1:
            Gp.remove_edge(edges[n][0], edges[n][1])
            return edges[n], Gp
        else:
            return -1, Gp
    else:
            return -1, Gp

def remove_edges(G,data, num_edges, component):
    """Uniformly sample (without replacement) `num_edges` edges from `G`."""
    edges_removed = []
    Gp = G.copy()
    data = deepcopy(data)
    while len(edges_removed) < num_edges:
        edge, Gp = delete_edge(Gp, component)
        if edge != -1: 
            edges_removed.append(edge) 
    
    edge_index = data.edge_index

    # calculate row indices to delete (both directions)
    mask_idx = [] 
    for edge in edges_removed:
        mask_idx.append(torch.where((torch.tensor(edge) == edge_index.T).all(axis=1))[0].item())
        mask_idx.append(torch.where((torch.tensor(edge[::-1]) == edge_index.T).all(axis=1))[0].item())

    # delete rows using a mask
    mask = torch.ones(edge_index.T.size(0))
    mask[mask_idx] = 0
    edge_index = edge_index.T[mask.bool()].T

    return edges_removed, Gp, edge_index

def get_augmented_adjacency(G, alpha):
    A = nx.adjacency_matrix(G)
    #generate diagonal degree matrix
    degrees = [val for (node, val) in G.degree()]
    a = np.matrix(degrees)
    D = np.diag(a.A1)

    #shift by alpha
    Id = np.identity(A.shape[0])
    A_ = A + alpha*Id
    D_ = D + alpha*Id

    #take square root and invert D
    D_inv = np.linalg.inv(np.sqrt(D_))
    #multiply 
    A_n = np.matmul(np.matmul(D_inv, A_), D_inv)
    
    return A_n

def load_result_df(filename = None):
    if filename is None:
        results = pd.DataFrame(columns = ['AddRatio', 'RemoveRatio', 'PerturbType',
        'Test_Accuracy', 'PerturbedTestAccuracy', 'AccuracyChange', 'DistAvgAc', 'DistAvgPc', 
         'DistMinAc', 'DistMinPc','DegreePc'])
    else: 
        results = pd.read_csv(filename, index_col = 0)
    return results


def string_to_tuple(edgesperturbed):

    z  = edgesperturbed[1:-1]
    z = iter(z.split(','))
    s = [ (a+ ',' + b).replace(" ", "") for a,b in zip(*([z]*2))]
    
    edges_perturbed = [make_tuple(i) for i in s]

    return edges_perturbed