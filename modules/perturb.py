import networkx as nx
import torch_geometric
import torch
from copy import deepcopy
import numpy as np


def randomly_perturb(G: nx.Graph, data: torch_geometric.data.data.Data, add: int=0, remove: int=0):

    Gp = G.copy()
    nodes = list(max(nx.connected_components(G), key=len))
    data = deepcopy(data)
    edge_index = data.edge_index
    
    edges_to_add = []
    edges_to_remove = []
    perturb_type = None
    
    num_edges = int(data.num_edges/ 2)

    if remove != 0:
        if remove < 1: 
            remove *= num_edges
        edges_to_remove, Gp, edge_index = delete_edge(Gp, data, remove, nodes)


    else:
        perturb_type = 'add'
    


    if add != 0:
        if add < 1: 
            add *= num_edges
        edges_to_add = add_edges(G, nodes, add)
        
        Gp.add_edges_from(edges_to_add)
        edges_to_add_t = torch.tensor(edges_to_add)
        edges_to_add_t = torch.cat([edges_to_add_t, edges_to_add_t[:, [1, 0]]])
        edge_index = torch.cat((edge_index.T, edges_to_add_t)).T

    else:
        perturb_type = 'remove'

    data.edge_index = edge_index
    edges_perturbed  = edges_to_add + edges_to_remove

    if perturb_type is None:
        perturb_type = 'addremove'

    return Gp, data, edges_to_add, edges_to_remove, perturb_type

def delete_edge(G, data, remove, nodes):
    
    data = deepcopy(data)
    edges_removed = set()
    Gp = G.copy()
    
    while len(edges_removed) < remove:
        
        i = np.random.choice(list([node for node in nodes if Gp.degree(node) > 1]), replace=False)

        if remove <= 3200:

            neigh = [n for n in Gp.neighbors(i) if Gp.degree(n) > 1]
        else:
            neigh = [n for n in Gp.neighbors(i)]

        if len(neigh) > 0:
            n = np.random.randint(0, len(neigh))
            Gp.remove_edge(i, neigh[n])
            edges_removed.add((i, neigh[n]))

            
        else:
            continue
    
    edge_index = data.edge_index
    # calculate row indices to delete (both directions)
    mask_idx = [] 
    edges_removed = list(edges_removed)
    for edge in edges_removed:
        mask_idx.append(torch.where((torch.tensor(edge) == edge_index.T).all(axis=1))[0].item())
        mask_idx.append(torch.where((torch.tensor(edge[::-1]) == edge_index.T).all(axis=1))[0].item())
    # delete rows using a mask
    mask = torch.ones(edge_index.T.size(0))
    mask[mask_idx] = 0
    edge_index = edge_index.T[mask.bool()].T

    return edges_removed, G, edge_index

def add_edges(G, nodes, add):

    edges_to_add = set()
    
    while len(edges_to_add) < add:
        i = np.random.choice(list(nodes), replace=False)
        neigh = set(nx.neighbors(G, i)) | set([i])
        choice = [node for node in nodes if node not in neigh]
        n = np.random.choice(list(choice), replace=False)
        edges_to_add.add((i, n))
    
    return list(edges_to_add)

