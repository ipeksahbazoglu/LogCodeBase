import networkx as nx
import torch_geometric
import torch
from copy import deepcopy
from modules.util import contains_isolates,remove_edges, sample_connected_nodes, sample_nodes

def randomly_perturb(G: nx.Graph, data: torch_geometric.data.data.Data,  add: int=0, remove: int=0, component = False) -> nx.Graph:
    """Add and remove edges uniformly at random.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    data: torch_geometric.data.data.Data
        Data that generated the graph.
    add : int, optional
        Number of edges to add, by default 0
    remove : int, optional
        Number of edges to remove, by default 0

    Returns
    -------
    nx.Graph
        Perturbed graph, perturbed data.
    """

    while True:
        Gp = G.copy()
        data = deepcopy(data)
        edge_index = data.edge_index
        
        edges_to_add = []
        edges_to_remove = []
        perturb_type = None

        if add != 0:
            if add < 1: 
                num_edges = int(data.num_edges/ 2)
                add *= num_edges
            if component != True:
                while len(edges_to_add) < add:
                    edge = sample_connected_nodes(Gp)
                    edge = (min(edge), max(edge))
                    if edge not in G.edges and edge not in edges_to_add:
                        edges_to_add.append(edge)
            else:
                 while len(edges_to_add) < add:
                    edge = sample_nodes(Gp)
                    edge = (min(edge), max(edge))
                    if edge not in G.edges and edge not in edges_to_add:
                        edges_to_add.append(edge)
        else:
            perturb_type = 'remove'


        if remove != 0:
            if remove < 1: 
                num_edges = int(data.num_edges/ 2 )
                remove *= num_edges
            
            edges_to_remove, Gp, edge_index = remove_edges(Gp, data, remove, component)
        else:
            perturb_type = 'add'
        
        Gp.add_edges_from(edges_to_add)

        if not contains_isolates(Gp):
            break
        
    if len(edges_to_add) > 0:    
        edges_to_add_t = torch.tensor(edges_to_add)
        edges_to_add_t = torch.cat([edges_to_add_t, edges_to_add_t[:, [1, 0]]])
        edge_index = torch.cat((edge_index.T, edges_to_add_t)).T

    data.edge_index = edge_index
    edges_perturbed  = edges_to_add + edges_to_remove

    if perturb_type is None:
        perturb_type = 'addremove'

    return Gp, data, edges_perturbed, perturb_type