
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx
from ast import literal_eval as make_tuple
import collections
import pickle


def load_data(name):
    
    dataset = Planetoid(root='data/', name= name, pre_transform=ToUndirected())
    G = to_networkx(dataset.data, node_attrs=['y'], to_undirected=True)
    nodes = max(nx.connected_components(G), key=len)

    if name == 'PubMed':
        with open('ddict-pubmed.pickle', 'rb') as f:
            ddict = pickle.load(f)
        data = change_mask(dataset, 'masks-pubmed.npz', dataset.data.num_nodes)
    elif name == 'cora':        
        ddict = dict(nx.all_pairs_shortest_path_length(G))
        data = change_mask(dataset, 'masks-cora.npz', dataset.data.num_nodes)
    
    elif name == 'citeseer':
        ddict = dict(nx.all_pairs_shortest_path_length(G))
        data = change_mask(dataset, 'masks-citeseer.npz', dataset.data.num_nodes) 

    return dataset, data, G, nodes, ddict

def change_mask(dataset, filename, num_nodes):

    masks = np.load(filename)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[masks['trainset']] = 1

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[masks['valset']] = 1

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[masks['testset']] = 1

    dataset.data.train_mask = train_mask
    dataset.data.test_mask = test_mask
    dataset.data.val_mask = val_mask

    return dataset.data



# def change_mask(dataset, nodes):
#     data = dataset.data
#     train_mask = data.train_mask.cpu().detach().numpy()
#     test_mask = data.test_mask.cpu().detach().numpy()
#     val_mask = data.val_mask.cpu().detach().numpy()

#     for i, val in enumerate(train_mask):
#         if i not in nodes:
#             train_mask[i] = False
#             test_mask[i] = False
#             val_mask[i] = False
#         else:
#             if (not val) and (not val_mask[i]):
#                 test_mask[i] = True
#     data.train_mask = torch.from_numpy(train_mask)
#     data.test_mask = torch.from_numpy(test_mask)
#     data.val_mask = torch.from_numpy(val_mask)

    return data

def load_result_df(filename = None):
    if filename is None:
        results = pd.DataFrame(columns = ['AddRatio', 'RemoveRatio', 'PerturbType',
        'PerturbedTestFScores', 'TP', 'FN', 'FP', 'DistMetric', 
        'DegreeMetric', 'PerturbedHomophily'])
    else: 
        results = pd.read_csv(filename, index_col = 0)
    return results


def homophily(data, nodes):
    labels = data.y.numpy()
    to_del = []
    
    for i, node in enumerate(data.edge_index.numpy()[0]):
        if node not in nodes: 
            to_del.append(i)

    edges = data.edge_index.numpy()
    edges = np.delete(edges, to_del, axis = 1)

    connected_labels_set = list(map(lambda x: labels[x], edges))
    connected_labels_set = np.array(connected_labels_set)
    label_connection_counts = []

    for i in range(7):
        connected_labels = connected_labels_set[:, np.where(connected_labels_set[0] == i)[0]]
        counter = dict(collections.Counter(connected_labels[1]))
        counter = add_missing_keys(counter, range(7))
        items = sorted(counter.items())
        items = [x[1] for x in items]
        label_connection_counts.append(items)
    label_connection_counts = np.array(label_connection_counts)
    homophily = label_connection_counts.diagonal().sum() / label_connection_counts.sum()
    
    return label_connection_counts, homophily

def add_missing_keys(counter, classes):
    for x in classes:
        if x not in counter.keys():
            counter[x] = 0
    return counter