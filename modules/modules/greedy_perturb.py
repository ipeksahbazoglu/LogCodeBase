from modules.util import homophily
from modules.graphutil import get_metrics, generate_row
import networkx as nx
import numpy as np
import torch_geometric
import torch
from copy import deepcopy
from modules.GCN import GCN
import pandas as pd
from torcheval.metrics.functional import multiclass_f1_score
from collections import Counter


def greedy_perturb(G, dataset, budget, model, addsample, rmsample, ddict, pathname):
      
    results = load_result_df(pathname)
    perturbed_edges = set()
    add_count = 0
    remove_count = 0
    
    Gp = G.copy()
    dp = deepcopy(dataset)
    Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    comp_edges = Gc.edges()
    p10 = round(len(comp_edges)*0.10)
    #p10 = round(len(comp_edges)*0.05)
    comp_nodes = list(Gc.nodes())

    maxloss = []
    maxedge = []

    while len(perturbed_edges) < budget:        

        edges = list(comp_edges - perturbed_edges)

        best_edge_a, loss_a = best_edge_add(Gp, dp, addsample, perturbed_edges, comp_nodes, model)

        best_edge_r, loss_r = best_edge_remove(Gp, dp, rmsample, edges, model)


        if loss_a > loss_r:
            add_count += 1
            perturbed_edges.add(best_edge_a)
            dp.data = add_edge_tensor(dp.data, best_edge_a)
            Gp.add_edge(best_edge_a[0], best_edge_a[1])
            maxloss.append(loss_a)
            maxedge.append(best_edge_a)
        else:
            remove_count +=1 
            perturbed_edges.add(best_edge_r)
            dp.data = remove_edge_tensor(dp.data, best_edge_r)
            Gp.remove_edge(best_edge_r[0], best_edge_r[1])
            maxloss.append(loss_r)
            maxedge.append(best_edge_r)
    
        checkpoint = divmod(len(perturbed_edges), p10)
        if checkpoint[1] == 0:
            # print('10 percent done')
            _ ,hp = homophily(dp.data, comp_nodes)
            dist_metric_avg, avg_degree =  get_metrics(G, list(perturbed_edges), ddict)
            row = generate_row(dataset.data, dp.data, (10*checkpoint[0]), model,hp,dist_metric_avg, avg_degree, \
                 add_count,remove_count)
            results.loc[(len(results))] = row
            results.to_csv(r'data/finalreportgreedyhigh.csv')
            print(maxloss, maxedge)

    return results, maxloss, maxedge


def best_edge_remove(G: nx.Graph, dataset: torch_geometric.datasets, samples: int, edges: list, model):
    if samples == 0:
        return None, 0
    
    best_edge = set()
    edges_rm = set()
    loss = 0
    
    for _ in range(samples):
        idx = np.random.choice(range(len(edges)), replace=False)
        u, v = edges[idx]
        
        G.remove_edge(u, v)
        if nx.number_of_isolates(G) == 0:
            edges_rm.add((u,v))
            dataset.data = remove_edge_tensor(dataset.data, [u,v])
            ploss = generate_loss(dataset.data, model, mask = 'test')
            dataset.data = add_edge_tensor(dataset.data, [u,v])
            
            if ploss > loss:
                loss = ploss
                best_edge = (u,v)

        G.add_edge(u, v)
        
    return best_edge, loss


def best_edge_add(G: nx.Graph, dataset: torch_geometric.datasets, samples: int, edges_perturbed: set, comp_nodes: list, model):
    
    if samples == 0:
        return None, 0
    
    edges_add = set()
    best_edge = None
    loss = 0
    
    for _ in range(samples):
        nodes = comp_nodes.copy()
        u, v = np.random.choice(nodes, 2, replace=False)
        u, v = min(u, v), max(u, v)
        if (not G.has_edge(u, v)) and ((u,v) not in edges_perturbed) :
            edges_add.add((u, v))
            dataset.data = add_edge_tensor(dataset.data, [u,v])
            ploss = generate_loss(dataset.data, model, mask = 'test')
            dataset.data = remove_edge_tensor(dataset.data, [u,v])

            if ploss > loss:
                loss = ploss
                best_edge = (u,v)
        
    return best_edge, loss

def remove_edge_tensor(data, edge):
    u = edge[0]; v = edge[1]

 
    edge_index = data.edge_index

    mask_idx = []
    mask_idx.append(torch.where((torch.tensor([u,v]) == edge_index.T).all(axis=1))[0].item())
    mask_idx.append(torch.where((torch.tensor([v,u]) == edge_index.T).all(axis=1))[0].item())

    mask = torch.ones(edge_index.T.size(0))
    mask[mask_idx] = 0
    
    data.edge_index = edge_index.T[mask.bool()].T

    return data

def add_edge_tensor(data, edge):
    u, v = edge
    edge_index = data.edge_index
    add_idx_t = torch.stack([torch.tensor([u,v]), torch.tensor([v,u])])
    data.edge_index  = torch.cat((edge_index.T, add_idx_t)).T
    
    return data

def load_model(filepath, dataset):
      model = GCN(dataset)
      model.load_state_dict(torch.load(filepath))
      model.eval()

      return model

def generate_loss(data, model, mask = 'val'):
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)
    if mask == 'val':
        t_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    elif mask == 'test': 
        t_loss = criterion(out[data.test_mask], data.y[data.test_mask])
    elif mask == 'val+test': 
        data.vt_mask = data.test_mask | data.val_mask
        t_loss = criterion(out[data.test_mask | data.val_mask], data.y[data.test_mask | data.val_mask])
    else:
        t_loss = criterion(out[data.train_mask], data.y[data.train_mask])

    return t_loss.item()


def generate_output(model, data, data_p):
      
      model.eval()
      h_p = model(data_p.x, data_p.edge_index)

      out_p = h_p.argmax(dim = 1)

      
      fscore = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask],
                           num_classes=7, average = None).flatten().numpy()
      
      f1_macro_p = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask], 
                          num_classes=7, average = 'macro').item()
      
      correct_nodes = (out_p[data.test_mask] == data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()
      mis_nodes = (out_p[data.test_mask] != data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()

      TP = [x[1] for x in sorted(Counter((data.y[data.test_mask])[correct_nodes].flatten().numpy()).items())]
      FN = [x[1] for x in sorted(Counter((data.y[data.test_mask])[mis_nodes].flatten().numpy()).items())]

      FP = [x[1] for x in sorted(Counter(out_p[data.test_mask][mis_nodes].flatten().numpy()).items())]
        




      return [ list(fscore) + [f1_macro_p], TP, FN, FP]

def load_result_df(filename = None):
    if filename is None:
        results = pd.DataFrame(columns= ['budget', 'samples', 'method', 'test_loss', 'train_loss', 
                                    'val_loss', 'fscore','TP','FN','FP', 'homophily', 
                                    'dist_metric', 'degree_metric', 'add_count', 'remove_count'])
    else: 
        results = pd.read_csv(filename, index_col = 0)
    return results

def generate_row(data, data_p, budget, model,hp,dist_metric_avg, avg_degree, \
                 add_count,remove_count):
    row = [budget, '50', 'maxtestloss']
    row.append(generate_loss(data_p, model, 'test'))
    row.append(generate_loss(data_p, model, 'train'))
    row.append(generate_loss(data_p, model, 'val'))
    row.extend(generate_output(model, data, data_p))
    row.append(hp) 
    row.append(dist_metric_avg); row.append(avg_degree); 
    row.append(add_count); row.append(remove_count)

    return row