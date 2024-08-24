import networkx as nx
from modules.perturb import randomly_perturb
from modules.GCN import generate_output
from modules.util import homophily
import networkx as nx
import numpy as np 

def get_degree_dict(G: nx.Graph, edges_perturbed: list):
    #get perturbed nodes
    if len(edges_perturbed) == 0:
        return 4.08, 0
    node_p = dict()
    degree_sum = 0
    for edge in edges_perturbed:
        for node in edge:
            if node not in node_p:
                degree =  G.degree(node)
                node_p[node] = degree
                degree_sum += degree

    #calculate avg degree of nodes perturbed
    avg_degree = degree_sum / len(node_p)
    
    return avg_degree, degree_sum

def find_path(G: nx.Graph, edge1: tuple, edge2: tuple, ddict: dict):
    
    sum_paths = ddict[edge1[0]][edge2[0]] + ddict[edge1[0]][edge2[1]] \
        + ddict[edge1[1]][edge2[0]] + ddict[edge1[1]][edge2[1]]
    
    mean_distance = sum_paths / 4
        

    return mean_distance

def get_distribution_dict(G: nx.Graph, edges_perturbed: list, ddict: dict):
    #find pairwise distance between all nodes involved in a pair of edges
    if len(edges_perturbed) == 0:
        return 5.51
    dist = dict()
    avg_distance = dict()

    if len(edges_perturbed)  == 1:
        distance = nx.shortest_path_length(G, edges_perturbed[0][0], edges_perturbed[0][1])
        avg_distance[(edges_perturbed[0])] = distance

    else:
        for i in range(len(edges_perturbed)):
            edge_i = edges_perturbed[i]
            for edge in edges_perturbed[(i+1):]:
                a_distance = find_path(G, edge_i, edge, ddict)
                avg_distance[(edge_i, edge)] = a_distance
        
    dist_metric_avg = sum(avg_distance.values()) / len(avg_distance)

    
    return dist_metric_avg

def get_metrics(G, edges_perturbed, ddict):

    dist_metric_avg = get_distribution_dict(G, edges_perturbed, ddict)
    avg_degree, _ = get_degree_dict(G, edges_perturbed)    
   
    return dist_metric_avg, avg_degree



def generate_row(G, add, remove, dataset, model, ddict, nodes):
    row = []
    row.append(add); row.append(remove)
    _, data_p, ea,er, perturb_type = randomly_perturb(G= G , data = dataset.data,remove = remove, add = add)
    row.append(perturb_type); 
    fscores, tp, fn, fp = generate_output(model, dataset, data_p)
    row.append(fscores); row.append(tp);row.append(fn);row.append(fp)
    dist_metric_avg, avg_degree =  get_metrics(G, ea + er, ddict)
    row.append(dist_metric_avg); row.append(avg_degree); 
    _, hp = homophily(data_p, nodes)
    row.append(hp)
    return row





