import networkx as nx
from modules.perturb import randomly_perturb
from modules.GCN import compare_accuracy, generate_output
import networkx as nx
from modules.GCN import compare_accuracy, generate_output

def find_path(G: nx.Graph, edge1: tuple, edge2: tuple, dist: dict):
    all_paths = []
    for node1 in edge1:
        for node2 in edge2:
            if ((node1, node2) in dist):
                distance = dist[(node1, node2)] 
                all_paths.append(distance)
                mean_distance = sum(all_paths)/len(all_paths)

                 
            else:
                distance = nx.shortest_path_length(G, node1, node2)
                dist[(node1, node2)]  = distance
                dist[(node2, node1)]  = distance
                all_paths.append(distance)
                mean_distance = sum(all_paths)/len(all_paths)
        

    return mean_distance, dist

def get_distribution_dict(G: nx.Graph, edges_perturbed: list):
    #find pairwise distance between all nodes involved in a pair of edges
    dist = dict()
    avg_distance = dict()
    if len(edges_perturbed)  == 1:
        distance = nx.shortest_path_length(G, edges_perturbed[0][0], edges_perturbed[0][1])
        avg_distance[(edges_perturbed[0])] = distance

    else:
        for i in range(len(edges_perturbed)):

            edge_i = edges_perturbed[i]
            for edge in edges_perturbed[(i+1):]:
                a_distance, dist = find_path(G, edge_i, edge, dist)
                avg_distance[(edge_i, edge)] = a_distance
        
    dist_metric_avg = sum(avg_distance.values()) / len(avg_distance)

    
    return dist_metric_avg

def get_degree_dict(G: nx.Graph, edges_perturbed: list):
    #get perturbed nodes
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


def get_metrics(G, edges_perturbed):

    dist_metric_avg = get_distribution_dict(G, edges_perturbed)
    avg_degree, degree_sum = get_degree_dict(G, edges_perturbed)    
   
    
    return dist_metric_avg, avg_degree

def generate_row(G, add, remove, dataset, modelpath):
    row = []
    row.append(add); row.append(remove)
    _, data_p, edges_perturbed, perturb_type = randomly_perturb(G= G , data = dataset.data,remove = remove, add = add)
    row.append(perturb_type); 
    h, h_p, _ = generate_output(modelpath, dataset, data_p)
    test_acc, test_p_acc = compare_accuracy(dataset.data, h, h_p)
    row.append(test_acc); row.append(test_p_acc)
    accuracy_drop = 100*((test_acc -test_p_acc) / test_acc)
    row.append(accuracy_drop)
    dist_metric_avg, avg_degree =  get_metrics(G, edges_perturbed)
    row.append(dist_metric_avg); row.append(avg_degree); 

    
    return row

    