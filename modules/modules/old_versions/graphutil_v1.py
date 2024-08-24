import networkx as nx
from modules.perturb import randomly_perturb
from modules.GCN import compare_accuracy, generate_output

def find_path(G: nx.Graph, edge1: tuple, edge2: tuple, dist: dict):
    all_paths = []
    for node1 in edge1:
        for node2 in edge2:
            if ((node1, node2) in dist):
                distance = dist[(node1, node2)] 
                all_paths.append(distance)
                min_distance = min(all_paths)
                mean_distance = sum(all_paths)/len(all_paths)

                 
            elif nx.has_path(G, node1, node2):
                distance = nx.shortest_path_length(G, node1, node2)
                dist[(node1, node2)]  = distance
                dist[(node2, node1)]  = distance
                all_paths.append(distance)
                min_distance = min(all_paths)
                mean_distance = sum(all_paths)/len(all_paths)
            
            else:
                min_distance = -1
                mean_distance = -1

    return min_distance, mean_distance, dist

def get_distribution_dict(G: nx.Graph, edges_perturbed: list):
    #find pairwise distance between all nodes involved in a pair of edges
    dist = dict()
    min_distance = dict()
    avg_distance = dict()

    for i in range(len(edges_perturbed)):
        edge_i = edges_perturbed[i]
        for edge in edges_perturbed[(i+1):]:
            m_distance, a_distance, dist = find_path(G, edge_i, edge, dist)
            if m_distance != -1:
                min_distance[(edge_i, edge)] = m_distance
            if a_distance != -1:
                avg_distance[(edge_i, edge)] = a_distance
    
    dist_metric_min = sum(min_distance.values()) / len(min_distance)
    dist_metric_avg = sum(avg_distance.values()) / len(avg_distance)

    
    return dist_metric_min, dist_metric_avg


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
    return node_p, avg_degree

def generate_row(G, add, remove, dataset, modelpath):
    row = []
    row.append(add); row.append(remove)
    _, data_p, edges_perturbed, perturb_type = randomly_perturb(G= G , data = dataset.data,remove = remove, add = add)
    row.append(perturb_type); row.append(edges_perturbed); 
    h, h_p, _ = generate_output(modelpath, dataset, data_p)
    row.append(h.detach().numpy()); row.append(h_p.detach().numpy())
    test_acc, test_p_acc = compare_accuracy(dataset.data, h, h_p)
    row.append(test_acc); row.append(test_p_acc)
    accuracy_drop = 100*((test_acc -test_p_acc) / test_acc)
    row.append(accuracy_drop)
    min_dist, avg_dist =  get_distribution_dict(G, edges_perturbed)
    _, avg_degree = get_degree_dict(G, edges_perturbed)
    row.append(min_dist); row.append(avg_dist); row.append(avg_degree)
    return row

    