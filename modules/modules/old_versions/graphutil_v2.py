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

                 
            else:
                distance = nx.shortest_path_length(G, node1, node2)
                dist[(node1, node2)]  = distance
                dist[(node2, node1)]  = distance
                all_paths.append(distance)
                min_distance = min(all_paths)
                mean_distance = sum(all_paths)/len(all_paths)
            

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
            min_distance[(edge_i, edge)] = m_distance
            avg_distance[(edge_i, edge)] = a_distance
    
    dist_metric_min = sum(min_distance.values()) / len(min_distance)
    dist_metric_avg = sum(avg_distance.values()) / len(avg_distance)

    
    return dist_metric_min, dist_metric_avg

def separate_components(G, edges):
    remaining = []
    connected = []
    for edge in edges:
        if nx.has_path(G, edges[0][0], edge[0]):
            connected.append(edge)
        else:
            remaining.append(edge)

    return remaining, connected

def connected_perturbations(G, edges_perturbed):
    remaining = edges_perturbed
    components = []

    while len(remaining) >=1 :
        remaining, connected = separate_components(G, remaining)
        components.append(connected)
    
    dist_min = []
    dist_avg = []

    for component in components:

        if len(component) > 1:
            dist_metric_min, dist_metric_avg = get_distribution_dict(G, component)
            dist_min.append(dist_metric_min / len(nx.node_connected_component(G, component[0][0])))
            dist_avg.append(dist_metric_avg / len(nx.node_connected_component(G, component[0][0])))
        else:
            dist_min.append(0)
            dist_avg.append(0)
            
    dist_metric_avg = (sum(dist_avg) / len(dist_avg))
    dist_metric_min = (sum(dist_min) / len(dist_min))
    
    return dist_metric_avg, dist_metric_min


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
    min_dist, avg_dist =  connected_perturbations(G, edges_perturbed)
    _, avg_degree = get_degree_dict(G, edges_perturbed)
    row.append(min_dist); row.append(avg_dist); row.append(avg_degree)
    return row

    