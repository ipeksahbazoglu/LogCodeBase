import networkx as nx
from modules.perturb import randomly_perturb
from modules.GNN import generate_output
from modules.util import homophily
import networkx as nx
import numpy as np 
from typing import List
import pandas as pd
from torch_geometric.datasets import planetoid


class GraphPerturbation:

    def __init__(self, G: nx.Graph, 
                 dataset: planetoid.Planetoid, 
                 max_add_budget: float, max_remove_budget:float, path = str):
        
        self.G = G
        self.path_length_lookup = dict(nx.all_pairs_shortest_path_length(self.G))
        self.nodes = max(nx.connected_components(G), key=len)
        self.dataset = dataset
        self.add_budget = max_add_budget
        self.remove_budget = max_remove_budget
        self.path = path

        self.result_df = pd.DataFrame(columns = ['AdditionBudget', 'RemovalBudget', 'PerturbationStrategy',
        'PerturbedTestFScores', 'TruePositive', 'FalseNegative', 'FalsePositive', 'DistanceMetric', 
        'DegreeMetric', 'PerturbedHomophily'])
        

    def __get_degree_metric(self, edge_list: List[tuple]):
        """Returns the average and total node degree of a list of edges in a graph.

        Args:
            G (nx.Graph): Graph.  
            edge_list (list): list of edges (sets of nodes).

        Returns:
            average node degree
        """
        nodes = set()
        degree_sum = 0
        for edge in edge_list:
            for node in edge:
                if node not in nodes:
                    nodes.add(node)
                    degree_sum += self.G.degree(node)

        #calculate avg degree of nodes perturbed
        degree_avg = degree_sum / len(nodes)
        
        return degree_avg

    def __find_path(self, edge1: tuple, edge2: tuple):
        """ Sums path lengths between each node pairing given two edges.
            Path lengths between nodes are precalculated and provided in a lookup dictionary.

        Args:
            edge1 (tuple): an edge defined as a tuple of two nodes.
            edge2 (tuple): another edge defined as a tuple of two nodes.
            path_length_lookup (dict): lookup dictionary for the path length between nodes.

        Returns:
            (float): returns the average path length between unique node pairings of two edges. 
        """
        paths = [
        self.path_length_lookup[edge1[0]][edge2[0]], self.path_length_lookup[edge1[0]][edge2[1]], 
        self.path_length_lookup[edge1[1]][edge2[0]], self.path_length_lookup[edge1[1]][edge2[1]] 
        ]
        
        return sum(paths) / len(paths)

    def __get_edge_distance_metric(self, edge_list: list):
        """ calculates distance metric for each edge pairing from the given edge list.  

        Args:
            G (nx.Graph): _description_
            edges_perturbed (list): _description_
            ddict (dict): _description_

        Returns:
            (float): average value for the distance metric 
        """

        distance_metric_dict = dict()

        if len(edge_list)  == 1:
            #if only on edge is perturbed, return the path length between two nodes forming the edge. 
            distance = self.path_length_lookup[edge_list[0][0]][edge_list[0][1]]
            distance_metric_dict[(edge_list[0])] = distance

        else:
            for i in range(len(edge_list)):
                edge_1 = edge_list[i]
                for edge_2 in edge_list[(i+1):]:
                    a_distance = self.__find_path(edge_1, edge_2)
                    distance_metric_dict[(edge_1, edge_2)] = a_distance
            
        return sum(distance_metric_dict.values()) / len(distance_metric_dict)


    def __get_metrics(self, edges_perturbed :list):

        dist_metric_avg = self.__get_edge_distance_metric(edges_perturbed)
        avg_degree = self.__get_degree_metric(edges_perturbed)    
    
        return dist_metric_avg, avg_degree



    def __generate_row(self, model, add_budget, remove_budget):
                row = []
                row.append(add_budget); row.append(remove_budget)
                _, data_p, ea,er, perturb_type = randomly_perturb(G= self.G , data = self.dataset.data,remove = remove_budget, add = add_budget)
                row.append(perturb_type); 
                fscores, tp, fn, fp = generate_output(model, self.dataset, data_p)
                row.append(fscores); row.append(tp);row.append(fn);row.append(fp)
                dist_metric_avg, degree_metric_avg =  self.__get_metrics(ea + er)
                row.append(dist_metric_avg); row.append(degree_metric_avg); 
                _, hp = homophily(data_p, self.nodes)
                row.append(hp)

                self.result_df.loc[len(self.result_df)] = row

    def generate_results(self, model, trials):
        for add_budget in np.arange(0, self.add_budget, 0.1):
            for remove_budget in np.arange(0, self.remove_budget, 0.1):
                if (add_budget == 0.) and (remove_budget == 0.):
                    continue
                for i in range(trials): 
                    self.__generate_row(model, add_budget, remove_budget)
                    if i % 20 == 0: 
                        self.result_df.to_csv(self.path, index = False)
        return self.result_df

    def get_result_df(self):
        return self.result_df




