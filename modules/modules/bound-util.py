import numpy as np

def get_Eu_node(E):
    return np.argmax(np.abs(np.array(E.sum(0))))

def get_remove_bound(E, G, Gp,node):
    """Calculate term relating to the edges removed from max node"""
    v_list = []
    for v in G.neighhbors(node):
        if (node, v) not in Gp.edges():
            v_list.append(v)
    Eu_removed= 0
    
    # for v in v_list:
    #     Eu_removed += np.abs(E[node, v])
    
    #neighbor with the smallest degree from unperturbed graph
    delta = np.min([G.degree(v) for v in G.neighbors(node)])
    
    #calculate bound term
    Eu_bound = len(v_list)/np.sqrt(delta*G.degree(node))

    return Eu_bound

def get_addition_bound(E, G, Gp, node):
    """Calculate term relating to the edges added to the max node"""
    v_list = []

    for v in Gp.neighhbors(node):
        if (node, v) not in G.edges():
            v_list.append(v)
    
    #find neighbor with the smallest degree in the perturbed graph

    delta = np.min([Gp.degree(v) for v in Gp.neighbors(node)])

    Eu_bound = len(v_list)/np.sqrt(delta*Gp.degree(node))

    return Eu_bound

def get_remain_bound(E, G, Gp, node):
    """Calculate term relating to the edges remaining of max node"""
    v_list = []

    for v in G.neighhbors(node):
        if (node, v) in Gp.edges():
            v_list.append(v)
    
    #find neighbor with the smallest degree in the perturbed graph
    alpha = get_alpha(G, Gp, node)
    delta = np.min([Gp.degree(v) for v in Gp.neighbors(node)])

    if alpha >=1:
        Eu_bound = None
    else:
        first_term = (alpha)/(1-alpha)
        second_term = len(v_list)/ np.sqrt(G.degree(node)*delta)
        Eu_bound = first_term*second_term
    return Eu_bound


def get_alpha(G, Gp, node):
    #change of degree around node
    node_list = [n for n in G.neighbors(node)]
    node_list.append(node)

    return np.max([np.abs(G.degree(n) - Gp.degree(n)) / G.degree(n) for n in node_list])
