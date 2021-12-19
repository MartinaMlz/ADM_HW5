import math
from utilities import get_total_graph, dijkstra
from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

'''
library that contains the function for functionality and visualization 2
'''

def degree_centrality(graph, node):
    """Get the degree centrality measure for a given node"""
    # Degree formula
    degree = graph.degree(node) / (graph.n_nodes-1)
    return degree

def closeness_centrality(graph, node):
    """Get the closeness centrality measure for a given node"""
    
    tot_nodes = graph.n_nodes - 1
    total_dist = 0
    
    if len(graph.adjacency[node])==0:
        # if the node is isolated we return 0
        return(0)
    
    # get distances between node and all the others
    distances = dijkstra(graph, node)['weigth']
    
    for n, dist in distances.items():
        if dist < math.inf:
            total_dist += distances[n]

    return tot_nodes / total_dist


def dijkstra2(graph, node):
    """Compute shortest path with Dijkstra"""
    # Initialize the code
    graph = deepcopy(graph)
    distances = dict.fromkeys(graph.nodes, math.inf)
    path = dict.fromkeys(graph.nodes)
    to_be_visited = list(graph.nodes)

    distances[node] = 0
    
    while(to_be_visited):
        min_ = to_be_visited[0]
        for n in to_be_visited:
            if distances[n] < distances[min_]:
                min_ = n
            
        neighbors = list(graph.adjacency[node])
        try:
            neighbors.remove(node)
        except:
            pass
        for n in neighbors:
            candidate = distances[min_] + graph.adjacency[min_][n]
            if candidate < distances[n]:
                distances[n] = candidate
                path[n] = min_
        to_be_visited.remove(min_)
    
    return path, distances

def nodes_path(previous, start_node, end_node):
    """"""
    path = []
    node = end_node
    
    while node != start_node:
        if node == None:
            return path
        path.append(node)
        node = previous[node]
    path.append(start_node)
    
    return list(reversed(path))


def total_shortest_paths(graph, node):
    """"""
    all_nodes = list(graph.nodes)
    sigma = 0
    for n in all_nodes:
        tot_path = 0
        tot_node_path = 0
        previous , _ = dijkstra2(graph, n)
        for x in previous.keys():
            if previous[x]!=None:
                tot_path += 1
            path = nodes_path(previous, n, x)
            if node in path:
                tot_node_path += 1
        try:
            sigma += tot_node_path / tot_path
        except ZeroDivisionError:
            pass
    
    return sigma


def betweenness_centrality(graph, node):
    """Get the betweenness centrality mesure for a given node"""
    tot_nodes = len(graph.nodes)
    constant = 2 / ((tot_nodes**2) - 3 * tot_nodes + 2)
    term = total_shortest_paths(graph, node)
    bet = term * constant
    return bet


def pagerank_vector(graph, alpha=0.85, max_iter=100, tol=1e-03):
    
    """Get the pagerank vector for a given graph"""
    node_encoding = {}
    for idx, node in enumerate(graph.nodes):
        node_encoding[node] = idx

    # compute adjacency matrix
    P = np.zeros((graph.n_nodes, graph.n_nodes))
    for node, idx in node_encoding.items():
        total_weight = sum(graph.adjacency[node].values())
        for adjacent_node, weight in graph.adjacency[node].items():
            P[idx, node_encoding[adjacent_node]] = weight/total_weight

    # compute the transpose pagerank matrix
    P = (1 - alpha) * np.ones((graph.n_nodes, graph.n_nodes)) / graph.n_nodes + alpha * P

    # renormalize the matrix
    P = P / np.broadcast_to(np.sum(P, axis = 1)[:, np.newaxis], (graph.n_nodes, graph.n_nodes))
    
    # starting vector
    q = np.ones((graph.n_nodes, 1)) / graph.n_nodes
    
    difference = tol + 1
    i = 0
    
    while ( i < max_iter and difference > tol ):
        
        i += 1
        
        P = np.linalg.matrix_power(P, 2)
        new_q = np.matmul(P, q)
        
        difference = np.linalg.norm(new_q - q)
        
        q = new_q
        
    
    
    pagerank_vector = {}
    for node, idx in node_encoding.items():
        pagerank_vector[node] = q[idx, 0]
    return(pagerank_vector)
    

def pagerank(graph, input_node, alpha=0.85, max_iter=100, tol=1e-03):
    """Get the pagerank score for a given node"""
    
    # compute the pagerank vector
    pagerank_vec = pagerank_vector(graph, alpha = alpha, max_iter = max_iter, tol = tol)
    
    return(pagerank_vec[input_node])



def functionality_2(node, start_time, end_time, metric, datasets):
    """
    Compute a given metric for a specific node. For the available metrics
    watch "metric" in Args.

    Args:
        node (int): Target node for the measure computation.
        start_time (int): Sarting time for the subGraph of the 
            original Graph.
        end_time (int): Ending time for the subGraph of the 
            original Graph.
        metric (int): Metric to compute. Available values:
            
            1: Betweenness Centrality
            2: Page Rank
            3: Closeness Centrality
            4: Degree Centrality
        
        dataset (tuple[MultiDiGraph]): List of datasets.
        
    Returns:
        list, float: Output of the required measu.

    """ 
    graph = get_total_graph(datasets, start_time, end_time)
    if(metric == 'BETWEENESS'):
        return betweenness_centrality(graph, node), graph
    elif(metric == 'PAGERANK'):
        return pagerank(graph, node), graph
    elif(metric == 'CLOSENESS CENTRALITY'):
        return closeness_centrality(graph, node), graph
    elif(metric == 'DEGREE CENTRALITY'):
        return degree_centrality(graph, node), graph
    
    
def plot_metric(graph, node, metric, figsize=(12, 10)):
    if(metric == 'BETWEENESS'):
        
        # Create the networkx DiGraph of the path
        path_graph = nx.DiGraph()
        
        # Calculate the paths
        paths = dijkstra(graph, node)
        # Add the edges
        for node in paths['path'].keys():
            if paths['path'][node]:
                path_graph.add_weighted_edges_from(paths['path'][node])
        # Arrange the nodes in a circular fashion
        pos = nx.shell_layout(path_graph) 
        plt.figure(figsize=figsize)
        
        # Draw the path in red
        nx.draw(path_graph, pos, with_labels=True, font_weight='bold',  node_color='r', edge_color='r',
                node_size=7000/len(path_graph.nodes), font_size=350/len(path_graph.nodes) )
        # Add the weights' labels
        labels = nx.get_edge_attributes(path_graph, "weight")
        # Draw the labels
        nx.draw_networkx_edge_labels(path_graph, pos, edge_labels=labels, font_size=150/len(path_graph.nodes))
        plt.axis('equal')
         
        plt.show()
    
    elif(metric == 'PAGERANK'):
        
        # convert to networkx graph for visualization
        H = graph.to_nx()

        p = pagerank_vector(graph)

        # set the colormap
        cmap = cm.get_cmap('viridis')


        # retrieve the node colors
        node_colors = [cmap(p[node]) for node in H.nodes()]


        fig = plt.figure(figsize = (18,10))

        # add grid specifications for the color legend
        gs = fig.add_gridspec(1, 18)

        # the plot of the graph
        ax = fig.add_subplot(gs[:,0:17])

        ax.set_title('Pagerank distribution')


        pos = nx.spring_layout(H)


        nx.draw_networkx_nodes(H, pos = pos, node_color = node_colors, ax = ax)
        nx.draw_networkx_edges(H, pos = pos, ax = ax, edge_color = 'gray')

        # the plot of the color legend
        ax1 = fig.add_subplot(gs[:,17])

        ax1.set_title('Score')

        # color gradient to plot the legend
        gradient = np.flip(np.linspace(0, 1, 256))
        gradient = np.transpose(np.vstack((gradient, gradient)))
        ax1.imshow(gradient, aspect='auto', cmap=cmap)

        # removing xticks from the legend
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # correctly setting the yticks in the legend
        ax1.yaxis.tick_right()
        ax1.set_yticks([0,256])
        ax1.set_yticklabels([1, 0])

        plt.show()
    
    elif(metric == 'CLOSENESS CENTRALITY'):
        # Create the networkx DiGraph of the path
        path_graph = nx.DiGraph()
        
        # Calculate the paths
        paths = dijkstra(graph, node)
        # Add the edges
        for node in paths['path'].keys():
            if paths['path'][node]:
                path_graph.add_weighted_edges_from(paths['path'][node])
        # Arrange the nodes in a circular fashion
        pos = nx.spring_layout(path_graph) 
        plt.figure(figsize=figsize)
        
        # Draw the path in red
        nx.draw(path_graph, pos, with_labels=True, font_weight='bold',  node_color='r', edge_color='r',
                node_size=7000/len(path_graph.nodes), font_size=350/len(path_graph.nodes) )
        # Add the weights' labels
        labels = nx.get_edge_attributes(path_graph, "weight")
        # Draw the labels
        nx.draw_networkx_edge_labels(path_graph, pos, edge_labels=labels, font_size=150/len(path_graph.nodes))
        plt.axis('equal')
         
        plt.show()
        

    elif(metric == 'DEGREE CENTRALITY'):    
        # Create the networkx DiGraph of the path
        plot_graph = nx.DiGraph()
        # Add the edges
        for node0 in graph.adjacency[node].keys(): 
            plot_graph.add_edge(node, node0, weight=graph.adjacency[node][node0])
        for node0 in graph.distant_neighbours[node].keys(): 
            plot_graph.add_edge(node0, node, weight=graph.distant_neighbours[node][node0])
        # Arrange the nodes in a circular fashion
        pos = nx.circular_layout(plot_graph) 
        plt.figure(figsize=figsize)
        
        # Draw the path in red
        nx.draw(plot_graph, pos, with_labels=True, font_weight='bold',  node_color='r', edge_color='r',
                node_size=5000/len(plot_graph.nodes), font_size=250/len(plot_graph.nodes) )
        # Add the weights' labels
        labels = nx.get_edge_attributes(plot_graph, "weight")
        # Draw the labels
        nx.draw_networkx_edge_labels(plot_graph, pos, edge_labels=labels, font_size=150/len(plot_graph.nodes))
        plt.axis('equal')
         
        plt.show()