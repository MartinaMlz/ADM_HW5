import random
import networkx as nx
import matplotlib.pyplot as plt
from utilities import get_total_graph, weightedDiGraph, dijkstra



def functionality_3(users_list, datasets=None, start_time=None, end_time=None, 
                    start_user=None, end_user=None, g=None, verbose=True):
    """
    Functionality 3: Given a list of nodes, compute the shortest pass passing 
                true the nodes in the given order.

    Args:
        users_list (list[int]): List of nodes.
        datasets (tuple[MultiDiGraph]): list of the three datasets.
        start_time (int, optional): Sarting time for the subGraph of the 
            original Graph. Defaults to None.
        end_time (int, optional): Ending time for the subGraph of the 
            original Graph. Defaults to None.
        start_user (int, optional): Source user for functionality 3. 
            Defaults to None.
        end_user (int, optional): Destination user for functionality 3. 
            Defaults to None.
        g (DiGraph, optional): Graph object. If provided, Defaults to None.
        verbose (bool, optional): Print functionality during the path 
            computation. Defaults to True.

    Raises:
        ValueError: If sart_user or end user not in users_list, 
            an error is raised.

    Returns:
        list[tuple[int]]: List of edges from the source node to the 
            destination node.

    """  
    # If weightedDiGraph object is not provided
    if g is None:
        # Check start_time and end_time are not None if weightedDiGraph is not provided
        assert datasets is not None, 'Datasets must be provided if a graph object is not given!'
        assert start_time is not None, "start_time can't be none if weightedDiGraph is not provided!"
        assert end_time is not None, "end_time can't be none if weightedDiGraph is not provided!"
        assert end_time > start_time, 'end_time is befor start_time!'
        
        # Compute the graph from the three files
        g = get_total_graph(datasets, start_time, end_time)

    # If start_user is not provided, take the first user in user_list as start_user
    if start_user is not None:
        try:
            # Remove start_user from the list and add it to the top
            start_user = users_list.pop(
                            users_list.index(start_user)
                            )
            users_list.insert(0, start_user)
        except ValueError:
            raise ValueError(f'start_user={start_user} not in users_list')
    # If end_user is not provided, take the last user in user_list as end_user
    if end_user is not None:
        try:
            # Remove end_user from the list and add it to the end
            end_user = users_list.pop(
                            users_list.index(end_user)
                            )
            users_list.append(end_user)
        except ValueError:
            raise ValueError(f'end_user={end_user} not in user_list')
    
    print(users_list[0], '--->', users_list[1:-1], '--->', users_list[-1])
    users_pairs = zip(users_list[:-1], users_list[1:])
    paths_list = []
    tot_weight = 0
    
    for pairs in users_pairs:
        dijk = dijkstra(g, *pairs)
        if not dijk['path']:
            print(f'Path: {pairs[0]} ---> {pairs[1]} NOT POSSIBLE')
            return "Not possible"
        if verbose:
            print(f'\nPath {pairs[0]} ---> {pairs[1]}:', dijk['path'],
                  '\nWeight:', dijk['weigth'])
        paths_list.extend(dijk['path'])
        tot_weight += dijk['weigth']
     
    if verbose:
            print('\nFull path:', paths_list, '\n\nFull dist:', tot_weight)
    return paths_list


def plot_path(path, figsize=(15, 15)):
    """
    Visualization function for functionality 3.

    Args:
        path (list[tuple[int]]): Path to visualize.

    Returns:
        None.

    """
      
    # Create the networkx DiGraph of the path
    path_graph = nx.DiGraph()
    # Add the edges
    path_graph.add_weighted_edges_from(path)
    # Arrange the nodes in a circular fashion
    pos = nx.circular_layout(path_graph) 
    plt.figure(figsize=figsize)
    
    # Draw the path in red
    nx.draw(path_graph, pos, with_labels=True, font_weight='bold',  node_color='r', edge_color='r',
            node_size=2000/len(path_graph.nodes), font_size=150/len(path_graph.nodes) )
    # Add the weights' labels
    labels = nx.get_edge_attributes(path_graph, "weight")
    # Draw the labels
    nx.draw_networkx_edge_labels(path_graph, pos, edge_labels=labels, font_size=150/len(path_graph.nodes))
    plt.axis('equal')
     
    plt.show()
    
def get_example_graph(n_nodes=1000, edges_range=(1, 10,), weights=[1, 2, 5], seed=432):
    """
    Random weightedDiGraph generator.

    Args:
        n_nodes (int, optional): Number of node for the graph. 
            Defaults to 1000.
        edges_range (tuple[int], optional): Range of possible edges for 
            each node. Defaults to (1, 10,).
        weights (list[int], optional): List of possible values for each edge. 
            Defaults to [1, 2, 5].
        seed (int, optional): Seed value for graph generation. Defaults to 432.

    Returns:
        graph (TYPE): DESCRIPTION.

    """
    graph = weightedDiGraph()
    random.seed(seed)
    
    # Add nodes
    for v in range(n_nodes):
        graph.add_node(v)
    
    # Add edges
    for v in graph.nodes:
        # For each node choose the number of edges
        n_edges = random.choice(range(*edges_range))
        
        # Connect the node with other nodes
        for _ in range(n_edges):
            # Select the destination node
            v_to = random.choice(graph.nodes)
            # Select the edge weight
            w = random.choice(weights) 
            # Add the edge to the graph
            graph.add_edge(v, v_to, w)
           
    return graph
