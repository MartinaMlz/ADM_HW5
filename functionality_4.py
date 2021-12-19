import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
import copy
import networkx as nx
import matplotlib.pyplot as plt
from utilities import *


'''
library that contains the function for functionality and visualization 4
'''


#####################################################################################################################
def plot_counterexample():
    '''
    plots a counterexample in the functionality 4 section in the notebook
    '''
    
    plt.figure(figsize = (20,5))
    DG = nx.DiGraph()
    DG.add_node('A',pos=(1,0))
    DG.add_node('C',pos=(2,-1))
    DG.add_node('D',pos=(2,1))
    DG.add_node('B',pos=(3,0))
    DG.add_weighted_edges_from([('A','C',1), ('C','D',1.5), ('D','A',3), ('D','B',3), ('B','C',1)])

    pos_attrs = {'s':(0,0.2), 'S': (4,0.2)}
    custom_node_attrs = {'s': 'Source', 'S': 'Sink'}

    subax1 = plt.subplot(131)
    pos=nx.get_node_attributes(DG,'pos')
    nx.draw(DG,pos, with_labels = True)
    nx.draw_networkx_labels(DG, pos_attrs, labels=custom_node_attrs)
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)


    DG2 = nx.DiGraph()
    DG2.add_node('A',pos=(1,0))
    DG2.add_node('C',pos=(2,-1))
    DG2.add_node('D',pos=(2,1))
    DG2.add_node('B',pos=(3,0))
    DG2.add_node('S',pos=(4,0))
    DG2.add_node('s',pos=(0,0))
    DG2.add_weighted_edges_from([('s', 'A', 10), ('B', 'S', 10), ('A','C',1), ('C','D',1.5), ('D','A',3), ('D','B',3), ('B','C',1)])

    pos_attrs = {'s':(0,0.2), 'S': (4,0.2)}
    custom_node_attrs = {'s': 'Source', 'S': 'Sink'}

    subax2 = plt.subplot(132)
    pos=nx.get_node_attributes(DG2,'pos')
    nx.draw(DG2,pos, with_labels = True)
    nx.draw_networkx_labels(DG2, pos_attrs, labels=custom_node_attrs)
    labels = nx.get_edge_attributes(DG2,'weight')
    nx.draw_networkx_edge_labels(DG2,pos,edge_labels=labels)

    DG3 = nx.DiGraph()
    DG3.add_node('A',pos=(1,0))
    DG3.add_node('C',pos=(2,-1))
    DG3.add_node('D',pos=(2,1))
    DG3.add_node('B',pos=(3,0))
    DG3.add_node('s',pos=(4,0))
    DG3.add_node('S',pos=(0,0))
    DG3.add_weighted_edges_from([('A', 'S', 10), ('s', 'B', 10), ('A','C',1), ('C','D',1.5), ('D','A',3), ('D','B',3), ('B','C',1)])

    pos_attrs = {'S':(0,0.2), 's': (4,0.2)}
    custom_node_attrs = {'s': 'Source', 'S': 'Sink'}

    subax3 = plt.subplot(133)
    pos=nx.get_node_attributes(DG3,'pos')
    nx.draw(DG3,pos, with_labels = True)
    nx.draw_networkx_labels(DG3, pos_attrs, labels=custom_node_attrs)
    labels = nx.get_edge_attributes(DG3,'weight')
    nx.draw_networkx_edge_labels(DG3,pos,edge_labels=labels)

    plt.show()
    
    return
#####################################################################################################################

def functionality_4(dataset_graphs, interval1, interval2, user1, user2):
    '''
    functionality 4:
    given two input intervals and two users, we build two associated graphs,
    we join them and we search for the smallest cut that disconnects the two users
    '''
    
    # get whether or not the intervals are disjoint
    is_disjoint, joined_intervals = parse_intervals(interval1, interval2)
    
    if is_disjoint:
        # if they are disjoint, we get the two graphs and we join them
        total_graph = get_total_graph(dataset_graphs, *interval1)
        total_graph.union(get_total_graph(dataset_graphs, *interval2))
    else:
        # if they are not disjoint, we get directly the final graph
        total_graph = get_total_graph(dataset_graphs, *joined_intervals)
    
    
    
    # make a copy of the graph
    graph = copy.deepcopy(total_graph)
    
    # find the edges to remove
    edges = find_min_cut(graph, user1, user2)
    
    # compute the total cost of the solution
    total_cost = 0
    for edge in edges:
        total_cost += total_graph.edges_list[edge]
    
    return total_graph, total_cost, edges


def parse_intervals(interval1, interval2):
    '''
    determines whether the input time intervals are disjoint or not
    and joins them in the first case
    '''
    
    if interval1[0] < interval2[0]:
        a1 = interval1[0]
        b1 = interval1[1]

        a2 = interval2[0]
        b2 = interval2[1]
    else:
        a1 = interval2[0]
        b1 = interval2[1]

        a2 = interval1[0]
        b2 = interval1[1]
    
    if b1 < a2:
        return True, None
    else:
        return False, [a1,b2]
    

def find_min_cut(graph, user1, user2):
    '''
    here we find the minimum edges to cut to disconnect the input users
    '''
    
    # we transform the graph to remove the antiparallel edges
    # as a first step to simplify the algorithm
    # 
    # and we remove the edge that connects a user to himself
    # (as they will not be relevant for the analysis of the flow)
    
    # retrieve the undirected edges
    antiparallel = graph.undirected_edges()
    removed_edges = {}
    
    # we use the fact that all the nodes are positive integers
    new_node = 0
    for a,b in antiparallel:
        # remove the edges in one of the directions
        weight = graph.remove_edge((a,b))
        
        if a != b:
            # we use negative integers to denote the added nodes
            new_node -= 1
            # we store the added node for future reference
            removed_edges[new_node] = (a,b)
            
            # we add a mock edge between a and b
            graph.add_edge(a,new_node,weight)
            graph.add_edge(new_node,b,weight)
    
    
    # we find the minimum cut from user1 to user2
    cut_1_to_2 = find_min_cut_A_to_B(graph, user1, user2)
    
    # we find the minimum cut from user1 to user2
    cut_2_to_1 = find_min_cut_A_to_B(graph, user2, user1)
    
    # we join the two cuts
    # this is the step in which we may lose the optimality
    total_edges = set.union(cut_1_to_2, cut_2_to_1)
    
    
    # among these edges, we substitute the newly added (the negative) nodes
    # and we restore the antiparallel edges for the final solution
    transformed_edges = set()
    for a,b in total_edges:
        # by construction there are no edges between two negative nodes
        if a < 0 or b < 0:
            new_node = min(a,b)
            transformed_edges.add(removed_edges[new_node])
        else:
            transformed_edges.add((a,b))
        
    return(transformed_edges)

def find_min_cut_A_to_B(graph, user1, user2):
    '''
    this function finds the minimum edge cut to disconnect
    user1 (the source) from user2 (the sink)
    '''
    # we add a 'supersource' that only connects to user1
    # and a 'supersink' that only connects to user2
    # so that the flow problem is well defined
    
    # we choose the weight of the edge from the supersource to user1 as some number
    # greater than the sum of all the outgoing weights from user1 (so that it won't affect the flow)
    source_weight = 1
    for weight in graph.adjacency[user1].values():
        source_weight += weight
    graph.add_edge('source', user1, source_weight)   # add the 'supersource'
    
    # it is sufficient to have sink_weight == source_weight for these
    # nodes to not influence the outcome of the algorithm
    sink_weight = source_weight
    graph.add_edge(user2, 'sink', sink_weight)   # add the 'supersink'
    
    # find the maximum flow in this flow problem
    flow = find_max_flow(graph)
    
    # find the minimum cut from the maximum flow
    selected_edges = find_min_cut_edges(graph, flow, source = 'source', sink = 'sink')
    
    # remove the added source and sink
    graph.remove_node('source')
    graph.remove_node('sink')
    
    return(selected_edges)

def find_max_flow(graph, source = 'source', sink = 'sink'):
    '''
    find maximum flow in the input flow problem
    
    we use edmond-karp algorithm
    '''
    # we initialize the flow dictionaries with all zeros
    # the keys of this dictionary will be the edges
    flow_dict = defaultdict(lambda : 0)
    
    # boolean variable to check if we found the maximum flow
    # (since a maximum flow is found when there are no augmenting paths)
    found_path = True
    
    while found_path:
        # search and return an augmenting path
        previous = search_augmenting_path(graph, flow_dict, source = source, sink = sink)
        
        # if the sink is not in the path, we did not find an augmenting path
        if sink not in previous.keys():
            found_path = False
        
        # if we found an augmenting path, we increase the flow accordingly
        if found_path:
            increase_flow(graph, flow = flow_dict, path = previous, source = source, sink = sink)
    
    
    return (flow_dict)

def search_augmenting_path(graph, flow, source, sink):
    '''
    this function searches an augmenting path in the residual graph
    '''
    # dictionary that will store the path
    # the keys will be the nodes of the path
    previous = {}
    
    # initialize residual graph
    residual_graph = weightedDiGraph()
    
    # we use BFS to search for a shortest path
    # (in this framework, it is sufficient to consider the residual
    #  graph as an unweighted graph)
    
    # we use a queue for BFS
    q = Queue()
    q.put(source)
    
    # BFS cycle
    while not q.empty():
        
        current_node = q.get()
        
        # we build the residual graph locally around the current node
        build_residual_graph(graph, residual_graph, current_node, flow)
        
        for node, capacity in residual_graph.adjacency[current_node].items():
            if node not in previous.keys():   # if the node is not already visited
                
                # the values associated to each path node is a dictionary
                # with two elements: the previous node in the path and the residual capacity of the edge
                previous[node] = {'previous' : current_node, 'capacity' : capacity}
                
                q.put(node)
            
            # we stop if we found the sink
            if node == sink:
                break
    
    return(previous)

def build_residual_graph(graph, residual_graph, current_node, flow):
    '''
    builds the residual graph around the input node
    '''
    
    # build the residual edges for the outgoing edges
    for node, weight in graph.adjacency[current_node].items():
        capacity = weight - flow[(current_node, node)]
        if capacity > 0:   # the edge is a residual edge only if it has positive residual capacity
            residual_graph.add_edge(current_node, node, capacity)
    
    # build the residual edges for the ingoing edges
    for node, weight in graph.distant_neighbours[current_node].items():
        capacity = flow[(node, current_node)]
        if capacity > 0:   # the edge is a residual edge only if it has positive residual capacity
            residual_graph.add_edge(node, current_node, capacity)
    
    return
    
def increase_flow(graph, flow, path, source, sink):
    '''
    increase the flow on the input augmenting path
    '''
    
    # we will run across the path backwards
    # and we stop at the source
    
    prev_node = path[sink]['previous']
    min_capacity = path[sink]['capacity']
    
    # here we build the edges that belong to the path
    # and we search for the residual capacity to add
    path_edges = [(prev_node, sink)]
    while prev_node != source:
        current_node = prev_node
        prev_node = path[prev_node]['previous']
        
        path_edges.append((prev_node, current_node))
        
        # the residual capacity is the minimum weight in the path
        if path[current_node]['capacity'] < min_capacity:
            min_capacity = path[current_node]['capacity']
    
    # now we increase the flow on all the path edges
    for edge in path_edges:
        
        if edge in graph.edges_list.keys():
            flow[edge] += min_capacity
        else:
            flow[edge] -= min_capacity
    
    return

def find_min_cut_edges(graph, flow, source, sink):
    '''
    finds the edges corresponding to the minimum cut
    using the maximum flow
    
    To find these edges we build the residual graph and we
    visit every node reachable from the source.
    Since the flow is maximum, the source and the sink are disconnected
    in the residual network.
    The minimum cut edges are then those between a visited and a non visited node.
    '''
    
    # dictionary to track the visited nodes
    visited = defaultdict(lambda : 0)
    
    # initialize the residual graph
    residual_graph = weightedDiGraph()
    
    # we use BFS to visit every reachable node
    
    # initialize the queue used in BFS
    q = Queue()
    q.put(source)
    visited[source] = 1
    
    while not q.empty():
        current_node = q.get()
        
        # we build the residual graph locally
        build_residual_graph(graph, residual_graph, current_node, flow)
        
        for node in residual_graph.adjacency[current_node].keys():
            if node not in visited.keys():
                q.put(node)
                visited[node] = 1
    
    # once every reachable node is marked as visited
    # we select the minimum cut edges
    selected_edges = set()
    for a,b in graph.edges_list.keys():
        if visited[a] ==1 and visited[b] == 0:
            selected_edges.add((a,b))
    
    return(selected_edges)


def plot_4(graph, edges, total_cost, user1, user2):
    '''
    this function shows a visualization of the links needed to be removed in order to disconnect the users

    input: graph,
           edges to remove,
           user1,
           user2
    '''
    
    print('-------------------------------------------------------------------------------------')
    if len(edges)==0:
        print(f'Users {user1} and {user2} are not connected. There are no edges to be removed.')
        return
    else:
        print(f'There are {len(edges)} edges to be removed, for a total cost of {total_cost}.')
    print('-------------------------------------------------------------------------------------')
    
    
    # convert input graph to networkx graph
    H = graph.to_nx()

    # we compute all the nodes that belong to a path
    # between user1 and user2 and use the induced subgraph for the visualization
    all_nodes_in_between = set()
    for path in nx.all_simple_paths(H, user1, user2):
        all_nodes_in_between.update(path)
    for path in nx.all_simple_paths(H, user2, user1):
        all_nodes_in_between.update(path)
    
    H = nx.DiGraph(H.subgraph(all_nodes_in_between))
    
    # we also remove the edges between the users and themselves for better visualization
    H.remove_edges_from([(user1, user1), (user2, user2)])
    
    
    # we fix the position of the users and their labels
    fixed_pos = { user1 : (-1,-1), user2 : (1,1)}
    label_pos = { user1 : (-1,-0.9), user2 : (1,0.9)}
    labels = {user1 : user1, user2 : user2}
    
    # we color-code the nodes and the edges
    node_colors = ["red" if node in [user1, user2] else "blue" for node in H.nodes()]
    node_sizes = list(map(lambda x: 250 if x=='red' else 50, node_colors))
    edge_colors = ['orange' if edge in edges else 'black' for edge in H.edges()]
    edge_widths = list(map(lambda x: 1.5 if x=='orange' else 0.5, edge_colors))
    
    # we label every edge to remove with its weight
    edge_labels = {}
    for a,b in edges:
        if (b,a) in edge_labels.keys():
            # because networkx does not allow proper edge label visualization for antiparallel edges
            # so we give the sum of the two weights in this last case
            edge_labels[(b,a)] += H.edges()[(a,b)]['weight']
        else:
            edge_labels[(a,b)] = H.edges()[(a,b)]['weight']
    
    for edge, weight in edge_labels.items():
        edge_labels[edge] = "{:.2f}".format(weight)
     
    
    pos = nx.spring_layout(H, pos = fixed_pos, fixed = fixed_pos, center = (0,0), scale = None)

    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot([0],[0],color = 'orange', label='edges to remove')
    
    nx.draw_networkx_nodes(H, pos = pos, node_color = node_colors, node_size = node_sizes)
    nx.draw_networkx_edges(H, pos = pos, edge_color = edge_colors, width = edge_widths, arrows = True, connectionstyle='arc3, rad = 0.01')
    nx.draw_networkx_labels(H, pos = label_pos , labels = labels, font_size=15, font_color='black')
    nx.draw_networkx_edge_labels(H, pos, edge_labels = edge_labels)
    
    ax.legend(loc='upper left')
    
    plt.show()
    
    return