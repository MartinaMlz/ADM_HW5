import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
import copy
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

'''
library that contains the custom graph and queue objects that we built
'''

# (global) standard working interval
standard_interval = [0, 3000000000]

# (global) weight parameters
a2q_weight = 0.5
c2a_weight = 0.2
c2q_weight = 0.3

# (global) datasets paths
filename_a2q = 'data/sx-stackoverflow-a2q.txt'
filename_c2a = 'data/sx-stackoverflow-c2a.txt'
filename_c2q = 'data/sx-stackoverflow-c2q.txt'

# (global) lenght of every dataset
dataset_length = {filename_a2q : 17823525, filename_c2a : 25405374, filename_c2q : 20268151}

class DiGraph():
    '''
    basic directed graph
    '''
    
    def __init__(self, default_attribute = None):
        """
        internally we save it both as an adjacency list, as a list of edges,
        and as a 'reverse' adjacency list for faster computation
        """
        
        # we use default dictionaries to be able to easily insert new nodes and edges
        # without overwriting and eventual already existing edge
        
        def default_None_dict():
            return defaultdict(lambda : default_attribute)
        
        self.edges_list = defaultdict(lambda : default_attribute)   # dictionary with (directed) edges as keys and a default dummy value
        
        self.adjacency = defaultdict(default_None_dict)             # dictionary with nodes as keys and dictionary as default value
                                                                    # the default value dictionary is a default dictionary that contains
                                                                    # neighbours as keys and a default dummy value
                
        self.distant_neighbours = defaultdict(default_None_dict)    # dictionary with nodes as keys and dictionary as default value
                                                                    # the default value dictionary is a default dictionary that contains
                                                                    # all the nodes that have the considered one as neighbours as keys
                                                                    # and a default dummy value
                
    
    @property
    def graph_type(self):
        return 'Directed Graph'
    
    def __repr__(self):
        """"""
        return 'DiGraph()'
    
    def __getitem__(self, key):
        '''
        G[i] are the neighbours of the node i
        '''
        return list(self.adjacency[key].keys())
    
    def add_node(self, node):
        '''
        adds a node in the graph
        '''
        self.adjacency[node]           # thanks to the defaultdict we just need to call
        self.distant_neighbours[node]  # a key to insert the node
    
    @property
    def nodes(self):
        '''
        returns the list of nodes
        '''
        return list(self.adjacency.keys())
    
    @property
    def edges(self):
        '''
        returns the list of edges
        '''
        return list(self.edges_list.keys())
    
    def adj(self):
        '''
        returns the adjacency list
        We convert the dictionary structure to as simple dictionary of lists
        '''
        return(dict(map(lambda x: (x[0], list(x[1].keys())), self.adjacency.items())))
    
    def add_edge(self, edge0, edge1):
        '''
        adds an edge in the graph
        '''
        # add the terminal nodes if they don't exist
        self.add_node(edge0)
        self.add_node(edge1)
        
        # add the edge
        self.adjacency[edge0][edge1]
        self.distant_neighbours[edge1][edge0]
        self.edges_list[(edge0, edge1,)]
    
    @property
    def n_nodes(self):
        '''
        number of nodes
        '''
        return len(self.adjacency)
    
    @property
    def n_edges(self):
        '''
        number of edges
        '''
        return len(self.edges_list)
    
    @property
    def get_average_links(self):
        '''
        average number of links per node
        '''
        return self.n_edges / self.n_nodes
    
    @property
    def get_density(self):
        '''
        density of a directed graph
        '''
        possible_edges = self.n_nodes * (self.n_nodes - 1)   # number of all possible edges
        return self.n_edges / possible_edges
    
    @property
    def isDense(self):
        '''
        we define a graph as dense if its density is at least 0.5
        '''
        is_dense = self.get_density >= 0.5
        return is_dense
    
    @property
    def isSparse(self):
        '''
        we define a graph as sparse if it's not dense
        '''
        return (not self.isDense)
    
    def add_edges_from(self, iterable):
        '''
        add edges from an iterable
        '''
        for edge in iterable:
            self.add_edge(*edge)
    
    def remove_edge(self, edge):
        '''
        removes the input edge from the graph
        '''
        self.adjacency[edge[0]].pop(edge[1])
        self.distant_neighbours[edge[1]].pop(edge[0])
        return(self.edges_list.pop(edge))
    
    def remove_node(self, node):
        '''
        removes the input node from the graph
        '''
        # removes all outgoing edges
        for arrive in self.adjacency[node].keys():
            self.edges_list.pop((node, arrive))
            self.distant_neighbours[arrive].pop(node)
        
        # removes all ingoing edges
        for start in self.distant_neighbours[node].keys():
            self.edges_list.pop((start, node))
            self.adjacency[start].pop(node)
        
        # removes the node
        self.distant_neighbours.pop(node)
        self.adjacency.pop(node)
        
    def indegree(self, node):
        '''
        returns the indegree of the input node
        '''
        return len(self.adjacency[node])
    
    def outdegree(self, node):
        '''
        returns the outdegree of the input node
        '''
        return len(self.distant_neighbours[node])
    
    def degree(self, node):
        '''
        returns the degree of the input node
        '''
        return self.indegree(node) + self.outdegree(node)
    
    def to_nx(self):
        '''
        converts a DiGraph() to a networkX digraph
        '''
        DG = nx.DiGraph()
        DG.add_nodes_from(self.adjacency.keys())
        DG.add_edges_from(self.edges_list.keys())
        return(DG)
    
    
class weightedDiGraph(DiGraph):
    '''
    weighted directed graph
    '''
    
    def __init__(self):
        '''
        we initialize the weighted digraph as a digraph with default attribute 0
        so the values in our default dictionaries will be the weight of the edges
        '''
        super().__init__( 0 )
    
    
    def __repr__(self):
        """"""
        return 'weightedDiGraph()'
    
    @property
    def graph_type(self):
        return 'Weighted Directed Graph'
    
    def __getitem__(self, key):
        '''
        G[i] are the tuples (neighbours, weight) of the node i
        '''
        return dict(self.adjacency[key])
    
    @property
    def edges(self):
        '''
        returns the list of tuples (edge, weight)
        '''
        return list(self.edges_list.items())
    
    def adj(self):
        '''
        returns the adjacency list
        We convert the dictionary structure to as simple dictionary of lists
        '''
        return(dict(map(lambda x: (x[0], dict(x[1])), self.adjacency.items())))
    
    def add_edge(self, edge0, edge1, weight):
        '''
        adds a weighted edge in the graph
        if the edge already exists, we sum the weights
        '''
        assert weight > 0, 'Weight must be greater than 0'
        
        # add the terminal nodes if they don't exist       
        self.add_node(edge0)
        self.add_node(edge1)
        
        # add the edge
        self.adjacency[edge0][edge1] += weight            # thanks to the default dictionaries we just need
        self.distant_neighbours[edge1][edge0] += weight   # to add the weight to the existing one
        self.edges_list[(edge0, edge1)] += weight         # (where the default is 0 if the edge does not exist)
    
    def edit_weight(self, edge, new_weight):
        '''
        replace the weight of the input edge
        '''
        edge0 = edge[0]
        edge1 = edge[1]
        
        self.adjacency[edge0][edge1] = new_weight
        self.distant_neighbours[edge1][edge0] = new_weight
        self.edges_list[(edge0, edge1)] = new_weight
        
    def union(self, another_weightedDiGraph):
        '''
        joins the graph with an input weighted graph
        (if an edge exists it adds the weight, otherwise it adds the edges)
        '''
        for edge, weight in another_weightedDiGraph.edges:
            self.add_edge(*edge, weight)
    
    def undirected_edges(self):
        '''
        returns the set of all the edges (considering (a,b) and (b,a) as the same edge)
        '''
        undirected_edges_list = set()
        for a,b in self.edges_list.keys():
            if (b,a) in self.edges_list.keys():
                if not ((a,b) in undirected_edges_list or (b,a) in undirected_edges_list):
                    undirected_edges_list.add((a,b))
        
        return(undirected_edges_list)
    
    @property
    def n_weighted_edges(self):
        '''
        number of edges according to weight
        '''
        total_weight = sum(self.edges_list.values())
        return total_weight
    
    @property
    def get_average_weighted_links(self):
        '''
        average number of links per node
        '''
        return self.n_weighted_edges / self.n_nodes
    
    def to_nx(self):
        '''
        converts a weightedDiGraph() to a networkX weighted digraph
        '''
        DG = nx.DiGraph()
        DG.add_nodes_from(self.adjacency.keys())
        DG.add_weighted_edges_from([(*edge,weight) for edge, weight in self.edges_list.items()])
        return(DG)
    

class MultiDiGraph():
    '''
    basic multidigraph class
    '''
    
    def __init__(self):
        """
        internally we just save it as a list of edges since this is the way in which our datasets are coded
        (and we don't need to actually use this class other than for the generation of the digraph)
        """
        
        self.edges_list = []
    
    
    def __repr__(self):
        """"""
        return 'DiMultiGraph()'
    
    @property
    def graph_type(self):
        return 'Directed MultiGraph'
    
    @property
    def edges(self):
        '''
        returns the list of edges
        '''
        return self.edges_list
    
    @property
    def n_edges(self):
        '''
        number of edges
        '''
        return len(self.edges_list)
    
    def add_edges_from(self, iterable):
        '''
        add edges from an iterable
        '''
        for edge in iterable:
            self.add_edge(*edge)
    
    def add_edge(self, edge0, edge1, timestamp):
        '''
        adds an edge in the graph
        '''
        self.edges_list.append((edge0, edge1, timestamp))
    
    def get_graph(self, filename, start_time = 0, end_time = 3000000000):
        """
        builds the multidigraph from an input file
        (selecting just the edges in the input interval, by default all edges are taken)
        
        the format for every row is user1 user2 timestamp
        where the rows are ordered by timestamp
        """
        
        with open(filename, 'r') as file:
            for row in tqdm(file, total = dataset_length[filename]):
                # process every row
                row = row.strip().split()
                
                # convert each element to integers
                start_node, end_node, timestamp = list(map(int, row))
                
                # only consider edges within a certain time interval
                if timestamp >= start_time and timestamp <= end_time:
                    self.add_edge(start_node, end_node, timestamp)
    
    
    def get_DiGraph_from_interval(self, start_time = None, end_time = None, weight = 1):
        '''
        builds a digraph that contains only the selected time interval
        (one should be sure to select an interval that is actually contained in the current graph)
        '''
        
        # if no interval is passed we take the whole graph
        if start_time == None:
            start_time = self.edges[0][2]
        if end_time == None:
            end_time = self.edges[-1][2]
        
        
        final_graph = weightedDiGraph()
        start_edge = search_timestamp(self.edges, start_time, mode = 'start')  # returns the index of the smallest edge
                                                                               # with a timestamp bigger or equal than start_time
        
        end_edge = search_timestamp(self.edges, end_time, mode = 'end')        # returns the index of the smallest edge
                                                                               # with a timestamp smaller than end_time
        
        # build the final graph
        for edge in self.edges[start_edge : end_edge]:
            final_graph.add_edge(*(edge[0:2]), weight)
        
        return(final_graph)
    


def search_timestamp(edges_list, time, mode):
    '''
    finds the timed edge in edges_list that is closest (based on the mode)
    to the input time
    
    if mode == 'start' the function returns the index of the smallest edge
    with a timestamp bigger or equal than the input time
    
    if mode == 'end' the function returns the index of the smallest edge
    with a timestamp smaller than the input time
    '''
    
    # basic checks on the input time
    if mode == 'start':
        if edges_list[0][2] >= time:
            return(0)
        elif edges_list[-1][2] < time:
            raise ValueError('Start time too high in search_timestamp()')
    elif mode == 'end':
        if edges_list[0][2] > time:
            raise ValueError('End time too low in search_timestamp()')
        elif edges_list[-1][2] <= time:
            return(len(edges_list))
    else:
        raise ValueError('Third argument in search_timestamp() must be either "start" or "end"')
    
    
    start_idx = 0
    end_idx = len(edges_list)
    
    # since the input edge list is ordered by timestamp
    # we implemented a binary search to find the index that we want
    
    # we keep two indices and progressively reduce the span of the search
    while (end_idx - start_idx) > 1:
        
        # take the middle of the current array
        middle_idx = (end_idx + start_idx)//2
        
        # select the interesting half of the current array
        if edges_list[middle_idx][2] < time:
            start_idx = middle_idx
            
        elif edges_list[middle_idx][2] > time:
            end_idx = middle_idx
            
        else: # if we found exactly the timestamp that we want we stop the search
            
            if mode == 'start':
                while (edges_list[middle_idx][2] == time):
                    middle_idx -= 1
                start_idx = middle_idx + 1
                return(start_idx)
            else:
                while (edges_list[middle_idx][2] == time):
                    middle_idx += 1
                start_idx = middle_idx
                return(start_idx)
    
    start_idx+=1
    
    return(start_idx)


def get_dataset_graphs(start_time = 0, end_time = 3000000000):
    # the datasets paths
    global filename_a2q
    global filename_c2a
    global filename_c2q
    
    # retrieve the first dataset from the disk
    print('retrieve the first dataset from the disk')
    dataset_graph_a2q = MultiDiGraph()
    dataset_graph_a2q.get_graph(filename_a2q, start_time, end_time)
    
    # retrieve the second dataset from the disk
    print('retrieve the second dataset from the disk')
    dataset_graph_c2a = MultiDiGraph()
    dataset_graph_c2a.get_graph(filename_c2a, start_time, end_time)
    
    # retrieve the third dataset from the disk
    print('retrieve the third dataset from the disk')
    dataset_graph_c2q = MultiDiGraph()
    dataset_graph_c2q.get_graph(filename_c2q, start_time, end_time)
    
    return dataset_graph_a2q, dataset_graph_c2a, dataset_graph_c2q


def get_total_graph(datasets, start_time = 0, end_time = 3000000000):  # time limits of our datasets
    '''
    This function takes in input an interval of time and returns the
    "total" weighted digraph associated with this interval of time
    '''
    
    # the weight parameters
    global a2q_weight
    global c2a_weight
    global c2q_weight
    
    dataset_graph_a2q, dataset_graph_c2a, dataset_graph_c2q = datasets
    
    # select the time interval in the first dataset and build the weighted graph
    total_graph = dataset_graph_a2q.get_DiGraph_from_interval(start_time = start_time, end_time = end_time, weight = a2q_weight)
    
    # join the second and the first dataset
    total_graph.union(dataset_graph_c2a.get_DiGraph_from_interval(start_time = start_time, end_time = end_time, weight = c2a_weight))
    
    # join the third and the other two
    total_graph.union(dataset_graph_c2q.get_DiGraph_from_interval(start_time = start_time, end_time = end_time, weight = c2q_weight))
    
    # we invert all the weight to reverse the priorities
    for edge, weight in total_graph.edges_list.items():
        new_weight = 1/weight
        total_graph.edit_weight(edge, new_weight)
    
    return(total_graph)







class Queue():
    '''
    priority queue
    '''
    
    def __init__(self, maxsize = 10**4):
        """
        internally we save it as a simple list
        and we keep track of the indices of the start and the end of the queue
        """
        
        self.size = maxsize + 1              # we keep the last element free to better manage the indices
        self.basic_list = [0] * self.size    # this list will contain the actual queue
        self.front = 0                       # the index of the front of the queue
        self.back = 0                        # the index of the first free index of the queue
    
    def empty(self):
        '''
        check if the queue is empty
        '''
        return( self.front == self.back )
    
    @property
    def occupied_space(self):
        '''
        returns the occupied space in the queue
        '''
        return((self.back - self.front)%self.size)
    
    @property
    def remaining_space(self):
        '''
        returns the space remained in the queue
        '''
        return(self.size - 1 - self.occupied_space)
    
    def resize(self):
        '''
        resizes the queue
        '''
        # we double the size of the queue
        maxsize = (self.size - 1) * 2
        newsize = maxsize + 1
        newlist = [0] * newsize
        occupied_space = self.occupied_space
        for i in range(occupied_space):
            newlist[i] = self.get()
        
        self.size = newsize
        self.basic_list = newlist
        self.front = 0
        self.back = occupied_space
        
        return
    
    def put(self, element):
        '''
        insert element at the end of the queue
        '''
        if self.remaining_space > 0:
            self.basic_list[self.back] = element
            self.back = (self.back + 1) % self.size
        else:
            self.resize()
            self.put(element)
    
    def pick(self):
        '''
        return the first element of the queue
        '''
        if not self.empty():
            return(self.basic_list[self.front])
        else:
            raise RuntimeError('Trying to get the first element of an empty queue')
    
    def get(self):
        '''
        return the first element of the queue and removes it
        '''
        if not self.empty():
            front_element = self.basic_list[self.front]
            self.front = (self.front + 1) % self.size
            return(front_element)
        else:
            raise RuntimeError('Trying to dequeue from an empty queue')

            
def get_timestamp(date_string):
    """Convert date string to timestamp"""
    date_format = "%Y-%m-%d"
    # Get datetime object from string
    dt = datetime.strptime(date_string, date_format).timetuple()
    # Return date as timestamp
    return time.mktime(dt)

def dijkstra(graph, start_node, target_node=None):
    """
    Compute shortest path from a source node to all the other nodes in the
    graph with the Dijkstra algorithm. Some shortest path could be None 
    If the graph is a diGraph or its a disconnected Graph.

    Args:
        graph (DiGraph): Graph object.
        start_node (int): Name of the source node.
        target_node (int, optional): Name of the destination node name. 
            Defaults to None.

    Returns:
        dict: Returns a dict with the paths to all the nodes in the graph 
        and their weights. If target_node is provided, returns only the
        path from the starting node to the target node.

    """
    
    # Get a copy of the graph and initialize the variables
    graph_ = copy.deepcopy(graph)
    visited = {}
    not_visited = dict.fromkeys(graph_.nodes, math.inf)
    not_visited[start_node] = 0
    paths = dict.fromkeys(graph_.nodes, [])
    
    # While there are elements in not_visited
    while not_visited:
        # Choose next_node as node with min weight among all the not visited nodes
        next_node = min(not_visited, key=not_visited.get)
        # Move the node from not_visited to visited
        visited[next_node] = not_visited.pop(next_node)
        # Get the neighbors of the selected node
        neighbors = graph_.adjacency[next_node]
        
        # For each node in neighbors...
        for node, weight in neighbors.items():
            if node in visited.keys():
                # ...update visited if the new path has a smaller weight
                if  visited[next_node] + weight < visited[node]:
                    visited[node] = visited[next_node] + weight
                    # Update also the path
                    paths[node] = paths[next_node] + [(next_node, node, weight)]
            else:
                # ...or update not visited
                if node in not_visited.keys():
                    if visited[next_node] + weight < not_visited[node]:
                        not_visited[node] = visited[next_node] + weight
                        # Update also the path
                        paths[node] = paths[next_node] + [(next_node, node, weight)]
     
    if target_node is None:
        # Return the shortest path and its weight from start_node to all the other nodes
        return dict(path=paths, weigth=visited)
    
    # Return the shortest path and its weight from start_node to target_node
    return dict(path=paths[target_node], weigth=visited[target_node])