import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
import copy
import networkx as nx
from IPython.core.display import display, HTML
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from utilities import *


'''
library that contains the function for functionality and visualization 1
'''

def functionality_1(dataset_graph):
    '''
    functionality 1:
    Given in input one of the 3 dataset multidigraphs
    we give in output:
        The dataset DiGraph
        Whether the graph is directed or not
        Number of users
        Number of answers/comments
        Average number of answers/comments per user
        Average number of links per user
        Density degree of the graph
        Whether the graph is sparse or dense
    '''
    global standard_interval
    
    graph = dataset_graph.get_DiGraph_from_interval(*standard_interval)
    
    graph_type = graph.graph_type
    
    n_users = graph.n_nodes
    
    n_answers_comments = graph.n_weighted_edges
    
    average_answers_comments = graph.get_average_weighted_links
    
    average_links = graph.get_average_links
    
    density = graph.get_density
    
    is_dense = graph.isDense
    
    return graph, graph_type, n_users, n_answers_comments, average_answers_comments, average_links, density, is_dense


def plot_1(graph, graph_type, n_users, n_answers_comments, average_answers_comments, average_links, density, is_dense):
    '''
    visualization for functionality 1
    '''
    
    display(HTML(f"<table class='table table-striped'> <tbody> <tr> <th scope='row'>Graph Type</th> <td>{graph_type}</td></tr> <tr> <th scope='row'>Number of users</th> <td>{n_users}</td> </tr> <tr> <th scope='row'>Number of answers/comments</th> <td>{n_answers_comments}</td> </tr>  <tr> <th scope='row'>Average number of answers/comments per user</th> <td>{'{:.2f}'.format(average_answers_comments)}</td> </tr> <tr> <th scope='row'>Average number of links per user</th> <td>{'{:.2f}'.format(average_links)}</td> </tr> <tr> <th scope='row'>Density degree of the graph</th> <td>{'{:.2e}'.format(density)}</td> </tr> <tr> <th scope='row'>Density/Sparsity</th> <td>The graph is {'Dense' if is_dense else 'Sparse'}</td> </tr> </tbody> </table>"))
    
    # convert to networkx graph for visualization
    H = graph.to_nx()
    
    # compute the node centrality for the density distribution
    normalization = H.number_of_nodes() - 1
    node_centrality = [H.degree(node)/normalization for node in H.nodes()]
    
    # renormalize the centrality between max and min
    max_centr = max(node_centrality)
    min_centr = min(node_centrality)
    node_centrality = list(map(lambda x: (x - min_centr)/(max_centr - min_centr), node_centrality))
    
    # set the colormap
    cmap = cm.get_cmap('viridis')
    
    # retrieve the node colors
    node_colors = list(map(lambda x: cmap(x), node_centrality))
    
    fig = plt.figure(figsize = (18,10))

    # add grid specifications for the color legend
    gs = fig.add_gridspec(1, 18)
    
    # the plot of the graph
    ax = fig.add_subplot(gs[:,0:17])
    
    ax.set_title('Density distribution')

    pos = nx.spring_layout(H)

    nx.draw_networkx_nodes(H, pos = pos, node_color = node_colors, ax = ax)
    nx.draw_networkx_edges(H, pos = pos, ax = ax, edge_color = 'gray')
    
    # the plot of the color legend
    ax1 = fig.add_subplot(gs[:,17])
    
    ax1.set_title('Centrality')
    
    # color gradient to plot the legend
    gradient = np.flip(np.linspace(0, 1, 256))
    gradient = np.transpose(np.vstack((gradient, gradient)))
    ax1.imshow(gradient, aspect='auto', cmap=cmap)
    
    # removing xticks from the legend
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # correctly setting the yticks in the legend
    ax1.yaxis.tick_right()
    ax1.set_yticks([0,256])
    ax1.set_yticklabels(['{:.2g}'.format(max_centr), '{:.2g}'.format(min_centr)])
    
    plt.show()
    
    return