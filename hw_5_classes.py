import pandas as pd
from collections import defaultdict

#SET PATH TO THE TXT FILE
data = pd.read_csv(r'sx-stackoverflow-c2q.txt', sep=' ')
df = data.head(1000).copy()

df.columns = ['start', 'to', 'ts']
df['ts2'] = pd.to_datetime(df['ts'], unit='s')


class Graph:
    def __init__(self):
        """"""
        self.nodes = set()
        self.edges = list()
        self.adjacency = defaultdict(list)
        
    def __repr__(self):
        """"""
        return 'Graph()'

    def __getitem__(self, key):
        return self.adjacency[key]

    def add_node(self, node):
        self.adjacency[node]
        self.nodes.add(node)

    
    def add_edge(self, edge0, edge):
        self.adjacency[edge0].append(edge)
        self.edges.append((edge0, edge,))
        
        
    def compute_graph(self, df):
        """"""
        for row in df.itertuples():
            self.add_node(row.start)
            self.add_edge(row.start, row.to)
                     
        self.n_nodes = len(self.nodes)
        self.n_edgess = len(self.edges)
        
    @property
    def get_average_links(self):
        return sum(map(len, self.adjacency.values())) / len(self.adjacency)
        
    
 
    
test = Graph()
test.compute_graph(df)
test[5]
test.get_average_links