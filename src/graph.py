import time
import os
import pickle
import itertools
import numpy as np
import torch
import torch.nn.functional as F

from _config import DATA_DIR

import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.nn import WLConv, GCNConv
from torch_geometric.nn.models import GAE


########################  Create graphs  ########################
def create_edges_graphs_sparse_relative_distance(X, ratio_diff_to_neighbor=0.05, max_degree=25):
    """
    Construct an UNDIRECTED graph: each node to its nearest neighbors within a certain distance
        distance = abs(node_i - node_j) < (95th - 5th percentile)*ratio_diff_to_neighbor then connect node_i with node_j

    Arguments
    - X (np.array): data matrix
    - ratio_diff_to_neighbor (float): percentage of the difference between the 95th and 5th percentiles of
        the values of the feature.
    - max_degree (int): maximum degree of a node

    Returns
    - list_of_edges (list of np.arrays): each np.array is a matrix of size 2 x num_edges, where each column is an edge
    """
    list_of_edges = []
    for feature_id in range(X.shape[1]):
        diff = ratio_diff_to_neighbor * (np.percentile(X[:, feature_id], 95, interpolation='midpoint') - np.percentile(X[:, feature_id], 5, interpolation='midpoint'))

        if diff == 0: # some features from 'lung' and other datasets have almost the same value, and the difference is 0
            diff = ratio_diff_to_neighbor * (X[:, feature_id].max() - X[:, feature_id].min()) + 1e-5 # add 1e-5 because some features have the same value and diff is 0

        # sort the nodes by their feature value
        nodes_and_values = list(zip(range(X.shape[0]), X[:, feature_id].tolist()))
        nodes_and_values.sort(key=lambda x: x[1])

        edges = []
        for node_id, value in nodes_and_values:
            # iterate left, right to all neighbors within diff
            left = node_id - 1
            right = node_id + 1

            for _ in range(max_degree):
                left_diff = abs(value - nodes_and_values[left][1]) if left >= 0 else float('inf')
                right_diff = abs(value - nodes_and_values[right][1]) if right < len(nodes_and_values) else float('inf')
                
                if left_diff <= right_diff and left_diff < diff:
                    edges.append([node_id, nodes_and_values[left][0]])
                    edges.append([nodes_and_values[left][0], node_id])

                    left -= 1
                elif right_diff <= left_diff and right_diff < diff:
                    edges.append([node_id, nodes_and_values[right][0]])
                    edges.append([nodes_and_values[right][0], node_id])

                    right += 1
                else:
                    break
            
        list_of_edges.append(np.array(edges).T)

    return list_of_edges


def create_edges_graphs_knn(X, k=5):
    """
    Construct a DIRECTED graph: each node to its k nearest neighbors

    Arguments
    - X (np.array): data matrix
    - k (int): number of nearest neighbors to connect to

    Returns
    - list_of_edges (list of np.arrays): each np.array is a matrix of size 2 x num_edges, where each column is an edge
    """
    print("Creating edges for kNN graphs...")

    assert k > 0 and k < X.shape[0]

    list_of_edges = []
    for feature_id in range(X.shape[1]):
        # sort the nodes by their feature value
        nodes_and_values = list(zip(range(X.shape[0]), X[:, feature_id].tolist()))
        nodes_and_values.sort(key=lambda x: x[1])

        edges = []
        for i, (node_id, value) in enumerate(nodes_and_values): # for every node
            left = i - 1
            right = i + 1

            for _ in range(k): # connect to k nearest neighbors
                left_diff = abs(value - nodes_and_values[left][1]) if left >= 0 else float('inf')
                right_diff = abs(value - nodes_and_values[right][1]) if right < len(nodes_and_values) else float('inf')
                
                if left_diff <= right_diff:
                    edges.append([node_id, nodes_and_values[left][0]])
                    left -= 1
                else:
                    edges.append([node_id, nodes_and_values[right][0]])
                    right += 1 
            
        list_of_edges.append(np.array(edges).T)

    return list_of_edges


def load_random_graph(dataset_name, repeat_id):
    """
    Load a random graph that was previously created and saved. 
    The number of nodes is based on the setup from WPFS paper (five fold cross validation, valid_percentage=0.1)
    ---> If one changes the size of the training data, these graphs won't be valid anymore (the number of nodes needs to be changed)

    Arguments
    - dataset_name (str): name of the dataset
    - repeat_id (int): the id of the generated graphs (because for each datasets there are multiple generated graphs)

    Returns
    - list_of_edges (list of np.arrays): each np.array is a matrix of size 2 x num_edges, where each column is an edge
    """
    print("Loading random graph for dataset {} and repeat {}".format(dataset_name, repeat_id))

    if dataset_name == 'cll':
        file_name = 'cll'
    elif dataset_name == 'lung':
        file_name = 'lung'
    elif dataset_name.startswith('metabric'):
        file_name = 'metabric'
    elif dataset_name == 'prostate':
        file_name = 'prostate'
    elif dataset_name == 'smk':
        file_name = 'smk'
    elif dataset_name.startswith('tcga'):
        file_name = 'tcga'
    elif dataset_name == 'toxicity':
        file_name = 'toxicity'

    # load the graph from pickle
    with open(os.path.join(DATA_DIR, 'edges_random_graphs_of_samples', f"{file_name}_repeat{repeat_id}.pkl"), 'rb') as f:
        return pickle.load(f)


def create_edges_random_graphs(num_nodes, proportion_edges, seed_random_edges, all_edges=None):
    """
    Construct an UNDIRECTED graph:
    It should be called once for each gene, with a different seed_random_edges to obtain different graphs for each gene.

    Arguments
    :param num_nodes (int): number of nodes in the graph
    :param proportion_edges (float): proportion of edges to select from all possible edges of a fully connected graph
    :param seed_random_edges (int): seed for sampling random edges
    :param all_edges (np.array of num_edges x 2): all possible edges of a fully connected graph

    Returns
    :edges (np.array of size 2 x num_edges): each column is an edge
    """
    assert num_nodes <= 300, "The function is too slow for large graphs."
    assert proportion_edges <= 1, "proportion_of_max_edges must be <= 1"

    if all_edges is None: # create all possible edges of a fully connected graph
        all_edges = np.array(list(itertools.combinations(range(num_nodes), 2)))    

    if proportion_edges < 0:
        proportion_edges = 0

    # number of edges to select from all possible edges of a fully connceted graph
    num_edges = int(proportion_edges * num_nodes * (num_nodes - 1) / 2)

    # create all possible edges
    np.random.RandomState(seed_random_edges).shuffle(all_edges)
    
    # sample num_edges from all possible edges of a fully connceted graph
    return all_edges[:num_edges].T


class GeneGraph(tg.data.Dataset):
    def __init__(self, root=None, data=None, transform=None):
        """
        :param data (np.array): list of tg.data.Data instances, each holding a graph
        """
        super().__init__(root=root, transform=transform)

        assert type(data)==list
        self.data = data

    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]

# ----------------------  Weisfeiler-Lehman  --------------------

class WLencoder(torch.nn.Module):
    # Weisfeiler-Lehman graph embeddings
    def __init__(self):
        super().__init__()

        self.wl1 = WLConv()

    def forward(self, x, edge_index):
        x = self.wl1(x, edge_index)

        return x

# create the gene graphs
def compute_WL_graph_embeddings(X, embeddings_size, ratio_diff_to_neighbor=0.05):
    """
    Arguments:
    :param X: the data matrix
    :param embeddings_size: the size of the embeddings
    :param ratio_diff_to_neighbor: the ratio of the difference between the 95th and 5th percentiles of
        the values of the feature to consider as a neighbor
    
    Returns:
    :embeddings (tensor embeddings_size x X.shape[1]): the embeddings for each gene
    :wl_colors: (tensor X.shape[0] x X.shape[1]): all colors returned by the WL embedding
    """
    model = WLencoder().cuda()

    graph_embeddings = torch.zeros((embeddings_size, X.shape[1]))
    wl_colors = torch.zeros((X.shape[0], X.shape[1]))
    now = time.time()
    print("Starting time to create gene graph...")
    edges = create_edges_graphs_sparse_relative_distance(X, ratio_diff_to_neighbor=ratio_diff_to_neighbor)
    for i in range(X.shape[1]):
        # create the graph of a gene
        data = tg.data.Data(x=torch.ones(X.shape[0], dtype=torch.long), edge_index=torch.tensor(edges[i]))

        final_node_colors = model.forward(data.x, data.edge_index)
        counts, _ = torch.histogram(final_node_colors.float(), bins=embeddings_size, density=True)

        graph_embeddings[:, i] = counts
        wl_colors[:, i] = final_node_colors
    
    print("Ending time to create gene graph {}".format(time.time() - now))

    return graph_embeddings, wl_colors


# ----------------------  Graph autoencoder  -------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=20, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def forward_and_global_pooling(self, batch):
        node_embeddings = self.forward(batch.x, batch.edge_index)

        # create the indexes of each nodes in the graph (because the global_mean_pooling will pool the nodes based on their indexes)
        batch_index = torch.arange(batch.num_graphs, dtype=torch.long, device=batch.x.device)

        # compute number of graphs
        nodes_per_graph = int(batch.x.shape[0] / batch.num_graphs)
        batch_index = batch_index.repeat_interleave(nodes_per_graph)

        return tg.nn.global_mean_pool(x=node_embeddings, batch=batch_index).T # (embeddings_size, num_graphs)


# ----------------------  Helper functions  -------------------

def create_gcondnet_graph_dataset(X_standardized, list_of_edges):
    """
    :param X_standardized: the standardised data matrix (e.g., 100 x 5000) used for creating the node features
    :param list_of_edges: list of the edges np.ndarrays (of size 2 x num_edges) of the graphs (created outside of this function)
    returns:
    - the dataset of all_graphs
    - the dataset of n_sample_graphs graphs
    """
    num_samples = X_standardized.shape[0]
    num_features = X_standardized.shape[1]

    assert num_features == len(list_of_edges), "The number of features should be equal to the number of sets of edges"

    now = time.time()
    print("Starting time to create feature graphs...")

    #### Create dataset
    feature_graphs = []
    for feature_id in range(num_features):
        node_features = torch.matmul(
                            torch.eye(num_samples),
                            torch.diag(torch.tensor(X_standardized[:, feature_id])))
    
        feature_graphs.append(tg.data.Data(
            x = node_features,
            edge_index = torch.tensor(list_of_edges[feature_id], dtype=torch.long),
        ))

    print("Ending time to create feature graph {}".format(time.time() - now))

    return GeneGraph(root=None, data=feature_graphs, transform=T.ToDevice('cuda'))


##### Compute graph embeddings
def compute_graph_embeddings(gnn, batch):
    """
    :param batch (torch_geometric.data.Batch). It holds multiple gene graphs as one large graph with disconnected components
    """
    # if the gnn has .encoder
    if hasattr(gnn, 'encoder'): # for GAE
        return gnn.encoder.forward_and_global_pooling(batch)
    else: # for vanilla GNN (without being wrapped in a GAE)
        return gnn.forward_and_global_pooling(batch)
