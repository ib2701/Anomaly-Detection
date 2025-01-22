import networkx as nx
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.io import loadmat




def ex1():
    # 1. Load the dataset and build the graph
    G = nx.Graph()
    with open('ca-AstroPh.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1500:
                break
            node1, node2 = map(int, line.split())
            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += 1
            else:
                G.add_edge(node1, node2, weight=1)

    # 2. Extract features from ca-AstroPh.txt
    features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)
        Ni = len(egonet) - 1
        Ei = egonet.size(weight=None)
        Wi = egonet.size(weight='weight')
        A = nx.to_numpy_array(egonet, weight='weight')
        eigenvalues, _ = eigh(A)
        lambda_wi = max(eigenvalues)
        features[node] = {'Ni': Ni, 'Ei': Ei, 'Wi': Wi, 'lambda_wi': lambda_wi}

    nx.set_node_attributes(G, features)

    # 3. Compute anomaly score
    X = np.array([[np.log(features[node]['Ei']), np.log(features[node]['Ni'])] for node in G.nodes()])
    y = np.array([features[node]['Wi'] for node in G.nodes()])
    reg = LinearRegression().fit(X, y)
    Cx_theta = reg.predict(X)
    scores = {node: (max(y[i], Cx_theta[i]) / min(y[i], Cx_theta[i])) * np.log(abs(y[i] - Cx_theta[i]) + 1) for i, node in enumerate(G.nodes())}

    # 4. Sort and draw graph
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    top_10_nodes = sorted_nodes[:10]
    colors = ['red' if node in top_10_nodes else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=colors, with_labels=True)
    plt.show()

    # 5. Modify anomaly score and draw
    lof = LocalOutlierFactor(n_neighbors=20)
    lof_scores = -lof.fit_predict(X)
    normalized_scores = {
        node: (scores[node] - min(scores.values())) / (max(scores.values()) - min(scores.values()))
        for node in G.nodes()}
    final_scores = {node: normalized_scores[node] + lof_scores[i] for i, node in
                    enumerate(G.nodes())}
    sorted_final_nodes = sorted(final_scores, key=final_scores.get, reverse=True)
    top_10_final_nodes = sorted_final_nodes[:10]
    final_colors = ['green' if node in top_10_final_nodes else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=final_colors, with_labels=True)
    plt.show()


def ex2():
    # 1. Generate a regular graph with 100 nodes, each with degree 3
    G1 = nx.random_regular_graph(3, 100)
    G2 = nx.connected_caveman_graph(10, 20)
    G = nx.union(G1, G2, rename=('G1-', 'G2-'))

    # Add random edges
    for _ in range(50):
        u = np.random.choice(G.nodes())
        v = np.random.choice(G.nodes())
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    nx.draw(G, with_labels=True)
    plt.show()

    # Extract features from the graph
    features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)
        Ni = len(egonet) - 1
        Ei = egonet.size(weight=None)
        Wi = egonet.size(weight='weight')
        A = nx.to_numpy_array(egonet, weight='weight')
        eigenvalues, _ = eigh(A)
        lambda_wi = max(eigenvalues)
        features[node] = {'Ni': Ni, 'Ei': Ei, 'Wi': Wi, 'lambda_wi': lambda_wi}

    nx.set_node_attributes(G, features)

    # Compute anomaly scores
    X = np.array(
        [[np.log(features[node]['Ei']), np.log(features[node]['Ni'])] for node in G.nodes()])
    y = np.array([features[node]['Wi'] for node in G.nodes()])
    reg = LinearRegression().fit(X, y)
    Cx_theta = reg.predict(X)
    scores = {node: (max(y[i], Cx_theta[i]) / min(y[i], Cx_theta[i])) * np.log(
        abs(y[i] - Cx_theta[i]) + 1) for i, node in enumerate(G.nodes())}

    # Sort and draw the graph
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    top_10_nodes = sorted_nodes[:10]
    colors = ['red' if node in top_10_nodes else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=colors, with_labels=True)
    plt.show()

    # 2. Generate HeavyVicinity anomalies
    G3 = nx.random_regular_graph(3, 100)
    G4 = nx.random_regular_graph(5, 100)
    G = nx.union(G3, G4, rename=('G3-', 'G4-'))

    # Assign weight 1 edges
    for edge in G.edges():
        G.add_edge(edge[0], edge[1], weight=1)

    # Pick 2 random nodes and add 10 to the weights of all edges in their egonets
    nodes = np.random.choice(G.nodes(), 2, replace=False)
    for node in nodes:
        egonet = nx.ego_graph(G, node)
        for edge in egonet.edges():
            G[edge[0]][edge[1]]['weight'] += 10

    # Extract features from the graph
    features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)
        Ni = len(egonet) - 1
        Ei = egonet.size(weight=None)
        Wi = egonet.size(weight='weight')
        A = nx.to_numpy_array(egonet, weight='weight')
        eigenvalues, _ = eigh(A)
        lambda_wi = max(eigenvalues)
        features[node] = {'Ni': Ni, 'Ei': Ei, 'Wi': Wi, 'lambda_wi': lambda_wi}

    nx.set_node_attributes(G, features)

    # Compute anomaly scores
    X = np.array(
        [[np.log(features[node]['Ei']), np.log(features[node]['Ni'])] for node in G.nodes()])
    y = np.array([features[node]['Wi'] for node in G.nodes()])
    reg = LinearRegression().fit(X, y)
    Cx_theta = reg.predict(X)
    scores = {node: (max(y[i], Cx_theta[i]) / min(y[i], Cx_theta[i])) * np.log(
        abs(y[i] - Cx_theta[i]) + 1) for i, node in enumerate(G.nodes())}

    # Sort and draw the graph with top 4 nodes
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    top_4_nodes = sorted_nodes[:4]
    colors = ['green' if node in top_4_nodes else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=colors, with_labels=True)
    plt.show()



def ex3():
    # Load the ACM dataset
    data = loadmat('ACM.mat')
    attributes = torch.tensor(data['Attributes'], dtype=torch.float)
    adj = data['Network']
    labels = torch.tensor(data['Label'], dtype=torch.long).squeeze()

    edge_index, _ = from_scipy_sparse_matrix(adj)
    adj = torch.tensor(adj.todense(), dtype=torch.float)

    class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(Encoder, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            return x

    class AttributeDecoder(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(AttributeDecoder, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, z, edge_index):
            z = self.conv1(z, edge_index)
            z = torch.relu(z)
            z = self.conv2(z, edge_index)
            return z

    class StructureDecoder(nn.Module):
        def __init__(self, in_channels):
            super(StructureDecoder, self).__init__()
            self.conv = GCNConv(in_channels, in_channels)

        def forward(self, z, edge_index):
            z = self.conv(z, edge_index)
            z = torch.relu(z)
            adj_reconstructed = torch.matmul(z, z.t())
            return adj_reconstructed

    class GraphAutoencoder(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GraphAutoencoder, self).__init__()
            self.encoder = Encoder(in_channels, hidden_channels, out_channels)
            self.attr_decoder = AttributeDecoder(out_channels, hidden_channels, in_channels)
            self.struct_decoder = StructureDecoder(out_channels)

        def forward(self, x, edge_index):
            z = self.encoder(x, edge_index)
            x_reconstructed = self.attr_decoder(z, edge_index)
            adj_reconstructed = self.struct_decoder(z, edge_index)
            return x_reconstructed, adj_reconstructed

    # Initialize model
    model = GraphAutoencoder(in_channels=attributes.shape[1], hidden_channels=128, out_channels=64)

#ex1()
ex2()
