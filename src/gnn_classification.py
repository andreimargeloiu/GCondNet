import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F

def create_knn_patient_edges(X, k=5, distance='cosine'):
	"""
	Connect each patient to its k nearest neighbours

	- X: data N x D
	- k: number of nearest neighbours
	- distance: distance metric (cosine, euclidean)
	"""
	num_patients = X.shape[0]
	edges = []

	for i in range(num_patients):
		distances = []

		for j in range(num_patients):
			if i!=j:
				if distance == 'cosine':
					distances.append((j, 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))))
				elif distance == 'euclidean':
					distances.append((j, np.linalg.norm(X[i] - X[j])))
				else:
					raise ValueError('distance metric not recognised')

		distances = sorted(distances, key=lambda x: x[1])
		for j in range(k): # take the smalles distances
			edges.append((distances[j][0], i)) # incoming edge to node i

	return np.array(edges).T


class GCN_Classifier(torch.nn.Module):
	def __init__(self, in_channels, out_channels, num_classes, dropout_rate=0.5):
		super().__init__()
		self.dropout_rate = dropout_rate

		self.conv1 = GCNConv(in_channels, 2*out_channels)
		self.conv2 = GCNConv(2*out_channels, out_channels)

		self.linear = torch.nn.Linear(out_channels, num_classes)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.conv2(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.linear(x)

		return x


class GATv2_Classifier(torch.nn.Module):
	def __init__(self, in_channels, out_channels, num_classes, dropout_rate=0.5, heads=4):
		super().__init__()
		self.dropout_rate = dropout_rate

		self.conv1 = GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout_rate)
		self.conv2 = GATv2Conv(out_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)

		self.linear = torch.nn.Linear(out_channels, num_classes)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.conv2(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.linear(x)

		return x