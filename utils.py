import os
import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
import scipy.sparse as sp
import random
import scanpy as sc
from operator import itemgetter


def graph_generations(features, threshold, feature_preprocess, labels):
    # take cell*protein matrix as input
    # output cell*cell graph
    features = features.astype(float)
    features_pd = pd.DataFrame(features.T)
    adj = features_pd.corr()
    adj_matrix = np.where(adj>threshold,1,0)

    if feature_preprocess:
        features = sp.coo_matrix(features)
        features, _ = preprocess_features(features)

    adj_matrix_sp = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((adj_matrix_sp.row, adj_matrix_sp.col)), dtype=torch.long)
    features = torch.tensor(features,dtype=torch.float32)

    data = Data(x=features, edge_index=edge_index, y=labels)
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)




def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized, sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

