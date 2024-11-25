# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:37:24 2024

@author: Ana
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import pickle
import pandas as pd

def create_adjacency_matrix(undirected_edges,directed_edges):
    adj_matrix = np.zeros((54, 54), dtype=int)
    for edge in undirected_edges:
        node1, node2 = edge
        adj_matrix[node1-1, node2-1] = 1  # Adjust for 0-indexing
        adj_matrix[node2-1, node1-1] = 1  # Undirected means symmetry
    for edge in directed_edges:
        node1, node2 = edge
        adj_matrix[node1-1, node2-1] = 1  # Directed, so only one direction
    
    return torch.tensor(adj_matrix, dtype=torch.float32)

undirected_edges = [
    (1, 2), (1, 4), (1, 7), (2, 3), (2, 4), (2, 5), (3, 5), (3, 21),
    (4, 5), (4, 7), (4, 9), (5, 6), (5, 9), (5, 21), (5, 22), (6, 8),
    (6, 9), (6, 10), (6, 11), (6, 17), (6, 18), (6, 22), (7, 8), (7, 9),
    (8, 9), (8, 11), (10, 11), (10, 15), (10, 18), (11, 12), (11, 15),
    (12, 13), (12, 15), (13, 14), (13, 15), (14, 15), (14, 16), (14, 17), (14, 18), (15, 18),
    (16, 17), (16, 19), (17, 18), (17, 19), (17, 20), (17, 22), (19, 20),
    (20, 21), (20, 22), (21, 22),
    (23, 24), (23, 26), (23, 29), (24, 25), (24, 26), (24, 27), (25, 27), (25, 43),
    (26, 27), (26, 29), (26, 31), (27, 28), (27, 31), (27, 43), (27, 44), (28, 30),
    (28, 31), (28, 32), (28, 33), (28, 39), (28, 40), (28, 44), (29, 30), (29, 31),
    (30, 31), (30, 33), (32, 33), (32, 37), (32, 40), (33, 34), (33, 37),
    (34, 35), (34, 37), (35, 36), (35, 37), (36, 37), (36, 38), (36, 39), (36, 40), (37, 40),
    (38, 39), (38, 41), (39, 40), (39, 41), (39, 42), (39, 44), (41, 42),
    (42, 43), (42, 44), (43, 44)
]
directed_edges = [
    (1, 45), (2, 45), (3, 45), (4, 45), (5, 45), (6, 45), (7, 45), (8, 45),
    (7, 47), (8, 47), (9, 47), (10, 46), (11, 46), (11, 48), (19, 50), (20, 50),
    (21, 50), (22, 50), (16, 51), (17, 51), (18, 51), (14, 51), (15, 51),
    (13, 52), (12, 53),
    (23, 45), (24, 45), (25, 45), (26, 45), (27, 45), (28, 45), (29, 45), (30, 45),
    (29, 47), (30, 47), (31, 47), (32, 46), (33, 46), (33, 48), (41, 50), (42, 50),
    (43, 50), (44, 50), (38, 51), (39, 51), (40, 51), (36, 51), (37, 51),
    (35, 52), (34, 53), (45,46),(47, 48), (48, 49), (50, 51), (46, 51),
    (49, 51), (51, 52), (52, 53),
    (54, 51), (2, 54), (3, 54), (5, 54), (6, 54), (21, 54), (22, 54),
    (24, 54), (25, 54), (27, 54), (28, 54), (43, 54), (44, 54)
]

a = create_adjacency_matrix(undirected_edges, directed_edges)
edge_index, edge_attr = dense_to_sparse(a)

with open("models/scaler_prec.pkl", "rb") as file:
    scaler_prec = pickle.load(file)
    
with open("models/scaler_temp.pkl", "rb") as file:
    scaler_temp = pickle.load(file)

with open("models/scaler_target.pkl", "rb") as file:
    scaler_flow = pickle.load(file)
    


def prepare_data_for_gnn(prec, temp, flow, m_stations, q_stations, lag):

    num_nodes = len(m_stations)*2 + len(q_stations)

    new_prec = prec[m_stations].fillna(0)
    new_temp = temp[m_stations].fillna(0)
    new_flow = flow[q_stations].fillna(0)
    
    scaled_flow = scaler_flow.transform(new_flow.values.flatten().reshape(-1,1))
    scaled_flow = pd.DataFrame(scaled_flow.reshape(new_flow.shape),columns=new_flow.columns)
    scaled_prec = scaler_prec.transform(new_prec.values.flatten().reshape(-1,1))
    scaled_prec = pd.DataFrame(scaled_prec.reshape(new_prec.shape),columns=new_prec.columns)
    scaled_temp = scaler_temp.transform(new_temp.values.flatten().reshape(-1,1))
    scaled_temp = pd.DataFrame(scaled_temp.reshape(new_temp.shape),columns=new_temp.columns)
    
    meteo = np.concatenate([scaled_prec, scaled_temp], axis=1)
    input_data = np.concatenate((meteo,scaled_flow),axis=1)
    
    num_features = input_data.shape[1] // num_nodes
    node_features_list = []

    for i in range(num_nodes):
        node_features_list.append(input_data[:, i * num_features:(i + 1) * num_features])

    node_features_tensor = torch.stack([torch.tensor(node_features, dtype=torch.float32) for node_features in node_features_list])
    graph_data = Data(x=node_features_tensor[:, 0:0 + lag, :].reshape(num_nodes, (input_data.shape[1] // num_nodes) * lag),  
                      edge_index=edge_index, edge_attr=edge_attr)
    
    return graph_data
    

    
    
    
    
    
