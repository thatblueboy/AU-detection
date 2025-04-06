from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
import pandas as pd
from datasets.utils.helpers import GraphGenerator


class BaseInMemoryGraphDataset(Dataset, ABC): 
    '''
    Makes stacked Graphs of 5 frames from onset to offset
    input: frames_path, depth_path, labels_path, landmarks(different for different AU), edges(different for different AU)
    '''
    def __init__(self, 
                 frames_path, 
                 labels_csv_path,
                 aus:list,
                 central_landmark,
                 landmarks, 
                 num_timesteps,
                 edges, 
                 noise = False,
                 noise_params = [0, 0.0015], #mean, std
                 noise_bounds = [-0.0010, 0.0010], #low, high #0.0010 suitable for normalization of 200
                 include_flipped=False,
                 self_loop_in_edge_index=True,
                 normalizing_factor=400,
                 image_size=(400, 400),
                 device='cpu'):
        
        assert num_timesteps in {5, 10, 15, 20}, "timesteps must be 5, 10, 15, or 20"

        landmarks = sorted(landmarks)
        self.graph_generator = GraphGenerator(landmarks, 
                                              normalizing_factor, 
                                              central_landmark)
        self.frames_path = frames_path
        self.labels_df = pd.read_csv(labels_csv_path)
        self.image_size = image_size
        self.edges = edges
        self.device = device
        self.central_landmark=central_landmark
        self.aus = sorted(aus)
        self.num_timesteps = num_timesteps
        self.num_nodes = len(landmarks)
        self.include_flipped = include_flipped
        self.noise_params = noise_params
        self.noise_bounds = noise_bounds
        self.noise = noise

        edge_list = []
        node_mapping = {node: i for i, node in enumerate(landmarks)}
        for src, dst in edges:
            edge_list.append((node_mapping[src], node_mapping[dst]))  # Forward edge
            edge_list.append((node_mapping[dst], node_mapping[src]))  # Reverse edge (undirected)

        if self_loop_in_edge_index:
            for i in range(len(landmarks)):
                edge_list.append((i, i))
             
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T

        self.generate_graphs()
        
        self.au_indices = [0] #which of au{list} is label

        self.X = self.X.to(self.device)
        self.y = self.all_au_labels.to(self.device)
        # self.y = self.y.to(self.device)

    @abstractmethod
    def generate_graphs(self):
        """Each subclass should implement this method to generate dataset-specific graphs."""
        pass

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        x = self.X[idx]  # Get the original graph node features
    
        # Apply noise if enabled
        if self.noise:
            mean, std = self.noise_params
            noise = torch.normal(mean=mean, std=std, size=x.shape, device=self.device)

            # Clamp noise to be within noise_bounds
            low, high = self.noise_bounds
            noise = torch.clamp(noise, min=low, max=high)

            # Add noise to the graph features
            x = x + noise     
        return x, self.edge_index, self.all_au_labels[idx, self.au_indices].squeeze(-1), self.subjects[idx]

    def reset_label(self, aus):
        self.au_indices = [self.aus.index(au) for au in aus]
