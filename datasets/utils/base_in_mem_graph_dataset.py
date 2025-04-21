import os
from abc import ABC, abstractmethod

import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset

from datasets.utils.helpers import GraphGenerator


class BaseInMemoryGraphDataset(Dataset, ABC): 
    '''
    Makes stacked Graphs of n frames from onset to offset
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
                 device='cpu',
                 processed_data_path=None):
        
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
        self.processed_data_path = processed_data_path

        edge_list = []
        node_mapping = {node: i for i, node in enumerate(landmarks)}
        for src, dst in edges:
            edge_list.append((node_mapping[src], node_mapping[dst]))  # Forward edge
            edge_list.append((node_mapping[dst], node_mapping[src]))  # Reverse edge (undirected)

        if self_loop_in_edge_index:
            for i in range(len(landmarks)):
                edge_list.append((i, i))
             
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T

        if processed_data_path is None:
            print("No processed data path provided. Generating graphs...")
            self.generate_graphs()
            
        elif not os.path.exists(processed_data_path) or os.path.getsize(processed_data_path) == 0:
            print(f"{processed_data_path} is missing or empty. Generating and saving new graphs...")
            self.generate_graphs()
            try:
                with open(processed_data_path, 'wb') as f:
                    pickle.dump({
                        'x': self.X,
                        'all_au_labels': self.all_au_labels,
                        'edge_index': self.edge_index,
                        'subjects': self.subjects
                    }, f)
                print(f"Data saved to {processed_data_path}")
            except Exception as e:
                print(f"Failed to save data: {e}")
            
        else:
            try:
                with open(processed_data_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                print(f"Data successfully loaded from {processed_data_path}")

                self.X = loaded_data.get('x')
                self.all_au_labels = loaded_data.get('all_au_labels')
                self.edge_index = loaded_data.get('edge_index')
                self.subjects = loaded_data.get('subjects')

            except Exception as e:
                print(f"An error occurred while loading the data: {e}")

        self.au_indices = [0] #which of au{list} is label

        self.X = self.X.to(self.device)
        self.all_au_labels = self.all_au_labels.to(self.device)

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
