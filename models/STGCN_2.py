import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv

class STGCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes, K, num_timesteps):
        super(STGCN_1, self).__init__()

        self.st_conv1 = STConv(num_nodes, 
                               in_channels,  # 3
                               hidden_channels=8,
                               out_channels=16, 
                               kernel_size=3, 
                               K=K)

        # Global Average Pooling instead of Flattening everything
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pools over (T, Nodes)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(16, 512),  # Output channels after pooling
            nn.ReLU(),  
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 16),  
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x, edge_index):
        batch_size, num_timesteps, num_nodes, _ = x.shape
        
        x = F.pad(x, (0, 0, 0, 0, 1, 1))  # Padding along time
        x = self.st_conv1(x, edge_index)
        x = F.relu(x)

        # Global Average Pooling
        x = self.global_pool(x)  # Shape: [batch, 16, 1, 1]
        x = x.view(batch_size, -1)  # Flatten to [batch, 16]

        # Pass through classifier
        y = self.classifier(x)
        return y.squeeze(-1)  # (batch_size, 1)
