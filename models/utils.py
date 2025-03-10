import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class STConv(nn.Module):
    '''
    '''
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        # Single Temporal Convolution
        self.temporal_conv = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size)
        )
        
        # Graph Convolution
        self.graph_conv = ChebConv(out_channels, out_channels, K=K, normalization=normalization, bias=bias)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        
        # Residual Connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
        
    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None):
        # Residual connection
        res = self.residual(X)
        
        # Temporal Convolution
        T = self.temporal_conv(X.permute(0, 3, 2, 1))
        T = F.relu(T)
        
        # Graph Convolution
        T = T.permute(0, 3, 1, 2)  # Reshape for graph conv
        batch_size, num_nodes, seq_len, channels = T.shape
        T_out = torch.zeros_like(T).to(T.device)
        for b in range(batch_size):
            for t in range(seq_len):
                T_out[b, :, t, :] = self.graph_conv(T[b, :, t, :], edge_index, edge_weight)
        
        # Batch Normalization
        T_out = self.batch_norm(T_out.permute(0, 2, 1, 3))
        T_out = T_out.permute(0, 2, 1, 3)
        
        return F.relu(T_out + res)
