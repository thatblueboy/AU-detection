import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STConv(nn.Module):
    '''
    Spatio Temporal GCN
    '''
    def __init__(self, in_channels, out_channels, num_edges, kernel_size=3, stride=1, residual=True):
        super(STConv, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                              padding=((kernel_size - 1) // 2, 0), stride=(stride, 1))
        self.tcn_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.padding = (kernel_size - 1) //2
        self.edge_weight = nn.Parameter(torch.ones(num_edges))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels) # per feature
            )

    def forward(self, x, edge_index):
        batch_size, num_timesteps, num_nodes, features = x.shape
        #residual
        res = self.residual(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        x_out = torch.zeros(batch_size, num_timesteps, num_nodes, self.gcn.out_channels, device=x.device)
        #spatial conv per batch per timestep
        for b in range(batch_size):        
            for t in range(num_timesteps):
                xt = x[b][t]
                xt = self.gcn(xt, edge_index, self.edge_weight)
                x_out[b][t] = xt

        x_out = x_out.permute(0, 3, 2, 1).contiguous()  # [batch, features, num_nodes, num_timesteps]
        x_out = self.gcn_bn(x_out) #per feature/channel
        x_out = self.relu(x_out)

        #temporal conv per batch
        x_out = self.tcn(x_out)
        x_out = self.tcn_bn(x_out)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        
        return self.relu(x_out +res)

class STGCN(nn.Module):
    def __init__(self, in_channels, num_nodes, num_edges, num_timesteps):
        super(STGCN, self).__init__()

        # Spatio-temporal convolution layers using STConv2
        self.st_conv1 = STConv(in_channels, out_channels=4, num_edges=num_edges, kernel_size=3)
        self.st_conv2 = STConv(4, out_channels=8, num_edges=num_edges, kernel_size=3)
        self.st_conv3 = STConv(8, out_channels=16, num_edges=num_edges, kernel_size=3)
        self.st_conv4 = STConv(16, out_channels=32, num_edges=num_edges, kernel_size=5)
        self.st_conv5 = STConv(32, out_channels=64, num_edges=num_edges, kernel_size=5)
        self.st_conv6 = STConv(64, out_channels=128, num_edges=num_edges, kernel_size=5)
        self.st_conv7 = STConv(128, out_channels=256, num_edges=num_edges, kernel_size=5)

        self.conv = nn.Conv2d(in_channels=num_timesteps, out_channels=1, kernel_size=(num_nodes, 1))

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  
            nn.ReLU(),  
            nn.Linear(128, 64),  
            nn.ReLU(),
            nn.Linear(64, 16),  
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Sigmoid()  # Binary classification
        )

    def forward(self, x, edge_index):
        batch_size, num_timesteps, num_nodes, _ = x.shape
        pad_size = 1  # Since kernel_size=3, we need to pad equally on both sides
        x = self.st_conv1(x, edge_index)
        x = self.st_conv2(x, edge_index)
        x = self.st_conv3(x, edge_index)
        x = self.st_conv4(x, edge_index)
        x = self.st_conv5(x, edge_index)
        x = self.st_conv6(x, edge_index)
        x = self.st_conv7(x, edge_index)
        
        # Global Average Pooling over nodes (keep batch & features)
        # x = torch.mean(x, dim=(1, 2))  # Reduce nodes dimension
        # [batch_size, num_timesteps, num_nodes, features]
        # [batch, channels, height, width]
        x = self.conv(x)

        # Flatten to [batch, time * channels]
        x_flat = x.view(batch_size, -1)
        
        # Pass through classifier
        y = self.classifier(x_flat)
        
        return y.squeeze(-1)  # (batch_size, 1)