import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl 
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from utils import idx_sample

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation,
        ])
    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1)  # row normalize
        return h

class Encoder_global(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.g = graph
    def forward(self, h):
        h = self.encoder(h)
        return h


class GlobalAutoencoder(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = Encoder_global(graph, in_dim, out_dim, activation)
        self.g = graph
    def forward(self, feats):
        z = self.encoder(feats)
        adj_reconstructed = torch.matmul(z, z.t())
        adj_reconstructed = torch.sigmoid(adj_reconstructed)
        return adj_reconstructed, z
    
class ADEncoder(nn.Module):
    def __init__(self, g, in_dim0, hidden_dim):
        super().__init__()
        # 添加自环
        self.g = dgl.add_self_loop(g)
        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.gconv = GraphConv(hidden_dim, hidden_dim, activation=None)
    
    def forward(self, x):
        h0 = self.mlp0(x)
        l1 = self.gconv(self.g, h0)
        return l1, h0

class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, use_final_relu: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.use_final_relu = use_final_relu  

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_final_relu:      
            x = F.relu(x)
        return x

class ADDecoders(nn.Module):
    def __init__(self, hidden_dim, feat_dim):
        super().__init__()
        self.feature_decoder = FNN(hidden_dim, hidden_dim, feat_dim, use_final_relu=False)
        # 度解码：保留 ReLU，确保非负
        self.degree_decoder  = FNN(hidden_dim, hidden_dim, 1, use_final_relu=True)
    def decode_features(self, z):
        return self.feature_decoder(z)
    
    def decode_degree(self, z):
        return F.relu(self.degree_decoder(z))

    

class AttributeDegreeAutoencoder(nn.Module):
    def __init__(self, graph, feat_dim, hidden_dim):
        super().__init__()
        self.g = graph
        self.encoder = ADEncoder(graph, feat_dim, hidden_dim)
        self.decoders = ADDecoders(hidden_dim, feat_dim)
        self.mse = nn.MSELoss(reduction='mean')

    @torch.no_grad()
    def _node_degrees(self):
        deg = self.g.out_degrees().float()
        if next(self.parameters()).is_cuda:
            deg = deg.cuda()
        return deg.view(-1, 1)

    def forward(self, x):
        z, h0 = self.encoder(x)
        x_hat = self.decoders.decode_features(z)  
        deg_hat = self.decoders.decode_degree(z)
        deg_gt = self._node_degrees()
  
        feat_err_node = (x - x_hat).pow(2).mean(dim=1, keepdim=True)  # [N,1]
        deg_err_node = (deg_gt - deg_hat).pow(2)                      # [N,1]
        
        # 批量损失
        feat_loss = self.mse(x, x_hat)
        deg_loss = self.mse(deg_gt, deg_hat)
        
        return {
            'z': z, 'h0': h0,
            'feat_loss': feat_loss,
            'deg_loss': deg_loss,
            'feat_err_node': feat_err_node,  # [N,1]
            'deg_err_node': deg_err_node,   # [N,1]
        }



