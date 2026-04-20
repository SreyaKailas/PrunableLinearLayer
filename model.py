import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PrunableLinear(nn.Module):
    """
    Custom Linear Layer that learns to prune its own weights 
    using a 'gate' mechanism.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # Standard weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate_scores: same shape as weights
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        # Start gates at a neutral state (Sigmoid(0) = 0.5)
        nn.init.constant_(self.gate_scores, 0.0) 

    def forward(self, x):
        # Transform scores to gates [0, 1] using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Standard linear operation
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    """
    A 3-layer feed-forward network using PrunableLinear layers for CIFAR-10.
    """
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10):
        super(SelfPruningNet, self).__init__()
        self.layer1 = PrunableLinear(input_dim, hidden_dim)
        self.layer2 = PrunableLinear(hidden_dim, hidden_dim)
        self.layer3 = PrunableLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_sparsity_loss(self):
        """Calculates L1 norm of all gates across the network."""
        total_l1 = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total_l1 += torch.sum(torch.sigmoid(m.gate_scores))
        return total_l1
