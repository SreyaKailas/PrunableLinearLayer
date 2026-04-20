import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Part 1: The "Prunable" Linear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate_scores: registered as a parameter for optimization
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.gate_scores, 0.5) # Initialize gates near the middle

    def forward(self, x):
        # Apply sigmoid to constrain gates between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication to get pruned weights
        pruned_weights = self.weight * gates
        
        # Standard linear operation: y = xA^T + b
        return F.linear(x, pruned_weights, self.bias)

# Part 2: The Neural Network Definition
class PruningNet(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10):
        super(PruningNet, self).__init__()
        self.layer1 = PrunableLinear(input_dim, hidden_dim)
        self.layer2 = PrunableLinear(hidden_dim, hidden_dim)
        self.layer3 = PrunableLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten CIFAR-10 images
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_sparsity_loss(self):
        # Calculate L1 norm (sum of values) of all gates across prunable layers
        total_l1 = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total_l1 += torch.sum(torch.sigmoid(m.gate_scores))
        return total_l1

    def get_sparsity_stats(self, threshold=1e-2):
        # Calculate percentage of weights where gate < threshold
        total_weights = 0
        pruned_weights = 0
        all_gates = []
        
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
                all_gates.extend(gates)
                total_weights += gates.size
                pruned_weights += np.sum(gates < threshold)
        
        sparsity_level = (pruned_weights / total_weights) * 100
        return sparsity_level, all_gates

# Part 3: Training and Evaluation Loop
def train_and_evaluate(lambd, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = PruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            class_loss = criterion(output, target)
            sparsity_loss = model.get_sparsity_loss()
            
            # Total Loss = ClassificationLoss + lambda * SparsityLoss
            total_loss = class_loss + lambd * sparsity_loss
            total_loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_state():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    sparsity, gates = model.get_sparsity_stats()
    return accuracy, sparsity, gates

# Main execution for different Lambda values
if __name__ == "__main__":
    lambdas = [1e-5, 1e-4, 1e-3]
    results = []

    for l in lambdas:
        print(f"Training with Lambda: {l}")
        acc, sp, gates = train_and_evaluate(l)
        results.append((l, acc, sp))
        
        # Save plot for the best model (usually medium lambda)
        if l == 1e-4:
            plt.hist(gates, bins=50)
            plt.title(f"Gate Distribution (Lambda={l})")
            plt.xlabel("Gate Value")
            plt.ylabel("Frequency")
            plt.savefig("gate_distribution.png")
            plt.show()

    print("\nSummary Table:")
    print("Lambda | Test Accuracy | Sparsity (%)")
    for r in results:
        print(f"{r[0]:.1e} | {r[1]:.2f}% | {r[2]:.2f}%")
