import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PrunableLinear

def evaluate_and_plot(model, test_loader, lambd, threshold=0.01):
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Calculate Accuracy
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)

    # 2. Calculate Sparsity & Collect Gate Values
    total_gates = 0
    pruned_gates = 0
    all_gate_values = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gate_values.extend(gates)
            total_gates += gates.size
            pruned_gates += np.sum(gates < threshold)

    sparsity = (pruned_gates / total_gates) * 100

    # 3. Plot Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_gate_values, bins=100, color='skyblue', edgecolor='black')
    plt.title(f"Gate Value Distribution ($\lambda$={lambd})")
    plt.xlabel("Gate Value (0 = Pruned, 1 = Active)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"gate_dist_lambda_{lambd}.png")
    
    return accuracy, sparsity
