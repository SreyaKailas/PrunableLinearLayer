import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SelfPruningNet

def train_model(lambd, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard CIFAR-10 preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), 
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), 
                             batch_size=1000, shuffle=False)

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"--- Training with Lambda: {lambd} ---")
    for epoch in range(epochs):
        model.train()
        total_running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            class_loss = criterion(output, target)
            sparsity_loss = model.get_sparsity_loss()
            
            loss = class_loss + lambd * sparsity_loss
            loss.backward()
            optimizer.step()
            total_running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_running_loss/len(train_loader):.4f}")

    return model, test_loader
