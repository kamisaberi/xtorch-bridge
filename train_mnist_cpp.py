import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import xtorch_bridge
import numpy as np


# Define SimpleCNN for parameter management (same as create_model.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Initialize model for parameter management
    model = SimpleCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Path to the TorchScript model
    model_path = "model.pt"

    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to NumPy arrays
        data_np = data.numpy()
        target_np = target.numpy().astype(np.int64)  # Ensure target is int64

        # Call C++ training function
        loss_np, grad_np = xtorch_bridge.train_model(model_path, data_np, target_np)

        # Update model parameters with gradients
        for i, param in enumerate(model.parameters()):
            param.grad = torch.from_numpy(grad_np[i])
        optimizer.step()
        optimizer.zero_grad()

        # Update the TorchScript model with new parameters
        state_dict = model.state_dict()
        traced_model = torch.jit.load(model_path)
        traced_model.load_state_dict(state_dict)
        traced_model.save(model_path)

        # Print loss
        print(f'Batch {batch_idx}, Loss: {loss_np.item():.4f}')

        # For demonstration, stop after a few batches
        if batch_idx >= 200:
            break

    print("Training completed!")


if __name__ == "__main__":
    main()
