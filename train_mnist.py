import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch_cpp
import numpy as np

# Define LeNet architecture (for reference)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Initialize LeNet model
    model = LeNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Extract model parameters as NumPy arrays
    model_params = [param.detach().numpy() for param in model.parameters()]





    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to NumPy arrays
        data_np = data.numpy()
        target_np = target.numpy().astype(np.int64)  # Ensure target is int64

        # Call C++ training function
        loss_np, grad_np = torch_cpp.train_lenet(model_params, data_np, target_np)

        # Update model parameters with gradients
        for i, param in enumerate(model.parameters()):
            param.grad = torch.from_numpy(grad_np[i])
            model_params[i] = param.detach().numpy()  # Update model_params for next iteration

        # PyTorch optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Print loss
        print(f'Batch {batch_idx}, Loss: {loss_np.item():.4f}')

        # For demonstration, stop after a few batches
        if batch_idx >= 200:
            break

    print("Training completed!")

if __name__ == "__main__":
    main()