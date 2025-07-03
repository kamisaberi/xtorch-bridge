import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_cpp
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Initialize LeNet model and optimizer
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop (1 epoch for simplicity)
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # Get model parameters as NumPy arrays
        model_params = [p.detach().numpy() for p in model.parameters()]  # [conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b, fc2.w, fc2.b, fc3.w, fc3.b]

        # Convert data and target to NumPy arrays
        data_np = data.numpy()  # [batch_size, 1, 28, 28]
        target_np = target.numpy()  # [batch_size]

        # Call C++ function
        loss_np, grad_np = torch_cpp.train_lenet(model_params, data_np, target_np)

        # Convert loss and gradients back to torch.Tensor
        loss = torch.tensor(loss_np.item())
        gradients = [torch.from_numpy(g) for g in grad_np]

        # Update model parameters with gradients
        with torch.no_grad():
            for param, grad in zip(model.parameters(), gradients):
                param.grad = grad
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx >= 200:  # Limit for quick testing
            break

    print("Training completed!")

if __name__ == "__main__":
    main()