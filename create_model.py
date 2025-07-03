import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # [batch_size, 16, 28, 28]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # [batch_size, 32, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)               # [batch_size, 32, 14, 14]
        self.fc1 = nn.Linear(32 * 14 * 14, 128)      # [batch_size, 128]
        self.fc2 = nn.Linear(128, 10)                # [batch_size, 10]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Initialize model
    model = SimpleCNN()
    model.eval()  # Set to evaluation mode for tracing

    # Load a sample input for tracing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    sample_input, _ = next(iter(loader))  # Get one sample input

    # Trace the model
    traced_model = torch.jit.trace(model, sample_input)

    # Save the traced model
    traced_model.save("model.pt")
    print("Model saved as model.pt")

if __name__ == "__main__":
    main()