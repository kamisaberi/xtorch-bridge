import torch
import torchvision
import torchvision.transforms as transforms
import xtorch_bridge
import os

MODEL_PATH = "lenet_model.pt"

def create_and_save_model():
    class LeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
            self.conv2 = torch.nn.Conv2d(6, 16, 5)
            self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
            self.fc2 = torch.nn.Linear(120, 84)
            self.fc3 = torch.nn.Linear(84, 10)
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

    if not os.path.exists(MODEL_PATH):
        print(f"Creating and saving TorchScript model to {MODEL_PATH}")
        model = LeNet()
        model.eval()
        scripted_model = torch.jit.script(model)
        scripted_model.save(MODEL_PATH)
    else:
        print(f"Using existing TorchScript model from {MODEL_PATH}")

def main():
    create_and_save_model()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    print("\n--- Initializing C++ ModelManager ---")
    cpp_model_manager = xtorch_bridge.ModelManager(MODEL_PATH)

    print("\n--- Creating C++ Optimizer ---")
    params_in_cpp = cpp_model_manager.get_parameters()
    cpp_optimizer = xtorch_bridge.SGDOptimizer(params_in_cpp, lr=0.01, momentum=0.9)
    print("C++ SGDOptimizer created successfully.")

    print("\n--- Starting C++ Accelerated Training Loop ---")
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            cpp_optimizer.zero_grad()
            loss = cpp_model_manager.train_batch(data, target)
            cpp_optimizer.step()

            total_loss += loss
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Avg Loss: {total_loss / (batch_idx+1):.4f}")

    print("\n--- Training Completed! ---")
    final_model_path = "lenet_model_trained.pt"
    cpp_model_manager.save(final_model_path)
    print(f"Final trained model saved to {final_model_path}")

if __name__ == "__main__":
    main()