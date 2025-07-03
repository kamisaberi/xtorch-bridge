import torch
import torchvision
import torchvision.transforms as transforms
import xtorch_bridge
import os

# Path to the TorchScript model
MODEL_PATH = "lenet_model.pt"

def create_and_save_model():
    """Creates a PyTorch LeNet model and saves it as a TorchScript file."""
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
        model.eval() # Important for tracing
        scripted_model = torch.jit.script(model)
        scripted_model.save(MODEL_PATH)
    else:
        print(f"Using existing TorchScript model from {MODEL_PATH}")

def main():
    # 1. Create the TorchScript model file if it doesn't exist
    create_and_save_model()

    # 2. Load the dataset in Python
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 3. Load the model INTO C++ memory ONCE.
    #    The `cpp_model_manager` now owns the model state.
    print("\n--- Initializing C++ ModelManager ---")
    cpp_model_manager = xtorch_bridge.ModelManager(MODEL_PATH)

    # 4. Get the parameters from C++ and create the C++ optimizer
    params_in_cpp = cpp_model_manager.get_parameters()
    cpp_optimizer = xtorch_bridge.SGDOptimizer(params_in_cpp, lr=0.01, momentum=0.9)
    print("C++ SGDOptimizer created.")

    # 5. The new, efficient training loop
    print("\n--- Starting C++ Accelerated Training Loop ---")
    for epoch in range(2): # Train for 2 epochs
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # The ONLY data transferred is the input batch.
            # No model params or gradients cross the Python/C++ boundary.

            # a. Clear gradients in C++
            cpp_optimizer.zero_grad()

            # b. Run forward/backward pass in C++
            #    The GIL is released during this call for performance.
            loss = cpp_model_manager.train_batch(data, target)

            # c. Update parameters in C++
            cpp_optimizer.step()

            total_loss += loss
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Avg Loss: {total_loss / (batch_idx+1):.4f}")

    print("\n--- Training Completed! ---")

    # 6. Save the final, trained model from C++
    final_model_path = "lenet_model_trained.pt"
    cpp_model_manager.save(final_model_path)
    print(f"Final trained model saved to {final_model_path}")

if __name__ == "__main__":
    main()