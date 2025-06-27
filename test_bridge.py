import torch
import torchvision
import torchvision.transforms as transforms
import time
import os
import sys

# ==============================================================================
#  STEP 1: IMPORT YOUR C++ BRIDGE
# ==============================================================================
# This is the main entry point. Thanks to your __init__.py, we can
# import the C++ classes and functions as if they were native Python.
try:
    from xtorch_bridge import fit, LeNet5, Module
except ImportError as e:
    print("=" * 80)
    print("FATAL: Could not import the 'xtorch_bridge' module.")
    print("Please ensure you have successfully installed it using 'pip install .'")
    print(f"Original Python error: {e}")
    print("=" * 80)
    sys.exit(1)

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001
# Use "cuda" if you have a GPU and installed the CUDA-enabled PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"

# The path to the model saved by your C++ trainer.
# This needs to match the logic in your C++ Trainer's save_checkpoint method.
# Let's assume the best model from the last epoch is saved.
# You might need to adjust the epoch number based on your training.
BEST_MODEL_FILENAME = f"model_best_model_epoch_{EPOCHS}.pt"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_FILENAME)


# ==============================================================================
#  PHASE 1: DATA PREPARATION (Pure Python)
# ==============================================================================
def prepare_data_loaders():
    """
    Uses standard PyTorch tools to prepare the data.
    This demonstrates that the data pipeline can remain entirely in Python.
    """
    print("--- PHASE 1: Preparing Data in Python ---")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST stats
    ])

    print("Downloading MNIST dataset (if needed)...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    print("Creating Python DataLoaders...")
    # Using multiple workers in the Python DataLoader pre-fetches data on separate
    # processes, feeding the C++ training loop as fast as possible.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True # Speeds up CPU to GPU data transfer
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2, # Can use a larger batch size for inference
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Data ready. Using device: {DEVICE.upper()}")
    return train_loader, test_loader


# ==============================================================================
#  PHASE 2: TRAINING (Hand-off to C++)
# ==============================================================================
def train_model_in_cpp(train_loader, val_loader):
    """
    Instantiates the C++ model and hands off the entire training loop
    to the high-performance C++ backend.
    """
    print("\n--- PHASE 2: Training in C++ via xtorch-bridge ---")

    # 1. Instantiate your C++ model directly from Python.
    cpp_model = LeNet5(num_classes=10)
    print(f"Successfully created C++ LeNet5 model: {cpp_model}")

    # The C++ Trainer will handle moving the model and data to the device.

    # 2. Call the C++ 'fit' function. This is the core of the bridge.
    print(f"\nHanding off to C++ for training for {EPOCHS} epochs...")
    start_time = time.time()

    xtorch_bridge.fit(
        model=cpp_model,
        train_loader=train_loader,
        val_loader=val_loader, # Pass the test_loader as a validation set
        max_epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    end_time = time.time()
    print("\n--- C++ Training Finished ---")
    print(f"Total C++ training time: {end_time - start_time:.2f} seconds.")


# ==============================================================================
#  PHASE 3: INFERENCE (Python using the C++ model)
# ==============================================================================
def run_inference_with_cpp_model(test_loader):
    """
    Demonstrates loading a saved C++ model and running inference from Python.
    """
    print("\n--- PHASE 3: Inference using the C++ Model from Python ---")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Checkpoint not found at '{BEST_MODEL_PATH}'.")
        print("Please ensure your C++ Trainer's checkpointing is enabled and the path is correct.")
        return

    # 1. Create a new model instance to load the state into.
    inference_model = LeNet5(num_classes=10)

    # 2. Load the state dictionary using standard PyTorch functions.
    # This works because your C++ module is compatible with torch::save.
    print(f"Loading saved model state from: {BEST_MODEL_PATH}")
    state_dict = torch.load(BEST_MODEL_PATH)
    inference_model.load_state_dict(state_dict)

    # Move model to the correct device and set to evaluation mode
    inference_model.to(DEVICE)
    inference_model.eval()

    correct = 0
    total = 0

    print("Running inference on the test set...")
    with torch.no_grad(): # Essential for inference performance
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            # 3. Call the forward pass of the C++ model directly from Python.
            #    This requires the 'forward' method to be exposed by pybind11,
            #    which is often automatic for `torch::nn::Module` inheritance.
            outputs = inference_model(data) # Use the callable __call__

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"\nInference Accuracy on {total} test images: {accuracy:.2f}%")


# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    train_loader, test_loader = prepare_data_loaders()

    train_model_in_cpp(train_loader, test_loader)

    # This part assumes your C++ trainer successfully saved a checkpoint.
    run_inference_with_cpp_model(test_loader)