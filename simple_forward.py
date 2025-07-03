import torch
import torch.nn as nn
import xtorch_bridge
import numpy as np

def main():
    # Define a simple PyTorch model: single linear layer (3 inputs, 2 outputs)
    model = nn.Linear(3, 2, bias=True)

    # Generate dummy input data: batch of 4 samples, each with 3 features
    input_data = torch.randn(4, 3, dtype=torch.float32)

    # Get model parameters (weights and bias) as NumPy arrays
    weights = model.weight.detach().numpy()  # Shape: [2, 3]
    bias = model.bias.detach().numpy()       # Shape: [2]
    model_params = [weights, bias]           # List of NumPy arrays for C++ function

    # Convert input to NumPy array
    input_np = input_data.numpy()

    # Call C++ function
    output_np = xtorch_bridge.forward_pass(model_params, input_np)

    # Convert output back to torch.Tensor for comparison
    output = torch.from_numpy(output_np)

    # Verify by running the same computation in Python
    expected_output = model(input_data)

    print("C++ Output (from xtorch_bridge.forward_pass):")
    print(output_np)
    print("\nExpected Output (from PyTorch):")
    print(expected_output.detach().numpy())

    # Check if results match
    if torch.allclose(output, expected_output, atol=1e-5):
        print("\nSuccess: C++ and PyTorch outputs match!")
    else:
        print("\nError: Outputs do not match!")

if __name__ == "__main__":
    main()