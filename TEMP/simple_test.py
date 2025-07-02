import torch
import xtorch_bridge

print("\n--- Python: Calling C++ function ---")
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = xtorch_bridge.add_tensors(a, b)

print("Result from C++:\n", c)
print("\nSUCCESS: The simple example works in your project structure!")