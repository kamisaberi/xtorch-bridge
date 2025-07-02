import sys
import os
import time

# ==============================================================================
#  TEST SCRIPT: Isolate the Crashing Import
# ==============================================================================

print("--- Python Script Starting ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("-" * 40)
# The flush=True is critical to ensure we see the message before a crash.
print("Step 1: Attempting to import 'torch'...", flush=True)

# --- Test 1: Does PyTorch itself load? ---
try:
    import torch
    print("SUCCESS: 'torch' imported successfully.", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"PyTorch C++ ABI setting: {torch._C._GLIBCXX_USE_CXX11_ABI}", flush=True)
except Exception as e:
    print(f"\nFATAL: Failed to import torch. The error is in your PyTorch installation, not xtorch-bridge.", flush=True)
    print(f"Error: {e}", flush=True)
    sys.exit(1)

print("-" * 40)
time.sleep(1) # Small pause to make the log easier to read

print("Step 2: Attempting to import 'xtorch_bridge' C++ backend...", flush=True)

# --- Test 2: Does your C++ module load? ---
try:
    # This is the line that will trigger the loading of your .so file
    # and all of its dependencies (libtorch.so, libxTorch.so, etc.)
    from xtorch_bridge import fit, LeNet5, Module
    print("SUCCESS: 'xtorch_bridge' imported successfully!", flush=True)
except Exception as e:
    print(f"\nFATAL: Failed to import xtorch_bridge. The crash is happening while loading your C++ module.", flush=True)
    print(f"This is almost certainly a runtime linking error (check LD_LIBRARY_PATH) or a C++ ABI mismatch.")
    print(f"Original error: {e}", flush=True)
    sys.exit(1)


print("-" * 40)
time.sleep(1)

print("--- All Imports Successful. Proceeding to run the program. ---", flush=True)

# If we get here, the rest of the original script can run
# This is a simplified version for testing.
try:
    print("\nInstantiating C++ LeNet5 model...", flush=True)
    model = LeNet5(num_classes=10)
    print("Model instantiated successfully.", flush=True)

    # We won't run the full training loop, just prove that the functions
    # are callable without crashing.
    print("\nTesting that the 'fit' function exists...", flush=True)
    print(f"Found function: {fit}")
    print("\n--- TEST COMPLETE: SUCCESS ---", flush=True)

except Exception as e:
    print(f"\nFATAL: An error occurred after import, while using the module.", flush=True)
    print(f"Error: {e}", flush=True)
    sys.exit(1)