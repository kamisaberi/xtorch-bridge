import ctypes
import os

# --- IMPORTANT ---
# PASTE THE FULL, ABSOLUTE PATH to your compiled .so file here.
# Get this path from the output of `pip show -f xtorch_bridge`.
SO_FILE_PATH = "/home/kami/Documents/cpp/xtorch-bridge/.venv/lib/python3.13/site-packages/xtorch_bridge/xtorch_bridge_impl.cpython-313-x86_64-linux-gnu.so"

print(f"Attempting to load: {SO_FILE_PATH}")

if not os.path.exists(SO_FILE_PATH):
    print("\nFATAL: The .so file does not exist at the specified path.")
    print("The installation failed to create the file. Check the build log for C++ errors.")
else:
    try:
        # This is a direct call to the OS's dynamic loader (dlopen).
        # It will give a much more specific error message.
        ctypes.CDLL(SO_FILE_PATH)
        print("\nSUCCESS! The library was loaded by the OS successfully.")
        print("This is very strange. The problem might be in the pybind11 module initialization.")

    except OSError as e:
        print("\nFAILURE! The OS failed to load the library.")
        print("This is the REAL error message we were looking for:")
        print("-" * 60)
        print(f"{e}")
        print("-" * 60)
        print("\nIf the error contains 'undefined symbol', you have a C++ ABI mismatch.")