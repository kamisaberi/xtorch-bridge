import sys
import os

# --- THIS IS THE CRUCIAL PART ---
#
# 1. Get the path to the directory CONTAINING your .so file.
#    You must PASTE THE PATH YOU FOUND in Step 3 here.
#    For example: './build/lib.linux-x86_64-cpython-310/xtorch_bridge'
#
#    NOTE: It's the path to the *directory*, not the file itself.
COMPILED_LIB_DIR = '/path/to/the/directory/containing/the/so/file' # <-- PASTE HERE

# 2. Add this directory to Python's path so it knows where to look.
print(f"Attempting to add to path: {os.path.abspath(COMPILED_LIB_DIR)}")
sys.path.append(os.path.abspath(COMPILED_LIB_DIR))

# 3. Now, try to import the module by its filename (without the .so part)
try:
    print("Attempting to import 'xtorch_bridge_impl'...")
    import xtorch_bridge_impl
    print("\nSUCCESS! The C++ module was loaded directly.")
    print("This means the .so file is valid and all its dependencies were found at runtime.")
    print("The problem is 100% in the packaging/installation step.")

except ImportError as e:
    print("\nFAILURE! The C++ module could not be loaded.")
    print("This means the .so file itself is broken or its dependencies cannot be found.")
    print("-" * 60)
    print(f"Original Error: {e}")
    print("-" * 60)
    print("\nNext Steps: Check your LD_LIBRARY_PATH or for a C++ ABI mismatch.")