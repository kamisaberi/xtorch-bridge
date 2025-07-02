#!/bin/bash
set -e # Exit immediately if a command fails

# 1. Activate the virtual environment to ensure we use the correct Python
#    (This might not be strictly necessary, but it's good practice)
source .venv/bin/activate

# 2. Clean up any previous build artifacts
echo "--- Cleaning up old build artifacts ---"
rm -rf build/

# 3. Create a new build directory
mkdir -p build && cd build

# 4. Run CMake with all the correct flags and paths
#    This is the command we proved works.
echo "--- Running CMake configuration ---"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/home/kami/libs/cpp/libtorch"

# 5. Run Make to compile the C++ code
echo "--- Compiling C++ code ---"
make -j8

# 6. Go back to the root directory
cd ..

# 7. Copy the final .so file into the Python package directory
#    This is the crucial step that setup.py was failing to do reliably.
echo "--- Copying .so file to Python package directory ---"
cp build/xtorch_bridge_impl.*.so xtorch_bridge/

echo "--- Build complete! The .so file is now in the xtorch_bridge/ directory. ---"