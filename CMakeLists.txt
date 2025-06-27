cmake_minimum_required(VERSION 3.18)
project(xtorch-bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
# You must tell CMake where to find your installed libraries by setting CMAKE_PREFIX_PATH.
# Example:
# cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/your/xtorch/install"

## 1. Find LibTorch
#find_package(Torch REQUIRED)

# 2. Find Python and pybind11
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# 3. Find your custom xtorch library

list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)
find_package(onnxruntime REQUIRED)






# --- Build the Python Module ---
# pybind11_add_module handles creating the Python extension module (.so or .pyd)
pybind11_add_module(xtorch_bridge
        src/bindings.cpp
        src/python_data_loader_bridge.cpp
)

# Specify include directories for this target
target_include_directories(xtorch_bridge PRIVATE
        # Its own public headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # The public headers from pybind11
        ${pybind11_INCLUDE_DIRS}
)

# Link the bridge against its dependencies
target_link_libraries(xtorch_bridge PRIVATE
        # The target exported by your xtorch library's install step
        xtorch::xtorch
        # The torch libraries
        ${TORCH_LIBRARIES}
)

# Optional: Add an install rule for the bridge itself
install(TARGETS xtorch_bridge
        LIBRARY DESTINATION xtorch_bridge # Installs to <prefix>/xtorch_bridge
)