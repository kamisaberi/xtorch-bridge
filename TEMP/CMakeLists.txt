cmake_minimum_required(VERSION 3.18)
project(xtorch_bridge_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
# These are found using the CMAKE_PREFIX_PATH from pyproject.toml
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# --- Build the Python Module ---
add_library(xtorch_bridge_impl MODULE src/bindings.cpp) # Only the simple example
pybind11_extension(xtorch_bridge_impl)

# --- Link Dependencies ---
target_link_libraries(xtorch_bridge_impl PRIVATE
        pybind11::module
        Torch::torch
)

# --- Installation & RPATH ---
# This tells CMake where to install the final .so file inside the Python wheel.
install(TARGETS xtorch_bridge_impl
        LIBRARY DESTINATION xtorch_bridge
)

# This bakes the runtime search paths directly into the .so file.
# It gets the path to libtorch from the CMAKE_PREFIX_PATH.
set(RPATH_DIRS "$ENV{LIBTORCH_PATH}/lib")
set_target_properties(xtorch_bridge_impl PROPERTIES
        INSTALL_RPATH "${RPATH_DIRS}"
)
message(STATUS "Baking RPATH into library: ${RPATH_DIRS}")