cmake_minimum_required(VERSION 3.18)
project(xtorch_bridge_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(xtorch_bridge_impl MODULE src/bindings.cpp src/python_data_loader_bridge.cpp)
pybind11_extension(xtorch_bridge_impl)

target_link_libraries(xtorch_bridge_impl PRIVATE
        pybind11::module
        xTorch::xTorch
        Torch::torch
        OpenCV::opencv_core
        CUDA::cudart
)

# This applies the RPATH variable passed in from setup.py
set_target_properties(xtorch_bridge_impl PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
)