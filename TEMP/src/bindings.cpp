#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// This is our simple C++ function that uses LibTorch.
torch::Tensor add_tensors_in_cpp(torch::Tensor a, torch::Tensor b) {
    std::cout << "[C++] The C++ function 'add_tensors_in_cpp' was called." << std::endl;
    return a + b;
}

// This is the pybind11 module definition.
// The name "xtorch_bridge_impl" MUST match the target name in CMakeLists.txt
// and the import name in __init__.py.
PYBIND11_MODULE(xtorch_bridge_impl, m) {
    m.doc() = "A simple C++ extension using LibTorch and pybind11";

    m.def(
        "add_tensors",           // The name of the function in Python
        &add_tensors_in_cpp,     // A pointer to our C++ function
        "Adds two PyTorch tensors in C++" // A docstring for the function
    );
}