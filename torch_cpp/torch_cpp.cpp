#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Added for Python list to std::vector conversion
#include <torch/torch.h>
#include <vector>

std::vector<float> matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, int rows_a, int cols_a, int cols_b) {
    // Convert input vectors to torch tensors
    torch::Tensor tensor_a = torch::from_blob(const_cast<float*>(a.data()), {rows_a, cols_a}, torch::kFloat32);
    torch::Tensor tensor_b = torch::from_blob(const_cast<float*>(b.data()), {cols_a, cols_b}, torch::kFloat32);

    // Perform matrix multiplication
    torch::Tensor result = torch::matmul(tensor_a, tensor_b);

    // Convert result back to std::vector<float>
    std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
    return output;
}

PYBIND11_MODULE(torch_cpp, m) {
    m.def("matrix_multiply", &matrix_multiply, "Perform matrix multiplication using LibTorch",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("rows_a"), pybind11::arg("cols_a"), pybind11::arg("cols_b"));
}