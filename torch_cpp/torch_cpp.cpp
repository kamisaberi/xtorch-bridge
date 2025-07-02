#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;

std::vector<float> matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, int rows_a, int cols_a, int cols_b) {
    torch::Tensor tensor_a = torch::from_blob(const_cast<float*>(a.data()), {rows_a, cols_a});
    torch::Tensor tensor_b = torch::from_blob(const_cast<float*>(b.data()), {cols_a, cols_b});
    torch::Tensor result = torch::matmul(tensor_a, tensor_b);
    std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
    return output;
}

PYBIND11_MODULE(torch_cpp, m) {
    m.doc() = "Python bindings for LibTorch matrix multiplication";
    m.def("matrix_multiply", &matrix_multiply, "Perform matrix multiplication using LibTorch",
          py::arg("a"), py::arg("b"), py::arg("rows_a"), py::arg("cols_a"), py::arg("cols_b"));
}