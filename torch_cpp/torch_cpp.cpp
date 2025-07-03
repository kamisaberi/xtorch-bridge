#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For Python list to std::vector conversion
#include <pybind11/numpy.h>  // For torch::Tensor conversion via NumPy
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

pybind11::array_t<float> forward_pass(const std::vector<pybind11::array_t<float>>& model_params, const pybind11::array_t<float>& input) {
    // Convert model_params (weights and bias) from NumPy arrays to torch::Tensor
    auto weights_info = model_params[0].request();
    auto bias_info = model_params[1].request();
    torch::Tensor weights = torch::from_blob(weights_info.ptr, {weights_info.shape[0], weights_info.shape[1]}, torch::kFloat32);
    torch::Tensor bias = torch::from_blob(bias_info.ptr, {bias_info.shape[0]}, torch::kFloat32);

    // Transpose weights to [3, 2] for correct matrix multiplication
    weights = weights.transpose(0, 1);  // From [2, 3] to [3, 2]

    // Convert input from NumPy array to torch::Tensor
    auto input_info = input.request();
    torch::Tensor input_tensor = torch::from_blob(input_info.ptr, {input_info.shape[0], input_info.shape[1]}, torch::kFloat32);

    // Perform linear layer computation: output = input @ weights + bias
    torch::Tensor output = torch::matmul(input_tensor, weights) + bias;

    // Convert output back to NumPy array
    auto output_array = pybind11::array_t<float>({output.size(0), output.size(1)});
    auto output_info = output_array.request();
    std::memcpy(output_info.ptr, output.data_ptr<float>(), output.numel() * sizeof(float));
    return output_array;
}

PYBIND11_MODULE(torch_cpp, m) {
    m.def("matrix_multiply", &matrix_multiply, "Perform matrix multiplication using LibTorch",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("rows_a"), pybind11::arg("cols_a"), pybind11::arg("cols_b"));
    m.def("forward_pass", &forward_pass, "Perform a forward pass with a linear layer",
          pybind11::arg("model_params"), pybind11::arg("input"));
}