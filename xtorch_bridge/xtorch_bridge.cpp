#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <stdexcept>

std::vector<float> matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, int rows_a, int cols_a, int cols_b) {
    torch::Tensor tensor_a = torch::from_blob(const_cast<float*>(a.data()), {rows_a, cols_a}, torch::kFloat32);
    torch::Tensor tensor_b = torch::from_blob(const_cast<float*>(b.data()), {cols_a, cols_b}, torch::kFloat32);
    torch::Tensor result = torch::matmul(tensor_a, tensor_b);
    std::vector<float> output(result.data_ptr<float>(), result.data_ptr<float>() + result.numel());
    return output;
}

pybind11::array_t<float> forward_pass(const std::vector<pybind11::array_t<float>>& model_params, const pybind11::array_t<float>& input) {
    auto weights_info = model_params[0].request();
    auto bias_info = model_params[1].request();
    torch::Tensor weights = torch::from_blob(weights_info.ptr, {weights_info.shape[0], weights_info.shape[1]}, torch::kFloat32);
    torch::Tensor bias = torch::from_blob(bias_info.ptr, {bias_info.shape[0]}, torch::kFloat32);
    weights = weights.transpose(0, 1);
    auto input_info = input.request();
    torch::Tensor input_tensor = torch::from_blob(input_info.ptr, {input_info.shape[0], input_info.shape[1]}, torch::kFloat32);
    torch::Tensor output = torch::matmul(input_tensor, weights) + bias;
    auto output_array = pybind11::array_t<float>({output.size(0), output.size(1)});
    auto output_info = output_array.request();
    std::memcpy(output_info.ptr, output.data_ptr<float>(), output.numel() * sizeof(float));
    return output_array;
}

std::tuple<pybind11::array_t<float>, std::vector<pybind11::array_t<float>>> train_lenet(
    const std::vector<pybind11::array_t<float>>& model_params,
    const pybind11::array_t<float>& input,
    const pybind11::array_t<float>& target) {
    // Convert model parameters to torch::Tensor
    std::vector<torch::Tensor> params;
    params.reserve(model_params.size());
    for (const auto& param : model_params) {
        auto param_info = param.request();
        std::vector<int64_t> shape;
        shape.reserve(param_info.ndim);
        for (pybind11::ssize_t i = 0; i < param_info.ndim; ++i) {
            shape.push_back(static_cast<int64_t>(param_info.shape[i]));
        }
        params.push_back(torch::from_blob(param_info.ptr, shape, torch::kFloat32).clone().set_requires_grad(true));
    }

    // Convert input to torch::Tensor
    auto input_info = input.request();
    std::vector<int64_t> input_shape;
    input_shape.reserve(input_info.ndim);
    for (pybind11::ssize_t i = 0; i < input_info.ndim; ++i) {
        input_shape.push_back(static_cast<int64_t>(input_info.shape[i]));
    }
    torch::Tensor input_tensor = torch::from_blob(input_info.ptr, input_shape, torch::kFloat32);

    // Convert target to torch::Tensor (cast from float to int64_t)
    auto target_info = target.request();
    std::vector<int64_t> target_shape;
    target_shape.reserve(target_info.ndim);
    for (pybind11::ssize_t i = 0; i < target_info.ndim; ++i) {
        target_shape.push_back(static_cast<int64_t>(target_info.shape[i]));
    }
    std::vector<int64_t> target_data(target_info.size);
    float* target_ptr = static_cast<float*>(target_info.ptr);
    for (size_t i = 0; i < target_info.size; ++i) {
        target_data[i] = static_cast<int64_t>(target_ptr[i]);
    }
    torch::Tensor target_tensor = torch::from_blob(target_data.data(), target_shape, torch::kInt64);

    // Release GIL for forward and backward pass
    pybind11::gil_scoped_release no_gil;

    // LeNet architecture: conv1 -> relu -> maxpool -> conv2 -> relu -> maxpool -> flatten -> fc1 -> relu -> fc2 -> relu -> fc3
    torch::Tensor x = input_tensor; // [batch_size, 1, 28, 28]
    x = torch::conv2d(x, params[0], params[1], /*stride=*/1, /*padding=*/2); // conv1: [batch_size, 6, 28, 28]
    x = torch::relu(x);
    x = torch::max_pool2d(x, 2, 2); // [batch_size, 6, 14, 14]
    x = torch::conv2d(x, params[2], params[3], /*stride=*/1); // conv2: [batch_size, 16, 10, 10]
    x = torch::relu(x);
    x = torch::max_pool2d(x, 2, 2); // [batch_size, 16, 5, 5]
    x = x.view({x.size(0), -1}); // flatten: [batch_size, 16*5*5=400]
    x = torch::linear(x, params[4], params[5]); // fc1: [batch_size, 120]
    x = torch::relu(x);
    x = torch::linear(x, params[6], params[7]); // fc2: [batch_size, 84]
    x = torch::relu(x);
    x = torch::linear(x, params[8], params[9]); // fc3: [batch_size, 10]

    // Compute cross-entropy loss
    torch::Tensor loss = torch::nll_loss(torch::log_softmax(x, 1), target_tensor);

    // Backward pass
    loss.backward();

    // Re-acquire GIL after computations
    pybind11::gil_scoped_acquire acquire_gil;

    // Collect gradients
    std::vector<pybind11::array_t<float>> grad_arrays;
    grad_arrays.reserve(params.size());
    for (const auto& param : params) {
        torch::Tensor grad = param.grad();
        auto grad_array = pybind11::array_t<float>(std::vector<ssize_t>(grad.sizes().begin(), grad.sizes().end()));
        auto grad_info = grad_array.request();
        if (grad.defined() && grad.numel() > 0) {
            std::memcpy(grad_info.ptr, grad.data_ptr<float>(), grad.numel() * sizeof(float));
        } else {
            std::fill_n(static_cast<float*>(grad_info.ptr), grad_info.size, 0.0f);
        }
        grad_arrays.push_back(grad_array);
    }

    // Convert loss to NumPy array
    auto loss_array = pybind11::array_t<float>(1); // Scalar array with size 1
    auto loss_info = loss_array.request();
    *static_cast<float*>(loss_info.ptr) = loss.item<float>();

    return std::make_tuple(loss_array, grad_arrays);
}

std::tuple<pybind11::array_t<float>, std::vector<pybind11::array_t<float>>> train_model(
    const std::string& model_path,
    const pybind11::array_t<float>& input,
    const pybind11::array_t<float>& target) {
    // Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.train(); // Enable training mode for gradient computation
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path + ": " + e.what());
    }

    // Convert input to torch::Tensor
    auto input_info = input.request();
    if (input_info.ndim != 4) {
        throw std::runtime_error("Input must be 4D [batch_size, 1, 28, 28], got " +
            std::to_string(input_info.ndim) + " dimensions");
    }
    std::vector<int64_t> input_shape;
    input_shape.reserve(input_info.ndim);
    for (pybind11::ssize_t i = 0; i < input_info.ndim; ++i) {
        input_shape.push_back(static_cast<int64_t>(input_info.shape[i]));
    }
    if (input_shape[1] != 1 || input_shape[2] != 28 || input_shape[3] != 28) {
        throw std::runtime_error("Input shape must be [batch_size, 1, 28, 28], got [" +
            std::to_string(input_shape[0]) + "," + std::to_string(input_shape[1]) + "," +
            std::to_string(input_shape[2]) + "," + std::to_string(input_shape[3]) + "]");
    }
    torch::Tensor input_tensor = torch::from_blob(input_info.ptr, input_shape, torch::kFloat32).clone().set_requires_grad(true);

    // Convert target to torch::Tensor (cast from float to int64_t)
    auto target_info = target.request();
    if (target_info.ndim != 1) {
        throw std::runtime_error("Target must be 1D [batch_size], got " +
            std::to_string(target_info.ndim) + " dimensions");
    }
    std::vector<int64_t> target_shape;
    target_shape.reserve(target_info.ndim);
    for (pybind11::ssize_t i = 0; i < target_info.ndim; ++i) {
        target_shape.push_back(static_cast<int64_t>(target_info.shape[i]));
    }
    std::vector<int64_t> target_data(target_info.size);
    float* target_ptr = static_cast<float*>(target_info.ptr);
    for (size_t i = 0; i < target_info.size; ++i) {
        target_data[i] = static_cast<int64_t>(target_ptr[i]);
        if (target_data[i] < 0 || target_data[i] > 9) {
            throw std::runtime_error("Target value at index " + std::to_string(i) +
                " is " + std::to_string(target_data[i]) + ", must be in [0, 9]");
        }
    }
    torch::Tensor target_tensor = torch::from_blob(target_data.data(), target_shape, torch::kInt64);

    // Release GIL for forward and backward pass
    pybind11::gil_scoped_release no_gil;

    // Forward pass
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    torch::Tensor output;
    try {
        output = module.forward(inputs).toTensor();
        if (output.dim() != 2 || output.size(1) != 10) {
            throw std::runtime_error("Output shape must be [batch_size, 10], got [" +
                std::to_string(output.size(0)) + "," + std::to_string(output.size(1)) + "]");
        }
    } catch (const c10::Error& e) {
        throw std::runtime_error("Forward pass failed: " + std::string(e.what()));
    }

    // Compute loss
    torch::Tensor loss;
    try {
        loss = torch::nll_loss(torch::log_softmax(output, 1), target_tensor);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Loss computation failed: " + std::string(e.what()));
    }

    // Backward pass
    try {
        loss.backward();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Backward pass failed: " + std::string(e.what()));
    }

    // Re-acquire GIL after computations
    pybind11::gil_scoped_acquire acquire_gil;

    // Collect gradients
    std::vector<pybind11::array_t<float>> grad_arrays;
    auto parameters = module.parameters();
    grad_arrays.reserve(parameters.size());
    for (const auto& param : parameters) {
        torch::Tensor grad = param.grad();
        std::vector<ssize_t> grad_shape(param.sizes().begin(), param.sizes().end()); // Use param shape if grad undefined
        auto grad_array = pybind11::array_t<float>(grad_shape);
        auto grad_info = grad_array.request();
        if (grad.defined() && grad.numel() > 0 && grad.sizes() == param.sizes()) {
            std::memcpy(grad_info.ptr, grad.data_ptr<float>(), grad.numel() * sizeof(float));
        } else {
            std::fill_n(static_cast<float*>(grad_info.ptr), grad_info.size, 0.0f);
        }
        grad_arrays.push_back(grad_array);
    }

    // Convert loss to NumPy array
    auto loss_array = pybind11::array_t<float>(1); // Scalar array with size 1
    auto loss_info = loss_array.request();
    *static_cast<float*>(loss_info.ptr) = loss.item<float>();

    return std::make_tuple(loss_array, grad_arrays);
}

PYBIND11_MODULE(torch_cpp, m) {
    m.def("matrix_multiply", &matrix_multiply, "Perform matrix multiplication using LibTorch",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("rows_a"), pybind11::arg("cols_a"), pybind11::arg("cols_b"));
    m.def("forward_pass", &forward_pass, "Perform a forward pass with a linear layer",
          pybind11::arg("model_params"), pybind11::arg("input"));
    m.def("train_lenet", &train_lenet, "Train LeNet on a batch of MNIST data",
          pybind11::arg("model_params"), pybind11::arg("input"), pybind11::arg("target"));
    m.def("train_model", &train_model, "Train a TorchScript model on a batch of data",
          pybind11::arg("model_path"), pybind11::arg("input"), pybind11::arg("target"));
}