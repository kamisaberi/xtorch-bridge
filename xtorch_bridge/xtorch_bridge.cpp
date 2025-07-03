#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // <--- THIS IS THE FIX. Add this line.
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <torch/csrc/utils/pybind.h>


namespace py = pybind11;

// ============================================================================
// C++ Optimizer
// ============================================================================
class SGDOptimizer {
public:
    SGDOptimizer(std::vector<torch::Tensor> params, double lr, double momentum = 0.0)
        : params_(params), lr_(lr), momentum_(momentum) {
        if (momentum_ > 0) {
            for (const auto& p : params_) {
                velocity_buffers_.push_back(torch::zeros_like(p));
            }
        }
    }

    void zero_grad() {
        for (auto& p : params_) {
            if (p.grad().defined()) {
                p.grad().detach_();
                p.grad().zero_();
            }
        }
    }

    void step() {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < params_.size(); ++i) {
            auto& p = params_[i];
            if (!p.grad().defined()) {
                continue;
            }
            auto d_p = p.grad();

            if (momentum_ > 0) {
                auto& buf = velocity_buffers_[i];
                buf = buf.mul(momentum_).add(d_p);
                d_p = buf;
            }
            p.add_(d_p, -lr_);
        }
    }

private:
    std::vector<torch::Tensor> params_;
    double lr_;
    double momentum_;
    std::vector<torch::Tensor> velocity_buffers_;
};


// ============================================================================
// C++ Model Manager
// ============================================================================
class ModelManager {
public:
    ModelManager(const std::string& model_path) {
        try {
            module_ = torch::jit::load(model_path);
            module_.train();
            std::cout << "[C++] Model loaded from " << model_path << " into C++ memory." << std::endl;
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
    }

    float train_batch(const torch::Tensor& input_tensor, const torch::Tensor& target_tensor) {
        std::vector<torch::jit::IValue> inputs = {input_tensor};
        auto output = module_.forward(inputs).toTensor();
        auto target_long = target_tensor.to(torch::kLong);
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), target_long);
        loss.backward();
        return loss.item<float>();
    }

    // A method to get all the parameters from the C++ model
    std::vector<torch::Tensor> get_parameters() {
        // --- THE FIX IS HERE ---
        std::vector<torch::Tensor> params_vec;
        for (const auto& param : module_.parameters()) {
            params_vec.push_back(param);
        }
        return params_vec;
    }

    void save(const std::string& save_path) {
        module_.save(save_path);
        std::cout << "[C++] Model state saved to " << save_path << std::endl;
    }

private:
    torch::jit::script::Module module_;
};


// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================
PYBIND11_MODULE(xtorch_bridge, m) {
    m.doc() = "A more advanced C++ extension with stateful objects.";

    py::class_<SGDOptimizer>(m, "SGDOptimizer")
        .def(py::init<std::vector<torch::Tensor>, double, double>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum")=0.0)
        .def("step", &SGDOptimizer::step, "Performs a single optimization step.")
        .def("zero_grad", &SGDOptimizer::zero_grad, "Clears the gradients of all optimized tensors.");

    py::class_<ModelManager>(m, "ModelManager")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("train_batch", &ModelManager::train_batch,
             "Performs a forward and backward pass on a single batch.",
             py::arg("input"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_parameters", &ModelManager::get_parameters, "Returns a list of the model's parameters.")
        .def("save", &ModelManager::save, "Saves the current model state.", py::arg("save_path"));
}