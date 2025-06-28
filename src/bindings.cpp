#include <pybind11/pybind11.h>
#include <xtorch/xtorch.h>
#include "xtorch_bridge/python_data_loader_bridge.h"

namespace py = pybind11;

/**
 * @brief The main function exposed to Python.
 *
 * This function acts as the C++ entry point. It receives Python objects,
 * wraps them in the appropriate C++ bridge classes, sets up the xt::Trainer,
 * releases the Python Global Interpreter Lock (GIL), and runs the C++
 * training loop.
 */
void fit_from_python(
    xt::Module& model,
    py::object train_loader_py,
    py::object val_loader_py,
    int max_epochs,
    double lr)
{
    std::cout << "[BRIDGE] Entered fit_from_python." << std::flush << std::endl;
    // 1. Create C++ bridge objects that wrap the Python DataLoaders
    PythonDataLoaderBridge train_loader_bridge(train_loader_py);
    std::cout << "[BRIDGE] Creating optimizer and trainer." << std::flush << std::endl;
    // Create an optional bridge for the validation loader
    std::optional<PythonDataLoaderBridge> val_loader_bridge;
    if (!val_loader_py.is_none()) {
        val_loader_bridge.emplace(val_loader_py);
    }

    // 2. Setup optimizer and trainer from your xtorch library
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lr));
    xt::Trainer trainer;
    trainer.set_max_epochs(max_epochs)
           .set_optimizer(optimizer)
           .set_loss_fn([](const auto& out, const auto& tgt) { return torch::nll_loss(out, tgt); });

    // 3. IMPORTANT: Release the Python GIL before running the C++ training loop
    py::gil_scoped_release release_gil;

    // 4. Call the templated fit method. The compiler automatically deduces the
    //    correct template arguments for PythonDataLoaderBridge.
    std::cout << "[BRIDGE] Calling trainer.fit()..." << std::flush << std::endl;
    trainer.fit(
        model,
        train_loader_bridge,
        val_loader_bridge.has_value() ? &(*val_loader_bridge) : nullptr, // Pass pointer or nullptr
        torch::kCPU // or torch::kCUDA
    );
    std::cout << "[BRIDGE] trainer.fit() returned." << std::flush << std::endl;
}

// =================================================================================
// PYBIND11 MODULE DEFINITION
// =================================================================================

PYBIND11_MODULE(xtorch_bridge_impl, m) {
    m.doc() = "A bridge to run xtorch C++ training from Python data loaders.";

    // Expose your C++ model classes to Python so they can be created and passed around.
    // They must be wrapped in a shared_ptr to be managed correctly by pybind11's memory model.
    // The second template argument (the "holder type") tells pybind11 how to manage the object.
    py::class_<xt::Module, std::shared_ptr<xt::Module>>(m, "Module")
        .doc() = "Base model class from the xtorch C++ library.";

    py::class_<xt::models::LeNet5, xt::Module, std::shared_ptr<xt::models::LeNet5>>(m, "LeNet5")
        .def(py::init<int>(), py::arg("num_classes"))
        .doc() = "LeNet5 model implemented in C++.";

    // Expose other models here...
    // py::class_<xt::models::ResNet, ... >(m, "ResNet").def(...);

    // Expose the main fit function, with keyword arguments for clarity in Python.
    m.def("fit", &fit_from_python, "Train a C++ model using Python DataLoaders",
          py::arg("model"),
          py::arg("train_loader"),
          py::arg("val_loader"),
          py::arg("max_epochs"),
          py::arg("lr")
    );
}