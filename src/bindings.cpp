#include <pybind11/pybind11.h>
#include <xtorch/xtorch.h> // Includes your trainer, models, etc.
#include "xtorch_bridge/python_data_loader_bridge.h"

namespace py = pybind11;

/**
 * @brief The C++ entry point called from Python.
 * @param model A shared_ptr to the xt::Module. This is the safe way to accept objects managed by pybind11.
 */
void fit_from_python(
    std::shared_ptr<xt::Module> model, // <-- FIX #1: Accept the shared_ptr directly
    py::object train_loader_py,
    py::object val_loader_py,
    int max_epochs,
    double lr)
{
	//THIS_IS_A_DELIBERATE_SYNTAX_ERROR_TO_TEST_THE_BUILD;
    // 1. Create C++ bridge objects that wrap the Python DataLoaders
    PythonDataLoaderBridge train_loader_bridge(train_loader_py);

    std::optional<PythonDataLoaderBridge> val_loader_bridge;
    if (!val_loader_py.is_none()) {
        val_loader_bridge.emplace(val_loader_py);
    }

    // 2. Setup optimizer and trainer. Note the '->' to access members of the pointed-to object.
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
    xt::Trainer trainer;
    trainer.set_max_epochs(max_epochs)
           .set_optimizer(optimizer)
           .set_loss_fn([](const auto& out, const auto& tgt) { return torch::nll_loss(out, tgt); });

    py::gil_scoped_release release_gil;

    // 3. Call the C++ trainer's fit method.
    // We now pass the shared_ptr directly if the trainer expects it, or dereference it (*model)
    // if the trainer still expects a reference. Let's assume the trainer expects a reference.
    trainer.fit(
        *model, // <-- Dereference the pointer to get the xt::Module& the trainer wants
        train_loader_bridge,
        val_loader_bridge.has_value() ? &(*val_loader_bridge) : nullptr,
        torch::kCPU
    );
}

// =================================================================================
// PYBIND11 MODULE DEFINITION
// =================================================================================

// FIX #2: The module name here MUST match the target name in CMakeLists.txt
PYBIND11_MODULE(xtorch_bridge_impl, module) {
    module.doc() = "A C++ bridge for xtorch training with Python DataLoaders.";

    // Define the Module class, managed by shared_ptr
    py::class_<xt::Module, std::shared_ptr<xt::Module>>(module, "Module")
        .doc() = "Base model class from the xtorch C++ library.";

    // Define the LeNet5 class, inheriting from Module, managed by shared_ptr
    py::class_<xt::models::LeNet5, xt::Module, std::shared_ptr<xt::models::LeNet5>>(module, "LeNet5")
        .def(py::init<int>(), py::arg("num_classes"))
        .doc() = "LeNet5 model implemented in C++.";

    // Expose the fit function. Pybind11 will handle the argument conversion.
    module.def("fit", &fit_from_python, "Train a C++ model using Python DataLoaders",
          py::arg("model"),
          py::arg("train_loader"),
          py::arg("val_loader"),
          py::arg("max_epochs"),
          py::arg("lr")
    );
}