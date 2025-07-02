#include <torch/extension.h>
#include <pybind11/stl.h>
#include "xt/models/BaseModel.h"
#include "xt/models/MLP.h"
#include "xt/train/Trainer.h"
#include "xt/dataloader/DataLoader.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<xt::models::BaseModel, std::shared_ptr<xt::models::BaseModel>>(m, "BaseModel");

    py::class_<xt::models::MLP, xt::models::BaseModel, std::shared_ptr<xt::models::MLP>>(m, "MLP")
        .def(py::init<int, int, int>());

    py::class_<xt::Trainer>(m, "Trainer")
        .def(py::init<>())
        .def("fit", [](xt::Trainer& self, std::shared_ptr<xt::models::BaseModel> model, py::object py_loader) {
            // Bind your xt::DataLoader here or simulate data loop from Python
            auto dataset = std::vector<std::pair<torch::Tensor, torch::Tensor>>();
            for (auto item : py_loader) {
                auto pair = item.cast<std::pair<torch::Tensor, torch::Tensor>>();
                dataset.push_back(pair);
            }

            xt::DataLoader<xt::Dataset> cpp_loader(dataset); // Assume you can construct from vector
            self.fit(model.get(), cpp_loader);
        });
}
