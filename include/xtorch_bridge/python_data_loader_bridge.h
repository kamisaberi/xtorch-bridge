#pragma once

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <vector>
#include <mutex>

namespace py = pybind11;

/**
 * @class PythonDataLoaderBridge
 * @brief Mimics the public interface of xt::dataloaders::ExtendedDataLoader.
 *
 * This class acts as a C++ wrapper around a Python `torch.utils.data.DataLoader`
 * object. It implements the necessary methods (`begin()`, `end()`, `next_batch()`, etc.)
 * so that it can be used as a drop-in replacement in the templated `xt::Trainer::fit`
 * method. It handles all the Python GIL management internally.
 */
class PythonDataLoaderBridge {
public:
    using BatchData = std::pair<torch::Tensor, torch::Tensor>;

    // Constructor takes the Python DataLoader object.
    explicit PythonDataLoaderBridge(py::object py_loader);
    ~PythonDataLoaderBridge();

    // These methods provide the "duck-typed" interface that xt::Trainer::fit expects.
    void reset_epoch();
    std::optional<BatchData> next_batch();

public: // Iterator support, mirroring xt::dataloaders::ExtendedDataLoader
    class Iterator {
    public:
        Iterator(PythonDataLoaderBridge* loader, bool end = false);
        const BatchData& operator*() const;
        BatchData& operator*();
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
    private:
        PythonDataLoaderBridge* loader_;
        bool is_end_;
        std::optional<BatchData> current_batch_opt_;
    };

    Iterator begin();
    Iterator end();

private:
    py::object loader_py_;      // The Python DataLoader object.
    py::object iterator_py_;    // The current Python iterator object.
    std::mutex py_mutex_;       // Mutex to protect access to py::objects if ever needed.
};