#include "xtorch_bridge/python_data_loader_bridge.h"
#include <pybind11/stl.h> // Required for casting Python lists/tuples

PythonDataLoaderBridge::PythonDataLoaderBridge(py::object py_loader)
    : loader_py_(py_loader) {
    // The first call to begin() will perform the initial reset.
}

PythonDataLoaderBridge::~PythonDataLoaderBridge() {
    // The py::object destructors will automatically handle reference counting.
    // Explicitly releasing can be good practice to control destruction order.
    std::lock_guard<std::mutex> lock(py_mutex_);
    iterator_py_.release();
    loader_py_.release();
}

void PythonDataLoaderBridge::reset_epoch() {
    std::lock_guard<std::mutex> lock(py_mutex_);
    // To interact with Python objects, we must hold the GIL.
    py::gil_scoped_acquire acquire;
    try {
        iterator_py_ = loader_py_.attr("__iter__")();
    } catch (const py::error_already_set& e) {
        // Propagate Python exceptions back up.
        throw;
    }
}

std::optional<PythonDataLoaderBridge::BatchData> PythonDataLoaderBridge::next_batch() {
    std::lock_guard<std::mutex> lock(py_mutex_);
    // Acquire the Python GIL to safely interact with Python objects.
    py::gil_scoped_acquire acquire;
    try {
        // Call the Python iterator's __next__() method.
        py::object batch_py = iterator_py_.attr("__next__")();

        // Cast the Python batch (e.g., a list or tuple of tensors) to C++ types.
        // This assumes the Python loader yields a list/tuple like [data, target].
        auto py_list = batch_py.cast<py::list>();
        if (py_list.size() != 2) {
            throw std::runtime_error("Python DataLoader must yield a list/tuple of 2 Tensors (data, target).");
        }

        torch::Tensor data = py_list[0].cast<torch::Tensor>();
        torch::Tensor target = py_list[1].cast<torch::Tensor>();

        return {{data, target}};

    } catch (const py::error_already_set& e) {
        // Python's `for` loop stops when a `StopIteration` exception is raised.
        // We catch it and return an empty optional to signal the end of the epoch.
        if (e.matches(PyExc_StopIteration)) {
            return std::nullopt;
        }
        // If it was some other Python error, re-throw it.
        throw;
    }
}


// --- Implementation of the Iterator (mostly boilerplate) ---

PythonDataLoaderBridge::Iterator::Iterator(PythonDataLoaderBridge* loader, bool end)
    : loader_(loader), is_end_(end) {
    if (loader_ && !is_end_) {
        // Prime the first batch when the iterator is created (for begin()).
        current_batch_opt_ = loader_->next_batch();
        if (!current_batch_opt_) {
            is_end_ = true; // Handle empty dataloader.
        }
    }
}

const PythonDataLoaderBridge::BatchData& PythonDataLoaderBridge::Iterator::operator*() const {
    if (!current_batch_opt_) throw std::runtime_error("Attempting to dereference an end iterator.");
    return *current_batch_opt_;
}

PythonDataLoaderBridge::BatchData& PythonDataLoaderBridge::Iterator::operator*() {
    if (!current_batch_opt_) throw std::runtime_error("Attempting to dereference an end iterator.");
    return *current_batch_opt_;
}

PythonDataLoaderBridge::Iterator& PythonDataLoaderBridge::Iterator::operator++() {
    if (loader_ && !is_end_) {
        current_batch_opt_ = loader_->next_batch();
        if (!current_batch_opt_) is_end_ = true; // Reached the end.
    } else {
        is_end_ = true;
    }
    return *this;
}

bool PythonDataLoaderBridge::Iterator::operator!=(const Iterator& other) const {
    // Two iterators are not equal if one is an end iterator and the other is not.
    return is_end_ != other.is_end_;
}

PythonDataLoaderBridge::Iterator PythonDataLoaderBridge::begin() {
    reset_epoch(); // Prepare for a new iteration.
    return Iterator(this, false);
}

PythonDataLoaderBridge::Iterator PythonDataLoaderBridge::end() {
    return Iterator(this, true); // Represents the end.
}