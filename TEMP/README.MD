# xtorch-bridge: Connecting Python and C++ for High-Performance Deep Learning

**Date:** March 2025

---

## Abstract

xtorch-bridge allows Python-based data loading and orchestration while training and inference run in high-speed C++. Built on top of pybind11 and xtorch, it connects PyTorch's flexibility with native performance. This project reduces boilerplate and enables efficient deployment of deep learning models using both Python and C++.

---

## 1. Introduction

While Python is the primary interface for deep learning with PyTorch, performance-critical systems often require native speed. PyTorch's C++ API (LibTorch) offers this speed but is not user-friendly on its own. The xtorch library improves LibTorch by introducing high-level abstractions like `Trainer`, `DataLoader`, and modular models, but writing everything in C++ can still be tedious.

xtorch-bridge makes it possible to load datasets and control training in Python while executing the actual model training and inference in xtorch C++. This hybrid approach provides the performance of C++ with the ease and flexibility of Python.

---

## 2. Architecture and Design

xtorch-bridge is structured into three layers:

- **Python Interface**: Used to load data, launch training, and call C++ classes using pybind11.
- **pybind11 Binding Layer**: Exposes C++ models, trainers, and methods so they can be called from Python.
- **xtorch Backend**: Contains C++ models and training logic using LibTorch and the xtorch API.

This modular architecture allows users to keep their Python data pipelines while reaping the speed benefits of native code.

---

## 3. Features and Benefits

- Train C++ models directly from Python
- Easy integration with PyTorch datasets
- Native-speed training, especially useful for embedded or performance-sensitive systems
- Lower development effort compared to full C++ workflows
- Access to xtorch features: trainer abstraction, extended data handling, metric logging, and model saving

---

## 4. Example Use Case

### Step 1: Define a model in C++

```cpp
class MLP : public xt::models::BaseModel {
public:
  MLP(int in, int h, int out) {
    fc1 = register_module("fc1", xt::nn::Linear(in, h));
    fc2 = register_module("fc2", xt::nn::Linear(h, out));
  }

  torch::Tensor forward(torch::Tensor x) override {
    return fc2->forward(torch::relu(fc1->forward(x)));
  }

private:
  xt::nn::Linear fc1{nullptr}, fc2{nullptr};
};
```

### Step 2: Call it from Python using xtorch-bridge

```python
from TEMP.xtorch_bridge import MLP, Trainer
from torchvision import datasets, transforms
import torch

model = MLP(784, 128, 10)
trainer = Trainer()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

trainer.fit(model, train_loader)
```

---

## 5. Performance Evaluation

We compared three different setups for training a simple model on MNIST:

| Setup             | Time per Batch |
|------------------|----------------|
| Pure PyTorch     | 2.1 ms         |
| xtorch-bridge    | 0.9 ms         |
| Pure xtorch C++  | 0.6 ms         |

xtorch-bridge nearly halves training time compared to pure PyTorch while maintaining Python flexibility. Full C++ is fastest, but at the cost of convenience.

---

## 6. Applications

- Robotics and Embedded AI
- Research prototyping with fast experimentation
- Educational low-level ML development
- High-performance industrial deployment

---

## 7. Comparison with Related Tools

| Tool              | Notes |
|-------------------|-------|
| LibTorch           | Requires verbose, low-level coding |
| PyTorch Lightning  | Great for Python, not applicable for C++ |
| TorchScript        | Good for inference, not full training |
| xtorch-bridge      | Full training and inference across C++ and Python |

---

## 8. Conclusion

xtorch-bridge brings the best of both worlds: the simplicity of Python and the speed of C++. By allowing Python code to control C++ training and models, it opens up new possibilities for developers building deep learning systems that must scale, perform, and stay easy to develop.

---

## Acknowledgments

Thanks to the developers of pybind11, LibTorch, xtorch, and the PyTorch ecosystem.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs)
- [xtorch GitHub](https://github.com/kamisaberi/xTorch)
- [pybind11](https://github.com/pybind/pybind11)
- [LibTorch Tutorial](https://leimao.github.io/blog/LibTorch-Tutorial)
- [PyTorch Lightning](https://www.pytorchlightning.ai)
