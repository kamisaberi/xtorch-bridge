# xtorch_bridge/loader.py
from torch.utils.cpp_extension import load

def load_xtorch_system():
    return load(
        name="xtorch_native",
        sources=["cpp/xtorch_bridge_wrapper.cpp", "xt/train/Trainer.cpp", "xt/models/MLP.cpp"],
        include_dirs=["xt"],
        extra_cflags=["-O3"],
        verbose=True
    )