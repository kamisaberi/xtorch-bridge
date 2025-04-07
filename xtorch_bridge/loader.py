from torch.utils.cpp_extension import load
import os

def load_xtorch_extension():
    module = load(
        name="xtorch_native",
        sources=["cpp/xtorch_wrapper.cpp"],  # add more files here if needed
        extra_cflags=["-O3"],
        verbose=True
    )
    return module