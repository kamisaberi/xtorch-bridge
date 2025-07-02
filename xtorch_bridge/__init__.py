# xtorch_bridge/__init__.py

try:
    # Import the C++ functions and classes from the compiled .so file
    from .xtorch_bridge_impl import fit, LeNet5, Module
    print("xtorch_bridge C++ backend imported successfully!")
except ImportError as e:
    raise ImportError(
        "Could not import the C++ backend of xtorch_bridge. "
        "Please ensure the package was built and installed correctly. "
        f"Original error: {e}"
    ) from e

# This controls what `from xtorch_bridge import *` does
__all__ = ["fit", "LeNet5", "Module"]