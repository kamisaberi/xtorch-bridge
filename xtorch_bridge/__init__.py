# This file makes 'xtorch_bridge' a Python package.

# When the user does 'import xtorch_bridge', this file is executed.
# We import the contents of our C++ module into this package's namespace.
# The C++ module will be built and placed in this directory by the build process.

try:
    # The name of the C++ module target in CMakeLists.txt
    from .xtorch_bridge_impl import fit, LeNet5, Module
except ImportError as e:
    # Provide a more helpful error message if the C++ module failed to build or import.
    raise ImportError(
        "Could not import the C++ backend of xtorch_bridge. "
        "Please check that the library was compiled correctly. "
        f"Original error: {e}"
    ) from e


# This controls what `from xtorch_bridge import *` does and helps tools like linters.
__all__ = ["fit", "LeNet5", "Module"]