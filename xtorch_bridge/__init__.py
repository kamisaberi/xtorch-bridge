# xtorch_bridge/__init__.py
print("--- Importing xtorch_bridge package ---")
try:
    # Import the C++ functions from the compiled .so file
    from .xtorch_bridge_impl import add_tensors
    print("--- C++ backend 'xtorch_bridge_impl' imported successfully! ---")
except ImportError as e:
    raise ImportError(
        "Could not import the C++ backend of xtorch_bridge. "
        "This is the critical failure point. Check the error below."
        f"\n--- Original error ---\n{e}"
    ) from e

# This controls what `from xtorch_bridge import *` does
__all__ = ["add_tensors"]