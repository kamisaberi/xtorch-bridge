# xtorch_bridge/__init__.py
try:
    # Import our new C++ classes
    from .xtorch_bridge import ModelManager, SGDOptimizer
    print("xtorch_bridge C++ backend imported successfully!")
except ImportError as e:
    raise ImportError(
        "Could not import the C++ backend of xtorch_bridge. "
        f"Original error:\n{e}"
    ) from e

__all__ = ["ModelManager", "SGDOptimizer"]