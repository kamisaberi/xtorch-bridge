from setuptools import setup, find_packages

# This setup.py is a shim to allow scikit-build-core to handle the build.
# All configuration is in pyproject.toml and CMakeLists.txt.
setup(
    # Although most config is in pyproject.toml, we can create a placeholder
    # package directory here that scikit-build will place the C++ module into.
    packages=["xtorch_bridge"]
)