from setuptools import setup, find_packages

# This setup.py assumes that the C++ extension has already been built
# by running the `build.sh` script, and the resulting .so file has
# been copied into the `xtorch_bridge` package directory.

setup(
    name="xtorch_bridge",
    version="0.1.0",
    author="Kami",
    description="A C++ bridge for xtorch training with Python DataLoaders.",
    packages=find_packages(),
    # This tells setuptools to include non-Python files (like .so)
    # found inside the package directory.
    include_package_data=True,
    # We explicitly list the package data to be sure.
    package_data={
        # "package_name": ["file_pattern"]
        "xtorch_bridge": ["*.so"],
    },
    zip_safe=False,
    python_requires=">=3.8",
)