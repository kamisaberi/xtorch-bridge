import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# A CMakeExtension needs a sourcedir instead of a file list.
# The name of the Extension must be the same as the name passed to
# pybind11_add_module in CMake.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in setuptools
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # We need to pass the CMAKE_PREFIX_PATH to the cmake command
        # You can add more paths here, separated by semicolons
        cmake_prefix_path = os.environ.get("CMAKE_PREFIX_PATH", "")
        if not cmake_prefix_path:
            # A default for libtorch, if the env var is not set. Adjust if needed.
            libtorch_path = "/home/kami/libs/cpp/libtorch"
            print(f"Warning: CMAKE_PREFIX_PATH not set. Defaulting to {libtorch_path}")
            cmake_prefix_path = libtorch_path

        # We also pass the ABI flag directly
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-D_GLIBCXX_USE_CXX11_ABI=1", # Force the ABI
        ]

        # Build arguments
        build_args = []

        # We can also add build configuration here
        cfg = "Debug" if self.debug else "Release"
        build_args += ["--config", cfg]
        build_args += ["--", "-j8"] # Use 8 cores for building

        # Now, we run the cmake commands that we know work
        print("--- Running CMake configuration ---")
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, check=True
        )
        print("--- Running CMake build ---")
        subprocess.run(
            ["cmake", "--build", "."] + build_args, check=True
        )

# The name of the package as it will be on PyPI
setup(
    name="xtorch_bridge",
    version="0.1.0",
    author="Kami",
    description="A C++ bridge for xtorch training with Python DataLoaders.",
    long_description="",
    # Tell setuptools about our C++ extension
    ext_modules=[CMakeExtension("xtorch_bridge.xtorch_bridge_impl")],
    # Tell setuptools to run our custom build command
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    packages=['xtorch_bridge'], # Find the package directory
)