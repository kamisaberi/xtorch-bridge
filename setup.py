import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# A custom class to handle our CMake extension
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

# A custom build command to run CMake
class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # The directory where the final .so file should be placed
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.resolve())

        # Check PyTorch's C++ ABI to ensure compatibility
        try:
            import torch
            torch_cxx11_abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        except ImportError:
            print("Warning: Could not import torch to check ABI. Defaulting to new ABI (1).")
            torch_cxx11_abi = "1"

        # Define the paths to your libraries
        libtorch_path = "/home/kami/libs/cpp/libtorch"
        xtorch_path = "/usr/local"  # Where your xTorch library is installed
        cmake_prefix_path = f"{libtorch_path};{xtorch_path}"

        print(f"--- Found PyTorch CXX11 ABI: {torch_cxx11_abi}")
        print(f"--- Using CMAKE_PREFIX_PATH: {cmake_prefix_path}")

        # Arguments to pass to the cmake command
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}",
        ]

        # RPATH setting to bake library paths into the .so file.
        # This makes LD_LIBRARY_PATH unnecessary at runtime.
        rpath = f"{libtorch_path}/lib:{xtorch_path}/lib:/opt/cuda/lib64"
        cmake_args.append(f"-DCMAKE_INSTALL_RPATH={rpath}")

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure and build the C++ extension
        print("-" * 10, "Configuring CMake", "-" * 10)
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )

        print("-" * 10, "Building extension", "-" * 10)
        subprocess.run(
            ["cmake", "--build", ".", "--", "-j8"], cwd=build_temp, check=True
        )

# Main setup() call
setup(
    name="xtorch_bridge",
    version="1.0.1", # Incremented version
    author="Kami",
    description="A C++ bridge for xtorch training with Python DataLoaders.",
    packages=['xtorch_bridge'],
    ext_modules=[CMakeExtension("xtorch_bridge.xtorch_bridge_impl")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.11",
)