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

# A custom build command that runs CMake
class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.resolve())

        # Check that we have the LIBTORCH_PATH environment variable
        libtorch_path = os.environ.get("LIBTORCH_PATH")
        if not libtorch_path:
            raise RuntimeError("The LIBTORCH_PATH environment variable must be set to your C++ LibTorch directory.")

        # Find pybind11's CMake directory
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        # Check PyTorch's C++ ABI
        try:
            import torch
            abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        except ImportError:
            print("Warning: Could not import torch. Defaulting to CXX11 ABI ON (1).")
            abi = "1"

        # Construct CMAKE_PREFIX_PATH for find_package
        cmake_prefix_path = f"{libtorch_path};/usr/local;{pybind11_cmake_dir}"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ]

        # This is the robust way to set the RPATH for the final library.
        rpath = f"{libtorch_path}/lib:/usr/local/lib"
        cmake_args.append(f"-DCMAKE_INSTALL_RPATH={rpath}")

        build_args = ["--", f"-j{os.cpu_count() or 1}"]
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake and Make
        subprocess.run(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", "."] + build_args, cwd=build_temp, check=True)

# Main setup call
setup(
    name="xtorch_bridge",
    version="4.0.0", # I am so sorry.
    author="Kami",
    description="A C++ bridge for xtorch training with Python DataLoaders.",
    packages=['xtorch_bridge'],
    ext_modules=[CMakeExtension("xtorch_bridge.xtorch_bridge_impl", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=['torch'],
    setup_requires=['pybind11>=2.11'], # pybind11 is needed during setup
    python_requires=">=3.11,<3.12",
)