from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

class CMakeBuild(build_ext):
    def run(self):
        libtorch_path = "/home/kami/libs/cpp/libtorch"
        if not os.path.exists(libtorch_path):
            raise RuntimeError(f"LibTorch not found at {libtorch_path}")

        build_dir = "build"
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={libtorch_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Try to get pybind11 CMake directory
        try:
            pybind11_cmake_dir = subprocess.check_output(
                [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
                text=True
            ).strip()
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
        except subprocess.CalledProcessError:
            print("Warning: Could not find pybind11 CMake directory via Python. Relying on CMake to find pybind11.")

        try:
            subprocess.check_call(["cmake", ".."] + cmake_args, cwd=build_dir)
            subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake failed: {e}")

        # Copy the built module to the package directory
        for ext in self.extensions:
            module_path = os.path.join(build_dir, self.get_ext_filename(ext.name))  # Changed to build/ directory
            self.copy_file(module_path, os.path.join("xtorch_bridge", self.get_ext_filename(ext.name)))

setup(
    name="xtorch_bridge",
    version="0.1.0",
    description="A simple Python package to run LibTorch C++ code",
    ext_modules=[Extension("xtorch_bridge", [])],  # Empty sources, handled by CMake
    cmdclass={"build_ext": CMakeBuild},
    packages=["xtorch_bridge"],
    package_dir={"xtorch_bridge": "xtorch_bridge"},
    install_requires=["pybind11>=2.12"],
)