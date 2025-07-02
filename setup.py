from setuptools import setup
from setuptools.command.build_ext import build_ext
import os
import subprocess

class CMakeBuild(build_ext):
    def run(self):
        libtorch_path = os.environ.get("LIBTORCH", "/path/to/libtorch")
        if not os.path.exists(libtorch_path):
            raise RuntimeError(f"LibTorch not found at {libtorch_path}")

        build_dir = "build"
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={libtorch_path}",
            f"-DPYTHON_EXECUTABLE={self.distribution.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        subprocess.check_call(["cmake", ".."] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir)

        # Copy the built module to the package directory
        for ext in self.extensions:
            module_path = os.path.join(build_dir, "torch_cpp", self.get_ext_filename(ext.name))
            self.copy_file(module_path, os.path.join("torch_cpp", self.get_ext_filename(ext.name)))

setup(
    name="torch_cpp",
    version="0.1.0",
    description="A simple Python package to run LibTorch C++ code",
    ext_modules=[Extension("torch_cpp", [])],  # Empty sources, handled by CMake
    cmdclass={"build_ext": CMakeBuild},
    packages=["torch_cpp"],
    package_dir={"torch_cpp": "torch_cpp"},
    install_requires=["pybind11>=2.12"],
)