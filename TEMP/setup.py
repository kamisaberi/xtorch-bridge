import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.resolve())
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        try:
            import torch
            abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
            print(f"--- Building with PyTorch CXX11 ABI: {abi} ---")
        except ImportError:
            abi = "1"
        libtorch_path = os.environ.get("LIBTORCH_PATH")
        if not libtorch_path:
            raise RuntimeError("The LIBTORCH_PATH environment variable must be set.")
        cmake_prefix_path = f"{libtorch_path};{pybind11_cmake_dir}"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ]
        rpath = f"{libtorch_path}/lib"
        cmake_args.append(f"-DCMAKE_INSTALL_RPATH={rpath}")
        build_args = ["--", f"-j{os.cpu_count() or 1}"]
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", "."] + build_args, cwd=build_temp, check=True)

setup(
    name="xtorch_bridge",
    version="1.0.0", # <-- THE FIX IS HERE
    author="Kami",
    description="A simple LibTorch + pybind11 example in the xtorch_bridge structure.",
    packages=['xtorch_bridge'],
    ext_modules=[CMakeExtension("xtorch_bridge.xtorch_bridge_impl", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=['torch'],
    setup_requires=['pybind11>=2.11'],
    python_requires=">=3.11,<3.12",
)