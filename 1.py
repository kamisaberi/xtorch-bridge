import pybind11
import subprocess
import sys
print(pybind11.get_cmake_dir())

pybind11_cmake_dir = subprocess.check_output(
    [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
    text=True
).strip()
