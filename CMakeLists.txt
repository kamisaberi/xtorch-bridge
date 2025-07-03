cmake_minimum_required(VERSION 3.18)
project(xtorch_bridge LANGUAGES CXX)

#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
    # Fallback: Try to find pybind11 in the Python virtual environment
    execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()
if (NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Please install it or set pybind11_DIR.")
endif()

# Find LibTorch
set(LIBTORCH_PATH "/home/kami/libs/cpp/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define the extension module
pybind11_add_module(xtorch_bridge
        xtorch_bridge/xtorch_bridge.cpp
)

# Link against LibTorch and Python
target_link_libraries(xtorch_bridge PRIVATE "${TORCH_LIBRARIES}")

# Set C++ standard
set_target_properties(xtorch_bridge PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Install the module to the xtorch_bridge package directory
install(TARGETS xtorch_bridge
        LIBRARY DESTINATION xtorch_bridge
)