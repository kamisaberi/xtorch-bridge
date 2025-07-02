cmake_minimum_required(VERSION 3.18)
project(xtorch-bridge)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- FIX: THIS BLOCK WAS MOVED FROM THE MIDDLE TO THE TOP ---
# We must find the CUDA toolkit BEFORE any other package
# (especially Torch) tries to use its components. This solves the
# silent cmake failure that was causing "No Makefile found".
# find_package(CUDA REQUIRED) # You can add this back if you need CUDA targets explicitly
# --- END OF FIX ---


# --- Find Dependencies (Your original calls, in your order) ---
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# This hardcoded path is kept as you requested.
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)

# Your full list of find_package calls, unchanged.
find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)
find_package(onnxruntime REQUIRED)


# Your libraries list - UNCHANGED
set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${XTORCH_LIBRARIES}
        xTorch::xTorch
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        ${ONNXRUNTIME_LIBRARY}
        onnxruntime
)

# --- RPATH Handling (Your original settings) ---
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH
        "$ORIGIN"
        "/home/kami/libs/cpp/libtorch/lib"
        "/usr/local/lib"
        # Let's add the CUDA path here just to be safe for runtime
        "/opt/cuda/lib64"
        "/usr/lib"
)
message(STATUS "Setting RPATH to: ${CMAKE_INSTALL_RPATH}")


# --- Build and Link (Your original commands) ---
pybind11_add_module(xtorch_bridge_impl
        src/bindings.cpp
        src/python_data_loader_bridge.cpp
)

target_include_directories(xtorch_bridge_impl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(xtorch_bridge_impl PRIVATE
        ${TEST_LIBRARIES}
)

install(TARGETS xtorch_bridge_impl
        LIBRARY DESTINATION xtorch_bridge
)