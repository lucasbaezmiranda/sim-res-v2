cmake_minimum_required(VERSION 3.15)
project(heat_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies -----------------------------------------------------------
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# --- Extension module -------------------------------------------------------
pybind11_add_module(heat_core solver.cpp)

target_link_libraries(heat_core PRIVATE Eigen3::Eigen)

target_compile_options(heat_core PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
)

# --- Info -------------------------------------------------------------------
message(STATUS "Python  : ${Python_VERSION} @ ${Python_EXECUTABLE}")
message(STATUS "pybind11: ${pybind11_VERSION}")
message(STATUS "Eigen3  : ${EIGEN3_VERSION_STRING}")
