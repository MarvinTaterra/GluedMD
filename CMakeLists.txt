cmake_minimum_required(VERSION 3.17)
project(GLUED VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where OpenMM is installed.
# Auto-detected from $CONDA_PREFIX when a conda env is active:
#   Linux/WSL:  $CONDA_PREFIX           (e.g. ~/miniconda3/envs/openmm_env)
#   Windows:    $CONDA_PREFIX/Library
# Override at cmake time: cmake -DOPENMM_DIR=/path/to/openmm ...
if(DEFINED ENV{CONDA_PREFIX})
    if(WIN32)
        set(_openmm_default "$ENV{CONDA_PREFIX}/Library")
    else()
        set(_openmm_default "$ENV{CONDA_PREFIX}")
    endif()
else()
    set(_openmm_default "/usr/local/openmm")
endif()
set(OPENMM_DIR "${_openmm_default}"
    CACHE PATH "OpenMM install prefix (contains include/ and lib/)")

# IMPORTANT: both paths are required — the second is the common-layer headers
# used by GPU kernel implementations (ComputeContext.h, ComputeArray.h, etc.)
include_directories(
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/common"
)
link_directories(
    "${OPENMM_DIR}/lib"
    "${OPENMM_DIR}/lib/plugins"
)

set(GLUED_VERSION "${PROJECT_VERSION}")

# macOS rpath
if(APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
endif()

# Install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH
            "Where to install the plugin" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH
            "Where to install the plugin" FORCE)
    endif()
endif()

# ---- Core library (openmmapi + serialization) --------------------------------
set(GLUED_LIBRARY_NAME OpenMMGlued)

set(API_INCLUDE_DIRS
    "openmmapi/include"
    "openmmapi/include/internal"
)

set(API_INCLUDE_FILES)
foreach(dir ${API_INCLUDE_DIRS})
    file(GLOB _h ${dir}/*.h)
    list(APPEND API_INCLUDE_FILES ${_h})
endforeach()

set(SOURCE_SUBDIRS openmmapi serialization)
set(SOURCE_FILES)
foreach(subdir ${SOURCE_SUBDIRS})
    file(GLOB _src "${subdir}/src/*.cpp")
    list(APPEND SOURCE_FILES ${_src})
    include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include")
endforeach()

add_library(${GLUED_LIBRARY_NAME} SHARED
    ${SOURCE_FILES} ${API_INCLUDE_FILES})

target_compile_definitions(${GLUED_LIBRARY_NAME} PRIVATE
    GLUED_BUILDING_SHARED_LIBRARY)

target_include_directories(${GLUED_LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include/internal>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/serialization/include>
)

target_link_libraries(${GLUED_LIBRARY_NAME} OpenMM)

install(TARGETS ${GLUED_LIBRARY_NAME}
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION lib   # .dll goes to lib on Windows
    ARCHIVE  DESTINATION lib)

# Install public headers
install(FILES ${API_INCLUDE_FILES}
    DESTINATION include)

# ---- Platform plugins --------------------------------------------------------
add_subdirectory(platforms/reference)

# Optional GPU platforms — only add if CUDA / OpenCL toolkits are found
find_package(CUDAToolkit QUIET)

# Optional libtorch for Stage 3.8 PyTorch CVs.
# Discovery runs BEFORE add_subdirectory(platforms/cuda) so that the imported
# targets created by find_package(Torch) (torch, torch_library, …) exist by
# the time the CUDA CMakeLists records them as link dependencies.
# The conda-installed libtorch config uses the deprecated FindCUDA module
# internally; pre-setting CUDA_TOOLKIT_ROOT_DIR satisfies it.
if(CUDAToolkit_FOUND)
    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR OR CUDA_TOOLKIT_ROOT_DIR STREQUAL "")
        foreach(_cuda_root /usr/local/cuda /usr/local/cuda-13.1 /usr/local/cuda-13.0
                           /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.0)
            if(EXISTS "${_cuda_root}/include/cuda.h")
                set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_root}" CACHE PATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()
    find_package(Torch QUIET HINTS
        "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/torch/share/cmake/Torch"
        "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/torch/share/cmake/Torch"
        "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/torch/share/cmake/Torch"
        "$ENV{CONDA_PREFIX}/share/cmake/Torch")
    if(Torch_FOUND)
        message(STATUS "Found libtorch ${Torch_VERSION} — PyTorch CVs enabled (CUDA platform)")
        set(GLUED_HAS_TORCH TRUE CACHE INTERNAL "")
        set(GLUED_TORCH_INCLUDES "${TORCH_INCLUDE_DIRS}" CACHE INTERNAL "")
        # Store the torch imported target name; it is defined at this point so
        # the CUDA subdirectory (processed immediately below) can link against it.
        set(GLUED_TORCH_LIBS "torch" CACHE INTERNAL "")
    else()
        message(STATUS "libtorch not found — PyTorch CVs disabled")
        set(GLUED_HAS_TORCH FALSE CACHE INTERNAL "")
        set(GLUED_TORCH_LIBS "" CACHE INTERNAL "")
    endif()
else()
    message(STATUS "CUDA toolkit absent — skipping libtorch discovery (PyTorch CVs disabled)")
    set(GLUED_HAS_TORCH FALSE CACHE INTERNAL "")
    set(GLUED_TORCH_LIBS "" CACHE INTERNAL "")
endif()

if(CUDAToolkit_FOUND)
    add_subdirectory(platforms/cuda)
else()
    message(STATUS "CUDA toolkit not found — skipping GPU CUDA platform")
endif()

# OpenCL: Apple deprecated OpenCL in 10.14 and Apple Silicon support is
# unreliable (loading libOpenMMOpenCL.dylib crashes on macOS CI runners).
# Skip OpenCL on Apple by default; users who want it can override with
# -DGLUED_BUILD_OPENCL=ON.
if(APPLE)
    option(GLUED_BUILD_OPENCL "Build the OpenCL platform plugin" OFF)
else()
    option(GLUED_BUILD_OPENCL "Build the OpenCL platform plugin" ON)
endif()

if(GLUED_BUILD_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND
            OR EXISTS "${OPENMM_DIR}/lib/plugins/libOpenMMOpenCL.so"
            OR EXISTS "${OPENMM_DIR}/lib/plugins/libOpenMMOpenCL.dylib"
            OR EXISTS "${OPENMM_DIR}/lib/plugins/OpenMMOpenCL.lib")
        add_subdirectory(platforms/opencl)
    else()
        message(STATUS "OpenCL not found — skipping GPU OpenCL platform")
    endif()
else()
    message(STATUS "GLUED_BUILD_OPENCL=OFF — skipping GPU OpenCL platform")
endif()

# ---- Python wrapper ----------------------------------------------------------
find_program(PYTHON_EXECUTABLE python)
find_program(SWIG_EXECUTABLE   swig)

if(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    option(GLUED_BUILD_PYTHON_WRAPPERS "Build Python wrappers" ON)
else()
    option(GLUED_BUILD_PYTHON_WRAPPERS "Build Python wrappers" OFF)
endif()

if(GLUED_BUILD_PYTHON_WRAPPERS)
    add_subdirectory(python)
endif()

# ---- Tests -------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
