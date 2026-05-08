set(GLUED_CUDA_LIBRARY_NAME OpenMMGluedCUDA)

set(SOURCE_FILES
    src/CudaGluedKernelFactory.cpp
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonGluedKernels.cpp
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonCVKernels.cpp
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonBiasKernels.cpp
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonSerialization.cpp
)

add_library(${GLUED_CUDA_LIBRARY_NAME} SHARED ${SOURCE_FILES})

# Pull in CUDA toolkit headers (cuda.h, cuda_runtime.h) so that OpenMM's
# CudaContext.h resolves without needing CPATH=/usr/local/cuda/include.
find_package(CUDAToolkit REQUIRED)

target_compile_definitions(${GLUED_CUDA_LIBRARY_NAME} PRIVATE
    GLUED_BUILDING_SHARED_LIBRARY)

target_include_directories(${GLUED_CUDA_LIBRARY_NAME} PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/platforms/common/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include/internal
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/common"
    "${OPENMM_DIR}/include/openmm/cuda"
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Locate the OpenMM CUDA platform library in a platform-agnostic way.
# On Linux/macOS: libOpenMMCUDA.so  On Windows: OpenMMCUDA.lib/.dll
find_library(OPENMM_CUDA_LIB
    NAMES OpenMMCUDA
    HINTS "${OPENMM_DIR}/lib/plugins" "${OPENMM_DIR}/lib"
    NO_DEFAULT_PATH)
if(NOT OPENMM_CUDA_LIB)
    message(FATAL_ERROR "OpenMMCUDA library not found under ${OPENMM_DIR}. "
            "Ensure OpenMM is installed with CUDA platform support.")
endif()

target_link_libraries(${GLUED_CUDA_LIBRARY_NAME}
    OpenMMGlued
    OpenMM
    ${OPENMM_CUDA_LIB})

install(TARGETS ${GLUED_CUDA_LIBRARY_NAME}
    LIBRARY  DESTINATION lib/plugins
    RUNTIME  DESTINATION lib/plugins
    ARCHIVE  DESTINATION lib/plugins)

if(GLUED_HAS_TORCH)
    target_link_libraries(${GLUED_CUDA_LIBRARY_NAME} ${GLUED_TORCH_LIBS})
    target_include_directories(${GLUED_CUDA_LIBRARY_NAME} PRIVATE ${GLUED_TORCH_INCLUDES})
    target_compile_definitions(${GLUED_CUDA_LIBRARY_NAME} PRIVATE GLUED_HAS_TORCH)
    # Suppress torch's many deprecation warnings
    target_compile_options(${GLUED_CUDA_LIBRARY_NAME} PRIVATE -Wno-deprecated-declarations)
endif()
