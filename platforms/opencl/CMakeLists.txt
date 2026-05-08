set(GLUED_OPENCL_LIBRARY_NAME OpenMMGluedOpenCL)

set(SOURCE_FILES
    src/OpenCLGluedKernelFactory.cpp
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonGluedKernels.cpp
)

add_library(${GLUED_OPENCL_LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_compile_definitions(${GLUED_OPENCL_LIBRARY_NAME} PRIVATE
    GLUED_BUILDING_SHARED_LIBRARY)

target_include_directories(${GLUED_OPENCL_LIBRARY_NAME} PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/platforms/common/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include/internal
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/common"
    "${OPENMM_DIR}/include/openmm/opencl"
)

# Locate the OpenMM OpenCL platform library in a platform-agnostic way.
# On Linux/macOS: libOpenMMOpenCL.so  On Windows: OpenMMOpenCL.lib/.dll
find_library(OPENMM_OPENCL_LIB
    NAMES OpenMMOpenCL
    HINTS "${OPENMM_DIR}/lib/plugins" "${OPENMM_DIR}/lib"
    NO_DEFAULT_PATH)
if(NOT OPENMM_OPENCL_LIB)
    message(FATAL_ERROR "OpenMMOpenCL library not found under ${OPENMM_DIR}. "
            "Ensure OpenMM is installed with OpenCL platform support.")
endif()

target_link_libraries(${GLUED_OPENCL_LIBRARY_NAME}
    OpenMMGlued
    OpenMM
    ${OPENMM_OPENCL_LIB})

install(TARGETS ${GLUED_OPENCL_LIBRARY_NAME}
    LIBRARY  DESTINATION lib/plugins
    RUNTIME  DESTINATION lib/plugins
    ARCHIVE  DESTINATION lib/plugins)
