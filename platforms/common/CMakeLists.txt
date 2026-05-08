# The common layer is compiled into each GPU platform plugin, not a standalone
# library.  GPU platform subdirectories include this directory's source files.
# This CMakeLists.txt exists only to define the INTERFACE include path.

add_library(GluedCommon INTERFACE)
target_include_directories(GluedCommon INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include/internal
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/common"
)
