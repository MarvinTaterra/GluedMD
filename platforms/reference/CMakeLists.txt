set(GLUED_REFERENCE_LIBRARY_NAME OpenMMGluedReference)

set(SOURCE_FILES
    src/ReferenceGluedKernels.cpp
    src/ReferenceGluedKernelFactory.cpp
)

add_library(${GLUED_REFERENCE_LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_compile_definitions(${GLUED_REFERENCE_LIBRARY_NAME} PRIVATE
    GLUED_BUILDING_SHARED_LIBRARY)

target_include_directories(${GLUED_REFERENCE_LIBRARY_NAME} PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include/internal
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm/reference"
)

target_link_libraries(${GLUED_REFERENCE_LIBRARY_NAME}
    OpenMMGlued
    OpenMM)

install(TARGETS ${GLUED_REFERENCE_LIBRARY_NAME}
    LIBRARY  DESTINATION lib/plugins
    RUNTIME  DESTINATION lib/plugins
    ARCHIVE  DESTINATION lib/plugins)
