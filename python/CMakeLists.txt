find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Locate Python includes via python3-config
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

set_property(SOURCE gluedplugin.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE gluedplugin.i PROPERTY SWIG_FLAGS
    "-I${OPENMM_DIR}/include")

configure_file(gluedplugin.i gluedplugin_configured.i @ONLY)

swig_add_library(gluedplugin
    TYPE MODULE
    LANGUAGE python
    SOURCES gluedplugin.i
)

swig_link_libraries(gluedplugin
    OpenMMGlued
    OpenMM
    Python3::Python
)

target_include_directories(gluedplugin PRIVATE
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/openmmapi/include/internal
    "${OPENMM_DIR}/include"
    Python3::NumPy
)

# Install Python module alongside openmm's site-packages
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gluedplugin.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
install(TARGETS gluedplugin
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
    RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/COLVARReporter.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ReplicaExchange.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/MultiGPUManager.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/glued.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/OPESConvergenceReporter.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}")
