find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Run collected tests through pytest, including module-level optional skips.
set(GLUED_TEST_RUNNER "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_ctest.py"
    --core "$<TARGET_FILE:OpenMMGlued>"
    --python-dir "${CMAKE_BINARY_DIR}/python"
    --python-dir "${CMAKE_SOURCE_DIR}/python")
foreach(platform Reference CUDA OpenCL)
    if(TARGET OpenMMGlued${platform})
        list(APPEND GLUED_TEST_RUNNER --plugin "$<TARGET_FILE:OpenMMGlued${platform}>")
    endif()
endforeach()

add_test(NAME pytest_suite COMMAND ${GLUED_TEST_RUNNER})
set_tests_properties(pytest_suite PROPERTIES TIMEOUT 900)

# Additional numerical checks live in main() and cannot be collected by pytest.
foreach(script test_api_smoke test_cv_energy test_cv_energy_vsite test_bias_multithermal)
    add_test(NAME ${script} COMMAND ${GLUED_TEST_RUNNER}
        --script "${CMAKE_CURRENT_SOURCE_DIR}/${script}.py")
    set_tests_properties(${script} PROPERTIES TIMEOUT 300)
endforeach()
