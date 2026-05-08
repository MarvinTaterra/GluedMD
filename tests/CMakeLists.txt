find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Helper: register a Python test with a shared PYTHONPATH and per-test timeout.
# Usage: add_python_test(<name> <filename> <timeout_seconds> [label...])
macro(add_python_test name file timeout)
    add_test(NAME ${name}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;$ENV{PYTHONPATH}"
        TIMEOUT     ${timeout}
        LABELS      "${ARGN}")
endmacro()

# ---- Stage 1 / 2 -------------------------------------------------------
add_python_test(smoke_test      test_api_smoke.py   120)
add_python_test(gpu_io_test     test_gpu_io.py      180)

# ---- Stage 3: CV kernels -----------------------------------------------
add_python_test(cv_distance         test_cv_distance.py         180)
add_python_test(cv_angle_dihedral   test_cv_angle_dihedral.py   240)
add_python_test(cv_com_rg           test_cv_com_rg.py           240)
add_python_test(cv_coordination     test_cv_coordination.py     240)
add_python_test(cv_rmsd             test_cv_rmsd.py             240)
add_python_test(cv_path             test_cv_path.py             300)
add_python_test(cv_expression       test_cv_expression.py       240)
add_python_test(cv_pytorch          test_cv_pytorch.py          300 requires_cuda requires_torch)
add_python_test(cv_position         test_cv_position.py         180)
add_python_test(cv_drmsd            test_cv_drmsd.py            240)
add_python_test(cv_contactmap       test_cv_contactmap.py       240)
add_python_test(cv_plane            test_cv_plane.py            240)
add_python_test(cv_projection       test_cv_projection.py       180)
add_python_test(cv_volume_cell      test_cv_volume_cell.py      180)
add_python_test(cv_dipole           test_cv_dipole.py           180)
add_python_test(cv_pca              test_cv_pca.py              240)
add_python_test(cv_puckering        test_cv_puckering.py        180)
add_python_test(cv_secondary_structure  test_cv_secondary_structure.py  240)
add_python_test(cv_ermsd  test_cv_ermsd.py  240)

# ---- Stage 4 / 5: bias methods -----------------------------------------
add_python_test(bias_harmonic           test_bias_harmonic.py           180)
add_python_test(bias_walls              test_bias_walls.py              180)
add_python_test(bias_linear             test_bias_linear.py             180)
add_python_test(bias_moving_restraint   test_bias_moving_restraint.py   240)
add_python_test(bias_metad              test_bias_metad.py              360)
add_python_test(bias_pbmetad            test_bias_pbmetad.py            360)
add_python_test(bias_opes               test_bias_opes.py               360)
add_python_test(bias_opes_expanded      test_bias_opes_expanded.py      240)
add_python_test(bias_abmd               test_bias_abmd.py               240)
add_python_test(bias_external           test_bias_external.py           240)
add_python_test(bias_ext_lagrangian     test_bias_ext_lagrangian.py     240)
add_python_test(bias_eds                test_bias_eds.py                240)

# ---- Stage 6.1: bias state serialization --------------------------------
add_python_test(bias_state_serialization  test_bias_state_serialization.py  300)

# ---- Stage 6: serialization and output ---------------------------------
add_python_test(xml_serialization       test_xml_serialization.py       120)
add_python_test(cv_query_and_reporter   test_cv_query_and_reporter.py   180)

# ---- Integration: ADP production system --------------------------------
add_python_test(integration_adp         test_integration_adp.py         600)
