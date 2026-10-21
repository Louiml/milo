# Run the script-level golden tests.
#
# Every `tests/scripts/<name>.milo` is run through the `milo` executable and
# its standard output is compared with `tests/scripts/<name>.expected`.
#
# Usage:
#   cmake -DMILO_BIN=<path> -DMILO_SCRIPT_DIR=<dir> -P run_scripts.cmake

if(NOT DEFINED MILO_BIN OR NOT DEFINED MILO_SCRIPT_DIR)
  message(FATAL_ERROR "MILO_BIN and MILO_SCRIPT_DIR must both be set")
endif()

file(GLOB milo_scripts "${MILO_SCRIPT_DIR}/*.milo")
list(SORT milo_scripts)

set(failures 0)

foreach(script IN LISTS milo_scripts)
  get_filename_component(name "${script}" NAME_WE)
  set(expected "${MILO_SCRIPT_DIR}/${name}.expected")

  if(NOT EXISTS "${expected}")
    message("MISSING  ${name} (no .expected file)")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()

  # Every script is run twice: once by the tree-walking evaluator and once
  # by the bytecode machine. They must agree with each other and with the
  # expected output.
  foreach(engine "" "--vm")
    execute_process(
      COMMAND "${MILO_BIN}" run ${engine} "${script}"
      WORKING_DIRECTORY "${MILO_SCRIPT_DIR}"
      OUTPUT_VARIABLE actual
      ERROR_VARIABLE errors
      RESULT_VARIABLE code)

    file(READ "${expected}" expected_text)
    string(STRIP "${expected_text}" expected_text)
    string(STRIP "${actual}" actual)

    if(NOT code EQUAL 0)
      message("FAIL     ${name} ${engine} (exit code ${code})")
      message("--- stderr ---")
      message("${errors}")
      math(EXPR failures "${failures} + 1")
    elseif(NOT actual STREQUAL expected_text)
      message("FAIL     ${name} ${engine}")
      message("--- expected ---")
      message("${expected_text}")
      message("--- actual ---")
      message("${actual}")
      math(EXPR failures "${failures} + 1")
    else()
      message("ok       ${name} ${engine}")
    endif()
  endforeach()
endforeach()

if(failures GREATER 0)
  message(FATAL_ERROR "${failures} script test(s) failed")
endif()
