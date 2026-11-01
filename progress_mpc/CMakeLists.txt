cmake_minimum_required(VERSION 3.8)
project(progress_mpc LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# acados is a manual install (the sim repo's install/install_acados.sh builds v0.4.4 into ~/acados).
if(NOT ACADOS_SOURCE_DIR)
  set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}")
endif()
if(NOT ACADOS_SOURCE_DIR)
  set(ACADOS_SOURCE_DIR "$ENV{HOME}/acados")
endif()
find_library(ACADOS_LIB acados HINTS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
find_library(HPIPM_LIB hpipm HINTS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
find_library(BLASFEO_LIB blasfeo HINTS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
find_package(ament_cmake_python QUIET)

if(NOT (ACADOS_LIB AND HPIPM_LIB AND BLASFEO_LIB))
  # Only the controller manager's MPC mode needs this package, so a workspace without acados still builds.
  if(ament_cmake_python_FOUND)
    message(WARNING "progress_mpc: acados not found in '${ACADOS_SOURCE_DIR}/lib' (set ACADOS_SOURCE_DIR); "
                    "building nothing. The controller's MPC mode will be unavailable.")
    find_package(ament_cmake REQUIRED)
    ament_package()
    return()
  endif()
  message(FATAL_ERROR "acados not found in '${ACADOS_SOURCE_DIR}/lib'; set ACADOS_SOURCE_DIR")
endif()
get_filename_component(ACADOS_LIB_DIR "${ACADOS_LIB}" DIRECTORY)

find_package(pybind11 REQUIRED)

# The solver and model code exported by scripts/export_acados.py.
file(GLOB_RECURSE GENERATED_C CONFIGURE_DEPENDS generated/acados/*.c)
list(APPEND GENERATED_C ${CMAKE_CURRENT_SOURCE_DIR}/generated/mpc_step.c)

add_library(progress_mpc_core STATIC ${GENERATED_C} src/progress_mpc.cpp)
target_include_directories(progress_mpc_core
  PUBLIC include
  PRIVATE
    generated
    generated/acados
    "${ACADOS_SOURCE_DIR}/include"
    "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
    "${ACADOS_SOURCE_DIR}/include/hpipm/include")
target_link_libraries(progress_mpc_core PUBLIC "${ACADOS_LIB}" "${HPIPM_LIB}" "${BLASFEO_LIB}" m)
target_compile_options(progress_mpc_core PRIVATE -O3 -Wall -Wno-unused-function)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE progress_mpc_core)
set_target_properties(_core PROPERTIES
  BUILD_RPATH "${ACADOS_LIB_DIR}"
  INSTALL_RPATH "${ACADOS_LIB_DIR}")

# AC4MPC (the actor- and critic-guided MPC): needs libtorch for the PPO networks, and the export of its OCP
# (scripts/export_ac4mpc.py). Without either it is skipped and the rest builds as before. libtorch is taken from
# the python `torch` package (TORCH_ROOT, or the `python3` on the path), the same one the controller's RL mode uses.
set(AC4MPC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated_ac4mpc)
set(AC4MPC_BUILD OFF)
if(EXISTS ${AC4MPC_DIR}/ac4mpc_casadi.c)
  if(NOT TORCH_ROOT)
    set(TORCH_ROOT "$ENV{TORCH_ROOT}")
  endif()
  if(NOT TORCH_ROOT)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_EXECUTABLE)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import os, torch; print(os.path.dirname(torch.__file__))"
                      OUTPUT_VARIABLE TORCH_ROOT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif()
  find_library(TORCH_LIB torch HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
  find_library(TORCH_CPU_LIB torch_cpu HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
  find_library(C10_LIB c10 HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
  if(TORCH_ROOT AND TORCH_LIB AND TORCH_CPU_LIB AND C10_LIB)
    set(AC4MPC_BUILD ON)
  else()
    message(WARNING "progress_mpc: libtorch not found (set TORCH_ROOT to the python torch package); AC4MPC is not built.")
  endif()
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated_ac4mpc)
  message(WARNING "progress_mpc: generated_ac4mpc/ has no export; run scripts/export_ac4mpc.py. AC4MPC is not built.")
endif()

if(AC4MPC_BUILD)
  file(GLOB_RECURSE AC4MPC_GENERATED_C CONFIGURE_DEPENDS ${AC4MPC_DIR}/acados/*.c)
  list(APPEND AC4MPC_GENERATED_C ${AC4MPC_DIR}/ac4mpc_casadi.c)
  add_library(ac4mpc_core STATIC ${AC4MPC_GENERATED_C} src/ac4mpc.cpp src/policy.cpp)
  target_include_directories(ac4mpc_core
    PUBLIC include
    PRIVATE
      src
      ${AC4MPC_DIR}
      ${AC4MPC_DIR}/acados
      "${ACADOS_SOURCE_DIR}/include"
      "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
      "${ACADOS_SOURCE_DIR}/include/hpipm/include"
      "${TORCH_ROOT}/include"
      "${TORCH_ROOT}/include/torch/csrc/api/include")
  target_link_libraries(ac4mpc_core PUBLIC progress_mpc_core "${ACADOS_LIB}" "${HPIPM_LIB}" "${BLASFEO_LIB}" m
                        -Wl,--no-as-needed "${TORCH_LIB}" "${TORCH_CPU_LIB}" "${C10_LIB}" pthread)
  target_compile_options(ac4mpc_core PRIVATE -O3 -Wall -Wno-unused-function)

  pybind11_add_module(_ac4mpc src/ac4mpc_bindings.cpp)
  target_link_libraries(_ac4mpc PRIVATE ac4mpc_core)
  set_target_properties(_ac4mpc PROPERTIES
    BUILD_RPATH "${ACADOS_LIB_DIR};${TORCH_ROOT}/lib"
    INSTALL_RPATH "${ACADOS_LIB_DIR};${TORCH_ROOT}/lib")
endif()

if(ament_cmake_python_FOUND)
  find_package(ament_cmake REQUIRED)
  ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR python/${PROJECT_NAME})
  install(TARGETS _core LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  if(AC4MPC_BUILD)
    install(TARGETS _ac4mpc LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  endif()
  if(BUILD_TESTING)
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(test_progress_mpc test/test_progress_mpc.py)
  endif()
  ament_package()
else()
  # Plain CMake build (no ROS): lay the package out in <build>/python so PYTHONPATH=<build>/python works.
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}")
  if(AC4MPC_BUILD)
    set_target_properties(_ac4mpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}")
  endif()
  configure_file(python/${PROJECT_NAME}/__init__.py "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}/__init__.py" COPYONLY)
endif()
