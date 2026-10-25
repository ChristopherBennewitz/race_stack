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

if(ament_cmake_python_FOUND)
  find_package(ament_cmake REQUIRED)
  ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR python/${PROJECT_NAME})
  install(TARGETS _core LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  if(BUILD_TESTING)
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(test_progress_mpc test/test_progress_mpc.py)
  endif()
  ament_package()
else()
  # Plain CMake build (no ROS): lay the package out in <build>/python so PYTHONPATH=<build>/python works.
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}")
  configure_file(python/${PROJECT_NAME}/__init__.py "${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}/__init__.py" COPYONLY)
endif()
