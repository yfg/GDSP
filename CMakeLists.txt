cmake_minimum_required(VERSION 3.1)
project(Sppart)
enable_language(CXX C Fortran)

set(CMAKE_CXX_STANDARD 11)
find_package(LAPACK)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Julia REQUIRED)
find_program(JULIA_BIN_PATH NAMES julia)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
configure_file(
  ${CMAKE_SOURCE_DIR}/bin/version.hpp.in
  ${CMAKE_BINARY_DIR}/bin/version.hpp
  @ONLY
)
# execute_process(COMMAND "julia" "-e 'print(joinpath(Sys.BINDIR, Base.DATAROOTDIR, \"julia\"))'" RESULT_VARIABLE JUL_RET OUTPUT_VARIABLE JULIA_SHARE)
# get_filename_component(JULIA_BIN_DIR ${JULIA_BIN_PATH} DIRECTORY)
# set(JULIA_INCLUDE_DIR "${JULIA_BIN_DIR}/../include/julia")
# set(JULIA_LIB_DIR "${JULIA_BIN_DIR}/../lib")
# message("JULIA_PATH: ${JULIA_BIN_PATH}")
# message("JULIA_INCLUDE_DIR: ${JULIA_INCLUDE_DIR}")

add_subdirectory(external/metis-5.1.0)
add_subdirectory(external/mt-metis-0.7.2)

add_executable(sppart_test bin/sppart_test.cpp)
add_executable(metis_test bin/metis_test.cpp)
add_executable(mtmetis_test bin/mtmetis_test.cpp)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Julia_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/external/
    ${PROJECT_SOURCE_DIR}/external/metis-5.1.0/include
    ${PROJECT_SOURCE_DIR}/external/mt-metis-0.7.2/include
    ${PROJECT_SOURCE_DIR}/external/nlohmann
    ${CMAKE_BINARY_DIR}/bin/
)
add_library(sppart_blas_lapack STATIC src/blas_lapack_interface.f90)
target_link_libraries(sppart_test sppart_blas_lapack ${LAPACK_LIBRARIES})
target_link_libraries(metis_test metis)
target_link_libraries(mtmetis_test mtmetis wildriver)

find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_EXE_LINKER_FLAGS "-L${Julia_LIBRARY_DIR} -Wl,-rpath,${Julia_LIBRARY_DIR} -ljulia")
