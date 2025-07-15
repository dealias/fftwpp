cmake_minimum_required(VERSION 3.15)

project( fftw++ LANGUAGES CXX )

option(TESTS "Build tests." ON)
option(EXAMPLES "Build examples." ON)

option(WRAPPERS "Building Fortran and C wrappers." OFF)
option(MPI_TESTS "Build MPI tests and examples." OFF)
option(MPI_FFTW "Build fftw MPI benchmarks." OFF)
option(ALL "Build everything." OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(FFTW_LIB fftw3 REQUIRED)
find_library(FFTW_OMP_LIB fftw3_omp REQUIRED)
find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
include_directories(${FFTW_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/)

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  add_compile_options(-Ofast -g -Wall -DNDEBUG -fomit-frame-pointer
    -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -fopenmp)
  if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GNU compiler flags")
  else()
    message(STATUS "Using Clang compiler flags")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Using Apple Clang compiler flags")

  if(EXISTS "/opt/local/lib" AND EXISTS "/opt/local/lib/libomp")
    message(STATUS "Using OpenMP via MacPorts")
    # Provide OpenMP hints BEFORE find_package()
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "omp")
    link_directories(/opt/local/lib /opt/local/lib/libomp)
    include_directories(/opt/local/include /opt/local/include/libomp)
  endif()

  add_compile_options(
    -O3 -ffast-math -g -Wall -DNDEBUG
    -fomit-frame-pointer -fstrict-aliasing
  )

elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  message(STATUS "Using Intel compiler flags")
  add_compile_options(-O3 -qopenmp -ansi-alias -malign-double -fp-model fast=2)

elseif( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
  message(STATUS "Using XL compiler flags")
  add_compile_options(-O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp)

endif()

set( lib_list fftw++ parallel )
foreach( libname ${lib_list} )
  add_library( ${libname} OBJECT ${libname}.cc )
  set_property(TARGET ${libname} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

set( lib_convtest_list convolve )
foreach( libname ${lib_convtest_list} )
  add_library( ${libname} OBJECT ${libname}.cc )
  set_property(TARGET ${libname} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories( ${libname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ )
endforeach()

set( lib_convtest_list_local options direct )
foreach( libname ${lib_convtest_list_local} )
  add_library( ${libname} OBJECT tests/${libname}.cc )
  target_include_directories( ${libname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} )
endforeach()

# Find OpenMP properly
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found")
endif()

set(FFTW_LIBS fftw3 fftw3_omp m ${OpenMP_CXX_LIBRARIES} CACHE INTERNAL "FFTW libraries")
set(FFTW_LIBS fftw3 fftw3_omp m ${OpenMP_CXX_LIBRARIES} CACHE INTERNAL "FFTW libraries")

set(FFTW_SOURCES fftw++.cc parallel.cc)
set(HYBRID_SOURCES ${FFTW_SOURCES} convolve.cc tests/direct.cc tests/options.cc)

add_library(fftw_helpers STATIC ${FFTW_SOURCES})
add_library(hybrid_helpers STATIC ${HYBRID_SOURCES})

target_link_libraries(fftw_helpers PRIVATE ${FFTW_LIB})
target_link_libraries(hybrid_helpers PRIVATE fftw_helpers ${FFTW_LIB})

if(TESTS OR ALL)
  message(STATUS "Building tests.")
  set(tests_directory "${CMAKE_BINARY_DIR}/tests")
  file(MAKE_DIRECTORY "${tests_directory}")
  add_subdirectory(tests)
else()
  message(STATUS "Skipping tests.")
endif()

if(EXAMPLES OR ALL)
  message(STATUS "Building examples.")
  set(examples_directory "${CMAKE_BINARY_DIR}/examples")
  file(MAKE_DIRECTORY "${examples_directory}")
  add_subdirectory(examples)
else()
  message(STATUS "Skipping examples.")
endif()

if(WRAPPERS OR ALL)
  message(STATUS "Building Fortran and C wrappers.")
  set(wrappers_directory "${CMAKE_BINARY_DIR}/wrappers")
  file(MAKE_DIRECTORY "${wrappers_directory}")
  add_subdirectory(wrappers)
else()
  message(STATUS "Skipping Fortran and C wrappers.")
endif()

if(MPI_TESTS OR MPI_FFTW OR ALL)
  find_package(MPI REQUIRED)

  include_directories(${CMAKE_SOURCE_DIR}/mpi ${CMAKE_SOURCE_DIR}/tests)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

  set(FFTW_MPI_SOURCES fftw++.cc parallel.cc mpi/mpitranspose.cc mpi/mpifftw++.cc)
  set(HYBRID_MPI_SOURCES ${FFTW_MPI_SOURCES} convolve.cc mpi/mpiconvolve.cc tests/options.cc)

  add_library(fftw_mpi_helpers STATIC ${FFTW_MPI_SOURCES})
  add_library(hybrid_mpi_helpers STATIC ${HYBRID_MPI_SOURCES})
  target_link_libraries(hybrid_mpi_helpers PRIVATE fftw_mpi_helpers)

  if(MPI_TESTS OR ALL)
    message(STATUS "Building MPI tests and examples.")
    set(mpi_tests_directory "${CMAKE_BINARY_DIR}/mpi/tests")
    file(MAKE_DIRECTORY "${mpi_tests_directory}")
    add_subdirectory(mpi/tests)
  else()
    message(STATUS "Skipping MPI tests and examples.")
  endif()

  if(MPI_FFTW OR ALL)
    message(STATUS "Building fftw MPI benchmarks.")
    set(mpi_fftw_directory "${CMAKE_BINARY_DIR}/mpi/fftw")
    file(MAKE_DIRECTORY "${mpi_fftw_directory}")
    add_subdirectory(mpi/fftw)
  else()
    message(STATUS "Skipping fftw MPI benchmarks.")
  endif()
else()
  message(STATUS "Skipping MPI tests and examples.")
  message(STATUS "Skipping fftw MPI benchmarks.")
endif()

add_custom_target(distclean
  COMMAND ${CMAKE_MAKE_PROGRAM} clean
  COMMAND ${CMAKE_COMMAND} -P  ${CMAKE_SOURCE_DIR}/distclean.cmake
)
