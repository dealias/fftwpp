cmake_minimum_required(VERSION 3.11)

project( fftw++_wrappers LANGUAGES CXX C Fortran )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  get_filename_component(mpi_directory "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(fftwpp_directory "${mpi_directory}" DIRECTORY)
  message(FATAL_ERROR "FFTW++ must be built from top-level directory, not the wrappers directory. From ${fftwpp_directory} use\ncmake .")
else()
  message(STATUS "Building fftw wrappers.")
endif()

# TODO: set C and FORTRAN flags

# C flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(STATUS "C Compiler: GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
    message(STATUS "C Compiler: Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp -Ofast")

elseif(CMAKE_C_COMPILER_ID MATCHES "XL")
    message(STATUS "C Compiler: IBM XL")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -qsmp")
else()
    message(WARNING "Unknown C compiler — using default flags")
endif()

# FORTRAN flags
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    message(STATUS "FORTRAN Compiler: GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    message(STATUS "FORTRAN Compiler: Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -openmp")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "XL")
    message(STATUS "FORTRAN Compiler: IBM XL")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -qsmp")
else()
    message(WARNING "Unknown FORTRAN compiler — using default flags")
endif()


set_source_files_properties( cfftw++.cc LANGUAGES C )
add_library( cfftw++ OBJECT cfftw++.cc )
set_property( TARGET cfftw++ PROPERTY POSITION_INDEPENDENT_CODE ON )
target_include_directories( cfftw++ SYSTEM PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../ )
add_library( _fftwpp SHARED cfftw++.cc )

target_include_directories( _fftwpp SYSTEM PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( _fftwpp PUBLIC
  fftw++
  convolve
  parallel
  ${FFTW_LIB}
  ${FFTW_OMP_LIB}
)

add_executable( cexample cexample.c )
set_source_files_properties( cexample.c LANGUAGES C )
set_target_properties(${cexample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${wrappers_directory}")
target_link_libraries( cexample PUBLIC
  fftw++
  convolve
  parallel
  cfftw++
  ${FFTW_LIB}
  ${FFTW_OMP_LIB}
)

add_library( fftwpp SHARED fftwpp.f90 )
target_include_directories( fftwpp SYSTEM PUBLIC ${FFTW_INCLUDEDIR} )

add_executable( fexample fexample.f90 )
set_source_files_properties( fexample.f90 LANGUAGES Fortran )
set_target_properties(fexample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${wrappers_directory}")
target_link_libraries( fexample PUBLIC
  fftwpp
  fftw++
  convolve
  parallel
  cfftw++
  ${FFTW_LIB}
  ${FFTW_OMP_LIB}
)
