cmake_minimum_required(VERSION 3.11)

project( fftwpp_examples LANGUAGES CXX C Fortran )

find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
link_libraries     (PkgConfig::FFTW)

# TODO: set C and FORTRAN flags
if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -Wall -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -fopenmp")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -ansi-alias -malign-double -fp-model fast=2")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp")
endif()

add_library( fftw++ OBJECT ../fftw++.cc )
set_property(TARGET fftw++ PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library( convolve OBJECT ../convolve.cc )
set_property(TARGET convolve PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library( parallel OBJECT ../parallel.cc )
set_property(TARGET parallel PROPERTY POSITION_INDEPENDENT_CODE ON)

set_source_files_properties( cfftw++.cc LANGUAGES C )
add_library( cfftw++ OBJECT cfftw++.cc )
set_property( TARGET cfftw++ PROPERTY POSITION_INDEPENDENT_CODE ON )
target_include_directories( cfftw++ SYSTEM PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../ )

add_library( _fftwpp SHARED cfftw++.cc )
target_include_directories( _fftwpp SYSTEM PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../ )
set_source_files_properties( cexample.c LANGUAGES C )
target_link_libraries( _fftwpp PUBLIC
  OpenMP::OpenMP_CXX
  PkgConfig::FFTW
  "-lfftw3_omp"
  parallel
  fftw++
  convolve)

add_executable( cexample cexample.c )
set_source_files_properties( cexample.c LANGUAGES C )
target_link_libraries( cexample PUBLIC
  OpenMP::OpenMP_CXX
  PkgConfig::FFTW
  "-lfftw3_omp"
  cfftw++
  fftw++
  parallel
  convolve )

add_library( fftwpp SHARED fftwpp.f90 )
set_source_files_properties( fftwpp.f90 LANGUAGES Fortran )
target_include_directories( fftwpp SYSTEM PUBLIC ${FFTW_INCLUDEDIR} )

add_executable( fexample fexample.f90 )
set_source_files_properties( fexample.f90 LANGUAGES Fortran )
target_link_libraries( fexample PUBLIC
  OpenMP::OpenMP_Fortran
  ${FFTW_LDFLAGS}
  "-lfftw3_omp"
  fftwpp
  cfftw++
  fftw++
  parallel
  convolve)
