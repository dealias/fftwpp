cmake_minimum_required(VERSION 3.11)

project( fftwpp_examples LANGUAGES CXX )

find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries     (PkgConfig::FFTW)

set( fft_list
  example0
  example0m
  example0mr
  example1
  example1r
  example2
  example2r
  example3
  example3r
  optimal
  )

set( conv_list
  exampleconv
  exampleconvh
  exampleconvr
  exampleconv2
  exampleconvh2
  exampleconvr2
  exampleconv3
  exampleconvh3
  exampleconvr3
  exampletranspose
  )

set( sample_list ${fft_list} ${conv_list})

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -Wall -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -fopenmp")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -ansi-alias -malign-double -fp-model fast=2")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp")
endif()

add_library( fftw++ OBJECT ../fftw++.cc )
add_library( convolve OBJECT ../convolve.cc )
add_library( parallel OBJECT ../parallel.cc )

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cc )

  target_include_directories(${sample} SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${FFTW_INCLUDE_DIRS})

  if (${sample} IN_LIST conv_list)
    target_link_libraries( ${sample} PUBLIC convolve parallel )
  endif()

  target_link_libraries( ${sample} PUBLIC
    OpenMP::OpenMP_CXX
    PkgConfig::FFTW
    "-lfftw3_omp"
    parallel
    fftw++)
  # NB: FFTW is missing the fftw3_omp link in the package config file, so we add it manually.

endforeach( sample )
