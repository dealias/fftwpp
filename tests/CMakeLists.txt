cmake_minimum_required(VERSION 3.11)

project( fftwpp_tests LANGUAGES CXX )

find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries     (PkgConfig::FFTW)

set( fft_list
  fft1
  fft1r
  fft2
  fft2r
  fft3
  fft3r
  mfft1
  mfft1r
  transpose
  )  

set( conv_list
  cconv2
  cconv
  conv3
  hybridconv
  hybridconvh
  tconv
  cconv3
  conv2
  conv
  hybridconv2
  hybridconvh2
  tconv2
  hybrid
  hybridh
  )

set( sample_list ${fft_list} ${conv_list})

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ansi-alias -malign-double -fp-model fast=2")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp")
endif()

foreach( sample ${sample_list} )
  
  if (${sample} IN_LIST fft_list) 
    add_executable( ${sample} ${sample}.cc ../fftw++.cc getopt.c)
  else()
    add_executable( ${sample} ${sample}.cc ../fftw++.cc getopt.c ../convolution ../explicit.cc ../direct.cc ../convolve.cc )
  endif()

  target_include_directories(${sample} SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${FFTW_INCLUDE_DIRS})

  target_link_libraries( ${sample} PUBLIC
    OpenMP::OpenMP_CXX
    PkgConfig::FFTW
    "-lfftw3_omp")
  # NB: FFTW is missing the fftw3_omp link in the package config file, so we add it manually.

endforeach( sample )
