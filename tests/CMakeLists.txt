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
  hybridconv
  hybridconvh
  hybridconvr
  hybridconv2
  hybridconvh2
  hybridconvr2
  hybridconv3
  hybridconvh3
  hybrid
  hybridh
  hybridr
  )

set( sample_list ${fft_list} ${conv_list} )

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -Wall -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -fopenmp")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -ansi-alias -malign-double -fp-model fast=2")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "XL" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp")
endif()

set( lib_list fftw++ parallel )
foreach( libname ${lib_list} )
  add_library( ${libname} OBJECT ../${libname}.cc )
endforeach()

set( lib_convtest_list convolve )
foreach( libname ${lib_convtest_list} )
  add_library( ${libname} OBJECT ../${libname}.cc )
  target_include_directories( ${libname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../ )
endforeach()

set( lib_convtest_list_local options direct )
foreach( libname ${lib_convtest_list_local} )
  add_library( ${libname} OBJECT ${libname}.cc )
  target_include_directories( ${libname} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/.. )
endforeach()

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cc )

  target_include_directories(${sample} SYSTEM PUBLIC
    ${FFTW_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
  )

  if (${sample} IN_LIST conv_list)
    target_link_libraries( ${sample} PRIVATE convolve ${lib_convtest_list} )
    target_link_libraries( ${sample} PRIVATE options direct ${lib_convtest_list_local} )
  endif()

  target_link_libraries( ${sample} PUBLIC
    OpenMP::OpenMP_CXX
    PkgConfig::FFTW
    "-lfftw3_omp"
    parallel
    fftw++ )
  # NB: FFTW is missing the fftw3_omp link in the package config file, so we add it manually.

endforeach( sample )
