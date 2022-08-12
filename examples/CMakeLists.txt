cmake_minimum_required(VERSION 3.11)

project( fftwpp_examples LANGUAGES CXX )

find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries     (PkgConfig::FFTW)

set( sample_list
  example0 )


foreach( sample ${sample_list} )
  add_executable( ${sample} ${sample}.cc ../fftw++.cc)

  target_include_directories(${sample} SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${FFTW_INCLUDE_DIRS})

  target_link_libraries( ${sample} PUBLIC OpenMP::OpenMP_CXX PkgConfig::FFTW)

    # #CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse


endforeach( sample )
