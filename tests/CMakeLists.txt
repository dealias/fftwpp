cmake_minimum_required(VERSION 3.15)

project( fftwpp_tests LANGUAGES CXX )

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries(PkgConfig::FFTW)

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
  hybridconvr3
  hybrid
  hybridh
  hybridr
  )

include_directories(${CMAKE_SOURCE_DIR}/../)

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

# Find OpenMP properly
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found")
endif()

set(FFTW_LIBS fftw3 fftw3_omp m ${OpenMP_CXX_LIBRARIES})

set(FFTW_SOURCES ../fftw++.cc ../parallel.cc)
set(HYBRID_SOURCES ${FFTW_SOURCES} ../convolve.cc direct.cc options.cc)

add_library(fftw_helpers STATIC ${FFTW_SOURCES})

add_library(hybrid_helpers STATIC ${HYBRID_SOURCES})
target_link_libraries(hybrid_helpers PRIVATE fftw_helpers)

foreach(fft IN LISTS fft_list)
  add_executable(${fft} ${fft}.cc)
  target_link_libraries(${fft} PRIVATE fftw_helpers ${FFTW_LIBS})
endforeach()

foreach(conv IN LISTS conv_list)
  add_executable(${conv} ${conv}.cc)
  target_link_libraries(${conv} PRIVATE hybrid_helpers ${FFTW_LIBS})
endforeach()
