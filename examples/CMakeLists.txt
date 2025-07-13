cmake_minimum_required(VERSION 3.15)

project( fftw++_examples LANGUAGES CXX )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  get_filename_component(fftwpp_directory "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  message(FATAL_ERROR "FFTW++ must be built from toplevel directory, not the examples directory. From ${fftwpp_directory} use\ncmake .")
else()
  message(STATUS "Building examples.")
endif()


set( fft_list
  example0
  example0r
  example0m
  example0mr
  example1
  example1r
  example2
  example2r
  example3
  example3r
  exampletranspose
  optimal
)

set( conv_list
  exampleconv
  exampleconv2
  exampleconv3
  exampleconvh
  exampleconvh2
  exampleconvh3
  exampleconvr
  exampleconvr2
  exampleconvr3
)

foreach(fft IN LISTS fft_list)
  add_executable(${fft} ${fft}.cc)
  set_target_properties(${fft} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${examples_directory}")
  target_link_libraries(${fft} PRIVATE fftw_helpers ${FFTW_LIBS})
endforeach()

foreach(conv IN LISTS conv_list)
  add_executable(${conv} ${conv}.cc)
  set_target_properties(${fft} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${examples_directory}")
  target_link_libraries(${conv} PRIVATE hybrid_helpers ${FFTW_LIBS})
endforeach()
