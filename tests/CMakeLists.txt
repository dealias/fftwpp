cmake_minimum_required(VERSION 3.15)
project( fftw++_tests LANGUAGES CXX )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  get_filename_component(fftwpp_directory "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  message(FATAL_ERROR "FFTW++ must be built from toplevel directory, not the tests directory. From ${fftwpp_directory} use\ncmake .")
endif()

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
  hybridconvroe2
  hybridconvroe3
  hybridconvrcm
  hybridconvrcm2
  hybrid
  hybridh
  hybridr
  )

foreach(fft IN LISTS fft_list)
  add_executable(${fft} ${fft}.cc)
  set_target_properties(${conv} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${tests_directory}")
  target_link_libraries(${fft} PRIVATE fftw_helpers ${FFTW_LIBS})
endforeach()

foreach(conv IN LISTS conv_list)
  add_executable(${conv} ${conv}.cc)
  set_target_properties(${conv} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${tests_directory}")
  target_link_libraries(${conv} PRIVATE hybrid_helpers ${FFTW_LIBS})
endforeach()
