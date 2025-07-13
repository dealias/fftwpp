cmake_minimum_required(VERSION 3.15)

project( fftw++_mpi_tests LANGUAGES CXX )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  get_filename_component(mpi_directory "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(fftwpp_directory "${mpi_directory}" DIRECTORY)
  message(FATAL_ERROR "FFTW++ must be built from toplevel directory, not the mpi tests directory. From ${fftwpp_directory} use\ncmake .")
else()
  message(STATUS "Building MPI tests.")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set( fft_list
  gather
  gatheryz
  gatherxy
  transpose
  fft2
  fft2r
  fft3
  fft3r
)

set( conv_list
  hybridconv2
  hybridconvh2
  hybridconvr2
  hybridconv3
  hybridconvh3
  hybridconvr3
  )

foreach(fft IN LISTS fft_list)
  add_executable(${fft}_mpi ${fft}.cc)
  target_link_libraries(${fft}_mpi PRIVATE fftw_mpi_helpers ${FFTW_LIBS} ${MPI_CXX_LIBRARIES} )
  set_target_properties(${fft}_mpi PROPERTIES OUTPUT_NAME ${fft})
  set_target_properties(${fft}_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${mpi_tests_directory}")
endforeach()

foreach(conv IN LISTS conv_list)
  add_executable(${conv}_mpi ${conv}.cc)
  target_link_libraries(${conv}_mpi PRIVATE hybrid_mpi_helpers ${FFTW_LIBS} ${MPI_CXX_LIBRARIES} )
  set_target_properties(${conv}_mpi PROPERTIES OUTPUT_NAME ${conv})
  set_target_properties(${conv}_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${mpi_tests_directory}")
endforeach()
