cmake_minimum_required(VERSION 3.15)

project( fftw++_mpi_fftw LANGUAGES CXX )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  get_filename_component(mpi_directory "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(fftwpp_directory "${mpi_directory}" DIRECTORY)
  message(FATAL_ERROR "FFTW++ must be built from top-level directory, not the mpi/fftw directory. From ${fftwpp_directory} use\ncmake .")
endif()

set( fft_list
  transpose
  fft2
  fft2r
  fft3
  fft3r
)

add_library(fftw_mpi_fftw_helpers STATIC exmpiutils.cc)

get_filename_component(MPI_LIBS_PATH ${MPI_CXX_LIBRARIES} DIRECTORY)
set(FFTW_MPI_LIBS fftw3_mpi)

foreach(fft IN LISTS fft_list)
  add_executable(${fft}_mpi_fftw ${fft}.cc)
  target_link_directories(${fft}_mpi_fftw PRIVATE ${MPI_LIBS_PATH})
  target_link_libraries(${fft}_mpi_fftw PRIVATE fftw_mpi_helpers ${FFTW_LIBS} ${FFTW_MPI_LIBS} ${MPI_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set_target_properties(${fft}_mpi_fftw PROPERTIES OUTPUT_NAME ${fft})
  set_target_properties(${fft}_mpi_fftw PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${mpi_explict_directory}")
endforeach()
