module fftwpp
  use iso_c_binding

  ! Interfaces for setting number of threads for convolution routines
  interface
     subroutine set_fftwpp_maxthreads(nthreads) &
          bind(c, name = 'set_fftwpp_maxthreads')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: nthreads
     end subroutine set_fftwpp_maxthreads
  end interface

  interface
     integer(c_int) function get_fftwpp_maxthreads() &
          bind(c, name = 'get_fftwpp_maxthreads')
       use iso_c_binding
       implicit none
     end function get_fftwpp_maxthreads
  end interface

  ! 1d complex non-centered convolution
  interface
     type(c_ptr) function cconv1d_create(m) &
          bind(c, name = 'fftwpp_create_conv1d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
     end function cconv1d_create
  end interface

  interface
     subroutine cconv1d_convolve(p, f, g) &
          bind(c, name='fftwpp_conv1d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv1d_convolve
  end interface

  interface
     subroutine delete_cconv1d(p) &
          bind(c, name = 'fftwpp_conv1d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p
     end subroutine delete_cconv1d
  end interface

  ! 1d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv1d_create(m) &
          bind(c, name = 'fftwpp_create_hconv1d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
     end function hconv1d_create
  end interface

  interface
     subroutine hconv1d_convolve(p, f, g) &
          bind(c, name='fftwpp_hconv1d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p, f, g
     end subroutine hconv1d_convolve
  end interface
  
  interface
     subroutine hconv1d_convolve_dot(p, f, g) &
          bind(c, name = 'fftwpp_hconv1d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv1d_convolve_dot
  end interface

  interface
     subroutine delete_hconv1d(p) &
          bind(c, name = 'fftwpp_hconv1d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv1d
  end interface

  ! 2d complex non-centered convolution
  interface
     type(c_ptr) function cconv2d_create(mx, my) &
          bind(c, name = 'fftwpp_create_conv2d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
     end function cconv2d_create
  end interface

  interface
     subroutine cconv2d_convolve(p, f, g) &
          bind(c, name = 'fftwpp_conv2d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p, f, g
     end subroutine cconv2d_convolve
  end interface

  interface
     subroutine delete_cconv2d(p) &
          bind(c, name = 'fftwpp_conv2d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_cconv2d
  end interface

  ! 2d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv2d_create(mx, my) &
          bind(c, name = 'fftwpp_create_hconv2d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
     end function hconv2d_create
  end interface

  interface
     subroutine hconv2d_convolve(p, f, g) &
          bind(c, name = 'fftwpp_hconv2d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv2d_convolve
  end interface

  interface
     subroutine delete_hconv2d(p) &
          bind(c, name = 'fftwpp_hconv2d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv2d
  end interface

  ! 3d complex non-centered convolution
  interface
     type(c_ptr) function cconv3d_create(mx, my, mz) &
          bind(c, name = 'fftwpp_create_conv3d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
     end function cconv3d_create
  end interface

  interface
     subroutine cconv3d_convolve(p, f, g) &
          bind(c, name='fftwpp_conv3d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p, f, g
     end subroutine cconv3d_convolve
  end interface

  interface
     subroutine delete_cconv3d(p) &
          bind(c, name = 'fftwpp_conv3d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_cconv3d
  end interface

  ! 3d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv3d_create(mx, my, mz) &
          bind(c, name = 'fftwpp_create_hconv3d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
     end function hconv3d_create
  end interface

  interface
     subroutine hconv3d_convolve(p, f, g) &
          bind(c, name = 'fftwpp_hconv3d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv3d_convolve
  end interface
  
  interface
     subroutine delete_hconv3d(p) &
          bind(c, name = 'fftwpp_hconv3d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv3d
  end interface

end module fftwpp
