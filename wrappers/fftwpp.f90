module fftwpp
  use iso_c_binding

  ! interfaces for setting number of threads for convolution routines
  interface
     subroutine set_fftwpp_maxthreads(nthreads) &
          bind(c, name='set_fftwpp_maxthreads')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: nthreads
     end subroutine set_fftwpp_maxthreads
  end interface

  interface
     integer(c_int) function get_fftwpp_maxthreads() &
          bind(c, name='get_fftwpp_maxthreads')
       use iso_c_binding
       implicit none
!       integer(c_int), intent(in), value :: nthreads
     end function get_fftwpp_maxthreads
  end interface

  ! 1d complex non-centered convolution
  interface
     type(c_ptr) function cconv1d_create(m) &
          bind(c, name='fftwpp_create_conv1d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
     end function cconv1d_create
  end interface

  interface
     type(c_ptr) function cconv1d_create_dot(m,mm) &
          bind(c, name='fftwpp_create_conv1d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m, mm
     end function cconv1d_create_dot
  end interface

  interface
     type(c_ptr) function cconv1d_create_work(m,u,v) &
          bind(c, name='fftwpp_create_conv1d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
       type(c_ptr), intent(in), value :: u, v
     end function cconv1d_create_work
  end interface

  interface
     type(c_ptr) function cconv1d_create_work_dot(m,u,v,mm) &
          bind(c, name='fftwpp_create_conv1d_work_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m, mm
       type(c_ptr), intent(in), value :: u, v
     end function cconv1d_create_work_dot
  end interface

  interface
     subroutine cconv1d_convolve(p,f,g) bind(c, name='fftwpp_conv1d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv1d_convolve
  end interface

  interface
     subroutine cconv1d_convolve_dot(p,f,g) & 
          bind(c, name='fftwpp_conv1d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv1d_convolve_dot
  end interface

  interface
     subroutine delete_cconv1d(p) bind(c, name='fftwpp_conv1d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p
     end subroutine delete_cconv1d
  end interface

  ! 1d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv1d_create(m) &
          bind(c, name='fftwpp_create_hconv1d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
     end function hconv1d_create
  end interface

  interface
     type(c_ptr) function hconv1d_create_dot(m,mm) &
          bind(c, name='fftwpp_create_hconv1d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m, mm
     end function hconv1d_create_dot
  end interface

  interface
     type(c_ptr) function hconv1d_create_work(m,u,v,w) &
          bind(c, name='fftwpp_create_hconv1d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m
       type(c_ptr), intent(in), value :: u, v,w
     end function hconv1d_create_work
  end interface

  interface
     type(c_ptr) function hconv1d_create_work_dot(m,u,v,w,mm) &
          bind(c, name='fftwpp_create_hconv1d_work_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: m, mm
       type(c_ptr), intent(in), value :: u, v,w
     end function hconv1d_create_work_dot
  end interface

  interface
     subroutine hconv1d_convolve(p,f,g) bind(c, name='fftwpp_hconv1d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv1d_convolve
  end interface
  
  interface
     subroutine hconv1d_convolve_dot(p,f,g) &
          bind(c, name='fftwpp_hconv1d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv1d_convolve_dot
  end interface


  interface
     subroutine delete_hconv1d(p) bind(c, name='fftwpp_hconv1d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv1d
  end interface

  ! 2d complex non-centered convolution
  interface
     type(c_ptr) function cconv2d_create(mx,my) &
          bind(c, name='fftwpp_create_conv2d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
     end function cconv2d_create
  end interface

  interface
     type(c_ptr) function cconv2d_create_dot(mx,my,mm) &
          bind(c, name='fftwpp_create_conv2d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my,mm
     end function cconv2d_create_dot
  end interface

  interface
     type(c_ptr) function cconv2d_create_work(mx,my,u1,u2,v1,v2) &
          bind(c, name='fftwpp_create_conv2d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
       type(c_ptr), intent(in), value :: u1, v1, u2, v2
     end function cconv2d_create_work
  end interface

  interface
     type(c_ptr) function cconv2d_create_work_dot(mx,my,u1,u2,v1,v2,mm) &
          bind(c, name='fftwpp_create_conv2d_work_dotf')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mm
       type(c_ptr), intent(in), value :: u1, v1, u2, v2
     end function cconv2d_create_work_dot
  end interface

  interface
     subroutine cconv2d_convolve(p,f,g) bind(c, name='fftwpp_conv2d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv2d_convolve
  end interface

  interface
     subroutine cconv2d_convolve_dot(p,f,g) &
          bind(c, name='fftwpp_conv2d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv2d_convolve_dot
  end interface

  interface
     subroutine delete_cconv2d(p) bind(c, name='fftwpp_conv2d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_cconv2d
  end interface

  ! 2d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv2d_create(mx,my) &
          bind(c, name='fftwpp_create_hconv2d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
     end function hconv2d_create
  end interface

  interface
     type(c_ptr) function hconv2d_create_dot(mx,my,mm) &
          bind(c, name='fftwpp_create_hconv2d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mm
     end function hconv2d_create_dot
  end interface

  interface
     type(c_ptr) function hconv2d_create_work(mx,my,u1,v1,w1,u2,v2) &
          bind(c, name='fftwpp_create_hconv2d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my
       type(c_ptr), intent(in), value :: u1, v1, w1, u2, v2
     end function hconv2d_create_work
  end interface

  interface
     type(c_ptr) function hconv2d_create_work_dot(mx,my,u1,v1,w1,u2,v2,mm) &
          bind(c, name='fftwpp_create_hconv2d_work_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mm
       type(c_ptr), intent(in), value :: u1, v1, w1, u2, v2
     end function hconv2d_create_work_dot
  end interface

  interface
     subroutine hconv2d_convolve(p,f,g) bind(c, name='fftwpp_hconv2d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv2d_convolve
  end interface

  interface
     subroutine hconv2d_convolve_dot(p,f,g) &
          bind(c, name='fftwpp_hconv2d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv2d_convolve_dot
  end interface

  interface
     subroutine delete_hconv2d(p) bind(c, name='fftwpp_hconv2d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv2d
  end interface

  ! 3d complex non-centered convolution
  interface
     type(c_ptr) function cconv3d_create(mx,my,mz) &
          bind(c, name='fftwpp_create_conv3d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
     end function cconv3d_create
  end interface

  interface
     type(c_ptr) function cconv3d_create_dot(mx,my,mz,mm) &
          bind(c, name='fftwpp_create_conv3d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz, mm
     end function cconv3d_create_dot
  end interface

  interface
     type(c_ptr) function cconv3d_create_work(mx,my,mz,u1,v1,u2,v2,u3,v3) &
          bind(c, name='fftwpp_create_conv3d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
       type(c_ptr), intent(in), value :: u1, v1, u2, v2, u3, v3
     end function cconv3d_create_work
  end interface

  interface
     type(c_ptr) function cconv3d_create_work_dot(mx,my,mz,u1,v1,u2,v2,u3,v3,mm)&
          bind(c, name='fftwpp_create_conv3d_work_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz, mm
       type(c_ptr), intent(in), value :: u1, v1, u2, v2, u3, v3
     end function cconv3d_create_work_dot
  end interface

  interface
     subroutine cconv3d_convolve(p,f,g) bind(c, name='fftwpp_conv3d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv3d_convolve
  end interface

  interface
     subroutine cconv3d_convolve_dot(p,f,g) &
          bind(c, name='fftwpp_conv3d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine cconv3d_convolve_dot
  end interface


  interface
     subroutine delete_cconv3d(p) bind(c, name='fftwpp_conv3d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_cconv3d
  end interface

  ! 3d Hermitian-symmetric entered convolution
  interface
     type(c_ptr) function hconv3d_create(mx,my,mz) &
          bind(c, name='fftwpp_create_hconv3d')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
     end function hconv3d_create
  end interface

  interface
     type(c_ptr) function hconv3d_create_dot(mx,my,mz,mm) &
          bind(c, name='fftwpp_create_hconv3d_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz, mm
     end function hconv3d_create_dot
  end interface

  interface
     type(c_ptr) function hconv3d_create_work(mx,my,mz,u1,v1,w1,u2,v2,u3,v3) &
          bind(c, name='fftwpp_create_hconv3d_work')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz
       type(c_ptr), intent(in), value :: u1, v1, w1, u2, v2, u3, v3
     end function hconv3d_create_work
  end interface

  interface
     type(c_ptr) function hconv3d_create_work_dot(mx,my,mz,u1,v1,w1,u2,v2,u3,v3,mm) &
          bind(c, name='fftwpp_create_hconv3d_work_dot')
       use iso_c_binding
       implicit none
       integer(c_int), intent(in), value :: mx, my, mz, mm
       type(c_ptr), intent(in), value :: u1, v1, w1, u2, v2, u3, v3
     end function hconv3d_create_work_dot
  end interface

  interface
     subroutine hconv3d_convolve(p,f,g) bind(c, name='fftwpp_hconv3d_convolve')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv3d_convolve
  end interface

  interface
     subroutine hconv3d_convolve_dot(p,f,g) &
          bind(c, name='fftwpp_hconv3d_convolve_dotf')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv3d_convolve_dot
  end interface
  
  interface
     subroutine delete_hconv3d(p) bind(c, name='fftwpp_hconv3d_delete')
       use iso_c_binding
       implicit none
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv3d
  end interface

end module fftwpp
