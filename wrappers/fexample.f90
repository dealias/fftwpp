program fexample
  use fftwpp_module

  implicit NONE
  integer :: m=8
  complex :: z =(2,3)
  complex, dimension(8) :: f
  ! FIXME: make f and g allocatable or whatever it is in Fortran.
   
  type(hconv1d_type) :: conv
  write(*,*) z

  call new_hconv1d(conv,m)
  
  !call del_hconv1d(conv) !segfaults, just like before, yo.

end program fexample
