program fexample
  use fftwpp_module

  implicit NONE
  integer :: m=8
  complex :: z =(2,3)

  type(hconv1d_type) :: conv
  write(*,*) z

  call new_hconv1d(conv,m)
  

end program fexample
