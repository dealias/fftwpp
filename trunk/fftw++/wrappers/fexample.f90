program fexample
  use fftwpp_module

  implicit NONE
  integer :: m, i
  complex :: z =(2,3)
  complex :: cf, cg
  double complex pf, pg
  real, pointer :: dp
  
  complex, dimension(:), allocatable :: f, g
   
  type(hconv1d_type) :: conv
  write(*,*) z

  m=8

  ! FIXME: need to align memory here
  allocate(f(m))
  allocate(g(m))

  do i=0,m-1
     f(i+1)=cmplx(i,(i+1))
     g(i+1)=cmplx(i,(2*i +1))
     print*,f(i+1)
  end do


  call new_hconv1d(conv,m)
  call conv_hconv1d(conv,cf,cg) ! FIXME: pass pointers to arrays
  !call del_hconv1d(conv) !FIXME: segfaults

  deallocate(f)
  deallocate(g)

end program fexample
