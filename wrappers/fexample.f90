program fexample
  use fftwpp_module

  use, intrinsic :: ISO_C_Binding !FIXME: use only.... to clean up namespace?
  implicit NONE
  include 'fftw3.f03' !FIXME: have to link the file to pwd right now. Makefile?
  integer(C_SIZE_T) :: mm

  
  integer :: m, i
  complex :: z =(2,3)
  complex :: cf, cg
  double complex pf, pg ! these should be ponters....
  ! double real, pointer :: pd
  complex*16, dimension(:), allocatable, target :: g, f
  
  type(hconv1d_type) :: conv
  complex(C_DOUBLE_COMPLEX), pointer :: ff(:), gg(:)
  complex(C_DOUBLE_COMPLEX), pointer :: arr(:)
  type(C_PTR) :: p
  
  write(*,*) loc(p)
!  p=create_complexAlign(m);
  p = fftw_alloc_complex(int(mm, C_SIZE_T)) ! allocate 
!  p = fftw_alloc_real(int(mm, C_SIZE_T)) ! allocate 
  write(*,*) loc(p)
  write(*,*) loc(arr)
  call c_f_pointer(p, arr, [mm])
  write(*,*) loc(arr)  

!  arr=1.0
!  write(*,*) arr(:)
!  write(*,*) p
  
  write(*,*) "asdfasdf"

  write(*,*) z

  m=8

  ! FIXME: need to align memory here
  allocate(f(m))
  allocate(g(m))
!  pd => f

  do i=0,m-1
     f(i+1)=cmplx(i,(i+1))
     g(i+1)=cmplx(i,(2*i +1))
     print*,f(i+1)
  end do


  call new_hconv1d(conv,m)
  !call conv_hconv1d(conv,cf,cg) ! FIXME: pass pointers to arrays
  !call del_hconv1d(conv) !FIXME: segfaults

  deallocate(f)
  deallocate(g)

end program fexample
