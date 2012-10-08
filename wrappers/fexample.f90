program fexample
  !use fftwpp_module
  use fftwpp

  use, intrinsic :: ISO_C_Binding !FIXME: use only.... to clean up namespace?
  implicit NONE
  include 'fftw3.f03' !FIXME: have to link the file to pwd right now. Makefile?
  integer(c_int) :: m
  integer :: i
  
!  type(hconv1d_type) :: conv
  complex(C_DOUBLE_COMPLEX), pointer :: f(:), g(:)
  type(C_PTR) :: pf, pg, pconv
 
  m=8 ! problem size

  write(*,*) "allocate memory:"
  write(*,*) associated(f)
  pf = fftw_alloc_complex(int(m, C_SIZE_T)) ! allocate 
  call c_f_pointer(pf, f, [m])
  pg = fftw_alloc_complex(int(m, C_SIZE_T)) ! allocate 
  call c_f_pointer(pg, g, [m])
  write(*,*) associated(f)

  ! initialize arrays
  do i=0,m-1
     f(i+1)=cmplx(i,(i+1))
     g(i+1)=cmplx(i,(2*i +1))
     print*,f(i+1)
  end do

  write(*,*)
  write(*,*) "Create convolution:"
  pconv=hconv1d_create(m)

  write(*,*)
  write(*,*) "convolve:"
  call hconv1d_convolve(pconv,pf,pg)

!  write(*,*)
!  write(*,*) "delete C++ object:"
!  call delete_hconv1d(pconv)! sefaults


  ! output of result
  write(*,*)
  write(*,*) "output:"
  do i=1,m
     print*,f(i)
  end do

  !deallocate(f) ! needs to be done with fftw_delete or some such thing.
  !deallocate(g)

end program fexample
