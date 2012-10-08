program fexample
  use fftwpp

  use, intrinsic :: ISO_C_Binding !FIXME: use only.... to clean up namespace?
  implicit NONE
  include 'fftw3.f03'
  integer(c_int) :: m
  integer :: i
  
  complex(C_DOUBLE_COMPLEX), pointer :: f(:), g(:)
  type(C_PTR) :: pf, pg, hconv1d, cconv1d
  
  write(*,*) "Example of calling fftw++ convolutions from Fortran:"

  m=8 ! problem size

  !allocate memory:
  pf = fftw_alloc_complex(int(m, C_SIZE_T))
  call c_f_pointer(pf, f, [m])
  pg = fftw_alloc_complex(int(m, C_SIZE_T))
  call c_f_pointer(pg, g, [m])

  ! initialize arrays
  call init(f,g,m)

  write(*,*)
  write(*,*) "input f:"
  call output(f,m)

  write(*,*)
  write(*,*) "input g:"
  call output(g,m)

  write(*,*)
  write(*,*) "1d non-centered complex convolution:"
  cconv1d=cconv1d_create(m)   !Create non-centered convolution:
  call cconv1d_convolve(cconv1d,pf,pg)   !convolve:
  call delete_cconv1d(cconv1d) !delete C++ object and free work memory

  call output(f,m)

  call init(f,g,m) ! reset input data

  write(*,*)
  write(*,*) "1d centered Hermitian-symmetric complex convolution:"
  hconv1d=hconv1d_create(m) !Create Hermitian-symmetric centered convolution:
  call hconv1d_convolve(hconv1d,pf,pg) ! convolve
  call delete_hconv1d(hconv1d)   !delete C++ object and free work memory
  call output(f,m)

  ! memory-aligned arrays need to be deleted using FFTW.
  call fftw_free(pf)
  call fftw_free(pg)

  contains
    
    subroutine init(f,g,m)
      use, intrinsic :: ISO_C_Binding
      implicit NONE
      integer :: i
      integer(c_int), intent(in) :: m
      complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:), g(:)
      do i=0,m-1
         f(i+1)=cmplx(i,(i+1))
         g(i+1)=cmplx(i,(2*i +1))
      end do
    end subroutine init
    
    subroutine output(f,m)
      use, intrinsic :: ISO_C_Binding
      implicit NONE
      integer :: i
      integer(c_int), intent(in) :: m
      complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:)
      do i=1,m
         print*,f(i)
      end do
    end subroutine output

end program fexample

