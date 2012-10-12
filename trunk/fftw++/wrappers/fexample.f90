program fexample
  use fftwpp

  use, intrinsic :: ISO_C_Binding !FIXME: use only.... to clean up namespace?
  implicit NONE
  include 'fftw3.f03'
  integer(c_int) :: m
  integer :: i, flag
  
  complex(C_DOUBLE_COMPLEX), pointer :: f(:), g(:)
  type(C_PTR) :: pf, pg, hconv1d, cconv1d
  
  write(*,*) "Example of calling fftw++ convolutions from Fortran:"

  m=8 ! problem size

  flag=0 ! return value for tests
  
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
  cconv1d=cconv1d_create(m)
  call cconv1d_convolve(cconv1d,pf,pg)
  call delete_cconv1d(cconv1d)

  ! FIXME: add output test.

  call output(f,m)

  call init(f,g,m) ! reset input data

  write(*,*)
  write(*,*) "1d centered Hermitian-symmetric complex convolution:"
  hconv1d=hconv1d_create(m)
  call hconv1d_convolve(hconv1d,pf,pg)
  call delete_hconv1d(hconv1d)

  ! FIXME: add output test.

  call output(f,m)
  i=hash1(f,m)

  ! memory-aligned arrays need to be deleted using FFTW.
  call fftw_free(pf)
  call fftw_free(pg)

  ! 2d non-centered complex convolution
  
  ! 2d centered Hermitian convolution

  ! 3d non-centered complex convolution
  
  ! 3d centered Hermitian convolution

  ! 1d centered Hermitian ternary convolution

  ! 2d centered Hermitian ternary convolution

  call EXIT(flag)

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

