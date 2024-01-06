program fexample
  use fftwpp ! access the wrapper for FFTW++
  use, intrinsic :: ISO_C_Binding
  implicit NONE
  include 'fftw3.f03'

  integer(c_int) :: nthreads
  integer(c_int) :: Lx, Ly, Lz
  integer(c_int) :: Hx, Hy, Hz
  integer(c_int) :: x0, y0
  integer(c_int) ::  i, j, k

  integer :: returnflag

  complex(C_DOUBLE_COMPLEX), pointer :: f(:), g(:)
  complex(C_DOUBLE_COMPLEX), pointer :: ff(:,:), gg(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: fff(:,:,:), ggg(:,:,:)

  type(C_PTR) :: pf, pg, pconv
  !type(C_PTR) :: pu, pv, pw, pu2, pv2, pu1, pv1, pw1, pu3, pv3

  returnflag=0 ! return value for tests

  write(*,*) "Example of calling fftw++ convolutions from Fortran:"

  nthreads = 2 ! specify number of threads

  call set_fftwpp_maxthreads(nthreads);


!!! hybridconv
  write(*,*) "1d noncentered complex convolution:"

  Lx = 8 ! problem size

  ! Allocate the memory using FFTW.
  pf = fftw_alloc_complex(int(Lx, C_SIZE_T))
  pg = fftw_alloc_complex(int(Lx, C_SIZE_T))

  ! Create the pointer to the convolution
  pconv = cconv1d_create(Lx)

  ! Make the FORTRAN pointer pf point to the same memory as f, and
  ! give it the correct shape.
  call c_f_pointer(pf, f, [Lx])
  call c_f_pointer(pg, g, [Lx])

  call init(f, g, Lx)
  write(*,*) "Input f:"
  call output(f, Lx)
  write(*,*) "Input g:"
  call output(g, Lx)

  call cconv1d_convolve(pconv, pf, pg)

  write(*,*) "Output:"
  call output(f, Lx)

  call delete_cconv1d(pconv)
  call fftw_free(pf)
  call fftw_free(pg)

  write(*,*)

!!! hybridconvh
  write(*,*) "1d centered Hermitian-symmetric complex convolution:"

  Hx = 4
  Lx = 2*Hx-1

  pf = fftw_alloc_complex(int(Lx, C_SIZE_T))
  pg = fftw_alloc_complex(int(Lx, C_SIZE_T))
  pconv = hconv1d_create(Lx)

   call c_f_pointer(pf, f, [Hx])
   call c_f_pointer(pg, g, [Hx])

   call init(f, g, Hx) ! Enforce Hermitian symmetry: DC mode must be real
   call fftwpp_HermitianSymmetrize(pf)
   call fftwpp_HermitianSymmetrize(pg)

   write(*,*) "Input f:"
   call output(f, Hx)
   write(*,*) "Input g:"
   call output(g, Hx)

   call hconv1d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output(f, Hx)

   call fftw_free(pf)
   call fftw_free(pg)
   call delete_hconv1d(pconv)

   write(*,*)

!!! hybridconv2
   write(*,*) "2d noncentered complex convolution:"
   Lx = 4
   Ly = 4

   pf = fftw_alloc_complex(int(Lx * Ly, C_SIZE_T))
   pg = fftw_alloc_complex(int(Lx * Ly, C_SIZE_T))
   pconv = cconv2d_create(Lx, Ly)

   call c_f_pointer(pf, ff, [Ly, Lx])
   call c_f_pointer(pg, gg, [Ly, Lx])

   call init2(ff, gg, Ly, Lx)

   write(*,*) "Input f:"
   call output2(ff, Ly, Lx)
   write(*,*) "Input g:"
   call output2(gg, Ly, Lx)

   call cconv2d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output2(ff, Lx, Ly)

   call delete_cconv2d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

!!! hybridconvh2
   write(*,*) "2d centered Hermitian-symmetric complex convolution:"
   Hx = 4
   Hy = 4
   Lx = 2 * Hx - 1
   Ly = 2 * Hy - 1

   pf = fftw_alloc_complex(int(Lx * Hy, C_SIZE_T))
   pg = fftw_alloc_complex(int(Lx * Hy, C_SIZE_T))
   pconv = hconv2d_create(Lx, Ly)

   call c_f_pointer(pf, ff, [Hy, Lx])
   call c_f_pointer(pg, gg, [Hy, Lx])

   call init2(ff, gg, Hy, Lx)

   x0=Lx/2
   call fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,pf)
   call fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,pg)

   write(*,*) "Input f:"
   call output2(ff, Hy, Lx)
   write(*,*) "Input g:"
   call output2(gg, Hy, Lx)

   call hconv2d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output2(ff, Hy, Lx)

   call delete_hconv2d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

!!! hybridconv3
   write(*,*) "3d noncentered complex convolution:"
   Lx = 4
   Ly = 4
   Lz = 4

   pf = fftw_alloc_complex(int(Lx * Ly * Lz, C_SIZE_T))
   pg = fftw_alloc_complex(int(Lx * Ly * Lz, C_SIZE_T))
   pconv = cconv3d_create(Lx, Ly, Lz)

   call c_f_pointer(pf, fff, [Lz, Ly, Lx])
   call c_f_pointer(pg, ggg, [Lz, Ly, Lx])
   call init3(fff, ggg, Lz, Ly, Lx)

   write(*,*) "Input f:"
   call output3(fff, Lz, Ly, Lx)
   write(*,*) "Input g:"
   call output3(ggg, Lz, Ly, Lx)

   call cconv3d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output3(fff, Lz, Ly, Lx)

   call delete_cconv3d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

!!! hybridconvh3
   write(*,*) "3d centered Hermitian-symmetric complex convolution:"
   Hx = 4
   Hy = 4
   Hz = 4
   Lx = 2 * Lx - 1
   Ly = 2 * Ly - 1
   Lz = 2 * Lz - 1

   pf = fftw_alloc_complex(int(Lx * Ly * Hz, C_SIZE_T))
   pg = fftw_alloc_complex(int(Lx * Ly * Hz, C_SIZE_T))
   pconv = hconv3d_create(Lx, Ly, Lz)

   call c_f_pointer(pf, fff, [Hz, Ly, Lx])
   call c_f_pointer(pg, ggg, [Hz, Ly, Lx])

   call init3(fff, ggg, Hz, Ly, Lx)

   x0=Lx/2
   y0=Ly/2
   call fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,pf)
   call fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,pg)

   write(*,*) "Input f:"
   call output3(fff, Hz, Ly, Lx)
   write(*,*) "Input g:"
   call output3(ggg, Hz, Ly, Lx)

   call hconv3d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output3(fff, Hz, Ly, Lx)

   call delete_hconv3d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

   call EXIT(returnflag)

contains
  subroutine init(f, g, m)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i
    integer(c_int), intent(in) :: m
    complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:), g(:)
    do i = 0 ,m - 1
       f(i+1)=cmplx(i+1,i+3)
       g(i+1)=cmplx(i+2,2*i+3)
    end do
  end subroutine init

  subroutine init2(f, g, Ly, Lx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: Ly, Lx
    complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:,:), g(:,:)
    do j = 0, Ly - 1
       do i = 0, Lx - 1
          f(j+1,i+1)=cmplx(i+1,j+3);
          g(j+1,i+1)=cmplx(i+2,2*j+3);
       end do
    end do
  end subroutine init2

  subroutine init3(f, g, Lz, Ly, Lx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: Lz, Ly, Lx
    complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:,:,:), g(:,:,:)
    do k = 0, Lz - 1
       do j = 0, Ly - 1
          do i = 0, Lx - 1
             f(k+1,j+1,i+1)=cmplx(i+1,j+3+k);
             g(k+1,j+1,i+1)=cmplx(i+k+1,2*j+3+k);
          end do
       end do
    end do
  end subroutine init3

  subroutine output(f, m)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i
    integer(c_int),intent(in) :: m
    complex(C_DOUBLE_COMPLEX),pointer,intent(inout) :: f(:)
    do i=1,m
       print*,f(i)
    end do
  end subroutine output

  subroutine output2(f, Ly, Lx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: j, i
    integer(c_int), intent(in) :: Ly, Lx
    complex(C_DOUBLE_COMPLEX), pointer, intent(in) :: f(:,:)
    do i = 1, Lx
       do j = 1, Ly
          print*,f(j,i)
       end do
       write(*,*)
    end do
  end subroutine output2

  subroutine output3(f, Lz, Ly, Lx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: Lz, Ly, Lx
    complex(C_DOUBLE_COMPLEX),pointer,intent(inout) :: f(:,:,:)
    do i=1,Lx
       do j=1,Ly
          do k=1,Lz
             print*,f(k,j,i)
          end do
          write(*,*)
       end do
       write(*,*)
    end do
  end subroutine output3

end program fexample
