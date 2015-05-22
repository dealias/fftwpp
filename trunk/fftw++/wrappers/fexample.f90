program fexample
  use fftwpp ! access the wrapper for FFTW++
  use, intrinsic :: ISO_C_Binding
  implicit NONE
  include 'fftw3.f03'

  integer(c_int) :: nthreads
  integer(c_int) :: mx, my, mz
  integer(c_int) :: mxp, myp
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


!!! cconv  
  write(*,*) "1d non-centered complex convolution:"

  mx = 8 ! problem size

  ! Allocate the memory using FFTW.
  pf = fftw_alloc_complex(int(mx, C_SIZE_T))
  pg = fftw_alloc_complex(int(mx, C_SIZE_T))

  ! Create the pointer to the convolution
  pconv = cconv1d_create(mx)

  ! Make the FORTRAN pointer pf point to the same memory as f, and
  ! give it the correct shape.
  call c_f_pointer(pf, f, [mx])
  call c_f_pointer(pg, g, [mx])

  call init(f, g, mx)
  write(*,*) "Input f:"
  call output(f, mx)
  write(*,*) "Input g:"
  call output(g, mx)
  
  call cconv1d_convolve(pconv, pf, pg)

  write(*,*) "Output:"
  call output(f, mx)

  call delete_cconv1d(pconv)
  call fftw_free(pf)
  call fftw_free(pg)

  write(*,*)

!!! hconv
  write(*,*) "1d centered Hermitian-symmetric complex convolution:"
   
  mx = 8 ! problem size
  
  pf = fftw_alloc_complex(int(mx, C_SIZE_T))
  pg = fftw_alloc_complex(int(mx, C_SIZE_T))
  pconv = hconv1d_create(mx)

   call c_f_pointer(pf, f, [mx])
   call c_f_pointer(pg, g, [mx])
   
   call init(f, g, mx)
   write(*,*) "Input f:"
   call output(f, mx)
   write(*,*) "Input g:"
   call output(g, mx)

   call hconv1d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output(f, mx)

   call fftw_free(pf)
   call fftw_free(pg)
   call delete_hconv1d(pconv)

   write(*,*)
   
!!! cconv2
   write(*,*) "2d non-centered complex convolution:"
   mx = 4
   my = 4
   
   pf = fftw_alloc_complex(int(mx * my, C_SIZE_T))
   pg = fftw_alloc_complex(int(mx * my, C_SIZE_T))
   pconv = cconv2d_create(mx, my)

   call c_f_pointer(pf, ff, [my, mx])
   call c_f_pointer(pg, gg, [my, mx])
   call init2(ff, gg, my, mx)
   
   write(*,*) "Input f:"
   call output2(ff, my, mx)
   write(*,*) "Input g:"
   call output2(gg, my, mx)
   
   call cconv2d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output2(ff, mx, my)

   call delete_cconv2d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)
   
!!! hconv2
   write(*,*) "2d centered complex convolution:"
   mx = 4
   my = 4
   mxp = 2 * mx - 1
   
   pf = fftw_alloc_complex(int(mxp * my, C_SIZE_T))
   pg = fftw_alloc_complex(int(mxp * my, C_SIZE_T))
   pconv = hconv2d_create(mx, my)

   call c_f_pointer(pf, ff, [my, mxp])
   call c_f_pointer(pg, gg, [my, mxp])

   call init2(ff, gg, my, mxp)
   
   write(*,*) "Input f:"
   call output2(ff, my, mxp)
   write(*,*) "Input g:"
   call output2(gg, my, mxp)

   call hconv2d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output2(ff, my, mxp)

   call delete_hconv2d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

!!! cconv3
   write(*,*) "3d non-centered complex convolution:"
   mx = 4
   my = 4
   mz = 4
   
   pf = fftw_alloc_complex(int(mx * my * mz, C_SIZE_T))
   pg = fftw_alloc_complex(int(mx * my * mz, C_SIZE_T))
   pconv = cconv3d_create(mx, my, mz)

   call c_f_pointer(pf, fff, [mz, my, mx])
   call c_f_pointer(pg, ggg, [mz, my, mx])
   call init3(fff, ggg, mz, my, mx)
   
   write(*,*) "Input f:"
   call output3(fff, mz, my, mx)
   write(*,*) "Input g:"
   call output3(ggg, mz, my, mx)
   
   call cconv3d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output3(fff, mz, my, mx)

   call delete_cconv3d(pconv)
   call fftw_free(pf)
   call fftw_free(pg)

   !!! hconv2
   write(*,*) "2d centered complex convolution:"
   mx = 4
   my = 4
   mz = 4
   mxp = 2 * mx - 1
   myp = 2 * my - 1
   
   pf = fftw_alloc_complex(int(mxp * myp * mz, C_SIZE_T))
   pg = fftw_alloc_complex(int(mxp * myp * mz, C_SIZE_T))
   pconv = hconv3d_create(mx, my, mz)

   call c_f_pointer(pf, fff, [mz, myp, mxp])
   call c_f_pointer(pg, ggg, [mz, myp, mxp])

   call init3(fff, ggg, mz, myp, mxp)
   
   write(*,*) "Input f:"
   call output3(fff, mz, myp, mxp)
   write(*,*) "Input g:"
   call output3(ggg, mz, myp, mxp)

   call hconv3d_convolve(pconv, pf, pg)

   write(*,*) "Output:"
   call output3(fff, mz, myp, mxp)

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
       f(i + 1) = cmplx(i, i + 1)
       g(i + 1) = cmplx(i, 2 * i + 1)
    end do
  end subroutine init

  subroutine init2(f, g, my, mx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: my, mx
    complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:,:), g(:,:)
    do j = 0, my - 1
       do i = 0, mx - 1
          f(j + 1, i + 1) = cmplx(i, j)
          g(j + 1, i + 1) = cmplx(2 * i, j + 1)
       end do
    end do
  end subroutine init2

  subroutine init3(f, g, mz, my, mx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: mz, my, mx
    complex(C_DOUBLE_COMPLEX), pointer, intent(inout) :: f(:,:,:), g(:,:,:)
    do k = 0, mz - 1
       do j = 0, my - 1
          do i = 0, mx - 1
             f(k + 1, j + 1, i + 1) = cmplx(i + k, j + k)
             g(k + 1, j + 1, i + 1) = cmplx(2 * i + k, j + 1 + k)
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

  subroutine output2(f, my, mx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: j, i
    integer(c_int), intent(in) :: my, mx
    complex(C_DOUBLE_COMPLEX), pointer, intent(in) :: f(:,:)
    do j = 1, my
       do i = 1, mx
          print*,f(j,i)
       end do
       write(*,*)
    end do
  end subroutine output2

  subroutine output3(f, mz, my, mx)
    use,intrinsic :: ISO_C_Binding
    implicit NONE
    integer :: i,j
    integer(c_int),intent(in) :: mz, my, mx
    complex(C_DOUBLE_COMPLEX),pointer,intent(inout) :: f(:,:,:)
    do k=1,mz
       do j=1,my
          do i=1,mx
             print*,f(k,j,i)
          end do
          write(*,*)
       end do
       write(*,*)
    end do
  end subroutine output3

end program fexample

