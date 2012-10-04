module fftwpp_module
!  use, intrinsic :: ISO_C_Binding, only: C_int, C_ptr, C_NULL_ptr
  use, intrinsic :: ISO_C_Binding !FIXME: use only.... to clean up namespace?
  implicit none
  private
  
  type hconv1d_type
     private
     type(C_ptr) :: object = C_NULL_ptr
  end type hconv1d_type
  
  interface
     function C_create_hconv1d(m) result(this) bind(C,name="fftwpp_create_conv1d")
       import
       type(C_ptr) :: this
       integer(C_int), value :: m
     end function C_create_hconv1d
  end interface
  
  interface
     subroutine C_delete_hconv1d(this) bind(C,name="fftwpp_conv1d_delete")
       import
 
       type(C_ptr), intent(inout) :: this
     end subroutine C_delete_hconv1d
  end interface

  interface
     subroutine C_convolve_hconv1d(this,f,g) bind(C,name="fftwpp_conv1d_convolve")
       import
!       complex(C_DOUBLE_COMPLEX), pointer, intent(in) :: f(:), g(:)
       type(C_PTR) :: f,g
       type(C_ptr), intent(inout) :: this
     end subroutine C_convolve_hconv1d
  end interface

  public :: hconv1d_type, new_hconv1d, del_hconv1d, conv_hconv1d
  
  interface new_hconv1d
     module procedure create_hconv1d
  end interface new_hconv1d

  interface del_hconv1d
     module procedure delete_hconv1d
  end interface del_hconv1d

  interface conv_hconv1d
     module procedure convolve_hconv1d
  end interface conv_hconv1d

contains

  subroutine create_hconv1d(this,m)
    type(hconv1d_type), intent(out) :: this
    integer(C_int), value :: m
    this%object = C_create_hconv1d(m)
  end subroutine create_hconv1d

  subroutine delete_hconv1d(this)
    type(hconv1d_type), intent(inout) :: this
    call C_delete_hconv1d(this%object)
    this%object = C_NULL_ptr
  end subroutine delete_hconv1d

  subroutine convolve_hconv1d(this,f,g)
    type(hconv1d_type), intent(inout) :: this
    !complex(C_DOUBLE_COMPLEX), pointer, intent(in) :: f(:), g(:)
    !complex(C_DOUBLE_COMPLEX), pointer, intent(in) :: f(:), g(:)
    type(C_PTR) :: f,g
    call C_convolve_hconv1d(this%object,f,g)
  end subroutine convolve_hconv1d



end module fftwpp_module

