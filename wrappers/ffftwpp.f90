module fftwpp_module
  use, intrinsic :: ISO_C_Binding, only: C_int, C_ptr, C_NULL_ptr
  implicit none
  private
  
  type hconv1d_type
     private
     type(C_ptr) :: object = C_NULL_ptr
  end type hconv1d_type
  !ImplicitConvolution_type
  
  interface
     function C_create_hconv1d(m) result(this) bind(C,name="fftwpp_create_conv1d")
       import
       type(C_ptr) :: this
       integer(C_int), value :: m
     end function C_create_hconv1d
  end interface

  public :: hconv1d_type, new_hconv1d
  
  interface new_hconv1d
     module procedure create_hconv1d
  end interface new_hconv1d

contains

  subroutine create_hconv1d(this,m)
    type(hconv1d_type), intent(out) :: this
    integer(C_int), value :: m
    this%object = C_create_hconv1d(m)
  end subroutine create_hconv1d

end module fftwpp_module
