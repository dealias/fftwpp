module fftwpp
  use iso_c_binding

  interface
     type(c_ptr) function hconv1d_create(m) bind(c, name='fftwpp_create_hconv1d')
       use iso_c_binding
       integer(c_int), intent(in), value :: m
     end function hconv1d_create
  end interface

  interface
     subroutine hconv1d_convolve(p,f,g) bind(c, name='fftwpp_hconv1d_convolve')
       use iso_c_binding
       type(c_ptr), intent(in), value :: p,f,g
     end subroutine hconv1d_convolve
  end interface

  interface
     subroutine delete_hconv1d(p) bind(c, name='fftwpp_hconv1d_delete')
       use iso_c_binding
       type(c_ptr), intent(in), value  :: p
     end subroutine delete_hconv1d
  end interface


end module fftwpp
