module fhash
  use iso_c_binding

  interface
     integer(c_int) function hash1(a,m) bind(c, name='hash')
       use iso_c_binding
       integer(c_int), intent(in), value :: m
       type(c_ptr), intent(in), value  :: a
     end function hash1
  end interface

end module fhash
