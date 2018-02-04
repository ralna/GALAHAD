!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_contrib
  use, intrinsic :: iso_c_binding
  use spral_ssids_datatypes, only : wp
  implicit none

  private
  public :: contrib_type

  ! This type represents a contribution block being passed between two
  ! subtrees. It exists in CPU memory, but provides a cleanup routine as
  ! memory management may differ between two subtrees being passed.
  ! (It would be nice and clean to have a procedure pointer for the cleanup,
  ! but alas Fortran/C interop causes severe problems, so we just have the
  ! owner value instead and if statements to call the right thing).
  type :: contrib_type
     logical :: ready = .false.
     integer :: n ! size of block
     real(C_DOUBLE), dimension(:), pointer :: val ! n x n lwr triangular matrix
     integer(C_INT) :: ldval
     integer(C_INT), dimension(:), pointer :: rlist ! row list
     integer :: ndelay
     integer(C_INT), dimension(:), pointer :: delay_perm
     real(C_DOUBLE), dimension(:), pointer :: delay_val
     integer :: lddelay
     integer :: owner ! cleanup routine to call: 0=cpu, 1=gpu
     ! Following are used by CPU to call correct cleanup routine
     logical(C_BOOL) :: posdef
     type(C_PTR) :: owner_ptr
  end type contrib_type
end module spral_ssids_contrib

! C function to get interesting components
subroutine spral_ssids_contrib_get_data(ccontrib, n, val, ldval, rlist, &
     ndelay, delay_perm, delay_val, lddelay) bind(C)
  use, intrinsic :: iso_c_binding
  use spral_ssids_contrib
  implicit none

  type(C_PTR), value :: ccontrib
  integer(C_INT), intent(out) :: n
  type(C_PTR), intent(out) :: val
  integer(C_INT), intent(out) :: ldval
  type(C_PTR), intent(out) :: rlist
  integer(C_INT), intent(out) :: ndelay
  type(C_PTR), intent(out) :: delay_perm
  type(C_PTR), intent(out) :: delay_val
  integer(C_INT), intent(out) :: lddelay

  type(contrib_type), pointer, volatile :: fcontrib

  if (c_associated(ccontrib)) then
     call c_f_pointer(ccontrib, fcontrib)

     do while (.not. fcontrib%ready)
        ! FIXME: make below a taskyield? (was: flush)
        !$omp taskyield
     end do

     n = fcontrib%n
     val = c_loc(fcontrib%val)
     ldval = fcontrib%ldval
     rlist = c_loc(fcontrib%rlist)
     ndelay = fcontrib%ndelay
     if (associated(fcontrib%delay_val)) then
        delay_perm = c_loc(fcontrib%delay_perm)
        delay_val = c_loc(fcontrib%delay_val)
     else
        delay_perm = c_null_ptr
        delay_val = c_null_ptr
     end if
     lddelay = fcontrib%lddelay
  end if
end subroutine spral_ssids_contrib_get_data
