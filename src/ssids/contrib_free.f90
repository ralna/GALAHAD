!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!
!> \brief Routines for freeing contrib_type.
!>
!> As it depends on routines defined by module that use the type, it needs
!> to be a seperate module to spral_ssids_contrib.
module spral_ssids_contrib_free
  use spral_ssids_contrib, only : contrib_type
  use spral_ssids_cpu_subtree, only : cpu_free_contrib
  use spral_ssids_gpu_subtree, only : gpu_free_contrib
  implicit none

contains
  subroutine contrib_free(contrib)
    implicit none
    type(contrib_type), intent(inout) :: contrib

    select case(contrib%owner)
    case (0) ! CPU
       call cpu_free_contrib(contrib%posdef, contrib%owner_ptr)
    case (1) ! GPU
       call gpu_free_contrib(contrib)
    case default
       ! This should never happen
       print *, "Unrecognised contrib owner ", contrib%owner
       stop -1
    end select
  end subroutine contrib_free
end module spral_ssids_contrib_free

! The C prototype for the following routine is in contrib.h
subroutine spral_ssids_contrib_free_dbl(ccontrib) bind(C)
  use, intrinsic :: iso_c_binding
  use spral_ssids_contrib_free
  implicit none

  type(C_PTR), value :: ccontrib

  type(contrib_type), pointer :: fcontrib

   if (c_associated(ccontrib)) then
      call c_f_pointer(ccontrib, fcontrib)
      call contrib_free(fcontrib)
   end if
end subroutine spral_ssids_contrib_free_dbl
