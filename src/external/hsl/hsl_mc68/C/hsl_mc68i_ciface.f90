! THIS VERSION: GALAHAD 5.0 - 2024-03-27 AT 09:10 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 8 _ C I F A C E   M O D U L E  -*-*-

module hsl_mc68_integer_ciface
   use hsl_kinds, only: ipc_, lp_, ip_, longc_
   use hsl_mc68_integer, only:              &
      f_mc68_control       => mc68_control, &
      f_mc68_info          => mc68_info,    &
      f_mc68_order         => mc68_order
   implicit none

   type, bind(C) :: mc68_control
      integer(ipc_) :: f_array_in ! 0 is false, otherwise true
      integer(ipc_) :: f_array_out! 0 is false, otherwise true
      integer(ipc_) :: min_l_workspace
      integer(ipc_) :: lp
      integer(ipc_) :: wp
      integer(ipc_) :: mp
      integer(ipc_) :: nemin
      integer(ipc_) :: print_level
      integer(ipc_) :: row_full_thresh
      integer(ipc_) :: row_search
   end type mc68_control

   type, bind(C) :: mc68_info
     integer(ipc_) :: flag
     integer(ipc_) :: iostat
     integer(ipc_) :: stat
     integer(ipc_) :: out_range
     integer(ipc_) :: duplicate
     integer(ipc_) :: n_compressions
     integer(ipc_) :: n_zero_eigs
     integer(longc_) :: l_workspace
     integer(ipc_) :: zb01_info
     integer(ipc_) :: n_dense_rows
   end type mc68_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_array_in, f_array_out, &
         min_l_workspace)
      type(mc68_control), intent(in) :: ccontrol
      type(f_mc68_control), intent(out) :: fcontrol
      logical(lp_), intent(out) :: f_array_in
      logical(lp_), intent(out) :: f_array_out
      integer(ip_), intent(out) :: min_l_workspace

      f_array_in                 = (ccontrol%f_array_in .ne. 0)
      f_array_out                = (ccontrol%f_array_out .ne. 0)
      min_l_workspace            = ccontrol%min_l_workspace
      fcontrol%lp                = ccontrol%lp
      fcontrol%wp                = ccontrol%wp
      fcontrol%mp                = ccontrol%mp
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%print_level       = ccontrol%print_level
      fcontrol%row_full_thresh   = ccontrol%row_full_thresh
      fcontrol%row_search        = ccontrol%row_search
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_mc68_info), intent(in) :: finfo
      type(mc68_info), intent(out) :: cinfo

      cinfo%flag              = finfo%flag
      cinfo%iostat            = finfo%iostat
      cinfo%stat              = finfo%stat
      cinfo%out_range         = finfo%out_range
      cinfo%duplicate         = finfo%duplicate
      cinfo%n_compressions    = finfo%n_compressions
      cinfo%n_zero_eigs       = finfo%n_zero_eigs
      cinfo%l_workspace       = finfo%l_workspace
      cinfo%zb01_info         = finfo%zb01_info
      cinfo%n_dense_rows      = finfo%n_dense_rows
   end subroutine copy_info_out
end module hsl_mc68_integer_ciface
