! THIS VERSION: GALAHAD 4.3 - 2024-02-09 AT 09:50 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 4 _ C I F A C E   M O D U L E  -*-*-

module hsl_mc64_real_ciface
   use hsl_kinds_real, only: ipc_, rpc_, lp_
   use hsl_mc64_real, only:                  &
      f_mc64_control       => mc64_control,  &
      f_mc64_info          => mc64_info,     &
      f_mc64_matching      => mc64_matching
   implicit none

   type, bind(C) :: mc64_control
      integer(ipc_) :: f_arrays
      integer(ipc_) :: lp
      integer(ipc_) :: wp
      integer(ipc_) :: sp
      integer(ipc_) :: ldiag
      integer(ipc_) :: checking
   end type mc64_control

   type, bind(C) :: mc64_info
      integer(ipc_) :: flag
      integer(ipc_) :: more
      integer(ipc_) :: strucrank
      integer(ipc_) :: stat
   end type mc64_info

contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(mc64_control), intent(in) :: ccontrol
      type(f_mc64_control), intent(out) :: fcontrol
      logical(lp_), intent(out) :: f_arrays

      f_arrays          = (ccontrol%f_arrays.ne.0)
      fcontrol%lp       = ccontrol%lp
      fcontrol%wp       = ccontrol%wp
      fcontrol%sp       = ccontrol%sp
      fcontrol%ldiag    = ccontrol%ldiag
      fcontrol%checking = ccontrol%checking
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_mc64_info), intent(in) :: finfo
      type(mc64_info), intent(out) :: cinfo

      cinfo%flag        = finfo%flag
      cinfo%more        = finfo%more
      cinfo%strucrank   = finfo%strucrank
      cinfo%stat        = finfo%stat
   end subroutine copy_info_out
end module hsl_mc64_real_ciface
