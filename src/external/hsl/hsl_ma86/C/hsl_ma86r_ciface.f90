! THIS VERSION: GALAHAD 5.0 - 2024-03-27 AT 09:10 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M A 8 6 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma86_real_ciface
   use hsl_kinds_real, only: ipc_, rpc_, lp_, longc_
   use hsl_ma86_real, only :                       &
      f_ma86_keep          => ma86_keep,           &
      f_ma86_control       => ma86_control,        &
      f_ma86_info          => ma86_info,           &
      f_ma86_analyse       => ma86_analyse,        &
      f_ma86_factor        => ma86_factor,         &
      f_ma86_factor_solve  => ma86_factor_solve,   &
      f_ma86_solve         => ma86_solve,          &
      f_ma86_finalise      => ma86_finalise,       &
      f_ma86_get_n__       => ma86_get_n__
   implicit none

   ! Data type for user controls
   type, bind(C) :: ma86_control
      ! C/Fortran interface related controls
      integer(ipc_) :: f_arrays ! 0 is false, otherwise is true
      ! Printing controls
      integer(ipc_) :: diagnostics_level
      integer(ipc_) :: unit_diagnostics
      integer(ipc_) :: unit_error
      integer(ipc_) :: unit_warning
      ! Controls used by ma86_analyse
      integer(ipc_) :: nemin
      integer(ipc_) :: nb
      ! Controls used by ma86_factor and ma86_factor_solve
      integer(ipc_) :: action ! 0 is false, otherwise is true
      integer(ipc_) :: nbi
      integer(ipc_) :: pool_size
      real(rpc_) :: small
      real(rpc_) :: static
      real(rpc_) :: u
      real(rpc_) :: umin
      integer(ipc_) :: scaling
   end type ma86_control

   !*************************************************

   ! data type for returning information to user.
   type, bind(C) :: ma86_info
      real(rpc_)  :: detlog
      integer(ipc_)  :: detsign
      integer(ipc_)  :: flag
      integer(ipc_)  :: matrix_rank
      integer(ipc_)  :: maxdepth
      integer(ipc_)  :: num_delay
      integer(longc_) :: num_factor
      integer(longc_) :: num_flops
      integer(ipc_)  :: num_neg
      integer(ipc_)  :: num_nodes
      integer(ipc_)  :: num_nothresh
      integer(ipc_)  :: num_perturbed
      integer(ipc_)  :: num_two
      integer(ipc_)  :: pool_size
      integer(ipc_)  :: stat
      real(rpc_)  :: usmall
   end type ma86_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma86_control), intent(in) :: ccontrol
      type(f_ma86_control), intent(out) :: fcontrol
      logical(lp_), intent(out) :: f_arrays

      f_arrays                   = (ccontrol%f_arrays .ne. 0)
      fcontrol%diagnostics_level = ccontrol%diagnostics_level
      fcontrol%unit_diagnostics  = ccontrol%unit_diagnostics
      fcontrol%unit_error        = ccontrol%unit_error
      fcontrol%unit_warning      = ccontrol%unit_warning
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%nb                = ccontrol%nb
      fcontrol%action            = (ccontrol%action .ne. 0)
      fcontrol%nbi               = ccontrol%nbi
      fcontrol%pool_size         = ccontrol%pool_size
      fcontrol%small             = ccontrol%small
      fcontrol%static            = ccontrol%static
      fcontrol%u                 = ccontrol%u
      fcontrol%umin              = ccontrol%umin
      fcontrol%scaling           = ccontrol%scaling
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_ma86_info), intent(in) :: finfo
      type(ma86_info), intent(out) :: cinfo

      cinfo%detlog = finfo%detlog
      cinfo%detsign = finfo%detsign
      cinfo%flag = finfo%flag
      cinfo%matrix_rank = finfo%matrix_rank
      cinfo%maxdepth = finfo%maxdepth
      cinfo%num_delay = finfo%num_delay
      cinfo%num_factor = finfo%num_factor
      cinfo%num_flops = finfo%num_flops
      cinfo%num_neg = finfo%num_neg
      cinfo%num_nodes = finfo%num_nodes
      cinfo%num_nothresh = finfo%num_nothresh
      cinfo%num_perturbed = finfo%num_perturbed
      cinfo%num_two = finfo%num_two
      cinfo%pool_size = finfo%pool_size
      cinfo%stat = finfo%stat
      cinfo%usmall = finfo%usmall
   end subroutine copy_info_out
end module hsl_ma86_real_ciface
