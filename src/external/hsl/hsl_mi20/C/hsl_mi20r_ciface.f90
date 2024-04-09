! THIS VERSION: GALAHAD 5.0 - 2024-03-27 AT 09:10 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M I 2 0 _ C I F A C E   M O D U L E  -*-*-

module hsl_mi20_real_ciface
   use hsl_kinds_real, only: ipc_, rpc_, lp_, C_BOOL
   use hsl_mi20_real, only:                      &
      f_mi20_data          => mi20_data,         &
      f_mi20_keep          => mi20_keep,         &
      f_mi20_control       => mi20_control,      &
      f_mi20_solve_control => mi20_solve_control,&
      f_mi20_info          => mi20_info,         &
      f_mi20_setup         => mi20_setup,        &
!     f_mi20_setup_csr     => mi20_setup_csr,    &
!     f_mi20_setup_csc     => mi20_setup_csc,    &
!     f_mi20_setup_coord   => mi20_setup_coord,  &
!     f_mi20_precondition  => mi20_precondition, &
!     f_mi20_solve         => mi20_solve,        &
      f_mi20_finalize      => mi20_finalize
   use hsl_zd11_real, only:                      &
      f_zd11_type          => zd11_type,         &
      f_zd11_put           => zd11_put
   implicit none

   type, bind(C) :: mi20_control
      integer(ipc_) :: f_arrays ! true (!=0) or false (==0)
      integer(ipc_) :: aggressive
      integer(ipc_) :: c_fail
      integer(ipc_) :: max_levels
      integer(ipc_) :: max_points
      real(rpc_) :: reduction
      integer(ipc_) :: st_method
      real(rpc_) :: st_parameter
      integer(ipc_) :: testing
      real(rpc_) :: trunc_parameter
      integer(ipc_) :: coarse_solver
      integer(ipc_) :: coarse_solver_its
      real(rpc_) :: damping
      real(rpc_) :: err_tol
      integer(ipc_) :: levels
      integer(ipc_) :: pre_smoothing
      integer(ipc_) :: smoother
      integer(ipc_) :: post_smoothing
      integer(ipc_) :: v_iterations
      integer(ipc_) :: print_level
      integer(ipc_) :: print
      integer(ipc_) :: error
      integer(ipc_) :: one_pass_coarsen
   end type mi20_control

   type, bind(C) :: mi20_solve_control
      real(rpc_) ::  abs_tol
      real(rpc_) ::  breakdown_tol
      integer(ipc_) :: gmres_restart
      logical(C_BOOL) :: init_guess
      integer(ipc_) :: krylov_solver
      integer(ipc_) :: max_its
      integer(ipc_) :: preconditioner_side
      real(rpc_) ::  rel_tol
   end type mi20_solve_control

   type, bind(C) :: mi20_info
      integer(ipc_) :: flag
      integer(ipc_) :: clevels
      integer(ipc_) :: cpoints
      integer(ipc_) :: cnnz
      integer(ipc_) :: stat
      integer(ipc_) :: getrf_info
      integer(ipc_) :: iterations
      real(rpc_) :: residual
   end type mi20_info

   type ciface_keep_type
      type(f_zd11_type) :: matrix
      type(f_mi20_data), dimension(:), allocatable :: data
      type(f_mi20_keep) :: keep
   end type ciface_keep_type
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(mi20_control), intent(in) :: ccontrol
      type(f_mi20_control), intent(out) :: fcontrol
      logical(lp_), intent(out) :: f_arrays

      f_arrays                = (ccontrol%f_arrays.ne.0)
      fcontrol%aggressive     = ccontrol%aggressive
      fcontrol%c_fail         = ccontrol%c_fail
      fcontrol%max_levels     = ccontrol%max_levels
      fcontrol%max_points     = ccontrol%max_points
      fcontrol%reduction      = ccontrol%reduction
      fcontrol%st_method      = ccontrol%st_method
      fcontrol%st_parameter   = ccontrol%st_parameter
      fcontrol%testing        = ccontrol%testing
      fcontrol%trunc_parameter= ccontrol%trunc_parameter
      fcontrol%coarse_solver  = ccontrol%coarse_solver
      fcontrol%coarse_solver_its = ccontrol%coarse_solver_its
      fcontrol%damping        = ccontrol%damping
      fcontrol%err_tol        = ccontrol%err_tol
      fcontrol%levels         = ccontrol%levels
      fcontrol%pre_smoothing  = ccontrol%pre_smoothing
      fcontrol%smoother       = ccontrol%smoother
      fcontrol%post_smoothing = ccontrol%post_smoothing
      fcontrol%v_iterations   = ccontrol%v_iterations
      fcontrol%print_level    = ccontrol%print_level
      fcontrol%print          = ccontrol%print
      fcontrol%error          = ccontrol%error
      fcontrol%one_pass_coarsen = (ccontrol%one_pass_coarsen.ne.0)
   end subroutine copy_control_in

   subroutine copy_solve_control_in(csolve_control, fsolve_control)
      type(mi20_solve_control), intent(in) :: csolve_control
      type(f_mi20_solve_control), intent(out) :: fsolve_control

      fsolve_control%abs_tol             = csolve_control%abs_tol
      fsolve_control%breakdown_tol       = csolve_control%breakdown_tol
      fsolve_control%gmres_restart       = csolve_control%gmres_restart
      fsolve_control%init_guess          = csolve_control%init_guess
      fsolve_control%krylov_solver       = csolve_control%krylov_solver
      fsolve_control%max_its             = csolve_control%max_its
      fsolve_control%preconditioner_side = csolve_control%preconditioner_side
      fsolve_control%rel_tol             = csolve_control%rel_tol
   end subroutine copy_solve_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_mi20_info), intent(in) :: finfo
      type(mi20_info), intent(out) :: cinfo

      cinfo%flag        = finfo%flag
      cinfo%clevels     = finfo%clevels
      cinfo%cpoints     = finfo%cpoints
      cinfo%cnnz        = finfo%cnnz
      cinfo%stat        = finfo%stat
      cinfo%getrf_info  = finfo%getrf_info
      cinfo%residual    = finfo%residual
      cinfo%iterations  = finfo%iterations
   end subroutine copy_info_out

end module hsl_mi20_real_ciface
