! THIS VERSION: 29/12/2021 AT 15:40:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 8 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma87_single_ciface
   use iso_c_binding
   use hsl_ma87_single, only :                           &
      f_ma87_keep             => ma87_keep,              &
      f_ma87_control          => ma87_control,           &
      f_ma87_info             => ma87_info,              &
      f_ma87_analyse          => ma87_analyse,           &
      f_ma87_factor           => ma87_factor,            &
      f_ma87_factor_solve     => ma87_factor_solve,      &
      f_ma87_solve            => ma87_solve,             &
      f_ma87_sparse_fwd_solve => ma87_sparse_fwd_solve,  &
      f_ma87_finalise         => ma87_finalise,          &
      f_ma87_get_n__          => ma87_get_n__
   implicit none

   ! Data type for user controls
   type, bind(C) :: ma87_control
      ! C/Fortran interface related controls
      integer(C_INT) :: f_arrays ! 0 is false, otherwise is true
      ! Printing controls
      integer(C_INT) :: diagnostics_level
      integer(C_INT) :: unit_diagnostics
      integer(C_INT) :: unit_error
      integer(C_INT) :: unit_warning
      ! Controls used by ma87_analyse
      integer(C_INT) :: nemin
      integer(C_INT) :: nb
      ! Controls used by ma87_factor and ma87_factor_solve
      integer(C_INT) :: pool_size
      real(C_FLOAT) :: diag_zero_minus
      real(C_FLOAT) :: diag_zero_plus
      character(C_CHAR), dimension(40) :: unused
   end type ma87_control

   !*************************************************

   ! data type for returning information to user.
   type, bind(C) :: ma87_info 
      real(C_FLOAT)  :: detlog
      integer(C_INT)  :: flag
      integer(C_INT)  :: maxdepth
      integer(C_LONG) :: num_factor
      integer(C_LONG) :: num_flops
      integer(C_INT)  :: num_nodes
      integer(C_INT)  :: pool_size
      integer(C_INT)  :: stat
      integer(C_INT)  :: num_zero
      character(C_CHAR), dimension(40) :: unused
   end type ma87_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma87_control), intent(in) :: ccontrol
      type(f_ma87_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

      f_arrays                   = (ccontrol%f_arrays .ne. 0)
      fcontrol%diagnostics_level = ccontrol%diagnostics_level
      fcontrol%unit_diagnostics  = ccontrol%unit_diagnostics
      fcontrol%unit_error        = ccontrol%unit_error
      fcontrol%unit_warning      = ccontrol%unit_warning
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%nb                = ccontrol%nb
      fcontrol%pool_size         = ccontrol%pool_size
      fcontrol%diag_zero_minus   = ccontrol%diag_zero_minus
      fcontrol%diag_zero_plus    = ccontrol%diag_zero_plus
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_ma87_info), intent(in) :: finfo
      type(ma87_info), intent(out) :: cinfo

      cinfo%detlog = finfo%detlog
      cinfo%flag = finfo%flag
      cinfo%maxdepth = finfo%maxdepth
      cinfo%num_factor = finfo%num_factor
      cinfo%num_flops = finfo%num_flops
      cinfo%num_nodes = finfo%num_nodes
      cinfo%pool_size = finfo%pool_size
      cinfo%stat = finfo%stat
      cinfo%num_zero = finfo%num_zero
   end subroutine copy_info_out
end module hsl_ma87_single_ciface
