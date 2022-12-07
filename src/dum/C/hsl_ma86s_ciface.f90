! THIS VERSION: 29/12/2021 AT 15:35:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 8 6 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma86_single_ciface
   use iso_c_binding
   use hsl_ma86_single, only :                     &
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
      integer(C_INT) :: f_arrays ! 0 is false, otherwise is true
      ! Printing controls
      integer(C_INT) :: diagnostics_level
      integer(C_INT) :: unit_diagnostics
      integer(C_INT) :: unit_error
      integer(C_INT) :: unit_warning
      ! Controls used by ma86_analyse
      integer(C_INT) :: nemin
      integer(C_INT) :: nb
      ! Controls used by ma86_factor and ma86_factor_solve
      integer(C_INT) :: action ! 0 is false, otherwise is true
      integer(C_INT) :: nbi
      integer(C_INT) :: pool_size
      real(C_FLOAT) :: small
      real(C_FLOAT) :: static
      real(C_FLOAT) :: u
      real(C_FLOAT) :: umin
      integer(C_INT) :: scaling
   end type ma86_control

   !*************************************************

   ! data type for returning information to user.
   type, bind(C) :: ma86_info 
      real(C_FLOAT)  :: detlog
      integer(C_INT)  :: detsign
      integer(C_INT)  :: flag
      integer(C_INT)  :: matrix_rank
      integer(C_INT)  :: maxdepth
      integer(C_INT)  :: num_delay
      integer(C_LONG) :: num_factor
      integer(C_LONG) :: num_flops
      integer(C_INT)  :: num_neg
      integer(C_INT)  :: num_nodes
      integer(C_INT)  :: num_nothresh
      integer(C_INT)  :: num_perturbed
      integer(C_INT)  :: num_two
      integer(C_INT)  :: pool_size
      integer(C_INT)  :: stat
      real(C_FLOAT)  :: usmall
   end type ma86_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma86_control), intent(in) :: ccontrol
      type(f_ma86_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

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
end module hsl_ma86_single_ciface
