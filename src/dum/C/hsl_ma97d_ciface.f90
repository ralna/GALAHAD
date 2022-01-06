! THIS VERSION: 29/12/2021 AT 15:40:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 9 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma97_double_ciface
   use iso_c_binding
   use hsl_ma97_double, only:                            &
      f_ma97_akeep            => ma97_akeep,             &
      f_ma97_control          => ma97_control,           &
      f_ma97_fkeep            => ma97_fkeep,             &
      f_ma97_info             => ma97_info,              &
      f_ma97_analyse          => ma97_analyse,           &
      f_ma97_analyse_coord    => ma97_analyse_coord,     &
      f_ma97_factor           => ma97_factor,            &
      f_ma97_factor_solve     => ma97_factor_solve,      &
      f_ma97_solve            => ma97_solve,             &
      f_ma97_free             => ma97_free,              &
      f_ma97_enquire_posdef   => ma97_enquire_posdef,    &
      f_ma97_enquire_indef    => ma97_enquire_indef,     &
      f_ma97_alter            => ma97_alter,             &
      f_ma97_solve_fredholm   => ma97_solve_fredholm,    &
      f_ma97_lmultiply        => ma97_lmultiply,         &
      f_ma97_sparse_fwd_solve => ma97_sparse_fwd_solve,  &
      f_ma97_get_n__          => ma97_get_n__,           &
      f_ma97_get_nz__         => ma97_get_nz__

   integer, parameter :: wp = C_DOUBLE ! pkg type
   integer, parameter :: rp = C_DOUBLE ! real type

   type, bind(C) :: ma97_control
      integer(C_INT) :: f_arrays ! true(!=0) or false(==0)
      integer(C_INT) :: action ! true(!=0) or false(==0)
      integer(C_INT) :: nemin
      real(rp) :: multiplier
      integer(C_INT) :: ordering
      integer(C_INT) :: print_level
      integer(C_INT) :: scaling
      real(rp) :: small
      real(rp) :: u
      integer(C_INT) :: unit_diagnostics
      integer(C_INT) :: unit_error
      integer(C_INT) :: unit_warning
      integer(C_LONG) :: factor_min
      integer(C_INT) :: solve_blas3
      integer(C_LONG) :: solve_min
      integer(C_INT) :: solve_mf
      real(rp) :: consist_tol
      integer(C_INT) :: ispare(5)
      real(rp) :: rspare(10)
   end type ma97_control

   type, bind(C) :: ma97_info
      integer(C_INT) :: flag
      integer(C_INT) :: flag68
      integer(C_INT) :: flag77
      integer(C_INT) :: matrix_dup
      integer(C_INT) :: matrix_rank
      integer(C_INT) :: matrix_outrange
      integer(C_INT) :: matrix_missing_diag
      integer(C_INT) :: maxdepth
      integer(C_INT) :: maxfront
      integer(C_INT) :: num_delay
      integer(C_LONG) :: num_factor
      integer(C_LONG) :: num_flops
      integer(C_INT) :: num_neg
      integer(C_INT) :: num_sup
      integer(C_INT) :: num_two
      integer(C_INT) :: ordering
      integer(C_INT) :: stat
      integer(C_INT) :: ispare(5)
      real(rp) :: rspare(10)
   end type ma97_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma97_control), intent(in) :: ccontrol
      type(f_ma97_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

      f_arrays                   = (ccontrol%f_arrays.ne.0)
      fcontrol%action            = (ccontrol%action.ne.0)
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%multiplier        = ccontrol%multiplier
      fcontrol%ordering          = ccontrol%ordering
      fcontrol%print_level       = ccontrol%print_level
      fcontrol%scaling           = ccontrol%scaling
      fcontrol%small             = ccontrol%small
      fcontrol%u                 = ccontrol%u
      fcontrol%unit_diagnostics  = ccontrol%unit_diagnostics
      fcontrol%unit_error        = ccontrol%unit_error
      fcontrol%unit_warning      = ccontrol%unit_warning
      fcontrol%factor_min        = ccontrol%factor_min
      fcontrol%solve_blas3       = (ccontrol%solve_blas3.ne.0)
      fcontrol%solve_min         = ccontrol%solve_min
      fcontrol%solve_mf          = (ccontrol%solve_mf.ne.0)
      fcontrol%consist_tol       = ccontrol%consist_tol
   end subroutine copy_control_in

   subroutine copy_info_out(finfo,cinfo)
      type(f_ma97_info), intent(in) :: finfo
      type(ma97_info), intent(out) :: cinfo

      cinfo%flag                 = finfo%flag
      cinfo%flag68               = finfo%flag68
      cinfo%flag77               = finfo%flag77
      cinfo%matrix_dup           = finfo%matrix_dup
      cinfo%matrix_rank          = finfo%matrix_rank
      cinfo%matrix_outrange      = finfo%matrix_outrange
      cinfo%matrix_missing_diag  = finfo%matrix_missing_diag
      cinfo%maxdepth             = finfo%maxdepth
      cinfo%maxfront             = finfo%maxfront
      cinfo%num_delay            = finfo%num_delay
      cinfo%num_factor           = finfo%num_factor
      cinfo%num_flops            = finfo%num_flops
      cinfo%num_neg              = finfo%num_neg
      cinfo%num_sup              = finfo%num_sup
      cinfo%num_two              = finfo%num_two
      cinfo%ordering             = finfo%ordering
      cinfo%stat                 = finfo%stat
   end subroutine copy_info_out

end module hsl_ma97_double_ciface
