! THIS VERSION: 29/12/2021 AT 15:30:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 4 8 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma48_double_ciface
   use iso_c_binding
   use hsl_ma48_double, only:                                           &
      f_ma48_factors                => ma48_factors,                    &
      f_ma48_control                => ma48_control,                    &
      f_ma48_ainfo                  => ma48_ainfo,                      &
      f_ma48_finfo                  => ma48_finfo,                      &
      f_ma48_sinfo                  => ma48_sinfo,                      &
      f_ma48_initialize             => ma48_initialize,                 &
      f_ma48_analyse                => ma48_analyse,                    &
      f_ma48_get_perm               => ma48_get_perm,                   &
      f_ma48_factorize              => ma48_factorize,                  &
      f_ma48_solve                  => ma48_solve,                      &
      f_ma48_finalize               => ma48_finalize,                   &
      f_ma48_special_rows_and_cols  => ma48_special_rows_and_cols,      &
      f_ma48_determinant            => ma48_determinant
   use hsl_zd11_double
   implicit none

   integer, parameter :: wp = C_DOUBLE

   type, bind(C) :: ma48_control
      integer(C_INT) :: f_arrays
      real(wp) :: multiplier
      real(wp) :: u
      real(wp) :: switch
      real(wp) :: drop
      real(wp) :: tolerance
      real(wp) :: cgce
      integer(C_INT) :: lp
      integer(C_INT) :: wp
      integer(C_INT) :: mp
      integer(C_INT) :: ldiag
      integer(C_INT) :: btf
      integer(C_INT) :: struct
      integer(C_INT) :: maxit
      integer(C_INT) :: factor_blocking
      integer(C_INT) :: solve_blas
      integer(C_INT) :: pivoting
      integer(C_INT) :: diagonal_pivoting
      integer(C_INT) :: fill_in
      integer(C_INT) :: switch_mode
   end type ma48_control

   type, bind(C) :: ma48_ainfo
      real(wp) :: ops
      integer(C_INT) :: flag
      integer(C_INT) :: more
      integer(C_LONG) :: lena_analyse
      integer(C_LONG) :: lenj_analyse
      integer(C_LONG) :: lena_factorize
      integer(C_LONG) :: leni_factorize
      integer(C_INT) :: ncmpa
      integer(C_INT) :: rank
      integer(C_LONG) :: drop
      integer(C_INT) :: struc_rank
      integer(C_LONG) :: oor
      integer(C_LONG) :: dup
      integer(C_INT) :: stat
      integer(C_INT) :: lblock
      integer(C_INT) :: sblock
      integer(C_LONG) :: tblock
   end type ma48_ainfo

   type, bind(C) :: ma48_finfo
      real(wp) :: ops
      integer(C_INT) :: flag
      integer(C_INT) :: more
      integer(C_LONG) :: size_factor
      integer(C_LONG) :: lena_factorize
      integer(C_LONG) :: leni_factorize
      integer(C_LONG) :: drop
      integer(C_INT) :: rank
      integer(C_INT) :: stat
   end type ma48_finfo

   type, bind(C) :: ma48_sinfo
      integer(C_INT) :: flag
      integer(C_INT) :: more
      integer(C_INT) :: stat
   end type ma48_sinfo
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma48_control), intent(in) :: ccontrol
      type(f_ma48_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

      f_arrays = (ccontrol%f_arrays.ne.0)
      fcontrol%multiplier = ccontrol%multiplier
      fcontrol%u = ccontrol%u
      fcontrol%switch = ccontrol%switch
      fcontrol%drop = ccontrol%drop
      fcontrol%tolerance = ccontrol%tolerance
      fcontrol%cgce = ccontrol%cgce
      fcontrol%lp = ccontrol%lp
      fcontrol%wp = ccontrol%wp
      fcontrol%mp = ccontrol%mp
      fcontrol%ldiag = ccontrol%ldiag
      fcontrol%btf = ccontrol%btf
      fcontrol%struct = (ccontrol%struct.ne.0)
      fcontrol%maxit = ccontrol%maxit
      fcontrol%factor_blocking = ccontrol%factor_blocking
      fcontrol%solve_blas = ccontrol%solve_blas
      fcontrol%pivoting = ccontrol%pivoting
      fcontrol%diagonal_pivoting = (ccontrol%diagonal_pivoting.ne.0)
      fcontrol%fill_in = ccontrol%fill_in
      fcontrol%switch_mode = (ccontrol%switch_mode.ne.0)
   end subroutine copy_control_in

   subroutine copy_ainfo_out(fainfo, cainfo)
      type(f_ma48_ainfo), intent(in) :: fainfo
      type(ma48_ainfo), intent(out) :: cainfo

      cainfo%ops = fainfo%ops
      cainfo%flag = fainfo%flag
      cainfo%more = fainfo%more
      cainfo%lena_analyse = fainfo%lena_analyse
      cainfo%lenj_analyse = fainfo%lenj_analyse
      cainfo%lena_factorize = fainfo%lena_factorize
      cainfo%leni_factorize = fainfo%leni_factorize
      cainfo%ncmpa = fainfo%ncmpa
      cainfo%rank = fainfo%rank
      cainfo%drop = fainfo%drop
      cainfo%struc_rank = fainfo%struc_rank
      cainfo%oor = fainfo%oor
      cainfo%dup = fainfo%dup
      cainfo%stat = fainfo%stat
      cainfo%lblock = fainfo%lblock
      cainfo%sblock = fainfo%sblock
      cainfo%tblock = fainfo%tblock
   end subroutine copy_ainfo_out

   subroutine copy_finfo_out(ffinfo, cfinfo)
      type(f_ma48_finfo), intent(in) :: ffinfo
      type(ma48_finfo), intent(out) :: cfinfo

      cfinfo%ops = ffinfo%ops
      cfinfo%flag = ffinfo%flag
      cfinfo%more = ffinfo%more
      cfinfo%size_factor = ffinfo%size_factor
      cfinfo%lena_factorize = ffinfo%lena_factorize
      cfinfo%leni_factorize = ffinfo%leni_factorize
      cfinfo%drop = ffinfo%drop
      cfinfo%rank = ffinfo%rank
      cfinfo%stat = ffinfo%stat
   end subroutine copy_finfo_out

   subroutine copy_sinfo_out(fsinfo, csinfo)
      type(f_ma48_sinfo), intent(in) :: fsinfo
      type(ma48_sinfo), intent(out) :: csinfo

      csinfo%flag = fsinfo%flag
      csinfo%more = fsinfo%more
      csinfo%stat = fsinfo%stat
   end subroutine copy_sinfo_out

end module hsl_ma48_double_ciface

