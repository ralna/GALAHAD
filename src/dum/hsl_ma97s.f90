! THIS VERSION: GALAHAD 4.0 - 2022-01-07 AT 13:00 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 9 7    M O D U L E  -*-*-*-*-*-

module hsl_MA97_single
    
   USE GALAHAD_SYMBOLS

   implicit none
   public :: ma97_get_n__, ma97_get_nz__

! Parameters (all private)
  integer, parameter, private  :: short = kind(0)
  integer(short), parameter, private  :: wp = kind(0.0)
  integer(short), parameter, private  :: long = selected_int_kind(18)
  real(wp), parameter, private :: one = 1.0
  real(wp), parameter, private :: zero = 0.0
  integer(short), parameter, private :: nemin_default = 8

  interface MA97_analyse
      module procedure MA97_analyse_single
  end interface

  interface MA97_analyse_coord
      module procedure MA97_analyse_coord_single
  end interface

  interface MA97_factor
      module procedure MA97_factor_single
  end interface

  interface MA97_factor_solve
      module procedure MA97_factor_solve_single
      module procedure MA97_factor_solve_one_single
  end interface

  interface MA97_solve
      module procedure MA97_solve_single
      module procedure MA97_solve_one_single
  end interface

  interface ma97_solve_fredholm
      module procedure MA97_solve_fredholm_single
   end interface ma97_solve_fredholm

   interface ma97_free
      module procedure free_akeep_single
      module procedure free_fkeep_single
   end interface ma97_free

  interface MA97_enquire_posdef
    module procedure MA97_enquire_posdef_single
  end interface

  interface MA97_enquire_indef
    module procedure MA97_enquire_indef_single
  end interface

  interface MA97_alter
    module procedure MA97_alter_single
  end interface

  interface ma97_lmultiply
     module procedure ma97_lmultiply_one_single
     module procedure ma97_lmultiply_mult_single
  end interface ma97_lmultiply

  interface ma97_sparse_fwd_solve
      module procedure ma97_sparse_fwd_solve_single
   end interface ma97_sparse_fwd_solve

  interface MA97_finalise
      module procedure MA97_finalise_single
  end interface

  interface ma97_get_n__
     module procedure ma97_get_n_single
  end interface ma97_get_n__

  interface ma97_get_nz__
     module procedure ma97_get_nz_single
  end interface ma97_get_nz__

  type MA97_control ! The scalar control of this type controls the action
    logical :: action = .true. ! pos_def = .false. only.
    real(wp) :: consist_tol = epsilon(one) 
    integer(long) :: factor_min = 20000000
    integer(short) :: nemin = nemin_default
    real(wp) :: multiplier = 1.1 
    integer :: ordering = 5 ! controls choice of ordering
    integer(short) :: print_level = 0 ! Controls diagnostic printing
    integer :: scaling = 0 ! controls use of scaling. 
    real(wp) :: small = tiny(one) ! Minimum pivot size
    logical :: solve_blas3 = .false. ! Use sgemm rather than sgemv in solve
    integer(long) :: solve_min = 100000 ! Minimum value of info%num_factor
    logical :: solve_mf = .false. ! Do we use s/n (false) or m/f (true) solve?
    real(wp) :: u = 0.01 ! Initial relative pivot threshold
    integer(short) :: unit_diagnostics = 6 ! unit for diagnostic printing.
    integer(short) :: unit_error = 6 ! unit number for error messages
    integer(short) :: unit_warning = 6 ! unit number for warning messages
    integer(long) :: min_subtree_work = 1e5 ! Minimum amount of work
    integer :: min_ldsrk_work = 1e4 ! Minimum amount of work to aim
  end type MA97_control

  type MA97_info ! The scalar info of this type returns information to user.
    integer(short) :: flag = 0     ! Value zero after successful entry.
    integer(short) :: flag68 = 0 ! error flag from hsl_mc68
    integer(short) :: flag77 = 0 ! error flag from mc77
    integer(short) :: matrix_dup = 0 ! Number of duplicated entries.
    integer(short) :: matrix_rank = 0 ! Rank of factorized matrix.
    integer(short) :: matrix_outrange = 0 ! Number of out-of-range entries.
    integer(short) :: matrix_missing_diag = 0 ! Number of missing diag. entries
    integer(short) :: maxdepth = 0 ! Maximum depth of the tree.
    integer(short) :: maxfront = 0 ! Maximum front size.
    integer(long)  :: num_factor = 0_long ! Number of entries in the factor.
    integer(long)  :: num_flops = 0_long ! Number of flops to calculate L.
    integer(short) :: num_delay = 0 ! Number of delayed eliminations.
    integer(short) :: num_neg = 0 ! Number of negative eigenvalues.
    integer(short) :: num_sup = 0 ! Number of supervariables. 
    integer(short) :: num_two = 0 ! Number of 2x2 pivots.
    integer :: ordering = 0 ! ordering actually used
    integer(short) :: stat = 0 ! STAT value (when available).
   end type MA97_info

  type MA97_akeep ! proper version has many componets!
    integer :: n
    integer :: ne
  end type MA97_akeep

  type MA97_fkeep ! proper version has many componets!
    integer :: n
  end type MA97_fkeep

contains

  subroutine MA97_analyse_single(check, n, ptr, row, akeep,                   &
                                 control, info, order)
   logical, intent(in) :: check
   integer, intent(in) :: n
   integer, intent(in) :: row(:), ptr(:)
   type (MA97_akeep), intent (out) :: akeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (out) :: info
   integer(short), OPTIONAL, intent (inout) :: order(:)

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_analyse with its HSL namesake ', /,        &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_analyse_single

  subroutine MA97_analyse_coord_single( n, ne, row, col, akeep,                &
                                        control, info, order)
   integer, intent(in) :: n, ne
   integer, intent(in) :: row(:), col(:)
   type (MA97_akeep), intent (out) :: akeep
   type (MA97_control), intent(in) :: control
   type (MA97_info), intent(out) :: info
   integer(short), OPTIONAL, intent (inout) :: order(:)

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_analyse_coord with its HSL namesake ', /,  &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_analyse_coord_single

  subroutine MA97_factor_single(matrix_type,val,akeep,fkeep,control,info,      &
                                scale,ptr,row)
   integer, intent(in) :: matrix_type 
   real(wp), intent(in) :: val(*)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(wp), intent(inout), optional :: scale(:)
   integer(short), intent(in), optional :: ptr(akeep%n+1)
   integer(short), intent(in), optional :: row(*)

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_factorize with its HSL namesake ', /,      &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_factor_single

  subroutine MA97_factor_solve_single(matrix_type,val,nrhs,x,lx,akeep,fkeep,   &
                                      control,info,scale,ptr,row)
   integer, intent(in) :: matrix_type 
   real(wp), intent(in) :: val(*)
   integer(short) :: lx, nrhs
   real(wp), intent(inout) :: x(lx,nrhs)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(wp), intent(inout), optional :: scale(:) 
   integer(short), intent(in), optional :: ptr(akeep%n+1)
   integer(short), intent(in), optional :: row(*)

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_factor_solve with its HSL namesake ', /,   &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_factor_solve_single

  subroutine MA97_factor_solve_one_single(matrix_type,val,x1,akeep,fkeep,      &
                                          control,info,scale,ptr,row)
   integer(short), intent(in) :: matrix_type 
   real(wp), intent(in) :: val(*)
   real(wp), intent(inout) :: x1(:) 
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(wp), intent(inout), optional :: scale(:)
   integer(short), intent(in), optional :: ptr(akeep%n+1)
   integer(short), intent(in), optional :: row(*)

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_factor_solve_one with its HSL namesake', /,&
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_factor_solve_one_single

  subroutine MA97_solve_single(nrhs,x,lx,akeep,fkeep,control,info,scale,job)
   integer(short), intent (in) :: nrhs, lx
   real(wp), intent (inout) :: x(lx,nrhs)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (in) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(wp), intent(in), optional :: scale(:)
   integer(short), optional, intent (in) :: job

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_solve with its HSL namesake ', /,          &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_solve_single

  subroutine MA97_solve_one_single(x,akeep,fkeep,control,info,scale,job)
   real(wp), intent (inout) :: x(:)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (in) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(wp), intent(in), optional :: scale(:)
   integer(short), optional, intent (in) :: job

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_solve_one with its HSL namesake ', /,      &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_solve_one_single

  subroutine MA97_solve_fredholm_single( nrhs, flag_out, x, ldx,               &
                                         akeep, fkeep, control, info )
   integer, intent(in) :: nrhs
   logical, intent(out) :: flag_out(nrhs)
   integer, intent(in) :: ldx
   real(wp), dimension(ldx,2*nrhs), intent(inout) :: x
   type(ma97_akeep), intent(in) :: akeep
   type(ma97_fkeep), intent(in) :: fkeep
   type(ma97_control), intent(in) :: control
   type(ma97_info), intent(out) :: info

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_solve_fredhom with its HSL namesake',      &
  &          /, ' and dependencies. See ', /,                                  &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option
  end subroutine MA97_solve_fredholm_single

  subroutine ma97_lmultiply_one_single(trans, x1, y1, akeep, fkeep,            &
                                       control, info)
     logical, intent(in) :: trans
     real(wp), dimension(:), intent(in) :: x1
     real(wp), dimension(:), intent(out) :: y1
     type(ma97_akeep), intent(in) :: akeep
     type(ma97_fkeep), intent(in) :: fkeep
     type(ma97_control), intent(in) :: control
     type(ma97_info), intent(out) :: info

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_lmultiply_mult with its HSL namesake',     &
  &          /, ' and dependencies. See ', /,                                  &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option
  end subroutine ma97_lmultiply_one_single

  subroutine ma97_lmultiply_mult_single(trans, k, x, ldx, y, ldy,              &
                                        akeep, fkeep, control, info)
     logical, intent(in) :: trans
     integer, intent(in) :: k
     integer, intent(in) :: ldx
     real(wp), dimension(ldx,k), intent(in) :: x
     integer, intent(in) :: ldy
     real(wp), dimension(ldy,k), intent(out) :: y
     type(ma97_akeep), intent(in) :: akeep
     type(ma97_fkeep), intent(in) :: fkeep
     type(ma97_control), intent(in) :: control
     type(ma97_info), intent(out) :: info

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_lmultiply_one with its HSL namesake',      &
  &          /, ' and dependencies. See ', /,                                  &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option
  end subroutine ma97_lmultiply_mult_single

  subroutine MA97_enquire_posdef_single(akeep,fkeep,control,info,d)
    type (MA97_akeep), intent (in) :: akeep
    type (MA97_fkeep), intent(in) :: fkeep
    type (MA97_control), intent (inout) :: control
    type (MA97_info), intent (inout) :: info
    real(wp), dimension( : ), intent(out) :: d

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_enquire_posdef with its HSL namesake', /,  &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_enquire_posdef_single

  subroutine MA97_enquire_indef_single(akeep,fkeep,control,info,piv_order,d)
    integer(short), optional, intent(out) :: piv_order(:)
    real(wp), optional, intent(out) :: d(:,:)
    type (MA97_akeep), intent (in) :: akeep
    type (MA97_fkeep), intent (in) :: fkeep
    type (MA97_control), intent (inout) :: control
    type (MA97_info), intent (inout) :: info

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_enquire_indef with its HSL namesake ', /,  &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_enquire_indef_single

  subroutine MA97_alter_single(d,akeep,fkeep,control,info)
    real(wp), intent (in) :: d(:,:)
    type (MA97_akeep), intent (in) :: akeep
    type (MA97_fkeep), intent (in) :: fkeep
    type (MA97_control), intent (inout) :: control
    type (MA97_info), intent (inout) :: info

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_alter with its HSL namesake ', /,          &
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine MA97_alter_single

  subroutine ma97_sparse_fwd_solve_single(nbi, bindex, b, order, lflag,        &
      nxi, xindex, x, akeep, fkeep, control, info)
   integer, intent(in) :: nbi
   integer, intent(in) :: bindex(:)
   real(wp), intent(in) :: b(:)
   integer, intent(in) :: order(:)
   logical, intent(inout), dimension(:) :: lflag
   integer, intent(out) :: nxi
   integer, intent(out) :: xindex(:)
   real(wp), intent(inout) :: x(:)
   type(ma97_akeep), intent(in) :: akeep
   type(ma97_fkeep), intent(in) :: fkeep
   type(ma97_control), intent(in) :: control
   type(ma97_info), intent(out) :: info
   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have HSL (formerly the Harwell Subroutine', /,            &
  &         ' Library), this option may be enabled by replacing the ', /,      &
  &         ' dummy subroutine MA97_sparse_fwd_solve with its HSL namesake ',/,&
  &         ' and dependencies. See ', /,                                      &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   info%flag = GALAHAD_unavailable_option

  end subroutine ma97_sparse_fwd_solve_single

  subroutine free_akeep_single(akeep)
     type(ma97_akeep), intent(inout) :: akeep
  end subroutine free_akeep_single

  subroutine free_fkeep_single(fkeep)
     type(ma97_fkeep), intent(inout) :: fkeep
  end subroutine free_fkeep_single

  subroutine MA97_finalise_single(akeep,fkeep)
    type (MA97_akeep), intent (inout) :: akeep
    type (MA97_fkeep), intent (inout) :: fkeep
  end subroutine MA97_finalise_single

pure integer function ma97_get_n_single(akeep)
   type(ma97_akeep), intent(in) :: akeep

   ma97_get_n_single = akeep%n
end function ma97_get_n_single

pure integer function ma97_get_nz_single(akeep)
   type(ma97_akeep), intent(in) :: akeep

   ma97_get_nz_single = akeep%ne
end function ma97_get_nz_single

end module hsl_MA97_single
