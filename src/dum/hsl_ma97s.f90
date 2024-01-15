! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 10:15 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 9 7    M O D U L E  -*-*-*-*-*-

module hsl_MA97_single

   USE GALAHAD_KINDS
   USE GALAHAD_SYMBOLS

   implicit none
   public :: ma97_get_n__, ma97_get_nz__

! Parameters (all private)
  real(sp_), parameter, private :: one = 1.0_sp_
  real(sp_), parameter, private :: zero = 0.0_sp_
  integer(ip_), parameter, private :: nemin_default = 8

  interface MA97_analyse
      module procedure analyse_single
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
      module procedure MA97_solve_mult_single
      module procedure MA97_solve_one_single
  end interface

  interface ma97_solve_fredholm
      module procedure MA97_solve_fredholm_single
   end interface ma97_solve_fredholm

   interface ma97_free
      module procedure free_akeep_single
      module procedure free_fkeep_single
   end interface ma97_free

   interface ma97_finalise
      module procedure finalise_both_single
   end interface ma97_finalise

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

  interface ma97_get_n__
     module procedure ma97_get_n_single
  end interface ma97_get_n__

  interface ma97_get_nz__
     module procedure ma97_get_nz_single
  end interface ma97_get_nz__

  type MA97_control ! The scalar control of this type controls the action
    logical :: action = .true. ! pos_def = .false. only.
    real(sp_) :: consist_tol = epsilon(one)
    integer(long_) :: factor_min = 20000000
    integer(ip_) :: nemin = nemin_default
    real(sp_) :: multiplier = 1.1
    integer(ip_) :: ordering = 5 ! controls choice of ordering
    integer(ip_) :: print_level = 0 ! Controls diagnostic printing
    integer(ip_) :: scaling = 0 ! controls use of scaling.
    real(sp_) :: small = tiny(one) ! Minimum pivot size
    logical :: solve_blas3 = .false. ! Use sgemm rather than sgemv in solve
    integer(long_) :: solve_min = 100000 ! Minimum value of info%num_factor
    logical :: solve_mf = .false. ! Do we use s/n (false) or m/f (true) solve?
    real(sp_) :: u = 0.01 ! Initial relative pivot threshold
    integer(ip_) :: unit_diagnostics = 6 ! unit for diagnostic printing.
    integer(ip_) :: unit_error = 6 ! unit number for error messages
    integer(ip_) :: unit_warning = 6 ! unit number for warning messages
    integer(long_) :: min_subtree_work = 1e5 ! Minimum amount of work
    integer(ip_) :: min_ldsrk_work = 1e4 ! Minimum amount of work to aim
  end type MA97_control

  type MA97_info ! The scalar info of this type returns information to user.
    integer(ip_) :: flag = 0     ! Value zero after successful entry.
    integer(ip_) :: flag68 = 0 ! error flag from hsl_mc68
    integer(ip_) :: flag77 = 0 ! error flag from mc77
    integer(ip_) :: matrix_dup = 0 ! Number of duplicated entries.
    integer(ip_) :: matrix_rank = 0 ! Rank of factorized matrix.
    integer(ip_) :: matrix_outrange = 0 ! Number of out-of-range entries.
    integer(ip_) :: matrix_missing_diag = 0 ! Number of missing diag. entries
    integer(ip_) :: maxdepth = 0 ! Maximum depth of the tree.
    integer(ip_) :: maxfront = 0 ! Maximum front size.
    integer(long_)  :: num_factor = 0_long_ ! Number of entries in the factor.
    integer(long_)  :: num_flops = 0_long_ ! Number of flops to calculate L.
    integer(ip_) :: num_delay = 0 ! Number of delayed eliminations.
    integer(ip_) :: num_neg = 0 ! Number of negative eigenvalues.
    integer(ip_) :: num_sup = 0 ! Number of supervariables.
    integer(ip_) :: num_two = 0 ! Number of 2x2 pivots.
    integer(ip_) :: ordering = 0 ! ordering actually used
    integer(ip_) :: stat = 0 ! STAT value (when available).
   end type MA97_info

  type MA97_akeep ! proper version has many componets!
    integer(ip_) :: n
    integer(ip_) :: ne
  end type MA97_akeep

  type MA97_fkeep ! proper version has many componets!
    integer(ip_) :: n
  end type MA97_fkeep

contains

  subroutine MA97_analyse_single(check, n, ptr, row, akeep,                    &
                                 control, info, order)
   logical, intent(in) :: check
   integer(ip_),  intent(in) :: n
   integer(ip_),  intent(in) :: row(:), ptr(:)
   type (MA97_akeep), intent (out) :: akeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (out) :: info
   integer(ip_), OPTIONAL, intent (inout) :: order(:)

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

  subroutine analyse_single(check, n, ptr, row, akeep,                         &
                                 control, info, order, val)
   logical, intent(in) :: check
   integer(ip_),  intent(in) :: n
   integer(ip_),  intent(in) :: row(:), ptr(:)
   type (MA97_akeep), intent (out) :: akeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (out) :: info
   integer(ip_), OPTIONAL, intent (inout) :: order(:)
   real(sp_), optional, intent(in) :: val(:)

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

  end subroutine analyse_single

  subroutine MA97_analyse_coord_single( n, ne, row, col, akeep,                &
                                        control, info, order)
   integer(ip_),  intent(in) :: n, ne
   integer(ip_),  intent(in) :: row(:), col(:)
   type (MA97_akeep), intent (out) :: akeep
   type (MA97_control), intent(in) :: control
   type (MA97_info), intent(out) :: info
   integer(ip_), OPTIONAL, intent (inout) :: order(:)

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
   integer(ip_),  intent(in) :: matrix_type
   real(sp_), intent(in) :: val(*)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(sp_), intent(inout), optional :: scale(:)
   integer(ip_), intent(in), optional :: ptr(akeep%n+1)
   integer(ip_), intent(in), optional :: row(*)

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
   integer(ip_),  intent(in) :: matrix_type
   real(sp_), intent(in) :: val(*)
   integer(ip_) :: lx, nrhs
   real(sp_), intent(inout) :: x(lx,nrhs)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(sp_), intent(inout), optional :: scale(:)
   integer(ip_), intent(in), optional :: ptr(akeep%n+1)
   integer(ip_), intent(in), optional :: row(*)

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
   integer(ip_),  intent(in) :: matrix_type
   real(sp_), intent(in) :: val(*)
   real(sp_), intent(inout) :: x1(:)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent(out) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(sp_), intent(inout), optional :: scale(:)
   integer(ip_), intent(in), optional :: ptr(akeep%n+1)
   integer(ip_), intent(in), optional :: row(*)

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
   integer(ip_), intent (in) :: nrhs, lx
   real(sp_), intent (inout) :: x(lx,nrhs)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (in) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(sp_), intent(in), optional :: scale(:)
   integer(ip_), optional, intent (in) :: job

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

  subroutine ma97_solve_mult_single(nrhs,x,ldx,akeep,fkeep,control,info,job)
   integer(ip_), intent(in) :: nrhs
   integer(ip_), intent(in) :: ldx
   real(sp_), dimension(ldx,nrhs), intent(inout) :: x
   type(ma97_akeep), intent(in) :: akeep
   type(ma97_fkeep), intent(in) :: fkeep
   type(ma97_control), intent(in) :: control
   type(ma97_info), intent(out) :: info
   integer(ip_), optional, intent(in) :: job
  end subroutine ma97_solve_mult_single

  subroutine MA97_solve_one_single(x,akeep,fkeep,control,info,scale,job)
   real(sp_), intent (inout) :: x(:)
   type (MA97_akeep), intent (in) :: akeep
   type (MA97_fkeep), intent (in) :: fkeep
   type (MA97_control), intent (in) :: control
   type (MA97_info), intent (inout) :: info
   real(sp_), intent(in), optional :: scale(:)
   integer(ip_), optional, intent (in) :: job

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
   integer(ip_),  intent(in) :: nrhs
   logical, intent(out) :: flag_out(nrhs)
   integer(ip_),  intent(in) :: ldx
   real(sp_), dimension(ldx,2*nrhs), intent(inout) :: x
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
     real(sp_), dimension(:), intent(in) :: x1
     real(sp_), dimension(:), intent(out) :: y1
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
     integer(ip_),  intent(in) :: k
     integer(ip_),  intent(in) :: ldx
     real(sp_), dimension(ldx,k), intent(in) :: x
     integer(ip_),  intent(in) :: ldy
     real(sp_), dimension(ldy,k), intent(out) :: y
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
    real(sp_), dimension( : ), intent(out) :: d
    type (MA97_akeep), intent (in) :: akeep
    type (MA97_fkeep), intent(in) :: fkeep
    type (MA97_control), intent (inout) :: control
    type (MA97_info), intent (inout) :: info

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
    integer(ip_), optional, intent(out) :: piv_order(:)
    real(sp_), optional, intent(out) :: d(:,:)
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
    real(sp_), intent (in) :: d(:,:)
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

  subroutine finalise_both_single(akeep, fkeep)
     type(ma97_akeep), intent(inout) :: akeep
     type(ma97_fkeep), intent(inout) :: fkeep
  end subroutine finalise_both_single

  subroutine ma97_sparse_fwd_solve_single(nbi, bindex, b, order, lflag,        &
      nxi, xindex, x, akeep, fkeep, control, info)
   integer(ip_),  intent(in) :: nbi
   integer(ip_),  intent(in) :: bindex(:)
   real(sp_), intent(in) :: b(:)
   integer(ip_),  intent(in) :: order(:)
   logical, intent(inout), dimension(:) :: lflag
   integer(ip_),  intent(out) :: nxi
   integer(ip_),  intent(out) :: xindex(:)
   real(sp_), intent(inout) :: x(:)
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

  subroutine MA97_finalise_both_single(akeep,fkeep)
    type (MA97_akeep), intent (inout) :: akeep
    type (MA97_fkeep), intent (inout) :: fkeep
  end subroutine MA97_finalise_both_single

pure integer function ma97_get_n_single(akeep)
   type(ma97_akeep), intent(in) :: akeep

   ma97_get_n_single = akeep%n
end function ma97_get_n_single

pure integer function ma97_get_nz_single(akeep)
   type(ma97_akeep), intent(in) :: akeep

   ma97_get_nz_single = akeep%ne
end function ma97_get_nz_single

end module hsl_MA97_single
