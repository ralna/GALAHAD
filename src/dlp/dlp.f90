! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ D L P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.6. January 30th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DLP_double

!      ------------------------------------------------------
!     |                                                      |
!     | Minimize the linear objective function               |
!     |                                                      |
!     |                    g^T x + f                         |
!     |                                                      |
!     | subject to the linear constraints and bounds         |
!     |                                                      |
!     |                 c_l <= A x <= c_u                    |
!     |                 x_l <=  x <= x_u                     |
!     |                                                      |
!     | using a dual gradient-projection method              |
!     |                                                      |
!     | Optionally, minimize instead the penalty function    |
!     |                                                      |
!     |   g^T x + rho || min( A x - c_l, c_u - A x, 0 )||_1  |
!     |                                                      |
!     | subject to the bound constraints x_l <= x <= x_u     |
!     |                                                      |
!      ------------------------------------------------------

!$    USE omp_lib
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural,           &
                                       STRING_ies, STRING_are, STRING_ordinal, &
                                       STRING_their, STRING_integer_6
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, DLP_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, DLP_data_type => QPD_data_type,                  &
                              DLP_AX => QPD_AX, DLP_HX => QPD_HX,              &
                              DLP_abs_AX => QPD_abs_AX, DLP_abs_HX => QPD_abs_HX
      USE GALAHAD_SORT_double, ONLY: SORT_inverse_permute
      USE GALAHAD_FDC_double
      USE GALAHAD_SLS_double
      USE GALAHAD_SCU_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_NORMS_double, ONLY: TWO_norm
      USE GALAHAD_DQP_double, DLP_control_type => DQP_control_type,            &
                              DLP_time_type => DQP_time_type,                  &
                              DLP_inform_type => DQP_inform_type
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DLP_initialize, DLP_read_specfile, DLP_solve,                  &
                DLP_next_perturbation, DLP_terminate, QPT_problem_type,        &
                SMT_type, SMT_put, SMT_get, DLP_Ax, DLP_data_type,             &
                DLP_dims_type, DLP_control_type, DLP_time_type, DLP_inform_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: relative_pivot_default = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: hzero = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: infeas = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: big_radius = ten ** 10
      REAL ( KIND = wp ), PARAMETER :: alpha_search = one
      REAL ( KIND = wp ), PARAMETER :: beta_search = half
      REAL ( KIND = wp ), PARAMETER :: mu_search = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: obj_unbounded = - epsmch ** ( - 2 )

    CONTAINS

!-*-*-*-*-*-   D L P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE DLP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for DLP. This routine should be called before
!  DLP_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble for DQP
!  inform   a structure containing output information. See preamble for DQP
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( DLP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DLP_control_type ), INTENT( OUT ) :: control
      TYPE ( DLP_inform_type ), INTENT( OUT ) :: inform

      CALL DQP_initialize( data, control, inform )
      RETURN

!  End of DLP_initialize

      END SUBROUTINE DLP_initialize

!-*-*-*-*-   D L P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE DLP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by DLP_initialize could (roughly)
!  have been set as:

! BEGIN DLP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  iterations-between-printing                       1
!  maximum-number-of-iterations                      1000
!  maximum-dimension-of-schur-complement             75
!  cauchy-only-until-change-level                    -1
!  maximum-number-of-steps-per-arc-search            -1
!  maximum-number-of-cg-iterations-per-iteration     1000
!  restore-problem-on-output                         2
!  dual-starting-point                               0
!  sif-file-device                                   52
!  penalty-weight                                    0.0D+0
!  infinity-value                                    1.0D+19
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-5
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-5
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-5
!  initial-perturbation                              0.1
!  perturbation-reduction-factor                     0.1
!  final-perturbation                                1.0D-6
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  cg-zero-curvature-threshold                       1.0D-15
!  identical-bounds-tolerance                        1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  direct-solution-of-subspace-problem               F
!  perform-exact-arc-search                          F
!  perform-subspace-arc-search                       T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  sils
!  definite-linear-equation-solver                   sils
!  unsymmetric-linear-equation-solver                gls
!  generate-sif-file                                 F
!  sif-file-name                                     DLPPROB.SIF
!  output-line-prefix                                ""
! END DLP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( DLP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'DLP'

      IF ( PRESENT( alt_specname ) ) THEN
        CALL DQP_read_specfile( control, device,                               &
                                alt_specname = TRIM( alt_specname ) )
      ELSE
        CALL DQP_read_specfile( control, device, main_specname = specname )
      END IF

      RETURN

      END SUBROUTINE DLP_read_specfile

!-*-*-*-*-*-*-*-*-*-   D L P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE DLP_solve( prob, data, control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear objective
!
!                      g^T x + f
!
!  where
!
!             (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a dual gradient-projection
!  method. The subroutine is particularly appropriate when A is sparse.
!
!  Optionally, minimize instead the penalty function
!
!    g^T x + f + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  subject to the bound constraints x_l <= x <= x_u
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to DLP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found. %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %A is a structure of type SMT_type used to hold the matrix A.
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %C_l, %C_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %C_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %C_u to a value larger than
!    control%infinity. On exit, %C_l and %C_u will most likely have been
!    reordered.
!
!   %Y is a REAL array of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers
!    corresponding to the general constraints c_l <= A x <= c_u.
!    On successful exit, it will contain the required vector of Lagrange
!    multipliers.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x_l <= x <= x_u). On successful exit, it will contain
!   the required vector of dual variables.
!
!  data is a structure of type DLP_data_type which holds private internal data
!
!  control is a structure of type DLP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to DLP_initialize. See the preamble
!   of DQP for details
!
!  inform is a structure of type DLP_inform_type that provides
!    information on exit from DLP_solve. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -8 The analytic center appears to be unbounded.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -11 The solve of a required linear system failed; the return status from
!       the factorization package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -17 The step is too small to make further impact.
!
!   -18 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!   -19 Too much time has passed. This may happen if control%cpu_time_limit or
!       control%clock_time_limit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!  On exit from DLP_solve, other components of inform are given in the
!   preamble for DQP
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set
!
!  X_stat is an optional  INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( DLP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DLP_control_type ), INTENT( IN ) :: control
      TYPE ( DLP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  Local variables

      INTEGER :: i, j, n_depen, nzc, nv, lbd, dual_starting_point
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: time_analyse, time_factorize
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: clock_analyse, clock_factorize
      REAL ( KIND = wp ) :: av_bnd, perturbation
!     REAL ( KIND = wp ) :: fixed_sum, xi
      REAL ( KIND = wp ), DIMENSION( 1 ) :: H_val
      LOGICAL :: composite_g, diagonal_h, identity_h, scaled_identity_h
      LOGICAL :: printi, remap_freed, reset_bnd, stat_required
      LOGICAL :: extrapolation_ok, initial
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( DLP_control_type ) :: dqp_control

!  functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering DLP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  initialize counts

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%factorization_status = 0
      inform%iter = - 1 ; inform%nfacts = - 1
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%obj = - one
      inform%non_negligible_pivot = zero
      inform%feasible = .FALSE.
!$    inform%threads = OMP_GET_MAX_THREADS( )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( X_stat )

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status
        GO TO 800
      END IF

!  if required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
        IF ( prob%gradient_kind == 0 ) THEN
          WRITE( control%out, "( ' G = zeros' )" )
        ELSE IF ( prob%gradient_kind == 1 ) THEN
          WRITE( control%out, "( ' G = ones' )" )
        ELSE
          WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" )                  &
            prob%G( : prob%n )
        END IF
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_l( : prob%n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_u( : prob%n )
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
        END IF
        WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_l( : prob%m )
        WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_u( : prob%m )
      END IF

!  check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        ELSE IF ( prob%X_u( i ) == prob%X_l( i )  ) THEN
        ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
          prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( /, A, '   **  Warning: one or more variable bounds reset ' )" )     &
         prefix

      reset_bnd = .FALSE.
      DO i = 1, prob%m
        IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        ELSE IF ( prob%C_u( i ) == prob%C_l( i ) ) THEN
        ELSE IF ( prob%C_u( i ) - prob%C_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%C_l( i ) + prob%C_u( i ) )
          prob%C_l( i ) = av_bnd ; prob%C_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( A, /, '   **  Warning: one or more constraint bounds reset ' )" )   &
          prefix

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

!  store the problem dimensions

      IF ( prob%new_problem_structure ) THEN
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF
        prob%Hessian_kind = 0
        data%h_ne = 0
      END IF

!  if the problem has no general constraints, so is a bound-constrained LP,
!  and can be solved explicitly

      IF ( data%a_ne <= 0 ) THEN
        IF ( printi ) WRITE( control%out,                                      &
          "( /, A, ' Solving bound-constrained LP -' )" ) prefix
        CALL QPD_solve_separable_BQP( prob, control%infinity,                  &
                                      obj_unbounded,  inform%obj,              &
                                      inform%feasible, inform%status,          &
                                      B_stat = X_stat( : prob%n ) )
        IF ( printi ) THEN
          CALL CLOCK_time( clock_now )
          WRITE( control%out,                                                  &
             "( A, ' On exit from QPD_solve_separable_BQP: status = ',         &
          &   I0, ', time = ', F0.2, /, A, ' objective value =', ES12.4 )",    &
            advance = 'no' ) prefix, inform%status, inform%time%clock_total    &
              + clock_now - clock_start, prefix, inform%obj
          IF ( PRESENT( X_stat ) ) THEN
            WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )  &
              COUNT( X_stat( : prob%n ) /= 0 ), prob%n
          ELSE
            WRITE( control%out, "( '' )" )
          END IF
        END IF
        inform%iter = 0 ; inform%non_negligible_pivot = zero
        inform%factorization_integer = 0 ; inform%factorization_real = 0

        IF ( printi ) then
          SELECT CASE( inform%status )
            CASE( GALAHAD_error_restrictions  ) ; WRITE( control%out,          &
              "( /, A, '  Warning - input paramters incorrect' )" ) prefix
            CASE( GALAHAD_error_primal_infeasible ) ; WRITE( control%out,      &
              "( /, A, '  Warning - the constraints appear to be',             &
             &   ' inconsistent' )" ) prefix
            CASE( GALAHAD_error_unbounded ) ; WRITE( control%out,              &
              "( /, A, '  Warning - problem appears to be unbounded from',     &
             & ' below' )") prefix
          END SELECT
        END IF
        IF ( inform%status /= GALAHAD_ok ) RETURN
        GO TO 800
      END IF

!  perform the preprocessing

      IF ( prob%new_problem_structure ) THEN
        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          GO TO 800
        END IF

!  record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform,                       &
                          prob, get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + REAL( time_now - time_record, wp )
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

          IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( A, ' status ', I0, ' after QPP_apply')" ) &
               prefix, data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) prefix, inform%status
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            GO TO 800
          END IF
        END IF
        data%trans = data%trans + 1
      END IF

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      time_analyse = inform%FDC_inform%time%analyse
      clock_analyse = inform%FDC_inform%time%clock_analyse
      time_factorize = inform%FDC_inform%time%factorize
      clock_factorize = inform%FDC_inform%time%clock_factorize

      IF ( prob%m > 0 .AND.                                                    &
           ( .NOT. data%tried_to_remove_deps .AND.                             &
              control%remove_dependencies ) ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
           "( /, A, 1X, I0, ' equalit', A, ' from ', I0, ' constraint', A )" ) &
              prefix, data%dims%c_equality,                                    &
              TRIM( STRING_ies( data%dims%c_equality ) ),                      &
              prob%m, TRIM( STRING_pleural( prob%m ) )

!  set control parameters

        data%FDC_control = control%FDC_control
        data%FDC_control%max_infeas = control%stop_abs_p

!  find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL FDC_find_dependent( prob%n, data%dims%c_equality,                 &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%dims%c_equality + 1 ),     &
                                 prob%C_l, n_depen, data%Index_C_freed,        &
                                 data%FDC_data, data%FDC_control,              &
                                 inform%FDC_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%find_dependent =                                           &
          inform%time%find_dependent + REAL( time_now - time_record, wp )
        inform%time%clock_find_dependent =                                     &
          inform%time%clock_find_dependent + clock_now - clock_record

!  record output parameters

        inform%status = inform%FDC_inform%status
        inform%non_negligible_pivot = inform%FDC_inform%non_negligible_pivot
        inform%alloc_status = inform%FDC_inform%alloc_status
        inform%factorization_status = inform%FDC_inform%factorization_status
        inform%factorization_integer = inform%FDC_inform%factorization_integer
        inform%factorization_real = inform%FDC_inform%factorization_real
        inform%bad_alloc = inform%FDC_inform%bad_alloc
        inform%nfacts = 1

        IF ( ( control%cpu_time_limit >= zero .AND.                            &
             REAL( time_now - time_start, wp ) > control%cpu_time_limit ) .OR. &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        END IF

        IF ( printi .AND. inform%non_negligible_pivot < thousand *             &
          control%FDC_control%SLS_control%absolute_pivot_tolerance )           &
            WRITE( control%out, "(                                             &
       &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,                 &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, A, ' ***  perhaps increase',                                      &
       &     ' FDC_control%SLS_control%absolute_pivot_tolerance from',         &
       &    ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )   &
           prefix, prefix, inform%non_negligible_pivot, prefix,                &
           control%FDC_control%SLS_control%absolute_pivot_tolerance, prefix

!  check for error exits

        IF ( inform%status /= 0 ) THEN

!  print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%error, "( ' ' )" )
            IF ( inform%status /= GALAHAD_ok ) WRITE( control%error,           &
                 "( A, '    ** Error return ', I0, ' from ', A )" )            &
               prefix, inform%status, 'FDC_dependent'
          END IF
          GO TO 700
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, A, ' The following ', I0, ' constraint',    &
         &  A, ' appear', A, ' to be dependent', /, ( 4X, 8I8 ) )" )           &
              prefix, n_depen, TRIM( STRING_pleural( n_depen ) ),              &
              TRIM( STRING_verb_pleural( n_depen ) ), data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y( : prob%m ) = zero
          prob%Z( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL DLP_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
                       prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
        inform%nfacts = 0
      END IF

      IF ( remap_freed ) THEN

!  some of the current constraints will be removed by freeing them

        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ', A,   &
         & ' dependent and will be temporarily removed' )" ) prefix, n_depen,  &
           TRIM( STRING_pleural( n_depen ) ), TRIM( STRING_are( n_depen ) )

!  allocate arrays to indicate which constraints have been freed

          array_name = 'DLP: data%C_freed'
          CALL SPACE_resize_array( n_depen, data%C_freed,                      &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  store the problem dimensions

        data%dims_save_freed = data%dims
        data%a_ne = prob%A%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before removal of dependecies: ', &
              &   /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

        data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1
        data%dims%x_s = 1 ; data%dims%x_e = prob%n
        data%dims%c_s = data%dims%x_e + 1
        data%dims%c_e = data%dims%x_e + data%dims%nc
        data%dims%c_b = data%dims%c_e - prob%m
        data%dims%y_s = data%dims%c_e + 1
        data%dims%y_e = data%dims%c_e + prob%m
        data%dims%y_i = data%dims%c_s + prob%m
        data%dims%v_e = data%dims%y_e

!  test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,            &
                              data%QPP_inform )
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          GO TO 800
        END IF

!  record revised array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions after removal of dependencies: ', &
             &    /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne
      END IF

!  compute the dimension of the KKT system

      data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1

!  arrays containing data relating to the composite vector ( x  c  y )
!  are partitioned as follows:

!   <---------- n --------->  <---- nc ------>  <-------- m --------->
!                             <-------- m --------->
!                        <-------- m --------->
!   -------------------------------------------------------------------
!   |                   |    |                 |    |                 |
!   |         x              |       c         |          y           |
!   |                   |    |                 |    |                 |
!   -------------------------------------------------------------------
!    ^                 ^    ^ ^               ^ ^    ^               ^
!    |                 |    | |               | |    |               |
!   x_s                |    |c_s              |y_s  y_i             y_e = v_e
!                      |    |                 |
!                     c_b  x_e               c_e

      data%dims%x_s = 1 ; data%dims%x_e = prob%n
      data%dims%c_s = data%dims%x_e + 1
      data%dims%c_e = data%dims%x_e + data%dims%nc
      data%dims%c_b = data%dims%c_e - prob%m
      data%dims%y_s = data%dims%c_e + 1
      data%dims%y_e = data%dims%c_e + prob%m
      data%dims%y_i = data%dims%c_s + prob%m
      data%dims%v_e = data%dims%y_e

!  ----------------
!  set up workspace
!  ----------------

      composite_g = prob%gradient_kind == 0 .OR. prob%gradient_kind == 1

      diagonal_h = .TRUE. ; identity_h = .FALSE. ; scaled_identity_h = .TRUE.

      CALL DQP_workspace( prob%m, prob%n, data%dims, prob%A, prob%H,           &
                          composite_g, diagonal_h,                             &
                          identity_h, scaled_identity_h, nv, lbd,              &
                          data%C_status, data%NZ_p, data%IUSED, data%INDEX_r,  &
                          data%INDEX_w, data%X_status, data%V_status,          &
                          data%X_status_old, data%C_status_old, data%C_active, &
                          data%X_active, data%CHANGES, data%ACTIVE_list,       &
                          data%ACTIVE_status, data%SOL, data%RHS, data%RES,    &
                          data%H_s, data%Y_l, data%Y_u, data%Z_l, data%Z_u,    &
                          data%VECTOR, data%BREAK_points, data%YC_l,           &
                          data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,          &
                          data%GY_u, data%GZ_l, data%GZ_u, data%V0, data%VT,   &
                          data%GV, data%G, data%PV, data%HPV,                  &
                          data%DV, data%V_bnd,                                 &
                          data%H_sbls, data%A_sbls, data%SCU_mat,              &
                          control, inform )

!  ===================================
!  Solve the problem as a perturbed QP
!  ===================================

      H_val( 1 ) = control%initial_perturbation
      dual_starting_point = control%dual_starting_point
      initial = .TRUE.
      dqp_control = control
!     dqp_control%rho = one
      dqp_control%factor_optimal_matrix = .TRUE.

!  loop over a sequence of decreasing perturbations until optimal

      DO
        IF ( printi ) WRITE( control%out, "( /,  A, 2X, 25( '-' ),             &
       &  ' perturbation = ', ES7.1, 1X, 25( '-' ) )" ) prefix, H_val( 1 )
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix, dqp_control, inform,    &
                               - 1, prob%gradient_kind,                        &
                               nv, lbd, data%m_ref, dual_starting_point,       &
                               data%clock_total, data%cpu_total,               &
                               data%SBLS_data, data%SLS_data,                  &
                               data%SCU_data, data%GLTR_data,                  &
                               data%SLS_control, data% SBLS_control,           &
                               data%GLTR_control, data%C_status,               &
                               data%NZ_p, data%IUSED, data%INDEX_r,            &
                               data%INDEX_w, data%X_status, data%V_status,     &
                               data%X_status_old, data%C_status_old,           &
                               data%refactor, data%m_active, data%n_active,    &
                               data%C_active, data%X_active, data%CHANGES,     &
                               data%ACTIVE_list, data%ACTIVE_status,           &
                               data%SOL, data%RHS, data%RES, data%H_s,         &
                               data%Y_l, data%Y_u, data%Z_l, data%Z_u,         &
                               data%VECTOR, data%BREAK_points, data%YC_l,      &
                               data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,     &
                               data%GY_u, data%GZ_l, data%GZ_u, data%V0,       &
                               data%VT, data%GV, data%G, data%PV,              &
                               data%HPV, data%DV,                              &
                               data%V_bnd, data%H_sbls, data%A_sbls,           &
                               data%C_sbls, data%SCU_mat, H_val = H_val,       &
                               C_stat = C_stat, X_stat = X_stat,               &
                               initial = initial )
        ELSE
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix, dqp_control, inform,    &
                               - 1, prob%gradient_kind,                        &
                               nv, lbd, data%m_ref, dual_starting_point,       &
                               data%clock_total, data%cpu_total,               &
                               data%SBLS_data, data%SLS_data,                  &
                               data%SCU_data, data%GLTR_data,                  &
                               data%SLS_control, data% SBLS_control,           &
                               data%GLTR_control, data%C_status,               &
                               data%NZ_p, data%IUSED, data%INDEX_r,            &
                               data%INDEX_w, data%X_status, data%V_status,     &
                               data%X_status_old, data%C_status_old,           &
                               data%refactor, data%m_active, data%n_active,    &
                               data%C_active, data%X_active, data%CHANGES,     &
                               data%ACTIVE_list, data%ACTIVE_status,           &
                               data%SOL, data%RHS, data%RES, data%H_s,         &
                               data%Y_l, data%Y_u, data%Z_l, data%Z_u,         &
                               data%VECTOR, data%BREAK_points, data%YC_l,      &
                               data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,     &
                               data%GY_u, data%GZ_l, data%GZ_u, data%V0,       &
                               data%VT, data%GV, data%G, data%PV,              &
                               data%HPV, data%DV,                              &
                               data%V_bnd, data%H_sbls, data%A_sbls,           &
                               data%C_sbls, data%SCU_mat,                      &
                               H_val = H_val, G = prob%G,                      &
                               C_stat = C_stat, X_stat = X_stat,               &
                               initial = initial )
        END IF
!write(6,"( A, ( 5ES12.4 ) )" ) ' X ', prob%X
!write(6,"( A, ( 10I4 ) )" ) ' X_stat ', X_stat
!write(6,"( A, ( 10I4 ) )" ) ' C_stat ', C_stat
        IF ( inform%status /= GALAHAD_ok ) EXIT

!  reduce the perturbation, and prepare to warm start if the value is above
!  its lower limit

        perturbation = H_val( 1 )
        CALL DLP_next_perturbation( prob%n, prob%m, data%dims, prob%A%val,     &
                                    prob%A%col, prob%A%ptr, prob%C_l,          &
                                    prob%C_u, prob%X_l, prob%X_u, prob%X,      &
                                    prob%Y, prob%Z, prob%C, prob%f, prefix,    &
                                    dqp_control, inform,                       &
                                    extrapolation_ok, perturbation,            &
                                    prob%gradient_kind, nv, data%m_ref,        &
                                    data%SBLS_data, data%SCU_data,             &
                                    data%GLTR_data, data% SBLS_control,        &
                                    data%GLTR_control, data%refactor,          &
                                    data%m_active, data%n_active,              &
                                    data%C_active, data%X_active,              &
                                    data%ACTIVE_status, data%SOL, data%RHS,    &
                                    data%RES, data%Y_l, data%Y_u, data%Z_l,    &
                                    data%Z_u, data%VECTOR, data%VT, data%GV,   &
                                    data%PV, data%DV, data%A_sbls,             &
                                    data%C_sbls, data%SCU_mat,                 &
                                    G = prob%G, C_stat = C_stat,               &
                                    X_stat = X_stat )

!  exit if the extrapolation leads to the optimal solution

        IF ( extrapolation_ok ) THEN
          IF ( printi ) WRITE( control%out,                                    &
               "( /, A, ' Extrapolated solution is optimal' )" ) prefix
          EXIT

!  the extrapolation is infeasible, so reduce the perturbation so the new
!  perturbed problem lies on a different face of the dual fesaible set

        ELSE
          H_val( 1 ) = MIN( H_val( 1 ) * control%perturbation_reduction,       &
                            0.99_wp * perturbation )
!         H_val( 1 ) = perturbation * control%perturbation_reduction

!  don't allow the perturbation to be too small

          IF ( H_val( 1 ) <= control%final_perturbation ) EXIT
          dual_starting_point = 0
          initial = .FALSE.
        END IF

      END DO

!  record the times taken

      inform%time%analyse = inform%time%analyse +                              &
        inform%FDC_inform%time%analyse - time_analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%FDC_inform%time%clock_analyse - clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%FDC_inform%time%factorize - time_factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%FDC_inform%time%clock_factorize - clock_factorize

!  if some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( stat_required ) THEN
          C_stat( prob%m + 1 : data%QPP_map_freed%m ) = 0
          CALL SORT_inverse_permute( data%QPP_map_freed%m,                     &
                                     data%QPP_map_freed%c_map,                 &
                                     IX = C_stat( : data%QPP_map_freed%m ) )
          X_stat( prob%n + 1 : data%QPP_map_freed%n ) = - 1
          CALL SORT_inverse_permute( data%QPP_map_freed%n,                     &
                                     data%QPP_map_freed%x_map,                 &
                                     IX = X_stat( : data%QPP_map_freed%n ) )
        END IF
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform, prob,           &
                          get_all = .TRUE.)
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO
      END IF
      data%tried_to_remove_deps = .FALSE.

!  retore the problem to its original form

  700 CONTINUE
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( stat_required ) THEN
          C_stat( prob%m + 1 : data%QPP_map%m ) = 0
          CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,       &
                                     IX = C_stat( : data%QPP_map%m ) )
          X_stat( prob%n + 1 : data%QPP_map%n ) = - 1
          CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,       &
                                     IX = X_stat( : data%QPP_map%n ) )
        END IF

!  full restore

        IF ( control%restore_problem >= 2 .OR. stat_required ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )

!  restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  recover solution

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_x = .TRUE., get_y = .TRUE.,                    &
                            get_z = .TRUE., get_c = .TRUE. )
        END IF

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  compute total time

  800 CONTINUE
      IF ( control%error > 0 .AND. control%print_level >= 1 )                 &
        CALL SYMBOLS_status( inform%status, control%error, prefix, 'DLP' )
      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=',                                            &
    &   /, A, 3X, ' =                          DLP total time            ',    &
    &             '             =',                                            &
    &   /, A, 3X, ' =', 24X, 0P, F12.2, 29x, '='                               &
    &   /, A, 3X, ' =    preprocess    analyse    factorize     solve    ',    &
    &             '   search    =',                                            &
    &   /, A, 3X, ' =', 5F12.2, 5X, '=',                                       &
    &   /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=') ")                                         &
        prefix, prefix, prefix, inform%time%clock_total, prefix, prefix,       &
        inform%time%clock_preprocess, inform%time%clock_analyse,               &
        inform%time%clock_factorize, inform%time%clock_solve,                  &
        inform%time%clock_search, prefix

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving DLP_solve ' )" ) prefix
      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -DLP_solve-', /,  A,                           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving DLP_solve ' )" ) prefix
      RETURN

!  non-executable statements

 2010 FORMAT( ' ', /, A, '    ** Error return ', I0, ' from DLP ' )

!  End of DLP_solve

      END SUBROUTINE DLP_solve

!-*-*-*-*-*-*-   D L P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE DLP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine DLP_initialize
!   control see Subroutine DLP_initialize
!   inform  see Subroutine DLP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( DLP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DLP_control_type ), INTENT( IN ) :: control
      TYPE ( DLP_inform_type ), INTENT( INOUT ) :: inform

      CALL DQP_terminate( data, control, inform )

      RETURN

!  End of subroutine DLP_terminate

      END SUBROUTINE DLP_terminate

!-*-*-*- D L P _ N E X T _ P E R T U R B A T I O N   S U B R O U T I N E  -*-*-

      SUBROUTINE DLP_next_perturbation( n, m, dims, A_val, A_col, A_ptr,       &
                                        C_l, C_u, X_l, X_u, X, Y, Z, C, f,     &
                                        prefix, control, inform,               &
                                        extrapolation_ok, perturbation,        &
                                        gradient_kind, nv, m_ref,              &
                                        SBLS_data, SCU_data, GLTR_data,        &
                                        SBLS_control,GLTR_control,             &
                                        refactor, m_active, n_active,          &
                                        C_active, X_active,                    &
                                        ACTIVE_status, SOL, RHS, RES,          &
                                        Y_l, Y_u, Z_l, Z_u, VECTOR, VT, GV,    &
                                        PV, DV, A_sbls, C_sbls, SCU_mat,       &
                                        G, C_stat, X_stat )

! -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  given a vector v_current = v(sigma), for sigma = sigma_current, where

!    ( sigma I  J_F^T ) (  x(sigma)  ) = ( - g  )                      (*)
!    (   J_F      0   ) ( - v(sigma) )   (  b_F )

!  find the smallest 0 <= sigma <= sigma_current for which y(sigma)
!  satisfies (*) and also has the correct "sign"

!  Note that (*) may be rewritten as

!    (  I   J_F^T ) ( sigma x(sigma) ) = (   - g     )
!    ( J_F    0   ) (   - v(sigma)   )   ( sigma b_F )

!  and thus that

!    x(sigma) = dx + ( sigma_current / sigma ) . ( x_current - dx ) and
!    v(sigma) = v_current - dv + ( sigma / sigma_current ) dv ,

!  where

!    ( sigma_current I   J_F^T ) (   dx ) = (  0  )                 (**)
!    (         J_F         0   ) ( - dv )   ( b_F )

!  Thus it remains to solve (**) using the current factorization, and to look
!  for sign changes to y(sigma) as sigma decreases from sigma_current, and
!  also to check that x(sigma) is itself feasible

! -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, gradient_kind, nv
      INTEGER, INTENT( INOUT ) :: m_ref
      INTEGER, INTENT( INOUT ) :: m_active, n_active
      TYPE ( DQP_dims_type ), INTENT( IN ) :: dims
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( INOUT ) :: perturbation
      LOGICAL, INTENT( INOUT ) :: refactor
      LOGICAL, INTENT( OUT ) :: extrapolation_ok
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: X_active
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: C_active
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: ACTIVE_status
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: RES
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: RHS
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( 1 : dims%c_l_end ) :: Y_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%x_free + 1:dims%x_l_end ) :: Z_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: VECTOR
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: VT
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: GV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: PV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: DV
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( DLP_control_type ), INTENT( IN ) :: control
      TYPE ( DLP_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: SBLS_data
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( GLTR_data_type ), INTENT( INOUT ) :: GLTR_data
      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: SBLS_control
      TYPE ( GLTR_control_type ), INTENT( INOUT ) :: GLTR_control
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_sbls
      TYPE ( SMT_type ), INTENT( INOUT ) :: C_sbls
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_stat
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G

!  Local variables

      INTEGER :: i, ii, j, l, out
      REAL ( KIND = wp ) :: d_v, d_x, vc, xc, pxd, qt, obj, rho, ratio
      REAL ( KIND = wp ) :: new_perturbation, this_perturbation, skip_tol
      LOGICAL :: printi, printt, printp, printd
      LOGICAL :: c_stat_required, x_stat_required, penalty_objective, skip

      out = control%out
      printi = out > 0 .AND. control%print_level >= 1
      printt = out > 0 .AND. control%print_level >= 2
      printp = out > 0 .AND. control%print_level >= 3
      printd = out > 0 .AND. control%print_level >= 5

      penalty_objective = control%rho > zero
      IF ( penalty_objective ) rho = control%rho

!  parts extracted and modified from DQP_solve_main

      IF ( printp ) WRITE( out, "( /, A, ' ** entrering DLP_next_perturbation',&
     &  ' with perturbation =', ES12.4 )" ) prefix, perturbation

      extrapolation_ok = .FALSE.

!  use a direct method
!  . . . . . . . . . .

      IF ( control%subspace_direct ) THEN
        IF ( refactor ) THEN

!  form the vector of primal-active right-hand sides of (S) in sol

          RHS( : n ) = zero

!  components from the general constraints

          ii = n
          DO j = 1, m_active
            ii = ii + 1 ; i = C_active( j )
            IF ( i < 0 ) THEN
              RHS( ii ) = C_l( - i )
            ELSE
              RHS( ii ) = C_u( i )
            END IF
          END DO

!  components from the simple bounds

          DO i = 1, n_active
            ii = ii + 1 ; j = X_active( i )
            IF ( j < 0 ) THEN
              RHS( ii ) = X_l( - j )
            ELSE
              RHS( ii ) = X_u( j )
            END IF
          END DO

!  find the Fredholm Alternative for the linear system (S); the solution will
!  be in sol

          SOL( : n + A_sbls%m ) = RHS( : n + A_sbls%m )
          IF ( printd ) WRITE( out, "( /, A, ' Fredholm rhs = ', ES12.4 )" )   &
            prefix, MAXVAL( ABS( RHS( : n + A_sbls%m ) ) )
          CALL SBLS_fredholm_alternative( n, A_sbls%m, A_sbls,                 &
                                          SBLS_data%efactors, SBLS_control,    &
                                          inform%SBLS_inform, SOL )
          IF ( inform%SBLS_inform%status < 0 ) THEN
            IF ( printi ) WRITE( out,                                          &
               "( A, ' SBLS_fredholm_alternative, status = ', I0 )" )          &
                 prefix, inform%SBLS_inform%status
            inform%status = inform%SBLS_inform%status
            GO TO 900
          END IF

!  check if the system is inconsistent

          IF ( inform%SBLS_inform%alternative ) THEN
            IF ( printp ) WRITE( out,                                          &
               "( A, ' SBLS_fredholm_alternative, inconsistent' )" ) prefix

!  the system is consistent

          ELSE
            IF ( printp ) WRITE( out, "( A, ' consistent solution',            &
           &       ' to perturbation reduction problem found ' )" ) prefix

!  if required, check the residuals

            IF ( printd ) THEN
              RES( : n + A_sbls%m ) = - RHS( : n + A_sbls%m )
              RES( : n ) = RES( : n ) + perturbation * SOL( : n )
              DO l = 1, A_sbls%ne
                i = n + A_sbls%ROW( l ) ; j = A_sbls%COL( l )
                RES( j ) = RES( j ) + A_sbls%VAL( l ) * SOL( i )
                RES( i ) = RES( i ) + A_sbls%VAL( l ) * SOL( j )
              END DO
              WRITE( out, "( A, ' Fredholm residual ', ES12.4 )" )             &
                prefix, MAXVAL( ABS( RES( : n + A_sbls%m ) ) )
            END IF

!  as sol contains - dv_F, replace sol by dv_F

            SOL( n + 1 : n + A_sbls%m ) = - SOL( n + 1 : n + A_sbls%m )
          END IF

!  alteratively ... try updating the existing factorization using the
!  Schur-complement method. That is, given the reference matrix, we form

!    K = ( K_0  K_+^T )
!        ( K_+    0   )

!  for suitably chosen K_+, and solve systems with K using factors of K_0
!  and the Schur complement K_+ K_0^{1} K_+^T of K_0 in K

        ELSE

!  form the vector of primal-active right-hand sides of (S) in sol

          RHS( : SCU_mat%n + SCU_mat%m ) = zero

!  components from the general constraints

          DO j = 1, m_active
            i = C_active( j )
            ii = n + ACTIVE_status( ABS( i ) )
            IF ( i < 0 ) THEN
              RHS( ii ) = C_l( - i )
            ELSE
              RHS( ii ) = C_u( i )
            END IF
          END DO

!  components from the simple bounds

          DO i = 1, n_active
            j = X_active( i )
            ii = n + ACTIVE_status( m + ABS( j ) )
            IF ( j < 0 ) THEN
              RHS( ii ) = X_l( - j )
            ELSE
              RHS( ii ) = X_u( j )
            END IF
          END DO

!  now solve (S)

          inform%scu_status = 1
          DO
            CALL SCU_solve( SCU_mat, SCU_data, RHS, SOL,                       &
                            VECTOR( : SCU_mat%n ), inform%scu_status )
            IF ( inform%scu_status <= 0 ) EXIT
            CALL SBLS_solve( n, m_ref, A_sbls, C_sbls, SBLS_data,              &
                             SBLS_control, inform%SBLS_inform,                 &
                             VECTOR( : SCU_mat%n ) )
            IF ( inform%SBLS_inform%status < 0 ) THEN
              IF ( printi ) WRITE( out, "( A, ' SBLS_solve, status = ', I0 )" )&
                prefix, inform%SBLS_inform%status
              inform%status = inform%SBLS_inform%status
              GO TO 900
            END IF
          END DO

!  untangle the dv_F component by copying it into RHS (and then back)

          l = n
          DO j = 1, m_active
            i = ABS( C_active( j ) )
            ii = n + ACTIVE_status( i )
            l = l + 1
            RHS( l ) = - SOL( ii )
          END DO
          DO i = 1, n_active
            j = ABS( X_active( i ) )
            ii = n + ACTIVE_status( m + j )
            l = l + 1
            RHS( l ) = - SOL( ii )
          END DO
          SOL( n + 1 : n + A_sbls%m ) = RHS( n + 1 : n + A_sbls%m )

!  regenerate the b_F component of RHS

          ii = n
          DO j = 1, m_active
            ii = ii + 1 ; i = C_active( j )
            IF ( i < 0 ) THEN
              RHS( ii ) = C_l( - i )
            ELSE
              RHS( ii ) = C_u( i )
            END IF
          END DO
          DO i = 1, n_active
            ii = ii + 1 ; j = X_active( i )
            IF ( j < 0 ) THEN
              RHS( ii ) = X_l( - j )
            ELSE
              RHS( ii ) = X_u( j )
            END IF
          END DO

!  if required, check the residuals

          IF ( printd ) THEN
            RES( : n + A_sbls%m ) = - RHS( : n + A_sbls%m )
            RES( : n ) = RES( : n ) + perturbation * SOL( : n )
            DO l = 1, A_sbls%ne
              i = n + A_sbls%ROW( l ) ; j = A_sbls%COL( l )
              RES( j ) = RES( j ) - A_sbls%VAL( l ) * SOL( i )
              RES( i ) = RES( i ) + A_sbls%VAL( l ) * SOL( j )
            END DO
            WRITE( out, "( A, ' Fredholm residual ', ES12.4 )" )               &
              prefix, MAXVAL( ABS( RES( : n + A_sbls%m ) ) )
          END IF
        END IF

!  use an iterative method
!  . . . . . . . . . . . .

      ELSE

!  record the dual gradient gc^d_k = b_F_k of q_k^d(y_F)

!  set the components of b_F_k from the general constraints

        l = 0
        DO j = 1, m_active
          l = l + 1 ; i = C_active( j )
          IF ( i < 0 ) THEN
            GV( l ) = C_l( - i )
          ELSE
            GV( l ) = C_u( i )
          END IF
        END DO

!  and the components from the simple bounds

        DO i = 1, n_active
          l = l + 1 ; j = X_active( i )
          IF ( j < 0 ) THEN
            GV( l ) = X_l( - j )
          ELSE
            GV( l ) = X_u( j )
          END IF
        END DO

!  make a copy of the dual gradient

        GV( : A_sbls%m ) = - GV( : A_sbls%m )
        DV( : A_sbls%m ) = GV( : A_sbls%m )

        GLTR_control = control%GLTR_control
!       GLTR_control%print_level = 1
        GLTR_control%f_0 = zero
        inform%GLTR_inform%status = 1

!  iteration to find the minimizer

        DO
          CALL GLTR_solve( A_sbls%m, big_radius, qt,                           &
                           SOL( n + 1 : n + A_sbls%m ), GV, PV, GLTR_data,     &
                           GLTR_control, inform%GLTR_inform )

!  branch as a result of inform%status

          SELECT CASE( inform%GLTR_inform%status )

!  form the preconditioned gradient

!         CASE( 2, 6 )

!  form the matrix-vector product

          CASE ( 3, 7 )

!  compute pv -> A A^T pv / sigma_current
!  ... first store the vector A^T pv in vt ...

            VT( : n ) = zero
            DO l = 1, A_sbls%ne
              i = A_sbls%row( l ) ; j = A_sbls%col( l )
              VT( j ) = VT( j ) + A_sbls%val( l ) * PV( i )
            END DO

!  ... then overwrite vt with vt / sigma_current ...

            VT( : n ) = VT( : n ) / perturbation

!  ... and finally compute pv = A vt

            PV( : A_sbls%m ) = zero
            DO l = 1, A_sbls%ne
              i = A_sbls%row( l ) ; j = A_sbls%col( l )
              PV( i ) = PV( i ) + A_sbls%val( l ) * VT( j )
            END DO

!  restart

          CASE ( 5 )
             GV( : A_sbls%m ) = DV( : A_sbls%m )

!  successful return

          CASE ( - 30, 0 )

!  compute J^T dv / sigma_current

            SOL( : n ) = zero
            DO l = 1, A_sbls%ne
              i = A_sbls%row( l ) ; j = A_sbls%col( l )
              SOL( j ) = SOL( j ) + A_sbls%val( l ) * SOL( n + i )
            END DO
            SOL( : n ) = SOL( : n ) / perturbation

            EXIT

!  error returns

          CASE DEFAULT
            IF ( printt ) WRITE( out, "( A, ' GLTR_solve exit status = ',      &
           &  I0 ) " ) prefix, inform%GLTR_inform%status
            EXIT
          END SELECT
        END DO
      END IF

!  find the largest sigma in [0, sigma_current ] for which
!    v(sigma) = v_current + ( sigma / sigma_current - 1 ) d_v is in
!  D = { v: (y_l,z_l) >= 0 & (y_u,z_u) <= 0 },
!  or, in the penalty case,
!  D = { v: 0 <= y_l <= rho, - rho <= y_u <= 0, z_l >= 0 & z_u <= 0 }

      new_perturbation = zero

!  components from the general constraints

      obj = zero
      l = n
      DO j = 1, m_active
        l = l + 1 ; i = C_active( j ) ; d_v = SOL( l )
        IF ( ABS( i ) <= dims%c_equality ) THEN
           IF ( i < 0 ) THEN
              obj = obj + ( Y_l( - i ) - d_v ) * C_l( - i )
            ELSE
              obj = obj + ( Y_l( i ) - d_v ) * C_l( i )
            END IF
          CYCLE
        END IF
        IF ( i < 0 ) THEN
          vc = Y_l( - i )
          obj = obj + ( vc - d_v ) * C_l( - i )
          IF ( penalty_objective ) THEN
            IF ( vc < d_v ) THEN
              this_perturbation = perturbation * ( d_v - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            ELSE IF ( vc - rho > d_v ) THEN
              this_perturbation = perturbation * ( d_v + rho - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE
            IF ( vc < d_v ) THEN
              this_perturbation = perturbation * ( d_v - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          END IF
        ELSE
          vc = Y_u( i )
          obj = obj + ( vc - d_v ) * C_u( i )
          IF ( penalty_objective ) THEN
            IF ( vc > d_v ) THEN
              this_perturbation = perturbation * ( d_v - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            ELSE IF ( vc - rho < d_v ) THEN
              this_perturbation = perturbation * ( d_v + rho - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE
            IF ( vc > d_v ) THEN
              this_perturbation = perturbation * ( d_v - vc ) / d_v
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          END IF
        END IF
      END DO

!  components from the simple bounds

      DO i = 1, n_active
        l = l + 1 ; j = X_active( i ) ; d_v = SOL( l )
        IF ( j < 0 ) THEN
          vc = Z_l( - j )
          obj = obj + ( vc - d_v ) * X_l( - j )
          IF ( vc < d_v ) THEN
            this_perturbation = perturbation * ( d_v - vc ) / d_v
            IF ( this_perturbation < perturbation )                            &
              new_perturbation = MAX( new_perturbation, this_perturbation )
          END IF
        ELSE
          vc = Z_u( j )
          obj = obj + ( vc - d_v ) * X_u( j )
          IF ( d_v < vc ) THEN
            this_perturbation = perturbation * ( d_v - vc ) / d_v
            IF ( this_perturbation < perturbation )                            &
              new_perturbation = MAX( new_perturbation, this_perturbation )
          END IF
        END IF
      END DO

!  find the largest sigma in [0, sigma_current ] for which
!    x(sigma) = d_x + ( sigma_current / sigma ) . ( x_current - d_x ) is in
!  P = { x:  x_l <= x <= x & c_l <= A x <= c_u }
!  or, in the penalty case,
!  P = { x:  x_l <= x <= x }

!  components from the simple bounds

      skip = .TRUE. ; skip_tol = thousand * epsmch
      DO j = dims%x_free + 1, n
        xc = X( j ) ; d_x = SOL( j )
        IF ( ABS( xc - d_x ) > skip_tol ) THEN
          skip = .FALSE.
          pxd = perturbation * ( xc - d_x )
          IF ( j <= dims%x_l_start - 1 ) THEN
            IF ( - d_x /= zero ) THEN
              this_perturbation = pxd / ( - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE IF ( j <= dims%x_u_start - 1 ) THEN
            IF ( X_l( j ) - d_x /= zero ) THEN
              this_perturbation = pxd / ( X_l( j ) - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE IF ( j <= dims%x_l_end ) THEN
            IF ( X_l( j ) - d_x /= zero ) THEN
              this_perturbation = pxd / ( X_l( j ) - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
            IF ( X_u( j ) - d_x /= zero ) THEN
              this_perturbation = pxd / ( X_u( j ) - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE IF ( j <= dims%x_u_end ) THEN
            IF ( X_u( j ) - d_x /= zero ) THEN
              this_perturbation = pxd / ( X_u( j ) - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          ELSE
            IF ( - d_x /= zero ) THEN
              this_perturbation = pxd / ( - d_x )
              IF ( this_perturbation < perturbation )                          &
                new_perturbation = MAX( new_perturbation, this_perturbation )
            END IF
          END IF
        END IF
      END DO

!  components from the general constraints

      IF ( .NOT. skip ) THEN
        DO i = dims%c_equality + 1, m
          xc = C( i ) ; d_x = zero
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            d_x = d_x + A_val( l ) * SOL( A_col( l ) )
          END DO
          IF ( ABS( xc - d_x ) > skip_tol ) THEN
            pxd = perturbation * ( xc - d_x )
            IF ( i <= dims%c_l_end ) THEN
              IF ( C_l( i ) - d_x /= zero ) THEN
                this_perturbation = pxd / ( C_l( i ) - d_x )
                IF ( this_perturbation < perturbation )                        &
                  new_perturbation = MAX( new_perturbation, this_perturbation )
              END IF
            END IF
            IF ( i >= dims%c_u_start ) THEN
              IF ( C_u( i ) - d_x /= zero ) THEN
                this_perturbation = pxd / ( C_u( i ) - d_x )
                IF ( this_perturbation < perturbation )                        &
                  new_perturbation = MAX( new_perturbation, this_perturbation )
              END IF
            END IF
          END IF
        END DO
      END IF

!  record the optimal solution if we have found it

      IF ( printi ) WRITE( out, "( /, A, '  new quadratic perturbation ',      &
     &   'should be no larger than', ES9.2 )" ) prefix, new_perturbation

      IF ( new_perturbation < infeas ) THEN
        extrapolation_ok = .TRUE.
        perturbation = zero
        c_stat_required = PRESENT( C_stat )
        x_stat_required = PRESENT( X_stat )
        IF ( x_stat_required ) X_stat  = 0
        IF ( c_stat_required ) THEN
          C_stat( : dims%c_equality ) = - 1
          C_stat( dims%c_equality + 1 : ) = 0
        END IF

!  compute the new X, Y and Z

        Y( : m ) = zero ; Z( : n ) = zero
        l = n
        DO j = 1, m_active
          l = l + 1 ; i = C_active( j )
          IF ( i < 0 ) THEN
            Y( - i ) = Y_l( - i ) - SOL( l )
            IF ( c_stat_required ) C_stat( - i ) = - 1
          ELSE
            Y( i ) = Y_u( i ) - SOL( l )
            IF ( c_stat_required ) C_stat( i ) = 1
          END IF
        END DO

        DO i = 1, n_active
          l = l + 1 ; j = X_active( i )
          IF ( j < 0 ) THEN
            Z( - j ) = Z_l( - j ) - SOL( l )
            IF ( - j < dims%x_l_start ) THEN
              X( - j ) = zero
            ELSE
              X( - j ) = X_l( - j )
            END IF
            IF ( x_stat_required ) X_stat( - j ) = - 1
          ELSE
            Z( j ) = Z_u( j ) - SOL( l )
            IF ( j > dims%x_u_end ) THEN
              X( j ) = zero
            ELSE
              X( j ) = X_u( j )
            END IF
            IF ( x_stat_required ) X_stat( j ) = 1
          END IF
        END DO

!  the solution at sigma = 0 is infeasible

      ELSE

!  compute the new x = d_x + ( sigma_current / sigma ) . ( x_current - d_x )

        ratio = perturbation / new_perturbation
        DO j = dims%x_free + 1, n
          X( j ) = d_x + ratio * ( X( j ) - SOL( j ) )
        END DO

!  compute the new v = v_current + ( sigma / sigma_current - 1 ) d_v

        ratio = new_perturbation / perturbation - one

!  components from the general constraints

        l = n
        DO j = 1, m_active
          l = l + 1 ; i = C_active( j )
          IF ( ABS( i ) <= dims%c_equality ) CYCLE
          IF ( i < 0 ) THEN
            Y_l( - i ) = Y_l( - i ) + ratio * SOL( l )
            Y( - i ) = Y_l( - i )
            IF ( c_stat_required ) C_stat( - i ) = - 1
          ELSE
            Y_u( i ) = Y_u( i ) + ratio * SOL( l )
            Y( i ) = Y_u( i )
            IF ( c_stat_required ) C_stat( i ) = 1
          END IF
        END DO

!  components from the simple bounds

        DO i = 1, n_active
          l = l + 1 ; j = X_active( i )
          IF ( j < 0 ) THEN
            Z_l( - j ) = Z_l( - j ) + ratio * SOL( l )
            Z( - j ) = Z_l( - j )
            IF ( x_stat_required ) X_stat( - j ) = - 1
          ELSE
            Z_u( j ) = Z_u( j ) + ratio * SOL( l )
            Z( j ) = Z_u( j )
            IF ( x_stat_required ) X_stat( j ) = 1
          END IF
        END DO

!  record the new maximum perturbation

        perturbation = new_perturbation
      END IF

!  compute the new C

      C = zero
      CALL DQP_AX( m, C, m, A_ptr( m + 1 ) - 1, A_val, A_col, A_ptr,           &
                   n, X, '+ ')

!  compute the objective function value

      IF ( gradient_kind == 0 ) THEN
        inform%obj = f
      ELSE IF ( gradient_kind == 1 ) THEN
        inform%obj = f + SUM( X )
      ELSE
        inform%obj = f + DOT_PRODUCT( X, G )
      END IF
      IF ( penalty_objective ) THEN
        DO i = 1, dims%c_equality
          inform%obj = inform%obj + rho * ABS( C( i ) -  C_l( i ) )
        END DO
        DO i = dims%c_equality + 1, dims%c_l_end
          inform%obj = inform%obj + rho * MAX( zero, C_l( i ) - C( i ) )
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          inform%obj = inform%obj + rho * MAX( zero, C( i ) - C_u( i ) )
        END DO
      END IF

      IF ( printp ) WRITE( out, "( /, A, ' ** leaving DLP_next_perturbation',  &
     &  ' with perturbation =', ES12.4 )" ) prefix, perturbation

      inform%status = GALAHAD_ok

  900 CONTINUE
      RETURN

!  end of subroutine DLP_next_perturbation

      END SUBROUTINE DLP_next_perturbation

!  end of module DLP

    END MODULE GALAHAD_DLP_double
