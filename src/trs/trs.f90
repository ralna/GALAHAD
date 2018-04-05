! THIS VERSION: GALAHAD 2.6 - 26/10/2014 AT 14:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ T R S  double  M O D U L E  *-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.3. January 7th, 2009
!   modified to handle constraints, February 7th, 2009
!   modified to incorporate SLS sparse-equation solver, February 1st, 2010
!   modified to allow dense factorization, October 25th, 2011
!   modified to include TRACE contract, October 26th, 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_TRS_double

!       -----------------------------------------------
!      |                                               |
!      | Solve the trust-region subproblem             |
!      |                                               |
!      |    minimize     1/2 <x, H x> + <c, x> + f     |
!      |    subject to      ||x||_M <= radius          |
!      |    or              ||x||_M  = radius          |
!      |    and optionally  A x = 0,                   |
!      |                                               |
!      | where ||x||_M^2 = <x, Mx> and M is diagonally |
!      | dominant, using a sparse matrix factorization |
!      |                                               |
!       -----------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_RAND_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_ROOTS_double, ONLY: ROOTS_quadratic, ROOTS_cubic
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_SLS_double
      USE GALAHAD_IR_double
      USE GALAHAD_MOP_double
      USE GALAHAD_LAPACK_interface, ONLY : SYEV, SYGV

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: TRS_initialize, TRS_read_specfile, TRS_solve, TRS_terminate,   &
                TRS_contract, TRS_solve_diagonal, TRS_orthogonal_solve,        &
                SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: history_max = 100
      INTEGER, PARAMETER :: max_degree = 3
      INTEGER, PARAMETER :: n_dense = 100
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: point4 = 0.4_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: six = 6.0_wp
      REAL ( KIND = wp ), PARAMETER :: sixth = one / six
      REAL ( KIND = wp ), PARAMETER :: twothirds = two /three
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: twentyfour = 24.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = half * HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

      REAL ( KIND = wp ), PARAMETER :: theta_ii = one
      REAL ( KIND = wp ), PARAMETER :: theta_eps = point01
      REAL ( KIND = wp ), PARAMETER :: theta_g = half
      REAL ( KIND = wp ), PARAMETER :: theta_n = half
      REAL ( KIND = wp ), PARAMETER :: theta_n_small = ten ** ( - 1 )
      REAL ( KIND = wp ), PARAMETER :: theta_n_tiny = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: theta_hard = 0.999_wp
      REAL ( KIND = wp ), PARAMETER :: gamma_eps = half
      REAL ( KIND = wp ), PARAMETER :: gamma = one
      REAL ( KIND = wp ), PARAMETER :: roots_tol = teneps
      LOGICAL :: roots_debug = .FALSE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TRS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  unit to write problem data into file problem_file

        INTEGER :: problem = 0

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  should the problem be solved by dense factorization? Possible values are
!   0     sparse factorization will be used
!   1     dense factorization will be used
!   other the choice is made automatically depending on the dimension & sparsity

        INTEGER :: dense_factorization = 0

!  how much of H has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_h = 2

!  how much of M has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_m = 2

!  how much of A has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_a = 2

!  the maximum number of factorizations (=iterations) allowed. -ve => no limit

        INTEGER :: max_factorizations = - 1

!  the number of inverse iterations performed in the "maybe hard" case

        INTEGER :: inverse_itmax = 2

!  maximum degree of Taylor approximant allowed

        INTEGER :: taylor_max_degree = 3

!  initial estimate of the Lagrange multipler

        REAL ( KIND = wp ) :: initial_multiplier = zero

!  lower and upper bounds on the multiplier, if known

        REAL ( KIND = wp ) :: lower = - half * HUGE( one )
        REAL ( KIND = wp ) :: upper =  HUGE( one )

!  stop when | ||x|| - radius | <=
!     max( stop_normal * radius, stop_absolute_normal )

        REAL ( KIND = wp ) :: stop_normal = epsmch
        REAL ( KIND = wp ) :: stop_absolute_normal = epsmch

!  stop when bracket on optimal multiplier <= stop_hard * max( bracket ends )

        REAL ( KIND = wp ) :: stop_hard  = epsmch

!  start inverse iteration when bracket on optimal multiplier <=
!    stop_start_invit_tol * max( bracket ends )

        REAL ( KIND = wp ) :: start_invit_tol = half

!  start full inverse iteration when bracket on multiplier <=
!    stop_start_invitmax_tol * max( bracket ends)

        REAL ( KIND = wp ) :: start_invitmax_tol = point1

!  is the solution is REQUIRED to lie on the boundary (i.e., is the constraint
!  an equality)?

        LOGICAL :: equality_problem= .FALSE.

!  ignore initial_multiplier?

        LOGICAL :: use_initial_multiplier = .FALSE.

!  should a suitable initial eigenvector should be chosen or should a previous
!  eigenvector may be used?

        LOGICAL :: initialize_approx_eigenvector  = .TRUE.

!  ignore the trust-region if H is positive definite

        LOGICAL :: force_Newton  = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  name of file into which to write problem data

        CHARACTER ( LEN = 30 ) :: problem_file =                               &
         'trs_problem.data' // REPEAT( ' ', 14 )

!  symmetric (indefinite) linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  definite linear equation solver

        CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
           "sils" // REPEAT( ' ', 26 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix  = '""                            '

!  control parameters for the Cholesky factorization and solution

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for iterative refinement

        TYPE ( IR_control_type ) :: IR_control

      END TYPE

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: TRS_data_type
        PRIVATE
        INTEGER :: m, npm, h_ne, m_ne, a_ne, m_end
        TYPE ( RAND_seed ) :: seed
!       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U, V, Y, Z, WORK
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: M_diag, M_offd
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_dense, X_dense
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Q_dense
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: M_dense
        TYPE ( SMT_type ) :: H_dense, A_dense
        LOGICAL :: get_initial_u = .TRUE.
        LOGICAL :: accurate = .TRUE.
        TYPE ( SMT_type ) :: H_lambda
        TYPE ( IR_data_type ) :: IR_data
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( TRS_control_type ) :: control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TRS_time_type

!  total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  CPU time spent building H + lambda * M

        REAL ( KIND = wp ) :: assemble = 0.0

!  CPU time spent reordering H + lambda * M prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  CPU time spent factorizing H + lambda * M

        REAL ( KIND = wp ) :: factorize = 0.0

!  CPU time spent solving linear systems inolving H + lambda * M

        REAL ( KIND = wp ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time spent building H + lambda * M

        REAL ( KIND = wp ) :: clock_assemble = 0.0

!  clock time spent reordering H + lambda * M prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  clock time spent factorizing H + lambda * M

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving H + lambda * M

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - -
!   history derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TRS_history_type

!  value of lambda

        REAL ( KIND = wp ) :: lambda = zero

!  corresponding value of ||x(lambda)||_M

        REAL ( KIND = wp ) :: x_norm = zero
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TRS_inform_type

!   reported return status:
!      0 the solution has been found
!     -1 an array allocation has failed
!     -2 an array deallocation has failed
!     -3 n and/or Delta is not positive
!     -9 the analysis phase of the factorization of H + lambda * M failed
!    -10 the factorization of H + lambda * M failed
!    -15 M does not appear to be strictly diagonally dominant
!    -16 ill-conditioning has prevented furthr progress

        INTEGER :: status = 0

!  STAT value after allocate failure

        INTEGER :: alloc_status = 0

!   the number of factorizations performed

        INTEGER :: factorizations = 0

!   the maximum number of entries in the factors

        INTEGER ( KIND = long ) :: max_entries_factors = 0

!  the number of (||x||_M,lambda) pairs in the history

        INTEGER :: len_history = 0

!  the value of the quadratic function

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the M-norm of x, ||x||_M

        REAL ( KIND = wp ) :: x_norm = zero

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = wp ) :: multiplier = zero

!  a lower bound max(0,-lambda_1), where lambda_1 is the left-most
!  eigenvalue of (H,M)

        REAL ( KIND = wp ) :: pole = zero

!  was a dense factorization used?

        LOGICAL :: dense_factorization = .FALSE.

!  has the hard case occurred?

        LOGICAL :: hard_case = .FALSE.

!  name of array that provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  time, history, cholesky and iterative_refinement information

        TYPE ( TRS_time_type ) :: time
        TYPE ( TRS_history_type ), DIMENSION( history_max ) :: history
        TYPE ( SLS_inform_type ) :: SLS_inform
        TYPE ( IR_inform_type ) :: IR_inform
      END TYPE

    CONTAINS

!-*-*-*-*-*-*-  T R S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE TRS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!
!  .  Set initial values for the TRS control parameters  .
!
!  Arguments:
!  =========
!
!   data     private internal data
!   control  a structure containing control information. See TRS_control_type
!   data     private internal data
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( TRS_DATA_TYPE ), INTENT( INOUT ) :: data
      TYPE ( TRS_CONTROL_TYPE ), INTENT( OUT ) :: control
      TYPE ( TRS_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  Set initial control parameter values

      control%stop_normal = epsmch ** 0.75
      control%stop_absolute_normal = epsmch ** 0.75
      control%stop_hard = epsmch ** 0.75

!  initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )

!  Set initial values for factorization controls and data

      control%SLS_control%ordering = 0

!  Ensure that TRS control values are passed on to SLS and IR

      control%SLS_control%error = control%error
      control%SLS_control%warning = control%out
      control%SLS_control%out = control%out
      control%SLS_control%statistics = control%out
!     control%SLS_control%print_level = control%print_level
      control%SLS_control%prefix = '" - SLS:"'

!  Set initial values for solve controls

      CALL IR_initialize( data%IR_data, control%IR_control, inform%IR_inform )
      control%IR_control%prefix = '" - IR:"'

!  Set initial data values

      data%get_initial_u = .TRUE.

      RETURN

!  End of subroutine TRS_initialize

      END SUBROUTINE TRS_initialize

!-*-*-*-*-   T R S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE TRS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by TRS_initialize could (roughly)
!  have been set as:

!  BEGIN TRS SPECIFICATIONS (DEFAULT)
!   error-printout-device                          6
!   printout-device                                6
!   problem-device                                 0
!   print-level                                    0
!   use-dense-factorization                        0
!   has-h-changed                                  2
!   has-m-changed                                  2
!   has-a-changed                                  2
!   factorization-limit                            -1
!   inverse-iteration-limit                        1
!   max-degree-taylor-approximant                  3
!   initial-multiplier                             0.0
!   lower-bound-on-multiplier                      0.0
!   upper-bound-on-multiplier                      1.0D+300
!   stop-normal-case                               1.0D-12
!   stop-hard-case                                 1.0D-12
!   start-inverse-iteration-tolerance              0.5
!   start-max-inverse-iteration-tolerance          0.01
!   equality-problem                               F
!   use-initial-multiplier                         F
!   initialize-approximate-eigenvector             T
!   force-Newton-if-positive-definite              F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   symmetric-linear-equation-solver               sils
!   definite-linear-equation-solver                sils
!   problem-file                                   trs_problem.data
!   output-line-prefix                             ""
!  END TRS SPECIFICATIONS

!  Dummy arguments

      TYPE ( TRS_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: problem = out + 1
      INTEGER, PARAMETER :: print_level = problem + 1
      INTEGER, PARAMETER :: dense_factorization = print_level + 1
      INTEGER, PARAMETER :: new_h = dense_factorization + 1
      INTEGER, PARAMETER :: new_m = new_h + 1
      INTEGER, PARAMETER :: new_a = new_m + 1
      INTEGER, PARAMETER :: max_factorizations = new_a + 1
      INTEGER, PARAMETER :: inverse_itmax = max_factorizations + 1
      INTEGER, PARAMETER :: taylor_max_degree = inverse_itmax + 1
      INTEGER, PARAMETER :: initial_multiplier = taylor_max_degree + 1
      INTEGER, PARAMETER :: lower = initial_multiplier + 1
      INTEGER, PARAMETER :: upper = lower + 1
      INTEGER, PARAMETER :: stop_normal = upper + 1
      INTEGER, PARAMETER :: stop_absolute_normal = stop_normal + 1
      INTEGER, PARAMETER :: stop_hard = stop_absolute_normal + 1
      INTEGER, PARAMETER :: start_invit_tol = stop_hard + 1
      INTEGER, PARAMETER :: start_invitmax_tol = start_invit_tol + 1
      INTEGER, PARAMETER :: equality_problem = start_invitmax_tol + 1
      INTEGER, PARAMETER :: use_initial_multiplier = equality_problem + 1
      INTEGER, PARAMETER :: initialize_approx_eigenvector =                    &
                              use_initial_multiplier + 1
      INTEGER, PARAMETER :: force_Newton = initialize_approx_eigenvector + 1
      INTEGER, PARAMETER :: space_critical = force_Newton + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: problem_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = problem_file + 1
      INTEGER, PARAMETER :: definite_linear_solver = symmetric_linear_solver + 1
      INTEGER, PARAMETER :: prefix = definite_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'TRS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( problem )%keyword = 'problem-device'
      spec( print_level )%keyword = 'print-level'
      spec( dense_factorization )%keyword = 'use-dense-factorization'
      spec( new_h )%keyword = 'has-h-changed'
      spec( new_m )%keyword = 'has-m-changed'
      spec( new_a )%keyword = 'has-a-changed'
      spec( max_factorizations )%keyword = 'factorization-limit'
      spec( inverse_itmax )%keyword = 'inverse-iteration-limit'
      spec( taylor_max_degree )%keyword = 'max-degree-taylor-approximant'

!  Real key-words

      spec( initial_multiplier )%keyword = 'initial-multiplier'
      spec( lower )%keyword = 'lower-bound-on-multiplier'
      spec( upper )%keyword = 'upper-bound-on-multiplier'
      spec( stop_normal )%keyword = 'stop-normal-case'
      spec( stop_absolute_normal )%keyword = 'stop-absolute-normal-case'
      spec( stop_hard )%keyword = 'stop-hard-case'
      spec( start_invit_tol )%keyword = 'start-inverse-iteration-tolerance'
      spec( start_invitmax_tol )%keyword =                                     &
        'start-max-inverse-iteration-tolerance'

!  Logical key-words

      spec( equality_problem )%keyword = 'equality-problem'
      spec( use_initial_multiplier )%keyword = 'use-initial-multiplier'
      spec( initialize_approx_eigenvector )%keyword =                          &
        'initialize-approximate-eigenvector'
      spec( force_Newton )%keyword = 'force-Newton-if-positive-definite'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( problem_file )%keyword = 'problem-file'
      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( definite_linear_solver )%keyword = 'definite-linear-equation-solver'
      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( problem ),                             &
                                  control%problem,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( dense_factorization ),                 &
                                  control%dense_factorization,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_h ),                               &
                                  control%new_h,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_m ),                               &
                                  control%new_m,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_a ),                               &
                                  control%new_a,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_factorizations ),                  &
                                  control%max_factorizations,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( inverse_itmax ),                       &
                                  control%inverse_itmax,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( taylor_max_degree ),                   &
                                  control%taylor_max_degree,                   &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( initial_multiplier ),                  &
                                  control%initial_multiplier,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( lower ),                               &
                                  control%lower,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( upper ),                               &
                                  control%upper,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_normal ),                         &
                                  control%stop_normal,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_absolute_normal ),                &
                                  control%stop_absolute_normal,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_hard ),                           &
                                  control%stop_hard,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( start_invit_tol ),                     &
                                  control%start_invit_tol,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( start_invitmax_tol ),                  &
                                  control%start_invitmax_tol,                  &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( equality_problem ),                    &
                                  control%equality_problem,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( use_initial_multiplier ),              &
                                  control%use_initial_multiplier,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( initialize_approx_eigenvector ),       &
                                  control%initialize_approx_eigenvector,       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( force_Newton ),                        &
                                  control%force_Newton,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( problem_file ),                        &
                                  control%problem_file,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),             &
                                  control%symmetric_linear_solver,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( definite_linear_solver ),              &
                                  control%definite_linear_solver,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read specfile data for SLS and IR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
        CALL IR_read_specfile( control%IR_control, device,                     &
                                alt_specname = TRIM( alt_specname ) // '-IR' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL IR_read_specfile( control%IR_control, device )
      END IF

      RETURN

!  End of subroutine TRS_read_specfile

      END SUBROUTINE TRS_read_specfile

!-*-*-*-*-*-*-*-*-*-*  T R S _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE TRS_solve( n, radius, f, C, H, X, data, control, inform,      &
                            M, A, Y )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the trust-region subproblem
!
!      minimize     1/2 <x, H x> + <c, x> + f
!      subject to    ||x||_M <= radius  or ||x||_M = radius
!      and optionally  A x = 0
!
!  where ||x||_M^2 = <x, Mx> and M is diagonally dominant, using a sparse
!  or dense matrix factorization
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!  see TRS_solve_main for details
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C
      TYPE ( SMT_type ), INTENT( IN ) :: H
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( TRS_data_type ), INTENT( INOUT ) :: data
      TYPE ( TRS_control_type ), INTENT( IN ) :: control
      TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: M
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: A
      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( * ) :: Y

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, l, m_dim, nb, nim1, lwork, sy_status
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record
      LOGICAL :: new_q
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER :: ILAENV
      EXTERNAL :: ILAENV

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for obvious errors

      IF ( n <= 0 ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
           "( A, ' n = ', I0, ' is too small ' )" ) prefix, n
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

      IF ( radius <= zero ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
          "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, radius
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  should the problem be solved by dense factorization? Possible values are
!   0     sparse factorization will be used
!   1     dense factorization will be used
!   other the choice is made automatically depending on the dimension & sparsity

      IF ( control%dense_factorization == 1 ) THEN
        inform%dense_factorization = .TRUE.
      ELSE IF ( control%dense_factorization == 0 ) THEN
        inform%dense_factorization = .FALSE.
      ELSE
        inform%dense_factorization = .FALSE.
        IF ( n <= n_dense ) inform%dense_factorization = .TRUE.
        IF ( SMT_get( H%type ) == 'DENSE' ) THEN
          IF ( PRESENT( M ) ) THEN
            IF ( PRESENT( A ) ) THEN
              IF ( SMT_get( M%type ) == 'DENSE' .AND.                          &
                   SMT_get( A%type ) == 'DENSE' )                              &
                inform%dense_factorization = .TRUE.
            ELSE
              IF ( SMT_get( M%type ) == 'DENSE' )                              &
                inform%dense_factorization = .TRUE.
            END IF
          ELSE
            IF ( PRESENT( A ) ) THEN
              IF ( SMT_get( A%type ) == 'DENSE' )                              &
                inform%dense_factorization = .TRUE.
            ELSE
              inform%dense_factorization = .TRUE.
            END IF
          END IF
        END IF
      END IF

!  a dense factorization will be used

      IF ( inform%dense_factorization ) THEN

!  if H or M have changed, compute D and Q

        IF ( control%new_h == 0 ) THEN
          IF ( PRESENT( M ) ) THEN
            IF ( control%new_m == 0 ) THEN
              new_q = .FALSE.
            ELSE
              new_q = .TRUE.
            END IF
          ELSE
            new_q = .FALSE.
          END IF
        ELSE
          new_q = .TRUE.
        END IF

        IF ( new_q ) THEN

!  set up space to store D and Q

          CALL SMT_put( data%H_dense%type, 'DIAGONAL', i )
          data%H_dense%n = n

!  allocate space to hold the required matrices H_dense and Q_dense

          array_name = 'trs: H_dense%val'
          CALL SPACE_resize_array( n, data%H_dense%val,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

          array_name = 'trs: Q_dense'
          CALL SPACE_resize_array( n, n, data%Q_dense,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

!  allocate space to hold the vectors X_dense and C_dense

          array_name = 'trs: X_dense'
          CALL SPACE_resize_array( n, data%X_dense,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

          array_name = 'trs: C_dense'
          CALL SPACE_resize_array( n, data%C_dense,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

!  copy H temporrily into the lower triangle of Q_dense

          data%Q_dense( : n, : n ) = zero
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              data%Q_dense( i, i ) = H%val( i )
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                data%Q_dense( i, j ) = H%val( l )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                data%Q_dense( i, H%col( l ) ) =                                &
                  data%Q_dense( i, H%col( l ) ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              data%Q_dense(  H%row( l ), H%col( l ) ) =                        &
                data%Q_dense(  H%row( l ), H%col( l ) ) + H%val( l )
            END DO
          END SELECT

!  allocate workspace

          nb = ILAENV( 1, 'DSYTRD', 'L', n, - 1, - 1, - 1 )
          lwork = MAX( 1, 3 * n - 1, ( nb + 2 ) * n )

          array_name = 'trs: work'
          CALL SPACE_resize_array( lwork, data%WORK,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

          IF ( PRESENT( M ) ) THEN

!  if necessary allocate space to hold the additional dense matrices M_dense

            array_name = 'trs: M_dense'
            CALL SPACE_resize_array( n, n, data%M_dense,                       &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910

!  copy M into the lower triangle of M_dense

            data%M_dense( : n, : n ) = zero
            SELECT CASE ( SMT_get( M%type ) )
            CASE ( 'DIAGONAL' )
              DO i = 1, n
                data%M_dense( i, i ) = M%val( i )
              END DO
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, n
                DO j = 1, i
                  l = l + 1
                  data%M_dense( i, j ) = M%val( l )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = M%ptr( i ), M%ptr( i + 1 ) - 1
                  data%M_dense( i, M%col( l ) ) =                              &
                    data%M_dense( i, M%col( l ) ) + M%val( l )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, M%ne
                data%M_dense(  M%row( l ), M%col( l ) ) =                      &
                  data%M_dense(  M%row( l ), M%col( l ) ) + M%val( l )
              END DO
            END SELECT

!  find the generalised spectral decomposition H Q = M Q D, where Q^T M Q = I

            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SYGV( 1, 'V','L', n, data%Q_dense( : n, : n ), n,             &
                       data%M_dense( : n, : n ), n, data%H_dense%val,          &
                       data%WORK( : lwork ), lwork, sy_status )

!  find the spectral decomposition H Q = Q D, where Q^T Q = I

          ELSE
            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SYEV( 'V','L', n, data%Q_dense( : n, : n ), n,                &
                        data%H_dense%val, data%WORK( : lwork ), lwork,         &
                        sy_status )
          END IF

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%factorize =                                              &
            inform%time%factorize + time_now - time_record
          inform%time%clock_factorize =                                        &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, 2010 ) prefix, clock_now - clock_record
          inform%factorizations = inform%factorizations + 1
          IF ( sy_status /= 0 ) GO TO 920

!  form c_dense = Q^T c

          DO i = 1, n
            data%C_dense( i ) = DOT_PRODUCT( data%Q_dense( : n , i ),  C( : n ))
          END DO
        END IF

        IF ( PRESENT( A ) ) THEN

!  if necessary allocate space to hold the additional dense matrices A_dense

          CALL SMT_put( data%A_dense%type, 'DENSE', i )
          m_dim = A%m
          data%A_dense%m = m_dim
          data%A_dense%n = n
          array_name = 'trs: A_dense%val'
          CALL SPACE_resize_array( m_dim * n, data%A_dense%val,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910

!  form A_dense = A Q

          data%A_dense%val = zero

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, m_dim
              nim1 = n * ( i - 1 )
              DO j = 1, n
                l = l + 1
                data%A_dense%val( nim1 + 1 : nim1 + n ) =                      &
                  data%A_dense%val( nim1 + 1 : nim1 + n ) +                    &
                    A%val( l ) * data%Q_dense( j, 1 : n )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m_dim
              nim1 = n * ( i - 1 )
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                data%A_dense%val( nim1 + 1 : nim1 + n ) =                      &
                  data%A_dense%val( nim1 + 1 : nim1 + n ) +                    &
                    A%val( l ) * data%Q_dense( j, 1 : n )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              nim1 = n * ( i - 1 )
              data%A_dense%val( nim1 + 1 : nim1 + n ) =                        &
                data%A_dense%val( nim1 + 1 : nim1 + n ) +                      &
                  A%val( l ) * data%Q_dense( j, 1 : n )
            END DO
          END SELECT

!  solve the dense problem

          CALL TRS_solve_main( n, radius, f, data%C_dense, data%H_dense,       &
                               data%X_dense, data, control, inform,            &
                               A = data%A_dense, Y = Y )
        ELSE
          CALL TRS_solve_diagonal( n, radius, f, data%C_dense,                 &
                                   data%H_dense%val, data%X_dense,             &
                                   control, inform )
        END IF

!  recover x = Q x_dense

        X( : n ) = MATMUL( data%Q_dense( : n , : n ), data%X_dense( : n ) )

!  record the overall time when a dense factorization is used

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%total = inform%time%total + time_now - time_start
        inform%time%clock_total =                                              &
          inform%time%clock_total + clock_now - clock_start

!  a sparse factorization will be used

      ELSE

!  solve the sparse problem

        CALL TRS_solve_main( n, radius, f, C, H, X, data, control, inform,     &
                             M, A, Y )
      END IF

!  ----
!  Exit
!  ----

      RETURN

!  -------------
!  General error
!  -------------

  910 CONTINUE
      IF ( control%out > 0 .AND. control%print_level > 0 )                     &
        WRITE( control%out, "( A, '   **  Error return ', I0,                  &
       & ' from TRS ' )" ) control%prefix, inform%status
      RETURN

!  ---------------------
!  Factorization failure
!  ---------------------

  920 CONTINUE
      IF ( control%out > 0 .AND. control%print_level > 1 ) WRITE( control%out, &
       "( A, ' error return from SYSV/SYGV: status = ', I0 )") prefix, sy_status
      inform%status = GALAHAD_error_factorization
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2010 FORMAT( A, ' time( SYSV/SYGV factorization ) = ', F0.2 )

!  End of subroutine TRS_solve

      END SUBROUTINE TRS_solve

!-*-*-*-*-*-*-  T R S _ S O L V E _ M A I N   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE TRS_solve_main( n, radius, f, C, H, X, data, control, inform, &
                                 M, A, Y )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the trust-region subproblem
!
!      minimize     1/2 <x, H x> + <c, x> + f
!      subject to    ||x||_M <= radius  or ||x||_M = radius
!      and optionally  A x = 0
!
!  where ||x||_M^2 = <x, Mx> and M is diagonally dominant, using a sparse
!  matrix factorization and secular iteration
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   n - the number of unknowns
!
!   radius - the trust-region radius
!
!   f - the value of constant term for the quadratic function
!
!   C - a vector of values for the linear term c
!
!   H -  a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of the symmetric matrix H. Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with the entries in each row in order of increasing
!                    column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!   X - the required solution vector x
!
!   data - private internal data
!
!   control - a structure containing control information. See TRS_control_type
!
!   inform - a structure containing information. See TRS_inform_type
!
!   M - an optional structure of type SMT_type used to hold the LOWER TRIANGULAR
!    part of the symmetric, DIAGONALLY DOMINANT matrix M. Four storage formats
!    are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       M%type( 1 : 10 ) = TRANSFER( 'COORDINATE', M%type )
!       M%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of M
!       M%val( : )   the values of the components of M
!       M%row( : )   the row indices of the components of M
!       M%col( : )   the column indices of the components of M
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       M%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', M%type )
!       M%val( : )   the values of the components of M, stored row by row
!       M%col( : )   the column indices of the components of M
!       M%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       M%type( 1 : 5 ) = TRANSFER( 'DENSE', M%type )
!       M%val( : )   the values of the components of M, stored row by row,
!                    with the entries in each row in order of increasing
!                    column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       M%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', M%type )
!       M%val( : )   the values of the diagonals of M, stored in order
!
!    If the argument M is absent, M will be assumed to be the identity matrix,
!    and thus ||x||_M = ||x||_2
!
!   A -  an optional structure of type SMT_type used to hold the matrix A.
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%m          the number of rows of A
!       A%ne         the number of nonzeros used to store A
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%m          the number of rows of A
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
!       A%m          the number of rows of A
!       A%val( : )   the values of the components of A, stored row by row,
!                    with the entries in each row in order of increasing
!                    column indicies.
!
!    If the argument A is absent, no linear constraints Ax = 0 will be imposed
!
!  Y - an optional array of dimension at least A%m that will be filled
!    with Lagrange multipliers for the constraints A x = 0, if present

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C
      TYPE ( SMT_type ), INTENT( IN ) :: H
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( TRS_data_type ), INTENT( INOUT ) :: data
      TYPE ( TRS_control_type ), INTENT( IN ) :: control
      TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: M
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: A
      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( * ) :: Y

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, l, it, i_max, j_max, out, nroots, n_invit, print_level
      INTEGER :: max_order, n_lambda, in_n
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record
      REAL ( KIND = wp ) :: lambda, lambda_l, lambda_u, delta_lambda, root_eps
      REAL ( KIND = wp ) :: alpha, H_f, H_f2, H_inf, utx, distx, val, rayleigh
      REAL ( KIND = wp ) :: c_norm, c_norm_over_radius, u_norm, v_norm2, w_norm2
      REAL ( KIND = wp ) :: beta, z_norm2, h_max, root1, root2, root3, curv, umu
      REAL ( KIND = wp ) :: hp, hm, mp, mm, lambda_min, lambda_max, diag_min
      REAL ( KIND = wp ) :: width, width_rel, lambda_s_l, lambda_plus
      REAL ( KIND = wp ) :: a_0, a_1, a_2, a_3, a_max
      REAL ( KIND = wp ), DIMENSION( 3 ) :: lambda_new
      REAL ( KIND = wp ), DIMENSION( 0 : max_degree ) :: x_norm2, pi_beta
      LOGICAL :: printi, printt, printd, psdef, try_zero, dummy, unit_m
      LOGICAL :: problem_file_exists, phase_1, constrained, set_y
      CHARACTER ( LEN = 1 ) :: region
      CHARACTER ( LEN = 80 ) :: array_name
      LOGICAL :: u_set = .FALSE.

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      unit_m = .NOT. PRESENT( M )
      constrained = PRESENT( A )

!  is the problem diagonal and unconstrained?

      IF ( SMT_get( H%type ) == 'DIAGONAL' .AND. unit_m .AND.                  &
           .NOT. constrained ) THEN
        CALL TRS_solve_diagonal( n, radius, f, C, H%val, X, control, inform )
        RETURN
      END IF

!  set initial values

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

      inform%hard_case = .FALSE.
      inform%pole = zero
      inform%IR_inform%status = 0
      inform%IR_inform%alloc_status = 0
      inform%IR_inform%bad_alloc = ''

      IF ( constrained ) THEN
        IF ( A%m <= 0 ) constrained = .FALSE.
      END IF
      IF ( constrained .AND. PRESENT( Y ) ) THEN
        set_y = .TRUE.
      ELSE
        set_y = .FALSE.
      END IF
      phase_1 = .TRUE.
      X = zero ; inform%x_norm = zero ; inform%obj = f

      data%control = control
      data%control%IR_control%record_residuals = .TRUE.
      IF ( data%control%initialize_approx_eigenvector )                        &
        data%get_initial_u = .TRUE.
      out = control%out

!  record desired output level

      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1
      printd = out > 0 .AND. print_level > 2

!  choose initial values for the control parameters for the factorization

      IF ( constrained ) THEN
        data%control%SLS_control%pivot_control = 1
        data%accurate = .TRUE.
      ELSE
        data%control%SLS_control%pivot_control = 2
      END IF
      data%control%inverse_itmax = MAX( 1, data%control%inverse_itmax )
      IF ( control%max_factorizations > 0 ) THEN
        data%control%max_factorizations = control%max_factorizations
      ELSE
        data%control%max_factorizations = HUGE( 0 )
      END IF

!  ---------------------------------------------------------------
!  Set up data structure for the matrix H(lambda) = H + lambda * M
!  ---------------------------------------------------------------

!  compute the space required to hold the matrix H ...

      IF ( data%control%new_h >= 2 ) THEN
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' )
          data%h_ne = n
        CASE ( 'DENSE' )
          data%h_ne = ( n * ( n + 1 ) ) / 2
        CASE ( 'SPARSE_BY_ROWS' )
          data%h_ne = H%ptr( n + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%h_ne = H%ne
        END SELECT
      END IF

!  ... and that required to hold the matrix M ...

      IF ( .NOT. unit_m ) THEN
        M%n = n ; M%m = n
        IF ( data%control%new_m >= 2 ) THEN
          SELECT CASE ( SMT_get( M%type ) )
          CASE ( 'DIAGONAL' )
            data%m_ne = n
          CASE ( 'DENSE' )
            data%m_ne = ( n * ( n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            data%m_ne = M%ptr( n + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%m_ne = M%ne
          END SELECT
        END IF
      ELSE
        data%m_ne = n
      END IF

!  ... and, if needed, to hold A

      IF ( constrained ) THEN
        IF ( data%control%new_a >= 2 ) THEN
          data%m = A%m ; data%npm = n + data%m
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            data%a_ne = data%m * n
          CASE ( 'SPARSE_BY_ROWS' )
            data%a_ne = A%ptr( data%m + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%a_ne = A%ne
          END SELECT
        END IF
      ELSE
        data%m = 0 ; data%npm = n ; data%a_ne = 0
      END IF
      data%m_end = data%h_ne + data%m_ne

      IF ( printt ) THEN
        WRITE( out, "( A, ' ||H|| = ', ES10.4 )" )                             &
          prefix, MAXVAL( ABS( H%val( : data%h_ne ) ) )
        IF ( .NOT. unit_m ) WRITE( out, "( A, ' ||M|| = ', ES10.4 )" )         &
          prefix, MAXVAL( ABS( M%val( : data%m_ne ) ) )
        IF ( constrained ) WRITE( out, "( A, ' ||A|| = ', ES10.4 )" )          &
          prefix, MAXVAL( ABS( A%val( : data%a_ne ) ) )
      END IF
!write(6,*)  H%val( : data%h_ne )
!stop
      IF ( data%control%new_h >= 2 .OR.                                        &
           ( .NOT. unit_m .AND. data%control%new_m >= 2 ) .OR.                 &
           ( constrained .AND. data%control%new_a >= 2 ) ) THEN

!  add space required for lambda * M and A

        data%H_lambda%n = n + data%m
        data%H_lambda%ne = data%h_ne + data%m_ne + data%a_ne

!  allocate space to hold the matrix in co-ordinate form

        array_name = 'trs: H_lambda%row'
        CALL SPACE_resize_array( data%H_lambda%ne, data%H_lambda%row,          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

        array_name = 'trs: H_lambda%col'
        CALL SPACE_resize_array( data%H_lambda%ne, data%H_lambda%col,          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

        array_name = 'trs: H_lambda%val'
        CALL SPACE_resize_array( data%H_lambda%ne, data%H_lambda%val,          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

        CALL SMT_put( data%H_lambda%type, 'COORDINATE', inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = GALAHAD_error_allocate
          GO TO 910
        END IF

!  fit the data from H into the coordinate storage scheme provided

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' )
          DO i = 1, n
            data%H_lambda%row( i ) = i ; data%H_lambda%col( i ) = i
          END DO
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1
              data%H_lambda%row( l ) = i ; data%H_lambda%col( l ) = j
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              data%H_lambda%row( l ) = i
              data%H_lambda%col( l ) = H%col( l )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          data%H_lambda%row( : data%h_ne ) = H%row( : data%h_ne )
          data%H_lambda%col( : data%h_ne ) = H%col( : data%h_ne )
        END SELECT

        IF ( .NOT. unit_m ) THEN

!  fit the data from M into the coordinate storage scheme provided if required

          SELECT CASE ( SMT_get( M%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              data%H_lambda%row( data%h_ne + i ) = i
              data%H_lambda%col( data%h_ne + i ) = i
            END DO
          CASE ( 'DENSE' )
            l = data%h_ne
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                data%H_lambda%row( l ) = i ; data%H_lambda%col( l ) = j
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = M%ptr( i ), M%ptr( i + 1 ) - 1
                data%H_lambda%row( data%h_ne + l ) = i
                data%H_lambda%col( data%h_ne + l ) = M%col( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            data%H_lambda%row( data%h_ne + 1 : data%h_ne + data%m_ne ) =       &
              M%row( : data%m_ne )
            data%H_lambda%col( data%h_ne + 1 : data%h_ne + data%m_ne ) =       &
              M%col( : data%m_ne )
          END SELECT
        ELSE

!  otherwise append the data for lambda * I

          DO i = 1, n
            data%H_lambda%row( data%h_ne + i ) = i
            data%H_lambda%col( data%h_ne + i ) = i
          END DO
        END IF

!  fit the data from A into the coordinate storage scheme provided if required

        IF ( constrained ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            l = data%m_end
            DO i = 1, data%m
              DO j = 1, n
                l = l + 1
                data%H_lambda%row( l ) = n + i ; data%H_lambda%col( l ) = j
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                data%H_lambda%row( data%m_end + l ) = n + i
                data%H_lambda%col( data%m_end + l ) = A%col( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            data%H_lambda%row( data%m_end + 1 : data%m_end + data%a_ne ) =     &
              n + A%row( : data%a_ne )
            data%H_lambda%col( data%m_end + 1 : data%m_end + data%a_ne ) =     &
              A%col( : data%a_ne )
          END SELECT
        END IF
      END IF

!  introduce the numerical values from A

      IF ( constrained .AND. ( data%control%new_h >= 2 .OR.                    &
           ( .NOT. unit_m .AND. data%control%new_m >= 2 ) .OR.                 &
           data%control%new_a >= 1 ) )                                         &
        data%H_lambda%val( data%m_end + 1 : data%H_lambda%ne )                 &
          = A%val( : data%a_ne )

!  output problem data

      IF ( control%problem > 0 ) THEN
        INQUIRE( FILE = control%problem_file, EXIST = problem_file_exists )
        IF ( problem_file_exists ) THEN
          OPEN( control%problem, FILE = control%problem_file,                  &
                FORM = 'FORMATTED', STATUS = 'OLD' )
          REWIND control%problem
        ELSE
          OPEN( control%problem, FILE = control%problem_file,                  &
                FORM = 'FORMATTED', STATUS = 'NEW' )
        END IF
        WRITE( control%problem, * ) n, COUNT( C( : n ) /= zero ),              &
          COUNT( H%val( : data%h_ne ) /= zero )
        WRITE( control%problem, * ) radius, f
        DO i = 1, n
          IF ( C( i ) /= zero ) WRITE( control%problem, * ) i, C( i )
        END DO
        DO l = 1, data%h_ne
          IF ( H%val( l ) /= zero ) WRITE( control%problem, * )                &
            data%H_lambda%row( l ), data%H_lambda%col( l ), H%val( l )
        END DO
        CLOSE( control%problem )
      END IF

      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%assemble = inform%time%assemble + time_now - time_start
      inform%time%clock_assemble =                                             &
        inform%time%clock_assemble + clock_now - clock_start
      IF ( printt ) WRITE( out, "( A,  ' time( assembly ) = ', F0.2 )" )       &
        prefix, clock_now - clock_start

!  set up linear equation solver-dependent data

      IF ( data%control%new_h >= 2 .OR.                                        &
           ( .NOT. unit_m .AND. data%control%new_m >= 2 ) .OR.                 &
           ( constrained .AND. data%control%new_a >= 2 ) ) THEN
        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( constrained ) THEN
          CALL SLS_initialize_solver( control%symmetric_linear_solver,         &
                                      data%SLS_data, inform%SLS_inform )
        ELSE
          CALL SLS_initialize_solver( control%definite_linear_solver,          &
                                      data%SLS_data, inform%SLS_inform )
        END IF
        inform%max_entries_factors = 0

!  Perform an analysis of the spasity pattern to identify a good
!  ordering for sparse factorization

        CALL SLS_analyse( data%H_lambda, data%SLS_data,                        &
                          data%control%SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse =                                            &
          inform%time%clock_analyse + clock_now - clock_record
        IF ( printt ) WRITE( out, 2000 ) prefix, clock_now - clock_record

!  test that the analysis succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( A, ' error return from ',               &
         &  'SLS_analyse: status = ', I0 )" ) prefix, inform%SLS_inform%status
          inform%status = GALAHAD_error_analysis ;  GO TO 910 ; END IF
      END IF

!  =====================
!  Array (re)allocations
!  =====================

!  allocate U, V, Y and Z

      array_name = 'trs: U'
      CALL SPACE_resize_array( data%npm, data%U,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'trs: V'
      CALL SPACE_resize_array( data%npm, data%V,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'trs: Y'
      CALL SPACE_resize_array( data%npm, data%Y,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'trs: Z'
      CALL SPACE_resize_array( data%npm, data%Z,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

!  compute the sums of the absolute values of off-diagonal terms of H (in Y),
!  its diagonal terms (in Z) and the square of its Frobenius norm. Also, compute
!  H u or H c (in V) and the largest off-diagonal (h_max) in row/column
!  i_max/j_max

      IF ( unit_m ) THEN
        data%Y( : n ) = zero ; data%Z( : n ) = zero ; H_f2 = zero
        data%V( : n ) = zero
        i_max = 0 ; j_max = 0 ; h_max = zero
        DO l = 1, data%h_ne
          i = data%H_lambda%row( l ) ; j = data%H_lambda%col( l )
          val = H%val( l )
          IF ( i == j ) THEN
            data%Z( i ) = data%Z( i ) + val
!           H_f2 = H_f2 + val ** 2
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( i )
            END IF
          ELSE
            data%Y( i ) = data%Y( i ) + ABS( val )
            data%Y( j ) = data%Y( j ) + ABS( val )
            H_f2 = H_f2 + two * val ** 2
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( j )
              data%V( j ) = data%V( j ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( j )
              data%V( j ) = data%V( j ) + val * data%U( i )
            END IF
            IF ( ABS( val ) > ABS( h_max ) ) THEN
              i_max = i ; j_max = j ; h_max = val
            END IF
          END IF
        END DO
        H_f2 = H_f2 + DOT_PRODUCT( data%Z( : n ), data%Z( : n ) )

!  compute the Frobenius and infinity norms of H

        H_f = SQRT( H_f2 )
        H_inf = MAXVAL( ABS( data%Z( : n ) ) + data%Y( : n ) )

!  compute the two-norm of c and the Rayleigh quotient u^T H u / u^T u or
!  c^T H c / c^T c as required

        c_norm = TWO_NORM( C )
        IF ( c_norm > zero ) THEN
          IF ( data%get_initial_u ) THEN
            rayleigh = DOT_PRODUCT( data%V( : n ), C ) / c_norm ** 2
          ELSE
            rayleigh = DOT_PRODUCT( data%V( : n ), data%U( : n ) )
          END IF
        ELSE
          rayleigh = zero
        END IF

!  compute the leftmost eigenvalue of the 2 by 2 sub-matrix in rows/columns
!  i_max and j_max

        IF ( n > 1 .AND. i_max /= 0 ) THEN
          a_0 = data%Z( i_max ) *  data%Z( j_max ) - h_max** 2
          a_1 = - data%Z( i_max ) -  data%Z( j_max )
          a_2 = one
          CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,              &
                                root1, root2, roots_debug )
          rayleigh = MIN( rayleigh, root1 )

!  do the same over all nonzeros

          dummy = .FALSE.
          IF ( dummy ) THEN
            DO l = 1, data%h_ne
              i = data%H_lambda%row( l ) ; j = data%H_lambda%col( l )
              val = data%H_lambda%val( l )
              IF ( i /= j ) THEN
                a_0 = - val ** 2
                a_1 = data%Z( i ) -  data%Z( j )
                a_2 = one
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      root1, root2, roots_debug )
                rayleigh = MIN( rayleigh, data%Z( i ) + root1 )
              END IF
            END DO
          END IF
        END IF

!  record the Gershgorin bounds on the eigenvalues, and the smallest diagonal

        lambda_min = MAX( MINVAL( data%Z( : n ) - data%Y( : n ) ),             &
                          - H_f, - H_inf )
        lambda_max = MIN( MAXVAL( data%Z( : n ) + data%Y( : n ) ),             &
                          H_f, H_inf )
        diag_min = MINVAL( data%Z( : n ) )

      ELSE

!  compute the sums of the absolute values of off-diagonal terms of H (in Y),
!  its diagonal terms (in Z) and H u or H c (in V)

        data%Y( : n ) = zero ; data%Z( : n ) = zero ; data%V( : n ) = zero
        i_max = 0 ; j_max = 0 ; h_max = zero
        DO l = 1, data%h_ne
          i = data%H_lambda%row( l ) ; j = data%H_lambda%col( l )
          val = H%val( l )
          IF ( i == j ) THEN
            data%Z( i ) = data%Z( i ) + val
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( i )
            END IF
          ELSE
            data%Y( i ) = data%Y( i ) + ABS( val )
            data%Y( j ) = data%Y( j ) + ABS( val )
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( j )
              data%V( j ) = data%V( j ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( j )
              data%V( j ) = data%V( j ) + val * data%U( i )
            END IF
          END IF
        END DO

!  compute u^T H u or c^T H c as required

        IF ( data%get_initial_u ) THEN
          curv = DOT_PRODUCT( data%V( : n ), C )
        ELSE
          curv = DOT_PRODUCT( data%V( : n ), data%U( : n ) )
        END IF

!  attempt an L B L^T factorization of M

        data%H_lambda%val( 1 : data%h_ne ) = zero
        data%H_lambda%val( data%h_ne + 1 : data%h_ne + data%m_ne ) =           &
          M%val(  1 : data%m_ne )

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( data%H_lambda, data%SLS_data,                      &
                            data%control%SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
        inform%factorizations = inform%factorizations + 1
        inform%max_entries_factors = MAX( inform%max_entries_factors,          &
                                          inform%SLS_inform%entries_in_factors )

!  test that the factorization succeeded

        IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
          psdef = .TRUE.
        ELSE IF ( inform%SLS_inform%status == GALAHAD_error_inertia ) THEN
          IF ( printd ) WRITE( out, "( A, ' error return from ',               &
         &  'SLS_factorize: status = ', I0 )" ) prefix, inform%SLS_inform%status
          GO TO 930
!         psdef = .FALSE.
        ELSE
          GO TO 920
        END IF

!  compute M^{-1} c (in V)

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        data%V( : n ) = C ; data%V( n + 1 : data%npm ) = zero
        CALL IR_solve( data%H_lambda, data%V( : data%npm ),                    &
                       data%IR_data, data%SLS_data,                            &
                       data%control%IR_control, data%control%SLS_control,      &
                       inform%IR_inform, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the M^-1 norm of c, while checking that M is positive definite

        c_norm = DOT_PRODUCT( data%V( : n ), C )
        IF ( c_norm < zero ) THEN
          IF ( printd ) WRITE( out, "( A, ' c^T M^-1 c =', ES12.4, ' < 0' )" ) &
            prefix, c_norm
          GO TO 930
        END IF
        c_norm = SQRT( c_norm )

!  allocate space to hold the diagonals and sums of absolute values of the
!  off diagonals of M

        array_name = 'trs: M_diag'
        CALL SPACE_resize_array( n, data%M_diag,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

        array_name = 'trs: M_offd'
        CALL SPACE_resize_array( n, data%M_offd,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

!  compute the sums of the absolute values of off-diagonal terms of M
!  (in M_offd), its diagonal terms (in M_diag) and M u or M c (in V)

        data%M_offd( : n ) = zero ; data%M_diag( : n ) = zero
        data%V( : n ) = zero
        i_max = 0 ; j_max = 0 ; h_max = zero
        DO l = data%h_ne + 1, data%h_ne + data%m_ne
          i = data%H_lambda%row( l ) ; j = data%H_lambda%col( l )
          val = data%H_lambda%val( l )
          IF ( i == j ) THEN
            data%M_diag( i ) = data%M_diag( i ) + val
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( i )
            END IF
          ELSE
            data%M_offd( i ) = data%M_offd( i ) + ABS( val )
            data%M_offd( j ) = data%M_offd( j ) + ABS( val )
            IF ( data%get_initial_u ) THEN
              data%V( i ) = data%V( i ) + val * C( j )
              data%V( j ) = data%V( j ) + val * C( i )
            ELSE
              data%V( i ) = data%V( i ) + val * data%U( j )
              data%V( j ) = data%V( j ) + val * data%U( i )
            END IF
          END IF
        END DO

!  compute the Rayleigh quotient u^T H u / u^T M u or c^T H c / c^T M c
!  as required

        IF ( data%get_initial_u ) THEN
          umu = DOT_PRODUCT( data%V( : n ), C )
          IF ( umu > zero ) THEN
            rayleigh = curv / umu
          ELSE
            rayleigh = zero
          END IF
        ELSE
          umu = DOT_PRODUCT( data%V( : n ), data%U( : n ) )
          IF ( umu > zero ) THEN
            rayleigh = curv / umu
          ELSE
            rayleigh = zero
          END IF
        END IF

!  ensure that M is strictly diagonally dominant

        lambda_min = infinity ; lambda_max = - infinity
        DO i = 1, n
          mm = data%M_diag( i ) - data%M_offd( i )
          IF ( mm <= zero ) THEN
            IF ( printd ) WRITE( out, "( A, ' Gershgorin lower bound =',       &
           &   ES12.4, ' < 0' )" ) prefix, mm
            GO TO 930
          END IF

!  find Gershgorin-like bounds on the generalised eigenvalues of H - lambda M

          hp = data%Z( i ) + data%Y( i )
          hm = data%Z( i ) - data%Y( i )
          mp = data%M_diag( i ) + data%M_offd( i )
          lambda_min = MIN( lambda_min, hm / mp, hm / mm )
          lambda_max = MAX( lambda_max, hp / mp, hp / mm )
        END DO

!  record the minimum relative diagonal

        diag_min = MINVAL( data%Z( : n ) / data%M_diag( : n ) )
      END IF
      IF ( printt ) WRITE( out, "( A, ' ||c|| = ', ES10.4 )" ) prefix, c_norm

!  The real line is partitioned into disjoint sets
!     N = { lambda: lambda <= max(0, -lambda_1(H))}
!     L = { lambda: max(0, -lambda_1(H)) < lambda <= lambda_optimal } and
!     G = { lambda: lambda > lambda_optimal };
!  for the equality problem, 0 plays no role in N and L.
!  The aim is to find a lambda in L, as generally then Newton's method
!  will converge both globally and ultimately quadratically. We also let
!     F = L union G
!
!  Construct values lambda_l and lambda_u for which lambda_l <= lambda_optimal
!   <= lambda_u, and ensure that all iterates satisfy lambda_l <= lambda
!   <= lambda_u

      IF ( c_norm > zero ) THEN
        c_norm_over_radius = c_norm / radius
        IF ( constrained ) THEN
          IF ( data%control%equality_problem ) THEN
            lambda_s_l = - infinity
            lambda_l = MAX( data%control%lower,                                &
                            - lambda_max )
!                           c_norm_over_radius - lambda_max )
            lambda_u = MIN( data%control%upper,                                &
                            c_norm_over_radius - lambda_min )
          ELSE
            lambda_s_l = zero
            lambda_l = MAX( data%control%lower, zero )
!                           c_norm_over_radius - lambda_max )
            lambda_u = MIN( data%control%upper,                                &
                            MAX( zero, c_norm_over_radius - lambda_min ) )
          END IF
        ELSE
          IF ( data%control%equality_problem ) THEN
            lambda_s_l = MAX( - rayleigh, - diag_min )
            lambda_l = MAX( data%control%lower, lambda_s_l,                    &
                            c_norm_over_radius - lambda_max )
            lambda_u = MIN( data%control%upper,                                &
                            c_norm_over_radius - lambda_min )
          ELSE
            lambda_s_l = MAX( zero, - rayleigh, - diag_min )
            lambda_l = MAX( data%control%lower, lambda_s_l,                    &
                            c_norm_over_radius - lambda_max )
            lambda_u = MIN( data%control%upper,                                &
                            MAX( zero, c_norm_over_radius - lambda_min ) )
          END IF
        END IF
      ELSE
        IF ( constrained ) THEN
          IF ( data%control%equality_problem ) THEN
            lambda_s_l = - infinity
            lambda_l = MAX( data%control%lower, - lambda_max )
            lambda_u = MIN( data%control%upper, - lambda_min )
          ELSE
            lambda_s_l = zero
            lambda_l = MAX( data%control%lower, zero, - lambda_max )
            lambda_u = MIN( data%control%upper, MAX( zero, - lambda_min ) )
          END IF
        ELSE
          IF ( data%control%equality_problem ) THEN
            lambda_s_l = MAX( - rayleigh, - diag_min )
            lambda_l = MAX( data%control%lower, - lambda_max, lambda_s_l )
            lambda_u = MIN( data%control%upper, - lambda_min )
          ELSE
            lambda_s_l = MAX( zero, - rayleigh, - diag_min )
            lambda_l = MAX( data%control%lower, zero, - lambda_max,            &
                            lambda_s_l )
            lambda_u = MIN( data%control%upper, MAX( zero, - lambda_min ) )
          END IF
        END IF
        inform%hard_case = .TRUE.
      END IF

      IF ( lambda_l > lambda_u ) THEN
        WRITE( out, "( ' lambda_l = ', ES12.4, ' > lambda_u = ', ES12.4 )" )   &
          lambda_l, lambda_u
        STOP
      END IF

!  assign the initial lambda

      IF ( control%force_Newton .AND. rayleigh > zero ) THEN
        lambda = zero
      ELSE IF ( data%control%use_initial_multiplier ) THEN
        IF ( data%control%initial_multiplier >= lambda_l .AND.                 &
             data%control%initial_multiplier <= lambda_u ) THEN
          lambda =  data%control%initial_multiplier
        ELSE
          lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                   &
                        lambda_l + theta_eps * ( lambda_u - lambda_l ) )
        END IF
      ELSE
        IF ( data%control%equality_problem ) THEN
          lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
        ELSE
          lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                   &
                        lambda_l + theta_eps * ( lambda_u - lambda_l ) )
        END IF
      END IF

!  if the problem has changed, restart the history of small steps

      IF ( data%control%new_h >= 1 .OR.                                        &
           ( .NOT. unit_m .AND. data%control%new_m >= 1 ) .OR.                 &
           ( constrained .AND. data%control%new_a >= 1 ) )                     &
        inform%len_history = 0

!  introduce the numerical values from H ...

      IF ( data%control%new_h >= 1 .OR.                                        &
           ( .NOT. unit_m .AND. data%control%new_m >= 2 ) .OR.                 &
           ( constrained .AND. data%control%new_a >= 2 ) )                     &
        data%H_lambda%val( : data%h_ne ) = H%val( : data%h_ne )

!  ... and from A

!      IF ( constrained .AND. ( data%control%new_h >= 2 .OR.                   &
!           ( .NOT. unit_m .AND. data%control%new_m >= 2 ) .OR.                &
!           data%control%new_a >= 1 ) )                                        &
!        data%H_lambda%val( data%m_end + 1 : data%H_lambda%ne )                &
!         = A%val( : data%a_ne )

      try_zero = lambda > zero .AND. lambda_l == zero
      region = ' '
      max_order = MAX( 1, MIN( max_degree, control%taylor_max_degree ) )
      root_eps = SQRT( epsmch )

      IF ( printt )                                                            &
        WRITE( out, "( A, 4X, 28( '-' ), ' phase one ', 28( '-' ) )" ) prefix

!write(6,*) ' hard ? ', inform%hard_case

!  start the main loop

      it = 0 ; in_n = 0

      DO
        it = it + 1

!  add lambda * M to H to form H(lambda)

 100    CONTINUE
        IF ( unit_m ) THEN
          data%H_lambda%val( data%h_ne + 1 : data%m_end ) = lambda
        ELSE
          data%H_lambda%val( data%h_ne + 1 : data%m_end ) =                    &
            lambda * M%val(  1 : data%m_ne )
        END IF

!  attempt an L B L^T factorization of H(lambda)

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( data%H_lambda, data%SLS_data,                      &
                            data%control%SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
        inform%factorizations = inform%factorizations + 1
        inform%max_entries_factors = MAX( inform%max_entries_factors,          &
                                          inform%SLS_inform%entries_in_factors )

!  test that the factorization succeeded

        IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
          IF ( data%control%SLS_control%pivot_control == 2 ) THEN
            psdef = .TRUE.
          ELSE
            psdef = inform%SLS_inform%negative_eigenvalues == data%m
          END IF
        ELSE
          IF ( constrained ) THEN
            IF ( inform%SLS_inform%rank /= data%npm ) THEN
              psdef = .FALSE.
            ELSE
              GO TO 920
            END IF
          ELSE
            IF ( inform%SLS_inform%status == GALAHAD_error_inertia ) THEN
              psdef = .FALSE.
            ELSE
              GO TO 920
            END IF
          END IF
        END IF

!  if H(lambda) is positive definite, solve  H(lambda) x = - c

!write(6,*) ' pos def ?', psdef
        IF ( psdef ) THEN
          IF ( .NOT. inform%hard_case ) THEN
            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            IF ( constrained ) THEN
              data%Y( : n ) = - C ; data%Y( n + 1 : data%npm ) = zero
              CALL IR_solve( data%H_lambda, data%Y( : data%npm ),              &
                             data%IR_data, data%SLS_data,                      &
                             data%control%IR_control,                          &
                             data%control%SLS_control,                         &
                             inform%IR_inform, inform%SLS_inform )
              X = data%Y( : n )
              IF ( set_y )  Y( : data%m ) = data%Y( n + 1 : data%npm )
            ELSE
              X = - C
              CALL IR_solve( data%H_lambda, X, data%IR_data, data%SLS_data,  &
                             data%control%IR_control,                        &
                             data%control%SLS_control,                       &
                             inform%IR_inform, inform%SLS_inform )
            END IF
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%solve = inform%time%solve + time_now - time_record
            inform%time%clock_solve =                                          &
              inform%time%clock_factorize + clock_now - clock_record
            IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  if there has been an increase in the residuals, H + lambda M is likely
!  singular, so perturb lambda and try again

            IF ( inform%IR_inform%norm_final_residual >                        &
                 inform%IR_inform%norm_initial_residual ) THEN
! write(6, "( ' *********** WARNING B - initial and final residuals are ',     &
!& 2ES12.4 )" ) inform%IR_inform%norm_initial_residual,                        &
!               inform%IR_inform%norm_final_residual
              lambda_l = lambda
              lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
              GO TO 100
            END IF

!  compute the M-norm of x, ||x||_M

            IF ( unit_m ) THEN
              inform%x_norm = TWO_NORM( X )
              x_norm2( 0 ) = inform%x_norm ** 2
            ELSE
              CALL mop_AX( one, M, X, zero, data%Y( : n ), 0,                  &
                           symmetric = .TRUE. )
              x_norm2( 0 ) = DOT_PRODUCT( X, data%Y( : n ) )
              IF ( x_norm2( 0 ) < zero ) THEN
                IF ( printd ) WRITE( out, "( A, ' ||x||_M^2 =',                &
           &       ES12.4, ' < 0' )" ) prefix, x_norm2( 0 )
                GO TO 930
              END IF
              inform%x_norm = SQRT( x_norm2( 0 ) )
            END IF
          END IF

!  debug printing

          IF ( printd ) THEN
            WRITE( out, "( A, 8X, 'lambda', 13X, 'x_norm', 15X, 'radius' )" )  &
              prefix
            WRITE( out, "( A, 3ES20.12 )") prefix, lambda, inform%x_norm, radius
            IF ( phase_1 ) THEN
              WRITE( out, "( A, ' interval width =', ES21.14 )")               &
                prefix, lambda_u - lambda_l
            ELSE
              WRITE( out, 2020 ) prefix
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )                          &
                prefix, '  ', it, lambda_l, lambda, lambda_u
            END IF
          END IF

!write(6,*) ' la, x, del', lambda, inform%x_norm, radius

!  if the Newton step lies within the trust region, exit

          IF ( lambda == zero .AND. inform%x_norm <= radius ) THEN
            IF ( printi ) WRITE( out, "( A,                                    &
           &    ' Interior stopping criteria satisfied' )" ) prefix
            inform%obj = f + half * DOT_PRODUCT( C, X )
            inform%status = GALAHAD_ok
            GO TO 900
          END IF

! if we wish to force the Newton solution, do so

          IF ( lambda == zero .AND. data%control%force_Newton ) THEN
            IF ( printi ) WRITE( out, "( A,                                    &
           &    ' forced Newton solution taken ' )" ) prefix
            inform%obj = f + half * DOT_PRODUCT( C, X )
            inform%status = GALAHAD_ok
            GO TO 900
          END IF

!  see if we are definitely in the hard case

!         IF ( inform%x_norm <= epsmch ) THEN
          IF ( inform%x_norm == zero ) THEN

!  reset the interval bounds (the secular equation is irrelevant)

            IF ( .NOT. inform%hard_case ) THEN
              IF ( constrained ) THEN
                IF ( data%control%equality_problem ) THEN
                  lambda_l = MAX( data%control%lower, - lambda_max )
                  lambda_u = MIN( data%control%upper, - lambda_min )
                ELSE
                  lambda_l = MAX( data%control%lower, zero, - lambda_max )
                  lambda_u = MIN( data%control%upper, MAX( zero, - lambda_min ))
                END IF
              ELSE
                IF ( data%control%equality_problem ) THEN
                  lambda_l = MAX( data%control%lower, - lambda_max, lambda_s_l )
                  lambda_u = MIN( data%control%upper, - lambda_min )
                ELSE
                  lambda_l = MAX( data%control%lower, zero, - lambda_max,      &
                                  lambda_s_l )
                  lambda_u = MIN( data%control%upper, MAX( zero, - lambda_min ))
                END IF
              END IF
              inform%hard_case = .TRUE.
            END IF

!  check that the solution isn't trivially zero

            IF ( lambda_l == zero .AND. lambda_u == zero ) THEN
              lambda = zero
              IF ( printi ) THEN
                WRITE( out, 2020 ) prefix
                WRITE( out, "( A, A2, I4, 3ES22.15 )" )                        &
                  prefix, ' ', 0, lambda_l, lambda, lambda_u
                WRITE( out, "( A,                                              &
               &    ' Hard-case stopping criteria satisfied.',                 &
               &    ' Interval width =', ES11.4 )" ) prefix, lambda_u - lambda_l
              END IF
              inform%obj = f
              inform%status = GALAHAD_ok
              GO TO 900
            END IF

            region = 'G'
            lambda_u = MIN( lambda_u, lambda )
            IF ( .NOT. phase_1 ) THEN
              phase_1 = .TRUE.
              IF ( printi ) WRITE( out, 2020 ) prefix
            ELSE
              IF ( printt .OR. ( printi .AND. it == 1 ) ) WRITE( out, 2020 )   &
                prefix
            END IF
            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )              &
              prefix, region, it, lambda_l, lambda, lambda_u
            n_lambda = 0
            GO TO 200
          END IF

!  --------------------------------------------------------------------
!  The current estimate gives a good approximation to the required root
!  --------------------------------------------------------------------

          IF ( ABS( inform%x_norm - radius ) <=                                &
                 MAX( data%control%stop_normal * radius,                       &
                      data%control%stop_absolute_normal ) ) THEN
            IF ( inform%x_norm > radius ) THEN
              region = 'L'
              lambda_l = MAX( lambda_l, lambda )
            ELSE
              region = 'G'
              lambda_u = MIN( lambda_u, lambda )
            END IF
            IF ( ( phase_1 .AND. printi ) .OR. printt .OR.                     &
                 ( printi .AND. it == 1 ) ) WRITE( out, 2030 ) prefix
            IF ( printi ) THEN
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,         &
              it, ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
              WRITE( out, "( A,                                                &
           &    ' Normal stopping criteria satisfied' )" ) prefix
            END IF
            inform%status = GALAHAD_ok
            EXIT
          END IF

!  check to see if the factorization limit has been exceeded

          IF ( inform%factorizations > data%control%max_factorizations ) THEN
            inform%multiplier = lambda ; inform%pole = lambda_s_l
            inform%obj =                                                       &
              f + half * ( DOT_PRODUCT( C, X ) - lambda * x_norm2( 0 ) )
            inform%status = GALAHAD_error_max_iterations ; GO TO 910
          END IF

!  determine which region the current lambda lies in

          IF ( inform%x_norm > radius ) THEN

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------

            region = 'L'
            lambda_l = MAX( lambda_l, lambda )

!  record that we are now in phase 2

            IF ( phase_1 ) THEN
              phase_1 = .FALSE.
              delta_lambda = zero
              IF ( printd ) THEN
                WRITE( out, 2020 ) prefix
                WRITE( out, "( A, A2, I4, 3ES22.15 )" )                        &
                  prefix, region, it, lambda_l, lambda, lambda_u
              END IF
              IF ( printt ) THEN
                WRITE( out, "( A, 4X, 28( '-' ), ' phase two ', 28( '-' ) )" ) &
                  prefix
              END IF
              IF ( printi ) WRITE( out, 2030 ) prefix
            ELSE
              IF ( printt .OR. ( printi .AND. it == 1 ) ) WRITE( out, 2030 )   &
                prefix
            END IF

!  a lambda in L has been found. It is now simply a matter of applying
!  a variety of Taylor-series-based methods starting from this lambda

            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" ) prefix,      &
              region, it, ABS( inform%x_norm - radius ), lambda,               &
              ABS( delta_lambda )

!  precaution against rounding producing lambda outside L

            IF ( lambda > lambda_u ) THEN
              WRITE( out, "( A,                                                &
           &    ' Error - too ill conditioned' )" ) prefix
              inform%status = GALAHAD_error_ill_conditioned
              EXIT
            END IF
          ELSE

!  ----------------------------
!  The current lambda lies in G
!  ----------------------------

            region = 'G'
            lambda_u = MIN( lambda_u, lambda )
            IF ( .NOT. phase_1 ) THEN
              phase_1 = .TRUE.
              IF ( printi ) WRITE( out, 2020 ) prefix
            ELSE
              IF ( printt .OR. ( printi .AND. it == 1 ) ) WRITE( out, 2020 )   &
                prefix
            END IF
            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )              &
              prefix, region, it, lambda_l, lambda, lambda_u

!  -----------------------------------------------------
!  The solution lies in the interior of the trust-region
!  -----------------------------------------------------

            IF ( lambda == zero .AND. .NOT. data%control%equality_problem ) THEN
              IF ( printi ) WRITE( out, "( A,                                  &
             &    ' Interior stopping criteria satisfied' )" ) prefix
              inform%obj = f + half * DOT_PRODUCT( C, X )
              inform%status = GALAHAD_ok
              GO TO 900
            END IF

!  record, for the future, values of lambda which give small ||x||

            IF ( inform%len_history < history_max ) THEN
              inform%len_history = inform%len_history + 1
              inform%history( inform%len_history )%lambda = lambda
              inform%history( inform%len_history )%x_norm = inform%x_norm
            END IF
          END IF

!  compute first derivatives of x^T M x

          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = M x

          IF ( data%accurate ) THEN
            IF ( unit_m ) data%Y( : n ) = X
            data%Z( : n ) = data%Y( : n )
            IF ( constrained ) data%Z( n + 1 : data%npm ) = zero
            CALL IR_solve( data%H_lambda, data%Z( : data%npm ),                &
                           data%IR_data, data%SLS_data,                        &
                           data%control%IR_control,                            &
                           data%control%SLS_control,                           &
                           inform%IR_inform, inform%SLS_inform )

!  find y so that L y = M x

          ELSE
            IF ( unit_m ) data%Y( : n ) = X
            CALL SLS_part_solve( 'L', data%Y( : n ), data%SLS_data,            &
                                 data%control%SLS_control, inform%SLS_inform )

!  find z so that D L z = D y = M x

            data%Z( : n ) = data%Y( : n )
            CALL SLS_part_solve( 'D', data%Z( : n ), data%SLS_data,            &
                                 data%control%SLS_control, inform%SLS_inform )
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  form ||w||^2 = y^T z = x^T L^-T D^-1 L^-1 x = x^T H^-1(lambda) x

          w_norm2 = DOT_PRODUCT( data%Y( : n ), data%Z( : n ) )

!  compute the first derivative of x_norm2 = x^T M x

          x_norm2( 1 ) = - two * w_norm2
!write(6,*) ' x_norm2  ', x_norm2(1)

!  compute pi_beta = ||x||^beta and its first derivative when beta = - 1

          beta = - one
          CALL TRS_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = - 1)

          delta_lambda = - ( pi_beta( 0 ) - ( radius ) ** beta ) / pi_beta( 1 )
          n_lambda = 1
!write(6,*) delta_lambda, ( pi_beta( 0 ) - ( radius ) ** beta ), pi_beta( 1 )
          lambda_new( n_lambda ) = lambda + delta_lambda

          IF ( ( max_order >= 3 .AND. region == 'L' ) .OR.                     &
               ( max_order >= 2 .AND. region == 'G' ) ) THEN
            IF ( .NOT. data%accurate ) THEN
              CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
              CALL SLS_part_solve( 'U', data%Z( : n ), data%SLS_data,          &
                                   data%control%SLS_control,                   &
                                   inform%SLS_inform )
              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_factorize + clock_now - clock_record
              IF ( printt )                                                    &
                WRITE( out, 2050 ) prefix, clock_now - clock_record
            END IF

!  form z^T M z

            IF ( unit_m ) THEN
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Z( : n ) )
            ELSE
              CALL mop_AX( one, M, data%Z( : n ), zero, data%Y( : n ), 0,      &
                           symmetric = .TRUE. )
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )
            END IF

!  compute the second derivative of x_norm2 = x^T M x

            x_norm2( 2 ) = six * z_norm2

            IF ( max_order >= 3 ) THEN
              CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = M x

              IF ( data%accurate ) THEN
                IF ( unit_m ) THEN
                  data%Y( : n ) = data%Z( : n )
                ELSE
                  data%Z( : n ) = data%Y( : n )
                END IF
                IF ( constrained ) data%Z( n + 1 : data%npm ) = zero
                CALL IR_solve( data%H_lambda, data%Z( : data%npm ),            &
                               data%IR_data, data%SLS_data,                    &
                               data%control%IR_control,                        &
                               data%control%SLS_control,                       &
                               inform%IR_inform, inform%SLS_inform )
!  find z so that L z = x'

              ELSE
                IF ( unit_m ) THEN
                  CALL SLS_part_solve( 'L', data%Z( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )

!  find y so that D L y = D z = x'

                  data%Y( : n ) = data%Z( : n )
                  CALL SLS_part_solve( 'D', data%Y( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )
!  find z so that L y = M x'

                ELSE
                  CALL SLS_part_solve( 'L', data%Y( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )

!  find y so that D L z = D y = x'

                  data%Z( : n ) = data%Y( : n )
                  CALL SLS_part_solve( 'D', data%Z( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )
                END IF
              END IF
              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_factorize + clock_now - clock_record
              IF ( printt ) WRITE( out, 2050 ) prefix,                         &
                clock_now - clock_record

!  form ||v||^2 = z^T y = x'^T L^-T D^-1 L^-1 x' = x'^T H^-1(lambda) x'

              v_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )

!  compute the third derivatives of x_norm2 = x^T M x

              x_norm2( 3 ) = - twentyfour * v_norm2
            END IF
          END IF

!  compute pi_beta = ||x||^beta and its derivatives for various beta
!  and the resulting Taylor series approximants

 200      CONTINUE
!write(6,*) ' x > del?', inform%x_norm > radius, ' hard ? ', inform%hard_case
          IF ( inform%x_norm > radius ) THEN

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------

            IF ( max_order >= 3 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

              beta = two
              CALL TRS_pi_derivs( max_order, beta, x_norm2( : max_order ),     &
                                  pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

              a_0 = pi_beta( 0 ) - ( radius ) ** beta
              a_1 = pi_beta( 1 )
              a_2 = half * pi_beta( 2 )
              a_3 = sixth * pi_beta( 3 )
              a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
              IF ( a_max > zero ) THEN
                a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                a_2 = a_2 / a_max ; a_3 = a_3 / a_max
              END IF
              CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,         &
                                root1, root2, root3, roots_debug )
              n_lambda = n_lambda + 1
              IF ( nroots == 3 ) THEN
                lambda_new( n_lambda ) = lambda + root3
!write(6,*) root3
              ELSE
                lambda_new( n_lambda ) = lambda + root1
!write(6,*) root1
              END IF

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.4

              beta = - point4
              CALL TRS_pi_derivs( max_order, beta, x_norm2( : max_order ),     &
                                  pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

              a_0 = pi_beta( 0 ) - ( radius ) ** beta
              a_1 = pi_beta( 1 )
              a_2 = half * pi_beta( 2 )
              a_3 = sixth * pi_beta( 3 )
              a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
              IF ( a_max > zero ) THEN
                a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                a_2 = a_2 / a_max ; a_3 = a_3 / a_max
              END IF
              CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,         &
                                root1, root2, root3, roots_debug )
              n_lambda = n_lambda + 1
              IF ( nroots == 3 ) THEN
!write(6,*) root3
                lambda_new( n_lambda ) = lambda + root3
              ELSE
!write(6,*) root1
                lambda_new( n_lambda ) = lambda + root1
              END IF
            END IF

!  record all of the estimates of the optimal lambda

            IF ( printd ) THEN
              WRITE( out, "( A, ' lambda_t (', I1, ')', 3ES20.13 )" )          &
                prefix, MAXLOC( lambda_new( : n_lambda ) ),                    &
                lambda_new( : MIN( 3, n_lambda ) )
              IF ( n_lambda > 3 ) WRITE( out, "( A, 13X, 3ES20.13 )" )         &
                prefix, lambda_new( 4 : MIN( 6, n_lambda ) )
              IF ( n_lambda > 6 ) WRITE( out, "( A, 13X, 3ES20.13 )" )         &
                prefix, lambda_new( 7 : MIN( 9, n_lambda ) )
            END IF

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
!write(6,*)  lambda_new( : n_lambda )
!write(6,*)  lambda_plus, lambda
            delta_lambda = lambda_plus - lambda
            lambda = lambda_plus

!  improve the lower bound if possible

            lambda_l = MAX( lambda_l, lambda_plus )

!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
!write(6,*) ABS( delta_lambda ), MAX( one, ABS( lambda ) )
              IF ( printi )                                                    &
                WRITE( out, "( A, ' Warning - step too small' )" ) prefix
              inform%status = GALAHAD_ok
              EXIT
            END IF
          ELSE

!  ----------------------------
!  The current lambda lies in G
!  ----------------------------

            IF ( .NOT. inform%hard_case ) THEN
              IF ( max_order >= 2 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.666

                beta = - twothirds
                CALL TRS_pi_derivs( 2, beta, x_norm2( : 2 ), pi_beta( : 2 ) )

!  compute the "quadratic Taylor approximaton" step (beta = - 0.666)

                a_0 = pi_beta( 0 ) - ( radius ) ** beta
                a_1 = pi_beta( 1 )
                a_2 = half * pi_beta( 2 )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      root1, root2, roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda + root1

                IF ( max_order >= 3 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.4

                  beta = - point4
                  CALL TRS_pi_derivs( max_order, beta, x_norm2( : max_order ), &
                                      pi_beta( : max_order ) )

!  Compute the "cubic Taylor approximaton" step (beta = - 0.4)

                  a_0 = pi_beta( 0 ) - ( radius ) ** beta
                  a_1 = pi_beta( 1 )
                  a_2 = half * pi_beta( 2 )
                  a_3 = sixth * pi_beta( 3 )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    root1, root2, root3, roots_debug )
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda + root1
                END IF
              END IF

!  record all of the estimates of the optimal lambda

              IF ( printd ) WRITE( out, "( A, ' lambda_t', 3ES20.12 )" )       &
                prefix, lambda_new( : n_lambda )

!  compute the best Taylor improvement

              lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
              delta_lambda = lambda_plus - lambda

!  improve the lower bound if possible

              lambda_l = MAX( lambda_l, lambda_plus )

!  if lambda = 0 hasn't yet been tried, do so

              IF ( try_zero ) THEN
                try_zero = .FALSE.
                IF ( MAXVAL( lambda_new( : n_lambda ) ) < zero ) THEN
                  lambda = zero
                  CYCLE
                END IF
              END IF

!  check that the best Taylor improvement is significant

              IF ( ABS( delta_lambda ) <                                       &
                   epsmch * MAX( one, ABS( lambda ) ) ) THEN
                IF ( lambda_u - lambda_l == zero ) THEN
                  inform%hard_case = .TRUE.
                ELSE
                  IF ( printi )                                                &
                    WRITE( out, "( A, ' Warning - step too small' )" ) prefix
                  inform%status = GALAHAD_ok
                  EXIT
                END IF
              END IF
            ELSE
              delta_lambda = zero
            END IF

!  - - - - - - - - - -
!  Potential hard case
!  - - - - - - - - - -

!  if it seems as if it may be necessary, build an estmate of the leftmost
!  eigenvalue and its vector u using inverse iteration

            IF ( inform%hard_case .OR.                                         &
                 lambda_u - lambda_s_l  < data%control%start_invit_tol *       &
                    MAX( ABS( lambda_l ), ABS( lambda_u ) ) ) THEN
              IF ( data%get_initial_u ) THEN
                data%get_initial_u = .FALSE.

!  start the inverse iteration with a random vector orthogonal to c

                DO i = 1, n
                  CALL RAND_random_real( data%seed, .FALSE., data%U( i ) )
                END DO
                IF ( c_norm > zero ) THEN
                  alpha = DOT_PRODUCT( C, data%U( : n ) ) / c_norm ** 2
                  data%U( : n ) = data%U( : n ) - alpha * C
                END IF

!  normalize u

                IF ( unit_m ) THEN
                  u_norm = TWO_NORM( data%U( : n ) )
                ELSE
                  CALL mop_AX( one, M, data%U( : n ), zero, data%Y( : n ),     &
                               0, symmetric = .TRUE. )
                  u_norm = DOT_PRODUCT( data%U( : n ), data%Y( : n ) )
                  IF ( u_norm < zero ) THEN
                    IF ( printd ) WRITE( out, "( A, ' ||u||_M^2 =',            &
                   &  ES12.4, ' < 0' )" ) prefix, u_norm
                    GO TO 930
                  END IF
                  u_norm = SQRT( u_norm )
                END IF
                data%U( : n ) = data%U( : n ) / u_norm
                u_set = .TRUE.
              ELSE

! perturb the current u by a small, random amount

                DO i = 1, n
                  CALL RAND_random_real( data%seed, .FALSE., val )
                  data%U( i ) = data%U( i ) +                                  &
                    MAX( one, ABS(  data%U( i ) ) ) * val * root_eps
                END DO
                u_set = .TRUE.
                IF ( unit_m ) THEN
                  u_norm = TWO_NORM( data%U( : n ) )
                  data%U( : n ) = data%U( : n ) / u_norm
                ELSE
                  CALL mop_AX( one, M, data%U( : n ), zero, data%Y( : n ),     &
                               0, symmetric = .TRUE. )
                  u_norm = DOT_PRODUCT( data%U( : n ), data%Y( : n ) )
                  IF ( u_norm < zero ) THEN
                    IF ( printd ) WRITE( out, "( A, ' ||u||_M^2 =',            &
                   &  ES12.4, ' < 0' )" ) prefix, u_norm
                    GO TO 930
                  END IF
                  u_norm = SQRT( u_norm )
                END IF
              END IF

!  decide how may iterations of inverse iteration to perform

              IF ( lambda_u - lambda_s_l < data%control%start_invitmax_tol *   &
                MAX( ABS( lambda_l ), ABS( lambda_u ) ) ) THEN
                n_invit = data%control%inverse_itmax
              ELSE
                n_invit = 1
              END IF

!  now perform a few iterations of inverse iteration

              DO i = 1, n_invit
                CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
                IF ( .NOT. unit_m ) THEN
                  data%U( : n ) = data%Y( : n ) / u_norm
                  u_set = .TRUE.
                END IF
                IF ( constrained ) data%U( n + 1 : data%npm ) = zero
                CALL IR_solve( data%H_lambda, data%U( : data%npm ),            &
                       data%IR_data, data%SLS_data,                            &
                       data%control%IR_control, data%control%SLS_control,      &
                       inform%IR_inform, inform%SLS_inform )
                CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
                inform%time%solve = inform%time%solve + time_now - time_record
                inform%time%clock_solve =                                      &
                  inform%time%clock_factorize + clock_now - clock_record
                IF ( printt ) WRITE( out, 2050 ) prefix, clock_now -clock_record

                IF ( unit_m ) THEN
                  u_norm = TWO_NORM( data%U( : n ) )
                ELSE
                  CALL mop_AX( one, M, data%U( : n ), zero, data%Y( : n ),     &
                               0, symmetric = .TRUE. )
                  u_norm = DOT_PRODUCT( data%U( : n ), data%Y( : n ) )
                  IF ( u_norm < zero ) THEN
                    IF ( printd ) WRITE( out, "( A, ' ||u||_M^2 =',            &
                   &  ES12.4, ' < 0' )" ) prefix, u_norm
                    GO TO 930
                  END IF
                  u_norm = SQRT( u_norm )
                END IF
                data%U( : n ) = data%U( : n ) / u_norm
                u_set = .TRUE.
                lambda_s_l = MAX( lambda_s_l,  lambda - one / u_norm )
              END DO

!  compute the Rayleigh quotient

              data%Y( : n ) = zero
              DO l = 1, data%h_ne
                i = data%H_lambda%row( l ) ; j = data%H_lambda%col( l )
                val = data%H_lambda%val( l )
                IF ( i == j ) THEN
                  data%Y( i ) = data%Y( i ) + val * data%U( i )
                ELSE
                  data%Y( i ) = data%Y( i ) + val * data%U( j )
                  data%Y( j ) = data%Y( j ) + val * data%U( i )
                END IF
              END DO
              rayleigh = DOT_PRODUCT( data%U( : n ), data%Y( : n ) )
              IF ( printd ) WRITE( out, "( A, ' rayleigh ', ES23.15 )" )       &
                prefix, rayleigh

!  adjust lambda_l to account for the Rayleigh quotient

              lambda_s_l = MAX( lambda_s_l, - rayleigh )
              lambda_l = MAX( lambda_l, lambda_s_l )

!  compute the next lambda - bias this towards lambda_l unless the hard case
!  is suspected

              width = ABS( lambda_u - lambda_l )
              width_rel = width / MAX( ABS( lambda_l ), ABS( lambda_u ) )
              IF ( inform%hard_case ) THEN
!               lambda = MIN( lambda_l + theta_g * width,                      &
                lambda = MIN( lambda_l + theta_hard * width,                   &
                           MAX( lambda_l, lambda_s_l + theta_ii *              &
                             ( width_rel ** 1.333 ) ) )
!                            ( width_rel ** ( two * n_invit - gamma_eps ) ) ) )
              ELSE
                lambda = MIN( lambda_l + theta_g * width,                      &
                           MAX( lambda_l, lambda_s_l + theta_ii *              &
                             ( width_rel ** ( two * n_invit - gamma_eps ) ) ) )
              END IF

!  end of potential hard case. If no inverse iteration was applied,
!  use safeguarded bisection

            ELSE
              IF ( lambda_plus >= lambda_l ) THEN
                lambda = lambda_plus
              ELSE
                IF ( data%control%equality_problem ) THEN
                  lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
                ELSE
                  lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),           &
                                lambda_l + theta_eps * ( lambda_u - lambda_l ) )
                END IF
              END IF
            END IF
          END IF
        ELSE

!  ----------------------------
!  The current lambda lies in N
!  ----------------------------

          IF ( printt .OR. ( printi .AND. it == 1 ) ) WRITE( out, 2020 ) prefix
          region = 'N'
          IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )                &
            prefix, region, it, lambda_l, lambda, lambda_u
          try_zero = .FALSE.
          lambda_s_l = MAX( lambda_s_l, lambda )
          lambda_l = lambda_s_l

!  compute the next lambda - bias this towards lambda_l unless the hard case
!  is suspected

          width = ABS( lambda_u - lambda_l )
          IF ( inform%hard_case ) THEN
            in_n = in_n + 1
            IF ( MOD( in_n, 3 ) == 2 ) THEN
              lambda = lambda_l + theta_n * width
            ELSE
              lambda = lambda_l + theta_hard * width
            END IF
          ELSE
            IF ( data%control%equality_problem ) THEN
              lambda = lambda_l + theta_n * width
            ELSE
              in_n = in_n + 1
              IF ( constrained ) THEN
                IF ( MOD( in_n, 3 ) == 1 ) THEN
                  lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),           &
                                lambda_l + theta_n * width )
                ELSE IF ( MOD( in_n, 3 ) == 2 ) THEN
                  lambda = lambda_l + theta_hard * width
                ELSE
                  lambda = lambda_l + theta_n_small * width
                END IF
              ELSE
                IF ( MOD( in_n, 2 ) == 1 ) THEN
                  lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),           &
                                lambda_l + theta_n * width )
                ELSE
                  lambda = lambda_l + theta_n_small * width
                END IF
              END IF
            END IF
          END IF
        END IF

!  - - - - -
!  Hard case
!  - - - - -

        IF ( lambda_u - lambda_s_l < data%control%stop_hard *                  &
             MAX( one, ABS( lambda_l ), ABS( lambda_u ) ) ) THEN
          IF ( printi ) THEN
            IF ( printt ) WRITE( out, 2020 ) prefix
            WRITE( out, "( A, A2, I4, 3ES22.15 )" )                            &
              prefix, region, it + 1, lambda_l, lambda, lambda_u
            WRITE( out, "( A,                                                  &
           &    ' Hard-case stopping criteria satisfied.',                     &
           &    ' Interval width =', ES11.4 )" ) prefix, lambda_u - lambda_l
          END IF

!  compute the step alpha so that X + alpha U lies on the trust-region
!  boundary and gives the smaller value of q

          IF ( u_set ) THEN
            IF ( unit_m ) THEN
              utx = DOT_PRODUCT( data%U( : n ), X ) / radius
            ELSE
!             CALL mop_AX( one, M, data%U( : n ), zero, data%Y( : n ),         &
              CALL mop_AX( one, M, X, zero, data%Y( : n ),                     &
                           0, symmetric = .TRUE. )
              utx = DOT_PRODUCT( data%U( : n ), data%Y( : n ) ) / radius
            END IF
            distx = ( radius - inform%x_norm ) * ( ( radius + inform%x_norm )  &
                       / radius )
            alpha = sign( distx / ( abs( utx ) +                               &
                          sqrt( utx ** 2 + distx / radius ) ), utx )

!  record the optimal values

            X = X + alpha * data%U( : n )
          END IF
          inform%obj = f + half * ( DOT_PRODUCT( C, X ) - lambda * radius ** 2 )
          inform%x_norm = radius
          inform%status = GALAHAD_ok
          inform%hard_case = .TRUE.
          GO TO 900
        END IF

!  - - - - - - - - -
!  Almost Hard case
!  - - - - - - - - -

        IF ( lambda_u - lambda_l < data%control%stop_hard *                    &
             MAX( one, ABS( lambda_l ), ABS( lambda_u ) ) ) THEN
          IF ( printi ) THEN
            IF ( printt .AND. .NOT. phase_1) WRITE( out, 2020 ) prefix
            WRITE( out, "( A, A2, I4, 3ES22.15 )" )                            &
              prefix, region, it + 1, lambda_l, lambda, lambda_u
            WRITE( out, "( A,                                                  &
           &    ' Almost hard-case stopping criteria satisfied.',              &
           &    ' Interval width =', ES11.4 )" ) prefix, lambda_u - lambda_l
          END IF
          inform%status = GALAHAD_ok
          EXIT
        END IF

!  End of main iteration loop

      END DO

!  Record the optimal obective value

      inform%obj = f + half * ( DOT_PRODUCT( C, X ) - lambda * x_norm2( 0 ) )

!  ----
!  Exit
!  ----

 900  CONTINUE
      inform%multiplier = lambda
      inform%pole = lambda_s_l
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      RETURN

!  -------------
!  General error
!  -------------

  910 CONTINUE
      IF ( printi ) WRITE( out, "( A, '   **  Error return ', I0,              &
     & ' from TRS ' )" ) prefix, inform%status
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  ---------------------
!  Factorization failure
!  ---------------------

  920 CONTINUE
      IF ( printi ) WRITE( out, "( A, ' error return from ',                   &
     &   'SLS_factorize: status = ', I0 )" ) prefix, inform%SLS_inform%status
      inform%status = GALAHAD_error_factorization
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  -----------------
!  Indefinite M-norm
!  -----------------

  930 CONTINUE
      IF ( printi ) WRITE( out,                                                &
         "( A, ' The matrix M provided for TRS appears not to be strictly',    &
        &   A, ' diagonally dominant '  )" ) prefix, prefix
      inform%status = GALAHAD_error_preconditioner
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2000 FORMAT( A, ' time( SLS_analyse ) = ', F0.2 )
 2010 FORMAT( A, ' time( SLS_factorize ) = ', F0.2 )
 2020 FORMAT( A, '    it       lambda_l               lambda ',                &
                 '              lambda_u' )
 2030 FORMAT( A, '    it     ||x||-radius             lambda ',                &
                 '              d_lambda' )
 2050 FORMAT( A, ' time( SLS_solve ) = ', F0.2 )

!  End of subroutine TRS_solve_main

      END SUBROUTINE TRS_solve_main

!-*-*-*-*-  T R S _ S O L V E _ D I A G O N A L   S U B R O U T I N E  -*-*-*-

      SUBROUTINE TRS_solve_diagonal( n, radius, f, C, H, X, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the trust-region subproblem
!
!      minimize     1/2 <x, H x> + <c, x> + f
!      subject to    ||x||_2 <= radius  or ||x||_2 = radius
!
!  where H is diagonal, using a secular iteration
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   n - the number of unknowns
!
!   radius - the trust-region radius
!
!   f - the value of constant term for the quadratic function
!
!   C - a vector of values for the linear term c
!
!   H -  a vector of values for the diagonal matrix H
!
!   X - the required solution vector x
!
!   control - a structure containing control information. See TRS_control_type
!
!   inform - a structure containing information. See TRS_inform_type
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C, H
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( TRS_control_type ), INTENT( IN ) :: control
      TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, it, out, nroots, print_level
      INTEGER :: max_order, n_lambda, i_hard, n_sing
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record
      REAL ( KIND = wp ) :: lambda, lambda_l, lambda_u, delta_lambda
      REAL ( KIND = wp ) :: alpha, utx, distx, x_big
      REAL ( KIND = wp ) :: c_norm, c_norm_over_radius, v_norm2, w_norm2
      REAL ( KIND = wp ) :: beta, z_norm2, root1, root2, root3
      REAL ( KIND = wp ) :: lambda_min, lambda_max, lambda_plus, c2
      REAL ( KIND = wp ) :: a_0, a_1, a_2, a_3, a_max
      REAL ( KIND = wp ), DIMENSION( 3 ) :: lambda_new
      REAL ( KIND = wp ), DIMENSION( 0 : max_degree ) :: x_norm2, pi_beta
      LOGICAL :: printi, printt, printd, problem_file_exists
      CHARACTER ( LEN = 1 ) :: region

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  output problem data

      IF ( control%problem > 0 ) THEN
        INQUIRE( FILE = control%problem_file, EXIST = problem_file_exists )
        IF ( problem_file_exists ) THEN
          OPEN( control%problem, FILE = control%problem_file,                  &
                FORM = 'FORMATTED', STATUS = 'OLD' )
          REWIND control%problem
        ELSE
          OPEN( control%problem, FILE = control%problem_file,                  &
                FORM = 'FORMATTED', STATUS = 'NEW' )
        END IF
        WRITE( control%problem, * ) n, COUNT( C( : n ) /= zero ),              &
          COUNT( H( : n ) /= zero )
        WRITE( control%problem, * ) radius, f
        DO i = 1, n
          IF ( C( i ) /= zero ) WRITE( control%problem, * ) i, C( i )
        END DO
        DO i = 1, n
          IF ( H( i ) /= zero ) WRITE( control%problem, * ) i, i, H( i )
        END DO
        CLOSE( control%problem )
      END IF

!  set initial values

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

      X = zero ; inform%x_norm = zero ; inform%obj = f
      inform%hard_case = .FALSE.
      delta_lambda = zero

!  record desired output level

      out = control%out
      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1
      printd = out > 0 .AND. print_level > 2

!  compute the two-norm of c and the extreme eigenvalues of H

      c_norm = TWO_NORM( C )
      lambda_min = MINVAL( H( : n ) )
      lambda_max = MAXVAL( H( : n ) )

      IF ( printt ) WRITE( out, "( A, ' ||c|| = ', ES10.4, ', ||H|| = ',       &
     &                             ES10.4, ', lambda_min = ', ES11.4 )" )      &
          prefix, c_norm, MAXVAL( ABS( H( : n ) ) ), lambda_min

      region = 'L'
      IF ( printt )                                                            &
        WRITE( out, "( A, 4X, 28( '-' ), ' phase two ', 28( '-' ) )" ) prefix
      IF ( printi ) WRITE( out, 2030 ) prefix

!  check for the trivial case

      IF ( c_norm == zero .AND. lambda_min >= zero ) THEN
        IF (  control%equality_problem ) THEN
          DO i = 1, n
            IF ( H( i ) == lambda_min ) THEN
              i_hard = i
              EXIT
            END IF
          END DO
          X( i_hard ) = one / radius
          inform%x_norm = radius
          inform%obj = f + lambda_min * radius ** 2
          lambda = - lambda_min
        ELSE
          lambda = zero
        END IF
        IF ( printi ) THEN
          WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,             &
          0, ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
          WRITE( out, "( A,                                                    &
       &    ' Normal stopping criteria satisfied' )" ) prefix
        END IF
        inform%status = GALAHAD_ok
        GO TO 900
      END IF

!  construct values lambda_l and lambda_u for which lambda_l <= lambda_optimal
!   <= lambda_u, and ensure that all iterates satisfy lambda_l <= lambda
!   <= lambda_u

      c_norm_over_radius = c_norm / radius
      IF ( control%equality_problem ) THEN
        lambda_l = MAX( control%lower,  - lambda_min,                          &
                        c_norm_over_radius - lambda_max )
        lambda_u = MIN( control%upper,                                         &
                        c_norm_over_radius - lambda_min )
      ELSE
        lambda_l = MAX( control%lower,  zero, - lambda_min,                    &
                        c_norm_over_radius - lambda_max )
        lambda_u = MIN( control%upper,                                         &
                        MAX( zero, c_norm_over_radius - lambda_min ) )
      END IF
      lambda = lambda_l

!  check for the "hard case"

      IF ( lambda == - lambda_min ) THEN
        c2 = zero
        inform%hard_case = .TRUE.
        DO i = 1, n
          IF ( H( i ) == lambda_min ) THEN
            IF ( ABS( C( i ) ) > epsmch * c_norm ) THEN
              inform%hard_case = .FALSE.
              c2 = c2 + C( i ) ** 2
            ELSE
              i_hard = i
            END IF
          END IF
        END DO

!  the hard case may occur

        IF ( inform%hard_case ) THEN
          DO i = 1, n
            IF ( H( i ) /= lambda_min ) THEN
              X( i )  = - C( i ) / ( H( i ) + lambda )
            ELSE
              X( i ) = zero
            END IF
          END DO
          inform%x_norm = TWO_NORM( X )

!  the hard case does occur

          IF ( inform%x_norm <= radius ) THEN
            IF ( inform%x_norm < radius ) THEN

!  compute the step alpha so that X + alpha E_i_hard lies on the trust-region
!  boundary and gives the smaller value of q

              utx = X( i_hard ) / radius
              distx = ( radius - inform%x_norm ) *                             &
                ( ( radius + inform%x_norm ) / radius )
              alpha = sign( distx / ( abs( utx ) +                             &
                            sqrt( utx ** 2 + distx / radius ) ), utx )

!  record the optimal values

              X( i_hard ) = X( i_hard ) + alpha
            END IF
            inform%x_norm = TWO_NORM( X )
            inform%obj =                                                       &
                f + half * ( DOT_PRODUCT( C, X ) - lambda * radius ** 2 )
            IF ( printi ) THEN
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,         &
              0, ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
              WRITE( out, "( A,                                                &
           &    ' Hard-case stopping criteria satisfied' )" ) prefix
            END IF
            inform%status = GALAHAD_ok
            GO TO 900

!  the hard case didn't occur after all

          ELSE
            inform%hard_case = .FALSE.

!  compute the first derivative of ||x|(lambda)||^2 - radius^2

            w_norm2 = zero
            DO i = 1, n
              IF ( H( i ) /= lambda_min )                                      &
                w_norm2 = w_norm2 + C( i ) ** 2 / ( H( i ) + lambda ) ** 3
            END DO
            x_norm2( 1 ) = - two * w_norm2

!  compute the Newton correction

            lambda =                                                           &
              lambda - ( inform%x_norm ** 2 - radius ** 2 ) / x_norm2( 1 )
            lambda_l = MAX( lambda_l, lambda )
          END IF

!  there is a singularity at lambda. Compute the point for which the
!  sum of squares of the singular terms is equal to radius^2

        ELSE
          lambda = lambda + SQRT( c2 ) / radius
          lambda_l = MAX( lambda_l, lambda )
        END IF
      END IF

!  the iterates will all be in the L region. Prepare for the main loop

      it = 0
      max_order = MAX( 1, MIN( max_degree, control%taylor_max_degree ) )
      inform%len_history = 0

!  start the main loop

      DO
        it = it + 1

!  if H(lambda) is positive definite, solve  H(lambda) x = - c

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        n_sing = COUNT( H( : n ) + lambda <= zero )
        IF ( n_sing == 0 ) THEN
          DO i = 1, n
            X( i )  = - C( i ) / ( H( i ) + lambda )
          END DO
        ELSE
          x_big = radius / SQRT( REAL( n_sing, wp ) )
          DO i = 1, n
            IF ( H( i ) + lambda > zero ) THEN
              X( i )  = - C( i ) / ( H( i ) + lambda )
            ELSE
              X( i )  = SIGN( x_big, - C( i ) )
            END IF
          END DO
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_factorize + clock_now - clock_record

!  compute the two-norm of x

        inform%x_norm = TWO_NORM( X )
        x_norm2( 0 ) = inform%x_norm ** 2

!  if the Newton step lies within the trust region, exit

        IF ( lambda == zero .AND. inform%x_norm <= radius ) THEN
          inform%obj = f + half * DOT_PRODUCT( C, X )
          inform%status = GALAHAD_ok
          region = 'L'
          IF ( printi ) THEN
            WRITE( out, "( A, A2, I4, 2ES22.15 )" ) prefix,                    &
              region, it, inform%x_norm - radius, lambda
            WRITE( out, "( A, ' Interior stopping criteria satisfied')" ) prefix
          END IF
          GO TO 900
        END IF

!  the current estimate gives a good approximation to the required root

        IF ( ABS( inform%x_norm - radius ) <=                                  &
               MAX( control%stop_normal * radius,                              &
                    control%stop_absolute_normal ) ) THEN
          IF ( inform%x_norm > radius ) THEN
            lambda_l = MAX( lambda_l, lambda )
          ELSE
            region = 'G'
            lambda_u = MIN( lambda_u, lambda )
          END IF
          IF ( printt .AND. it > 1 ) WRITE( out, 2030 ) prefix
          IF ( printi ) THEN
            WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,           &
            it, ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
            WRITE( out, "( A,                                                  &
         &    ' Normal stopping criteria satisfied' )" ) prefix
          END IF
          inform%status = GALAHAD_ok
          EXIT
        END IF

        lambda_l = MAX( lambda_l, lambda )

!  debug printing

        IF ( printd ) THEN
          WRITE( out, "( A, 8X, 'lambda', 13X, 'x_norm', 15X, 'radius' )" )    &
            prefix
          WRITE( out, "( A, 3ES20.12 )") prefix, lambda, inform%x_norm, radius
        ELSE IF ( printi ) THEN
          IF ( printt .AND. it > 1 ) WRITE( out, 2030 ) prefix
          WRITE( out, "( A, A2, I4, 3ES22.15 )" ) prefix, region, it,          &
            ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
        END IF

!  record, for the future, values of lambda which give small ||x||

        IF ( inform%len_history < history_max ) THEN
          inform%len_history = inform%len_history + 1
          inform%history( inform%len_history )%lambda = lambda
          inform%history( inform%len_history )%x_norm = inform%x_norm
        END IF

!  a lambda in L has been found. It is now simply a matter of applying
!  a variety of Taylor-series-based methods starting from this lambda

!  precaution against rounding producing lambda outside L

        IF ( lambda > lambda_u ) THEN
          inform%status = GALAHAD_error_ill_conditioned
          EXIT
        END IF

!  compute first derivatives of x^T M x

!  form ||w||^2 = x^T H^-1(lambda) x

        w_norm2 = zero
        DO i = 1, n
          w_norm2 = w_norm2 + C( i ) ** 2 / ( H( i ) + lambda ) ** 3
        END DO

!  compute the first derivative of x_norm2 = x^T M x

        x_norm2( 1 ) = - two * w_norm2

!  compute pi_beta = ||x||^beta and its first derivative when beta = - 1

        beta = - one
        CALL TRS_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = - 1)

        delta_lambda = - ( pi_beta( 0 ) - ( radius ) ** beta ) / pi_beta( 1 )

        n_lambda = 1
        lambda_new( n_lambda ) = lambda + delta_lambda

        IF ( max_order >= 3 ) THEN

!  compute the second derivative of x^T x

          z_norm2 = zero
          DO i = 1, n
            z_norm2 = z_norm2 + C( i ) ** 2 / ( H( i ) + lambda ) ** 4
          END DO
          x_norm2( 2 ) = six * z_norm2

!  compute the third derivatives of x^T x

          v_norm2 = zero
          DO i = 1, n
            v_norm2 = v_norm2 + C( i ) ** 2 / ( H( i ) + lambda ) ** 5
          END DO
          x_norm2( 3 ) = - twentyfour * v_norm2

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

          beta = two
          CALL TRS_pi_derivs( max_order, beta, x_norm2( : max_order ),         &
                              pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

          a_0 = pi_beta( 0 ) - ( radius ) ** beta
          a_1 = pi_beta( 1 )
          a_2 = half * pi_beta( 2 )
          a_3 = sixth * pi_beta( 3 )
          a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
          IF ( a_max > zero ) THEN
            a_0 = a_0 / a_max ; a_1 = a_1 / a_max
            a_2 = a_2 / a_max ; a_3 = a_3 / a_max
          END IF
          CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,             &
                            root1, root2, root3, roots_debug )
          n_lambda = n_lambda + 1
          IF ( nroots == 3 ) THEN
            lambda_new( n_lambda ) = lambda + root3
          ELSE
            lambda_new( n_lambda ) = lambda + root1
          END IF

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.4

          beta = - point4
          CALL TRS_pi_derivs( max_order, beta, x_norm2( : max_order ),         &
                              pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

          a_0 = pi_beta( 0 ) - ( radius ) ** beta
          a_1 = pi_beta( 1 )
          a_2 = half * pi_beta( 2 )
          a_3 = sixth * pi_beta( 3 )
          a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
          IF ( a_max > zero ) THEN
            a_0 = a_0 / a_max ; a_1 = a_1 / a_max
            a_2 = a_2 / a_max ; a_3 = a_3 / a_max
          END IF
          CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,             &
                            root1, root2, root3, roots_debug )
          n_lambda = n_lambda + 1
          IF ( nroots == 3 ) THEN
            lambda_new( n_lambda ) = lambda + root3
          ELSE
            lambda_new( n_lambda ) = lambda + root1
          END IF
        END IF

!  record all of the estimates of the optimal lambda

        IF ( printd ) THEN
          WRITE( out, "( A, ' lambda_t (', I1, ')', 3ES20.13 )" )              &
            prefix, MAXLOC( lambda_new( : n_lambda ) ),                        &
            lambda_new( : MIN( 3, n_lambda ) )
          IF ( n_lambda > 3 ) WRITE( out, "( A, 13X, 3ES20.13 )" )             &
            prefix, lambda_new( 4 : MIN( 6, n_lambda ) )
        END IF

!  compute the best Taylor improvement

        lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
        delta_lambda = lambda_plus - lambda
        lambda = lambda_plus

!  improve the lower bound if possible

        lambda_l = MAX( lambda_l, lambda_plus )

!  check that the best Taylor improvement is significant

        IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
          IF ( printi ) WRITE( out, "( A, ' normal exit with no ',             &
         &                     'significant Taylor improvement' )" ) prefix
          inform%status = GALAHAD_ok
          EXIT
        END IF

!  End of main iteration loop

      END DO

!  Record the optimal obective value

      inform%obj = f + half * ( DOT_PRODUCT( C, X ) - lambda * x_norm2( 0 ) )

!  ----
!  Exit
!  ----

 900  CONTINUE
      inform%multiplier = lambda
      inform%pole = MAX( zero, - lambda_min )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2030 FORMAT( A, '    it     ||x||-radius             lambda ',                &
                 '              d_lambda' )

!  End of subroutine TRS_solve_diagonal

      END SUBROUTINE TRS_solve_diagonal

!-*-*-*-*-*-  T R S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE TRS_terminate( data, control, inform )

!  ...........................................
!  .                                         .
!  .  Deallocate arrays at end of TRS_solve  .
!  .                                         .
!  ...........................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine TRS_initialize
!   inform    see Subroutine TRS_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( TRS_data_type ), INTENT( INOUT ) :: data
      TYPE ( TRS_control_type ), INTENT( IN ) :: control
      TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'trs: M_diag'
      CALL SPACE_dealloc_array( data%M_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: M_offd'
      CALL SPACE_dealloc_array( data%M_offd,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: V'
      CALL SPACE_dealloc_array( data%V,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: H_dense%val'
      CALL SPACE_dealloc_array( data%H_dense%val,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: Q_dense'
      CALL SPACE_dealloc_array( data%Q_dense,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: X_dense'
      CALL SPACE_dealloc_array( data%X_dense,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: C_dense'
      CALL SPACE_dealloc_array( data%C_dense,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: work'
      CALL SPACE_dealloc_array( data%WORK,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: M_dense'
      CALL SPACE_dealloc_array( data%M_dense,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: A_dense%val'
      CALL SPACE_dealloc_array( data%A_dense%val,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'trs: data%SLS_data'
      END IF

!  Deallocate all arrays allocated within IR

      CALL IR_terminate( data%IR_data, control%IR_control, inform%IR_inform )
      IF ( inform%IR_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'trs: data%IR_data'
      END IF

      RETURN

!  End of subroutine TRS_terminate

      END SUBROUTINE TRS_terminate

!-*-*-*-  G A L A H A D -  T R S _ c o n t r a c t  S U B R O U T I N E -*-*-*-

     SUBROUTINE TRS_contract( n, radius, lambda, G, S, sigma_lower,            &
                              sigma_upper, sigma_target, gamma_lambda,         &
                              radius_reduce, have_solved_next, path,           &
                              data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Perform the TRACE CONTRACT calculation to determine a new radius
!  following an unsuccessful trust-region iteration (see Curtis et al, p9)

!  Requires: sigma_lower <= sigma_target <= sigma_upper

!  Input:
!   n - problem dimension
!   G - gradient
!   lambda - current TR multiplier
!   S - current TR minimizer
!   sigma_lower, sigma_upper, sigma_target - lower bound, upper bound and
!     target values for sigma
!   gamma_lambda - lambda increase factor
!   radius_reduce - trust-region radius reduction factor
!   data, control, inform - see TRS_initialize

!  Output
!   radius - next trust-region radius
!   lambda -  next TR multiplier (have_solved_next=.TRUE.) or an upper bound
!     on the multiplier (have_solved_next=.FALSE., input lambda is lower bound)
!   S - next TR minimizer (have_solved_next = .TRUE.) or approximation
!     (have_solved_next = .FALSE.)
!   have_solved_next - have we already solved the next TR problem?
!   path - which line of Curtis et al's is executed (27,29,37,39)
!   data, inform - see TRS_initialize

!  Parts extracted from GALAHAD module TRS

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     INTEGER, INTENT( OUT ) :: path
     REAL ( KIND = wp ), INTENT( IN ) :: sigma_lower, sigma_upper, sigma_target
     REAL ( KIND = wp ), INTENT( IN ) :: gamma_lambda, radius_reduce
     REAL ( KIND = wp ), INTENT( INOUT ) :: lambda
     REAL ( KIND = wp ), INTENT( OUT ) :: radius
     LOGICAL, INTENT( OUT ) :: have_solved_next
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: S
     TYPE ( TRS_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRS_control_type ), INTENT( IN ) :: control
     TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: it, n_lambda, max_order, out, print_level, nroots
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record
      REAL ( KIND = wp ) :: norm_s_input, norm_s, delta_lambda, oos, oos2
      REAL ( KIND = wp ) :: lambda_input, lambda_trial, lambda_plus
      REAL ( KIND = wp ) :: radius_trial, v_norm2, w_norm2, z_norm2, target
      REAL ( KIND = wp ) :: sigma, a_0, a_1, a_2, a_3, a_max, beta
      REAL ( KIND = wp ), DIMENSION( 3 ) :: roots
      REAL ( KIND = wp ), DIMENSION( 9 ) :: lambda_new
      REAL ( KIND = wp ), DIMENSION( 0 : max_degree ) :: x_norm2
      REAL ( KIND = wp ), DIMENSION( 0 : max_degree ) :: pi_beta, theta_beta
      LOGICAL :: printi, printt, printd, pos_def

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  record desired output level

      out = control%out
      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1
      printd = out > 0 .AND. print_level > 2

      IF ( sigma_lower > sigma_upper ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 910
      END IF
      sigma = MAX( sigma_lower, MIN( sigma_target, sigma_upper ) )
      have_solved_next = .FALSE.

!  reccord useful constants

      max_order = MAX( 1, MIN( max_degree, control%taylor_max_degree ) )
      oos = one / sigma ; oos2 = oos * oos

!  record the input lambda and the norm of s(lambda)

      lambda_input = lambda
      norm_s_input = TWO_NORM( S )

!  choose a larger lambda, lambda_trial

      IF ( lambda_input < sigma_lower * norm_s_input ) THEN
        lambda_trial = lambda_input + SQRT( sigma_lower * TWO_NORM( G ) )
      ELSE
        lambda_trial = lambda_input * gamma_lambda
      END IF

!  record the diagonal perturbation of H in H(lambda_trial)

      data%H_lambda%val( data%h_ne + 1 : data%m_end ) = lambda_trial

!  attempt an L B L^T factorization of H(lambda_trial)

      CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_factorize( data%H_lambda, data%SLS_data,                        &
                          data%control%SLS_control, inform%SLS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_record
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + clock_now - clock_record
      inform%factorizations = inform%factorizations + 1
      inform%max_entries_factors = MAX( inform%max_entries_factors,            &
                                        inform%SLS_inform%entries_in_factors )

!  test that the factorization succeeded

      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        IF ( data%control%SLS_control%pivot_control == 2 ) THEN
          pos_def = .TRUE.
        ELSE
          pos_def = inform%SLS_inform%negative_eigenvalues == 0
        END IF
      ELSE
        pos_def = .FALSE.
      END IF
      IF ( .NOT. pos_def ) GO TO 920

!  if H(lambda_trial) is positive definite, solve
!    H(lambda_trial) s(lambda_trial) = - g

      CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
      S = - G
      CALL IR_solve( data%H_lambda, S, data%IR_data, data%SLS_data,            &
                     data%control%IR_control,                                  &
                     data%control%SLS_control,                                 &
                     inform%IR_inform, inform%SLS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%solve = inform%time%solve + time_now - time_record
      inform%time%clock_solve =                                                &
        inform%time%clock_factorize + clock_now - clock_record
      IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the norm of the s(lambda_trial)

      radius_trial = TWO_NORM( S )

!  if lambda_input is large enough relative to the norm of s(lambda_input),
!  set the desired radius so that s(lambda_input) lies sufficiently
!  within a ball with this radius

      IF ( lambda_input >= sigma_lower * norm_s_input ) THEN
!       radius = MAX( radius_trial, radius_reduce * norm_s_input )
        IF ( radius_trial >= radius_reduce * norm_s_input ) THEN
          lambda = lambda_trial
          radius = radius_trial
          have_solved_next = .TRUE.
          path = 37
        ELSE
          lambda = lambda_trial
          radius = radius_reduce * norm_s_input
          have_solved_next = .FALSE.
          path = 39
        END IF

!  lambda_input is too small relative to the norm of s(lambda_input) ?

      ELSE

!  is lambda_trial small enough relative to the norm of s(lamda_trial) ?

        IF ( lambda_trial <= sigma_upper * radius_trial ) THEN
          lambda = lambda_trial
          radius = radius_trial
          have_solved_next = .TRUE.
          path = 27

!  lambda_trial is too large. Compute a lambda in (lambda_input, lambda_trial)
!  for which sigma_lower <= lambda / ||s(lambda) <= sigma_upper

        ELSE

!  since Newton-like methods converge from the left, aim for the root with
!  sigma = sigma_target starting from lambda_input, stopping as soon as
!    ||s(lambda)|| <= lambda / sigma_lower

!  start the main loop

          it = 0
          lambda = MAX( lambda_input, teneps )
          DO
            it = it + 1

!  add lambda * M to H to form H(lambda)

            data%H_lambda%val( data%h_ne + 1 :  data%m_end ) = lambda

!  attempt an L B L^T factorization of H(lambda)

            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SLS_factorize( data%H_lambda, data%SLS_data,                  &
                                data%control%SLS_control, inform%SLS_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%factorize =                                            &
              inform%time%factorize + time_now - time_record
            inform%time%clock_factorize =                                      &
              inform%time%clock_factorize + clock_now - clock_record
            IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
            inform%factorizations = inform%factorizations + 1
            inform%max_entries_factors = MAX( inform%max_entries_factors,      &
                                       inform%SLS_inform%entries_in_factors )

!  test that the factorization succeeded

            IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
              IF ( data%control%SLS_control%pivot_control == 2 ) THEN
                pos_def = .TRUE.
              ELSE
                pos_def = inform%SLS_inform%negative_eigenvalues == 0
              END IF
            ELSE
              pos_def = .FALSE.
            END IF
            IF ( .NOT. pos_def ) GO TO 920

!  if H(lambda) is positive definite, solve  H(lambda) s(lambda) = - g

            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            S = - G
            CALL IR_solve( data%H_lambda, S, data%IR_data, data%SLS_data,      &
                           data%control%IR_control,                            &
                           data%control%SLS_control,                           &
                           inform%IR_inform, inform%SLS_inform )

            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%solve = inform%time%solve + time_now - time_record
            inform%time%clock_solve =                                          &
              inform%time%clock_factorize + clock_now - clock_record
            IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the norm of s(lambda)

            norm_s = TWO_NORM( S )
            x_norm2( 0 ) = norm_s ** 2

!  stop - the current estimate gives a good approximation to the required root

            IF ( norm_s <= lambda / sigma_lower ) THEN
              IF ( printt .OR. ( printi .AND. it == 1 ) )  &
                WRITE( out, 2030 ) prefix
              IF ( printi ) THEN
                WRITE( out, "( A, A2, I4, 3ES22.15 )" ) prefix, 'L', it,       &
                  ABS( norm_s - target ), lambda, ABS( delta_lambda )
                WRITE( out, "( A,                                              &
            &    ' Normal stopping criteria satisfied' )" ) prefix
              END IF
              inform%status = GALAHAD_ok
              EXIT
            END IF

!  compute the target value lambda / sigma_target

            target = lambda / sigma

!  check to see if the factorization limit has been exceeded

            IF ( inform%factorizations > data%control%max_factorizations ) THEN
              inform%multiplier = lambda
              inform%status = GALAHAD_error_max_iterations ; GO TO 910
            END IF

            IF ( printt .OR. ( printi .AND. it == 1 ) ) WRITE( out, 2030 )     &
              prefix

!  compute first derivatives of x^T x

            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = s

            IF ( data%accurate ) THEN
              data%Y( : n ) = S
              data%Z( : n ) = data%Y( : n )
              CALL IR_solve( data%H_lambda, data%Z( : data%npm ),              &
                             data%IR_data, data%SLS_data,                      &
                             data%control%IR_control,                          &
                             data%control%SLS_control,                         &
                             inform%IR_inform, inform%SLS_inform )

!  find y so that L y = s

            ELSE
              data%Y( : n ) = S
              CALL SLS_part_solve( 'L', data%Y( : n ), data%SLS_data,          &
                                 data%control%SLS_control, inform%SLS_inform )

!  find z so that D L z = D y = s

              data%Z( : n ) = data%Y( : n )
              CALL SLS_part_solve( 'D', data%Z( : n ), data%SLS_data,          &
                                 data%control%SLS_control, inform%SLS_inform )
            END IF
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%solve = inform%time%solve + time_now - time_record
            inform%time%clock_solve =                                          &
              inform%time%clock_factorize + clock_now - clock_record
            IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  form ||w||^2 = y^T z = x^T L^-T D^-1 L^-1 x = x^T H^-1(lambda) x

            w_norm2 = DOT_PRODUCT( data%Y( : n ), data%Z( : n ) )

!  compute the first derivative of x_norm2 = x^T x

            x_norm2( 1 ) = - two * w_norm2

!  count the number of corrections computed

            n_lambda = 0

!  compute Taylor approximants of degree one

!  compute pi_beta = ||x||^beta and its first derivative when beta = 2

            beta = two
            CALL TRS_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = 2)

            a_0 = pi_beta( 0 ) - target ** 2
            a_1 = pi_beta( 1 ) - two * lambda * oos2
            a_2 = - oos2
            a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
            IF ( a_max > zero ) THEN
              a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
            END IF
            CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,            &
                                  roots( 1 ), roots( 2 ), roots_debug )
            lambda_plus = lambda + roots( 2 )
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and its first derivative when beta = 1

            beta = one
            CALL TRS_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = 1)

            delta_lambda = - ( pi_beta( 0 ) - target ) / ( pi_beta( 1 ) - oos )
            lambda_plus = lambda + delta_lambda
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and its first derivative when beta = - 1

            beta = - one
            CALL TRS_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = -1)

            a_0 = pi_beta( 0 ) * lambda - sigma
            a_1 = pi_beta( 0 ) + lambda * pi_beta( 1 )
            a_2 = pi_beta( 1 )
            a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
            IF ( a_max > zero ) THEN
              a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
            END IF
            CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,            &
                                  roots( 1 ), roots( 2 ), roots_debug )
            lambda_plus = lambda + roots( 2 )
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!           WRITE( out, "( ' lambda, delta ', 2ES22.14)") lambda, delta_lambda

            IF ( max_order >= 3 ) THEN
              IF ( .NOT. data%accurate ) THEN
                CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
                CALL SLS_part_solve( 'U', data%Z( : n ), data%SLS_data,        &
                                     data%control%SLS_control,                 &
                                     inform%SLS_inform )
                CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
                inform%time%solve = inform%time%solve + time_now - time_record
                inform%time%clock_solve =                                      &
                  inform%time%clock_factorize + clock_now - clock_record
                IF ( printt )                                                  &
                  WRITE( out, 2050 ) prefix, clock_now - clock_record
              END IF

!  form z^T z

              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Z( : n ) )

!  compute the second derivative of x_norm2 = x^T x

              x_norm2( 2 ) = six * z_norm2

              IF ( max_order >= 3 ) THEN
                CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = x

                IF ( data%accurate ) THEN
                  data%Y( : n ) = data%Z( : n )
                  CALL IR_solve( data%H_lambda, data%Z( : data%npm ),          &
                                 data%IR_data, data%SLS_data,                  &
                                 data%control%IR_control,                      &
                                 data%control%SLS_control,                     &
                                 inform%IR_inform, inform%SLS_inform )

!  find z so that L z = x'

                ELSE
                  CALL SLS_part_solve( 'L', data%Z( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )

!  find y so that D L y = D z = x'

                  data%Y( : n ) = data%Z( : n )
                  CALL SLS_part_solve( 'D', data%Y( : n ), data%SLS_data,      &
                                       data%control%SLS_control,               &
                                       inform%SLS_inform )
                END IF
                CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
                inform%time%solve =                                            &
                  inform%time%solve + time_now - time_record
                inform%time%clock_solve =                                      &
                  inform%time%clock_factorize + clock_now - clock_record
                IF ( printt )                                                  &
                  WRITE( out, 2050 ) prefix, clock_now - clock_record

!  form ||v||^2 = z^T y = x'^T L^-T D^-1 L^-1 x' = x'^T H^-1(lambda) x'

                v_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )

!  compute the third derivatives of x_norm2 = x^T x

                x_norm2( 3 ) = - twentyfour * v_norm2
              END IF
            END IF

!  compute pi_beta = ||x||^beta and its derivatives for various beta
!  and the resulting Taylor series approximants

!  for Taylor approximants of degree larger than one

            IF ( max_order >= 3 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

              beta = two
              CALL TRS_pi_derivs( 3, beta, x_norm2( : 3 ), pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

              a_0 = pi_beta( 0 ) - target ** 2
              a_1 = pi_beta( 1 ) - two * lambda * oos2
              a_2 = half * pi_beta( 2 ) - oos2
              a_3 = sixth * pi_beta( 3 )
              a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
              IF ( a_max > zero ) THEN
                a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                a_2 = a_2 / a_max ; a_3 = a_3 / a_max
              END IF
              CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,         &
                                roots( 1 ), roots( 2 ), roots( 3 ),            &
                                roots_debug )
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda + roots( 1 )

!  compute pi_beta = ||x||^beta and its derivatives when beta = 1

              beta = one
              CALL TRS_pi_derivs( 3, beta, x_norm2( : 3 ), pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 1)

              a_0 = pi_beta( 0 ) - target
              a_1 = pi_beta( 1 ) - oos
              a_2 = half * pi_beta( 2 )
              a_3 = sixth * pi_beta( 3 )
              a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
              IF ( a_max > zero ) THEN
                a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                a_2 = a_2 / a_max ; a_3 = a_3 / a_max
              END IF
              CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,         &
                                roots( 1 ), roots( 2 ), roots( 3 ),            &
                                roots_debug )
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda + roots( 1 )

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^beta and
!  their derivatives when beta = - 0.4

              beta = - point4
              CALL TRS_pi_derivs( 3, beta, x_norm2( : 3 ), pi_beta( : 3 ) )
              CALL TRS_theta_derivs( 3, beta, lambda, sigma,                   &
                                     theta_beta( : 3 )  )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

              a_0 = pi_beta( 0 ) - theta_beta( 0 )
              a_1 = pi_beta( 1 ) - theta_beta( 1 )
              a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
              a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3 ) )
              a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
              IF ( a_max > zero ) THEN
                a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                a_2 = a_2 / a_max ; a_3 = a_3 / a_max
              END IF
              CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,         &
                                roots( 1 ), roots( 2 ), roots( 3 ),            &
                                roots_debug )
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda + roots( 1 )
            END IF

!  record all of the estimates of the optimal lambda

            IF ( printd ) THEN
              WRITE( out, "( A, ' lambda_t (', I1, ')', 3ES20.12 )" )          &
                prefix, MAXLOC( lambda_new( : n_lambda ) ),                    &
                lambda_new( : MIN( 3, n_lambda ) )
              IF ( n_lambda > 3 ) WRITE( out, "( A, 13X, 3ES20.12 )" )         &
                prefix, lambda_new( 4 : MIN( 6, n_lambda ) )
            END IF

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
            delta_lambda = lambda_plus - lambda
            lambda = lambda_plus

!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) <                                         &
                 epsmch * MAX( one, ABS( lambda ) ) ) THEN
              IF ( printi ) WRITE( out, "( A, ' normal exit with no ',         &
             &                   'significant Taylor improvement' )" ) prefix
              EXIT
            END IF

!  end of main iteration loop

          END DO

!  compute the radius as the norm of the solution s(lambda)

          radius = norm_s
          have_solved_next = .TRUE.
          path = 29
        END IF
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  -------------
!  General error
!  -------------

  910 CONTINUE
      IF ( printi ) WRITE( out, "( A, '   **  Error return ', I0,              &
     & ' from TRS_contract' )" ) prefix, inform%status
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  ---------------------
!  Factorization failure
!  ---------------------

  920 CONTINUE
      IF ( printi ) WRITE( out, "( A, ' error return from ',                   &
     &   'SLS_factorize: status = ', I0 )" ) prefix, inform%SLS_inform%status
      inform%status = GALAHAD_error_factorization
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2010 FORMAT( A, ' time( SLS_factorize ) = ', F0.2 )
 2030 FORMAT( A, '    it    ||x||-target              lambda ',                &
                 '              d_lambda' )
 2050 FORMAT( A, ' time( SLS_solve ) = ', F0.2 )

!  End of subroutine TRS_contract

     END SUBROUTINE TRS_contract

!-*-*-*-*-*-*-  T R S _ P I _ D E R I V S   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE TRS_pi_derivs( max_order, beta, x_norm2, pi_beta )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute pi_beta = ||x||^beta and its derivatives
!
!  Arguments:
!  =========
!
!  Input -
!   max_order - maximum order of derivative
!   beta - power
!   x_norm2 - (0) value of ||x||^2,
!             (i) ith derivative of ||x||^2, i = 1, max_order
!  Output -
!   pi_beta - (0) value of ||x||^beta,
!             (i) ith derivative of ||x||^beta, i = 1, max_order
!
!  Extracted wholesale from module GALAHAD_RQS
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: max_order
      REAL ( KIND = wp ), INTENT( IN ) :: beta, x_norm2( 0 : max_order )
      REAL ( KIND = wp ), INTENT( OUT ) :: pi_beta( 0 : max_order )

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      REAL ( KIND = wp ) :: hbeta

      hbeta = half * beta
      pi_beta( 0 ) = x_norm2( 0 ) ** hbeta
      IF ( hbeta == one ) THEN
        pi_beta( 1 ) = x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = x_norm2( 2 )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = x_norm2( 3 )
      ELSE IF ( hbeta == two ) THEN
        pi_beta( 1 ) = two * x_norm2( 0 ) * x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = two * ( x_norm2( 1 ) ** 2 + x_norm2( 0 ) * x_norm2( 2 ) )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = two *                                                   &
          ( x_norm2( 0 ) * x_norm2( 3 ) + three * x_norm2( 1 ) * x_norm2( 2 ) )
      ELSE
        pi_beta( 1 )                                                           &
          = hbeta * ( x_norm2( 0 ) ** ( hbeta - one ) ) * x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - two ) ) *           &
          ( ( hbeta - one ) * x_norm2( 1 ) ** 2 + x_norm2( 0 ) * x_norm2( 2 ) )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - three ) ) *         &
          ( x_norm2( 3 ) * x_norm2( 0 ) ** 2 + ( hbeta - one ) *               &
            ( three * x_norm2( 0 ) * x_norm2( 1 ) * x_norm2( 2 ) +             &
              ( hbeta - two ) * x_norm2( 1 ) ** 3 ) )
      END IF
      RETURN

!  End of subroutine TRS_pi_derivs

      END SUBROUTINE TRS_pi_derivs

!-*-*-*-*-*  T R S _ T H E T A _ D E R I V S   S U B R O U T I N E   *-*-*-*-*-

      SUBROUTINE TRS_theta_derivs( max_order, beta, lambda, sigma,            &
                                     theta_beta )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute theta_beta = (lambda/sigma)^beta and its derivatives
!
!  Arguments:
!  =========
!
!  Input -
!   max_order - maximum order of derivative
!   beta - power
!   lambda, sigma - lambda and sigma
!  Output -
!   theta_beta - (0) value of (lambda/sigma)^beta,
!             (i) ith derivative of (lambda/sigma)^beta, i = 1, max_order
!
!  Extracted wholesale from module GALAHAD_RQS
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: max_order
      REAL ( KIND = wp ), INTENT( IN ) :: beta, lambda, sigma
      REAL ( KIND = wp ), INTENT( OUT ) :: theta_beta( 0 : max_order )

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      REAL ( KIND = wp ) :: los, oos

      los = lambda / sigma
      oos = one / sigma

!if ( los <= zero ) write( 6,*) ' l, s ',  lambda, sigma, beta
      theta_beta( 0 ) = los ** beta
      IF ( beta == one ) THEN
        theta_beta( 1 ) = oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = zero
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = zero
      ELSE IF ( beta == two ) THEN
        theta_beta( 1 ) = two * los * oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = oos ** 2
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = zero
      ELSE
        theta_beta( 1 ) = beta * ( los ** ( beta - one ) ) * oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = beta * ( los ** ( beta - two ) ) *                   &
                          ( beta - one ) * oos ** 2
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = beta * ( los ** ( beta - three ) ) *                 &
                          ( beta - one ) * ( beta - two ) * oos ** 3
      END IF
      RETURN

!  End of subroutine TRS_theta_derivs

      END SUBROUTINE TRS_theta_derivs

!-  G A L A H A D - T R S _ o r t h o g o n a l _ s o l v e  S U B R O U T I N E

      SUBROUTINE TRS_orthogonal_solve( n, C, H, X, k_max,                      &
                                       data, control, inform )

! ------------------------------------------------------------------------

!  Aim: find, successively

!  x_k+1 = arg min q(x) = 1/2 <x, H x> + <c, x> + f : X_k^T x = 0
!  where X_k = (x_1, .., x_k)

!  Explanation:

!  => (   H    X_k ) ( x_k+1 ) = - ( c )
!     ( X^T_k   0  ) (  y_k  )     ( 0 )

!   x_1   = - H^{-1} c &
!   x_k+1 = - H^{-1} c - H^{-1} X_k y_k
!         = x_1 - Z_k  y_k

!  where Z_k = (z_1, ..., z_k) = H^{-1} X_k
!            = ( H^{-1} x_1, ...,  H^{-1} x_k )

!  Also X^T_k x_k+1 = 0 =>

!   X^T_k Z_k y_k = X^T_k x_1 = ||x_1||^2 e_1

!  Let H = L_H L^T_H and W_k = L_H^{-1} X_k so that w_k = W_k e_k =>

!   X^T_k Z_k = X^T_k L^-T L_H^{-1} X_k = W^T_k W_k = L_k L^T_k

!  with L_k lower triangular with last row (l_k-1^T,lam_k)

!  => W^T_k W_k = ( W^T_k-1 W_k-1   W^T_k-1 w_k ) =
!                 (  w^T_k W_k-1     w^T_k w_k  )
!     L_k L^T_k = ( L_k-1 L^T_k-1         L_k-1 l_k-1        )
!                 ( l^T_k-1 L^T_k-1  l^T_k-1 l_k-1 + lam_k^2 )

!  i.e., L_k-1 l_k-1 = v_k-1 and lam_k^2 = nu_k - l^T_k-1 l_k-1
!  where v_k-1 = W^T_k-1 w_k and nu_k = w^T_k w_k

!  v_k-1  = W^T_k-1 w_k =  X^T_k-1 L_H^{-T} L_H^{-1} x_k = X^T_k-1 H^{-1} x_k
!       = Z^T_k-1 x_k and
!  nu_k = w^T_k w_k
!       = z^T_k x_k

!  Given L_k, find y_k via  L_k L^T_k y_k = ||x_1||^2 e_1, i.e.
!  L_k u_k = ||x_1||^2 e_1 and L^T_k y_k = u_k =>

!   u^T_k = (u^T_k-1,mu_k), where mu_k = - u_k-1^T l_k-1 / lam_k

! Thus .... to summarize

! ALGORITHM ORTHOGONAL_SOLVE

! -1. Set Z_0 = 0 and u_0 = 0

 ! 0. Factorize H and compute x_1 = - H^-1 c

! for k = 1, ..., k_max - 1

! 1. Set z_k = H^{-1} x_k and Z_k = ( Z_k-1 : z_k )

! 2. Compute v_k-1 = Z^T_k-1 x_k and nu_k = z^T_k x_k

! 3. Find new row of (l^T_k-1,lam_k) of L_k via

!     L_k-1 l_k-1 = v_k-1 and lam_k^2 = nu_k - l^T_k-1 l_k-1

! 4. If k = 1, set mu_1 = x^T_1 x_1 / lam_1
!    else mu_k = - u_k-1^T l_k-1 / lam_k

! 5. Find  L^T_k y_k = u_k, where u^T_k = (`u^T_k-1,mu_k)

! 6. Set x_k+1 = x_1 - Z_k y_k

! end for

! total effort: k_max solves with H, k_max * ( k_max + 1 ) inner products
! of order n to form Z^T_k x_k and Z_k y_k + lesser terms

! ------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, k_max
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = wp ), INTENT( OUT), DIMENSION( n, k_max ) :: X

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, k
     REAL ( KIND = wp ), DIMENSION( k_max ) :: U, Y
     REAL ( KIND = wp ), DIMENSION( k_max, k_max ) :: LT
     REAL ( KIND = wp ), DIMENSION( n, k_max ) :: Z
     TYPE ( TRS_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRS_control_type ), INTENT( IN ) :: control
     TYPE ( TRS_inform_type ), INTENT( INOUT ) :: inform

!  factorize H

     CALL SLS_analyse( data%H_lambda, data%SLS_data,                           &
                       data%control%SLS_control, inform%SLS_inform )
     CALL SLS_factorize( data%H_lambda, data%SLS_data,                         &
                         data%control%SLS_control, inform%SLS_inform )

!    U = 0.0_wp
!    Y = 0.0_wp
!    LT = 0.0_wp
!    Z = 0.0_wp

!  compute x_1 = - H^-1 c

     X( : , 1 ) = - C
     CALL SLS_solve( data%H_lambda, X( : , 1 ), data%SLS_data,                 &
                     data%control%SLS_control, inform%SLS_inform )

!  main loop

!write(6,*) ' x ', 1, X( : , 1 )
     DO k = 1, k_max - 1

!  set z_k = H^{-1} x_k and Z_k = ( Z_k-1 : z_k )

       Z( : , k ) = X( : , k )
       CALL SLS_solve( data%H_lambda, Z( : , k ), data%SLS_data,               &
                       data%control%SLS_control, inform%SLS_inform )

!write(6,*) ' z ', k, Z( : , k )

!  store v_k-1 = Z^T_k-1 x_k and nu_k = z^T_k x_k in the kth column of LT

       DO i = 1, k
         LT( i , k ) = DOT_PRODUCT( Z( : , i ), X( : , k ) )
       END DO

!  find the new row (l^T_k-1,lam_k) of L_k (i.e, k-th column of LT) via
!    L_k-1 l_k-1 = v_k-1 ...

!write(6,*) ' k ', k
       IF ( k > 1 ) THEN
         LT( 1, k ) = LT( 1, k ) / LT( 1, 1 )
         DO i = 2, k - 1
           LT( i, k ) = ( LT( i, k )                                           &
             - DOT_PRODUCT( LT( : i - 1, i ), LT( : i - 1, k ) ) ) / LT( i, i )
         END DO

!  ... and lam_k^2 = nu_k - l^T_k-1 l_k-1

         IF ( LT( k, k ) < DOT_PRODUCT( LT( : k - 1, k ), LT( : k - 1, k ) ) ) &
           write(6,*) ' error ', LT( k, k ),                                   &
             DOT_PRODUCT( LT( : k - 1, k ), LT( : k - 1, k ) )
         LT( k, k ) =                                                          &
           SQRT( LT( k, k ) - DOT_PRODUCT( LT( : k - 1, k ), LT( : k - 1, k ) ))
       ELSE
         LT( 1, 1 ) = SQRT( LT( 1, 1 ) )
       END IF

!  if k = 1, set mu_1 = x^T_1 x_1 / lam_1

       IF ( k == 1 ) THEN
         U( 1 ) = DOT_PRODUCT(  X( : , 1 ),  X( : , 1 ) ) / LT( 1, 1 )

!  otherwise set mu_k = - u_k-1^T l_k-1 / lam_k

       ELSE
         U( k )                                                                &
           =  - DOT_PRODUCT(  U( : k - 1 ),  LT( : k - 1 , k ) ) / LT( k, k )
       END IF

!  find  L^T_k y_k = u_k, where u^T_k = (u^T_k-1,mu_k)

       Y( : k ) = U( : k )
       DO i = k, 1, - 1
         Y( i ) = Y( i ) / LT( i, i )
         IF ( i > 1 ) &
         Y( : i - 1 ) = Y( : i - 1 ) - Y( i ) * LT( : i - 1, i )
       END DO

!  set x_k+1 = x_1 - Z_k y_k

       X( : , k + 1 ) = X( : , 1 )
       DO i = 1, k
         X( : , k + 1 ) = X( : , k + 1 ) - Z( : , i ) * Y( i )
       END DO

!write(6,*) ' x ', k+1, X( : , k+1 )

!write(6,*) ' u ', u( : k )
!DO i = 1, k
!  write(6,*) 'LT col ', i, LT( : i, i )
!END DO
!DO i = 1, k
!  write(6,*) 'Z col ', i, Z( 1:2, i )
!END DO

     END DO

     RETURN

!  end of subroutine TRS_orthogonal_solve

     END SUBROUTINE TRS_orthogonal_solve

!-*-*-*-*-*-  End of G A L A H A D _ T R S  double  M O D U L E  *-*-*-*-*-*-

   END MODULE GALAHAD_TRS_double
