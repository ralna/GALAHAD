! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 13:20 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P A  M O D U L E  -*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. May 21st 1999
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html
!

   MODULE GALAHAD_QPA_double

!     -----------------------------------------------
!     |                                             |
!     | Solve the l_1 quadratic program             |
!     |                                             |
!     |    minimize   1/2 x(T) H x + g(T) x + f     |
!     |                + rho_g min( A x - c_l , 0 ) |
!     |                + rho_g max( A x - c_u , 0 ) |
!     |                + rho_b min(  x - x_l , 0 )  |
!     |                + rho_b max(  x - x_u , 0 )  |
!     |                                             |
!     | using an active-set method based on         |
!     | projected conjugate-gradients               |
!     |                                             |
!     -----------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_NORMS_double
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_RAND_double
      USE GALAHAD_ROOTS_double, ONLY : ROOTS_quadratic
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_are,             &
                                       STRING_exponent, STRING_real_7
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
         SORT_heapsort_smallest, SORT_inplace_permute, SORT_inverse_permute
      USE GALAHAD_SLS_double
      USE GALAHAD_SCU_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, QPA_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, QPA_data_type => QPD_data_type,                  &
                              QPA_HX => QPD_HX

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPA_initialize, QPA_read_specfile, QPA_solve, QPA_terminate,   &
                QPA_solve_qp, QPA_solve_main, QPA_remove_dependent,            &
                QPA_factorize_reference, QPA_new_reference_set, QPA_HX,        &
                QPA_ir, QPA_pcg, QPA_add_constraint, QPA_delete_constraint,    &
                QPA_data_type, QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: hzero = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: k_diag = one
      REAL ( KIND = wp ), PARAMETER :: curv_min = epsmch
      LOGICAL, PARAMETER :: exact = .TRUE.
      LOGICAL :: roots_debug = .FALSE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: QPA_partition_type
        INTEGER :: n_free = - 1
        INTEGER :: n_fixed = - 1
        INTEGER :: c_ref = - 1
        INTEGER :: m_ref = - 1
        INTEGER :: k_ref = - 1
        INTEGER :: n_ref = - 1
        INTEGER :: k_free_od = - 1
        INTEGER :: k_free_d = - 1
        INTEGER :: k_free_p = - 1
        INTEGER :: k_fixed_od = - 1
        INTEGER :: k_fixed_d = - 1
      END TYPE

      TYPE, PUBLIC :: QPA_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   any printing will start on this iteration

        INTEGER :: start_print = - 1

!   any printing will stop on this iteration

        INTEGER :: stop_print = - 1

!   at most maxit inner iterations are allowed

        INTEGER :: maxit = 1000

!   the factorization to be used. Possible values are

!      0  automatic
!      1  Schur-complement factorization
!      2  augmented-system factorization

        INTEGER :: factor = 0

!   the maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization

        INTEGER :: max_col = 35

!   the maximum permitted size of the Schur complement before a
!    refactorization is performed

        INTEGER :: max_sc = 75

!   an initial guess as to the integer workspace required by SLS      (OBSOLETE)

        INTEGER :: indmin = 1000

!   an initial guess as to the real workspace required by SLS         (OBSOLETE)

        INTEGER :: valmin = 1000

!   the maximum number of iterative refinements allowed               (OBSOLETE)

        INTEGER :: itref_max = 1

!   the infeasibility will be checked for improvement every
!    infeas_check_interval iterations (see infeas_g_improved_by_factor
!    and infeas_b_improved_by_factor below)

        INTEGER :: infeas_check_interval = 100

!   the maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1
!
        INTEGER :: cg_maxit = - 1

!   the preconditioner to be used for the CG is defined by precon.
!    Possible values are

!      0  automatic
!      1  no preconditioner, i.e, the identity within full factorization
!      2  full factorization
!      3  band within full factorization
!      4  diagonal using the barrier terms within full factorization

        INTEGER :: precon = 0

!   the semi-bandwidth of a band preconditioner, if appropriate

        INTEGER :: nsemib = 5

!   if the ratio of the number of nonzeros in the factors of the reference
!    matrix to the number of nonzeros in the matrix itself exceeds
!    full_max_fill, and the preconditioner is being selected automatically
!    (precon = 0), a banded approximation will be used instead

        INTEGER :: full_max_fill = 10

!   the constraint deletion strategy to be used. Possible values are:
!
!      0  most violated of all
!      1  LIFO (last in, first out)
!      k  LIFO(k) most violated of the last k in LIFO

        INTEGER :: deletion_strategy = 0

!   indicate whether and how much of the input problem should be restored
!    on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

!   the frequency at which residuals will be monitored

        INTEGER :: monitor_residuals = 1
!
!   indicates whether a cold or warm start should be made. Possible values are
!
!     0 warm start - the values set in C_stat and B_stat indicate which
!       constraints will be included in the initial working set.
!     1 cold start from the value set in X; constraints active
!       at X will determine the initial working set.
!     2 cold start with no active constraints
!     3 cold start with only equality constraints active
!     4 cold start with as many active constraints as possible

        INTEGER :: cold_start = 3

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   any constraint violated by less than feas_tol will be considered to be
!    satisfied

        REAL ( KIND = wp ) :: feas_tol = epsmch

!   if the objective function value is smaller than obj_unbounded, it will be
!    flagged as unbounded from below.

        REAL ( KIND = wp ) :: obj_unbounded = one / epsmch

!   if the problem is currently infeasible and solve_qp (see below) is .TRUE.,
!    the current penalty parameter for the general constraints will be
!    increased by increase_rho_g_factor when needed

        REAL ( KIND = wp ) :: increase_rho_g_factor = two

!   if the infeasibility of the general constraints has not dropped by a factor
!     of infeas_g_improved_by_factor over the previous infeas_check_interval
!     iterations, the current corresponding penalty parameter will be increased

        REAL ( KIND = wp ) :: infeas_g_improved_by_factor = 0.75_wp

!   if the problem is currently infeasible and solve_qp or solve_within_bounds
!     (see below) is .TRUE., the current penalty parameter for the simple bound
!     constraints will be increased by increase_rho_b_factor when needed

        REAL ( KIND = wp ) :: increase_rho_b_factor = two

!   if the infeasibility of the simple bounds has not dropped by a factor of
!    infeas_b_improved_by_factor over the previous infeas_check_interval
!    iterations, the current corresponding penalty parameter will be increased
!
        REAL ( KIND = wp ) :: infeas_b_improved_by_factor = 0.75_wp

!   the threshold pivot used by the matrix factorization.
!    See the documentation for SLS for details                        (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol = epsmch

!   the threshold pivot used by the matrix factorization when attempting to
!    detect linearly dependent constraints.

        REAL ( KIND = wp ) :: pivot_tol_for_dependencies = half

!   any pivots smaller than zero_pivot in absolute value will be regarded to be
!    zero when attempting to detect linearly dependent constraints    (OBSOLETE)

        REAL ( KIND = wp ) :: zero_pivot = epsmch

!   the search direction is considered as an acceptable approximation
!    to the minimizer of the model if the gradient of the model in the
!    preconditioning(inverse) norm is less than
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )

        REAL ( KIND = wp ) :: inner_stop_relative = point01
        REAL ( KIND = wp ) :: inner_stop_absolute = epsmch

!   any dual variable or Lagrange multiplier which is less than multiplier_tol
!    outside its optimal interval will be regarded as being acceptable when
!    checking for optimality

        REAL ( KIND = wp ) :: multiplier_tol = epsmch

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!   any problem bound with the value zero will be treated as if it were a
!    general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if solve_qp is .TRUE., the value of prob%rho_g and prob%rho_b will be
!    increased as many times as are needed to ensure that the output
!    solution is feasible

        LOGICAL :: solve_qp = .FALSE.

!   if solve_within_bounds is  .TRUE., the value of prob%rho_b will be
!    increased as many times as are needed to ensure that the output
!    solution is feasible with respect to the simple bounds

        LOGICAL :: solve_within_bounds = .FALSE.

!   if randomize is .TRUE., the constraint bounds will be perturbed by
!    small random quantities during the first stage of the solution
!    process. Any randomization will ultimately be removed. Randomization
!    helps when solving degenerate problems

        LOGICAL :: randomize = .TRUE.

!   if %array_syntax_worse_than_do_loop is true, f77-style do loops will be
!    used rather than f90-style array syntax for vector operations    (OBSOLETE)

        LOGICAL :: array_syntax_worse_than_do_loop = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!  indefinite linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  definite linear equation solver

!       CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
!          "sils" // REPEAT( ' ', 26 )

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "QPAPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  component specifically for parametric problems (not used at present)

        LOGICAL :: each_interval = .FALSE.

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPA_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent analysing the required matrices prior to factorizatio

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = WP ) :: clock_preprocess = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = WP ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = WP ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = WP ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPA_inform_type

!  return status. See QPB_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of major iterations required

        INTEGER :: major_iter = - 1

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the total number of conjugate gradient iterations required

        INTEGER :: cg_iter = - 1

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ):: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = 0

!  the total number of factorizations which were modified to ensure that the
!   matrix was an appropriate preconditioner

        INTEGER :: nmods = 0

!  the number of infeasible general constraints

        INTEGER :: num_g_infeas = - 1

!  the number of infeasible simple-bound constraints

        INTEGER :: num_b_infeas = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by QPB_solve

        REAL ( KIND = wp ) :: obj = biginf

!  the 1-norm of the infeasibility of the general constraints

        REAL ( KIND = wp ) :: infeas_g = biginf

!  the 1-norm of the infeasibility of the simple-bound constraints

        REAL ( KIND = wp ) :: infeas_b = biginf

!  the merit function value = obj + rho_g * infeas_g + rho_b * infeas_b

        REAL ( KIND = wp ) :: merit = biginf

!  timings (see above)

        TYPE ( QPA_time_type ) :: time

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: SLS_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   Q P A _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QPA. This routine should be called before
!  QPA_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  control  a structure output informationan. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPA_control_type ), INTENT( OUT ) :: control
      TYPE ( QPA_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  Real parameters

      control%feas_tol = epsmch ** 0.75
      control%obj_unbounded = - epsmch ** ( - 2.0 )
      control%pivot_tol = point1 * epsmch ** 0.5
      control%zero_pivot = epsmch ** 0.75
      control%multiplier_tol = epsmch ** 0.5
      control%inner_stop_absolute = epsmch ** 0.5

!  initalize SLS components

      CALL SLS_INITIALIZE( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )

      control%SLS_control%relative_pivot_tolerance = control%pivot_tol
      control%SLS_control%zero_tolerance = control%zero_pivot
      control%SLS_control%prefix = '" - SLS:"                    '

      RETURN

!  End of QPA_initialize

      END SUBROUTINE QPA_initialize

!-*-*-*-*-   Q P A _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QPA_initialize could (roughly)
!  have been set as:

! BEGIN QPA SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      1000
!  start-print                                       -1
!  stop-print                                        -1
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  maximum-dimension-of-schur-complement             75
!  initial-integer-workspace                         1000
!  initial-real-workspace                            1000
!  maximum-refinements                               1
!  maximum-infeasible-iterations-before-rho-increase 100
!  maximum-number-of-cg-iterations                   -1
!  preconditioner-used                               0
!  semi-bandwidth-for-band-preconditioner            5
!  full-max-fill-ratio                               10
!  deletion-strategy                                 0
!  restore-problem-on-output                         2
!  residual-monitor-interval                         1
!  cold-start-strategy                               3
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  feasibility-tolerance                             1.0D-12
!  minimum-objective-before-unbounded                -1.0D+32
!  increase-rho-g-factor                             2.0
!  increase-rho-b-factor                             2.0
!  infeasible-g-required-improvement-factor          0.75
!  infeasible-b-required-improvement-factor          0.75
!  pivot-tolerance-used                              1.0D-8
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  multiplier-tolerance                              1.0D-8
!  inner-iteration-relative-accuracy-required        0.0
!  inner-iteration-absolute-accuracy-required        1.0E-8
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  treat-zero-bounds-as-general                      F
!  solve-qp                                          F
!  solve-within-bounds                               F
!  temporarily-perturb-constraint-bounds             T
!  array-syntax-worse-than-do-loop                   F
!  generate-sif-file                                 F
!  symmetric-linear-equation-solver                  sils
!  sif-file-name                                     QPAPROB.SIF
! END QPA SPECIFICATIONS

!  Dummy arguments

      TYPE ( QPA_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 45
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'QPA'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec(  5 )%keyword = 'start-print'
      spec(  6 )%keyword = 'stop-print'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  9 )%keyword = 'maximum-dimension-of-schur-complement'
      spec( 10 )%keyword = 'initial-integer-workspace'
      spec( 11 )%keyword = 'initial-real-workspace'
      spec( 12 )%keyword = 'maximum-refinements'
      spec( 13 )%keyword = 'maximum-infeasible-iterations-before-rho-increase'
      spec( 14 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 15 )%keyword = 'preconditioner-used'
      spec( 16 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 17 )%keyword = 'full-max-fill-ratio'
      spec( 18 )%keyword = 'deletion-strategy'
      spec( 19 )%keyword = 'restore-problem-on-output'
      spec( 20 )%keyword = 'residual-monitor-interval'
      spec( 21 )%keyword = 'cold-start-strategy'
      spec( 41 )%keyword = 'sif-file-device'

!  Real key-words

      spec( 22 )%keyword = 'infinity-value'
      spec( 23 )%keyword = 'feasibility-tolerance'
      spec( 24 )%keyword = 'minimum-objective-before-unbounded'
      spec( 25 )%keyword = 'increase-rho-g-factor'
      spec( 26 )%keyword = 'increase-rho-b-factor'
      spec( 27 )%keyword = 'infeasible-g-required-improvement-factor'
      spec( 28 )%keyword = 'infeasible-b-required-improvement-factor'
      spec( 29 )%keyword = 'pivot-tolerance-used'
      spec( 30 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 31 )%keyword = 'zero-pivot-tolerance'
      spec( 32 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 33 )%keyword = 'inner-iteration-absolute-accuracy-required'
      spec( 34 )%keyword = 'multiplier-tolerance'
      spec( 40 )%keyword = 'maximum-cpu-time-limit'
      spec( 43 )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( 35 )%keyword = 'treat-zero-bounds-as-general'
      spec( 36 )%keyword = 'solve-qp'
      spec( 37 )%keyword = 'solve-within-bounds'
      spec( 38 )%keyword = 'temporarily-perturb-constraint-bounds'
      spec( 39 )%keyword = 'array-syntax-worse-than-do-loop'
      spec( 42 )%keyword = 'generate-sif-file'

!  Character key-words

      spec( 44 )%keyword = 'sif-file-name'
      spec( 45 )%keyword = 'symmetric-linear-equation-solver'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( 1 ), control%error,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 2 ), control%out,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 3 ), control%print_level,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 4 ), control%maxit,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 5 ), control%start_print,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 6 ), control%stop_print,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 7 ), control%factor,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 8 ), control%max_col,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 9 ), control%max_sc,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 10 ), control%indmin,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 11 ), control%valmin,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 12 ), control%itref_max,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 13 ), control%infeas_check_interval,   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 14 ), control%cg_maxit,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 15 ), control%precon,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 16 ), control%nsemib,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 17 ), control%full_max_fill,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 18 ), control%deletion_strategy,       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 19 ), control%restore_problem,         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 20 ), control%monitor_residuals,       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 21 ), control%cold_start,              &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 41 ), control%sif_file_device,         &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( 22 ), control%infinity,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 23 ), control%feas_tol,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 24 ), control%obj_unbounded,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 25 ), control%increase_rho_g_factor,   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 26 ), control%increase_rho_g_factor,   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 27 ),                                  &
                                  control%infeas_g_improved_by_factor,         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 28 ),                                  &
                                  control%infeas_b_improved_by_factor,         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 29 ), control%pivot_tol,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 30 ),                                  &
                                  control%pivot_tol_for_dependencies,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 31 ), control%zero_pivot,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 32 ), control%inner_stop_relative,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 33 ), control%inner_stop_absolute,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 34 ), control%multiplier_tol,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 40 ), control%cpu_time_limit,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 43 ), control%clock_time_limit,        &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( 35 ),                                  &
                                  control%treat_zero_bounds_as_general,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 36 ), control%solve_qp,                &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 37 ), control%solve_within_bounds,     &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 38 ), control%randomize,               &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 39 ),                                  &
                                  control%array_syntax_worse_than_do_loop,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 42 ), control%generate_sif_file,       &
                                  control%error )

!  Set character value

      CALL SPECFILE_assign_value( spec( 44 ), control%sif_file_name,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 45 ), control%symmetric_linear_solver, &
                                  control%error )

!  Read the specfiles for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

      RETURN

      END SUBROUTINE QPA_read_specfile

!-*-*-*-*-*-*-**-*-   Q P A _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE QPA_solve( prob, C_stat, B_stat, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize   1/2 x(T) H x + g(T) x + f
!                  + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!                  + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  where x is a vector of n components ( x_1, .... , x_n ), f, rho_g/rho_b are
!  constants, g is an n-vector, H is a symmetric matrix, A is an m by n matrix,
!  and any of the bounds c_l, c_u, x_l, x_u may be infinite, using an active
!  set method. The subroutine is particularly appropriate when A and H are
!  sparse, and when we do not anticipate a large number of active set
!  changes prior to optimality
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
!    to be solved since the last call to QPA_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H. Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
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
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!    On exit, G will most likely have been reordered.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %rho_g is a REAL variable, which must be set by the user to the
!    required value of the penalty parameter for the general constraints
!
!   %rho_b is a REAL variable, which must be set by the user to the
!    required value of the penalty parameter for the simple bound constraints
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
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
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
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPA_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; C_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   C_stat( i ) < 0, the i-th constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPA_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   B_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type QPA_data_type which holds private internal data
!
!  control is a structure of type QPA_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QPA_initialize. See the preamble
!   for details
!
!  inform is a structure of type QPA_inform_type that provides
!    information on exit from QPA_solve. The component status
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
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!          prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', 'DIAGONAL'}
!       has been violated.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -18 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!   -23 an entry from the strict upper triangle of H has been input.
!
!  On exit from QPA_solve, the other components of inform are as described
!   in the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      INTEGER, INTENT( INOUT ), DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
      TYPE ( QPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( OUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, l, lbd, m_link, k_n_max, lbreak, n_depen
      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, type, n_pcg
      REAL ( KIND = wp ) :: a_x, a_norms
      REAL :: time_start, time_record, time_now, time_inner_start
      REAL ( KIND = WP ) :: clock_start, clock_record, clock_now, clock_inner_start
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' -- entering QPA_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  Initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Initialize counts

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%major_iter = 0 ; inform%iter = 0 ; inform%cg_iter = 0 ;
      inform%nfacts = 0 ; inform%nmods = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%factorization_status = 0
      inform%obj = prob%f

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_H( prob%H%type ) .OR.                             &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status
        GO TO 800
      END IF

!  If required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
        WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )       &
            prob%H%val( : prob%n )
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' H (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%H%col( j ), prob%H%val( j ),                           &
                j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' H (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, prob%H%ne)
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

!  Set Hessian and gradient types to generic - this may change in future

      prob%Hessian_kind = - 1 ; prob%gradient_kind = - 1

!  ===========================
!  Preprocess the problem data
!  ===========================

      prob%Y = zero ; prob%Z = zero
      data%new_problem_structure = prob%new_problem_structure
      IF ( prob%new_problem_structure ) THEN
        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general
        IF ( control%randomize )                                               &
          data%QPP_control%treat_zero_bounds_as_general = .TRUE.

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /, A,    &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Perform the preprocessing

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims,                          &
                          prob, .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I3, ' after QPP_reorder ')" )&
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == GALAHAD_error_upper_entry )                    &
             WRITE( control%error, "( /, A, '  Warning - an entry from ',      &
            &  'strict upper triangle of H given ' )" ) prefix
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          GO TO 800
        END IF

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions after preprocessing: ', /, A,     &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        prob%new_problem_structure = .FALSE.

!  Recover the problem dimensions after preprocessing

      ELSE
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_apply( data%QPP_map, data%QPP_inform, prob, get_all = .TRUE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I3, ' after QPP_apply ')" )  &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        END IF
      END IF

!  Permute initial working sets if provided

      IF ( control%cold_start == 0 ) THEN
        CALL SORT_inplace_permute( data%QPP_map%m, data%QPP_map%c_map,        &
                                   IX = C_stat( : data%QPP_map%m ) )
        CALL SORT_inplace_permute( data%QPP_map%n, data%QPP_map%x_map,        &
                                   IX = B_stat( : data%QPP_map%n ) )
      END IF

!  ===========================================================================
!  Check to see if the constraints in the working set are linearly independent
!  ===========================================================================

!  Allocate workspace arrays

      lbreak = prob%m + data%dims%c_l_end - data%dims%c_u_start +              &
               prob%n - data%dims%x_free + data%dims%x_l_end -                 &
               data%dims%x_u_start + 2

      array_name = 'qpa: data%IBREAK'
      CALL SPACE_resize_array( lbreak, data%IBREAK, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%RES_l'
      CALL SPACE_resize_array( 1, data%dims%c_l_end, data%RES_l,               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%RES_u'
      CALL SPACE_resize_array( data%dims%c_u_start, prob%m, data%RES_u,        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%A_norms'
      CALL SPACE_resize_array( prob%m, data%A_norms, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Compute the initial residuals

      DO i = 1, data%dims%c_u_start - 1
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'l', i, a_x, prob%C_l( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_u_start, data%dims%c_l_end
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'l', i, a_x, prob%C_l( i )
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_l_end + 1, prob%m
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

!  If necessary, determine which constraints occur in the reference set

!  cold start from the value set in X; constraints active
!  at X will determine the initial working set.

      IF ( control%cold_start == 1 ) THEN

!  constraints with lower bounds

        DO i = 1, data%dims%c_u_start - 1
          IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_l( i )
            C_stat( i ) = - 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  constraints with both lower and upper bounds

        DO i = data%dims%c_u_start, data%dims%c_l_end
          IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_l( i )
            C_stat( i ) = - 1
          ELSE IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_u( i )
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  constraints with upper bounds

        DO i = data%dims%c_l_end + 1, prob%m
          IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_u( i )
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  free variables

        B_stat( : data%dims%x_free ) = 0

!  simple non-negativity

        DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
          IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i )
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from below

        DO i = data%dims%x_l_start, data%dims%x_u_start - 1
          IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from below and above

        DO i = data%dims%x_u_start, data%dims%x_l_end
          IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
            B_stat( i ) = - 1
          ELSE IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_u( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from above

        DO i = data%dims%x_l_end + 1, data%dims%x_u_end
          IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_u( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple non-positivity

        DO i = data%dims%x_u_end + 1, prob%n
          IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  cold start with only equality constraints active

      ELSE IF ( control%cold_start == 3 ) THEN
        B_stat = 0
        C_stat( : MIN( data%dims%c_equality, prob%n ) ) = 1
        C_stat( MIN( data%dims%c_equality, prob%n ) + 1 : ) = 0

!  cold start with as many active constraints as possible

      ELSE IF ( control%cold_start == 4 ) THEN
        B_stat = 0 ; C_stat = 0
        l = 0

!  equality constraints

        DO i = 1,  data%dims%c_equality
          IF ( l > prob%n ) EXIT
          C_stat( i ) = - 1
          l = l + 1
        END DO

!  simple bound from below

        DO i = data%dims%x_free + 1, data%dims%x_l_end
          IF ( l > prob%n ) EXIT
          B_stat( i ) = - 1
          l = l + 1
        END DO

!  simple bound from above

        DO i = data%dims%x_l_end + 1, data%dims%x_u_end
          IF ( l > prob%n ) EXIT
          B_stat( i ) = 1
          l = l + 1
        END DO

!  constraints with lower bounds

        DO i = data%dims%c_equality + 1, data%dims%c_l_end
          IF ( l > prob%n ) EXIT
          C_stat( i ) = - 1
          l = l + 1
        END DO

!  constraints with upper bounds

        DO i = data%dims%c_l_end + 1, prob%m
          IF ( l > prob%n ) EXIT
          C_stat( i ) = 1
          l = l + 1
        END DO

!  cold start with no active constraints

      ELSE IF ( control%cold_start /= 0 ) THEN
        B_stat = 0 ; C_stat = 0
      ELSE
        DO i = 1, prob%n
          IF ( B_stat( i ) < 0 ) THEN
            B_stat( i ) = - 1
          ELSE IF ( B_stat( i ) > 0 ) THEN
            B_stat( i ) = 1
          END IF
        END DO
        DO i = 1, prob%m
          IF ( C_stat( i ) < 0 ) THEN
            C_stat( i ) = - 1
          ELSE IF ( C_stat( i ) > 0 ) THEN
            C_stat( i ) = 1
          END IF
        END DO
      END IF

!  Remove any dependent working constraints
!  ========================================

      data%SLS_control = control%SLS_control

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL QPA_remove_dependent( prob%n, prob%m, prob%A%val, prob%A%col,       &
                                 prob%A%ptr, data%K, data%SLS_data,            &
                                 data%SLS_control, C_stat, B_stat,             &
                                 data%IBREAK, data%P, data%SOL, data%D,        &
                                 prefix, control, inform, n_depen )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%preprocess =                                               &
        inform%time%preprocess + time_now - time_record
      inform%time%clock_preprocess =                                         &
        inform%time%clock_preprocess + clock_now - clock_record

!  Allocate more real workspace arrays

      array_name = 'qpa: data%H_s'
      CALL SPACE_resize_array( prob%n, data%H_s, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Check for error exits

      IF ( inform%status /= 0 ) THEN

!  On error exit, compute the current objective function value

        data%H_s( : prob%n ) = zero
        CALL QPA_HX( data%dims, prob%n, data%H_s( : prob%n ),                  &
                      prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,                &
                      prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
        inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),                   &
                                         data%H_s( : prob%n ) )                &
                     + DOT_PRODUCT( prob%X( : prob%n ),                        &
                                    prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

        IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
          WRITE( control%out, "( ' ' )" )
          IF ( inform%status /= 0 ) WRITE( control%error,                      &
             "( A, '   **  Error return ', I0, ' from ', A ) " )               &
            prefix, inform%status, 'QPA_remove_dependent'
        END IF
        GO TO 750

!       IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
!         WRITE( control%out, "(/, ' The following ',I7,' constraints appear', &
!      &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

      END IF

!  Continue allocating workspace arrays

!     m_link = MIN( prob%m + prob%n - data%dims%x_free, prob%n )
      m_link = prob%m + prob%n - data%dims%x_free
      k_n_max = prob%n + m_link

!  Allocate real workspace

      array_name = 'qpa: data%BREAKP'
      CALL SPACE_resize_array( lbreak, data%BREAKP, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%A_s'
      CALL SPACE_resize_array( prob%m, data%A_s, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%PERT'
      CALL SPACE_resize_array( prob%m + prob%n, data%PERT, inform%status,      &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%GRAD'
      CALL SPACE_resize_array( prob%n, data%GRAD, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%VECTOR'
      CALL SPACE_resize_array( k_n_max, data%VECTOR, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%RHS'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%RHS,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%S'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%S,               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%B'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%B,               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%RES'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%RES,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%S_perm'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%S_perm,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%DX'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%DX,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%RES_print'
      CALL SPACE_resize_array( k_n_max + control%max_sc, data%RES_print,       &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( control%precon >= 0 ) THEN
        n_pcg = prob%n
      ELSE
        n_pcg = 0
      END IF

      array_name = 'qpa: data%R_pcg'
      CALL SPACE_resize_array( n_pcg, data%R_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%X_pcg'
      CALL SPACE_resize_array( n_pcg, data%X_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%P_pcg'
      CALL SPACE_resize_array( n_pcg, data%P_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Allocate integer workspace arrays

      array_name = 'qpa: data%SC'
      CALL SPACE_resize_array( control%max_sc + 1, data%SC, inform%status,     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%REF'
      CALL SPACE_resize_array( m_link, data%REF, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%C_up_or_low'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_l_end,         &
             data%C_up_or_low, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%X_up_or_low'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_l_end,         &
             data%X_up_or_low, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%PERM'
      CALL SPACE_resize_array(  k_n_max + control%max_sc, data%PERM,           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Find the total length of the control%max_sc largest rows

      DO i = 1, prob%m
        data%IBREAK( i ) = prob%A%ptr( i ) - prob%A%ptr( i + 1 )
      END DO
      CALL SORT_heapsort_build( prob%m, data%IBREAK( : prob%m ),               &
                                inform%status )
      lbd = 0
      DO i = 1, MIN( control%max_sc, prob%m )
        ii = prob%m - i + 1
        CALL SORT_heapsort_smallest( ii, data%IBREAK( : ii ), inform%status )
        lbd = lbd - data%IBREAK( ii )
      END DO
      IF ( control%max_sc > prob%m )                                           &
        lbd = lbd + control%max_sc - prob%m

!  Allocate arrays

      array_name = 'qpa: data%SCU_mat%BD_col_start'
      CALL SPACE_resize_array( control%max_sc + 1, data%SCU_mat%BD_col_start,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%SCU_mat%BD_val'
      CALL SPACE_resize_array( lbd, data%SCU_mat%BD_val, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%SCU_mat%BD_row'
      CALL SPACE_resize_array( lbd, data%SCU_mat%BD_row, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%DIAG'
      CALL SPACE_resize_array( 2, K_n_max, data%DIAG, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  decide on appropriate initial preconditioners and factorizations

      data%auto_prec = control%precon == 0
      data%auto_fact = control%factor == 0

!  If the Hessian has semi-bandwidth smaller than nsemib and the preconditioner
!  is to be picked automatically, use the full Hessian. Otherwise, use the
!  Hessian of the specified semi-bandwidth.

      IF ( data%auto_prec ) THEN
        data%prec_hist = 2

!  prec_hist indicates which factors are currently being used. Possible values:
!   1 full factors used
!   2 band factors used
!   3 diagonal factors used (as a last resort)

!  Check to see if the Hessian is banded

 dod :  DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = data%dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_free

          CASE ( 2 )

            hd_start  = data%dims%x_free + 1
            hd_end    = data%dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = data%dims%x_l_start
            hd_end    = data%dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = data%dims%x_u_start
            hd_end    = data%dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_end

          CASE ( 5 )

            hd_start  = data%dims%x_l_end + 1
            hd_end    = data%dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_end

          CASE ( 6 )

            hd_start  = data%dims%x_u_end + 1
            hd_end    = data%dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = prob%n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, prob%n )
          DO i = hd_start, hd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 2
              IF ( ABS( i - prob%H%col( l ) ) > control%nsemib ) THEN
                data%prec_hist = 1
                EXIT dod
              END IF
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, prob%n )
          DO i = hnd_start, hnd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              IF ( ABS( i - prob%H%col( l ) ) > control%nsemib ) THEN
                data%prec_hist = 1
                EXIT dod
              END IF
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT

        END DO dod

      END IF

!  =================
!  Solve the problem
!  =================

      data%SLS_control = control%SLS_control

      CALL CPU_TIME( time_inner_start ) ; CALL CLOCK_TIME( clock_inner_start )
      CALL QPA_solve_qp( data%dims, prob%n, prob%m,                            &
                         prob%H%val, prob%H%col, prob%H%ptr,                   &
                         prob%G, prob%f, prob%rho_g, prob%rho_b, prob%A%val,   &
                         prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, prob%X_l, &
                         prob%X_u, prob%X, prob%Y, prob%Z, C_stat, B_stat,     &
                         m_link, K_n_max, lbreak, data%RES_l, data%RES_u,      &
                         data%A_norms, data%H_s, data%BREAKP, data%A_s,        &
                         data%PERT, data%GRAD, data%VECTOR, data%RHS, data%S,  &
                         data%B, data%RES, data%S_perm, data%DX, n_pcg,        &
                         data%R_pcg, data%X_pcg, data%P_pcg, data%Abycol_val,  &
                         data%Abycol_row, data%Abycol_ptr, data%S_val,         &
                         data%S_row, data%S_col, data%S_colptr, data%IBREAK,   &
                         data%SC, data%REF, data%RES_print, data%DIAG,         &
                         data%C_up_or_low, data%X_up_or_low, data%PERM,        &
                         data%P, data%SOL, data%D,                             &
                         data%SLS_data, data%SLS_control,                      &
                         data%SCU_mat, data%SCU_info, data%SCU_data, data%K,   &
                         data%seed, time_inner_start, clock_inner_start,       &
                         data%start_print, data%stop_print,                    &
                         data%prec_hist, data%auto_prec, data%auto_fact,       &
                         printi, prefix, control, inform )

!  Restore the problem to its original form

  750 CONTINUE
      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,           &
                                 IX = C_stat( : data%QPP_map%m ) )
      B_stat( prob%n + 1 : data%QPP_map%n ) = - 1
      CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,           &
                                 IX = B_stat( : data%QPP_map%n ) )

!  Full restore

      IF ( control%restore_problem >= 2 ) THEN
        CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,                 &
                          get_all = .TRUE. )

!  Restore vectors and scalars

      ELSE IF ( control%restore_problem == 1 ) THEN
        CALL QPP_restore( data%QPP_map, data%QPP_inform,                       &
                           prob, get_g = .TRUE.,                               &
                           get_x = .TRUE., get_x_bounds = .TRUE.,              &
                           get_y = .TRUE., get_z = .TRUE.,                     &
                           get_c = .TRUE., get_c_bounds = .TRUE. )

!  Solution recovery

      ELSE
        CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,                 &
                          get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,      &
                          get_c = .TRUE. )
      END IF
!     CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%preprocess = inform%time%preprocess + time_now - time_record
      inform%time%clock_preprocess =                                           &
        inform%time%clock_preprocess + clock_now - clock_record
      prob%new_problem_structure = data%new_problem_structure

!  Compute total time

  800 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
           "( /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',            &
          &   /, 14X, ' =', 10X,  'QPA timing statistics', 8X, '=',            &
          &   /, 14X, ' =         total         preprocess      =',            &
          &   /, 14X, ' =', 2X, 0P, F12.2, 4X, F12.2, 9X, '=',                 &
          &   /, 14X, ' =     analyse   factorize    solve      =',            &
          &   /, 14X, ' =', 3F11.2, 6x, '=',                                   &
          &   /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=' )" )        &
        inform%time%total, inform%time%preprocess,                             &
        inform%time%analyse, inform%time%factorize, inform%time%solve

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPA_solve ' )" )
      RETURN

!  Allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
         "( A, ' ** Message from -QPA_solve-', /,                              &
        &   A, ' Allocation error, for ', A, /, A, ' status = ', I0 )" )       &
        prefix, prefix, inform%bad_alloc, prefix, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPA_solve ' )" )
      RETURN

!  Non-executable statements

 2010 FORMAT( ' ', /, A, '   **  Error return ',I3,' from QPA ' )

!  End of QPA_solve

      END SUBROUTINE QPA_solve

! -*-*-*-*-*-*-*-   Q P A _ S O L V E _ Q P   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE QPA_solve_qp( dims, n, m, H_val, H_col, H_ptr, G, f, rho_g,   &
                               rho_b, A_val, A_col, A_ptr, C_l, C_u, X_l, X_u, &
                               X, Y, Z, C_stat, B_stat, m_link, K_n_max,       &
                               lbreak, RES_l, RES_u, A_norms, H_s, BREAKP,     &
                               A_s, PERT, GRAD, VECTOR, RHS, S, B, RES,        &
                               S_perm, DX, n_pcg, R_pcg, X_pcg, P_pcg,         &
                               Abycol_val, Abycol_row, Abycol_ptr, S_val,      &
                               S_row, S_col, S_colptr, IBREAK, SC, REF,        &
                               RES_print, DIAG, C_up_or_low, X_up_or_low,      &
                               PERM, P, SOL, D, SLS_data, SLS_control,         &
                               SCU_mat, SCU_info, SCU_data, K,                 &
                               seed, time_inner_start, clock_inner_start,      &
                               start_print, stop_print, prec_hist, auto_prec,  &
                               auto_fact, printi, prefix, control, inform,     &
                               G_p, X_p, Y_p, Z_p )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize 1/2 x(T) H x + g(T) x + f
!               + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!               + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  where x is a vector of n components ( x_1, .... , x_n ), f, rho_g/rho_b are
!  constant, g is an n-vector, H is a symmetric matrix, A is an m by n matrix,
!  and any of the bounds c_l, c_u, x_l, x_u may be infinite, using an active
!  set method. The subroutine is particularly appropriate when A and H are
!  sparse, and when we do not anticipate a large number of active set
!  changes prior to optimality
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
      INTEGER, INTENT( IN ) :: n, m, m_link, k_n_max, lbreak, n_pcg
      INTEGER, INTENT( INOUT ) :: start_print, stop_print, prec_hist
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( INOUT ) :: rho_g, rho_b
      REAL, INTENT( IN ) :: time_inner_start
      REAL ( KIND = wp ), INTENT( IN ) :: clock_inner_start
      LOGICAL, INTENT( IN ) :: printi
      LOGICAL, INTENT( INOUT ) :: auto_prec, auto_fact
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
      INTEGER, INTENT( OUT ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( OUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( OUT ),                                                  &
               DIMENSION( dims%c_u_start : dims%c_l_end ) :: C_up_or_low
      INTEGER, INTENT( OUT ),                                                  &
               DIMENSION( dims%x_u_start : dims%x_l_end ) :: X_up_or_low
      INTEGER, INTENT( OUT ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row, Abycol_ptr
      INTEGER, ALLOCATABLE, DIMENSION( : )  :: S_row, S_col, S_colptr
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: P
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( :, : ) :: D
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_norms
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lbreak ) :: BREAKP
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: H_s, GRAD
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( k_n_max ) :: VECTOR
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m + n ) :: PERT
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( k_n_max + control%max_sc ) :: RHS, S, B,  &
                                     RES, S_perm, DX, RES_print
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, k_n_max ) :: DIAG
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( n_pcg ) :: R_pcg, X_pcg, P_pcg
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Abycol_val, S_val
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

      REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( n ) :: G_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: Z_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: Y_p

!  Local variables

      INTEGER :: i, pass, icrit, ncrit, initial_seed
      REAL ( KIND = wp ) :: feas_tol, feas_val, last_infeas_g, last_infeas_b
      REAL ( KIND = wp ) :: damp, randm, rand_max
      LOGICAL :: got_sol, warm_start

      got_sol = .FALSE. ; warm_start = control%cold_start /= 1
      feas_tol = control%feas_tol ; feas_val = epsmch ** 0.4

!  If required, perturb constraint bounds by small pseudo-random numbers

      IF ( control%randomize .AND. m > 0 ) THEN
        rand_max = epsmch ** 0.5
!       rand_max = epsmch ** 0.375
!       rand_max = epsmch ** 0.75
!       rand_max = epsmch ** 0.25
!       rand_max = zero
        initial_seed = 1234567
        CALL RAND_set_seed( seed, initial_seed )
        IF ( printi ) WRITE( control%out,                                      &
             "( /, A, 16X, '   !!!!! RANDOMIZING RHS !!!!! ' )" ) prefix

!  general constraints

        DO i = dims%c_equality + 1, dims%c_l_end
          CALL RAND_random_real(  seed, .TRUE., randm )
          c_l( i ) = c_l( i ) - rand_max * randm
        END DO

        DO i = dims%c_u_start, m
          CALL RAND_random_real(  seed, .TRUE., randm )
          c_u( i ) = c_u( i ) + rand_max * randm
        END DO

!  simple bounds

        DO i = dims%x_l_start,  dims%x_l_end
          CALL RAND_random_real( seed, .TRUE., randm )
          x_l( i ) = x_l( i ) - rand_max * randm
        END DO

        DO i = dims%x_u_start,  dims%x_u_end
          CALL RAND_random_real( seed, .TRUE., randm )
          x_u( i ) = x_u( i ) + rand_max * randm
        END DO

      END IF

!  Solution loop: gradually reduce any perturbations
!  =============

      pass = 0 ; icrit = 0 ; ncrit = 9
      last_infeas_g = biginf / two ; last_infeas_b = biginf / two
      DO
        pass = pass + 1
        CALL QPA_solve_main( dims, n, m, H_val, H_col, H_ptr, G, f, rho_g,     &
                       rho_b, A_val, A_col, A_ptr, C_l, C_u, X_l, X_u, X, Y, Z,&
                       C_stat, B_stat, m_link, K_n_max, lbreak, RES_l, RES_u,  &
                       A_norms, H_s, BREAKP, A_s, PERT, GRAD, VECTOR, RHS, S,  &
                       B, RES, S_perm, DX, n_pcg, R_pcg, X_pcg, P_pcg,         &
                       Abycol_val, Abycol_row, Abycol_ptr, S_val, S_row, S_col,&
                       S_colptr, IBREAK, SC, REF, RES_print, DIAG, C_up_or_low,&
                       X_up_or_low, PERM, P, SOL, D, SLS_data, SLS_control,    &
                       SCU_mat, SCU_info, SCU_data, K, warm_start,             &
                       time_inner_start, clock_inner_start,                    &
                       last_infeas_g, last_infeas_b,                           &
                       start_print, stop_print, prec_hist, auto_prec,          &
                       auto_fact, pass, feas_tol, prefix, control, inform,     &
                       G_p = G_p, X_p = X_p, Y_p = Y_p, Z_p = Z_p )

        IF ( inform%status < GALAHAD_ok .AND.                                  &
             inform%status /=  GALAHAD_error_unbounded ) THEN
          IF ( control%randomize .AND. m > 0 ) THEN

!  Unperturb the constraints

            IF ( got_sol ) THEN
              damp = one - damp
            ELSE
              IF ( printi ) WRITE( control%out,                                &
                   "( /, A, 16X, '   !!!!! REMOVING RANDOMIZATION !!!!! ' )" ) &
                 prefix
              damp = one
            END IF
!           CALL RANDOM_SEED( put = initial_seed )
            CALL RAND_set_seed( seed, initial_seed )

!  general constraints

            DO i = dims%c_equality + 1, dims%c_l_end
              CALL RAND_random_real( seed, .TRUE., randm )
              c_l( i ) = c_l( i ) + damp * rand_max * randm
            END DO

            DO i = dims%c_u_start, m
              CALL RAND_random_real( seed, .TRUE., randm )
              c_u( i ) = c_u( i ) - damp * rand_max * randm
            END DO

!  simple bounds

            DO i = dims%x_l_start,  dims%x_l_end
              CALL RAND_random_real( seed, .TRUE., randm )
              x_l( i ) = x_l( i ) + damp * rand_max * randm
            END DO

            DO i = dims%x_u_start,  dims%x_u_end
              CALL RAND_random_real( seed, .TRUE., randm )
              x_u( i ) = x_u( i ) - damp * rand_max * randm
            END DO
          END IF
          EXIT
        END IF

        IF ( printi ) WRITE( control%out, "( '' )" )
        IF ( inform%status == 0 ) THEN
          IF ( .NOT. ( control%solve_qp .OR. control%solve_within_bounds )     &
               .OR. got_sol                                                    &
               .OR. ( control%solve_qp .AND.                                   &
                      inform%infeas_g + inform%infeas_b <= feas_val )          &
               .OR. ( .NOT. control%solve_qp .AND.                             &
                      control%solve_within_bounds .AND.                        &
                      inform%infeas_b <= feas_val ) ) THEN
            IF ( got_sol ) THEN

!  Unperturb the constraints

              IF ( control%randomize .AND. m > 0 ) THEN
                CALL RAND_set_seed( seed, initial_seed )
                damp = one - damp

!  general constraints

                DO i = dims%c_equality + 1, dims%c_l_end
                  CALL RAND_random_real( seed, .TRUE., randm )
                  c_l( i ) = c_l( i ) + damp * rand_max * randm
                END DO

                DO i = dims%c_u_start, m
                  CALL RAND_random_real( seed, .TRUE., randm )
                  c_u( i ) = c_u( i ) - damp * rand_max * randm
                END DO

!  simple bounds

                DO i = dims%x_l_start,  dims%x_l_end
                  CALL RAND_random_real( seed, .TRUE., randm )
                  x_l( i ) = x_l( i ) + damp * rand_max * randm
                END DO

                DO i = dims%x_u_start,  dims%x_u_end
                  CALL RAND_random_real( seed, .TRUE., randm )
                  x_u( i ) = x_u( i ) - damp * rand_max * randm
                END DO
              END IF
              EXIT
            END IF

!  Unperturb the constraints

            IF ( control%randomize .AND. m > 0 ) THEN
              CALL RAND_set_seed( seed, initial_seed )
              IF ( printi ) WRITE( control%out,                                &
                   "( A, 16X, '   !!!!! REMOVING RANDOMIZATION !!!!! ' )" )    &
                prefix
              damp = one - point1 * MIN( one, control%feas_tol / rand_max )

!  general constraints

              DO i = dims%c_equality + 1, dims%c_l_end
                CALL RAND_random_real( seed, .TRUE., randm )
                c_l( i ) = c_l( i ) + damp * rand_max * randm
              END DO

              DO i = dims%c_u_start, m
                CALL RAND_random_real( seed, .TRUE., randm )
                c_u( i ) = c_u( i ) - damp * rand_max * randm
              END DO

!  simple bounds

              DO i = dims%x_l_start,  dims%x_l_end
                CALL RAND_random_real( seed, .TRUE., randm )
                x_l( i ) = x_l( i ) + damp * rand_max * randm
              END DO

              DO i = dims%x_u_start,  dims%x_u_end
                CALL RAND_random_real( seed, .TRUE., randm )
                x_u( i ) = x_u( i ) - damp * rand_max * randm
              END DO

            END IF

!  Tighten the infeasibility tolerance

            feas_tol = epsmch ** 0.75
            IF ( printi ) WRITE( control%out,                                  &
                 "( A, 16X, ' tightening infeasibility tolerance ...')" )      &
              prefix
            got_sol = .TRUE.
          ELSE

!  Check to see that the infeasibility is decreasing

            IF ( control%solve_qp ) THEN
              IF ( inform%infeas_g + inform%infeas_b                           &
                   > 0.99_wp * ( last_infeas_g + last_infeas_b ) ) THEN
                icrit = icrit + 1
                IF ( icrit >= ncrit ) THEN
                  IF ( printi ) WRITE( control%out, 2160 ) prefix, ncrit, prefix
                  inform%status = GALAHAD_error_primal_infeasible
                  EXIT
                END IF
              ELSE
                icrit = 0
              END IF
              last_infeas_g = inform%infeas_g
              last_infeas_b = inform%infeas_b
              rho_g = control%increase_rho_g_factor * rho_g
              rho_b = control%increase_rho_b_factor * rho_b
            ELSE IF ( control%solve_within_bounds ) THEN
              IF ( inform%infeas_b > 0.99_wp * last_infeas_b ) THEN
                icrit = icrit + 1
                IF ( icrit >= ncrit ) THEN
                  IF ( printi ) WRITE( control%out, 2160 ) prefix, ncrit, prefix
                  inform%status = GALAHAD_error_primal_infeasible
                  EXIT
                END IF
              ELSE
                icrit = 0
              END IF
              last_infeas_b = inform%infeas_b
              rho_b = control%increase_rho_b_factor * rho_b
            ELSE
              EXIT
            END IF
          END IF
        ELSE

!  Check to see that the infeasibility is decreasing

          IF ( control%solve_qp ) THEN
            IF ( inform%infeas_g + inform%infeas_b <= feas_val .AND.           &
                 inform%status == GALAHAD_error_unbounded ) THEN
              icrit = icrit + 1
              IF ( icrit >= ncrit ) EXIT
            ELSE IF ( inform%infeas_g + inform%infeas_b                        &
                      > 0.99_wp * ( last_infeas_g + last_infeas_b ) ) THEN
              icrit = icrit + 1
              IF ( icrit >= ncrit ) THEN
                IF ( printi ) WRITE( control%out, 2160 ) prefix, ncrit, prefix
                inform%status = GALAHAD_error_primal_infeasible
                EXIT
              END IF
            ELSE
              icrit = 0
            END IF
            last_infeas_g = inform%infeas_g
            last_infeas_b = inform%infeas_b
            rho_g = control%increase_rho_g_factor * rho_g
            rho_b = control%increase_rho_b_factor * rho_b
          ELSE IF ( control%solve_within_bounds ) THEN
            IF ( inform%infeas_b <= feas_val .AND.                             &
                 inform%status == GALAHAD_error_unbounded ) THEN
              icrit = icrit + 1
              IF ( icrit >= ncrit ) EXIT
            ELSE IF ( inform%infeas_b > 0.99_wp * last_infeas_b ) THEN
              icrit = icrit + 1
              IF ( icrit >= ncrit ) THEN
                IF ( printi ) WRITE( control%out, 2160 ) prefix, ncrit, prefix
                inform%status = GALAHAD_error_primal_infeasible
                EXIT
              END IF
            ELSE
              icrit = 0
            END IF
            last_infeas_b = inform%infeas_b
            rho_b = control%increase_rho_b_factor * rho_b
          ELSE
            EXIT
          END IF
        END IF
        IF ( printi ) WRITE( control%out, "( A, 16X, ' infeasibility =',       &
       &  ES11.4, ', restarting ' )" ) prefix, inform%infeas_g + inform%infeas_b
        warm_start = .TRUE.
      END DO

      RETURN

!  Non-executable statements

 2160 FORMAT( /, A, ' Constraint violations have not decreased',               &
                 ' substantially over', I0, ' major iterations. ',             &
              /, A, ' Problem possibly infeasible, terminating run. ' )

!  End of QPA_solve_qp

      END SUBROUTINE QPA_solve_qp

!-*-*-*-*-*-    Q P A _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_solve_main( dims, n, m,                                   &
                                 H_val, H_col, H_ptr, G, f, rho_g, rho_b,      &
                                 A_val, A_col, A_ptr, C_l, C_u, X_l, X_u,      &
                                 X, Y, Z, C_stat, B_stat, m_link, K_n_max,     &
                                 lbreak, RES_l, RES_u, A_norms, H_s, BREAKP,   &
                                 A_s, PERT, GRAD, VECTOR, RHS, S, B, RES,      &
                                 S_perm, DX, n_pcg, R_pcg, X_pcg, P_pcg,       &
                                 Abycol_val, Abycol_row, Abycol_ptr, S_val,    &
                                 S_row, S_col, S_colptr, IBREAK, SC, REF,      &
                                 RES_print, DIAG, C_up_or_low, X_up_or_low,    &
                                 PERM, P, SOL, D, SLS_data, SLS_control,       &
                                 SCU_mat, SCU_info, SCU_data, K, warm_start,   &
                                 time_start, clock_start,                      &
                                 best_infeas_g, best_infeas_b,                 &
                                 start_print, stop_print, prec_hist,           &
                                 auto_prec, auto_fact, pass, feas_tol, prefix, &
                                 control, inform, G_p, X_p, Y_p, Z_p )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize 1/2 x(T) H x + g(T) x + f
!               + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!               + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  where x is a vector of n components ( x_1, .... , x_n ), f, rho_g/rho_b are
!  constant, g is an n-vector, H is a symmetric matrix, A is an m by n matrix,
!  and any of the bounds c_l, c_u, x_l, x_u may be infinite, using an active
!  set method. The subroutine is particularly appropriate when A and H are
!  sparse, and when we do not anticipate a large number of active set
!  changes prior to optimality
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type QPA_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. The following components must be set:
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %x_free is an INTEGER variable, which must be set by the user to the
!    number of free variables. RESTRICTION: %x_free >= 0
!
!   %x_l_start is an INTEGER variable, which must be set by the user to the
!    index of the first variable with a nonzero lower (or lower range) bound.
!    RESTRICTION: %x_l_start >= %x_free + 1
!
!   %x_l_end is an INTEGER variable, which must be set by the user to the
!    index of the last variable with a nonzero lower (or lower range) bound.
!    RESTRICTION: %x_l_end >= %x_l_start
!
!   %x_u_start is an INTEGER variable, which must be set by the user to the
!    index of the first variable with a nonzero upper (or upper range) bound.
!    RESTRICTION: %x_u_start >= %x_l_start
!
!   %x_u_end is an INTEGER variable, which must be set by the user to the
!    index of the last variable with a nonzero upper (or upper range) bound.
!    RESTRICTION: %x_u_end >= %x_u_start
!
!   %c_equality is an INTEGER variable, which must be set by the user to the
!    number of equality constraints, m. RESTRICTION: %c_equality >= 0
!
!   %c_l_start is an INTEGER variable, which must be set by the user to the
!    index of the first inequality constraint with a lower (or lower range)
!    bound. RESTRICTION: %c_l_start = %c_equality + 1
!    (strictly, this information is redundant!)
!
!   %c_l_end is an INTEGER variable, which must be set by the user to the
!    index of the last inequality constraint with a lower (or lower range)
!    bound. RESTRICTION: %c_l_end >= %c_l_start
!
!   %c_u_start is an INTEGER variable, which must be set by the user to the
!    index of the first inequality constraint with an upper (or upper range)
!    bound. RESTRICTION: %c_u_start >= %c_l_start
!    (strictly, this information is redundant!)
!
!   %c_u_end is an INTEGER variable, which must be set by the user to the
!    index of the last inequality constraint with an upper (or upper range)
!    bound. RESTRICTION: %c_u_end = %m
!    (strictly, this information is redundant!)
!
!   %h_diag_end_free is an INTEGER variable, which must be set by the user to
!    the index of the last free variable whose for which the Hessian has a
!    diagonal entry
!
!   %h_diag_end_nonneg is an INTEGER variable, which must be set by the user to
!    the index of the last nonnegative variable whose for which the Hessian has
!    a diagonal entry
!
!   %h_diag_end_lower is an INTEGER variable, which must be set by the user to
!    the index of the last flower-bounded variable whose for which the Hessian
!    has a diagonal entry
!
!   %h_diag_end_range is an INTEGER variable, which must be set by the user to
!    the index of the last range variable whose for which the Hessian has a
!    diagonal entry
!
!   %h_diag_end_upper is an INTEGER variable, which must be set by the user to
!    the index of the last upper-bounded variable whose for which the Hessian
!    has a diagonal entry
!
!   %h_diag_end_nonpos is an INTEGER variable, which must be set by the user to
!    the index of the last  nonpositive variable whose for which the Hessian
!    has a diagonal entry
!
!   %np1 is an INTEGER variable, which must be set by the user to the
!    value n + 1
!
!   %npm is an INTEGER variable, which must be set by the user to the
!    value n + m
!
!   %nc is an INTEGER variable, which must be set by the user to the
!    value dims%c_u_end - dims%c_l_start + 1
!
!   %x_s is an INTEGER variable, which must be set by the user to the
!    value 1
!
!   %x_e is an INTEGER variable, which must be set by the user to the
!    value n
!
!   %c_s is an INTEGER variable, which must be set by the user to the
!    value dims%x_e + 1
!
!   %c_e is an INTEGER variable, which must be set by the user to the
!    value dims%x_e + nc
!
!   %c_b is an INTEGER variable, which must be set by the user to the
!    value dims%c_e - m
!
!   %y_s is an INTEGER variable, which must be set by the user to the
!    value dims%c_e + 1
!
!   %y_i is an INTEGER variable, which must be set by the user to the
!    value dims%c_s + m
!
!   %y_e is an INTEGER variable, which must be set by the user to the
!    value dims%c_e + m
!
!   %v_e is an INTEGER variable, which must be set by the user to the
!    value dims%y_e
!
!  f is a REAL variable, which must be set by the user to the value of
!   the constant term f in the objective function.
!   This argument is not altered by the subroutine
!
!  rho_g and rho_b are REAL variables, which must be set by the
!   user to the values of the penalty parameters, rho_g and rho_b.
!
!  G is a REAL array of length n, which must be set by
!   the user to the value of the gradient, g, of the linear term of the
!   quadratic objective function. The i-th component of G, i = 1, ....,
!   n, should contain the value of g_i.  The contents of this argument
!   are not altered by the subroutine
!
!  A_* is used to hold the matrix A by rows. In particular:
!      A_col( : )   the column indices of the components of A
!      A_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row.
!      A_val( : )   the values of the components of A
!
!  H_* is used to hold the LOWER TRIANGLULAR PART of H by rows. In particular:
!      H_col( : )   the column indices of the components of H
!      H_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row.
!      H_val( : )   the values of the components of H
!
!   NB. Each off-diagonal pair of nonzeros should be represented
!   by a single component of H.
!
!  X_l, X_u are REAL arrays of length n, which must be set by the user to the
!   values of the arrays x_l and x_u of lower and upper bounds on x. Any
!   bound X_l( i ) or X_u( i ) larger than or equal to biginf in absolute value
!   will be regarded as being infinite (see the entry control%biginf).
!   Thus, an infinite lower bound may be specified by setting the appropriate
!   component of X_l to a value smaller than -biginf, while an infinite
!   upper bound can be specified by setting the appropriate element of X_u
!   to a value larger than biginf. If X_u( i ) < X_l( i ), X_u( i ) will be
!   reset to X_l( i ). Otherwise, the contents of these arguments are not
!   altered by the subroutine
!
!  C_l, C_u are  REAL array of length m, which must be set by the user to the
!  values of the arrays bl and bu of lower and upper bounds on A x.
!   Any bound bl_i or bu_i larger than or equal to biginf in absolute value
!   will be regarded as being infinite (see the entry control%biginf).
!   Thus, an infinite lower bound may be specified by setting the appropriate
!   component of C_u to a value smaller than -biginf, while an infinite
!   upper bound can be specified by setting the appropriate element of BU
!   to a value larger than biginf. If C_u( i ) < C_l( i ), C_u( i ) will be
!   reset to C_u( i ). Otherwise, the contents of these arguments are not
!   altered by the subroutine
!
!  X is a REAL array of length n, which must be set by
!   the user on entry to QPA_solve to give an initial estimate of the
!   optimization parameters, x. The i-th component of X should contain
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need
!   not satisfy the simple bound constraints and may be perturbed by
!   QPA_solve prior to the start of the minimization.  On exit from
!   QPA_solve, X will contain the best estimate of the optimization
!   parameters found
!
!  Y is a REAL array of length m, which need not be set on entry.
!   On exit, the i-th component of Y contains the best estimate of the
!   the Lagrange multiplier connected to constraint i.
!
!  Z is a REAL array of length n, which need not be set on entry.
!   On exit, the i-th component of Z contains the best estimate of the
!   the Dual variable connected to simple bound constraint i.
!
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPA_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%warm_start must be set .TRUE. on entry; C_stat
!   need not be set if control%warm_start is .FALSE. . On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   C_stat( i ) < 0, the i-th constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPA_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%warm_start must be set .TRUE. on entry; B_stat
!   need not be set if control%warm_start is .FALSE. . On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   B_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  control is a structure of type QPA_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QPA_initialize. See QPA_initialize
!   for details
!
!  inform is a structure of type QPA_inform_type that provides
!    information on exit from QPA_solve. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!     1 The objective function is unbounded below along the line
!       starting at X and pointing in the direction ??
!
!     2 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!     3 one of the restrictions
!          n     >=  1
!          m     >=  0
!       has been violated.
!
!     4 The step is too small to make further impact.
!
!     5 The Newton residuals are larger than the current measure of
!       optimality so further progress is unlikely.
!
!     6 The Hessian matrix is non-convex in the manifold defined by
!       the linear constraints.
!
!    < 0 An allocation error occured
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, m_link, k_n_max, lbreak, pass, n_pcg
      INTEGER, INTENT( INOUT ) :: start_print, stop_print, prec_hist
      REAL ( KIND = wp ), INTENT( IN ) :: f, feas_tol
      REAL ( KIND = wp ), INTENT( IN ) :: best_infeas_g, best_infeas_b
      REAL ( KIND = wp ), INTENT( INOUT ) :: rho_g, rho_b
      REAL, INTENT( IN ) :: time_start
      REAL ( KIND = wp ), INTENT( IN ) :: clock_start
      LOGICAL, INTENT( IN ) :: warm_start
      LOGICAL, INTENT( INOUT ) :: auto_prec, auto_fact
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
      INTEGER, INTENT( INOUT ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( OUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( OUT ),                                                  &
               DIMENSION( dims%c_u_start : dims%c_l_end ) :: C_up_or_low
      INTEGER, INTENT( OUT ),                                                  &
               DIMENSION( dims%x_u_start : dims%x_l_end ) :: X_up_or_low
      INTEGER, INTENT( OUT ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row, Abycol_ptr
      INTEGER, ALLOCATABLE, DIMENSION( : )  :: S_row, S_col, S_colptr
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: P
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( :, : ) :: D
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_norms
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lbreak ) :: BREAKP
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: H_s, GRAD
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( k_n_max ) :: VECTOR
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m + n ) :: PERT
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( k_n_max + control%max_sc ) :: RHS, S, B,  &
                                     RES, S_perm, DX, RES_print
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, k_n_max ) :: DIAG
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( n_pcg ) :: R_pcg, X_pcg, P_pcg
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Abycol_val, S_val
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

      REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( n ) :: G_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: Z_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: Y_p

!  Parameters

! ===========================================================================
!
!  Constraints: 1, .., m are general constraints
!               m+1, .., m+n might be simple bounds on variables
!
!  Reference constraints: those constraints which are active on refactorization
!  Other constraints: those constraints which are not reference constraints
!
!                                  Constraint:
!  C_stat( i ) = j   j > 0         active other pointing to SC(j)
!                    j = 0         inactive other
!                    j < 0         reference pointing to REF
!
!  B_stat( i ) = j   j > 0         active other pointing to SC(j)
!                    j = 0         inactive other
!                    j < 0         reference pointing to REF
!
!  SC( j ) = i   i in (0,m]        active other pointing to C_stat(i)
!                i in (m,m+n]      active other pointing to B_stat(i-m)
!                i in [-m,0)       inactive other pointing to C_stat(-i)
!                i in [-m-n,-m)    inactive other pointing to B_stat(-i-m)
!                i = 0             artificial row (dead inactive reference)
!
!  REF( j ) = i  i in (0,m]        active reference pointing to C_stat(i)
!                i in (m,m+n]      active reference pointing to B_stat(i-m)
!                i < 0             inactive reference pointing to SC
!
! In other words:
!
!  Active other:       C_stat +            => SC +  => C_stat
!                      B_stat +            => SC ++ => B_stat
!  Active reference:   C_stat -  => REF +           => C_stat
!                      B_stat -  => REF ++          => B_stat
!  Inactive reference: C_stat -  => REF -  => SC -  => C_stat
!                      B_stat -  => REF -  => SC -- => B_stat
!
! ===========================================================================

!  Local variables

      INTEGER :: i, ii, j, jj, ll, scu_status, out, error, iii
      INTEGER :: m_active, linesearch_inform, j_min, itref_max, n_all
      INTEGER :: iter, pcount, j_add, j_del, active, pcg_iter, pcg_status
      INTEGER :: cg_maxit, print_level, max_col, s_plus, s_minus, dof, inactive
      INTEGER :: factor, precon, nsemib, icount, QPA_delete_constraint_status
      INTEGER :: jumpto_factorize_reference, QPA_add_constraint_status
      REAL ( KIND = wp ) :: g_s, s_hs, t_opt, mult, hmax, G_perturb
      REAL ( KIND = wp ) :: last_infeas_g, last_infeas_b
      REAL ( KIND = wp ) :: a_x, mult_min, mult_max, ats, mult_zero, res_max
      REAL ( KIND = wp ) :: inner_stop_absolute, inner_stop_relative
      REAL ( KIND = wp ) :: s_norm, y_norm, too_small_a_s
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      LOGICAL :: set_printt, set_printi, set_printm, set_printd, set_printe
      LOGICAL :: printt, printi, printm, printd, printe, warmer_start
      LOGICAL :: stop_now, negative_curvature, G_eq_H, minor_start
      LOGICAL :: x_r0, y0, x_f0, z0, still_adding
      LOGICAL :: move_infeasible, move_infeas, check_dependent
      CHARACTER ( LEN = 1 ) :: mo
      CHARACTER ( LEN = 10 ) :: addel
      CHARACTER ( LEN = 12 ) :: sc_data
      TYPE ( QPA_partition_type ) :: K_part

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) n, m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) f
        WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) G( : n )
          WRITE( control%out, "( ' H (row-wise) = ' )" )
          DO i = 1, n
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, H_col( j ), H_val( j ),                                     &
                j = H_ptr( i ), H_ptr( i + 1 ) - 1 )
          END DO
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" ) X_l( : n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" ) X_u( : n )
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, A_col( j ), A_val( j ),                                     &
                j = A_ptr( i ), A_ptr( i + 1 ) - 1 )
          END DO
        WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" ) C_l( : m )
        WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" ) C_u( : m )
      END IF

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
       WRITE( control%out, "( A, ' entering QPA_solve_main ' )" ) prefix

      pcount = 0 ; print_level = 0

!  ===========================
!  Control the output printing
!  ===========================

      IF ( control%start_print < 0 ) THEN
        start_print = 0
      ELSE
        start_print = control%start_print
      END IF

      IF ( control%stop_print < 0 ) THEN
        stop_print = control%maxit
      ELSE
        stop_print = control%stop_print
      END IF

      out = control%out ; error = control%error
      set_printe = error > 0 .AND. control%print_level >= 1

!  Initialize counts

      iter = 0
      inform%factorization_integer = 0 ; inform%factorization_real = 0

!  Basic single line of output per iteration

      set_printi = out > 0 .AND. control%print_level >= 1

!  As per printi, but with additional timings for various operations

      set_printt = out > 0 .AND. control%print_level >= 2

!  As per printm, but with checking of residuals, etc

      set_printm = out > 0 .AND. control%print_level >= 3

!  Full debugging printing with significant arrays printed

      set_printd = out > 0 .AND. control%print_level >= 4

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printm = set_printm ; printd = set_printd
        print_level = control%print_level
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printm = .FALSE. ; printd = .FALSE.
        print_level = 0
      END IF

!  Ensure that precon has a reasonable value

      precon = control%precon
      IF ( precon >= 6 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A, ' precon = ', I0, ' out of range [0,5]. Reset to 3') ")        &
            prefix, precon
        precon = 3
      END IF

!  Do the same for factor

      factor = control%factor
      IF ( factor < 0 .OR. factor > 2 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A, ' factor = ', I0, ' out of range [0,2]. Reset to 2') ")        &
            prefix, precon
        factor = 2
      END IF

      nsemib = control%nsemib

!  Compute the initial gradient

      PERT = zero
      GRAD = G

      DO i = 1, n
        DO ii = H_ptr( i ), H_ptr( i + 1 ) - 1
          j = H_col( ii )
          GRAD( i ) = GRAD( i ) + H_val( ii ) * X( j )
          IF ( i /= j ) GRAD( j ) = GRAD( j ) + H_val( ii ) * X( i )
        END DO
      END DO
      inform%obj = f + half * DOT_PRODUCT( GRAD + G, X )
      max_col = control%max_col
      IF ( max_col < 0 ) max_col = n

!  Determine which constraints occur in the reference set

      warmer_start = warm_start
      K_part%m_ref = 0

!  general equalities

      DO i = 1, dims%c_equality
        IF ( C_stat( i ) /= 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          C_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = i
        END IF
      END DO

!  general inequalities bounded from below

      DO i = dims%c_equality + 1, dims%c_u_start - 1
        IF ( C_stat( i ) < 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          C_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = i
        ELSE
          C_stat( i ) = 0
        END IF
      END DO

!  general inequalities bounded both from below and above

      DO i = dims%c_u_start, dims%c_l_end
        C_up_or_low( i ) = C_stat( i )
        IF ( C_stat( i ) /= 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          C_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = i
        END IF
      END DO

!  general inequalities bounded from above

      DO i = dims%c_l_end + 1, m
        IF ( C_stat( i ) > 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          C_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = i
        ELSE
          C_stat( i ) = 0
        END IF
      END DO
      K_part%c_ref = K_part%m_ref

!  simple non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        ii = m + i
        IF ( B_stat( i ) < 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          B_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = ii
        END IF
      END DO

!  simple bound from below

      DO i = dims%x_l_start, dims%x_u_start - 1
        ii = m + i
        IF ( B_stat( i ) < 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          B_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = ii
        END IF
      END DO

!  simple bound from below and above

      DO i = dims%x_u_start, dims%x_l_end
        ii = m + i
        X_up_or_low( i ) = B_stat( i )
        IF ( B_stat( i ) /= 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          B_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = ii
        END IF
      END DO

!  simple bound from above

      DO i = dims%x_l_end + 1, dims%x_u_end
        ii = m + i
        IF ( B_stat( i ) > 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          B_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = ii
        END IF
      END DO

!  simple non-positivity

      DO i = dims%x_u_end + 1, n
        ii = m + i
        IF ( B_stat( i ) > 0 ) THEN
          K_part%m_ref = K_part%m_ref + 1
          B_stat( i ) = - K_part%m_ref
          REF( K_part%m_ref ) = ii
        END IF
      END DO

!     WRITE( out, "( ' c_stat ', /, ( 10I5 ) )" ) C_stat( : m )
!     WRITE( out, "( ' b_stat ', /, ( 10I5 ) )" ) B_stat( : n )
!     WRITE( out, "( ' ref ', /, ( 10I5 ) )" ) REF( : K_part%m_ref )

      IF ( printd ) THEN
        IF ( K_part%c_ref > 0 )                                                &
          WRITE( out, "( ' REF(c) ', /, ( 10I5 ) )" ) REF( : K_part%c_ref )
        IF ( K_part%m_ref >= K_part%c_ref )                                    &
          WRITE( out, "( ' REF(b) ', /, ( 10I5 ) )" )                          &
            REF( K_part%c_ref + 1 : K_part%m_ref ) - m
      END IF

! Set up the initial reference matrix

      K_part%k_ref = n + K_part%m_ref ; SCU_mat%n = K_part%k_ref

!  ------------------------------------------------------------------------
!                      Start of Major iteration
!  ------------------------------------------------------------------------

      mult_zero = - control%multiplier_tol
!     tiny_cosine = epsmch ** 0.75
      G_perturb = zero
      addel = '          '

      SLS_control%relative_pivot_tolerance =                                   &
        control%SLS_control%relative_pivot_tolerance

      still_adding = .TRUE.
      stop_now = .FALSE.
      check_dependent = .FALSE.
      mult_max = zero
      t_opt = zero
      icount = 0 ; last_infeas_g = best_infeas_g ; last_infeas_b = best_infeas_b

      major : DO

!  Check to see if the iteration limit has been exceeded

        IF ( inform%iter > control%maxit ) THEN
          inform%status = GALAHAD_error_max_iterations
          EXIT major
        END IF

        inform%major_iter = inform%major_iter + 1

!  Check that the CPU time limit has not been reached

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - time_start > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit ; EXIT major
        END IF

!  Set automatic choices for the factorizations and preconditioner

        dof = n - K_part%m_ref
        IF ( auto_fact ) factor = 0
        IF ( auto_prec ) THEN
          IF ( dof <= 1 ) THEN
            precon = 3
            nsemib = 0
          ELSE IF ( prec_hist == 1 ) THEN
            precon = 1
            nsemib = control%nsemib
          ELSE IF ( prec_hist == 2 ) THEN
            precon = 3
            nsemib = control%nsemib
          ELSE
            precon = 3
            nsemib = 0
          END IF
        END IF

        IF ( printi )                                                          &
         WRITE( out, "( /, A, 18X, ' ======================================',  &
       &                /, A, 18X, '      Start of Major iteration', I5,       &
       &                /, A, 18X, '               (pass', I5, ')',            &
       &                /, A, 18X, '   general pen. parameter =', ES11.4,      &
       &                /, A, 18X, '     bound pen. parameter =', ES11.4,      &
       &                /, A, 18X, '       maximum multiplier =', ES11.4,      &
       &                /, A, 18X, ' ======================================')")&
       &        prefix, prefix, inform%major_iter, prefix, pass,               &
                prefix, rho_g, prefix, rho_b, prefix, mult_max, prefix

!  Form and factorize the reference matrix

        jumpto_factorize_reference = 0
  10    CONTINUE

        CALL QPA_factorize_reference(                                          &
                    dims, n, m, jumpto_factorize_reference,                    &
                    k_n_max, print_level, m_link, max_col, factor, precon,     &
                    nsemib, hmax, G_perturb, out, prec_hist, printi, printt,   &
                    printe, G_eq_H, auto_prec, auto_fact,                      &
                    check_dependent, mo, PERM, REF, C_stat, B_stat,            &
                    A_ptr, A_col, H_ptr, H_col, A_val, H_val, K, K_part, S,    &
                    Abycol_row, Abycol_ptr, S_row, S_col, S_colptr,            &
                    Abycol_val, S_val, DIAG, SLS_data, SLS_control,            &
                    prefix, control, inform )

        IF ( jumpto_factorize_reference == 1 ) THEN
          RETURN
        ELSE IF ( jumpto_factorize_reference == 2 ) THEN
          GO TO 410
        END IF

! Initialize the Schur complement

        SCU_mat%m = 0
        SCU_mat%class = 2 ; SCU_mat%m_max = control%max_sc
        SCU_mat%BD_col_start( 1 ) = 1
        scu_status = 1
        CALL SCU_factorize( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),          &
                            scu_status, SCU_info )
        m_active = K_part%m_ref
        s_minus = 0 ; s_plus = 0
        j_del = 0 ; j_add = 0

!  Prepare for minor iteration

        inner_stop_absolute = control%inner_stop_absolute
        inner_stop_relative = control%inner_stop_relative
        minor_start = .TRUE.

        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' There ', A, 1X, I0, ' degree', A, ' of freedom' )" )     &
            prefix, TRIM( STRING_are( dof ) ), dof,                            &
            TRIM( STRING_pleural( dof ) )
        itref_max = control%itref_max

!  ------------------------------------------------------------------------
!                      Start of Minor iteration
!  ------------------------------------------------------------------------

        minor : DO

          IF ( warmer_start ) THEN
            IF ( printi ) WRITE( out, "( /, A, 31X, 'Warm start' )" ) prefix
          ELSE
            IF ( iter == 0 .AND. printi )                                      &
              WRITE( out, "( /, 31X, 'Cold start' )" )
          END IF

          dof = n - m_active
          IF ( printt ) WRITE( out, "( /, A, ' dof = ', I0 )" ) prefix, dof
          IF ( dof < 0 ) WRITE( out, "( A, ' dof = ', I0, ' iter = ', I0 )" )  &
            prefix, dof, iter
          IF ( iter == 0 ) THEN

            inform%num_g_infeas =                                              &
              COUNT( ABS( RES_l( : dims%c_equality ) ) > feas_tol ) +          &
              COUNT( RES_l( dims%c_equality + 1 : ) < - feas_tol ) +           &
              COUNT( RES_u < - feas_tol )
            inform%num_b_infeas =                                              &
              COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < - feas_tol ) +&
              COUNT( X( dims%x_l_start : dims%x_l_end )                        &
                     - X_l( dims%x_l_start : dims%x_l_end ) < - feas_tol ) +   &
              COUNT( X_u( dims%x_u_start : dims%x_u_end )                      &
                     - X( dims%x_u_start : dims%x_u_end ) < - feas_tol ) +     &
              COUNT( X( dims%x_u_end + 1 : n ) > feas_tol )

            inform%infeas_g = SUM( ABS( RES_l( : dims%c_equality ) ) ) -       &
              SUM( MIN( RES_l( dims%c_equality + 1 : ), zero ) ) -             &
              SUM( MIN( RES_u, zero ) )
            inform%infeas_b = -                                                &
              SUM( MIN( X( dims%x_free + 1 : dims%x_l_start - 1 ), zero ) ) -  &
              SUM( MIN( X( dims%x_l_start : dims%x_l_end )                     &
                        - X_l( dims%x_l_start : dims%x_l_end ), zero ) ) -     &
              SUM( MIN( X_u( dims%x_u_start : dims%x_u_end )                   &
                        - X( dims%x_u_start : dims%x_u_end ), zero ) ) -       &
              SUM( MIN( - X( dims%x_u_end + 1 : n ), zero ) )

            inform%merit = inform%obj + rho_g * inform%infeas_g                &
                                      + rho_b * inform%infeas_b

            IF ( printt .OR. ( printi .AND. pcount == 0 ) )                    &
              WRITE( out, 2000 ) prefix, precon, prefix
            IF ( printi ) THEN
              CALL CLOCK_TIME( clock_now )
              WRITE( out, 2060 ) prefix, inform%iter, inform%merit,            &
                STRING_exponent( inform%infeas_g + inform%infeas_b ),          &
                inform%num_g_infeas + inform%num_b_infeas,                     &
                STRING_real_7( clock_now - clock_start )
            END IF
            pcount = MOD( pcount + 1, 50 )
          END IF

!  Check to see if the iteration limit has been exceeded

!         WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" ) X
!         WRITE( out, "( ' lower residuals ', /, ( 5ES12.4 ) )" ) RES_l
!         WRITE( out, "( ' upper residuals ', /, ( 5ES12.4 ) )" ) RES_u
!         WRITE( out, "( ' ref ', /, ( 10I5 ) )" ) REF( : K_part%m_ref )
!         WRITE( out, "( ' sc ', /, ( 10I5 ) )" ) SC( : SCU_mat%m )

          inform%iter = inform%iter + 1 ; iter = iter + 1

!  Control printing

          IF ( inform%iter == start_print ) THEN
            pcount = 0
            printe = set_printe ; printi = set_printi ; printt = set_printt
            printm = set_printm ; printd = set_printd
            print_level = control%print_level
          END IF

          IF ( inform%iter == stop_print + 1 ) THEN
            printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
            printm = .FALSE. ; printd = .FALSE.
            print_level = 0
          END IF

          IF ( inform%iter > control%maxit + 1 ) THEN
            inform%status = GALAHAD_error_max_iterations ; EXIT major
          END IF

!  Check that the CPU time limit has not been reached

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - time_start > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - clock_start > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit ; EXIT major
          END IF

!  Check to see if the infeasibility warrants further action

          icount = MOD( icount + 1, control%infeas_check_interval )
!         write(6,"( i5 )" ) icount
          IF ( icount == control%infeas_check_interval - 1 ) THEN

            IF ( control%solve_qp ) THEN
!             write(6,"( ' last, now ', 2ES12.4)") inform%infeas_g,last_infeas_g
              IF ( inform%num_g_infeas > 0 .AND.                               &
                   inform%infeas_g > MAX( feas_tol ** 0.5,                     &
                   control%infeas_g_improved_by_factor * last_infeas_g ) ) THEN
                rho_g = control%increase_rho_g_factor * rho_g
                IF ( printi ) WRITE( out, "( /, 5X, '... little improvement',  &
               &  ' in infeasibility over the previous', I5, ' iterations', /, &
               &  8X, ' so increasing rho_g to ', ES12.4, / )")                &
                control%infeas_check_interval, rho_g
              END IF
              last_infeas_g = inform%infeas_g
            END IF

            IF ( control%solve_qp .OR.  control%solve_within_bounds ) THEN
              IF ( inform%num_b_infeas > 0 .AND.                               &
                   inform%infeas_b > MAX( feas_tol ** 0.5,                     &
                   control%infeas_b_improved_by_factor * last_infeas_b ) ) THEN
                rho_b = control%increase_rho_b_factor * rho_b
                IF ( printi ) WRITE( out, "( /, 5X, '... little improvement',  &
               &  ' in infeasibility over the previous', I5, ' iterations', /, &
               &  8X, ' so increasing rho_b to ', ES12.4, / )")                &
                control%infeas_check_interval, rho_b
              END IF
              last_infeas_b = inform%infeas_b
            END IF

          END IF

          IF ( printm )                                                        &
            WRITE ( out, "( /, A, ' =============================== iter ',    &
           &   I0, ' =============================== ' )" ) prefix, inform%iter

!  Warm start
!  ==========

!  Ensure that the initial point satisfies the working set of constraints

          IF ( warmer_start ) THEN
!           write(6,"('============ warmer start !!! ')" )
            warmer_start = .FALSE.

!  Since this is the first iteration, solve the current EQP

            IF ( inform%iter <= 1 ) THEN
              S( : n ) = - G
              DO i = 1, K_part%m_ref
                j =  REF( i )
                IF ( j <= m ) THEN
                  IF ( j < dims%c_u_start ) THEN
                    S( n + i ) = C_l( j )
                  ELSE IF ( j > dims%c_l_end ) THEN
                    S( n + i ) = C_u( j )
                  ELSE IF ( j >= dims%c_u_start ) THEN
                    IF ( C_up_or_low( j ) == 1 ) THEN
                      S( n + i ) = C_u( j )
                    ELSE
                      S( n + i ) = C_l( j )
                    END IF
                  END IF
                ELSE
                  j = j - m
                  IF ( j < dims%x_l_start ) THEN
                    S( n + i ) = zero
                  ELSE IF ( j < dims%x_u_start ) THEN
                    S( n + i ) = X_l( j )
                  ELSE IF ( j > dims%x_u_end ) THEN
                    S( n + i ) = zero
                  ELSE IF ( j > dims%x_l_end ) THEN
                    S( n + i ) = X_u( j )
                  ELSE IF ( j >= dims%x_u_start ) THEN
                    IF ( X_up_or_low( j ) == 1 ) THEN
                      S( n + i ) = X_u( j )
                    ELSE
                      S( n + i ) = X_l( j )
                    END IF
                  END IF
                END IF
              END DO
!             write( 6, "( I6, ES12.4 )" ) ( i, S( i ), i = 1, K_part%k_ref )

!  Find a point that satisfies the constraints

              S_perm( PERM( : K_part%k_ref ) ) = S( : K_part%k_ref )
              x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.
              CALL QPA_ir( K, SLS_data, K_part, S_perm( : K_part%k_ref ),      &
                           B( : K_part%k_ref ), RES( : K_part%k_ref ),         &
                           x_r0, y0, x_f0, z0, SLS_control, itref_max + 1,     &
                           out, printm, RES_print, inform )
              inform%factorization_status = inform%status
              IF ( printm ) WRITE( out, "( ' ' )" )

              S( : K_part%k_ref ) = S_perm( PERM( : K_part%k_ref ) )
              X = S( : n )

!  From this point, try to minimize the quadratic on the current working set

              IF ( .NOT. G_eq_H ) THEN
                inner_stop_relative = zero
                inner_stop_absolute = SQRT( EPSILON( one ) )
                cg_maxit = control%cg_maxit
                X_pcg = X
                RHS( : n ) = - G
                iii = itref_max + 1
                CALL QPA_pcg( dims, n, S, RHS, R_pcg, X_pcg, P_pcg, S_perm,    &
                               RES, B, DX, VECTOR, PERM, SCU_mat, SCU_data,    &
                               K, K_part, H_val, H_col, H_ptr, prefix, control,&
                               SLS_control, SLS_data, print_level - 1,         &
                               cg_maxit, dof, .TRUE., inner_stop_absolute,     &
                               inner_stop_relative, iii, inform,               &
                               pcg_iter, negative_curvature, pcg_status,       &
                               RES_print( : K_part%n_free + K_part%c_ref ) )
                itref_max = iii - 1

                IF ( pcg_status < 0 ) THEN
                  IF ( printt ) WRITE( out, "( /, A,                           &
                 &  ' Warning return from QPA_pcg, status = ', I0 )")          &
                    prefix, pcg_status
                  IF ( pcg_status < - 2 ) THEN

!  If a negative inner product has occured on the first minor iteration,
!  modify the preconditioner and restart

                    IF ( pcg_status == - 3 .AND. minor_start ) THEN
                      IF ( mo == ' ' ) THEN
                        inform%nmods = inform%nmods + 1
                        mo = 'm'
                        K%ne = K_part%k_free_p
                      END IF
                      G_perturb = G_perturb + hmax
                      K%val( K_part%k_free_od + 1 : K_part%k_free_p )          &
                        = K%val( K_part%k_free_od + 1 : K_part%k_free_p ) + hmax
                      IF ( printi ) WRITE( out, 2100 ) prefix, prefix,         &
                        G_perturb, SLS_control%relative_pivot_tolerance
                      G_eq_H = .FALSE.
                      SLS_control%relative_pivot_tolerance = MIN( half,        &
                         ten * SLS_control%relative_pivot_tolerance )
                      pcount = 0 ; minor_start = .FALSE.
                      IF ( K_part%k_free_p == K_part%k_free_d ) THEN
                        jumpto_factorize_reference = 2
                        GO TO 10
                      ELSE
                        jumpto_factorize_reference = 1
                        GO TO 10
                      END IF
                    END IF
                    warmer_start = .TRUE.
                    EXIT minor
                  END IF
                END IF
                IF ( pcg_status == 0 .OR. pcg_status == 1 ) X = S( : n )
                inner_stop_absolute = control%inner_stop_absolute
                inner_stop_relative = control%inner_stop_relative
              END IF

!  Since this is not the first iteration, simply try to move back onto the
!  current working set

            ELSE
              DO i = 1, K_part%m_ref
                j = REF( i )
                IF ( j > 0 ) THEN
                  IF ( j <= m ) THEN
                    IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                      IF ( C_up_or_low( j ) == - 1 ) THEN
                        a_x = - C_l( j )
                      ELSE
                        a_x = - C_u( j )
                      END IF
                    ELSE IF ( j > dims%c_l_end ) THEN
                      a_x = - C_u( j )
                    ELSE
                      a_x = - C_l( j )
                    END IF
                    ats = ABS( a_x )
                    DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                      a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                      ats = ats + ABS( A_val( ii ) * X( A_col( ii ) ) )
                    END DO
                    IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )            &
                      WRITE( out, "( I7, 'c is an active reference, residual,',&
                     &               ' size = ',  2ES12.4 )" ) j, a_x, ats
                  ELSE
                    jj = j - m
                    IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                      IF ( X_up_or_low( jj ) == - 1 ) THEN
                        a_x = X( jj ) - X_l( jj )
                        ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
                      ELSE
                        a_x = X( jj ) - X_u( jj )
                        ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
                      END IF
                    ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                      a_x = X( jj ) - X_u( jj )
                      ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
                    ELSE IF ( jj < dims%x_u_start .AND.                        &
                              jj >= dims%x_l_start ) THEN
                      a_x = X( jj ) - X_l( jj )
                      ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
                    ELSE
                      a_x = X( jj )
                      ats = ABS( X( jj ) )
                    END IF
                    IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )            &
                      WRITE( out, "( I7, 'b is an active reference, residual,',&
                     &               ' size = ', 2ES12.4 )" ) jj, a_x, ats
                  END IF
                  S( n + i ) = - a_x
                ELSE
                  S( n + i ) = zero
                END IF
              END DO

              S( : n ) = zero

!  Find a point that satisfies the constraints

              n_all = SCU_mat%n + SCU_mat%m

              S_perm( PERM( : n_all ) ) = S( : n_all )
              iii = itref_max
              CALL QPA_iterative_refinement(                                   &
                                  K, SCU_mat, SCU_data, S_perm( : n_all ),     &
                                  RES( : n_all ), B( : n_all ), S( : n_all ),  &
                                  DX( : n_all ), VECTOR( : SCU_mat%n ),        &
                                  SLS_control, SLS_data, K_part, .FALSE.,      &
                                  iii, out, printm, RES_print, inform )

              S( : n_all ) = RES( PERM( : n_all ) )

              X = X + S( : n )

            END IF

!  Compute the initial residuals

            DO i = 1, dims%c_u_start - 1
              a_x = zero
              DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                a_x = a_x + A_val( ii ) * X( A_col( ii ) )
              END DO
              RES_l( i ) = a_x - C_l( i )
            END DO

            DO i = dims%c_u_start, dims%c_l_end
              a_x = zero
              DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                a_x = a_x + A_val( ii ) * X( A_col( ii ) )
              END DO
              RES_l( i ) = a_x - C_l( i )
              RES_u( i ) = C_u( i ) - a_x
            END DO

            DO i = dims%c_l_end + 1, m
              a_x = zero
              DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                a_x = a_x + A_val( ii ) * X( A_col( ii ) )
              END DO
              RES_u( i ) = C_u( i ) - a_x
            END DO

            res_max = zero

!  reference constraints

            DO i = 1, K_part%m_ref
              j = REF( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                END IF
                res_max = MAX( res_max, ABS( a_x ) )
              END IF

            END DO

!  other constraints

            DO i = 1, SCU_mat%m
              j = SC( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                END IF
                res_max = MAX( res_max, ABS( a_x ) )
              END IF
            END DO

            IF ( printm ) WRITE( out, "( ' maximum residual of',              &
           &     ' working constituents is ', ES10.2 )" ) res_max

!  Compute the initial gradient

            GRAD = G
            DO i = 1, n
              DO ii = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( ii )
                GRAD( i ) = GRAD( i ) + H_val( ii ) * X( j )
                IF ( i /= j ) GRAD( j ) = GRAD( j ) + H_val( ii ) * X( i )
              END DO
            END DO
            inform%obj = f + half * DOT_PRODUCT( GRAD + G, X )

            linesearch_inform = 0

            inform%num_g_infeas =                                              &
              COUNT( ABS( RES_l( : dims%c_equality ) ) > feas_tol ) +          &
              COUNT( RES_l( dims%c_equality + 1 : ) < - feas_tol ) +           &
              COUNT( RES_u < - feas_tol )
            inform%num_b_infeas =                                              &
              COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < - feas_tol ) +&
              COUNT( X( dims%x_l_start : dims%x_l_end )                        &
                     - X_l( dims%x_l_start : dims%x_l_end ) < - feas_tol ) +   &
              COUNT( X_u( dims%x_u_start : dims%x_u_end )                      &
                     - X( dims%x_u_start : dims%x_u_end ) < - feas_tol ) +     &
              COUNT( X( dims%x_u_end + 1 : n ) > feas_tol )

            inform%infeas_g = SUM( ABS( RES_l( : dims%c_equality ) ) ) -       &
              SUM( MIN( RES_l( dims%c_equality + 1 : ), zero ) ) -             &
              SUM( MIN( RES_u, zero ) )
            inform%infeas_b = -                                                &
              SUM( MIN( X( dims%x_free + 1 : dims%x_l_start - 1 ), zero ) ) -  &
              SUM( MIN( X( dims%x_l_start : dims%x_l_end )                     &
                        - X_l( dims%x_l_start : dims%x_l_end ), zero ) ) -     &
              SUM( MIN( X_u( dims%x_u_start : dims%x_u_end )                   &
                        - X( dims%x_u_start : dims%x_u_end ), zero ) ) -       &
              SUM( MIN( - X( dims%x_u_end + 1 : n ), zero ) )

            inform%merit = inform%obj + rho_g * inform%infeas_g                &
                                      + rho_b * inform%infeas_b

            IF ( printt .OR. ( printi .AND. pcount == 0 ) )                    &
               WRITE( out, 2000 ) prefix, precon, prefix
            IF ( printi ) THEN
              CALL CLOCK_TIME( clock_now )
              IF ( G_eq_H .OR. inform%iter > 1 ) THEN
                WRITE( out, 2060 ) prefix, inform%iter, inform%merit,          &
                  STRING_exponent( inform%infeas_g + inform%infeas_b ),        &
                  inform%num_g_infeas + inform%num_b_infeas,                   &
                  STRING_real_7( clock_now - clock_start )
              ELSE
                WRITE( out, 2061 ) prefix, inform%iter, inform%merit,          &
                  STRING_exponent( inform%infeas_g + inform%infeas_b ),        &
                  inform%num_g_infeas + inform%num_b_infeas, pcg_iter,         &
                  STRING_real_7( clock_now - clock_start )
              END IF
            END IF
            pcount = MOD( pcount + 1, 50 )
            addel = '          '
            CYCLE

          ELSE

!  -=-=-=-=-=-=-=-=-=-=-=-=-
!  Find the search direction
!  -=-=-=-=-=-=-=-=-=-=-=-=-

  200       CONTINUE
            n_all = SCU_mat%n + SCU_mat%m

!  Set up the right-hand side

            RHS( : n ) = - GRAD
!           RHS( n + 1 : n_all ) = zero

!  equality constraints

            DO i = 1, dims%c_equality
              IF ( i == - j_del ) CYCLE
              ll = C_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( RES_l( i ) < zero ) THEN
                DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                  j = A_col( ii )
                  RHS( j ) = RHS( j ) + rho_g * A_val( ii )
                END DO
              ELSE IF ( RES_l( i ) > zero ) THEN
                DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                  j = A_col( ii )
                  RHS( j ) = RHS( j ) - rho_g * A_val( ii )
                END DO
              END IF
            END DO

!  constraints with lower bounds

            DO i = dims%c_equality + 1, dims%c_u_start - 1
              IF ( i == - j_del ) THEN
                IF ( move_infeasible ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) + rho_g * A_val( ii )
                  END DO
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = C_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( RES_l( i ) < zero ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) + rho_g * A_val( ii )
                  END DO
                END IF
              END IF
            END DO

!  constraints with lower and upper bounds

            DO i = dims%c_u_start, dims%c_l_end
              IF ( i == ABS( j_del ) ) THEN
                IF ( move_infeasible ) THEN
                  IF ( j_del < 0 ) THEN
                    DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                      j = A_col( ii )
                      RHS( j ) = RHS( j ) + rho_g * A_val( ii )
                    END DO
                  ELSE
                    DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                      j = A_col( ii )
                      RHS( j ) = RHS( j ) - rho_g * A_val( ii )
                    END DO
                  END IF
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = C_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( RES_l( i ) < zero ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) + rho_g * A_val( ii )
                  END DO
                ELSE IF ( RES_u( i ) < zero ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) - rho_g * A_val( ii )
                  END DO
                END IF
              END IF
            END DO

!  constraints with upper bounds

            DO i = dims%c_l_end + 1, m
              IF ( i == j_del ) THEN
                IF ( move_infeasible ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) - rho_g * A_val( ii )
                  END DO
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = C_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( RES_u( i ) < zero ) THEN
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( ii )
                    RHS( j ) = RHS( j ) - rho_g * A_val( ii )
                  END DO
                END IF
              END IF
            END DO

!  simple non-negativity

            DO i = dims%x_free + 1, dims%x_l_start - 1
              IF ( m + i == - j_del ) THEN
                IF ( move_infeasible ) THEN
                  RHS( i ) = RHS( i ) + rho_b
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = B_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( X( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) + rho_b
                END IF
              END IF
            END DO

!  simple bound from below

            DO i = dims%x_l_start, dims%x_u_start - 1
              IF ( m + i == - j_del ) THEN
                IF ( move_infeasible ) THEN
                  RHS( i ) = RHS( i ) + rho_b
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = B_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( X( i ) - X_l( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) + rho_b
                END IF
              END IF
            END DO

!  simple bound from below and above

            DO i = dims%x_u_start, dims%x_l_end
              IF ( ABS( m + i ) == j_del ) THEN
                IF ( move_infeasible ) THEN
                  IF ( j_del < 0 ) THEN
                    RHS( i ) = RHS( i ) + rho_b
                  ELSE
                    RHS( i ) = RHS( i ) - rho_b
                  END IF
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = B_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( X( i ) - X_l( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) + rho_b
                ELSE IF ( X_u( i ) - X( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) - rho_b
                END IF
              END IF
            END DO

!  simple bound from above

            DO i = dims%x_l_end + 1, dims%x_u_end
              IF ( m + i == j_del ) THEN
                IF ( move_infeasible ) THEN
                  RHS( i ) = RHS( i ) - rho_b
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = B_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( X_u( i ) - X( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) - rho_b
                END IF
              END IF
            END DO

!  simple non-positivity

            DO i = dims%x_u_end + 1, n
              IF ( m + i == j_del ) THEN
                IF ( move_infeasible ) THEN
                  RHS( i ) = RHS( i ) - rho_b
                ELSE
                  CYCLE
                END IF
              ELSE
                ll = B_stat( i )
                IF ( ll > 0 ) CYCLE
                IF ( ll < 0 ) THEN
                  IF ( REF( - ll ) > 0 ) CYCLE
                END IF
                IF ( - X( i ) < zero ) THEN
                  RHS( i ) = RHS( i ) - rho_b
                END IF
              END IF
            END DO

!  Solve for the Newton correction
!  -------------------------------

!  Use a direct method to obtain a KKT point

            IF ( G_eq_H ) THEN

              S_perm( PERM( : n ) ) = RHS( : n )
              S_perm( PERM( n + 1 : n_all ) ) = zero

              iii = itref_max
              CALL QPA_iterative_refinement(                                   &
                                  K, SCU_mat, SCU_data, S_perm( : n_all ),     &
                                  RES( : n_all ), B( : n_all ), S( : n_all ),  &
                                  DX( : n_all ), VECTOR( : SCU_mat%n ),        &
                                  SLS_control, SLS_data, K_part, .TRUE.,       &
                                  iii, out, printm, RES_print, inform )

              S( : n_all ) = RES( PERM( : n_all ) )
              pcg_iter = 0
            ELSE

!  Use an iterative method to minimize the quadratic model

              IF ( printm ) WRITE( out,                                        &
               "(/, '   |--------------------------------------------------|', &
             &   /, '   |        start to solve equality subproblem        |', &
             &   / )" )

              IF ( still_adding ) THEN
                cg_maxit = 1
              ELSE
                cg_maxit = control%cg_maxit
              END IF
              CALL CPU_TIME( time_record ) ; CALL CLOCK_TIME( clock_record )

              iii = itref_max
              CALL QPA_pcg( dims, n, S, RHS, R_pcg, X_pcg, P_pcg, S_perm,      &
                             RES, B, DX, VECTOR, PERM, SCU_mat, SCU_data,      &
                             K, K_part, H_val, H_col, H_ptr, prefix, control,  &
                             SLS_control, SLS_data, print_level - 1,           &
                             cg_maxit, dof, .FALSE.,                           &
                             inner_stop_absolute, inner_stop_relative, iii,    &
                             inform, pcg_iter, negative_curvature, pcg_status, &
                             RES_print( : K_part%n_free + K_part%c_ref ) )

              IF ( pcg_status < 0 ) THEN
                IF ( printt ) WRITE( out, "( /, A,                             &
               &  ' Warning return from QPA_pcg, status = ', I0 )" )           &
                  prefix, pcg_status
                IF ( pcg_status < - 2 ) THEN

!  If a negative inner product has occured on the first minor iteration,
!  modify the preconditioner and restart

                  IF ( pcg_status == - 3 .AND. minor_start ) THEN
                    IF ( mo == ' ' ) THEN
                      inform%nmods = inform%nmods + 1
                      mo = 'm'
                      K%ne = K_part%k_free_p
                    END IF
                    G_perturb = G_perturb + hmax
                    K%val( K_part%k_free_od + 1 : K_part%k_free_p )            &
                      = K%val( K_part%k_free_od + 1 : K_part%k_free_p ) + hmax
                    IF ( printi ) WRITE( out, 2100 ) prefix, prefix,           &
                      G_perturb, SLS_control%relative_pivot_tolerance
                    G_eq_H = .FALSE.
                    SLS_control%relative_pivot_tolerance = MIN( half,          &
                       ten * SLS_control%relative_pivot_tolerance )
                    pcount = 0 ; minor_start = .FALSE.
                    IF ( K_part%k_free_p == K_part%k_free_d ) THEN
                      jumpto_factorize_reference = 2
                      GO TO 10
                    ELSE
                      jumpto_factorize_reference = 1
                      GO TO 10
                    END IF
                  END IF

                  IF ( printt .OR. ( printi .AND. pcount == 0 ) )              &
                    WRITE( out, 2000 ) prefix, precon, prefix
!                 write(6,*) ' minor_start, pcount ', minor_start, pcount

                  IF ( printi ) THEN
                    CALL CLOCK_TIME( clock_now )
                    WRITE( out, 2050 ) prefix, inform%iter, inform%merit,      &
                      t_opt,                                                   &
                      STRING_exponent( inform%infeas_g + inform%infeas_b ),    &
                      inform%num_g_infeas + inform%num_b_infeas, pcg_iter,     &
                      sc_data, addel, STRING_real_7( clock_now - clock_start )
                  END IF
                  warmer_start = .TRUE.
                  EXIT minor
                END IF
              END IF
              inform%cg_iter = inform%cg_iter + pcg_iter

              IF ( printm ) WRITE( out,                                        &
               "(/, '   |           equality subproblem solved             |', &
         &       /, '   |--------------------------------------------------|', &
         &         / )" )

              CALL CPU_TIME( time_now ) ; CALL CLOCK_TIME( clock_now )
              time_now = time_now - time_record
              inform%time%solve = inform%time%solve + time_now
              inform%time%clock_solve =                                        &
                inform%time%clock_solve + clock_now - clock_record
              IF ( printt ) WRITE( out, "( A, ' solve time = ', F10.2 )" )     &
                prefix, time_now
            END IF

! If the search direction is essentially rounding errors, reset it to zero

            s_norm = SQRT( DOT_PRODUCT( S( : n ), S( : n ) ) )
            IF ( printt ) WRITE( out, "( A, ' s vs y ', 2ES12.4 )" ) prefix,   &
               s_norm, SQRT( SUM( S( n + 1 : n_all ) ** 2 ) )
            IF ( s_norm /= zero ) THEN
              y_norm = SQRT( SUM( S( n + 1 : n_all ) ** 2 ) )
              IF ( s_norm <= y_norm * epsmch ** 0.66 ) THEN
                IF ( printt ) WRITE( out, "( ' tiny S = ', ES10.4, ' (',       &
               & 'compared with Y = ', ES10.4, ') has been reset to zero' )" ) &
                  s_norm, y_norm
                S( : n ) = zero
                s_norm = zero
              END IF
            END IF

            IF ( stop_now ) THEN
              WRITE( 6, "( 's,rhs ', /, ( 2ES12.4 ) )" )                       &
                           ( S( i ), RHS( i ), i = 1, n_all )
              STOP
            END IF
            minor_start = .FALSE.

!  Check the residual

            IF ( inform%factorization_status /= 0 ) THEN
              IF ( printt ) WRITE ( out, "( /, A, '  on exit from SCU_solve', &
             &  ', status = ', I0 )" ) prefix, inform%factorization_status
              EXIT minor
            END IF

            IF ( s_norm /= zero ) THEN

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Linesearch along the search direction
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Reset small potential changes in S to zero

              too_small_a_s = epsmch ** 0.5

! Compute A_s

              A_s = zero
              DO i = 1, m
                DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                  A_s( i ) = A_s( i ) + A_val( ii ) * S( A_col( ii ) )
                END DO

!  Reset small potential changes to zero

!               IF ( ABS( A_s( i ) ) < too_small_a_s )                         &
!                 A_s( i ) = zero
              END DO

!  Check that the search direction really lies in N(A)

              IF ( MOD( inform%iter, control%monitor_residuals ) == 0 ) THEN
                res_max = zero
                DO i = 1, K_part%m_ref
                  j = REF( i )
                  IF ( j > 0 ) THEN
                    IF ( j <= m ) THEN
                      a_x = A_s( j )
                    ELSE
                      a_x = S( j - m )
                    END IF
                    res_max = MAX( res_max, ABS( a_x ) )
                  END IF
                END DO
                DO i = 1, SCU_mat%m
                  j = SC( i )
                  IF ( j > 0 ) THEN
                    IF ( j <= m ) THEN
                      a_x = A_s( j )
                    ELSE
                      a_x = S( j - m )
                    END IF
                    res_max = MAX( res_max, ABS( a_x ) )
                  END IF
                END DO

                IF ( res_max > too_small_a_s * MAX( one, s_norm ) ) THEN
                  IF ( printt ) WRITE( out, "( ' working A_s = ', ES12.4,      &
                 &  ' exceeds maximum allowed ', ES12.4  )" )                  &
                    res_max, too_small_a_s
                  itref_max = itref_max + 1
                  IF ( itref_max <= 3 ) THEN
                    GO TO 200
                  ELSE
                    addel = '         '
                    warmer_start = .TRUE.
                    EXIT minor
                  END IF
                ELSE
                  IF ( printm ) WRITE( out, "( ' maximum working A_s = ',      &
                 &                     ES12.4 )" ) res_max
                END IF
              END IF

! Compute H_s

              H_s = zero
              DO i = 1, n
                DO ii = H_ptr( i ), H_ptr( i + 1 ) - 1
                  j = H_col( ii )
                  H_s( i ) = H_s( i ) + H_val( ii ) * S( j )
                  IF ( i /= j ) H_s( j ) = H_s( j ) + H_val( ii ) * S( i )
                END DO
              END DO

              IF ( printd ) write( out, "( ' slope ', ES12.4 )" ) &
                              - DOT_PRODUCT( RHS( : n ), S( : n ) )
              g_s = DOT_PRODUCT( GRAD, S( : n ) )
              s_hs = DOT_PRODUCT( S( : n ), H_s )

!  Perform the linesearch

              CALL QPA_linesearch( dims, n, m, inform%obj, g_s, s_hs, s_norm,  &
                                  rho_g, rho_b, X, X_l, X_u, RES_l, RES_u,     &
                                  S( : n ), A_s, A_norms,                      &
                                  C_stat, B_stat, REF, IBREAK, BREAKP, lbreak, &
                                  m_link, out, printt, printm, printd, t_opt,  &
                                  too_small_a_s, active, linesearch_inform )

!  The problem is unbounded from below

              IF ( linesearch_inform == - 2 ) THEN
                IF ( printi ) WRITE( out,                                      &
                  "( /, ' Problem is unbounded from below: stopping')" )
                inform%status = GALAHAD_error_unbounded
                EXIT major
              END IF

!  If we have truncated the CG early, but not picked up any constraints, make
!  sure that we solve the CG more accurately in future

              IF ( still_adding .AND. linesearch_inform == - 1 ) THEN
                linesearch_inform = 0
              END IF
              still_adding = linesearch_inform > 0

!  Update X and the residuals

              inform%obj = inform%obj + t_opt * ( g_s + half * t_opt * s_hs )

!  Restore s

              X = X + t_opt * S( : n )

              RES_l = RES_l + t_opt * A_s( : dims%c_l_end )
              RES_u = RES_u - t_opt * A_s( dims%c_u_start : m )
              GRAD = GRAD + t_opt * H_s

              IF ( printt ) THEN
                res_max = zero
                IF ( printd ) WRITE( out,                                      &
               &   "( '       <   --------- residuals ----------> ', /,        &
               &      '    i   recurred    actual      difference ' )" )
                DO i = 1, dims%c_u_start - 1
                  a_x = zero
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                  res_max = MAX( res_max, ABS( - RES_l( i ) + a_x - C_l( i ) ) )
                  IF ( printd ) WRITE( out, "( i5, 'l', 3ES12.4 )" ) i,        &
                    RES_l( i ), a_x - C_l( i ),                                &
                    ABS( - RES_l( i ) + a_x - C_l( i ) )
                END DO

                DO i = dims%c_u_start, dims%c_l_end
                  a_x = zero
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                  res_max = MAX( res_max, ABS( - RES_l( i ) + a_x - C_l( i ) ),&
                                          ABS( - RES_u( i ) + C_u( i ) - a_x ) )
                  IF ( printd ) WRITE( out, "( i5, 'l', 3ES12.4 )" ) i,        &
                    RES_l( i ), a_x - C_l( i ),                                &
                    ABS( - RES_l( i ) + a_x - C_l( i ) )
                  IF ( printd ) WRITE( out, "( i5, 'u', 3ES12.4 )" ) i,        &
                    RES_u( i ), C_u( i ) - a_x,                                &
                    ABS( - RES_u( i ) + C_u( i ) - a_x )
                END DO

                DO i = dims%c_l_end + 1, m
                  a_x = zero
                  DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                  res_max = MAX( res_max, ABS( - RES_u( i ) + C_u( i ) - a_x ) )
                  IF ( printd ) WRITE( out, "( i5, 'u', 3ES12.4 )" ) i,        &
                    RES_u( i ), C_u( i ) - a_x,                                &
                    ABS( - RES_u( i ) + C_u( i ) - a_x )
                END DO

                WRITE( out, "( ' ++ errors in computed residuals', ES12.4 )" ) &
                  res_max
              END IF
            ELSE
              linesearch_inform = - 1
            END IF
          END IF

!  Compute the number of infeasibilities, their sum, and the overal merit value

          inform%num_g_infeas =                                                &
            COUNT( ABS( RES_l( : dims%c_equality ) ) > feas_tol ) +            &
            COUNT( RES_l( dims%c_equality + 1 : ) < - feas_tol ) +             &
            COUNT( RES_u < - feas_tol )
          inform%num_b_infeas =                                                &
            COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < - feas_tol ) +  &
            COUNT( X( dims%x_l_start : dims%x_l_end )                          &
                   - X_l( dims%x_l_start : dims%x_l_end ) < - feas_tol ) +     &
            COUNT( X_u( dims%x_u_start : dims%x_u_end )                        &
                   - X( dims%x_u_start : dims%x_u_end ) < - feas_tol ) +       &
            COUNT( X( dims%x_u_end + 1 : n ) > feas_tol )

          inform%infeas_g = SUM( ABS( RES_l( : dims%c_equality ) ) ) -         &
            SUM( MIN( RES_l( dims%c_equality + 1 : ), zero ) ) -               &
            SUM( MIN( RES_u, zero ) )
          inform%infeas_b = -                                                  &
            SUM( MIN( X( dims%x_free + 1 : dims%x_l_start - 1 ), zero ) ) -    &
            SUM( MIN( X( dims%x_l_start : dims%x_l_end )                       &
                      - X_l( dims%x_l_start : dims%x_l_end ), zero ) ) -       &
            SUM( MIN( X_u( dims%x_u_start : dims%x_u_end )                     &
                      - X( dims%x_u_start : dims%x_u_end ), zero ) ) -         &
            SUM( MIN( - X( dims%x_u_end + 1 : n ), zero ) )

          inform%merit = inform%obj + rho_g * inform%infeas_g                  &
                                    + rho_b * inform%infeas_b

          move_infeasible = .FALSE.
          sc_data = '            '

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Update the Schur complement following a change in the working set
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  A constraint has been encountered and will be added

          IF ( linesearch_inform > 0 ) THEN

            inner_stop_absolute = control%inner_stop_absolute
            inner_stop_relative = control%inner_stop_relative

!  Add a constraint
!  ================

            CALL QPA_add_constraint( QPA_add_constraint_status, control,       &
                     inform, dims, n, m, K_part, active, out, k_n_max, m_link, &
                     itref_max, j_add, j_del, scu_status, m_active, s_plus,    &
                     s_minus, printt, printm, printd, addel, sc_data, C_stat,  &
                     B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low, B, RES,  &
                     RES_print, VECTOR, PERT, A_ptr, A_col, A_val, SCU_mat,    &
                     SCU_info, SCU_data, K, SLS_control, SLS_data )

!  Check for error exits

            SELECT CASE( QPA_add_constraint_status )
            CASE ( 1 )
              IF ( printt .OR. ( printi .AND. pcount == 0 ) )                  &
                WRITE( out, 2000 ) prefix, precon, prefix
              WRITE( sc_data, "( I4, 8X )" ) SCU_mat%m
              IF ( printi ) THEN
                CALL CLOCK_time( clock_now )
                WRITE( out, 2050 ) prefix, inform%iter, inform%merit,          &
                  t_opt, STRING_exponent( inform%infeas_g + inform%infeas_b ), &
                  inform%num_g_infeas + inform%num_b_infeas, pcg_iter,         &
                  sc_data, addel, STRING_real_7( clock_now - clock_start )
              END IF
              EXIT minor
            CASE ( 2 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              IF ( printe ) WRITE ( out, 2120 ) prefix, scu_status
              EXIT minor
            CASE ( 3 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              IF ( printe ) WRITE ( out, 2130 ) prefix, scu_status
              EXIT minor
            CASE ( 4 )
              WRITE( out, 2140 ) prefix, s_minus, s_plus,                      &
                                 SCU_info%inertia( 1 : 2 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              EXIT minor
            END SELECT

!  ====================
!  Check for optimality
!  ====================

          ELSE IF ( linesearch_inform == - 1 ) THEN

            inner_stop_absolute = MAX( point1 * inner_stop_absolute,           &
                                       SQRT( EPSILON( one ) ) )
            inner_stop_relative = point1 * inner_stop_relative

!  The minimizer for the current working set has been found.
!  Choose a constraint to delete, by looking for constraints with
!  Lagrange multipliers whose values lie outside [0,rho_g] or [0,rho_b]
!  as appropriate

!  Pick the appropriate deletion strategy

            SELECT CASE( control%deletion_strategy )

!  Most-violated multiplier strategy used

            CASE ( 0 )

              mult_min = mult_zero ; mult_max = zero
              j_min = 0

!  consider constraints in the Schur complement

              DO i = SCU_mat%m, 1, - 1
                j = SC( i )
                IF ( j > 0 ) THEN
                  mult = - S( K_part%k_ref + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_min ) THEN
                    mult_min = mult ; j_min = j ; move_infeasible = move_infeas
                    IF ( printd ) THEN
                       write(out,"( ' mult_min, j ', ES10.2, I6 )" ) mult, j
                    END IF
                  END IF
                END IF
              END DO

!  now consider constraints in the reference set

              DO i = K_part%m_ref, 1, - 1
                j = REF( i )
                IF ( j > 0 ) THEN
                  mult = - S( n + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_min ) THEN
                    mult_min = mult ; j_min = j ; move_infeasible = move_infeas
                    IF ( printd ) THEN
                       write(out,"( ' mult_min, j ', ES10.2, I6 )" ) mult, j
                    END IF
                  END IF
                END IF
              END DO

              IF ( j_min /= 0 ) THEN
                j = j_min
                IF ( printt ) THEN
                  IF ( j <= m ) THEN
                    WRITE( out, 2220 ) prefix, 'y', j, mult_min
                  ELSE
                    WRITE( out, 2220 ) prefix, 'z', j - m, mult_min
                  END IF
                END IF
                GO TO 20
              END IF

!  LIFO strategy used

            CASE ( 1 )

              mult_max = zero

!  consider constraints in the Schur complement

              DO i = SCU_mat%m, 1, - 1
                j = SC( i )
                IF ( j > 0 ) THEN
                  mult = - S( K_part%k_ref + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_zero ) THEN
                    move_infeasible = move_infeas
                    IF ( printt ) THEN
                      IF ( j <= m ) THEN
                        WRITE( out, 2220 ) prefix, 'y', j, mult
                      ELSE
                        WRITE( out, 2220 ) prefix, 'z', j - m, mult
                      END IF
                    END IF
                    GO TO 20
                  END IF
                END IF
              END DO

!  now consider constraints in the reference set

              DO i = K_part%m_ref, 1, - 1
                j = REF( i )
                IF ( j > 0 ) THEN
                  mult = - S( n + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_zero ) THEN
                    move_infeasible = move_infeas
                    IF ( printt ) THEN
                      IF ( j <= m ) THEN
                        WRITE( out, 2220 ) prefix, 'y', j, mult
                      ELSE
                        WRITE( out, 2220 ) prefix, 'z', j - m, mult
                      END IF
                    END IF
                    GO TO 20
                  END IF
                END IF
              END DO

!  LIFO(k) strategy used

            CASE DEFAULT

              mult_min = zero ; mult_max = zero
              j_min = 0

              ll = 0

!  consider constraints in the Schur complement

              DO i = SCU_mat%m, 1, - 1
                j = SC( i )
                IF ( j > 0 ) THEN
                  mult = - S( K_part%k_ref + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_zero ) THEN
                    IF ( mult < mult_min ) THEN
                      mult_min = mult ; j_min = j
                      move_infeasible = move_infeas
                    END IF
                    ll = ll + 1
                    IF ( ll >= control%deletion_strategy ) THEN
                      j = j_min
                      IF ( printt ) THEN
                        IF ( j <= m ) THEN
                          WRITE( out, 2220 ) prefix, 'y', j, mult_min
                        ELSE
                          WRITE( out, 2220 ) prefix, 'z', j - m, mult_min
                        END IF
                      END IF
                      GO TO 20
                    END IF
                  END IF
                END IF
              END DO

!  now consider constraints in the reference set

              DO i = K_part%m_ref, 1, - 1
                j = REF( i )
                IF ( j > 0 ) THEN
                  mult = - S( n + i )
                  IF ( j <= m ) THEN
                    IF ( j <= dims%c_equality ) THEN
                      CYCLE
                    ELSE IF ( j > dims%c_l_end ) THEN
                      mult = - mult
                    ELSE IF ( j >= dims%c_u_start ) THEN
                      IF ( C_up_or_low( j ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_g
                    IF ( move_infeas ) mult = rho_g - mult
                  ELSE
                    jj = j - m
                    IF ( jj > dims%x_l_end ) THEN
                      mult = - mult
                    ELSE IF ( jj >= dims%x_u_start ) THEN
                      IF ( X_up_or_low( jj ) == 1 ) mult = - mult
                    END IF
                    mult_max = MAX( mult_max, mult )
                    move_infeas = mult > rho_b
                    IF ( move_infeas ) mult = rho_b - mult
                  END IF
                  IF ( mult < mult_zero ) THEN
                    IF ( mult < mult_min ) THEN
                      mult_min = mult ; j_min = j
                      move_infeasible = move_infeas
                    END IF
                    ll = ll + 1
                    IF ( ll >= control%deletion_strategy ) THEN
                      j = j_min
                      IF ( printt ) THEN
                        IF ( j <= m ) THEN
                          WRITE( out, 2220 ) prefix, 'y', j, mult_min
                        ELSE
                          WRITE( out, 2220 ) prefix, 'z', j - m, mult_min
                        END IF
                      END IF
                      GO TO 20
                    END IF
                  END IF
                END IF
              END DO

              IF ( j_min /= 0 ) THEN
                j = j_min
                IF ( printt ) THEN
                  IF ( j <= m ) THEN
                    WRITE( out, 2220 ) prefix, 'y', j, mult_min
                  ELSE
                    WRITE( out, 2220 ) prefix, 'z', j - m, mult_min
                  END IF
                END IF
                GO TO 20
              END IF

            END SELECT

!  If the are no multipliers outside [0,rho_g] or [0,rho_b]
!  (as appropriate), we have found the optimal solution
!  --------------------------------------------------------

            inform%status = GALAHAD_ok

!  Record the Lagrange multipliers

            Y = zero ; Z = zero

!  Set the multipliers for violated constraints to +/- rho_g or rho_b
!  as appropriate.

!  equality constraints

            DO i = 1, dims%c_equality
              ll = C_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( RES_l( i ) < zero ) THEN
                Y( i ) = rho_g
              ELSE IF ( RES_l( i ) > zero ) THEN
                Y( i ) = - rho_g
              END IF
            END DO

!  constraints with lower bounds

            DO i = dims%c_equality + 1, dims%c_u_start - 1
              ll = C_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( RES_l( i ) < zero ) Y( i ) = rho_g
            END DO

!  constraints with lower and upper bounds

            DO i = dims%c_u_start, dims%c_l_end
              ll = C_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( RES_l( i ) < zero ) THEN
                Y( i ) = rho_g
              ELSE IF ( RES_u( i ) < zero ) THEN
                Y( i ) = - rho_g
              END IF
            END DO

!  constraints with upper bounds

            DO i = dims%c_l_end + 1, m
              ll = C_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( RES_u( i ) < zero ) Y( i ) = - rho_g
            END DO

!  simple non-negativity

            DO i = dims%x_free + 1, dims%x_l_start - 1
              ll = B_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( X( i ) < zero ) Z( i ) = rho_b
            END DO

!  simple bound from below

            DO i = dims%x_l_start, dims%x_u_start - 1
              ll = B_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( X( i ) - X_l( i ) < zero ) Z( i ) = rho_b
            END DO

!  simple bound from below and above

            DO i = dims%x_u_start, dims%x_l_end
              ll = B_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( X( i ) - X_l( i ) < zero ) THEN
                Z( i ) = rho_b
              ELSE IF ( X_u( i ) - X( i ) < zero ) THEN
                Z( i ) = - rho_b
              END IF
            END DO

!  simple bound from above

            DO i = dims%x_l_end + 1, dims%x_u_end
              ll = B_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( X_u( i ) - X( i ) < zero ) Z( i ) = - rho_b
            END DO

!  simple non-positivity

            DO i = dims%x_u_end + 1, n
              ll = B_stat( i )
              IF ( ll > 0 ) CYCLE
              IF ( ll < 0 ) THEN
                IF ( REF( - ll ) > 0 ) CYCLE
              END IF
              IF ( - X( i ) < zero ) Z( i ) = - rho_b
            END DO

!  Now set the multipliers for constraints in the working set

            DO i = 1, K_part%m_ref
              j = REF( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  Y( j ) = - S( n + i )
                ELSE
                  Z( j - m ) = - S( n + i )
                END IF
              END IF
            END DO

            DO i = 1, SCU_mat%m
              j = SC( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  Y( j ) = - S( K_part%k_ref + i )
                ELSE
                  Z( j - m ) = - S( K_part%k_ref + i )
                END IF
              END IF
            END DO

            IF ( printt .OR. ( printi .AND. pcount == 0 ) ) WRITE( out, 2000 ) &
              prefix, precon, prefix
            IF ( printi ) THEN
              CALL CLOCK_time( clock_now )
              WRITE( out, 2050 ) prefix, inform%iter, inform%merit, t_opt,     &
                STRING_exponent( inform%infeas_g + inform%infeas_b ),          &
                inform%num_g_infeas + inform%num_b_infeas, pcg_iter, sc_data,  &
                addel, STRING_real_7( clock_now - clock_start )
            END IF

            IF ( printi ) WRITE( out, "( /, A, 18X,                            &
       &             ' ======================================',                &
       &     /, A, 18X, '         Optimal Solution found',                     &
       &     /, A, 18X, '   general pen. parameter =', ES11.4,                 &
       &     /, A, 18X, '     bound pen. parameter =', ES11.4,                 &
       &     /, A, 18X, '       maximum multiplier =', ES11.4,                 &
       &     /, A, 18X, ' ======================================' )" )         &
              prefix, prefix, prefix, rho_g, prefix, rho_b, prefix, mult_max,  &
              prefix
            EXIT major

!  There is a multiplier outside [0,rho_g] (or [0,rho_b])
!  ----------------------------------------------------

  20        CONTINUE

!  Remove a constraint from the Schur complement

            inactive = j
            CALL QPA_delete_constraint( QPA_delete_constraint_status, control, &
                   inform, dims, n, m, K_part, inactive, out, k_n_max, m_link, &
                   itref_max, j_add, j_del, scu_status, m_active, s_plus,      &
                   s_minus, printt, printm, printd, printe, addel, sc_data,    &
                   C_stat, B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low,    &
                   B, RES, RES_print, VECTOR, PERT, SCU_mat,                   &
                   SCU_info, SCU_data, K, SLS_control, SLS_data )

!  Check for error exits

            SELECT CASE( QPA_delete_constraint_status )
            CASE ( 1 )
              IF ( printt .OR. ( printi .AND. pcount == 0 ) )                  &
                WRITE( out, 2000 ) prefix, precon, prefix
              IF ( printi ) THEN
                CALL CLOCK_time( clock_now )
                WRITE( out, 2050 ) prefix, inform%iter, inform%merit,          &
                t_opt, STRING_exponent( inform%infeas_g + inform%infeas_b ),   &
                inform%num_g_infeas + inform%num_b_infeas, pcg_iter, sc_data,  &
                addel, STRING_real_7( clock_now - clock_start )
              END IF
              EXIT minor
            CASE ( 2 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              IF ( printe ) WRITE ( out, 2120 ) prefix, scu_status
              EXIT minor
            CASE ( 3 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              IF ( printe ) WRITE ( out, 2130 ) prefix, scu_status
              EXIT minor
            CASE ( 4 )
              WRITE( out, 2140 ) prefix, s_minus, s_plus,                      &
                                 SCU_info%inertia( 1 : 2 )
              IF ( printe ) WRITE ( out, 2062 ) prefix, inform%iter, addel
              EXIT minor
            END SELECT

          ELSE
            j_add = 0
            j_del = 0
            inner_stop_absolute = MAX( point1 * inner_stop_absolute,           &
                                       SQRT( EPSILON( one ) ) )
            inner_stop_relative = point1 * inner_stop_relative
          END IF

!  Make sure that all is well (this should NEVER be executed!)

          DO i = 1, SCU_mat%m
            ii = SC( i )
            IF ( ii > 0 ) THEN
              IF ( ii <= m ) THEN
                jj = C_stat( ii )
              ELSE
                jj = B_stat( ii - m )
              END IF
              IF ( jj /= i ) THEN
                IF ( ii <= m ) THEN
                  WRITE( out, "( 'i, SC, C_stat ', 3i5 )" ) i, ii, jj
                ELSE
                  WRITE( out, "( 'i, SC, B_stat ', 3i5 )" ) i, ii, jj
                END IF
                STOP
              END IF
            ELSE IF ( ii < 0 ) THEN
              IF ( ii >= - m ) THEN
                jj = - C_stat( - ii )
              ELSE
                jj = - B_stat( - ii - m )
              END IF
              ll = - REF( jj )
              IF ( ll /= i ) THEN
                IF ( ii >= - m ) THEN
                  WRITE( out, "( 'i, - SC, - C_stat, - REF ', 4i5 )" )         &
                    i, ii, jj, ll
                ELSE
                  WRITE( out, "( 'i, - SC, - B_stat, - REF ', 4i5 )" )         &
                    i, ii, jj, ll
                END IF
                STOP
              END IF
            END IF
          END DO

!  Monitor the residuals of active constraints

          IF ( MOD( inform%iter, control%monitor_residuals ) == 0 ) THEN

            res_max = zero

!  reference constraints

            DO i = 1, K_part%m_ref
              j = REF( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                END IF
                res_max = MAX( res_max, ABS( a_x ) )
              END IF

            END DO

!  other constraints

            DO i = 1, SCU_mat%m
              j = SC( i )
              IF ( j > 0 ) THEN
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                  END DO
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                END IF
                res_max = MAX( res_max, ABS( a_x ) )
              END IF
            END DO

            IF ( res_max > epsmch ** 0.333 ) THEN
              IF ( printi ) THEN
                CALL CLOCK_time( clock_now )
                WRITE( out, 2050 ) prefix, inform%iter, inform%merit,          &
                  t_opt, STRING_exponent( inform%infeas_g + inform%infeas_b ), &
                  inform%num_g_infeas + inform%num_b_infeas, pcg_iter,         &
                  sc_data, addel, STRING_real_7( clock_now - clock_start )
              END IF
              addel = '         '
              IF ( printi ) WRITE( out, "( /, ' maximum residual of',          &
             &     ' working constituents is ', ES10.2 )" ) res_max
              warmer_start = .TRUE.
              EXIT minor
            ELSE
              IF ( printm ) WRITE( out, "( ' maximum residual of',             &
           &       ' working constituents is ', ES10.2 )" ) res_max
            END IF
          END IF

!  Print active constraint residuals if required

!  reference constraints

          IF ( printm ) THEN
            DO i = 1, K_part%m_ref
              j = REF( i )
              IF ( j > 0 ) THEN
                ats = zero
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                    ats = ats + A_val( ii ) * S( A_col( ii ) )
                  END DO
                  IF ( abs( a_x ) > ten ** ( - 5 ) )                           &
                    WRITE( out, "( I7, 'c is an active reference, residual = ',&
                 &                 ES12.4 )" ) j, a_x
                  IF ( j /= j_add ) THEN
                    IF ( abs( ats ) > ten ** ( - 5 ) )                         &
                      WRITE( out, "( I7, 'c is an active reference, a(t)s  = ',&
                 &                   ES12.4 )" ) j, ats
                  ELSE
                    IF ( printm )                                              &
                      WRITE( out, "( I7, 'c is an active reference, a(t)s  = ',&
                 &                   ES12.4 )" ) j, ats
                  END IF
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                  ats = S( jj )
                  IF ( abs( a_x ) > ten ** ( - 5 ) )                           &
                    WRITE( out, "( I7, 'b is an active reference, residual = ',&
                 &               ES12.4 )" ) jj, a_x
                  IF ( j /= j_add ) THEN
                    IF ( abs( ats ) > ten ** ( - 5 ) )                         &
                      WRITE( out, "( I7, 'b is an active reference, a(t)s  = ',&
                 &                   ES12.4 )" ) jj, ats
                  ELSE
                    IF ( printm )                                              &
                      WRITE( out, "( I7, 'b is an active reference, a(t)s  = ',&
                 &                   ES12.4 )" ) jj, ats
                  END IF
                END IF
              END IF
            END DO

!  other constraints

            DO i = 1, SCU_mat%m
              j = SC( i )
              IF ( j > 0 ) THEN
                ats = zero
                IF ( j <= m ) THEN
                  IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                    IF ( C_up_or_low( j ) == - 1 ) THEN
                      a_x = - C_l( j )
                    ELSE
                      a_x = - C_u( j )
                    END IF
                  ELSE IF ( j > dims%c_l_end ) THEN
                    a_x = - C_u( j )
                  ELSE
                    a_x = - C_l( j )
                  END IF
                  DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                    a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                    ats = ats + A_val( ii ) * S( A_col( ii ) )
                  END DO
                  IF ( abs( a_x ) > ten ** ( - 5 ) )                           &
                  WRITE( out, "( I7, 'c is an active other,     residual = ',  &
                 &               ES12.4 )" ) j, a_x
                  IF ( j /= j_add ) THEN
                    IF ( abs( ats ) > ten ** ( - 5 ) )                         &
                      WRITE( out, "( I7, 'c is an active other,     a(t)s  = ',&
                 &                   ES12.4 )" ) j, ats
                  ELSE
                    IF ( printm )                                              &
                      WRITE( out, "( I7, 'c is an active other,     a(t)s  = ',&
                 &                   ES12.4 )" ) j, ats
                  END IF
                ELSE
                  jj = j - m
                  IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                    IF ( X_up_or_low( jj ) == - 1 ) THEN
                      a_x = X( jj ) - X_l( jj )
                    ELSE
                      a_x = X( jj ) - X_u( jj )
                    END IF
                  ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                    a_x = X( jj ) - X_u( jj )
                  ELSE IF ( jj < dims%x_u_start .AND.                          &
                            jj >= dims%x_l_start ) THEN
                    a_x = X( jj ) - X_l( jj )
                  ELSE
                    a_x = X( jj )
                  END IF
                  ats = S( jj )
                  IF ( abs( a_x ) > ten ** ( - 5 ) )                           &
                  WRITE( out, "( I7, 'b is an active other,     residual = ',  &
                 &               ES12.4 )" ) jj, a_x
                  IF ( j /= j_add ) THEN
                    IF ( abs( ats ) > ten ** ( - 5 ) )                         &
                      WRITE( out, "( I7, 'b is an active other,     a(t)s  = ',&
                 &                   ES12.4 )" ) jj, ats
                  ELSE
                    IF ( printm )                                              &
                      WRITE( out, "( I7, 'b is an active other,     a(t)s  = ',&
                 &                   ES12.4 )" ) jj, ats
                  END IF
                END IF
              END IF
            END DO
          END IF

          IF ( printt .OR. ( printi .AND. ( minor_start .OR. pcount == 0 ) )   &
               ) WRITE( out, 2000 ) prefix, precon, prefix
          IF ( printi .AND. .NOT. (  minor_start .AND. pcount == 0 ) ) THEN
            CALL CLOCK_time( clock_now )
            WRITE( out, 2050 )  prefix, inform%iter, inform%merit, t_opt,      &
                STRING_exponent( inform%infeas_g + inform%infeas_b ),          &
                inform%num_g_infeas + inform%num_b_infeas, pcg_iter, sc_data,  &
                addel, STRING_real_7( clock_now - clock_start )
          END IF
          pcount = MOD( pcount + 1, 50 )
          addel = '         '
        END DO minor

  410   CONTINUE

!  ------------------------------------------------------------------------
!                      End of Minor iteration
!  ------------------------------------------------------------------------

!  Prepare for the next major iteration by computing the new reference set

        CALL QPA_new_reference_set( control, inform, prefix, n, m, K_part,     &
                                    SCU_mat, out, m_link, pcount, printd,      &
                                    warmer_start, check_dependent, C_stat,     &
                                    B_stat, SC, REF, IBREAK, A_ptr,            &
                                    A_col, A_val, K, P, SOL, D,                &
                                    SLS_data, SLS_control )

      END DO major  ! end of minimization

!  ------------------------------------------------------------------------
!                      End of Major iteration
!  ------------------------------------------------------------------------

!  Confirm optimal value found

      IF ( inform%status == 0 ) THEN

!  Print active constraint residuals if required

       IF ( printi ) WRITE( out,                                               &
          "( /, A, 18X, 'verifying optimality and feasibility ...' )" ) prefix

!  reference constraints

        DO i = 1, K_part%m_ref
          j = REF( i )
          IF ( j > 0 ) THEN
            IF ( j <= m ) THEN
              IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                IF ( C_up_or_low( j ) == - 1 ) THEN
                  a_x = - C_l( j )
                ELSE
                  a_x = - C_u( j )
                END IF
              ELSE IF ( j > dims%c_l_end ) THEN
                a_x = - C_u( j )
              ELSE
                a_x = - C_l( j )
              END IF
              ats = ABS( a_x )
              DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                ats = ats + ABS( A_val( ii ) * X( A_col( ii ) ) )
              END DO
              IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )                &
                WRITE( out, "( I7, 'c is an active reference, residual,',    &
               &               ' size = ',  2ES12.4 )" ) j, a_x, ats
            ELSE
              jj = j - m
              IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                IF ( X_up_or_low( jj ) == - 1 ) THEN
                  a_x = X( jj ) - X_l( jj )
                  ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
                ELSE
                  a_x = X( jj ) - X_u( jj )
                  ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
                END IF
              ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                a_x = X( jj ) - X_u( jj )
                ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
              ELSE IF ( jj < dims%x_u_start .AND.                          &
                        jj >= dims%x_l_start ) THEN
                a_x = X( jj ) - X_l( jj )
                ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
              ELSE
                a_x = X( jj )
                ats = ABS( X( jj ) )
              END IF
              IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )                &
                WRITE( out, "( I7, 'b is an active reference, residual,',    &
               &               ' size = ', 2ES12.4 )" ) jj, a_x, ats
            END IF
            S( n + i ) = - a_x
          ELSE
            S( n + i ) = zero
          END IF
        END DO

!  other constraints

        DO i = 1, SCU_mat%m
          j = SC( i )
          IF ( j > 0 ) THEN
            IF ( j <= m ) THEN
              IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
                IF ( C_up_or_low( j ) == - 1 ) THEN
                  a_x = - C_l( j )
                ELSE
                  a_x = - C_u( j )
                END IF
              ELSE IF ( j > dims%c_l_end ) THEN
                a_x = - C_u( j )
              ELSE
                a_x = - C_l( j )
              END IF
              ats = ABS( a_x )
              DO ii = A_ptr( j ), A_ptr( j + 1 ) - 1
                a_x = a_x + A_val( ii ) * X( A_col( ii ) )
                ats = ats + ABS( A_val( ii ) * X( A_col( ii ) ) )
              END DO
              IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )                &
                WRITE( out, "( I7, 'c is an active other,     residual,',    &
               &               ' size = ',  2ES12.4 )" ) j, a_x, ats
            ELSE
              jj = j - m
              IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
                IF ( X_up_or_low( jj ) == - 1 ) THEN
                  a_x = X( jj ) - X_l( jj )
                  ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
                ELSE
                  a_x = X( jj ) - X_u( jj )
                  ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
                END IF
              ELSE IF ( jj > dims%x_l_end .AND. jj <= dims%x_u_end ) THEN
                a_x = X( jj ) - X_u( jj )
                ats = ABS( X( jj ) ) + ABS( X_u( jj ) )
              ELSE IF ( jj < dims%x_u_start .AND. jj >= dims%x_l_start ) THEN
                a_x = X( jj ) - X_l( jj )
                ats = ABS( X( jj ) ) + ABS( X_l( jj ) )
              ELSE
                a_x = X( jj )
                ats = ABS( X( jj ) )
              END IF
              IF ( printm .AND. abs( a_x ) > ten ** ( - 8 ) )                 &
                WRITE( out, "( I7, 'b is an active other,     residual,',     &
               &               ' size = ',  2ES12.4 )" ) j, a_x, ats
            END IF
            S( K_part%k_ref + i ) = - a_x
          ELSE
            S( K_part%k_ref + i ) = zero
          END IF
        END DO
        S( : n ) = zero

!  Find a point that satisfies the constraints

        n_all = SCU_mat%n + SCU_mat%m

        S_perm( PERM( : n_all ) ) = S( : n_all )
        iii = itref_max
        CALL QPA_iterative_refinement(                                        &
                            K, SCU_mat, SCU_data, S_perm( : n_all ),          &
                            RES( : n_all ), B( : n_all ), S( : n_all ),       &
                            DX( : n_all ), VECTOR( : SCU_mat%n ),             &
                            SLS_control, SLS_data, K_part, .FALSE.,           &
                            iii, out, printm, RES_print, inform )

        S( : n_all ) = RES( PERM( : n_all ) )
        X = X + S( : n )

!  From this point, try to minimize the quadratic on the current working set

!       IF ( .NOT. G_eq_H ) THEN
        IF ( inform%status == - 20 ) THEN

!  Compute the resulting residuals

          DO i = 1, dims%c_u_start - 1
            a_x = zero
            DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
              a_x = a_x + A_val( ii ) * X( A_col( ii ) )
            END DO
            RES_l( i ) = a_x - C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            a_x = zero
            DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
              a_x = a_x + A_val( ii ) * X( A_col( ii ) )
            END DO
            RES_l( i ) = a_x - C_l( i )
            RES_u( i ) = C_u( i ) - a_x
          END DO

          DO i = dims%c_l_end + 1, m
            a_x = zero
            DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
              a_x = a_x + A_val( ii ) * X( A_col( ii ) )
            END DO
            RES_u( i ) = C_u( i ) - a_x
          END DO

!  Compute the resulting gradient

          GRAD = G
          DO i = 1, n
            DO ii = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( ii )
              GRAD( i ) = GRAD( i ) + H_val( ii ) * X( j )
              IF ( i /= j ) GRAD( j ) = GRAD( j ) + H_val( ii ) * X( i )
            END DO
          END DO
          inform%obj = f + half * DOT_PRODUCT( GRAD + G, X )

          linesearch_inform = 0

          inform%num_g_infeas =                                              &
            COUNT( ABS( RES_l( : dims%c_equality ) ) > feas_tol ) +          &
            COUNT( RES_l( dims%c_equality + 1 : ) < - feas_tol ) +           &
            COUNT( RES_u < - feas_tol )
          inform%num_b_infeas =                                                &
            COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < - feas_tol ) +  &
            COUNT( X( dims%x_l_start : dims%x_l_end )                          &
                   - X_l( dims%x_l_start : dims%x_l_end ) < - feas_tol ) +     &
            COUNT( X_u( dims%x_u_start : dims%x_u_end )                        &
                   - X( dims%x_u_start : dims%x_u_end ) < - feas_tol ) +       &
            COUNT( X( dims%x_u_end + 1 : n ) > feas_tol )

          inform%infeas_g = SUM( ABS( RES_l( : dims%c_equality ) ) ) -         &
            SUM( MIN( RES_l( dims%c_equality + 1 : ), zero ) ) -               &
            SUM( MIN( RES_u, zero ) )
          inform%infeas_b = -                                                  &
            SUM( MIN( X( dims%x_free + 1 : dims%x_l_start - 1 ), zero ) ) -    &
            SUM( MIN( X( dims%x_l_start : dims%x_l_end )                       &
                      - X_l( dims%x_l_start : dims%x_l_end ), zero ) ) -       &
            SUM( MIN( X_u( dims%x_u_start : dims%x_u_end )                     &
                      - X( dims%x_u_start : dims%x_u_end ), zero ) ) -         &
            SUM( MIN( - X( dims%x_u_end + 1 : n ), zero ) )

          inform%merit = inform%obj + rho_g * inform%infeas_g                  &
                                    + rho_b * inform%infeas_b

          IF ( printt .OR. ( printi .AND. pcount == 0 ) ) &
             WRITE( out, 2000 ) prefix, precon, prefix
          IF ( printi .AND. .NOT. (  minor_start .AND. pcount == 0 ) ) THEN
            CALL CLOCK_time( clock_now )
            IF ( G_eq_H ) THEN
              WRITE( out, 2060 ) prefix, inform%iter, inform%merit,            &
                STRING_exponent( inform%infeas_g + inform%infeas_b ),          &
                inform%num_g_infeas + inform%num_b_infeas,                     &
                STRING_real_7( clock_now - clock_start )
            ELSE
              WRITE( out, 2061 ) prefix, inform%iter, inform%merit,            &
                STRING_exponent( inform%infeas_g + inform%infeas_b ),          &
                inform%num_g_infeas + inform%num_b_infeas, pcg_iter,           &
                STRING_real_7( clock_now - clock_start )
            END IF
          END IF

          inner_stop_relative = zero
          inner_stop_absolute = SQRT( EPSILON( one ) )
          cg_maxit = control%cg_maxit
          X_pcg = X
          RHS( : n ) = - G
          iii = itref_max + 1
          CALL QPA_pcg( dims, n, S, RHS, R_pcg, X_pcg, P_pcg, S_perm,          &
                         RES, B, DX, VECTOR, PERM, SCU_mat, SCU_data,          &
                         K, K_part, H_val, H_col, H_ptr, prefix, control,      &
                         SLS_control,                                          &
                         SLS_data, print_level - 1, cg_maxit, dof, .TRUE.,     &
                         inner_stop_absolute, inner_stop_relative, iii,        &
                         inform, pcg_iter, negative_curvature, pcg_status,     &
                         RES_print( : K_part%n_free + K_part%c_ref ) )
          itref_max = iii - 1

          IF ( pcg_status < 0 ) THEN
            IF ( printt ) WRITE( out, "( /, A,                                 &
           &  ' Warning return from QPA_pcg, status = ', I0 )" )               &
              prefix, pcg_status
          END IF
          IF ( pcg_status == 0 .OR. pcg_status == 1 ) X = S( : n )
          inner_stop_absolute = control%inner_stop_absolute
          inner_stop_relative = control%inner_stop_relative
        END IF

!  Compute the resulting residuals

        DO i = 1, dims%c_u_start - 1
          a_x = zero
          DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
            a_x = a_x + A_val( ii ) * X( A_col( ii ) )
          END DO
          RES_l( i ) = a_x - C_l( i )
        END DO

        DO i = dims%c_u_start, dims%c_l_end
          a_x = zero
          DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
            a_x = a_x + A_val( ii ) * X( A_col( ii ) )
          END DO
          RES_l( i ) = a_x - C_l( i )
          RES_u( i ) = C_u( i ) - a_x
        END DO

        DO i = dims%c_l_end + 1, m
          a_x = zero
          DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
            a_x = a_x + A_val( ii ) * X( A_col( ii ) )
          END DO
          RES_u( i ) = C_u( i ) - a_x
        END DO

!  Compute the resulting gradient

        GRAD = G
        DO i = 1, n
          DO ii = H_ptr( i ), H_ptr( i + 1 ) - 1
            j = H_col( ii )
            GRAD( i ) = GRAD( i ) + H_val( ii ) * X( j )
            IF ( i /= j ) GRAD( j ) = GRAD( j ) + H_val( ii ) * X( i )
          END DO
        END DO
        inform%obj = f + half * DOT_PRODUCT( GRAD + G, X )

        linesearch_inform = 0

        inform%num_g_infeas =                                                  &
          COUNT( ABS( RES_l( : dims%c_equality ) ) > feas_tol ) +              &
          COUNT( RES_l( dims%c_equality + 1 : ) < - feas_tol ) +               &
          COUNT( RES_u < - feas_tol )
        inform%num_b_infeas =                                                  &
          COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < - feas_tol ) +    &
          COUNT( X( dims%x_l_start : dims%x_l_end )                            &
                 - X_l( dims%x_l_start : dims%x_l_end ) < - feas_tol ) +       &
          COUNT( X_u( dims%x_u_start : dims%x_u_end )                          &
                 - X( dims%x_u_start : dims%x_u_end ) < - feas_tol ) +         &
          COUNT( X( dims%x_u_end + 1 : n ) > feas_tol )

        inform%infeas_g = SUM( ABS( RES_l( : dims%c_equality ) ) ) -           &
          SUM( MIN( RES_l( dims%c_equality + 1 : ), zero ) ) -                 &
          SUM( MIN( RES_u, zero ) )
        inform%infeas_b = -                                                    &
          SUM( MIN( X( dims%x_free + 1 : dims%x_l_start - 1 ), zero ) ) -      &
          SUM( MIN( X( dims%x_l_start : dims%x_l_end )                         &
                    - X_l( dims%x_l_start : dims%x_l_end ), zero ) ) -         &
          SUM( MIN( X_u( dims%x_u_start : dims%x_u_end )                       &
                    - X( dims%x_u_start : dims%x_u_end ), zero ) ) -           &
          SUM( MIN( - X( dims%x_u_end + 1 : n ), zero ) )

        inform%merit = inform%obj + rho_g * inform%infeas_g                    &
                                  + rho_b * inform%infeas_b

        IF ( printi ) WRITE( out, 2000 ) prefix, precon, prefix
!       write(6,*) ' minor_start, pcount ', minor_start, pcount
        IF ( printi .AND. .NOT. (  minor_start .AND. pcount == 0 ) ) THEN
          CALL CLOCK_time( clock_now )
          IF ( G_eq_H ) THEN
            WRITE( out, 2060 ) prefix, inform%iter, inform%merit,              &
              STRING_exponent( inform%infeas_g + inform%infeas_b ),            &
              inform%num_g_infeas + inform%num_b_infeas,                       &
              STRING_real_7( clock_now - clock_start )
          ELSE
            WRITE( out, 2061 ) prefix, inform%iter, inform%merit,              &
              STRING_exponent( inform%infeas_g + inform%infeas_b ),            &
              inform%num_g_infeas + inform%num_b_infeas, pcg_iter,             &
              STRING_real_7( clock_now - clock_start )
          END IF
        END IF

      END IF

      IF ( PRESENT( G_p ) .AND. PRESENT( X_p ) .AND.                          &
           PRESENT( Y_p ) .AND. PRESENT( Z_p ) ) THEN

!  Optionally, find the solution to the system

!       (    H  A_active^T ) ( x_p ) = - ( g_p )
!       ( A_active     0   ) ( y_p )     (  0  )

!  (required by the GALAHAD module fastr)

        inner_stop_relative = zero
        inner_stop_absolute = SQRT( EPSILON( one ) )
        cg_maxit = control%cg_maxit
        RHS( : n ) = - G_p
        iii = itref_max + 1
        CALL QPA_pcg( dims, n, S, RHS, R_pcg, X_pcg, P_pcg, S_perm,            &
                       RES, B, DX, VECTOR, PERM, SCU_mat, SCU_data,            &
                       K, K_part, H_val, H_col, H_ptr, prefix, control,        &
                       SLS_control, SLS_data, print_level - 1,                 &
                       cg_maxit, dof, .FALSE., inner_stop_absolute,            &
                       inner_stop_relative, iii, inform,                       &
                       pcg_iter, negative_curvature, pcg_status,               &
                       RES_print( : K_part%n_free + K_part%c_ref ) )
        itref_max = iii - 1
        X_p = S( : n )

!  Now set the parametric multipliers for constraints in the working set

        Y_p = zero ; Z_p = zero
        DO i = 1, K_part%m_ref
          j = REF( i )
          IF ( j > 0 ) THEN
            IF ( j <= m ) THEN
              Y_p( j ) = - S( n + i )
            ELSE
              Z_p( j - m ) = - S( n + i )
            END IF
          END IF
        END DO

        DO i = 1, SCU_mat%m
          j = SC( i )
          IF ( j > 0 ) THEN
            IF ( j <= m ) THEN
              Y_p( j ) = - S( K_part%k_ref + i )
            ELSE
              Z_p( j - m ) = - S( K_part%k_ref + i )
            END IF
          END IF
        END DO
      END IF

!  Set C_stat for exit

      DO i = 1, dims%c_u_start - 1
        ll = C_stat( i )
        IF ( ll > 0 ) THEN
          C_stat( i ) = - 1
        ELSE IF ( ll < 0 ) THEN
          IF ( REF( - ll ) > 0 ) THEN
            C_stat( i ) = - 1
          ELSE
            C_stat( i ) = 0
          END IF
        END IF
      END DO

      DO i = dims%c_u_start, dims%c_l_end
        C_stat( i ) = C_up_or_low( i )
      END DO

      DO i = dims%c_l_end + 1, m
        ll = C_stat( i )
        IF ( ll > 0 ) THEN
          C_stat( i ) = 1
        ELSE IF ( ll < 0 ) THEN
          IF ( REF( - ll ) > 0 ) THEN
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END IF
      END DO

!  Set B_stat for exit

      B_stat( : dims%x_free ) = 0
      DO i = dims%x_free + 1, dims%x_u_start - 1
        ll = B_stat( i )
        IF ( ll > 0 ) THEN
          B_stat( i ) = - 1
        ELSE IF ( ll < 0 ) THEN
          IF ( REF( - ll ) > 0 ) THEN
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END IF
      END DO

      DO i = dims%x_u_start, dims%x_l_end
        B_stat( i ) = X_up_or_low( i )
      END DO

      DO i = dims%x_l_end + 1, n
        ll = B_stat( i )
        IF ( ll > 0 ) THEN
          B_stat( i ) = 1
        ELSE IF ( ll < 0 ) THEN
          IF ( REF( - ll ) > 0 ) THEN
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END IF
      END DO

!  Compute total time

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPA_solve_main ' )" )
      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, '               pr =', I2, '       violation',             &
              '      Schur complt             CPU',                            &
              /, A, '   iter   merit fn.  step  size     #',                   &
                 ' CG it dim +ve -ve   change    time', / )
 2050 FORMAT( A, I7, ES12.4, ES8.1, 1X, A3, I6, I6, A12, A10, A7 )
 2060 FORMAT( A, I7, ES12.4, 8X, 1X, A3, I6, 28X, A7 )
 2061 FORMAT( A, I7, ES12.4, 8X, 1X, A3, I6, I6, 22X, A7 )
 2062 FORMAT( A, I7, 54X, A )
 2100 FORMAT( /, A, ' Preconditioner is inappropriate as it gave',             &
                 ' a negative inner-product ', //, A,                          &
                 ' Perturbing G', :, ' by ', ES12.4, ' and restarting',        &
                 ' (pivtol = ', ES9.2, ')', / )
 2120 FORMAT( /, A, '  on exit from SCU_append,   status = ', I0 )
 2130 FORMAT( /, A, '  on exit from SCU_delete,   status = ', I0 )
 2140 FORMAT( /, A, ' =+=> Inertia should be (', I0, ',', I0, ',  0)',         &
                              ' but is    (', I0, ',', I0, ',  0)' )
 2220 FORMAT( A, '   ', A1, '(', I0, ') = ', ES12.4 )

!  End of QPA_solve_main

      END SUBROUTINE QPA_solve_main

!-*-*-*-*-*-*-   Q P A _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_terminate( data, control, inform )

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine QPA_initialize
!   control see Subroutine QPA_initialize
!   inform  see Subroutine QPA_solve

!  Dummy arguments

      TYPE ( QPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: scu_status
      CHARACTER ( LEN = 80 ) :: array_name

      inform%status = GALAHAD_ok

!  Deallocate all allocated integer arrays

      array_name = 'qpa: data%SC'
      CALL SPACE_dealloc_array( data%SC,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%REF'
      CALL SPACE_dealloc_array( data%REF,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%IBREAK'
      CALL SPACE_dealloc_array( data%IBREAK,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%C_up_or_low'
      CALL SPACE_dealloc_array( data%C_up_or_low,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%X_up_or_low'
      CALL SPACE_dealloc_array( data%X_up_or_low,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%PERM'
      CALL SPACE_dealloc_array( data%PERM,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%Abycol_row'
      CALL SPACE_dealloc_array( data%Abycol_row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%Abycol_ptr'
      CALL SPACE_dealloc_array( data%Abycol_ptr,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S_row'
      CALL SPACE_dealloc_array( data%S_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S_col'
      CALL SPACE_dealloc_array( data%S_col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S_colptr'
      CALL SPACE_dealloc_array( data%S_colptr,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%K%row'
      CALL SPACE_dealloc_array( data%K%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%K%col'
      CALL SPACE_dealloc_array( data%K%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all allocated real arrays

      array_name = 'qpa: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%BREAKP'
      CALL SPACE_dealloc_array( data%BREAKP,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%RES_l'
      CALL SPACE_dealloc_array( data%RES_l,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%RES_u'
      CALL SPACE_dealloc_array( data%RES_u,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%A_s'
      CALL SPACE_dealloc_array( data%A_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%A_norms'
      CALL SPACE_dealloc_array( data%A_norms,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%PERT'
      CALL SPACE_dealloc_array( data%PERT,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%H_s'
      CALL SPACE_dealloc_array( data%H_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%GRAD'
      CALL SPACE_dealloc_array( data%GRAD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S_perm'
      CALL SPACE_dealloc_array( data%S_perm,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%R_pcg'
      CALL SPACE_dealloc_array( data%R_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%X_pcg'
      CALL SPACE_dealloc_array( data%X_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%P_pcg'
      CALL SPACE_dealloc_array( data%P_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%Abycol_val'
      CALL SPACE_dealloc_array( data%Abycol_val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%S_val'
      CALL SPACE_dealloc_array( data%S_val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%K%val'
      CALL SPACE_dealloc_array( data%K%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%RES_print'
      CALL SPACE_dealloc_array( data%RES_print,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%DIAG'
      CALL SPACE_dealloc_array( data%DIAG,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%DX'
      CALL SPACE_dealloc_array( data%DX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within SCU

      array_name = 'qpa: data%SCU_mat%BD_col_start'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_col_start,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%SCU_mat%BD_val'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_val,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpa: data%SCU_mat%BD_row'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_row,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL SCU_terminate( data%SCU_data, scu_status, data%SCU_info )

      IF ( scu_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'qpa: data%SCU_data'
        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE ( control%out, &
             "( ' on exit from SCU_terminate,     status = ', I3 )" ) scu_status
      END IF

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF
      IF ( inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'qpa: data%SLS_data'
        IF ( control%error > 0 )                                               &
             WRITE( control%error, 2900 ) 'data%SLS_data',                     &
                                          inform%SLS_inform%alloc_status
      END IF

      RETURN

!  Deallocate QPP internal arrays

      CALL QPP_terminate( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Non-executable statement

 2900 FORMAT( ' ** Message from -QPA_terminate-', /,                           &
                 ' Allocation error, for ', A, ', status = ', I0 )

!  End of subroutine QPA_terminate

      END SUBROUTINE QPA_terminate

!-*-*-*-*-*-    Q P A _ L I N E S E A R C H    S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_linesearch( dims, n, m,                                   &
                                 f, g_s, s_hs, s_norm, rho_g, rho_b,           &
                                 X, X_l, X_u, RES_l, RES_u, S, A_s, A_norms,   &
                                 C_stat, B_stat, REF, IBREAK, BREAKP, lbreak,  &
                                 m_link, out, print_1line, print_detail,       &
                                 print_debug, t_opt, too_small, active,        &
                                 inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the first local minimizer of the function
!
!     1/2 x(T) H x + c(T) x
!        + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!        + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  along the arc x(t) = x + t s  (t >= 0)
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  H is a symmetric matrix, and A is an m by n matrix.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type QPA_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. See QPA_solve_main for details.
!
!  f is a REAL variable, which must be set by the user to the value
!   of 1/2 <x, H x> + <c, x> at x
!
!  g_s is a REAL variable, which must be set by the user to the value
!   of the inner product <s, Hx + c>
!
!  s_hs is a REAL variable, which must be set by the user to the value
!   of the inner product <s, Hs>
!
!  s_norm is a REAL variable, which must be set by the user to the value
!   of the two norm of s, ||s||_2
!
!  rho_g is a REAL variable, which must be set by the user to the value
!   of the penalty parameter rho_g for the general constraints
!
!  rho_b is a REAL variable, which must be set by the user to the value
!   of the penalty parameter rho_g for the simple bound constraints
!
!  X is a REAL array of length n, which must be set by the user to the
!   value of x
!
!  X_l is a REAL array of length n, that must be set by
!   the user to the value of x_l for all components which have lower bounds
!
!  X_u is a REAL array of  length n, that must be set
!   by the user to the value of x_u for all components which have upper bounds
!
!  RES_l is a REAL array of extent 1:dims%c_l_end, that must be set by
!   the user to the value of A x - c_l for all components which have
!   lower bounds/are equalities
!
!  RES_u is a REAL array of extent dims%c_u_start : m, that must be
!   set by the user to the value of c_u - A x for all components which have
!   upper bounds
!
!  S is a REAL array of length n, which must be set by the user to the
!   value of s
!
!  A_s is a REAL array of length m, which must be set by the user to the
!   value of A s
!
!  A_norms is a REAL array of length m, which must be set by the user to
!   the estimates of the norms of the rows of $A$.
!
!  C_stat, B_stat, REF are INTEGER arrays described in QPA_solve
!
!  IBREAK is an INTEGER workspace array of length lbreak
!
!  BREAKP is a REAL workspace array of length lbreak
!
!  lbreak is an INTEGER that must be at least
!   m + dims%x_l_end - dims%x_u_start + 1
!
!  t_opt  is a REAL variable, which gives the required value of t on exit
!
!  inform is an INTEGER variable, which gives the exit status. Possible
!   values are:
!
!    0     the minimizer given in t_opt occurs between breakpoints after first
!    1     the minimizer given in t_opt occurs at the breakpoint indicated by
!          the variable active
!   -1     the minimizer given in t_opt occurs before the first breakpoint
!   -2     the function is unbounded from below. Ignore the value in t_opt
!   -3     the value m is negative. Ignore the value in t_opt
!
!  active is an INTEGER variable, which gives the index of the constraint
!   that has become active. Possible values are:
!
!   0      the minimizer occurs between breakpoints
!   i in (0,m] the minimizer given in t_opt occurs at a breakpoint at which
!          the i-th constraint is active
!   i in [-m,0) the minimizer given in t_opt occurs at a breakpoint at which
!          the -i-th constraint is active (on its upper bound)
!   i in (m,m+n] the minimizer given in t_opt occurs at a breakpoint at which
!          the i-m-th bound is active
!   i in [-m+n,-m) the minimizer given in t_opt occurs at a breakpoint at which
!          the -i-m-th bound is active (on its upper bound)
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, lbreak, m_link, out
      INTEGER, INTENT( OUT ) :: active, inform
      LOGICAL, INTENT( IN ) :: print_1line, print_detail, print_debug
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, s_norm
      REAL ( KIND = wp ), INTENT( IN ) :: rho_g, rho_b, too_small
      REAL ( KIND = wp ), INTENT( OUT ) :: t_opt
      INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_norms
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, S
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lbreak ) :: BREAKP
      INTEGER, INTENT( IN ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( IN ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( IN ), DIMENSION( n ) :: B_stat

!  Local variables

      INTEGER :: i, l, nbreak, inheap, iter, ibreakp, nbreak_total
      INTEGER :: cluster_start, cluster_end, cluster_best, active_ok
      REAL ( KIND = wp ) :: as, res, val, slope, curv, slope_old, eval, eslope
      REAL ( KIND = wp ) :: t_break, t_star, feasep, epsqrt, infeas_g, infeas_b
      REAL ( KIND = wp ) :: fun, gradient, t_pert, cos_best, t_old
      REAL ( KIND = wp ) :: pert_val, pert_eps, cosine, tiny_cosine, val_old
      REAL ( KIND = wp ) :: breakp_max, slope_infeas_g, slope_infeas_b

      LOGICAL :: beyond_first_breakpoint, recover
      CHARACTER ( LEN = 14 ) :: cluster

      active = 0 ; t_opt = zero
      IF ( m < 0 ) THEN
         inform = - 3
         RETURN
      END IF

      iter = 0 ; nbreak = 0 ; t_break = zero
      epsqrt = SQRT( epsmch )
      tiny_cosine = epsmch ** 0.75

!  Find the distance to each constraint boundary, and the slope of this function
!  =============================================================================

      IF ( exact ) THEN
        t_pert = zero
      ELSE
        t_pert = - epsmch
      END IF
      infeas_g = zero ; slope_infeas_g = zero
      infeas_b = zero ; slope_infeas_b = zero
      breakp_max = zero

!  equality constraints
!  --------------------

      DO i = 1, dims%c_equality
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = RES_l( i )
        infeas_g = infeas_g + ABS( res )

        as = A_s( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as
        IF ( res + t_pert * as > zero ) slope_infeas_g = slope_infeas_g + as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res > zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / ( s_norm * A_norms( i ) )
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                          &
            i, res, as,  - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' const e ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IF ( res /= zero ) THEN
          IBREAK( nbreak ) = - i
        ELSE
          IBREAK( nbreak ) = i
        END IF
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  constraints with lower bounds
!  -----------------------------

      DO i = dims%c_equality + 1, dims%c_l_end
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = RES_l( i )
        IF ( res < zero ) infeas_g = infeas_g - res

        as = A_s( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / ( s_norm * A_norms( i ) )
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                           &
            i, res, as,  - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' const l ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  constraints with upper bounds
!  -----------------------------

      DO i = dims%c_u_start, m
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

! IF ( print_detail .AND. RES_u( i ) < 0 ) write(6,*) '+RES_u ', i, RES_u( i )

        res = RES_u( i )
        IF ( res < zero ) infeas_g = infeas_g - res

        as = - A_s( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as
!       IF ( res + t_pert * as < zero ) write(6,*) ' slope_term = u ', i

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / ( s_norm * A_norms( i ) )
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                           &
            i, res, as,  - res / as, cosine
        END IF
!       IF ( cosine < tiny_cosine ) CYCLE

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' const u ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple non-negativity
!  ---------------------

      DO i = dims%x_free + 1, dims%x_l_start - 1
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = X( i )
        IF ( res < zero ) infeas_b = infeas_b - res

        as = S( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_b = slope_infeas_b - as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / s_norm
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rbound ', i5, 4ES12.4 )" )                           &
            i, res, as,  - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' bound n ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = m + i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple bound from below
!  -----------------------

      DO i = dims%x_l_start, dims%x_l_end
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = X( i ) - X_l( i )
        IF ( res < zero ) infeas_b = infeas_b - res

        as = S( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_b = slope_infeas_b - as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / s_norm
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rbound ', i5, 4ES12.4 )" )                          &
            i, res, as,  - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' bound l ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = m + i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple bound from above
!  -----------------------

      DO i = dims%x_u_start, dims%x_u_end
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = X_u( i ) - X( i )
        IF ( res < zero ) infeas_b = infeas_b - res

        as = - S( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_b = slope_infeas_b - as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / s_norm
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( ' bound ', i5, 4ES12.4 )" )                           &
            i, res, as,  - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' bound u ', i5, 4ES12.4 )" )                         &
            i, res, as,  - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - m - i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple non-positivity
!  ---------------------

      DO i = dims%x_u_end + 1, n
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        res = - X( i )
        IF ( res < zero ) infeas_b = infeas_b - res

        as = - S( i )
        IF ( ABS( as ) < too_small ) CYCLE

        IF ( res + t_pert * as < zero ) slope_infeas_b = slope_infeas_b - as

!  Find if the step will change the status of the constraint

        IF ( ( as > zero .AND. res >= zero ) .OR.                              &
             ( as < zero .AND. res < zero ) ) CYCLE
        cosine = ABS( as ) / s_norm
        IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rbound ', i5, 4ES12.4 )" )                           &
            i, res, as, - res / as, cosine
        END IF

!  Find the breakpoint

        IF ( print_debug ) THEN
          WRITE( out, "( ' bound p ', i5, 4ES12.4 )" )                         &
            i, res, as, - res / as, cosine
        END IF
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - m - i
        BREAKP( nbreak ) = - res / as
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

      nbreak_total = nbreak

!  Record the initial function, slope and curvature

      val = f + rho_g * infeas_g + rho_b * infeas_b
      slope = g_s + rho_g * slope_infeas_g + rho_b * slope_infeas_b
      curv = s_hs

      IF ( print_detail ) THEN
        CALL QPA_p_val_and_slope( dims, n, m,                                  &
                                  f, g_s, s_hs, rho_g, rho_b, X, X_l,          &
                                  X_u, RES_l, RES_u, S, A_s, 0.0_wp, t_pert,   &
                                  too_small, REF, m_link, C_stat, B_stat,      &
                                  eval, eslope )
        write( out, 2010 ) '  val', val, eval
      END IF

!  Record the function value and gradient at (just on the other side of)
!  the initial point

      fun = val ; gradient = slope ; recover = .FALSE.

!  Order the breakpoints in increasing size using a heapsort. Build the heap

      CALL SORT_heapsort_build( nbreak, BREAKP, inheap, INDA = IBREAK )
      cluster_start = 1
      cluster_end = 0
      cluster = '      0      0'

!  =======================================================================
!  Start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces
!  =======================================================================

      beyond_first_breakpoint = .FALSE.
      inform = - 1
      DO

!  ---------------------------------------------------------------
!  The piecewise quadratic function within the current interval is
!    val + slope * t + 0.5 * curv * t**2
!  ---------------------------------------------------------------

!       DO i = 1, dims%c_l_end
!         WRITE( out, "(' lower: i = ', I6, ' res = ', ES12.4 )")              &
!                i, RES_l( i ) + t_break * A_s( i )
!       END DO
!       DO i = dims%c_u_start, m
!         WRITE( out, "(' upper: i = ', I6, ' res = ', ES12.4 )")              &
!                i, RES_u( i ) - t_break * A_s( i )
!       END DO

!  Print details of the piecewise quadratic in the next interval

        iter = iter + 1
        IF ( ( print_1line .AND. cluster_end == 0 ) .OR. print_detail )        &
          WRITE( out, 2000 )
        IF ( print_1line ) WRITE( out, "( 3X, I7, ES12.4, A14, 3ES12.4 )" )    &
           iter, t_break, cluster, fun, gradient, curv

!  If the gradient of the unvariate function increases, exit

        IF ( gradient > gzero ) THEN
          IF ( inform == 0 ) t_opt = t_break
          EXIT
        END IF

!  If the gradient of the univariate function is small and its curvature
!  is positive, exit

        IF ( ABS( gradient ) <= gzero ) THEN
          IF ( curv > - hzero ) THEN
            IF ( inform == 0 ) t_opt = t_break
            EXIT
          END IF
        END IF

!  Find the next breakpoint

        t_old = t_break
        IF ( nbreak > 0 ) THEN
          t_break = BREAKP( 1 )
          CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, INDA = IBREAK )
          cluster_end = cluster_end + 1
          cluster_start = cluster_end
        ELSE
          t_break = biginf
        END IF

!  If the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

        IF ( curv > zero ) THEN
          IF ( print_detail ) WRITE( out, "( ' slope, curv ', 2ES12.4 )" )     &
             slope,  curv
          t_star = - slope / curv

!  If the line minimum occurs before the breakpoint, the line minimum gives
!  the required minimizer. Exit

          IF ( nbreak == 0 .OR. t_star < t_break ) THEN
            t_opt = t_star
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star

!  Calculate the function value for the piecewise quadratic

            fun = val + t_opt * ( slope + half * t_opt * curv )
            val = val + half * t_opt * slope

            IF ( print_detail ) WRITE( out, 2000 )
            IF ( print_1line ) WRITE( out, &
              "( 3X, I7, ES12.4, A14, 3ES12.4 )" ) &
                 iter, t_opt, '      -      -', fun, zero, curv
            IF ( print_debug ) THEN
              eval = QPA_p_val( dims, n, m,                                    &
                                f, g_s, s_hs, rho_g, rho_b, X, X_l, X_u,       &
                                RES_l, RES_u, S, A_s, t_opt, too_small )
              write( out, 2010 ) '  val', fun, eval
            END IF
            IF ( beyond_first_breakpoint ) inform = 0
            EXIT
          ELSE
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star
          END IF
        ELSE

          IF ( print_detail ) WRITE( out, 2040 ) t_old, t_break

!  Exit if the function is unbounded from below

          IF ( nbreak == 0 ) THEN
            t_opt = biginf
            IF ( print_detail ) WRITE( out, 2000 )
            IF ( print_1line ) WRITE( out, &
                                      "( 3X, I7, 5X, 'inf', 22X, '-inf')" ) iter
            inform = - 2
            EXIT
          END IF

        END IF

!  Update the univariate function and slope values

        slope_old = slope

!  Record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

!       pert_val = MAX( epsmch ** 0.5, 0.001_wp * t_break )
        pert_val = MAX( teneps, 0.001_wp * t_break )
        pert_eps = epsmch

!       pert_val = epsmch * 100.0_wp
!       pert_eps = epsmch
!       pert_val = zero
!       pert_eps = zero
!       pert_val = epsmch ** 0.75
!       pert_eps = epsmch

        feasep = t_break + pert_val
        t_pert = pert_val + pert_eps
        t_break = feasep

        IF ( t_break < breakp_max ) THEN

          DO
            ibreakp = IBREAK( nbreak )

            IF ( ibreakp < 0 ) THEN
              ibreakp = - ibreakp
              IF ( ibreakp <= dims%c_equality ) THEN
                slope = slope + rho_g * ABS( A_s( ibreakp ) )
              END IF
            END IF

!  Update the slope

            IF ( ibreakp <= m ) THEN
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'C', IBREAK( nbreak ), BREAKP( nbreak )
               slope = slope + rho_g * ABS( A_s( ibreakp ) )
            ELSE
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'B', IBREAK( nbreak ) - m, BREAKP( nbreak )
               slope = slope + rho_b * ABS( S( ibreakp - m ) )
            END IF

!  If the last breakpoint has been passed, exit

            nbreak = nbreak - 1
            IF( nbreak == 0 ) EXIT

!  Determine if other terms become active at the breakpoint

            IF ( BREAKP( 1 ) >= feasep ) EXIT
            CALL SORT_heapsort_smallest( nbreak, BREAKP( : nbreak ), inheap,   &
                                    INDA = IBREAK )
            cluster_end = cluster_end + 1
          END DO

!  Compute the function value and gradient at (just on the other side of)
!  the breakpoint

          fun = val + t_break * ( slope_old + half * t_break * curv )
          gradient = slope + t_break * curv
          WRITE( cluster, "( 2I7 )" ) cluster_start, cluster_end

          IF ( print_detail ) THEN
            CALL QPA_p_val_and_slope( dims, n, m,                              &
                                      f, g_s, s_hs, rho_g, rho_b, X,           &
                                      X_l, X_u, RES_l, RES_u, S, A_s, t_break, &
                                      t_pert, too_small, REF, m_link, C_stat,  &
                                      B_stat, eval, eslope )
            write( out, 2010 ) '  val', fun, eval
            write( out, 2010 ) 'slope', gradient, eslope
          END IF

!  Fit the new quadratic so that it's value is fun at the breakpoint

          val = val + t_break * ( slope_old - slope )

!  Check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivative should be recomputed.

          IF ( ABS( slope ) < - epsqrt * slope_old ) THEN
            IF ( print_debug )                                                 &
              WRITE( out, "( ' recompute line derivative ... ' )" )
            CALL QPA_p_val_and_slope( dims, n, m,                              &
                                      f, g_s, s_hs, rho_g, rho_g, X,           &
                                      X_l, X_u, RES_l, RES_u, S, A_s, t_break, &
                                      t_pert, too_small, REF, m_link, C_stat,  &
                                      B_stat, val, slope )

            gradient = slope + t_break * curv
            IF ( print_debug )                                                 &
              WRITE( out, "( ' val, slope ', 2ES22.14 )" ) val, slope
          ENDIF

        ELSE

!  Special case: all the remaining breakpoints are reached

          IF ( print_detail )                                                  &
            WRITE( out, "( ' all remaining breakpoints reached' )" )

!  Compute the function value and gradient at (just on the other side of)
!  the breakpoint

          CALL QPA_p_val_and_slope( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X, &
                                    X_l, X_u, RES_l, RES_u, S, A_s, t_break,   &
                                    t_pert, too_small, REF, m_link, C_stat,    &
                                    B_stat, val, slope )
          gradient = slope + t_break * curv
          nbreak = 0
          cluster_end = nbreak_total
        END IF

        beyond_first_breakpoint = .TRUE.

!  See if the current cluster yields a sufficiently independent
!  constraint gradient

        active_ok = QPA_most_independent( m, n,                      &
                                          nbreak_total - cluster_end + 1,      &
                                          nbreak_total - cluster_start + 1,    &
                                          s_norm, IBREAK, lbreak, S, A_s,      &
                                          A_norms, cluster_best, cos_best,     &
                                          out, print_debug )

!  It does

        IF ( active_ok /= 0 ) THEN
          active = active_ok
          t_opt = BREAKP( cluster_best )
          inform = 1
          IF ( print_debug ) THEN
            IF ( ABS( active ) <= m ) THEN
              WRITE( out, 2030 ) 'C', ABS( active ), cos_best
            ELSE
              WRITE( out, 2030 ) 'B', ABS( active ) - m, cos_best
            END IF
          END IF
        END IF

!  ================
!  End of main loop
!  ================

      END DO

      IF ( inform == - 2 ) RETURN

!  Check to ensure that rounding has not caused an increase in the objective
!  value

      val = QPA_p_val( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X, X_l, X_u,    &
                       RES_l, RES_u, S, A_s, t_opt, too_small )
      recover =                                                                &
        f + rho_g * infeas_g + rho_b * infeas_b + epsmch ** 0.33 <= val

      IF ( .NOT. recover ) RETURN
      IF ( print_detail ) WRITE( out,                                          &
           "( ' *** predicted vs actual function values =', /, 2ES22.14,       &
  &        /, ' .... being more careful ... ' )" )                             &
        f + rho_g * infeas_g + rho_b * infeas_b, val

!  ==========================================================================
!  This part of the code is to cope with the possibility that rounding errors
!  have so dominated the search that a descent point has not been found. A
!  more cautious search will be performed.
!  ==========================================================================

      iter = 0 ; t_break = zero ; active = 0 ; t_opt = zero

!  Compute the initial function, slope and curvature

      CALL QPA_p_val_and_slope( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X,     &
                                X_l, X_u, RES_l, RES_u, S, A_s, 0.0_wp, t_pert,&
                                too_small, REF, m_link, C_stat, B_stat,        &
                                val, slope )
      curv = s_hs

!  Record the function value and gradient at (just on the other side of)
!  the initial point

      fun = val ; gradient = slope ; val_old = val

      nbreak = nbreak_total
      cluster_start = 1
      cluster_end = 0
      cluster = '      0      0'

!  =========================================================================
!  Start the main recovery loop to find the first local minimizer of the
!  piecewise quadratic function. Consider the problem over successive pieces
!  =========================================================================

      beyond_first_breakpoint = .FALSE.
      inform = - 1
      DO

!  ---------------------------------------------------------------
!  The piecewise quadratic function within the current interval is
!    val + slope * t + 0.5 * curv * t**2
!  ---------------------------------------------------------------

!  Print details of the piecewise quadratic in the next interval

        iter = iter + 1
        IF ( ( print_1line .AND. cluster_end == 0 ) .OR. print_detail )        &
          WRITE( out, 2000 )
        IF ( print_1line ) WRITE( out, "( 3X, I7, ES12.4, A14, 3ES12.4 )" )    &
           iter, t_break, cluster, fun, gradient, curv

!  If the gradient of the unvariate function increases, exit

        IF ( gradient > gzero ) THEN
          IF ( inform == 0 ) t_opt = t_break
          EXIT
        END IF

!  If the gradient of the univariate function is small and its curvature
!  is positive, exit

        IF ( ABS( gradient ) <= gzero ) THEN
          IF ( curv > - hzero ) THEN
            IF ( inform == 0 ) t_opt = t_break
            EXIT
          END IF
        END IF

!  Find the next breakpoint

        t_old = t_break
        IF ( nbreak > 0 ) THEN
          t_break = BREAKP( nbreak )
          cluster_end = cluster_end + 1
          cluster_start = cluster_end
        ELSE
          t_break = biginf
        END IF

!  If the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

        IF ( curv > zero ) THEN
          IF ( print_detail ) WRITE( out, "( ' slope, curv ', 2ES12.4 )" )     &
             slope,  curv
          t_star = - slope / curv

!  If the line minimum occurs before the breakpoint, the line minimum gives
!  the required minimizer. Exit

          IF ( nbreak == 0 .OR. t_star < t_break ) THEN
            t_opt = t_star
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star

!  Calculate the function value for the piecewise quadratic

            val = QPA_p_val( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X,        &
                             X_l, X_u, RES_l, RES_u, S, A_s, t_opt, too_small )

!  If the function value has risen in the current interval, search the
!  interval for a better value, and exit

            IF ( val_old < val ) THEN
              CALL QPA_linesearch_interval( dims, n, m,                        &
                                            f, g_s, s_hs, rho_g, rho_b,        &
                                            X, X_l, X_u, RES_l, RES_u, S, A_s, &
                                            t_old, val_old, t_opt, val,        &
                                            too_small, out, print_detail )
            END IF

            IF ( print_detail ) WRITE( out, 2000 )
            IF ( print_1line ) WRITE( out, &
              "( 3X, I7, ES12.4, A14, 3ES12.4 )" ) &
                 iter, t_opt, '      -      -', fun, zero, curv
            IF ( print_debug ) THEN
              eval = QPA_p_val( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X, X_l,&
                                X_u, RES_l, RES_u, S, A_s, t_opt, too_small )
              WRITE( out, 2010 ) '  val', fun, eval
            END IF
            IF ( beyond_first_breakpoint ) inform = 0
            EXIT
          ELSE
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star
          END IF
        ELSE

          IF ( print_detail ) WRITE( out, 2040 ) t_old, t_break

!  Exit if the function is unbounded from below

          IF ( nbreak == 0 ) THEN
            t_opt = biginf
            IF ( print_detail ) WRITE( out, 2000 )
            IF ( print_1line ) WRITE( out, &
                                      "( 3X, I7, 5X, 'inf', 22X, '-inf')" ) iter
            inform = - 2
            EXIT
          END IF

        END IF

!  Update the univariate function and slope values

        slope_old = slope

!  Record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

        pert_val = MAX( teneps, 0.001_wp * t_break )
        pert_eps = epsmch

        feasep = t_break + pert_val
        t_pert = pert_val + pert_eps
        t_break = feasep

        IF ( feasep < breakp_max ) THEN
          DO

!  Update the slope

            IF ( ibreakp <= m ) THEN
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'C', IBREAK( nbreak ), BREAKP( nbreak )
            ELSE
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'B', IBREAK( nbreak ) - m, BREAKP( nbreak )
            END IF

!  If the last breakpoint has been passed, exit

            nbreak = nbreak - 1
            IF( nbreak == 0 ) EXIT

!  Determine if other terms become active at the breakpoint

            IF ( BREAKP( nbreak ) >= feasep ) EXIT
            cluster_end = cluster_end + 1
          END DO

        ELSE

!  Special case: all the remaining breakpoints are reached

          nbreak = 0
          cluster_end = nbreak_total
        END IF

!  Compute the function value and gradient at (just on the other side of)
!  the breakpoint

        CALL QPA_p_val_and_slope( dims, n, m,                                  &
                                  f, g_s, s_hs, rho_g, rho_b, X, X_l,          &
                                  X_u, RES_l, RES_u, S, A_s, t_break, t_pert,  &
                                  too_small, REF, m_link, C_stat, B_stat,      &
                                  val, slope )
        gradient = slope + t_break * curv

        IF ( print_debug )                                                     &
          WRITE( out, "( ' val_old, val ', 2ES22.14 )" ) val_old, val
        IF ( val_old < val ) THEN
          t_opt = t_break
          CALL QPA_linesearch_interval( dims, n, m,                            &
                                        f, g_s, s_hs, rho_g, rho_b, X,         &
                                        X_l, X_u, RES_l, RES_u, S, A_s,        &
                                        t_old, val_old, t_opt, val,            &
                                        too_small, out, print_detail )
          EXIT
        END IF
        val_old = val

        beyond_first_breakpoint = .TRUE.

!  See if the current cluster yields a sufficiently independent
!  constraint gradient

        active_ok = QPA_most_independent( m, n,                                &
                                          nbreak_total - cluster_end + 1,      &
                                          nbreak_total - cluster_start + 1,    &
                                          s_norm, IBREAK, lbreak, S, A_s,      &
                                          A_norms, cluster_best, cos_best,     &
                                          out, print_debug )

!  It does

        IF ( active_ok /= 0 ) THEN
          active = active_ok
          t_opt = BREAKP( cluster_best )
          inform = 1
          IF ( print_debug ) THEN
            IF ( ABS( active ) <= m ) THEN
              WRITE( out, 2030 ) 'C', ABS( active), cos_best
            ELSE
              WRITE( out, 2030 ) 'B', ABS( active) - m, cos_best
            END IF
          END IF
        END IF

!  =========================
!  End of main recovery loop
!  =========================

      END DO

      RETURN

!  Non-executable statements

 2000 FORMAT( /, '  **  iter break point      cluster      ', &
                 ' val       slope        curv ', / )
 2010 FORMAT( 1X, A5, '(est,true) = ', 2ES22.14 )
 2020 FORMAT( ' breakpoint for ', A1, '-term ', I7, ' reached, step = ', ES12.4)
 2030 FORMAT( ' breakpoint for ', A1, '-term ', I7, ' is acceptable,',         &
              ' cosine = ', ES12.4 )
 2040 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, ']' )
 2050 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, &
              '], stationary point = ', ES12.4 )

!  End of subroutine QPA_linesearch

      END SUBROUTINE QPA_linesearch

!-*-   Q P A _ L I N E S E A R C H _ I N T E R V A L   S U B R O U T I N E   -*-

      SUBROUTINE QPA_linesearch_interval( dims, n, m,                          &
                                          f, g_s, s_hs, rho_g, rho_b, X,       &
                                          X_l, X_u,  RES_l, RES_u, S, A_s,     &
                                          t_lower, val_lower, t_opt, val_opt,  &
                                          too_small, out, print_debug )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the local minimizer of the function
!
!     1/2 x(T) H x + c(T) x
!        + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!        + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  along the arc x(t) = x + t s  in the interval (t_lower, t_opt)
!  returning the minimizer in t_opt,
!  where x is a vector of n components ( x_1, .... , x_n ),
!  H is a symmetric matrix, and A is an m by n matrix.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, out
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho_g, rho_b, too_small
      REAL ( KIND = wp ), INTENT( INOUT ) :: t_lower, val_lower, t_opt, val_opt
      LOGICAL, INTENT( IN ) :: print_debug
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, S
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u

!  Local variables

      REAL ( KIND = wp ) :: t_new, val_new, min_interval
      min_interval = epsmch ** 0.25

      IF ( print_debug )                                                       &
        WRITE( out, "( '      t_lower                t          ',             &
     &                 '|    val_lower              val         ' )" )

!  A dumb bisection algorithm - this shouldn't get used very often,
!  so this is probably not catastrophic (famous last words)!

      DO
        IF ( print_debug )                                                     &
          WRITE( out, "( 4ES20.12 )" ) t_lower, t_opt, val_lower, val_opt
        IF ( ( t_opt - t_lower <= min_interval .AND. t_lower > zero ) .OR.     &
               t_opt - t_lower <= epsmch ) THEN
          IF ( val_opt > val_lower ) THEN
            t_opt = t_lower ; val_opt = val_lower
          END IF
          EXIT
        END IF
        t_new = half * ( t_lower + t_opt )
        val_new = QPA_p_val( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X, X_l,   &
                             X_u, RES_l, RES_u, S, A_s, t_new, too_small )
        IF ( val_opt < val_lower ) THEN
          t_lower = t_new ; val_lower = val_new
        ELSE
          t_opt = t_new ; val_opt = val_new
        END IF
      END DO

      RETURN

!  End of subroutine QPA_linesearch_interval

      END SUBROUTINE QPA_linesearch_interval

!-*-*-*-   Q P A _ M O S T _ I N D E P E N D E N T   F U N C T I O N   -*-*-*-

      FUNCTION QPA_most_independent( m, n, cluster_start, cluster_end, s_norm, &
                                     IBREAK, lbreak, S, A_s, A_norms,          &
                                     cluster_best, cos_best, out, print_debug )

      INTEGER :: QPA_most_independent

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the index of the gradient a_i which has the largest cosine
!  with a direction s amongst those whose indices lie in the
!  set IBREAK( i ), i = cluster_start, cluster_end. The directional
!  derivatives <a_i,s>, and the norms of the a_i are input in A_s and A_norms
!  while ||s||_2 is input in s_norm
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, lbreak, cluster_start, cluster_end, out
      INTEGER, INTENT( OUT ) :: cluster_best
      LOGICAL, INTENT( IN ) :: print_debug
      REAL ( KIND = wp ), INTENT( IN ) :: s_norm
      INTEGER, INTENT( IN ), DIMENSION( lbreak ) :: IBREAK
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s, A_norms
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
      REAL ( KIND = wp ), INTENT( OUT ) :: cos_best

!  Local variables

      INTEGER :: i, j
      REAL ( KIND = wp ) :: cos_current, small_cos, small_cosine

      small_cosine = epsmch ** 0.4
      cos_best = zero
      cluster_best = 0
      QPA_most_independent = 0
      small_cos = s_norm * small_cosine

!  Run through the cluster, ignoring any terms that result in small cosines

      DO i = cluster_start, cluster_end
        j = ABS( IBREAK( i ) )
        IF ( j <= m ) THEN
          cos_current = ABS( A_s( j ) / A_norms( j ) )
        ELSE
          cos_current = ABS( S( j - m ) )
        END IF

        IF ( print_debug )                                                     &
          WRITE( out, "( ' cosine for term ', I5, ' is ', ES12.4 )" )          &
            j, cos_current / s_norm

        IF ( cos_current < small_cos ) CYCLE

!  The normal of this member of the cluster currently makes the largest
!  cosine with the search direction

        IF ( cos_current > cos_best ) THEN
          cos_best = cos_current
          cluster_best = i
          QPA_most_independent = IBREAK( i )
        END IF
      END DO
      cos_best =  cos_best / s_norm
      IF ( print_debug )                                                       &
        WRITE( out, "( ' cosine and s for term ', I5, ' is ', 2ES12.4 )" )     &
          ABS( QPA_most_independent ), cos_best, s_norm

      RETURN

!  End of function QPA_most_independent

      END FUNCTION QPA_most_independent

!-*-*-*-*-*-*-*-*-*-   Q P A _ P _ V A L   F U N C T I O N   -*-*-*-*-*-*-*-*-*-

      FUNCTION QPA_p_val( dims, n, m, f, g_s, s_hs, rho_g, rho_b, X,           &
                          X_l, X_u, RES_l, RES_u, S, A_s, t, too_small )
      REAL ( KIND = wp ) QPA_p_val

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the penalty function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho_g, rho_b
      REAL ( KIND = wp ), INTENT( IN ) :: t, too_small
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, S
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: as, infeas_g, infeas_b

      infeas_g = zero ; infeas_b = zero

!  equality constraints

      DO i = 1, dims%c_equality
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g + ABS( RES_l( i ) )
        ELSE
          infeas_g = infeas_g + ABS( RES_l( i ) + t * as )
        END IF
      END DO

!  constraints with lower bounds

      DO i = dims%c_equality + 1, dims%c_l_end
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g - MIN( RES_l( i ), zero )
        ELSE
          infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
        END IF
      END DO

!  constraints with upper bounds

      DO i = dims%c_u_start, m
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g - MIN( RES_u( i ), zero )
        ELSE
          infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
        END IF
      END DO

!  simple non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X( i ) + t * as, zero )
        END IF
      END DO

!  simple bound from below

      DO i = dims%x_l_start, dims%x_l_end
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X( i ) - X_l( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X( i ) - X_l( i ) + t * as, zero )
        END IF
      END DO

!  simple bound from above

      DO i = dims%x_u_start, dims%x_u_end
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X_u( i ) - X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X_u( i ) - X( i ) - t * as, zero )
        END IF
      END DO

!  simple non-positivity

      DO i = dims%x_u_end + 1, n
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( - X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( - X( i ) - t * as, zero )
        END IF
      END DO

      QPA_p_val = f + t * ( g_s + half * t * s_hs )                            &
                    + rho_g * infeas_g + rho_b * infeas_b

      RETURN

!  End of function QPA_p_val

      END FUNCTION QPA_p_val

!-*-*-*-   Q P A _ P _ V A L _ A N D _ S L O P E   S U B R O U T I  N E -*-*-*-

      SUBROUTINE QPA_p_val_and_slope( dims, n, m, f, g_s, s_hs, rho_g, rho_b,  &
                                      X, X_l, X_u, RES_l, RES_u, S, A_s, t,    &
                                      t_pert, too_small, REF, m_link,          &
                                      C_stat, B_stat, val, slope )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value and slope (in the direction S) of the penalty function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( IN ) :: m_link
      INTEGER, INTENT( IN ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( IN ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( IN ), DIMENSION( n ) :: B_stat
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho_g, rho_b
      REAL ( KIND = wp ), INTENT( IN ) :: t, t_pert, too_small
      REAL ( KIND = wp ), INTENT( OUT ) :: val, slope
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, S
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_end ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_u_start : m ) ::  RES_u

!  Local variables

      INTEGER :: i, l
      REAL ( KIND = wp ) :: tp, as, infeas_g, infeas_b, slope_g, slope_b

      IF ( exact ) THEN
        tp = t
      ELSE
        tp = t + t_pert
      END IF
      infeas_g = zero ; slope_g = zero ; infeas_b = zero ; slope_b = zero

!  equality constraints

      DO i = 1, dims%c_equality
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g + ABS( RES_l( i ) )
        ELSE
          infeas_g = infeas_g + ABS( RES_l( i ) + t * as )
          l = C_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( RES_l( i ) + tp * as < zero ) THEN
            slope_g = slope_g - as
          ELSE IF ( RES_l( i ) + tp * as > zero ) THEN
            slope_g = slope_g + as
          END IF
        END IF
      END DO

!  constraints with lower bounds

      DO i = dims%c_equality + 1, dims%c_l_end
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g - MIN( RES_l( i ), zero )
        ELSE
          infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
          l = C_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( RES_l( i ) + tp * as < zero ) THEN
            slope_g = slope_g - as
          END IF
        END IF
      END DO

!  constraints with upper bounds

      DO i = dims%c_u_start, m
        as = A_s( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_g = infeas_g - MIN( RES_u( i ), zero )
        ELSE
          infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
          l = C_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( RES_u( i ) - tp * as < zero ) THEN
            slope_g = slope_g + as
          END IF
        END IF
      END DO

!  simple non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X( i ) + t * as, zero )
          l = B_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( X( i ) + tp * as < zero ) THEN
            slope_b = slope_b - as
          END IF
        END IF
      END DO

!  simple bound from below

      DO i = dims%x_l_start, dims%x_l_end
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X( i ) - X_l( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X( i ) - X_l( i ) + t * as, zero )
          l = B_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( X( i ) - X_l( i ) + tp * as < zero ) THEN
            slope_b = slope_b - as
          END IF
        END IF
      END DO

!  simple bound from above

      DO i = dims%x_u_start, dims%x_u_end
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( X_u( i ) - X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( X_u( i ) - X( i ) - t * as, zero )
          l = B_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( X_u( i ) - X( i ) - tp * as < zero ) THEN
            slope_b = slope_b + as
          END IF
        END IF
      END DO

!  simple non-positivity

      DO i = dims%x_u_end + 1, n
        as = S( i )
        IF ( ABS( as ) < too_small ) THEN
          infeas_b = infeas_b - MIN( - X( i ), zero )
        ELSE
          infeas_b = infeas_b - MIN( - X( i ) - t * as, zero )
          l = B_stat( i )
          IF ( l > 0 ) CYCLE
          IF ( l < 0 ) THEN
            IF ( REF( - l ) > 0 ) CYCLE
          END IF
          IF ( - X( i ) - tp * as < zero ) THEN
            slope_b = slope_b + as
          END IF
        END IF
      END DO

      val = f + t * ( g_s + half * t * s_hs )                                  &
              + rho_g * infeas_g + rho_b * infeas_b
      slope = ( g_s + t * s_hs ) + rho_g * slope_g + rho_b * slope_b

      RETURN

!  End of subroutine QPA_p_val_and_slope

      END SUBROUTINE QPA_p_val_and_slope

!-*-*-   Q P A _ R E M O V E _ D E P E N D E N T  S U B R O U T I N E   -*-*-

      SUBROUTINE QPA_remove_dependent( n, m, A_val, A_col, A_ptr, K,           &
                                       SLS_data, SLS_control, C_stat, B_stat,  &
                                       WORKING, P, SOL, D, prefix, control,    &
                                       inform, n_depen, C_given, X_given )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Determine which, if any, of the gradients of constraint in the working set
!  are dependent and, optionally if so, if they are consistent with a given
!  right-hand side
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: n_depen
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( IN ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( OUT ), DIMENSION( m ) :: WORKING
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: C_given
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: X_given

!  allocatable arrays

      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: P
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( :, : ) :: D
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, jj, l, ll, nroots, out, n_free, m_working
      INTEGER :: piv, pmax, pmin
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: root1, root2, dmax, dmin, rmax, rmin, big, diag
      REAL ( KIND = wp ) :: u, tolerance, scale, res, res_max, a_abs
      LOGICAL ::  twobytwo, scale_a, rhs
      CHARACTER ( LEN = 80 ) :: array_name

      out = control%out

!  Analyse the sparsity pattern. Find the number of free variables

      n_free = COUNT( B_stat == 0 )

!  Find the number of constraints in the working set, and set the dimensions
!  of K. Also record the indices of the working constraints

      m_working = 0
      K%ne = n_free
      DO i = 1, m
        IF ( C_stat( i ) /= 0 ) THEN
          m_working = m_working + 1
          WORKING( m_working ) = i
          DO j =  A_ptr( i ), A_ptr( i + 1 ) - 1
            IF ( B_stat( A_col( j ) ) == 0 ) K%ne = K%ne + 1
          END DO
        END IF
      END DO
      K%n = n_free + m_working

      IF ( out > 0 .AND. control%print_level >= 1 ) THEN
        WRITE( out, 2000 ) prefix, m_working, m, prefix, n - n_free, n
        WRITE( out,                                                            &
         "( /, A, 7( ' -' ), ' test for rank defficiency ', 6( ' - ' ) )" )    &
           prefix
      END IF

!  Special case: all variables on bounds

      IF ( n_free == 0 ) THEN
        IF ( m_working /= 0 ) THEN
          IF ( out > 0 .AND. control%print_level >= 1 )                        &
            WRITE( out, "( /, A, '  all general constraints are dependent' )" )&
              prefix
          C_stat = 0
        END IF
        n_depen = m_working
        GO TO 800
      END IF

!  Allocate temporary workspace

      array_name = 'qpa: data%P'
      CALL SPACE_resize_array( MAX( n, K%n ), P, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      rhs = PRESENT( X_given ) .AND. PRESENT( C_given )

      IF ( rhs ) THEN
        array_name = 'qpa: data%SOL'
        CALL SPACE_resize_array( K%n, SOL, inform%status,                      &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  Order the free variables

      n_free = 0
      DO i = 1, n
        IF ( B_stat( i ) == 0 ) THEN
          n_free = n_free + 1
          P( i ) = n_free
        ELSE
          P( i ) = 0
        END IF
      END DO

!  Allocate the arrays for the analysis phase

      array_name = 'qpa: data%K%row'
      CALL SPACE_resize_array( K%ne, K%row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%K%col'
      CALL SPACE_resize_array( K%ne, K%col, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%K%val'
      CALL SPACE_resize_array( K%ne, K%val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Put A into K and compute the right-hand side to verify consistency

      scale_a = .TRUE.
      ll = n_free ; ii = n_free
      DO i = 1, m
        IF ( C_stat( i ) /= 0 ) THEN
          ii = ii + 1
          IF ( rhs ) SOL( ii ) = C_given( i )
          scale = one
          IF ( scale_a ) THEN
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( P( A_col( l ) ) > 0 ) scale = MAX( scale, ABS( A_val( l ) ) )
            END DO
          END IF
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            j = A_col( l )
            jj = P( j )
            IF ( jj > 0 ) THEN
              ll = ll + 1
              K%row( ll ) = ii ; K%col( ll ) = jj
              K%val( ll ) = A_val( l ) / scale
            ELSE
              IF ( rhs ) SOL( ii ) = SOL( ii ) - A_val( l ) * X_given( j )
            END IF
          END DO
          IF ( rhs ) SOL( ii ) = SOL( ii ) / scale
        END IF
      END DO

!  Put the diagonal into K

      IF ( n_free + 1 <= K%ne ) THEN
        diag = one
        DO i = n_free + 1, K%ne
          a_abs = ABS( K%val( i ) )
          IF ( a_abs > zero )  diag = MIN( diag, a_abs )
        END DO
      ELSE
        diag = one
      END IF
      diag = MAX( SQRT( epsmch ), diag )
      DO i = 1, n_free
!       K%row( i ) = i ; K%col( i ) = i ; K%val( i ) = one
        K%row( i ) = i ; K%col( i ) = i ; K%val( i ) = diag
      END DO

!     write(6,*) K%n, K%ne
!     DO i = 1,  K%ne
!       write(6,"( 2I8, ES22.14 )" ) K%row( i ), K%col( i ), K%val( i )
!     END DO

!  Analyse the sparsity pattern of the matrix

      CALL SMT_put( K%type, 'COORDINATE', i )

      piv = SLS_control%pivot_control
      u = SLS_control%relative_pivot_tolerance
      tolerance = SLS_control%zero_tolerance
      SLS_control%pivot_control = 1
      SLS_control%relative_pivot_tolerance = control%pivot_tol_for_dependencies
      SLS_control%zero_tolerance = epsmch ** 2

      CALL SLS_initialize_solver( control%symmetric_linear_solver,             &
                                  SLS_data, inform%SLS_inform )
      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_analyse( K, SLS_data, SLS_control, inform%SLS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%analyse = inform%time%analyse + time_now - time_record
      inform%time%clock_analyse =                                              &
        inform%time%clock_analyse + time_now - time_record
      inform%factorization_status = inform%SLS_inform%status
      inform%factorization_integer = inform%SLS_inform%integer_size_factors
      inform%factorization_real = inform%SLS_inform%real_size_factors

!  Check for error returns

      IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,                &
         "(  A, ' SLS: analysis complete:      status = ', I0 )" )             &
             prefix, inform%SLS_inform%status
      IF ( inform%factorization_status < 0 ) THEN
         inform%status = GALAHAD_error_analysis ; GO TO 990
      ELSE IF ( inform%factorization_status > 0 ) THEN
        IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,              &
           "( /, A, ' ** Warning ', I0, ' from SLS_analyse' )" )               &
          prefix, inform%factorization_status
      END IF

      IF ( out > 0 .AND. control%print_level >= 2 )                            &
        WRITE( out, "( A, ' ** analysis time = ', F10.2, /, A,                 &
       &     ' real/integer space required for factors ', 2I0 )" )             &
          prefix, time_now - time_record,                                      &
          prefix, inform%factorization_real, inform%factorization_integer

!  Factorize the matrix

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_factorize( K, SLS_data, SLS_control, inform%SLS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_record
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + time_now - time_record
      inform%factorization_status = inform%SLS_inform%status

!  Record the storage required

      inform%nfacts = inform%nfacts + 1
      inform%factorization_integer = inform%SLS_inform%integer_size_factors
      inform%factorization_real = inform%SLS_inform%real_size_factors

!  Test that the factorization succeeded

      IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,                &
        "( A, ' SLS: factorization complete: status = ', I0 )" ) prefix,       &
           inform%SLS_inform%status
      IF ( inform%SLS_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_factorization ; GO TO 990
      END IF

      IF ( inform%SLS_inform%rank < K%n .AND.                                  &
          out > 0 .AND. control%print_level >= 1)                              &
        WRITE( out, "( A, ' ** Matrix has ', I0, ' zero eigenvalue', A )" )    &
          prefix, K%n - inform%SLS_inform%rank,                                &
          TRIM( STRING_pleural( inform%SLS_inform%rank ) )

      IF ( control%out > 0 .AND. control%print_level >= 2 )                    &
        WRITE( control%out, "( A, ' ** factorize time = ', F10.2 ) " )         &
          prefix, time_now - time_record

! Allocate temporary workspace

      array_name = 'qpa: data%D'
      CALL SPACE_resize_array( 2, K%n, D, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Determine the block diagonal part of the factors and the pivot order

      CALL SLS_enquire( SLS_data, inform%SLS_inform, PIVOTS = P, D = D )

!  Find the largest diagonal (of the inverse)

      dmax = zero  ; twobytwo = .FALSE.

!  Loop over the diagonal blocks

      DO i = 1, inform%SLS_inform%rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i < inform%SLS_inform%rank ) THEN

!  A 2x2 block

          IF ( P( i ) < 0 ) THEN
            twobytwo = .TRUE.
            CALL ROOTS_quadratic( D( 1, i ) * D( 1, i + 1 ) - D( 2, i ) ** 2,  &
              - D( 1, i ) - D( 1, i + 1 ), one, epsmch, nroots, root1, root2,  &
              roots_debug )
            IF ( root1 /= zero ) dmax = MAX( ABS( one / root1 ), dmax )
            IF ( root2 /= zero ) dmax = MAX( ABS( one / root2 ), dmax )

!  A 1x1 block

          ELSE
            IF ( D( 1, i ) /= zero ) dmax = MAX( ABS( one / D( 1, i ) ), dmax )
          END IF
        ELSE

!  The final 1x1 block

          IF ( D( 1, i ) /= zero ) dmax = MAX( ABS( one / D( 1, i ) ), dmax )
        END IF
      END DO
      IF ( out > 0 .AND. control%print_level >= 3 )                            &
          WRITE( out, "( A, ' largest diagonal = ', ES12.4 )" ) prefix, dmax

!  Compute the smallest and largest eigenvalues of the block diagonal factor

      n_depen = 0 ; twobytwo = .FALSE.
      big = one / ( MAX( dmax, one ) * MAX( control%zero_pivot, epsmch ) )

      dmax = zero ; dmin = HUGE( one )

!  Loop over the diagonal blocks

      DO i = 1, inform%SLS_inform%rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i < inform%SLS_inform%rank ) THEN

!  A 2x2 block

          IF ( P( i ) < 0 ) THEN
            twobytwo = .TRUE.

            CALL ROOTS_quadratic( D( 1, i ) * D( 1, i + 1 ) - D( 2, i ) ** 2,  &
              - D( 1, i ) - D( 1, i + 1 ), one, epsmch, nroots, root1, root2,  &
              roots_debug )
            rmax = MAX( ABS( root1 ), ABS( root2 ) )
            dmax = MAX( rmax, dmax )
            rmin = MIN( ABS( root1 ), ABS( root2 ) )
            dmin = MIN( rmin, dmin )

            pmax = MAX( ABS( P( i ) ), ABS( P( i + 1 ) ) )
            pmin = MIN( ABS( P( i ) ), ABS( P( i + 1 ) ) )

            IF ( pmin <= n_free ) THEN
              IF ( rmax >= big ) THEN
                n_depen = n_depen + 1
                C_stat( WORKING( pmax - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE(  out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )   &
                   pmax - n_free, pmin - n_free, one / rmax
                END IF
              ELSE IF ( rmin == zero ) THEN
                n_depen = n_depen + 1
                C_stat( WORKING( pmax - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = infinity' )" )    &
                   pmax - n_free, pmin - n_free
                END IF
              ELSE
                IF ( out > 0 .AND. control%print_level >= 5 )                  &
                  WRITE(  out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )   &
                   pmax - n_free, pmin - n_free, one / rmax
              END IF

              IF ( rmin >= big ) THEN
                n_depen = n_depen + 1
                C_stat( WORKING( pmin - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )    &
                    pmin - n_free, pmax - n_free, one / rmin
                END IF
              ELSE IF ( rmax == zero ) THEN
                n_depen = n_depen + 1
                C_stat( WORKING( pmin - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = infinity' )" )    &
                    pmin - n_free, pmax - n_free
                END IF
              ELSE
                IF ( out > 0 .AND. control%print_level >= 5 )                  &
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )    &
                    pmin - n_free, pmax - n_free, one / rmin
              END IF
            ELSE
              IF ( rmax >= big .OR. rmin == zero ) THEN
                n_depen = n_depen + 1
                C_stat( WORKING( pmax - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE(  out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )   &
                   pmax - n_free, pmin - n_free, one / rmax
                END IF
                n_depen = n_depen + 1
                C_stat( WORKING( pmin - n_free ) ) = 0
                IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = infinity' )" )    &
                    pmin - n_free, pmax - n_free
                END IF
              ELSE
                IF ( out > 0 .AND. control%print_level >= 5 )                  &
                  WRITE(  out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )   &
                   pmax - n_free, pmin - n_free, one / rmax
                IF ( out > 0 .AND. control%print_level >= 5 )                  &
                  WRITE( out, "( '2x2 block ', 2i7, ' eval = ', ES12.4 )" )    &
                    pmin - n_free, pmax - n_free, one / rmin
              END IF
            END IF

            IF ( rhs ) THEN
              IF ( ABS( root2 ) >= big .OR.                                    &
                   root1 == zero .OR. root2 == zero ) THEN
                IF ( ABS( root1 ) >= big .OR.                                  &
                     ( root1 == zero .AND. root2 == zero ) ) THEN
                  D( 1, i ) = zero ;  D( 2, i ) = zero ; D( 1, i + 1 ) = zero
                END IF
              END IF
            END IF

!  A 1x1 block

          ELSE
            dmax = MAX( ABS( D( 1, i ) ), dmax )
            dmin = MIN( ABS( D( 1, i ) ), dmin )
            IF ( ABS( D( 1, i ) ) >= big ) THEN
              n_depen = n_depen + 1
              C_stat( WORKING( P( i ) - n_free ) ) = 0
              IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                WRITE( out, "( '1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )   &
                  P( i ) - n_free,  one / ABS( D( 1, i ) )
              END IF
              IF ( rhs ) D( 1, i ) = zero
            ELSE IF ( D( 1, i ) == zero ) THEN
              n_depen = n_depen + 1
              C_stat( WORKING( P( i ) - n_free ) ) = 0
              IF ( out > 0 .AND. control%print_level >= 3 ) THEN
                WRITE( out, "( '1x1 block ', i7, 7x, ' eval = infinity' )" )   &
                  P( i ) - n_free
              END IF
            ELSE
              IF ( out > 0 .AND. control%print_level >= 5 )                    &
                WRITE( out, "( '1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )   &
                  P( i ) - n_free,  one / ABS( D( 1, i ) )
            END IF
          END IF
        ELSE

!  The final 1x1 block

          dmax = MAX( ABS( D( 1, i ) ), dmax )
          dmin = MIN( ABS( D( 1, i ) ), dmin )
          IF ( ABS( D( 1, i ) ) >= big ) THEN
            n_depen = n_depen + 1
            C_stat( WORKING( P( i ) - n_free ) ) = 0
            IF ( out > 0 .AND. control%print_level >= 3 ) THEN
              WRITE( out, "( '1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )     &
                P( i ) - n_free, one / ABS( D( 1, i ) )
            END IF
            IF ( rhs ) D( 1, i ) = zero
          ELSE IF ( D( 1, i ) == zero ) THEN
            n_depen = n_depen + 1
            C_stat( WORKING( P( i ) - n_free ) ) = 0
            IF ( out > 0 .AND. control%print_level >= 3 ) THEN
              WRITE( out, "( '1x1 block ', i7, 7x, ' eval = infinity ' )" )    &
                P( i ) - n_free
            END IF
          ELSE
            IF ( out > 0 .AND. control%print_level >= 5 )                      &
              WRITE( out, "( '1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )     &
                P( i ) - n_free, one / ABS( D( 1, i ) )
          END IF
        END IF
      END DO

!  Any null blocks

      DO i = inform%SLS_inform%rank + 1, K%n
         n_depen = n_depen + 1
         C_stat( WORKING( P( i ) - n_free ) ) = 0
!        IF ( rhs ) D( 1, i ) = zero
         dmax = HUGE( one )
         IF ( out > 0 .AND. control%print_level >= 3 )                         &
          WRITE( out, "( '1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )         &
            P( i ) - n_free, zero
      END DO

      IF ( out > 0 .AND. control%print_level >= 1 ) THEN
        IF ( dmin == zero .OR. dmax == zero ) THEN
          WRITE( out,"( A,' 1/ smallest,largest block eigenvalues =',2ES12.4)")&
            prefix, dmin, dmax
        ELSE
          WRITE( out, "( A, ' smallest,largest block eigenvalues =',2ES12.4)") &
            prefix, one / dmax, one / dmin
        END IF
        WRITE( out,"( A, 1X, I0, ' constraint', A, ' appear to be dependent')")&
          prefix, n_depen, TRIM( STRING_pleural( n_depen ) )
      END IF

      IF ( rhs ) THEN

!  Reset "small" pivots to zero

!       CALL SLS_alter_d( SLS_data, D, inform%SLS_inform )

!  Check to see if the constraints are consistent

        SOL( : n_free ) = zero

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_solve( K, SOL, SLS_data, control%SLS_control,                 &
                        inform%SLS_inform )
        CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_solve + time_now - time_record

!  Reorder the free variables

        n_free = 0
        DO i = 1, n
          IF ( B_stat( i ) == 0 ) THEN
            n_free = n_free + 1
            P( i ) = n_free
          ELSE
            P( i ) = 0
          END IF
        END DO

!  Compute the residuals

        res_max = zero
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN
            res = C_given( i )
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              j = A_col( l )
              jj = P( j )
              IF ( jj > 0 ) THEN
                res = res - A_val( l ) * SOL( jj )
              ELSE
                res = res - A_val( l ) * X_given ( j )
              END IF
            END DO
            res_max = MAX( res_max, ABS( res ) )
          END IF
        END DO

        IF ( res_max <= control%feas_tol ) THEN
          IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,            &
            "( A, ' constraints are consistent: maximum infeasibility = ',     &
          &    ES12.4 )" ) prefix, res_max
        ELSE
          IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,            &
            "( A, ' constraints are inconsistent: maximum infeasibility = ',   &
          &    ES12.4, /, A, 31X, ' is larger than control%feas_tol ' )" )     &
            prefix, res_max, prefix
          inform%status = GALAHAD_error_primal_infeasible ; GO TO 800
        END IF

      END IF

!  Reset the pivot tolerance and zero pivoting threshold to their initial values

  800 CONTINUE
      inform%status = GALAHAD_ok

      SLS_control%pivot_control = piv
      SLS_control%relative_pivot_tolerance = u
      SLS_control%zero_tolerance = tolerance

      IF ( out > 0 .AND. control%print_level >= 1 ) THEN
        WRITE( out, "( /, A, 6( ' -' ), ' end of test for rank defficiency',   &
       &               5( ' - ' ) )" ) prefix
        WRITE( out, 2000 ) prefix, m_working - n_depen, m, prefix, n - n_free, n
      END IF

      RETURN

!  Allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      IF ( out > 0 .AND. control%print_level >= 1 )                            &
       WRITE( out, "( ' ** Message from -QPA_remove_dependent-',               &
     &   /, ' Allocation error, for ', A, /, ' status = ', I0 ) " )            &
        inform%bad_alloc, inform%alloc_status

!  Error return

  990 CONTINUE
      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, 1X, I0, ' in working set from ', I0, ' constraints',       &
              /, A, 1X, I0, ' in working set from ', I0, ' simple bounds' )

!  End of QPA_remove_dependent

      END SUBROUTINE QPA_remove_dependent

!-*-  Q P A _ I T E R A T I V E _ R E F I N E M E N T  S U B R O U T I N E -*-

      SUBROUTINE QPA_iterative_refinement( K, SCU_mat, SCU_data, RHS, X,       &
                                           B, RES, DX, VECTOR, SLS_control,    &
                                           SLS_data, K_part, zeq0, itref_max,  &
                                           out, print, RES_print, inform )

!  Solve the system of equations

!     / K  C^T \ / x1 \ _ / rhs1 \
!     \ C   D  / \ x2 / ~ \ rhs2 /

!  using the Schur complement method. The vectors
!  B, RES and DX are used as workspace

!  Dummy arguments

      INTEGER, INTENT( IN ) :: itref_max, out
      LOGICAL, INTENT( IN ) :: zeq0, print
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( SCU_matrix_type ), INTENT( IN ) :: SCU_mat
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION ( SCU_mat%n + SCU_mat%m ) :: RHS
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION ( SCU_mat%n + SCU_mat%m ) :: X, B, RES, DX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( SCU_mat%n ) :: VECTOR
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( K_part%n_free + K_part%c_ref ) :: RES_print
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, l, jumpto, scu_status, iter
      REAL ( KIND = wp ) :: val
      LOGICAL :: x_r0, y0, x_f0, z0
      LOGICAL :: x_r, y, x_f, z

!  Solve the linear system

      IF ( print ) WRITE( out, "( ' norm rhs      = ', ES12.4 )" )             &
        MAXVAL( ABS( RHS ) )

      scu_status = 1 ; jumpto = 1
      DO
        CALL SCU_solve( SCU_mat, SCU_data, RHS, X, VECTOR, scu_status )
        IF ( scu_status <= 0 ) THEN
          inform%factorization_status = scu_status
          EXIT
        END IF
        IF ( jumpto == 1 .AND. zeq0 ) THEN
          x_r0 = .FALSE. ; y0 = .TRUE. ; x_f0 = .FALSE. ; z0 = .TRUE.
        ELSE
          x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.
        END IF

        CALL QPA_ir( K, SLS_data, K_part, VECTOR, B, RES, x_r0, y0, x_f0, z0,  &
                     SLS_control, itref_max - 1, out, print, RES_print, inform )
        IF ( inform%status /= 0 ) THEN
          inform%factorization_status = inform%status
          EXIT
        END IF
        jumpto = jumpto + 1
      END DO

!  Refine the solution

      DO iter = 1, itref_max

!  Compute the residuals

        x_r = .FALSE. ; y = .FALSE. ; x_f = .FALSE. ; z = .FALSE.
        CALL QPA_K_residuals( K, K_part, X, RHS, RES, x_r, y, x_f, z )
        RES( SCU_mat%n + 1 : SCU_mat%n + SCU_mat%m ) =                         &
         RHS( SCU_mat%n + 1 : SCU_mat%n + SCU_mat%m )

        DO ii = 1, SCU_mat%m
          i = SCU_mat%n + ii
          DO l = SCU_mat%BD_col_start( ii ),                                   &
                 SCU_mat%BD_col_start( ii + 1 ) - 1
            j = SCU_mat%BD_row( l )
            val = SCU_mat%BD_val( l )
            IF ( j > SCU_mat%n + SCU_mat%m )                                   &
              write( 6,*) ' ----- ', j, SCU_mat%n + SCU_mat%m
            IF ( j > SCU_mat%n + SCU_mat%m ) stop
            RES( i ) = RES( i ) - val * X( j )
            IF ( i /= j ) RES( j ) = RES( j ) - val * X( i )
          END DO
        END DO
        IF ( print ) WRITE( out, "( ' norm residual = ', ES12.4 )" )           &
            MAXVAL( ABS( RES ) )

!  Find the correction

        scu_status = 1 ; jumpto = 1
        DO
          CALL SCU_solve( SCU_mat, SCU_data, RES, DX, VECTOR, scu_status )
          IF ( scu_status <= 0 ) THEN
            inform%factorization_status = scu_status
            EXIT
          END IF
          IF ( jumpto == 1 .AND. zeq0 ) THEN
            x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .TRUE.
          ELSE
            x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.
          END IF
          CALL QPA_ir( K, SLS_data, K_part, VECTOR, B, RES, x_r0, y0, x_f0,    &
                       z0, SLS_control, itref_max - 1, out, print, RES_print,  &
                       inform )
          IF ( inform%status /= 0 ) THEN
            inform%factorization_status = inform%status
            EXIT
          END IF
          jumpto = jumpto + 1
        END DO

!  Refine the solution using the correction

        X = X + DX
      END DO

!  Compute the final residuals

      IF ( print ) THEN
        x_r = .FALSE. ; y = .FALSE. ; x_f = .FALSE. ; z = .FALSE.
        CALL QPA_K_residuals( K, K_part, X, RHS, RES, x_r, y, x_f, z )
        RES( SCU_mat%n + 1 : SCU_mat%n + SCU_mat%m ) =                         &
         RHS( SCU_mat%n + 1 : SCU_mat%n + SCU_mat%m )

        DO ii = 1, SCU_mat%m
          i = SCU_mat%n + ii
          DO l = SCU_mat%BD_col_start( ii ),                                   &
                 SCU_mat%BD_col_start( ii + 1 ) - 1
            j = SCU_mat%BD_row( l )
            val = SCU_mat%BD_val( l )
            RES( i ) = RES( i ) - val * X( j )
            IF ( i /= j ) RES( j ) = RES( j ) - val * X( i )
          END DO
        END DO

        WRITE( out, "( ' norm residual = ', ES12.4 )" ) MAXVAL( ABS( RES ) )
      END IF

      RETURN

!  End of QPA_iterative_refinement

      END SUBROUTINE QPA_iterative_refinement

!-*-*-*-*-*-*-*-*-*-  Q P A _ _ i r  S U B R O U T I N E -*-*-*-*-*-*-*-*-*-

      SUBROUTINE QPA_ir( K, SLS_data, K_part, X, B, RES, x_r0, y0, x_f0, z0,   &
                         SLS_control, itref_max, out, print, RES_print, inform )

!  Solve the reference linear system K x = b, with b input in X, and x
!  output in X, using iterative refinement. B and RES are used as workspace

!  Dummy arguments

      INTEGER, INTENT( IN ) :: itref_max, out
      LOGICAL, INTENT( IN ) :: print
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( K_part%k_ref ) :: X
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( K_part%k_ref ) :: B, RES
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( K_part%n_free + K_part%c_ref ) :: RES_print
      LOGICAL, INTENT( IN ) :: x_r0, y0, x_f0, z0
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: iter
      LOGICAL :: x_r, y, x_f, z, xs_r, ys, xs_f, zs

!  Solve the linear system

      IF ( itref_max >= 1 .OR. print ) B = X
      x_r = x_r0 ; y = y0 ; x_f = x_f0 ; z = z0
      IF ( print ) WRITE( out, "( '  subnorm rhs      = ', ES12.4 )" )         &
          MAXVAL( ABS( B ) )
      CALL QPA_block_solve( K, SLS_data, K_part, X, x_r, y, x_f, z,            &
                            print, RES_print, out, SLS_control, inform )

      IF ( z ) THEN
        IF ( .NOT. x_r .OR. .NOT. y ) THEN
          xs_r = .FALSE. ; ys = .FALSE.
        ELSE
          xs_r = x_r ; ys = y
        END IF
        xs_f = .TRUE.
      ELSE
        xs_r = .FALSE. ; ys = .FALSE. ; xs_f = .FALSE.
      END IF
      zs = .FALSE.

!  Refine the solution

      IF ( .NOT. ( x_r .AND. y .AND. z ) ) THEN

        DO iter = 1, itref_max

!  Compute the residuals

          IF ( z ) THEN
            IF ( .NOT. x_r .OR. .NOT. y ) THEN
              x_r = .FALSE. ; y = .FALSE.
            END IF
            x_f = .TRUE. ; z = .FALSE.
          ELSE
            x_r = .FALSE. ; y = .FALSE. ; x_f = .FALSE.
          END IF

          CALL QPA_K_residuals( K, K_part, X, B, RES, xs_r, ys, xs_f, zs )

          IF ( print ) WRITE( out, "( '  subnorm residual = ', ES12.4 )" )     &
              MAXVAL( ABS( RES ) )

!  Find the correction

          IF ( .NOT. x_r .OR. .NOT. y ) THEN
            x_r = .FALSE. ; y = .FALSE.
          END IF
          x_f = .FALSE. ; z = .TRUE.
          CALL QPA_block_solve( K, SLS_data, K_part, RES, x_r, y, x_f, z,      &
                                print, RES_print, out, SLS_control, inform )

!  Refine the solution using the correction

          X = X + RES

          IF ( z ) THEN
            IF ( .NOT. x_r .OR. .NOT. y ) THEN
              xs_r = .FALSE.
              ys = .FALSE.
            ELSE
              xs_r = xs_r .AND. x_r
              ys = ys .AND. y
            END IF
            x_f = .FALSE.
            z = .TRUE.
          ELSE
            xs_r = .FALSE. ; ys = .FALSE. ; xs_f = .FALSE.
          END IF
        END DO
      END IF

!  Compute the final residuals

      IF ( print ) THEN

        CALL QPA_K_residuals( K, K_part, X, B, RES, xs_r, ys, xs_f, zs )
        WRITE( out, "( '  subnorm residual = ', ES12.4 )" ) MAXVAL( ABS( RES ) )

      END IF

      RETURN

!  End of QPA_ir

      END SUBROUTINE QPA_ir

!-*-*-*-*-*-*-*-*-*-  Q P A _ B L O C K _ S O L V E   -*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE QPA_block_solve( K, SLS_data, K_part, X, b_r, c, b_f, d,      &
                                  print, RES_print, out, SLS_control, inform )

!  Solve the block system

!   (  G_r   A_r^T | G_o^T    ) ( x_r )   ( b_r )
!   (  A_r         |  A_f     ) (  y  )   (  c  )
!   (-------------------------) ( --- ) = ( --- )
!   (  G_o   A_f^T |  G_f   I ) ( x_f )   ( b_f )
!   (              |   I      ) (  z  )   (  d  )

!  given a factorization of the block

!   ( G_r  A_r^T )
!   ( A_r        )

!  The logical arguments b_r, c, b_f, d should be set .TRUE. if all the
!  components of the corresponding vectors have the value zero, and .FALSE.
!  otherwise. The RHS should be input in X, and the solution overwrites X

!  Dummy arguments

      INTEGER, INTENT( IN ) :: out
      LOGICAL, INTENT( IN ) :: print
      TYPE ( SMT_TYPE ), INTENT( IN ) :: K
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      LOGICAL, INTENT( IN ) :: b_r, c, b_f, d
      REAL ( KIND = wp ), INTENT( INOUT ) :: X( K_part%k_ref )
      REAL ( KIND = wp ), INTENT( OUT ) ::                                     &
                          RES_print( K_part%n_free + K_part%c_ref )
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, l, y_e, x_fs, z_s
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: swap, val

!  The rhs is input in X, and the solution overwrites X on output

      y_e = K_part%n_free + K_part%c_ref ; x_fs = y_e
      z_s = x_fs + K_part%n_fixed

!  Adjust the right-hand side to form

!    ( b_r )    ( b_r )   ( G_o^T )
!    (  c  ) <- (  c  ) - (  A_f  ) d
!    ( b_f )    ( b_f )   (  L^T  )

!  where G_f = L + D + L^T

      IF ( .NOT. d ) THEN
        DO l = K_part%k_free_p + 1, K_part%k_fixed_od
          i = K%row( l ) + K_part%n_fixed
          X( K%col( l ) ) = X( K%col( l ) ) - K%val( l ) * X( i )
        END DO
      END IF

!  Solve
!   (  G_r    A_r^T  ) ( x_r ) = ( b_r )
!   (  A_r           ) (  y  )   (  c  )

      IF ( .NOT. ( b_r .AND. c .AND. d ) ) THEN
        IF ( K%n > 0 ) THEN
          IF ( print ) THEN
            RES_print = X( : y_e )
            WRITE( out, "( '  rhs = ', ES12.4 )" ) MAXVAL( ABS( RES_print ) )
          END IF

          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SLS_solve( K, X, SLS_data, SLS_control, inform%SLS_inform )
          CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + time_now - time_record

          IF ( print ) THEN
            DO l = 1, K_part%k_free_od
              i = K%row( l ) ; j =  K%col( l ) ; val = K%val( l )
              RES_print( i ) = RES_print( i ) - val * X( j )
              RES_print( j ) = RES_print( j ) - val * X( i )
            END DO

            DO l =  K_part%k_free_od + 1,  K_part%k_free_d
              i = K%row( l )
              RES_print( i ) = RES_print( i ) - K%val( l ) * X( K%col( l ) )
            END DO

            IF (  K_part%k_free_d <  K_part%k_free_p ) THEN
              IF ( k%val(  K_part%k_free_p ) /= zero ) THEN
                DO l = K_part%k_free_d + 1, K_part%k_free_p
                  i = K%row( l )
                  RES_print( i ) = RES_print( i )                              &
                                    - K%val( l ) * X( K%col( l ) )
                END DO
              END IF
            END IF
          END IF
        END IF
      END IF

!  Swap z = b_f and x_f = d

      IF ( .NOT. ( b_f .AND. d ) ) THEN
        DO l = 1, K_part%n_fixed
          swap = X( x_fs + l )
          X( x_fs + l ) = X( z_s + l )
          X( z_s + l ) = swap
        END DO
      END IF

!  Update z <- z - G_o x_r - A_f^T y - ( L + D ) x_f

      IF ( .NOT. ( b_r .AND. c .AND. d ) ) THEN
        DO l = K_part%k_free_p + 1, K_part%k_fixed_d
          i = K%row( l ) + K_part%n_fixed
          X( i ) = X( i ) - K%val( l ) * X( K%col( l ) )
        END DO
      END IF

      RETURN

!  End of QPA_block_solve

      END SUBROUTINE QPA_block_solve

!-*-*-*-*-*-*-  Q P A _ K _ R E S I D U A L S _ S U B R O U T I N E -*-*-*-*-

      SUBROUTINE QPA_K_residuals( K, K_part, X, B, RES, x_r, y, x_f, z )

!  Compute the residuals

!           (  G_r   A_r^T | G_o^T   ) ( x_r )
!           (  A_r         |  A_f    ) (  y  )
! res = b - (------------------------) ( --- )
!           (  G_o   A_f^T |  G_f  I ) ( x_f )
!           (              |   I     ) (  z  )

!  The logical arguments  x_r, y, x_f, z should be set .TRUE. if all the
!  components of the corresponding vectors have the value zero, and .FALSE.
!  otherwise

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      LOGICAL, INTENT( IN ) :: x_r, y, x_f, z
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( K_part%k_ref ) :: X, B
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( K_part%k_ref ) :: RES

!  Local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: val

      RES = B

!  Include the terms from K

      IF ( .NOT. ( x_r .AND. y ) ) THEN
        DO l = 1, K_part%k_free_od
          i = K%row( l ) ; j =  K%col( l ) ; val = K%val( l )
          RES( i ) = RES( i ) - val * X( j )
          RES( j ) = RES( j ) - val * X( i )
        END DO

        DO l =  K_part%k_free_od + 1,  K_part%k_free_d
          i = K%row( l )
          RES( i ) = RES( i ) - K%val( l ) * X( K%col( l ) )
        END DO

        IF (  K_part%k_free_d <  K_part%k_free_p ) THEN
          IF ( k%val(  K_part%k_free_p ) /= zero ) THEN
            DO l = K_part%k_free_d + 1, K_part%k_free_p
              i = K%row( l )
              RES( i ) = RES( i ) - K%val( l ) * X( K%col( l ) )
            END DO
          END IF
        END IF
      END IF

      DO l =  K_part%k_free_p + 1,  K_part%k_fixed_od
        i = K%row( l ) ; j =  K%col( l ) ; val = K%val( l )
        RES( i ) = RES( i ) - val * X( j )
        RES( j ) = RES( j ) - val * X( i )
      END DO

      IF ( .NOT. x_f ) THEN
        DO l =  K_part%k_fixed_od + 1,  K_part%k_fixed_d
          i = K%row( l )
          RES( i ) = RES( i ) - K%val( l ) * X( K%col( l ) )
        END DO
      END IF

!  Include the terms from the fixed variables

      IF ( .NOT. x_f ) THEN
        RES( K_part%n_ref + 1 : K_part%n_ref + K_part%n_fixed ) =              &
          RES( K_part%n_ref + 1 : K_part%n_ref + K_part%n_fixed ) -            &
            X( K%n + 1 : K%n + K_part%n_fixed )
      END IF

      IF ( .NOT. z ) THEN
        RES( K%n + 1 : K%n + K_part%n_fixed ) =                                &
          RES( K%n + 1 : K%n + K_part%n_fixed ) -                              &
            X( K_part%n_ref + 1 : K_part%n_ref + K_part%n_fixed )
      END IF

      RETURN

!  End of subroutine QPA_K_residuals

      END SUBROUTINE QPA_K_residuals

!-*-*-*-*-*-*-*-*-*-*-*-  Q P A _ M A 2 7 _ L O W E R  -*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE QPA_lower( row, col, i, j )

!  Assign the larger of i and j to row, and the smaller to column

!  Dummy arguments

      INTEGER, INTENT( IN ) :: i, j
      INTEGER, INTENT( OUT ) :: row, col

      IF ( i >= j ) THEN
        row = i
        col = j
      ELSE
        row = j
        col = i
      END IF

      RETURN

!  End of QPA_lower

      END SUBROUTINE QPA_lower

!-*-*-*-*-*-*-*-*-*-*-  Q P A _ P C G   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE QPA_pcg( dims, n, SOL, RHS, R, X, P, R_perm, G_perm, B, DX,   &
                          HP, PERM, SCU_mat, SCU_data, K, K_part, H_val,       &
                          H_col, H_ptr, prefix, control, SLS_control,          &
                          SLS_data, print_level,  cg_maxit, dof, start_from_x, &
                          inner_stop_absolute, inner_stop_relative, itref_max, &
                          inform, iter, negative_curvature, status, RES_print )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   dims     dimensions, see QPA_solve
!   X        the vector of unknowns. Need not be set on entry.
!            On exit, the best value found so far
!   RHS      the vector c
!   R        the residual vector H x + c
!   P, HP, R_perm, RES, B, G and DX  internal workspace vector
!   control  a structure containing control information. See QPA_initialize.
!            Only the components out, error, stop_relative, and
!            stop_absolute need be given
!   print_level  the level of output required. <= 0 gives no output
!   cg_maxit the maximum number of CG iterations allowed
!   dof      the number of degrees of freedom
!   iter     the number of iterations performed
!   negative_curvature  if true, X contains a direction of negative curvature
!   status     the output status. Possible exit values are:
!               0 the solution has been found
!              -1 the iteration limit has been exceeded
!              -2 negative curvature has been encountered
!              -3 the preconditioning matrix M appears to be indefinite
!              -4 n is not positive
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, print_level, cg_maxit, dof
      INTEGER, INTENT( INOUT ) :: itref_max
      INTEGER, INTENT( OUT ) :: status, iter
      LOGICAL, INTENT( IN ) :: start_from_x
      LOGICAL, INTENT( OUT ) :: negative_curvature
      REAL ( KIND = wp ),                                                      &
             INTENT( IN ) :: inner_stop_absolute, inner_stop_relative
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      TYPE ( SCU_matrix_type ), INTENT( IN ) :: SCU_mat
      INTEGER, INTENT( IN ), DIMENSION( SCU_mat%n + SCU_mat%m ) :: PERM
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RHS
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: R, P
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION ( SCU_mat%n + SCU_mat%m ) :: SOL, R_perm, G_perm, B, DX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( SCU_mat%n ) :: HP
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( K_part%n_free + K_part%c_ref ) :: RES_print
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: itmax, n_all
      REAL ( KIND = wp ) :: alpha, beta, curv, rminvr, rminvr_old, f
      REAL ( KIND = wp ) :: stop_tol, rayleigh, pgnorm, pmp, norm_p, norm_p_tiny
      LOGICAL :: printi, printm, printd
      INTEGER, PARAMETER :: ir_max = 2

!  Check for obvious errors

      iter = 0
      negative_curvature = .FALSE.
      IF ( n <= 0 ) THEN ; status = - 4 ; GO TO 900 ; END IF

      norm_p_tiny = epsmch ** 0.75
      n_all = SCU_mat%n + SCU_mat%m
      itmax = cg_maxit ; IF ( itmax < 0 ) itmax = dof
      printi = control%out > 0 .AND. print_level >= 1
      printm = control%out > 0 .AND. print_level >= 2
      printd = control%out > 0 .AND. print_level >= 3
      R_perm( PERM( n + 1 : n_all ) ) = zero

!  If starting from a given x, find the value and gradient at this point

      IF ( start_from_x ) THEN
        HP( : n ) = zero
        CALL QPA_HX( dims, n, HP( : n ), H_ptr( n + 1 ) - 1,                  &
                      H_val, H_col, H_ptr, X, '+' )
        f = DOT_PRODUCT( X, HP( : n ) ) - half * DOT_PRODUCT( X, RHS )
        R = - RHS + HP( : n )
      ELSE
        f = zero
      END IF

!  ===========================
!  Start of the main iteration
!  ===========================

main: DO

!  ----------------------------------
!  Obtain the preconditioned residual
!  ----------------------------------

      IF ( printd ) WRITE( control%out,                                        &
         "( '   |................. precondition  ..................| ' )" )

      DO
        IF ( iter > 0 .OR. start_from_x ) THEN
          R_perm( PERM( : n ) ) = R
        ELSE
          R_perm( PERM( : n ) ) = - RHS
        END IF

        CALL QPA_iterative_refinement( K, SCU_mat, SCU_data, R_perm, G_perm,   &
                                        B, SOL, DX, HP, SLS_control,           &
                                        SLS_data, K_part, .TRUE., itref_max,   &
                                        control%out, printm, RES_print, inform )
        SOL( : n ) = G_perm( PERM( : n ) )

!  Obtain the scaled norm of the residual

        IF ( iter > 0 .OR. start_from_x ) THEN
          rminvr = DOT_PRODUCT( R, SOL( : n ) )
        ELSE
          rminvr = - DOT_PRODUCT( RHS, SOL( : n ) )
        END IF

        IF ( ABS( rminvr ) < teneps ) rminvr = zero
        IF ( rminvr >= zero ) EXIT
        IF ( itref_max > ir_max - 1 ) THEN
          IF ( printi ) write( control%out,                                    &
            "( ' inner product = ', ES12.4 )" ) rminvr
          status = - 3
          EXIT main
        ELSE
          itref_max = itref_max + 1
          IF ( printi ) write( control%out,                                    &
            "( ' increasing itref_max to ', I3 )" ) itref_max
        END IF
      END DO
      pgnorm = SIGN( SQRT( ABS( rminvr ) ), rminvr )

      IF ( iter > 0 ) THEN
        beta = rminvr / rminvr_old
      ELSE

!  Compute the stopping tolerance

        stop_tol = MAX( inner_stop_relative * pgnorm, inner_stop_absolute )
        IF ( printi ) WRITE( control%out, "( /, A, ' stopping tolerance = ',   &
       &                    ES12.4 )" ) prefix, stop_tol
      END IF

!  Print details of the latest iteration

      IF ( printi ) THEN
        IF ( MOD( iter, 25 ) == 0 .OR.printm ) WRITE( control%out, 2000 ) prefix

        IF ( iter /= 0 ) THEN
          WRITE( control%out, "( A, I9, ES16.8, 3ES9.2, ES10.2 )" )            &
                 prefix, iter, f, pgnorm, alpha, norm_p, rayleigh
        ELSE
          WRITE( control%out, "( A, I9, ES16.8, ES9.2, 5X, '-',                &
        &         8X, '-', 9X, '-' )" ) prefix, iter, f, pgnorm
        END IF
      END IF

!  Test for an interior approximate solution

      IF ( pgnorm <= stop_tol ) THEN
        IF ( printi ) WRITE( control%out,                                      &
          "( A, ' pgnorm ', ES12.4, ' < ', ES12.4 )" ) prefix, pgnorm, stop_tol
        status = GALAHAD_ok ; EXIT
      END IF

      IF ( iter > 0 ) THEN

!  Test to see that iteration limit has not been exceeded

        IF ( iter >= itmax ) THEN ; status = - 1 ; EXIT ; END IF

!  Obtain the search direction P

        P = - SOL( : n ) + beta * P
        pmp = rminvr + pmp * beta * beta
        IF ( printd ) WRITE( control%out,                                      &
             "( ' beta ', ES12.4 )" ) beta
      ELSE

!  Special case for the first iteration

        P = - SOL( : n )
        pmp = rminvr
      END IF

      rminvr_old = rminvr

!  Compute the 2-norm of the search direction

      norm_p = NRM2( n, P, 1 ) !TWO_NORM( n, P, 1 )

!  Test for convergence

      IF ( norm_p <= norm_p_tiny ) THEN
        IF ( printi ) WRITE( control%out,                                      &
         "( A, ' pnorm ', ES12.4, ' < ', ES12.4 )" ) prefix, norm_p, norm_p_tiny
        status = GALAHAD_ok ; EXIT
      END IF

      iter = iter + 1

!  ------------------------------
!  Obtain the product of H with p
!  ------------------------------

!  Obtain the curvature

      IF ( printd ) WRITE( control%out,                                        &
         "( A, '   |............ matrix-vector product ...............| ' )" ) &
           prefix

      HP( : n ) = zero
      CALL QPA_HX( dims, n, HP( : n ), H_ptr( n + 1 ) - 1,                     &
                    H_val, H_col, H_ptr, P, '+' )

!  Compute the curvature and the Rayleigh quotient

      curv = DOT_PRODUCT( P, HP( : n ) )
      rayleigh = curv / pmp

!  Exit if the curvature is not positive

      IF ( rayleigh <= curv_min ) THEN ; status = - 2 ; EXIT ; END IF

!  Obtain the stepsize

      alpha = rminvr / curv

!  Update the estimate of the solution and its residual

      IF ( iter > 1 .OR. start_from_x ) THEN
        X = X + alpha * P
        R = R + alpha * HP( : n )
      ELSE
        X = alpha * P
        R = - RHS + alpha * HP( : n )
      END IF

!  Update the function value

      f = f + alpha * ( half * alpha * curv - rminvr )

!  =========================
!  End of the main iteration
!  =========================

      END DO main

!  ===============
!  Exit conditions
!  ===============

  900 CONTINUE
      SELECT CASE( status )

!  Successful returns

      CASE( 0 )
        IF ( iter > 0 .OR. start_from_x ) THEN
          SOL( : n ) = X
        ELSE
          SOL( : n ) = zero
        END IF
        SOL( n + 1 : ) = - G_perm( PERM( n + 1 : ) )

!  Too many iterations

      CASE( - 1 )
        IF ( printi )                                                          &
          WRITE( control%out, "( /, ' Iteration limit exceeded ' ) " )
        IF ( iter > 0 .OR. start_from_x ) THEN
          SOL( : n ) = X
        ELSE
          SOL( : n ) = zero
        END IF
        SOL( n + 1 : ) = - G_perm( PERM( n + 1 : ) )

!  Negative curvature encountered

      CASE( - 2 )
        negative_curvature = .TRUE.
        SOL( : n ) = P
        SOL( n + 1 : ) = - G_perm( PERM( n + 1 : ) )
        IF ( printm ) WRITE( control%out, 2000 )
        IF ( printi )                                                          &
          WRITE( control%out, "( A, I9, 8X, '-', 12X, 2( '-', 8X ), '-', 3X,   &
         &                        ES10.2 )" ) prefix, iter, rayleigh

!  Unsuccessful returns

      CASE( - 3 )
!       WRITE( control%out,  &
        IF ( control%error > 0 .AND. print_level >= 1 ) WRITE( control%error,  &
        "( /, A, ' The preconditioner appears to be indefinite.',              &
     &        ' Inner product = ', ES12.4  )" ) prefix, rminvr

      CASE( - 4 )
        IF ( control%error > 0 .AND. print_level >= 1 ) WRITE( control%error,  &
           "( A, ' n = ', I0, ' is not positive ' )" ) prefix, n
      END SELECT

      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, '     Iter        f         pgnorm    step   ',            &
                    ' norm p     curv')

!  End of subroutine QPA_pcg

      END SUBROUTINE QPA_pcg

!-*-*-*-*-*-*-*-   Q P A _ F O R M _ S _ C   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE QPA_form_Schur_complement(                                    &
                          n_free, m, k_free_od, K_row, K_col, K_val,           &
                          D_row, D_col, D_val,                                 &
                          Abycol_ne, Abycol_val, Abycol_row, Abycol_ptr,       &
                          S_ne, S_val, S_row, S_col, S_colptr,                 &
                          ierr, factor, max_col, prefix,                       &
                          control_print_level, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  This subroutine computes the Schur-complement matrix
!            S = A D(inv) A(trans)
!  given the matrix A and vector of weights D

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!  m is the number of rows in matrix A.
!  n is the number of columns in matrix A, and the number of rows and
!    columns in matrix S.
!  Abycol_ne is the number of entries in matrix A. It is the length of
!     arrays Abycol_val and Abycol_row.
!  Abycol_val holds the values of the entries in matrix A.
!  Abycol_row holds the row indices of the corresponding entries in A
!  Abycol_ptr points to the start of each column of A.
!  ls is the length of arrays S and S_row.
!  S_val is set to the values of the entries in the lower triangle of S,
!   ordered by columns
!  S_row is set to the row numbers of the entries in the lower triangle of S
!  S_colptr is set to point to the start of each column of S.
!  ierr is set by the routine to the code specifying the type of input
!       error:
!            = 0    no input errors;
!            = 1    n less than or equal to zero;
!            = 2    m less than or equal to zero;
!            = 3    ls less than Abycol_ne+m;
!            = 4    ls less than the number of entries in S (the
!                   number required is in nes);
!            = 5    allocation error
!  nes is the actual number of entries in S
!  COL_count is the number of entries in each column of A
!  ROW_count is the number of entries in each row in A and, later,
!            the number of entries in each row of the lower triangle of S

!  Based on Iain Duff's routine MC35

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n_free, m, k_free_od
      INTEGER, INTENT( IN ) :: control_print_level, max_col
      INTEGER, INTENT( INOUT ) :: factor
      INTEGER, INTENT( OUT ) :: Abycol_ne, S_ne, ierr
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row, Abycol_ptr
      INTEGER, ALLOCATABLE, DIMENSION( : )  :: S_row, S_col, S_colptr
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )  :: Abycol_val, S_val
      INTEGER, INTENT( IN ), DIMENSION( k_free_od ) :: K_row, K_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( k_free_od ) :: K_val
      INTEGER, INTENT( INOUT ), DIMENSION( n_free ) :: D_row, D_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n_free ) :: D_val
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, i1, i2, ii, ij, ipos, j, j1, j2, jj, max_len
      INTEGER :: icount, idist, k, k1, k2, kk, alloc_status, a_ne, ls
      REAL ( KIND = wp ) :: amult
      LOGICAL :: error
      CHARACTER ( LEN = 80 ) :: array_name

      Abycol_ne = k_free_od

!  Check for input errors

      IF ( n_free <= 0 ) THEN
        ierr = 1
        IF ( control%error > 0 .AND. control_print_level >= 1 )                &
          WRITE( control%error,                                                &
           "( A, ' ** Error return from QPA_form_Schur_complement', /,         &
         &    A, '    Value of n (number of rows of A) is set to ', I0, /,     &
         &    A, '    but n must be at least 1 ' )" )                          &
                prefix, prefix, n_free, prefix
        RETURN
      END IF

      IF ( m < 0 ) THEN
        ierr = 2
        IF ( control%error > 0 .AND. control_print_level >= 1 )                &
          WRITE( control%error,                                                &
           "( A, ' ** Error return from QPA_form_Schur_complement', /,         &
         &    A, '    Value of m (number of columns of A) is set to ', I0, /,  &
         &    A, '    but m must be at least 1 ' )" )                          &
                prefix, prefix, m,  prefix
        RETURN
      END IF

!  Ensure that the diagonals of K are ordered correctly

      DO i = 1, n_free
        IF ( D_row( i ) /= i ) THEN
          write(6,"( ' diagonals unordered ', 2I6 )") D_row( i ), i
          stop
        END IF
        IF ( D_col( i ) /= i ) THEN
          write(6,"( ' diagonals unordered ', 2I6 )") D_col( i ), i
          stop
        END IF
      END DO

!  Allocate the arrays for the analysis phase

      a_ne = k_free_od

      array_name = 'qpa: data%Abycol_val'
      CALL SPACE_resize_array( a_ne, Abycol_val, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%Abycol_row'
      CALL SPACE_resize_array( a_ne, Abycol_row, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%Abycol_ptr'
      CALL SPACE_resize_array( n_free + 1, Abycol_ptr, inform%status,          &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( m > 0 ) THEN

!  Reorder A so that its entries are ordered by columns

!  count the number of nonzeros in each column

        Abycol_ptr( : n_free ) = 0
        DO ii = 1, k_free_od
          Abycol_ptr( K_col( ii ) ) = Abycol_ptr( K_col( ii ) ) + 1
        END DO

!  set the starting addresses for each column

        j = 1
        DO i = 1, n_free
          ii = j
          j = j + Abycol_ptr( i )
          Abycol_ptr( i ) = ii
        END DO

!  move the entries from A to B

        DO i = 1, K_free_od
          j = K_col( i )
          jj = Abycol_ptr( j )
          Abycol_row( jj ) = K_row( i ) - n_free
          Abycol_val( jj ) = K_val( i )
          Abycol_ptr( j ) = Abycol_ptr( j ) + 1
        END DO

!  reset the starting addresses

        DO i = n_free, 1, - 1
          Abycol_ptr( i + 1 ) = Abycol_ptr( i )
        END DO
        Abycol_ptr( 1 ) = 1

!  Compute the length of the largest column as well as the average length

        max_len =                                                              &
          MAXVAL( Abycol_ptr( 2 : n_free + 1 ) - Abycol_ptr( 1 : n_free ) )
        IF ( control%error > 0 .AND. control_print_level >= 1 )                &
           WRITE( control%error, "( A, ' maximum, average column lengths: ',   &
          &   I0, ' & ', 0P, F0.1, /, A, ' number of columns longer than ',    &
          &   I0, ' is ', I0 )" ) prefix, max_len,                             &
          float( ( Abycol_ptr( n_free + 1 ) - 1 ) ) / float( n_free ),         &
          prefix, max_col,                                                     &
          COUNT( Abycol_ptr( 2 : n_free+1 ) - Abycol_ptr( : n_free ) > max_col )
      ELSE
        max_len = 0
      END IF

!  Check that the largest column is not too long

      IF ( factor == 0 .AND. max_len > max_col .AND. m > control%max_sc ) THEN
        IF ( control%error > 0 .AND. control_print_level >= 1 )                &
          WRITE( control%error,                                                &
           "( A, ' ** The maximum column length in A is larger than', /, A,    &
         &    '    max_col =', I0, ' - abandon the Schur-complement', /, A,    &
         &    '    factorization in favour of one of the augmented matrix')" ) &
            prefix, prefix, max_col, prefix
        ierr = 5
        factor = 2
        DEALLOCATE( Abycol_val, Abycol_row, Abycol_ptr )
        RETURN
      END IF

!  Continue allocating the arrays for the analysis phase

      ls = max( a_ne + m, 2 * a_ne, control%valmin )

      array_name = 'qpa: data%S_row'
      CALL SPACE_resize_array( ls, S_row, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%S_col'
      CALL SPACE_resize_array( ls, S_col, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%S_val'
      CALL SPACE_resize_array( ls, S_val, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%S_colptr'
      CALL SPACE_resize_array( m + 1, S_colptr, inform%status,                 &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Permute A so that its entries appear by columns, with the entries in
!  each column sorted by increasing row number. Also form the data
!  structure required to hold S = A D(inv) A(transpose)

   10 CONTINUE

      IF ( m > 0 ) THEN

        ierr = 0

!  Set up column counts

        D_col( : n_free ) =                                                    &
          Abycol_ptr( 2 : n_free + 1 ) - Abycol_ptr( : n_free )

!  Set up row counts

        D_row( : m ) = 0
        DO i = 1, Abycol_ne
          j = Abycol_row( i )
          D_row( j ) = D_row( j ) + 1
        END DO

!  Set up column pointer

        S_colptr( 1 ) = ls - Abycol_ne + 1
        DO i = 1, m
          S_colptr( i + 1 ) = S_colptr( i ) + D_row( i )
        END DO

!  Generate structure of A by columns in S/S_row

        j1 = 1
        DO i = 1, n_free
          j2 = j1 + D_col( i ) - 1
          DO jj = j1, j2
            j = Abycol_row( jj )
            ipos = S_colptr( j )
            S_val( ipos ) = Abycol_val( jj )
            S_row( ipos ) = i
            S_colptr( j ) = ipos + 1
          END DO
          j1 = j2 + 1
        END DO

!  Regenerate structure of A by columns but in row order within each column.
!  Order the row indices in Abycol_row using S_val/S_row to create
!  Abycol_val/Abycol_row

        i1 = ls - Abycol_ne + 1
        DO j = 1, m
          ij = m - j + 1

!  Reset S_colptr after the last loop altered it

          IF ( ij > 1 ) S_colptr( ij ) = S_colptr( ij - 1 )
          i2 = i1 + D_row( j ) - 1
          DO ii = i1, i2
            i = S_row( ii )
            ipos = Abycol_ptr( i )
            Abycol_row( ipos ) = j
            Abycol_val( ipos ) = S_val( ii )
            Abycol_ptr( i ) = ipos + 1
          END DO
          i1 = i2 + 1
        END DO
        S_colptr( 1 ) = 1 + ls - Abycol_ne

!  Reset Abycol_ptr since it has been altered

        DO j = n_free, 2, - 1
          Abycol_ptr( j ) = Abycol_ptr( j - 1 )
        END DO
        Abycol_ptr( 1 ) = 1

!  Now assemble S

        error = .FALSE.
        S_ne = 0
        icount = 0
        idist = 0
        k1 = S_colptr( 1 )

!  Form row i of S from a linear combination of rows of A

        DO i = 1, m

!  row i of A is non-empty

          k2 = k1 + D_row( i ) - 1
          D_row( i ) = 0

!  Scan row i of A(trans) to find which rows of A contribute to row i of S

          DO kk = k1, k2
            k = S_row( kk )
            amult = S_val( kk ) / D_val( k )
            j1 = Abycol_ptr( k )
            j2 = j1 + D_col( k ) - 1

!  The column indices for the entries in each row are in order

            Abycol_ptr( k ) = j1 + 1
            D_col( k ) = j2 - j1

!  Scan row k of A and add multiple of it to row i of S

            DO jj = j1, j2
              j = Abycol_row( jj )
              IF ( error ) THEN

!  Only count the size of arrays needed because of the error

                IF ( S_colptr( j ) < 0 ) CYCLE
                D_row( i ) = D_row( i ) + 1
                ipos = D_row( i )
                S_colptr( j ) = - D_row( i )
                IF ( ipos + idist > kk ) THEN
                 icount = icount + 1
                 idist = idist - 1
                END IF

                S_row( ipos ) = j
                CYCLE

              END IF

              IF ( S_colptr( j ) >= 0 ) THEN

!  First contribution to row i of S

                D_row( i ) = D_row( i ) + 1

!  S_colptr is used to obtain offset for column positions in present row

                S_colptr( j ) = - D_row( i )
                ipos = S_ne + D_row( i )
                IF ( ipos > kk ) THEN

!  Insufficient room in S to continue calculation, so only calculate
!  the size required

                  error = .TRUE.

!  Overwrite the beginning of S_row since it is not now needed.

                  ij = D_row( i )
                  S_row( 1 : ij - 1 ) = S_row( S_ne + 1 : S_ne + ij - 1 )
                  S_ne = 0

!  idist holds the difference between the current position in S_row
!  and the position that it would be if S_row was long enough

                  idist = kk - ij

!  icount is the amount by which S_row is too short

                  icount = 1
                  ipos = ij
                  S_row( ipos ) = j
                  CYCLE

                END IF

                S_row( ipos ) = j
                S_val( ipos ) = Abycol_val( jj ) * amult
                CYCLE

              END IF
              ipos = S_ne - S_colptr( j )
              S_val( ipos ) = S_val( ipos ) + Abycol_val( jj) * amult
            END DO
          END DO

!  Reset S_colptr

          IF ( error ) S_ne = 0
          j1 = S_ne + 1
          S_ne = S_ne + D_row( i )
          S_colptr( S_row( j1 : S_ne ) ) = 0

          IF ( error ) THEN
           idist = idist + D_row( i )
           S_ne = 0
          END IF

          k1 = k2 + 1
        END DO

!  Set pointer arrays for the start of each column of A and S

        IF ( .NOT. error ) THEN
          S_colptr( 1 ) = 1
          DO i = 1, m
            S_colptr( i + 1 ) = S_colptr( i ) + D_row( i )
          END DO
        END IF

        DO j = n_free, 2, - 1
          Abycol_ptr( j ) = Abycol_ptr( j - 1 )
        END DO
        Abycol_ptr( 1 ) = 1

        IF ( icount > 0 ) THEN
          S_ne = icount + ls

!  Insufficient space. Allocate more and retry

          DEALLOCATE( S_row, S_col, S_val )
          ls = S_ne
          ALLOCATE( S_row( ls ), S_col( ls ), S_val( ls ), STAT = alloc_status )

          IF ( alloc_status /= 0 ) THEN
            ierr = 5 ; GO TO 900
          END IF

          GO TO 10
        END IF

!  Record the column numbers of each entry in A A^T

        DO i = 1, m
          S_col( S_colptr( i ) : S_colptr( i + 1 ) - 1 ) = i
        END DO

!  reset the row and column numbers for the diagonal terms

        DO i = 1, n_free
          D_row( i ) = i ; D_col( i ) = i
        END DO

      ELSE
        ierr = 0
        S_ne = 0
      END IF

  900 CONTINUE
      ierr = 5
      RETURN

!  End of QPA_form_Schur_complement

      END SUBROUTINE QPA_form_Schur_complement

!-*-   Q P A _ F A C T O R I Z E _ R E F E R E N C E   S U B R O U T I N E   -*-

      SUBROUTINE QPA_factorize_reference(                                      &
                    dims, n, m, jumpto, k_n_max, print_level, &
                    m_link, max_col, factor, precon, nsemib, hmax, G_perturb,  &
                    out, prec_hist, printi, printt, printe, G_eq_H, auto_prec, &
                    auto_fact, check_dependent, mo, PERM, REF, C_stat, B_stat, &
                    A_ptr, A_col, H_ptr, H_col, A_val, H_val, K, K_part, S,    &
                    Abycol_row, Abycol_ptr, S_row, S_col, S_colptr,            &
                    Abycol_val, S_val, DIAG, SLS_data, SLS_control,            &
                    prefix, control, inform )

!  Form and factorize the reference matrix

!  Dummy arguments

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, k_n_max, print_level, m_link, max_col, out
      INTEGER, INTENT( INOUT ) :: jumpto, factor, precon, nsemib, prec_hist
      REAL ( KIND = wp ), INTENT( INOUT ) :: hmax
      REAL ( KIND = wp ), INTENT( INOUT ) :: G_perturb
      LOGICAL, INTENT( IN ) :: printi, printt, printe
      LOGICAL, INTENT( INOUT ) :: G_eq_H
      LOGICAL, INTENT( INOUT ) :: auto_prec, auto_fact, check_dependent
      CHARACTER ( LEN = 1 ), INTENT( INOUT ) :: mo

      INTEGER, INTENT( INOUT ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, INTENT( IN ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( QPA_partition_type ), INTENT( INOUT ) :: K_part
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( k_n_max + control%max_sc ) :: S
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, k_n_max ) :: DIAG
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row, Abycol_ptr
      INTEGER, ALLOCATABLE, DIMENSION( : )  :: S_row, S_col, S_colptr
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Abycol_val, S_val

      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

!  Local variables

      INTEGER :: i, ii, j, jj, l, n_perturb
      INTEGER :: i_free, i_fixed, ib_ref, ic_ref, S_ne, ierr, zeig, Abycol_ne
      INTEGER :: type, hd_start, hd_end, hnd_start, hnd_end, nroots
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      LOGICAL ::  modify_k, twobytwo
      REAL ( KIND = wp ) :: root1, root2, big
      CHARACTER ( LEN = 80 ) :: array_name

!  Skip to relevaant section if the structure is already known

      IF ( jumpto == 1 ) THEN
        GO TO 20
      ELSE IF ( jumpto == 2 ) THEN
        GO TO 30
      END IF

!  Permute the variables
!  ---------------------

!  Find a permutation of the rows and columns of the reference matrix
!  so that they are ordered as follows: free variables, Lagrange multipliers,
!  fixed variables, dual variables

      K_part%n_free = COUNT( B_stat == 0 )
      K_part%n_fixed = n - K_part%n_free

      i_free =  dims%x_free
      i_fixed = K_part%n_free + K_part%c_ref
      ib_ref = n + K_part%c_ref
      ic_ref = K_part%n_free

      DO i = 1, dims%x_free
        PERM( i ) = i
      END DO

      DO i = dims%x_free + 1, n
        IF ( B_stat( i ) == 0 ) THEN
          i_free = i_free + 1
          PERM( i ) = i_free
        END IF
      END DO

      i_fixed = K_part%n_free + K_part%c_ref
      DO i = 1, K_part%m_ref
        j = REF( i )
        IF ( j <= m ) THEN
          ic_ref = ic_ref + 1
          PERM( n + i ) = ic_ref
        ELSE
          ib_ref = ib_ref + 1
          PERM( n + i ) = ib_ref
          i_fixed = i_fixed + 1
          PERM( j - m ) = i_fixed
        END IF
      END DO

      DO i = K_part%k_ref + 1, k_n_max + control%max_sc
        PERM( i ) = i
      END DO

!  Print a header indicating the method selected

  10  CONTINUE

      IF ( printi ) THEN
        SELECT CASE( precon )
          CASE( : - 1, 1 ) ; WRITE( control%out,                               &
            "( /, A, ' Full Hessian ' )" ) prefix
          CASE( 0 ) ; WRITE( control%out,                                      &
            "( /, A, ' Automatic preconditioner ' )" ) prefix
          CASE( 2 ) ; WRITE( control%out,                                      &
             "( /, A, ' Identity Hessian ' )" ) prefix
          CASE( 3 ) ; WRITE( control%out,                                      &
             "( /, A, ' Band (semi-bandwidth ', I0, ') Hessian ' )")           &
               prefix, nsemib
          CASE( 4 ) ; WRITE( control%out,                                      &
             "( /, A, ' Identity Hessian for free variables, full for',        &
            &         ' remainder ' )" ) prefix
          CASE( 5 ) ; WRITE( control%out,                                      &
             "( /, A, ' Band (semi-bandwidth ', I0, ') Hessian for free',      &
            &         ' variables, full for remainder ' )" )  prefix, nsemib
        END SELECT

        IF ( factor == 0 .OR. factor == 1 ) THEN
          WRITE( control%out, "( A, '  Schur-complement factorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" )                               &
            prefix, SLS_control%relative_pivot_tolerance
        ELSE
          WRITE( control%out, "( A, '  Augmented system faxtorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" )                               &
            prefix, SLS_control%relative_pivot_tolerance
        END IF
      END IF

!  See how much room is needed for the reference matrix
!  ----------------------------------------------------

      K_part%k_free_od = 0 ; K_part%k_free_d = 0 ; K_part%k_free_p = 0
      K_part%k_fixed_od = 0 ; K_part%k_fixed_d = 0

!  Terms for the given approximation to the Hessian:

      SELECT CASE( precon )

!  The full Hessian
!  ................

      CASE( : - 1, 1 )

        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                IF ( B_stat( H_col( l ) ) == 0 ) THEN ! ** free variable
                  K_part%k_free_od = K_part%k_free_od + 1
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i ) - 1
              K_part%k_fixed_d = K_part%k_fixed_d + 1
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                IF ( B_stat( H_col( l ) ) == 0 ) THEN ! ** free variable
                  K_part%k_free_od = K_part%k_free_od + 1
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i )
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  The identity matrix
!  ...................

      CASE( 2 )

        K_part%k_free_d = K_part%n_free
        K_part%k_fixed_d = K_part%n_fixed

!  A band from the full Hessian
!  ............................

      CASE( 3 )

        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l )
                IF ( ABS( ii - PERM( j ) ) <= nsemib ) THEN
                  IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                    K_part%k_free_od = K_part%k_free_od + 1
                  ELSE ! ** fixed variable
                    K_part%k_fixed_od = K_part%k_fixed_od + 1
                  END IF
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                IF ( ABS( ii - PERM( H_col( l ) ) ) <= nsemib )                &
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
              END DO
              K_part%k_fixed_d = K_part%k_fixed_d + 1
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                IF ( ABS( ii - PERM( j ) ) <= nsemib ) THEN
                  IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                    K_part%k_free_od = K_part%k_free_od + 1
                  ELSE ! ** fixed variable
                    K_part%k_fixed_od = K_part%k_fixed_od + 1
                  END IF
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                IF ( ABS( ii - PERM( H_col( l ) ) ) <= nsemib )                &
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
              END DO
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  The identity for the free part of the Hessian; the entire fixed part is used
!  ............................................................................

      CASE( 4 )

!       IF ( printi ) WRITE( out, 2440 )
        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                IF ( B_stat( H_col( l ) ) /= 0 ) THEN ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i ) - 1
              K_part%k_fixed_d = K_part%k_fixed_d + 1
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                IF ( B_stat( j ) /= 0 ) THEN ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i )
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  A band from the free part of the full Hessian; the entire fixed part is used
!  ............................................................................

      CASE( 5 )

        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  IF ( ABS( ii - PERM( j ) ) <= nsemib )                       &
                    K_part%k_free_od = K_part%k_free_od + 1
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i ) - 1
              K_part%k_fixed_d = K_part%k_fixed_d + 1
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  IF ( ABS( ii - PERM( j ) ) <= nsemib )                       &
                    K_part%k_free_od = K_part%k_free_od + 1
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
            ELSE  ! ** fixed variable
              K_part%k_fixed_od = K_part%k_fixed_od +                          &
                                    H_ptr( i + 1 ) - H_ptr( i )
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

      END SELECT

!  Constraint Jacobian terms
!  .........................

!  general constraints

      DO i = 1, m
        IF ( C_stat( i ) /= 0 ) THEN ! ** working constraint
          l = COUNT( B_stat( A_col( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) == 0 )
          K_part%k_free_od = K_part%k_free_od + l
          K_part%k_fixed_od = K_part%k_fixed_od +                              &
            A_ptr( i + 1 ) - A_ptr( i ) - l
        END IF
      END DO

!  Compute the total space required for the reference matrix

      K%n  = K_part%n_free + K_part%c_ref
      K%ne = K_part%k_free_od + K_part%k_free_d + K_part%k_free_p +            &
             K_part%k_fixed_od + K_part%k_fixed_d

      IF ( printt )                                                            &
        WRITE( out, "( /, A, ' n, n_free, n_fixed            = ', 3I8,         &
     &                 /, A, ' k_free_od, k_free_d, k_free_p = ', 3I8,         &
     &                 /, A, ' k_fixed_od, k_fixed_d         = ', 2I8,         &
     &                 /, A, ' c_ref, m_ref                  = ', 2I8  )" )    &
        prefix, n, K_part%n_free, K_part%n_fixed, prefix, K_part%k_free_od,    &
        K_part%k_free_d, K_part%k_free_p, prefix, K_part%k_fixed_od,           &
        K_part%k_fixed_d, prefix, K_part%c_ref, K_part%m_ref

!  Allocate arrays to store the reference matrix.

      array_name = 'qpa: data%K%row'
      CALL SPACE_resize_array( K%ne, K%row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%K%col'
      CALL SPACE_resize_array( K%ne, K%col, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpa: data%K%val'
      CALL SPACE_resize_array( K%ne, K%val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Now move the relevant data into the reference matrix
!  ----------------------------------------------------

!  The rows/columns/values components will be partitioned as follows:

!   <-            free components           -> | <- remaining components ->
!  -------------------------------------------------------------------------
!  | Off Diagonals | Diagonals | Perturbations | Off Diagonals | Diagonals |
!  -------------------------------------------------------------------------
!                 ^           ^               ^               ^           ^
!                 |           |               |               |           |
!              k_free_od  k_free_d         k_free_p       k_fixed_od   k_fixed_d

!  Terms for the given approximation to the Hessian:

      K_part%k_fixed_d = K_part%k_fixed_od + K_part%k_free_p +                 &
        K_part%k_free_d + K_part%k_free_od
      K_part%k_fixed_od = K_part%k_free_p + K_part%k_free_d + K_part%k_free_od
      K_part%k_free_p = K_part%k_free_d + K_part%k_free_od
      K_part%k_free_d = K_part%k_free_od
      K_part%k_free_od = 0

!  S is temporarily used to sum the absolute values of the off-diagonal
!  terms and the larger of minus the diagonal and zero

      S( : K%n ) =  ten ** ( - 5 )

      SELECT CASE( precon )

!  The full Hessian
!  ................

      CASE( : - 1, 1 )

        G_eq_H = .TRUE.
        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  K_part%k_free_od = K_part%k_free_od + 1
                  CALL QPA_lower( K%row( K_part%k_free_od ),                   &
                                   K%col( K_part%k_free_od ), ii, jj )
                  K%val( K_part%k_free_od ) = H_val( l )
                  S( ii ) = S( ii ) + ABS( H_val( l ) )
                  S( jj ) = S( jj ) + ABS( H_val( l ) )
                  IF ( factor == 0 .OR. factor == 1 ) THEN
                    IF ( printi ) WRITE( control%out, 2220 ) prefix
                    factor = 2
                  END IF
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
              K%row( K_part%k_free_d ) = ii
              K%col( K_part%k_free_d ) = ii
              K%val( K_part%k_free_d ) = H_val( l )
              S( ii ) = S( ii ) + MAX( - H_val( l ), zero )
              IF ( ( factor == 0 .OR. factor == 1 ) .AND.                      &
                     H_val( l ) == zero ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )
                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
              K_part%k_fixed_d = K_part%k_fixed_d + 1
              K%row( K_part%k_fixed_d ) = ii
              K%col( K_part%k_fixed_d ) = ii
              K%val( K_part%k_fixed_d ) = H_val( l )
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  K_part%k_free_od = K_part%k_free_od + 1
                  CALL QPA_lower( K%row( K_part%k_free_od ),                   &
                                   K%col( K_part%k_free_od ), ii, jj )
                  K%val( K_part%k_free_od ) = H_val( l )
                  S( ii ) = S( ii ) + ABS( H_val( l ) )
                  S( jj ) = S( jj ) + ABS( H_val( l ) )
                  IF ( factor == 0 .OR. factor == 1 ) THEN
                    IF ( printi ) WRITE( control%out, 2220 ) prefix
                    factor = 2
                  END IF
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
              K%row( K_part%k_free_p ) = ii
              K%col( K_part%k_free_p ) = ii
              K%val( K_part%k_free_p ) = zero
              IF ( factor == 0 .OR. factor == 1 ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )

                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  The identity matrix
!  ...................

      CASE( 2 )

        G_eq_H = .FALSE.
        DO i = 1, n
          ii = PERM( i )
          IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
            K_part%k_free_d = K_part%k_free_d + 1
            K%row( K_part%k_free_d ) = ii
            K%col( K_part%k_free_d ) = ii
            K%val( K_part%k_free_d ) = k_diag
          ELSE  ! ** fixed variable
            K_part%k_fixed_d = K_part%k_fixed_d + 1
            K%row( K_part%k_fixed_d ) = ii
            K%col( K_part%k_fixed_d ) = ii
            K%val( K_part%k_fixed_d ) = k_diag
          END IF
        END DO

!  A band from the full Hessian
!  ............................

      CASE( 3 )

        G_eq_H = .TRUE.
        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                IF ( ABS( ii - jj ) <= nsemib ) THEN
                  IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                    K_part%k_free_od = K_part%k_free_od + 1
                    CALL QPA_lower( K%row( K_part%k_free_od ),                 &
                                     K%col( K_part%k_free_od ), ii, jj )
                    K%val( K_part%k_free_od ) = H_val( l )
                    S( ii ) = S( ii ) + ABS( H_val( l ) )
                    S( jj ) = S( jj ) + ABS( H_val( l ) )
                    IF ( factor == 0 .OR. factor == 1 ) THEN
                      IF ( printi ) WRITE( control%out, 2220 ) prefix
                      factor = 2
                    END IF
                  ELSE ! ** fixed variable
                    K_part%k_fixed_od = K_part%k_fixed_od + 1
                    CALL QPA_lower( K%row( K_part%k_fixed_od ),                &
                                     K%col( K_part%k_fixed_od ), ii, jj )
                    K%val( K_part%k_fixed_od ) = H_val( l )
                  END IF
                ELSE
                  G_eq_H = .FALSE.
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
              K%row( K_part%k_free_d ) = ii
              K%col( K_part%k_free_d ) = ii
              K%val( K_part%k_free_d ) = H_val( l )
              S( ii ) = S( ii ) + MAX( - H_val( l ), zero )
              IF ( ( factor == 0 .OR. factor == 1 ) .AND.                      &
                     H_val( l ) == zero ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                IF ( ABS( ii - jj ) <= nsemib ) THEN
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                ELSE
                  G_eq_H = .FALSE.
                END IF
              END DO
              K_part%k_fixed_d = K_part%k_fixed_d + 1
              K%row( K_part%k_fixed_d ) = ii
              K%col( K_part%k_fixed_d ) = ii
              K%val( K_part%k_fixed_d ) = H_val( l )
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                IF ( ABS( ii - jj ) <= nsemib ) THEN
                  IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                    K_part%k_free_od = K_part%k_free_od + 1
                    CALL QPA_lower( K%row( K_part%k_free_od ),                 &
                                     K%col( K_part%k_free_od ), ii, jj )
                    K%val( K_part%k_free_od ) = H_val( l )
                    S( ii ) = S( ii ) + ABS( H_val( l ) )
                    S( jj ) = S( jj ) + ABS( H_val( l ) )
                    IF ( factor == 0 .OR. factor == 1 ) THEN
                      IF ( printi ) WRITE( control%out, 2220 ) prefix
                      factor = 2
                    END IF
                  ELSE ! ** fixed variable
                    K_part%k_fixed_od = K_part%k_fixed_od + 1
                    CALL QPA_lower( K%row( K_part%k_fixed_od ),                &
                                     K%col( K_part%k_fixed_od ), ii, jj )
                    K%val( K_part%k_fixed_od ) = H_val( l )
                  END IF
                ELSE
                  G_eq_H = .FALSE.
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
              K%row( K_part%k_free_p ) = ii
              K%col( K_part%k_free_p ) = ii
              K%val( K_part%k_free_p ) = zero
              IF ( factor == 0 .OR. factor == 1 ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                IF ( ABS( ii - jj ) <= nsemib ) THEN
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                ELSE
                  G_eq_H = .FALSE.
                END IF
              END DO
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  The identity for the free part of the Hessian; the entire fixed part is used
!  ............................................................................

      CASE( 4 )

        G_eq_H = .TRUE.
        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) /= 0 ) THEN ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
              K%row( K_part%k_free_d ) = ii
              K%col( K_part%k_free_d ) = ii
              K%val( K_part%k_free_d ) = k_diag
              G_eq_H = .FALSE.
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )
                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
              K_part%k_fixed_d = K_part%k_fixed_d + 1
              K%row( K_part%k_fixed_d ) = ii
              K%col( K_part%k_fixed_d ) = ii
              K%val( K_part%k_fixed_d ) = H_val( l )
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) /= 0 ) THEN ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
              K%row( K_part%k_free_d ) = ii
              K%col( K_part%k_free_d ) = ii
              K%val( K_part%k_free_d ) = k_diag
              G_eq_H = .FALSE.
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )
                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

!  A band from the free part of the full Hessian; the entire fixed part is used
!  ............................................................................

      CASE( 5 )

        G_eq_H = .TRUE.
        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  IF ( ABS( ii - jj ) <= nsemib ) THEN
                    K_part%k_free_od = K_part%k_free_od + 1
                    CALL QPA_lower( K%row( K_part%k_free_od ),                 &
                                     K%col( K_part%k_free_od ), ii, jj )
                    K%val( K_part%k_free_od ) = H_val( l )
                    S( ii ) = S( ii ) + ABS( H_val( l ) )
                    S( jj ) = S( jj ) + ABS( H_val( l ) )
                    IF ( factor == 0 .OR. factor == 1 ) THEN
                      IF ( printi ) WRITE( control%out, 2220 ) prefix
                      factor = 2
                    END IF
                  ELSE
                    G_eq_H = .FALSE.
                  END IF
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_d = K_part%k_free_d + 1
              K%row( K_part%k_free_d ) = ii
              K%col( K_part%k_free_d ) = ii
              K%val( K_part%k_free_d ) = H_val( l )
              S( ii ) = S( ii ) + MAX( - H_val( l ), zero )
              IF ( ( factor == 0 .OR. factor == 1 ) .AND.                      &
                     H_val( l ) == zero ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )
                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
              K_part%k_fixed_d = K_part%k_fixed_d + 1
              K%row( K_part%k_fixed_d ) = ii
              K%col( K_part%k_fixed_d ) = ii
              K%val( K_part%k_fixed_d ) = H_val( l )
            END IF
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            ii = PERM( i )
            IF ( B_stat( i ) == 0 ) THEN  ! ** free variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                IF ( B_stat( j ) == 0 ) THEN ! ** free variable
                  IF ( ABS( ii - jj ) <= nsemib ) THEN
                    K_part%k_free_od = K_part%k_free_od + 1
                    CALL QPA_lower( K%row( K_part%k_free_od ),                 &
                                     K%col( K_part%k_free_od ), ii, jj )
                    K%val( K_part%k_free_od ) = H_val( l )
                    S( ii ) = S( ii ) + ABS( H_val( l ) )
                    S( jj ) = S( jj ) + ABS( H_val( l ) )
                    IF ( factor == 0 .OR. factor == 1 ) THEN
                      IF ( printi ) WRITE( control%out, 2220 ) prefix
                      factor = 2
                    END IF
                  ELSE
                    G_eq_H = .FALSE.
                  END IF
                ELSE ! ** fixed variable
                  K_part%k_fixed_od = K_part%k_fixed_od + 1
                  CALL QPA_lower( K%row( K_part%k_fixed_od ),                  &
                                   K%col( K_part%k_fixed_od ), ii, jj )
                  K%val( K_part%k_fixed_od ) = H_val( l )
                END IF
              END DO
              K_part%k_free_p = K_part%k_free_p + 1
              K%row( K_part%k_free_p ) = ii
              K%col( K_part%k_free_p ) = ii
              K%val( K_part%k_free_p ) = zero
              IF ( factor == 0 .OR. factor == 1 ) THEN
                IF ( printi ) WRITE( control%out, 2200 ) prefix
                factor = 2
              END IF
            ELSE  ! ** fixed variable
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l ) ; jj = PERM( j )
                K_part%k_fixed_od = K_part%k_fixed_od + 1
                CALL QPA_lower( K%row( K_part%k_fixed_od ),                    &
                                 K%col( K_part%k_fixed_od ), ii, jj )
                K%val( K_part%k_fixed_od ) = H_val( l )
              END DO
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO

      END SELECT

      hmax = MAX( MAXVAL( ABS( S( : K%n ) ) ), one )

!  Constraint Jacobian terms
!  .........................

      DO jj = 1, K_part%c_ref
        i = REF( jj )
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          IF ( B_stat( A_col( l ) ) == 0 ) THEN ! ** free variable
            K_part%k_free_od = K_part%k_free_od + 1
            CALL QPA_lower( K%row( K_part%k_free_od ),                         &
                             K%col( K_part%k_free_od ),                        &
                             PERM( n + jj ), PERM( A_col( l ) ) )
            K%val( K_part%k_free_od ) = A_val( l )
          ELSE  ! ** fixed variable
            K_part%k_fixed_od = K_part%k_fixed_od + 1
            CALL QPA_lower( K%row( K_part%k_fixed_od ),                        &
                             K%col( K_part%k_fixed_od ),                       &
                             PERM( n + jj ), PERM( A_col( l ) ) )
            K%val( K_part%k_fixed_od ) = A_val( l )
          END IF
        END DO
      END DO

!  Record the dimension and number of nonzeros in the "free" part of the
!  reference matrix

      K%ne = K_part%k_free_d
      K_part%n_ref = n + K_part%c_ref

      IF ( printi .AND. K_part%k_free_od + K_part%n_free /= K_part%k_free_d)   &
        THEN ; WRITE( out, "( ' too few diagonals ', I0, 1X, I0 )" )           &
          K_part%k_free_od + K_part%n_free, K_part%k_free_d
        factor = 2
      END IF

      IF ( factor == 0 .OR. factor == 1 ) THEN
        IF ( K_part%n_free > 0 .AND. K_part%c_ref >= 0 ) THEN
          CALL QPA_form_Schur_complement(                                      &
                          K_part%n_free, K_part%c_ref, K_part%k_free_od,       &
                          K%row( : K_part%k_free_od ),                         &
                          K%col( : K_part%k_free_od ),                         &
                          K%val( : K_part%k_free_od ),                         &
                          K%row( K_part%k_free_od + 1 : K_part%k_free_d ),     &
                          K%col( K_part%k_free_od + 1 : K_part%k_free_d ),     &
                          K%val( K_part%k_free_od + 1 : K_part%k_free_d ),     &
                          Abycol_ne, Abycol_val, Abycol_row, Abycol_ptr,       &
                          S_ne, S_val, S_row, S_col, S_colptr,                 &
                          ierr, factor, max_col, prefix,                       &
                          print_level, control, inform )
          IF ( ierr == 5 ) THEN
            IF ( printi ) WRITE( control%out, 2210 ) prefix
            factor = 2
          END IF
        END IF
      END IF

!  Analyse the sparsity pattern of the free sub-block of the reference matrix

      mo = ' ' ; G_perturb = zero ; n_perturb = 0
      SLS_control%relative_pivot_tolerance =                                   &
        control%SLS_control%relative_pivot_tolerance

  20  CONTINUE
      IF ( K%n > 0 ) THEN
!CNTLA%ldiag = 10000 ; CNTLA%sp = 89
!CNTLA%lp = 89 ; CNTLA%wp = 89 ; CNTLA%mp = 89
!write(89,*) K%n, K%ne
!write(89,*) K%row( : K%ne )
!write(89,*) K%col( : K%ne )
!write(89,*) K%val( : K%ne )

        CALL SMT_put( K%type, 'COORDINATE', i )
        SLS_control = control%SLS_control
!        IF ( factor == 0 .OR. factor == 1 ) THEN
!          SLS_control%pivot_control = 2
!          CALL SLS_initialize_solver( control%definite_linear_solver,         &
!                                      SLS_data, inform%SLS_inform )
!        ELSE
          CALL SLS_initialize_solver( control%symmetric_linear_solver,         &
                                      SLS_data, inform%SLS_inform )
!        END IF

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_analyse( K, SLS_data, SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse =                                            &
          inform%time%clock_analyse + time_now - time_record
        inform%factorization_status = inform%SLS_inform%status
        inform%factorization_integer = inform%SLS_inform%integer_size_factors
        inform%factorization_real = inform%SLS_inform%real_size_factors

        IF ( printi ) THEN
          WRITE( out, "(  A, ' SLS: analysis complete:      status = ', I0 )" )&
            prefix, inform%SLS_inform%status
        ELSE IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printe ) WRITE( control%error, 2000 )                           &
            prefix, inform%SLS_inform%status, 'SLS_analyse'
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 )                  &
          WRITE( control%out, "( A, ' ** analysis time = ', F10.2 ) " )        &
            prefix, time_now - time_record

!  Check for error returns

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( inform%status == GALAHAD_error_allocate .OR.                    &
               inform%status == GALAHAD_error_deallocate ) THEN
            IF ( auto_prec ) THEN
              IF ( prec_hist == 1 ) THEN
                precon = 2
                GO TO 10
              ELSE IF ( nsemib > 0 ) THEN
                nsemib = 0
                GO TO 10
              END IF
            ELSE IF ( auto_fact ) THEN
              IF ( factor /= 2 ) THEN
                IF ( printi ) WRITE( control%out, 2220 ) prefix
                factor = 2
                GO TO 10
              END IF
            END IF
          END IF
          inform%status = GALAHAD_error_analysis ; jumpto = 1 ; RETURN
        END IF

        IF ( printi ) WRITE( control%out, "( /, A, ' nnz, real/integer space', &
       &  ' anticipated for factors: ', I0, ', ', I0, ' & ', I0 )" ) prefix,   &
             K%ne, inform%factorization_real, inform%factorization_integer

!  Check that there is too much fill in

        IF ( auto_prec .AND. precon == 1 ) THEN
          IF ( MAX( inform%factorization_integer, inform%factorization_real )  &
               > control%full_max_fill * K%ne ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( /, ' ... too much fill-in, changing preconditioner ... ' )" )
            precon = 3
            GO TO 10
          END IF
        END IF

      ELSE
        inform%factorization_integer = 0
        inform%factorization_real = 0
      END IF

!  Factorize the reference matrix

  30  CONTINUE
      IF ( K%n > 0 ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( K, SLS_data, SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + time_now - time_record
        inform%factorization_status = inform%SLS_inform%status

!  Record the storage required

        inform%nfacts = inform%nfacts + 1
        inform%factorization_integer = inform%SLS_inform%integer_size_factors
        inform%factorization_real = inform%SLS_inform%real_size_factors

        IF ( printi ) THEN
          WRITE( out,                                                          &
          "( A, ' SLS: factorization complete: status = ', I0 )" ) prefix,     &
            inform%SLS_inform%status
        ELSE IF ( printe ) THEN
          WRITE( control%error, 2000 )                                         &
            prefix, inform%SLS_inform%status, 'SLS_factorize'
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 )                  &
          WRITE( control%out, "( A, ' ** factorize time = ', F10.2 ) " )       &
            prefix, time_now - time_record

!  Test that the factorization succeeded

        zeig = 0
        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( inform%status == GALAHAD_error_allocate .OR.                    &
               inform%status == GALAHAD_error_deallocate ) THEN
            IF ( auto_prec ) THEN
              IF ( prec_hist == 1 ) THEN
                precon = 2
                GO TO 10
              ELSE IF ( nsemib > 0 ) THEN
                nsemib = 0
                GO TO 10
              END IF
            ELSE IF ( auto_fact ) THEN
              IF ( factor /= 2 ) THEN
                 IF ( printi ) WRITE( control%out, 2220 ) prefix
                 factor = 2
                GO TO 10
              END IF
            END IF
          END IF
          inform%status = GALAHAD_error_factorization ; jumpto = 1 ; RETURN

!  Record warning conditions

        ELSE IF (inform%SLS_inform%status > 0 ) THEN
          IF ( printt ) WRITE( control%out, 2100 )                             &
                          prefix, inform%SLS_inform%status, 'SLS_factorize'
        END IF

        zeig = K%n - inform%SLS_inform%rank
        IF ( zeig > 0 .AND. printt ) WRITE( control%out,                       &
          "( A, ' ** Matrix has ', I0, ' zero eigenvalues ' )" ) prefix, zeig

        IF ( printi ) WRITE( control%out, "( /, A, ' nnz, real/integer space', &
       &  ' required for factors: ', I0, ', ', I0, ' & ', I0 )" ) prefix,      &
             K%ne, inform%factorization_real, inform%factorization_integer

!  Check that there is too much fill in

        IF ( auto_prec .AND. precon == 1 ) THEN
          IF ( MAX( inform%factorization_integer, inform%factorization_real )  &
               > control%full_max_fill * K%ne ) THEN
            IF ( printi ) WRITE( control%out,                                  &
              "( /, ' ... too much fill-in, changing preconditioner ... ' )" )
            precon = 3
            GO TO 10
          END IF
        END IF

!  Test that the problem is convex in the null-space of the constraints

        modify_k = zeig + inform%SLS_inform%negative_eigenvalues > K_part%c_ref

!  As a precaution, check that the pivots are not too small

        IF ( .NOT. modify_k ) THEN

! Allocate temporary workspace

!         ALLOCATE( DIAG( 2, K%n ) )

!  Determine the block diagonal part of the factors

          CALL SLS_enquire( SLS_data, inform%SLS_inform, D = DIAG )

!  Compute the smallest and largest eigenvalues of the block diagonal factor

!         big = one / MAX( control%zero_pivot, epsmch )
          big = one / epsmch

!  Loop over the diagonal blocks

          twobytwo = .FALSE.
          DO i = 1, K%n
            IF ( twobytwo ) THEN
              twobytwo = .FALSE.
              CYCLE
            END IF
            IF ( i < K%n ) THEN

!  A 2x2 block

              IF ( DIAG( 2, i ) /= zero ) THEN
                twobytwo = .TRUE.
                CALL ROOTS_quadratic(                                          &
                         DIAG( 1, i ) * DIAG( 1, i + 1 ) - DIAG( 2, i ) ** 2,  &
                         - DIAG( 1, i ) - DIAG( 1, i + 1 ), one,               &
                         epsmch, nroots, root1, root2, roots_debug )
                IF ( ABS( root1 ) >= big .OR. root1 == zero .OR.               &
                     ABS( root2 ) >= big .OR. root2 == zero ) THEN
                  modify_k = .TRUE.
                  zeig = 1
                  EXIT
                END IF

!  A 1x1 block

              ELSE
                root1 = DIAG( 1, i )
                IF ( ABS( root1 ) >= big .OR. root1 == zero ) THEN
                  modify_k = .TRUE.
                  zeig = 1
                  EXIT
                END IF
              END IF
            ELSE

!  The final 1x1 block

              root1 = DIAG( 1, i )
              IF ( ABS( root1 ) >= big .OR. root1 == zero ) THEN
                modify_k = .TRUE.
                zeig = 1
                EXIT
              END IF
            END IF
          END DO
          IF ( printt .AND. modify_k ) WRITE( control%out,                     &
            "( A, ' ** Matrix has at least one zero eigenvalue ' )" ) prefix
        END IF

        IF ( modify_k ) THEN
          IF ( mo == ' ' ) THEN
            inform%nmods = inform%nmods + 1
            mo = 'm'
            K%ne = K_part%k_free_p
          END IF
          G_perturb = G_perturb + hmax
          K%val( K_part%k_free_od + 1 : K_part%k_free_p )                      &
            = K%val( K_part%k_free_od + 1 : K_part%k_free_p ) + hmax
          IF ( printi ) WRITE( out,                                            &
            "( /, A, ' Preconditioner is inappropriate as it has ',            &
          &   /, A, 1X, I0, ' negative and ', I0,                              &
          &    ' zero eigenvalues as opposed to ', I0, ' negative one', A,     &
          &   //, A, ' Perturbing G', :, ' by', ES14.6, ' and restarting',     &
          &          ' (pivtol = ', ES9.2, ')', / )" )                         &
            prefix, prefix, inform%SLS_inform%rank, zeig, K_part%c_ref,        &
            TRIM( STRING_pleural( K_part%c_ref ) ), prefix, G_perturb,         &
            SLS_control%relative_pivot_tolerance
          n_perturb = n_perturb + 1
          IF ( n_perturb > 5 ) THEN
            IF ( precon == 2 ) THEN
              IF ( printi ) WRITE( control%out, "( /, A, ' terminating - ',    &
             &       ' constraints are too ill conditioned ' )" ) control%prefix
              inform%status = GALAHAD_error_ill_conditioned
              RETURN
            END IF
            precon = 2
            GO TO 10
          END IF
          SLS_control%relative_pivot_tolerance =                               &
            MIN( half, ten * SLS_control%relative_pivot_tolerance )

          G_eq_H = .FALSE.
          IF (  zeig > 0 .AND. check_dependent ) THEN
            jumpto = 2 ; RETURN
          END IF
          IF ( K_part%k_free_p == K_part%k_free_d ) THEN
            GO TO 30
          ELSE
            GO TO 20
          END IF
        END IF
      END IF
      check_dependent = .FALSE.

      inform%status = GALAHAD_ok ; jumpto = 0
      RETURN

!  Allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate ; jumpto = 1
      IF ( printi ) WRITE( control%out,                                        &
         "( A, ' ** Message from -QPA_factorize_reference-', /, A,             &
      &        ' Allocation error, for ', A, ', status = ', I0 )" )            &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPA_solve_main ' )" )

      RETURN

!  Non-executable statements

 2000 FORMAT( A, '   **  Error return ', I0, ' from ', A15 )
 2100 FORMAT( A, '   **  Warning ', I0, ' from ', A15 )
 2200 FORMAT( /, A, ' zero diagonal - switching to augmented system method ', /)
 2210 FORMAT( /, A, ' Schur complement failure ... switching to augmented',    &
                 ' system method ', / )
 2220 FORMAT( /, A, ' off-diagonal term - switching to augmented system method')

!  End of QPA_factorize_reference

      END SUBROUTINE QPA_factorize_reference

!-*-*-*-*-   Q P A _ N E W _ R E F E R E N C E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPA_new_reference_set( control, inform, prefix, n, m, K_part, &
                                        SCU_mat, out, m_link, pcount, printd,  &
                                        warmer_start, check_dependent, C_stat, &
                                        B_stat, SC, REF, WORKING, A_ptr,       &
                                        A_col, A_val, K, P, SOL, D,            &
                                        SLS_data, SLS_control )

!  Use the current components of SC and REF to determine a new reference set

!  Dummy arguments

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_partition_type ), INTENT( INOUT ) :: K_part
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      INTEGER, INTENT( IN ) :: n, m, out, m_link
      INTEGER, INTENT( OUT ) :: pcount
      LOGICAL, INTENT( IN ) :: printd, warmer_start
      LOGICAL, INTENT( INOUT ) :: check_dependent

      INTEGER, INTENT( OUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( IN ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( INOUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( OUT ), DIMENSION( m ) :: WORKING
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: P
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( :, : ) :: D
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

!  Local variables

      INTEGER :: i, ii, jj, j, l, new_m_ref, n_depen

      pcount = 0
      C_stat = 0 ; B_stat = 0 ; K_part%c_ref = 0

!  Find the new reference set

      new_m_ref = 0
      DO i = 1, K_part%m_ref
        j = REF( i )
        IF ( j > 0 ) THEN
          new_m_ref = new_m_ref + 1
          REF( new_m_ref ) = j
          IF ( j <= m ) THEN
            C_stat( j ) = - new_m_ref
            K_part%c_ref = K_part%c_ref + 1
          ELSE
            B_stat( j - m ) = - new_m_ref
          END IF
        END IF
      END DO

      DO i = 1, SCU_mat%m
        j = SC( i )
        IF ( j > 0 ) THEN
          new_m_ref = new_m_ref + 1
          REF( new_m_ref ) = j
          IF ( j <= m ) THEN
            C_stat( j ) = - new_m_ref
            K_part%c_ref = K_part%c_ref + 1
          ELSE
            B_stat( j - m ) = - new_m_ref
          END IF
        END IF
      END DO

!  Now, reorder the references set so that general constraints appear
!  before bounds

      ii = new_m_ref
 order: DO i = 1, new_m_ref
        j = REF( i )
        IF ( j <= m ) CYCLE

!  A bound occurs before a general constraint

        DO jj = ii, i + 1, - 1
          l = REF( jj )
          IF ( l > m ) CYCLE

!  A general constraint occurs after a bound; swap them

          REF( i ) = l
          REF( jj ) = j
          B_stat( j - m ) = - jj
          C_stat( l ) = - i
          ii = jj - 1
          IF ( i >= ii ) EXIT order
          CYCLE order
        END DO
        EXIT order
      END DO order

      IF ( warmer_start .OR. check_dependent ) THEN
        ii = 0
        DO
          CALL QPA_remove_dependent( n, m, A_val, A_col, A_ptr, K, SLS_data,   &
                                     SLS_control, C_stat, B_stat, WORKING, P,  &
                                     SOL, D, prefix, control, inform, n_depen )
          IF ( n_depen == 0 ) THEN
            IF ( ii > 0 ) THEN

!  Update the reference set to account for removed bounds

              K_part%m_ref = new_m_ref
              K_part%c_ref = 0
              new_m_ref = 0
              DO i = 1, K_part%m_ref
                j = REF( i )
                IF ( j <= m ) THEN
                  IF ( C_stat( j ) == 0 .AND. printd )                         &
                    WRITE( out, "( ' constraint ', i7, ' rejected ' )" ) j
                  IF ( C_stat( j ) == 0 ) CYCLE
                ELSE
                  IF ( B_stat( j - m ) == 0 .AND. printd )                     &
                    WRITE( out, "( ' bound ', i7, ' rejected ' )" ) j - m
                  IF ( B_stat( j - m ) == 0 ) CYCLE
                END IF
                new_m_ref = new_m_ref + 1
                REF( new_m_ref ) = j
                IF ( j <= m ) THEN
                  C_stat( j ) = - new_m_ref
                  K_part%c_ref = K_part%c_ref + 1
                ELSE
                  B_stat( j - m ) = - new_m_ref
                END IF
              END DO
            END IF
            EXIT
          END IF
!         WRITE(6,"('new_m_ref', I6)" ) new_m_ref
          ii = ii + 1
        END DO
        check_dependent = .FALSE.
      ELSE
        check_dependent = .TRUE.
      END IF

      IF ( printd ) THEN
        WRITE( out, "( ' REF(c) ', /, ( 10I5 ) )" ) REF( : K_part%c_ref )
        WRITE( out, "( ' REF(b) ', /, ( 10I5 ) )" )                            &
          REF( K_part%c_ref + 1 : new_m_ref ) - m
      END IF

! Set up the reference matrix

      SCU_mat%m = 0
      K_part%m_ref = new_m_ref
      K_part%k_ref = n + K_part%m_ref ; SCU_mat%n = K_part%k_ref

      RETURN

!  End of QPA_new_reference_set

      END SUBROUTINE QPA_new_reference_set

!-*-*-*-*-*-*-*-*-*-   Q P A _ A D D _ C O N S T R A I N T   -*-*-*-*-*-*-*-*-*-

      SUBROUTINE QPA_add_constraint( QPA_add_constraint_status, control,       &
                     inform, dims, n, m, K_part, active, out, k_n_max, m_link, &
                     itref_max, j_add, j_del, scu_status, m_active, s_plus,    &
                     s_minus, printt, printm, printd, addel, sc_data, C_stat,  &
                     B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low, B, RES,  &
                     RES_print, VECTOR, PERT, A_ptr, A_col, A_val, SCU_mat,    &
                     SCU_info, SCU_data, K, SLS_control, SLS_data )

!  Adjust the Schur complement when adding a constraint to the working set

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      INTEGER, INTENT( IN ) :: n, m, active, out, k_n_max, m_link, itref_max
      INTEGER, INTENT( OUT ) :: QPA_add_constraint_status
      INTEGER, INTENT( OUT ) :: j_add, j_del, scu_status
      INTEGER, INTENT( INOUT ) :: m_active, s_plus, s_minus
      LOGICAL, INTENT( IN ) :: printt, printm, printd
      CHARACTER ( LEN = 10 ), INTENT( INOUT ) :: addel
      CHARACTER ( LEN = 12 ), INTENT( OUT)  :: sc_data
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( INOUT ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( INOUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( IN ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, INTENT( INOUT ),                                                &
               DIMENSION( dims%c_u_start : dims%c_l_end ) :: C_up_or_low
      INTEGER, INTENT( INOUT ),                                                &
               DIMENSION( dims%x_u_start : dims%x_l_end ) :: X_up_or_low
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION ( k_n_max + control%max_sc ) :: B, RES, RES_print
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( k_n_max ) :: VECTOR
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m + n ) :: PERT
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control

!  Local variables

      INTEGER :: i, ii, j, jj, ll, lll, nbd
      LOGICAL :: on_up, x_r0, y0, x_f0, z0
      CHARACTER ( LEN = 2 ) :: uplow

      j_del = 0
      j = active
      IF ( j < 0 ) THEN
        on_up = .TRUE.
        j = - j
      ELSE
        on_up = .FALSE.
      END IF
      IF ( j <= m ) THEN
        ii = C_stat( j )
      ELSE
        ii = B_stat( j - m )
      END IF
      IF ( ii < 0 ) ll = - REF( - ii )

!  Add a constraint
!  ================

      j_add = j
      m_active = m_active + 1

      IF ( j <= m ) THEN

!  general constraint

        jj = j
        IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
          IF ( on_up ) THEN
            C_up_or_low( j ) = 1
            uplow = 'cU'
          ELSE
            C_up_or_low( j ) = - 1
            uplow = 'cL'
          END IF
        ELSE IF ( j > dims%c_l_end ) THEN
          uplow = 'cu'
        ELSE IF ( j <= dims%c_equality ) THEN
          uplow = 'ce'
        ELSE
          uplow = 'cl'
        END IF
      ELSE

!  simple bound

        jj = j - m
        IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
          IF ( on_up ) THEN
            X_up_or_low( jj ) = 1
            uplow = 'bU'
          ELSE
            X_up_or_low( jj ) = - 1
            uplow = 'bL'
          END IF
        ELSE IF ( jj > dims%x_l_end ) THEN
          uplow = 'bu'
        ELSE
          uplow = 'bl'
        END IF
      END IF
      WRITE( addel, "( SP, I7, A2, A1 )" ) jj, uplow, ' '

!  Constraint has never been in the reference set
!  ----------------------------------------------

!     write(6,*) ' add '
      IF ( ii == 0 ) THEN

        s_plus = s_plus + 1
        IF ( j <= m ) THEN
          C_stat( j ) = SCU_mat%m + 1
        ELSE
          B_stat( j - m ) = SCU_mat%m + 1
        END IF
        SC( SCU_mat%m + 1 ) = j

        IF ( printd ) WRITE( out, "( ' constraint ', i5, ' added in position ',&
       &                              i5 )" ) j, SCU_mat%m + 1
        IF ( printm .AND. PERT( j ) /= zero )                                  &
          WRITE( out, "( ' ==> previously deleted reference ', I5,             &
       &    ' would have been reintroduced,', /,                               &
       &        '     but H has been modified' )" ) j

!  Exit minor iteration when SC is too large

        IF ( SCU_mat%m == control%max_sc ) THEN
          SCU_mat%m = SCU_mat%m + 1
          WRITE( sc_data, "( I4, 8X )" ) SCU_mat%m
          addel = '          '
          QPA_add_constraint_status = 1
          RETURN
        END IF

!  Constraint j will be added

        nbd = SCU_mat%BD_col_start( SCU_mat%m + 1 ) - 1
        IF ( j <= m ) THEN

!  general constraint

          x_r0 = .TRUE. ; x_f0 = .TRUE.
          DO lll = A_ptr( j ), A_ptr( j + 1 ) - 1
            nbd = nbd + 1
            SCU_mat%BD_row( nbd ) = PERM( A_col( lll ) )
            SCU_mat%BD_val( nbd ) = A_val( lll )
            IF ( B_stat( A_col( lll ) ) >= 0 ) THEN
              x_r0 = .FALSE.
            ELSE
              x_f0 = .FALSE.
            END IF
          END DO
          y0 = .TRUE. ; z0 = .TRUE.

        ELSE

!  simple bound

          nbd = nbd + 1
          SCU_mat%BD_row( nbd ) = PERM( jj )
          SCU_mat%BD_val( nbd ) = one
          x_r0 = .FALSE. ; y0 = .TRUE. ; x_f0 = .TRUE. ; z0 = .TRUE.
        END IF
        SCU_mat%BD_col_start( SCU_mat%m + 2 ) = nbd + 1

!  Update the Schur complement

        scu_status = 1
        IF ( printt ) WRITE ( out, "( '  adding with SCU_append ' )" )
        DO
          CALL SCU_append( SCU_mat, SCU_data, VECTOR, scu_status, SCU_info )
          IF ( scu_status <= 0 ) EXIT
          CALL QPA_ir( K, SLS_data, K_part, VECTOR, B, RES, x_r0, y0, x_f0, z0,&
                       SLS_control, itref_max, out, printm, RES_print, inform )
          x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.

          inform%factorization_status = inform%status
        END DO
        IF ( scu_status /= 0 ) THEN
          IF ( scu_status == - 9 ) SCU_mat%m = SCU_mat%m - 1
          QPA_add_constraint_status = 2
          RETURN
        END IF
      ELSE

!  Constraint was once in the reference set
!  ----------------------------------------

        s_minus = s_minus - 1

        IF ( printm ) WRITE( out, "( ' previously deleted reference ', I5,     &
       &                     ' to be reintroduced. In SC as ', I5 )" ) j, ll

        REF( - ii ) = j
        DO i = ll, SCU_mat%m - 1
           j = SC( i + 1 )
           IF ( j > 0 ) THEN  ! ** active other
             IF ( j <= m ) THEN ! ** general constraint
               C_stat( j ) = C_stat( j ) - 1
             ELSE ! ** simple bound
               B_stat( j - m ) = B_stat( j - m ) - 1
             END IF
           ELSE IF ( j < 0 ) THEN ! ** inactive reference
             IF ( j >= - m ) THEN ! ** general constraint
               ii = - C_stat( - j )
             ELSE ! ** simple bound
               ii = - B_stat( - j - m )
             END IF
             REF( ii ) = REF( ii ) + 1
           END IF
           SC( i ) = j
        END DO

!  Update the Schur complement

        IF ( printt ) WRITE ( out, "( '  adding with SCU_delete ' )" )
        CALL SCU_delete( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),             &
                         scu_status, SCU_info, ll )
        IF ( scu_status /= 0 ) THEN
          QPA_add_constraint_status = 3
          RETURN
        END IF

      END IF

      IF ( SCU_info%inertia( 1 ) /= s_minus ) THEN
        QPA_add_constraint_status = 4
        RETURN
      END IF
      WRITE( sc_data, "( 3I4 )" ) SCU_mat%m, SCU_info%inertia( 1 : 2 )

      QPA_add_constraint_status = GALAHAD_ok
      RETURN

!  End of QPA_add_constraint

      END SUBROUTINE QPA_add_constraint


!-*-*-*-*-*-*-*-   Q P A _ D E L E T E _ C O N S T R A I N T   -*-*-*-*-*-*-*-

      SUBROUTINE QPA_delete_constraint( QPA_delete_constraint_status, control, &
                   inform, dims, n, m, K_part, inactive, out, k_n_max, m_link, &
                   itref_max, j_add, j_del, scu_status, m_active, s_plus,      &
                   s_minus, printt, printm, printd, printe, addel, sc_data,    &
                   C_stat, B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low,    &
                   B, RES, RES_print, VECTOR, PERT, SCU_mat,                   &
                   SCU_info, SCU_data, K, SLS_control, SLS_data )

!  Adjust the Schur complement when deleting a constraint from the working set

      TYPE ( QPA_control_type ), INTENT( IN ) :: control
      TYPE ( QPA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      TYPE ( QPA_partition_type ), INTENT( IN ) :: K_part
      INTEGER, INTENT( IN ) :: n, m, inactive, out, k_n_max, m_link, itref_max
      INTEGER, INTENT( OUT ) :: QPA_delete_constraint_status
      INTEGER, INTENT( OUT ) :: j_add, j_del, scu_status
      INTEGER, INTENT( INOUT ) :: m_active, s_plus, s_minus
      LOGICAL, INTENT( IN ) :: printt, printm, printd, printe
      CHARACTER ( LEN = 10 ), INTENT( OUT ) :: addel
      CHARACTER ( LEN = 12 ), INTENT( OUT ) :: sc_data
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( INOUT ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( INOUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( IN ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, INTENT( INOUT ),                                                &
               DIMENSION( dims%c_u_start : dims%c_l_end ) :: C_up_or_low
      INTEGER, INTENT( INOUT ),                                                &
               DIMENSION( dims%x_u_start : dims%x_l_end ) :: X_up_or_low
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION ( k_n_max + control%max_sc ) :: B, RES, RES_print
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( k_n_max ) :: VECTOR
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m + n ) :: PERT
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control

!  Local variables

      INTEGER :: i, ii, j, jj, jjj, ll, nbd
      LOGICAL :: x_r0, y0, x_f0, z0
      CHARACTER ( LEN = 2 ) :: uplow

      j = inactive
      IF ( j <= m ) THEN
        ii = C_stat( j )
      ELSE
        ii = B_stat( j - m )
      END IF

!  Delete a constraint
!  ===================

      m_active = m_active - 1

      j_add = 0
      IF ( j <= m ) THEN

!  general constraint

        jj = j
        IF ( j >= dims%c_u_start .AND. j <= dims%c_l_end ) THEN
          IF ( C_up_or_low( j ) == - 1 ) THEN
            uplow = 'cL'
            j_del = - j
          ELSE
            uplow = 'cU'
            j_del = j
          END IF
          C_up_or_low( j ) = 0
        ELSE IF ( j > dims%c_l_end ) THEN
          uplow = 'cu'
          j_del = j
        ELSE IF ( j <= dims%c_equality ) THEN
          uplow = 'ce'
          j_del = - j
        ELSE
          uplow = 'cl'
          j_del = - j
        END IF
      ELSE

!  simple bound

        jj = j - m
        IF ( jj >= dims%x_u_start .AND. jj <= dims%x_l_end ) THEN
          IF ( X_up_or_low( jj ) == - 1 ) THEN
            uplow = 'bL'
            j_del = - j
          ELSE
            uplow = 'bU'
            j_del = j
          END IF
          X_up_or_low( jj ) = 0
        ELSE IF ( jj > dims%x_l_end ) THEN
          uplow = 'bu'
          j_del = j
        ELSE
          uplow = 'bl'
          j_del = - j
        END IF
      END IF
      WRITE( addel, "( SP, I7, A2, A1 )" ) - jj, uplow, ' '

      IF ( printd ) WRITE( out,                                                &
        "( ' constraint ', i5, ' deleted from position ', I5 )" ) j, ii

      IF ( ii > 0 ) THEN

!  Constraint is in the Schur complement
!  -------------------------------------

        s_plus = s_plus - 1

        IF ( j <= m ) THEN
          C_stat( j ) = 0
        ELSE
          B_stat( j - m ) = 0
        END IF

        DO i = ii, SCU_mat%m - 1
          jjj = SC( i + 1 )
          IF ( jjj > 0 ) THEN  ! ** active other
            IF ( jjj <= m ) THEN ! ** general constraint
              C_stat( jjj ) = C_stat( jjj ) - 1
            ELSE ! ** simple bound
              B_stat( jjj - m ) = B_stat( jjj - m ) - 1
            END IF
          ELSE IF ( jjj < 0 ) THEN ! ** inactive reference
            IF ( jjj >= - m ) THEN ! ** general constraint
              ll = - C_stat( - jjj )
            ELSE ! ** simple bound
              ll = - B_stat( - jjj - m )
            END IF
            REF( ll ) = REF( ll ) + 1
          END IF
          SC( i ) = jjj
        END DO

!  Update the Schur complement

        IF ( printt ) WRITE ( out, "( '  deleting with SCU_delete ' )" )
        CALL SCU_delete( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),             &
                         scu_status, SCU_info, ii )
        IF ( scu_status /= 0 ) THEN
          QPA_delete_constraint_status = 3
          RETURN
        END IF
      ELSE

!  Constraint is not in the Schur complement
!  -----------------------------------------

        s_minus = s_minus + 1

!  Delete the constraint by adding an appropriate row/column
!  to the Schur complement

        IF ( j <= m ) THEN
          jj = - C_stat( j )
          x_r0 = .TRUE. ; y0 = .FALSE. ; x_f0 = .TRUE. ; z0 = .TRUE.
        ELSE
          jj = - B_stat( j - m )
          x_r0 = .TRUE. ; y0 = .TRUE. ; x_f0 = .TRUE. ; z0 = .FALSE.
        END IF

        REF( jj ) = - ( SCU_mat%m + 1 )
        SC( SCU_mat%m + 1 ) = - j

!  Exit when SC is too large

        IF ( SCU_mat%m == control%max_sc ) THEN
          SCU_mat%m = SCU_mat%m + 1
          WRITE( sc_data, "( I4, 8X )" ) SCU_mat%m
          addel = '         '
          QPA_delete_constraint_status = 1
          RETURN
        END IF

!  Constraint j will be removed

        nbd = SCU_mat%BD_col_start( SCU_mat%m + 1 )
        SCU_mat%BD_row( nbd ) = PERM( n - ii )
        SCU_mat%BD_val( nbd ) = one
        SCU_mat%BD_col_start( SCU_mat%m + 2 ) = nbd + 1

!  Update the Schur complement

        scu_status = 1
        IF ( printt ) WRITE ( out, "( '  deleting with SCU_append ' )" )
        DO
          CALL SCU_append( SCU_mat, SCU_data, VECTOR, scu_status, SCU_info)
          IF ( scu_status <= 0 ) EXIT
          CALL QPA_ir( K, SLS_data, K_part, VECTOR, B, RES, x_r0, y0, x_f0, z0,&
                       SLS_control, itref_max, out, printm, RES_print, inform )
          x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.
          inform%factorization_status = inform%status
        END DO
        IF ( SCU_info%inertia( 1 ) == s_minus .AND. scu_status /= 0            &
             .AND. printe ) WRITE ( out, 2020 ) scu_status
        IF ( scu_status == - 9 ) THEN
          IF ( printm ) WRITE ( out, "( /, '  ... modifying preconditioner ' )")
        ELSE IF ( scu_status /= 0 ) THEN
          QPA_delete_constraint_status = 2
          RETURN
        END IF
      END IF

!  If the matrix is no longer regular, add an appropriate diagonal term to
!  the Schur complement

      IF ( SCU_info%inertia( 1 ) /= s_minus ) THEN
        addel( 10 : 10 ) = 'm'
        CALL SCU_increase_diagonal( SCU_data, PERT( j ), SCU_info )
        IF ( printm ) WRITE ( out,                                             &
          "( ' =*=*=*=*=*= Hessian modified: diagonal = ', ES12.4 )" ) PERT( j )

!  Add the diagonal perturbation to the Schur-complement storage

        nbd = SCU_mat%BD_col_start( SCU_mat%m + 1 )
        SCU_mat%BD_row( nbd ) = SCU_mat%n + SCU_mat%m
        SCU_mat%BD_val( nbd ) = PERT( j )
        SCU_mat%BD_col_start( SCU_mat%m + 1 ) = nbd + 1
      END IF

      IF ( SCU_info%inertia( 1 ) /= s_minus ) THEN
        QPA_delete_constraint_status = 4
        RETURN
      END IF
      WRITE( sc_data, "( 3I4 )" ) SCU_mat%m, SCU_info%inertia( 1 : 2 )

      QPA_delete_constraint_status = GALAHAD_ok
      RETURN

!  Non-executable statements

 2020 FORMAT( /, '  on exit from SCU_append,   status = ', I3 )

!  End of QPA_delete_constraint

      END SUBROUTINE QPA_delete_constraint

!  End of module QPA_double

   END MODULE GALAHAD_QPA_double
