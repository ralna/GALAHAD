! THIS VERSION: GALAHAD 3.0 - 10/04/2017 AT 08:50 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ D Q P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 2.5. August 1st 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DQP_double

!     ----------------------------------------------------------
!     |                                                        |
!     | Minimize the quadratic objective function              |
!     |                                                        |
!     |         q(x) = 1/2 x^T H x + g^T x + f                 |
!     |                                                        |
!     | or linear/seprable objective function                  |
!     |                                                        |
!     |     s(x) = 1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                                        |
!     | subject to the linear constraints and bounds           |
!     |                                                        |
!     |             c_l <= A x <= c_u                          |
!     |             x_l <=  x <= x_u                           |
!     |                                                        |
!     | for some positive definite Hessian H or diagonal       |
!     | matrix W using a dual gradient-projection method       |
!     |                                                        |
!     | Optionally, minimize instead the penalty function      |
!     |                                                        |
!     |    q(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1    |
!     |                                                        |
!     | or                                                     |
!     |                                                        |
!     |    s(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1    |
!     |                                                        |
!     | subject to the bound constraints x_l <= x <= x_u       |
!     |                                                        |
!     ----------------------------------------------------------

!$    USE omp_lib
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_verb_pleural,    &
                                       STRING_ies, STRING_are, STRING_ordinal, &
                                       STRING_their, STRING_integer_6
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, DQP_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, DQP_data_type => QPD_data_type,                  &
                              DQP_AX => QPD_AX, DQP_HX => QPD_HX,              &
                              DQP_abs_AX => QPD_abs_AX, DQP_abs_HX => QPD_abs_HX
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
                               SORT_heapsort_smallest, SORT_inverse_permute
      USE GALAHAD_FDC_double
      USE GALAHAD_SLS_double
      USE GALAHAD_SCU_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_NORMS_double, ONLY: TWO_norm
      USE GALAHAD_CHECKPOINT_double
      USE GALAHAD_RPD_double, ONLY: RPD_inform_type, RPD_write_qp_problem_data

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DQP_initialize, DQP_read_specfile, DQP_solve, DQP_solve_main,  &
                DQP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,   &
                DQP_Ax, DQP_data_type, DQP_dims_type, DQP_workspace

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: sp = KIND( 1.0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: relative_pivot_default = 0.01_wp
!     REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 10 )
!     REAL ( KIND = wp ), PARAMETER :: hzero = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 20 )
      REAL ( KIND = wp ), PARAMETER :: hzero = zero
      REAL ( KIND = wp ), PARAMETER :: big_radius = ten ** 10
!     REAL ( KIND = wp ), PARAMETER :: big_radius = ten ** 20
      REAL ( KIND = wp ), PARAMETER :: alpha_search = one
      REAL ( KIND = wp ), PARAMETER :: beta_search = half
      REAL ( KIND = wp ), PARAMETER :: mu_search = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: obj_unbounded = - epsmch ** ( - 2 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DQP_control_type

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

!   printing will only occur every print_gap iterations

        INTEGER :: print_gap = 1

!   which starting point should be used for the dual problem

!     -1 user supplied comparing primal vs dual variables
!      0 user supplied
!      1 minimize linearized dual
!      2 minimize simplified quadratic dual
!      3 all free (= all active primal costraints)
!      4 all fixed on bounds (= no active primal costraints)

       INTEGER :: dual_starting_point = 0

!   at most maxit inner iterations are allowed

        INTEGER :: maxit = 1000

!   the maximum permitted size of the Schur complement before a refactorization
!    is performed (used in the case where there is no Fredholm Alternative,
!    0 = refactor every iteration)

        INTEGER :: max_sc = 100

!   a subspace step will only be taken when the current Cauchy step has
!    changed no more than than cauchy_only active constraints; the subspace
!    step will always be taken if cauchy_only < 0

       INTEGER :: cauchy_only = - 1

!   how many iterations are allowed per arc search (-ve = as many as required)

        INTEGER :: arc_search_maxit = - 1

!   how many CG iterations to perform per DQP iteration (-ve reverts to n+1)

       INTEGER :: cg_maxit = 1000

!   once a potentially optimal subspace has been found, investigate it
!      0 as per an ordinary subspace
!      1 by increasing the maximum number of allowed CG iterations
!      2 by switching to a direct method

       INTEGER :: explore_optimal_subspace = 0

!   indicate whether and how much of the input problem
!    should be restored on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!    specifies the unit number to write generated QPLIB file describing the
!     current problem

        INTEGER :: qplib_file_device = 53

!    the penalty weight, rho. The general constraints are not enforced
!     explicitly, but instead included in the objective as a penalty term
!     weighted by rho when rho > 0. If rho <= 0, the general constraints are
!     explicit (that is, there is no penalty term in the objective function)

        REAL ( KIND = wp ) :: rho = zero

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   the required absolute and relative accuracies for the primal infeasibility

        REAL ( KIND = wp ) :: stop_abs_p = epsmch
        REAL ( KIND = wp ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

        REAL ( KIND = wp ) :: stop_abs_d = epsmch
        REAL ( KIND = wp ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

        REAL ( KIND = wp ) :: stop_abs_c = epsmch
        REAL ( KIND = wp ) :: stop_rel_c = epsmch

!  the CG iteration will be stopped as soon as the current norm of the
!  preconditioned gradient is smaller than
!    max( stop_cg_relative * initial preconditioned gradient, stop_cg_absolute )

       REAL ( KIND = wp ) :: stop_cg_relative = ten ** ( - 2 )
       REAL ( KIND = wp ) :: stop_cg_absolute = epsmch

!  threshold below which curvature is regarded as zero if CG is used

       REAL ( KIND = wp ) :: cg_zero_curvature = ten * epsmch

!  maximum growth factor allowed without a refactorization

       REAL ( KIND = wp ) :: max_growth = ten ** 7

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!  ------------ for  DLP only ------------

!   the initial penalty weight

        REAL ( KIND = wp ) :: initial_perturbation = point1

!   the penalty weight reduction factor

        REAL ( KIND = wp ) :: perturbation_reduction = point1

!   the final penalty weight

        REAL ( KIND = wp ) :: final_perturbation = ten ** ( - 6 )

!   are the factors of the optimal augmented matrix required?

        LOGICAL :: factor_optimal_matrix = .FALSE.

!  ---------------------------------------

!   the equality constraints will be preprocessed to remove any linear
!    dependencies if true

        LOGICAL :: remove_dependencies = .TRUE.

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if %exact_arc_search is true, an exact piecewise arc search will be
!     performed. Otherwise an ineaxt search using a backtracing Armijo
!     strategy will be employed

        LOGICAL :: exact_arc_search = .TRUE.

!   if %subspace_direct is true, the subspace step will be calculated
!    using a direct (factorization) method, while if it is false, an
!    iterative (conjugate-gradient) method will be used.

        LOGICAL :: subspace_direct = .FALSE.

!   if %subspace_alternate is true, the subspace step will alternate
!    between a direct (factorization) method and an iterative
!    (GLTR conjugate-gradient) method. This will override %subspace_direct

        LOGICAL :: subspace_alternate = .FALSE.

!   if %subspace_arc_search is true, a piecewise arc search will be performed
!     along the subspace step. Otherwise the search will stop at the first
!     constraint encountered

        LOGICAL :: subspace_arc_search = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!   if %generate_qplib_file is .true. if a QPLIB file describing the current
!    problem is to be generated

        LOGICAL :: generate_qplib_file = .FALSE.

!  indefinite linear equation solver set in symmetric_linear_solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "ma57" // REPEAT( ' ', 26 )

!  definite linear equation solver

        CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
           "sils" // REPEAT( ' ', 26 )

!  unsymmetric linear equation solver

        CHARACTER ( LEN = 30 ) :: unsymmetric_linear_solver =                  &
           "gls" // REPEAT( ' ', 27 )

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "DQPPROB.SIF"  // REPEAT( ' ', 18 )

!  name of generated QPLIB file containing input problem

        CHARACTER ( LEN = 30 ) :: qplib_file_name =                            &
         "DQPPROB.qplib"  // REPEAT( ' ', 16 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for GLTR

        TYPE ( GLTR_control_type ) :: GLTR_control

      END TYPE DQP_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DQP_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ):: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

!  the CPU time spent in the linesearch

        REAL ( KIND = wp ) :: search = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = wp ) :: clock_preprocess = 0.0

!  the clock time spent detecting linear dependencies

        REAL ( KIND = wp ) :: clock_find_dependent = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = wp ) :: clock_solve = 0.0

!  the clock time spent in the linesearch

        REAL ( KIND = wp ) :: clock_search = 0.0
      END TYPE DQP_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DQP_inform_type

!  return status. See DQP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the total number of iterations required

        INTEGER :: cg_iter = 0

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = - 1

!  the number of threads used

        INTEGER :: threads = 1

!  the value of the objective function at the best estimate of the solution
!   determined by DQP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  the smallest pivot that was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  checkpoints(i) records the iteration at which the criticality measures
!   first fall below 10**-i, i = 1, ..., 16 (-1 means not achieved)

!      INTEGER, DIMENSION( 16 ) :: checkpoints = - 1
       INTEGER, DIMENSION( 16 ) :: checkpointsIter = - 1
       REAL ( KIND = wp ), DIMENSION( 16 ) :: checkpointsTime = - one

!  timings (see above)

        TYPE ( DQP_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  return information from GLTR

        TYPE ( GLTR_info_type ) :: GLTR_inform

!  inform parameters for SCU

        INTEGER :: scu_status = 0
        TYPE ( SCU_info_type ) :: SCU_inform

!  inform parameters for RPD

        TYPE ( RPD_inform_type ) :: RPD_inform
      END TYPE DQP_inform_type

    CONTAINS

!-*-*-*-*-*-   D Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE DQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for DQP. This routine should be called before
!  DQP_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( DQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DQP_control_type ), INTENT( OUT ) :: control
      TYPE ( DQP_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set real control parameters

      control%stop_abs_p = epsmch ** 0.33
      control%stop_rel_p = epsmch ** 0.33
      control%stop_abs_c = epsmch ** 0.33
      control%stop_rel_c = epsmch ** 0.33
      control%stop_abs_d = epsmch ** 0.33
      control%stop_rel_d = epsmch ** 0.33

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%max_infeas = control%stop_abs_p
      control%FDC_control%prefix = '" - FDC:"                     '

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
!     control%SBLS_control%perturb_to_make_definite = .FALSE.
!     control%SBLS_control%preconditioner = 2
      control%SBLS_control%factorization = 2
      control%SBLS_control%prefix = '" - SBLS:"                    '

!  Set GLTR control parameters

      CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,              &
                            inform%GLTR_inform )
!     control%GLTR_control%unitm = .FALSE.
!     control%GLTR_control%Lanczos_itmax = 5
      control%GLTR_control%steihaug_toint = .TRUE.
      control%GLTR_control%prefix = '" - GLTR:"                    '
      control%GLTR_control%rminvr_zero = epsmch ** 1.25

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of DQP_initialize

      END SUBROUTINE DQP_initialize

!-*-*-*-*-   D Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE DQP_read_specfile( control, device, alt_specname,             &
                                    main_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by DQP_initialize could (roughly)
!  have been set as:

! BEGIN DQP SPECIFICATIONS (DEFAULT)
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
!  explore-optimal-subspace                          0
!  restore-problem-on-output                         2
!  dual-starting-point                               0
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  penalty-weight                                    0.0D+0
!  infinity-value                                    1.0D+19
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-5
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-5
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-5
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  cg-zero-curvature-threshold                       1.0D-15
!  maximum-growth-before-refactorization             1.0D+7
!  identical-bounds-tolerance                        1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  direct-solution-of-subspace-problem               F
!  alternate-solution-of-subspace-problem            F
!  perform-exact-arc-search                          F
!  perform-subspace-arc-search                       T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  sils
!  definite-linear-equation-solver                   sils
!  unsymmetric-linear-equation-solver                gls
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  sif-file-name                                     DQPPROB.SIF
!  qplib-file-name                                   DQPPROB.qplib
!  output-line-prefix                                ""
! END DQP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( DQP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname
      CHARACTER( LEN = * ), OPTIONAL :: main_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: start_print = print_level + 1
      INTEGER, PARAMETER :: stop_print = start_print + 1
      INTEGER, PARAMETER :: print_gap = stop_print + 1
      INTEGER, PARAMETER :: maxit = print_gap + 1
      INTEGER, PARAMETER :: max_sc = maxit + 1
      INTEGER, PARAMETER :: cauchy_only = max_sc + 1
      INTEGER, PARAMETER :: arc_search_maxit = cauchy_only + 1
      INTEGER, PARAMETER :: cg_maxit = arc_search_maxit + 1
      INTEGER, PARAMETER :: explore_optimal_subspace = cg_maxit + 1
      INTEGER, PARAMETER :: dual_starting_point = explore_optimal_subspace + 1
      INTEGER, PARAMETER :: restore_problem = dual_starting_point + 1
      INTEGER, PARAMETER :: sif_file_device = restore_problem + 1
      INTEGER, PARAMETER :: qplib_file_device = sif_file_device + 1
      INTEGER, PARAMETER :: rho = qplib_file_device + 1
      INTEGER, PARAMETER :: infinity = rho + 1
      INTEGER, PARAMETER :: stop_abs_p = infinity + 1
      INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
      INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
      INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
      INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
      INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
      INTEGER, PARAMETER :: stop_cg_relative = stop_rel_c + 1
      INTEGER, PARAMETER :: stop_cg_absolute = stop_cg_relative + 1
      INTEGER, PARAMETER :: cg_zero_curvature = stop_cg_absolute + 1
      INTEGER, PARAMETER :: max_growth = stop_cg_absolute + 1
      INTEGER, PARAMETER :: identical_bounds_tol = max_growth + 1
      INTEGER, PARAMETER :: cpu_time_limit = identical_bounds_tol + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: initial_perturbation = clock_time_limit + 1
      INTEGER, PARAMETER :: perturbation_reduction = initial_perturbation + 1
      INTEGER, PARAMETER :: final_perturbation = perturbation_reduction + 1
      INTEGER, PARAMETER :: remove_dependencies = final_perturbation + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general                       &
                              = remove_dependencies + 1
      INTEGER, PARAMETER :: subspace_direct = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: subspace_alternate = subspace_direct + 1
      INTEGER, PARAMETER :: exact_arc_search = subspace_alternate + 1
      INTEGER, PARAMETER :: subspace_arc_search = exact_arc_search + 1
      INTEGER, PARAMETER :: space_critical = subspace_arc_search + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: generate_qplib_file = generate_sif_file + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = generate_qplib_file + 1
      INTEGER, PARAMETER :: definite_linear_solver = symmetric_linear_solver + 1
      INTEGER, PARAMETER :: unsymmetric_linear_solver                          &
                              = definite_linear_solver + 1
      INTEGER, PARAMETER :: sif_file_name = unsymmetric_linear_solver + 1
      INTEGER, PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER, PARAMETER :: prefix = qplib_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'DQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( print_gap )%keyword = 'iterations-between-printing'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( max_sc )%keyword = 'maximum-dimension-of-schur-complement'
      spec( cauchy_only )%keyword = 'cauchy-only-until-change-level'
      spec( arc_search_maxit )%keyword                                         &
        = 'maximum-number-of-steps-per-arc-search'
      spec( cg_maxit )%keyword = 'maximum-number-of-cg-iterations-per-iteration'
      spec( explore_optimal_subspace )%keyword = 'explore-optimal-subspace'
      spec( dual_starting_point )%keyword = 'dual-starting-point'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( rho )%keyword = 'penalty-weight'
      spec( infinity )%keyword = 'infinity-value'
      spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
      spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
      spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
      spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
      spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
      spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
      spec( stop_cg_relative )%keyword = 'cg-relative-accuracy-required'
      spec( stop_cg_absolute )%keyword = 'cg-absolute-accuracy-required'
      spec( cg_zero_curvature )%keyword = 'cg-zero-curvature-threshold'
      spec( max_growth )%keyword = 'maximum-growth-before-refactorization'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'
      spec( initial_perturbation )%keyword = 'initial-perturbation'
      spec( perturbation_reduction )%keyword = 'perturbation-reduction-factor'
      spec( final_perturbation )%keyword = 'final-perturbation'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword                             &
        = 'treat-zero-bounds-as-general'
      spec( subspace_direct )%keyword = 'direct-solution-of-subspace-problem'
      spec( subspace_alternate )%keyword                                       &
        = 'alternate-solution-of-subspace-problem'
      spec( exact_arc_search )%keyword = 'perform-exact-arc-search'
      spec( subspace_arc_search )%keyword = 'perform-subspace-arc-search'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( symmetric_linear_solver )%keyword                                  &
        = 'symmetric-linear-equation-solver'
      spec( definite_linear_solver )%keyword = 'definite-linear-equation-solver'
      spec( unsymmetric_linear_solver )%keyword                                &
        = 'unsymmetric-linear-equation-solver'
      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( main_specname ) ) THEN
        CALL SPECFILE_read( device, main_specname, spec, lspec, control%error )
      ELSE IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_sc ),                               &
                                 control%max_sc,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cauchy_only ),                          &
                                 control%cauchy_only,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( arc_search_maxit ),                     &
                                 control%arc_search_maxit,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cg_maxit ),                             &
                                 control%cg_maxit,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( explore_optimal_subspace ),             &
                                 control%explore_optimal_subspace,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dual_starting_point ),                  &
                                 control%dual_starting_point,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( restore_problem ),                      &
                                 control%restore_problem,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_device ),                    &
                                 control%qplib_file_device,                    &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( rho ),                                  &
                                 control%rho,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_p ),                           &
                                 control%stop_abs_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_p ),                           &
                                 control%stop_rel_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_d ),                           &
                                 control%stop_abs_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_d ),                           &
                                 control%stop_rel_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_c ),                           &
                                 control%stop_abs_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_c ),                           &
                                 control%stop_rel_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_relative ),                     &
                                 control%stop_cg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_absolute ),                     &
                                 control%stop_cg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cg_zero_curvature ),                    &
                                 control%cg_zero_curvature,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_growth ),                           &
                                 control%max_growth,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_perturbation ),                 &
                                 control%initial_perturbation,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( perturbation_reduction ),               &
                                 control%perturbation_reduction,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( final_perturbation ),                   &
                                 control%final_perturbation,                   &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( remove_dependencies ),                  &
                                 control%remove_dependencies,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subspace_direct ),                      &
                                 control%subspace_direct,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subspace_alternate ),                   &
                                 control%subspace_alternate,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( exact_arc_search ),                     &
                                 control%exact_arc_search,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subspace_arc_search ),                  &
                                 control%subspace_arc_search,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_qplib_file ),                  &
                                 control%generate_qplib_file,                  &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),              &
                                 control%symmetric_linear_solver,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),              &
                                 control%SBLS_control%symmetric_linear_solver, &
                                 control%error )
     CALL SPECFILE_assign_value( spec( definite_linear_solver ),               &
                                 control%definite_linear_solver,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( definite_linear_solver ),               &
                                 control%SBLS_control%definite_linear_solver,  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( unsymmetric_linear_solver ),            &
                                 control%unsymmetric_linear_solver,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( unsymmetric_linear_solver ),            &
                               control%SBLS_control%unsymmetric_linear_solver, &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_name ),                      &
                                 control%qplib_file_name,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfile for FDC

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FDC_read_specfile( control%FDC_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FDC' )
      ELSE
        CALL FDC_read_specfile( control%FDC_control, device )
      END IF
      control%FDC_control%max_infeas = control%stop_abs_p

!  Read the specfile for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

!  Read the specfile for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

!  Read the specfile for GLTR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL GLTR_read_specfile( control%GLTR_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-GLTR')
      ELSE
        CALL GLTR_read_specfile( control%GLTR_control, device )
      END IF

      RETURN

      END SUBROUTINE DQP_read_specfile

!-*-*-*-*-*-*-*-*-*-   D Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE DQP_solve( prob, data, control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic objective
!
!        q(x) = 1/2 x^T H x + g^T x + f
!
!  or the linear/separable objective
!
!        s(x) = 1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
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
!      q(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  or
!
!      s(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
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
!    to be solved since the last call to DQP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the positive definite Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H. Seven storage formats are permitted:
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
!    v) scaled identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 15) = 'SCALED-IDENTITY'
!       H%val( 1 )  the value assigned to each diagonal of H
!
!    vi) identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = 'IDENTITY'
!
!    vii) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component H_lm below
!       ** N.B. YET TO BE IMPLEMENTED **
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!   ** N.B. YET TO BE IMPLEMENTED **
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %target_kind is an INTEGER variable that defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by %X0 (see below)
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
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
!  data is a structure of type DQP_data_type which holds private internal data
!
!  control is a structure of type DQP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to DQP_initialize. See the preamble
!   for details
!
!  inform is a structure of type DQP_inform_type that provides
!    information on exit from DQP_solve. The component status
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
!  On exit from DQP_solve, other components of inform are given in the preamble
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
!  X_stat is an optional  INTEGER array of length n, which if present will be
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
      TYPE ( DQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DQP_control_type ), INTENT( IN ) :: control
      TYPE ( DQP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  Local variables

      INTEGER :: i, j, l, n_depen, nzc, nv, lbd, dual_starting_point
      REAL ( KIND = wp ) :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: time_analyse, time_factorize
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: clock_analyse, clock_factorize
      REAL ( KIND = wp ) :: av_bnd
!     REAL ( KIND = wp ) :: fixed_sum, xi
      LOGICAL :: composite_g, diagonal_h, identity_h, scaled_identity_h
      LOGICAL :: printi, remap_freed, reset_bnd, stat_required
      LOGICAL :: separable_bqp
      CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering DQP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

! -------------------------------------------------------------------
!  If desired, generate a QPLIB file for problem passed

      IF ( control%generate_qplib_file ) THEN
        CALL RPD_write_qp_problem_data( prob, control%qplib_file_name,         &
                    control%qplib_file_device, inform%rpd_inform )
      END IF

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
        IF ( prob%Hessian_kind == 0 ) THEN
          WRITE( control%out, "( ' W = zeros' )" )
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          WRITE( control%out, "( ' W = ones ' )" )
          IF ( prob%target_kind == 0 ) THEN
            WRITE( control%out, "( ' X0 = zeros ' )" )
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( control%out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" )               &
              prob%X0( : prob%n )
          END IF
        ELSE IF ( prob%Hessian_kind == 2 ) THEN
          WRITE( control%out, "( ' W = ', /, ( 5ES12.4 ) )" )                  &
            prob%WEIGHT( : prob%n )
          IF ( prob%target_kind == 0 ) THEN
            WRITE( control%out, "( ' X0 = zeros ' )" )
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( control%out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" )               &
              prob%X0( : prob%n )
          END IF
        ELSE
          IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
            WRITE( control%out, "( ' H  = I' )" )
          ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
            WRITE( control%out, "( ' H  =', ES12.4, ' * I' )" ) prob%H%val( 1 )
          ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )     &
              prob%H%val( : prob%n )
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )        &
              prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            WRITE( control%out, "( ' H (row-wise) = ' )" )
            DO i = 1, prob%n
              WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                 &
                ( i, prob%H%col( j ), prob%H%val( j ),                         &
                  j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
            END DO
          ELSE
            WRITE( control%out, "( ' H (co-ordinate) = ' )" )
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
            ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ),               &
              i = 1, prob%H%ne)
          END IF
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

! trivial checks that H is positive definite ... more sophisticated checks
! occur later

      IF ( prob%Hessian_kind < 0 ) THEN
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'IDENTITY' )
        CASE ( 'SCALED_IDENTITY' )
          IF ( prob%H%val( 1 ) <= zero ) THEN
            inform%status = GALAHAD_error_inertia ; GO TO 800
          END IF
        CASE ( 'DIAGONAL' )
          IF ( COUNT( prob%H%val( :  prob%n ) <= zero ) > 0 ) THEN
            inform%status = GALAHAD_error_inertia ; GO TO 800
          END IF
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( i == j .AND. prob%H%val( l ) <= zero ) THEN
                inform%status = GALAHAD_error_inertia ; GO TO 800
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
!         DO i = 1, prob%n
!           DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
!             j = prob%H%col( l )
!             IF ( i == j .AND. prob%H%val( l ) <= zero ) THEN
!               inform%status = GALAHAD_error_inertia ; GO TO 800
!             END IF
!           END DO
!         END DO
        CASE ( 'COORDINATE' )
!         DO l = 1, prob%H%ne
!           i = prob%H%row( l ) ; j = prob%H%col( l )
!           IF ( i == j .AND. prob%H%val( l ) <= zero ) THEN
!             inform%status = GALAHAD_error_inertia ; GO TO 800
!           END IF
!         END DO
        END SELECT
      ELSE IF ( prob%Hessian_kind == 0 ) THEN
        inform%status = GALAHAD_error_inertia ; GO TO 800
      ELSE IF ( prob%Hessian_kind >= 2 ) THEN
        IF ( COUNT( prob%WEIGHT( : prob%n ) == zero ) > 0 ) THEN
         inform%status = GALAHAD_error_inertia ; GO TO 800
        END IF
      END IF

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

      IF ( prob%new_problem_structure ) THEN

!  store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( prob%Hessian_kind < 0 ) THEN
          IF ( SMT_get( prob%H%type ) == 'LBFGS' ) THEN
            data%h_ne = 0
            inform%status = GALAHAD_not_yet_implemented
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error,                                            &
             & "( A, ' LBFGS Hessian not yet implemented ' )" ) prefix
            GO TO 800
          ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
            data%h_ne = 0
          ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
            data%h_ne = 1
          ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            data%h_ne = prob%n
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          ELSE
            data%h_ne = prob%H%ne
          END IF
        ELSE
          data%h_ne = 0
        END IF
      END IF

!  if the problem has no general constraints, check to see if it is separable

      IF ( data%a_ne <= 0 ) THEN
        separable_bqp = .TRUE.
        IF ( prob%Hessian_kind < 0 ) THEN
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, prob%n
              DO j = 1, i
                l = l + 1
                IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                  separable_bqp = .FALSE. ; GO TO 10
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, prob%n
              DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                j = prob%H%col( l )
                IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                  separable_bqp = .FALSE. ; GO TO 10
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, prob%H%ne
              i = prob%H%row( l ) ; j = prob%H%col( l )
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                separable_bqp = .FALSE. ; GO TO 10
              END IF
            END DO
          END SELECT
        END IF
 10     CONTINUE

!  the problem is a separable bound-constrained QP. Solve it explicitly

!separable_bqp = .FALSE.
        IF ( separable_bqp ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( /, A, ' Solving separable bound-constrained QP -' )" ) prefix
          CALL QPD_solve_separable_BQP( prob, control%infinity,                &
                                        obj_unbounded,  inform%obj,            &
                                        inform%feasible, inform%status,        &
                                        B_stat = X_stat( : prob%n ) )
          IF ( printi ) THEN
            CALL CLOCK_time( clock_now )
            WRITE( control%out,                                                &
               "( A, ' On exit from QPD_solve_separable_BQP: status = ',       &
            &   I0, ', time = ', F0.2, /, A, ' objective value =', ES12.4 )",  &
              advance = 'no' ) prefix, inform%status, inform%time%clock_total  &
                + clock_now - clock_start, prefix, inform%obj
            IF ( PRESENT( X_stat ) ) THEN
              WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )&
                COUNT( X_stat( : prob%n ) /= 0 ), prob%n
            ELSE
              WRITE( control%out, "( '' )" )
            END IF
          END IF
          inform%iter = 0 ; inform%non_negligible_pivot = zero
          inform%factorization_integer = 0 ; inform%factorization_real = 0

          IF ( printi ) then
            SELECT CASE( inform%status )
              CASE( GALAHAD_error_restrictions  ) ; WRITE( control%out,        &
                "( /, A, '  Warning - input paramters incorrect' )" ) prefix
              CASE( GALAHAD_error_primal_infeasible ) ; WRITE( control%out,    &
                "( /, A, '  Warning - the constraints appear to be',           &
               &   ' inconsistent' )" ) prefix
              CASE( GALAHAD_error_unbounded ) ; WRITE( control%out,            &
                "( /, A, '  Warning - problem appears to be unbounded from',   &
               & ' below' )") prefix
            END SELECT
          END IF
          IF ( inform%status /= GALAHAD_ok ) RETURN
          GO TO 800
        END IF
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
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
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

        IF ( prob%Hessian_kind < 0 ) THEN
          IF ( SMT_get( prob%H%type ) == 'LBFGS' ) THEN
            data%h_ne = 0
            inform%status = GALAHAD_not_yet_implemented
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error,                                            &
             & "( A, ' LBFGS Hessian not yet implemented ' )" ) prefix
            GO TO 800
          ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
            data%h_ne = 0
          ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
            data%h_ne = 1
          ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            data%h_ne = prob%n
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          ELSE
            data%h_ne = prob%H%ne
          END IF
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
            inform%time%preprocess + time_now - time_record
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
          inform%time%find_dependent + time_now - time_record
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
               time_now - time_start > control%cpu_time_limit ) .OR.           &
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
            WRITE( control%out, "( ' ' )" )
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
          CALL DQP_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
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

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ', A,   &
         & ' dependent and will be temporarily removed' )" ) prefix, n_depen,  &
           TRIM( STRING_pleural( n_depen ) ), TRIM( STRING_are( n_depen ) )

!  allocate arrays to indicate which constraints have been freed

          array_name = 'DQP: data%C_freed'
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
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
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

      IF ( prob%Hessian_kind >= 1 ) THEN
        composite_g = prob%target_kind /= 0
      ELSE
        composite_g = prob%gradient_kind == 0 .OR. prob%gradient_kind == 1
      END IF

      IF ( prob%hessian_kind < 0 ) THEN
        diagonal_h = .TRUE. ; identity_h = .TRUE.
        scaled_identity_h = SMT_get( prob%H%type ) == 'SCALED_IDENTITY'
        IF ( prob%Hessian_kind < 0 ) THEN
  H_loop: DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                diagonal_h = .FALSE. ; EXIT H_loop
              END IF
              IF ( i == j .AND. prob%H%val( l ) /= one ) identity_h = .FALSE.
            END DO
          END DO H_loop
        END IF
      ELSE IF ( prob%hessian_kind == 0 ) THEN
        diagonal_h = .TRUE. ; identity_h = .TRUE.
        scaled_identity_h = .FALSE.
      ELSE IF ( prob%hessian_kind == 1 ) THEN
        diagonal_h = .TRUE. ; identity_h = .TRUE.
        scaled_identity_h = .FALSE.
      ELSE
        diagonal_h = .TRUE. ; identity_h = .FALSE.
        scaled_identity_h = .TRUE.
      END IF

      CALL DQP_workspace( prob%m, prob%n, data%dims, prob%A, prob%H,           &
                          composite_g, diagonal_h, identity_h,                 &
                          scaled_identity_h, nv, lbd,                          &
                          data%C_status, data%NZ_p, data%IUSED, data%INDEX_r,  &
                          data%INDEX_w, data%X_status, data%V_status,          &
                          data%X_status_old, data%C_status_old, data%C_active, &
                          data%X_active, data%CHANGES, data%ACTIVE_list,       &
                          data%ACTIVE_status, data%SOL, data%RHS, data%RES,    &
                          data%H_s, data%Y_l, data%Y_u, data%Z_l, data%Z_u,    &
                          data%VECTOR, data%BREAK_points, data%YC_l,           &
                          data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,          &
                          data%GY_u, data%GZ_l, data%GZ_u, data%V0, data%VT,   &
                          data%GV, data%G, data%PV, data%HPV, data%DV,         &
                          data%V_bnd, data%H_sbls, data%A_sbls, data%SCU_mat,  &
                          control, inform )

!  =================
!  Solve the problem
!  =================

      dual_starting_point = control%dual_starting_point

!write(6,*) ' h, t, g ', prob%Hessian_kind, prob%target_kind, prob%gradient_kind
      IF ( prob%Hessian_kind == 1 ) THEN
        IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 C_stat = C_stat, X_stat = X_stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind, G = prob%G,   &
                                 C_stat = C_stat, X_stat = X_stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 X0 = prob%X0, C_stat = C_stat,                &
                                 X_stat = X_stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 X0 = prob%X0, G = prob%G,                     &
                                 C_stat = C_stat, X_stat = X_stat )
          END IF
        END IF
      ELSE IF ( prob%Hessian_kind == 2 ) THEN
        IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT,                         &
                                 C_stat = C_stat, X_stat = X_stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, G = prob%G,             &
                                 C_stat = C_stat, X_stat = X_stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, X0 = prob%X0,           &
                                 C_stat = C_stat, X_stat = X_stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, control, inform,      &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%SOL, data%RHS, data%RES, data%H_s,       &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%VECTOR, data%BREAK_points, data%YC_l,    &
                                 data%YC_u, data%ZC_l, data%ZC_u, data%GY_l,   &
                                 data%GY_u, data%GZ_l, data%GZ_u, data%V0,     &
                                 data%VT, data%GV, data%G, data%PV,            &
                                 data%HPV, data%DV,                            &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, X0 = prob%X0,           &
                                 G = prob%G, C_stat = C_stat, X_stat = X_stat )
          END IF
        END IF
      ELSE IF ( prob%Hessian_kind /= 0 ) THEN
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix, control, inform,        &
                               prob%Hessian_kind, prob%gradient_kind,          &
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
                               H_val = prob%H%val, H_col = prob%H%col,         &
                               H_ptr = prob%H%ptr,                             &
                               C_stat = C_stat, X_stat = X_stat )
        ELSE
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix, control, inform,        &
                               prob%Hessian_kind, prob%gradient_kind,          &
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
                               H_val = prob%H%val, H_col = prob%H%col,         &
                               H_ptr = prob%H%ptr, G = prob%G,                 &
                               C_stat = C_stat, X_stat = X_stat )
        END IF
      END IF

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
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
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
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  compute total time

  800 CONTINUE
      IF ( control%error > 0 .AND. control%print_level >= 1 )                  &
        CALL SYMBOLS_status( inform%status, control%error, prefix, 'DQP' )
      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=',                                            &
    &   /, A, 3X, ' =                          DQP total time            ',    &
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
        WRITE( control%out, "( A, ' leaving DQP_solve ' )" ) prefix
      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -DQP_solve-', /,  A,                           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving DQP_solve ' )" ) prefix
      RETURN

!  non-executable statements

 2010 FORMAT( ' ', /, A, '    ** Error return ', I0, ' from DQP ' )

!  End of DQP_solve

      END SUBROUTINE DQP_solve

!-*-*-*-*-*-   D Q P _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE DQP_solve_main( n, m, dims, A_val, A_col, A_ptr,              &
                                 C_l, C_u, X_l, X_u, X, Y, Z, C, f,            &
                                 prefix, control, inform,                      &
                                 Hessian_kind, gradient_kind, nv, lbd, m_ref,  &
                                 dual_starting_point, clock_total, cpu_total,  &
                                 SBLS_data, SLS_data, SCU_data, GLTR_data,     &
                                 SLS_control, SBLS_control, GLTR_control,      &
                                 C_status, NZ_p, IUSED, INDEX_r, INDEX_w,      &
                                 X_status, V_status, X_status_old,             &
                                 C_status_old, refactor, m_active, n_active,   &
                                 C_active, X_active, CHANGES,                  &
                                 ACTIVE_list, ACTIVE_status, SOL, RHS, RES,    &
                                 H_s, Y_l, Y_u, Z_l, Z_u, VECTOR,              &
                                 BREAK_points, YC_l, YC_u, ZC_l, ZC_u, GY_l,   &
                                 GY_u, GZ_l, GZ_u, V0, VT, GV, GC, PV, HPV,    &
                                 DV, V_bnd, H_sbls, A_sbls, C_sbls, SCU_mat,   &
                                 H_val, H_col, H_ptr, WEIGHT, target_kind,     &
                                 X0, G, C_stat, X_stat, initial )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimizes the quadratic objective function
!
!        q(x) = 1/2 x^T H x + g^T x + f
!
!  or the linear/separable objective function
!
!        s(x) = 1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  subject to the constraints
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!    and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a dual gradient-projection
!  method. The subroutine is particularly appropriate when A is sparse.
!
!  Optionally, minimize instead the penalty function
!
!      q(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  or
!
!      s(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  subject to the bound constraints x_l <= x <= x_u
!
!  In order that many of the internal computations may be performed
!  efficiently, it is required that
!
!  * the variables are ordered so that their bounds appear in the order
!
!    free                      x
!    non-negativity      0  <= x
!    lower              x_l <= x
!    range              x_l <= x <= x_u   (x_l < x_u)
!    upper                     x <= x_u
!    non-positivity            x <=  0
!
!    Fixed variables are not permitted (ie, x_l < x_u for range variables).
!
!  * the constraints are ordered so that their bounds appear in the order
!
!    equality           c_l  = A x
!    lower              c_l <= A x
!    range              c_l <= A x <= c_u
!    upper                     A x <= c_u
!
!    Free constraints are not permitted (ie, at least one of c_l and c_u
!    must be finite). Bounds with the value zero are not treated separately.
!
!  These transformations may be effected, in place, using the module
!  GALAHAD_QPP. The same module may subsequently used to recover the solution.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!  m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!  dims is a structure of type DQP_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. The following components must be set:
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
!    value dims%x_e + dims%nc
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
!  A_* is used to hold the matrix A by rows. In particular:
!      A_col( : )   the column indices of the components of A
!      A_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row.
!      A_val( : )   the values of the components of A
!
!  C_l, C_u are REAL arrays of length m, which must be set by the user to
!   the values of the arrays x_l and x_u of lower and upper bounds on x, ordered
!   as described above (strictly only C_l( dims%c_l_start : dims%c_l_end )
!   and C_u( dims%c_u_start : dims%c_u_end ) need be set, as the other
!   components are ignored!).
!
!  X_l, X_u are REAL arrays of length n, which must be set by the user to
!   the values of the arrays x_l and x_u of lower and upper bounds on x, ordered
!   as described above (strictly only X_l( dims%x_l_start : dims%x_l_end )
!   and X_u( dims%x_u_start : dims%x_u_end ) need be set, as the other
!   components are ignored!).
!
!  X is a REAL array of length n, which must be set by the user on entry to
!   DQP_solve to give an initial estimate of the optimization parameters, x.
!   The i-th component of X should contain the initial estimate of x_i, for
!   i = 1, .... , n.  The estimate need not satisfy the simple bound
!   constraints and may be perturbed by DQP_solve prior to the start of the
!   minimization.  On exit from DQP_solve, X will contain the best estimate of
!   the optimization parameters found
!
!  Y is a REAL array of length m, which must be set by the user on entry to
!   DQP_solve to give an initial estimates of the optimal Lagrange multipiers,
!   y. The i-th component of Y should contain the initial estimate of y_i, for
!   i = 1, .... , m.  The Lagrange multiplier for any constraint with both
!   infinite lower and upper bounds need not be set. On exit from DQP_solve,
!   Y will contain the best estimate of the Lagrange multipliers found
!
!  Z, is a REAL array of length n, which must be set by on entry to DQP_solve
!   to hold the values of the the dual variables associated with the simple
!   bound constraints. The dual variable for any variable with both
!   infinite lower and upper bounds need not be set. On exit from
!   DQP_solve, Z will contain the best estimates obtained
!
!  C is a REAL array of length m, which need not be set on entry. On exit,
!   the i-th component of C will contain (A * x)_i, for i = 1, .... , m.
!
!  control and inform are exactly as for DQP_solve
!
!  Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero. WEIGHT (see below) need not be set
!
!     1  all the weights will be one. WEIGHT (see below) need not be set
!
!    >1  the weights will be those given by WEIGHT (see below)
!
!     any other value - the Hessian will be given by H (see below)
!
!   WEIGHT is an optional REAL array, which need only be included if
!    Hessian_kind is > 1. If this is so, it must be of length at least
!    n, and contain the weights W for the objective function.
!
!   X0 is an optional REAL array, which need only be included if
!    Hessian_kind is not 0. If this is so, it must be of length at least
!    n, and contain the shifts X^0 for the objective function.
!
!   H_* is OPTIONALly used to hold the lower triangle of the matrix H by rows.
!    In particular:
!      H_col( : )   the column indices of the components of H
!      H_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row.
!      H_val( : )   the values of the components of H
!    If H_ptr or H_col is absent, H will be presumed to be a scalar
!    multiple, H_val(1), of the identity. If additionally H_val is absent,
!    H will be presumed to be the identity
!
!   gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term will be zero, and the analytic centre of the
!        feasible region will be found if in addition Hessian_kind is 0.
!        G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        G (see below) need not be set
!
!     any other value - the gradients will be those given by G (see below)
!
!   G is an optional REAL array, which need only be included if
!    gradient_kind is not 0 or 1. If this is so, it must be of length at least
!    n, and contain the gradient term g for the objective function.
!

!   dual_starting_point is an INTEGER variable that specifies which starting
!    point should be used for the dual problem. Possible values are

!      0 user supplied
!      1 minimize linearized dual
!      2 minimize simplified quadratic dual
!      3 all free (= all active primal costraints)
!      4 all fixed on bounds (= no active primal costraints)

!  cpu_total is a REAL variable that gives a running total of the CPU time
!   spent in the package. It should be set on entry to the current total
!   CPU time, and on output will give the updated value

!  clock_total is a REAL variable that gives a running total of the clock time
!   spent in the package. It should be set on entry to the current total
!   clock time, and on output will give the updated value

!  The remaining arguments are used as internal workspace, and need not be
!  set on entry. These (and their minimum lengths where appropriate) are
!
!  INTEGER C_status(m)
!  INTEGER NZ_p(nv)
!  INTEGER IUSED(n)
!  INTEGER INDEX_r(n)
!  INTEGER INDEX_w(n)
!  INTEGER X_status(n)
!  INTEGER V_status(nv)
!  INTEGER X_status_old(n)
!  INTEGER C_status_old(m)
!  INTEGER X_active(len_n_active)
!  INTEGER C_active(len_m_active)
!  REAL SOL(n+len_m_active+len_n_active+control%max_sc)
!  REAL RHS(n+len_m_active+len_n_active+control%max_sc)
!  REAL RES(n+len_m_active+len_n_active+control%max_sc)
!  REAL H_s(n)
!  REAL Y_l(1:dims%c_l_end)
!  REAL Y_u(dims%c_u_start:dims%c_u_end)
!  REAL Z_l(dims%x_free+1:dims%x_l_end)
!  REAL Z_u(dims%x_u_start:n)
!  REAL BREAK_points(nv)
!  REAL YC_l(1:dims%c_l_end)
!  REAL YC_u(dims%c_u_start:dims%c_u_end)
!  REAL ZC_l(dims%x_free + 1:dims%x_l_end)
!  REAL ZC_u(dims%x_u_start:n)
!  REAL GY_l(1:dims%c_l_end)
!  REAL GY_u(dims%c_u_start:dims%c_u_end)
!  REAL GZ_l(dims%x_free+1:dims%x_l_end)
!  REAL GZ_u(dims%x_u_start:n)
!  REAL V0(nv)
!  REAL VT(MAX(n,nv))
!  REAL GV(nv)
!  REAL PV(nv)
!  REAL DV(nv)
!  REAL V_bnd(nv,2)
!  when composite_g:
!   REAL GC(n)
!  when diagonal_h:
!    when scaled_identity_h:
!      REAL H_sbls%val(1)
!    else:
!      REAL H_sbls%val(n)
!  else:
!    INTEGER H_sbls%ptr(n + 1)
!    INTEGER H_sbls%col(h_ne)
!    REAL H_sbls%val(h_ne)
!  when control%max_sc > 0:
!    INTEGER CHANGES(m+n)
!    INTEGER ACTIVE_list(m+n+control%max_sc)
!    INTEGER ACTIVE_status(m+n)
!    REAL VECTOR(m+2*n)
!    INTEGER SCU_mat%BD_col_start(control%max_sc+1)
!    INTEGER SCU_mat%BD_row(lbd)
!    REAL SCU_mat%BD_val(lbd)
!  INTEGER A_sbls%row(max_ne_active)
!  INTEGER A_sbls%col(max_ne_active)
!  REAL A_sbls%val(max_ne_active)
!  INTEGER C_sbls%row(0)
!  INTEGER C_sbls%col(0)
!  REAL C_sbls%val(0)
!
!  as well as internally defined derived types
!
!  SLS_data of type SLS_data_type
!  SBLS_data of type SBLS_data_type
!  GLTR_data of type GLTR_data_type
!  SCU_data of type SCU_data_type
!
!  where
!
!  h_ne = H%ptr( n + 1 ) - 1
!  max_ne_active = 2 * ( A%ptr( m + 1 ) + n - 1 )
!  len_m_active = dims%c_u_start - 1 +  m - dims%c_l_end
!                   + 2 * ( dims%c_l_end - dims%c_u_start + 1 )
!  len_n_active = dims%x_u_start - dims%x_free - 1 + n - dims%x_l_end
!                   + 2 * ( dims%x_l_end - dims%x_u_start + 1 )
!  nv = dims%c_l_end + ( dims%c_u_end - dims%c_u_start + 1 )
!         + ( dims%x_l_end - dims%x_free ) + ( prob%n - dims%x_u_start + 1 )
!  lbd = # entries in the largest control%max_sc rows of A
!  composite_g is true if and only if either
!    Hessian_kind >= 1 and target_kind /= 0 or
!    Hessian_kind <= 0 and prob%gradient_kind = 0 or 1
!  diagonal_h is true if and only if the Hessian is diagonal
!  scaled_identity_h is true if and only if the Hessian is a scaled identity
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( DQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, Hessian_kind, gradient_kind
      INTEGER, INTENT( OUT ) :: m_active, n_active
      INTEGER, INTENT( IN ), OPTIONAL :: target_kind
      REAL ( KIND = wp ), INTENT( IN ) :: f
      LOGICAL, INTENT( IN ), OPTIONAL :: initial
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL  :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( : ), OPTIONAL  :: H_col
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_stat
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( : ), OPTIONAL  :: H_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: WEIGHT, X0
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( DQP_control_type ), INTENT( IN ) :: control
      TYPE ( DQP_inform_type ), INTENT( INOUT ) :: inform

      LOGICAL, INTENT( INOUT ) :: refactor
      INTEGER, INTENT( IN ) :: dual_starting_point
      INTEGER, INTENT( INOUT ) :: nv, lbd, m_ref
      REAL ( KIND = wp ), INTENT( INOUT ) :: cpu_total, clock_total
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_status, C_status_old
      INTEGER, INTENT( INOUT ), DIMENSION( nv ) :: NZ_p, V_status
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IUSED, INDEX_r, INDEX_w
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X_status, X_status_old
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: X_active
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: C_active
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: CHANGES
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: ACTIVE_list
      INTEGER, INTENT( INOUT ), DIMENSION( * ) :: ACTIVE_status
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: RES
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: RHS
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: H_s
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( 1 : dims%c_l_end ) :: Y_l, YC_l, GY_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, YC_u, GY_u
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%x_free + 1:dims%x_l_end ) :: Z_l, ZC_l, GZ_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
               DIMENSION( dims%x_u_start : n ) :: Z_u, ZC_u, GZ_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: VECTOR
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: BREAK_points
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: V0
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: VT
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: GV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: PV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: DV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: HPV
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv , 2 ) :: V_bnd
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: GC
      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_sbls
      TYPE ( SMT_type ), INTENT( INOUT ) :: C_sbls
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: SBLS_data
      TYPE ( GLTR_data_type ), INTENT( INOUT ) :: GLTR_data
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: SBLS_control
      TYPE ( GLTR_control_type ), INTENT( INOUT ) :: GLTR_control

!  Local variables

      INTEGER :: a_ne, h_ne, i, ii, im, j, l, m_sbls, m_subspace
      INTEGER :: out, error, start_print, stop_print, print_level, ip, mpn
      INTEGER :: yl_start, yl_end, yu_start, yu_end, change, change_subspace
      INTEGER :: zl_start, zl_end, zu_start, zu_end, ce_start, ce_end
      INTEGER :: start_ce, start_yl, start_yu, start_zl, start_zu
      INTEGER :: arc_search_iter, l_start, u_start, print_gap
      INTEGER :: max_row_length, added, deleted, len_list, no_change
      REAL ( KIND = wp ) :: time_record, time_start, time_now
      REAL ( KIND = wp ) :: clock_record, clock_start, clock_now, sl, slope
      REAL ( KIND = wp ) :: a_max, h_max, xi, curv, alpha, dual_g_norm, dual_f
      REAL ( KIND = wp ) :: stop_d, step_max, feas_tol, q0, qt, qc, val
      REAL ( KIND = wp ) :: norm_pv, alpha_subspace, sigma, c_solve, t_solve
      REAL ( KIND = wp ) :: f_all, root_hd, growth, rho, primal_infeasibility
      REAL ( KIND = wp ) :: stop_reasonable, h_scale( 1 )
!     REAL ( KIND = wp ) :: stop_p, stop_c
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: printt, printi, printe, printd, printw, set_printp, printp
      LOGICAL :: stat_required, dolid, diagonal_h, composite_g
      LOGICAL :: identity_h, scaled_identity_h, fresh_start, first_iteration
      LOGICAL :: subspace_direct, subspace_direct_save, penalty_objective
      CHARACTER ( LEN = 1 ) :: skip
      CHARACTER ( LEN = 80 ) :: array_name

!  debug variables

!      INTEGER :: k, n_free, nvar_l, nvar_u, nnonnz, jumpto
!      INTEGER, ALLOCATABLE, DIMENSION( : ) :: INONNZ
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: QV
!      TYPE ( DQP_CAUCHY_data_type ) :: CAUCHY_data

!     INTEGER :: sif = 50
!     LOGICAL :: generate_sif = .TRUE.
!     LOGICAL :: generate_sif = .FALSE.

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering DQP_solve_main ' )" ) prefix

!  there must be a Hessian

      IF ( Hessian_kind == 0 ) THEN
        GO TO 920
      ELSE IF ( Hessian_kind > 1 ) THEN
        IF ( .NOT. PRESENT( WEIGHT ) ) GO TO 920
        IF ( COUNT( WEIGHT( : n ) == zero ) > 0 )  GO TO 920
      END IF

!  detemine what kind of Hessian is to be stored

      identity_h = Hessian_kind == 1 .OR. ( Hessian_kind < 0 .AND.             &
          .NOT. PRESENT( H_val ) )
!     scaled_identity_h = Hessian_kind > 1 .OR. ( Hessian_kind < 0 .AND.       &
!         ( .NOT. PRESENT( H_col ) .OR. .NOT. PRESENT( H_ptr ) ) )

      scaled_identity_h = .FALSE.
      IF ( Hessian_kind > 1 ) THEN
        IF ( COUNT( WEIGHT( : n ) /= WEIGHT( 1 ) ) == 0 ) THEN
          scaled_identity_h = .TRUE.
          h_scale( 1 ) = WEIGHT( 1 ) ** 2
        END IF
      ELSE IF ( Hessian_kind < 0 ) THEN
        IF ( .NOT. PRESENT( H_col ) .OR. .NOT. PRESENT( H_ptr ) ) THEN
          scaled_identity_h = .TRUE.
          h_scale( 1 ) = H_val( 1 )
        END IF
      END IF

!  the Hessian must be positive definite

     IF ( .NOT. identity_h ) THEN
       IF ( scaled_identity_h .AND. Hessian_kind < 0 ) THEN
         IF ( H_val( 1 ) <= zero ) GO TO 920
       END IF
     END IF

!  see how to start

      IF ( PRESENT( initial ) ) THEN
        fresh_start = initial
      ELSE
        fresh_start = .TRUE.
      END IF

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

!      IF ( generate_sif ) THEN
!        CALL QPD_SIF( prob, "DQP_out", sif, control%infinity, .TRUE. )
!      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  ===========================
!  Control the output printing
!  ===========================

      IF ( fresh_start ) inform%iter = 0
      first_iteration = .TRUE.

      IF ( control%print_gap < 2 ) THEN
        print_gap = 1
      ELSE
        print_gap = control%print_gap
      END IF

      print_level = 0
      IF ( control%start_print < 0 ) THEN
        start_print = 0
      ELSE
        start_print = control%start_print
      END IF

      IF ( control%stop_print < 0 ) THEN
        stop_print = control%maxit + 1
      ELSE
        stop_print = control%stop_print
      END IF

      error = control%error ; out = control%out

      set_printe = error > 0 .AND. control%print_level >= 1

!  basic single line of output per iteration

      set_printi = out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional timings for various operations

      set_printt = out > 0 .AND. control%print_level >= 2

!  as per printt but also with an indication of where in the code we are

      set_printp = out > 0 .AND. control%print_level >= 3

!  as per printp but also with details of innner iterations

      set_printw = out > 0 .AND. control%print_level >= 4

!  full debugging printing with significant arrays printed

      set_printd = out > 0 .AND. control%print_level >= 5

!  start setting control parameters

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printp = set_printp ;
        printw = set_printw ; printd = set_printd
        print_level = control%print_level
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printp = .FALSE. ;
        printw = .FALSE. ; printd = .FALSE.
        print_level = 0
      END IF

      SBLS_control = control%SBLS_control
      SBLS_control%symmetric_linear_solver =                                   &
        control%symmetric_linear_solver
      SBLS_control%definite_linear_solver =                                    &
        control%definite_linear_solver
      SBLS_control%unsymmetric_linear_solver =                                 &
        control%unsymmetric_linear_solver
      IF ( SBLS_control%factorization < 0 .OR.                                 &
           SBLS_control%factorization > 3 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A,' factor = ', I0, ' out of range [0,3]. Reset to 0' )" )        &
          prefix, SBLS_control%factorization
        SBLS_control%factorization = 0
      END IF

!  if there are no variables, exit

      IF ( n == 0 ) THEN
        i = COUNT( ABS( C_l( : dims%c_equality ) ) > control%stop_abs_p ) +    &
            COUNT( C_l( dims%c_l_start : dims%c_l_end ) > control%stop_abs_p)+ &
            COUNT( C_u( dims%c_u_start : dims%c_u_end ) < - control%stop_abs_p )
        inform%dual_infeasibility = zero
        inform%complementary_slackness = zero
        IF ( i == 0 ) THEN
          inform%primal_infeasibility = zero
          inform%status = GALAHAD_ok
        ELSE
          inform%primal_infeasibility = MAX(                                   &
            MAXVAL( ABS( C_l( : dims%c_equality ) ) ),                         &
            MAXVAL( MAX( C_l( dims%c_l_start : dims%c_l_end ), zero ) ),       &
            MAXVAL( MAX( - C_u( dims%c_u_start : dims%c_u_end ), zero ) ) )
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 810
        END IF
        C = zero ; Y = zero
        f_all = f ; dual_g_norm = zero
        GO TO 600
      END IF

!  print input matrix for debugging

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) n, m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) f
        IF ( gradient_kind == 0 ) THEN
          WRITE( control%out, "( ' G = zeros' )" )
        ELSE IF ( gradient_kind == 1 ) THEN
          WRITE( control%out, "( ' G = ones' )" )
        ELSE
          WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) G( : n )
        END IF
        IF ( Hessian_kind == 1 ) THEN
          WRITE( control%out, "( ' W = ones ' )" )
          IF ( target_kind == 0 ) THEN
            WRITE( control%out, "( ' X0 = zeros ' )" )
          ELSE IF ( target_kind == 1 ) THEN
            WRITE( control%out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" ) X0( : n )
          END IF
        ELSE IF ( Hessian_kind == 2 ) THEN
          WRITE( control%out, "( ' W = ', /, ( 5ES12.4 ) )" ) WEIGHT( : n )
          IF ( target_kind == 0 ) THEN
            WRITE( control%out, "( ' X0 = zeros ' )" )
          ELSE IF ( target_kind == 1 ) THEN
            WRITE( control%out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" ) X0( : n )
          END IF
        ELSE IF ( Hessian_kind /= 0 ) THEN
          IF ( identity_h ) THEN
            WRITE( control%out, "( ' H  = I ' )" )
          ELSE IF ( scaled_identity_h ) THEN
            WRITE( control%out, "( ' H  =', ES12.4, ' * I ' )" ) h_scale( 1 )
          ELSE
            WRITE( control%out, "( ' H (row-wise) = ' )" )
            DO i = 1, n
              WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                 &
               ( i, H_col( j ), H_val( j ), j = H_ptr( i ), H_ptr( i + 1 ) - 1 )
            END DO
          END IF
        END IF
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" ) X_l( : n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" ) X_u( : n )
        IF ( m > 0 ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, A_col( j ), A_val( j ), j = A_ptr( i ), A_ptr( i + 1 ) - 1 )
          END DO
          WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" ) C_l( : m )
          WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" ) C_u( : m )
        END IF
      END IF

!  record array sizes, and see if H may be stored more efficiently

      a_ne = A_ptr( m + 1 ) - 1
      diagonal_h = .TRUE.
      IF ( Hessian_kind < 0 .AND. .NOT. identity_h ) THEN
        identity_h = .TRUE.
        IF ( scaled_identity_h ) THEN
          IF ( h_scale( 1 ) == one ) THEN
            h_ne = 0
          ELSE
            h_ne = 1
            identity_h = .FALSE.
          END IF
        ELSE
!         identity_h = .FALSE.
          h_ne = H_ptr( n + 1 ) - 1
 H_loop : DO i = 1, n
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              IF ( i /= j .AND. H_val( l ) /= zero ) THEN
                diagonal_h = .FALSE. ; EXIT H_loop
              END IF
              IF ( i == j .AND. H_val( l ) /= one ) identity_h = .FALSE.
            END DO
          END DO H_loop
        END IF
      ELSE
        h_ne = 0
      END IF
      IF ( identity_h ) scaled_identity_h = .FALSE.

!  find the largest components of A and H

      IF ( a_ne > 0 ) THEN
        a_max = MAXVAL( ABS( A_val( : a_ne ) ) )
      ELSE
        a_max = zero
      END IF

      IF ( hessian_kind < 0 ) THEN
        IF ( identity_h ) THEN
          h_max = one
        ELSE
          h_max = MAXVAL( ABS( H_val( : h_ne ) ) )
        END IF
      ELSE IF ( Hessian_kind == 1 ) THEN
        h_max = one
      ELSE
        h_max = MAXVAL( WEIGHT( : n ) ** 2 )
      END IF

      IF ( printi ) WRITE( out, "( /, A, '  maximum element of A =', ES11.4,   &
    &                              /, A, '  maximum element of H =', ES11.4 )")&
        prefix, a_max, prefix, h_max

!  are the general constraints to be handled using a penalty function?

      penalty_objective = control%rho > zero
      IF ( penalty_objective ) THEN
        rho = control%rho
        IF ( printi ) WRITE( out, "( /, A,                                     &
       &  '  general constraints penalized by ', ES8.2 )" ) prefix, rho
      ELSE
        rho = zero
      END IF

!  decide whether the gradient is to be treated as composite, and if so, set it

      IF (  Hessian_kind >= 1 ) THEN
        composite_g = target_kind /= 0
      ELSE
        composite_g = gradient_kind == 0 .OR. gradient_kind == 1
      END IF

      IF ( composite_g ) THEN
        array_name = 'dqp: GC'
        IF ( gradient_kind == 0 ) THEN
          GC( : n ) = zero
        ELSE IF ( gradient_kind == 1 ) THEN
          GC( : n ) = one
        ELSE
          GC( : n ) = G( : n )
        END IF
        IF ( Hessian_kind == 1 ) THEN
          IF ( target_kind == 1 ) THEN
            GC( : n ) = GC( : n ) - one
          ELSE IF ( target_kind /= 0 ) THEN
            GC( : n ) = GC( : n ) - X0( : n )
          END IF
        ELSE IF ( Hessian_kind > 1 ) THEN
          IF ( target_kind == 1 ) THEN
            GC( : n ) = GC( : n ) - WEIGHT( : n ) ** 2
          ELSE IF ( target_kind /= 0 ) THEN
            GC( : n ) = GC( : n ) - X0( : n ) * WEIGHT( : n ) ** 2
          END IF
        END IF
      END IF

!  compute the objective gradient and constant term

      f_all = f
      IF ( Hessian_kind == 1 ) THEN
        IF ( target_kind == 1 ) THEN
          f_all = f + half * REAL( n, wp )
        ELSE  IF ( target_kind /= 0 ) THEN
          f_all = f + half * SUM( X0( : n ) ** 2 )
        END IF
      ELSE IF ( Hessian_kind == 2 ) THEN
        IF ( target_kind == 1 ) THEN
          f_all = f + half * SUM( WEIGHT( : n ) ** 2 )
        ELSE  IF ( target_kind /= 0 ) THEN
          f_all = f + half * SUM( ( WEIGHT( : n ) * X0( : n ) ) ** 2 )
        END IF
      END IF

!  compute the initial objective value

      IF ( Hessian_kind == 1 ) THEN
        inform%obj = f_all + half * SUM( X ** 2 )
      ELSE IF ( Hessian_kind == 2 ) THEN
        inform%obj = f_all + half * SUM( ( WEIGHT * X ) ** 2 )
      ELSE
        IF ( identity_h ) THEN
          curv = DOT_PRODUCT( X( : n ), X( : n ) )
        ELSE IF ( scaled_identity_h ) THEN
          curv = h_scale( 1 ) * DOT_PRODUCT( X( : n ), X( : n ) )
        ELSE
          curv = zero
          DO i = 1, n
            xi = X( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              IF ( i == j ) THEN
                curv = curv + xi * xi * H_val( l )
              ELSE
                curv = curv + two * xi * X( j ) * H_val( l )
              END IF
            END DO
          END DO
          IF ( printd )                                                        &
            WRITE( out, "( A, A6, /, ( 4( 2I5, ES10.2 ) ) )" ) prefix,         &
           &  ' h ', ( ( i, H_col( l ), H_val( l ), l = H_ptr( i ),            &
              H_ptr( i + 1 ) - 1 ), i = 1, n )
        END IF
        inform%obj = f_all + half * curv
      END IF

      IF ( composite_g ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( GC( : n ), X )
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
      END IF

      mpn = m + n

!   (dual) variables v will be stored as follows:

!   -----------------------------------------------------------
!   | c_equality  | c_lower  | c_upper  | x_lower  | x_upper  |
!   -----------------------------------------------------------
!    |           | |        | |        | |        | |        |
!    ce_start    | yl_start | yu_start | zl_start | zu_start |
!                ce_end     yl_end     yu_end     zl_end     zu_end = nv

      ce_start = 1
      ce_end = dims%c_equality
      yl_start = dims%c_l_start
      yl_end = dims%c_l_end
      yu_start = yl_end + 1
      yu_end = yl_end + dims%c_u_end - dims%c_u_start + 1
      zl_start = yu_end + 1
      zl_end = yu_end + dims%x_l_end - dims%x_free
      zu_start = zl_end + 1
      zu_end = zl_end + n - dims%x_u_start + 1

      IF ( nv < zu_end ) THEN
        inform%status = GALAHAD_error_integer_ws ; GO TO 900
      END IF
      nv = zu_end

      start_ce = 0
      start_yl = 0
      start_yu = dims%c_u_start - yu_start
      start_zl = dims%x_free + 1 - zl_start
      start_zu = dims%x_u_start - zu_start

      IF ( printt )                                                            &
        WRITE( out, "( ' components of V: ', I0, ':', I0, ' (equality) ',      &
       &           /, 1X, I0, ':', I0, ' (c_l) ', I0, ':', I0, ' (c_u) ',      &
       &              1X, I0, ':', I0, ' (x_l) ', I0, ':', I0, ' (x_u) ' )" )  &
         1, dims%c_equality, dims%c_l_start, yl_end, yu_start, yu_end,         &
         zl_start, zl_end, zu_start, zu_end

!  discover how many entries there are in the largest control%max_sc rows
!  of A if SCU solves may be required

      IF ( control%max_sc > 0 ) THEN
        X_status = 0 ; X_status( 1 ) = n
        max_row_length = 0
        DO i = 1, m
          j =  A_ptr( i + 1 ) -  A_ptr( i )
          IF ( j > 0 ) THEN
            X_status( j ) = X_status( j ) + 1
            max_row_length = MAX( max_row_length, j )
          END IF
        END DO
        l = control%max_sc
        ii = 0
        DO i = max_row_length, 1, - 1
          j = MIN( X_status( i ), l )
          ii = ii + j * i
          l = l - j
          IF ( l == 0 ) EXIT
        END DO

        IF ( lbd < ii ) THEN
          inform%status = GALAHAD_error_integer_ws ; GO TO 900
        END IF
        lbd = ii
      END IF

!  set control parameters

      stat_required = PRESENT( C_stat ) .AND. PRESENT( X_stat )
      IF ( stat_required ) THEN
        X_stat  = 0
        C_stat( : dims%c_equality ) = - 1
        C_stat( dims%c_equality + 1 : ) = 0
      END IF

      IF ( diagonal_h ) THEN

!  if the Hessain is a scaled identity matrix, record it

        IF ( identity_h ) THEN
          root_hd = one

          array_name = 'dqp: H_sbls%type'
          CALL SMT_put( H_sbls%type, 'IDENTITY', inform%alloc_status )
          IF ( inform%status /= 0 ) GO TO 910

!  if the Hessain is a scaled identity matrix, record it

        ELSE IF ( scaled_identity_h ) THEN
          root_hd = SQRT( h_scale( 1 ) )

          array_name = 'dqp: H_sbls%type'
          CALL SMT_put( H_sbls%type, 'SCALED_IDENTITY', inform%alloc_status )
          IF ( inform%status /= 0 ) GO TO 910
          H_sbls%val( 1 ) = h_scale( 1 )

!  if the Hessain is diagonal, record it

        ELSE
          H_sbls%n = n ; H_sbls%m = n ; H_sbls%ne = n
          array_name = 'dqp: H_sbls%type'
          CALL SMT_put( H_sbls%type, 'DIAGONAL', inform%alloc_status )
          IF ( inform%status /= 0 ) GO TO 910

          IF ( Hessian_kind < 0 ) THEN
            H_sbls%val( : n ) = zero
            DO i = 1, n
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                IF ( i == H_col( l ) )                                         &
                  H_sbls%val( i ) = H_sbls%val( i ) + H_val( l )
              END DO
            END DO
          ELSE IF ( Hessian_kind == 1 ) THEN
!           H_sbls%val( : n ) = one
          ELSE
            H_sbls%val( : n ) = WEIGHT( : n ) ** 2
          END IF

!  check that the diagonal Hessian is positive definite

          IF ( Hessian_kind /= 1 ) THEN
            IF ( COUNT( H_sbls%val( : n ) <= zero ) > 0 ) THEN
              inform%status = GALAHAD_error_inertia ; GO TO 900
            END IF
          END IF
        END IF
        IF ( printt ) WRITE( out, "( A, ' Hessian is diagonal' )" ) prefix

!  if the Hessain is not diagonal, record and process it

      ELSE
        H_sbls%n = n ; H_sbls%m = n
        H_sbls%ne = H_ptr( n + 1 ) - 1
        array_name = 'dqp: H_sbls%type'
        CALL SMT_put( H_sbls%type, 'SPARSE_BY_ROWS', inform%alloc_status )
        IF ( inform%status /= 0 ) GO TO 910

!  store H in H_sbls

        H_sbls%ptr( : n + 1 ) = H_ptr( : n + 1 )
        H_sbls%col( : H_sbls%ne ) = H_col( : h_ne )
        H_sbls%val( : H_sbls%ne ) = H_val( : h_ne )

!do i = 1, n
!  do l = H_sbls%ptr( i ), H_sbls%ptr( i + 1 ) - 1
!    write(6,"( ' H: i, j, val ', 2I7, ES12.4 )" ) &
!      i, H_sbls%col( l ), H_sbls%val( l )
!  end do
!end do

!  -----------
!  factorize H
!  -----------

!  order the rows/columns of H prior to factorization

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_initialize_solver( control%definite_linear_solver,            &
                                    SLS_data, inform%SLS_inform )
        SLS_control = control%SLS_control
        SLS_control%pivot_control = 2
        CALL SLS_analyse( H_sbls, SLS_data, SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse =                                            &
          inform%time%clock_analyse + clock_now - clock_record

        IF ( printt ) WRITE( out,                                              &
             "( A, ' H nnz(matrix,predicted factors) = ', I0, ', ', I0,        &
          &  /, A, ' SLS: analysis (solver ', A, ') of H complete:',           &
          & ' status = ', I0 )" )                                              &
               prefix, h_ne, inform%SLS_inform%real_size_factors, prefix,      &
             TRIM( control%definite_linear_solver ), inform%SLS_inform%status
        IF ( printt .AND. inform%SLS_inform%out_of_range > 0 ) WRITE( out,     &
            "( A, ' ** warning: ', I0, ' entry', A, ' of H out of range' )" )  &
               prefix, inform%SLS_inform%out_of_range,                         &
               STRING_ies( inform%SLS_inform%out_of_range )
        IF ( inform%SLS_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_analysis ; GO TO 900
        END IF

!  obtain the factors of H

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( H_sbls, SLS_data, SLS_control, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%nfacts = inform%nfacts + 1
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record

        IF ( printt ) WRITE( out,                                              &
             "( A, ' H nnz(matrix,factors) = ', I0, ', ', I0,                  &
          &  /, A, ' SLS: factorization (solver ', A, ') of H complete:',      &
          & ' status = ', I0 )" ) prefix, h_ne,                                &
             inform%SLS_inform%entries_in_factors, prefix,                     &
             TRIM( control%definite_linear_solver ), inform%SLS_inform%status
        IF ( printt ) WRITE( out, "( A, ' H%n = ', I0, ', rank = ', I0,        &
     &                 ', #-ve eigenvalues = ', I0 )" ) prefix, H_sbls%n,      &
          inform%SLS_inform%rank, inform%SLS_inform%negative_eigenvalues

        IF ( inform%SLS_inform%status == GALAHAD_error_inertia ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
        ELSE IF ( inform%SLS_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_factorization ; GO TO 900
        END IF
      END IF

!  set up space for A and C

      A_sbls%n = n
      array_name = 'dqp: A_sbls%type'
      CALL SMT_put( A_sbls%type, 'COORDINATE', inform%alloc_status )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'dqp: C_sbls%type'
      CALL SMT_put( C_sbls%type, 'ZERO', inform%alloc_status )
      IF ( inform%status /= 0 ) GO TO 910

!  record the dual bounds

      IF ( penalty_objective ) THEN
        V_bnd( ce_start : ce_end, 1 ) = - rho
        V_bnd( yl_start : yl_end, 1 ) = zero
        V_bnd( yu_start : yu_end, 1 ) = - rho

        V_bnd( ce_start : ce_end, 2 ) = rho
        V_bnd( yl_start : yl_end, 2 ) = rho
        V_bnd( yu_start : yu_end, 2 ) = zero
      ELSE
        V_bnd( ce_start : ce_end, 1 ) = - ten * control%infinity
        V_bnd( yl_start : yl_end, 1 ) = zero
        V_bnd( yu_start : yu_end, 1 ) = - ten * control%infinity

        V_bnd( ce_start : ce_end, 2 ) = ten * control%infinity
        V_bnd( yl_start : yl_end, 2 ) = ten * control%infinity
        V_bnd( yu_start : yu_end, 2 ) = zero
      END IF

      V_bnd( zl_start : zl_end, 1 ) = zero
      V_bnd( zu_start : zu_end, 1 ) = - ten * control%infinity

      V_bnd( zl_start : zl_end, 2 ) = ten * control%infinity
      V_bnd( zu_start : zu_end, 2 ) = zero

!  --------------------------------
!  assign the primal starting point
!  --------------------------------

!  the variable is a non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        X( i ) = MAX( X( i ), zero )
      END DO

!  the variable has just a lower bound

      DO i = dims%x_l_start, dims%x_u_start - 1
        X( i ) = MAX( X( i ), X_l( i ) )
      END DO

!  the variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end

!  check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN
          inform%status = GALAHAD_error_bad_bounds ; GO TO 700
        END IF
        X( i ) = MIN( MAX( X( i ), X_l( i ) ), X_u( i ) )
      END DO

!  the variable has just an upper bound

      DO i = dims%x_l_end + 1, dims%x_u_end
        X( i ) = MIN( X( i ), X_u( i ) )
      END DO

!  the variable is a non-positivity

      DO i = dims%x_u_end + 1, n
        X( i ) = MIN( X( i ), zero )
      END DO

!  compute the value of the constraints, and their residuals

      IF ( m > 0 ) THEN
        C( : dims%c_equality ) = - C_l( : dims%c_equality )
        C( dims%c_l_start : dims%c_u_end ) = zero
        CALL DQP_AX( m, C, m, a_ne, A_val, A_col, A_ptr, n, X, '+ ' )

!  the constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1

!  compute an appropriate initial value for the slack variable

          C( i ) = MAX( C( i ), C_l( i ) )
        END DO

!  the constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end

!  check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

          IF ( C_u( i ) - C_l( i ) <= epsmch ) THEN
            inform%status = GALAHAD_error_bad_bounds ; GO TO 700
          END IF
          C( i ) = MIN( MAX( C( i ), C_l( i ) ), C_u( i ) )
        END DO

!  the constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          C( i ) = MIN( C( i ), C_u( i ) )
        END DO
      END IF

!  ------------------------------
!  assign the dual starting point
!  ------------------------------

      C_status = 0 ; X_status = 0
      stop_d = control%stop_abs_d

!  .....................................................
!  dual approximation starting point (see notation below):
!    minimize v^T g^d + 1/2 weight || v ||^2 s.t. v in D
!  .....................................................

      IF ( dual_starting_point == 1 .OR. dual_starting_point == 2 ) THEN

!  compute r = g and store in sol

        IF ( composite_g ) THEN
          SOL( : n ) = GC( : n )
        ELSE
          SOL( : n ) = G( : n )
        END IF

!  find the primal variables by solving H x = r

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( identity_h ) THEN
!         SOL( : n ) = SOL( : n )
        ELSE IF ( scaled_identity_h ) THEN
          SOL( : n ) = SOL( : n ) / h_scale( 1 )
        ELSE IF ( diagonal_h ) THEN
          SOL( : n ) = SOL( : n ) / H_sbls%val( : n )
        ELSE
          CALL SLS_solve( H_sbls, SOL, SLS_data, SLS_control, inform%SLS_inform)
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_solve + clock_now - clock_record

!  compute g^d = - J x - b

        DO i = 1, dims%c_l_end
          GY_l( i ) = - C_l( i )
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            GY_l( i )= GY_l( i ) - A_val( l ) * SOL( A_col( l ) )
          END DO
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          GY_u( i ) = - C_u( i )
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            GY_u( i ) = GY_u( i ) - A_val( l ) * SOL( A_col( l ) )
          END DO
        END DO
        DO j = dims%x_free + 1, dims%x_l_end
          GZ_l( j ) = - SOL( j ) - X_l( j )
        END DO
        DO j = dims%x_u_start, n
          GZ_u( j ) = - SOL( j ) - X_u( j )
        END DO

!  weight is so that max dual variable has value +/-1

        IF ( dual_starting_point == 1 ) THEN
          sigma =                                                              &
            MAX( MAXVAL( ABS( GY_l( : dims%c_l_end ) ) ),                      &
                 MAXVAL( ABS( GY_u( dims%c_u_start : dims%c_u_end ) ) ),       &
                 MAXVAL( ABS( GZ_l( dims%x_free + 1 : dims%x_l_end ) ) ),      &
                 MAXVAL( ABS( GZ_u( dims%x_u_start : n ) ) ) )

!  weight is a sample r J H^{-1} J^T r for some unit r.
!  Compute r = J^T e / ||e|| and store in sol

        ELSE
          SOL( : n ) = zero
          DO i = 1, dims%c_u_start - 1
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + ABS( A_val( l ) )
            END DO
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + ABS( A_val( l ) ) * two
            END DO
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + ABS( A_val( l ) )
            END DO
          END DO
          DO j = dims%x_free + 1, dims%x_u_start - 1
            SOL( j ) = SOL( j ) + one
          END DO
          DO j = dims%x_u_start, dims%x_l_end
            SOL( j ) = SOL( j ) + two
          END DO
          DO j = dims%x_l_end + 1, n
            SOL( j ) = SOL( j ) + one
          END DO
!         SOL( : n ) = SOL( : n ) / REAL( nv, kind = wp )

!  solve L x = r

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          IF ( identity_h ) THEN
!           SOL( : n ) = SOL( : n )
          ELSE IF ( scaled_identity_h ) THEN
            SOL( : n ) = SOL( : n ) / root_hd
          ELSE IF ( diagonal_h ) THEN
            SOL( : n ) = SOL( : n ) / SQRT( H_sbls%val( : n ) )
          ELSE
            CALL SLS_part_solve( 'S', SOL( : n ), SLS_data,                    &
                                 SLS_control, inform%SLS_inform )
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

!  compute sigma = x^T x as a 'sample' Hessian

          sigma = DOT_PRODUCT( SOL( : n ), SOL( : n ) )
        END IF

!  variables with lower bounds

        DO i = dims%x_free + 1, dims%x_l_end
          IF ( GZ_l( i ) < zero ) THEN
            Z_l( i ) = - GZ_l( i ) / sigma
            IF ( Z_l( i ) > stop_d ) X_status( i ) = 1
          ELSE
            Z_l( i ) = zero
          END IF
        END DO

!  variables with upper bounds

        DO i = dims%x_u_start, n
          IF ( GZ_u( i ) > zero ) THEN
            Z_u( i ) = - GZ_u( i ) / sigma
            IF ( Z_u( i ) < - stop_d ) X_status( i ) = X_status( i ) + 2
          ELSE
            Z_u( i ) = zero
          END IF
        END DO

!  equality constraints

        IF ( m > 0 ) THEN
          DO i = 1, dims%c_equality
            IF ( GY_l( i ) < zero ) THEN
              Y_l( i ) = - GY_l( i ) / sigma
            ELSE IF ( GY_l( i ) > zero ) THEN
              Y_l( i ) = - GY_l( i ) / sigma
            ELSE
              Y_l( i ) = zero
            END IF
          END DO
          C_status(: dims%c_equality ) = - 1

!  constraints with lower bounds

          DO i = dims%c_l_start, dims%c_l_end
            IF ( GY_l( i ) < zero ) THEN
              Y_l( i ) = - GY_l( i ) / sigma
              IF ( Y_l( i ) > stop_d ) C_status( i ) = 1
            ELSE
              Y_l( i ) = zero
            END IF
          END DO

!  constraints with upper bounds

          DO i = dims%c_u_start, dims%c_u_end
            IF ( GY_u( i ) > zero ) THEN
              Y_u( i ) = - GY_u( i ) / sigma
              IF ( Y_u( i ) < - stop_d ) C_status( i ) = C_status( i ) + 2
            ELSE
              Y_u( i ) = zero
            END IF
          END DO
        END IF

!  ..................................
!  all variables free of their bounds
!  ..................................

      ELSE IF ( dual_starting_point == 3 ) THEN

!  variables with lower bounds

        DO i = dims%x_free + 1, dims%x_l_end
          Z_l( i ) = MAX( Z( i ), one )
          IF ( Z_l( i ) > stop_d ) X_status( i ) = 1
        END DO

!  variables with upper bounds

        DO i = dims%x_u_start, n
          Z_u( i ) = MIN( Z( i ), - one )
          IF ( Z_u( i ) < - stop_d ) X_status( i ) = X_status( i ) + 2
        END DO

        IF ( m > 0 ) THEN
          Y_l( : dims%c_equality ) = Y( : dims%c_equality )
          C_status(: dims%c_equality ) = - 1

!  constraints with lower bounds

          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = MAX( Y( i ), one )
            IF ( Y_l( i ) > stop_d ) C_status( i ) = 1
          END DO

!  constraints with upper bounds

          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = MIN( Y( i ), - one )
            IF ( Y_u( i ) < - stop_d ) C_status( i ) = C_status( i ) + 2
          END DO
        END IF

!  ...................................
!  all variables fixed on their bounds
!  ...................................

      ELSE IF ( dual_starting_point == 4 ) THEN

!  variables with lower bounds

        DO i = dims%x_free + 1, dims%x_l_end
          Z_l( i ) = zero
          IF ( Z_l( i ) > stop_d ) X_status( i ) = 1
        END DO

!  variables with upper bounds

        DO i = dims%x_u_start, n
          Z_u( i ) = zero
          IF ( Z_u( i ) < - stop_d ) X_status( i ) = X_status( i ) + 2
        END DO

        IF ( m > 0 ) THEN
          Y_l( : dims%c_equality ) = Y( : dims%c_equality )
          C_status(: dims%c_equality ) = - 1

!  constraints with lower bounds

          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = zero
            IF ( Y_l( i ) > stop_d ) C_status( i ) = 1
          END DO

!  constraints with upper bounds

          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = zero
            IF ( Y_u( i ) < - stop_d )  C_status( i ) = C_status( i ) + 2
          END DO
        END IF

!  ........................................
!  primal-dual user-supplied starting point
!  ........................................

      ELSE IF ( dual_starting_point == - 1 ) THEN

!  variables with lower bounds

        DO i = dims%x_free + 1, dims%x_l_end
          Z_l( i ) = MAX( Z( i ), zero )
          val = MAX( X( i ) - X_l( i ), zero )
          IF ( val < Z_l( i ) ) THEN
            X_status( i ) = 1
          ELSE
            Z_l( i ) = zero
          END IF
        END DO

!  variables with upper bounds

        DO i = dims%x_u_start, n
          Z_u( i ) = MIN( Z( i ), zero )
          val = MAX( X_u( i ) - X( i ), zero )
          IF ( val < - Z_u( i ) ) THEN
            X_status( i ) = X_status( i ) + 2
          ELSE
            Z_u( i ) = zero
          END IF
        END DO

        IF ( m > 0 ) THEN
          Y_l( : dims%c_equality ) = Y( : dims%c_equality )
          C_status(: dims%c_equality ) = - 1

!  constraints with lower bounds

          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = MAX( Y( i ), zero )
            val = MAX( C( i ) - C_l( i ), zero )
            IF ( val < Y_l( i ) ) THEN
              C_status( i ) = 1
            ELSE
              Y_l( i ) = zero
            END IF
          END DO

!  constraints with upper bounds

          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = MIN( Y( i ), zero )
            val = MAX( C_u( i ) - C( i ), zero )
            IF ( val < - Y_u( i ) ) THEN
              C_status( i ) = C_status( i ) + 2
            ELSE
              Y_u( i ) = zero
            END IF
          END DO
        END IF
!write(99,"( ( I6, I6 ) )" ) ( i, X_status( i ), i = dims%x_free + 1, n )

!  ............................
!  user-supplied starting point
!  ............................

      ELSE

!  variables with lower bounds

        DO i = dims%x_free + 1, dims%x_l_end
          Z_l( i ) = MAX( Z( i ), zero )
          IF ( Z_l( i ) > stop_d ) X_status( i ) = 1
        END DO

!  variables with upper bounds

        DO i = dims%x_u_start, n
          Z_u( i ) = MIN( Z( i ), zero )
          IF ( Z_u( i ) < - stop_d ) X_status( i ) = X_status( i ) + 2
        END DO

        IF ( m > 0 ) THEN
          Y_l( : dims%c_equality ) = Y( : dims%c_equality )
          C_status(: dims%c_equality ) = - 1

!  constraints with lower bounds

          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = MAX( Y( i ), zero )
            IF ( Y_l( i ) > stop_d ) C_status( i ) = 1
          END DO

!  constraints with upper bounds

          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = MIN( Y( i ), zero )
            IF ( Y_u( i ) < - stop_d ) C_status( i ) = C_status( i ) + 2
          END DO
        END IF
      END IF
! write(6,*) m, COUNT( C_status(: m ) /= 0 ), n, COUNT( X_status(: n ) /= 0 )
      C_status_old( : m ) = C_status( : m )
      X_status_old( : n ) = X_status( : n )
!write(6,*) ' C_status ', C_status( : m )
!write(6,*) ' X_status ', X_status( : n )

!  print details of the starting point if required ...

!     IF ( .TRUE. ) THEN
      IF ( printd ) THEN
        WRITE( out, "( /, A, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X,      &
       &       'z_l', 9X, 'z_u     stat')") prefix
        DO i = 1, dims%x_free
          WRITE( out, "( A, I6, ES12.4, 4( '      -     ' ), I5 )" ) prefix,   &
            i, X( i ), X_status( i )
        END DO
        DO i = dims%x_free + 1, dims%x_l_start - 1
          WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,               &
         &  '      -     ', I5 )" ) prefix, i, X( i ), zero, Z_l( i ),         &
            X_status( i )
        END DO
        DO i = dims%x_l_start, dims%x_u_start - 1
          WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,               &
         &  '      -     ', I5 )" ) prefix, i, X( i ), X_l( i ),               &
            Z_l( i ), X_status( i )
        END DO
        DO i = dims%x_u_start, dims%x_l_end
          WRITE( out, "( A, I6, 5ES12.4, I5 )" )                               &
             prefix, i, X( i ), X_l( i ), X_u( i ), Z_l( i ),                  &
             Z_u( i ), X_status( i )
        END DO
        DO i = dims%x_l_end + 1, dims%x_u_end
          WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,                &
         &  '      -     ', ES12.4, I5 )" ) prefix, i, X( i ), X_u( i ),       &
            Z_u( i ), X_status( i )
        END DO
        DO i = dims%x_u_end + 1, n
          WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,                &
         &  '      -     ',  ES12.4, I5 )" ) prefix, i, X( i ), zero,          &
            Z_u( i ), X_status( i )
        END DO

!  ... and of the constraints

        IF ( m > 0 ) THEN
          WRITE( out, "( /, A, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X,     &
         &     'y_l', 9X, 'y_u     stat' )") prefix
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( A, I6, 4ES12.4, 12X, I5 )" ) prefix, i, C( i ),     &
              C_l( i ), C_u( i ), Y_l( i ), C_status( i )
          END DO
          DO i = dims%c_l_start, dims%c_u_start - 1
            WRITE( out,  "( A, I6, 2ES12.4, '      -     ', ES12.4,            &
           &  '      -     ', I5 )" ) prefix, i, C( i ), C_l( i ),             &
              Y_l( i ), C_status( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            WRITE( out, "( A, I6, 5ES12.4, I5 )" ) prefix,                     &
              i, C( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i ),               &
              C_status( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4, '      -',   &
            &  '     ', ES12.4, I5 )" ) prefix, i, C( i ), C_u( i ),           &
              Y_u( i ), C_status( i )
          END DO
        END IF
      END IF

!  set the feasibility tolerance and step bound

      feas_tol = epsmch
      step_max = SQRT( SQRT( HUGE( one ) ) )
      stop_reasonable = epsmch ** 0.5
      refactor = .TRUE.
      subspace_direct = control%subspace_direct
      change_subspace = nv
      no_change = 0
      alpha = one
      alpha_subspace = zero

!  Notation:
!  ========

!  v = ( y_l ), J = ( A ), b = ( c_l ),
!      ( y_u )      ( A )      ( c_u )
!      ( z_l )      ( I )      ( x_l )
!      ( z_u )      ( I )      ( x_u )

!  D = { v: (y_l,z_l) >= 0 & (y_u,z_u) <= 0 },
!  P_D [v] is the projection of v into D, and
!    q_d(v) = 1/2 ( J^T v - g )^T H^{-1} ( J^T v - g ) - <b, v> - f
!  is the dual objective function

!  N.B. For the penalty function case, we have instead
!  D = { v: 0 <= y_l <= rho, - rho <= y_u <= 0, z_l >= 0 & z_u <= 0 }

!  ---------------------------------------------------------------------
!  ---------------------- Start of Major Iteration ---------------------
!  ---------------------------------------------------------------------

      DO

!write(6,*) ' Y_l(5) ', Y_l(5)
!write(6,*) 'y_l', Y_l( 1 : dims%c_l_end )
!write(6,*) 'y_u', Y_u( dims%c_u_start : dims%c_u_end )
!write(6,*) 'z_l', Z_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) 'z_u', Z_u( dims%x_u_start : n )

!  ======================================================================
!  STEP 1: -*-*-*-*-*-*-*-   compute the dual gradient  -*-*-*-*-*-*-*-*-
!  ======================================================================

        IF ( printp ) WRITE( out, "( /, A, 20( '=' ), ' step 1, iteration ',   &
       &  I0, ', nv = ', I7, 1X, 6( '=' ) )" ) prefix, inform%iter + 1, nv

!  compute the dual residual r_k = J^T v_k - g and store in sol

!write(6,*) ' composite_g ', composite_g
        IF ( composite_g ) THEN
          SOL( : n ) = - GC( : n )
        ELSE
          SOL( : n ) = - G( : n )
        END IF
        DO i = 1, dims%c_u_start - 1
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            SOL( j ) = SOL( j ) + A_val( l ) * Y_l( i )
          END DO
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            SOL( j ) = SOL( j ) + A_val( l ) * ( Y_l( i ) + Y_u( i ) )
          END DO
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            SOL( j ) = SOL( j ) + A_val( l ) * Y_u( i )
          END DO
        END DO
        DO j = dims%x_free + 1, dims%x_u_start - 1
          SOL( j ) = SOL( j ) + Z_l( j )
        END DO
        DO j = dims%x_u_start, dims%x_l_end
          SOL( j ) = SOL( j ) + Z_l( j ) + Z_u( j )
        END DO
        DO j = dims%x_l_end + 1, n
          SOL( j ) = SOL( j ) + Z_u( j )
        END DO

!  find the primal variables by solving H x_k = r_k

        X( : n ) = SOL( : n )
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( identity_h ) THEN
!         X( : n ) = X( : n )
        ELSE IF ( scaled_identity_h ) THEN
          X( : n ) = X( : n ) / h_scale( 1 )
        ELSE IF ( diagonal_h ) THEN
          X( : n ) = X( : n ) / H_sbls%val( : n )
        ELSE
          CALL SLS_solve( H_sbls, X, SLS_data, SLS_control, inform%SLS_inform )
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_solve + clock_now - clock_record

!  compute the dual gradient g^d_k = J x_k - b of q^d(v) at v_k, the
!  dual objective function q_d(v) = 1/2 x_k^T r_k - <b,v> - f, and the
!  dual optimality measure || P_D[ v_k - g^d_k ] - v_k ||

        dual_f = half * DOT_PRODUCT( X( : n ), SOL( : n ) ) - f_all
        dual_g_norm = zero

!write(6,*) ' penalty_objective ', penalty_objective
!  GY_l = dual gradient wrt Y_l

        DO i = 1, dims%c_l_end
          GY_l( i ) = - C_l( i )
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            GY_l( i ) = GY_l( i ) + A_val( l ) * X( A_col( l ) )
          END DO
          dual_f = dual_f - C_l( i ) * Y_l( i )
          IF ( penalty_objective ) THEN
            IF ( i <= dims%c_equality ) THEN
              dual_g_norm = MAX( dual_g_norm, ABS(                             &
                 MIN( MAX( Y_l( i ) - GY_l( i ), - rho ), rho ) - Y_l( i ) ) )
            ELSE
              dual_g_norm = MAX( dual_g_norm, ABS(                             &
                 MIN( MAX( Y_l( i ) - GY_l( i ), zero ), rho ) - Y_l( i ) ) )
            END IF
          ELSE
            IF ( i <= dims%c_equality ) THEN
              dual_g_norm = MAX( dual_g_norm, ABS( GY_l( i ) ) )
            ELSE
              dual_g_norm = MAX( dual_g_norm, ABS(                             &
                 MAX( Y_l( i ) - GY_l( i ), zero ) - Y_l( i ) ) )
            END IF
          END IF
        END DO

!  GY_u = dual gradient wrt Y_u

        DO i = dims%c_u_start, dims%c_u_end
          GY_u( i ) = - C_u( i )
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            GY_u( i ) = GY_u( i ) + A_val( l ) * X( A_col( l ) )
          END DO
          dual_f = dual_f - C_u( i ) * Y_u( i )
          IF ( penalty_objective ) THEN
            dual_g_norm = MAX( dual_g_norm, ABS(                               &
                MAX( MIN( Y_u( i ) - GY_u( i ), zero ), - rho ) - Y_u( i ) ) )
          ELSE
            dual_g_norm = MAX( dual_g_norm, ABS(                               &
                MIN( Y_u( i ) - GY_u( i ), zero ) - Y_u( i ) ) )
          END IF
        END DO

!  GZ_l = dual gradient wrt Z_l

        DO j = dims%x_free + 1, dims%x_l_end
          GZ_l( j ) = X( j ) - X_l( j )
          dual_f = dual_f - X_l( j ) * Z_l( j )
          dual_g_norm = MAX( dual_g_norm, ABS(                                 &
             MAX( Z_l( j ) - GZ_l( j ), zero ) - Z_l( j ) ) )
        END DO

!  GZ_u = dual gradient wrt Z_u

        DO j = dims%x_u_start, n
          GZ_u( j ) = X( j ) - X_u( j )
          dual_f = dual_f - X_u( j ) * Z_u( j )
          dual_g_norm = MAX( dual_g_norm, ABS(                                 &
             MIN( Z_u( j ) - GZ_u( j ), zero ) - Z_u( j ) ) )
        END DO

!write(6,*) ' dual_g_norm ', dual_g_norm
!write(6,*) 'gy_l', GY_l( 1 : dims%c_l_end )
!write(6,*) 'gy_u', GY_u( dims%c_u_start : dims%c_u_end )
!write(6,*) 'gz_l', GZ_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) 'gz_u', GZ_u( dims%x_u_start : n )

! write(6,*) ' dual objective, pr gradient = ', dual_f, dual_g_norm

!  compute stopping tolerances on the first iteration

        IF ( first_iteration ) THEN
          stop_d = MAX( control%stop_abs_p,                                    &
                        control%stop_rel_p * dual_g_norm )
          IF ( printi ) WRITE( out,                                            &
            "(  /, A, '  Primal convergence tolerance =', ES11.4 )" )          &
              prefix, stop_d
        END IF

!  =======================================================================
!  STEP 2: -*-*-*-*-*-*-*-*-*-   test for optimality   -*-*-*-*-*-*-*-*-*-
!  =======================================================================

        IF ( printp ) WRITE( out, "( /, A, 20( '=' ), ' step 2, iteration ',   &
       & I0, 1X, 20( '=' ) )" ) prefix, inform%iter + 1

!  print a summary of the iteration

        CALL CLOCK_TIME( clock_now )
        clock_now = clock_now - clock_start + clock_total
        IF ( printi ) THEN
          IF ( first_iteration ) THEN
            first_iteration = .FALSE.
            WRITE( out, 2000 ) prefix, prefix
            WRITE( out, 2020 ) prefix, inform%iter,                            &
              dual_f, dual_g_norm, clock_now
          ELSE
            IF ( printt .OR. ( printi .AND.                                    &
               inform%iter == start_print ) ) WRITE( out, 2000 ) prefix, prefix
            WRITE( out, 2030 ) prefix, inform%iter, skip,                      &
             dual_f, dual_g_norm, m_sbls, change, alpha,                       &
             m_subspace, change_subspace, alpha_subspace, clock_now
          END IF
        END IF

!  test for optimality

        CALL CPU_TIME( time_record  )
        CALL CHECKPOINT( inform%iter, REAL( time_record - time_start, sp ),    &
                         dual_g_norm, inform%checkpointsIter,                  &
                         inform%checkpointsTime, 1, 16 )

        IF ( dual_g_norm <= stop_d .OR.                                        &
             ( no_change > 2 .AND. dual_g_norm <= stop_reasonable ) ) THEN

!  if required, form and factorize the matrix K_0 = (   H     J^T_F_k )
!                                                   ( J_F_k      0    )

          IF ( control%factor_optimal_matrix .AND. control%subspace_direct     &
               .AND. change_subspace > 0 ) THEN

!  form the matrix of primal-active rows of A

!  components from the general constraints

            m_sbls = 0 ; A_sbls%ne = 0

            DO j = 1, m_active
              m_sbls = m_sbls + 1
              i = ABS( C_active( j ) )
              DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                A_sbls%ne = A_sbls%ne + 1
                A_sbls%row( A_sbls%ne ) = m_sbls
                A_sbls%col( A_sbls%ne ) = A_col( l )
                A_sbls%val( A_sbls%ne ) = A_val( l )
              END DO
            END DO

!  components from the simple bounds

            DO i = 1, n_active
              m_sbls = m_sbls + 1
              A_sbls%ne = A_sbls%ne + 1
              A_sbls%row( A_sbls%ne ) = m_sbls
              A_sbls%col( A_sbls%ne ) = ABS( X_active( i ) )
              A_sbls%val( A_sbls%ne ) = one
            END DO
            A_sbls%m = m_sbls

!write(6,*) m_ref, COUNT( A_sbls%row( :  A_sbls%ne ) > m_ref )
!do i = 1, A_sbls%ne
!  if ( A_sbls%row( i ) > m_ref ) write( 6, * ) m_ref, A_sbls%row( i ), i
!end do

!  factorize the matrix K_0

  210       CONTINUE
!write(6,*) ' refactor',  refactor, ' n, m_sbls ', n, m_sbls
            IF ( printw )                                                      &
              WRITE( out, "( /, A, ' ** enter form_and_factor' )" ) prefix
            CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SBLS_form_and_factorize( n, m_sbls, H_sbls, A_sbls, C_sbls,   &
                                          SBLS_data, SBLS_control,             &
                                          inform%SBLS_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            IF ( printw )                                                      &
              WRITE( out, "( /, A, ' ** form_and_factor complete' )" ) prefix
            inform%time%factorize                                              &
              = inform%time%factorize + time_now - time_record
            inform%time%clock_factorize                                        &
              = inform%time%clock_factorize + clock_now - clock_record
            inform%nfacts = inform%nfacts + 1

            IF ( printd )                                                      &
              CALL SBLS_eigs( SBLS_data, out, inform%SBLS_inform )

            IF ( printt ) WRITE( out, "( /, A, ' on exit from SBLS: status = ',&
           &    I0, ', time = ', F0.2 )" )                                     &
                 prefix, inform%SBLS_inform%status, clock_now - clock_record

!  check for success

            IF ( inform%SBLS_inform%status < 0 ) THEN
              IF ( printi ) THEN
                IF ( inform%SBLS_inform%status == GALAHAD_error_analysis )     &
                  WRITE( out, "( A, ' error in SLS_analyse called from',       &
                 &  ' SBLS_form_and_factorize, status = ', I0 )" )             &
                    prefix, inform%SBLS_inform%SLS_inform%status
                IF ( inform%SBLS_inform%status == GALAHAD_error_factorization )&
                  WRITE( out, "( A, ' error in SLS_factorize called from',     &
                 &  ' SBLS_form_and_factorize, status = ', I0 )" )             &
                    prefix, inform%SBLS_inform%SLS_inform%status
              END IF

              IF ( inform%SBLS_inform%status == GALAHAD_error_factorization    &
                   .AND. SBLS_control%factorization /= 2 ) THEN
                SBLS_control%factorization = 2
                IF ( printi ) WRITE( out, "( A, ' retrying SLS_factorize' )" ) &
                  prefix
                GO TO 210
              ELSE
                inform%status = GALAHAD_error_factorization ; GO TO 900
              END IF

!  check for singularity

            ELSE IF ( inform%SBLS_inform%rank_def ) THEN
              IF ( printt ) WRITE( out, "( A, '  ** warning ** the matrix is', &
             &                        ' not of full rank, nullity = ', I0 )" ) &
                prefix, n + m_sbls - inform%SBLS_inform%rank
              skip = 'N'

!  the matrix is full rank

            ELSE
              IF ( printt )                                                    &
                WRITE( out, "( A, ' the matrix is of full rank' )" ) prefix
              skip = ' '
            END IF

          END IF
          inform%status = GALAHAD_ok ; GO TO 600
        END IF

!  check that some progress has been made

        IF ( ( alpha == zero .AND. alpha_subspace == zero ) .OR.               &
             ( no_change > 2 .AND. dual_g_norm <= stop_reasonable ) ) THEN
          inform%status = GALAHAD_no_progress ; GO TO 600
        END IF

!  test to see if more than maxit iterations have been performed

        inform%iter = inform%iter + 1
        IF ( inform%iter > control%maxit ) THEN
          inform%status = GALAHAD_error_max_iterations ; GO TO 600
        END IF

!  check that the CPU time limit has not been reached

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        time_now = time_now - time_start + cpu_total
        clock_now = clock_now - clock_start + clock_total

        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now > control%cpu_time_limit ) .OR.                        &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit ; GO TO 600
        END IF

        IF ( ( inform%iter >= start_print .AND. inform%iter < stop_print )     &
             .AND. MOD( inform%iter - start_print, print_gap ) == 0 ) THEN
          printe = set_printe ; printi = set_printi ; printt = set_printt
          printw = set_printw ; printd = set_printd
          print_level = control%print_level
        ELSE
          printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
          printw = .FALSE. ; printd = .FALSE.
          print_level = 0
        END IF

!  ======================================================================
!  STEP 3: -*-*-*-*-*-  compute the gradient arc minimizer  -*-*-*-*-*-*-
!  ======================================================================

        IF ( printp ) WRITE( out, "( /, A, 20( '=' ), ' step 3, iteration ',   &
       & I0, 1X, 20( '=' ) )" ) prefix, inform%iter

!  find the arc minimizer

!    v^C_k = P_D[ v_k + alpha^C_k d_k ],

!  where d = - g^d_k and

!    alpha^C_k = arg min q^D( P_D[ v_k + alpha d_k ] )

!  record the initial point

        V0( ce_start : yl_end ) = Y_l( 1 : dims%c_l_end )
        V0( yu_start : yu_end ) = Y_u( dims%c_u_start : dims%c_u_end )
        V0( zl_start : zl_end ) = Z_l( dims%x_free + 1: dims%x_l_end )
        V0( zu_start : zu_end ) = Z_u( dims%x_u_start : n )

!  record the dual objective value

        q0 = dual_f

!  record the dual gradient g_k

        GV( ce_start : yl_end ) = GY_l( 1 : dims%c_l_end )
        GV( yu_start : yu_end ) = GY_u( dims%c_u_start: dims%c_u_end )
        GV( zl_start : zl_end ) = GZ_l( dims%x_free + 1:dims%x_l_end )
        GV( zu_start : zu_end ) = GZ_u( dims%x_u_start : n )

!  compute the search direction d_k = - g_k / || g_k ||

        norm_pv = MAXVAL( ABS( GV( : nv ) ) )
        PV( : nv ) = - GV( : nv )  / norm_pv

!  set the initial variable status

        V_status( : nv ) = 0

        ip = print_level
!       ip = 4
!       ip = 11

!write(6,"( ' V0 ', /, ( 5ES16.8 ) )" ) V0( ce_start : zu_end )
!write(6,"( ' PV ', /, ( 5ES16.8 ) )" ) PV( : nv )
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        t_solve = inform%time%solve ; c_solve = inform%time%clock_solve
        IF ( control%exact_arc_search ) THEN
          IF ( identity_h ) THEN
            CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV, V_bnd, q0,    &
                     A_ptr, A_col, A_val, V_status, start_ce, start_yl,        &
                     start_yu, start_zl, start_zu, ce_end, yl_end, yu_end,     &
                     zl_end, zu_end, step_max, feas_tol, qc,                   &
                     NZ_p, IUSED, INDEX_r, INDEX_w,                            &
                     out, ip, prefix, BREAK_points, control%infinity,          &
                     control%arc_search_maxit, arc_search_iter, alpha,         &
                     DV, RHS, SOL, RES, H_s, HPV,                              &
                     diagonal_h, scaled_identity_h, identity_h,                &
                     inform%time%solve, inform%time%clock_solve,               &
                     inform%status )
          ELSE IF ( scaled_identity_h ) THEN
            CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV, V_bnd, q0,    &
                     A_ptr, A_col, A_val, V_status, start_ce, start_yl,        &
                     start_yu, start_zl, start_zu, ce_end, yl_end, yu_end,     &
                     zl_end, zu_end, step_max, feas_tol, qc,                   &
                     NZ_p, IUSED, INDEX_r, INDEX_w,                            &
                     out, ip, prefix, BREAK_points, control%infinity,          &
                     control%arc_search_maxit, arc_search_iter, alpha,         &
                     DV, RHS, SOL, RES, H_s, HPV,                              &
                     diagonal_h, scaled_identity_h, identity_h,                &
                     inform%time%solve, inform%time%clock_solve,               &
                     inform%status, HESSIAN = H_sbls )
          ELSE IF ( diagonal_h ) THEN
            CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV, V_bnd, q0,    &
                     A_ptr, A_col, A_val, V_status, start_ce, start_yl,        &
                     start_yu, start_zl, start_zu, ce_end, yl_end, yu_end,     &
                     zl_end, zu_end, step_max, feas_tol, qc,                   &
                     NZ_p, IUSED, INDEX_r, INDEX_w,                            &
                     out, ip, prefix, BREAK_points, control%infinity,          &
                     control%arc_search_maxit, arc_search_iter, alpha,         &
                     DV, RHS, SOL, RES, H_s, HPV,                              &
                     diagonal_h, scaled_identity_h, identity_h,                &
                     inform%time%solve, inform%time%clock_solve,               &
                     inform%status, HESSIAN = H_sbls )
          ELSE
            CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV, V_bnd, q0,    &
                     A_ptr, A_col, A_val, V_status, start_ce, start_yl,        &
                     start_yu, start_zl, start_zu, ce_end, yl_end, yu_end,     &
                     zl_end, zu_end, step_max, feas_tol, qc,                   &
                     NZ_p, IUSED, INDEX_r, INDEX_w,                            &
                     out, ip, prefix, BREAK_points, control%infinity,          &
                     control%arc_search_maxit, arc_search_iter, alpha,         &
                     DV, RHS, SOL, RES, H_s, HPV,                              &
                     diagonal_h, scaled_identity_h, identity_h,                &
                     inform%time%solve, inform%time%clock_solve,               &
                     inform%status, HESSIAN = H_sbls,                          &
                     SLS_data = SLS_data, SLS_control = SLS_control,           &
                     SLS_inform = inform%SLS_inform )
          END IF
        ELSE
          IF ( identity_h .OR. scaled_identity_h ) THEN
            IF ( composite_g ) THEN
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       GC, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),    &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            ELSE
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       G, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),     &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            END IF
          ELSE IF ( diagonal_h ) THEN
            IF ( composite_g ) THEN
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       GC, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),    &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            ELSE
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       G, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),     &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            END IF
          ELSE
            IF ( composite_g ) THEN
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       GC, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),    &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, SLS_data = SLS_data,                     &
                       SLS_control = SLS_control,                              &
                       SLS_inform = inform%SLS_inform )
            ELSE
              CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qc, alpha,  &
                       G, q0, Y_l, Y_u, Z_l, Z_u, C_l( 1 : dims%c_l_end ),     &
                       C_u( dims%c_u_start : dims%c_u_end ),                   &
                       X_l( dims%x_free + 1 : dims%x_l_end ),                  &
                       X_u( dims%x_u_start : n ),                              &
                       ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,   &
                       zl_start, zl_end, zu_start, zu_end,                     &
                       A_ptr, A_col, A_val, step_max, feas_tol,                &
                       V_status, arc_search_iter,                              &
                       out, ip, prefix, DV, RHS, H_s,                          &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, SLS_data = SLS_data,                     &
                       SLS_control = SLS_control,                              &
                       SLS_inform = inform%SLS_inform )
            END IF
          END IF
        END IF
!write(6,"( ' VT ', /, ( 5ES16.8 ) )" ) VT( ce_start : zu_end )
!write(6,"( ' V_status ', /, ( 5I5 ) )" ) V_status( : nv )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%search = inform%time%search + time_now - time_record       &
          + t_solve - inform%time%solve
        inform%time%clock_search = inform%time%clock_search                    &
          + clock_now - clock_record + c_solve - inform%time%clock_solve

!  write(6, "( ' q after arc search ', ES12.4 )" ) qc

!  record the arc minimizer

        YC_l( 1 : dims%c_l_end ) = VT( ce_start : yl_end )
        YC_u( dims%c_u_start : dims%c_u_end ) = VT( yu_start: yu_end )
        ZC_l( dims%x_free + 1 : dims%x_l_end ) = VT( zl_start : zl_end )
        ZC_u( dims%x_u_start : n ) = VT( zu_start : zu_end )

        IF (  inform%status == GALAHAD_error_primal_infeasible ) GO TO 700

!  if required compute the dual objective value. Record sol = J^T v_k - g

!       IF ( .TRUE. ) THEN
!write(6,*) ' composite_g ', composite_g
        IF ( printw ) THEN
          IF ( composite_g ) THEN
            SOL( : n ) = - GC( : n )
          ELSE
            SOL( : n ) = - G( : n )
          END IF
          DO i = 1, dims%c_u_start - 1
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * YC_l( i )
            END DO
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * ( YC_l( i ) + YC_u( i ) )
            END DO
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * YC_u( i )
            END DO
          END DO
          DO j = dims%x_free + 1, dims%x_u_start - 1
            SOL( j ) = SOL( j ) + ZC_l( j )
          END DO
          DO j = dims%x_u_start, dims%x_l_end
            SOL( j ) = SOL( j ) + ZC_l( j ) + ZC_u( j )
          END DO
          DO j = dims%x_l_end + 1, n
            SOL( j ) = SOL( j ) + ZC_u( j )
          END DO

!  find the primal variables by solving H x_k = r_k

          X( : n ) = SOL( : n )
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          IF ( identity_h ) THEN
!           X( : n ) = X( : n )
          ELSE IF ( scaled_identity_h ) THEN
            X( : n ) = X( : n ) / h_scale( 1 )
          ELSE IF ( diagonal_h ) THEN
            X( : n ) = X( : n ) / H_sbls%val( : n )
          ELSE
            CALL SLS_solve( H_sbls, X, SLS_data, SLS_control, inform%SLS_inform)
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

!  compute the dual objective function q_d(v) = 1/2 x_k^T r_k - <b,v> - f

          dual_f = half * DOT_PRODUCT( X( : n ), SOL( : n ) ) - f_all
          DO i = 1, dims%c_l_end
            dual_f = dual_f - C_l( i ) * YC_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            dual_f = dual_f - C_u( i ) * YC_u( i )
          END DO
          DO j = dims%x_free + 1, dims%x_l_end
            dual_f = dual_f - X_l( j ) * ZC_l( j )
          END DO
          DO j = dims%x_u_start, n
            dual_f = dual_f - X_u( j ) * ZC_u( j )
          END DO
          WRITE( out, "( A, ' dual obj after arc search (computed,',           &
         &               ' recurred) =', 2ES14.6 )" ) prefix, dual_f, qc
        END IF
!write(6,*) ' dual_f ', dual_f

!  record the status of the general constraints and simple bounds
!   _status = 0  primal inactive
!           = 1  primal active at lower bound
!           = 2  primal active at upper bound

        C_status = 0 ; X_status = 0
        m_active = 0 ;  n_active = 0 ; m_sbls = 0 ; change = 0
        DO ii = ce_start, yl_end
          i = ii
          IF ( V_status( ii ) == 0 ) THEN ! lower
            m_sbls = m_sbls + 1 ; C_status( i ) = 1
            m_active = m_active + 1
            C_active( m_active ) = - i
          END IF
        END DO

        DO ii = yu_start, yu_end
          i = ii - yu_start + dims%c_u_start
          IF ( V_status( ii ) == 0 ) THEN ! upper
            m_sbls = m_sbls + 1 ; C_status( i ) = C_status( i ) + 2
            m_active = m_active + 1
            C_active( m_active ) = i
          END IF
        END DO

        DO ii = zl_start, zl_end
          i = ii - zl_start + dims%x_free + 1
          IF ( V_status( ii ) == 0 ) THEN ! lower
            m_sbls = m_sbls + 1 ; X_status( i ) = 1
            n_active = n_active + 1
            X_active( n_active ) = - i
          END IF
        END DO

        DO ii = zu_start, zu_end
          i = ii - zu_start + dims%x_u_start
          IF ( V_status( ii ) == 0 ) THEN ! upper
            m_sbls = m_sbls + 1 ;  X_status( i ) = X_status( i ) + 2
            n_active = n_active + 1
            X_active( n_active ) = i
          END IF
        END DO

!write(6,*) ' C_status ', C_status( : m )
!write(6,*) ' X_status ', X_status( : n )
!if ( n_active > 0 ) write(6,*) ' X_active ', X_active( : n_active )

!  record the number of changes and, if required, the list of those that
!  have changed: CHANGES(:added) are added and CHANGES(m+n-deleted:m+n)
!  are removed, with CHANGES() <= m -> constraint CHANGES() that changes and
!  CHANGES() > m -> bound CHANGES()-m that changes

        IF ( control%max_sc > 0 ) THEN
          added = 0 ; deleted = 0
          DO i = 1, m
            IF ( C_status( i ) /= C_status_old( i ) ) THEN
              IF ( C_status( i ) == 0 ) THEN
                CHANGES( mpn - deleted ) = i
!               write(6,*) 'c', mpn - deleted, i
!               write(6,*) 'c del', i
                deleted = deleted + 1
              ELSE
                added = added + 1
                CHANGES( added ) = i
!               write(6,*) 'c add', i
              END IF
            END IF
          END DO
          DO i = 1, n
            IF ( X_status( i ) /= X_status_old( i ) ) THEN
              change = change + 1
              IF ( X_status( i ) == 0 ) THEN
                CHANGES( mpn - deleted ) = m + i
!               write(6,*) 'x', mpn - deleted, m + i
!               write(6,*) 'x del', m + i
                deleted = deleted + 1
              ELSE
                added = added + 1
                CHANGES( added ) = m + i
!               write(6,*) 'x add', m + i
              END IF
           END IF
          END DO
          change = added + deleted
!         write(6,*) ' deleted ', deleted
        ELSE
          change = COUNT( X_status( : n ) /= X_status_old( : n ) ) +           &
                   COUNT( C_status( : m ) /= C_status_old( : m ) )
        END IF
        C_status_old = C_status ; X_status_old = X_status
!write(6,*) ' max_sc, change ', control%max_sc, change

!write(6,*) 'yc_l', YC_l( 1 : dims%c_l_end )
!write(6,*) 'yc_u', YC_u( dims%c_u_start : dims%c_u_end )
!write(6,*) 'zc_l', ZC_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) 'zc_u', ZC_u( dims%x_u_start : n )

        IF ( printw ) WRITE( out, "( A, 1X, A, I0, A, I0, A, I0 )" ) prefix,   &
          ' n active = ', n_active, ', m active = ', m_active,                 &
          ', total active = ', m_sbls

!  ensure that the Schur complement is updated to account for the changes
!  in the active set

        IF ( subspace_direct .AND. .NOT. refactor .AND.                        &
             change > 0 .AND. .NOT. control%subspace_alternate ) THEN

!  if the number of additions and implied deletions exceeds the permitted size
!  of the Schur complement, restart with a new reference factorization

          IF ( deleted > 0 ) THEN
            i = COUNT( ACTIVE_status( CHANGES( mpn - deleted + 1 : mpn ) )     &
                         <= m_ref )
          ELSE
            i = 0
          END IF
          IF ( SCU_mat%m + added + i > SCU_mat%m_max ) THEN
            refactor = .TRUE. ; GO TO 390
          END IF

!         write(6,*) ' add, del ', added, deleted
!  remove the constraints that are to be deleted

          DO j = 0, deleted - 1
            i = CHANGES( mpn - j )
            ii = ACTIVE_status( i )

!  the constraint lies in the reference set

            IF ( ii <= m_ref ) THEN
              IF ( SCU_mat%m + 1 > SCU_mat%m_max ) THEN
                refactor = .TRUE. ; GO TO 390
              END IF

!  remove the comstraint by adding a row whose single nonzero value, one,
!  lies in the column of K corresponding to the outgoing constraint

              l = SCU_mat%BD_col_start( SCU_mat%m + 1 )
              IF ( l + 1 > lbd ) THEN
                refactor = .TRUE. ; GO TO 390
              END IF
              SCU_mat%BD_row( l ) = n + ii
              SCU_mat%BD_val( l ) = one
              l = l + 1
              SCU_mat%BD_col_start( SCU_mat%m + 2 ) = l

!  update the Schur complement

              inform%scu_status = 1
              DO
                CALL SCU_append( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),   &
                                 inform%scu_status, inform%SCU_inform )
                IF ( inform%scu_status <= 0 ) EXIT
                growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) )
                CALL SBLS_solve( n, m_ref, A_sbls, C_sbls, SBLS_data,        &
                                 SBLS_control, inform%SBLS_inform,           &
                                 VECTOR( : SCU_mat%n ) )
                growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) ) / growth
                IF ( growth > control%max_growth ) THEN
                  refactor = .TRUE. ; GO TO 390
                END IF
                IF ( inform%SBLS_inform%status < 0 ) THEN
                  IF ( printi ) WRITE( out,                                  &
                    "( ' SBLS_solve, status = ', I0 )" )                     &
                      inform%SBLS_inform%status
                  inform%status = inform%SBLS_inform%status
                  GO TO 900
                END IF
              END DO

!  check to ensure that the updated matrix is non singular; if it is, restart
!  with a new reference factorization

              IF ( inform%scu_status < 0 ) THEN
!write(6,*) ' SCU_append, status = ', inform%scu_status
                refactor = .TRUE. ; GO TO 390
              END IF
!write(6,*) ' m ', SCU_mat%m

!  update ACTIVE_list and update ACTIVE_status accordingly

              len_list = len_list + 1
              ACTIVE_status( i ) = 0
              ACTIVE_list( ii ) = 0
              ACTIVE_list( len_list ) = 0

!  the constraint was added since the reference set was established

            ELSE
              CALL SCU_delete( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),       &
                               inform%scu_status, inform%SCU_inform,           &
                               ii - m_ref )
!write(6,*) ' m ', SCU_mat%m

!  check to ensure that the updated matrix is non singular; if it is, restart
!  with a new reference factorization

              IF ( inform%scu_status < 0 ) THEN
!write(6,*) ' SCU_delete, status = ', inform%scu_status
                refactor = .TRUE. ; GO TO 390
              END IF

!  decrement ACTIVE_list and update ACTIVE_status accordingly

              ACTIVE_status( i ) = 0
              DO l = ii, len_list - 1
                ACTIVE_list( l ) = ACTIVE_list( l + 1 )
                IF ( ACTIVE_list( l ) > 0 )                                  &
                  ACTIVE_status( ACTIVE_list( l ) ) = l
              END DO
              len_list = len_list - 1
            END IF
          END DO

!  if the number of additions exceeds the permitted size of the Schur
!  complement, restart with a new reference factorization

          IF ( SCU_mat%m + added > SCU_mat%m_max ) THEN
            refactor = .TRUE. ; GO TO 390
          END IF

!  append the constraints that are to be added

          DO j = 1, added
            i = CHANGES( j )
!write(6,*) ' constraint ', i
!  record the data for the incoming constraint in K_+

            l = SCU_mat%BD_col_start( SCU_mat%m + 1 )
            IF ( i <= m ) THEN  !  add a linear constraint
              IF ( l + A_ptr( i + 1 ) - A_ptr( i ) > lbd ) THEN
                refactor = .TRUE. ; GO TO 390
              END IF
              DO ii = A_ptr( i ), A_ptr( i + 1 ) - 1
                SCU_mat%BD_row( l ) = A_col( ii )
                SCU_mat%BD_val( l ) = A_val( ii )
                l = l + 1
              END DO
            ELSE ! add a simple-bound constraint
              IF ( l + 1 > lbd ) THEN
                refactor = .TRUE. ; GO TO 390
              END IF
              SCU_mat%BD_row( l ) = i - m
              SCU_mat%BD_val( l ) = one
              l = l + 1
            END IF
            SCU_mat%BD_col_start( SCU_mat%m + 2 ) = l

!do l = SCU_mat%BD_col_start( SCU_mat%m + 1 ), &
! SCU_mat%BD_col_start( SCU_mat%m + 2 ) - 1
!write(6,*) ' row, val ',  SCU_mat%BD_row( l ),  SCU_mat%BD_val( l )
!end do

!  update the Schur complement

            inform%scu_status = 1
            DO
              CALL SCU_append( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),       &
                               inform%scu_status, inform%SCU_inform )
              IF ( inform%scu_status <= 0 ) EXIT
              growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) )
              CALL SBLS_solve( n, m_ref, A_sbls, C_sbls, SBLS_data,            &
                               SBLS_control, inform%SBLS_inform,               &
                               VECTOR( : SCU_mat%n ) )
              growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) ) / growth
              IF ( growth > control%max_growth ) THEN
                refactor = .TRUE. ; GO TO 390
              END IF
              IF ( inform%SBLS_inform%status < 0 ) THEN
                IF ( printi ) WRITE( out,                                      &
                  "( ' SBLS_solve, status = ', I0 )" )                         &
                    inform%SBLS_inform%status
                inform%status = inform%SBLS_inform%status
                GO TO 900
              END IF
            END DO

!  if the updated matrix is singular, restart with a new reference factorization

            IF ( inform%scu_status < 0 ) THEN
!write(6,*) ' SCU_append, status = ', inform%scu_status
!write(6,*) ' refactor ', inform%scu_status
              refactor = .TRUE. ; GO TO 390
            END IF
!write(6,*) SCU_mat%m
!write(6,*) ' m ', SCU_mat%m

!  increment ACTIVE_list and update ACTIVE_status accordingly

            len_list = len_list + 1
            ACTIVE_status( i ) = len_list
            ACTIVE_list( len_list ) = i
!write(6,*) ' active ', ACTIVE_list( : len_list )
          END DO
        END IF

!  if desired, only compute the subspace step when the active set appears to
!  have settled down

  390   CONTINUE
        IF ( control%cauchy_only >= 0 .AND. change > control%cauchy_only ) THEN
          m_subspace = m_sbls
          change_subspace = 0
          alpha_subspace = zero
          skip = ' '
          Y_l( 1 : dims%c_l_end ) = YC_l( 1 : dims%c_l_end )
          Y_u( dims%c_u_start : dims%c_u_end )                                 &
            = YC_u( dims%c_u_start : dims%c_u_end )
          Z_l( dims%x_free + 1 : dims%x_l_end )                                &
            = ZC_l( dims%x_free + 1 : dims%x_l_end )
          Z_u( dims%x_u_start : n ) = ZC_u( dims%x_u_start : n )
          IF ( printp ) WRITE( out, "( /, A, ' skipping subspace step as the', &
         &  ' active set has not settled sufficiently' )" ) prefix
          CYCLE
        END IF

!  ======================================================================
!  STEP 4: -*-*-*-*-*-*-*-   compute the subspace step  -*-*-*-*-*-*-*-*-
!  ======================================================================

        IF ( printp ) WRITE( out, "( /, A, 20( '=' ), ' step 4, iteration ',   &
       & I0, 1X, 20( '=' ) )" ) prefix, inform%iter

!  4a. find the improved subspace search direction
!  ...............................................

!    delta v_k = arg min q^D(v^C_k + delta v) : delta v_A_k = 0

!  where the dual active set A_k = { i : (v^C_k)_i = "on bound" }.

!  This is equivalent to solving the linear system

!    (   H     J^T_F_k ) (    delta x_k  ) = ( J^T_F_k v^C_F_k - g  )        (S)
!    ( J_F_k      0    ) ( - delta v_F_k )   (       b_F_k          )

!  where the primal active set F_k is the complement set to A_k or, if (S)
!  is inconsistent, finding a direction of linear infinite descent
!  delta v_k such that J_F_k^T delta v_k = 0 and delta v_k^T b_F_k > 0

!  form the matrix of primal-active rows of A

!  components from the general constraints

        m_sbls = 0 ; A_sbls%ne = 0

        DO j = 1, m_active
          m_sbls = m_sbls + 1
          i = ABS( C_active( j ) )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            A_sbls%ne = A_sbls%ne + 1
            A_sbls%row( A_sbls%ne ) = m_sbls
            A_sbls%col( A_sbls%ne ) = A_col( l )
            A_sbls%val( A_sbls%ne ) = A_val( l )
          END DO
        END DO

!  components from the simple bounds

        DO i = 1, n_active
          m_sbls = m_sbls + 1
          A_sbls%ne = A_sbls%ne + 1
          A_sbls%row( A_sbls%ne ) = m_sbls
          A_sbls%col( A_sbls%ne ) = ABS( X_active( i ) )
          A_sbls%val( A_sbls%ne ) = one
        END DO
        A_sbls%m = m_sbls

        dolid = .FALSE.
        skip = ' '

!write(6,*) m_ref, COUNT( A_sbls%row( :  A_sbls%ne ) > m_ref )
!do i = 1, A_sbls%ne
!  if ( A_sbls%row( i ) > m_ref ) write( 6, * ) m_ref, A_sbls%row( i ), i
!end do

        IF ( COUNT( A_sbls%row( :  A_sbls%ne ) > m_ref ) > 0 ) refactor = .TRUE.

!  use a direct method to find the subspace
!  . . . . . . . . . . . . . . . . . . . .

        subspace_direct_save = subspace_direct
        IF ( change == 0 .AND. change_subspace == 0 ) THEN
          no_change = no_change + 1
        ELSE
          no_change = 0
        END IF
        IF ( no_change > 0 .AND. .NOT. subspace_direct .AND.                   &
             control%explore_optimal_subspace >= 2 ) THEN
          subspace_direct = .TRUE.
          refactor = .TRUE.
          IF ( printw ) WRITE( out,                                            &
            "( A, ' direct exploration of subspace' )" ) prefix
        END IF

        IF ( subspace_direct ) THEN
          IF ( control%subspace_alternate )                                    &
             subspace_direct = .NOT. subspace_direct

  410     CONTINUE
!write(6,*) ' refactor',  refactor, ' n, m_sbls ', n, m_sbls
          IF ( refactor ) THEN
            IF ( printw )                                                      &
              WRITE( out, "( /, A, ' ** enter form_and_factor' )" ) prefix
            CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SBLS_form_and_factorize( n, m_sbls, H_sbls, A_sbls, C_sbls,   &
                                          SBLS_data, SBLS_control,             &
                                          inform%SBLS_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            IF ( printw )                                                      &
              WRITE( out, "( /, A, ' ** form_and_factor complete' )" ) prefix
            inform%time%factorize                                              &
              = inform%time%factorize + time_now - time_record
            inform%time%clock_factorize                                        &
              = inform%time%clock_factorize + clock_now - clock_record
            inform%nfacts = inform%nfacts + 1

            IF ( printd )                                                      &
              CALL SBLS_eigs( SBLS_data, out, inform%SBLS_inform )

            IF ( printt ) WRITE( out, "( /, A, ' on exit from SBLS: status = ',&
           &    I0, ', time = ', F0.2 )" )                                     &
                 prefix, inform%SBLS_inform%status, clock_now - clock_record

!  check for success

            IF ( inform%SBLS_inform%status < 0 ) THEN
              IF ( printi ) THEN
                IF ( inform%SBLS_inform%status == GALAHAD_error_analysis )     &
                  WRITE( out, "( A, ' error in SLS_analyse called from',       &
                 &  ' SBLS_form_and_factorize, status = ', I0 )" )             &
                    prefix, inform%SBLS_inform%SLS_inform%status
                IF ( inform%SBLS_inform%status == GALAHAD_error_factorization )&
                  WRITE( out, "( A, ' error in SLS_factorize called from',     &
                 &  ' SBLS_form_and_factorize, status = ', I0 )" )             &
                    prefix, inform%SBLS_inform%SLS_inform%status
              END IF

              IF ( inform%SBLS_inform%status == GALAHAD_error_factorization    &
                   .AND. SBLS_control%factorization /= 2 ) THEN
                SBLS_control%factorization = 2
                IF ( printi ) WRITE( out, "( A, ' retrying SLS_factorize' )" ) &
                  prefix
                GO TO 410
              END IF

              Y_l( 1 : dims%c_l_end ) = YC_l( 1 : dims%c_l_end )
              Y_u( dims%c_u_start : dims%c_u_end )                             &
                = YC_u( dims%c_u_start : dims%c_u_end )
              Z_l( dims%x_free + 1 : dims%x_l_end )                            &
                = ZC_l( dims%x_free + 1 : dims%x_l_end )
              Z_u( dims%x_u_start : n ) = ZC_u( dims%x_u_start : n )
              alpha = zero
              skip = 'S'
              CYCLE

!           inform%status = inform%SBLS_inform%status
!           GO TO 900

!  check for singularity

            ELSE IF ( inform%SBLS_inform%rank_def ) THEN
              IF ( printt ) WRITE( out, "( A, '  ** warning ** the matrix is', &
             &                        ' not of full rank, nullity = ', I0 )" ) &
                prefix, n + m_sbls - inform%SBLS_inform%rank
              skip = 'N'

!  if the matrix is full rank, initialize the Schur complement if required

            ELSE
              IF ( printt )                                                    &
                WRITE( out, "( A, ' the matrix is of full rank' )" ) prefix
              skip = ' '
              IF ( control%max_sc > 0 ) THEN
                m_ref = m_sbls
                refactor = .FALSE.
                SCU_mat%class = 2
                SCU_mat%n = n + m_sbls
                SCU_mat%m = 0 ; SCU_mat%m_max = control%max_sc
                SCU_mat%BD_col_start( 1 ) = 1
                inform%scu_status = 1
                CALL SCU_factorize( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),  &
                                    inform%scu_status, inform%SCU_inform )

!  the current matrix K_0 from (S) is said to be the "reference" matrix, and
!  will be used as the basis for forthcoming Schur complement updates

!  initialize ACTIVE_list and ACTIVE_status -

!   ACTIVE_list(:len_list) gives the list of constraints that are currently
!   active. Any entry 0 corresponds to a constraint that was active at some
!   time since the current reference matrix was established. Any entry whose
!   value is <= m corresponds to a general constraint, while those > m
!   correspond to variable value - m
!
!   ACTIVE_status(:m) gives the status of each constraint. The status = 0 for
!   an inactive constraint, while > 0 points to the position in ACTIVE_list
!   of an active constraint. ACTIVE_status(m+1:m+n) does the same for
!   simple-bound constraints

                len_list = 0
                ACTIVE_status( : mpn ) = 0

                DO j = 1, m_active
                  len_list = len_list + 1
                  i = C_active( j )
                  IF ( i > 0 ) THEN
                    ACTIVE_status( i ) = len_list
                    ACTIVE_list( len_list ) = i
                  ELSE
!                   ACTIVE_status( - i ) = - len_list
                    ACTIVE_status( - i ) = len_list
                    ACTIVE_list( len_list ) = - i
                  END IF
                END DO

                DO i = 1, n_active
                  len_list = len_list + 1
                  j = X_active( i )
                  IF ( j > 0 ) THEN
                    ACTIVE_status( j + m ) = len_list
                    ACTIVE_list( len_list ) = j + m
                  ELSE
!                   ACTIVE_status( - j + m ) = - len_list
                    ACTIVE_status( - j + m ) = len_list
                    ACTIVE_list( len_list ) = - j + m
                  END IF
                END DO
!write(6,*) ' active ', ACTIVE_list( : len_list )
              END IF
            END IF

!  form the vector of primal-active right-hand sides of (S) in sol

            IF ( composite_g ) THEN
              RHS( : n ) = - GC( : n )
            ELSE
              RHS( : n ) = - G( : n )
            END IF

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

!  N.B. Careful, if there is a penalty term, some of the fixed duals may
!  not be zero!

            IF ( penalty_objective ) THEN
              DO i = 1, dims%c_u_start - 1
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * YC_l( i )
                END DO
              END DO
              DO i = dims%c_u_start, dims%c_l_end
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * ( YC_l( i ) + YC_u( i ) )
                END DO
              END DO
              DO i = dims%c_l_end + 1, dims%c_u_end
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * YC_u( i )
                END DO
              END DO
            ELSE
              DO im = 1, m_active
                i = C_active( im )
                IF ( i < 0 ) THEN
                  DO l = A_ptr( - i ), A_ptr( - i + 1 ) - 1
                    j = A_col( l )
                    RHS( j ) = RHS( j ) + A_val( l ) * YC_l( - i )
                  END DO
                ELSE
                  DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( l )
                    RHS( j ) = RHS( j ) + A_val( l ) * YC_u( i )
                  END DO
                END IF
              END DO
            END IF

!  components from the simple bounds

            DO i = 1, n_active
              ii = ii + 1 ; j = X_active( i )
              IF ( j < 0 ) THEN
                RHS( - j ) = RHS( - j ) + ZC_l( - j )
                RHS( ii ) = X_l( - j )
              ELSE
                RHS( j ) = RHS( j ) + ZC_u( j )
                RHS( ii ) = X_u( j )
              END IF
            END DO

!  find the Fredholm Alternative for the linear system (S); the solution will
!  be in sol

            SOL( : n + m_sbls ) = RHS( : n + m_sbls )
!WRITE(6, "( /, A, ' Fredholm rhs = ', ES12.4 )" ) &
!  prefix, MAXVAL( ABS( RHS( : n + m_sbls ) ) )
!           IF ( .TRUE. ) THEN
            IF ( inform%SBLS_inform%rank_def ) THEN
              IF ( printd ) WRITE( out, "( /, A, ' Fredholm RHS = ', ES12.4 )")&
                prefix, MAXVAL( ABS( RHS( : n + m_sbls ) ) )
              CALL SBLS_fredholm_alternative( n, m_sbls, A_sbls,               &
                                              SBLS_data%efactors,              &
                                              SBLS_control,                    &
                                              inform%SBLS_inform, SOL )
              refactor = .TRUE.
            ELSE
              IF ( printd ) WRITE( out, "( /, A, ' RHS = ', ES12.4 )" )        &
                prefix, MAXVAL( ABS( RHS( : n + m_sbls ) ) )
              growth = MAXVAL( ABS( SOL( : n + m_sbls ) ) ) + epsmch
!WRITE(6, "( /, A, ' sol = ', ES12.4 )" ) &
!  prefix, MAXVAL( ABS( SOL( : n + m_sbls ) ) )
              CALL SBLS_solve( n, m_sbls, A_sbls, C_sbls, SBLS_data,           &
                               SBLS_control, inform%SBLS_inform, SOL )
!WRITE(6, "( /, A, ' sol = ', ES12.4 )" ) &
!  prefix, MAXVAL( ABS( SOL( : n + m_sbls ) ) )
              growth = MAXVAL( ABS( SOL( : n + m_sbls ) ) ) / growth
              IF ( growth > control%max_growth ) refactor = .TRUE.
            END IF
            IF ( inform%SBLS_inform%status < 0 ) THEN
              IF ( printi ) WRITE( out,                                        &
                 "( ' SBLS_fredholm_alternative, status = ', I0 )" )           &
                inform%SBLS_inform%status
              inform%status = inform%SBLS_inform%status
              GO TO 900
            END IF

!  check if there is direction of linear infinite descent

            dolid = inform%SBLS_inform%alternative
            IF ( dolid ) THEN
              skip = 'D'
              IF ( printp ) WRITE( out, "( A, ' direction of linear',          &
             &               ' infinite descent found ' )" ) prefix

!  calculate the slope

              slope = - DOT_PRODUCT( SOL( :  n + m_sbls ),                     &
                                     RHS( :  n + m_sbls ) )
              curv = zero
              IF ( slope > zero ) THEN
                slope = - slope
                SOL( n + 1 : n + m_sbls ) = - SOL( n + 1 : n + m_sbls )
              END IF

!  if required, check the residuals

!             IF ( .TRUE. ) THEN
              IF ( printd ) THEN
                RES( : n + m_sbls ) = zero
                IF ( identity_h ) THEN
                  RES( : n ) = RES( : n ) + SOL( : n )
                ELSE IF ( scaled_identity_h ) THEN
                  RES( : n ) = RES( : n ) + h_scale( 1 ) * SOL( : n )
                ELSE IF ( diagonal_h ) THEN
                  RES( : n ) = RES( : n ) + H_sbls%val( : n ) * SOL( : n )
                ELSE
                  DO i = 1, n
                    DO l = H_sbls%ptr( i ), H_sbls%ptr( i + 1 ) - 1
                      j = H_sbls%col( l )
                      RES( j ) = RES( j ) + H_sbls%val( l ) * SOL( i )
                      IF ( i /= j )                                            &
                        RES( i ) = RES( i ) + H_sbls%val( l ) * SOL( j )
                    END DO
                  END DO
                END IF
                DO l = 1, A_sbls%ne
                  i = n + A_sbls%ROW( l ) ; j = A_sbls%COL( l )
                  RES( j ) = RES( j ) + A_sbls%VAL( l ) * SOL( i )
                  RES( i ) = RES( i ) + A_sbls%VAL( l ) * SOL( j )
                END DO
                WRITE( out, "( A, ' Fredholm residual, slope ', 2ES12.4 )" )   &
                  prefix, MAXVAL( ABS( RES( : n + m_sbls ) ) ), slope
              END IF

!  the system is consistent

            ELSE
              IF ( printp ) WRITE( out, "( A, ' consistent solution',          &
             &                ' to subspace problem found ' )" ) prefix

!  if required, check the residuals

!             IF ( .TRUE. ) THEN
              IF ( printd ) THEN
                RES( : n + m_sbls ) = - RHS( : n + m_sbls )
                IF ( identity_h ) THEN
                  RES( : n ) = RES( : n ) + SOL( : n )
                ELSE IF ( scaled_identity_h ) THEN
                  RES( : n ) = RES( : n ) + h_scale( 1 ) * SOL( : n )
                ELSE IF ( diagonal_h ) THEN
                  RES( : n ) = RES( : n ) + H_sbls%val( : n ) * SOL( : n )
                ELSE
                  DO i = 1, n
                    DO l = H_sbls%ptr( i ), H_sbls%ptr( i + 1 ) - 1
                      j = H_sbls%col( l )
                      RES( j ) = RES( j ) + H_sbls%val( l ) * SOL( i )
                      IF ( i /= j )                                            &
                        RES( i ) = RES( i ) + H_sbls%val( l ) * SOL( j )
                    END DO
                  END DO
                END IF
                DO l = 1, A_sbls%ne
                  i = n + A_sbls%ROW( l ) ; j = A_sbls%COL( l )
                  RES( j ) = RES( j ) + A_sbls%VAL( l ) * SOL( i )
                  RES( i ) = RES( i ) + A_sbls%VAL( l ) * SOL( j )
                END DO
                WRITE( out, "( A, ' Fredholm residual ', ES12.4 )" )           &
                  prefix, MAXVAL( ABS( RES( : n + m_sbls ) ) )
!               WRITE( out, "( 5ES12.4 )" ) RES( : n )
!               WRITE( out, "( 5ES12.4 )" ) RES( n + 1 : n + m_sbls )
              END IF

!  as sol contains - delta v_F_k, replace sol by delta v_k_F_k

              SOL( n + 1 : n + m_sbls ) = - SOL( n + 1 : n + m_sbls )
            END IF

!  alteratively ... try updating the existing factorization using the
!  Schur-complement method. That is, given the reference matrix, we form

!    K = ( K_0  K_+^T )
!        ( K_+    0   )

!  for suitably chosen K_+, and solve systems with K using factors of K_0
!  and the Schur complement K_+ K_0^{1} K_+^T of K_0 in K

          ELSE
            skip = 'R'

!  form the vector of primal-active right-hand sides of (S) in sol

            IF ( composite_g ) THEN
              RHS( : n ) = - GC( : n )
            ELSE
              RHS( : n ) = - G( : n )
            END IF
            RHS( n + 1 : SCU_mat%n + SCU_mat%m ) = zero

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

            IF ( penalty_objective ) THEN
              DO i = 1, dims%c_u_start - 1
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * YC_l( i )
                END DO
              END DO
              DO i = dims%c_u_start, dims%c_l_end
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * ( YC_l( i ) + YC_u( i ) )
                END DO
              END DO
              DO i = dims%c_l_end + 1, dims%c_u_end
                DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  RHS( j ) = RHS( j ) + A_val( l ) * YC_u( i )
                END DO
              END DO
            ELSE
              DO im = 1, m_active
                i = C_active( im )
                IF ( i < 0 ) THEN
                  DO l = A_ptr( - i ), A_ptr( - i + 1 ) - 1
                    j = A_col( l )
                    RHS( j ) = RHS( j ) + A_val( l ) * YC_l( - i )
                  END DO
                ELSE
                  DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( l )
                    RHS( j ) = RHS( j ) + A_val( l ) * YC_u( i )
                  END DO
                END IF
              END DO
            END IF

!  components from the simple bounds

            DO i = 1, n_active
              j = X_active( i )
              ii = n + ACTIVE_status( m + ABS( j ) )
              IF ( j < 0 ) THEN
                RHS( - j ) = RHS( - j ) + ZC_l( - j )
                RHS( ii ) = X_l( - j )
              ELSE
                RHS( j ) = RHS( j ) + ZC_u( j )
                RHS( ii ) = X_u( j )
              END IF
            END DO

!  now solve (S)

!write(6,*) ' mat_m ', SCU_mat%m, MAXVAL( ABS( RHS( : SCU_mat%n + SCU_mat%m )))

            inform%scu_status = 1
            DO
!write(6,*) 'scu_stat ', inform%scu_status
!write(6,*) 'm_ref, m_sbls ', m_ref, m_sbls, SCU_mat%n, n + m_ref
              CALL SCU_solve( SCU_mat, SCU_data, RHS, SOL,                     &
                              VECTOR( : SCU_mat%n ), inform%scu_status )
              IF ( inform%scu_status <= 0 ) EXIT
!WRITE(6, "( /, A, ' RHS = ', ES12.4 )" ) &
!  prefix, MAXVAL( ABS(  VECTOR( : SCU_mat%n ) ) )
              growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) )
              CALL SBLS_solve( n, m_ref, A_sbls, C_sbls, SBLS_data,            &
                               SBLS_control, inform%SBLS_inform,               &
                               VECTOR( : SCU_mat%n ) )
!WRITE(6, "( /, A, ' sol = ', ES12.4 )" ) &
!  prefix, MAXVAL( ABS(  VECTOR( : SCU_mat%n ) ) )
              IF ( inform%SBLS_inform%status < 0 ) THEN
                IF ( printi ) WRITE( out, "( ' SBLS_solve, status = ', I0 )" ) &
                  inform%SBLS_inform%status
                inform%status = inform%SBLS_inform%status
                GO TO 900
              END IF
              growth = MAXVAL( ABS( VECTOR( : SCU_mat%n ) ) ) / growth
              IF ( growth > control%max_growth ) THEN
                refactor = .TRUE. ; GO TO 410
              END IF
            END DO

!  untangle the delta v_F_k component by copying it into RHS (and then back)

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
            SOL( n + 1 : n + m_sbls ) = RHS( n + 1 : n + m_sbls )

!  regenerate the b_F_k component of RHS

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
!           IF ( .TRUE. ) THEN
              RES( : n + m_sbls ) = - RHS( : n + m_sbls )
              DO i = 1, n
                DO l = H_sbls%ptr( i ), H_sbls%ptr( i + 1 ) - 1
                  j = H_sbls%col( l )
                  RES( j ) = RES( j ) + H_sbls%val( l ) * SOL( i )
                  IF ( i /= j ) RES( i ) = RES( i ) + H_sbls%val( l ) * SOL( j )
                END DO
              END DO
              DO l = 1, A_sbls%ne
                i = n + A_sbls%ROW( l ) ; j = A_sbls%COL( l )
                RES( j ) = RES( j ) - A_sbls%VAL( l ) * SOL( i )
                RES( i ) = RES( i ) + A_sbls%VAL( l ) * SOL( j )
              END DO
              WRITE( out, "( A, ' Fredholm residual ', ES12.4 )" )             &
                prefix, MAXVAL( ABS( RES( : n + m_sbls ) ) )
            END IF
          END IF

!  calculate the slope = - dy^T A H^^-1 A^T dy ... first store the vector
!  A^T dy in vt ...

!write(6,*) ' dolid ', dolid
!         IF ( printd .AND. .NOT. dolid ) THEN
          IF ( .TRUE. .AND. .NOT. dolid ) THEN
            VT( : n ) = zero
            DO l = 1, A_sbls%ne
              i = A_sbls%row( l ) ; j = A_sbls%col( l )
              VT( j ) = VT( j ) + A_sbls%val( l ) * SOL( n + i )
            END DO

!  ... then solve L x = vt and overwrite vt with x ...

            CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
            IF ( identity_h ) THEN
!             VT( : n ) = VT( : n )
            ELSE IF ( scaled_identity_h ) THEN
              VT( : n ) = VT( : n ) / root_hd
            ELSE IF ( diagonal_h ) THEN
              VT( : n ) = VT( : n ) / SQRT( H_sbls%val( : n ) )
            ELSE
              CALL SLS_part_solve( 'S', VT( : n ), SLS_data,                   &
                                   SLS_control, inform%SLS_inform )
            END IF
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%solve = inform%time%solve + time_now - time_record
            inform%time%clock_solve =                                          &
              inform%time%clock_solve + clock_now - clock_record

! ... finally obtain slope = - ||vt||^2

            slope = - DOT_PRODUCT(  VT( : n ),  VT( : n ) )
            curv = - slope
!write(6,*) ' slope ', slope
          END IF
!write(6,"( 'sol_y ', / ( 5ES12.4 ) )" ) SOL( n + 1 : n + m_sbls )

!  use an iterative method to find the subspace
!  . . . . . . . . . . . . . . . . . . . . . . .

        ELSE
          IF ( control%subspace_alternate )                                    &
             subspace_direct = .NOT. subspace_direct

!  compute the dual residual r_k = g - J_F_k^T v_k  and store in SOL

          IF ( composite_g ) THEN
            SOL( : n ) = GC( : n )
          ELSE
            SOL( : n ) = G( : n )
          END IF

!  components from the general constraints

          IF ( penalty_objective ) THEN
            DO i = 1, dims%c_u_start - 1
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                SOL( j ) = SOL( j ) - A_val( l ) * YC_l( i )
              END DO
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                SOL( j ) = SOL( j ) - A_val( l ) * ( YC_l( i ) + YC_u( i ) )
              END DO
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                SOL( j ) = SOL( j ) - A_val( l ) * YC_u( i )
              END DO
            END DO
          ELSE
            DO im = 1, m_active
              i = C_active( im )
              IF ( i < 0 ) THEN
                DO l = A_ptr( - i ), A_ptr( - i + 1 ) - 1
                  j = A_col( l )
                  SOL( j ) = SOL( j ) - A_val( l ) * YC_l( - i )
                END DO
              ELSE
                DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  SOL( j ) = SOL( j ) - A_val( l ) * YC_u( i )
                END DO
              END IF
            END DO
          END IF

!  components from the simple bounds

          DO i = 1, n_active
            j = X_active( i )
            IF ( j < 0 ) THEN
              SOL( - j ) = SOL( - j ) - ZC_l( - j )
            ELSE
              SOL( j ) = SOL( j ) - ZC_u( j )
            END IF
          END DO

!  find the primal variables by solving H xc_k = r_k, overwriting xc_k in SOL

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          IF ( identity_h ) THEN
!           SOL( : n ) = SOL( : n )
          ELSE IF ( scaled_identity_h ) THEN
            SOL( : n ) = SOL( : n ) / h_scale( 1 )
          ELSE IF ( diagonal_h ) THEN
            SOL( : n ) = SOL( : n ) / H_sbls%val( : n )
          ELSE
            CALL SLS_solve( H_sbls, SOL( : n ), SLS_data,                      &
                            SLS_control, inform%SLS_inform )
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

!  record the dual gradient gc^d_k = J_F_k xc_k + b_F_k of q_k^d(y_F) at vc_k

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

!  now add the term J_F_k xc_k

          DO l = 1, A_sbls%ne
            i = A_sbls%row( l ) ; j = A_sbls%col( l )
            GV( i ) = GV( i ) + A_sbls%val( l ) * SOL( j )
          END DO

!  make a copy of the dual gradient

          GV( : m_sbls ) = - GV( : m_sbls )
          DV( : m_sbls ) = GV( : m_sbls )

          GLTR_control = control%GLTR_control
!         GLTR_control%print_level = 1
          GLTR_control%f_0 = qc
          IF ( no_change > 0 .AND. control%explore_optimal_subspace == 1 ) THEN
            GLTR_control%itmax = - 1
            IF ( printw ) WRITE( out,                                          &
              "( A, ' more thorough exploration of subspace' )" ) prefix
          END IF
          inform%GLTR_inform%status = 1

!  iteration to find the minimizer

          DO
            CALL GLTR_solve( m_sbls, big_radius, qt, SOL( n + 1 : n + m_sbls ),&
                             GV, PV, GLTR_data, GLTR_control,                  &
                             inform%GLTR_inform )

!  branch as a result of inform%status

            SELECT CASE( inform%GLTR_inform%status )

!  form the preconditioned gradient

!           CASE( 2, 6 )

!  form the matrix-vector product

            CASE ( 3, 7 )

!  compute pv -> A H^^-1 A^T pv ... first store the vector A^T pv in vt ...

              VT( : n ) = zero
              DO l = 1, A_sbls%ne
                i = A_sbls%row( l ) ; j = A_sbls%col( l )
                VT( j ) = VT( j ) + A_sbls%val( l ) * PV( i )
              END DO

!  ... then solve H x = vt and overwrite vt with x ...

              CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
              IF ( identity_h ) THEN
!               VT( : n ) = VT( : n )
              ELSE IF ( scaled_identity_h ) THEN
                VT( : n ) = VT( : n ) / h_scale( 1 )
              ELSE IF ( diagonal_h ) THEN
                VT( : n ) = VT( : n ) / H_sbls%val( : n )
              ELSE
                CALL SLS_solve( H_sbls, VT( : n ), SLS_data,                   &
                                SLS_control, inform%SLS_inform )
              END IF
              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_solve + clock_now - clock_record

!  ... and finally compute pv = A vt

              PV( : m_sbls ) = zero
              DO l = 1, A_sbls%ne
                i = A_sbls%row( l ) ; j = A_sbls%col( l )
                PV( i ) = PV( i ) + A_sbls%val( l ) * VT( j )
              END DO

!  restart

            CASE ( 5 )
               GV( : m_sbls ) = DV( : m_sbls )

!  successful return

            CASE ( - 30, 0 )
               EXIT

!  error returns

            CASE DEFAULT
              IF ( printt ) WRITE( out, "( A, ' GLTR_solve exit status = ',    &
             &  I0 ) " ) prefix, inform%GLTR_inform%status
              EXIT
            END SELECT
          END DO

!  scale the setp if it is a direction of linear infinite descent

          dolid = inform%GLTR_inform%iter_pass2 > 0
          IF ( dolid .AND. control%exact_arc_search ) THEN
            norm_pv = MAXVAL( ABS( SOL( n + 1 : n + m_sbls ) ) )
            IF ( norm_pv > zero )                                              &
              SOL( n + 1 : n + m_sbls ) = SOL( n + 1 : n + m_sbls ) / norm_pv
          END IF

!  compute the slope and curvature

          IF ( printw .AND. GLTR_control%itmax == - 1 ) WRITE( out,            &
             "( /, A, 1X, I0, ' GLTR iterations required' )" )                 &
               prefix, inform%GLTR_inform%iter
          inform%cg_iter = inform%cg_iter + inform%GLTR_inform%iter
          IF ( dolid ) skip = 'D'
          IF ( .TRUE. ) THEN
!         IF ( printd ) THEN
            slope = DOT_PRODUCT( DV( : m_sbls ), SOL( n + 1 : n + m_sbls ) )
            IF ( dolid ) THEN
              curv = zero
            ELSE
              curv = - slope
            END IF
!           write(6,*) ' slope, curv = ', slope, curv
          END IF

!         i = print_level
!         i = 4
!         IF ( scaled_identity_h ) THEN
!           CALL DQP_CG( n, m_sbls, diagonal_h, scaled_identity_h, identity_h, &
!                        H_sbls, GV, qc, A_sbls, out, i,                       &
!                        prefix, SOL( n + 1 : n + m_sbls ), qt,                &
!                        arc_search_iter, inform%status, V0, DV,               &
!                        H_s PV, VT, control,                                  &
!                        H_diag = h_scale( 1 ) ) )
!         ELSE IF ( diagonal_h ) THEN
!           CALL DQP_CG( n, m_sbls, diagonal_h, scaled_identity_h, identity_h, &
!                        H_sbls, GV, qc, A_sbls, out, i,                       &
!                        prefix, SOL( n + 1 : n + m_sbls ), qt,                &
!                        arc_search_iter, inform%status, V0, DV,               &
!                        H_s PV, VT, control,                                  &
!                        H_diag = H_sbls%val )
!         ELSE
!           CALL DQP_CG( n, m_sbls, diagonal_h, scaled_identity_h, identity_h, &
!                        H_sbls, GV, qc, A_sbls, out, i,                       &
!                        prefix, SOL( n + 1 : n + m_sbls ), qt,                &
!                        arc_search_iter, inform%status, V0, DV,               &
!                        H_s PV, VT, control,                                  &
!                        SLS_data = SLS_data,                                  &
!                        SLS_control = SLS_control,                            &
!                        SLS_inform = inform%SLS_inform )
!         END IF

        END IF

!  4b. step along search direction using either an arc or line search
!  ..................................................................

!  4b(i). perform an arc search
!  ----------------------------

!  find v_{k+1} = P_D[ v^C_k + alpha_k delta v_k ], where
!  alpha_k = arg min q^D( P_D[ v^C_k + alpha delta v_k ] )

        subspace_direct = subspace_direct_save
        IF ( control%subspace_arc_search ) THEN

!  record the initial point

          V0( ce_start : yl_end ) = YC_l( 1 : dims%c_l_end )
          V0( yu_start : yu_end ) = YC_u( dims%c_u_start : dims%c_u_end )
          V0( zl_start : zl_end ) = ZC_l( dims%x_free + 1 : dims%x_l_end )
          V0( zu_start : zu_end ) = ZC_u( dims%x_u_start : n )

!  record the dual objective value

          q0 = qc

!  compute the search direction d_k

          PV( : nv ) = zero

!  components from the general constraints

          l_start = ce_start - 1
          u_start = yu_start - dims%c_u_start
          l = n
          DO j = 1, m_active
            l = l + 1 ; i = C_active( j )
            IF ( i < 0 ) THEN
              PV( l_start - i ) = SOL( l )
            ELSE
              PV( u_start + i ) = SOL( l )
            END IF
          END DO

!  components from the simple bounds

          l_start = zl_start - dims%x_free - 1
          u_start = zu_start - dims%x_u_start
          DO i = 1, n_active
            l = l + 1 ; j = X_active( i )
            IF ( j < 0 ) THEN
              PV( l_start - j ) = SOL( l )
            ELSE
              PV( u_start + j ) = SOL( l )
            END IF
          END DO

!  compute the dual residual r_k = J^T vc_k - g and store in U

          IF ( composite_g ) THEN
            RES( : n ) = - GC( : n )
          ELSE
            RES( : n ) = - G( : n )
          END IF
          DO i = 1, dims%c_u_start - 1
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              RES( j ) = RES( j ) + A_val( l ) * YC_l( i )
            END DO
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              RES( j ) = RES( j ) + A_val( l ) * ( YC_l( i ) + YC_u( i ) )
            END DO
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              RES( j ) = RES( j ) + A_val( l ) * YC_u( i )
            END DO
          END DO
          DO j = dims%x_free + 1, dims%x_u_start - 1
            RES( j ) = RES( j ) + ZC_l( j )
          END DO
          DO j = dims%x_u_start, dims%x_l_end
            RES( j ) = RES( j ) + ZC_l( j ) + ZC_u( j )
          END DO
          DO j = dims%x_l_end + 1, n
            RES( j ) = RES( j ) + ZC_u( j )
          END DO

!  find the primal variables by solving H x_k = r_k, overwriting x_k in RES

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          IF ( identity_h ) THEN
!           RES( : n ) = RES( : n )
          ELSE IF ( scaled_identity_h ) THEN
            RES( : n ) = RES( : n ) / h_scale( 1 )
          ELSE IF ( diagonal_h ) THEN
            RES( : n ) = RES( : n ) / H_sbls%val( : n )
          ELSE
            CALL SLS_solve( H_sbls, RES, SLS_data,                             &
                            SLS_control, inform%SLS_inform )
          END IF
!write(6,"(A, /, (5ES16.8))" ) ' x ', RES( : n )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

!  record the dual gradient g^d_k = J x_k - b of q^d(v) at vc_k in GV

          j = ce_start
          DO i = 1, dims%c_l_end
            val = - C_l( i )
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              val = val + A_val( l ) * RES( A_col( l ) )
            END DO
            GV( j ) = val
            j = j + 1
          END DO
          j = yu_start
          DO i = dims%c_u_start, dims%c_u_end
            val = - C_u( i )
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              val = val + A_val( l ) * RES( A_col( l ) )
            END DO
            GV( j ) = val
            j = j + 1
          END DO
          i = zl_start
          DO j = dims%x_free + 1, dims%x_l_end
            GV( i ) = RES( j ) - X_l( j )
            i = i + 1
          END DO
          i = zu_start
          DO j = dims%x_u_start, n
            GV( i ) = RES( j ) - X_u( j )
            i = i + 1
          END DO

!write(6,"(A, /, (5ES16.8))" ) ' pv ', PV
!write(6,"(A, /, (5ES16.8))" ) ' gv ', GV

          slope = DOT_PRODUCT( PV( : nv ), GV( : nv ) )
!write(6,*) ' slope ', slope
          IF ( slope > zero ) THEN
            IF ( printp )                                                      &
              WRITE( out, "( A, ' flip sign of search direction ' )" ) prefix
            slope = - slope
            PV( : nv ) = - PV( : nv )
          END IF
!write(6,"(A, /, (5ES16.8))" ) ' v0 ', V0
!write(6,"(A, /, (5ES16.8))" ) ' pv ', PV
!write(6,"(A, /, (5ES16.8))" ) ' v0+pv ', V0+PV

!  scale the search direction

          norm_pv = MAXVAL( ABS( PV( : nv ) ) )
          IF ( norm_pv > zero ) PV( : nv ) = PV( : nv ) / norm_pv

!write(6,*) ' p^T g ', slope
!write(6,*) ' gy_l ', GY_l( 1 : dims%c_l_end )
!write(6,*) ' gy_u ', GY_u( dims%c_u_start : dims%c_u_end )
!write(6,*) ' gz_l ', GZ_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) ' gz_u ', GZ_u( dims%x_u_start : n )
!write(6,*) ' g ', GV( : nv )
!write(6,*) ' p ', PV( : nv )
!write(6,*) ' bnd_l ', V_bnd( : nv, 1 )
!write(6,*) ' bnd_u ', V_bnd( : nv, 2 )
!write(6,*) ' q0 ', q0
!write(6,*) ' V0 ', V0

!  set the initial variable status

          V_status( : nv ) = 0

          ip = print_level
!         ip = 4
!         ip = 11
!write(6,"( ' V0 ', /, ( 5ES16.8 ) )" ) V0( ce_start : zu_end )
!write(6,"( ' PV ', /, ( 5ES16.8 ) )" ) PV( : nv )

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          t_solve = inform%time%solve ; c_solve = inform%time%clock_solve
          IF ( control%exact_arc_search ) THEN
            IF ( identity_h ) THEN
              CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV,             &
                       V_bnd, q0, A_ptr, A_col, A_val, V_status,               &
                       start_ce, start_yl, start_yu, start_zl, start_zu,       &
                       ce_end, yl_end, yu_end, zl_end, zu_end, step_max,       &
                       feas_tol, qt, NZ_p, IUSED,                              &
                       INDEX_r, INDEX_w, out, ip, prefix,                      &
                       BREAK_points, control%infinity,                         &
                       control%arc_search_maxit,                               &
                       arc_search_iter, alpha_subspace,                        &
                       DV, RHS, SOL, RES, H_s, HPV,                            &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status )
            ELSE IF ( scaled_identity_h ) THEN
              CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV,             &
                       V_bnd, q0, A_ptr, A_col, A_val, V_status,               &
                       start_ce, start_yl, start_yu, start_zl, start_zu,       &
                       ce_end, yl_end, yu_end, zl_end, zu_end, step_max,       &
                       feas_tol, qt, NZ_p, IUSED,                              &
                       INDEX_r, INDEX_w, out, ip, prefix,                      &
                       BREAK_points, control%infinity,                         &
                       control%arc_search_maxit,                               &
                       arc_search_iter, alpha_subspace,                        &
                       DV, RHS, SOL, RES, H_s, HPV,                            &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            ELSE IF ( diagonal_h ) THEN
              CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV,             &
                       V_bnd, q0, A_ptr, A_col, A_val, V_status,               &
                       start_ce, start_yl, start_yu, start_zl, start_zu,       &
                       ce_end, yl_end, yu_end, zl_end, zu_end, step_max,       &
                       feas_tol, qt, NZ_p, IUSED,                              &
                       INDEX_r, INDEX_w, out, ip, prefix,                      &
                       BREAK_points, control%infinity,                         &
                       control%arc_search_maxit,                               &
                       arc_search_iter, alpha_subspace,                        &
                       DV, RHS, SOL, RES, H_s, HPV,                            &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls )
            ELSE
              CALL DQP_exact_arc_search( nv, n, m, V0, PV, VT, GV,             &
                       V_bnd, q0, A_ptr, A_col, A_val, V_status,               &
                       start_ce, start_yl, start_yu, start_zl, start_zu,       &
                       ce_end, yl_end, yu_end, zl_end, zu_end, step_max,       &
                       feas_tol, qt, NZ_p, IUSED,                              &
                       INDEX_r, INDEX_w, out, ip, prefix,                      &
                       BREAK_points, control%infinity,                         &
                       control%arc_search_maxit,                               &
                       arc_search_iter, alpha_subspace,                        &
                       DV, RHS, SOL, RES, H_s, HPV,                            &
                       diagonal_h, scaled_identity_h, identity_h,              &
                       inform%time%solve, inform%time%clock_solve,             &
                       inform%status, HESSIAN = H_sbls,                        &
                       SLS_data = SLS_data, SLS_control = SLS_control,         &
                       SLS_inform = inform%SLS_inform )
            END IF
          ELSE
            IF ( identity_h .OR. scaled_identity_h ) THEN
              IF ( composite_g ) THEN
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, GC, q0, YC_l, YC_u, ZC_l, ZC_u,       &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         HESSIAN = H_sbls )
              ELSE
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, G, q0, YC_l, YC_u, ZC_l, ZC_u,        &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         HESSIAN = H_sbls )
              END IF
            ELSE IF ( diagonal_h ) THEN
              IF ( composite_g ) THEN
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, GC, q0, YC_l, YC_u, ZC_l, ZC_u,       &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         HESSIAN = H_sbls )
              ELSE
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, G, q0, YC_l, YC_u, ZC_l, ZC_u,        &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         HESSIAN = H_sbls )
              END IF
            ELSE
              IF ( composite_g ) THEN
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, GC, q0, YC_l, YC_u, ZC_l, ZC_u,       &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         SLS_data = SLS_data,                                  &
                         SLS_control = SLS_control,                            &
                         SLS_inform = inform%SLS_inform )
              ELSE
                CALL DQP_inexact_arc_search( dims, nv, n, m, PV, VT, qt,       &
                         alpha_subspace, G, q0, YC_l, YC_u, ZC_l, ZC_u,        &
                         C_l( 1 : dims%c_l_end ),                              &
                         C_u( dims%c_u_start : dims%c_u_end ),                 &
                         X_l( dims%x_free + 1 : dims%x_l_end ),                &
                         X_u( dims%x_u_start : n ),                            &
                         ce_start, ce_end, yl_start, yl_end, yu_start, yu_end, &
                         zl_start, zl_end, zu_start, zu_end,                   &
                         A_ptr, A_col, A_val, step_max, feas_tol,              &
                         V_status, arc_search_iter, out, ip, prefix,           &
                         DV, RHS, H_s,                                         &
                         diagonal_h, scaled_identity_h, identity_h,            &
                         inform%time%solve, inform%time%clock_solve,           &
                         inform%status,                                        &
                         SLS_data = SLS_data,                                  &
                         SLS_control = SLS_control,                            &
                         SLS_inform = inform%SLS_inform )
              END IF
            END IF
          END IF
!write(6,"( ' VT ', /, ( 5ES16.8 ) )" ) VT( ce_start : zu_end )
!write(6,"( ' V_status ', /, ( 5I5 ) )" ) V_status( : nv )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%search = inform%time%search + time_now - time_record     &
            + t_solve - inform%time%solve
          inform%time%clock_search = inform%time%clock_search                  &
            + clock_now - clock_record + c_solve - inform%time%clock_solve

!  record the new point

          Y_l( 1 : dims%c_l_end ) = VT( ce_start : yl_end )
          Y_u( dims%c_u_start : dims%c_u_end ) = VT( yu_start : yu_end )
          Z_l( dims%x_free + 1 : dims%x_l_end ) = VT( zl_start : zl_end )
          Z_u( dims%x_u_start : n ) = VT( zu_start : zu_end )

          IF (  inform%status == GALAHAD_error_primal_infeasible ) GO TO 700

!  record the status of the general constraints and simple bounds

!write(6,*) ' V_status ', V_status

          C_status = 0 ; X_status = 0
          m_active = 0 ;  n_active = 0

          DO ii = ce_start, yl_end
            i = ii
            IF ( V_status( ii ) == 0 ) THEN
              m_active = m_active + 1 ; C_status( i ) = 1
            END IF
          END DO

          DO ii = yu_start, yu_end
            i = ii - yu_start + dims%c_u_start
            IF ( V_status( ii ) == 0 ) THEN
              m_active = m_active + 1
              C_status( i ) = C_status( i ) + 2
            END IF
          END DO

          DO ii = zl_start, zl_end
            i = ii - zl_start + dims%x_free + 1
            IF ( V_status( ii ) == 0 ) THEN
              n_active = n_active + 1 ; X_status( i ) = 1
            END IF
          END DO

          DO ii = zu_start, zu_end
            i = ii - zu_start + dims%x_u_start
            IF ( V_status( ii ) == 0 ) THEN
              n_active = n_active + 1
              X_status( i ) = X_status( i ) + 2
            END IF
          END DO

!write(6,*) 'y_l', Y_l( 1 : dims%c_l_end )
!write(6,*) 'y_u', Y_u( dims%c_u_start : dims%c_u_end )
!write(6,*) 'z_l', Z_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) 'z_u', Z_u( dims%x_u_start : n )

          m_subspace = n_active + m_active
          change_subspace = COUNT( X_status( : n ) /= X_status_old( : n ) )    &
                              + COUNT( C_status( : m ) /= C_status_old( : m ) )

          IF ( printw ) WRITE( out, "( A, 1X, A, I0, A, I0, A, I0 )" ) prefix, &
            ' n active = ', n_active, ', m active = ', m_active,               &
            ', total subspace active = ', m_subspace

!  4b(ii). perform a line search
!  -----------------------------

!  find v_{k+1} = v^C_k + alpha_k delta v_k, where alpha_k is the largest
!  alpha in [0,1] for which v^C_k + alpha delta v_k is in D (see notation)

        ELSE
          alpha_subspace = one

!  components from the general constraints

          l = n
          DO j = 1, m_active
            l = l + 1 ; i = C_active( j ) ; sl = SOL( l )
            IF ( ABS( i ) <= dims%c_equality ) CYCLE
            IF ( i < 0 ) THEN
              IF ( sl < zero )                                                 &
                alpha_subspace = MIN( alpha_subspace, - YC_l( - i ) / sl )
            ELSE
              IF ( sl > zero )                                                 &
                alpha_subspace = MIN( alpha_subspace, - YC_u( i ) / sl )
            END IF
          END DO

!  components from the simple bounds

          DO i = 1, n_active
            l = l + 1 ; j = X_active( i ) ; sl = SOL( l )
            IF ( j < 0 ) THEN
              IF ( sl < zero )                                                 &
                alpha_subspace = MIN( alpha_subspace, - ZC_l( - j ) / sl )
            ELSE
              IF ( sl > zero )                                                 &
                alpha_subspace = MIN( alpha_subspace, - ZC_u( j ) / sl )
            END IF
          END DO

          qt = qc + alpha_subspace * ( slope + half * alpha_subspace * curv )
          IF ( printp ) WRITE( out, "( A, ' q after line search = ', ES12.4,   &
         & ' with step =', ES12.4 )" ) prefix,  qt, alpha_subspace

!  move to the new point

          Y_l( 1 : dims%c_l_end ) = YC_l( 1 : dims%c_l_end )
          Y_u( dims%c_u_start : dims%c_u_end )                                 &
            = YC_u( dims%c_u_start : dims%c_u_end )
          Z_l( dims%x_free + 1 : dims%x_l_end )                                &
            = ZC_l( dims%x_free + 1 : dims%x_l_end )
          Z_u( dims%x_u_start : n ) = ZC_u( dims%x_u_start : n )

!  components from the general constraints

          l = n
          DO j = 1, m_active
            l = l + 1 ; i = C_active( j )
            IF ( i < 0 ) THEN
              Y_l( - i ) = Y_l( - i ) + alpha_subspace * SOL( l )
            ELSE
              Y_u( i ) = Y_u( i ) + alpha_subspace * SOL( l )
            END IF
          END DO

!  components from the simple bounds

         DO i = 1, n_active
            l = l + 1 ; j = X_active( i )
            IF ( j < 0 ) THEN
              Z_l( - j ) = Z_l( - j ) + alpha_subspace * SOL( l )
            ELSE
              Z_u( j ) = Z_u( j ) + alpha_subspace * SOL( l )
            END IF
          END DO

          IF ( alpha_subspace == one ) THEN
            change_subspace = change ; m_subspace = m_sbls
          ELSE
            change_subspace = change + 1 ; m_subspace = m_sbls + 1
          END IF
        END IF

!  if required compute the dual objective value. Record sol = J^T v_k - g

!       IF ( .TRUE. ) THEN
        IF ( printw ) THEN
!write(6,*) 'y_l', Y_l( 1 : dims%c_l_end )
!write(6,*) 'y_u', Y_u( dims%c_u_start : dims%c_u_end )
!write(6,*) 'z_l', Z_l( dims%x_free + 1 : dims%x_l_end )
!write(6,*) 'z_u', Z_u( dims%x_u_start : n )
          IF ( composite_g ) THEN
            SOL( : n ) = - GC( : n )
          ELSE
            SOL( : n ) = - G( : n )
          END IF
          DO i = 1, dims%c_u_start - 1
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * Y_l( i )
            END DO
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * ( Y_l( i ) + Y_u( i ) )
            END DO
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              SOL( j ) = SOL( j ) + A_val( l ) * Y_u( i )
            END DO
          END DO
          DO j = dims%x_free + 1, dims%x_u_start - 1
            SOL( j ) = SOL( j ) + Z_l( j )
          END DO
          DO j = dims%x_u_start, dims%x_l_end
            SOL( j ) = SOL( j ) + Z_l( j ) + Z_u( j )
          END DO
          DO j = dims%x_l_end + 1, n
            SOL( j ) = SOL( j ) + Z_u( j )
          END DO

!  find the primal variables by solving H x_k = r_k

          X( : n ) = SOL( : n )
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          IF ( identity_h ) THEN
!           X( : n ) = X( : n )
          ELSE IF ( scaled_identity_h ) THEN
            X( : n ) = X( : n ) / h_scale( 1 )
          ELSE IF ( diagonal_h ) THEN
            X( : n ) = X( : n ) / H_sbls%val( : n )
          ELSE
            CALL SLS_solve( H_sbls, X, SLS_data, SLS_control,                  &
                            inform%SLS_inform )
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

!  compute the dual objective function q_d(v) = 1/2 x_k^T r_k - <b,v> - f

          dual_f = half * DOT_PRODUCT( X( : n ), SOL( : n ) ) - f_all
          val = dual_f
          DO i = 1, dims%c_l_end
            dual_f = dual_f - C_l( i ) * Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            dual_f = dual_f - C_u( i ) * Y_u( i )
          END DO
          DO j = dims%x_free + 1, dims%x_l_end
            dual_f = dual_f - X_l( j ) * Z_l( j )
          END DO
          DO j = dims%x_u_start, n
            dual_f = dual_f - X_u( j ) * Z_u( j )
          END DO
          val = dual_f - val
          WRITE( out, "( A, ' dual obj after subspace step (computed,',        &
         &               ' recurred) =', 2ES14.6 )" ) prefix, dual_f, qt
        END IF

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Book keeping  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!       CALL SBLS_terminate( SBLS_data, SBLS_control,                          &
!                            inform%SBLS_inform )
!       CALL SBLS_tiny_terminate( SBLS_data, SBLS_control,                     &
!                                 inform%SBLS_inform )

!  ---------------------------------------------------------------------
!  ---------------------- End of Major Iteration -----------------------
!  ---------------------------------------------------------------------

      END DO

!  print details of the solution obtained

  600 CONTINUE

!  Compute the final objective function value

      IF ( Hessian_kind == 1 ) THEN
        inform%obj = f_all + half * SUM( X ** 2 )
      ELSE IF ( Hessian_kind == 2 ) THEN
        inform%obj = f_all + half * SUM( ( WEIGHT * X ) ** 2 )
      ELSE
        IF ( identity_h ) THEN
          curv = DOT_PRODUCT( X( : n ), X( : n ) )
        ELSE IF ( scaled_identity_h ) THEN
          curv = h_scale( 1 ) * DOT_PRODUCT( X( : n ), X( : n ) )
        ELSE
          curv = zero
          DO i = 1, n
            xi = X( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              IF ( i == j ) THEN
                curv = curv + xi * xi * H_val( l )
              ELSE
                curv = curv + two * xi * X( j ) * H_val( l )
              END IF
            END DO
          END DO
          IF ( printd .AND. n > 0 )                                            &
            WRITE( out, "( A, A6, /, ( 4( 2I5, ES10.2 ) ) )" ) prefix,         &
           &  ' h ', ( ( i, H_col( l ), H_val( l ), l = H_ptr( i ),            &
              H_ptr( i + 1 ) - 1 ), i = 1, n )
        END IF
        inform%obj = f_all + half * curv
      END IF

      IF ( composite_g ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( GC( : n ), X )
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
      END IF

!  Exit

 700  CONTINUE

!  Set the Lagrange multipliers

      DO i = 1, dims%c_u_start - 1
        Y( i ) = Y_l( i )
      END DO

      DO i = dims%c_u_start, dims%c_l_end
        IF ( ABS( Y_l( i ) ) <= ABS( Y_u( i ) ) ) THEN
          Y( i ) = Y_u( i )
        ELSE
          Y( i ) = Y_l( i )
        END IF
      END DO

      DO i = dims%c_l_end + 1, m
        Y( i ) = Y_u( i )
      END DO

!  Set the dual variables

      Z( : dims%x_free ) = zero
      DO i = dims%x_free + 1, dims%x_u_start - 1
        Z( i ) = Z_l( i )
      END DO

      DO i = dims%x_u_start, dims%x_l_end
        IF ( ABS( Z_l( i ) ) <= ABS( Z_u( i ) ) ) THEN
          Z( i ) = Z_u( i )
        ELSE
          Z( i ) = Z_l( i )
        END IF
      END DO

      DO i = dims%x_l_end + 1, n
        Z( i ) = Z_u( i )
      END DO

!  Compute the values of the constraints

      IF ( m > 0 ) THEN
        C( : m ) = zero
        CALL DQP_AX( m, C( : m ), m, a_ne, A_val, A_col, A_ptr, n, X, '+ ')
        inform%primal_infeasibility =                                          &
          MAX( zero, MAXVAL( ABS( C_l( : dims%c_equality ) -                   &
                                  C(: dims%c_equality ) ) ),                   &
                     MAXVAL( C_l(  dims%c_l_start : dims%c_l_end ) -           &
                             C(  dims%c_l_start : dims%c_l_end ) ),            &
                     MAXVAL( C( dims%c_u_start : dims%c_u_end ) -              &
                             C_u( dims%c_u_start : dims%c_u_end ) ) )
        primal_infeasibility = zero
        DO i = 1, dims%c_equality
          inform%obj = inform%obj + rho * ABS( C( i ) -  C_l( i ) )
          primal_infeasibility                                                 &
            = primal_infeasibility + ABS( C( i ) -  C_l( i ) )
        END DO
        DO i = dims%c_equality + 1, dims%c_l_end
          inform%obj = inform%obj + rho * MAX( zero, C_l( i ) - C( i ) )
          primal_infeasibility                                                 &
            = primal_infeasibility + MAX( zero, C_l( i ) - C( i ) )
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          inform%obj = inform%obj + rho * MAX( zero, C( i ) - C_u( i ) )
          primal_infeasibility                                                 &
            = primal_infeasibility + MAX( zero, C( i ) - C_u( i ) )
        END DO
      ELSE
        inform%primal_infeasibility = zero
      END IF

      IF ( printi ) THEN
        WRITE( out, "( /, A, '  Final objective function value is', ES22.14,   &
      &       /, A, '  Total number of iterations = ', I0,                     &
      &       /, A, '  Norm of primal infeasibility is', ES11.4,               &
      &       /, A, '  Norm of dual infeasibility is', ES11.4 )" )             &
          prefix, inform%obj, prefix, inform%iter,                             &
          prefix, inform%primal_infeasibility, prefix, dual_g_norm
      END IF

!  estimate the variable and constraint exit status

      IF ( stat_required ) THEN
        DO i = 1, n
          IF ( X_status( i ) == 1 .OR. X_status( i ) == 3 ) THEN
            X_stat( i ) = - 1
          ELSE IF ( X_status( i ) > 0 ) THEN
            X_stat( i ) = 1
          ELSE
            X_stat( i ) = 0
          END IF
        END DO

        DO i = 1, m
          IF ( C_status( i ) == - 1 .OR. C_status( i ) == 1 .OR.               &
               C_status( i ) == 3 ) THEN
            C_stat( i ) = - 1
          ELSE IF ( C_status( i ) > 0 ) THEN
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  print details of the optimal point if required ...

      IF ( printd ) THEN
!     IF ( .TRUE. ) THEN
        IF ( n > 0 ) THEN
          WRITE( out, "( /, A, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X,    &
       &       'z_l', 9X, 'z_u     stat')") prefix
          DO i = 1, dims%x_free
            WRITE( out, "( A, I6, ES12.4, 4( '      -     ' ), I5 )" ) prefix, &
              i, X( i ), X_status( i )
          END DO
          DO i = dims%x_free + 1, dims%x_l_start - 1
            WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,             &
           &  '      -     ', I5 )" ) prefix, i, X( i ), zero, Z_l( i ),       &
              X_status( i )
          END DO
          DO i = dims%x_l_start, dims%x_u_start - 1
            WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,             &
           &  '      -     ', I5 )" ) prefix, i, X( i ), X_l( i ),             &
              Z_l( i ), X_status( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            WRITE( out, "( A, I6, 5ES12.4, I5 )" )                             &
               prefix, i, X( i ), X_l( i ), X_u( i ), Z_l( i ),                &
               Z_u( i ), X_status( i )
          END DO
          DO i = dims%x_l_end + 1, dims%x_u_end
            WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,              &
           &  '      -     ', ES12.4, I5 )" ) prefix, i, X( i ), X_u( i ),     &
              Z_u( i ), X_status( i )
          END DO
          DO i = dims%x_u_end + 1, n
            WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,              &
           &  '      -     ',  ES12.4, I5 )" ) prefix, i, X( i ), zero,        &
              Z_u( i ), X_status( i )
          END DO
        END IF

!  ... and of the constraints

        IF ( m > 0 ) THEN
          WRITE( out, "( /, A, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X,     &
         &     'y_l', 9X, 'y_u     stat' )") prefix
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( A, I6, 4ES12.4, 12X, I5 )" ) prefix, i, C( i ),     &
              C_l( i ), C_u( i ), Y_l( i ), C_status( i )
          END DO
          DO i = dims%c_l_start, dims%c_u_start - 1
            WRITE( out,  "( A, I6, 2ES12.4, '      -     ', ES12.4,            &
           &  '      -     ', I5 )" ) prefix, i, C( i ), C_l( i ),             &
              Y_l( i ), C_status( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            WRITE( out, "( A, I6, 5ES12.4, I5 )" ) prefix,                     &
              i, C( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i ),               &
              C_status( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,              &
            & '      -     ', ES12.4, I5 )" ) prefix, i, C( i ), C_u( i ),     &
              Y_u( i ), C_status( i )
          END DO
        END IF
      END IF

!  count the number of active constraints/bounds

        IF ( printw )                                                          &
          WRITE( out, "( A, ' indicators: n_active/n, m_active/m ', 4I7 )" )   &
             prefix, COUNT( X_stat /= 0 ), n, COUNT( C_stat /= 0 ), m
       END IF

!  If necessary, print warning messages

  810 CONTINUE
      IF ( printi ) then
        SELECT CASE( inform%status )
        CASE( GALAHAD_error_restrictions  ) ; WRITE( out, "( /, A,             &
       & '  Warning - input paramters incorrect' )" ) prefix
        CASE( GALAHAD_error_bad_bounds ) ; WRITE( out, "( /, A,                &
       &  '  Warning - the constraints are inconsistent' )" ) prefix
        CASE( GALAHAD_error_primal_infeasible ) ; WRITE( out, "( /, A,         &
       &  '  Warning - the constraints appear to be inconsistent' )" ) prefix
        CASE( GALAHAD_error_factorization ) ; WRITE( out, "( /, A,             &
       &   '  Warning - factorization failure' )" ) prefix
        CASE( GALAHAD_error_ill_conditioned ) ; WRITE( out, "( /, A,           &
       &   '  Warning - no further progress possible' )"  ) prefix
        CASE( GALAHAD_error_tiny_step ) ; WRITE( out, "( /, A,                 &
       &   '  Warning - step too small to make progress,',                     &
       &   ' problem maybe infeasible' )" ) prefix
        CASE( GALAHAD_error_max_iterations ) ; WRITE( out, "( /, A,            &
       &   '  Warning - iteration bound exceeded' )" ) prefix
        CASE( GALAHAD_error_unbounded ) ; WRITE( out, "( /, A,                 &
       &   '  Warning - problem appears to be unbounded from below' )") prefix
        END SELECT

        IF ( subspace_direct .OR. control%subspace_alternate ) THEN
          WRITE( out, "( A, '  Direct subspace solver used' )" ) prefix
          IF ( inform%SBLS_inform%preconditioner /= 2 )                        &
            WRITE( out, "( A, 2X, I0, ' projected CG iterations taken ' )" )   &
            prefix, inform%SBLS_inform%iter_pcg
          IF ( inform%SBLS_inform%factorization == 0 .OR.                      &
               inform%SBLS_inform%factorization == 1 ) THEN
            WRITE( control%out, "( A, '  Schur-complement factorization is',   &
           &       ' used (pivot tol =', ES9.2, ')' )" ) prefix,               &
              SBLS_control%SLS_control%relative_pivot_tolerance
          ELSE
            WRITE( control%out, "( A, '  Augmented system factorization is',   &
           &       ' used (pivot tol =', ES9.2, ')' )" ) prefix,               &
              SBLS_control%SLS_control%relative_pivot_tolerance
          END IF
          WRITE( out, "( A, '  Linear system solver ', A,                      &
         &               ' (preconditioner = ', I0, ') is used' )" )           &
              prefix, TRIM( SBLS_control%symmetric_linear_solver ),            &
              inform%SBLS_inform%preconditioner
        END IF
        IF ( .NOT. subspace_direct .OR. control%subspace_alternate ) THEN
          WRITE( out, "( A, '  Iterative subspace solver used, ', A,           &
       &    ' iterations taken in total' )" ) prefix,                          &
          TRIM( STRING_integer_6( inform%cg_iter ) )
        END IF
        WRITE( out, "( A, '  Definite linear system solver ', A,               &
       &               ' (preconditioner = ', I0, ') is used' )" )             &
            prefix, TRIM( control%definite_linear_solver ),                    &
            inform%SBLS_inform%preconditioner
      END IF
      GO TO 990

!  error

  900 CONTINUE
      GO TO 990

!  allocation error

  910 CONTINUE
      inform%status = GALAHAD_error_allocate
      IF ( printe ) WRITE( control%error,                                      &
        "( A, ' ** Message from -DQP_solve-', /,  A,                           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status
      GO TO 990

!  inertia error

  920 CONTINUE
      inform%status = GALAHAD_error_inertia

!  record exit times

  990 CONTINUE
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving DQP_solve_main ' )" ) prefix
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      cpu_total = cpu_total + time_now - time_start
      clock_total = clock_total + clock_now - clock_start
      RETURN

!  Non-executable statements

!2000 FORMAT( /, A, ' Iter   p-feas  d-feas com-slk    obj   ',                &
!               '  step   target    time' )
 2000 FORMAT( /, A, 26X, ' <-   arcsearch   ->  <-    subsapce   ->',          &
              /, A, ' Iter    dual obj   p-feas  active   +/-   step ',        &
                ' active   +/-   step    time' )
 2020 FORMAT( A, I5, 1X, ES12.4, ES8.1, '       -     -    -         -',       &
              '     -    -  ', 0P, F8.2 )
 2030 FORMAT( A, I5, A1, ES12.4, ES8.1, 2( I8, I6, ES7.0 ), 0P, F8.2 )

!  End of DQP_solve_main

      END SUBROUTINE DQP_solve_main

!-*-*-*-*-*-*-   D Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE DQP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine DQP_initialize
!   control see Subroutine DQP_initialize
!   inform  see Subroutine DQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( DQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( DQP_control_type ), INTENT( IN ) :: control
      TYPE ( DQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: scu_status
      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by FDC

      CALL FDC_terminate( data%FDC_data, data%FDC_control,                     &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FDC_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      inform%status = inform%SLS_inform%status
      IF ( inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'DQP: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      inform%status = inform%SBLS_inform%status
      IF ( inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'DQP: data%SBLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate GLTR internal arrays

      CALL GLTR_terminate( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )
      IF ( inform%GLTR_inform%status /= GALAHAD_ok )                           &
        inform%status = inform%GLTR_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate SCU internal arrays

      CALL SCU_terminate( data%SCU_data, scu_status, inform%SCU_inform )
      IF ( scu_status /= GALAHAD_ok ) inform%status = GALAHAD_error_deallocate
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate QPP internal arrays

      CALL QPP_terminate( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated for the preprocessing stage

      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      IF ( data%QPP_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = 'DQP: data%QPP'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'dqp: data%H_sbls%ptr'
      CALL SPACE_dealloc_array( data%H_sbls%ptr,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%H_sbls%col'
      CALL SPACE_dealloc_array( data%H_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%A_sbls%row'
      CALL SPACE_dealloc_array( data%A_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%A_sbls%col'
      CALL SPACE_dealloc_array( data%A_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%A_sbls%val'
      CALL SPACE_dealloc_array( data%A_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_sbls%row'
      CALL SPACE_dealloc_array( data%C_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_sbls%col'
      CALL SPACE_dealloc_array( data%C_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_sbls%val'
      CALL SPACE_dealloc_array( data%C_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%X_status'
      CALL SPACE_dealloc_array( data%X_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_status'
      CALL SPACE_dealloc_array( data%C_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%V_status'
      CALL SPACE_dealloc_array( data%V_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%X_status_old'
      CALL SPACE_dealloc_array( data%X_status_old,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_status_old'
      CALL SPACE_dealloc_array( data%C_status_old,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%X_active'
      CALL SPACE_dealloc_array( data%X_active,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%C_active'
      CALL SPACE_dealloc_array( data%C_active,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%CHANGES'
      CALL SPACE_dealloc_array( data%CHANGES,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%ACTIVE_list'
      CALL SPACE_dealloc_array( data%ACTIVE_list,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%ACTIVE_status'
      CALL SPACE_dealloc_array( data%ACTIVE_status,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%NZ_p'
      CALL SPACE_dealloc_array( data%NZ_p,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%IUSED'
      CALL SPACE_dealloc_array( data%IUSED,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%INDEX_r'
      CALL SPACE_dealloc_array( data%INDEX_r,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%INDEX_w'
      CALL SPACE_dealloc_array( data%INDEX_w,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%BREAK_points'
      CALL SPACE_dealloc_array( data%BREAK_points,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%Y_l'
      CALL SPACE_dealloc_array( data%Y_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%Y_u'
      CALL SPACE_dealloc_array( data%Y_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%Z_l'
      CALL SPACE_dealloc_array( data%Z_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%Z_u'
      CALL SPACE_dealloc_array( data%Z_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%YC_l'
      CALL SPACE_dealloc_array( data%YC_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%YC_u'
      CALL SPACE_dealloc_array( data%YC_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%ZC_l'
      CALL SPACE_dealloc_array( data%ZC_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%ZC_u'
      CALL SPACE_dealloc_array( data%ZC_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%GY_l'
      CALL SPACE_dealloc_array( data%GY_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%GY_u'
      CALL SPACE_dealloc_array( data%GY_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%GZ_l'
      CALL SPACE_dealloc_array( data%GZ_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%GZ_u'
      CALL SPACE_dealloc_array( data%GZ_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%V0'
      CALL SPACE_dealloc_array( data%V0,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%VT'
      CALL SPACE_dealloc_array( data%VT,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%G'
      CALL SPACE_dealloc_array( data%G,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%GV'
      CALL SPACE_dealloc_array( data%GV,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%PV'
      CALL SPACE_dealloc_array( data%PV,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%HPV'
      CALL SPACE_dealloc_array( data%HPV,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%DV'
      CALL SPACE_dealloc_array( data%DV,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%H_s'
      CALL SPACE_dealloc_array( data%H_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: data%V_bnd'
      CALL SPACE_dealloc_array( data%V_bnd,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine DQP_terminate

      END SUBROUTINE DQP_terminate

!-*-*-*-  D Q P _ E X A C T _ A R C _ S E A R C H   S U B R O U T I N E  -*-*-*-

      SUBROUTINE DQP_exact_arc_search( nv, n, m, V_0, P, V_t, G, BND, f,       &
                   A_ptr, A_col, A_val, V_status, start_ce, start_yl,          &
                   start_yu, start_zl, start_zu, ce_end, yl_end, yu_end,       &
                   zl_end, zu_end, t_max, feas_tol, q_t, NZ_p, IUSED,          &
                   INDEX_r, INDEX_w, out, print_level, prefix, BREAK_points,   &
                   bnd_inf, max_iter, iter, t_arc_minimizer, D, R, W, U, H,    &
                   HP, diagonal_h, scaled_identity_h, identity_h, solve,       &
                   clock_solve, status, HESSIAN,                               &
                   SLS_data, SLS_control, SLS_inform )

!  Find the arc minimizer in the direction P from V_0 for a given quadratic
!  function within a box shaped region

!  If we define the arc v(t) = projection of v_0 + t*p into the box region

!     BND(*,1) <= v(*) <= BND(*,2),

!  the arc minimizer is the first local minimizer of the quadratic function

!     1/2 (v-v_0)^T H_d (v-v_0) + g_d^T (v-v_0) + f

!  for points lying on v(t), with 0 <= t <= t_max. Here

!    H_d = J H^{-1} J^T and g_d is given

!  The value of the array V_status gives the status of the variables

!  IF V_status( I ) = 0, the I-th variable is free
!  IF V_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF V_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF V_status( I ) = 3, the I-th variable is permanently fixed
!  IF V_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first n_free entries
!  of the array NZ_p

!  At the initial point, variables within feas_tol of their bounds and
!  for which the search direction P points out of the box will be fixed

!  Based on CAUCHY_get_exact_gcp from LANCELOT B

!  ------------------------- dummy arguments (to be updated!) -----------------

!  nv     (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  V_0    (REAL array of length at least nv) the point V_0 from which the search
!          arc commences. ** this variable is not altered by the subroutine
!  P      (REAL array of length at least n) contains the values of the
!          components of the vector P. On entry, P must contain the initial
!          direction of the search arc
!  V_t    (REAL array of length at least nv) the current estimate of the
!         arc minimizer
!  G      (REAL array of length at least nv) the coefficients of
!          the linear term g_d in the quadratic function
!          ** this variable is not altered by the subroutine
!  BND    (two dimensional REAL array with leading dimension nv and second
!          dimension 2) the lower (BND(*,1)) and upper (BND(*,2)) bounds on
!          the variables. ** this variable is not altered by the subroutine
!  V_status (INTEGER array of length at least nv) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          V_status should be set as follows:
!          If V_status( i ) = 0, the i-th variable is free
!          If V_status( i ) = 1, the i-th variable is on its lower bound
!          If V_status( i ) = 2, the i-th variable is on its upper bound
!          If V_status( i ) = 3, 4, the i-th variable is fixed at V_t(i)
!  f      (REAL) the value of the quadratic at V_0, see above.
!          ** this variable is not altered by the subroutine
!  feas_tol (REAL) a tolerance on feasibility of V_0, see above.
!          ** this variable is not altered by the subroutine.
!  q_t     (REAL) the value of the piecewise quadratic function at the current
!          estimate of the arc minimizer
!  NZ_p   (INTEGER array of length at least nv) workspace
!  INDEX_w (INTEGER array of length at least nv) workspace
!  out    (INTEGER) the fortran output channel number to be used
!  max_iter  (INTEGER) the maximum number of iterations allowed
!  iter   (INTEGER) the number of iterations performed
!  bnf_inf (REAL) any BND larger than bnd_inf in modulus is infinite
!  t_arc_minimizer (REAL) the minimizing value of t
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  BREAK_points (REAL) workspace that must be preserved between calls

!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN ):: nv, n, m, max_iter, out, print_level
      INTEGER, INTENT( INOUT ):: iter
      INTEGER, INTENT( IN ) :: start_ce, start_yl, start_yu, start_zl, start_zu
      INTEGER, INTENT( IN ) :: ce_end, yl_end, yu_end, zl_end, zu_end
      INTEGER, INTENT( INOUT ):: status
      REAL ( KIND = wp ), INTENT( IN ):: t_max, feas_tol, bnd_inf
      REAL ( KIND = wp ), INTENT( INOUT ):: f, q_t, t_arc_minimizer
      REAL ( KIND = wp ), INTENT( INOUT ):: solve, clock_solve
      LOGICAL, INTENT( IN ) :: diagonal_h, scaled_identity_h, identity_h
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, DIMENSION( nv ), INTENT( INOUT ) :: V_status, NZ_p
      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: IUSED, INDEX_r, INDEX_w
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nv, 2 ) :: BND
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nv ) :: V_0, G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: V_t
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: R, W, U, H
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: D, P, HP
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: BREAK_points
      TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: HESSIAN
      TYPE ( SLS_data_type ), OPTIONAL, INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), OPTIONAL, INTENT( IN ) :: SLS_control
      TYPE ( SLS_inform_type ), OPTIONAL, INTENT( INOUT ) :: SLS_inform

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, V_0, P, G, t_max, BND, f, feas_tol, out, print_level

!  If the i-th variable is required to be fixed at its initial value, V_0(i),
!   V_status(i) must be set to 3 or 4

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, ii, j, k, l, ibreak, insort, nvar_l, nvar_u
      INTEGER :: n_free, n_freed, n_break, n_zero, n_fix, nnz_r, nnz_w
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: t, t_star, feasep, beta, tk, q_t1, q_t1_old, q_t2
      REAL ( KIND = wp ) :: tbreak, deltat, root_hd, gp, val, php, epstl2
      LOGICAL :: xlower, xupper, printp, printw, printd, printdd, recomp
!     LOGICAL :: recomp

      status = GALAHAD_ok

!  on entry, set constants

      printp = print_level >= 3 .AND. out > 0
      printw = print_level >= 4 .AND. out > 0
      printd = print_level >= 5 .AND. out > 0
      printdd = print_level > 10 .AND. out > 0
!     printd = .TRUE.

      IF ( printp ) WRITE( out, "( /, A, ' ** arc search entered (nv = ', I0,  &
     &  ') ** ' )" ) prefix, nv
      n_break = 0 ; n_freed = 0 ; n_zero = nv + 1
      epstl2 = ten * epsmch
      tbreak = zero

!     IF ( printp ) THEN
      IF ( .FALSE. ) THEN
        i = COUNT(  V_status( : nv ) == 0 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are free' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, ' 1 variable is free' )" ) prefix
        END IF
        i = COUNT(  V_status( : nv ) == 1 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on lower bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its lower bound' )" ) prefix
        END IF
        i = COUNT(  V_status( : nv ) == 2 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on upper bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its upper bound' )" ) prefix
        END IF
        i = COUNT(  V_status( : nv ) >= 3 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are fixed' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, 1X, ' 1 variable is fixed' )" ) prefix
        END IF
      END IF

      IF ( print_level >= 100 ) THEN
        DO i = 1, nv
          WRITE( out, "( A, ' Var low V up P ', I6, 4ES12.4 )" )               &
            prefix, i, BND( i, 1 ), V_0( i ), BND( i, 2 ), P( i )
        END DO
      END IF

!  initialize d

      D( : nv ) = P( : nv )

!  find the status of the variables

      IF ( printdd ) WRITE( out,                                               &
        "( A, '    nv     BND_l       V_0         BND_u        P' )" ) prefix

      n_fix = 0
      DO i = 1, nv
        IF ( printdd ) WRITE( out, "( A, I6, 5ES12.4 )" )                      &
          prefix, i, BND( i, 1 ), V_0( i ), BND( i, 2 ), P( i )

!  check to see whether the variable is fixed

        IF ( V_status( i ) <= 2 ) THEN
          V_status( i ) = 0
          xupper = BND( i, 2 ) - V_0( i ) <= feas_tol
          xlower = V_0( i ) - BND( i, 1 ) <= feas_tol

!  the variable lies between its bounds. Check to see if the search
!  direction is zero

          IF ( .NOT. ( xupper .OR. xlower ) ) THEN
            IF ( ABS( P( i ) ) > epsmch ) GO TO 110
            n_zero = n_zero - 1
            NZ_p( n_zero ) = i
!write(6,*) 'NZ(', n_zero, ')=', i, ' nzero '
          ELSE

!  the variable lies close to its lower bound

            IF ( xlower ) THEN
              IF ( P( i ) > epsmch ) THEN
!write(6,*) ' variable ', i, ' freed from lower bound'
                n_freed = n_freed + 1
                GO TO 110
              END IF
              V_status( i ) = 1

!  the variable lies close to its upper bound

            ELSE
              IF ( P( i ) < - epsmch ) THEN
!write(6,*) ' variable ', i, ' freed from upper bound'
                n_freed = n_freed + 1
                GO TO 110
              END IF
              V_status( i ) = 2
            END IF
          END IF
        END IF

!  set the search direction to zero

        V_t( i ) = V_0( i )
        P( i ) = zero
        D( i ) = zero
        n_fix = n_fix + 1
        CYCLE
  110   CONTINUE

!  if the variable is free, set up the pointers to the nonzeros in the vector
!  p ready for calculating q = H * p

        n_break = n_break + 1
!write(6,*) 'NZ(', n_break, ')=', i
        NZ_p( n_break ) = i
      END DO

!  record the number of free variables at the starting point

      n_free = n_break ; nvar_u = n_free ; q_t = f

!  if all of the variables are fixed, exit

      IF ( printp ) WRITE( out, "( /, A, 1X, I0, ' dual variable', A,          &
     &  ' freed from ', A, ' bound', A, ', ', I0, ' variable', A, ' remain', A,&
     &  ' fixed,', /, A, ' of which ', I0, 1X, A, ' between bounds' )" )       &
        prefix, n_freed, TRIM( STRING_pleural( n_freed ) ),                    &
        TRIM( STRING_their( n_freed ) ), TRIM( STRING_pleural( n_freed ) ),    &
        nv - n_break, TRIM( STRING_pleural( nv - n_break ) ),                  &
        TRIM( STRING_verb_pleural( nv - n_break ) ),                           &
        prefix, nv - n_zero + 1, TRIM( STRING_are( nv - n_zero + 1 ) )
      IF ( n_break == 0 ) GO TO 600

!  find the breakpoints for the piecewise linear arc (the distances
!  to the boundary)

      DO j = 1, n_break
        i = NZ_p( j )
        IF ( P( i ) > epsmch ) THEN
          IF ( BND( i, 2 ) >= bnd_inf ) THEN
            t = t_max
          ELSE
            t = ( BND( i, 2 ) - V_0( i ) ) / P( i )
          END IF
        ELSE
          IF ( BND( i, 1 ) <= - bnd_inf ) THEN
            t = t_max
          ELSE
            t = ( BND( i, 1 ) - V_0( i ) ) / P( i )
          END IF
        END IF
        BREAK_points( j ) = t
      END DO

!  order the breakpoints in increasing size using a heapsort. Build the heap

      CALL SORT_heapsort_build( n_break, BREAK_points, insort, INDA = NZ_p )

!  compute w = J^T p

      nvar_l = 1 ; nvar_u = n_free

      W( : n ) = zero
      DO k = nvar_l, nvar_u
        ii = NZ_p( k )
        IF ( ii <= 0 .OR. ii > nv ) THEN
          IF ( printdd ) WRITE( out, "( ' extended index ', I0,                &
         &   ' larger than nv = ', I0 )" ) ii, nv
        ELSE IF ( ii <= ce_end ) THEN
          i = ii + start_ce
          IF ( printdd )                                                       &
            WRITE( out, "( ' product involves equality c ', I0 )" ) i
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            W( j ) = W( j ) + A_val( l ) * P( ii )
          END DO
        ELSE IF ( ii <= yl_end ) THEN
          i = ii + start_yl
          IF ( printdd )                                                       &
            WRITE( out, "( ' product involves lower c ', I0 )" ) i
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            W( j ) = W( j ) + A_val( l ) * P( ii )
          END DO
        ELSE IF ( ii <= yu_end ) THEN
          i = ii + start_yu
          IF ( printdd )                                                       &
            WRITE( out, "( ' product involves upper c ', I0 )" ) i
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l )
            W( j ) = W( j ) + A_val( l ) * P( ii )
          END DO
        ELSE IF ( ii <= zl_end ) THEN
          i = ii + start_zl
          IF ( printdd )                                                       &
            WRITE( out, "( ' product involves lower x ', I0 )" ) i
          W( i ) = W( i ) + P( ii )
        ELSE IF ( ii <= zu_end ) THEN
          i = ii + start_zu
          IF ( printdd )                                                       &
            WRITE( out, "( ' product involves upper x ', I0 )" ) i
          W( i ) = W( i ) + P( ii )
        END IF
      END DO

!  solve L w_k = r_k

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      IF ( identity_h ) THEN
!       W( : n ) = W( : n )
      ELSE IF ( scaled_identity_h ) THEN
        root_hd = SQRT( HESSIAN%val( 1 ) )
        W( : n ) = W( : n ) / root_hd
      ELSE IF ( diagonal_h ) THEN
        W( : n ) = W( : n ) / SQRT( HESSIAN%val( : n ) )
      ELSE
        CALL SLS_part_solve( 'S', W( : n ), SLS_data, SLS_control, SLS_inform )
      END IF

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      solve = solve + time_now - time_record
      clock_solve = clock_solve + clock_now - clock_record

!  initialize h

      H( : n ) = W( : n )

!  set all components of r to zero, and flag these in iused

      W( : n ) = zero ; R( : n ) = zero ; IUSED( : n ) = 0

!  calculate the function value (q_t), first derivative (q_t1) and
!  second derivative (q_t2) of the univariate piecewise quadratic
!  function at the start of the piecewise linear arc

      q_t1 = zero
      DO i = 1, n_free
        q_t1 = q_t1 + G( NZ_p( i ) ) * P( NZ_p( i ) )
      END DO
      q_t2 = DOT_PRODUCT( H( : n ), H( : n ) )
      IF ( q_t2 < hzero ) q_t2 = zero

      IF ( printdd ) WRITE( out,                                               &
        "( A, ' Current search direction ', /, ( 6X, 4( I6, ES12.4 ) ) )" )    &
         prefix, ( NZ_p( i ), P( NZ_p( i ) ), i = 1, nvar_u )

!  ---------
!  main loop
!  ---------

      iter = 0

!  start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces

      DO
        IF ( printw .OR. ( printp .AND. iter == 0 ) ) WRITE( out, 2000 ) prefix

!  print details of the piecewise quadratic in the next interval

        IF ( printp ) WRITE( out, "( A, 2I7, ES16.8, 3ES12.4 )" )              &
          prefix, iter, n_fix, q_t, q_t1, q_t2, tbreak
        IF ( printw ) WRITE( out,                                              &
          "( /, A, ' Piece', I5, ': f, G & H at start point', 4ES11.3 )" )     &
          prefix, iter, q_t, q_t1, q_t2, tbreak

        n_fix = 0
        iter = iter + 1

!  if the gradient of the univariate function increases, exit

        IF ( q_t1 > gzero ) GO TO 600

!  exit if the iteration limit has been exceeded

        IF ( max_iter >= 0 .AND. iter > max_iter ) GO TO 600

!  record the value of the last breakpoint

        tk = tbreak

!  find the next breakpoint ( end of the piece )

        tbreak = BREAK_points( 1 )
        CALL SORT_heapsort_smallest( n_break, BREAK_points, insort,            &
                                     INDA = NZ_p )

!  compute the length of the current piece

        deltat = MIN( tbreak, t_max ) - tk

!  print details of the breakpoint

        IF ( printw ) THEN
          WRITE( out, "( /, A, ' Next break point =', ES11.4, /, A,            &
         &  ' Maximum step     =', ES11.4 )" ) prefix, tbreak, prefix,         &
            t_max
        END IF

!  if the gradient of the univariate function is small and its curvature
!  is positive, exit

        IF ( ABS( q_t1 ) <= gzero ) THEN
          IF ( q_t2 > - hzero .OR. deltat >= HUGE( one ) ) THEN
            t_arc_minimizer = tk
            GO TO 600
          ELSE
            t_arc_minimizer = tbreak
          END IF
        ELSE

!  if the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

          IF ( q_t2 > zero ) THEN
            t_star = - q_t1 / q_t2
            IF ( printw ) WRITE( out,                                          &
              "( A, ' Stationary point =', ES11.4 )" ) prefix, tk + t_star

!           IF ( t_star > deltat ) THEN
!             write(6,*) ' t_star > deltat', t_star, deltat, ABS(t_star-deltat)
!           ELSE
!             write(6,*) ' t_star < deltat', t_star, deltat, ABS(t_star-deltat)
!           END IF

!  if the line minimum occurs before the breakpoint, the line minimum gives
!  the arc minimizer. Exit

            t_arc_minimizer = MIN( tk + t_star, tbreak )
            IF ( t_star < deltat ) THEN
              deltat = t_star
              GO TO 500
            END IF
          ELSE
            t_arc_minimizer = tbreak
          END IF
        END IF

!  if the arc minimizer occurs at t_max, exit.

        IF ( t_arc_minimizer >= t_max ) THEN
          t_arc_minimizer = t_max
          deltat = t_max - tk
          status = GALAHAD_error_primal_infeasible
          GO TO 500
        END IF

!  update the univariate function values

        q_t = q_t + deltat * ( q_t1 + half * deltat * q_t2 )

!  record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

        feasep = tbreak + epstl2

!  move the appropriate variable(s) to their bound(s)

        DO
          n_fix = n_fix + 1
          ibreak = NZ_p( n_break )
          n_break = n_break - 1
          IF ( printd ) WRITE( out, "( A, ' Variable ', I0,                    &
         &  ' is fixed, step =', ES12.4 )" ) prefix, ibreak, tbreak

!  indicate the status of the newly fixed variable

          IF ( P( ibreak ) < zero ) THEN
            V_status( ibreak ) = 1
          ELSE
            V_status( ibreak ) = 2
          END IF
          D( ibreak ) = zero

!  if all of the remaining search direction is zero, return

          IF ( n_break == 0 ) THEN
            DO j = 1, nvar_u
              i = NZ_p( j )

!  move the variable onto its bound

              V_t( i ) = BND( i, V_status( i ) )

!  store the step from the initial point to the arc minimizer in p

              P( i ) = V_t( i ) - V_0( i )
            END DO
            nvar_u = 0
            GO TO 600
          END IF

!  determine if other variables hit their bounds at the breakpoint

          IF (  BREAK_points( 1 ) >= feasep  ) EXIT
          CALL SORT_heapsort_smallest( n_break, BREAK_points, insort,          &
                                       INDA = NZ_p )
        END DO

!  update u and beta

        IF ( iter == 1  ) THEN
          U( : n ) = zero
          beta = deltat
        ELSE
          DO j = 1, nnz_w
            i = INDEX_w( j )
            U( i ) = U( i ) + beta * W( i )
            W( i ) = zero
          END DO
          beta = beta + deltat
        END IF

!  compute r = J^T p

        nvar_l = n_break + 1
        nnz_r = 0
        DO k = nvar_l, nvar_u
          ii = NZ_p( k )
          IF ( ii <= 0 .OR. ii > nv ) THEN
            IF ( printdd ) WRITE( out, "( ' extended index ', I0,              &
           &   ' larger than nv = ', I0 )" ) ii, nv
          ELSE IF ( ii <= ce_end ) THEN
            i = ii + start_ce
            IF ( printdd )                                                     &
              WRITE( out, "( ' product involves equality c ', I0 )" ) i
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              IF ( IUSED( j ) < iter ) THEN
                nnz_r = nnz_r + 1 ; INDEX_r( nnz_r ) = j ; IUSED( j ) = iter
              END IF
              R( j ) = R( j ) + A_val( l ) * P( ii )
            END DO
          ELSE IF ( ii <= yl_end ) THEN
            i = ii + start_yl
            IF ( printdd )                                                     &
              WRITE( out, "( ' product involves lower c ', I0 )" ) i
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              IF ( IUSED( j ) < iter ) THEN
                nnz_r = nnz_r + 1 ; INDEX_r( nnz_r ) = j ; IUSED( j ) = iter
              END IF
              R( j ) = R( j ) + A_val( l ) * P( ii )
            END DO
          ELSE IF ( ii <= yu_end ) THEN
            i = ii + start_yu
            IF ( printdd)                                                      &
              WRITE( out, "( ' product involves upper c ', I0 )" ) i
            DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
              j = A_col( l )
              IF ( IUSED( j ) < iter ) THEN
                nnz_r = nnz_r + 1 ; INDEX_r( nnz_r ) = j ; IUSED( j ) = iter
              END IF
              R( j ) = R( j ) + A_val( l ) * P( ii )
            END DO
          ELSE IF ( ii <= zl_end ) THEN
            i = ii + start_zl
            IF ( printdd )                                                     &
              WRITE( out, "( ' product involves lower x ', I0 )" ) i
            IF ( IUSED( i ) < iter ) THEN
              nnz_r = nnz_r + 1 ; INDEX_r( nnz_r ) = i ; IUSED( i ) = iter
            END IF
            R( i ) = R( i ) + P( ii )
          ELSE IF ( ii <= zu_end ) THEN
            i = ii + start_zu
            IF ( printdd )                                                     &
              WRITE( out, "( ' product involves upper x ', I0 )" ) i
            IF ( IUSED( i ) < iter ) THEN
              nnz_r = nnz_r + 1 ; INDEX_r( nnz_r ) = i ; IUSED( i ) = iter
            END IF
            R( i ) = R( i ) + P( ii )
          END IF
        END DO

!  solve L w_k = r_k

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( identity_h ) THEN
          DO j = 1, nnz_r
            i = INDEX_r( j )
            W( i ) = R( i )
            INDEX_w( j ) = i
          END DO
          nnz_w = nnz_r
        ELSE IF ( scaled_identity_h ) THEN
          DO j = 1, nnz_r
            i = INDEX_r( j )
            W( i ) = R( i ) / root_hd
            INDEX_w( j ) = i
          END DO
          nnz_w = nnz_r
        ELSE IF ( diagonal_h ) THEN
          DO j = 1, nnz_r
            i = INDEX_r( j )
            W( i ) = R( i ) / SQRT( HESSIAN%val( i ) )
            INDEX_w( j ) = i
          END DO
          nnz_w = nnz_r
        ELSE
          CALL SLS_sparse_forward_solve( nnz_r, INDEX_r, R, nnz_w, INDEX_w, W, &
                                         SLS_data, SLS_control, SLS_inform )
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        solve = solve + time_now - time_record
        clock_solve = clock_solve + clock_now - clock_record

!  reset nonzero components of r to zero

        R( INDEX_r( : nnz_r ) ) = zero

!  update the first and second derivatives of the univariate function

        q_t1_old = q_t1
        q_t1 = q_t1 + deltat * q_t2

!  include the contributions from the variables which have just been fixed

        DO j = nvar_l, nvar_u
          i = NZ_p( j )
          q_t1 = q_t1 - P( i ) * G( i )

!  move the variable onto its bound

          V_t( i ) = BND( i, V_status( i ) )
!write(6,"( I6, 5ES12.4 ) )" ) i, ABS( V_t( i ) - V_0( i ) ), V_0( i ), &
! BND( i, 1 ), BND( i, 2 ), P( i )

!  store the step from the initial point to the arc minimizer in p

          P( i ) = V_t( i ) - V_0( i )
        END DO

        DO j = 1, nnz_w
          i = INDEX_w( j )
          q_t1 = q_t1 - W( i ) * ( U( i ) + beta * H( i ) )
          q_t2 = q_t2 + W( i ) * ( W( i ) - two * H( i ) )
        END DO

!  update h

        DO j = 1, nnz_w
          i = INDEX_w( j )
          H( i ) = H( i ) - W( i )
        END DO

!  reset the number of free variables

        nvar_u = n_break

!  check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivatives will be recomputed

        recomp = ABS( q_t1 ) < - SQRT( epsmch ) * q_t1_old .OR. q_t2 <= zero

!  if required, compute the true line gradient and curvature.

        IF ( recomp .OR. printw ) THEN

!  next compute the matrix-vector product u = H_d * v_t = J H^{-1} J^T p.
!  First, compute r = J^T p

          DO k = 1, nvar_u
            ii = NZ_p( k )
            IF ( ii <= 0 .OR. ii > nv ) THEN
              IF ( printdd ) WRITE( out, "( ' extended index ', I0,            &
             &   ' larger than nv = ', I0 )" ) ii, nv
            ELSE IF ( ii <= ce_end ) THEN
              i = ii + start_ce
              IF ( printdd )                                                   &
                WRITE( out, "( ' product involves equality c ', I0 )" ) i
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                R( j ) = R( j ) + A_val( l ) * P( ii )
              END DO
            ELSE IF ( ii <= yl_end ) THEN
              i = ii + start_yl
              IF ( printdd )                                                   &
                WRITE( out, "( ' product involves lower c ', I0 )" ) i
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                R( j ) = R( j ) + A_val( l ) * P( ii )
              END DO
            ELSE IF ( ii <= yu_end ) THEN
              i = ii + start_yu
              IF ( printdd )                                                   &
                WRITE( out, "( ' product involves upper c ', I0 )" ) i
              DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
                j = A_col( l )
                R( j ) = R( j ) + A_val( l ) * P( ii )
              END DO
            ELSE IF ( ii <= zl_end ) THEN
              i = ii + start_zl
              IF ( printdd )                                                   &
                WRITE( out, "( ' product involves lower x ', I0 )" ) i
              R( i ) = R( i ) + P( ii )
            ELSE IF ( ii <= zu_end ) THEN
              i = ii + start_zu
              IF ( printdd )                                                   &
                WRITE( out, "( ' product involves upper x ', I0 )" ) i
              R( i ) = R( i ) + P( ii )
            END IF
          END DO

!  next set r -> H^{-1} r

          IF ( identity_h ) THEN
            IF ( printdd ) WRITE( out, "( ' identity h' )" )
!           R( : n ) = R( : n )
          ELSE IF ( scaled_identity_h ) THEN
            IF ( printdd ) WRITE( out, "( ' scaled identity h' )" )
            R( : n ) = R( : n ) / HESSIAN%val( 1 )
          ELSE IF ( diagonal_h ) THEN
            IF ( printdd ) WRITE( out, "( ' diagonal h' )" )
            R( : n ) = R( : n ) / HESSIAN%val( : n )
          ELSE
            IF ( printdd ) WRITE( out, "( ' general h' )" )
            CALL SLS_solve( HESSIAN, R, SLS_data, SLS_control, SLS_inform )
          END IF

!  finally, compute hp = J r

          DO ii = 1, yu_end
            IF ( ii <= ce_end ) THEN
              i = ii + start_ce
              IF ( printdd )                                                 &
                WRITE( out, "( ' product involves equality c ', I0 )" ) i
            ELSE IF ( ii <= yl_end ) THEN
              i = ii + start_yl
              IF ( printdd )                                                 &
                WRITE( out, "( ' product involves lower c ', I0 )" ) i
            ELSE
              i = ii + start_yu
              IF ( printdd )                                                 &
                WRITE( out, "( ' product involves upper c ', I0 )" ) i
            END IF
            val = zero
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              val = val + A_val( l ) * R( A_col( l ) )
            END DO
            HP( ii ) = val
          END DO

          DO ii = yu_end + 1, nv
            IF ( ii <= zl_end ) THEN
              i = ii + start_zl
              IF ( printdd )                                                 &
                WRITE( out, "( ' product involves lower x ', I0 )" ) i
            ELSE
              i = ii + start_zu
              IF ( printdd )                                                 &
                WRITE( out, "( ' product involves upper x ', I0 )" ) i
            END IF
            HP( ii ) = R( i )
          END DO

!  remember to re-initialise r = 0

          R( : n ) = zero

!  compute v_t

          V_t( NZ_p( : nvar_u ) ) =                                            &
            V_0( NZ_p( : nvar_u ) ) + tbreak * P( NZ_p( : nvar_u ) )

         IF ( printdd ) THEN
           WRITE( out, "( A, ' Current search direction ', /,                  &
          &  ( 6X, 4( I6, ES12.4 ) ) )" )                                      &
              prefix, ( NZ_p( i ), P( NZ_p( i ) ), i = 1, nvar_u )
           WRITE( out,  "( A, ' G ', /, ( 6X, 6( ES12.4 ) ) )" )               &
              prefix, ( G( i ), i = 1, nv )
           WRITE( out, "( A, ' HP ', /, ( 6X, 6( ES12.4 ) ) )" )               &
              prefix, ( HP( i ), i = 1, nv )
           WRITE( out,  "( A, ' v_t ', /, ( 6X, 6( ES12.4 ) ) )" )             &
             prefix, ( V_t( i ) , i = 1, nv )
         END IF

!  compute gp = hp^T v_t + p^T g_d and php = hp^T p

          gp = DOT_PRODUCT( HP, ( V_t - V_0 ) ) ; php = zero
          DO j = 1, nvar_u
            i = NZ_p( j )
            gp = gp + P( i ) * G( i )
            php = php + P( i ) * HP( i )
          END DO

          IF ( printw ) WRITE( out, "( /, A, ' Calculated q_t1 and q_t2 =',    &
         &  2ES22.14, /, A, ' Recurred   q_t1 and q_t2 =', 2ES22.14 )" )       &
              prefix, gp, php, prefix, q_t1, q_t2
          IF ( recomp ) THEN
            q_t1 = gp ; q_t2 = php
          END IF
        END IF

!  jump back to calculate the next breakpoint

      END DO

!  ----------------
!  end of main loop
!  ----------------

!  step to the arc minimizer

  500 CONTINUE

!  calculate the function value for the piecewise quadratic

      q_t = q_t + deltat * ( q_t1 + half * deltat * q_t2 )
      IF ( printw ) WRITE( out, 2000 ) prefix
      IF ( printp ) WRITE( out, "( A, 2I7, ES16.8, 24X, ES12.4 )" )            &
        prefix, iter, n_fix, q_t, t_arc_minimizer
      IF ( printp ) WRITE( out,                                                &
       "( /, A, ' Function value at the arc minimizer ', ES12.4 )" ) prefix, q_t

!  the arc minimizer has been found. Set the array p to the step from the
!  initial point to the arc minimizer

  600 CONTINUE
      P( NZ_p( : nvar_u ) ) = t_arc_minimizer * P( NZ_p( : nvar_u ) )
      V_t( NZ_p( : nvar_u ) ) = V_0( NZ_p( : nvar_u ) ) + P( NZ_p( : nvar_u ) )
!write(6,*) MAXVAL( ABS( P( NZ_p( : nvar_u ) ) ) )
!  record that variables whose gradients were zero at the initial
!  point are free variables

!     DO j = n_zero, nv
!       n_free = n_free + 1
!       NZ_p( n_free ) = NZ_p( j )
!     END DO

!  set return conditions

!     nvar_l = 1 ; nvar_u = n_free

      RETURN

!  non-executable statement

 2000 FORMAT( /, A, ' Segment fixed       model        gradient   curvature',  &
              '     step' )

!  End of subroutine DQP_exact_arc_search

      END SUBROUTINE DQP_exact_arc_search

!-*-*-  D Q P _ I N E X A C T _ A R C _ S E A R C H   S U B R O U T I N E  -*-*-

      SUBROUTINE DQP_inexact_arc_search( dims, nv, n, m, P, V_t, q_t, t,       &
                   G, q_0, Y_l, Y_u, Z_l, Z_u, C_l, C_u, X_l, X_u,             &
                   ce_start, ce_end, yl_start, yl_end, yu_start, yu_end,       &
                   zl_start, zl_end, zu_start, zu_end,                         &
                   A_ptr, A_col, A_val, t_max, feas_tol,                       &
                   V_status, iter, out, print_level,  prefix,                  &
                   D, S, H, diagonal_h, scaled_identity_h, identity_h, solve,  &
                   clock_solve, status, HESSIAN,                               &
                   SLS_data, SLS_control, SLS_inform )

!  Find an approximation to the arc minimizer in the direction p from v_0
!  for a given quadratic function within a box shaped region

!  If we define the arc v(t) = projection of v_0 + t * p into the box region

!     v_l <= v <= v_u,

!  the arc minimizer is the first local minimizer of the quadratic function

!     q(v) = 1/2 (v-v_0)^T H_d (v-v_0) + g_d^T (v-v_0) + f

!  for points lying on v(t), with 0 <= t <= t_max. Here

!    H_d = J H^{-1} J^T and g_d = J H^{-1} ( J^T v_0 - g ) - d

!  If v = v_0 + s and H = L L^T, we have that

!    q(v) = 1/2 s^T J H^{-1} J s + s^T ( J H^{-1} ( J^T v_0 - g ) - d )
!         = 1/2 h^T t + h^T q - s^T d,

!  where L h = J^T s and L q = J^T v_0 - g

!  A suitable inexact arc search is defined as follows:

!  1) If the minimizer of q(x) along x_0 + t * p lies on the search arc,
!     this is the required point. Otherwise,

!  2) Starting from some specified t_0, construct a decreasing sequence
!     of values t_1, t_2, t_3, .... . Given 0 < mu < 1, pick the first
!     t_i (i = 0, 1, ...) for which the Armijo condition

!        q(v(t_i)) <= linear(x(t_i),mu) = f + mu * g^T (v(t_i) - v_0)

!     is satisfied. v_0 + t_i * p is then the required point

!  The value of the array V_status gives the status of the variables

!  IF V_status( I ) = 0, the I-th variable is free
!  IF V_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF V_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF V_status( I ) = 3, the I-th variable is permanently fixed
!  IF V_status( I ) = 4, the I-th variable is fixed at some other value

!  At the initial point, variables within feas_tol of their bounds and
!  for which the search direction P points out of the box will be fixed

!  Based on CAUCHY_get_inexact_gcp from LANCELOT B

!  ------------------------- dummy arguments (to be updated!) -----------------

!  nv     (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  V_0    (REAL array of length at least nv) the point v_0 from which the
!          search arc commences. ** this variable is not altered by the
!          subroutine
!  P      (REAL array of length at least n) contains the values of the
!          components of the vector P. On entry, P must contain the initial
!          direction of the search arc
!  V_t    (REAL array of length at least nv) the current estimate of the
!          arc minimizer
!  q_t     (REAL) the value of the piecewise quadratic function at the current
!          estimate of the arc minimizer
!  t      (REAL) the minimizing value of t
!  G      (REAL array of length at least nv) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  BND    (two dimensional REAL array with leading dimension nv and second
!          dimension 2) the lower (BND(*,1)) and upper (BND(*,2)) bounds on
!          the variables. ** this variable is not altered by the subroutine
!  V_status (INTEGER array of length at least nv) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          V_status should be set as follows:
!          If V_status( i ) = 0, the i-th variable is free
!          If V_status( i ) = 1, the i-th variable is on its lower bound
!          If V_status( i ) = 2, the i-th variable is on its upper bound
!          If V_status( i ) = 3, 4, the i-th variable is fixed at V_t(i)
!  q_0     (REAL) the value of the quadratic at V_0, see above.
!          ** this variable is not altered by the subroutine
!  feas_tol (REAL) a tolerance on feasibility of V_0, see above.
!          ** this variable is not altered by the subroutine.
!  INDEX_w (INTEGER array of length at least nv) workspace
!  out    (INTEGER) the fortran output channel number to be used
!  iter   (INTEGER) the number of iterations performed
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  BREAK_points (REAL) workspace that must be preserved between calls

!  ------------------ end of dummy arguments --------------------------

      TYPE ( DQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ):: nv, n, m, out, print_level
      INTEGER, INTENT( INOUT ):: iter
      INTEGER, INTENT( IN ) :: ce_start, ce_end, yl_start, yl_end, yu_start
      INTEGER, INTENT( IN ) :: yu_end, zl_start, zl_end, zu_start, zu_end
      INTEGER, INTENT( INOUT ):: status
      REAL ( KIND = wp ), INTENT( IN ):: t_max, feas_tol, q_0
      REAL ( KIND = wp ), INTENT( INOUT ):: q_t, t
      REAL ( KIND = wp ), INTENT( INOUT ):: solve, clock_solve
      LOGICAL, INTENT( IN ) :: diagonal_h, scaled_identity_h, identity_h
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, DIMENSION( nv ), INTENT( INOUT ) :: V_status
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( 1 : dims%c_l_end ) :: Y_l, C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, C_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: Z_l, X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_u_start : n ) :: Z_u, X_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: V_t
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nv ) :: D, P
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: S, H
      TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: HESSIAN
      TYPE ( SLS_data_type ), OPTIONAL, INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), OPTIONAL, INTENT( IN ) :: SLS_control
      TYPE ( SLS_inform_type ), OPTIONAL, INTENT( INOUT ) :: SLS_inform

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, P, G, t_max, f, feas_tol, out, print_level

!  If the i-th variable is required to be fixed at its initial value
!   V_status(i) must be set to 3 or 4

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, il, iu, j, jl, ju, l, n_free, n_zero
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: slope, curvature, t_first, t_last, t_break
      REAL ( KIND = wp ) :: di, l_t, q_old, root_hd
      LOGICAL :: printp
!     LOGICAL :: printw, printd
      INTEGER, PARAMETER :: itmax = 100
      LOGICAL, PARAMETER :: forward = .TRUE.

!  on entry, set constants

      printp = print_level >= 3 .AND. out > 0
!     printw = print_level >= 4 .AND. out > 0
!     printd = print_level > 10 .AND. out > 0

      iter = 0 ; t = zero

!  if p is zero, exit

      IF ( MAXVAL( ABS( P( : nv ) ) ) == zero ) THEN
        q_t = q_0
        IF ( printp ) WRITE( out, 2000 ) prefix, prefix, iter, zero, q_t
        GO TO 600
      END IF

!  compute the original step d, checking to see if each variable is fixed
!  or on its bound with the input step moving infeasible

      D( ce_start : ce_end ) = P( ce_start : ce_end )
      n_free = ce_end ; n_zero = nv + 1

      il = yl_start - 1
      DO i = dims%c_l_start, dims%c_l_end
        il = il + 1
        IF ( V_status( il ) <= 2 ) THEN
          IF ( Y_l( i ) > feas_tol ) THEN
            IF ( ABS( P( il ) ) > epsmch ) THEN
              D( il ) = P( il ) ; n_free = n_free + 1
            ELSE
              D( il ) = zero ; n_zero = n_zero - 1
            END IF
          ELSE
            IF ( P( il ) >= epsmch ) THEN
              D( il ) = P( il ) ; n_free = n_free + 1
              n_free = n_free + 1
            ELSE
              D( il ) = zero ; n_zero = n_zero - 1
            END IF
          END IF
        ELSE
          D( il ) = zero ; n_zero = n_zero - 1
        END IF
      END DO

      iu = yu_start - 1
      DO i = dims%c_u_start, dims%c_u_end
        iu = iu + 1
        IF ( V_status( iu ) <= 2 ) THEN
          IF ( Y_u( i ) < - feas_tol ) THEN
            IF ( ABS( P( iu ) ) > epsmch ) THEN
              D( iu ) = P( iu ) ; n_free = n_free + 1
            ELSE
              D( iu ) = zero ; n_zero = n_zero - 1
            END IF
          ELSE
            IF ( P( iu ) <= - epsmch ) THEN
              D( iu ) = P( iu ) ; n_free = n_free + 1
            ELSE
              D( iu ) = zero ; n_zero = n_zero - 1
            END IF
          END IF
        ELSE
          D( iu ) = zero ; n_zero = n_zero - 1
        END IF
      END DO

      jl = zl_start - 1
      DO j =  dims%x_free + 1, dims%x_l_end
        jl = jl + 1
        IF ( V_status( jl ) <= 2 ) THEN
          IF ( Z_l( j ) > feas_tol ) THEN
            IF ( ABS( P( jl ) ) > epsmch ) THEN
              D( jl ) = P( jl ) ; n_free = n_free + 1
            ELSE
              D( jl ) = zero ; n_zero = n_zero - 1
            END IF
          ELSE
            IF ( P( jl ) >= epsmch ) THEN
              D( jl ) = P( jl ) ; n_free = n_free + 1
              n_free = n_free + 1
            ELSE
              D( jl ) = zero ; n_zero = n_zero - 1
            END IF
          END IF
        ELSE
          D( jl ) = zero ; n_zero = n_zero - 1
        END IF
      END DO

      ju = zu_start - 1
      DO j = dims%x_u_start, n
        ju = ju + 1
        IF ( V_status( ju ) <= 2 ) THEN
          IF ( Z_u( j ) < - feas_tol ) THEN
            IF ( ABS( P( ju ) ) > epsmch ) THEN
              D( ju ) = P( ju ) ; n_free = n_free + 1
            ELSE
              D( ju ) = zero ; n_zero = n_zero - 1
            END IF
          ELSE
            IF ( P( ju ) <= - epsmch ) THEN
              D( ju ) = P( ju ) ; n_free = n_free + 1
            ELSE
              D( ju ) = zero ; n_zero = n_zero - 1
            END IF
          END IF
        ELSE
          D( ju ) = zero ; n_zero = n_zero - 1
        END IF
      END DO

      IF ( printp ) WRITE( out,                                                &
        "( /, A, ' ----------- inexact arcsearch -------------', //,           &
       &  A, I8, ' variables free of their bounds ', /, A, I8,                 &
       &  ' variables fixed ' )" ) prefix, prefix, n_free, prefix, nv - n_free

!  if all of the variables are fixed, exit

      IF ( n_free == 0 ) THEN
        q_t = q_0
        IF ( printp ) WRITE( out, 2000 ) prefix, prefix, iter, zero, q_t
        GO TO 600
      END IF

!  compute the dual residual r = J^T v - g and store in h

      H( : n ) = - G( : n )
      DO i = 1, dims%c_u_start - 1
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l ) ; H( j ) = H( j ) + A_val( l ) * Y_l( i )
        END DO
      END DO
      DO i = dims%c_u_start, dims%c_l_end
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l )
          H( j ) = H( j ) + A_val( l ) * ( Y_l( i ) + Y_u( i ) )
        END DO
      END DO
      DO i = dims%c_l_end + 1, dims%c_u_end
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l ) ; H( j ) = H( j ) + A_val( l ) * Y_u( i )
        END DO
      END DO
      DO j = dims%x_free + 1, dims%x_u_start - 1
        H( j ) = H( j ) + Z_l( j )
      END DO
      DO j = dims%x_u_start, dims%x_l_end
        H( j ) = H( j ) + Z_l( j ) + Z_u( j )
      END DO
      DO j = dims%x_l_end + 1, n
        H( j ) = H( j ) + Z_u( j )
      END DO

!  solve L h = r and overwrite h in H

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      IF ( identity_h ) THEN
!       H( : n ) = H( : n )
      ELSE IF ( scaled_identity_h ) THEN
        root_hd = SQRT( HESSIAN%val( 1 ) )
        H( : n ) = H( : n ) / root_hd
      ELSE IF ( diagonal_h ) THEN
        H( : n ) = H( : n ) / SQRT( HESSIAN%val( : n ) )
      ELSE
        CALL SLS_part_solve( 'S', H, SLS_data, SLS_control, SLS_inform )
      END IF
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      solve = solve + time_now - time_record
      clock_solve = clock_solve + clock_now - clock_record

!  compute J^T d and store in S

      S( : n ) = zero
      il = ce_start ; iu = yu_start ; jl = zl_start ; ju = zu_start
      DO i = 1, dims%c_u_start - 1
        di = D( il ) ; il = il + 1
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
        END DO
      END DO
      DO i = dims%c_u_start, dims%c_l_end
        di = D( il ) + D( iu ) ; il = il + 1 ; iu = iu + 1
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
        END DO
      END DO
      DO i = dims%c_l_end + 1, dims%c_u_end
        di = D( iu ) ; iu = iu + 1
        DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
          j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
        END DO
      END DO
      DO j = dims%x_free + 1, dims%x_u_start - 1
        S( j ) = S( j ) + D( jl ) ; jl = jl + 1
      END DO
      DO j = dims%x_u_start, dims%x_l_end
        S( j ) = S( j ) + D( jl ) + D( ju ) ; jl = jl + 1 ; ju = ju + 1
      END DO
      DO j = dims%x_l_end + 1, n
        S( j ) = S( j ) + D( ju ) ; ju = ju + 1
      END DO

!  solve L s = J^T d and overwrite s in S

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      IF ( identity_h ) THEN
!       S( : n ) = S( : n )
      ELSE IF ( scaled_identity_h ) THEN
        S( : n ) = S( : n ) / root_hd
      ELSE IF ( diagonal_h ) THEN
        S( : n ) = S( : n ) / SQRT( HESSIAN%val( : n ) )
      ELSE
        CALL SLS_part_solve( 'S', S, SLS_data, SLS_control, SLS_inform )
      END IF
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      solve = solve + time_now - time_record
      clock_solve = clock_solve + clock_now - clock_record

!  compute the slope along d

      slope = DOT_PRODUCT( S( : n ), H( : n ) )                                &
        - DOT_PRODUCT( D( ce_start : yl_end ),                                 &
                       C_l( 1 : dims%c_l_end ) )                               &
        - DOT_PRODUCT( D( yu_start : yu_end ),                                 &
                       C_u( dims%c_u_start : dims%c_u_end ) )                  &
        - DOT_PRODUCT( D( zl_start : zl_end ),                                 &
                       X_l( dims%x_free + 1 : dims%x_l_end ) )                 &
        - DOT_PRODUCT( D( zu_start : zu_end ),                                 &
                       X_u( dims%x_u_start : n ) )

!  compute the curvature along d

      curvature = DOT_PRODUCT( S( : n ), S( : n ) )

      IF ( printp )                                                            &
        WRITE( out, "( /, A, ' initial slope and curvature ', 2ES12.4 )" )     &
          prefix, slope, curvature

!  compute the objective function at v + d

!    q_t = q_0 + slope + half * curvature

!  compute the maximum feasible distance, t_first, allowable in the direction d.
!  Also compute t_last, the largest breakpoint

      il = yl_start ; iu = yu_start ; jl = zl_start ; ju = zu_start
      t_first = infinity ; t_last = zero
      DO i = dims%c_l_start, dims%c_l_end
        IF ( D( il ) < zero ) THEN
          t_break = - Y_l( i ) / D( il )
          t_first = MIN( t_first, t_break ) ;  t_last = MAX( t_last, t_break )
        END IF
        il = il + 1
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        IF ( D( iu ) > zero ) THEN
          t_break = - Y_u( i ) / D( iu )
          t_first = MIN( t_first, t_break ) ;  t_last = MAX( t_last, t_break )
        END IF
        iu = iu + 1
      END DO
      DO j =  dims%x_free + 1, dims%x_l_end
        IF ( D( jl ) < zero ) THEN
          t_break = - Z_l( j ) / D( jl )
          t_first = MIN( t_first, t_break ) ;  t_last = MAX( t_last, t_break )
        END IF
        jl = jl + 1
      END DO
      DO j = dims%x_u_start, n
        IF ( D( ju ) > zero ) THEN
          t_break = - Z_u( j ) / D( ju )
          t_first = MIN( t_first, t_break ) ;  t_last = MAX( t_last, t_break )
        END IF
        ju = ju + 1
      END DO
      IF ( t_last < t_first ) t_last = t_first

      IF ( printp ) WRITE( out, 2000 ) prefix, prefix, iter, zero, q_0
      iter = iter + 1

!  check that the curvature is positive

      IF ( curvature > zero ) THEN

!  compute the minimizer, t, of quad(v(t)) in the direction p

        t = - slope / curvature

!  compare the values of t and t_first. If the calculated minimizer is the
!  arc miminizer, exit

        IF ( t <= t_first ) THEN

!  the arc minimizer occured in the first interval. Record the point and the
!  value of the quadratic at the point

          q_t = q_0 + t * ( slope + half * t * curvature )
          D( ce_start : zu_end ) = t * D( ce_start : zu_end )

!  print details of the arc minimizer

          IF ( printp )                                                        &
            WRITE( out, 2010 ) prefix, iter, t, q_t
          GO TO 600
        ELSE
          q_t = q_0 + t_first * ( slope + half * t_first * curvature )
          l_t = q_0 + mu_search * t_first * slope
          IF ( printp )                                                        &
            WRITE( out, 2010 ) prefix, iter, t_first, q_t, l_t
!         IF ( printp ) WRITE( out,                                            &
!        "( A, 21X, I6, '  1st line mimimizer infeasible. Step = ', ES10.2 )" )&
!             prefix, iter, t

!  ensure that the initial value of t for backtracking is no larger than
!  alpha_search times the step to the first line minimum

          IF ( forward ) THEN
            t = t_first
          ELSE
            t = MIN( alpha_search * t, t_last, t_max )
          END IF
        END IF
      ELSE
        IF ( forward ) THEN
          t = t_first
          q_t = q_0 + t_first * ( slope + half * t_first * curvature )
        ELSE
          t = t_last
        END IF
      END IF

!  -----------------------
!  The remaining intervals
!  -----------------------

!  the calculated minimizer is infeasible; prepare to backtrack from t until
!  an approximate arc minimizer is found

!  --------------------------------
!  Start of the main iteration loop
!  --------------------------------

      DO

!  compute the step d = P( x + td ) - x

        D( ce_start : ce_end ) = t * P( ce_start : ce_end )
        D( yl_start : yl_end ) = MAX( Y_l( dims%c_l_start : dims%c_l_end )     &
                                        + t * P( yl_start : yl_end ), zero )   &
                                   - Y_l( dims%c_l_start : dims%c_l_end )
        D( yu_start : yu_end ) = MIN( Y_u( dims%c_u_start : dims%c_u_end )     &
                                        + t * P( yu_start : yu_end ), zero )   &
                                   - Y_u( dims%c_u_start : dims%c_u_end )
        D( zl_start : zl_end ) = MAX( Z_l( dims%x_free + 1 : dims%x_l_end )    &
                                        + t * P( zl_start : zl_end ), zero )   &
                                   - Z_l( dims%x_free + 1 : dims%x_l_end )
        D( zu_start : zu_end ) = MIN( Z_u( dims%x_u_start : n )                &
                                        + t * P( zu_start : zu_end ), zero )   &
                                   - Z_u( dims%x_u_start : n )

!  compute J^T d and store in S

        S( : n ) = zero
        il = ce_start ; iu = yu_start ; jl = zl_start ; ju = zu_start
        DO i = 1, dims%c_u_start - 1
          di = D( il ) ; il = il + 1
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
          END DO
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          di = D( il ) + D( iu ) ; il = il + 1 ; iu = iu + 1
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
          END DO
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          di = D( iu ) ; iu = iu + 1
          DO l = A_ptr( i ) , A_ptr( i + 1 ) - 1
            j = A_col( l ) ; S( j ) = S( j ) + A_val( l ) * di
          END DO
        END DO
        DO j = dims%x_free + 1, dims%x_u_start - 1
          S( j ) = S( j ) + D( jl ) ; jl = jl + 1
        END DO
        DO j = dims%x_u_start, dims%x_l_end
          S( j ) = S( j ) + D( jl ) + D( ju ) ; jl = jl + 1 ; ju = ju + 1
        END DO
        DO j = dims%x_l_end + 1, n
          S( j ) = S( j ) + D( ju ) ; ju = ju + 1
        END DO

!  solve L s = J^T d and overwrite s in S

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( identity_h ) THEN
!         S( : n ) = S( : n )
        ELSE IF ( scaled_identity_h ) THEN
          S( : n ) = S( : n ) / root_hd
        ELSE IF ( diagonal_h ) THEN
          S( : n ) = S( : n ) / SQRT( HESSIAN%val( : n ) )
        ELSE
          CALL SLS_part_solve( 'S', S, SLS_data, SLS_control, SLS_inform )
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        solve = solve + time_now - time_record
        clock_solve = clock_solve + clock_now - clock_record

!  compute the slope along d

        slope = DOT_PRODUCT( S( : n ), H( : n ) )                              &
          - DOT_PRODUCT( D( ce_start : yl_end ),                               &
                         C_l( 1 : dims%c_l_end ) )                             &
          - DOT_PRODUCT( D( yu_start : yu_end ),                               &
                         C_u( dims%c_u_start : dims%c_u_end ) )                &
          - DOT_PRODUCT( D( zl_start : zl_end ),                               &
                         X_l( dims%x_free + 1 : dims%x_l_end ) )               &
          - DOT_PRODUCT( D( zu_start : zu_end ),                               &
                         X_u( dims%x_u_start : n ) )

!  compute the curvature along d

        curvature = DOT_PRODUCT( S( : n ), S( : n ) )

!  compute the objective function at v + d

        IF ( forward ) q_old = q_t
        q_t = q_0 + slope + half * curvature

!  compute linear(v+d,mu)

        l_t = q_0 + mu_search * slope

!  print details of the current point

        iter = iter + 1
        IF ( printp ) WRITE( out, 2010 ) prefix, iter, t, q_t, l_t
        IF ( iter > itmax ) EXIT

!  compare q(x(t)) with linear(x(t),mu). If x(t) satisfies the Armijo condition
!  and thus qualifies as an approximate arch mimizer, exit

!  ..............
!  forward search
!  ..............

        IF ( forward ) THEN

!  if the Armijo rule is violated, move back to the previous step and exit

          IF ( q_t > l_t ) THEN
            t = t * beta_search
            q_t = q_old
            D( ce_start : ce_end ) = t * P( ce_start : ce_end )
            D( yl_start : yl_end )                                             &
              = MAX( Y_l( dims%c_l_start : dims%c_l_end )                      &
                       + t * P(  yl_start : yl_end ), zero )                   &
                  - Y_l( dims%c_l_start : dims%c_l_end )
            D( yu_start : yu_end )                                             &
              = MIN( Y_u( dims%c_u_start : dims%c_u_end )                      &
                       + t * P( yu_start : yu_end ), zero )                    &
                  - Y_u( dims%c_u_start : dims%c_u_end )
            D( zl_start : zl_end )                                             &
              = MAX( Z_l( dims%x_free + 1 : dims%x_l_end )                     &
                       + t * P( zl_start : zl_end ), zero )                    &
                  - Z_l( dims%x_free + 1 : dims%x_l_end )
            D( zu_start : zu_end )                                             &
              = MIN( Z_u( dims%x_u_start : n )                                 &
                       + t * P( zu_start : zu_end ), zero )                    &
                  - Z_u( dims%x_u_start : n )

            EXIT
          END IF

!  have we passed the last breakpoint?

          IF ( t > t_last ) GO TO 600

!  is the dual problem unbounded (i.e., primal infeasible)?

          IF ( t > t_max ) THEN
            status = GALAHAD_error_primal_infeasible
            GO TO 700
          END IF

!  increase the step size

          t = t / beta_search

!  ...............
!  backward search
!  ...............

        ELSE

!  if the Armijo rule is satisfied, exit

          IF ( q_t <= l_t ) EXIT

!  decrease the step size

          t = t * beta_search
        END IF
      END DO

!  ------------------------------
!  End of the main iteration loop
!  ------------------------------

!  an approximation to the arc minimizer has been found. Set the
!  array P to the step from the initial point to the approximate minimizer

  600 CONTINUE
      status = GALAHAD_ok

!  record the step and final point, and the values and number of free variables

  700 CONTINUE
      P( ce_start : zu_end ) = D( ce_start : zu_end )

      il = ce_start ; iu = yu_start ; jl = zl_start ; ju = zu_start ; n_free = 0
      DO i = 1, dims%c_equality
        V_t( il ) = Y_l( i ) + P( il )
        V_status( il ) = 0
        il = il + 1
      END DO
      DO i = dims%c_l_start, dims%c_l_end
        V_t( il ) = Y_l( i ) + P( il )
        IF ( V_t( il ) >= feas_tol ) THEN
          V_status( il ) = 0
        ELSE
          V_status( il ) = 1
        END IF
        il = il + 1
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        V_t( iu ) = Y_u( i ) + P( iu )
        IF ( V_t( iu ) <= - feas_tol ) THEN
          V_status( iu ) = 0
        ELSE
          V_status( iu ) = 2
        END IF
        iu = iu + 1
      END DO
      DO j =  dims%x_free + 1, dims%x_l_end
        V_t( jl ) = Z_l( j ) + P( jl )
        IF ( V_t( jl ) >= feas_tol ) THEN
          V_status( jl ) = 0
        ELSE
          V_status( jl ) = 1
        END IF
        jl = jl + 1
      END DO
      DO j = dims%x_u_start, n
        V_t( ju ) = Z_u( j ) + P( ju )
        IF ( V_t( ju ) <= - feas_tol ) THEN
          V_status( ju ) = 0
        ELSE
          V_status( ju ) = 2
        END IF
        ju = ju + 1
      END DO

!  set return conditions

      RETURN

!  non-executable statements

 2000 FORMAT( /, A, ' ** arcsearch entered  iter     step     ',               &
            '    q(step)       l(step,mu)', /, A, 21X, I6, ES12.4, ES16.8 )
 2010 FORMAT( A, 21X, I6, ES12.4, 2ES16.8 )

!  End of subroutine DQP_inexact_arc_search

      END SUBROUTINE DQP_inexact_arc_search

!-*-*-*-*-*-*-*-*-*-*-  D Q P _ C G   S U B R O U T I N E  -*-*-*-*-*-*-*-*-*-*-
!
!      SUBROUTINE DQP_CG( n, m, diagonal_h, scaled_identity_h, identity_h,     &
!                         H, G, f, A, out, print_level, prefix, V, q, iter,    &
!                         status, R, S, HS, PR, SOL, control, SLS_data,        &
!                         SLS_control, SLS_inform, H_diag )
!
!!  Approximate the minimizer of the quadratic function
!
!!    q(v) = 1/2 < v, Bv > + < g, v > + f
!
!!  when B = A H^{-1} A^T is positive semi definite. If q in unbounded from
!!  below, find a vector v for which q(v) decreases without bound
!
!!  ------------------------- dummy arguments --------------------------
!
!!  n      (INTEGER) the number of independent variables.
!!          ** this variable is not altered by the subroutine
!!  G      (REAL array of length at least nv) the coefficients of
!!          the linear term in the quadratic function
!!          ** this variable is not altered by the subroutine
!!  f      (REAL) the value of the quadratic at V = 0, see above.
!!          ** this variable is not altered by the subroutine
!!  print_level (INTEGER) allows detailed printing. If print_level is larger
!!          than 4, detailed output from the routine will be given. Otherwise,
!!          no output occurs
!!  V      (REAL array of length at least n) the estimate of the minimizer
!!  q      (REAL) the value of the piecewise quadratic function at the current
!!          estimate of the arc minimizer
!!  iter   (INTEGER) the number of iterations performed
!
!!  ------------------ end of dummy arguments --------------------------
!
!      INTEGER, INTENT( IN ):: n, m, out, print_level
!      INTEGER, INTENT( INOUT ):: iter
!      INTEGER, INTENT( OUT ):: status
!      REAL ( KIND = wp ), INTENT( INOUT ):: f, q
!      LOGICAL, INTENT( IN ) :: diagonal_h, scaled_identity_h, identity_h
!      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: G
!      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: V, R, S, HS, PR
!      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: SOL
!      TYPE ( SMT_type ), INTENT( IN ) :: H, A
!      TYPE ( DQP_control_type ), INTENT( IN ) :: control
!      TYPE ( SLS_data_type ), OPTIONAL, INTENT( INOUT ) :: SLS_data
!      TYPE ( SLS_control_type ), OPTIONAL, INTENT( IN ) :: SLS_control
!      TYPE ( SLS_inform_type ), OPTIONAL, INTENT( INOUT ) :: SLS_inform
!      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_diag
!
!!  INITIALIZATION:
!
!!  On the initial call to the subroutine the following variables MUST BE SET
!!  by the user:
!
!!      n, G, f, A_*, out, print_level
!
!!-----------------------------------------------
!!   L o c a l   V a r i a b l e s
!!-----------------------------------------------
!
!      INTEGER :: i, j, l
!      REAL ( KIND = wp ) :: alpha, beta, gnrmsq, old_gnrmsq
!      REAL ( KIND = wp ) :: stop_cg, pnrmsq, curvature
!      LOGICAL :: printp, printw
!
!!  on entry, set constants
!
!      printp = print_level >= 3 .AND. out > 0
!      printw = print_level >= 4 .AND. out > 0
!
!      IF ( printp ) WRITE( out, "( /, A, ' ** CG entered ** ' )" ) prefix
!
!      status = GALAHAD_ok
!
!!  start from v = 0
!
!       V( : m ) = zero ;  q = f
!       R( : m ) = G( : m )
!
!!  - - - - - - - - - -
!!  Start the CG loop
!!  - - - - - - - - - -
!
!       DO iter = 1, control%cg_maxit + 1
!
!!  obtain the preconditioned residual pg
!
!         PR( : m ) = R( : m )
!         gnrmsq =  DOT_PRODUCT( PR( : m ), R( : m ) )
!
!!  compute the CG stopping tolerance
!
!         IF (  iter == 1 )                                                     &
!           stop_cg = MAX( SQRT( ABS( gnrmsq ) ) * control%stop_cg_relative,    &
!                          control%stop_cg_absolute )
!
!!  print details of the current iteration
!
!         IF ( printw ) THEN
!           IF ( iter == 1 ) THEN
!             WRITE( out, "( /, A, '    required gradient =', ES8.1, /, A,      &
!            &    '    iter     model    proj grad    curvature     step')" )   &
!             prefix, stop_cg, prefix
!             WRITE( out,                                                       &
!               "( A, 1X, I7, 2ES12.4, '      -            -     ' )" )         &
!               prefix, iter, q, SQRT( ABS( gnrmsq ) )
!           ELSE
!             WRITE( out, "( A, 1X, I7, 4ES12.4 )" )                            &
!              prefix, iter, q, SQRT( ABS( gnrmsq ) ), curvature, alpha
!           END IF
!         END IF
!
!!  if the gradient of the model is sufficiently small or if the CG iteration
!!  limit is exceeded, exit; record the CG direction
!
!         IF ( SQRT( ABS( gnrmsq ) ) <= stop_cg ) EXIT
!
!!  compute the search direction, p_free, and the square of its length
!
!         IF ( iter > 1 ) THEN
!           beta = gnrmsq / old_gnrmsq
!           S( : m ) = - PR( : m ) + beta * S( : m )
!           pnrmsq = gnrmsq + pnrmsq * beta ** 2
!         ELSE
!           S( : m ) = - PR( : m )
!           pnrmsq = gnrmsq
!         END IF
!
!!  save the norm of the preconditioned gradient
!
!         old_gnrmsq = gnrmsq
!
!!  compute HS = A H^^-1 A^T s ... first store the vector A^T s in sol ...
!
!         SOL( : n ) = zero
!         DO l = 1, A%ne
!           i = A%row( l ) ; j = A%col( l )
!           SOL( j ) = SOL( j ) + A%val( l ) * S( i )
!         END DO
!
!!  ... then solve H x = sol and overwrite sol with x ...
!
!         IF ( identity_h ) THEN
!!          SOL( : n ) = SOL( : n )
!         ELSE IF ( scaled_identity_h ) THEN
!           SOL( : n ) = SOL( : n ) / H_diag( 1 )
!         ELSE IF ( diagonal_h ) THEN
!           SOL( : n ) = SOL( : n ) / H_diag( : n )
!         ELSE
!           CALL SLS_solve( H, SOL( : n ), SLS_data, SLS_control, SLS_inform )
!!          CALL SLS_solve( H, SOL( : n ), SLS_data, scontrol, SLS_inform )
!         END IF
!
!!  ... and finally compute HS = A sol
!
!         HS( : m ) = zero
!         DO l = 1, A%ne
!           i = A%row( l ) ; j = A%col( l )
!           HS( i ) = HS( i ) + A%val( l ) * SOL( j )
!         END DO
!
!!  compute the curvature s^T ( J H^{-1} J^T ) s along the search direction
!
!         curvature = DOT_PRODUCT( HS( : m ), S( : m ) ) / pnrmsq
!
!!  if the curvature is positive, compute the step to the minimizer of
!!  the objective along the search direction
!
!         IF ( curvature > control%cg_zero_curvature ) THEN
!           alpha = old_gnrmsq / curvature
!
!!  otherwise, the objective is unbounded ....
!
!         ELSE IF ( curvature >= - control%cg_zero_curvature ) THEN
!           IF ( printw ) WRITE( out, "( /, A, ' zero curvature = ', ES12.4 )" )&
!             prefix, curvature
!           V( : m ) = S( : m )
!           q = f + DOT_PRODUCT( S( : m ), G( : m ) )
!           EXIT
!         ELSE
!           status = GALAHAD_error_inertia
!           EXIT
!         END IF
!
!!  update the objective value
!
!         q = q + alpha * ( - old_gnrmsq + half * alpha * curvature )
!
!!  update the estimate of the solution
!
!         V( : m ) = V( : m ) + alpha * S( : m )
!
!!  update the gradient/residual at the estimate of the solution
!
!         R( : m ) = R( : m ) + alpha * HS( : m )
!
!!  compute HS = A H^^-1 A^T s ... first store the vector A^T s in sol ...
!
!         SOL( : n ) = zero
!         DO l = 1, A%ne
!           i = A%row( l ) ; j = A%col( l )
!           SOL( j ) = SOL( j ) + A%val( l ) * R( i )
!         END DO
!
!!  ... then solve H x = sol and overwrite sol with x ...
!
!         IF ( identity_h ) THEN
!!          SOL( : n ) = SOL( : n )
!         ELSE IF ( scaled_identity_h ) THEN
!           SOL( : n ) = SOL( : n ) / H_diag( 1 )
!         ELSE IF ( diagonal_h ) THEN
!           SOL( : n ) = SOL( : n ) / H_diag( : n )
!         ELSE
!           CALL SLS_solve( H, SOL( : n ), SLS_data, SLS_control, SLS_inform )
!!          CALL SLS_solve( H, SOL( : n ), SLS_data, scontrol, SLS_inform )
!         END IF
!
!!  ... and finally compute HS = A sol
!
!         PR( : m ) = zero
!         DO l = 1, A%ne
!           i = A%row( l ) ; j = A%col( l )
!           PR( i ) = PR( i ) + A%val( l ) * SOL( j )
!         END DO
!
!!  compute the curvature s^T ( J H^{-1} J^T ) s along the search direction
!
!         curvature = DOT_PRODUCT( PR( : m ), R( : m ) ) / &
!                     DOT_PRODUCT( R( : m ), R( : m ) )
!       END DO
!
!!  - - - - - - - - -
!!  End the CG loop
!!  - - - - - - - - -
!
!      RETURN
!
!!  End of subroutine DQP_CG
!
!      END SUBROUTINE DQP_CG

!-*-*-*-*-*-*-   D Q P _ w o r k s p a c e   S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE DQP_workspace( m, n, dims, A, H, composite_g, diagonal_h,     &
                                identity_h, scaled_identity_h, nv, lbd,        &
                                C_status, NZ_p, IUSED, INDEX_r, INDEX_w,       &
                                X_status, V_status, X_status_old,              &
                                C_status_old, C_active, X_active, CHANGES,     &
                                ACTIVE_list, ACTIVE_status, SOL, RHS, RES,     &
                                H_s, Y_l, Y_u, Z_l, Z_u, VECTOR,               &
                                BREAK_points, YC_l, YC_u, ZC_l, ZC_u, GY_l,    &
                                GY_u, GZ_l, GZ_u, V0, VT, GV, G, PV, HPV, DV,  &
                                V_bnd, H_sbls, A_sbls, SCU_mat,                &
                                control, inform )

!  allocate workspace arrays for use in DQP_solve_main

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n
      INTEGER, INTENT( OUT ) :: lbd, nv
      TYPE ( DQP_dims_type ), INTENT( IN ) :: dims
      LOGICAL, INTENT( IN ) :: composite_g
      LOGICAL, INTENT( IN ) :: diagonal_h, identity_h, scaled_identity_h
      TYPE ( SMT_type ), INTENT( IN ) :: A, H
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: C_status
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: NZ_p
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: IUSED
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: INDEX_r
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: INDEX_w
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: X_status
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: V_status
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: X_status_old
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: C_status_old
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: X_active
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: C_active
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: CHANGES
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: ACTIVE_list
      INTEGER, ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: ACTIVE_status
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: SOL
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: RES
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: RHS
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: H_s
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: Y_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: Y_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: Z_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: Z_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ),                        &
                                       DIMENSION( : ) :: VECTOR
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ),                        &
                                       DIMENSION( : ) :: BREAK_points
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: YC_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: YC_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: ZC_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: ZC_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: GY_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: GY_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: GZ_l
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: GZ_u
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: V0
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: VT
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: GV
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: PV
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: HPV
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: DV
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ),                        &
                                       DIMENSION( : , : ) :: V_bnd
      REAL ( KIND = wp ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) :: G
      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_sbls
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( DQP_control_type ), INTENT( IN ) :: control
      TYPE ( DQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, l, len_m_active, len_n_active, h_ne
      INTEGER :: max_ne_active, max_row_length
      CHARACTER ( LEN = 80 ) :: array_name

!  set array lengths

      len_m_active = dims%c_u_start - 1 +                                      &
          2 * ( dims%c_l_end - dims%c_u_start + 1 ) +  m - dims%c_l_end
      len_n_active = dims%x_u_start - dims%x_free - 1 +                        &
         2 * ( dims%x_l_end - dims%x_u_start + 1 ) + n - dims%x_l_end
      max_ne_active = 2 * ( A%ptr( m + 1 ) + n - 1 )
      nv = dims%c_l_end + ( dims%c_u_end - dims%c_u_start + 1 ) +              &
            ( dims%x_l_end - dims%x_free ) + ( n - dims%x_u_start + 1 )

!  allocate workspace arrays

      array_name = 'dqp: C_status'
      CALL SPACE_resize_array( m, C_status,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: NZ_p'
      CALL SPACE_resize_array( nv, NZ_p,                                       &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: IUSED'
      CALL SPACE_resize_array( n, IUSED,                                       &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: INDEX_r'
      CALL SPACE_resize_array( n, INDEX_r,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: INDEX_w'
      CALL SPACE_resize_array( n, INDEX_w,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: X_status'
      CALL SPACE_resize_array( n, X_status,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: V_status'
      CALL SPACE_resize_array( nv, V_status,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: X_status_old'
      CALL SPACE_resize_array( n, X_status_old,                                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: C_status_old'
      CALL SPACE_resize_array( m, C_status_old,                                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: X_active'
      CALL SPACE_resize_array( len_n_active, X_active,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: C_active'
      CALL SPACE_resize_array( len_m_active, C_active,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: SOL'
      CALL SPACE_resize_array( n + len_m_active + len_n_active                 &
               + control%max_sc, SOL,                                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: RHS'
      CALL SPACE_resize_array( n + len_m_active + len_n_active                 &
               + control%max_sc, RHS,                                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: RES'
      CALL SPACE_resize_array( n + len_m_active + len_n_active                 &
               + control%max_sc, RES,                                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: H_s'
      CALL SPACE_resize_array( n, H_s,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: Y_l'
      CALL SPACE_resize_array( 1, dims%c_l_end, Y_l,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: Y_u'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, Y_u,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: Z_l'
      CALL SPACE_resize_array( dims%x_free + 1, dims%x_l_end, Z_l,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: Z_u'
      CALL SPACE_resize_array( dims%x_u_start, n, Z_u,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: BREAK_points'
      CALL SPACE_resize_array( nv, BREAK_points,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: YC_l'
      CALL SPACE_resize_array( 1, dims%c_l_end, YC_l,                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: YC_u'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, YC_u,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: ZC_l'
      CALL SPACE_resize_array( dims%x_free + 1, dims%x_l_end, ZC_l,            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: ZC_u'
      CALL SPACE_resize_array( dims%x_u_start, n, ZC_u,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: GY_l'
      CALL SPACE_resize_array( 1, dims%c_l_end, GY_l,                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: GY_u'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, GY_u,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: GZ_l'
      CALL SPACE_resize_array( dims%x_free + 1, dims%x_l_end, GZ_l,            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: GZ_u'
      CALL SPACE_resize_array( dims%x_u_start, n, GZ_u,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'dqp: V0'
      CALL SPACE_resize_array( nv, V0,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: VT'
      CALL SPACE_resize_array( MAX( n, nv ), VT,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: GV'
      CALL SPACE_resize_array( nv, GV,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: PV'
      CALL SPACE_resize_array( nv, PV,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: HPV'
      CALL SPACE_resize_array( nv, HPV,                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: DV'
      CALL SPACE_resize_array( nv, DV,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: V_bnd'
      CALL SPACE_resize_array( nv, 2, V_bnd,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      IF ( composite_g ) THEN
        array_name = 'dqp: G'
        CALL SPACE_resize_array( n, G,                                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN
      ELSE
        array_name = 'dqp: G'
        CALL SPACE_resize_array( 0, G,                                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN
      END IF

!  allocate space for the SBLS solves

      IF ( diagonal_h ) THEN
        array_name = 'dqp: H_sbls%ptr'
        CALL SPACE_resize_array( 0, H_sbls%ptr,                                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

        array_name = 'dqp: H_sbls%col'
        CALL SPACE_resize_array( 0, H_sbls%col,                                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

        IF ( scaled_identity_h ) THEN
          array_name = 'dqp: H_sbls%val'
          CALL SPACE_resize_array( 1, H_sbls%val,                              &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN
        ELSE IF ( identity_h ) THEN
          array_name = 'dqp: H_sbls%val'
          CALL SPACE_resize_array( 0, H_sbls%val,                              &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN
        ELSE
          array_name = 'dqp: H_sbls%val'
          CALL SPACE_resize_array( n, H_sbls%val,                              &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN
        END IF
      ELSE
        h_ne = H%ptr( n + 1 ) - 1
        array_name = 'dqp: H_sbls%ptr'
        CALL SPACE_resize_array( n + 1, H_sbls%ptr,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

        array_name = 'dqp: H_sbls%col'
        CALL SPACE_resize_array( h_ne, H_sbls%col,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

        array_name = 'dqp: H_sbls%val'
        CALL SPACE_resize_array( h_ne, H_sbls%val,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN
      END IF

      array_name = 'dqp: A_sbls%row'
      CALL SPACE_resize_array( max_ne_active, A_sbls%row,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: A_sbls%col'
      CALL SPACE_resize_array( max_ne_active, A_sbls%col,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'dqp: A_sbls%val'
      CALL SPACE_resize_array( max_ne_active, A_sbls%val,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  allocate space for the SCU solves if they may be required

      IF ( control%max_sc > 0 ) THEN

!  discover how many entries there are in the largest control%max_sc rows

        IF ( n > 0 ) THEN
          X_status = 0 ; X_status( 1 ) = n
          max_row_length = 0
          DO i = 1, m
            j = A%ptr( i + 1 ) - A%ptr( i )
            IF ( j > 0 ) THEN
              X_status( j ) = X_status( j ) + 1
              max_row_length = MAX( max_row_length, j )
            END IF
          END DO
          l = control%max_sc
          lbd = 0
          DO i = max_row_length, 1, - 1
            j = MIN( X_status( i ), l )
            lbd = lbd + j * i
            l = l - j
            IF ( l == 0 ) EXIT
          END DO
        END IF
        array_name = 'dqp: CHANGES'
        CALL SPACE_resize_array( m + n, CHANGES,                               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: ACTIVE_list'
        CALL SPACE_resize_array( m + n + control%max_sc, ACTIVE_list,          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: ACTIVE_status'
        CALL SPACE_resize_array( m + n, ACTIVE_status,                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: VECTOR'
        CALL SPACE_resize_array( m + 2 * n, VECTOR,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_col_start'
        CALL SPACE_resize_array( control%max_sc + 1, SCU_mat%BD_col_start,     &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_val'
        CALL SPACE_resize_array( lbd, SCU_mat%BD_val, inform%status,           &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_row'
        CALL SPACE_resize_array( lbd, SCU_mat%BD_row, inform%status,           &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      ELSE
        array_name = 'dqp: CHANGES'
        CALL SPACE_resize_array( 0, CHANGES,                                   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: ACTIVE_list'
        CALL SPACE_resize_array( 0, ACTIVE_list,                               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: ACTIVE_status'
        CALL SPACE_resize_array( 0, ACTIVE_status,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: VECTOR'
        CALL SPACE_resize_array( 0, VECTOR,                                    &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_col_start'
        CALL SPACE_resize_array( 0, SCU_mat%BD_col_start,                      &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_val'
        CALL SPACE_resize_array( 0, SCU_mat%BD_val, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'dqp: SCU_mat%BD_row'
        CALL SPACE_resize_array( 0, SCU_mat%BD_row, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

      RETURN

!  End of subroutine DQP_workspace

      END SUBROUTINE DQP_workspace

!  End of module DQP

    END MODULE GALAHAD_DQP_double
