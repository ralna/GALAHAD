! THIS VERSION: GALAHAD 2.4 - 02/03/2011 AT 09:15 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ L S Q P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   started life as GALAHAD_CLS ~ 2000
!   originally released pre GALAHAD Version 1.0. April 10th 2001
!   update released with GALAHAD Version 2.0. November 1st 2005
!   modified to enable sbls in GALAHAD Version 2.4. April 16th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LSQP_double

!     ------------------------------------------------
!     |                                              |
!     | Minimize the linear/separable objective      |
!     |                                              |
!     |  1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some (possibly zero) diagonal matrix W,  |
!     | using an infeasible-point primal-dual method |
!     |                                              |
!     ------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_ies, STRING_are
      USE GALAHAD_QPP_double, LSQP_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, LSQP_data_type => QPD_data_type,                 &
                              LSQP_AX => QPD_AX

      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
         SORT_heapsort_smallest, SORT_inverse_permute
      USE GALAHAD_FDC_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_ROOTS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LSQP_initialize, LSQP_read_specfile, LSQP_solve,               &
                LSQP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,  &
                LSQP_A_by_cols, LSQP_Ax,                                       &
                LSQP_data_type, LSQP_dims_type, LSQP_indicators

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: no_last = - 1000
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
      REAL ( KIND = wp ), PARAMETER :: tenm7 = ten ** ( - 7 )
      REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: ten5 = ten ** 5
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSQP_control_type

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
!      2  augmented-system factorization                              (OBSOLETE)

        INTEGER :: factor = 0

!   the maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization                          (OBSOLETE)

        INTEGER :: max_col = 35

!   an initial guess as to the integer workspace required by SBLS     (OBSOLETE)

        INTEGER :: indmin = 1000

!   an initial guess as to the real workspace required by SBLS        (OBSOLETE)

        INTEGER :: valmin = 1000

!   the maximum number of iterative refinements allowed               (OBSOLETE)

        INTEGER :: itref_max = 1

!   the number of iterations for which the overall infeasibility
!     of the problem is not reduced by at least a factor %reduce_infeas
!     before the problem is flagged as infeasible (see reduce_infeas)

        INTEGER :: infeas_max = 200

!   the initial value of the barrier parameter will not be changed for the
!     first muzero_fixed iterations
!
        INTEGER :: muzero_fixed = 1

!   indicate whether and how much of the input problem
!    should be restored on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

!   specifies the type of indicator function used. Pssible values are

!     1 primal indicator: constraint active <=> distance to nearest bound
!         <= %indicator_p_tol
!     2 primal-dual indicator: constraint active <=> distance to nearest bound
!        <= %indicator_tol_pd * size of corresponding multiplier
!     3 primal-dual indicator: constraint active <=> distance to nearest bound
!        <= %indicator_tol_tapia * distance to same bound at previous iteration

        INTEGER :: indicator_type = 3

!   should extrapolation be used to track the central path? Possible values are

!     0 never
!     1 after the final major iteration
!     2 at each major iteration

        INTEGER :: extrapolate = 0

!    the maximum number of previous path points to use when fitting the data

        INTEGER :: path_history = 1

!    the maximum order of path derivative to use
!
        INTEGER :: path_derivatives = 5

!    the order of (Puiseux) series to fit to the path data: <=0 to fit all data

        INTEGER :: fit_order = - 1

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   the required accuracy for the primal infeasibility

        REAL ( KIND = wp ) :: stop_p = epsmch

!   the required accuracy for the dual infeasibility

        REAL ( KIND = wp ) :: stop_d = epsmch

!   the required accuracy for the complementarity

        REAL ( KIND = wp ) :: stop_c = epsmch

!   initial primal variables will not be closer than prfeas from their bounds

        REAL ( KIND = wp ) :: prfeas = one

!   initial dual variables will not be closer than dufeas from their bounds
!
        REAL ( KIND = wp ) :: dufeas = one

!   the initial value of the barrier parameter. If muzero is not positive,
!    it will be reset to an appropriate value

        REAL ( KIND = wp ) :: muzero = - one

!   if the overall infeasibility of the problem is not reduced by at least a
!    factor reduce_infeas over %infeas_max iterations, the problem is flagged
!    as infeasible (see infeas_max)

        REAL ( KIND = wp ) :: reduce_infeas = one - point01

!   if W=0 and the potential function value is smaller than
!         potential_unbounded * number of one-sided bounds,
!     the analytic center will be flagged as unbounded

        REAL ( KIND = wp ) :: potential_unbounded = - 10.0_wp

!   the threshold pivot used by the matrix factorization.
!    See the documentation for SBLS for details                       (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol = epsmch

!   the threshold pivot used by the matrix factorization when attempting to
!    detect linearly dependent constraints.
!    See the documentation for SBLS for details                       (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol_for_dependencies = half

!   any pivots smaller than zero_pivot in absolute value will be regarded to be
!    zero when attempting to detect linearly dependent constraints    (OBSOLETE)

        REAL ( KIND = wp ) :: zero_pivot = epsmch

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!  start terminal extrapolation when mu reaches mu_min

        REAL ( KIND = wp ) :: mu_min = ten ** ( - 5 )

!   if %indicator_type = 1, a constraint/bound will be
!    deemed to be active <=> distance to nearest bound <= %indicator_p_tol

        REAL ( KIND = wp ) :: indicator_tol_p = epsmch

!   if %indicator_type = 2, a constraint/bound will be deemed to be active
!     <=> distance to nearest bound
!        <= %indicator_tol_pd * size of corresponding multiplier

        REAL ( KIND = wp ) :: indicator_tol_pd = 1.0_wp

!   if %indicator_type = 3, a constraint/bound will be deemed to be active
!     <=> distance to nearest bound
!        <= %indicator_tol_tapia * distance to same bound at previous iteration

        REAL ( KIND = wp ) :: indicator_tol_tapia = 0.9_wp

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!   the equality constraints will be preprocessed to remove any linear
!    dependencies if true

        LOGICAL :: remove_dependencies = .TRUE.

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if %just_feasible is true, the algorithm will stop as soon as a feasible
!     point is found. Otherwise, the optimal solution to the problem will be
!     found

        LOGICAL :: just_feasible  = .FALSE.

!   if %getdua, is true, advanced initial values are obtained for the
!    dual variables

        LOGICAL :: getdua = .FALSE.

!  If extrapolation is to be used, decide between Puiseux and Taylor series

        LOGICAL :: puiseux = .TRUE.

!   if %feasol is true, the final solution obtained will be perturbed so that
!    variables close to their bounds are moved onto these bounds

        LOGICAL :: feasol = .FALSE.

!   if %balance_initial_complentarity is true, the initial complemetarity
!    is required to be balanced
!
        LOGICAL :: balance_initial_complentarity = .FALSE.

!  if %use_corrector, a corrector step will be used

        LOGICAL :: use_corrector = .FALSE.

!   if %array_syntax_worse_than_do_loop is true, f77-style do loops will be
!    used rather than f90-style array syntax for vector operations   (OBSOLETE)

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

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "LSQPPROB.SIF"  // REPEAT( ' ', 18 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSQP_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

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
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSQP_inform_type

!  return status. See LSQP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = - 1

!  the total number of "wasted" function evaluations during the linesearch

        INTEGER :: nbacts = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by LSQP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the logarithmic potential function
!      sum -log(distance to constraint boundary)

        REAL ( KIND = wp ) :: potential

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  timings (see above)

        TYPE ( LSQP_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L S Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LSQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LSQP. This routine should be called before
!  LSQP_solve
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

      TYPE ( LSQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSQP_control_type ), INTENT( OUT ) :: control
      TYPE ( LSQP_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set real control parameters

      control%stop_p = epsmch ** 0.33
      control%stop_c = epsmch ** 0.33
      control%stop_d = epsmch ** 0.33
      control%pivot_tol = epsmch ** 0.75
      control%zero_pivot = epsmch ** 0.75
      control%indicator_tol_p = control%stop_p

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%max_infeas = control%stop_p
      control%FDC_control%prefix = '" - FDC:"                     '

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%preconditioner = 2
      control%SBLS_control%prefix = '" - SBLS:"                    '
      control%SBLS_control%SLS_control%relative_pivot_tolerance =              &
        control%pivot_tol
      control%SBLS_control%SLS_control%zero_tolerance = control%zero_pivot

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of LSQP_initialize

      END SUBROUTINE LSQP_initialize

!-*-*-*-*-   L S Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LSQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LSQP_initialize could (roughly)
!  have been set as:

! BEGIN LSQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      1000
!  start-print                                       -1
!  stop-print                                        -1
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  initial-integer-workspace                         1000
!  initial-real-workspace                            1000
!  maximum-refinements                               1
!  maximum-poor-iterations-before-infeasible         200
!  barrier-fixed-until-iteration                     1
!  indicator-type-used                               3
!  restore-problem-on-output                         0
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  mininum-initial-primal-feasibility                1000.0
!  mininum-initial-dual-feasibility                  1000.0
!  initial-barrier-parameter                         -1.0
!  poor-iteration-tolerance                          0.98
!  minimum-potential-before-unbounded                -10.0
!  pivot-tolerance-used                              1.0D-12
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  identical-bounds-tolerance                        1.0D-15
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  just-find-feasible-point                          F
!  fix-barrier-parameter-throughout                  F
!  use-corrector-step                                F
!  balance-initial-complentarity                     F
!  get-advanced-dual-variables                       F
!  move-final-solution-onto-bound                    F
!  array-syntax-worse-than-do-loop                   F
!  generate-sif-file                                 F
!  sif-file-name                                     LSQPPROB.SIF

! END LSQP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( LSQP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: start_print = print_level + 1
      INTEGER, PARAMETER :: stop_print = start_print + 1
      INTEGER, PARAMETER :: maxit = stop_print + 1
      INTEGER, PARAMETER :: factor = maxit + 1
      INTEGER, PARAMETER :: max_col = factor + 1
      INTEGER, PARAMETER :: indmin = max_col + 1
      INTEGER, PARAMETER :: valmin = indmin + 1
      INTEGER, PARAMETER :: itref_max = valmin + 1
      INTEGER, PARAMETER :: infeas_max = itref_max + 1
      INTEGER, PARAMETER :: muzero_fixed = infeas_max + 1
      INTEGER, PARAMETER :: restore_problem = muzero_fixed + 1
      INTEGER, PARAMETER :: indicator_type = restore_problem + 1
      INTEGER, PARAMETER :: extrapolate = indicator_type + 1
      INTEGER, PARAMETER :: path_history = extrapolate + 1
      INTEGER, PARAMETER :: path_derivatives = path_history + 1
      INTEGER, PARAMETER :: fit_order = path_derivatives + 1
      INTEGER, PARAMETER :: sif_file_device = fit_order + 1
      INTEGER, PARAMETER :: infinity = sif_file_device + 1
      INTEGER, PARAMETER :: stop_p = infinity + 1
      INTEGER, PARAMETER :: stop_d = stop_p + 1
      INTEGER, PARAMETER :: stop_c = stop_d + 1
      INTEGER, PARAMETER :: prfeas = stop_c + 1
      INTEGER, PARAMETER :: dufeas = prfeas + 1
      INTEGER, PARAMETER :: muzero = dufeas + 1
      INTEGER, PARAMETER :: reduce_infeas = muzero + 1
      INTEGER, PARAMETER :: potential_unbounded = reduce_infeas + 1
      INTEGER, PARAMETER :: pivot_tol = potential_unbounded + 1
      INTEGER, PARAMETER :: pivot_tol_for_dependencies = pivot_tol + 1
      INTEGER, PARAMETER :: zero_pivot = pivot_tol_for_dependencies + 1
      INTEGER, PARAMETER :: identical_bounds_tol = zero_pivot + 1
      INTEGER, PARAMETER :: mu_min = identical_bounds_tol + 1
      INTEGER, PARAMETER :: indicator_tol_p = mu_min + 1
      INTEGER, PARAMETER :: indicator_tol_pd = indicator_tol_p + 1
      INTEGER, PARAMETER :: indicator_tol_tapia = indicator_tol_pd + 1
      INTEGER, PARAMETER :: cpu_time_limit = indicator_tol_tapia + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: remove_dependencies = clock_time_limit + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general =                     &
                              remove_dependencies + 1
      INTEGER, PARAMETER :: just_feasible = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: getdua = just_feasible + 1
      INTEGER, PARAMETER :: puiseux = getdua + 1
      INTEGER, PARAMETER :: feasol = puiseux + 1
      INTEGER, PARAMETER :: balance_initial_complentarity = feasol + 1
      INTEGER, PARAMETER :: use_corrector = balance_initial_complentarity + 1
      INTEGER, PARAMETER :: array_syntax_worse_than_do_loop = use_corrector + 1
      INTEGER, PARAMETER :: space_critical = array_syntax_worse_than_do_loop + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
      INTEGER, PARAMETER :: prefix = sif_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LSQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( factor )%keyword = 'factorization-used'
      spec( max_col )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec( indmin )%keyword = 'initial-integer-workspace'
      spec( valmin )%keyword = 'initial-real-workspace'
      spec( itref_max )%keyword = 'maximum-refinements'
      spec( infeas_max )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( muzero_fixed )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( indicator_type )%keyword = 'indicator-type-used'
      spec( extrapolate )%keyword = 'extrapolate-solution'
      spec( path_history )%keyword = 'path-history-length'
      spec( path_derivatives )%keyword = 'path-derivatives-used'
      spec( fit_order )%keyword = 'path-fit-order'
      spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( stop_p )%keyword = 'primal-accuracy-required'
      spec( stop_d )%keyword = 'dual-accuracy-required'
      spec( stop_c )%keyword = 'complementary-slackness-accuracy-required'
      spec( prfeas )%keyword = 'mininum-initial-primal-feasibility'
      spec( dufeas )%keyword = 'mininum-initial-dual-feasibility'
      spec( muzero )%keyword = 'initial-barrier-parameter'
      spec( reduce_infeas )%keyword = 'poor-iteration-tolerance'
      spec( potential_unbounded )%keyword = 'minimum-potential-before-unbounded'
      spec( pivot_tol )%keyword = 'pivot-tolerance-used'
      spec( pivot_tol_for_dependencies )%keyword =                             &
        'pivot-tolerance-used-for-dependencies'
      spec( zero_pivot )%keyword = 'zero-pivot-tolerance'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( mu_min )%keyword = 'minimum-barrier-before-final-extrapolation'
      spec( indicator_tol_p )%keyword = 'primal-indicator-tolerance'
      spec( indicator_tol_pd )%keyword = 'primal-dual-indicator-tolerance'
      spec( indicator_tol_tapia )%keyword = 'tapia-indicator-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( just_feasible )%keyword = 'just-find-feasible-point'
      spec( getdua )%keyword = 'get-advanced-dual-variables'
      spec( puiseux )%keyword = 'puiseux-extrapolation'
      spec( feasol )%keyword = 'move-final-solution-onto-bound'
      spec( balance_initial_complentarity )%keyword =                          &
        'balance-initial-complentarity'
      spec( use_corrector )%keyword = 'use-corrector-step'
      spec( array_syntax_worse_than_do_loop )%keyword =                        &
        'array-syntax-worse-than-do-loop'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( prefix )%keyword = 'sif-file-name'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( factor ),                               &
                                 control%factor,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_col ),                              &
                                 control%max_col,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indmin ),                               &
                                 control%indmin,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( valmin ),                               &
                                 control%valmin,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( itref_max ),                            &
                                 control%itref_max,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infeas_max ),                           &
                                 control%infeas_max,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero_fixed ),                         &
                                 control%muzero_fixed,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( restore_problem ),                      &
                                 control%restore_problem,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_type ),                       &
                                 control%indicator_type,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( extrapolate ),                          &
                                 control%extrapolate,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( path_history ),                         &
                                 control%path_history,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( path_derivatives ),                     &
                                 control%path_derivatives,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( fit_order ),                            &
                                 control%fit_order,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_p ),                               &
                                 control%stop_p,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_d ),                               &
                                 control%stop_d,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_c ),                               &
                                 control%stop_c,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prfeas ),                               &
                                 control%prfeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dufeas ),                               &
                                 control%dufeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero ),                               &
                                 control%muzero,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduce_infeas ),                        &
                                 control%reduce_infeas,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( potential_unbounded ),                  &
                                 control%potential_unbounded,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_tol ),                            &
                                 control%pivot_tol,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_tol_for_dependencies ),           &
                                 control%pivot_tol_for_dependencies,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_pivot ),                           &
                                 control%zero_pivot,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_min ),                               &
                                 control%mu_min,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_p ),                      &
                                 control%indicator_tol_p,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_pd ),                     &
                                 control%indicator_tol_pd,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_tapia ),                  &
                                 control%indicator_tol_tapia,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( remove_dependencies ),                  &
                                 control%remove_dependencies,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( just_feasible ),                        &
                                 control%just_feasible,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( getdua ),                               &
                                 control%getdua,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( puiseux ),                              &
                                 control%puiseux,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasol ),                               &
                                 control%feasol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( balance_initial_complentarity ),        &
                                 control%balance_initial_complentarity,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( use_corrector ),                        &
                                 control%use_corrector,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( array_syntax_worse_than_do_loop ),      &
                                 control%array_syntax_worse_than_do_loop,      &
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

!  Set character values

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
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
      control%FDC_control%max_infeas = control%stop_p

!  Read the specfile for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

      RETURN

      END SUBROUTINE LSQP_read_specfile

!-*-*-*-*-*-*-*-*-*-   L S Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE LSQP_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear/separable objective
!
!        1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse.
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
!    to be solved since the last call to LSQP_initialize, and .FALSE. if
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
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     any other value - the weights will be those given by %WEIGHT (see below)
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is not 0.
!    If this is so, it must be of length at least %n, and contain the
!    weights X^0 for the objective function.
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
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
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
!  data is a structure of type LSQP_data_type which holds private internal data
!
!  control is a structure of type LSQP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LSQP_initialize. See the preamble
!   for details
!
!  inform is a structure of type LSQP_inform_type that provides
!    information on exit from LSQP_solve. The component status
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
!  See preamble for other components returned
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
!  B_stat is an optional  INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LSQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
      TYPE ( LSQP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i, j, n_depen, nzc, n_sbls
      INTEGER :: dy_l_lower, dy_l_upper, dy_u_lower, dy_u_upper
      INTEGER :: dz_l_lower, dz_l_upper, dz_u_lower, dz_u_upper
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: fixed_sum, av_bnd
      LOGICAL :: printi, remap_freed, reset_bnd, stat_required
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering LSQP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .FALSE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  Initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Initialize counts

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%factorization_status = 0
      inform%iter = - 1 ; inform%nfacts = - 1 ; inform%nbacts = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%obj = - one ; inform%potential = infinity
      inform%non_negligible_pivot = zero
      inform%feasible = .FALSE.
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
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
          WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" )                 &
            prob%X0( : prob%n )
        ELSE
          WRITE( control%out, "( ' W = ', /, ( 5ES12.4 ) )" )                  &
            prob%WEIGHT( : prob%n )
          WRITE( control%out, "( ' X0 = ', /, ( 5ES12.4 ) )" )                 &
            prob%X0( : prob%n )
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

!  Check that problem bounds are consistent; reassign any pair of bounds
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

!  Record the objective function value for any fixed variables

      fixed_sum = zero
      IF ( prob%Hessian_kind == 1 ) THEN
        DO i = 1, prob%n
          IF ( prob%X_l( i ) == prob%X_u( i ) ) fixed_sum = fixed_sum +        &
               ( prob%X_l( i ) - prob%X0( i ) ) ** 2
        END DO
      ELSE IF ( prob%Hessian_kind == 2 ) THEN
        DO i = 1, prob%n
          IF ( prob%X_l( i ) == prob%X_u( i ) ) fixed_sum = fixed_sum +        &
               ( prob%WEIGHT( i ) * ( prob%X_l( i ) - prob%X0( i ) ) ) ** 2
        END DO
      END IF

!  Allocate sufficient workspace to hold a null Hessian (if needed)

      array_name = 'lsqp: data%IW'
      CALL SPACE_resize_array( prob%n + 1, data%IW,                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

      IF ( prob%new_problem_structure ) THEN
        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                       &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
             & ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  Perform the preprocessing.

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

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

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions after preprocessing: ', /, A,     &
             & ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prefix, prefix, prob%n, prob%m, data%a_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  Recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform,                       &
                          prob, get_f = .TRUE., get_g = .TRUE.,                &
                          get_x_bounds = .TRUE., get_c_bounds = .TRUE.,        &
                          get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,      &
                          get_c = .TRUE., get_A = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

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

      IF ( prob%m > 0 .AND.                                                    &
           ( .NOT. data%tried_to_remove_deps .AND.                             &
              control%remove_dependencies ) ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
           "( /, A, 1X, I0, ' equalit', A, ' from ', I0, ' constraint', A )" ) &
              prefix, data%dims%c_equality,                                    &
              TRIM( STRING_ies( data%dims%c_equality ) ),                      &
              prob%m, TRIM( STRING_pleural( prob%m ) )

!  Set control parameters

        data%FDC_control = control%FDC_control
        data%FDC_control%max_infeas = control%stop_p

!  Find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1
        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
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

!  Record output parameters

        inform%status = inform%FDC_inform%status
        inform%alloc_status = inform%FDC_inform%alloc_status
        inform%non_negligible_pivot = inform%FDC_inform%non_negligible_pivot
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
          control%FDC_control%SLS_control%absolute_pivot_tolerance)            &
            WRITE( control%out, "(                                             &
       &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,                 &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, A, ' ***  perhaps increase',                                      &
       &     ' FDC_control%SLS_control%absolute_pivot_tolerance from',         &
       &    ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )   &
           prefix, prefix, inform%non_negligible_pivot, prefix,                &
           control%FDC_control%SLS_control%absolute_pivot_tolerance, prefix

!  Check for error exits

        IF ( inform%status /= 0 ) THEN

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= GALAHAD_ok ) WRITE( control%error,           &
                 "( A, '   **  Error return ', I0, ' from ', A )" )            &
               prefix, inform%status, 'LSQP_dependent'
          END IF
          GO TO 700
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, A, ' The following ', I0, ' constraint',    &
         &  A, ' appear', A, ' to be dependent', /, ( 4X, 8I8 ) )" )           &
              prefix, n_depen, TRIM( STRING_are( n_depen ) ),                  &
              TRIM( STRING_pleural( n_depen ) ), data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  Special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y( : prob%m ) = zero
          prob%Z( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL LSQP_AX( prob%m, prob%C( : prob%m ), prob%m,                    &
                        prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,              &
                        prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  Some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ', A,   &
         & ' dependent and will be temporarily removed' )" ) prefix, n_depen,  &
           TRIM( STRING_pleural( n_depen ) ), TRIM( STRING_are( n_depen ) )

!  Allocate arrays to indicate which constraints have been freed

          array_name = 'lsqp: data%C_freed'
          CALL SPACE_resize_array( n_depen, data%C_freed,                      &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Free the constraint bounds as required

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

!  Store the problem dimensions

        data%dims_save_freed = data%dims
        data%a_ne = prob%A%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before removal of dependecies: ', &
              &   /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  Perform the preprocessing

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

!  Test for satisfactory termination

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

!  Record revised array lengths

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

!  Compute the dimension of the KKT system

      data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1

!  Arrays containing data relating to the composite vector ( x  c  y )
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

!  Allocate real workspace

      array_name = 'lsqp: data%HX'
      CALL SPACE_resize_array( data%dims%v_e, data%HX,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%GRAD_L'
      CALL SPACE_resize_array( data%dims%c_e, data%GRAD_L,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DIST_X_l'
      CALL SPACE_resize_array(  data%dims%x_l_start, data%dims%x_l_end,        &
             data%DIST_X_l,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DIST_X_u'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_u_end,         &
             data%DIST_X_u,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%Z_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%Z_l,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%Z_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n, data%Z_u,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%BARRIER_X'
      CALL SPACE_resize_array( data%dims%x_free + 1, prob%n, data%BARRIER_X,   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%Y_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%Y_l,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DY_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DY_l,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DIST_C_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DIST_C_l,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%Y_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%Y_u,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DY_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DY_u,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DIST_C_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DIST_C_u,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end, data%C, &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%BARRIER_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%BARRIER_C,                                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%SCALE_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%SCALE_C,                                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DELTA'
      CALL SPACE_resize_array( data%dims%v_e, data%DELTA,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%RHS'
      CALL SPACE_resize_array( data%dims%v_e, data%RHS,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DZ_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%DZ_l,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%DZ_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n, data%DZ_u,         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Allocate optional extra arrays

      IF ( control%use_corrector ) THEN
        array_name = 'lsqp: data%DELTA_cor'
        CALL SPACE_resize_array( data%dims%v_e, data%DELTA_cor,                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lsqp: data%DY_cor_l'
        CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,       &
               data%DY_cor_l,                                                  &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dy_l_lower = data%dims%c_l_start
        dy_l_upper = data%dims%c_l_end

        array_name = 'lsqp: data%DY_cor_u'
        CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,       &
               data%DY_cor_u,                                                  &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dy_u_lower = data%dims%c_u_start
        dy_u_upper = data%dims%c_u_end

        array_name = 'lsqp: data%DZ_cor_l'
        CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,      &
               data%DZ_cor_l,                                                  &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dz_l_lower = data%dims%x_free + 1
        dz_l_upper = data%dims%x_l_end

        array_name = 'lsqp: data%DZ_cor_u'
        CALL SPACE_resize_array( data%dims%x_u_start, prob%n, data%DZ_cor_u,   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dz_u_lower = data%dims%x_u_start
        dz_u_upper = prob%n

      ELSE
        array_name = 'lsqp: data%DELTA_cor'
        CALL SPACE_resize_array( 0, data%DELTA_cor,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lsqp: data%DY_cor_l'
        CALL SPACE_resize_array( 0, data%DY_cor_l,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dy_l_lower = 1
        dy_l_upper = 0

        array_name = 'lsqp: data%DY_cor_u'
        CALL SPACE_resize_array( 0, data%DY_cor_u,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dy_u_lower = 1
        dy_u_upper = 0

        array_name = 'lsqp: data%DZ_cor_l'
        CALL SPACE_resize_array( 0, data%DZ_cor_l,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dz_l_lower = 1
        dz_l_upper = 0

        array_name = 'lsqp: data%DZ_cor_u'
        CALL SPACE_resize_array( 0, data%DZ_cor_u,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        dz_u_lower = 1
        dz_u_upper = 0
      END IF

      IF ( stat_required ) THEN
        array_name = 'lsqp: data%H_s'
        CALL SPACE_resize_array( prob%n, data%H_s,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lsqp: data%A_s'
        CALL SPACE_resize_array( prob%m, data%A_s,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lsqp: data%Y_last'
        CALL SPACE_resize_array( prob%m, data%Y_last,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lsqp: data%Z_last'
        CALL SPACE_resize_array( prob%n, data%Z_last,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      n_sbls =  prob%n + data%dims%nc

!  H will be in diagonal form

      CALL SMT_put( data%H_sbls%type, 'DIAGONAL', inform%alloc_status )
      data%H_sbls%n = n_sbls
      data%H_sbls%ne = n_sbls

!  allocate space for H

      array_name = 'lsqp: data%H_sbls%val'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  A will be in coordinate form

      CALL SMT_put( data%A_sbls%type, 'COORDINATE', inform%alloc_status )
      data%A_sbls%n = n_sbls
      data%A_sbls%m = prob%m
      data%A_sbls%ne = data%a_ne + data%dims%nc

!  allocate space for A

      array_name = 'lsqp: data%A_sbls%row'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%col'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%val'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of A in coordinate form ...

      DO i = 1, prob%m
        data%A_sbls%row( prob%A%ptr( i ) : prob%A%ptr( i + 1 ) - 1 ) = i
      END DO
      data%A_sbls%col( : data%a_ne ) = prob%A%col( : data%a_ne )
      data%A_sbls%val( : data%a_ne ) = prob%A%val( : data%a_ne )

!  ... and include the coodinates corresponding to the slack variables

      DO i = 1, data%dims%nc
        data%A_sbls%row( data%a_ne + i ) = data%dims%c_equality + i
        data%A_sbls%col( data%a_ne + i ) = prob%n + i
      END DO

!  the zero matrix C will be in zero form

      CALL SMT_put( data%C_sbls%type, 'ZERO', inform%alloc_status )

!  =================
!  Solve the problem
!  =================

      data%SBLS_control = control%SBLS_control
      data%SBLS_control%preconditioner = 2
      data%SBLS_control%get_norm_residual = .TRUE.
      data%SBLS_control%perturb_to_make_definite = .FALSE.

!  constraint/variable exit ststus required

      IF ( stat_required ) THEN
        IF ( prob%Hessian_kind == 0 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  G = prob%G,                                  &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          END IF
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  X0 = prob%X0,                                &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  X0 = prob%X0, G = prob%G,                    &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  WEIGHT = prob%WEIGHT, X0 = prob%X0,          &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  WEIGHT = prob%WEIGHT, X0 = prob%X0,          &
                                  G = prob%G,                                  &
                                  C_last = data%A_s, X_last = data%H_s,        &
                                  Y_last = data%Y_last, Z_last = data%Z_last,  &
                                  C_stat = C_stat, B_Stat = B_Stat )
          END IF
        END IF

!  constraint/variable exit ststus not required

      ELSE
        IF ( prob%Hessian_kind == 0 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  G = prob%G )
          END IF
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  X0 = prob%X0 )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  X0 = prob%X0, G = prob%G )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  WEIGHT = prob%WEIGHT, X0 = prob%X0 )
          ELSE
            CALL LSQP_solve_main( data%dims, prob%n, prob%m,                   &
                                  prob%A%val, prob%A%col, prob%A%ptr,          &
                                  prob%C_l, prob%C_u, prob%X_l, prob%X_u,      &
                                  prob%C, prob%X, prob%Y, prob%Z, data%HX,     &
                                  data%GRAD_L, data%DIST_X_l, data%DIST_X_u,   &
                                  data%Z_l, data%Z_u, data%BARRIER_X,          &
                                  data%Y_l, data%DY_l,                         &
                                  data%DIST_C_l, data%Y_u, data%DY_u,          &
                                  data%DIST_C_u, data%C, data%BARRIER_C,       &
                                  data%SCALE_C, data%DELTA, data%RHS,          &
                                  data%DZ_l, data%DZ_u,                        &
                                  prob%f, data%DELTA_cor,                      &
                                  data%H_sbls, data%A_sbls, data%C_sbls,       &
                                  data%DY_cor_l, dy_l_lower, dy_l_upper,       &
                                  data%DY_cor_u, dy_u_lower, dy_u_upper,       &
                                  data%DZ_cor_l, dz_l_lower, dz_l_upper,       &
                                  data%DZ_cor_u, dz_u_lower, dz_u_upper,       &
                                  data%SBLS_data, prefix,                      &
                                  control, inform, data%SBLS_control,          &
                                  prob%Hessian_kind, prob%gradient_kind,       &
                                  WEIGHT = prob%WEIGHT, X0 = prob%X0,          &
                                  G = prob%G )
          END IF
        END IF
      END IF
      inform%time%analyse = inform%time%analyse +                              &
        inform%FDC_inform%time%analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%FDC_inform%time%clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%FDC_inform%time%factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%FDC_inform%time%clock_factorize

      inform%obj = inform%obj + half * fixed_sum

!     write(6,*) ' c_stat ', C_stat( : prob%m )

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( stat_required ) THEN
          C_stat( prob%m + 1 : data%QPP_map_freed%m ) = 0
          CALL SORT_inverse_permute( data%QPP_map_freed%m,                     &
                                     data%QPP_map_freed%c_map,                 &
                                     IX = C_stat( : data%QPP_map_freed%m ) )
          B_stat( prob%n + 1 : data%QPP_map_freed%n ) = - 1
          CALL SORT_inverse_permute( data%QPP_map_freed%n,                     &
                                     data%QPP_map_freed%x_map,                 &
                                     IX = B_stat( : data%QPP_map_freed%n ) )
        END IF
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform, prob,           &
                          get_all = .TRUE.)
!       CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO
      END IF
      data%tried_to_remove_deps = .FALSE.

!  Retore the problem to its original form

  700 CONTINUE
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        data%IW( : prob%n + 1 ) = 0
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( stat_required ) THEN
          C_stat( prob%m + 1 : data%QPP_map%m ) = 0
          CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,       &
                                     IX = C_stat( : data%QPP_map%m ) )
          B_stat( prob%n + 1 : data%QPP_map%n ) = - 1
          CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,       &
                                     IX = B_stat( : data%QPP_map%n ) )
        END IF

!  Full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform,                     &
                            prob, get_f = .TRUE., get_g = .TRUE.,              &
                            get_x_bounds = .TRUE., get_c_bounds = .TRUE.,      &
                            get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,    &
                            get_c = .TRUE., get_A = .TRUE., get_H = .TRUE. )

!  Restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  Recover solution

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_x = .TRUE., get_y = .TRUE.,                    &
                            get_z = .TRUE., get_c = .TRUE. )
        END IF
!       CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  Compute total time

  800 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',   &
    &   /, A, 14X, ' =                  LSQP total time                  =',   &
    &   /, A, 14X, ' =', 16X, 0P, F12.2, 23x, '='                              &
    &   /, A, 14X, ' =    preprocess    analyse    factorize     solve   =',   &
    &   /, A, 14X, ' =', 4F12.2, 3x, '=',                                      &
    &   /, A, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=') ")&
        prefix, prefix, prefix, inform%time%total, prefix, prefix,             &
        inform%time%preprocess, inform%time%analyse, inform%time%factorize,    &
        inform%time%solve, prefix

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving LSQP_solve ' )" ) prefix
      RETURN

!  Allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -LSQP_solve-', /,  A,                          &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving LSQP_solve ' )" ) prefix
      RETURN

!  Non-executable statements

 2010 FORMAT( ' ', /, A, '   **  Error return ', I0, ' from LSQP ' )

!  End of LSQP_solve

      END SUBROUTINE LSQP_solve

!-*-*-*-*-*-   L S Q P _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LSQP_solve_main( dims, n, m, A_val, A_col, A_ptr,             &
                                  C_l, C_u, X_l, X_u, C_RES, X, Y, Z,          &
                                  HX, GRAD_L,                                  &
                                  DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X,     &
                                  Y_l, DY_l, DIST_C_l, Y_u, DY_u, DIST_C_u,    &
                                  C, BARRIER_C, SCALE_C, DELTA, RHS, DZ_l,     &
                                  DZ_u, f, DELTA_cor,                          &
                                  H_sbls, A_sbls, C_sbls,                      &
                                  DY_cor_l, dy_l_lower, dy_l_upper,            &
                                  DY_cor_u, dy_u_lower, dy_u_upper,            &
                                  DZ_cor_l, dz_l_lower, dz_l_upper,            &
                                  DZ_cor_u, dz_u_lower, dz_u_upper,            &
                                  SBLS_data, prefix, control, inform,          &
                                  SBLS_control,                                &
                                  Hessian_kind, gradient_kind, WEIGHT, X0, G,  &
                                  C_last, X_last, Y_last, Z_last,              &
                                  C_stat, B_Stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimizes the linear/separable quadratic objective function
!
!    1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  subject to the constraints
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!    and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse.
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
!  dims is a structure of type LSQP_data_type, whose components hold SCALAR
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
!  C_RES is a REAL array of length m, which need not be set on entry. On exit,
!   the i-th component of C_RES will contain (A * x)_i, for i = 1, .... , m.
!
!  X is a REAL array of length n, which must be set by
!   the user on entry to LSQP_solve to give an initial estimate of the
!   optimization parameters, x. The i-th component of X should contain
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need
!   not satisfy the simple bound constraints and may be perturbed by
!   LSQP_solve prior to the start of the minimization.  Any estimate which is
!   closer to one of its bounds than control%prfeas may be reset to try to
!   ensure that it is at least control%prfeas from its bounds. On exit from
!   LSQP_solve, X will contain the best estimate of the optimization
!   parameters found
!
!  Y is a REAL array of length m, which must be set by the user
!   on entry to LSQP_solve to give an initial estimates of the
!   optimal Lagrange multipiers, y. The i-th component of Y
!   should contain the initial estimate of y_i, for i = 1, .... , m.
!   Any estimate which is smaller than control%dufeas may be
!   reset to control%dufeas. The dual variable for any variable with both
!   On exit from LSQP_solve, Y will contain the best estimate of
!   the Lagrange multipliers found
!
!  Z, is a REAL array of length n, which must be set by
!   on entry to LSQP_solve to hold the values of the the dual variables
!   associated with the simple bound constraints.
!   Any estimate which is smaller than control%dufeas may be
!   reset to control%dufeas. The dual variable for any variable with both
!   infinite lower and upper bounds need not be set. On exit from
!   LSQP_solve, Z will contain the best estimates obtained
!
!  control and inform are exactly as for LSQP_solve
!
!  Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. WEIGHT (see below) need not be set
!
!     1  all the weights will be one. WEIGHT (see below) need not be set
!
!     any other value - the weights will be those given by WEIGHT (see below)
!
!   WEIGHT is an optional REAL array, which need only be included if
!    Hessian_kind is not 0 or 1. If this is so, it must be of length at least
!    n, and contain the weights W for the objective function.
!
!   X0 is an optional REAL array, which need only be included if
!    Hessian_kind is not 0. If this is so, it must be of length at least
!    n, and contain the shifts X^0 for the objective function.
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
!  The remaining arguments are used as internal workspace, and need not be
!  set on entry
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, Hessian_kind, gradient_kind
      INTEGER, INTENT( IN ) :: dy_l_lower, dy_l_upper, dy_u_lower, dy_u_upper
      INTEGER, INTENT( IN ) :: dz_l_lower, dz_l_upper, dz_u_lower, dz_u_upper
      REAL ( KIND = wp ), INTENT( IN ) :: f
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: B_stat
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: WEIGHT, X0
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C_RES
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: Y_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: Z_last
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%v_e ) :: DELTA, RHS
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%v_e ) :: HX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: GRAD_L
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) ::  Z_l, DZ_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_u_start : n ) :: Z_u, DZ_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DY_l, DIST_C_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DY_u, DIST_C_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, BARRIER_C, SCALE_C

      REAL ( KIND = wp ), DIMENSION( dy_l_lower : dy_l_upper ) :: DY_cor_l
      REAL ( KIND = wp ), DIMENSION( dy_u_lower : dy_u_upper ) :: DY_cor_u
      REAL ( KIND = wp ), DIMENSION( dz_l_lower : dz_l_upper ) :: DZ_cor_l
      REAL ( KIND = wp ), DIMENSION( dz_u_lower : dz_u_upper ) :: DZ_cor_u
      REAL ( KIND = wp ), DIMENSION( : ) :: DELTA_cor

      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls, A_sbls
      TYPE ( SMT_type ), INTENT( IN ) :: C_sbls

      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
      TYPE ( LSQP_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: SBLS_data
      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: SBLS_control

!  Parameters

      REAL ( KIND = wp ), PARAMETER :: eta = tenm4
      REAL ( KIND = wp ), PARAMETER :: sigma_max = point01
!     REAL ( KIND = wp ), PARAMETER :: sigma_max = point1
      REAL ( KIND = wp ), PARAMETER :: gamma_b0 = tenm5
      REAL ( KIND = wp ), PARAMETER :: gamma_f0 = tenm5
!     REAL ( KIND = wp ), PARAMETER :: gamma_b0 = 0.9
!     REAL ( KIND = wp ), PARAMETER :: gamma_f0 = 0.9
      REAL ( KIND = wp ), PARAMETER :: degen_tol = tenm5

!  Local variables

      INTEGER :: a_ne, i, l, start_print, stop_print, print_level
      INTEGER :: j, nbnds, nbnds_x, nbnds_c, muzero_fixed
      INTEGER :: nbact
      INTEGER :: out, error, it_best, infeas_max
      REAL :: time_record, time_start, time_now, time_solve
      REAL ( KIND = wp ) :: clock_record, clock_start, clock_now, clock_solve
      REAL ( KIND = wp ) :: pjgnrm, mu, pmax, amax, gamma_f, nu
      REAL ( KIND = wp ) :: alpha, alpha_b, alpha_f, sigma, slope, gamma_b
      REAL ( KIND = wp ) :: gi, merit, merit_model, merit_trial
      REAL ( KIND = wp ) :: res_prim, res_dual, slknes, slkmin, potential_trial
      REAL ( KIND = wp ) :: slknes_x, slknes_c, slkmax_x, slkmax_c, slknes_req
      REAL ( KIND = wp ) :: slkmin_x, slkmin_c, merit_best, reduce_infeas
      REAL ( KIND = wp ) :: prfeas, dufeas, p_min, p_max, d_min, d_max
      REAL ( KIND = wp ) :: step, one_minus_alpha
      REAL ( KIND = wp ) :: pmax_cor, pivot_tol, relative_pivot_tol, balance
!     REAL ( KIND = wp ) :: errorc, errorg
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: set_printp, printt, printi, printp, printe, printd, printw
      LOGICAL :: use_corrector, maxpiv, stat_required
      LOGICAL :: get_stat, use_scale_c = .FALSE.
      CHARACTER ( LEN = 1 ) :: re, mo, co
      INTEGER :: sif = 50
!     LOGICAL :: generate_sif = .TRUE.
      LOGICAL :: generate_sif = .FALSE.

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering LSQP_solve_main ' )" ) prefix

!  move to argument list

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) n, m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) f
        IF ( PRESENT( G ) )                                                    &
          WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) G( : n )
        IF ( hessian_kind == 1 ) THEN
          WRITE( control%out, "( ' H  = 1.0' )" )
        ELSE IF ( hessian_kind /= 0 ) THEN
          WRITE( control%out, "( ' H  =' )" )
          WRITE( control%out, "( ( 4( ES12.4 ) ) )" ) ( WEIGHT( i ), i = 1, n )
        END IF
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

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( generate_sif .AND. PRESENT( G ) ) THEN
        WRITE( sif, "( 'NAME          LSQP_OUT', //, 'VARIABLES', / )" )
        DO i = 1, n
          WRITE( sif, "( '    X', I8 )" ) i
        END DO

        WRITE( sif, "( /, 'GROUPS', / )" )
        DO i = 1, n
          IF ( G( i ) /= zero )                                                &
            WRITE( sif, "( ' N  OBJ      ', ' X', I8, ' ', ES12.5 )" ) i, G( i )
        END DO
        DO i = 1, dims%c_l_start - 1
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
          END DO
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
          END DO
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
          END DO
        END DO

        WRITE( sif, "( /, 'CONSTANTS', / )" )
        DO i = 1, dims%c_l_end
          IF ( C_l( i ) /= zero )                                              &
          WRITE( sif, "( '    RHS      ', ' C', I8, ' ', ES12.5 )" ) i, C_l( i )
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          IF ( C_u( i ) /= zero )                                              &
          WRITE( sif, "( '    RHS      ', ' C', I8, ' ', ES12.5 )" ) i, C_u( i )
        END DO

        IF ( dims%c_u_start <= dims%c_l_end ) THEN
          WRITE( sif, "( /, 'RANGES', / )" )
          DO i = dims%c_u_start, dims%c_l_end
            WRITE( sif, "( '    RANGE    ', ' C', I8, ' ', ES12.5 )" )        &
              i, C_u( i ) - C_l( i )
          END DO
        END IF

        IF ( dims%x_free /= 0 .OR. dims%x_l_start <= n ) THEN
          WRITE( sif, "( /, 'BOUNDS', /, ' FR BND       ''DEFAULT''' )" )
          DO i = dims%x_free + 1, dims%x_l_start - 1
            WRITE( sif, "( ' LO BND       X', I8, ' ', ES12.5 )" ) i, zero
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            WRITE( sif, "( ' LO BND       X', I8, ' ', ES12.5 )" ) i, X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            WRITE( sif, "( ' UP BND       X', I8, ' ', ES12.5 )" ) i, X_u( i )
          END DO
          DO i = dims%x_u_end + 1, n
            WRITE( sif, "( ' UP BND       X', I8, ' ', ES12.5 )" ) i, zero
          END DO
        END IF

        WRITE( sif, "( /, 'START POINT', / )" )
        DO i = 1, n
          IF ( X( i ) /= zero )                                                &
            WRITE( sif, "( ' V  START    ', ' X', I8, ' ', ES12.5 )" ) i, X( i )
        END DO

        WRITE( sif, "( /, 'ENDATA' )" )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  Initialize time

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  ===========================
!  Control the output printing
!  ===========================

      print_level = 0
      IF ( control%start_print < 0 ) THEN
        start_print = - 1
      ELSE
        start_print = control%start_print
      END IF

      IF ( control%stop_print < 0 ) THEN
        stop_print = control%maxit
      ELSE
        stop_print = control%stop_print
      END IF

      error = control%error ; out = control%out

      set_printe = error > 0 .AND. control%print_level >= 1

!  Basic single line of output per iteration

      set_printi = out > 0 .AND. control%print_level >= 1

!  As per printi, but with additional timings for various operations

      set_printt = out > 0 .AND. control%print_level >= 2

!  As per printt but also with an indication of where in the code we are

      set_printp = out > 0 .AND. control%print_level >= 3

!  As per printp but also with details of innner iterations

      set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

      set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printp = set_printp ; printw = set_printw ; printd = set_printd
        print_level = control%print_level
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printp = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
        print_level = 0
      END IF

      IF ( SBLS_control%factorization < 0 .OR.                                 &
           SBLS_control%factorization > 2 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A,' factor = ', I0, ' out of range [0,2]. Reset to 0' )" )        &
          prefix, SBLS_control%factorization
        SBLS_control%factorization = 0
      END IF

!  If there are no variables, exit

      IF ( n == 0 ) THEN
        i = COUNT( ABS( C_l( : dims%c_equality ) ) > control%stop_p ) +        &
            COUNT( C_l( dims%c_l_start : dims%c_l_end ) > control%stop_p ) +   &
            COUNT( C_u( dims%c_u_start : dims%c_u_end ) < - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        C_RES = zero ; Y = zero
        inform%obj = zero
        GO TO 810
      END IF

!  Record array size

      a_ne = A_ptr( m + 1 ) - 1

!  Set control parameters

      muzero_fixed = control%muzero_fixed
      prfeas = MAX( control%prfeas, epsmch )
      dufeas = MAX( control%dufeas, epsmch )
      reduce_infeas = MAX( epsmch,                                             &
                           MIN( control%reduce_infeas ** 2, one - epsmch ) )
      infeas_max = MAX( 0, control%infeas_max )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )
      IF ( stat_required ) THEN
        B_stat  = 0
        C_stat( : dims%c_equality ) = - 1
        C_stat( dims%c_equality + 1 : ) = 0
      END IF
      get_stat = .FALSE.

!  If required, write out the problem

      IF ( printd ) WRITE( out, "( A, A6, /, ( 4( 2I5, ES10.2 ) ) )" ) prefix, &
     &  ' a ', ( ( i, A_col( l ), A_val( l ), l = A_ptr( i ),                  &
          A_ptr( i + 1 ) - 1 ), i = 1, m )

      IF ( control%balance_initial_complentarity ) THEN
        IF ( control%muzero <= zero ) THEN
          balance = one
        ELSE
          balance = control%muzero
        END IF
      END IF

!  Record the initial point, move the starting point away from any bounds,
!  and move that for dual variables away from zero

      nbnds_x = 0

!  The variable is free

      IF ( printd ) THEN
        WRITE( out, "( /, A, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X,      &
       &       'z_l', 9X, 'z_u')") prefix
        DO i = 1, dims%x_free
          WRITE( out, "( A, I6, ES12.4, 4( '      -    ') )" ) prefix, i, X( i )
        END DO
      END IF

!  The variable is a non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        nbnds_x = nbnds_x + 1
        X( i ) = MAX( X( i ), prfeas )
        IF ( control%balance_initial_complentarity ) THEN
          Z_l( i ) = balance / X( i )
        ELSE
          Z_l( i ) = MAX( ABS( Z( i ) ), dufeas )
        END IF
        IF ( printd ) WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,   &
       &  '      -     ' )" ) prefix, i, X( i ), zero, Z_l( i )
      END DO

!  The variable has just a lower bound

      DO i = dims%x_l_start, dims%x_u_start - 1
        nbnds_x = nbnds_x + 1
        X( i ) = MAX( X( i ), X_l( i ) + prfeas )
        DIST_X_l( i ) = X( i ) - X_l( i )
        IF ( control%balance_initial_complentarity ) THEN
          Z_l( i ) = balance / DIST_X_l( i )
        ELSE
          Z_l( i ) = MAX( ABS( Z( i ) ), dufeas )
        END IF
        IF ( printd ) WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,   &
       &  '      -     ' )" ) prefix, i, X( i ), X_l( i ), Z_l( i )
      END DO

!  The variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN
          inform%status = GALAHAD_error_bad_bounds
          GO TO 700
        END IF
        nbnds_x = nbnds_x + 2
        IF ( X_l( i ) + prfeas >= X_u( i ) - prfeas ) THEN
          X( i ) = half * ( X_l( i ) + X_u( i ) )
        ELSE
          X( i ) = MIN( MAX( X( i ), X_l( i ) + prfeas ), X_u( i ) - prfeas )
        END IF
        DIST_X_l( i ) = X( i ) - X_l( i ) ; DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( control%balance_initial_complentarity ) THEN
          Z_l( i ) = balance / DIST_X_l( i )
          Z_u( i ) = - balance / DIST_X_u( i )
        ELSE
          Z_l( i ) = MAX(   ABS( Z( i ) ),   dufeas )
          Z_u( i ) = MIN( - ABS( Z( i ) ), - dufeas )
        END IF
        IF ( printd ) WRITE( out, "( A, I6, 5ES12.4 )" )                       &
             prefix, i, X( i ), X_l( i ), X_u( i ), Z_l( i ), Z_u( i )
      END DO

!  The variable has just an upper bound

      DO i = dims%x_l_end + 1, dims%x_u_end
        nbnds_x = nbnds_x + 1
        X( i ) = MIN( X( i ), X_u( i ) - prfeas )
        DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( control%balance_initial_complentarity ) THEN
          Z_u( i ) = - balance / DIST_X_u( i )
        ELSE
          Z_u( i ) = MIN( - ABS( Z( i ) ), - dufeas )
        END IF
        IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,    &
       &  '      -     ', ES12.4 )" ) prefix, i, X( i ), X_u( i ), Z_u( i )
      END DO

!  The variable is a non-positivity

      DO i = dims%x_u_end + 1, n
        nbnds_x = nbnds_x + 1
        X( i ) = MIN( X( i ), - prfeas )
        IF ( control%balance_initial_complentarity ) THEN
          Z_u( i ) = balance / X( i )
        ELSE
          Z_u( i ) = MIN( - ABS( Z( i ) ), - dufeas )
        END IF
        IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,    &
       &  '      -     ',  ES12.4 )" ) prefix, i, X( i ), zero, Z_u( i )
      END DO

!  Compute the value of the constraint, and their residuals

      nbnds_c = 0
      IF ( m > 0 ) THEN
        C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
        C_RES( dims%c_l_start : dims%c_u_end ) = zero
        CALL LSQP_AX( m, C_RES, m, a_ne, A_val, A_col, A_ptr,                  &
                      n, X, '+ ' )
        IF ( printd ) THEN
          WRITE( out, "( /, A, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X,     &
         &     'y_l', 9X, 'y_u' )") prefix
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( A, I6, 3ES12.4 )" )                                 &
              prefix, i, C_RES( i ), C_l( i ), C_u( i )
          END DO
        END IF

!  The constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          nbnds_c = nbnds_c + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MAX( C_RES( i ) / SCALE_C( i ), C_l( i ) + prfeas )
          DIST_C_l( i ) = C( i ) - C_l( i )
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          IF ( control%balance_initial_complentarity ) THEN
            Y_l( i ) = balance / DIST_C_l( i )
          ELSE
            Y_l( i ) = MAX( ABS( SCALE_C( i ) * Y( i ) ),  dufeas )
          END IF
          IF ( printd ) WRITE( out,  "( A, I6, 2ES12.4, '      -     ',       &
         &  ES12.4, '      -    ' )" ) prefix, i, C_RES( i ), C_l( i ), Y_l( i )
        END DO

!  The constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

          IF ( C_u( i ) - C_l( i ) <= epsmch ) THEN
            inform%status = GALAHAD_error_bad_bounds
            GO TO 700
          END IF
          nbnds_c = nbnds_c + 2

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )
          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          IF ( C_l( i ) + prfeas >= C_u( i ) - prfeas ) THEN
            C( i ) = half * ( C_l( i ) + C_u( i ) )
          ELSE
            C( i ) = MIN( MAX( C_RES( i ) / SCALE_C( i ), C_l( i ) + prfeas ), &
                               C_u( i ) - prfeas )
          END IF
          DIST_C_l( i ) = C( i ) - C_l( i )
          DIST_C_u( i ) = C_u( i ) - C( i )
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          IF ( control%balance_initial_complentarity ) THEN
            Y_l( i ) = balance / DIST_C_l( i )
            Y_u( i ) = - balance / DIST_C_u( i )
          ELSE
            Y_l( i ) = MAX(   ABS( SCALE_C( i ) * Y( i ) ),   dufeas )
            Y_u( i ) = MIN( - ABS( SCALE_C( i ) * Y( i ) ), - dufeas )
          END IF
          IF ( printd ) WRITE( out, "( A, I6, 5ES12.4 )" )                     &
            prefix, i, C_RES( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i )
        END DO

!  The constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          nbnds_c = nbnds_c + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MIN( C_RES( i ) / SCALE_C( i ), C_u( i ) - prfeas )
          DIST_C_u( i ) = C_u( i ) - C( i )
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          IF ( control%balance_initial_complentarity ) THEN
            Y_u( i ) = - balance / DIST_C_u( i )
          ELSE
            Y_u( i ) = MIN( - ABS( SCALE_C( i ) * Y( i ) ), - dufeas )
          END IF
          IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,  &
         &  '      -     ', ES12.4 )") prefix, i, C_RES( i ), C_u( i ), Y_u( i )
        END DO
        res_prim = MAXVAL( ABS( C_RES( : dims%c_u_end ) ) )
      ELSE
        res_prim = zero
      END IF

!  Set useful pointers to stop Compaq optimization inefficiencies

!      ptr_GRAD_L => GRAD_L( dims%x_s : dims%x_e )
!      ptr_HX => HX( : n )
!      ptr_HX_v => HX( : dims%v_e )
!      ptr_DELTA_x => DELTA( dims%x_s : dims%x_e )
!     ptr_DELTA_c => DELTA( dims%c_s : dims%c_e )
!      ptr_DELTA_y => DELTA( dims%y_s : dims%y_e )
!      ptr_RHS_x => RHS( dims%x_s : dims%x_e )
!      ptr_RHS_c => RHS( dims%c_s : dims%c_e )
!      ptr_RHS_y => RHS( dims%y_s : dims%y_e )
      IF ( control%use_corrector ) THEN
!        ptr_DELTA_cor_x => DELTA_cor( dims%x_s : dims%x_e )
!        ptr_DELTA_cor_c => DELTA_cor( dims%c_s : dims%c_e )
!        ptr_DELTA_cor_y => DELTA_cor( dims%y_s : dims%y_e )
      END IF

!  Find the max-norm of the residual

      nbnds = nbnds_x + nbnds_c
      IF ( printi .AND. m > 0 .AND. dims%c_l_start <= dims%c_u_end )           &
        WRITE( out, "( A, ' largest/smallest scale factor ', 2ES12.4 )" )      &
          prefix, MAXVAL( SCALE_C ), MINVAL( SCALE_C )

!  Compute the complementary slackness

!       DO i = dims%x_free + 1, dims%x_l_start - 1
!         write(6,"(I6, ' x lower', 2ES12.4)" ) i, X( i ), Z_l( i )
!       END DO
!       DO i = dims%x_l_start, dims%x_l_end
!         write(6,"(I6, ' x lower', 2ES12.4)" ) i, DIST_X_l( i ), Z_l( i )
!       END DO
!       DO i = dims%x_u_start, dims%x_u_end
!         write(6,"(I6, ' x upper', 2ES12.4)" ) i, - DIST_X_u( i ), Z_u( i )
!       END DO
!       DO i = dims%x_u_end + 1, n
!         write(6,"(I6, ' x upper', 2ES12.4)" ) i, X( i ), Z_u( i )
!       END DO

!       DO i = dims%c_l_start, dims%c_l_end
!         write(6,"(I6, ' c lower', 2ES12.4)" ) i, DIST_C_l( i ), Y_l( i )
!       END DO
!       DO i = dims%c_u_start, dims%c_u_end
!         write(6,"(I6, ' c upper', 2ES12.4)" ) i, - DIST_C_u( i ), Y_u( i )
!       END DO

      slknes_x = DOT_PRODUCT( X( dims%x_free + 1 : dims%x_l_start - 1 ),       &
                              Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ) +  &
                 DOT_PRODUCT( DIST_X_l( dims%x_l_start : dims%x_l_end ),       &
                              Z_l( dims%x_l_start : dims%x_l_end ) ) -         &
                 DOT_PRODUCT( DIST_X_u( dims%x_u_start : dims%x_u_end ),       &
                              Z_u( dims%x_u_start : dims%x_u_end ) ) +         &
                 DOT_PRODUCT( X( dims%x_u_end + 1 : n ),                       &
                              Z_u( dims%x_u_end + 1 : n ) )
      slknes_c = DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),       &
                              Y_l( dims%c_l_start : dims%c_l_end ) ) -         &
                 DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),       &
                              Y_u( dims%c_u_start : dims%c_u_end ) )
      slknes = slknes_x + slknes_c

      slkmin_x = MIN( MINVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) *      &
                              Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ),   &
                      MINVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) *      &
                              Z_l( dims%x_l_start : dims%x_l_end ) ),          &
                      MINVAL( - DIST_X_u( dims%x_u_start : dims%x_u_end ) *    &
                              Z_u( dims%x_u_start : dims%x_u_end ) ),          &
                      MINVAL( X( dims%x_u_end + 1 : n ) *                      &
                              Z_u( dims%x_u_end + 1 : n ) ) )
      slkmin_c = MIN( MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *      &
                              Y_l( dims%c_l_start : dims%c_l_end ) ),          &
                      MINVAL( - DIST_C_u( dims%c_u_start : dims%c_u_end ) *    &
                              Y_u( dims%c_u_start : dims%c_u_end ) ) )
      slkmin = MIN( slkmin_x, slkmin_c )

      slkmax_x = MAX( MAXVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) *      &
                              Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ),   &
                      MAXVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) *      &
                              Z_l( dims%x_l_start : dims%x_l_end ) ),          &
                      MAXVAL( - DIST_X_u( dims%x_u_start : dims%x_u_end ) *    &
                              Z_u( dims%x_u_start : dims%x_u_end ) ),          &
                      MAXVAL( X( dims%x_u_end + 1 : n ) *                      &
                              Z_u( dims%x_u_end + 1 : n ) ) )
      slkmax_c = MAX( MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *      &
                              Y_l( dims%c_l_start : dims%c_l_end ) ),          &
                      MAXVAL( - DIST_C_u( dims%c_u_start : dims%c_u_end ) *    &
                              Y_u( dims%c_u_start : dims%c_u_end ) ) )

!     WRITE(6,2100) ' >0 ', X( dims%x_free + 1 : dims%x_l_start - 1 )
!     WRITE(6,2100) ' >l ', DIST_X_l( dims%x_l_start : dims%x_l_end )
!     WRITE(6,2100) ' <u ', DIST_X_u( dims%x_u_start : dims%x_u_end )
!     WRITE(6,2100) ' <0 ', - X( dims%x_u_end + 1 : n )
!     stop



      p_min = MIN( MINVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) ),        &
                   MINVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) ),        &
                   MINVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) ),        &
                   MINVAL( - X( dims%x_u_end + 1 : n ) ),                      &
                   MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

      p_max = MAX( MAXVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) ),        &
                   MAXVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) ),        &
                   MAXVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) ),        &
                   MAXVAL( - X( dims%x_u_end + 1 : n ) ),                      &
                   MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

      d_min = MIN( MINVAL(   Z_l( dims%x_free + 1 : dims%x_l_end ) ),          &
                   MINVAL( - Z_u( dims%x_u_start : n ) ),                      &
                   MINVAL(   Y_l( dims%c_l_start : dims%c_l_end ) ),           &
                   MINVAL( - Y_u( dims%c_u_start : dims%c_u_end ) ) )

      d_max = MAX( MAXVAL(   Z_l( dims%x_free + 1 : dims%x_l_end ) ),          &
                   MAXVAL( - Z_u( dims%x_u_start : n ) ),                      &
                   MAXVAL(   Y_l( dims%c_l_start : dims%c_l_end ) ),           &
                   MAXVAL( - Y_u( dims%c_u_start : dims%c_u_end ) ) )

!  Record the slackness and the deviation from the central path

      IF ( nbnds_x > 0 ) THEN
        slknes_x = slknes_x / nbnds_x
      ELSE
        slknes_x = zero
      END IF

      IF ( nbnds_c > 0 ) THEN
        slknes_c = slknes_c / nbnds_c
      ELSE
        slknes_c = zero
      END IF

      IF ( nbnds > 0 ) THEN
        gamma_f = gamma_f0 * slknes ; slknes = slknes / nbnds
        gamma_b = gamma_b0 * slkmin / slknes
      ELSE
        gamma_f = zero ; slknes = zero ; gamma_b = zero
      END IF

      IF ( printt .AND. nbnds > 0 ) WRITE( out, 2130 )                         &
        prefix, slknes, prefix, slknes_x, prefix, slknes_c, prefix, slkmin_x,  &
        slkmax_x, prefix, slkmin_c, slkmax_c, prefix, p_min, p_max, prefix,    &
        d_min, d_max

!  Compute the initial objective value

      IF ( Hessian_kind == 0 ) THEN
        inform%obj = f
      ELSE IF ( Hessian_kind == 1 ) THEN
        inform%obj = f + half * SUM( ( X - X0 ) ** 2 )
      ELSE
        inform%obj = f + half * SUM( ( WEIGHT * ( X - X0 ) ) ** 2 )
      END IF

      IF ( gradient_kind == 1 ) THEN
        inform%obj = inform%obj + SUM( X )
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
      END IF

!  Find the largest components of A

      IF ( a_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : a_ne ) ) )
      ELSE
        amax = zero
      END IF

      IF ( printi ) WRITE( out, "( A, '  maximum element of A = ', ES12.4 )" ) &
        prefix, amax

!  Test to see if we are feasible

      inform%feasible = res_prim <= control%stop_p
      pjgnrm = infinity

      IF ( inform%feasible ) THEN
        IF ( printi ) WRITE( out, 2070 ) prefix
        IF ( control%just_feasible ) THEN
          inform%status = GALAHAD_ok
          GO TO 500
        END IF
        IF ( Hessian_kind == 0 .AND. gradient_kind == 0 )                      &
          inform%potential = LSQP_potential_value( dims, n,                    &
                                   X, DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u )
      END IF

!  Set the initial barrier parameter

      sigma = sigma_max ; nu = one
      IF ( control%muzero < zero ) THEN
        mu = sigma * slknes
      ELSE
        mu = control%muzero
      END IF
      slknes_req = slknes

!  Compute the gradient of the Lagrangian function.

      CALL LSQP_Lagrangian_gradient( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,     &
                                     a_ne, A_val, A_col, A_ptr,                &
                                     DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,   &
                                     GRAD_L( dims%x_s : dims%x_e ),            &
                                     control%getdua, dufeas,                   &
                                     Hessian_kind, gradient_kind,              &
                                     WEIGHT = WEIGHT, X0 = X0, G = G )

!  Evaluate the merit function

      merit = LSQP_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,          &
                                 DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,       &
                                 GRAD_L( dims%x_s : dims%x_e ), C_RES, res_dual)

!  Prepare for the major iteration

      inform%iter = 0 ; inform%nfacts = 0
      IF ( printt ) WRITE( out, "( /, A, ' merit function value = ',           &
     &     ES12.4 )" ) prefix, merit

      IF ( n == 0 ) THEN
        inform%status = GALAHAD_ok ; GO TO 600
      END IF
      merit_best = merit ; it_best = 0

!  Test for convergence

      IF ( res_prim <= control%stop_p .AND. res_dual <= control%stop_d .AND.   &
           slknes_req <= control%stop_c ) THEN
        inform%status = GALAHAD_ok ; GO TO 600
      END IF

!  ===================================================
!  Analyse the sparsity pattern of the required matrix
!  ===================================================

      re = ' ' ; mo = ' ' ;  nbact = 0
      pivot_tol = SBLS_control%SLS_control%relative_pivot_tolerance
      relative_pivot_tol = pivot_tol
      maxpiv = pivot_tol >= half

      IF ( printi ) WRITE( out,                                                &
          "(  /, A, '  Primal    convergence tolerance = ', ES12.4,            &
         &    /, A, '  Dual      convergence tolerance = ', ES12.4,            &
         &    /, A, '  Slackness convergence tolerance = ', ES12.4 )" )        &
          prefix, control%stop_p, prefix, control%stop_d, prefix, control%stop_c

!  complete A

      DO i = 1, dims%nc
        A_sbls%val( a_ne + i ) = - SCALE_C( dims%c_equality + i )
      END DO

!  ---------------------------------------------------------------------
!  ---------------------- Start of Major Iteration ---------------------
!  ---------------------------------------------------------------------

      DO

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Test for Optimality   -*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  Print a summary of the iteration

        CALL CLOCK_TIME( clock_now ) ; clock_now = clock_now - clock_start
        IF ( printi ) THEN
          IF ( inform%iter > 0 ) THEN
            IF ( printt .OR. ( printi .AND.                                    &
               inform%iter == start_print ) ) WRITE( out, 2000 ) prefix
            WRITE( out, 2030 ) prefix, inform%iter, re, res_prim, res_dual,    &
             slknes_req, inform%obj, alpha, co, mo, mu, nbact, clock_now
          ELSE
            WRITE( out, 2000 ) prefix
            WRITE( out, 2020 ) prefix, inform%iter, re, res_prim, res_dual,    &
              slknes_req, inform%obj, mu, clock_now
          END IF

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' X ', X
            WRITE( out, 2100 ) prefix, ' Z_l ',Z_l( dims%x_l_start:dims%x_l_end)
            WRITE( out, 2100 ) prefix, ' Z_u ',Z_u( dims%x_u_start:dims%x_u_end)
          END IF
        END IF

!  Test for optimality

        IF ( res_prim <= control%stop_p .AND. res_dual <= control%stop_d .AND. &
             slknes_req <= control%stop_c ) THEN
          inform%status = GALAHAD_ok ; GO TO 600
        END IF

!  Test to see if more than maxit iterations have been performed

        inform%iter = inform%iter + 1
        IF ( inform%iter > control%maxit ) THEN
          inform%status = GALAHAD_error_max_iterations ; GO TO 600
        END IF

!  Check that the CPU time limit has not been reached

        CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now )
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - time_start > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit ; GO TO 600
        END IF

        IF ( inform%iter == start_print ) THEN
          printe = set_printe ; printi = set_printi ; printt = set_printt
          printw = set_printw ; printd = set_printd
          print_level = control%print_level
        END IF

        IF ( inform%iter == stop_print + 1 ) THEN
          printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
          printw = .FALSE. ; printd = .FALSE.
          print_level = 0
        END IF

!       WRITE( 6, "( ' start, stop print, iter ', 3I8 )" )                     &
!         start_print, stop_print, inform%iter

!  Test to see whether the method has stalled

        IF ( merit <= reduce_infeas * merit_best ) THEN
          merit_best = merit
          it_best = 0
        ELSE
          it_best = it_best + 1
          IF ( it_best > infeas_max ) THEN
            IF ( inform%feasible ) THEN
              inform%status = GALAHAD_error_no_center ; GO TO 600
            ELSE
              IF ( printi ) WRITE( out, "( /, A, ' =============== the prob',  &
             &  'lem appears to be infeasible ====================== ', / )" ) &
               prefix
              inform%status = GALAHAD_error_primal_infeasible ; GO TO 600
            END IF
          END IF
        END IF

!  Test to see if the potential function appears to be unbounded from below

!       IF ( nbnds == 0 .AND. inform%feasible .AND. Hessian_kind == 0 .AND.    &
        IF ( inform%feasible .AND. Hessian_kind == 0 .AND.                     &
             gradient_kind == 0 ) THEN
          IF ( inform%potential < control%potential_unbounded *                &
               ( ( dims%x_l_end - dims%x_free ) +                              &
               ( n -  dims%x_u_start + 1 ) +                                   &
               ( dims%c_l_end - dims%c_l_start + 1 ) +                         &
               ( dims%c_u_end - dims%c_u_start + 1 ) ) ) THEN
            inform%status = GALAHAD_error_no_center ; GO TO 600
          END IF

!  Compute the Hessian matrix of the barrier terms

!  Special case for the analytic center

!  problem variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            BARRIER_X( i ) =  mu / X( i ) ** 2
          END DO
          DO i = dims%x_l_start, dims%x_u_start - 1
            BARRIER_X( i ) = mu / DIST_X_l( i ) ** 2
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            BARRIER_X( i ) = mu / DIST_X_l( i ) ** 2                           &
                             + mu / DIST_X_u( i ) ** 2
          END DO
          DO i = dims%x_l_end + 1, dims%x_u_end
            BARRIER_X( i ) = mu / DIST_X_u( i ) ** 2
          END DO
          DO i = dims%x_u_end + 1, n
            BARRIER_X( i ) = mu / X( i ) ** 2
          END DO

!  slack variables:

          BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero
          DO i = dims%c_l_start, dims%c_u_start - 1
            BARRIER_C( i ) = mu / DIST_C_l( i ) ** 2
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            BARRIER_C( i ) = mu / DIST_C_l( i ) ** 2                           &
                             + mu / DIST_C_u( i ) ** 2
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            BARRIER_C( i ) = mu / DIST_C_u( i ) ** 2
          END DO

!  General case

        ELSE

!  problem variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( ABS( X( i ) ) <= degen_tol .AND. printw )                     &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, X( i ), Z_l( i )
            BARRIER_X( i ) = Z_l( i ) / X( i )
          END DO
          DO i = dims%x_l_start, dims%x_u_start - 1
            IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, DIST_X_l( i ), Z_l( i )
            BARRIER_X( i ) = Z_l( i ) / DIST_X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, DIST_X_l( i ), Z_l( i )
            IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, DIST_X_u( i ), Z_u( i )
            BARRIER_X( i ) = Z_l( i ) / DIST_X_l( i ) - Z_u( i ) / DIST_X_u( i )
          END DO
          DO i = dims%x_l_end + 1, dims%x_u_end
            IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, DIST_X_u( i ), Z_u( i )
            BARRIER_X( i ) = - Z_u( i ) / DIST_X_u( i )
          END DO
          DO i = dims%x_u_end + 1, n
            IF ( ABS( X( i ) ) <= degen_tol .AND. printw )                     &
              WRITE( 6, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
                prefix, i, X( i ), Z_u( i )
            BARRIER_X( i ) = Z_u( i ) / X( i )
          END DO

!  slack variables:

          BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero
          DO i = dims%c_l_start, dims%c_u_start - 1
            IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
                prefix, i, DIST_C_l( i ), Y_l( i )
            BARRIER_C( i ) = Y_l( i ) / DIST_C_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
                prefix, i, DIST_C_l( i ), Y_l( i )
            IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
                prefix, i, DIST_C_u( i ), Y_u( i )
            BARRIER_C( i ) = Y_l( i ) / DIST_C_l( i ) - Y_u( i ) / DIST_C_u( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
                prefix, i, DIST_C_u( i ), Y_u( i )
            BARRIER_C( i ) = - Y_u( i ) / DIST_C_u( i )
          END DO
        END IF

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-*-      Factorization      -*-*-*-*-*-*-*-*-*-
!  =====================================================================

        mo = ' '

!  Only refactorize if B has changed

        re = 'r'

!  Include the values of the barrier terms

        IF ( Hessian_kind == 0 ) THEN
          H_sbls%val( 1 : dims%x_free ) = zero
          H_sbls%val(dims%x_free + 1 : n ) = BARRIER_X
        ELSE IF ( Hessian_kind == 1 ) THEN
          H_sbls%val( 1 : dims%x_free ) = one
          H_sbls%val( dims%x_free + 1 : n ) = one + BARRIER_X
        ELSE
          H_sbls%val( 1 : dims%x_free ) = WEIGHT( : dims%x_free ) ** 2
          H_sbls%val( dims%x_free + 1 : n ) =                                  &
            WEIGHT( dims%x_free + 1 : ) ** 2 + BARRIER_X
        END IF
        H_sbls%val( dims%c_s : dims%c_e ) = BARRIER_C

! ::::::::::::::::::::::::::::::
!  Factorize the required matrix
! ::::::::::::::::::::::::::::::

        IF ( printw ) WRITE( out, "( A,                                        &
       &  ' ......... factorization of KKT matrix ............... ' )" ) prefix
        CALL CLOCK_time( clock_record )
        CALL SBLS_form_and_factorize( A_sbls%n, A_sbls%m, H_sbls, A_sbls,      &
          C_sbls, sbls_data, SBLS_control, inform%SBLS_inform )
        inform%time%analyse = inform%time%analyse +                            &
          inform%SBLS_inform%SLS_inform%time%analyse
        inform%time%clock_analyse = inform%time%clock_analyse +                &
          inform%SBLS_inform%SLS_inform%time%clock_analyse
        inform%time%factorize = inform%time%factorize +                        &
          inform%SBLS_inform%SLS_inform%time%factorize
        inform%time%clock_factorize = inform%time%clock_factorize +            &
          inform%SBLS_inform%SLS_inform%time%clock_factorize
        time_solve = 0.0 ; clock_solve = 0.0

        IF ( printw ) WRITE( out, "( A,                                        &
       &  ' ............... end of factorization ............... ' )" ) prefix

        inform%nfacts = inform%nfacts + 1

!  Test that the factorization succeeded

        inform%factorization_status = inform%SBLS_inform%status
        IF ( inform%factorization_status == GALAHAD_error_preconditioner ) THEN
            IF ( printi ) WRITE( out,                                          &
               "( A, ' wrong inertia ... proceeding with caution ' )" ) prefix
        ELSE IF ( inform%factorization_status < 0 ) THEN
          IF ( printe ) WRITE( error, "( A, '   **  Error return ', I0,        &
         &  ' from ', A )" ) prefix, inform%factorization_status,              &
            'SBLS_form_and_factorize'

!  It didn't. We might have run out of options

          IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
            inform%status = GALAHAD_error_factorization ; GO TO 700

!  ... or we may change the method

          ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
            pivot_tol = relative_pivot_tol
            maxpiv = pivot_tol >= half
            SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
            SBLS_control%factorization = 2
            IF ( printi )  WRITE( out,                                         &
              "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

          ELSE
            pivot_tol = half
            maxpiv = .TRUE.
            SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
            SBLS_control%factorization = 2
            IF ( printi ) WRITE( out,                                          &
               "( A, ' Pivot tolerance increased ' )" ) prefix
          END IF
          alpha = zero ; nbact = 0
          inform%factorization_integer = - 1
          inform%factorization_real = - 1
          CYCLE

!  Record warning conditions

        ELSE
          IF (inform%factorization_status > 0 ) THEN
            IF ( printt ) THEN
              WRITE( out, "( A, '   **  Warning ', I0, ' from ', A )" )        &
              prefix, inform%SBLS_inform%status, 'SBLS_form_andfactorize'
            END IF
          END IF

!  Record the storage required

          inform%factorization_integer =                                       &
            inform%SBLS_inform%SLS_inform%integer_size_necessary
          inform%factorization_real =                                          &
            inform%SBLS_inform%SLS_inform%real_size_necessary

        END IF

        IF ( printt ) WRITE( out,                                              &
          "( A, ' real/integer space used for factors ', 2I10 )" ) prefix,     &
            inform%factorization_real, inform%factorization_integer

        IF ( inform%SBLS_inform%perturbed ) THEN
          SBLS_control%new_h = 2
        ELSE
          SBLS_control%new_h = 1
        END IF
        SBLS_control%new_a = 0
        SBLS_control%new_c = 0

        CALL CLOCK_time( clock_now )
        IF ( printt ) THEN
          WRITE( out, "( A, ' ** factorize time = ', F10.2 ) " )               &
            prefix, clock_now - clock_record
          WRITE( out, "( A, 1X, I0, ' integer and ', I0, ' real words needed', &
         &    ' for factorization' )" ) prefix, inform%factorization_integer,  &
                                        inform%factorization_real
        END IF

!  =======
!  STEP 1:
!  =======

!  =======================================================================
!  -*-*-*-   Obtain the Primal-Dual (Predictor) Search Direction -*-*-*-*-
!  =======================================================================

!  :::::::::::::::::::::::::::::::::
!  Set up the right-hand-side vector
!  :::::::::::::::::::::::::::::::::

        IF ( printd ) WRITE( out, 2100 )                                       &
          prefix, ' GRAD_L', GRAD_L( dims%x_s : dims%x_e )

!  Problem variables:

        RHS( : dims%x_free ) = - GRAD_L( : dims%x_free )
        DO i = dims%x_free + 1, dims%x_l_start - 1
          RHS( i ) = - GRAD_L( i ) + mu / X( i )
        END DO
        DO i = dims%x_l_start, dims%x_u_start - 1
          RHS( i ) = - GRAD_L( i ) + mu / DIST_X_l( i )
        END DO
        DO i = dims%x_u_start, dims%x_l_end
          RHS( i ) = - GRAD_L( i ) + mu / DIST_X_l( i ) - mu / DIST_X_u( i )
        END DO
        DO i = dims%x_l_end + 1, dims%x_u_end
          RHS( i ) = - GRAD_L( i ) - mu / DIST_X_u( i )
        END DO
        DO i = dims%x_u_end + 1, n
          RHS( i ) = - GRAD_L( i ) + mu / X( i )
        END DO

!  Slack variables:

        DO i = dims%c_l_start, dims%c_u_start - 1
          RHS( dims%c_b + i ) = - Y( i ) + mu / DIST_C_l( i )
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          RHS( dims%c_b + i ) = - Y( i ) + mu / DIST_C_l( i )                  &
                                       - mu / DIST_C_u( i )
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          RHS( dims%c_b + i ) = - Y( i ) - mu / DIST_C_u( i )
        END DO

!  Include the constraint infeasibilities

        RHS( dims%y_s : dims%y_e ) = - C_RES( : dims%c_u_end )
        DELTA = RHS

        IF ( printd ) THEN
          WRITE( out, 2100 ) prefix, ' RHS_x ', RHS( dims%x_s : dims%x_e )
          IF ( m > 0 )                                                         &
            WRITE( out, 2100 ) prefix, ' RHS_y ', RHS( dims%y_s : dims%y_e )
        END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the primal-dual direction for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Solve  ( H  A^T ) ( Dx^pd ) = - ( grad b )
!         ( A   0  ) ( Dy^pd )     (   r    )

        IF ( printw ) WRITE( out,                                              &
             "( A, ' ............... compute step  ............... ' )" ) prefix

!  Use a direct method

        DELTA( : A_sbls%n + A_sbls%m ) = RHS( : A_sbls%n + A_sbls%m )
        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls,                   &
                         sbls_data, SBLS_control, inform%SBLS_inform, DELTA )
        CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
        time_solve = time_solve + time_now - time_record
        clock_solve = clock_solve + clock_now - clock_record
        IF ( printt ) WRITE( out, "( A, ' ** solve time = ', F10.2 )" )        &
          prefix, clock_now - clock_record

        inform%status = inform%SBLS_inform%status
        IF ( inform%status /= GALAHAD_ok ) GO TO 700

!  Compute the residual of the linear system

!       CALL LSQP_residual( dims, n, m, dims%v_e, a_ne, A_val, A_col, A_ptr,   &
!                           DELTA( dims%x_s : dims%x_e ),                      &
!                           DELTA( dims%c_s : dims%c_e ),                      &
!                           DELTA( dims%y_s : dims%y_e ),                      &
!                           RHS( dims%x_s : dims%x_e ),                        &
!                           RHS( dims%c_s : dims%c_e ),                        &
!                           RHS( dims%y_s : dims%y_e ), HX( : dims%v_e ),      &
!                           BARRIER_X, BARRIER_C, SCALE_C, errorg, errorc,     &
!                           print_level, prefix, control, Hessian_kind,        &
!                           WEIGHT = WEIGHT )

!  Check to see if the problem is unbounded from below

        IF ( inform%feasible .AND. inform%SBLS_inform%rank_def .AND.           &
!            MAXVAL( ABS( HX( : n ) - RHS( dims%x_s : dims%x_e ) ) ) >         &
             inform%SBLS_inform%norm_residual > epsmch ** 0.5 ) THEN
          inform%status = GALAHAD_error_unbounded ; GO TO 600
        END IF

!  If the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely. Exit

!       IF ( SQRT( SUM( ( HX( : dims%v_e ) - RHS ) ** 2 ) ) > merit ) THEN
        IF ( inform%SBLS_inform%norm_residual > merit ) THEN

!  It didn't. We might have run out of options

          IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
            inform%status = GALAHAD_error_ill_conditioned ; GO TO 600

!  ... or we may change the method

          ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
            pivot_tol = relative_pivot_tol
            maxpiv = pivot_tol >= half
            SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
            SBLS_control%factorization = 2
            IF ( printi ) WRITE( out,                                          &
              "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

          ELSE
            pivot_tol = half
            maxpiv = .TRUE.
            SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
            SBLS_control%factorization = 2
            IF ( printi )                                                      &
              WRITE( out, "( A, ' Pivot tolerance increased ' )" ) prefix
          END IF
          alpha = zero ; nbact = 0
          CYCLE
        END IF

        IF ( printw ) WRITE( out,                                              &
             "( A, ' ............... step computed ............... ' )" ) prefix

        IF ( printd ) THEN
          WRITE( out, 2120 ) prefix, mu
          WRITE( out, 2100 ) prefix, ' DX ', DELTA( dims%x_s : dims%x_e )
          IF ( m > 0 )                                                         &
            WRITE( out, 2100 ) prefix, ' DY ', DELTA( dims%y_s : dims%y_e )
        END IF

!  =======
!  STEP 2:
!  =======

! ::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the search directions for the dual variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Problem variables:

!       l = 0
        DO i = dims%x_free + 1, dims%x_l_start - 1
          DZ_l( i ) =   ( mu - Z_l( i ) * ( X( i ) + DELTA( i ) ) ) / X( i )
!         IF ( ABS( one + DELTA( i ) / X( i ) ) < 0.001 ) l = l + 1
        END DO
        DO i = dims%x_l_start, dims%x_l_end
          DZ_l( i ) =   ( mu - Z_l( i ) *                                      &
                           ( DIST_X_l( i ) + DELTA( i ) ) ) / DIST_X_l( i )
!         IF ( ABS( one + DELTA( i ) / DIST_X_l( i ) ) < 0.001 ) l = l + 1
        END DO

        DO i = dims%x_u_start, dims%x_u_end
          DZ_u( i ) = - ( mu + Z_u( i ) *                                      &
                          ( DIST_X_u( i ) - DELTA( i ) ) ) / DIST_X_u( i )
!         IF ( ABS( one - DELTA( i ) / DIST_X_u( i ) ) < 0.001 ) l = l + 1
        END DO

        DO i = dims%x_u_end + 1, n
          DZ_u( i ) =   ( mu - Z_u( i ) * ( X( i ) + DELTA( i ) ) ) / X( i )
!         IF ( ABS( one + DELTA( i ) / X( i ) )  < 0.001 ) l = l + 1
        END DO
!       write(6,*) l, ' degenerate variable(s)'

!  Slack variables:

        DO i = dims%c_l_start, dims%c_l_end
          DY_l( i ) =   ( mu - Y_l( i ) *                                      &
                    ( DIST_C_l( i ) + DELTA( dims%c_b + i ) ) ) / DIST_C_l( i )
        END DO

        DO i = dims%c_u_start, dims%c_u_end
          DY_u( i ) = - ( mu + Y_u( i ) *                                      &
                    ( DIST_C_u( i ) - DELTA( dims%c_b + i ) ) ) / DIST_C_u( i )
        END DO

        IF ( printd ) THEN
          WRITE( out, 2100 ) prefix, ' DZ_l ', DZ_l(dims%x_free+1:dims%x_l_end)
          WRITE( out, 2100 ) prefix, ' DZ_u ', DZ_u( dims%x_u_start : n )
        END IF

!  Calculate the norm of the search direction

        pmax = MAX( MAXVAL( ABS( DELTA( dims%x_s : dims%x_e ) ) ),             &
                    MAXVAL( ABS( DELTA( dims%c_s : dims%c_e ) ) ),             &
                    MAXVAL( ABS( DELTA( dims%y_s : dims%y_e ) ) ),             &
                    MAXVAL( ABS( DZ_l( dims%x_free + 1 : dims%x_l_end ) ) ),   &
                    MAXVAL( ABS( DZ_u( dims%x_u_start  : n ) ) ),              &
                    MAXVAL( ABS( DY_l( dims%c_l_start  : dims%c_l_end ) ) ),   &
                    MAXVAL( ABS( DY_u( dims%c_u_start  : dims%c_u_end ) ) ) )

        IF ( printp ) WRITE( out, "( /, A,                                     &
       &  '  Norm of (predictor) search direction = ', ES12.4 )" ) prefix, pmax

!  ========
!  STEP 1b:
!  ========

        IF ( control%use_corrector ) THEN

!  =======================================================================
!  -*-*-*-   Obtain the Primal-Dual (Corrector) Search Direction -*-*-*-*-
!  =======================================================================

!  :::::::::::::::::::::::::::::::::
!  Set up the right-hand-side vector
!  :::::::::::::::::::::::::::::::::

!  Problem variables:

          RHS( : dims%x_free ) = zero
          DO i = dims%x_free + 1, dims%x_l_start - 1
            RHS( i ) = - DELTA( i ) * DZ_l( i ) / X( i )
          END DO
          DO i = dims%x_l_start, dims%x_u_start - 1
            RHS( i ) = - DELTA( i ) * DZ_l( i ) / DIST_X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            RHS( i ) = - DELTA( i ) * DZ_l( i ) / DIST_X_l( i )                &
                       + DELTA( i ) * DZ_u( i ) / DIST_X_u( i )
          END DO
          DO i = dims%x_l_end + 1, dims%x_u_end
            RHS( i ) =   DELTA( i ) * DZ_u( i ) / DIST_X_u( i )
          END DO
          DO i = dims%x_u_end + 1, n
            RHS( i ) = - DELTA( i ) * DZ_u( i ) / X( i )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            RHS( dims%c_b + i ) =                                              &
              - DELTA( dims%c_b + i ) * DY_l( i ) / DIST_C_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            RHS( dims%c_b + i ) =                                              &
              - DELTA( dims%c_b + i ) * DY_l( i ) / DIST_C_l( i )              &
              + DELTA( dims%c_b + i ) * DY_u( i ) / DIST_C_u( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            RHS( dims%c_b + i ) =                                              &
                DELTA( dims%c_b + i ) * DY_u( i ) / DIST_C_u( i )
          END DO

!  Include the constraint infeasibilities

          RHS( dims%y_s : dims%y_e ) = zero
          DELTA_cor = RHS

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' RHS_cor_x ', RHS( dims%x_s : dims%x_e )
            IF ( m > 0 )                                                       &
              WRITE( out, 2100 ) prefix, ' RHS_cor_y ', RHS( dims%y_s:dims%y_e )
          END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the corrector direction for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

          IF ( printw ) WRITE( out,                                            &
             "( A, ' ............... compute step  ............... ' )" ) prefix

!  Use a direct method

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls,                 &
                           sbls_data, SBLS_control, inform%SBLS_inform,        &
                           DELTA_cor )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          time_solve = time_solve + time_now - time_record
          clock_solve = clock_solve + clock_now - clock_record
          IF ( printt ) WRITE( out, "( A, ' ** solve time = ', F10.2 )" )      &
            prefix, clock_now - clock_record

          inform%status = inform%SBLS_inform%status
          IF ( inform%status /= GALAHAD_ok ) GO TO 700

!  Compute the residual of the linear system

!         CALL LSQP_residual( dims, n, m, dims%v_e, a_ne, A_val, A_col, A_ptr, &
!                             DELTA_cor( dims%x_s : dims%x_e ),                &
!                             DELTA_cor( dims%c_s : dims%c_e ),                &
!                             DELTA_cor( dims%y_s : dims%y_e ),                &
!                             RHS( dims%x_s : dims%x_e ),                      &
!                             RHS( dims%c_s : dims%c_e ),                      &
!                             RHS( dims%y_s : dims%y_e ), HX( : dims%v_e ),    &
!                             BARRIER_X, BARRIER_C, SCALE_C,                   &
!                             errorg, errorc, print_level, prefix, control,    &
!                             Hessian_kind, WEIGHT = WEIGHT )

!  Check to see if the problem is unbounded from below

          IF ( inform%feasible .AND. inform%SBLS_inform%rank_def .AND.         &
!              MAXVAL( ABS( HX( : n ) - RHS( dims%x_s : dims%x_e ) ) )         &
               inform%SBLS_inform%norm_residual > epsmch ** 0.5 ) THEN
            inform%status = GALAHAD_error_unbounded ; GO TO 600
          END IF

!  If the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely. Exit

!         IF ( SQRT( SUM( ( HX( : dims%v_e ) - RHS ) ** 2 ) ) > merit ) THEN
          IF ( inform%SBLS_inform%norm_residual > merit ) THEN

!  It didn't. We might have run out of options

            IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
              inform%status = GALAHAD_error_ill_conditioned ; GO TO 600

!  ... or we may change the method

            ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
              pivot_tol = relative_pivot_tol
              maxpiv = pivot_tol >= half
              SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
              SBLS_control%factorization = 2
              IF ( printi ) WRITE( out,                                        &
                "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

            ELSE
              pivot_tol = half
              maxpiv = .TRUE.
              SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
              SBLS_control%factorization = 2
            IF ( printi )                                                      &
              WRITE( out, "( A, ' Pivot tolerance increased ' )" ) prefix
            END IF
            alpha = zero ; nbact = 0
            CYCLE
          END IF

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ............... step computed ............... ' )" ) prefix

          IF ( printd ) THEN
            WRITE( out, 2120 ) prefix, mu
            WRITE( out, 2100 ) prefix, ' DX_cor ', DELTA_cor( dims%x_s:dims%x_e)
            IF ( m > 0 ) WRITE( out, 2100 ) prefix,                            &
              ' DY_cor ', DELTA_cor( dims%y_s : dims%y_e )
          END IF

!  ========
!  STEP 2b:
!  ========

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the corrector search directions for the dual variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Problem variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            DZ_cor_l( i ) = - ( DZ_l( i ) * DELTA( i ) +                       &
                                 Z_l( i ) * DELTA_cor( i ) ) / X( i )
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            DZ_cor_l( i ) = - ( DZ_l( i ) * DELTA( i ) +                       &
                                 Z_l( i ) * DELTA_cor( i ) ) / DIST_X_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            DZ_cor_u( i ) =   ( DZ_u( i ) * DELTA( i ) +                       &
                                 Z_u( i ) * DELTA_cor( i ) ) / DIST_X_u( i )
          END DO

          DO i = dims%x_u_end + 1, n
            DZ_cor_u( i ) = - ( DZ_u( i ) * DELTA( i ) +                       &
                                 Z_u( i ) * DELTA_cor( i ) ) / X( i )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_l_end
            DY_cor_l( i ) = - ( DY_l( i ) * DELTA( dims%c_b + i ) +            &
              Y_l( i ) * DELTA_cor( dims%c_b + i ) ) / DIST_C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            DY_cor_u( i ) =   ( DY_u( i ) * DELTA( dims%c_b + i ) +            &
              Y_u( i ) * DELTA_cor( dims%c_b + i ) ) / DIST_C_u( i )
          END DO

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' DZ_cor_l ',                           &
              DZ_cor_l( dims%x_free + 1 : dims%x_l_end )
            WRITE( out, 2100 ) prefix, ' DZ_cor_u ', DZ_cor_u( dims%x_u_start:n)
          END IF

!  Calculate the norm of the search direction

          pmax_cor = MAX( MAXVAL( ABS( DELTA_cor( dims%x_s : dims%x_e ) ) ),   &
            MAXVAL( ABS( DELTA_cor( dims%c_s : dims%c_e ) ) ),                 &
            MAXVAL( ABS( DELTA_cor( dims%y_s : dims%y_e ) ) ),                 &
            MAXVAL( ABS( DZ_cor_l( dims%x_free + 1 : dims%x_l_end ) ) ),       &
            MAXVAL( ABS( DZ_cor_u( dims%x_u_start  : n ) ) ),                  &
            MAXVAL( ABS( DY_cor_l( dims%c_l_start  : dims%c_l_end ) ) ),       &
            MAXVAL( ABS( DY_cor_u( dims%c_u_start  : dims%c_u_end ) ) ) )

          IF ( printp ) WRITE( out, "( /, A, '  Norm of (corrector)',          &
         &                ' search direction = ', ES12.4 )" ) prefix, pmax_cor
        END IF

!  Check to see whether to use a corrector step based on the relative
!  sizes of the predictor and corrector

        IF ( control%use_corrector ) THEN
!         IF ( pmax_cor < ten * pmax ) THEN
            use_corrector = .TRUE.
!         ELSE
!           use_corrector = .FALSE.
!         END IF
        ELSE
          use_corrector = .FALSE.
        END IF

        IF ( use_corrector ) THEN
          co = 'c'
          IF ( printp ) WRITE( out, "( A, ' ** corrector used' )" ) prefix
        ELSE
          co = ' '
        END IF
        inform%time%solve = inform%time%solve + time_solve
        inform%time%clock_solve = inform%time%clock_solve + clock_solve

!  =======
!  STEP 3:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Line search   -*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  Perform a line-search to find a point X + alpha DX that
!  sufficiently reduces the merit function

        IF ( printw ) WRITE( out,                                              &
             "( A, ' .............. get steplength  .............. ' )" ) prefix

!  Form the vector H dx + A(trans) dy

        IF ( Hessian_kind == 0 ) THEN
          HX( dims%x_s : dims%x_e ) = zero
        ELSE IF ( Hessian_kind == 1 ) THEN
          HX( dims%x_s : dims%x_e ) = DELTA( dims%x_s : dims%x_e )
        ELSE
          HX( dims%x_s : dims%x_e ) =                                          &
            DELTA( dims%x_s : dims%x_e ) * WEIGHT( dims%x_s : dims%x_e ) ** 2
        END IF
        IF ( m > 0 ) CALL LSQP_AX( n, HX( : n ), m, a_ne, A_val, A_col, A_ptr, &
                                   m, DELTA( dims%y_s : dims%y_e ), '+T' )

!  Perform a line-search to find a point X + alpha DX which
!  sufficiently reduces the measure of potential

        IF ( inform%feasible .AND. Hessian_kind == 0 .AND.                     &
             gradient_kind == 0 ) THEN

!  Find the largest possible feasible stepsize

          alpha = infinity
          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( DELTA( i ) < zero ) alpha = MIN( alpha, - X( i ) / DELTA( i ) )
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            IF ( DELTA( i ) < zero )                                           &
              alpha = MIN( alpha, - DIST_X_l( i ) / DELTA( i ) )
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            IF ( DELTA( i ) > zero )                                           &
              alpha = MIN( alpha, DIST_X_u( i ) / DELTA( i ) )
          END DO

          DO i = dims%x_u_end + 1, n
            IF ( DELTA( i ) > zero ) alpha = MIN( alpha, - X( i ) / DELTA( i ) )
          END DO

          DO i = dims%c_l_start, dims%c_l_end
            IF ( DELTA(  dims%c_b + i ) < zero )                               &
              alpha = MIN( alpha, - DIST_C_l( i ) / DELTA( dims%c_b + i ) )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            IF ( DELTA(  dims%c_b + i ) > zero )                               &
              alpha = MIN( alpha, DIST_C_u( i ) / DELTA( dims%c_b + i ) )
          END DO

!  A step of no larger than one will be attempted

          alpha = MIN( one, 0.9999_wp * alpha )

          IF ( printt ) WRITE( out, "( /, A, '       ***  Linesearch      ',   &
         &                        '  step trial centr model centr ' )" ) prefix

          nbact = 0 ; step = alpha
          DO

!  Calculate the distances to the bounds at the trial point

            X = X + step * DELTA( dims%x_s : dims%x_e )

            DO i = dims%x_l_start, dims%x_l_end
              DIST_X_l( i ) = DIST_X_l( i ) + step * DELTA( i )
            END DO

            DO i = dims%x_u_start, dims%x_u_end
              DIST_X_u( i ) = DIST_X_u( i ) - step * DELTA( i )
            END DO

!  Do the same for the slacks

            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l( i ) = DIST_C_l( i ) + step * DELTA( dims%c_b + i )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              DIST_C_u( i ) = DIST_C_u( i ) - step * DELTA( dims%c_b + i )
            END DO

!  Ensure that the measure of potential has decreased

            one_minus_alpha = one - alpha
            potential_trial = LSQP_potential_value( dims, n,                   &
                                   X, DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u )

!  Check to see if the Amijo criterion is satisfied. If not, halve the
!  steplength

            IF ( printt ) WRITE( out, "( A, 22X, 3ES12.4 )" )                  &
              prefix, alpha, potential_trial, inform%potential

            IF ( potential_trial <= inform%potential ) EXIT
            alpha = alpha * half ;  step = - alpha ; nbact = nbact + 1
            IF ( alpha < epsmch ) THEN
              inform%status = GALAHAD_error_tiny_step
              GO TO 500
            END IF
          END DO

!  Update the Lagrange multipliers

          Y = Y - alpha * DELTA( dims%y_s : dims%y_e )

!  Calculate the new dual variables

          DO i = dims%x_free + 1, dims%x_l_start - 1
            Z_l( i ) = mu / X( i )
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            Z_l( i ) = mu / DIST_X_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            Z_u( i ) = - mu / DIST_X_u( i )
          END DO

          DO i = dims%x_u_end + 1, n
            Z_u( i ) = mu / X( i )
          END DO

!  Do the same for the Lagrange multipliers

          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = mu / DIST_C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = - mu / DIST_C_u( i )
          END DO

          inform%potential = potential_trial
          inform%nbacts = inform%nbacts + nbact

!  Perform a line-search to find a point X + alpha DX that
!  sufficiently reduces the merit function

        ELSE

!  Find the maximum step which keeps the complementarity and feasibilty balanced

          CALL LSQP_compute_maxstep( dims, n, m, Z_l, Z_u, DZ_l, DZ_u,         &
                                     X, DIST_X_l, DIST_X_u,                    &
                                     DELTA( dims%x_s : dims%x_e ),             &
                                     Y_l, Y_u, DY_l, DY_u, DIST_C_l, DIST_C_u, &
                                     DELTA( dims%c_s : dims%c_e ), gamma_b,    &
                                     gamma_f, nu, nbnds,                       &
                                     alpha_b, alpha_f, print_level,            &
                                     control, inform )

          IF ( inform%status /= GALAHAD_ok ) GO TO 500
          IF ( printt ) WRITE( out, "( A, ' alpha_b, alpha_f = ', 2ES12.4 )" ) &
                                               prefix, alpha_b, alpha_f

!  A step of no larger than one will be attempted

          alpha = MIN( one, alpha_b, alpha_f )

! ::::::::::::::::::::::::::::::::::::::::::::
!  Record the slope along the search direction
! ::::::::::::::::::::::::::::::::::::::::::::

          slope = - ( merit - mu * nbnds )
          IF ( printt ) WRITE( out, "( A, '  Value and slope = ', 1P, 2D12.4)")&
            prefix, merit, slope

! ::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Use a backtracking line-search, starting from alpha
! ::::::::::::::::::::::::::::::::::::::::::::::::::::

          IF ( printt ) WRITE( out, "( /, A, '       ***  Linesearch    ',     &
         &                     '    step trial value model value ' )" ) prefix

          nbact = 0 ; step = alpha
          DO

!  The sqaure of the norm of the new residual should be smaller than a
!  linear model

            merit_model = merit + alpha * eta * slope

!  Calculate the distances to the bounds and the dual variables at the
!  trial point

            X = X + step * DELTA( dims%x_s : dims%x_e )
            Y = Y - step * DELTA( dims%y_s : dims%y_e )

            DO i = dims%x_free + 1, dims%x_l_start - 1
              Z_l( i ) = Z_l( i ) + step * DZ_l( i )
            END DO

            DO i = dims%x_l_start, dims%x_l_end
              DIST_X_l( i ) = DIST_X_l( i ) + step * DELTA( i )
              Z_l( i ) = Z_l( i ) + step * DZ_l( i )
            END DO

            DO i = dims%x_u_start, dims%x_u_end
              DIST_X_u( i ) = DIST_X_u( i ) - step * DELTA( i )
              Z_u( i ) = Z_u( i ) + step * DZ_u( i )
            END DO

            DO i = dims%x_u_end + 1, n
              Z_u( i ) = Z_u( i ) + step * DZ_u( i )
            END DO

!  Do the same for the slacks and their duals

            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l( i ) = DIST_C_l( i ) + step * DELTA( dims%c_b + i )
              Y_l( i ) = Y_l( i ) + step * DY_l( i )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              DIST_C_u( i ) = DIST_C_u( i ) - step * DELTA( dims%c_b + i )
              Y_u( i ) = Y_u( i ) + step * DY_u( i )
            END DO

!  Evaluate the merit function at the new point

            one_minus_alpha = one - alpha
            merit_trial = LSQP_merit_value(                                    &
                            dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,              &
                            DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,            &
                            GRAD_L( dims%x_s : dims%x_e ) + alpha * HX( : n ), &
                            one_minus_alpha * C_RES, res_dual )
            IF ( printt ) WRITE( out, "( A, 22X, 3ES12.4 )" )                  &
              prefix, alpha, merit_trial, merit_model

!  Check to see if the Amijo criterion is satisfied. If not, halve the
!  steplength

            IF ( merit_trial <= merit_model ) EXIT
            alpha = alpha * half ;  step = - alpha ; nbact = nbact + 1
            IF ( alpha < epsmch ) THEN
              IF ( inform%iter - 1 > muzero_fixed ) THEN
                inform%status = GALAHAD_error_tiny_step
                GO TO 500
              ELSE
                muzero_fixed = inform%iter - 2
                EXIT
              END IF
            END IF
          END DO
          merit = merit_trial

          inform%nbacts = inform%nbacts + nbact
        END IF

!  Update the slack variables

        IF ( use_corrector ) THEN
          C = C + alpha * ( DELTA( dims%c_s : dims%c_e ) +                     &
                alpha * DELTA_cor( dims%c_s : dims%c_e ) )
        ELSE
          C = C + alpha * DELTA( dims%c_s : dims%c_e )
        END IF

!  Update the values of the merit function, the gradient of the Lagrangian,
!  and the constraint residuals

        GRAD_L( dims%x_s : dims%x_e ) = GRAD_L( dims%x_s : dims%x_e ) +        &
          alpha * HX( dims%x_s : dims%x_e )

        IF ( use_corrector ) THEN
          DO i = dims%x_free + 1, dims%x_l_end
            GRAD_L( i ) = GRAD_L( i ) + alpha * alpha * DZ_cor_l( i )
          END DO
          DO i = dims%x_u_start, n
            GRAD_L( i ) = GRAD_L( i ) + alpha * alpha * DZ_cor_u( i )
          END DO
        END IF

        C_RES = one_minus_alpha * C_RES

!  Update the norm of the constraint residual

        res_prim = one_minus_alpha * res_prim
        nu = one_minus_alpha * nu

!  Evaluate the merit function if not already done

        IF ( inform%feasible .AND. Hessian_kind == 0 .AND.                     &
                  gradient_kind == 0 ) THEN
          merit = LSQP_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,      &
                            DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,            &
                            GRAD_L( dims%x_s : dims%x_e ), C_RES, res_dual )
        END IF

!  Compute the objective function value

        IF ( Hessian_kind == 0 ) THEN
          inform%obj = f
        ELSE IF ( Hessian_kind == 1 ) THEN
          inform%obj = f + half * SUM( ( X - X0 ) ** 2 )
        ELSE
          inform%obj = f + half * SUM( ( WEIGHT * ( X - X0 ) ) ** 2 )
        END IF

        IF ( gradient_kind == 1 ) THEN
          inform%obj = inform%obj + SUM( X )
        ELSE IF ( gradient_kind /= 0 ) THEN
          inform%obj = inform%obj + DOT_PRODUCT( G, X )
        END IF

        IF ( m > 0 ) THEN
          C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
          C_RES( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
          CALL LSQP_AX( m, C_RES, m, a_ne, A_val, A_col, A_ptr,      &
                        n, X, '+ ' )

          IF ( printt ) WRITE( out, "( A, '  Constraint residual ', ES12.4 )" )&
            prefix, MAXVAL( ABS( C_RES ) )
!         WRITE( 6, "( ' rec, cal cres = ', 2ES12.4 )" )                       &
!           res_prim, MAXVAL( ABS( C_RES ) )
          IF ( res_prim < MAXVAL( ABS( C_RES ) ) ) THEN
            res_prim = MAXVAL( ABS( C_RES ) )
          END IF
        END IF

!  Compute the complementary slackness, and the min/max components
!  of the primal/dual infeasibilities

!       DO i = dims%x_free + 1, dims%x_l_start - 1
!         write(6,"(I6, ' x lower', 2ES12.4)" ) i, X( i ), Z_l( i )
!       END DO
!       DO i = dims%x_l_start, dims%x_l_end
!         write(6,"(I6, ' x lower', 2ES12.4)" ) i, DIST_X_l( i ), Z_l( i )
!       END DO
!       DO i = dims%x_u_start, dims%x_u_end
!         write(6,"(I6, ' x upper', 2ES12.4)" ) i, - DIST_X_u( i ), Z_u( i )
!       END DO
!       DO i = dims%x_u_end + 1, n
!         write(6,"(I6, ' x upper', 2ES12.4)" ) i, X( i ), Z_u( i )
!       END DO

!       DO i = dims%c_l_start, dims%c_l_end
!         write(6,"(I6, ' c lower', 2ES12.4)" ) i, DIST_C_l( i ), Y_l( i )
!       END DO
!       DO i = dims%c_u_start, dims%c_u_end
!         write(6,"(I6, ' c upper', 2ES12.4)" ) i, - DIST_C_u( i ), Y_u( i )
!       END DO

        slknes_x = DOT_PRODUCT( X( dims%x_free + 1 : dims%x_l_start - 1 ),     &
                                Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ) +&
                   DOT_PRODUCT( DIST_X_l( dims%x_l_start : dims%x_l_end ),     &
                                Z_l( dims%x_l_start : dims%x_l_end ) ) -       &
                   DOT_PRODUCT( DIST_X_u( dims%x_u_start : dims%x_u_end ),     &
                                Z_u( dims%x_u_start : dims%x_u_end ) ) +       &
                   DOT_PRODUCT( X( dims%x_u_end + 1 : n ),                     &
                              Z_u( dims%x_u_end + 1 : n ) )
        slknes_c = DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),     &
                                Y_l( dims%c_l_start : dims%c_l_end ) ) -       &
                   DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),     &
                                Y_u( dims%c_u_start : dims%c_u_end ) )
        slknes = slknes_x + slknes_c

        slkmin_x = MIN( MINVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) *    &
                                Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ), &
                        MINVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) *    &
                                Z_l( dims%x_l_start : dims%x_l_end ) ),        &
                        MINVAL( - DIST_X_u( dims%x_u_start : dims%x_u_end ) *  &
                                Z_u( dims%x_u_start : dims%x_u_end ) ),        &
                        MINVAL( X( dims%x_u_end + 1 : n ) *                    &
                                Z_u( dims%x_u_end + 1 : n ) ) )
        slkmin_c = MIN( MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *    &
                                Y_l( dims%c_l_start : dims%c_l_end ) ),        &
                        MINVAL( - DIST_C_u( dims%c_u_start : dims%c_u_end ) *  &
                                Y_u( dims%c_u_start : dims%c_u_end ) ) )
        slkmin = MIN( slkmin_x, slkmin_c )

        slkmax_x = MAX( MAXVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) *    &
                                Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ), &
                        MAXVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) *    &
                                Z_l( dims%x_l_start : dims%x_l_end ) ),        &
                        MAXVAL( - DIST_X_u( dims%x_u_start : dims%x_u_end ) *  &
                                Z_u( dims%x_u_start : dims%x_u_end ) ),        &
                        MAXVAL( X( dims%x_u_end + 1 : n ) *                    &
                                Z_u( dims%x_u_end + 1 : n ) ) )
        slkmax_c = MAX( MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *    &
                                Y_l( dims%c_l_start : dims%c_l_end ) ),        &
                        MAXVAL( - DIST_C_u( dims%c_u_start : dims%c_u_end ) *  &
                                Y_u( dims%c_u_start : dims%c_u_end ) ) )

        p_min = MIN( MINVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) ),      &
                     MINVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) ),      &
                     MINVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) ),      &
                     MINVAL( - X( dims%x_u_end + 1 : n ) ),                    &
                     MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),      &
                     MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

        p_max = MAX( MAXVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) ),      &
                     MAXVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) ),      &
                     MAXVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) ),      &
                     MAXVAL( - X( dims%x_u_end + 1 : n ) ),                    &
                     MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),      &
                     MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

        d_min = MIN( MINVAL(   Z_l( dims%x_free + 1 : dims%x_l_end ) ),        &
                     MINVAL( - Z_u( dims%x_u_start : n ) ),                    &
                     MINVAL(   Y_l( dims%c_l_start : dims%c_l_end ) ),         &
                     MINVAL( - Y_u( dims%c_u_start : dims%c_u_end ) ) )

        d_max = MAX( MAXVAL(   Z_l( dims%x_free + 1 : dims%x_l_end ) ),        &
                     MAXVAL( - Z_u( dims%x_u_start : n ) ),                    &
                     MAXVAL(   Y_l( dims%c_l_start : dims%c_l_end ) ),         &
                     MAXVAL( - Y_u( dims%c_u_start : dims%c_u_end ) ) )

        IF ( nbnds_x > 0 ) THEN
          slknes_x = slknes_x / nbnds_x
        ELSE
          slknes_x = zero
        END IF

        IF ( nbnds_c > 0 ) THEN
          slknes_c = slknes_c / nbnds_c
        ELSE
          slknes_c = zero
        END IF
        IF ( nbnds > 0 ) THEN
          slknes = slknes / nbnds
          slknes_req = slknes
        ELSE
          slknes = zero
        END IF

        IF ( printt .AND. nbnds > 0 ) WRITE( out, 2130 )                       &
          prefix, slknes, prefix, slknes_x, prefix, slknes_c, prefix, slkmin_x,&
          slkmax_x, prefix, slkmin_c, slkmax_c, prefix, p_min, p_max, prefix,  &
          d_min, d_max

        IF ( printd ) THEN
          WRITE( out, "( A, ' primal-dual -vs- primal dual variables' )") prefix
          WRITE( out, "( A, ' lower ', /, ( 2( I6, 2ES12.4 ) ) )" )            &
            prefix, ( i, Z_l( i ), mu / X( i ),                                &
              i = dims%x_free + 1, dims%x_l_start - 1 ),                       &
            ( i, Z_l( i ), mu / DIST_X_l( i ),                                 &
              i =  dims%x_l_start, dims%x_l_end )
          WRITE( out, "( A, ' upper ', /, ( 2( I6, 2ES12.4 ) ) )" )            &
            prefix, ( i, Z_u( i ),  - mu / DIST_X_u( i ),                      &
              i = dims%x_u_start, dims%x_u_end ),                              &
            ( i, Z_u( i ), mu / X( i ),                                        &
              i = dims%x_u_end + 1, n )
        END IF

!  Test to see if we are feasible

        IF ( res_prim <= control%stop_p ) THEN
          IF ( control%just_feasible ) THEN
            inform%status = GALAHAD_ok
            inform%feasible = .TRUE.
            IF ( printi ) THEN
              CALL CLOCK_TIME( clock_now )
              WRITE( out, 2070 ) prefix
              WRITE( out, 2030 ) prefix, inform%iter, re, res_prim, res_dual,  &
                slknes_req, zero, alpha, co, mo, mu, nbact,                    &
                clock_now - clock_start
              IF ( printt ) WRITE( out, 2000 ) prefix
            END IF
            GO TO 500
          END IF

          IF ( .NOT. inform%feasible ) THEN
            IF ( printi ) WRITE( out, 2070 ) prefix
            inform%feasible = .TRUE.
            IF ( Hessian_kind == 0 .AND. gradient_kind == 0 ) THEN
              IF ( slkmin_x >= epsmch .AND. slkmin_c >= epsmch ) THEN
                inform%potential = LSQP_potential_value( dims, n, X, DIST_X_l, &
                                                  DIST_X_u, DIST_C_l, DIST_C_u )
              ELSE
                inform%potential = infinity
              END IF
            END IF
          END IF

          res_prim = zero ; nu = zero
          C_RES = zero
        END IF

!  =======
!  STEP 5:
!  =======

        IF ( get_stat ) THEN

!  Estimate the variable and constraint exit status

          CALL LSQP_indicators( dims, n, m, C_l, C_u, C_last, C,               &
                                DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,       &
                                DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,        &
                                Y_last, Z_last,                                &
                                control, C_stat = C_stat, B_stat = B_stat )

!  Count the number of active constraints/bounds

          IF ( printt )                                                        &
            WRITE( out, "( A, ' indicators: n_active/n, m_active/m ', 4I7 )" ) &
               prefix, COUNT( B_stat /= 0 ), n, COUNT( C_stat /= 0 ), m
        END IF

!  Compute the new penalty parameter

        sigma = sigma_max
!       mu = sigma * slknes
        IF ( inform%iter > muzero_fixed ) mu = sigma * slknes
        IF ( mu < one .AND. stat_required ) THEN
          get_stat = .TRUE.
          C_last( dims%c_l_start : dims%c_u_end )                              &
            = C( dims%c_l_start : dims%c_u_end )
          X_last = X

          DO i = dims%c_l_start, dims%c_u_start - 1
            Y_last( i ) = Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( DIST_C_l( i ) <= DIST_C_u( i ) ) THEN
              Y_last( i ) = Y_l( i )
            ELSE
              Y_last( i ) = Y_u( i )
            END IF
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            Y_last( i ) = Y_u( i )
          END DO

          Z_last( : dims%x_free ) = zero
          DO i = dims%x_free + 1, dims%x_u_start - 1
            Z_last( i ) = Z_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( DIST_X_l( i ) <= DIST_X_u( i ) ) THEN
              Z_last( i ) = Z_l( i )
            ELSE
              Z_last( i ) = Z_u( i )
            END IF
          END DO
          DO i = dims%x_l_end + 1, n
            Z_last( i ) = Z_u( i )
          END DO
        END IF

!  Compute the projected gradient of the Lagrangian function

        pjgnrm = zero
        DO i = 1, n
          gi = GRAD_L( i )
          IF ( gi < zero ) THEN
            gi = - MIN( ABS( X_u( i ) - X( i ) ), - gi )
          ELSE
            gi = MIN( ABS( X_l( i ) - X( i ) ), gi )
          END IF
          pjgnrm = MAX( pjgnrm, ABS( gi ) )
        END DO

        IF ( printd ) THEN
          WRITE( out, 2100 ) prefix, ' DIST_X_l ',                             &
            X( dims%x_free + 1 : dims%x_l_start - 1 ), DIST_X_l
          WRITE( out, 2100 ) prefix, ' DIST_X_u ',                             &
            DIST_X_u, - X( dims%x_u_end + 1 : n )
          WRITE( out, "( ' ' )" )
        END IF

        IF ( printd ) WRITE( out, 2110 ) prefix, pjgnrm, prefix, res_prim
      END DO

!  ---------------------------------------------------------------------
!  ---------------------- End of Major Iteration -----------------------
!  ---------------------------------------------------------------------

  500 CONTINUE

!  Print details of the solution obtained

  600 CONTINUE

!  Compute the final objective function value

      IF ( Hessian_kind == 0 ) THEN
        inform%obj = f
      ELSE IF ( Hessian_kind == 1 ) THEN
        inform%obj = f + half * SUM( ( X - X0 ) ** 2 )
      ELSE
        inform%obj = f + half * SUM( ( WEIGHT * ( X - X0 ) ) ** 2 )
      END IF

      IF ( gradient_kind == 1 ) THEN
        inform%obj = inform%obj + SUM( X )
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
      END IF

      IF ( printi ) THEN
        WRITE( out, "( /, A, '  Final objective function value ', ES22.14,     &
      &       /, A, '  Total number of iterations = ', I0,                     &
      &       /, A, '  Total number of backtracks = ', I0 )" )                 &
          prefix, inform%obj, prefix, inform%iter, prefix, inform%nbacts
        WRITE( out, 2110 ) prefix, pjgnrm, prefix, res_prim
        IF ( control%getdua ) WRITE( out,                                      &
         "( /, A, ' Advanced starting point used for dual variables' )" ) prefix
        IF ( SBLS_control%factorization == 0 .OR.                              &
             SBLS_control%factorization == 1 ) THEN
          WRITE( control%out, "( A, '  Schur-complement factorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" ) prefix,                       &
            SBLS_control%SLS_control%relative_pivot_tolerance
        ELSE
          WRITE( control%out, "( A, '  Augmented system factorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" ) prefix,                       &
            SBLS_control%SLS_control%relative_pivot_tolerance
        END IF
      END IF

!  If required, make the solution exactly complementary

      IF ( control%feasol ) THEN
        DO i = dims%x_free + 1, dims%x_l_start - 1
          IF ( ABS( Z_l( i ) ) < ABS( X( i ) ) ) THEN
            Z_l( i ) = zero
          ELSE
            X( i ) = X_l( i )
          END IF
        END DO

        DO i = dims%x_l_start, dims%x_l_end
          IF ( ABS( Z_l( i ) ) < ABS( DIST_X_l( i ) ) ) THEN
            Z_l( i ) = zero
          ELSE
            X( i ) = X_l( i )
          END IF
        END DO

        DO i = dims%x_u_start, dims%x_u_end
          IF ( ABS( Z_u( i ) ) < ABS( DIST_X_u( i ) ) ) THEN
            Z_u( i ) = zero
          ELSE
            X( i ) = X_u( i )
          END IF
        END DO

        DO i = dims%x_u_end + 1, n
          IF ( ABS( Z_u( i ) ) < ABS( X( i ) ) ) THEN
            Z_u( i ) = zero
          ELSE
            X( i ) = X_u( i )
          END IF
        END DO
      END IF

!  Exit

 700  CONTINUE

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

!  Unscale the constraint bounds

      DO i = dims%c_l_start, dims%c_l_end
        C_l( i ) = C_l( i ) * SCALE_C( i )
      END DO

      DO i = dims%c_u_start, dims%c_u_end
        C_u( i ) = C_u( i ) * SCALE_C( i )
      END DO

!  Compute the values of the constraints

      C_RES( : m ) = zero
      CALL LSQP_AX( m, C_RES( : m ), m, a_ne, A_val, A_col,                    &
                    A_ptr, n, X, '+ ')
      IF ( printi .AND. m > 0 ) THEN
        WRITE( out, "( A, '  Constraint residual ', ES12.4 )" ) prefix,        &
             MAX( zero, MAXVAL( ABS( C_l( : dims%c_equality ) -                &
                                     C_RES(: dims%c_equality ) ) ),            &
                        MAXVAL( C_l(  dims%c_l_start : dims%c_l_end ) -        &
                                C_RES(  dims%c_l_start : dims%c_l_end ) ),     &
                        MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -       &
                                C_u( dims%c_u_start : dims%c_u_end ) ) )
      END IF

!  If necessary, print warning messages

  810 CONTINUE

      sbls_data%last_preconditioner = no_last
      sbls_data%last_factorization = no_last

      IF ( printi ) then

        SELECT CASE( inform%status )
          CASE( GALAHAD_error_restrictions  ) ; WRITE( out, "( /, A,           &
         & '  Warning - input paramters incorrect' )" ) prefix
          CASE( GALAHAD_error_no_center ) ; WRITE( out, "( /, A,               &
         & '  Warning - the analytic centre appears to be unbounded' )" ) prefix
          CASE( GALAHAD_error_bad_bounds ) ; WRITE( out, "( /, A,              &
         &  '  Warning - the constraints are inconsistent' )" ) prefix
          CASE( GALAHAD_error_primal_infeasible ) ; WRITE( out, "( /, A,       &
         &  '  Warning - the constraints appear to be inconsistent' )" ) prefix
          CASE( GALAHAD_error_factorization ) ; WRITE( out, "( /, A,           &
         &   '  Warning - factorization failure' )" ) prefix
          CASE( GALAHAD_error_ill_conditioned ) ; WRITE( out, "( /, A,         &
         &   '  Warning - no further progress possible' )"  ) prefix
          CASE( GALAHAD_error_tiny_step ) ; WRITE( out, "( /, A,               &
         &   '  Warning - step too small to make further progress' )" ) prefix
          CASE( GALAHAD_error_max_iterations ) ; WRITE( out, "( /, A,          &
         &   '  Warning - iteration bound exceeded' )" ) prefix
          CASE( GALAHAD_error_unbounded ) ; WRITE( out, "( /, A,               &
         &   '  Warning - problem unbounded from below' )" ) prefix
        END SELECT

      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving LSQP_solve_main ' )" ) prefix

      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Iter   p-feas  d-feas com-slk   obj    ',                &
                '  step      mu    bac    time' )
 2020 FORMAT( A, I5, A1, 3ES8.1, ES9.1, '     -    ', ES7.1,                   &
            '   -', 0P, F9.2 )
 2030 FORMAT( A, I5, A1, 3ES8.1, ES9.1, ES8.1, 2A1, ES7.1, I4, 0P, F9.2 )
 2070 FORMAT( /, A, ' ====================== feasible point found',            &
                    ' ======================= ', / )
 2100 FORMAT( A, A, 7ES10.2, /, ( 10X, 7ES10.2 ) )
 2110 FORMAT( /, A, '  Norm of projected gradient is ', ES12.4,                &
              /, A, '  Norm of infeasibility is      ', ES12.4 )
 2120 FORMAT( A, ' Penalty parameter = ', ES12.4 )
 2130 FORMAT( A, 21X, ' == >  mu estimated   = ', ES10.2, /,                   &
              A, 21X, '       mu_x estimated = ', ES10.2, /,                   &
              A, 21X, '       mu_c estimated = ', ES10.2, /,                   &
              A, 21X, ' min/max slackness_x = ', 2ES12.4, /,                   &
              A, 21X, ' min/max slackness_c = ', 2ES12.4, /,                   &
              A, 14X, ' min/max primal feasibility = ', 2ES12.4, /,            &
              A, 14X, ' min/max dual   feasibility = ', 2ES12.4 )

!  End of LSQP_solve_main

      END SUBROUTINE LSQP_solve_main

!-*-*-*-*-*-*-   L S Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LSQP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine LSQP_initialize
!   control see Subroutine LSQP_initialize
!   inform  see Subroutine LSQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
      TYPE ( LSQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by FDC

      CALL FDC_terminate( data%FDC_data, data%FDC_control,                     &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FDC_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      inform%status = inform%SBLS_inform%status
      IF ( inform%SBLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'lsqp: data%SBLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

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
        inform%bad_alloc = 'lsqp: data%QPP'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'lsqp: data%INDEX_C_freed'
      CALL SPACE_dealloc_array( data%INDEX_C_freed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%HX'
      CALL SPACE_dealloc_array( data%HX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%GRAD_L'
      CALL SPACE_dealloc_array( data%GRAD_L,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DIST_X_l'
      CALL SPACE_dealloc_array( data%DIST_X_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DIST_X_u'
      CALL SPACE_dealloc_array( data%DIST_X_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Z_l'
      CALL SPACE_dealloc_array( data%Z_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Z_u'
      CALL SPACE_dealloc_array( data%Z_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%BARRIER_X'
      CALL SPACE_dealloc_array( data%BARRIER_X,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Y_l'
      CALL SPACE_dealloc_array( data%Y_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DY_l'
      CALL SPACE_dealloc_array( data%DY_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DIST_C_l'
      CALL SPACE_dealloc_array( data%DIST_C_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Y_u'
      CALL SPACE_dealloc_array( data%Y_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DY_u'
      CALL SPACE_dealloc_array( data%DY_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DIST_C_u'
      CALL SPACE_dealloc_array( data%DIST_C_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%BARRIER_C'
      CALL SPACE_dealloc_array( data%BARRIER_C,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%SCALE_C'
      CALL SPACE_dealloc_array( data%SCALE_C,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DELTA'
      CALL SPACE_dealloc_array( data%DELTA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DZ_l'
      CALL SPACE_dealloc_array( data%DZ_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DZ_u'
      CALL SPACE_dealloc_array( data%DZ_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!     array_name = 'lsqp: data%C_last'
!     CALL SPACE_dealloc_array( data%C_last,                                   &
!        inform%status, inform%alloc_status, array_name = array_name,          &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND.                                &
!          inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%X_last'
      CALL SPACE_dealloc_array( data%X_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Y_last'
      CALL SPACE_dealloc_array( data%Y_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%Z_last'
      CALL SPACE_dealloc_array( data%Z_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DELTA_cor'
      CALL SPACE_dealloc_array( data%DELTA_cor,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DY_cor_l'
      CALL SPACE_dealloc_array( data%DY_cor_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DY_cor_u'
      CALL SPACE_dealloc_array( data%DY_cor_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DZ_cor_l'
      CALL SPACE_dealloc_array( data%DZ_cor_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%DZ_cor_u'
      CALL SPACE_dealloc_array( data%DZ_cor_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%A_sbls%row'
      CALL SPACE_dealloc_array( data%A_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%A_sbls%col'
      CALL SPACE_dealloc_array( data%A_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%A_sbls%val'
      CALL SPACE_dealloc_array( data%A_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lsqp: data%H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine LSQP_terminate

      END SUBROUTINE LSQP_terminate

!-*-  L S Q P _ L A G R A N G I A N _ G R A D I E N T   S U B R O U T I N E  -*-

      SUBROUTINE LSQP_Lagrangian_gradient( dims, n, m, X, Y, Y_l, Y_u,         &
                                           Z_l, Z_u, a_ne, A_val, A_col, A_ptr,&
                                           DIST_X_l, DIST_X_u, DIST_C_l,       &
                                           DIST_C_u, GRAD_L, getdua, dufeas,   &
                                           Hessian_kind, gradient_kind,        &
                                           WEIGHT, X0, G )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the gradient of the Lagrangian function
!
!  GRAD_L = W*W*( x - x0 ) - A(transpose) y
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, Hessian_kind, gradient_kind
      REAL ( KIND = wp ), INTENT( IN ) :: dufeas
      LOGICAL, INTENT( IN ) :: getdua
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD_L
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: Z_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u
      INTEGER, INTENT( IN ) :: a_ne
      INTEGER, INTENT( IN ), DIMENSION( a_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( a_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: WEIGHT, X0
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: gi

!  Add the product A( transpose ) y to the gradient of the quadratic

      IF ( Hessian_kind == 0 ) THEN
        GRAD_L = zero
      ELSE IF ( Hessian_kind == 1 ) THEN
        GRAD_L = X - X0
      ELSE
        GRAD_L = ( WEIGHT ** 2 ) * ( X - X0 )
      END IF
      IF ( gradient_kind == 1 ) THEN
        GRAD_L = GRAD_L + one
      ELSE IF ( gradient_kind /= 0 ) THEN
        GRAD_L = GRAD_L + G
      END IF

      CALL LSQP_AX( n, GRAD_L, m, a_ne, A_val, A_col, A_ptr, m, Y, '-T' )

!  If required, obtain suitable "good" starting values for the dual
!  variables ( see paper )

      IF ( getdua ) THEN

!  Problem variables:

!  The variable is a non-negativity

        DO i = dims%x_free + 1, dims%x_l_start - 1
          Z_l( i ) = MAX( dufeas, GRAD_L( i ) / ( one + X( i ) ** 2 ) )
        END DO

!  The variable has just a lower bound

        DO i = dims%x_l_start, dims%x_u_start - 1
          Z_l( i ) = MAX( dufeas, GRAD_L( i ) / ( one + DIST_X_l( i ) ** 2 ) )
        END DO

!  The variable has both lower and upper bounds

        DO i = dims%x_u_start, dims%x_l_end
          gi = GRAD_L( i )
          IF ( ABS( gi ) <= dufeas ) THEN
            Z_l( i ) = dufeas ; Z_u( i ) = - dufeas
          ELSE IF ( gi > dufeas ) THEN
            Z_l( i ) = ( gi + dufeas ) / ( one + DIST_X_l( i ) ** 2 )
            Z_u( i ) = - dufeas
          ELSE
            Z_l( i ) = dufeas
            Z_u( i ) = ( gi - dufeas ) / ( one + DIST_X_u( i ) ** 2 )
          END IF
        END DO

!  The variable has just an upper bound

        DO i = dims%x_l_end + 1, dims%x_u_end
          Z_u( i ) = MIN( - dufeas, GRAD_L( i ) / ( one + DIST_X_u( i ) ** 2 ) )
        END DO

!  The variable is a non-positivity

        DO i = dims%x_u_end + 1, n
          Z_u( i ) = MIN( - dufeas, GRAD_L( i ) / ( one + X( i ) ** 2 ) )
        END DO

!  Slack variables:

!  The variable has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          Y_l( i ) = MAX( dufeas, - Y( i ) / ( one + DIST_C_l( i ) ** 2 ) )
        END DO

!  The variable has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          gi = - Y( i )
          IF ( ABS( gi ) <= dufeas ) THEN
            Y_l( i ) = dufeas ; Y_u( i ) = - dufeas
          ELSE IF ( gi > dufeas ) THEN
            Y_l( i ) = ( gi + dufeas ) / ( one + DIST_C_l( i ) ** 2 )
            Y_u( i ) = - dufeas
          ELSE
            Y_l( i ) = dufeas
            Y_u( i ) = ( gi - dufeas ) / ( one + DIST_C_u( i ) ** 2 )
          END IF
        END DO

!  The variable has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          Y_u( i ) = MIN( - dufeas, - Y( i ) / ( one + DIST_C_u( i ) ** 2 ) )
        END DO
      END IF

      RETURN

!  End of LSQP_Lagrangian_gradient

      END SUBROUTINE LSQP_Lagrangian_gradient

!-*-*-*-  L S Q P _ P O T E N T I A L _ V A L U E   S U B R O U T I N E  -*-*-*-

      FUNCTION LSQP_potential_value( dims, n, X, DIST_X_l, DIST_X_u,          &
                                     DIST_C_l, DIST_C_u )
      REAL ( KIND = wp ) LSQP_potential_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the potential function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u

! Compute the potential terms

      LSQP_potential_value =                                                   &
        - SUM( LOG( X( dims%x_free + 1 : dims%x_l_start - 1 ) ) )              &
        - SUM( LOG( DIST_X_l ) ) - SUM( LOG( DIST_X_u ) )                      &
        - SUM( LOG( - X( dims%x_u_end + 1 : n ) ) )                            &
        - SUM( LOG( DIST_C_l ) ) - SUM( LOG( DIST_C_u ) )

      RETURN

!  End of LSQP_potential_value

      END FUNCTION LSQP_potential_value

!-*-*-*-*-*-   L S Q P _ M E R I T _ V A L U E   F U N C T I O N   -*-*-*-*-*-*-

      FUNCTION LSQP_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,         &
                                 DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,       &
                                 GRAD_L, C_RES, res_dual )
      REAL ( KIND = wp ) LSQP_merit_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the merit function
!
!     | < z_l . ( x - x_l ) > +  < z_u . ( x_u - x ) > +
!       < y_l . ( c - c_l ) > +  < y_u . ( c_u - c ) > | +
!       || ( GRAD_L - z_l - z_u ) ||
!       || (   y - y_l - y_u    ) ||
!       || (  A x - SCALE_c * c ) ||
!
!  where GRAD_L = W*W*( x - x0 ) - A(transpose) y is the gradient
!  of the Lagrangian
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( OUT ) :: res_dual
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, GRAD_L
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_x_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: Z_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y, C_RES
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DIST_C_u

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: res_cs

!  Compute in the l_2-norm

      res_dual = SUM( GRAD_L( : dims%x_free ) ** 2 ) ; res_cs = zero

!  Problem variables:

      DO i = dims%x_free + 1, dims%x_l_start - 1
        res_dual = res_dual + ( GRAD_L( i ) - Z_l( i ) ) ** 2
        res_cs = res_cs + Z_l( i ) * X( i )
      END DO
      DO i = dims%x_l_start, dims%x_u_start - 1
        res_dual = res_dual + ( GRAD_L( i ) - Z_l( i ) ) ** 2
        res_cs = res_cs + Z_l( i ) * DIST_X_l( i )
      END DO
      DO i = dims%x_u_start, dims%x_l_end
        res_dual = res_dual + ( GRAD_L( i ) - Z_l( i ) - Z_u( i ) ) ** 2
        res_cs = res_cs + Z_l( i ) * DIST_X_l( i ) - Z_u( i ) * DIST_X_u( i )
      END DO
      DO i = dims%x_l_end + 1, dims%x_u_end
        res_dual = res_dual + ( GRAD_L( i ) - Z_u( i ) ) ** 2
        res_cs = res_cs - Z_u( i ) * DIST_X_u( i )
      END DO
      DO i = dims%x_u_end + 1, n
        res_dual = res_dual + ( GRAD_L( i ) - Z_u( i ) ) ** 2
        res_cs = res_cs + Z_u( i ) * X( i )
      END DO

!  Slack variables:

      DO i = dims%c_l_start, dims%c_u_start - 1
        res_dual = res_dual + ( Y( i ) - Y_l( i ) ) ** 2
        res_cs = res_cs + Y_l( i ) * DIST_C_l( i )
      END DO
      DO i = dims%c_u_start, dims%c_l_end
        res_dual = res_dual + ( Y( i ) - Y_l( i ) - Y_u( i ) ) ** 2
        res_cs = res_cs + Y_l( i ) * DIST_C_l( i ) - Y_u( i ) * DIST_C_u( i )
      END DO
      DO i = dims%c_l_end + 1, dims%c_u_end
        res_dual = res_dual + ( Y( i ) - Y_u( i ) ) ** 2
        res_cs = res_cs - Y_u( i ) * DIST_C_u( i )
      END DO

      LSQP_merit_value = ABS( res_cs ) + SQRT( res_dual + SUM( C_RES ** 2 ) )
      res_dual = SQRT( res_dual )

      RETURN

!  End of function LSQP_merit_value

      END FUNCTION LSQP_merit_value

!!-*-*-*-*-*-*-   L S Q P _ R E S I D U A L   S U B R O U T I N E   -*-*-*-*-*-*-
!
!      SUBROUTINE LSQP_residual( dims, n, m, l_res, a_ne, A_val, A_col, A_ptr,  &
!                                DX, DC, DY, RHS_x, RHS_c, RHS_y, RES,          &
!                                BARRIER_X, BARRIER_C, SCALE_C, errorg, errorc, &
!                                print_level, prefix, control, Hessian_kind,    &
!                                WEIGHT )
!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!!  Compute the residual of the linear system
!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!!  Dummy arguments
!
!      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
!      INTEGER, INTENT( IN ) :: n, m, a_ne, l_res, Hessian_kind, print_level
!      REAL( KIND = wp ), INTENT( OUT ) :: errorg, errorc
!      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( l_res ) :: RES
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RHS_x, DX
!      REAL ( KIND = wp ), INTENT( IN ),                                        &
!             DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: RHS_y, DY
!      REAL ( KIND = wp ), INTENT( IN ),                                        &
!           DIMENSION( dims%c_l_start : m ) :: RHS_c, DC, BARRIER_C, SCALE_C
!      INTEGER, INTENT( IN ), DIMENSION( a_ne ) :: A_col
!      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( a_ne ) :: A_val
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: WEIGHT
!      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
!      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
!
!!  Local variables
!
!      INTEGER :: i
!      REAL ( KIND = wp ) :: res_tol
!
!      res_tol = epsmch ** 0.5
!
!!  Initalize RES as the zero vector
!
!      RES( dims%y_s : dims%y_e ) = zero
!
!!  Remember the barrier terms, and any diagonal perturbations
!
!      IF ( Hessian_kind == 0 ) THEN
!        RES( : dims%x_free ) = zero
!        RES( dims%x_free + 1 : dims%x_e ) =                                    &
!             BARRIER_X * DX( dims%x_free + 1 : )
!      ELSE IF ( Hessian_kind == 1 ) THEN
!        RES( : dims%x_free ) = DX( : dims%x_free )
!        RES( dims%x_free + 1 : dims%x_e ) =                                    &
!            ( one + BARRIER_X ) * DX( dims%x_free + 1 : )
!      ELSE
!        RES( : dims%x_free ) =                                                 &
!           ( WEIGHT( : dims%x_free ) ** 2 ) * DX( : dims%x_free )
!        RES( dims%x_free + 1 : dims%x_e ) =                                    &
!           ( WEIGHT( dims%x_free + 1 : dims%x_e ) ** 2 +                       &
!             BARRIER_X ) * DX( dims%x_free + 1 : )
!      END IF
!      RES( dims%c_s : dims%c_e ) = BARRIER_C * DC
!
!!  Include the contribution from A and A^T
!
!      CALL LSQP_AX( n, RES( dims%x_s : dims%x_e ), m, a_ne, A_val, A_col,      &
!                    A_ptr, m, DY, '+T' )
!      CALL LSQP_AX( m, RES( dims%y_s : dims%y_e ), m, a_ne, A_val, A_col,      &
!                    A_ptr, n, DX, '+ ' )
!
!!  Include the contribution from the slack variables
!
!      RES( dims%c_s : dims%c_e ) =                                             &
!        RES( dims%c_s : dims%c_e ) - SCALE_C * DY( dims%c_l_start : m )
!      RES( dims%y_i : dims%y_e ) =                                             &
!        RES( dims%y_i : dims%y_e ) - SCALE_C * DC
!
!!  Find the largest residual and component of the search direction
!
!      IF ( control%out > 0 .AND. print_level >= 2 ) THEN
!        errorg = MAX( MAXVAL( ABS( RES( dims%x_s : dims%x_e ) - RHS_x ) ),     &
!                      MAXVAL( ABS( RES( dims%c_s : dims%c_e ) - RHS_c ) ) )
!        IF ( print_level >= 4 ) THEN
!          DO i = 1, n
!            IF ( ABS( RES( i ) - RHS_x( i ) ) > res_tol )                      &
!              WRITE( control%out, 2010 )                                       &
!                prefix, 'X', i, RES( i ), RHS_x( i )
!          END DO
!          DO i = dims%c_l_start, dims%c_u_end
!            IF ( ABS( RES( dims%c_b + i ) - RHS_c( i ) ) > res_tol )           &
!              WRITE( control%out, 2010 )                                       &
!                prefix, 'C', i, RES( dims%c_b + i ), RHS_c( i )
!          END DO
!        END IF
!        IF ( m > 0 ) THEN
!          errorc = MAXVAL( ABS( RES( dims%y_s : dims%y_e ) - RHS_y ) )
!        ELSE
!          errorc = zero
!        END IF
!        IF ( print_level >= 4 ) THEN
!          DO i = 1, m
!            IF ( ABS( RES( dims%y_s + i - 1 ) - RHS_y( i ) ) > res_tol )       &
!              WRITE( control%out, 2010 )                                       &
!                prefix, 'C', I, RES( dims%y_s + i - 1 ), RHS_y( i )
!          END DO
!        END IF
!        WRITE( control%out, "( ' ',                                            &
!       & /, A, '    ***  Max component of gradient  residuals = ', ES12.4,     &
!       & /, A, '    ***  Max component of contraint residuals = ', ES12.4,     &
!       & /, A, '    ***  Max component of search direction    = ', ES12.4 )" ) &
!           prefix, errorg, prefix, errorc, prefix,                             &
!           MAX( MAXVAL( ABS( DX ) ), MAXVAL( ABS( DC ) ), MAXVAL( ABS( DY ) ) )
!      END IF
!      RETURN
!
!!  Non-executable statements
!
! 2010 FORMAT( A,  ' ', A1, '-residual', I6, ' lhs = ', ES12.4,' rhs = ', ES12.4 )
!
!!  End of subroutine LSQP_residual
!
!      END SUBROUTINE LSQP_residual

!-*-*-*-  L S Q P _ C O M P U T E _ M A X S T E P   S U B R O U T I N E  -*-*-*-

      SUBROUTINE LSQP_compute_maxstep( dims, n, m, Z_l, Z_u, DZ_l, DZ_u,       &
                                       X, DIST_X_l, DIST_X_u, DX,              &
                                       Y_l, Y_u, DY_l, DY_u,                   &
                                       DIST_C_l, DIST_C_u, DC,                 &
                                       gamma_b, gamma_f, nu, nbnds,            &
                                       alpha_max_b, alpha_max_f,               &
                                       print_level, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the maximum allowable stepsizes alpha_max_b, which balances the
!  complementarity ie, such that
!
!      min (x-l)_i(z_l)_i - (gamma_b / nbds)( <x-l,z_l> + <x-u,z_u> ) >= 0
!       i
!  and
!      min (x-u)_i(z_u)_i - (gamma_b / nbds)( <x-l,z_l> + <x-u,z_u> ) >= 0 ,
!       i
!
!  and alpha_max_f, which favours feasibility over complementarity,
!  ie, such that
!
!      <x-l,z_l> + <x-u,z_u> >= nu * gamma_f
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, nbnds, print_level
      REAL ( KIND = wp ), INTENT( IN ) :: gamma_b, gamma_f, nu
      REAL ( KIND = wp ), INTENT( OUT ) :: alpha_max_b, alpha_max_f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, DX
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: Z_l, DZ_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : n ) :: Z_u, DZ_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DY_l, DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DY_u, DIST_C_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : m ) :: DC
      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
      TYPE ( LSQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, nroots

!  Local variables

      REAL ( KIND = wp ) :: compc, compl, compq, coef0, coef1, coef2
      REAL ( KIND = wp ) :: coef0_f, coef1_f, coef2_f
      REAL ( KIND = wp ) :: root1, root2, tol, alpha, alp, nu_gamma_f

      alpha_max_b = infinity ; alpha_max_f = infinity
      inform%status = GALAHAD_ok
      IF ( nbnds == 0 ) RETURN
      tol = epsmch ** 0.75

!  ================================================
!             part to compute alpha_max_b
!  ================================================

!  Compute the coefficients for the quadratic expression
!  for the overall complementarity

      coef0_f = zero ; coef1_f = zero ; coef2_f = zero
      DO i = dims%x_free + 1, dims%x_l_start - 1
        coef0_f = coef0_f + X( i ) * Z_l( i )
        coef1_f = coef1_f + X( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
        coef2_f = coef2_f + DX( i ) * DZ_l( i )
      END DO
      DO i = dims%x_l_start, dims%x_l_end
        coef0_f = coef0_f + DIST_X_l( i ) * Z_l( i )
        coef1_f = coef1_f + DIST_X_l( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
        coef2_f = coef2_f + DX( i ) * DZ_l( i )
      END DO
      DO i = dims%x_u_start, dims%x_u_end
        coef0_f = coef0_f - DIST_X_u( i ) * Z_u( i )
        coef1_f = coef1_f - DIST_X_u( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
        coef2_f = coef2_f + DX( i ) * DZ_u( i )
      END DO
      DO i = dims%x_u_end + 1, n
        coef0_f = coef0_f + X( i ) * Z_u( i )
        coef1_f = coef1_f + X( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
        coef2_f = coef2_f + DX( i ) * DZ_u( i )
      END DO
      DO i = dims%c_l_start, dims%c_l_end
        coef0_f = coef0_f + DIST_C_l( i ) * Y_l( i )
        coef1_f = coef1_f + DIST_C_l( i ) * DY_l( i ) + DC( i ) * Y_l( i )
        coef2_f = coef2_f + DC( i ) * DY_l( i )
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        coef0_f = coef0_f - DIST_C_u( i ) * Y_u( i )
        coef1_f = coef1_f - DIST_C_u( i ) * DY_u( i ) + DC( i ) * Y_u( i )
        coef2_f = coef2_f + DC( i ) * DY_u( i )
      END DO

!  Scale these coefficients

      compc = - gamma_b * coef0_f / nbnds ; compl = - gamma_b * coef1_f / nbnds
      compq = - gamma_b * coef2_f / nbnds
!     write(6,"( ' gamma_b ', ES12.4, I6 )" ) gamma_b, nbnds
!     write( 6, "( 3ES10.2 )" )  compq, compl, compc

!  Compute the coefficients for the quadratic expression
!  for the individual complementarity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        coef0 = compc + X( i ) * Z_l( i )
        coef1 = compl + X( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
        coef2 = compq + DX( i ) * DZ_l( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      DO i = dims%x_l_start, dims%x_l_end
        coef0 = compc + DIST_X_l( i ) * Z_l( i )
        coef1 = compl + DIST_X_l( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
        coef2 = compq + DX( i ) * DZ_l( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      DO i = dims%x_u_start, dims%x_u_end
        coef0 = compc - DIST_X_u( i ) * Z_u( i )
        coef1 = compl - DIST_X_u( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
        coef2 = compq + DX( i ) * DZ_u( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      DO i = dims%x_u_end + 1, n
        coef0 = compc + X( i ) * Z_u( i )
        coef1 = compl + X( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
        coef2 = compq + DX( i ) * DZ_u( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      DO i = dims%c_l_start, dims%c_l_end
        coef0 = compc + DIST_C_l( i ) * Y_l( i )
        coef1 = compl + DIST_C_l( i ) * DY_l( i ) + DC( i ) * Y_l( i )
        coef2 = compq + DC( i ) * DY_l( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      DO i = dims%c_u_start, dims%c_u_end
        coef0 = compc - DIST_C_u( i ) * Y_u( i )
        coef1 = compl - DIST_C_u( i ) * DY_u( i ) + DC( i ) * Y_u( i )
        coef2 = compq + DC( i ) * DY_u( i )
        coef0 = MAX( coef0, zero )
        CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2,  &
                              .FALSE. )
!       write( 6, "( 3ES10.2, 2ES22.14 )" )  coef2, coef1, coef0, root1, root2
        IF ( nroots == 2 ) THEN
          IF ( coef2 > zero ) THEN
            IF ( root2 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = root2
          END IF
        ELSE IF ( nroots == 1 ) THEN
          IF ( root1 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = infinity
        END IF
        IF ( alpha < alpha_max_b ) alpha_max_b = alpha
      END DO

      IF ( - compc <= epsmch ** 0.75 ) alpha_max_b = 0.99_wp * alpha_max_b

!  An error has occured. Investigate

      IF ( alpha_max_b <= zero ) THEN
        IF ( control%out > 0 .AND. print_level >= 2 )                          &
          WRITE( control%out, 2020 ) alpha_max_b
        DO i = dims%x_free + 1, dims%x_l_start - 1
          coef0 = compc + X( i ) * Z_l( i )
          coef1 = compl + X( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
          coef2 = compq + DX( i ) * DZ_l( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'L', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                  'X', i, 'L', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'X', i, 'L', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'X', i, 'L', alpha
            END IF
          END IF
        END DO
        DO i = dims%x_l_start, dims%x_l_end
          coef0 = compc + DIST_X_l( i ) * Z_l( i )
          coef1 = compl + DIST_X_l( i ) * DZ_l( i ) + DX( i ) * Z_l( i )
          coef2 = compq + DX( i ) * DZ_l( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'L', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'L', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'X', i, 'L', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'X', i, 'L', alpha
            END IF
          END IF
        END DO
        DO i = dims%x_u_start, dims%x_u_end
          coef0 = compc - DIST_X_u( i ) * Z_u( i )
          coef1 = compl - DIST_X_u( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
          coef2 = compq + DX( i ) * DZ_u( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'U', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'U', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'X', i, 'U', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'X', i, 'U', alpha
            END IF
          END IF
        END DO
        DO i = dims%x_u_end + 1, n
          coef0 = compc + X( i ) * Z_u( i )
          coef1 = compl + X( i ) * DZ_u( i ) + DX( i ) * Z_u( i )
          coef2 = compq + DX( i ) * DZ_u( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'U', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'X', i, 'U', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'X', i, 'U', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'X', i, 'U', alpha
            END IF
          END IF
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          coef0 = compc + DIST_C_l( i ) * Y_l( i )
          coef1 = compl + DIST_C_l( i ) * DY_l( i ) + DC( i ) * Y_l( i )
          coef2 = compq + DC( i ) * DY_l( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'C', i, 'L', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'C', i, 'L', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'C', i, 'L', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'C', i, 'L', alpha
            END IF
          END IF
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          coef0 = compc - DIST_C_u( i ) * Y_u( i )
          coef1 = compl - DIST_C_u( i ) * DY_u( i ) + DC( i ) * Y_u( i )
          coef2 = compq + DC( i ) * DY_u( i )
          CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,       &
                                root2, .FALSE. )
          IF ( nroots == 2 ) THEN
            IF ( coef2 > zero ) THEN
               IF ( root2 > zero ) THEN
                  alpha = root1
               ELSE
                  alpha = infinity
               END IF
            ELSE
               alpha = root2
            END IF
          ELSE IF ( nroots == 1 ) THEN
            IF ( root1 > zero ) THEN
               alpha = root1
            ELSE
               alpha = infinity
            END IF
          ELSE
            alpha = infinity
          END IF
          IF ( alpha == alpha_max_b ) THEN
            IF ( control%out > 0 .AND. print_level >= 2 ) THEN
               IF ( nroots == 2 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'C', i, 'U', coef0, coef1, coef2, root1, root2
               ELSE IF ( nroots == 1 ) THEN
                 WRITE( control%out, 2000 )                                    &
                   'C', i, 'U', coef0, coef1, coef2, root1
               ELSE
                 WRITE( control%out, 2000 ) 'C', i, 'U', coef0, coef1, coef2
               END IF
               WRITE( control%out, 2010 ) 'C', i, 'U', alpha
            END IF
          END IF
        END DO

        DO i = dims%x_free + 1, dims%x_l_start - 1
          coef0 = X( i ) * Z_l( i )
          coef1 = DX( i ) * Z_l( i ) + X( i ) * DZ_l( i )
          coef2 = DX( i ) * DZ_l( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'X', i, 'L', alp, alpha
        END DO
        DO i = dims%x_l_start, dims%x_l_end
          coef0 = DIST_X_l( i ) * Z_l( i )
          coef1 = DX( i ) * Z_l( i ) + DIST_X_l( i ) * DZ_l( i )
          coef2 = DX( i ) * DZ_l( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'X', i, 'L', alp, alpha
        END DO
        DO i = dims%x_u_start, dims%x_u_end
          coef0 = - DIST_X_u( i ) * Z_u( i )
          coef1 = DX( i ) * Z_u( i ) - DIST_X_u( i ) * DZ_u( i )
          coef2 = DX( i ) * DZ_u( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'X', i, 'U', alp, alpha
        END DO
        DO i = dims%x_u_end + 1, n
          coef0 = X( i ) * Z_u( i )
          coef1 = DX( i ) * Z_u( i ) + X( i ) * DZ_u( i )
          coef2 = DX( i ) * DZ_u( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'X', i, 'U', alp, alpha
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          coef0 = DIST_C_l( i ) * Y_l( i )
          coef1 = DC( i ) * Y_l( i ) + DIST_C_l( i ) * DY_l( i )
          coef2 = DC( i ) * DY_l( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'C', i, 'L', alp, alpha
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          coef0 = - DIST_C_u( i ) * Y_u( i )
          coef1 = DC( i ) * Y_u( i ) - DIST_C_u( i ) * DY_u( i )
          coef2 = DC( i ) * DY_u( i )
          alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 )
          IF ( control%out > 0 .AND. print_level >= 2 )                        &
            WRITE( control%out, 2030 ) 'C', i, 'U', alp, alpha
        END DO
        alp = alpha_max_b ; alpha = compc + alp * ( compl + alp * compq )
        IF ( control%out > 0 .AND. print_level >= 2 ) THEN
          WRITE( control%out, 2040 ) alpha
          WRITE( control%out, 2020 ) alpha_max_b
        END IF
        WRITE( control%out, "( ' -ve step, no further progress possible ' )" )
        inform%status = GALAHAD_error_tiny_step
        RETURN
      END IF

!  ================================================
!             part to compute alpha_max_f
!  ================================================

      nu_gamma_f = nu * gamma_f

!  Compute the coefficients for the quadratic expression
!  for the overall complementarity, remembering to first
!  subtract the term for the feasibility

      coef0_f = coef0_f - nu_gamma_f
      coef1_f = coef1_f + nu_gamma_f

!  Compute the coefficients for the quadratic expression
!  for the individual complementarity
!
      CALL ROOTS_quadratic( coef0_f, coef1_f, coef2_f, tol,                    &
                            nroots, root1, root2, .FALSE. )
      IF ( nroots == 2 ) THEN
        IF ( coef2_f > zero ) THEN
          IF ( root2 > zero ) THEN
            alpha = root1
          ELSE
            alpha = infinity
          END IF
        ELSE
          alpha = root2
        END IF
      ELSE IF ( nroots == 1 ) THEN
        IF ( root1 > zero ) THEN
          alpha = root1
        ELSE
          alpha = infinity
        END IF
      ELSE
        alpha = infinity
      END IF
      IF ( alpha < alpha_max_f ) alpha_max_f = alpha
      IF ( - compc <= epsmch ** 0.75 ) alpha_max_f = 0.99_wp * alpha_max_f

      RETURN

!  Non-executable statements

 2000 FORMAT( A1, I6, A1,' coefs', 3ES12.4,' roots', 2ES12.4 )
 2010 FORMAT( A1, I6, A1,' alpha', ES12.4 )
 2020 FORMAT( ' alpha_min ', ES12.4 )
 2030 FORMAT( A1, I6, A1,' value at ', ES12.4,' = ', ES12.4 )
 2040 FORMAT( ' .vs. ', ES12.4 )

!  End of subroutine LSQP_compute_maxstep

      END SUBROUTINE LSQP_compute_maxstep

!-*-*-*-*-*-*-   L S Q P _ A _ B Y _ C O L S   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LSQP_A_by_cols( n, m, a_ne, A_val, A_col, A_ptr, B_val,       &
                                 B_row, B_colptr )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Takes a matrix A stored by co-ordinates, and returns the same matrix
!  as B stored by columns

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, a_ne
      INTEGER, INTENT( IN ), DIMENSION( a_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( a_ne ) :: B_row
      INTEGER, INTENT( OUT ), DIMENSION( n + 1 ) :: B_colptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( a_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( a_ne ) :: B_val

!  Local variables

      INTEGER :: i, j, k, l

!  count the number of nonzeros in each column

      B_colptr( : n ) = 0
      DO l = 1, a_ne
        B_colptr( A_col( l ) ) = B_colptr( A_col( l ) ) + 1
      END DO

!  set the starting addresses for each column

      j = 1
      DO i = 1, n
        l = j
        j = j + B_colptr( i )
        B_colptr( i ) = l
      END DO

!  move the entries from A to B

      DO i = 1, m
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          j = A_col( l )
          k = B_colptr( j )
          B_val( k ) = A_val( l )
          B_row( k ) = i
          B_colptr( j ) = B_colptr( j ) + 1
        END DO
      END DO

!  reset the starting addresses

      DO i = n, 1, - 1
        B_colptr( i + 1 ) = B_colptr( i )
      END DO
      B_colptr( 1 ) = 1

      RETURN

!  End of LSQP_A_by_cols

      END SUBROUTINE LSQP_A_by_cols

!-*-*-*-*-*-   L S Q P _ I N D I C A T O R S   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE LSQP_indicators( dims, n, m, C_l, C_u, C_last, C,              &
                                 DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,      &
                                 DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,       &
                                 Y_last, Z_last, control, C_stat, B_stat )

!  ---------------------------------------------------------------------------

!  Compute indicatirs for active simnple bounds and general constraints

!  C_stat is an INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set

!  B_stat is an INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set

!  ---------------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_last, Z_last
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_last, Y_last
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l, Y_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u, Y_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_l_start : dims%c_u_end ) :: C
      REAL ( KIND = wp ), INTENT(IN ),                                        &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) ::  Z_l
      REAL ( KIND = wp ), INTENT( IN ),                                       &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      TYPE ( LSQP_control_type ), INTENT( IN ) :: control
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: B_stat

!  Local variables

      INTEGER :: i

!     IF ( printd ) WRITE(  control%out,                                     &
!       "( /, ' Constraints : ', /, '                   ',                   &
!    &   '        <------ Bounds ------> ', /                                &
!    &   '      # name       state      Lower       Upper     Multiplier' )" )
!     DO i = dims%c_equality + 1, m
!       IF ( printd ) WRITE(  control%out,"( 2I7, 4ES12.4 )" ) i,            &
!         C_stat( i ), C( i ), C_l( i ), C_u( i ), Y( i )
!     END DO

!     IF ( printd ) WRITE(  control%out,                                     &
!        "( /, ' Solution : ', /,'                    ',                     &
!       &    '        <------ Bounds ------> ', /                            &
!       &    '      # name       state      Lower       Upper       Dual' )" )
!     DO i = dims%x_free + 1, n
!       IF ( printd ) WRITE(  control%out,"( 2I7, 4ES12.4 )" ) i,            &
!         B_stat( i ), X( i ), X_l( i ), X_u( i ), Z( i )
!     END DO

!  equality constraints

      C_stat( : dims%c_equality ) = - 1

!  free variables

      B_stat( : dims%x_free ) = 0

!  Compute the required indicator

!  ----------------------------------
!  Type 1 ("primal") indicator used:
!  ----------------------------------

!    a variable/constraint will be "inactive" if
!        distance to nearest bound > indicator_p_tol
!    for some constant indicator_p_tol close-ish to zero

      IF ( control%indicator_type == 1 ) THEN
        DO i = dims%c_equality + 1, m
          IF ( ABS( C( i ) - C_l( i ) ) < control%indicator_tol_p ) THEN
            IF ( ABS( Y_l( i ) ) < control%indicator_tol_p ) THEN
              C_stat( i ) = - 2
            ELSE
              C_stat( i ) = - 1
            END IF
          ELSE IF ( ABS( C( i ) - C_u( i ) ) < control%indicator_tol_p ) THEN
            IF ( ABS( Y_u( i ) ) < control%indicator_tol_p ) THEN
              C_stat( i ) = 2
            ELSE
              C_stat( i ) = 1
            END IF
          ELSE
            C_stat( i ) = 0
          END IF
        END DO
        DO i = dims%x_free + 1, n
          IF ( ABS( X( i ) - X_l( i ) ) < control%indicator_tol_p ) THEN
            IF ( ABS( Z_l( i ) ) < control%indicator_tol_p ) THEN
              B_stat( i ) = - 2
            ELSE
              B_stat( i ) = - 1
            END IF
          ELSE IF ( ABS( X( i ) - X_u( i ) ) < control%indicator_tol_p ) THEN
            IF ( ABS( Z_u( i ) ) < control%indicator_tol_p ) THEN
              B_stat( i ) = 2
            ELSE
              B_stat( i ) = 1
            END IF
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  --------------------------------------
!  Type 2 ("primal-dual") indicator used:
!  --------------------------------------

!    a variable/constraint will be "inactive" if
!        distance to nearest bound
!          > indicator_tol_pd * size of corresponding multiplier
!    for some constant indicator_tol_pd close-ish to one.

      ELSE IF ( control%indicator_type == 2 ) THEN

!  constraints with lower bounds

        DO i = dims%c_l_start, dims%c_u_start - 1
          IF ( DIST_C_l( i ) > control%indicator_tol_pd * Y_l( i ) ) THEN
            C_stat( i ) = 0
          ELSE
            C_stat( i ) = - 1
          END IF
        END DO

!  constraints with both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          IF ( DIST_C_l( i ) <= DIST_C_u( i ) ) THEN
            IF ( DIST_C_l( i ) > control%indicator_tol_pd * Y_l( i ) ) THEN
              C_stat( i ) = 0
            ELSE
              C_stat( i ) = - 1
            END IF
          ELSE
            IF ( DIST_C_u( i ) > - control%indicator_tol_pd * Y_u( i ) ) THEN
              C_stat( i ) = 0
            ELSE
              C_stat( i ) = 1
            END IF
          END IF
        END DO

!  constraints with upper bounds

        DO i = dims%c_l_end + 1, m
          IF ( DIST_C_u( i ) > - control%indicator_tol_pd * Y_u( i ) ) THEN
            C_stat( i ) = 0
          ELSE
            C_stat( i ) = 1
          END IF
        END DO

!  simple non-negativity

        DO i = dims%x_free + 1, dims%x_l_start - 1
          IF ( X( i ) > control%indicator_tol_pd * Z_l( i ) ) THEN
            B_stat( i ) = 0
          ELSE
            B_stat( i ) = - 1
          END IF
        END DO

!  simple bound from below

        DO i = dims%x_l_start, dims%x_u_start - 1
          IF ( DIST_X_l( i ) > control%indicator_tol_pd * Z_l( i ) ) THEN
            B_stat( i ) = 0
          ELSE
            B_stat( i ) = - 1
          END IF
        END DO

!  simple bound from below and above

        DO i = dims%x_u_start, dims%x_l_end
          IF ( DIST_X_l( i ) <= DIST_X_u( i ) ) THEN
            IF ( DIST_X_l( i ) > control%indicator_tol_pd * Z_l( i ) ) THEN
              B_stat( i ) = 0
            ELSE
              B_stat( i ) = - 1
            END IF
          ELSE
            IF ( DIST_x_u( i ) > - control%indicator_tol_pd * Z_u( i ) ) THEN
              B_stat( i ) = 0
            ELSE
              B_stat( i ) = 1
            END IF
          END IF
        END DO

!  simple bound from above

        DO i = dims%x_l_end + 1, dims%x_u_end
          IF ( DIST_x_u( i ) > - control%indicator_tol_pd * Z_u( i ) ) THEN
            B_stat( i ) = 0
          ELSE
            B_stat( i ) = 1
          END IF
        END DO

!  simple non-positivity

        DO i = dims%x_u_end + 1, n
          IF ( - X( i ) > - control%indicator_tol_pd * Z_u( i ) ) THEN
            B_stat( i ) = 0
          ELSE
            B_stat( i ) = 1
          END IF
        END DO

!  --------------------------------
!  Type 3 ("Tapia") indicator used:
!  --------------------------------

!    a variable/constraint will be "inactive" if
!        distance to nearest bound now
!          > indicator_tol_tapia * distance to same bound at previous iteration
!    for some constant indicator_tol_tapia close-ish to one.

      ELSE IF ( control%indicator_type == 3 ) THEN

!  constraints with lower bounds

        DO i = dims%c_l_start, dims%c_u_start - 1
!WRITE( 6, "( 'i,dc,dc-,ratio', I6,3ES12.4)" )  i, C( i ) - C_l( i ), &
!C_last( i ) - C_l( i ) , ( C( i ) - C_l( i ) ) / (  C_last( i ) - C_l( i ) )
          IF ( ABS( C( i ) - C_l( i ) ) > control%indicator_tol_tapia *        &
               ABS( C_last( i ) - C_l( i ) ) ) THEN
            C_stat( i ) = 0
          ELSE
            IF ( ABS( Y_l( i ) / Y_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              C_stat( i ) = - 1
            ELSE
!             write(6,*) i, ABS( Y_l( i ) / Y_last( i ) )
              C_stat( i ) = - 2
            END IF
          END IF
        END DO

!  constraints with both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          IF ( DIST_C_l( i ) <= DIST_C_u( i ) ) THEN
            IF ( ABS( C( i ) - C_l( i ) ) > control%indicator_tol_tapia *      &
                 ABS( C_last( i ) - C_l( i ) ) ) THEN
              C_stat( i ) = 0
            ELSE
              IF ( ABS( Y_l( i ) / Y_last( i ) )                               &
                   > control%indicator_tol_tapia ) THEN
                C_stat( i ) = - 1
              ELSE
!               write(6,*) i, ABS( Y_l( i ) / Y_last( i ) )
                C_stat( i ) = - 2
              END IF
            END IF
          ELSE
            IF ( ABS( C( i ) - C_u( i ) ) > control%indicator_tol_tapia *      &
                 ABS( C_last( i ) - C_u( i ) ) )  THEN
              C_stat( i ) = 0
            ELSE
              IF ( ABS( Y_u( i ) / Y_last( i ) )                               &
                   > control%indicator_tol_tapia ) THEN
                C_stat( i ) = 1
              ELSE
!               write(6,*) i, ABS( Y_u( i ) / Y_last( i ) )
                C_stat( i ) = 2
              END IF
            END IF
          END IF
        END DO

!  constraints with upper bounds

        DO i = dims%c_l_end + 1, m
!WRITE( 6, "( 'i,dc,dc-,ratio', I6,3ES12.4)" )  i, C_u( i ) - C( i ), &
!C_u( i ) - C_last( i ), ( C_u( i ) - C( i ) ) / ( C_u( i ) - C_last( i ) )
          IF ( ABS( C( i ) - C_u( i ) ) > control%indicator_tol_tapia *        &
               ABS( C_last( i ) - C_u( i ) ) )  THEN
            C_stat( i ) = 0
          ELSE
            IF ( ABS( Y_u( i ) / Y_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              C_stat( i ) = 1
            ELSE
!             write(6,*) i, ABS( Y_u( i ) / Y_last( i ) )
              C_stat( i ) = 2
            END IF
          END IF
        END DO

!  simple non-negativity

        DO i = dims%x_free + 1, dims%x_l_start - 1
          IF ( ABS( X( i ) ) > control%indicator_tol_tapia *                   &
               ABS( X_last( i ) ) ) THEN
            B_stat( i ) = 0
          ELSE
            IF ( ABS( Z_l( i ) / Z_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              B_stat( i ) = - 1
            ELSE
!             write(6,*) i, ABS( Z_l( i ) / Z_last( i ) )
              B_stat( i ) = - 2
            END IF
          END IF
        END DO

!  simple bound from below

        DO i = dims%x_l_start, dims%x_u_start - 1
          IF ( ABS( X( i ) - X_l( i ) ) > control%indicator_tol_tapia *        &
               ABS( X_last( i ) - X_l( i ) ) ) THEN
            B_stat( i ) = 0
          ELSE
            IF ( ABS( Z_l( i ) / Z_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              B_stat( i ) = - 1
            ELSE
!             write(6,*) i, ABS( Z_l( i ) / Z_last( i ) )
              B_stat( i ) = - 2
            END IF
          END IF
        END DO

!  simple bound from below and above

        DO i = dims%x_u_start, dims%x_l_end
          IF ( DIST_X_l( i ) <= DIST_X_u( i ) ) THEN
            IF ( ABS( X( i ) - X_l( i ) ) > control%indicator_tol_tapia *      &
                 ABS( X_last( i ) - X_l( i ) ) ) THEN
              B_stat( i ) = 0
            ELSE
              IF ( ABS( Z_l( i ) / Z_last( i ) )                               &
                   > control%indicator_tol_tapia ) THEN
                B_stat( i ) = - 1
              ELSE
!               write(6,*) i, ABS( Z_l( i ) / Z_last( i ) )
                B_stat( i ) = - 2
              END IF
            END IF
          ELSE
            IF ( ABS( X( i ) - X_u( i ) ) > control%indicator_tol_tapia *      &
                 ABS( X_last( i ) - X_u( i ) ) ) THEN
              B_stat( i ) = 0
            ELSE
              IF ( ABS( Z_u( i ) / Z_last( i ) )                               &
                   > control%indicator_tol_tapia ) THEN
                B_stat( i ) = 1
              ELSE
!               write(6,*) i, ABS( Z_u( i ) / Z_last( i ) )
                B_stat( i ) = 2
              END IF
            END IF
          END IF
        END DO

!  simple bound from above

        DO i = dims%x_l_end + 1, dims%x_u_end
            IF ( ABS( X( i ) - X_u( i ) ) > control%indicator_tol_tapia *      &
                 ABS( X_last( i ) - X_u( i ) ) ) THEN
            B_stat( i ) = 0
          ELSE
            IF ( ABS( Z_u( i ) / Z_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              B_stat( i ) = 1
            ELSE
!             write(6,*) i, ABS( Z_u( i ) / Z_last( i ) )
              B_stat( i ) = 2
            END IF
          END IF
        END DO

!  simple non-positivity

        DO i = dims%x_u_end + 1, n
          IF ( ABS( X( i ) ) > control%indicator_tol_tapia *                   &
               ABS( X_last( i ) ) ) THEN
            B_stat( i ) = 0
          ELSE
            IF ( ABS( Z_u( i ) / Z_last( i ) )                                 &
                 > control%indicator_tol_tapia ) THEN
              B_stat( i ) = 1
            ELSE
!             write(6,*) i, ABS( Z_u( i ) / Z_last( i ) )
              B_stat( i ) = 2
            END IF
          END IF
        END DO
      ELSE
      END IF

!  End of LSQP_indicators

      END SUBROUTINE LSQP_indicators

!  End of module LSQP

   END MODULE GALAHAD_LSQP_double
