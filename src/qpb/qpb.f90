! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 13:20 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ Q P B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. October 17th 1997
!   update released with GALAHAD Version 2.0. February 16th 2005
!   modified to enable sbls and Puiseux extrapolation in GALAHAD Version 2.4. 
!    May 8th 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QPB_double

!     -------------------------------------------------
!     |                                               |
!     | Solve the quadratic program                   |
!     |                                               |
!     |    minimize     1/2 x(T) H x + g(T) x + f     |
!     |    subject to     c_l <= A x <= c_u           |
!     |                   x_l <=  x  <= x_u           |
!     |                                               |
!     | using an interior-point trust-region approach |
!     |                                               |
!     -------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_NORMS_double
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_ies, STRING_are
      USE GALAHAD_QPT_double
      USE GALAHAD_QPP_double
      USE GALAHAD_QPD_double, QPB_data_type => QPD_data_type,                  &
                              QPB_HX => QPD_HX, QPB_AX => QPD_AX
      USE GALAHAD_LSQP_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_FDC_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_FIT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPB_initialize, QPB_read_specfile, QPB_solve, QPB_terminate,   &
                QPB_solve_main, QPB_feasible_for_BQP, QPB_data_type,           &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: max_real_store_ratio = 100
      INTEGER, PARAMETER :: max_integer_store_ratio = 100
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: res_large = one
      REAL ( KIND = wp ), PARAMETER :: remote = ten ** 10
      REAL ( KIND = wp ), PARAMETER :: bar_min = zero
!     REAL ( KIND = wp ), PARAMETER :: z_min = ten ** ( - 12 )
      REAL ( KIND = wp ), PARAMETER :: z_min = epsmch

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: QPB_control_type

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

!   the maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1
!
        INTEGER :: cg_maxit = 200

!   the preconditioner to be used for the CG is defined by precon. 
!    Possible values are

!      0  automatic 
!      1  no preconditioner, i.e, the identity within full factorization
!      2  full factorization
!      3  band within full factorization
!      4  diagonal using the barrier terms within full factorization  (OBSOLETE)

        INTEGER :: precon = 0

!   the semi-bandwidth of a band preconditioner, if appropriate       (OBSOLETE)

        INTEGER :: nsemib = 5

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

        INTEGER :: extrapolate = 2

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

!   tolerances used to terminate the inner iteration (for given mu):
!     dual feasibility <= MAX( theta_d * mu ** beta, 0.99 * stop_d )
!     complementarity  <= MAX( theta_c * mu ** beta, 0.99 * stop_d )

        REAL ( KIND = wp ) :: theta_d = one
        REAL ( KIND = wp ) :: theta_c = one
        REAL ( KIND = wp ) :: beta = 1.01_wp

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

!   if the objective function value is smaller than obj_unbounded, it will be 
!    flagged as unbounded from below.

        REAL ( KIND = wp ) :: obj_unbounded = - one / epsmch

!   the threshold pivot used by the matrix factorization.
!    See the documentation for SBLS for details                       (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol = epsmch

!   the threshold pivot used by the matrix factorization when attempting to 
!    detect linearly dependent constraints.
!    See the documentation for FDC for details                        (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol_for_dependencies = half

!   any pivots smaller than zero_pivot in absolute value will be regarded to be
!    zero when attempting to detect linearly dependent constraints    (OBSOLETE)

        REAL ( KIND = wp ) :: zero_pivot = epsmch

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than 
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   the search direction is considered as an acceptable approximation
!    to the minimizer of the model if the gradient of the model in the 
!    preconditioning(inverse) norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )

        REAL ( KIND = wp ) :: inner_stop_relative = point01
        REAL ( KIND = wp ) :: inner_stop_absolute = epsmch

!   the initial trust-region radius

        REAL ( KIND = wp ) :: initial_radius  = - one

!  start terminal extrapolation when mu reaches mu_min

        REAL ( KIND = wp ) :: mu_min = ten ** ( - 5 )

!   a search direction which gives at least inner_fraction_opt times the 
!    optimal model decrease will be found

        REAL ( KIND = wp ) :: inner_fraction_opt = point1

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

!   if %center is true, the algorithm will use the analytic center
!    of the feasible set as its initial feasible point. Otherwise, a feasible
!    point as close as possible to the initial point will be used. We recommend
!    using the analytic center

        LOGICAL :: center = .TRUE.

!   if %primal, is true, a primal barrier method will be used in  place of the 
!    primal-dual method

        LOGICAL :: primal = .FALSE.

!  If extrapolation is to be used, decide between Puiseux and Taylor series

        LOGICAL :: puiseux = .TRUE.

!   if %feasol is true, the final solution obtained will be perturbed so that
!    variables close to their bounds are moved onto these bounds

        LOGICAL :: feasol = .FALSE.

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

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "QPBPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for LSQP

        TYPE ( LSQP_control_type ) :: LSQP_control

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for GLTR

        TYPE ( GLTR_control_type ) :: GLTR_control        

!  control parameters for FIT

        TYPE ( FIT_control_type ) :: FIT_control        
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPB_time_type

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

!  the total CPU time spent in the initial-point phase of the package

        REAL ( KIND = wp ) :: phase1_total = 0.0

!  the CPU time spent analysing the required matrices prior to factorization
!    in the inital-point phase

        REAL ( KIND = wp ) :: phase1_analyse = 0.0

!  the CPU time spent factorizing the required matrices in the inital-point 
!    phase

        REAL ( KIND = wp ) :: phase1_factorize = 0.0

!  the CPU time spent computing the search direction in the inital-point phase

        REAL ( KIND = wp ) :: phase1_solve = 0.0

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

!  the total clock time spent in the initial-point phase of the package

        REAL ( KIND = wp ) :: clock_phase1_total = 0.0

!  the clock time spent analysing the required matrices prior to factorization
!    in the inital-point phase

        REAL ( KIND = wp ) :: clock_phase1_analyse = 0.0

!  the clock time spent factorizing the required matrices in the inital-point 
!    phase

        REAL ( KIND = wp ) :: clock_phase1_factorize = 0.0

!  the clock time spent computing the search direction in the inital-point phase

        REAL ( KIND = wp ) :: clock_phase1_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: QPB_inform_type

!  return status. See QPB_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the total number of conjugate gradient iterations required

        INTEGER :: cg_iter = - 1

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER  ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER  ( KIND = long ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = 0

!  the total number of "wasted" function evaluations during the linesearch

        INTEGER :: nbacts = 0

!  the total number of factorizations which were modified to ensure that the 
!   matrix was an appropriate preconditioner

        INTEGER :: nmods = 0

!  the value of the objective function at the best estimate of the solution 
!   determined by QPB_solve

        REAL ( KIND = wp ) :: obj = infinity

!  the smallest pivot which was not judged to be zero when detecting linearly 
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  timings (see above)

        TYPE ( QPB_time_type ) :: time

!  inform parameters for LSQP

        TYPE ( LSQP_inform_type ) :: LSQP_inform

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  return information from GLTR

        TYPE ( GLTR_info_type ) :: GLTR_inform

!  return information from FIT

        TYPE ( FIT_inform_type ) :: FIT_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   Q P B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QPB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QPB. This routine should be called before
!  QPB_solve
! 
!  --------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( QPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPB_control_type ), INTENT( OUT ) :: control        
      TYPE ( QPB_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Real parameters

      control%stop_p  = epsmch ** 0.33
      control%stop_c  = epsmch ** 0.33
      control%stop_d  = epsmch ** 0.33
      control%obj_unbounded = - epsmch ** ( - 2 )
      control%pivot_tol = epsmch ** 0.75
      control%zero_pivot = epsmch ** 0.75
      control%indicator_tol_p = control%stop_p
      control%inner_stop_absolute = SQRT( epsmch )

!  Set LSQP control parameters

      CALL LSQP_initialize( data, control%LSQP_control, inform%LSQP_inform )

!  Reset relevant LSQP control parameters

      control%LSQP_control%FDC_control%max_infeas = control%stop_p
      control%LSQP_control%indicator_type = control%indicator_type
      control%LSQP_control%indicator_tol_p = control%indicator_tol_p
      control%LSQP_control%indicator_tol_pd = control%indicator_tol_pd
      control%LSQP_control%indicator_tol_tapia = control%indicator_tol_tapia
      control%LSQP_control%feasol = .FALSE.
      control%LSQP_control%prefix = '" - LSQP:"                    '

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%max_infeas = control%stop_p
      control%FDC_control%prefix = '" - FDC:"                     '

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%prefix = '" - SBLS:"                    '
      control%SBLS_control%SLS_control%relative_pivot_tolerance =              &
        control%pivot_tol
      control%SBLS_control%SLS_control%zero_tolerance = control%zero_pivot

!  Set GLTR control parameters

      CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,              &
                            inform%GLTR_inform )
      control%GLTR_control%unitm = .FALSE.
      control%GLTR_control%Lanczos_itmax = 5
      control%GLTR_control%prefix = '" - GLTR:"                    '

!  Set FIT control parameters

      CALL FIT_initialize( data%FIT_data, control%FIT_control,                 &
                           inform%FIT_inform )

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN  

!  End of QPB_initialize

      END SUBROUTINE QPB_initialize

!-*-*-*-*-   Q P B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QPB_initialize could (roughly) 
!  have been set as:

! BEGIN QPB SPECIFICATIONS (DEFAULT)
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
!  maximum-number-of-cg-iterations                   200
!  preconditioner-used                               0
!  semi-bandwidth-for-band-preconditioner            5
!  indicator-type-used                               3
!  extrapolate-solution                              1
!  path-history-length                               1
!  path-derivatives-used                             5
!  path-fit-order                                    -1
!  restore-problem-on-output                         0
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  inner-dual-stop-tolerance                         1.0
!  inner-complementarity-stop-tolerance              1.0
!  inner-stop-power                                  1.01
!  mininum-initial-primal-feasibility                1.0
!  mininum-initial-dual-feasibility                  1.0
!  initial-barrier-parameter                         -1.0
!  poor-iteration-tolerance                          0.98
!  minimum-objective-before-unbounded                -1.0D+32
!  pivot-tolerance-used                              1.0D-12
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  identical-bounds-tolerance                        1.0D-15
!  initial-trust-region-radius                       -1.0
!  inner-iteration-fraction-optimality-required      0.1
!  inner-iteration-relative-accuracy-required        0.01
!  inner-iteration-absolute-accuracy-required        1.0E-8
!  minimum-barrier-before-final-extrapolation        1.0D-5
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  start-at-analytic-center                          T
!  primal-barrier-used                               F
!  puiseux-extrapolation                             T
!  move-final-solution-onto-bound                    F
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     QPBPROB.SIF
!  output-line-prefix                                ""
! END QPB SPECIFICATIONS

!  Dummy arguments

      TYPE ( QPB_control_type ), INTENT( INOUT ) :: control        
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
      INTEGER, PARAMETER :: cg_maxit = infeas_max + 1
      INTEGER, PARAMETER :: precon = cg_maxit + 1
      INTEGER, PARAMETER :: nsemib = precon + 1
      INTEGER, PARAMETER :: restore_problem = nsemib + 1
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
      INTEGER, PARAMETER :: theta_d = stop_c + 1
      INTEGER, PARAMETER :: theta_c = theta_d + 1
      INTEGER, PARAMETER :: beta = theta_c + 1
      INTEGER, PARAMETER :: prfeas = beta + 1
      INTEGER, PARAMETER :: dufeas = prfeas + 1
      INTEGER, PARAMETER :: muzero = dufeas + 1
      INTEGER, PARAMETER :: reduce_infeas = muzero + 1
      INTEGER, PARAMETER :: obj_unbounded = reduce_infeas + 1
      INTEGER, PARAMETER :: pivot_tol = obj_unbounded + 1
      INTEGER, PARAMETER :: pivot_tol_for_dependencies = pivot_tol + 1
      INTEGER, PARAMETER :: zero_pivot = pivot_tol_for_dependencies + 1
      INTEGER, PARAMETER :: identical_bounds_tol = zero_pivot + 1
      INTEGER, PARAMETER :: inner_stop_relative = identical_bounds_tol + 1
      INTEGER, PARAMETER :: inner_stop_absolute = inner_stop_relative + 1
      INTEGER, PARAMETER :: initial_radius = inner_stop_absolute + 1
      INTEGER, PARAMETER :: mu_min = initial_radius + 1
      INTEGER, PARAMETER :: inner_fraction_opt = mu_min + 1
      INTEGER, PARAMETER :: indicator_tol_p = inner_fraction_opt + 1
      INTEGER, PARAMETER :: indicator_tol_pd = indicator_tol_p + 1
      INTEGER, PARAMETER :: indicator_tol_tapia = indicator_tol_pd + 1
      INTEGER, PARAMETER :: cpu_time_limit = indicator_tol_tapia + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: remove_dependencies = clock_time_limit + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general =                     &
                              remove_dependencies + 1
      INTEGER, PARAMETER :: center = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: primal = center + 1
      INTEGER, PARAMETER :: puiseux = primal + 1
      INTEGER, PARAMETER :: feasol = puiseux + 1
      INTEGER, PARAMETER :: array_syntax_worse_than_do_loop = feasol + 1
      INTEGER, PARAMETER :: space_critical = array_syntax_worse_than_do_loop + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
      INTEGER, PARAMETER :: prefix = sif_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'QPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

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
      spec( cg_maxit )%keyword = 'maximum-number-of-cg-iterations'
      spec( precon )%keyword = 'preconditioner-used'
      spec( nsemib )%keyword = 'semi-bandwidth-for-band-preconditioner'
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
      spec( theta_d )%keyword = 'inner-dual-stop-tolerance'
      spec( theta_c )%keyword = 'inner-complementarity-stop-tolerance'
      spec( beta  )%keyword = 'inner-stop-power'
      spec( prfeas )%keyword = 'mininum-initial-primal-feasibility'
      spec( dufeas )%keyword = 'mininum-initial-dual-feasibility'
      spec( muzero )%keyword = 'initial-barrier-parameter'
      spec( reduce_infeas )%keyword = 'poor-iteration-tolerance'
      spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
      spec( pivot_tol )%keyword = 'pivot-tolerance-used'
      spec( pivot_tol_for_dependencies )%keyword =                             &
        'pivot-tolerance-used-for-dependencies'
      spec( zero_pivot )%keyword = 'zero-pivot-tolerance'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( inner_stop_relative )%keyword =                                    &
        'inner-iteration-relative-accuracy-required'
      spec( inner_stop_absolute )%keyword =                                    &
        'inner-iteration-absolute-accuracy-required'
      spec( initial_radius )%keyword = 'initial-trust-region-radius'
      spec( mu_min )%keyword = 'minimum-barrier-before-final-extrapolation'
      spec( inner_fraction_opt )%keyword =                                     &
        'inner-iteration-fraction-optimality-required'
      spec( indicator_tol_p )%keyword = 'primal-indicator-tolerance'
      spec( indicator_tol_pd )%keyword = 'primal-dual-indicator-tolerance'
      spec( indicator_tol_tapia )%keyword = 'tapia-indicator-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( center )%keyword = 'start-at-analytic-center'
      spec( primal )%keyword = 'primal-barrier-used'
      spec( puiseux )%keyword = 'puiseux-extrapolation'
      spec( feasol )%keyword = 'move-final-solution-onto-bound'
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
     CALL SPECFILE_assign_value( spec( cg_maxit ),                             &
                                 control%cg_maxit,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( precon ),                               &
                                 control%precon,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( nsemib ),                               &
                                 control%nsemib,                               &
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
     CALL SPECFILE_assign_value( spec( theta_d ),                              &
                                 control%theta_d,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( theta_c ),                              &
                                 control%theta_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( beta ),                                 &
                                 control%beta,                                 &
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
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
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
     CALL SPECFILE_assign_value( spec( inner_stop_relative ),                  &
                                 control%inner_stop_relative,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( inner_stop_absolute ),                  &
                                 control%inner_stop_absolute,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_radius ),                       &
                                 control%initial_radius,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_min ),                               &
                                 control%mu_min,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( inner_fraction_opt ),                   &
                                 control%inner_fraction_opt,                   &
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
     CALL SPECFILE_assign_value( spec( center ),                               &
                                 control%center,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( primal ),                               &
                                 control%primal,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( puiseux ),                              &
                                 control%puiseux,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasol ),                               &
                                 control%feasol,                               &
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

!  Make sure that inifinity is set consistently

      control%LSQP_control%infinity = control%infinity

!  Read the specfile for LSQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL LSQP_read_specfile( control%LSQP_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-LSQP')
      ELSE
        CALL LSQP_read_specfile( control%LSQP_control, device )
      END IF

!  Reset relevant LSQP control parameters

      control%LSQP_control%indicator_type = control%indicator_type
      control%LSQP_control%indicator_tol_p = control%indicator_tol_p
      control%LSQP_control%indicator_tol_pd = control%indicator_tol_pd
      control%LSQP_control%indicator_tol_tapia = control%indicator_tol_tapia
      control%LSQP_control%feasol = .FALSE.
    
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

!  Read the specfile for GLTR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL GLTR_read_specfile( control%GLTR_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-GLTR')
      ELSE
        CALL GLTR_read_specfile( control%GLTR_control, device )
      END IF

!  Read the specfile for FIT

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FIT_read_specfile( control%FIT_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FIT' )
      ELSE
        CALL FIT_read_specfile( control%FIT_control, device )
      END IF

      RETURN

      END SUBROUTINE QPB_read_specfile

!-*-*-*-*-*-*-*-*-   Q P B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE QPB_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix, 
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
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
!    to be solved since the last call to QPB_initialize, and .FALSE. if
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
!  data is a structure of type QPB_data_type which holds private internal data
!
!  control is a structure of type QPB_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QPB_initialize. See QPB_initialize 
!   for details
!
!  inform is a structure of type QPB_inform_type that provides 
!    information on exit from QPB_solve. The component status 
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
!        prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', , 'DIAGONAL'}
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.

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
!   -23 an entry from the strict upper triangle of H has been input.
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
      TYPE ( QPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPB_control_type ), INTENT( IN ) :: control
      TYPE ( QPB_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i, j, l, tiny_x, tiny_c, n_depen, n_more_depen
      INTEGER :: nzc, n_sbls
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: time_fd
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now, clock_fd
      REAL ( KIND = wp ) :: tol, f, av_bnd
      LOGICAL :: printi, first_pass, center, reset_bnd, lsqp
      LOGICAL :: remap_fixed, remap_freed, remap_more_freed, stat_required
      LOGICAL :: diagonal_qp, convex_diagonal_qp
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( LSQP_control_type ) :: LSQP_control

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering QPB_solve ' )" ) prefix

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

      inform%obj = prob%f
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )

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

!  Check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix,  inform%status 
          GO TO 800 
        ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
        ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
          prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO   
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( /, A, '   **  Warning: one or more variable bounds reset' )" ) prefix

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
       "( A /, '   **  Warning: one or more constraint bounds reset' )" ) prefix

!  Set Hessian and gradient types to generic - this may change in future

      prob%Hessian_kind = - 1 ; prob%gradient_kind = - 1

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
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
             &   ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                 &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  Perform the preprocessing

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                     &
                          data%QPP_inform, data%dims, prob,                   &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
  
!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I3, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status 
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == GALAHAD_error_upper_entry )                    &
            WRITE( control%error, 2020 ) prefix 
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
             & ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prefix, prefix, prob%n, prob%m, data%a_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  Recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform, prob,                 &
                          get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record
 
!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( A, ' status ', I3, ' after QPP_apply')" ) &
               prefix, data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) prefix, inform%status 
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            GO TO 800 
          END IF 
        END IF 
        data%trans = data%trans + 1

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
      END IF

!  Special case: no free variables

      IF ( prob%n == 0 ) THEN
        prob%Y( : prob%m ) = zero
        remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
        n_depen = 0
        inform%obj = prob%f
        i = COUNT( ABS( prob%C_l( : data%dims%c_equality ) ) >                 &
              control%stop_p ) +                                               &
            COUNT( prob%C_l( data%dims%c_l_start : data%dims%c_l_end ) >       &
              control%stop_p ) +                                               &
            COUNT( prob%C_u( data%dims%c_u_start : data%dims%c_u_end ) <       &
              - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        GO TO 700
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
        inform%non_negligible_pivot =                                          &
          inform%FDC_inform%non_negligible_pivot
        inform%factorization_status =                                          &
           inform%FDC_inform%factorization_status
        inform%factorization_integer =                                         &
          inform%FDC_inform%factorization_integer
        inform%factorization_real =                                            &
          inform%FDC_inform%factorization_real
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
             data%FDC_control%SLS_control%absolute_pivot_tolerance )           &
            WRITE( control%out, "(                                             &
       &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,                 &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, A, ' ***  perhaps increase LSQP_control%',                        &
       &     'FDC_control%SLS_control%absolute_pivot_tolerance from',          &
       &  ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )     &
           prefix, prefix, inform%non_negligible_pivot, prefix,                &
           data%FDC_control%SLS_control%absolute_pivot_tolerance, prefix

!  Check for error exits

        IF ( inform%status /= GALAHAD_ok ) THEN

!  Allocate arrays to hold the matrix vector product

          array_name = 'qpb: data%HX'
          CALL SPACE_resize_array( prob%n, data%HX,                            &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  On error exit, compute the current objective function value

          data%HX( : prob%n ) = zero
          CALL QPB_HX( data%dims, prob%n, data%HX( : prob%n ),                 &
                       prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                       prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
          inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),                 &
                                           data%HX( : prob%n ) )               &
                       + DOT_PRODUCT( prob%X( : prob%n ),                      &
                                      prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= GALAHAD_ok ) WRITE( control%error, 2040 )    &
              prefix, inform%status, 'LSQP_dependent'
          END IF
          GO TO 750
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
          prob%Z( : prob%n ) = prob%G( : prob%n )
          CALL QPB_HX( data%dims, prob%n, prob%Z( : prob%n ),                  &
                       prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                       prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
          prob%C( : prob%m ) = zero
          CALL QPB_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                        prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,              &
                        prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
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

        array_name = 'qpb: data%C_freed'
        CALL SPACE_resize_array( n_depen, data%C_freed,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_C_freed( i )
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
        data%h_ne = prob%H%ne 

        IF ( printi ) WRITE( control%out,                                      &
           "( /, A, ' problem dimensions before removal of dependencies:', /,  &
     &        A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" ) &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

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
            WRITE( control%out, "( A, ' status ', I3, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status 
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == GALAHAD_error_upper_entry )                    &
            WRITE( control%error, 2020 ) prefix
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
           "( /, A, ' problem dimensions after removal of dependencies: ', /,  &
     &      A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

      END IF

!  Special case: Bound-constrained QP

      IF ( data%a_ne == 0 .AND. data%h_ne /= 0 .AND. control%extrapolate > 0   &
           .AND. printi ) WRITE( control%error,                                &
        "( A, ' >->->-> turned diagonal bqp solver off for testing ')" ) prefix

      IF ( data%a_ne == 0 .AND. data%h_ne /= 0 .AND.                           &
           control%extrapolate <= 0 ) THEN

!  Check to see if the Hessian is diagonal

        diagonal_qp = .TRUE.
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'DIAGONAL' ) 
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                diagonal_qp = .FALSE. ; GO TO 3
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                diagonal_qp = .FALSE. ; GO TO 3
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
              diagonal_qp = .FALSE. ; GO TO 3
            END IF
          END DO
        END SELECT
  3     CONTINUE

        remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
!       IF ( diagonal_qp ) WRITE( 6, * ) ' turning off diagonal qp detection
!       IF ( .FALSE. ) THEN
        IF ( diagonal_qp ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( /, A, ' Solving separable bound-constrained QP -' )" ) prefix
          IF ( PRESENT( B_stat ) ) THEN
            CALL QPD_solve_separable_BQP( prob, control%infinity,              &
                                          control%obj_unbounded,  inform%obj,  &
                                          inform%feasible, inform%status,      &
                                          B_stat = B_stat( : prob%n ) )
          ELSE
            CALL QPD_solve_separable_BQP( prob, control%infinity,              &
                                          control%obj_unbounded,  inform%obj,  &
                                          inform%feasible, inform%status )
          END IF
          IF ( printi ) THEN
            CALL CLOCK_time( clock_now )
            WRITE( control%out,                                                &
               "( A, ' On exit from QPD_solve_separable_BQP: status = ', I0,   &
            &   ', time = ', F0.2, /, A, ' objective value =', ES12.4 )",      &
              advance = 'no' ) prefix, inform%status, inform%time%clock_total  &
                + clock_now - clock_start, prefix, inform%obj
            IF ( PRESENT( B_stat ) ) THEN
              WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )&
                COUNT( B_stat( : prob%n ) /= 0 ), prob%n
            ELSE
              WRITE( control%out, "( '' )" )
            END IF
          END IF
          inform%iter = 0 ; inform%non_negligible_pivot = zero
          inform%factorization_integer = 0 ; inform%factorization_real = 0
          GO TO 700
        ELSE
          CALL QPB_feasible_for_BQP( prob, data, control, inform )
        END IF

!  General case: QP or LP

      ELSE
        first_pass = .TRUE.
        center = control%center
        f = prob%f
    
!  Check to see if the Hessian is diagonal

        convex_diagonal_qp = .TRUE.

        IF ( convex_diagonal_qp .AND. control%extrapolate > 0 ) THEN
          IF ( printi ) WRITE( control%error,                                  &
        "( A, ' >->->-> turned convex diagonal qp solver off for testing ')" ) &
             prefix
          convex_diagonal_qp = .FALSE. ; GO TO 5
        END IF

        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'DIAGONAL' ) 
          IF ( COUNT( prob%H%val( : prob%n ) < zero ) > 0 )                    &
            convex_diagonal_qp = .FALSE.
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.               &
                   ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
                convex_diagonal_qp = .FALSE. ; GO TO 5
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.               &
                   ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
                convex_diagonal_qp = .FALSE. ; GO TO 5
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.                 &
                 ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
              convex_diagonal_qp = .FALSE. ; GO TO 5
            END IF
          END DO
        END SELECT
  5     CONTINUE

        IF ( data%h_ne == 0 .OR. convex_diagonal_qp ) THEN
          prob%gradient_kind = 2
          IF ( NRM2( prob%n, prob%G, 1 ) <= epsmch ) THEN
            prob%gradient_kind = 0
          ELSE IF ( NRM2( prob%n, prob%G - one, 1 ) <= epsmch ) THEN
            prob%gradient_kind = 1
          END IF
        ELSE
          prob%gradient_kind = 0
        END IF

 10     CONTINUE

!  ==============================
!  Find an initial feasible point
!  ==============================

        LSQP_control = control%LSQP_control
        LSQP_control%infeas_max = control%infeas_max
        LSQP_control%restore_problem = control%restore_problem 
        LSQP_control%infinity = control%infinity
        LSQP_control%stop_p = control%stop_p
        LSQP_control%stop_c = control%stop_c
        LSQP_control%stop_d = control%stop_d
        LSQP_control%muzero = control%muzero
        LSQP_control%reduce_infeas = control%reduce_infeas
        LSQP_control%identical_bounds_tol = control%identical_bounds_tol
        LSQP_control%remove_dependencies = control%remove_dependencies
        LSQP_control%treat_zero_bounds_as_general =                            &
          control%treat_zero_bounds_as_general
        LSQP_control%feasol = .FALSE.
        LSQP_control%identical_bounds_tol =                                    &
          control%LSQP_control%identical_bounds_tol
        LSQP_control%prefix = '" - LSQP:"                    '
        IF ( printi ) THEN
          IF ( data%h_ne == 0 ) THEN
            WRITE( control%out, "( /, A, ' ============= since there is no',   &
           &   ' Hessian, solve as an LP =========== ' )" ) prefix
          ELSE
            WRITE( control%out, "( /, A, ' ==================== feasible',     &
           &   ' point phase ===================== ' )" ) prefix
          END IF
          LSQP_control%print_level = control%print_level
          LSQP_control%start_print = control%start_print
          LSQP_control%stop_print = control%stop_print
          LSQP_control%out = control%out
          LSQP_control%error = control%error
        END IF

!  Either find the solution to the LP (if there is no Hessian term) ...

        IF ( data%h_ne == 0 ) THEN
          lsqp = .TRUE.
          LSQP_control%just_feasible = .FALSE.
          LSQP_control%maxit = control%maxit
          LSQP_control%prfeas = MAX( LSQP_control%prfeas, control%prfeas )
          LSQP_control%dufeas = MAX( LSQP_control%dufeas, control%dufeas )
          prob%Hessian_kind = 0

!  .. or find the solution to the Diagonal QP (if the Hessian is diagonal) ...

        ELSE IF ( convex_diagonal_qp ) THEN
          lsqp = .TRUE.
          LSQP_control%just_feasible = .FALSE.
          LSQP_control%maxit = control%maxit
          LSQP_control%prfeas = MAX( LSQP_control%prfeas, control%prfeas )
          LSQP_control%dufeas = MAX( LSQP_control%dufeas, control%dufeas )
          prob%Hessian_kind = 2

          array_name = 'qpb: prob%X0'
          CALL SPACE_resize_array( prob%n, prob%X0,                            &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'qpb: prob%WEIGHT'
          CALL SPACE_resize_array( prob%n, prob%WEIGHT,                        &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'DIAGONAL' ) 
            prob%WEIGHT( : prob%n ) = prob%H%val( : prob%n )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, prob%n
              DO j = 1, i
                l = l + 1
                IF ( i == j ) prob%WEIGHT( i ) = prob%H%val( l )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            prob%WEIGHT( : prob%n ) = zero
            DO i = 1, prob%n
              DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                j = prob%H%col( l )
                IF ( i == j )                                                  &
                  prob%WEIGHT( i ) = prob%WEIGHT( i ) + prob%H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            prob%WEIGHT( : prob%n ) = zero
            DO l = 1, prob%H%ne
              i = prob%H%row( l ) ; j = prob%H%col( l )
              IF ( i == j )                                                    &
                prob%WEIGHT( i ) = prob%WEIGHT( i ) + prob%H%val( l )
            END DO
          END SELECT
          prob%WEIGHT( : prob%n ) = SQRT( prob%WEIGHT( : prob%n ) )
          prob%X0( : prob%n ) = zero

!  .. or find a centered feasible point ...

        ELSE IF ( center ) THEN
          lsqp = .FALSE.
          LSQP_control%just_feasible = .FALSE.
          LSQP_control%prfeas = control%prfeas
          LSQP_control%dufeas = control%dufeas
          prob%Hessian_kind = 0
          prob%f = zero

!  .. or minimize the distance to the nearest feasible point

        ELSE
          lsqp = .FALSE.
          LSQP_control%just_feasible = .TRUE.
          LSQP_control%prfeas = control%prfeas
          LSQP_control%dufeas = control%dufeas
          prob%Hessian_kind = 1

          array_name = 'qpb: prob%X0'
          CALL SPACE_resize_array( prob%n, prob%X0,                            &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'qpb: data%X0'
          CALL SPACE_resize_array( prob%n, data%X0,                            &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          data%X0 = prob%X( : prob%n )
          prob%X0( : prob%n ) = data%X0( : prob%n )
          prob%f = zero
        END IF

        time_fd = inform%LSQP_inform%time%find_dependent
        clock_fd = inform%LSQP_inform%time%clock_find_dependent

!  phase 1

        IF ( lsqp ) THEN
          LSQP_control%indicator_type = control%indicator_type
          LSQP_control%indicator_tol_p = control%indicator_tol_p
          LSQP_control%indicator_tol_pd = control%indicator_tol_pd
          LSQP_control%indicator_tol_tapia = control%indicator_tol_tapia
          IF ( PRESENT( C_stat ) .AND. PRESENT( B_stat ) ) THEN
            CALL LSQP_solve( prob, data, LSQP_control, inform%LSQP_inform,     &
                             C_stat = C_stat( : prob%m ),                      &
                             B_stat = B_stat( : prob%n ) )
          ELSE
            CALL LSQP_solve( prob, data, LSQP_control, inform%LSQP_inform )
          END IF
        ELSE
          CALL LSQP_solve( prob, data, LSQP_control, inform%LSQP_inform )
        END IF
     
!  record times for phase 1

        inform%time%phase1_total = inform%LSQP_inform%time%total
        inform%time%clock_phase1_total = inform%LSQP_inform%time%clock_total
  
        inform%time%phase1_analyse = inform%LSQP_inform%time%analyse
        inform%time%clock_phase1_analyse =                                     &
          inform%LSQP_inform%time%clock_analyse
        inform%time%phase1_factorize = inform%LSQP_inform%time%factorize
        inform%time%clock_phase1_factorize =                                   &
          inform%LSQP_inform%time%clock_factorize
        inform%time%phase1_solve = inform%LSQP_inform%time%solve
        inform%time%clock_phase1_solve =                                       &
          inform%LSQP_inform%time%clock_solve
   
        inform%time%find_dependent = inform%time%find_dependent +              &
          inform%LSQP_inform%time%find_dependent - time_fd
        inform%time%clock_find_dependent = inform%time%clock_find_dependent +  &
          inform%LSQP_inform%time%clock_find_dependent - clock_fd
   
!  check for successful phase 1

        inform%status = inform%LSQP_inform%status 
        IF ( inform%status == GALAHAD_error_allocate .OR.                      &
             inform%status == GALAHAD_error_deallocate ) THEN
             inform%alloc_status = inform%LSQP_inform%alloc_status
             inform%bad_alloc = inform%LSQP_inform%bad_alloc
          GO TO 920
        END IF

        inform%feasible = inform%LSQP_inform%feasible
        IF ( inform%status == GALAHAD_error_upper_entry .OR.                   &
             inform%status == GALAHAD_error_factorization .OR.                 &
             inform%status == GALAHAD_error_ill_conditioned .OR.               &
             inform%status == GALAHAD_error_tiny_step .OR.                     &
             inform%status == GALAHAD_error_max_iterations .OR.                &
             inform%status == GALAHAD_error_no_center .OR.                     &
           ( inform%status == GALAHAD_error_unbounded .AND. .NOT. lsqp ) ) THEN
          IF ( inform%feasible ) THEN
            inform%status = GALAHAD_ok
          ELSE
            IF ( first_pass .AND. .NOT. lsqp ) THEN
              center = .NOT. center
              first_pass = .FALSE.
              IF ( printi ) WRITE( control%out,                                &
            "( /, A, ' .... have a second attempt at getting feasible ....')" )&
               prefix
              GO TO 10
            END IF
          END IF
        END IF
        prob%f = f
!       IF ( .NOT. lsqp .AND. .NOT. center ) NULLIFY( prob%X0 )
        prob%Hessian_kind = - 1 ; prob%gradient_kind = - 1
  
        inform%alloc_status = inform%LSQP_inform%alloc_status 
        inform%factorization_status = inform%LSQP_inform%factorization_status 

        IF ( printi ) THEN
          WRITE( control%out, "( /, A, 1X, I0, ' integer and ', I0,            &
        &  ' real words required for the factorization' )" )                   &
           prefix, inform%LSQP_inform%factorization_integer,                   &
           inform%LSQP_inform%factorization_real
          IF ( lsqp ) THEN
            WRITE( control%out, "( /, A, ' ====================== end of',     &
             &  ' solution phase ====================== ' )" ) prefix
          ELSE
            WRITE( control%out, "( /, A, ' ===================== end of',      &
             & ' feasible point phase  ==================== ' )" ) prefix
          END IF
        END IF

!  Check for error exits

        IF ( inform%status /= GALAHAD_ok ) THEN

!  On error exit, compute the current objective function value

          IF ( inform%status /= GALAHAD_error_restrictions .AND.               &
               inform%status /= GALAHAD_error_allocate ) THEN
            data%HX( : prob%n ) = zero
            CALL QPB_HX( data%dims, prob%n, data%HX( : prob%n ),               &
                         prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,             &
                         prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
            inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),               &
                                             data%HX( : prob%n ) )             &
                              + DOT_PRODUCT( prob%X( : prob%n ),               &
                                        prob%G( : prob%n ) ) + prob%f
          END IF

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= GALAHAD_ok ) WRITE( control%error, 2040 )    &
              prefix, inform%status, 'LSQP_solve'
          END IF
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          GO TO 700
        END IF

!  Exit if the problem is an LP

!       IF ( prob%gradient_kind == 2 ) THEN
        IF ( lsqp ) THEN
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          inform%iter = inform%LSQP_inform%iter
          inform%factorization_real = inform%LSQP_inform%factorization_real
          inform%factorization_integer =                                       &
            inform%LSQP_inform%factorization_integer
          inform%nfacts = inform%LSQP_inform%nfacts
          inform%nbacts = inform%LSQP_inform%nbacts
          inform%obj = inform%LSQP_inform%obj
          GO TO 700
        END IF

!  ============================
!  Initial feasible point found
!  ============================

!  Check to see if any variables/constraints are flagged as being fixed

        tol = MIN( control%stop_p / ten, SQRT( epsmch ) )
        tiny_x = COUNT( prob%X( data%dims%x_free + 1 : data%dims%x_l_start - 1)&
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_l_start: data%dims%x_l_end ) -     &
                        prob%X_l( data%dims%x_l_start : data%dims%x_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_u_start: data%dims%x_u_end ) -     &
                        prob%X_u( data%dims%x_u_start : data%dims%x_u_end )    &
                        > - tol ) +                                            &
                 COUNT( prob%X( data%dims%x_u_end + 1 : prob%n )          &
                        > - tol )
  
        tiny_c = COUNT( prob%C( data%dims%c_l_start: data%dims%c_l_end ) -     &
                        prob%C_l( data%dims%c_l_start : data%dims%c_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%C( data%dims%c_u_start: data%dims%c_u_end ) -     &
                        prob%C_u( data%dims%c_u_start : data%dims%c_u_end )    &
                        > - tol )
  
        remap_fixed = tiny_x > 0 .OR. tiny_c > 0
        IF ( remap_fixed ) THEN

!  Some of the current variables/constraints will be fixed

          IF ( control%error > 0 .AND. control%print_level >= 1 )              &
            WRITE( control%out, "( /, A, ' -> ', I0, ' further variable', A,   &
           & ' and ', I0, ' further constraint', A, ' will be fixed' )" )      &
               prefix, tiny_x, TRIM( STRING_pleural( tiny_x ) ),               &
               tiny_c, TRIM( STRING_pleural( tiny_c ) )

!  Allocate arrays to record the bounds which will be altered

          IF ( tiny_x > 0 ) THEN
            array_name = 'qpb: data%X_fixed'
            CALL SPACE_resize_array( tiny_x, data%X_fixed,                     &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpb: data%Index_X_fixed'
            CALL SPACE_resize_array( tiny_x, data%Index_X_fixed,               &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF
  
          IF ( tiny_c > 0 ) THEN
            array_name = 'qpb: data%C_fixed'
            CALL SPACE_resize_array( tiny_c, data%C_fixed,                     &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
  
            array_name = 'qpb: data%Index_C_fixed'
            CALL SPACE_resize_array( tiny_c, data%Index_C_fixed,               &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  Fix the problem bounds as required

          IF ( tiny_x > 0 ) THEN
            tiny_x = 0
    
            DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i )
              IF ( prob%X( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) = zero
              END IF
            END DO
    
            DO i = data%dims%x_l_start, data%dims%x_u_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) =  prob%X_l( i )
              END IF
            END DO
    
            DO i = data%dims%x_u_start, data%dims%x_l_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) =  prob%X_l( i )
              ELSE IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO
  
            DO i = data%dims%x_l_end + 1, data%dims%x_u_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO
    
            DO i = data%dims%x_u_end + 1, prob%n
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', - prob%X( i )
              IF ( prob%X( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) = zero
              END IF
            END DO
          END IF

!  Do the same for the constraint bounds

          IF ( tiny_c > 0 ) THEN
            tiny_c = 0
    
            DO i = data%dims%c_l_start, data%dims%c_u_start - 1
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_u( i ) =  prob%C_l( i )
              END IF
            END DO
    
            DO i = data%dims%c_u_start, data%dims%c_l_end
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_u( i ) =  prob%C_l( i )
              ELSE IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO
    
            DO i = data%dims%c_l_end + 1, data%dims%c_u_end
              IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO
          END IF
  
          CALL QPP_initialize( data%QPP_map_fixed, data%QPP_control )
          data%QPP_control%infinity = control%infinity
          data%QPP_control%treat_zero_bounds_as_general =                      &
            control%treat_zero_bounds_as_general

!  Store the problem dimensions

          data%dims_save_fixed = data%dims
          data%a_ne = prob%A%ne 
          data%h_ne = prob%H%ne 
  
          IF ( printi ) WRITE( control%out,                                    &
                 "( /, A, ' problem dimensions before preprocessing: ', /, A,  &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" ) &
                 prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne
  
!  Perform the preprocessing

          CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
          CALL QPP_reorder( data%QPP_map_fixed, data%QPP_control,              &
                            data%QPP_inform, data%dims, prob,                  &
                            .FALSE., .FALSE., .FALSE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
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
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')")&
               prefix, data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) prefix, inform%status 
            IF ( control%out > 0 .AND. control%print_level > 0 .AND.           &
                 inform%status == GALAHAD_error_upper_entry )                  &
              WRITE( control%error, 2020 ) prefix
            CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map, data%QPP_control,                &
                                data%QPP_inform )
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

          IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            data%h_ne = prob%n
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          ELSE
            data%h_ne = prob%H%ne 
          END IF

          IF ( printi ) WRITE( control%out,                                    &
                 "( /, A, ' problem dimensions after preprocessing: ', /, A,   &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" ) &
                 prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  ====================================================================
!  Check to see if the equality constraints remain linearly independent
!  ====================================================================

          IF ( prob%m > 0 .AND. control%remove_dependencies ) THEN
            IF ( control%out > 0 .AND. control%print_level >= 1 )              &
              WRITE( control%out, "( /, A, 1X, I0,                             &
             & ' equalit', A, ' from ', I0, ' constraint', A )" )              &
                  prefix, data%dims%c_equality,                                &
                  TRIM( STRING_ies( data%dims%c_equality ) ),                  &
                  prob%m, TRIM( STRING_pleural( prob%m ) )

!  Find any dependent rows

            nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1 
            CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record ) 
            CALL FDC_find_dependent( prob%n, data%dims%c_equality,             &
                                     prob%A%val( : nzc ),                      &
                                     prob%A%col( : nzc ),                      &
                                     prob%A%ptr( : data%dims%c_equality + 1 ), &
                                     prob%C_l,                                 &
                                     n_more_depen, data%Index_C_more_freed,    &
                                     data%FDC_data, data%FDC_control,          &
                                     inform%FDC_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
            inform%time%find_dependent =                                       &
              inform%time%find_dependent + time_now - time_record
            inform%time%clock_find_dependent =                                 &
              inform%time%clock_find_dependent + clock_now - clock_record

!  Record output parameters

            inform%status = inform%FDC_inform%status
            inform%alloc_status = inform%FDC_inform%alloc_status
            inform%non_negligible_pivot =                                      &
             MIN( inform%FDC_inform%non_negligible_pivot,                      &
                  inform%non_negligible_pivot )
            inform%factorization_status =                                      &
              inform%FDC_inform%factorization_status
            inform%factorization_integer =                                     &
              inform%FDC_inform%factorization_integer
            inform%factorization_real =                                        &
              inform%FDC_inform%factorization_real
            inform%bad_alloc = inform%FDC_inform%bad_alloc
            inform%nfacts = 1

            IF ( ( control%cpu_time_limit >= zero .AND.                        &
                   time_now - time_start > control%cpu_time_limit ) .OR.       &
                 ( control%clock_time_limit >= zero .AND.                      &
                   clock_now - clock_start > control%clock_time_limit ) ) THEN
              inform%status = GALAHAD_error_cpu_limit
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) prefix, inform%status 
              GO TO 800 
            END IF 

            IF ( printi .AND. inform%non_negligible_pivot < thousand *         &
                 data%FDC_control%SLS_control%absolute_pivot_tolerance )       &
                WRITE( control%out, "(                                         &
           &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,             &
           &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',&
           &  /, A, ' ***  perhaps increase LSQP_control%',                    &
           &     'FDC_control%SLS_control%absolute_pivot_tolerance from',      &
           &  ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" ) &
               prefix, prefix, inform%non_negligible_pivot, prefix,            &
               data%FDC_control%SLS_control%absolute_pivot_tolerance, prefix

            IF ( inform%status /= GALAHAD_ok ) THEN

!  Allocate arrays to hold the matrix vector product

              array_name = 'qpb: data%HX'
              CALL SPACE_resize_array( prob%n, data%HX,                        &
                     inform%status, inform%alloc_status,                       &
                     array_name = array_name,                                  &
                     deallocate_error_fatal = control%deallocate_error_fatal,  &
                     exact_size = control%space_critical,                      &
                     bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= GALAHAD_ok ) GO TO 900
        
!  On error exit, compute the current objective function value

              data%HX( : prob%n ) = zero
              CALL QPB_HX( data%dims, prob%n, data%HX( : prob%n ),             &
                            prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,          &
                            prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
              inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),             &
                                               data%HX( : prob%n ) )           &
                           + DOT_PRODUCT( prob%X( : prob%n ),                  &
                                          prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

              IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
                WRITE( control%out, "( ' ' )" )
                IF ( inform%status /= GALAHAD_ok ) WRITE( control%error, 2040 )&
                 prefix, inform%status, 'LSQP_dependent'
              END IF
              GO TO 750
            END IF
    
            IF ( control%out > 0 .AND. control%print_level >= 2                &
                 .AND. n_more_depen > 0 )                                      &
              WRITE( control%out, "(/, A, ' The following ', I0, ' constraint',&
             &  A, ' appear', A, ' to be dependent', /, ( 4X, 8I8 ) )" )       &
                  prefix, n_more_depen, TRIM( STRING_are( n_more_depen ) ),    &
                  TRIM( STRING_pleural( n_more_depen ) ),                      &
                  data%Index_C_more_freed
            remap_more_freed = n_more_depen > 0
          ELSE
            remap_more_freed = .FALSE.
          END IF
    
          IF ( remap_more_freed ) THEN

!  Some of the current constraints will be removed by freeing them

            IF ( control%error > 0 .AND. control%print_level >= 1 )            &
              WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ',  &
         &  A, ' dependent and will be temporarily removed' )" ) prefix,       &
              n_more_depen, TRIM( STRING_pleural( n_more_depen ) ),            &
                TRIM( STRING_are( n_more_depen ) )

!  Allocate arrays to indicate which constraints have been freed

            array_name = 'qpb: data%C_more_freed'
            CALL SPACE_resize_array( n_more_depen, data%C_more_freed,          &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Free the constraint bounds as required

            DO i = 1, n_more_depen
              j = data%Index_C_more_freed( i )
              data%C_more_freed( i ) = prob%C_l( j )
              prob%C_l( j ) = - control%infinity
              prob%C_u( j ) = control%infinity
              prob%Y( j ) = zero
            END DO
    
            CALL QPP_initialize( data%QPP_map_more_freed, data%QPP_control )
            data%QPP_control%infinity = control%infinity
            data%QPP_control%treat_zero_bounds_as_general =                    &
              control%treat_zero_bounds_as_general

!  Store the problem dimensions

            data%dims_save_more_freed = data%dims
            data%a_ne = prob%A%ne 
            data%h_ne = prob%H%ne 
    
            IF ( printi ) WRITE( control%out,                                  &
              "( /, A, ' problem dimensions before removal of dependecies: ',  &
             &   /, A, ' n = ', I0, ' m = ', I0, ' a_ne = ', I0, ' h_ne = ',   &
             &      I0 )" ) prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne
    
!  Perform the preprocessing

            CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
            CALL QPP_reorder( data%QPP_map_more_freed, data%QPP_control,       &
                              data%QPP_inform, data%dims, prob,                &
                              .FALSE., .FALSE., .FALSE. )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
            inform%time%preprocess =                                           &
              inform%time%preprocess + time_now - time_record
            inform%time%clock_preprocess =                                     &
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
              IF ( control%out > 0 .AND. control%print_level >= 5 )            &
                WRITE( control%out, "( A, ' status ', I0,                      &
               & ' after QPP_reorder')") prefix, data%QPP_inform%status
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) prefix, inform%status 
              IF ( control%out > 0 .AND. control%print_level > 0 .AND.         &
                   inform%status == GALAHAD_error_upper_entry )                &
                 WRITE( control%error, 2020 ) prefix
              CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,   &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,        &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,        &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map, data%QPP_control,              &
                                  data%QPP_inform )
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

            IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
              data%h_ne = prob%n
            ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
              data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
            ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
              data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
            ELSE
              data%h_ne = prob%H%ne 
            END IF
    
            IF ( printi ) WRITE( control%out, "( /, A,                         &
             & ' problem dimensions after removal of dependencies: ', / A,     &
             & ' n = ', I0, ' m = ', I0, ' a_ne = ', I0, ' h_ne = ', I0 )" )   &
                   prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne
  
          END IF

!  Experiment!!

!         GO TO 10
        ELSE
          remap_more_freed = .FALSE.
        END IF
      END IF

!  Allocate additional real workspace

      array_name = 'qpb: data%DZ_l'
      CALL SPACE_resize_array( data%dims%x_l_start, data%dims%x_l_end,         &
             data%DZ_l,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DZ_u'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_u_end,         &
             data%DZ_u,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%GRAD'
      CALL SPACE_resize_array( prob%n, data%GRAD,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%X_trial'
      CALL SPACE_resize_array( prob%n, data%X_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%GRAD_X_phi'
      CALL SPACE_resize_array( prob%n, data%GRAD_X_phi,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%GRAD_C_phi'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%GRAD_C_phi,                                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%S'
      CALL SPACE_resize_array( data%dims%c_e, data%S,                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_trial'
      CALL SPACE_resize_array( prob%m, data%Y_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( stat_required ) THEN
        array_name = 'qpb: data%H_s'
        CALL SPACE_resize_array( prob%n, data%H_s,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qpb: data%A_s'
        CALL SPACE_resize_array( prob%m, data%A_s,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qpb: data%Y_last'
        CALL SPACE_resize_array( prob%m, data%Y_last,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'qpb: data%Z_last'
        CALL SPACE_resize_array( prob%n, data%Z_last,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( control%extrapolate > 0 ) THEN
        data%hist = control%path_history
        data%deriv = control%path_derivatives
        IF ( control%fit_order > 0 ) THEN
          data%order = control%fit_order
        ELSE
          data%order = data%hist * ( data%deriv + 1 ) - 1
        END IF
      ELSE
        data%hist = 0
        data%deriv = 0
        data%order = 0
      END IF
      data%len_hist = 0

      array_name = 'qpb: data%BINOMIAL'
      CALL SPACE_resize_array( 0, data%deriv - 1, data%deriv, data%BINOMIAL,   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%fit_mu'
      CALL SPACE_resize_array( data%order + 1, data%fit_mu,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%fit_f'
      CALL SPACE_resize_array( data%order + 1, data%fit_f,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%X_coef'
      CALL SPACE_resize_array( 0, data%order, prob%n, data%X_coef,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%C_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_l_start,             &
             data%dims%c_u_end, data%C_coef,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_coef'
      CALL SPACE_resize_array( 0, data%order, prob%m, data%Y_coef,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_l_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_l_start,             &
             data%dims%c_l_end, data%Y_l_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_u_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_u_start,             &
             data%dims%c_u_end, data%Y_u_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_l_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%x_free + 1,            &
             data%dims%x_l_end, data%Z_l_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_u_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%x_u_start,             &
             prob%n, data%Z_u_coef,                                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%list_hist'
      CALL SPACE_resize_array( data%hist, data%list_hist,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%mu_hist'
      CALL SPACE_resize_array( data%hist, data%mu_hist,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%X_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, prob%n, data%X_hist,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%C_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_l_start,  &
             data%dims%c_u_end, data%C_hist,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, prob%m, data%Y_hist,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_l_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_l_start,  &
             data%dims%c_l_end, data%Y_l_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_u_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_u_start,  &
             data%dims%c_u_end, data%Y_u_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_l_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%x_free + 1, &
             data%dims%x_l_end, data%Z_l_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_u_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%x_u_start,  &
             prob%n, data%Z_u_hist,                                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      n_sbls =  prob%n + data%dims%nc

!  H will be in coordinate form

      CALL SMT_put( data%H_sbls%type, 'COORDINATE', inform%alloc_status )
      data%H_sbls%n = n_sbls
      data%H_sbls%m = n_sbls
      data%H_sbls%ne = data%h_ne + n_sbls

!  allocate space for H

      array_name = 'lsqp: data%H_sbls%row'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%col'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%val'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of the barrier terms in coordinate form ...

      DO i = 1, n_sbls
        data%H_sbls%row( i ) = i ; data%H_sbls%col( i ) = i
      END DO

!  ... and add the components of H

      DO i = 1, prob%n
        data%H_sbls%row( n_sbls + prob%H%ptr( i ) :                            &
                         n_sbls + prob%H%ptr( i + 1 ) - 1 ) = i
      END DO
      data%H_sbls%col( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%col( : data%h_ne ) 
      data%H_sbls%val( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%val( : data%h_ne )

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

!  overlaps: DY_l => DIST_C_l_trial 
!            DY_u => DIST_C_u_trial
!            DELTA => VECTOR
!            RHS( : c_e ) => R 
!            DZ_l( x_l_start : x_l_end ) => DIST_X_l_trial
!            DZ_u( x_u_start : x_u_end ) => DIST_X_u_trial

      data%SBLS_control = control%SBLS_control
      data%SBLS_control%get_norm_residual = .TRUE.
      data%GLTR_control = control%GLTR_control

      IF ( stat_required ) THEN
        CALL QPB_solve_main( data%dims, prob%n, prob%m,                        &
                             prob%H%val, prob%H%col, prob%H%ptr,               &
                             prob%G, prob%f, prob%A%val, prob%A%col,           &
                             prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,         &
                             prob%X_u, prob%C, prob%X, prob%Y, prob%Z,         &
                             data%X_trial, data%Y_trial,                       &
                             data%HX, data%GRAD_L, data%DIST_X_l,              &
                             data%DIST_X_u, data%Z_l, data%Z_u,                &
                             data%BARRIER_X, data%Y_l, data%DY_l,              &
                             data%DIST_C_l, data%Y_u, data%DY_u,               &
                             data%DIST_C_u, data%C, data%BARRIER_C,            &
                             data%SCALE_C, data%DELTA,                         &
                             data%H_sbls, data%A_sbls, data%C_sbls,            &
                             data%RHS( : data%dims%v_e ),                      &
                             data%GRAD, data%GRAD_X_phi, data%GRAD_C_phi,      &
!                            data%DZ_l( data%dims%x_l_start :                  &
!                                       data%dims%x_l_end ),                   &
!                            data%DZ_u( data%dims%x_u_start :                  &
!                                       data%dims%x_u_end ), data%S,           &
                             data%DZ_l, data%DZ_u, data%S,                     &
                             data%hist, data%deriv, data%order, data%len_hist, &
                             data%BINOMIAL, data%fit_mu, data%fit_f,           &
                             data%X_coef, data%C_coef, data%Y_coef,            &
                             data%Y_l_coef, data%Y_u_coef,                     &
                             data%Z_l_coef, data%Z_u_coef,                     &
                             data%list_hist, data%mu_hist,                     &
                             data%X_hist, data%C_hist, data%Y_hist,            &
                             data%Y_l_hist, data%Y_u_hist,                     &
                             data%Z_l_hist, data%Z_u_hist,                     &
                             data%SBLS_data, data%GLTR_data, data%FIT_data,    &
                             prefix, control, inform, data%SBLS_control,       &
                             data%GLTR_control,                                &
                             C_last = data%A_s, X_last = data%H_s,             &
                             Y_last = data%Y_last, Z_last = data%Z_last,       &
                             C_stat = C_stat, B_Stat = B_Stat )
      ELSE
        CALL QPB_solve_main( data%dims, prob%n, prob%m,                        &
                             prob%H%val, prob%H%col, prob%H%ptr,               &
                             prob%G, prob%f, prob%A%val, prob%A%col,           &
                             prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,         &
                             prob%X_u, prob%C, prob%X, prob%Y, prob%Z,         &
                             data%X_trial, data%Y_trial,                       &
                             data%HX, data%GRAD_L, data%DIST_X_l,              &
                             data%DIST_X_u, data%Z_l, data%Z_u,                &
                             data%BARRIER_X, data%Y_l, data%DY_l,              &
                             data%DIST_C_l, data%Y_u, data%DY_u,               &
                             data%DIST_C_u, data%C, data%BARRIER_C,            &
                             data%SCALE_C, data%DELTA,                         &
                             data%H_sbls, data%A_sbls, data%C_sbls,            &
                             data%RHS( : data%dims%v_e ),                      &
                             data%GRAD, data%GRAD_X_phi, data%GRAD_C_phi,      &
!                            data%DZ_l( data%dims%x_l_start :                  &
!                                       data%dims%x_l_end ),                   &
!                            data%DZ_u( data%dims%x_u_start :                  &
!                                       data%dims%x_u_end ), data%S,           &
                             data%DZ_l, data%DZ_u, data%S,                     &
                             data%hist, data%deriv, data%order, data%len_hist, &
                             data%BINOMIAL, data%fit_mu, data%fit_f,           &
                             data%X_coef, data%C_coef, data%Y_coef,            &
                             data%Y_l_coef, data%Y_u_coef,                     &
                             data%Z_l_coef, data%Z_u_coef,                     &
                             data%list_hist, data%mu_hist,                     &
                             data%X_hist, data%C_hist, data%Y_hist,            &
                             data%Y_l_hist, data%Y_u_hist,                     &
                             data%Z_l_hist, data%Z_u_hist,                     &
                             data%SBLS_data, data%GLTR_data, data%FIT_data,    &
                             prefix, control, inform, data%SBLS_control,       &
                             data%GLTR_control )
      END IF
      inform%time%analyse = inform%time%analyse +                              &
        inform%LSQP_inform%time%analyse +                                      &
        inform%FDC_inform%time%analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%LSQP_inform%time%clock_analyse +                                &
        inform%FDC_inform%time%clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%LSQP_inform%time%factorize +                                    &
        inform%FDC_inform%time%factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%LSQP_inform%time%clock_factorize +                              &
        inform%FDC_inform%time%clock_factorize

!  If some of the constraints were freed having first been fixed during 
!  the computation, refix them now

  700 CONTINUE 
      IF ( remap_more_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_restore( data%QPP_map_more_freed, data%QPP_inform,            &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,         &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_more_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_more_depen
          j = data%Index_c_more_freed( i )
          prob%C_l( j ) = data%C_more_freed( i )
          prob%C_u( j ) = data%C_more_freed( i )
        END DO
      END IF

!  If some of the variables/constraints were fixed during the computation,
!  free them now

      IF ( remap_fixed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_restore( data%QPP_map_fixed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_fixed

!  Release the temporarily fixed problem bounds

        DO i = 1, tiny_x
          j = data%Index_X_fixed( i )
          IF ( j > 0 ) THEN
            prob%X_l( j ) = data%X_fixed( i )
            IF ( PRESENT( B_stat ) ) THEN
              IF ( B_stat( j ) < 0 ) THEN
                B_stat( j ) = - B_stat( j )
!               prob%Z( j ) = -  prob%Z( j )
              END IF  
            END IF
          ELSE
            prob%X_u( - j ) = data%X_fixed( i )
          END IF
        END DO
       
!  Do the same for the constraint bounds

        DO i = 1, tiny_c
          j = data%Index_C_fixed( i )
          IF ( j > 0 ) THEN
            prob%C_l( j ) = data%C_fixed( i )
            IF ( PRESENT( C_stat ) ) THEN
              IF ( C_stat( j ) < 0 ) THEN
                C_stat( j ) = - C_stat( j )
!               prob%Y( j ) = -  prob%Y( j )
              END IF  
            END IF  
          ELSE
            prob%C_u( - j ) = data%C_fixed( i )
          END IF
        END DO

      END IF

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_C_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO

      END IF
      data%tried_to_remove_deps = .FALSE.

!  Retore the problem to its original form

  750 CONTINUE 
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  Full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )

!  Restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  Solution recovery

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                             get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,   &
                             get_c = .TRUE. )
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
     &     "( /, A, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',            &
     &        '-=-=-=-==-=-=-=-=-=-=-=-=-=-=',                                 &
     &        /, A, ' =', 24X,  'QPB timing statistics', 25X, '=',             &
     &        /, A, ' =               total        preprocess ',               &
     &           '          phase 1              =',                           &
     &        /, A, ' =', 8X, 0P, F12.2, 6X, F12.2, 6X, F12.2, 14X, '='        &
     &        /, A, ' =      analyse   factorize     solve',                   &
     &           '    analyse  factorize      solve  =',                       &
     &        /, A, ' =', 2X, 6F11.2, 2X, '=',                                 &
     &        /, A, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',            &
     &           '-=-=-=-=-==-=-=-=-=-=-=-=-=-=' )" )                          &
        prefix, prefix, prefix, prefix,                                        &
        inform%time%total, inform%time%preprocess, inform%time%phase1_total,   &
        prefix, prefix,                                                        &
        inform%time%analyse, inform%time%factorize, inform%time%solve,         &
        inform%time%phase1_analyse, inform%time%phase1_factorize,              &
        inform%time%phase1_solve, prefix

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QPB_solve ' )" ) prefix
      RETURN  

!  Allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( printi ) WRITE( control%out, "( A, ' ** Message from -QPB_solve-',  &
     &  /, A, ' Allocation error, for ', A, /, A, ' status = ', I0 )" )        &
        prefix, prefix, inform%bad_alloc, prefix, inform%alloc_status

!  Compute total time

  920 CONTINUE 
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start 
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start 

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QPB_solve ' )" ) prefix
      RETURN  

!  Non-executable statements

 2010 FORMAT( ' ', /, A, '   **  Error return ', I0, ' from QPB ' ) 
 2020 FORMAT( /, A, '  Warning: an entry from strict upper triangle of H given')
 2040 FORMAT( A, '   **  Error return ', I0, ' from ', A ) 

!  End of QPB_solve

      END SUBROUTINE QPB_solve

!-*-*-*-*-   Q P B _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QPB_solve_main( dims, n, m,                                   &
                                 H_val, H_col, H_ptr, G, f, A_val, A_col,      &
                                 A_ptr, C_l, C_u, X_l, X_u, C_RES, X, Y, Z,    &
                                 X_trial, Y_trial, HX, GRAD_L,                 &
                                 DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X,      &
                                 Y_l, DIST_C_l_trial, DIST_C_l, Y_u,           &
                                 DIST_C_u_trial, DIST_C_u, C, BARRIER_C,       &
                                 SCALE_C, VECTOR, H_sbls, A_sbls, C_sbls, R,   &
                                 GRAD, GRAD_X_phi, GRAD_C_phi,                 &
                                 DIST_X_l_trial, DIST_X_u_trial, S,            &
                                 hist, deriv, order, len_hist, BINOMIAL,       &
                                 fit_mu, fit_f, X_coef, C_coef, Y_coef,        &
                                 Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,       &
                                 list_hist, mu_hist,                           &
                                 X_hist, C_hist, Y_hist,                       &
                                 Y_l_hist, Y_u_hist, Z_l_hist, Z_u_hist,       &
                                 SBLS_data, GLTR_data, FIT_data, prefix,       &
                                 control, inform, SBLS_control, GLTR_control,  &
                                 C_last, X_last, Y_last, Z_last,               &
                                 C_stat, B_Stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix, 
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
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
!    Within each category, the variables are further ordered so that those 
!    with non-zero diagonal Hessian entries occur before the remainder
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
!  dims is a structure of type QPB_data_type, whose components hold SCALAR
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
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. 
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
!  G is a REAL array of length n, which must be set by
!   the user to the value of the gradient, g, of the linear term of the
!   quadratic objective function. The i-th component of G, i = 1, ....,
!   n, should contain the value of g_i.  
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
!   the i-th component of C_RES will contain (A*x)_i, for i = 1, .... , m. 
!
!  X is a REAL array of length n, which must be set by
!   the user on entry to QPB_solve to give an initial estimate of the 
!   optimization parameters, x. The i-th component of X should contain 
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need 
!   not satisfy the simple bound constraints and may be perturbed by 
!   QPB_solve prior to the start of the minimization.  Any estimate which is 
!   closer to one of its bounds than control%prfeas may be reset to try to
!   ensure that it is at least control%prfeas from its bounds. On exit from 
!   QPB_solve, X will contain the best estimate of the optimization 
!   parameters found
!  
!  Y is a REAL array of length m, which must be set by the user
!   on entry to QPB_solve to give an initial estimates of the
!   optimal Lagrange multipiers, y. The i-th component of Y 
!   should contain the initial estimate of y_i, for i = 1, .... , m.  
!   Any estimate which is smaller than control%dufeas may be 
!   reset to control%dufeas. The dual variable for any variable with both
!   On exit from QPB_solve, Y will contain the best estimate of
!   the Lagrange multipliers found
!  
!  Z, is a REAL array of length n, which must be set by
!   on entry to QPB_solve to hold the values of the the dual variables 
!   associated with the simple bound constraints. 
!   Any estimate which is smaller than control%dufeas may be 
!   reset to control%dufeas. The dual variable for any variable with both
!   infinite lower and upper bounds need not be set. On exit from
!   QPB_solve, Z will contain the best estimates obtained
!  
!  control and inform are exactly as for QPB_solve
!
!  The remaining arguments are used as internal workspace, and need not be 
!  set on entry
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, hist, deriv, order
      INTEGER, INTENT( INOUT ) :: len_hist
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: B_stat
      REAL ( KIND = wp ), INTENT( IN ):: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, X_l, X_u, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( n ) :: X_trial, GRAD, GRAD_X_phi
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( m ) :: C_RES, Y_trial
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: Y_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: Z_last
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%v_e ) :: HX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: S
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: GRAD_L
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l, DIST_X_l_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u, DIST_X_u_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) ::  Z_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DIST_C_l,      &
                                                           DIST_C_l_trial 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DIST_C_u,      &
                                                           DIST_C_u_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, BARRIER_C,       &
                                                           SCALE_C, GRAD_C_phi
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%v_e ) :: VECTOR, R
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : deriv - 1 , deriv ) :: BINOMIAL
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( order + 1 ) :: fit_mu, fit_f
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, n ) :: X_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, dims%c_l_start : dims%c_u_end ) :: C_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, m ) :: Y_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, dims%c_l_start : dims%c_l_end ) ::  Y_l_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, dims%c_u_start : dims%c_u_end ) ::  Y_u_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, dims%x_free + 1 : dims%x_l_end ) :: Z_l_coef
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( 0 : order, dims%x_u_start : n ) :: Z_u_coef
      INTEGER, INTENT( OUT ), DIMENSION( hist ) :: list_hist
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( hist ) :: mu_hist
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, n ) :: X_hist                               
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, dims%c_l_start : dims%c_u_end ) :: C_hist 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, m ) :: Y_hist                               
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, dims%c_l_start : dims%c_l_end ) :: Y_l_hist 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, dims%c_u_start : dims%c_u_end ) :: Y_u_hist 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, dims%x_free + 1 : dims%x_l_end ) :: Z_l_hist
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( hist, 0 : deriv, dims%x_u_start : n ) :: Z_u_hist 

      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls, A_sbls
      TYPE ( SMT_type ), INTENT( IN ) :: C_sbls

      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( QPB_control_type ), INTENT( IN ) :: control
      TYPE ( QPB_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: SBLS_control
      TYPE ( GLTR_control_type ), INTENT( INOUT ) :: GLTR_control

      TYPE ( SBLS_data_type ), INTENT( INOUT )  :: SBLS_data
      TYPE ( GLTR_data_type ), INTENT( INOUT )  :: GLTR_data
      TYPE ( FIT_data_type ), INTENT( INOUT ) :: FIT_data

!  Parameters

      INTEGER, PARAMETER :: history = 5
      REAL, PARAMETER :: max_ratio = 2.0
      REAL ( KIND = wp ), PARAMETER :: eta_1 = tenm2
      REAL ( KIND = wp ), PARAMETER :: eta_2 = point9
      REAL ( KIND = wp ), PARAMETER :: sigma = point1
!     REAL ( KIND = wp ), PARAMETER :: sigma = 0.01_wp
!     REAL ( KIND = wp ), PARAMETER :: sigma = 0.8_wp
      REAL ( KIND = wp ), PARAMETER :: theta_min = ten ** 20
      REAL ( KIND = wp ), PARAMETER :: nu_1 = point01
      REAL ( KIND = wp ), PARAMETER :: eta = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: hmin = one
      REAL ( KIND = wp ), PARAMETER :: radius_max = ten ** 20

!  Local variables

      INTEGER :: a_ne, h_ne, i, j, kd, l, nbacts
      INTEGER :: start_print, stop_print, cg_maxit, seq, print_level
      INTEGER :: nbnds, out, error, cg_iter, fact_hist
      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, type
      INTEGER :: current
      INTEGER :: n_degree, n_degree_p, inner_iteration
      REAL :: time_record, time_start, time_now
      REAL ( KIND = wp ) :: clock_record, clock_start, clock_now
      REAL :: time_iter, time_last, time_mean, time_ratio
      REAL :: time_kkt, time_itsol, time_hist( 0 : history - 1 ) = 0.0
      REAL ( KIND = wp ) :: mu, amax, teneps, ared, prered, old_mu, res_fail
      REAL ( KIND = wp ) :: alpha, hmax, delta, zeta, c_feasmin
      REAL ( KIND = wp ) :: model, obj_trial, radius, old_radius
      REAL ( KIND = wp ) :: dufeas, ratio, norm_c, norm_d, norm_d_alt
      REAL ( KIND = wp ) :: obj, phi, phi_trial, theta_c, theta_d, rhs
      REAL ( KIND = wp ) :: phi_model, phi_slope, obj_slope, obj_curv, xx, zz 
      REAL ( KIND = wp ) :: p_min, p_max, d_min, d_max, relative_pivot_tol
      REAL ( KIND = wp ) :: initial_radius, small_x
!     REAL ( KIND = wp ) :: comp_min, comp_max
      REAL ( KIND = wp ) :: x_p, z_l_p, z_u_p, c_p, y_l_p, y_u_p
      REAL ( KIND = wp ) :: mu_extrapolate, norm_cd, old_norm_cd
      REAL ( KIND = wp ) :: x_norm, c_norm, y_norm, z_norm, step_max

      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe 
      LOGICAL :: one_fact, primal_hessian, got_time_kkt
      LOGICAL :: first_iteration, got_ratio, start_major, new_prec
      LOGICAL :: auto, full_iteration, scaled_c, set_z, get_factors
      LOGICAL :: new_fact, refact, primal, stat_required, get_stat, revert
      LOGICAL :: successful_iteration, allow_extrapolate
      LOGICAL :: use_scale_c = .FALSE.

      CHARACTER ( LEN = 1 ) :: re, mo, bdry
      CHARACTER ( LEN = 7 ) :: series

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering QPB_solve_main ' )" ) prefix

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )
      time_last = time_start

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

      out = control%out ; error = control%error 
      set_printe = error > 0 .AND. control%print_level >= 1

!  Basic single line of output per iteration

      set_printi = out > 0 .AND. control%print_level >= 1 

!  As per printi, but with additional timings for various operations

      set_printt = out > 0 .AND. control%print_level >= 2 

!  As per printm, but with checking of residuals, etc

      set_printm = out > 0 .AND. control%print_level >= 3 

!  As per printm but also with an indication of where in the code we are

      set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

      set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printm = set_printm ; printw = set_printw ; printd = set_printd
        print_level = control%print_level
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
        print_level = 0
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
        inform%obj = f
        GO TO 810
      END IF 

!  Record array sizes

      a_ne = A_ptr( m + 1 ) - 1
      h_ne = H_ptr( n + 1 ) - 1

!  Set control parameters

      auto = SBLS_control%preconditioner == 0

      cg_maxit = control%cg_maxit
      IF ( cg_maxit < 0 ) cg_maxit = dims%c_b + 2
      seq = 0

      dufeas = control%dufeas 
      relative_pivot_tol = SBLS_control%SLS_control%relative_pivot_tolerance
      initial_radius = control%initial_radius
      scaled_c = .TRUE.

      primal = control%primal ; primal_hessian = primal
      start_major = .TRUE.
      small_x = epsmch
      teneps = 10.0 * epsmch
      c_feasmin = SQRT( epsmch )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )
      IF ( stat_required ) THEN
        B_stat  = 0
        C_stat( : dims%c_equality ) = - 1
        C_stat( dims%c_equality + 1 : ) = 0
      END IF
      get_stat = .FALSE.

!  Compute the binomial coefficients b_i^k = b_i^{k-1} + b_{i-1}^{k-1} 

      IF ( control%extrapolate > 0 ) THEN
        BINOMIAL( 0, 1 ) = one
        DO j = 2, deriv
          BINOMIAL( j - 1, j - 1 ) = one
          BINOMIAL( 0, j ) = one
          DO i = 1, j - 1
            BINOMIAL( i, j ) = BINOMIAL( i, j - 1 ) + BINOMIAL( i - 1, j - 1 )
          END DO
        END DO
      END IF

!  Initialize counts

      nbacts = 0

!  If required, write out the problem

      IF ( printd ) THEN
        WRITE( out, 2180 ) prefix, ' g ', ( G( i ), i = 1, n )
        WRITE( out, 2190 ) prefix, ' A ', ( ( i, A_col( l ), A_val( l ),       &
                           l = A_ptr( i ), A_ptr( i + 1 ) - 1 ), i = 1, m )
        WRITE( out, 2190 ) prefix, ' H ', ( ( i, H_col( l ), H_val( l ),       &
                           l = H_ptr( i ), H_ptr( i + 1 ) - 1 ), i = 1, n )
      END IF 

!  Record the initial point, move the starting point away from any bounds, 
!  and move that for dual variables away from zero

      nbnds = 0
      set_z = .FALSE.

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
        IF ( X( i ) <= small_x ) THEN 
!         write(6,"('i,X, small',I5, 2ES12.4)") i, X(i), small_x
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        nbnds = nbnds + 1
        Z_l( i ) = MAX( Z( i ), dufeas )
        IF ( printd ) WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,   &
       &  '      -     ' )" ) prefix, i, X( i ), zero, Z_l( i )
      END DO

!  The variable has just a lower bound

      DO i = dims%x_l_start, dims%x_u_start - 1
        IF ( X( i ) - X_l( i ) <= small_x ) THEN 
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        nbnds = nbnds + 1
        Z_l( i ) = MAX( Z( i ), dufeas )
        DIST_X_l( i ) = X( i ) - X_l( i )
        IF ( printd ) WRITE( out, "( A, I6, 2ES12.4, '      -     ', ES12.4,   &
       &  '      -     ' )" ) prefix, i, X( i ), X_l( i ), Z_l( i )
      END DO

!  The variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN 
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        nbnds = nbnds + 2
        IF ( X( i ) - X_l( i ) <= small_x .OR.                                 &
             X_u( i ) - X( i ) <= small_x ) THEN 
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        Z_l( i ) = MAX(   ABS( Z( i ) ),   dufeas )  
        Z_u( i ) = MIN( - ABS( Z( i ) ), - dufeas )
        DIST_X_l( i ) = X( i ) - X_l( i ) ; DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( printd ) WRITE( out, "( A, I6, 5ES12.4 )" )                       &
             prefix, i, X( i ), X_l( i ), X_u( i ), Z_l( i ), Z_u( i )
      END DO

!  The variable has just an upper bound

      DO i = dims%x_l_end + 1, dims%x_u_end
        nbnds = nbnds + 1
        IF ( X_u( i ) - X( i ) <= small_x ) THEN 
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        Z_u( i ) = MIN( Z( i ), - dufeas ) 
        DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,    &
       &  '      -     ', ES12.4 )" ) prefix, i, X( i ), X_u( i ), Z_u( i )
      END DO

!  The variable is a non-positivity

      DO i = dims%x_u_end + 1, n
        nbnds = nbnds + 1
        IF ( - X( i ) <= small_x ) THEN 
          inform%status = GALAHAD_error_primal_infeasible
          GO TO 700
        END IF
        Z_u( i ) = MIN( Z( i ), - dufeas ) 
        IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,    &
       &  '      -     ',  ES12.4 )" ) prefix, i, X( i ), zero, Z_u( i )
      END DO

      DIST_X_l_trial = DIST_X_l ; DIST_X_u_trial = DIST_X_u
      set_z = .TRUE.

!  Compute the value of the constraint, and their residuals

      IF ( m > 0 ) THEN
        R( : m ) = zero
        CALL QPB_AX( m, R( : m ), m, a_ne, A_val, A_col, A_ptr, n, X, '+ ' )

        IF ( printd ) THEN
          WRITE( out, "( /, A, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X,     &
         &     'y_l', 9X, 'y_u' )") prefix
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( A, I6, 3ES12.4 )" )                                 &
             prefix, i, R( i ), C_l( i ), C_u( i )
          END DO
        END IF

!  The constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          nbnds = nbnds + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MAX( R( i ) / SCALE_C( i ),                                 &
                        C_l( i ) + c_feasmin * MAX( one, ABS( C_l( i ) ) ) )
          DIST_C_l( i ) = C( i ) - C_l( i )
          Y_l( i ) = MAX( Y( i ), dufeas )
          IF ( printd ) WRITE( out,  "( A, I6, 2ES12.4, '      -     ',        &
         &   ES12.4, '      -     ' )" ) prefix, i, C( i ), C_l( i ), Y_l( i )
        END DO

!  The constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          nbnds = nbnds + 2

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i ) 
          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MIN( C_u( i ) - c_feasmin * MAX( one, ABS( C_u( i ) ) ),    &
                        MAX( R( i ) / SCALE_C( i ),                            &
                          C_l( i ) + c_feasmin * MAX( one, ABS( C_l( i ) ) ) ) )
          DIST_C_l( i ) = C( i ) - C_l( i ) 
          DIST_C_u( i ) = C_u( i ) - C( i )
          Y_l( i ) = MAX( Y( i ),   dufeas )
          Y_u( i ) = MIN( Y( i ), - dufeas )
          IF ( printd ) WRITE( out, "( A, I6, 5ES12.4 )" )                     &
            prefix, i, C( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i )
        END DO

!  The constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          nbnds = nbnds + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MIN( R( i ) / SCALE_C( i ),                                 &
                        C_u( i ) - c_feasmin * MAX( one, ABS( C_u( i ) ) ) )
          DIST_C_u( i ) = C_u( i ) - C( i )
          Y_u( i ) = MIN( Y( i ), - dufeas ) 
          IF ( printd ) WRITE( out, "( A, I6, ES12.4, '      -     ', ES12.4,  &
         &  '      -     ', ES12.4 )" ) prefix, i, C( i ), C_u( i ), Y_u( i )
        END DO
      END IF

      scaled_c = .TRUE.
      IF ( printi .AND. m > 0 .AND. dims%c_l_start <= dims%c_u_end )           &
        WRITE( out, "( A, ' largest/smallest scale factor ', 2ES12.4 )" )      &
          prefix, MAXVAL( SCALE_C ), MINVAL( SCALE_C )

!  Find the largest components of A and H

      IF ( a_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : a_ne ) ) )
      ELSE
        amax = zero
      END IF

      IF ( h_ne > 0 ) THEN
        hmax = MAX( hmin, MAXVAL( ABS( H_val( : h_ne ) ) ) )
      ELSE
        hmax = hmin
      END IF

      IF ( printi ) WRITE( out, "( A, '  maximum element of A =', ES11.4,     &
     &  ' maximum element of H =', ES11.4 ) " ) prefix, amax, hmax 

!  Compute the product between H and x

      HX( : n ) = zero
      CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr, X, '+' )

!  Now, calculate the value .... 

      obj = half * DOT_PRODUCT( X, HX( : n ) ) + DOT_PRODUCT( X, G )
      inform%obj = obj + f

!  ... and gradient of the objective function

      GRAD = HX( : n ) + G

!  complete A

      DO i = 1, dims%nc
        A_sbls%val( a_ne + i ) = - SCALE_C( dims%c_equality + i )
      END DO

!  ===============
!  Outer iteration
!  ===============

!  Initialize penalty parameter, mu

      norm_c = DOT_PRODUCT( X( dims%x_free + 1 : dims%x_l_start - 1 ),         &
                            Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ) +    &
               DOT_PRODUCT( DIST_X_l( dims%x_l_start : dims%x_l_end ),         &
                            Z_l( dims%x_l_start : dims%x_l_end ) ) -           &
               DOT_PRODUCT( DIST_X_u( dims%x_u_start : dims%x_u_end ),         &
                            Z_u( dims%x_u_start : dims%x_u_end ) ) +           &
               DOT_PRODUCT( X( dims%x_u_end + 1 : n ),                         &
                            Z_u( dims%x_u_end + 1 : n ) ) +                    &
               DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),         &
                            Y_l( dims%c_l_start : dims%c_l_end ) ) -           &
               DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),         &
                            Y_u( dims%c_u_start : dims%c_u_end ) )

      IF ( nbnds > 0 ) THEN
        norm_c = ten ** ( 2 * ( NINT( LOG10( norm_c / nbnds ) ) / 2  ) )
      ELSE
        norm_c = zero
      END IF

      IF ( control%muzero < zero ) THEN
        mu = norm_c
      ELSE
        mu = control%muzero
      END IF

!  If the starting point is very close to one of its bounds, 
!  the feasible region likely has no interior. Be cautious

      old_mu = mu ; zeta = point01 * point01

!  Initialize convergence tolerances, theta_c, theta_d and theta_e

      theta_c = MIN( theta_min, control%theta_c * mu )
      theta_d = MIN( theta_min, control%theta_d * mu )

!  Prepare for the major iteration

      inform%iter = 0 ; inform%nfacts = 0 ; ratio = - one
      got_time_kkt = .FALSE. ; time_kkt = 0.0

!  Update GLTR control data

      GLTR_control%stop_relative = control%inner_stop_relative
      GLTR_control%stop_absolute = control%inner_stop_absolute
      GLTR_control%fraction_opt = control%inner_fraction_opt
      GLTR_control%error = control%error
      GLTR_control%out = control%out
      GLTR_control%print_level = print_level - 1
      GLTR_control%itmax = control%cg_maxit

!  If the Hessian is diagonal and the preconditioner is to be picked
!  automatically, start with the full Hessian. Otherwise, use the
!  Hessian of the barrier function

      one_fact = .FALSE.
      new_fact = .TRUE.
      IF ( printi ) WRITE( out, "( ' ' )" )
      IF ( auto ) THEN
        IF ( printi ) WRITE( out, 2400 ) prefix
        SBLS_control%preconditioner = 2 ; fact_hist = 4

!  fact_hist indicates which factors are currently being used. Possible values:
!   1 barrier factors used
!   2 full factors used
!   3 barrier factors used for the final time
!   4 full factors used for the final time
!   5 barrier factors as a last resort

!  Check to see if the Hessian is diagonal

 dod :  DO type = 1, 6
        
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
            IF ( H_ptr( i + 1 ) /= H_ptr( i ) + 1 ) THEN
              SBLS_control%preconditioner = 5 ; fact_hist = 1
              EXIT dod
            END IF
          END DO
          IF ( hd_end == n ) EXIT
    
!  rows without a diagonal entry
    
          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            IF ( H_ptr( i + 1 ) /= H_ptr( i ) ) THEN
              SBLS_control%preconditioner = 5 ; fact_hist = 1
              EXIT dod
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO dod

      ELSE

!  Check to see if the Hessian is diagonal

 dod2:  DO type = 1, 6
        
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
            IF ( H_ptr( i + 1 ) /= H_ptr( i ) + 1 ) THEN
              EXIT dod2
            END IF
          END DO
          IF ( hd_end == n ) EXIT
    
!  rows without a diagonal entry
    
          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            IF ( H_ptr( i + 1 ) /= H_ptr( i ) ) THEN
              EXIT dod2
            END IF
          END DO
          IF ( hnd_end == n ) EXIT
        END DO dod2

      END IF

      refact = .TRUE. ; re = ' ' ; mo = ' '
      SBLS_control%new_h = 2
      new_prec = .FALSE. ; full_iteration = .FALSE. ; got_ratio = .FALSE.
      successful_iteration = .FALSE.

      cg_iter = 0 ; inform%cg_iter = cg_iter

!  Compute the value of the barrier function, phi

      phi = QPB_barrier_value( dims, n, obj, X, DIST_X_l, DIST_X_u,            &
                                DIST_C_l, DIST_C_u, mu )

      GLTR_control%boundary = .FALSE.
      inform%GLTR_inform%status = 1
      inform%GLTR_inform%negative_curvature = .TRUE.

      DO  ! outer iteration loop

!  The vectors x, z_l and z_u satisfy
!  a) A x = b
!  b) X_l < x < X_u, z_l > 0, z_u < 0

!  Set the initial trust-region radius radius
   
        IF ( initial_radius <= zero ) THEN
          radius = ( ten ** 3 ) * MAX( one, mu )
        ELSE
          radius = MIN( initial_radius, radius_max )
        END IF
        old_radius = radius
        old_norm_cd = infinity

!  ===============
!  Inner iteration
!  ===============

        inner_iteration = 0
 inner: DO  ! inner iteration loop

          CALL CPU_TIME( time_now ) ; time_now = time_now - time_start
          time_iter = time_now - time_last ; time_last = time_now

!         WRITE( out, "( ' time_iter ', F10.2 )" ) time_iter

!  Estimate the time for an iteration with the full factorization

          IF ( auto ) THEN
            IF( fact_hist == 2 .AND.                                           &
               ( .NOT. got_time_kkt .AND. full_iteration ) ) THEN
              got_time_kkt = .TRUE.
              time_kkt = time_iter
            END IF

!  Check the effectiveness of the preconditioner over the previous iteration

            IF ( .NOT. start_major ) THEN 
              IF ( seq > 0 ) THEN
                 time_mean = SUM( time_hist( 0 : MIN( seq - 1, history - 1 ) ))&
                             / REAL( MIN( seq, history ) )
                 IF ( time_mean > 0.0 ) THEN
                   time_ratio = time_iter / time_mean
                 ELSE
                   time_ratio = 0.0
                 END IF
              ELSE
                 time_mean = 0.0 ; time_ratio = 0.0
              END IF
   
              time_hist( MOD( seq, history ) ) = time_iter
              seq = seq + 1

!  The previous preconditioner appears to be ineffective. Try another

              SELECT CASE( fact_hist ) 

!  The time/iteration has significantly increased. See if a full factorization
!  is better

              CASE ( 1 )
                IF ( seq >= history .AND. time_ratio > max_ratio ) THEN
                  IF ( printi )                                                &
                    WRITE( out, "( /, A, '  Iteration time (', 0P, F9.2,       &
                   &  ') exceeds', /, A, '    average time (',     F9.2,       &
                   &   ') by more than a factor of', F9.2 )" )                 &
                     prefix, time_iter, prefix, time_mean, max_ratio
                  new_prec = .TRUE.
                  time_itsol = time_iter
                  SBLS_control%preconditioner = 2 ; fact_hist = 2
                END IF

!  The full factorization is more expensive than the barrier factorization.
!  Revert to the latter

              CASE ( 2 )
                IF ( got_time_kkt .AND. time_kkt > time_itsol ) THEN
                  IF ( printi ) WRITE( out, "( /, A, '  Time (', 0P, F9.2,     &
                 &    ') for full factorization exceeds ', /, A,               &
                 & '  time (',     F9.2, ') for barrier factorization' )" )    &
                    prefix, time_kkt, prefix, time_itsol
                  new_prec = .TRUE.
                  SBLS_control%preconditioner = 5 ; fact_hist = 3
                END IF

!  The barrier factorization is more expensive than the full factorization.
!  Revert to the latter

              CASE ( 3 )
                IF ( time_iter > time_kkt ) THEN
                  IF ( printi ) WRITE( out, "( /, A, '  Time (', 0P, F9.2,     &
                 &   ') for barrier factorization exceeds', /, A,              &
                 &   '  time (',     F9.2, ') for full factorization')" )      &
                   prefix, time_iter, prefix, time_kkt
                  new_prec = .TRUE.
                  SBLS_control%preconditioner = 2 ; fact_hist = 4
                END IF
              END SELECT
            END IF
          END IF

!  Compute the derivatives of the barrier function, and the norm of the
!  complentarity, || (X-L) z_l - \mu e, (X-U) z_u - \mu e \|

!  Problem variables:

          p_min = infinity ; p_max = zero ; d_min = infinity ; d_max = zero 
          norm_c = zero

          GRAD_X_phi( : dims%x_free ) = GRAD( : dims%x_free )
          DO i = dims%x_free + 1, dims%x_l_start - 1
            GRAD_X_phi( i ) = GRAD( i ) - mu / X( i )
            p_min = MIN( p_min, X( i ) )
            p_max = MAX( p_max, X( i ) )
            d_min = MIN( d_min, Z_l( i ) )
            d_max = MAX( d_max, Z_l( i ) )
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( X( i ) * Z_l( i ) - mu )
          END DO

          DO i = dims%x_l_start, dims%x_u_start - 1
            GRAD_X_phi( i ) = GRAD( i ) - mu / DIST_X_l( i )
            p_min = MIN( p_min, DIST_X_l( i ) )
            p_max = MAX( p_max, DIST_X_l( i ) )
            d_min = MIN( d_min, Z_l( i ) )
            d_max = MAX( d_max, Z_l( i ) )
            IF ( ABS( DIST_X_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS(   DIST_X_l( i ) * Z_l( i ) - mu )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            GRAD_X_phi( i ) = GRAD( i ) - mu / DIST_X_l( i )                   &
                                        + mu / DIST_X_u( i )
            p_min = MIN( p_min, DIST_X_l( i ), DIST_X_u( i ) )
            p_max = MAX( p_max, DIST_X_l( i ), DIST_X_u( i ) )
            d_min = MIN( d_min, Z_l( i ), - Z_u( i ) )
            d_max = MAX( d_max, Z_l( i ), - Z_u( i ) )
            IF ( MIN( ABS( DIST_X_l( i ) ), ABS( DIST_X_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) - mu ),    &
                                    ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, MAX( ABS(   DIST_X_l( i ) * Z_l( i ) - mu ),      &
!                     ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
          END DO

          DO i = dims%x_l_end + 1, dims%x_u_end
            GRAD_X_phi( i ) = GRAD( i ) + mu / DIST_X_u( i )
            p_min = MIN( p_min, DIST_X_u( i ) )
            p_max = MAX( p_max, DIST_X_u( i ) )
            d_min = MIN( d_min, - Z_u( i ) )
            d_max = MAX( d_max, - Z_u( i ) )
            IF ( ABS( DIST_X_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( - DIST_X_u( i ) * Z_u( i ) - mu )
          END DO

          DO i = dims%x_u_end + 1, n
            GRAD_X_phi( i ) = GRAD( i ) - mu / X( i )
            p_min = MIN( p_min, - X( i ) )
            p_max = MAX( p_max, - X( i ) )
            d_min = MIN( d_min, - Z_u( i ) )
            d_max = MAX( d_max, - Z_u( i ) )
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( X( i ) * Z_l( i ) - mu )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            GRAD_C_phi( i ) = - mu / DIST_C_l( i )
            p_min = MIN( p_min, DIST_C_l( i ) )
            p_max = MAX( p_max, DIST_C_l( i ) )
            d_min = MIN( d_min, Y_l( i ) )
            d_max = MAX( d_max, Y_l( i ) )
            IF ( ABS( DIST_C_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( DIST_C_l( i ) * Y_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, ABS( DIST_C_l( i ) * Y_l( i ) - mu )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            GRAD_C_phi( i ) = - mu / DIST_C_l( i ) + mu / DIST_C_u( i )
            p_min = MIN( p_min, DIST_C_l( i ), DIST_C_u( i ) )
            p_max = MAX( p_max, DIST_C_l( i ), DIST_C_u( i ) )
            d_min = MIN( d_min, Y_l( i ), - Y_u( i ) )
            d_max = MAX( d_max, Y_l( i ), - Y_u( i ) )
            IF ( MIN( ABS( DIST_C_l( i ) ), ABS( DIST_C_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_C_l( i ) * Y_l( i ) - mu ),    &
                                    ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, MAX( ABS(   DIST_C_l( i ) * Y_l( i ) - mu ),      &
!                     ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            GRAD_C_phi( i ) = mu / DIST_C_u( i )
            p_min = MIN( p_min, DIST_C_u( i ) )
            p_max = MAX( p_max, DIST_C_u( i ) )
            d_min = MIN( d_min, - Y_u( i ) )
            d_max = MAX( d_max, - Y_u( i ) )
            IF ( ABS( DIST_C_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) 
          END DO

          IF ( printt ) WRITE( out, 2160 )                                     &
            prefix, p_min, p_max, prefix, d_min, d_max

!  Build the model of the barrier function
!  ---------------------------------------

!  Construct the quadratic model
!  m(s) = phi + <s,grad phi> + 1/2 <s,(Hess f + (X-L)(-2) + (U-X)(-2))s>

!  Compute the Hessian matrix of the barrier terms

!  problem variables:

          IF ( primal_hessian ) THEN
            DO i = dims%x_free + 1, dims%x_l_start - 1
              BARRIER_X( i ) = MAX( bar_min, old_mu / X( i ) ** 2 )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              BARRIER_X( i ) = MAX( bar_min,old_mu / DIST_X_l( i ) ** 2 )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              BARRIER_X( i ) = MAX( bar_min, old_mu / DIST_X_l( i ) ** 2 +     &
                                             old_mu / DIST_X_u( i ) ** 2 )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              BARRIER_X( i ) = MAX( bar_min, old_mu / DIST_X_u( i ) ** 2 )
            END DO
            DO i = dims%x_u_end + 1, n
              BARRIER_X( i ) = MAX( bar_min, old_mu / X( i ) ** 2 )
            END DO
          ELSE
            DO i = dims%x_free + 1, dims%x_l_start - 1
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / X( i ) )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / DIST_X_l( i ) )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / DIST_X_l( i ) -        &
                                             Z_u( i ) / DIST_X_u( i ) )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              BARRIER_X( i ) = MAX( bar_min, - Z_u( i ) / DIST_X_u( i ) )
            END DO
            DO i = dims%x_u_end + 1, n
              BARRIER_X( i ) = MAX( bar_min, Z_u( i ) / X( i ) )
            END DO
          END IF

!  slack variables:

!         BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero

          IF ( primal_hessian ) THEN
            DO i = dims%c_l_start, dims%c_u_start - 1
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_l( i ) ** 2 )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_l( i ) ** 2 +     &
                                             old_mu / DIST_C_u( i ) ** 2 )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_u( i ) ** 2 )
            END DO
          ELSE
            DO i = dims%c_l_start, dims%c_u_start - 1
              BARRIER_C( i ) = MAX( bar_min, Y_l( i ) / DIST_C_l( i ) )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              BARRIER_C( i ) = MAX( bar_min, Y_l( i ) / DIST_C_l( i ) -        &
                                             Y_u( i ) / DIST_C_u( i ) )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              BARRIER_C( i ) = MAX( bar_min, - Y_u( i ) / DIST_C_u( i ) )
            END DO
          END IF

!  If required, form and factorize the preconditioner
!  --------------------------------------------------

          mo = ' '
          IF ( refact ) THEN 

!  Only refactorize if M has changed

            IF ( one_fact .AND. .NOT. new_fact ) THEN
              re = ' ' 
            ELSE
              re = 'r'
              new_fact = .FALSE.

!  The previous preconditioner appears to be ineffective. Try another

              IF ( new_prec ) THEN
                seq = 0
                IF ( printi ) THEN
                  WRITE( out, "( /, A, ' *** changing preconditioner ... ',    &
                 &  / )" ) prefix
                  WRITE( out, 2400 ) prefix
                END IF
                SBLS_control%new_h = 2

                IF ( printt ) WRITE( out,                                      &
                 "( A, ' previous int, real space used ', 2I12 )") prefix,     &
                  inform%factorization_integer, inform%factorization_real
                new_prec = .FALSE.
              END IF

              CALL CLOCK_time( clock_record ) 
              IF ( printw ) WRITE( out, "( A,                                  &
             &  ' ............... factorization ...............' )" ) prefix

              get_factors = .TRUE.

!  Set the diagonal terms

              H_sbls%val( : dims%x_free ) = zero
              H_sbls%val( dims%x_free + 1 : dims%x_e ) = BARRIER_X
              H_sbls%val( dims%c_s : dims%c_e ) = BARRIER_C

! :::::::::::::::::::::::::::::
!  Factorize the preconditioner
! :::::::::::::::::::::::::::::

              IF ( get_factors ) THEN
                IF ( SBLS_control%preconditioner /= 5 ) THEN
                  CALL SBLS_form_and_factorize( A_sbls%n, A_sbls%m,            &
                     H_sbls, A_sbls, C_sbls, sbls_data, SBLS_control,          &
                     inform%SBLS_inform )
                ELSE
                  S( : dims%x_free ) = H_sbls%val( : dims%x_free ) + one
                  S( dims%x_free + 1 : dims%c_e ) =                            &
                    H_sbls%val( dims%x_free + 1 : dims%c_e ) 
                   CALL SBLS_form_and_factorize( A_sbls%n, A_sbls%m,           &
                       H_sbls, A_sbls, C_sbls, sbls_data, SBLS_control,        &
                       inform%SBLS_inform, D = S )
                END IF
                inform%nfacts = inform%nfacts + 1 
                inform%time%analyse = inform%time%analyse +                    &
                  inform%SBLS_inform%SLS_inform%time%analyse
                inform%time%clock_analyse = inform%time%clock_analyse +        &
                  inform%SBLS_inform%SLS_inform%time%clock_analyse
                inform%time%factorize = inform%time%factorize +                &
                  inform%SBLS_inform%SLS_inform%time%factorize
                inform%time%clock_factorize = inform%time%clock_factorize +    &
                  inform%SBLS_inform%SLS_inform%time%clock_factorize

                IF ( printw ) WRITE( out, "( A, ' ...............',            &
               &  ' end of factorization ...............' )" ) prefix

!  Test that the factorization succeeded

                inform%factorization_status = inform%SBLS_inform%status
                IF ( inform%factorization_status < 0 ) THEN
                  IF ( printe ) WRITE( control%error,                          &
                   "( A, '   **  Error return ', I0, ' from ', A )" ) prefix,  &
                    inform%factorization_status, 'SBLS_form_and_factorize'
                  IF ( inform%factorization_status == - 1 .OR.                 &
                       inform%factorization_status == - 2 ) THEN
                    IF ( auto .AND. SBLS_control%preconditioner /= 5 ) THEN
                      refact = .TRUE. ; new_prec = .TRUE.
                      SBLS_control%new_h = 2
                      SBLS_control%preconditioner = 5 ; fact_hist = 5
                      CYCLE inner
                    END IF
                  ELSE
                    IF ( SBLS_control%SLS_control%relative_pivot_tolerance     &
                         < half .AND. SBLS_control%factorization == 2 ) THEN
                      IF ( SBLS_control%SLS_control%relative_pivot_tolerance < &
                           point01 ) THEN
                        SBLS_control%SLS_control%relative_pivot_tolerance      &
                          = point01
                      ELSE
                        IF ( SBLS_control%SLS_control%relative_pivot_tolerance &
                             < point1 ) THEN
                          SBLS_control%SLS_control%relative_pivot_tolerance    &
                            = point1
                        ELSE
                          SBLS_control%SLS_control%relative_pivot_tolerance    &
                            = half
                        END IF
                      END IF
                      refact = .TRUE. ; new_fact = .TRUE.
                      IF ( printi ) WRITE( out, "( A, ' increasing relative',  &
                     &  ' pivot tolerance to ', ES12.4, / )" ) prefix,         &
                        SBLS_control%SLS_control%relative_pivot_tolerance
                      CYCLE inner
                    ELSE IF ( SBLS_control%factorization /= 2 ) THEN
                      SBLS_control%SLS_control%relative_pivot_tolerance =      &
                        relative_pivot_tol
                      SBLS_control%factorization = 2
                      refact = .TRUE. ; new_fact = .TRUE. ; new_prec = .TRUE.
                      SBLS_control%new_h = 2
                      IF ( printi ) WRITE( out,                                &
                        "( A, ' changing to augmented system ', / )" ) prefix
                      CYCLE inner
                    END IF
                  END IF
                  inform%status = GALAHAD_error_factorization
                  GO TO 700

!  Record warning conditions

                ELSE IF ( inform%factorization_status > 0 ) THEN
                  IF ( printt ) WRITE( control%out,                            &
                    "( A, '   **  Warning ', I0, ' from ', A1,                 &
                 &  ' ** Matrix has ', I0, ' zero eigenvalue', A )" ) prefix,  &
                      inform%factorization_status, 'SBLS_form_and_factorize',  &
                      dims%v_e - inform%SBLS_inform%rank, TRIM(                &
                        STRING_pleural( dims%v_e - inform%SBLS_inform%rank ) )
                END IF 

                SBLS_control%new_h = 1 ; SBLS_control%new_a = 0
                SBLS_control%new_c = 0
                allow_extrapolate = inform%SBLS_inform%preconditioner == 2     &
                 .AND. inform%SBLS_inform%perturbed
              END IF
              old_mu = mu

              IF ( printt ) WRITE( out, "( A, I0, ' integer and ', I0,         &
             &  ' real words needed for factorization' )" ) prefix,            &
               inform%factorization_integer, inform%factorization_real
              IF ( dims%v_e > 0 .AND. printt ) THEN
                CALL CLOCK_time( clock_now ) 
                 WRITE( out, "( A, ' factorize time = ', F10.2, /, A,          &
               & ' real/integer space used for factors ', I0, 1X, I0 )" )      &
                    prefix, clock_now - clock_record, prefix,                  &
                    inform%SBLS_inform%factorization_real,                     &
                    inform%SBLS_inform%factorization_integer
              END IF
            END IF

!  Ensure that the projection is only computed once
                  
            IF ( SBLS_control%preconditioner == 1 ) one_fact = .TRUE.
            full_iteration = .TRUE.
          ELSE 
            re = ' ' 
            full_iteration = .FALSE.
          END IF 

!  Check for convergence of the inner iteration
!  --------------------------------------------

!  Check if x, z_l, z_u satisfy the convergence tests

!  a) A x = b
!  b) X_l < x < X_u, z_l > 0, z_u < 0
!  c) || (X-L) z_l - \mu e, (X-U) z_u - \mu e \|_2 <= theta_c
!  d) || GRAD - z_l - z_u ||_M <= theta_d
!  e) leftmost eigenvalue of
!         N(trans)( H + (X-L)(inv)Z_l + (X-U)(inv)Z_u)N >= - theta_e

!  where GRAD = grad phi + mu ( (X-L)(inv) - (U-X)(inv) ) e = grad f
!  and N is an orthononormal basis for the null-space of A

!  Compute GRAD - z_l - z_u

!  Problem variables

          VECTOR( : dims%x_free ) = GRAD( : dims%x_free )

          DO i = dims%x_free + 1, dims%x_u_start - 1
            VECTOR( i ) = GRAD( i ) - Z_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            VECTOR( i ) = GRAD( i ) - Z_l( i ) - Z_u( i )
          END DO

          DO i = dims%x_l_end + 1, n
            VECTOR( i ) = GRAD( i ) - Z_u( i )
          END DO

!  Slack variables

          DO i = dims%c_l_start, dims%c_u_start - 1
            VECTOR( dims%c_b + i ) = - Y_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            VECTOR( dims%c_b + i ) = - Y_l( i ) - Y_u( i )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            VECTOR( dims%c_b + i ) = - Y_u( i )
          END DO

          CALL QPB_AX( n, VECTOR( : n ), m, a_ne, A_val,                       &
                        A_col, A_ptr, m, Y, '-T' )
          VECTOR( dims%c_s : dims%c_e ) = VECTOR( dims%c_s : dims%c_e ) +      &
                                SCALE_C * Y( dims%c_l_start : dims%c_u_end )
          VECTOR( dims%y_s : dims%y_e ) = zero

!  Calculate || GRAD - A^T y - z_l - z_u ||_2

          norm_d_alt = SQRT( ABS( SUM( VECTOR( : dims%c_e ) ** 2 ) ) )
!   write(6,"( ' d_alt ', ES12.4 )" )  norm_d_alt

!  Calculate || GRAD - z_l - z_u ||_M

          IF ( printw ) WRITE( out,                                            &
            "( A, ' .............. get multipliers .............. ' )" ) prefix

          IF ( m > 0 ) THEN
            S = VECTOR( : dims%c_e )
            res_fail = MAX( point1 * MAXVAL( ABS( VECTOR ) ), res_large )
            SBLS_control%affine = .TRUE.
            CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls, SBLS_data,    &
                             SBLS_control, inform%SBLS_inform, VECTOR )

!  Take appropriate action if the residual is too large

            IF ( inform%SBLS_inform%norm_residual > res_fail ) THEN
              IF ( printi .AND. SBLS_control%preconditioner > 0 )              &
                CALL SBLS_cond( SBLS_data, out, inform%SBLS_inform )
              IF ( printi ) WRITE( out, 2340 ) prefix,                         &
                inform%SBLS_inform%norm_residual, res_large
              IF ( SBLS_control%SLS_control%relative_pivot_tolerance < half    &
                   .AND. SBLS_control%factorization == 2 ) THEN
                IF ( SBLS_control%SLS_control%relative_pivot_tolerance <       &
                     point01 ) THEN
                  SBLS_control%SLS_control%relative_pivot_tolerance = point01
                ELSE
                  IF ( SBLS_control%SLS_control%relative_pivot_tolerance <     &
                       point1 ) THEN
                    SBLS_control%SLS_control%relative_pivot_tolerance = point1
                  ELSE
                    SBLS_control%SLS_control%relative_pivot_tolerance = half
                  END IF
                END IF
                refact = .TRUE. ; new_fact = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( A, ' increasing pivot tolerance to ', ES12.4, / )" )      &
                  prefix, SBLS_control%SLS_control%relative_pivot_tolerance
                CYCLE inner
              ELSE IF ( SBLS_control%factorization /= 2 ) THEN
                SBLS_control%SLS_control%relative_pivot_tolerance =            &
                  relative_pivot_tol
                SBLS_control%factorization = 2
                refact = .TRUE. ; new_fact = .TRUE. 
                SBLS_control%new_h = 2 ; new_prec = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( A, ' changing to augmented system method ', / )" ) prefix
                CYCLE inner
              ELSE
                inform%status = GALAHAD_error_ill_conditioned
                GO TO 700
              END IF
            END IF
            IF ( inform%status /= GALAHAD_ok ) GO TO 700   

            norm_d = SQRT( ABS( DOT_PRODUCT( S, VECTOR( : dims%c_e ) ) ) )

            IF ( inform%SBLS_inform%norm_residual > control%stop_d             &
                 .AND. auto ) THEN
              IF ( ( fact_hist == 1 .OR. fact_hist == 3 ) .AND.                &
                   ( SBLS_control%factorization == 0 .OR.                      &
                     SBLS_control%factorization == 1 ) ) THEN
                 SBLS_control%factorization = 2
                 refact = .TRUE. ; new_prec = .TRUE.
                 SBLS_control%new_h = 2
                 IF ( printi ) WRITE( out, 2340 ) prefix,                      &
                   inform%SBLS_inform%norm_residual, control%stop_d
                 CYCLE
              END IF
            END IF
            norm_d = MIN( norm_d, norm_d_alt )
!           norm_d = norm_d_alt
          ELSE
            norm_d = norm_d_alt
          END IF

!  Print a summary of the iteration

          CALL CLOCK_TIME( clock_now ) ; clock_now = clock_now - clock_start
          IF ( printi ) THEN
            bdry = ' '
            IF ( .NOT. start_major ) THEN 
              IF ( ABS( inform%GLTR_inform%mnormx - old_radius ) < teneps)     &
                 bdry = 'b'
              IF ( printt .OR. GLTR_control%print_level > 0 .OR. ( printi      &
                 .AND. inform%iter == start_print ) ) WRITE( out, 2000 ) prefix
              IF ( got_ratio ) THEN
                 WRITE( out, 2020 ) prefix, inform%iter, re, norm_d, norm_c,   &
                                    inform%obj, mo, ratio, old_radius, bdry,   &
                                    nbacts, cg_iter, clock_now
              ELSE
                 WRITE( out, 2030 ) prefix, inform%iter, re, norm_d, norm_c,   &
                                    inform%obj, mo, old_radius, bdry,          &
                                    nbacts, cg_iter, clock_now
              END IF
            ELSE
              IF ( printi ) WRITE( out, 2130 )                                 &
                prefix, prefix, mu, theta_c, theta_d, prefix
              WRITE( out, 2000 ) prefix
              WRITE( out, 2010 ) prefix, inform%iter, re, norm_d, norm_c,      &
                                 inform%obj, mo, old_radius, clock_now
            END IF 
            IF ( printd ) THEN 
              WRITE( out, 2100 ) prefix, ' X ', X
              WRITE( out, 2100 ) prefix, ' Z_l ', Z_l
              WRITE( out, 2100 ) prefix, ' Z_u ', Z_u
            END IF 
          END IF 
          start_major = .FALSE.

!  Record the Lagrange multipliers

          Y = Y + VECTOR( dims%y_s : dims%y_e )
!         WRITE( 6, "( ' norm Y = ', ES22.14 )") MAXVAL( ABS( Y ) )

!  Test for convergence

          norm_cd =  norm_c + norm_d
          IF ( ( norm_c <= theta_c .AND. norm_d <= theta_d .AND. .NOT.         &
               inform%GLTR_inform%negative_curvature .AND. .NOT.               &
               ( control%extrapolate > 1 .AND. inner_iteration == 0 ) ) .OR.   &
               ( control%extrapolate > 0 .AND. inner_iteration > 2 .AND.       &
                 mu < 1.01_wp * control%mu_min .AND. successful_iteration .AND.&
                 norm_cd > 0.9_wp * old_norm_cd ) ) THEN
            inform%status = GALAHAD_ok
            inform%GLTR_inform%status = 1
            start_major = .TRUE.
            got_ratio = .FALSE.
            full_iteration = .FALSE.
            EXIT
          END IF
          old_norm_cd = norm_cd

!  Test to see if more than maxit iterations have been performed

          inform%iter = inform%iter + 1 
          inner_iteration = inner_iteration + 1
          IF ( inform%iter > control%maxit ) THEN 
            inform%status = GALAHAD_error_max_iterations ; GO TO 600 
          END IF 

!  Check that the CPU time limit has not been reached

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - time_start > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - clock_start > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit ; GO TO 600
          END IF 

          IF ( inform%iter == start_print ) THEN
            printe = set_printe ; printi = set_printi ; printt = set_printt
            printm = set_printm ; printw = set_printw ; printd = set_printd
            print_level = control%print_level
            GLTR_control%print_level = print_level - 1
          END IF
 
          IF ( inform%iter == stop_print + 1 ) THEN
            printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
            printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
            print_level = 0 ; GLTR_control%print_level = 0
          END IF

!       WRITE( 6, "( ' start, stop print, iter ', 3I8 )" )                     &
!         start_print, stop_print, inform%iter

!  Test to see if the objective appears to be unbounded from below

          IF ( inform%obj < control%obj_unbounded ) THEN 
            inform%status = GALAHAD_error_unbounded ; GO TO 600 
          END IF 

!  Compute the trial step
!  ----------------------

          IF ( printw ) WRITE( out,                                            &
           "( A, ' ............... compute step  ............... ' )" ) prefix

!  Compute a trial step, s, to ``sufficiently'' reduce the model within 
!  the region defined by the intersection of the affine constraints A s = 0
!  and the trust region || s ||_M <= radius

!  Compute the derivatives of the Lagrangian function

          GRAD_L( dims%x_s : dims%x_e ) = GRAD_X_phi
          CALL QPB_AX( n, GRAD_L( dims%x_s : dims%x_e ), m, a_ne,              &
                        A_val, A_col, A_ptr, m, Y, '-T' )
          GRAD_L( dims%c_s : dims%c_e ) = GRAD_C_phi +                         &
                                SCALE_C * Y( dims%c_l_start : dims%c_u_end )

          IF ( printm ) WRITE( out, "( A, ' norm GRAD_L ', ES12.4 ) " )        &
                               prefix, MAXVAL( ABS( GRAD_L ) )
!  Set initial data

          R( : dims%c_e ) = GRAD_L
          first_iteration = .TRUE.

          IF ( printm ) WRITE( out,                                            &
         "(/, A, '   |------------------------------------------------------|',&
       &   /, A, '   |        start to solve trust-region subproblem        |',&
       &   / )" ) prefix, prefix

          CALL CPU_TIME( time_record ) ; CALL CLOCK_TIME( clock_record )

!  Iteration

  100     CONTINUE
          CALL GLTR_solve( dims%c_e, radius, model, S, R( : dims%c_e ),        &
                           VECTOR( : dims%c_e ), GLTR_data,                    &
                           GLTR_control, inform%GLTR_inform )

!  Check for error returns

          SELECT CASE( inform%GLTR_inform%status )

!  Successful return

          CASE ( GALAHAD_ok )

!  Warnings

          CASE ( GALAHAD_warning_on_boundary, GALAHAD_error_max_iterations )
            IF ( printt ) WRITE( out, "( /, A,                                 &
           &  ' Warning return from GLTR, status = ', I0 )" )                  &
             prefix, inform%GLTR_inform%status
          
!  Allocation errors

           CASE ( GALAHAD_error_allocate )
             inform%status = GALAHAD_error_allocate
             inform%alloc_status = inform%GLTR_inform%alloc_status
             inform%bad_alloc = inform%GLTR_inform%bad_alloc
             GO TO 920

!  Deallocation errors

           CASE ( GALAHAD_error_deallocate )
             inform%status = GALAHAD_error_deallocate
             inform%alloc_status = inform%GLTR_inform%alloc_status
             inform%bad_alloc = inform%GLTR_inform%bad_alloc
             GO TO 920

!  Error return

          CASE DEFAULT
             inform%status = inform%GLTR_inform%status
            IF ( printt ) WRITE( out, "( /, A,                                 &
           &  ' Error return from GLTR, status = ', I0 )" )                    &
              prefix, inform%GLTR_inform%status

!  Find the preconditioned gradient

          CASE ( 2, 6 )
            IF ( printw ) WRITE( out,                                          &
             "( A, ' ............... precondition  ............... ' )" ) prefix

            VECTOR( dims%y_s : dims%y_e ) = zero

            res_fail = MAX( point1 * MAXVAL( ABS( VECTOR ) ), res_large )
            SBLS_control%affine = .TRUE.
            CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls, sbls_data,    &
                             SBLS_control, inform%SBLS_inform, VECTOR )

!  Take appropriate action if the residual is too large

            IF ( inform%SBLS_inform%norm_residual > res_fail ) THEN
              IF ( printi .AND. SBLS_control%preconditioner > 0 )              &
                CALL SBLS_cond( SBLS_data, out, inform%SBLS_inform )
              IF ( printi ) WRITE( out, 2340 ) prefix,                         &
                inform%SBLS_inform%norm_residual, res_large
              IF ( SBLS_control%SLS_control%relative_pivot_tolerance < half    &
                   .AND. SBLS_control%factorization == 2 ) THEN
                IF ( SBLS_control%SLS_control%relative_pivot_tolerance <       &
                     point01 ) THEN
                  SBLS_control%SLS_control%relative_pivot_tolerance = point01
                ELSE
                  IF ( SBLS_control%SLS_control%relative_pivot_tolerance <     &
                       point1 ) THEN
                    SBLS_control%SLS_control%relative_pivot_tolerance = point1
                  ELSE
                    SBLS_control%SLS_control%relative_pivot_tolerance = half
                  END IF
                END IF
                refact = .TRUE. ; new_fact = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( A, ' increasing pivot tolerance to ', ES12.4, / )" )      &
                    prefix, SBLS_control%SLS_control%relative_pivot_tolerance
                CYCLE inner
              ELSE IF ( SBLS_control%factorization /= 2 ) THEN
                SBLS_control%SLS_control%relative_pivot_tolerance =            &
                  relative_pivot_tol
                SBLS_control%factorization = 2
                refact = .TRUE. ; new_fact = .TRUE. 
                SBLS_control%new_h = 2 ; new_prec = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( A, ' changing to augmented system method ', / )" ) prefix
                CYCLE inner
              ELSE
                inform%status = GALAHAD_error_ill_conditioned
                GO TO 700
              END IF
            END IF
            IF ( inform%status /= GALAHAD_ok ) GO TO 700   
            
            IF ( inform%SBLS_inform%norm_residual > control%stop_d             &
                 .AND. auto ) THEN
              IF ( ( fact_hist == 1 .OR. fact_hist == 3 ) .AND.                &
                   ( SBLS_control%factorization == 0 .OR.                      &
                     SBLS_control%factorization == 1 ) ) THEN
                SBLS_control%factorization = 2 ; refact = .TRUE.
                SBLS_control%new_h = 2 ; new_prec = .TRUE.
                IF ( printi ) WRITE( out, 2340 ) prefix,                       &
                   inform%SBLS_inform%norm_residual, control%stop_d
                CYCLE
              END IF
              IF ( first_iteration ) norm_d =                                  &
                SQRT( ABS( DOT_PRODUCT( GRAD_l, VECTOR( : dims%c_e ) ) ) )
            END IF
            GO TO 100

!  Form the product of VECTOR with H

          CASE ( 3, 7 )

            IF ( printw ) WRITE( out,                                          &
             "( A, ' ............ matrix-vector product .......... ' )" ) prefix

!  Compute the largest error in the residuals

            HX( : dims%x_free ) = zero
            HX( dims%x_free + 1 : n ) =                                        &
              BARRIER_X * VECTOR( dims%x_free + 1 : n )
            CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr,        &
                         VECTOR( : n ), '+' )
            HX( dims%c_s : dims%c_e ) =                                        &
              BARRIER_C * VECTOR( dims%c_s : dims%c_e )

!  Print the residuals if required 

            IF ( printm .AND. m > 0 ) THEN
              HX( dims%y_s : dims%y_i - 1 ) = zero
              HX( dims%y_i : dims%y_e ) =                                      &
                - SCALE_C * VECTOR( dims%c_s : dims%c_e )
              CALL QPB_AX( m, HX( dims%y_s : dims%y_e ), m, a_ne, A_val,       &
                            A_col, A_ptr, n, VECTOR( : n ), '+ ' )
              WRITE( out, "( A, ' constraint residual ', ES12.4 )" )           &
                prefix, MAXVAL( ABS( HX( dims%y_s : dims%y_e ) ) )
            END IF
            VECTOR( : dims%c_e ) = HX( : dims%c_e )
          
            GO TO 100

!  Reform the initial residual

          CASE ( 5 )
          
            IF ( printw ) WRITE( out,                                          &
             "( A, ' ................. restarting ................ ' )" ) prefix

            R( : dims%c_e ) = GRAD_L
            GO TO 100

          END SELECT

!  End of iteration

          IF ( printm ) WRITE( out,                                            &
         "(/, A, '   |           trust-region subproblem solved             |',&
       &   /, A, '   |------------------------------------------------------|',&
       &   / )" ) prefix, prefix

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ............... step computed ...............' )" ) prefix

          CALL CPU_TIME( time_now ) ; CALL CLOCK_TIME( clock_now )
          time_now = time_now - time_record
          IF ( printt ) WRITE( out, "( A, ' solve time = ', F10.2 )" )         &
            prefix, time_now
          inform%time%solve = inform%time%solve + time_now
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

          cg_iter = inform%GLTR_inform%iter
          inform%cg_iter = inform%cg_iter + cg_iter

!  Record the updated Lagrange multipliers

          Y_trial = Y + VECTOR( dims%y_s : dims%y_e )

!  If the overall search direction is unlikely to make a significant
!  impact on the residual, exit

          IF ( inform%GLTR_inform%mnormx <= teneps ) THEN
            inform%status = GALAHAD_error_tiny_step
            inform%GLTR_inform%status = 1
            got_ratio = .FALSE.
            full_iteration = .FALSE.

!  Update the Lagrange multipliers

            Y = Y_trial

!  Update the dual variables so that z_l > 0 and z_u > 0

!  Problem variables

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

!  Slack variables

            DO i = dims%c_l_start, dims%c_l_end
              Y_l( i ) = mu / DIST_C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              Y_u( i ) = - mu / DIST_C_u( i )
            END DO       

            EXIT
          END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Find the largest feasible step for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

          IF ( printw ) WRITE( out,                                            &
            "( A, ' .............. get steplength  ..............' )" ) prefix

          step_max = infinity

!  Problem variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( S( i ) < zero ) step_max = MIN( step_max, - X( i ) / S( i ) ) 
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            IF ( S( i ) < zero )                                               &
              step_max = MIN( step_max, - DIST_X_l( i ) / S( i ) ) 
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            IF ( S( i ) > zero )                                               &
              step_max = MIN( step_max, DIST_X_u( i ) / S( i ) ) 
          END DO 

          DO i = dims%x_u_end + 1, n
            IF ( S( i ) > zero ) step_max = MIN( step_max, - X( i ) / S( i ) ) 
          END DO 

!  Slack variables:

          DO i = dims%c_l_start, dims%c_l_end
            IF ( S( dims%c_b + i ) < zero )                                    &
              step_max = MIN( step_max, - DIST_C_l( i ) / S( dims%c_b + i ) ) 
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            IF ( S( dims%c_b + i ) > zero )                                    &
              step_max = MIN( step_max, DIST_C_u( i ) / S( dims%c_b + i ) ) 
          END DO 

!  Test whether the new point is acceptable
!  ----------------------------------------   

!  If x + s - X_l >= zeta(x - l) and X_u - x - s >= zeta(u - x),
!  compute phi at x+s and define the ratio
!       (phi(x+s) - phi(x))/m(x+s) - m(x))

          got_ratio = .TRUE.

          DO i = dims%x_free + 1, dims%x_l_start - 1

!  Check that x + s >= zeta(x)

            IF ( X( i ) + S( i ) < zeta * X( i ) ) THEN
               got_ratio = .FALSE.
               EXIT
            END IF
          END DO

          IF ( got_ratio ) THEN
            DO i = dims%x_l_start, dims%x_l_end

!  Calculate x + s - x_l

              DIST_X_l_trial( i ) = DIST_X_l( i ) + S( i ) 

!  Check that x + s - x_l >= zeta(x - x_l)

              IF ( DIST_X_l_trial( i ) < zeta * DIST_X_l( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%x_u_start, dims%x_u_end

!  Calculate x_u - x - s

              DIST_X_u_trial( i ) = DIST_X_u( i ) - S( i ) 

!  Check that x_u - x - s >= zeta(x_u - x)

              IF ( DIST_X_u_trial( i ) < zeta * DIST_X_u( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%x_u_end + 1, n

!  Check that x + s <= zeta(x)

              IF ( X( i ) + S( i ) > zeta * X( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

!  Do the same for the slack variables

!  Calculate x + s - l

          IF ( got_ratio ) THEN
            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l_trial( i ) = DIST_C_l( i ) + S( dims%c_b + i ) 


!  Check that x + s - X_l >= zeta(x - l)

              IF ( DIST_C_l_trial( i ) < zeta * DIST_C_l( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%c_u_start, dims%c_u_end

!  Calculate X_u - x - s

              DIST_C_u_trial( i ) = DIST_C_u( i ) - S( dims%c_b + i ) 

!  Check that X_u - x - s >= zeta(u - x)

              IF ( DIST_C_u_trial( i ) < zeta * DIST_C_u( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

!  The new point is feasible. Calculate the new value of the objective function

          IF ( got_ratio ) THEN
      
!  Calculate x + s

            X_trial = X + S( dims%x_s : dims%x_e )
            nbacts = 0
            GLTR_control%boundary = .FALSE.
            inform%GLTR_inform%status = 1

!  Compute the product between H and x + s

            HX( : n ) = zero
            CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr,        &
                        X_trial, '+' )

!  Now evaluate the objective function ...

            obj_trial = half * DOT_PRODUCT( X_trial, HX( : n ) ) +             &
                               DOT_PRODUCT( X_trial, G )

!  ... and the barrier function

            phi_trial = QPB_barrier_value( dims, n, obj_trial, X_trial,        &
                                            DIST_X_l_trial, DIST_X_u_trial,    &
                                            DIST_C_l_trial, DIST_C_u_trial,    &
                                            mu )

!  Compute the ratio of actual to predicted reduction over the current iteration

            ared   = ( phi - phi_trial ) + MAX( one, ABS( phi ) ) * teneps
            prered = - model + MAX( one, ABS( phi ) ) * teneps
            IF ( ABS( ared ) < teneps .AND. ABS( phi ) > teneps )              &
               ared = prered
            ratio = ared / prered

            IF ( printt ) WRITE( out,                                          &
               "( A, ' ared, pred ', 2ES12.4 ) " ) prefix, ared, prered
          ELSE
            nbacts = 1 ; GLTR_control%boundary = .TRUE.
          END IF

!  If ratio >= eta_1, the iteration was successful

          IF ( got_ratio .AND. ratio >= eta_1 ) THEN
            successful_iteration = .TRUE.
            IF ( printw ) WRITE( out,                                          &
            "( A, ' ............... successful step ...............' )" ) prefix

!  Compute new dual variables so that z_l > 0 and z_u < 0
!  ------------------------------------------------------

!  Problem variables:

!  For the lower bounds ...
 
            DO i = dims%x_free + 1, dims%x_l_start - 1
              Z_l( i ) = MAX( z_min, ( mu - Z_l( i ) * S( i ) ) / X( i ),      &
                                nu_1 * MIN( one, Z_l( i ), mu / X_trial( i ) ) )
            END DO

            DO i = dims%x_l_start, dims%x_l_end
              Z_l( i ) = MAX( z_min,                                           &
                              ( mu - Z_l( i ) * S( i ) ) / DIST_X_l( i ),      &
                           nu_1 * MIN( one,                                    &
                                       Z_l( i ), mu / DIST_X_l_trial( i ) ) )
            END DO
     
!  .... and now the upper bounds

            DO i = dims%x_u_start, dims%x_u_end
              Z_u( i ) = MIN( - z_min,                                         &
                              ( - mu + Z_u( i ) * S( i ) ) / DIST_X_u( i ),    &
                           nu_1 * MAX( - one,                                  &
                                       Z_u( i ), - mu / DIST_X_u_trial( i ) ) )
            END DO       

            DO i = dims%x_u_end + 1, n
              Z_u( i ) = MIN( - z_min, ( mu - Z_u( i ) * S( i ) ) / X( i ),    &
                           nu_1 * MAX( - one, Z_u( i ), mu / X_trial( i ) ) )
            END DO       

!  Now replace x by x + s

            X = X_trial
            DIST_X_l = DIST_X_l_trial
            DIST_X_u = DIST_X_u_trial

!  Slack variables:

!  For the lower bounds ...
 
            DO i = dims%c_l_start, dims%c_l_end
              Y_l( i ) = MAX( z_min, ( mu - Y_l( i ) * S( dims%c_b + i ) )     &
                                / DIST_C_l( i ), nu_1 * MIN( one,              &
                                       Y_l( i ), mu / DIST_C_l_trial( i ) ) )
            END DO
     
!  .... and now the upper bounds

            DO i = dims%c_u_start, dims%c_u_end
              Y_u( i ) = MIN( - z_min, ( - mu + Y_u( i ) * S( dims%c_b + i ) ) &
                                / DIST_C_u( i ), nu_1 * MAX( - one,            &
                                       Y_u( i ), - mu / DIST_C_u_trial( i ) ) )
            END DO       

!  Now replace c by c + s

            C = C + S( dims%c_s : dims%c_e )
            DIST_C_l = DIST_C_l_trial
            DIST_C_u = DIST_C_u_trial
            Y = Y_trial

!  Update the objective and barrier function values

            phi  = phi_trial
            obj  = obj_trial
            inform%obj = obj + f

!  Compute the derivatives of the objective function

            GRAD = HX( : n ) + G
            refact = .TRUE.
          ELSE
            successful_iteration = .FALSE.
            IF ( printw ) WRITE( out,                                          &
            "( A, ' .............. unsuccessful step ..............' )" ) prefix

!  As we have not achieved a sufficient reduction, use the Nocedal-Yuan 
!  technique to achieve one. To do this, perform a line-search to find
!  a point x + alpha s which sufficiently reduces the barrier function

!  Find the largest feasible step for x

            alpha = one
   
!  Problem variables:

            DO i = dims%x_free + 1, dims%x_l_start - 1
              IF( S( i ) < zero ) alpha = MIN( alpha, - X( i ) / S( i ) )
            END DO
            DO i = dims%x_l_start, dims%x_l_end
              IF( S( i ) < zero ) alpha = MIN( alpha, - DIST_X_l( i ) / S( i ) )
            END DO

            DO i = dims%x_u_start, dims%x_u_end
              IF( S( i ) > zero ) alpha = MIN( alpha, DIST_X_u( i ) / S( i ) ) 
            END DO 
            DO i = dims%x_u_end + 1, n
              IF( S( i ) > zero ) alpha = MIN( alpha, - X( i ) / S( i ) ) 
            END DO 

!  Slack variables:

            DO i = dims%c_l_start, dims%c_l_end
              IF ( S( dims%c_b + i ) < zero )                                  &
                alpha = MIN( alpha, - DIST_C_l( i ) / S( dims%c_b + i ) )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              IF ( S( dims%c_b + i ) > zero )                                  &
                alpha = MIN( alpha, DIST_C_u( i ) / S( dims%c_b + i ) ) 
            END DO 

!  A step of no larger than zeta of the distance to the nearest 
!  bound will be attempted

            alpha = ( one - zeta ) * alpha
!           alpha = half * alpha

! ::::::::::::::::::::::::::::::::::::::::::::
!  Record the slope along the search direction
! ::::::::::::::::::::::::::::::::::::::::::::

!  Compute the product between H and s

            HX( : n ) = zero
            CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr,        &
                          S( : n ), '+' )

!  Compute the slope and curvature of the objective function

            obj_slope = DOT_PRODUCT( GRAD( : n ), S( : n ) ) 
            obj_curv  = half * DOT_PRODUCT( HX( : n ), S( : n ) ) 

!  Compute the slope of the barrier function

            phi_slope = DOT_PRODUCT( GRAD_L, S ) 

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Use a backtracking line-search, starting from alpha_v
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::

            IF ( printt ) WRITE( out, "( /, A, ' value = ', ES12.4,            &
           &    ' slope = ', ES12.4 )" ) prefix, phi, phi_slope
            IF ( printw ) WRITE( out,                                          &
              "( A, ' ................ linesearch ................ ' )" ) prefix
            IF ( printt ) WRITE( out, "( /, A, '       ***  Linesearch    ',   &
           & ' step      trial value           model value ' )" ) prefix
            DO

!  The barrier value should  be smaller than a linear model

              phi_model = phi + alpha * eta * phi_slope

!  Calculate the distances of x + alpha s from the bounds

              X_trial = X + alpha * S( : n )

!  Problem variables:

              DO i = dims%x_l_start, dims%x_l_end
                DIST_X_l_trial( i ) = DIST_X_l( i ) + alpha * S( i ) 
              END DO 
              DO i = dims%x_u_start, dims%x_u_end
                DIST_X_u_trial( i ) = DIST_X_u( i ) - alpha * S( i ) 
              END DO 
        
!  Slack variables:

              DO i = dims%c_l_start, dims%c_l_end
                DIST_C_l_trial( i ) = DIST_C_l( i ) + alpha * S( dims%c_b + i )
              END DO 
              DO i = dims%c_u_start, dims%c_u_end
                DIST_C_u_trial( i ) = DIST_C_u( i ) - alpha * S( dims%c_b + i )
              END DO 
        
              obj_trial = obj + alpha * ( obj_slope + alpha * obj_curv )
              phi_trial = QPB_barrier_value( dims, n, obj_trial, X_trial,      &
                                              DIST_X_l_trial, DIST_X_u_trial,  &
                                              DIST_C_l_trial, DIST_C_u_trial,  &
                                              mu )

              IF ( printt ) WRITE( out, "( A, '                   ',           &
             &      ES12.4, 2ES22.14 )" ) prefix, alpha, phi_trial, phi_model 

!  Check to see if the Armijo criterion is satisfied. If not, halve the 
!  steplength

              IF ( phi_trial <= phi_model ) EXIT
              alpha = half * alpha ;  nbacts = nbacts + 1 
              IF ( alpha < epsmch ) THEN
                 inform%status = GALAHAD_error_tiny_step
                 GO TO 500
              END IF
            END DO
            inform%nbacts = inform%nbacts + nbacts
            refact = .TRUE.

!  Update the objective and barrier function values

            phi  = phi_trial ; obj  = obj_trial ; inform%obj = obj + f

!  Update the primal variables and derivatives of the objective function

            X = X + alpha * S( dims%x_s : dims%x_e )
            C = C + alpha * S( dims%c_s : dims%c_e )
            GRAD = GRAD + alpha * HX( : n ) 

!  Update the distances to the bounds

            DIST_X_l = DIST_X_l_trial ; DIST_X_u = DIST_X_u_trial
            DIST_C_l = DIST_C_l_trial ; DIST_C_u = DIST_C_u_trial

          END IF
!         write(6,"( ' X, C ', 2ES12.4 )" ) X(2), C(79)

          IF ( printm .AND. m > 0 ) THEN 
            R( : dims%c_equality ) = - C_l( : dims%c_equality )
            R( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
            CALL QPB_AX( m, R( : m ), m, a_ne, A_val, A_col, A_ptr, n, X, '+ ' )
            WRITE( out, "( /, A, '  Constraint residual ', ES14.6,             &
           &                  '  objective value ', ES14.6  )" )               &
                   prefix, MAXVAL( ABS( R( : m ) ) ), inform%obj
          END IF

!  Update the trust-region radius
!  ------------------------------   

          old_radius = radius

!  If ratio >= eta_2, possibly increase radius

          IF ( got_ratio .AND. ratio >= eta_2 ) THEN
            radius = MIN( radius_max,                                          &
                          radius * MIN( two, half * ( step_max + one ) ),      &
                          MAX( radius, two * inform%GLTR_inform%mnormx ) )

!  If eta_2 > ratio >= eta_1, replace radius by something 
!  in [gamma_2 radius, radius] 

          ELSE IF ( got_ratio .AND. ratio >= eta_1 ) THEN
            
!  If eta_1 > ratio, replace radius by something
!  in [gamma_1 radius, gamma_2 radius] 

          ELSE
            delta = one
  410       CONTINUE
            inform%GLTR_inform%status = 1
            delta = half * delta
            IF ( alpha <= delta ) GO TO 410
            radius = delta * radius
          END IF

        END DO inner  ! end of inner iteration loop

!  ======================
!  End of Inner iteration
!  ======================

!  Compare the primal and primal-dual dual variables

        IF ( printd ) THEN
          DO i = dims%x_free + 1, dims%x_l_start - 1
            WRITE( out," ( A, ' z_l, dz_l = ', 2ES12.4 )" ) prefix,            &
              Z_l( i ),    mu / X( i ) - Z_l( i )
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            WRITE( out," ( A, ' z_l, dz_l = ', 2ES12.4 )" ) prefix,            &
              Z_l( i ),    mu / DIST_X_l( i ) - Z_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            WRITE( out," ( A, ' z_u, dz_u = ', 2ES12.4 )" ) prefix,            &
              Z_u( i ),  - mu / DIST_X_u( i ) - Z_u( i ) 
          END DO
          DO i = dims%x_u_end + 1, n
            WRITE( out," ( A, ' z_u, dz_u = ', 2ES12.4 )" ) prefix,            &
              Z_u( i ),  mu / X( i ) - Z_u( i ) 
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            WRITE( out," ( A, ' y_l, dy_l, y = ', 3ES12.4 )" ) prefix,         &
              Y_l( i ),    mu / DIST_C_l( i ) - Y_l( i ), Y( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            WRITE( out," ( A, ' y_u, dy_u, y = ', 3ES12.4 )" ) prefix,         &
              Y_u( i ),  - mu / DIST_C_u( i ) - Y_u( i ), Y( i )
          END DO
        END IF

  500   CONTINUE

!!$!  Problem variables
!!$
!!$          VECTOR( : dims%x_free ) = GRAD( : dims%x_free )
!!$
!!$          DO i = dims%x_free + 1, dims%x_u_start - 1
!!$            VECTOR( i ) = GRAD( i ) - Z_l( i )
!!$          END DO
!!$
!!$          DO i = dims%x_u_start, dims%x_l_end
!!$            VECTOR( i ) = GRAD( i ) - Z_l( i ) - Z_u( i )
!!$          END DO
!!$
!!$          DO i = dims%x_l_end + 1, n
!!$            VECTOR( i ) = GRAD( i ) - Z_u( i )
!!$          END DO
!!$
!!$!  Slack variables
!!$
!!$          DO i = dims%c_l_start, dims%c_u_start - 1
!!$            VECTOR( dims%c_b + i ) = - Y_l( i )
!!$          END DO
!!$
!!$          DO i = dims%c_u_start, dims%c_l_end
!!$            VECTOR( dims%c_b + i ) = - Y_l( i ) - Y_u( i )
!!$          END DO
!!$
!!$          DO i = dims%c_l_end + 1, dims%c_u_end
!!$            VECTOR( dims%c_b + i ) = - Y_u( i )
!!$          END DO
!!$
!!$          CALL QPB_AX( n, VECTOR( : n ), m, a_ne, A_val,                    &
!!$                        A_col, A_ptr, m, Y, '-T' )
!!$          VECTOR( dims%c_s : dims%c_e ) = VECTOR( dims%c_s : dims%c_e ) +   &
!!$                                SCALE_C * Y( dims%c_l_start : dims%c_u_end )
!!$          VECTOR( dims%y_s : dims%y_e ) = zero
!!$
!!$write(6,*) ' res x', MAXVAL( ABS( VECTOR( dims%x_s : dims%x_e ) ) ) 
!!$write(6,*) ' res c', MAXVAL( ABS( VECTOR( dims%c_s : dims%c_e ) ) )

        IF ( get_stat ) THEN

!  Estimate the variable and constraint exit status

          CALL LSQP_indicators( dims, n, m, C_l, C_u, C_last, C,               &
                                DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,       &
                                DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,        &
                                Y_last, Z_last,                                &
                                control%LSQP_control, C_stat = C_stat,         &
                                B_stat = B_stat )

!  Count the number of active constraints/bounds

          IF ( printt )                                                        &
            WRITE( out, "( A, ' indicators: n_active/n, m_active/m ', 4I7 )" ) &
              prefix, COUNT( B_stat /= 0 ), n, COUNT( C_stat /= 0 ), m
        END IF

!  Do we wish to extrapolate on current data values?

        IF ( control%extrapolate > 0 .AND. allow_extrapolate ) THEN

!  If required, record path values. How many values do we currently have
!  and in which order?

          current = MOD( len_hist, hist ) + 1
          len_hist = len_hist + 1

          DO i = MIN( len_hist, hist ) - 1, 1, - 1
            list_hist( i + 1 ) = list_hist( i )
          END DO
          list_hist( 1 ) = current

!  record the current mu

          mu_hist( current ) = mu

!  record the current variables v(mu)

          X_hist( current, 0, : n ) = X( : n )
          C_hist( current, 0, dims%c_l_start : dims%c_u_end ) =                &
            C( dims%c_l_start : dims%c_u_end )
          Y_hist( current, 0, : m ) = Y( : m )
          Y_l_hist ( current, 0, dims%c_l_start : dims%c_l_end ) =             &
            Y_l( dims%c_l_start : dims%c_l_end )
          Y_u_hist ( current, 0, dims%c_u_start : dims%c_u_end ) =             &
            Y_u( dims%c_u_start : dims%c_u_end )
          Z_l_hist( current, 0, dims%x_free + 1 : dims%x_l_end ) =             &
            Z_l( dims%x_free + 1 : dims%x_l_end )
          Z_u_hist ( current, 0, dims%x_u_start : n ) =                        &
            Z_u( dims%x_u_start : n )

!  record the current derivatives of the variables v(mu)

          DO kd = 1, deriv

!  to find the k-th derivatives ( x^k, c^k, y^k, z_l^k, z_u^k, y_l^k, y_u^k ),
!  solve the equations

!   (  H       A^T   -I    -I               ) (  x^k  )   (   0   )
!   (          -I                 -I   -I   ) (  c^k  )   (   0   )
!   (  A   -I                               ) ( -y^k  )   (   0   )
!   (  Z_l         X-X_l                    ) ( z_l^k ) = ( r_l^k )
!   ( -Z_u               X_u-X              ) ( z_u^k )   ( r_u^k )
!   (      Y_l                  C-C_l       ) ( y_l^k )   ( s_l^k )
!   (     -Y_u                        C_u-C ) ( y_u^k )   ( s_u^k )

!  where r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}   (store in z_l^k)
!        r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}   (store in z_u^k)
!        s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}   (store in y_l^k)
!   and  s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}   (store in y_u^k)

!  for k > 1 or ( r_l^1, r_u^1, s_l^1, s_u^1 ) = ( 1, 1, 1, 1 )
!  and b_i^k is the binomial coefficient "k choose i"

!  Compute ( r_l^k, r_u^k, s_l^k, s_u^k )

            IF ( kd > 1 ) THEN
              DO j = dims%x_free + 1, dims%x_l_end
                rhs = zero
                DO i = 1, kd - 1
                  rhs = rhs + BINOMIAL( i, kd ) *                              &
                    X_hist( current, i, j ) * Z_l_hist( current, kd - i, j )
                END DO
                Z_l_hist( current, kd, j ) = - rhs
              END DO

              DO j = dims%x_u_start, n
                rhs = zero
                DO i = 1, kd - 1
                  rhs = rhs + BINOMIAL( i, kd ) *                              &
                    X_hist( current, i, j ) * Z_u_hist( current, kd - i, j )
                END DO
                Z_u_hist( current, kd, j ) = rhs
              END DO

              DO j = dims%c_l_start, dims%c_l_end
                rhs = zero
                DO i = 1, kd - 1
                  rhs = rhs + BINOMIAL( i, kd ) *                              &
                    C_hist( current, i, j ) * Y_l_hist( current, kd - i, j )
                END DO
                Y_l_hist( current, kd, j ) = - rhs
              END DO

              DO j = dims%c_u_start, dims%c_u_end 
                rhs = zero
                DO i = 1, kd - 1
                  rhs = rhs + BINOMIAL( i, kd ) *                              &
                    C_hist( current, i, j ) * Y_u_hist( current, kd - i, j )
                END DO
                Y_u_hist( current, kd, j ) = rhs
              END DO
            ELSE
              Z_l_hist( current, 1, dims%x_free + 1 : dims%x_l_end ) = one
              Z_u_hist( current, 1, dims%x_u_start : n ) = - one
              Y_l_hist( current, 1, dims%c_l_start : dims%c_l_end ) = one
              Y_u_hist( current, 1, dims%c_u_start : dims%c_u_end ) = - one
            END IF

!  On writing 
!        z_l^k = (X-X_l)^-1 [ r_l^k - Z_l x^k ]
!        z_u^k = (X_u-X)^-1 [ r_u^k + Z_u x^k ]
!        y_l^k = (C-C_l)^-1 [ s_l^k - Y_l c^k ]
!   and  y_u^k = (C_u-C)^-1 [ s_u^k + Y_u c^k ], we find

!  ( H + (X-X_l)^-1 Z_l                 A^T ) (  x^k )   ( (X-X_l)^-1 r_l^k + )
!  (   - (X_u-X)^-1 Z_u                     ) (      )   ( (X_u-X)^-1 r_u^k   ) 
!  (                     (C-C_l)^-1 Y_l  -I ) (  c^k ) = ( (C-C_l)^-1 s_l^k + )
!  (                   - (C_u-C)^-1 Y_u     ) (      )   ( (C_u-C)^-1 s_u^k   ) 
!  (       A                -I              ) ( -y^k )   (          0         )

!  Build the right-hand side vector

            VECTOR( : dims%x_free ) = zero
            DO i = dims%x_free + 1, dims%x_l_start - 1
              VECTOR( i ) = Z_l_hist( current, kd, i ) / X( i )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              VECTOR( i ) = Z_l_hist( current, kd, i ) / DIST_X_l( i )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              VECTOR( i ) = Z_l_hist( current, kd, i ) / DIST_X_l( i )         &
                            + Z_u_hist( current, kd, i ) / DIST_X_u( i )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              VECTOR( i ) = Z_u_hist( current, kd, i ) / DIST_X_u( i )
            END DO
            DO i = dims%x_u_end + 1, n
              VECTOR( i ) = - Z_u_hist( current, kd, i ) / X( i )
            END DO

            DO i = dims%c_l_start, dims%c_u_start - 1
              VECTOR( dims%c_b + i ) =                                         &
                Y_l_hist( current, kd, i ) / DIST_C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              VECTOR( dims%c_b + i ) =                                         &
                Y_l_hist( current, kd, i ) / DIST_C_l( i )                     &
                + Y_u_hist( current, kd, i ) / DIST_C_u( i )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              VECTOR( dims%c_b + i ) =                                         &
                Y_u_hist( current, kd, i ) / DIST_C_u( i )
            END DO
            VECTOR( dims%y_s : dims%y_e ) = zero

!  solve the system to obtain ( x^k, c^k, y^k )

            res_fail = MAX( point1 * MAXVAL( ABS( VECTOR ) ), res_large )
            SBLS_control%affine = .TRUE.
            CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls, sbls_data,    &
                             SBLS_control, inform%SBLS_inform, VECTOR )

!  record the k-th derivative of the current variables v(mu)

            X_hist( current, kd, : n ) = VECTOR( dims%x_s : dims%x_e )
            C_hist( current, kd, dims%c_l_start : dims%c_u_end ) =             &
              VECTOR( dims%c_s : dims%c_e )
            Y_hist( current, kd, : m ) = - VECTOR( dims%y_s : dims%y_e )
            DO i = dims%x_free + 1, dims%x_l_start - 1
              Z_l_hist( current, kd, i ) = ( Z_l_hist( current, kd, i ) -      &
                 Z_l( i ) * X_hist( current, kd, i ) ) / X( i )
            END DO

            DO i = dims%x_l_start, dims%x_l_end
              Z_l_hist( current, kd, i ) = ( Z_l_hist( current, kd, i ) -      &
                 Z_l( i ) * X_hist( current, kd, i ) ) / DIST_X_l( i )
            END DO

            DO i = dims%x_u_start, dims%x_u_end
              Z_u_hist( current, kd, i ) = ( Z_u_hist( current, kd, i ) +      &
                 Z_u( i ) * X_hist( current, kd, i ) ) / DIST_X_u( i )
            END DO       

            DO i = dims%x_u_end + 1, n
              Z_u_hist( current, kd, i ) = - ( Z_u_hist( current, kd, i ) +    &
                 Z_u( i ) * X_hist( current, kd, i ) ) / X( i )
            END DO       

            DO i = dims%c_l_start, dims%c_l_end
              Y_l_hist( current, kd, i ) = ( Y_l_hist( current, kd, i ) -      &
                Y_l( i ) * C_hist( current, kd, i ) ) / DIST_C_l( i )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              Y_u_hist( current, kd, i ) = ( Y_u_hist( current, kd, i ) +      &
                Y_u( i ) * C_hist( current, kd, i ) ) / DIST_C_u( i )
            END DO       

!!$!  Problem variables
!!$
!!$          VECTOR( : n ) = zero
!!$          CALL QPB_HX( dims, n, VECTOR( : n ), h_ne, H_val, H_col, H_ptr,   &
!!$                       X_hist( current, kd, : n ), '+' )
!!$
!!$          CALL QPB_AX( n, VECTOR( : n ), m, a_ne, A_val,                    &
!!$                        A_col, A_ptr, m, Y_hist( current, kd, : m ), '-T' )
!!$
!!$
!!$          DO i = dims%x_free + 1, dims%x_u_start - 1
!!$            VECTOR( i ) = VECTOR( i ) -  Z_l_hist( current, kd, i )
!!$          END DO
!!$
!!$          DO i = dims%x_u_start, dims%x_l_end
!!$            VECTOR( i ) = VECTOR( i ) - Z_l_hist( current, kd, i )          &
!!$                                      - Z_u_hist( current, kd, i )
!!$          END DO
!!$
!!$!  Slack variables
!!$
!!$          DO i = dims%x_l_end + 1, n
!!$            VECTOR( i ) = VECTOR( i ) -  Z_u_hist( current, kd, i )
!!$          END DO
!!$
!!$          DO i = dims%c_l_start, dims%c_u_start - 1
!!$            VECTOR( dims%c_b + i ) = - Y_l_hist( current, kd, i )
!!$          END DO
!!$
!!$          DO i = dims%c_u_start, dims%c_l_end
!!$            VECTOR( dims%c_b + i ) = - Y_l_hist( current, kd, i )           &
!!$                                     - Y_u_hist( current, kd, i )
!!$          END DO
!!$
!!$          DO i = dims%c_l_end + 1, dims%c_u_end
!!$            VECTOR( dims%c_b + i ) = - Y_u_hist( current, kd, i )
!!$          END DO
!!$          VECTOR( dims%c_s : dims%c_e ) = VECTOR( dims%c_s : dims%c_e ) +   &
!!$               SCALE_C * Y_hist( current, kd, dims%c_l_start : dims%c_u_end )
!!$
!!$!  dual variables
!!$
!!$          VECTOR( dims%y_s : dims%y_e ) = zero
!!$          VECTOR( dims%y_i : dims%y_e ) =                                   &
!!$            - SCALE_C * C_hist( current, kd, dims%c_l_start : dims%c_u_end )
!!$          CALL QPB_AX( m, VECTOR( dims%y_s : dims%y_e ), m, a_ne, A_val,    &
!!$                       A_col, A_ptr, n, X_hist( current, kd, : n ), '+ ' )

!!$x_norm = MAXVAL( ABS( X_hist( current, kd, : n ) ) )
!!$write(6,*) '||x||', x_norm
!!$c_norm = MAXVAL( ABS( C_hist( current, kd, dims%c_l_start : dims%c_u_end ) ))
!!$write(6,*) '||c||', c_norm
!!$y_norm= MAX( MAXVAL( ABS( Y_hist( current, kd, : m ) ) ),          &
!!$  MAXVAL( ABS( Y_l_hist( current, kd, dims%c_l_start : dims%c_l_end ) ) ),  &
!!$  MAXVAL( ABS( Y_u_hist( current, kd, dims%c_u_start : dims%c_u_end ) ) ) )
!!$write(6,*) '||y||', y_norm
!!$z_norm = MAX(                                                       &
!!$  MAXVAL( ABS( Z_l_hist( current, kd, dims%x_free + 1 : dims%x_l_end ) ) ), &
!!$  MAXVAL( ABS( Z_u_hist( current, kd, dims%x_u_start : n ) ) ) )
!!$write(6,*) '||z||', z_norm

!!$write(6,*) 'deriv', kd, ' res x',                                           &
!!$  MAXVAL( ABS( VECTOR( dims%x_s : dims%x_e ) ) ) /                          &
!!$  MAX( one, x_norm, y_norm, z_norm )
!!$write(6,*) 'deriv', kd, ' res c',                                           &
!!$  MAXVAL( ABS( VECTOR( dims%c_s : dims%c_e ) ) ) /                          &
!!$  MAX( one, y_norm )
!!$write(6,*) 'deriv', kd, ' res y',                                           &
!!$  MAXVAL( ABS( VECTOR( dims%y_s : dims%y_e ) ) ) /                          &
!!$  MAX( one, x_norm, c_norm )

          END DO

!  compute the Puiseux series that interpolates the "window" of function and 
!  first deriv derivatives at the past hist points on the trajectory. 
!  Record the trajectory parameters, mu

          n_degree = 0
          DO l = 1, MIN( len_hist, hist )
            j = list_hist( l )
            n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
            fit_mu( n_degree + 1 : n_degree_p ) = mu_hist( j )
            n_degree = n_degree_p
            IF ( n_degree >= order + 1 ) EXIT
          END DO

!  consider each variable x separately

          DO i = 1, n

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p )                               &
                = X_hist( j, 0 : n_degree_p - n_degree - 1, i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              X_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              X_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  consider each dual variable z_l separately

          DO i = dims%x_free + 1, dims%x_l_end

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series for z_l ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                Z_l_hist( j, 0 : n_degree_p - n_degree - 1, i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              Z_l_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              Z_l_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  likewise, consider each dual variable z_u separately

          DO i = dims%x_u_start, n

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series for z_u ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                Z_u_hist( j, 0 : n_degree_p - n_degree - 1, i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              Z_u_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              Z_u_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  consider each variable c separately

          DO i = dims%c_l_start, dims%c_u_end

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                C_hist( j, 0 : n_degree_p - n_degree - 1 , i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              C_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              C_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  consider each Lagrange multiplier y separately

          DO i = 1, m

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                Y_hist( j, 0 : n_degree_p - n_degree - 1 , i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_coef( 0 : n_degree - 1, i ),   &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  consider each dual variable y_l separately

          DO i = dims%c_l_start, dims%c_l_end

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series for y_l ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                Y_l_hist( j, 0 : n_degree_p - n_degree - 1, i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_l_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_l_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )
            END IF
          END DO

!  likewise, consider each dual variable y_u separately

          DO i = dims%c_u_start, dims%c_u_end

!  record the function and derivative values over the window and 
!  determine the coefficients of the Puiseux series for y_u ...

            n_degree = 0
            DO l = 1, MIN( len_hist, hist )
              j = list_hist( l )
              n_degree_p = MIN( n_degree + deriv + 1, order + 1 )
              fit_f( n_degree + 1 : n_degree_p ) =                             &
                Y_u_hist( j, 0 : n_degree_p - n_degree - 1, i )
              n_degree = n_degree_p
              IF ( n_degree >= order + 1 ) EXIT
            END DO

            IF ( control%puiseux ) THEN
              CALL FIT_puiseux_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_u_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

!  ... or the Hermite (Taylor) series

            ELSE
              CALL FIT_hermite_interpolation( n_degree, fit_mu, fit_f,         &
                                              Y_u_coef( 0 : n_degree - 1, i ), &
                                              FIT_data, control%FIT_control,   &
                                              inform%FIT_inform )

            END IF
          END DO

!  print the extrapolated complementarity at the solution ...

!!$      IF ( control%puiseux ) THEN
!!$        mu_extrapolate = SQRT( sigma * mu )
!!$       ELSE
!!$        mu_extrapolate = sigma * mu
!!$      END IF
!!$
!!$      CALL QPB_new_comp( mu_extrapolate, dims, n, m, X_l, X_u, C_l, C_u,       &
!!$                         X_coef, C_coef, Z_l_coef, Z_u_coef,                   &
!!$                         Y_l_coef, Y_u_coef,                                   &
!!$                         order, n_degree, comp_min, comp_max )
!!$
!!$      IF ( comp_min > zero ) THEN
!!$        write(6,"( ' min, target, max = ', 3ES12.4 )") &
!!$           SQRT(comp_min), mu_extrapolate, SQRT(comp_max)
!!$
!!$        IF ( control%puiseux ) THEN
!!$          mu_extrapolate = SQRT( 0.1_wp * sigma * mu )
!!$         ELSE
!!$          mu_extrapolate = 0.1_wp * sigma * mu
!!$        END IF
!!$
!!$        CALL QPB_new_comp( mu_extrapolate, dims, n, m, X_l, X_u, C_l, C_u,     &
!!$                           X_coef, C_coef, Z_l_coef, Z_u_coef,                 &
!!$                           Y_l_coef, Y_u_coef,                                 &
!!$                           order, n_degree, comp_min, comp_max )
!!$
!!$        IF ( comp_min > zero ) THEN
!!$          write(6,"( ' min, target, max = ', 3ES12.4 )") &
!!$             SQRT(comp_min), mu_extrapolate, SQRT(comp_max)
!!$
!!$          IF ( control%puiseux ) THEN
!!$            mu_extrapolate = SQRT( 0.01_wp * sigma * mu )
!!$           ELSE
!!$            mu_extrapolate = 0.01_wp * sigma * mu
!!$          END IF
!!$
!!$          mu_extrapolate = SQRT( 0.01 * sigma * mu )
!!$          CALL QPB_new_comp( mu_extrapolate, dims, n, m, X_l, X_u, C_l, C_u,   &
!!$                             X_coef, C_coef, Z_l_coef, Z_u_coef,               &
!!$                             Y_l_coef, Y_u_coef,                               &
!!$                             order, n_degree, comp_min, comp_max )
!!$          IF ( comp_min > zero ) THEN
!!$            write(6,"( ' min, target, max = ', 3ES12.4 )") &
!!$               SQRT(comp_min), mu_extrapolate, SQRT(comp_max)
!!$          ELSE
!!$            IF ( control%puiseux ) THEN
!!$              mu_extrapolate = SQRT( 0.1_wp * sigma * mu )
!!$             ELSE
!!$              mu_extrapolate = 0.1_wp * sigma * mu
!!$            END IF
!!$          END IF
!!$        ELSE
!!$          IF ( control%puiseux ) THEN
!!$            mu_extrapolate = SQRT( sigma * mu )
!!$           ELSE
!!$            mu_extrapolate = sigma * mu
!!$          END IF
!!$        END IF
!!$      END IF

          IF ( printd ) THEN
            IF ( control%puiseux ) THEN
               series = 'puiseux'
             ELSE
               series = 'taylor '
            END IF
            WRITE( out, "( A, ' complementarity when mu = ', ES12.4 )" )       &
              prefix, zero

            DO i = 1, dims%x_free
              WRITE( out, 2501 ) prefix, TRIM( series ), i, X_coef( 0, i )
            END DO
            DO i = dims%x_free + 1, dims%x_l_end
              WRITE( out, 2500 ) prefix, 'lower', TRIM( series ), i,           &
                Z_l_coef( 0, i ) * (  X_coef( 0, i ) - X_l( i ) ),             &
                   (  X_coef( 0, i ) - X_l( i ) ), Z_l_coef( 0, i )
            END DO
            DO i = dims%x_u_start, n
              WRITE( out, 2500 ) prefix, 'upper', TRIM( series ), i,           &
                - Z_u_coef( 0, i ) * (  X_u( i ) - X_coef( 0, i ) ),           &
                 (  X_u( i ) - X_coef( 0, i ) ), Z_u_coef( 0, i )
            END DO
            DO i = 1, m
              WRITE( out, 2503 ) prefix, TRIM( series ), i, Y_coef( 0, i )
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              WRITE( out, 2502 ) prefix, 'lower', TRIM( series ), i,           &
                Y_l_coef( 0, i ) * (  C_coef( 0, i ) - C_l( i ) ),             &
                 (  C_coef( 0, i ) - C_l( i ) ), Y_l_coef( 0, i )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              WRITE( out, 2502 ) prefix, 'upper', TRIM( series ), i,           &
                - Y_u_coef( 0, i ) * (  C_u( i ) - C_coef( 0, i ) ),           &
                 (  C_u( i ) - C_coef( 0, i ) ), Y_u_coef( 0, i )
            END DO
          END IF

!  ... and at the next point on the path

          IF ( printd ) THEN
            IF ( control%puiseux ) THEN
              series = 'puiseux'
              mu_extrapolate = SQRT( sigma * mu )
             ELSE
              series = 'taylor '
              mu_extrapolate = sigma * mu
            END IF
            WRITE( out, "( ' complementarity when mu = ', ES12.4 )" ) sigma * mu

            DO i = 1, dims%x_free
              WRITE( out, 2501 ) prefix, TRIM( series ), i,                    &
                 FIT_evaluate_polynomial( n_degree,                            &
                  X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            END DO
            DO i = dims%x_free + 1, dims%x_l_end
              xx = FIT_evaluate_polynomial( n_degree,                          &
                      X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              zz = FIT_evaluate_polynomial( n_degree,                          &
                      Z_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              WRITE( out, 2500 ) prefix, 'lower', TRIM( series ), i,           &
                  zz * ( xx - X_l( i ) ), ( xx - X_l( i ) ), zz
            END DO
            DO i = dims%x_u_start, n
              xx = FIT_evaluate_polynomial( n_degree,                          &
                      X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              zz = FIT_evaluate_polynomial( n_degree,                          &
                      Z_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              WRITE( out, 2500 ) prefix, 'upper', TRIM( series ), i,           &
                zz * ( xx - X_u( i ) ), ( X_u( i ) - xx ), zz
            END DO
            DO i = 1, m
              WRITE( out, 2503 ) prefix, TRIM( series ), i,                    &
                FIT_evaluate_polynomial( n_degree,                             &
                     Y_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              xx = FIT_evaluate_polynomial( n_degree,                          &
                      C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              zz = FIT_evaluate_polynomial( n_degree,                          &
                      Y_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              WRITE( out, 2502 ) prefix, 'lower', TRIM( series ), i,           &
                 zz * ( xx - C_l( i ) ), ( xx - C_l( i ) ), zz
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              xx = FIT_evaluate_polynomial( n_degree,                          &
                      C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              zz = FIT_evaluate_polynomial( n_degree,                          &
                      Y_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
              WRITE( out, 2502 ) prefix, 'upper', TRIM( series ), i,           &
                 zz * ( xx - C_u( i ) ), ( C_u( i ) - xx ), zz
            END DO
          END IF

!  If mu is small enough, take one last lunge at the solution using the
!  Puiseux/Taylor estimate

          IF ( mu < 1.01_wp * control%mu_min ) THEN

            IF ( control%puiseux ) THEN
               series = 'Puiseux'
             ELSE
               series = 'Taylor '
            END IF

            norm_c = zero
            DO i = dims%x_free + 1, dims%x_l_end
              norm_c = MAX( norm_c,                                            &
                ABS( Z_l_coef( 0, i ) * (  X_coef( 0, i ) - X_l( i ) ) ) )
            END DO
            DO i = dims%x_u_start, n
              norm_c = MAX( norm_c,                                            &
                ABS( Z_u_coef( 0, i ) * (  X_coef( 0, i ) - X_u( i ) ) ) )
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              norm_c = MAX( norm_c,                                            &
                ABS( Y_l_coef( 0, i ) * (  C_coef( 0, i ) - C_l( i ) ) ) )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              norm_c = MAX( norm_c,                                            &
                ABS( Y_u_coef( 0, i ) * (  C_coef( 0, i ) - C_u( i ) ) ) )
            END DO

!  if the lunge is unsuccessful, retore the existing solution

            IF ( norm_c <= control%stop_c ) THEN
              IF ( printi ) WRITE( out, "( /, A, ' -:-:- final ', A,           &
               &     ' lunge for the solution -:-:-' )" ) prefix, TRIM( series )

!  primal variables:

              X( 1 : n ) = X_coef( 0, 1 : n )
              Z_l( dims%x_free + 1 : dims%x_l_end ) =                          &
                Z_l_coef( 0, dims%x_free + 1 : dims%x_l_end )
              Z_u( dims%x_u_start : n ) = Z_u_coef( 0, dims%x_u_start : n  )

!  slack variables:

              Y( 1 : m ) = Y_coef( 0, 1 : m )
              C( dims%c_l_start : dims%c_u_end ) =                             &
                C_coef( 0, dims%c_l_start : dims%c_u_end )
              Y_l( dims%c_l_start : dims%c_l_end ) =                           &
                Y_l_coef( 0, dims%c_l_start : dims%c_l_end )
              Y_u( dims%c_u_start : dims%c_u_end ) =                           &
                Y_u_coef( 0, dims%c_u_start : dims%c_u_end )

!  recompute the distances to the bounds 

              DO i = dims%x_l_start, dims%x_l_end
                DIST_X_l( i ) = X( i ) - X_l( i )
              END DO
              DO i = dims%x_u_start, dims%x_u_end
                DIST_X_u( i ) = X_u( i ) - X( i )
              END DO
              DO i = dims%c_l_start, dims%c_l_end
                DIST_C_l( i ) = C( i ) - C_l( i )
              END DO
              DO i = dims%c_u_start, dims%c_u_end
                DIST_C_u( i ) = C_u( i ) - C( i )
              END DO

!  compute the complementary slackness

              norm_c = zero

!  primal variables:

              DO i = dims%x_free + 1, dims%x_l_start - 1
                IF ( ABS( X( i ) ) < remote )                                  &
                  norm_c = MAX( norm_c, ABS( X( i ) * Z_l( i ) ) )
              END DO

              DO i = dims%x_l_start, dims%x_u_start - 1
                IF ( ABS( DIST_X_l( i ) ) < remote )                           &
                  norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ) )
              END DO

              DO i = dims%x_u_start, dims%x_l_end
                IF ( MIN( ABS( DIST_X_l( i ) ),                                &
                          ABS( DIST_X_u( i ) ) ) < remote )                    &
                  norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ),     &
                                        ABS( - DIST_X_u( i ) * Z_u( i ) ) )
              END DO

              DO i = dims%x_l_end + 1, dims%x_u_end
                IF ( ABS( DIST_X_u( i ) ) < remote )                           &
                  norm_c = MAX( norm_c, ABS( - DIST_X_u( i ) * Z_u( i ) ) )
              END DO

              DO i = dims%x_u_end + 1, n
                IF ( ABS( X( i ) ) < remote )                                  &
                  norm_c = MAX( norm_c, ABS( X( i ) * Z_u( i ) ) )
              END DO

!  slack variables:

              DO i = dims%c_l_start, dims%c_u_start - 1
                IF ( ABS( DIST_C_l( i ) ) < remote )                           &
                  norm_c = MAX( norm_c, ABS( DIST_C_l( i ) * Y_l( i ) ) )
              END DO

              DO i = dims%c_u_start, dims%c_l_end
                IF ( MIN( ABS( DIST_C_l( i ) ),                                &
                          ABS( DIST_C_u( i ) ) ) < remote )                    &
                  norm_c = MAX( norm_c, ABS(   DIST_C_l( i ) * Y_l( i ) ),     &
                                        ABS( - DIST_C_u( i ) * Y_u( i ) ) )
              END DO

              DO i = dims%c_l_end + 1, dims%c_u_end
                IF ( ABS( DIST_C_u( i ) ) < remote )                           &
                  norm_c = MAX( norm_c, ABS( - DIST_C_u( i ) * Y_u( i ) ) )
              END DO

!  compute the product between H and x

              HX( : n ) = zero
              CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr,      &
                           X, '+' )

!  now, calculate the value .... 

              obj = half * DOT_PRODUCT( X, HX( : n ) ) + DOT_PRODUCT( X, G )
              inform%obj = obj + f
  
!  ... and gradient of the objective function

              GRAD = HX( : n ) + G
              refact = .TRUE.

            ELSE
              IF ( printi ) WRITE( out, "( /, A, ' -:-:- ', A,                 &
             &  ' lunge for the solution abandoned -:-:-', //, A,              &
             &  ' achived and required complementarity = ', 2ES12.4 )" )       &
               prefix, TRIM( series ), prefix, norm_c, control%stop_c
            END IF
          END IF
        END IF

!  Check for termination of the outer iteration

        IF ( norm_c <= control%stop_c .AND.                                    &
             norm_d <= control%stop_d ) THEN
          norm_c = zero

!  primal variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_l( i ) ) )
          END DO

          DO i = dims%x_l_start, dims%x_u_start - 1
            IF ( ABS( DIST_X_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ) )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            IF ( MIN( ABS( DIST_X_l( i ) ), ABS( DIST_X_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ),         &
                                    ABS( - DIST_X_u( i ) * Z_u( i ) ) )
          END DO

          DO i = dims%x_l_end + 1, dims%x_u_end
            IF ( ABS( DIST_X_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_X_u( i ) * Z_u( i ) ) )
          END DO

          DO i = dims%x_u_end + 1, n
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_u( i ) ) )
          END DO

!  slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            IF ( ABS( DIST_C_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( DIST_C_l( i ) * Y_l( i ) ) )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            IF ( MIN( ABS( DIST_C_l( i ) ), ABS( DIST_C_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_C_l( i ) * Y_l( i ) ),         &
                                    ABS( - DIST_C_u( i ) * Y_u( i ) ) )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            IF ( ABS( DIST_C_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_C_u( i ) * Y_u( i ) ) )
          END DO

          IF ( printt ) WRITE( out, 2160 )                                     &
            prefix, p_min, p_max, prefix, d_min, d_max

          IF ( norm_c <= control%stop_c ) THEN
            inform%status = GALAHAD_ok
            EXIT
          END IF
        END IF

        IF ( printm .AND. m > 0 ) THEN 
          R( : dims%c_equality ) = - C_l( : dims%c_equality )
          R( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
          CALL QPB_AX( m, R( : m ), m, a_ne, A_val, A_col, A_ptr, n, X, '+ ' )
          WRITE( out, "( /, A, '  Constraint residual ', ES14.6,               &
         &                  '  objective value ', ES14.6  )" )                 &
                 prefix, MAXVAL( ABS( R( : m ) ) ), inform%obj
        END IF

!  Update penalty parameter, mu

        old_mu = mu ; mu = sigma * mu
        IF (  old_mu > zero ) THEN
          zeta = sigma * mu / old_mu 
        ELSE
          zeta = zero
        END IF

        IF ( control%extrapolate > 1 .AND. allow_extrapolate ) THEN

!  try to use the Puiseux and Taylor expansions to find the next starting point

          revert = .FALSE.
          IF ( control%puiseux ) THEN
            mu_extrapolate = SQRT( mu )
          ELSE
            mu_extrapolate = mu
          END IF

!  primal and dual variables

          DO i = 1, dims%x_free
            X( i ) = FIT_evaluate_polynomial( n_degree,                        &
                           X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
          END DO
          DO i = dims%x_free + 1, dims%x_u_start - 1
            x_p = FIT_evaluate_polynomial( n_degree,                           &
                        X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            z_l_p = FIT_evaluate_polynomial( n_degree,                         &
                        Z_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( x_p >  X_l( i ) .AND. z_l_p > zero ) THEN
              X( i ) = x_p
              Z_l( i ) = z_l_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            x_p = FIT_evaluate_polynomial( n_degree,                           &
                        X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            z_l_p = FIT_evaluate_polynomial( n_degree,                         &
                        Z_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            z_u_p = FIT_evaluate_polynomial( n_degree,                         &
                        Z_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( x_p >  X_l( i ) .AND. z_l_p > zero .AND.                      &
                 x_p <  X_u( i ) .AND. z_u_p < zero ) THEN
              X( i ) = x_p
              Z_l( i ) = z_l_p
              Z_u( i ) = z_u_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO
          DO i = dims%x_l_end + 1, n
            x_p = FIT_evaluate_polynomial( n_degree,                           &
                        X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            z_u_p = FIT_evaluate_polynomial( n_degree,                         &
                        Z_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( x_p <  X_u( i ) .AND. z_u_p < zero ) THEN
              X( i ) = x_p
              Z_u( i ) = z_u_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO

!  slack variables and Lagrange multipliers

          DO i = 1, m
            Y( i ) = FIT_evaluate_polynomial( n_degree,                        &
                           Y_coef( 0 : n_degree - 1, i ), mu_extrapolate )
          END DO

          DO i = dims%c_l_start, dims%c_u_start - 1
            c_p = FIT_evaluate_polynomial( n_degree,                           &
                        C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            y_l_p = FIT_evaluate_polynomial( n_degree,                         &
                        Y_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( c_p >  C_l( i ) .AND. y_l_p > zero ) THEN
              C( i ) = c_p
              Y_l( i ) = y_l_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            c_p = FIT_evaluate_polynomial( n_degree,                           &
                        C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            y_l_p = FIT_evaluate_polynomial( n_degree,                         &
                        Y_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            y_u_p = FIT_evaluate_polynomial( n_degree,                         &
                        Y_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( c_p >  C_l( i ) .AND. y_l_p > zero .AND.                      &
                 c_p <  C_u( i ) .AND. y_u_p < zero ) THEN
              C( i ) = c_p
              Y_l( i ) = y_l_p
              Y_u( i ) = y_u_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            c_p = FIT_evaluate_polynomial( n_degree,                           &
                        C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            y_u_p = FIT_evaluate_polynomial( n_degree,                         &
                        Y_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
            IF ( c_p <  C_u( i ) .AND. y_u_p < zero ) THEN
              C( i ) = c_p
              Y_u( i ) = y_u_p
            ELSE
              revert = .TRUE. ; GO TO 510
            END IF
          END DO

!  if the Puiseux/Taylor point is infeasible, revert to the current point

  510     CONTINUE
          IF ( revert ) THEN
            IF ( printi ) THEN
              IF ( control%puiseux ) THEN
                WRITE( out, "( /, A,                                           &
               & ' -:-:-  unsuccessful Puiseux extrapolation  -:-:-' )" ) prefix
              ELSE
                WRITE( out, "( /, A,                                           &
               & ' -:-:-  unsuccessful Taylor extrapolation  -:-:-' )" ) prefix
              END IF
            END IF
            X( : n ) = X_hist( current, 0, : n ) 
            C( dims%c_l_start : dims%c_u_end ) =                               &
              C_hist( current, 0, dims%c_l_start : dims%c_u_end ) 
            Y( : m ) = Y_hist( current, 0, : m ) 
            Y_l( dims%c_l_start : dims%c_l_end ) =                             &
              Y_l_hist ( current, 0, dims%c_l_start : dims%c_l_end ) 
            Y_u( dims%c_u_start : dims%c_u_end ) =                             &
              Y_u_hist ( current, 0, dims%c_u_start : dims%c_u_end ) 
            Z_l( dims%x_free + 1 : dims%x_l_end ) =                            &
              Z_l_hist( current, 0, dims%x_free + 1 : dims%x_l_end ) 
            Z_u( dims%x_u_start : dims%x_u_end ) =                             &
              Z_u_hist ( current, 0, dims%x_u_start : dims%x_u_end ) 

!  use the Puiseux/Taylor point

          ELSE
            IF ( printi ) THEN
              IF ( control%puiseux ) THEN
                WRITE( out, "( /, A,                                           &
               & ' -:-:-   successful Puiseux extrapolation   -:-:-' )" ) prefix
              ELSE
                WRITE( out, "( /, A,                                           &
               & ' -:-:-   successful Taylor extrapolation   -:-:-' )" ) prefix
              END IF
            END IF
            DO i = dims%x_l_start, dims%x_l_end
              DIST_X_l( i ) = X( i ) - X_l( i )
            END DO
            DO i = dims%x_u_start, dims%x_u_end
              DIST_X_u( i ) = X_u( i ) - X( i )
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l( i ) = C( i ) - C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              DIST_C_u( i ) = C_u( i ) - C( i )
            END DO

!  compute the product between H and x

            HX( : n ) = zero
            CALL QPB_HX( dims, n, HX( : n ), h_ne, H_val, H_col, H_ptr, X, '+' )

!  now, calculate the value .... 

            obj = half * DOT_PRODUCT( X, HX( : n ) ) + DOT_PRODUCT( X, G )
            inform%obj = obj + f
  
!  ... and gradient of the objective function

            GRAD = HX( : n ) + G
            refact = .TRUE.

!  compute the value of the barrier function, phi

            phi = QPB_barrier_value( dims, n, obj, X, DIST_X_l, DIST_X_u,      &
                                     DIST_C_l, DIST_C_u, mu )
            IF ( printm .AND. m > 0 ) THEN 
              R( : dims%c_equality ) = - C_l( : dims%c_equality )
              R( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
              CALL QPB_AX( m, R( : m ), m, a_ne, A_val, A_col, A_ptr, n,       &
                           X, '+ ' )
              WRITE( out, "( /, A, '  Constraint residual ', ES14.6,           &
             &                  '  objective value ', ES14.6  )" )             &
                     prefix, MAXVAL( ABS( R( : m ) ) ), inform%obj
            END IF
          END IF
        END IF

!  record X and C

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

!  record the norms of the various components

         x_norm = MAXVAL( ABS( X( : n ) ) )
         c_norm = MAXVAL( ABS( C( dims%c_l_start : dims%c_u_end ) ) )
         y_norm = MAX( MAXVAL( ABS( Y( : m ) ) ),                              &
                       MAXVAL( ABS( Y_l( dims%c_l_start : dims%c_l_end ) ) ),  &
                       MAXVAL( ABS( Y_u( dims%c_u_start : dims%c_u_end ) ) ) )
         z_norm = MAX( MAXVAL( ABS( Z_l( dims%x_free + 1 : dims%x_l_end ) ) ), &
                       MAXVAL( ABS( Z_u( dims%x_u_start : n ) ) ) )
         IF ( printt ) WRITE( out, "( A, ' ||x,c,y,z|| = ', 4ES12.4 )" )       &
            prefix, x_norm, c_norm, y_norm, z_norm

!  Update convergence tolerances, theta_c, theta_d and theta_e

        IF ( mu < 1.01_wp * control%mu_min .AND.                               &
             control%extrapolate > 0 .AND. allow_extrapolate) THEN
!         theta_c = ten * epsmch
!         theta_d = ten * epsmch
          theta_c = ( ten ** 3 ) * epsmch 
          theta_d = ( ten ** 2 ) * epsmch * MAX( x_norm, y_norm, z_norm )
        ELSE
          theta_c = MIN( theta_min,                                            &
                      MAX( control%theta_c * mu ** control%beta,               &
                           point99 * control%stop_c ) )
          theta_d = MIN( theta_min,                                            &
                      MAX( control%theta_d * mu ** control%beta,               &
                           point99 * control%stop_d ) )
        END IF

!  Recompute the value of the barrier function, phi

        phi = QPB_barrier_value( dims, n, obj, X, DIST_X_l, DIST_X_u,          &
                                  DIST_C_l, DIST_C_u, mu )

        full_iteration = .FALSE.

        IF ( ( inform%status == GALAHAD_error_ill_conditioned .OR.             &
               inform%status == GALAHAD_error_tiny_step ) .AND.                &
               old_mu > point1 * control%stop_c ) THEN
          inform%status = GALAHAD_ok
          start_major = .TRUE.
          got_ratio = .FALSE.
          successful_iteration = .FALSE.
        END IF
        IF ( inform%status /= GALAHAD_ok ) EXIT

!  ======================
!  End of outer iteration
!  ======================

      END DO   ! end of outer iteration loop

  600 CONTINUE 

!  Print details of the solution obtained

      IF ( printi ) WRITE( out,                                                &
            "(  /, A, '  Final objective function value =', ES22.14,           &
           &    /, A, '  Total number of iterations = ', I0,                   &
           &    /, A, '  Total number of c.g. its = ', I0 )" )                 &
       prefix, inform%obj, prefix, inform%iter, prefix, inform%cg_iter

        IF ( get_stat ) THEN

!  Estimate the variable and constraint exit status

          CALL LSQP_indicators( dims, n, m, C_l, C_u, C_last, C,               &
                                DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,       &
                                DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,        &
                                Y_last, Z_last,                                &
                                control%LSQP_control, C_stat = C_stat,         &
                                B_stat = B_stat )

!  Count the number of active constraints/bounds

          IF ( printt )                                                        &
            WRITE( out, "( A, ' indicators: n_active/n, m_active/m ', 4I7 )" ) &
              prefix, COUNT( B_stat /= 0 ), n, COUNT( C_stat /= 0 ), m
        END IF

!  If required, make the solution exactly complementary

!  Problem variables

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

!  slack variables

        DO i = dims%c_l_start, dims%c_l_end
          IF ( ABS( Y_l( i ) ) < ABS( DIST_C_l( i ) ) ) THEN 
            Y_l( i ) = zero 
          ELSE 
            C( i ) = C_l( i ) 
          END IF 
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          IF ( ABS( Y_u( i ) ) < ABS( DIST_C_u( i ) ) ) THEN 
            Y_u( i ) = zero 
          ELSE 
            C( i ) = C_u( i ) 
          END IF 
        END DO
      END IF 

  700 CONTINUE

!  Set the dual variables

      IF ( set_z ) THEN
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
      END IF

!  Unscale the constraint bounds

      IF ( scaled_c ) THEN
        DO i = dims%c_l_start, dims%c_l_end
          C_l( i ) = C_l( i ) * SCALE_C( i )
        END DO
  
        DO i = dims%c_u_start, dims%c_u_end
          C_u( i ) = C_u( i ) * SCALE_C( i )
        END DO
      END IF

!  compute the constraint residuals

      C_RES = zero
      CALL QPB_AX( m, C_RES, m, a_ne, A_val, A_col, A_ptr, n, X, '+ ')
      IF ( printi .AND. m > 0 )                                                &
        WRITE( out, "( A, '  Constraint residual =', ES12.4 )" ) prefix,       &
             MAX( zero, MAXVAL( ABS( C_l( : dims%c_equality) -                 &
                                     C_RES(: dims%c_equality ) ) ),            &
                        MAXVAL( C_l(  dims%c_l_start : dims%c_l_end ) -        &
                                C_RES(  dims%c_l_start : dims%c_l_end ) ),     &
                        MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -       &
                                C_u( dims%c_u_start : dims%c_u_end ) ) )     

!  compute the complementarity

      norm_c = zero
      DO i = dims%x_free + 1, dims%x_l_end
        norm_c = MAX( norm_c, ABS( Z_l( i ) * (  X( i ) - X_l( i ) ) ) )
      END DO
      DO i = dims%x_u_start, n
        norm_c = MAX( norm_c, ABS( Z_u( i ) * (  X( i ) - X_u( i ) ) ) )
      END DO
      DO i = dims%c_l_start, dims%c_l_end
        norm_c = MAX( norm_c, ABS( Y_l( i ) * (  C( i ) - C_l( i ) ) ) )
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        norm_c = MAX( norm_c, ABS( Y_u( i ) * (  C( i ) - C_u( i ) ) ) )
      END DO

      IF ( printi )                                                            &
        WRITE( out, "( A, '  Complementarity =', ES12.4 )" ) prefix, norm_c

!  compute the dual feasibility, GRAD - A^T y - z_l - z_u

!write(6,*) ' g ', MAXVAL( ABS( GRAD( : n ) ) )
!write(6,*) ' z_l ', MAXVAL( ABS( Z_l( dims%x_free + 1 : dims%x_l_end ) ) )
!write(6,*) ' z_u ', MAXVAL( ABS( Z_u( dims%x_u_start : n ) ) )
!write(6,*) ' y_l ', MAXVAL( ABS( Y_l( dims%c_l_start : dims%c_l_end ) ) )
!write(6,*) ' y_u ', MAXVAL( ABS( Y_u( dims%c_u_start : dims%c_u_end ) ) )

      VECTOR( : dims%x_free ) = GRAD( : dims%x_free )

      DO i = dims%x_free + 1, dims%x_u_start - 1
        VECTOR( i ) = GRAD( i ) - Z_l( i )
      END DO

      DO i = dims%x_u_start, dims%x_l_end
        VECTOR( i ) = GRAD( i ) - Z_l( i ) - Z_u( i )
      END DO

      DO i = dims%x_l_end + 1, n
        VECTOR( i ) = GRAD( i ) - Z_u( i )
      END DO

      DO i = dims%c_l_start, dims%c_u_start - 1
        VECTOR( dims%c_b + i ) = - Y_l( i )
      END DO

      DO i = dims%c_u_start, dims%c_l_end
        VECTOR( dims%c_b + i ) = - Y_l( i ) - Y_u( i )
      END DO

      DO i = dims%c_l_end + 1, dims%c_u_end
        VECTOR( dims%c_b + i ) = - Y_u( i )
      END DO

      CALL QPB_AX( n, VECTOR( : n ), m, a_ne, A_val,                       &
                    A_col, A_ptr, m, Y, '-T' )
      VECTOR( dims%c_s : dims%c_e ) = VECTOR( dims%c_s : dims%c_e ) +      &
                            SCALE_C * Y( dims%c_l_start : dims%c_u_end )

!do i = 1, dims%c_e
!write(6,*) i, VECTOR( i )
!end do
      norm_d_alt = SQRT( ABS( SUM( VECTOR( : dims%c_e ) ** 2 ) ) )

      IF ( printi )                                                            &
        WRITE( out, "( A, '  Dual feasibility =', ES12.4 )" ) prefix, norm_d_alt

!  If necessary, print warning messages

  810 CONTINUE
      IF ( printi ) then

        SELECT CASE( inform%status )
          CASE( GALAHAD_error_restrictions ) ; WRITE( out, "(/, A,             &
         &  '  Warning - input paramters incorrect' )" ) prefix
          CASE( GALAHAD_error_deallocate ) ; WRITE( out, "(/, A,               &
         &  '  Warning - deallocation error' )"  ) prefix
          CASE( GALAHAD_error_bad_bounds ) ; WRITE( out, "(/, A,               &
         &  '  Warning - the constraints are inconsistent' )" ) prefix
          CASE( GALAHAD_error_primal_infeasible ) ; WRITE( out, "(/, A,        &
         & '  Warning - the constraints appear to be inconsistent' )" ) prefix
          CASE( GALAHAD_error_factorization ) ; WRITE( out, "(/, A,            &
         & '  Warning - factorization failure' )" ) prefix
          CASE( GALAHAD_error_ill_conditioned ) ; WRITE( out, "(/, A,          &
         & '  Warning - no further progress possible' )" ) prefix
          CASE( GALAHAD_error_tiny_step ) ; WRITE( out, "(/, A,                &
         & '  Warning - step too small to make further progress' )" ) 
          CASE( GALAHAD_error_max_iterations ) ; WRITE( out, "(/, A,           &
         & '  Warning - iteration bound exceeded' )"  ) prefix
          CASE( GALAHAD_error_unbounded ) ; WRITE( out, "(/, A,                &
         & '  Warning - objective unbounded below' )" ) prefix
        END SELECT

        IF ( auto ) WRITE( out, 2400 ) prefix
        SELECT CASE( SBLS_control%preconditioner )
          CASE( 1 ) ; WRITE( out, "( A, '  Identity Hessian ' )" ) prefix
          CASE( 2 ) ; WRITE( out, "( A, '  Full Hessian ' )" ) prefix
          CASE( 3 ) ; WRITE( out, "( A, '  Diagonal Hessian ' )" ) prefix
          CASE( 4 ) ; WRITE( out,                                              &
             "( A, '  Band (semi-bandwidth ', I0, ') Hessian ' )" ) prefix,    &
                SBLS_control%semi_bandwidth
          CASE( 5 ) ; WRITE( out, "( A, '  Barrier Hessian ' )" ) prefix
        END SELECT

        IF ( SBLS_control%factorization == 0 .OR.                              &
             SBLS_control%factorization == 1 ) THEN
          WRITE( control%out, "( A, '  Schur-complement factorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" )                               &
            prefix, SBLS_control%SLS_control%relative_pivot_tolerance
        ELSE
          WRITE( control%out, "( A, '  Augmented system factorization used ',  &
         &       '(pivot tol =', ES9.2, ')' )" )                               &
            prefix, SBLS_control%SLS_control%relative_pivot_tolerance
        END IF

      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving QPB_solve_main ' )" ) prefix

  920 CONTINUE 
      RETURN  

!  Non-executable statements

 2000 FORMAT( /, A, ' Iter   d-feas com-slk    obj     ratio   radius',        &
             ' nbacts cgits      time' ) 
 2010 FORMAT( A, I5, A1, 2ES8.1, ES9.1, A1, '    -   ', ES8.1, 1X,             &
            '     -     -', 0P, F10.2 ) 
 2020 FORMAT( A, I5, A1, 2ES8.1, ES9.1, A1, 2ES8.1, A1, 2I6, 0P, F10.2 ) 
 2030 FORMAT( A, I5, A1, 2ES8.1, ES9.1, A1, '    -   ', ES8.1, A1, 2I6,        &
              0P, F10.2 )
 2100 FORMAT( A, A, 7ES10.2, /, ( 10X, 7ES10.2 ) ) 
 2130 FORMAT( /, A, ' ', 33( '=-' ), '=',                                      &
              /, A, '  mu = ', ES12.4, '  theta_c = ', ES12.4,                 &
                 '  theta_d = ', ES12.4,                                       &
              /, A, ' ', 33( '=-' ), '=' )
 2160 FORMAT( /, A, ' min/max primal = ', 2ES12.4,                             &
              /, A, ' min/max dual   = ', 2ES12.4 )
 2180 FORMAT( A, A6, /, ( 8ES10.2 ) )
 2190 FORMAT( A, A6, /, ( 4( 2I5, ES10.2 ) ) )
 2340 FORMAT( /, A, '  norm of residual', ES12.4, ' for barrier',              &
               ' factorization exceeds ', ES12.4 )
 2400 FORMAT( A, '  Automatic preconditioner ' )
 2500 FORMAT( A, 1X, A, 1X, A, 1X, I6, 'x cs', ES12.4, ' pr, du', 2ES18.10 )
 2501 FORMAT( A, 6X, 1X, A, 1X, I6, 'x   ', 12X, ' pr    ', ES18.10 )
 2502 FORMAT( A, 1X, A, 1X, A, 1X, I6, 'c cs', ES12.4, ' pr, du', 2ES18.10 )
 2503 FORMAT( A, 6X, 1X, A, 1X, I6, 'y   ', 12X, ' pr    ', ES18.10 )

!  End of QPB_solve_main

      END SUBROUTINE QPB_solve_main

!-*-*-*-*-*-*-   Q P B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE QPB_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine QPB_initialize
!   control see Subroutine QPB_initialize
!   inform  see Subroutine QPB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPB_control_type ), INTENT( IN ) :: control        
      TYPE ( QPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by LSQP

      CALL LSQP_terminate( data, control%LSQP_control, inform%LSQP_inform )
      IF ( inform%LSQP_inform%status /= GALAHAD_ok )                           &
        inform%status = inform%LSQP_inform%status 
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated by FDC

      CALL FDC_terminate( data%FDC_data, control%FDC_control,                  &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FDC_inform%status 
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate GLTR internal arrays

      CALL GLTR_terminate( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )
      IF ( inform%GLTR_inform%status /= GALAHAD_ok )                           &
        inform%status = inform%GLTR_inform%status 
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate FIT internal arrays

      CALL FIT_terminate( data%FIT_data, control%FIT_control,                  &
                           inform%FIT_inform )
      IF ( inform%FIT_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FIT_inform%status 
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

      CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,           &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'qpb: data%X_fixed'
      CALL SPACE_dealloc_array( data%X_fixed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%C_fixed'
      CALL SPACE_dealloc_array( data%C_fixed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Index_X_fixed'
      CALL SPACE_dealloc_array( data%Index_X_fixed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Index_C_fixed'
      CALL SPACE_dealloc_array( data%Index_C_fixed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%GRAD'
      CALL SPACE_dealloc_array( data%GRAD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%X_trial'
      CALL SPACE_dealloc_array( data%X_trial,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%X0'
      CALL SPACE_dealloc_array( data%X0,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%GRAD_X_phi'
      CALL SPACE_dealloc_array( data%GRAD_X_phi,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%GRAD_C_phi'
      CALL SPACE_dealloc_array( data%GRAD_C_phi,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%X_coef'
      CALL SPACE_dealloc_array( data%X_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%C_coef'
      CALL SPACE_dealloc_array( data%C_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_coef'
      CALL SPACE_dealloc_array( data%Y_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_l_coef'
      CALL SPACE_dealloc_array( data%Y_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_u_coef'
      CALL SPACE_dealloc_array( data%Y_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Z_l_coef'
      CALL SPACE_dealloc_array( data%Z_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Z_u_coef'
      CALL SPACE_dealloc_array( data%Z_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%BINOMIAL'
      CALL SPACE_dealloc_array( data%BINOMIAL,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%list_hist'
      CALL SPACE_dealloc_array( data%list_hist,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%mu_hist'
      CALL SPACE_dealloc_array( data%mu_hist,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%X_hist'
      CALL SPACE_dealloc_array( data%X_hist,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%C_hist'
      CALL SPACE_dealloc_array( data%C_hist,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_hist'
      CALL SPACE_dealloc_array( data%Y_hist,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_l_hist'
      CALL SPACE_dealloc_array( data%Y_l_hist,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Y_u_hist'
      CALL SPACE_dealloc_array( data%Y_u_hist,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Z_l_hist'
      CALL SPACE_dealloc_array( data%Z_l_hist,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpb: data%Z_u_hist'
      CALL SPACE_dealloc_array( data%Z_u_hist,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine QPB_terminate

      END SUBROUTINE QPB_terminate

!-*-*-*-*-  Q P B _ B A R R I E R _ V A L U E   S U B R O U T I N E   -*-*-*-*

      FUNCTION QPB_barrier_value( dims, n, objf, X, DIST_X_l, DIST_X_u,        &
                                   DIST_C_l, DIST_C_u, mu )
      REAL ( KIND = wp ) QPB_barrier_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the barrier function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: mu 
      REAL ( KIND = wp ), INTENT( IN ) :: objf
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u

!  Local variables

      INTEGER :: i

! Compute the barrier terms

      QPB_barrier_value = zero

!  Problem variables: 

      DO i = dims%x_free + 1, dims%x_l_start - 1
        QPB_barrier_value = QPB_barrier_value + LOG( X( i ) ) 
      END DO 
      DO i = dims%x_l_start, dims%x_l_end
        QPB_barrier_value = QPB_barrier_value + LOG( DIST_X_l( i ) ) 
      END DO 
      DO i = dims%x_u_start, dims%x_u_end
        QPB_barrier_value = QPB_barrier_value + LOG( DIST_X_u( i ) ) 
      END DO 
      DO i = dims%x_u_end + 1, n
        QPB_barrier_value = QPB_barrier_value + LOG( - X( i ) ) 
      END DO 

!  Slack variables: 

      DO i = dims%c_l_start, dims%c_l_end
        QPB_barrier_value = QPB_barrier_value + LOG( DIST_C_l( i ) ) 
      END DO 
      DO i = dims%c_u_start, dims%c_u_end
        QPB_barrier_value = QPB_barrier_value + LOG( DIST_C_u( i ) ) 
      END DO 

!  Form the barrier function

      QPB_barrier_value = objf - mu * QPB_barrier_value

      RETURN  

!  End of QPB_barrier_value

      END FUNCTION QPB_barrier_value
 
!-*-*-*-  Q P B _ F E A S I B L E _ F O R _ B Q P  S U B R O U T I N E  -*-*-*-

      SUBROUTINE QPB_feasible_for_BQP( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute suitable well-centered initial values for the primal and dual
!  variables
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPB_control_type ), INTENT( IN ) :: control        
      TYPE ( QPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: prfeas, dufeas
      LOGICAL :: printi, printd
      CHARACTER ( LEN = 80 ) :: array_name

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering QPB_feasible_for_BQP ' )" )

!  Set initial timing breakdowns

      inform%time%total = 0.0 ; inform%time%analyse = 0.0
      inform%time%factorize = 0.0 ; inform%time%solve = 0.0

!  Initialize counts

      inform%status = GALAHAD_ok ; inform%feasible = .FALSE.
      inform%alloc_status = 0 ; inform%factorization_status = 0
      inform%time%phase1_analyse = 0.0
      inform%time%phase1_factorize = 0.0
      inform%time%phase1_solve = 0.0
      inform%time%analyse = inform%time%phase1_analyse
      inform%time%factorize = inform%time%phase1_factorize
      inform%time%solve = inform%time%phase1_solve
      inform%time%find_dependent = 0.0

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1 

!  Full debugging printing with significant arrays printed

      printd = control%out > 0 .AND. control%print_level >= 5

!  Feasibility tolerances

      prfeas = MAX( control%prfeas, epsmch )
      dufeas = MAX( control%dufeas, epsmch )

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

      array_name = 'qpb: data%X_trial'
      CALL SPACE_resize_array( prob%n, data%X_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_trial'
      CALL SPACE_resize_array( prob%m, data%Y_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%HX'
      CALL SPACE_resize_array( data%dims%v_e, data%HX,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%GRAD_L'
      CALL SPACE_resize_array( data%dims%c_e, data%GRAD_L,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DZ_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%DZ_l,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DIST_X_l'
      CALL SPACE_resize_array( data%dims%x_l_start, data%dims%x_l_end,         &
             data%DIST_X_l,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DZ_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n, data%DZ_u,         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DIST_X_u'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_u_end,         &
             data%DIST_X_u,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%Z_l,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n, data%Z_u,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%BARRIER_X'
      CALL SPACE_resize_array( data%dims%x_free + 1, prob%n, data%BARRIER_X,   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%Y_l,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DY_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DY_l,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DIST_C_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DIST_C_l,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%Y_u,                                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DY_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DY_u,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DIST_C_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DIST_C_u,                                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end, data%C, &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%BARRIER_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%BARRIER_C,                                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%SCALE_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%SCALE_C,                                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%DELTA'
      CALL SPACE_resize_array( data%dims%v_e, data%DELTA,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%RHS'
      CALL SPACE_resize_array( data%dims%v_e, data%RHS,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  =============================
!   Set suitable initial values
!  =============================

      IF ( printd ) THEN
        WRITE( control%out,                                                    &
        "( /, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X, 'z_l', 9X, 'z_u')" )
        DO i = 1, data%dims%x_free
          WRITE( control%out, "( I6, ES12.4, 4( '      -     ') )" )           &
            i, prob%X( i )
        END DO
      END IF

!  The variable is a non-negativity

      DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
        prob%X( i ) = MAX( prob%X( i ), prfeas )
        data%Z_l( i ) = MAX( ABS( prob%Z( i ) ), dufeas )
        IF ( printd ) WRITE( control%out, "( I6, 2ES12.4, '      -     ',      &
       &  ES12.4, '      -     ' )" ) i, prob%X( i ), zero, data%Z_l( i )
      END DO

!  The variable has just a lower bound

      DO i = data%dims%x_l_start, data%dims%x_u_start - 1
        prob%X( i ) = MAX( prob%X( i ), prob%X_l( i ) + prfeas )
        data%Z_l( i ) = MAX( ABS( prob%Z( i ) ), dufeas )
        data%DIST_X_l( i ) = prob%X( i ) - prob%X_l( i )
        IF ( printd ) WRITE( control%out, "( I6, 2ES12.4, '      -     ',      &
       &  ES12.4, '      -     ' )" ) i, prob%X( i ), prob%X_l( i ),           &
                                      data%Z_l( i )
      END DO

!  The variable has both lower and upper bounds

      DO i = data%dims%x_u_start, data%dims%x_l_end

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( prob%X_u( i ) - prob%X_l( i ) <= epsmch ) THEN 
          inform%status = GALAHAD_error_bad_bounds ; RETURN
        END IF
        IF ( prob%X_l( i ) + prfeas >= prob%X_u( i ) - prfeas ) THEN 
          prob%X( i ) = half * ( prob%X_l( i ) + prob%X_u( i ) ) 
        ELSE 
          prob%X( i ) = MIN( MAX( prob%X( i ), prob%X_l( i ) + prfeas ),       &
                             prob%X_u( i ) - prfeas ) 
        END IF 
        data%Z_l( i ) = MAX(   ABS( prob%Z( i ) ),   dufeas )  
        data%Z_u( i ) = MIN( - ABS( prob%Z( i ) ), - dufeas )
        data%DIST_X_l( i ) = prob%X( i )                                       &
          - prob%X_l( i ) ; data%DIST_X_u( i ) = prob%X_u( i ) - prob%X( i )
        IF ( printd ) WRITE( control%out, "( I6, 5ES12.4 )" ) i, prob%X( i ),  &
          prob%X_l( i ), prob%X_u( i ), data%Z_l( i ), data%Z_u( i )
      END DO

!  The variable has just an upper bound

      DO i = data%dims%x_l_end + 1, data%dims%x_u_end
        prob%X( i ) = MIN( prob%X( i ), prob%X_u( i ) - prfeas )
        data%Z_u( i ) = MIN( - ABS( prob%Z( i ) ), - dufeas ) 
        data%DIST_X_u( i ) = prob%X_u( i ) - prob%X( i )
        IF ( printd ) WRITE( control%out, "( I6, ES12.4, '      -     ',       &
       &  ES12.4, '      -     ', ES12.4 )" ) i, prob%X( i ), prob%X_u( i ),   &
                                              data%Z_u( i )
      END DO

!  The variable is a non-positivity

      DO i = data%dims%x_u_end + 1, prob%n
        prob%X( i ) = MIN( prob%X( i ), - prfeas )
        data%Z_u( i ) = MIN( - ABS( prob%Z( i ) ), - dufeas ) 
        IF ( printd ) WRITE( control%out, "( I6, ES12.4, '      -     ',       &
       &  ES12.4, '      -     ',  ES12.4 )" ) i, prob%X( i ), zero,           &
                                               data%Z_u( i )
      END DO

!  Prepare for exit

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPB_feasible_for_BQP ' )" )

      RETURN  

!  Allocation error

  900 CONTINUE 
      inform%status = GALAHAD_error_allocate
      IF ( printi ) WRITE( control%out, 2900 )                                 &
        inform%bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPB_feasible_for_BQP ' )" )

      RETURN  

!  Non-executable statements

 2900 FORMAT( ' ** Message from -QPB_feasible_for_BQP-', /,                    &
              ' Allocation error, for ', A, /, ' status = ', I6 ) 

!  End of QPB_feasible_for_BQP

      END SUBROUTINE QPB_feasible_for_BQP

!!$      SUBROUTINE QPB_new_comp( mu_extrapolate, dims, n, m, X_l, X_u, C_l, C_u, &
!!$                               X_coef, C_coef, Z_l_coef, Z_u_coef,             &
!!$                               Y_l_coef, Y_u_coef,                             &
!!$                               order, n_degree, comp_min, comp_max )
!!$
!!$      INTEGER, INTENT( IN ) :: n, m, order, n_degree
!!$      REAL ( KIND = wp ), INTENT( IN ) :: mu_extrapolate
!!$      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
!!$      REAL ( KIND = wp ), INTENT( IN ),                                        &
!!$        DIMENSION( 0 : order, n ) :: X_coef
!!$      REAL ( KIND = wp ), INTENT( IN ),                                        &
!!$        DIMENSION( 0 : order, dims%c_l_start : dims%c_u_end ) :: C_coef
!!$      REAL ( KIND = wp ), INTENT( IN ),                                       &
!!$        DIMENSION( 0 : order, dims%x_free + 1 : dims%x_l_end ) :: Z_l_coef
!!$      REAL ( KIND = wp ), INTENT( IN ),                                       &
!!$        DIMENSION( 0 : order, dims%x_u_start : n ) :: Z_u_coef
!!$      REAL ( KIND = wp ), INTENT( IN ),                                       &
!!$        DIMENSION( 0 : order, dims%c_l_start : dims%c_l_end ) ::  Y_l_coef
!!$      REAL ( KIND = wp ), INTENT( IN ),                                       &
!!$        DIMENSION( 0 : order, dims%c_u_start : dims%c_u_end ) ::  Y_u_coef
!!$
!!$      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
!!$      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
!!$      REAL ( KIND = wp ), INTENT( OUT ) :: comp_min, comp_max
!!$
!!$      INTEGER :: i
!!$      REAL ( KIND = wp ) :: comp, xx, zz
!!$
!!$      comp_min = infinity
!!$      comp_max = zero
!!$
!!$      DO i = dims%x_free + 1, dims%x_l_end
!!$        xx = FIT_evaluate_polynomial( n_degree,                          &
!!$                X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        zz = FIT_evaluate_polynomial( n_degree,                          &
!!$                Z_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        comp = zz * ( xx - X_l( i ) )
!!$        comp_min = MIN( comp_min, comp )
!!$        comp_max = MAX( comp_max, comp )
!!$      END DO
!!$
!!$      DO i = dims%x_u_start, n
!!$        xx = FIT_evaluate_polynomial( n_degree,                          &
!!$                X_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        zz = FIT_evaluate_polynomial( n_degree,                          &
!!$                Z_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        comp = zz * ( xx - X_u( i ) )
!!$        comp_min = MIN( comp_min, comp )
!!$        comp_max = MAX( comp_max, comp )
!!$      END DO
!!$
!!$      DO i = dims%c_l_start, dims%c_l_end
!!$        xx = FIT_evaluate_polynomial( n_degree,                          &
!!$                C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        zz = FIT_evaluate_polynomial( n_degree,                          &
!!$                Y_l_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        comp = zz * ( xx - C_l( i ) )
!!$        comp_min = MIN( comp_min, comp )
!!$        comp_max = MAX( comp_max, comp )
!!$      END DO
!!$
!!$      DO i = dims%c_u_start, dims%c_u_end
!!$        xx = FIT_evaluate_polynomial( n_degree,                          &
!!$                C_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        zz = FIT_evaluate_polynomial( n_degree,                          &
!!$                Y_u_coef( 0 : n_degree - 1, i ), mu_extrapolate )
!!$        comp = zz * ( xx - C_u( i ) )
!!$        comp_min = MIN( comp_min, comp )
!!$        comp_max = MAX( comp_max, comp )
!!$      END DO
!!$
!!$      END SUBROUTINE QPB_new_comp

!  End of module GALAHAD_QPB

   END MODULE GALAHAD_QPB_double
