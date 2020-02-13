! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ W C P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   started life as part of GALAHAD_CLS ~2000, which mutated into part of
!   GALAHAD LSQP, released pre GALAHAD Version 1.0. April 10th 2001.
!   update extracted and released with GALAHAD Version 2.0.November 1st 2005
!   modified to enable sbls in GALAHAD Version 2.4. May 3rd 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_WCP_double

!     --------------------------------------------------
!     |                                                |
!     | Find a well-centered point within the polytope |
!     |                                                |
!     |          c_l <= A x <= c_u                     |
!     |          x_l <=  x <= x_u                      |
!     |                                                |
!     | using an infeasible-point primal-dual method   |
!     |                                                |
!     --------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_STRING
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPT_double
      USE GALAHAD_QPP_double, WCP_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, WCP_data_type => QPD_data_type,                  &
                              WCP_AX => QPD_AX
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
                               SORT_heapsort_smallest, SORT_inverse_permute
      USE GALAHAD_ROOTS_double
      USE GALAHAD_FDC_double
      USE GALAHAD_SBLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: WCP_initialize, WCP_read_specfile, WCP_solve,                  &
                WCP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,   &
                WCP_data_type, WCP_dims_type

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
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm1 = ten ** ( - 1 )
      REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
      REAL ( KIND = wp ), PARAMETER :: tenm3 = ten ** ( - 3 )
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
      REAL ( KIND = wp ), PARAMETER :: tenm7 = ten ** ( - 7 )
      REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 8 )
      REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: ten2 = ten ** 2
      REAL ( KIND = wp ), PARAMETER :: ten3 = ten ** 3
      REAL ( KIND = wp ), PARAMETER :: ten4 = ten ** 4
      REAL ( KIND = wp ), PARAMETER :: ten5 = ten ** 5
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: mu_tol = ten ** 2
      LOGICAL :: roots_debug = .FALSE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: WCP_control_type

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

!   how to choose the initial point. Possible values are

!      0  the values input in X, shifted to be at least prfeas from
!         their nearest bound, will be used
!      1  the nearest point to the "bound average" 0.5(X_l+X_u) that satisfies
!          the linear constraints will be used

        INTEGER :: initial_point = 0

!   the factorization to be used. Possible values are

!      0  automatic
!      1  Schur-complement factorization
!      2  augmented-system factorization                              (OBSOLETE)

        INTEGER :: factor = 0

!   the maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization                          (OBSOLETE)

        INTEGER :: max_col = 35

!   an initial guess as to the integer workspace required by SBLS     (OBSOLETE)

        INTEGER :: indmin = 10000

!   an initial guess as to the real workspace required by SBLS        (OBSOLETE)

        INTEGER :: valmin = 10000

!   the maximum number of iterative refinements allowed               (OBSOLETE)

        INTEGER :: itref_max = 1

!   the number of iterations for which the overall infeasibility of the
!    problem is not reduced by at least a factor %required_infeas_reduction
!    before the problem is flagged as infeasible (see required_infeas_reduction)

        INTEGER :: infeas_max = 200

!  the strategy used to reduce relaxed constraint bounds. Possible values are

!    0 do not perturb the constraints
!    1 reduce all perturbations by the same amount with linear reduction
!    2 reduce each perturbation as much as possible with linear reduction
!    3 reduce all perturbations by the same amount with superlinear reduction
!    4 reduce each perturbation as much as possible with superlinear reduction

        INTEGER :: perturbation_strategy = 2

!   indicate whether and how much of the input problem should be restored
!     on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

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

!   the target value of the barrier parameter. If mu_target is not positive,
!    it will be reset to an appropriate value

        REAL ( KIND = wp ) :: mu_target = - one

!   the complemtary slackness x_i.z_i will be judged to lie within an
!    acceptable margin around its target value mu as soon as
!      mu_accept_fraction * mu <= x_i.z_i <= ( 1 / mu_accept_fraction ) * mu;
!    the perturbations will be reduced as soon as all of the complemtary
!    slacknesses x_i.z_i lie within acceptable bounds. mu_accept_fraction
!    will be reset to ensure that it lies in the interval (0,1]

        REAL ( KIND = wp ) :: mu_accept_fraction = one

!   the target value of the barrier parameter will be increased by
!    mu_increase_factor for infeasible constraints every time the
!    perturbations are adjusted
!
        REAL ( KIND = wp ) :: mu_increase_factor = two

!   if the overall infeasibility of the problem is not reduced by at least
!    a factor required_infeas_reduction over %infeas_max iterations, the
!    problem is flagged as infeasible (see infeas_max)

        REAL ( KIND = wp ) :: required_infeas_reduction = one - point01

!   any primal or dual variable that is less feasible than implicit_tol will
!     be regarded as defining an implicit constraint

        REAL ( KIND = wp ) :: implicit_tol = epsmch

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

!   the constraint bounds will initially be relaxed by %perturb_start;
!    this perturbation will subsequently be reduced to zero.
!    If perturb_start < 0, the amount by which the bounds are relaxed will
!    be computed automatically

        REAL ( KIND = wp ) :: perturb_start = - one

!   the test for rank defficiency will be to factorize
!    ( alpha_scale I  A^T )
!    (       A          0 )

        REAL ( KIND = wp ) :: alpha_scale = point01

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   the constraint perturbation will be reduced as follows:
!
!    - if the variable lies outside a bound, the corresponding perturbation
!      will be reduced to
!        reduce_perturb_factor * current pertubation
!           + ( 1 - reduce_perturb_factor ) * violation
!    - otherwise, if the variable lies within insufficiently_feasible of its
!      bound the pertubation will be reduced to
!        reduce_perturb_multiplier * current pertubation
!    - otherwise if will be set to zero

        REAL ( KIND = wp ) :: reduce_perturb_factor = 0.25_wp
        REAL ( KIND = wp ) :: reduce_perturb_multiplier = point01
        REAL ( KIND = wp ) :: insufficiently_feasible = epsmch

!   if the maximum constraint pertubation is smaller than
!    perturbation_small and the violation is smaller than implicit_tol, the
!    method will deduce that there is a feasible point but no interior

        REAL ( KIND = wp ) :: perturbation_small = - one

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

!   if %balance_initial_complementarity is .true. the initial complemetarity
!    will be balanced

        LOGICAL :: balance_initial_complementarity = .FALSE.

!  if %use_corrector, a corrector step will be used

        LOGICAL :: use_corrector = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %record_x_status is true, the array inform%X_status will be allocated
!     and the status of the bound constraints will be reported on exit.

        LOGICAL :: record_x_status = .TRUE.

!   if %record_c_status is true, the array inform%C_status will be allocated
!     and the status of the general constraints will be reported on exit.

        LOGICAL :: record_c_status = .TRUE.

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

      TYPE, PUBLIC :: WCP_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorizatio

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

      TYPE, PUBLIC :: WCP_inform_type

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

!  the number of general constraints that lie on (one) of their bounds for all
!   feasible solutions

        INTEGER :: c_implicit = 0

!  the number of variables that lie on (one) of their bounds for all
!   feasible solutions

        INTEGER :: x_implicit = 0

!  the number of Lagrange multipliers for general constraints that lie on
!   (one) of their bounds for all feasible solutions

        INTEGER :: y_implicit = 0

!  the number of dual variables that lie on (one) of their bounds for all
!   feasible solutions

        INTEGER :: z_implicit = 0

!  the value of the objective function at the best estimate of the solution
!   determined by LSQP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  if control%record_x_status is true, %X_status will be allocated
!    and the status of the bound constraints will be reported on exit.
!    In this case, possible values of %X_status(i) are as follows:
!       0  the variable lies between its bounds
!      -1  the variable lies on its lower bound for all feasible points
!       1  the variable lies on its upper bound for all feasible points
!      -2  the variable never lies on its lower bound at any feasible point
!       2  the variable never lies on its upper bound at any feasible point
!       3  the bounds are equal, and the variable takes this value for
!          all feasible points
!      -3  the variable never lies on either bound at any feasible point

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status

!  if control%record_c_status is true, %C_status will be allocated
!    and the status of the general constraints will be reported on exit.
!    In this case, possible values of inform%C_status(i) are as follows:
!       0  the constraint lies between its bounds
!      -1  the constraint lies on its lower bound for all feasible points
!          and may be fixed at this value and removed from the problem
!       1  the constraint lies on its upper bound for all feasible points
!          and may be fixed at this value and removed from the problem
!      -2  the constraint never lies on its lower bound at any feasible point
!          and the bound may be removed from the problem
!       2  the constraint never lies on its upper bound at any feasible point
!          and the bound may be removed from the problem
!       3  the bounds are equal, and the constraint takes this value for
!          all feasible points
!      -3  the constraint never lies on either bound at any feasible point
!          and the constraint may be removed from the problem
!       4  the constraint is implied by the others and may be removed
!          from the problem

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status

!  timings (see above)

        TYPE ( WCP_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

      END TYPE

   CONTAINS

!-*-*-*-*-*-   W C P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE WCP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for WCP. This routine should be called before
!  WCP_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( WCP_data_type ), INTENT( INOUT ) :: data
      TYPE ( WCP_control_type ), INTENT( OUT ) :: control
      TYPE ( WCP_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set real control parameters

      control%stop_p = epsmch ** 0.33
      control%stop_c = epsmch ** 0.33
      control%stop_d = epsmch ** 0.33
      control%implicit_tol = epsmch ** 0.33

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

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of WCP_initialize

      END SUBROUTINE WCP_initialize

!-*-*-*-*-   W C P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE WCP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by WCP_initialize could (roughly)
!  have been set as:

! BEGIN WCP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       1
!  maximum-number-of-iterations                      1000
!  start-print                                       -1
!  stop-print                                        -1
!  initial-point-used                                1
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  initial-integer-workspace                         10000
!  initial-real-workspace                            10000
!  maximum-refinements                               1
!  maximum-poor-iterations-before-infeasible         200
!  perturbation-strategy                             2
!  restore-problem-on-output                         2
!  infinity-value                                    1.0D+10
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  initial-bound-perturbation                        -1.0
!  perturbation-small                                -1.0
!  reduce-perturbation-factor                        0.25
!  reduce-perturbation-multiplier                    0.01
!  insufficiently-feasible-tolerance                 0.01
!  implicit-variable-tolerance                       1.0D-5
!  mininum-initial-primal-feasibility                1.0
!  mininum-initial-dual-feasibility                  1.0
!  target-barrier-parameter                          -1.0
!  target-barrier-accept-fraction                    1.0
!  increase-barrier-parameter-by                     2.0
!  required-infeasibility-reduction                  0.99
!  pivot-tolerance-used                              1.0D-12
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  alpha-scaling-tolerance                           1.0D-2
!  identical-bounds-tolerance                        1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  just-find-feasible-point                          F
!  use-corrector-step                                F
!  balance-initial-complementarity                   F
!  record-x-status                                   T
!  record-c-status                                   T
! END WCP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( WCP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 47
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'WCP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec(  5 )%keyword = 'start-print'
      spec(  6 )%keyword = 'stop-print'
      spec( 32 )%keyword = 'initial-point-used'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 12 )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( 38 )%keyword = 'perturbation-strategy'
      spec( 13 )%keyword = 'restore-problem-on-output'

!  Real key-words

      spec( 14 )%keyword = 'infinity-value'
      spec( 15 )%keyword = 'primal-accuracy-required'
      spec( 16 )%keyword = 'dual-accuracy-required'
      spec( 17 )%keyword = 'complementary-slackness-accuracy-required'
      spec( 18 )%keyword = 'mininum-initial-primal-feasibility'
      spec( 19 )%keyword = 'mininum-initial-dual-feasibility'
      spec( 20 )%keyword = 'target-barrier-parameter'
      spec( 21 )%keyword = 'poor-iteration-tolerance'
      spec( 22 )%keyword = 'initial-bound-perturbation'
      spec( 23 )%keyword = 'pivot-tolerance-used'
      spec( 24 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 25 )%keyword = 'zero-pivot-tolerance'
      spec( 30 )%keyword = 'alpha-scaling-tolerance'
      spec( 39 )%keyword = 'implicit-variable-tolerance'
      spec( 26 )%keyword = 'identical-bounds-tolerance'
      spec( 40 )%keyword = 'insufficiently-feasible-tolerance'
      spec( 41 )%keyword = 'reduce-perturbation-factor'
      spec( 42 )%keyword = 'reduce-perturbation-multiplier'
      spec( 43 )%keyword = 'perturbation-small'
      spec( 46 )%keyword = 'target-barrier-accept-fraction'
      spec( 47 )%keyword = 'increase-barrier-parameter-by'
      spec( 35 )%keyword = 'maximum-cpu-time-limit'
      spec( 31 )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( 27 )%keyword = 'remove-linear-dependencies'
      spec( 28 )%keyword = 'treat-zero-bounds-as-general'
      spec( 29 )%keyword = 'just-find-feasible-point'
      spec( 33 )%keyword = 'balance-initial-complementarity'
      spec( 34 )%keyword = 'use-corrector-step'
      spec( 36 )%keyword = 'space-critical'
      spec( 37 )%keyword = 'deallocate-error-fatal'
      spec( 44 )%keyword = 'record-x-status'
      spec( 45 )%keyword = 'record-c-status'

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
      CALL SPECFILE_assign_value( spec( 32 ), control%initial_point,           &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 7 ), control%factor,                   &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 8 ), control%max_col,                  &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 9 ), control%indmin,                   &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 10 ), control%valmin,                  &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 11 ), control%itref_max,               &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 12 ), control%infeas_max,              &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 38 ), control%perturbation_strategy,   &
                                   control%error )
      CALL SPECFILE_assign_value( spec( 13 ), control%restore_problem,         &
                                   control%error )

!  Set real values


      CALL SPECFILE_assign_value( spec( 14 ), control%infinity,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 15 ), control%stop_p,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 16 ), control%stop_d,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 17 ), control%stop_c,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 18 ), control%prfeas,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 19 ), control%dufeas,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 20 ), control%mu_target,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 21 ),                                  &
                                  control%required_infeas_reduction,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 22 ), control%perturb_start,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 23 ), control%pivot_tol,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 24 ),                                  &
                                  control%pivot_tol_for_dependencies,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 25 ), control%zero_pivot,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 30 ), control%alpha_scale,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 39 ), control%implicit_tol,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 40 ), control%insufficiently_feasible, &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 41 ), control%reduce_perturb_factor,   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 42 ),                                  &
                                  control%reduce_perturb_multiplier,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 26 ), control%identical_bounds_tol,    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 43 ), control%perturbation_small,      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 46 ), control%mu_accept_fraction,      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 47 ), control%mu_increase_factor,      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 35 ), control%cpu_time_limit,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 31 ), control%clock_time_limit,        &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( 27 ), control%remove_dependencies,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 28 ),                                  &
                                  control%treat_zero_bounds_as_general,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 29 ), control%just_feasible,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 33 ),                                  &
                                  control%balance_initial_complementarity,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 34 ), control%use_corrector,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 36 ), control%space_critical,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 37 ),                                  &
                                  control%deallocate_error_fatal,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 44 ), control%record_x_status,         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 45 ), control%record_c_status,         &
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
      control%SBLS_control%preconditioner = 2

      RETURN

      END SUBROUTINE WCP_read_specfile

!-*-*-*-*-*-*-*-*-*-   W C P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE WCP_solve( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Finds a well-centred feasible point for the system
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
!    to be solved since the last call to WCP_initialize, and .FALSE. if
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
!        feasible region will be found.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
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
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %C_l, %C_u are REAL arrays of length %m, which must be set by the user
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
!   %Y_l, %Y_u are REAL arrays of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers
!    corresponding to the general constraints c_l <= A x and A x  <= c_u
!    respectively. Y_l should be positive and Y_u should be negative.
!    On successful exit, they will contain the required vectors of Lagrange
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
!   %Z_l, %Z_u are REAL arrays of length %m, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    corresponding to the simple bounds x_l <= x and x  <= x_u
!    respectively. Z_l should be positive and Z_u should be negative.
!    On successful exit, they will contain the required vectors of dual
!    variables.
!
!  data is a structure of type WCP_data_type which holds private internal data
!
!  control is a structure of type WCP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default
!   values for the elements may be set by a call to WCP_initialize.
!   See preamble for details
!
!  inform is a structure of type WCP_inform_type that provides
!    information on exit from WCP_solve. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!   - 1 one of the restrictions
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -2 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -4 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!    -5 The constraints are inconsistent.
!
!    -6 The constraints appear to have no feasible point.
!
!    -7 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!    -8 The problem is so ill-conditoned that further progress is impossible.
!
!  On exit from WCP_solve, other components of inform are as described in
!  the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( WCP_data_type ), INTENT( INOUT ) :: data
      TYPE ( WCP_control_type ), INTENT( IN ) :: control
      TYPE ( WCP_inform_type ), INTENT( OUT ) :: inform

!  Local variables

      INTEGER :: i, j, a_ne, n_depen, nbnds, lbreak, lbnds, nzc, n_sbls
      INTEGER :: dy_l_lower, dy_l_upper, dy_u_lower, dy_u_upper
      INTEGER :: dz_l_lower, dz_l_upper, dz_u_lower, dz_u_upper
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: fixed_sum, av_bnd
      LOGICAL :: printi, remap_freed, reset_bnd, implicit
      CHARACTER ( LEN = 80 ) :: array_name

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering WCP_solve ' )" )

!  Initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Initialize counts

      inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter = - 1 ; inform%nfacts = - 1
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%obj = - one ; inform%non_negligible_pivot = zero
      inform%feasible = .FALSE. ; inform%factorization_status = 0

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) inform%status
        GO TO 800
      END IF
      prob%Hessian_kind = 0

!  If required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', 2I8 )" ) prob%n, prob%m
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
         ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne )
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
            WRITE( control%error, 2010 ) inform%status
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
        "( ' ', /, '   **  Warning: one or more variable bounds reset ' )" )

      reset_bnd = .FALSE.
      DO i = 1, prob%m
        IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
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
        "( ' ', /, '   **  Warning: one or more constraint bounds reset ' )" )

!  Record the objective function value for any fixed variables

      fixed_sum = zero

!  Allocate additional workspace

      IF ( control%record_x_status ) THEN
        array_name = 'wcp: inform%X_status'
        CALL SPACE_resize_array( prob%n, inform%X_status, inform%status,       &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900
        inform%X_status( :  prob%n ) = 0
      END IF

      IF ( control%record_c_status ) THEN
        array_name = 'wcp: inform%C_status'
        CALL SPACE_resize_array( prob%m, inform%C_status, inform%status,       &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900
        inform%C_status( :  prob%m ) = 0
      END IF

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
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before preprocessing: ', /,          &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

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

        IF ( data%QPP_inform%status /= 0 ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" )   &
             data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          GO TO 800
        END IF

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions after preprocessing: ', /,           &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

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

          IF ( data%QPP_inform%status /= 0 ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( ' status ', I3, ' after QPP_apply ')" )   &
               data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status
            GO TO 800
          END IF
        END IF
        data%trans = data%trans + 1
      END IF

!  Special case: no free variables

      IF ( prob%n == 0 ) THEN
        prob%Y_l( : prob%m ) = zero ; prob%Y_u( : prob%m ) = zero
        prob%Z_l( : prob%n ) = zero ; prob%Z_u( : prob%n ) = zero
        prob%C( : prob%m ) = zero
        CALL WCP_AX( prob%m, prob%C( : prob%m ), prob%m,                       &
                     prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,                 &
                     prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
        GO TO 700
      END IF

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      IF ( .NOT. data%tried_to_remove_deps .AND. control%remove_dependencies ) &
        THEN
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
            "( /, 1X, I0, ' equalities from ', I0, ' constraints ' )" )        &
            data%dims%c_equality, prob%m

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
            WRITE( control%error, 2010 ) inform%status
          GO TO 800
        END IF
        IF ( printi .AND. inform%non_negligible_pivot < thousand *             &
          control%SBLS_control%SLS_control%absolute_pivot_tolerance )          &
            WRITE( control%out, "(                                             &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                       &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, ' ***  perhaps increase SLS_control%absolute_pivot_tolerance ',   &
       &  'from', ES11.4,'  ***', /, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )") &
           inform%non_negligible_pivot,                                        &
           control%SBLS_control%SLS_control%absolute_pivot_tolerance

!  Check for error exits

        IF ( inform%status /= 0 ) THEN

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= 0 )                                          &
              WRITE( control%error, 2020 ) inform%status, 'WCP_dependent'
          END IF
          GO TO 700
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, ' The following ',I0,' constraints appear', &
       &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  Special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y_l( : prob%m ) = zero ; prob%Y_u( : prob%m ) = zero
          prob%Z_l( : prob%n ) = zero ; prob%Z_u( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL WCP_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
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
          WRITE( control%out, "( /, ' -> ', I0, ' constraints are',            &
         & ' dependent and will be temporarily removed' )" ) n_depen

!  Allocate arrays to indicate which constraints have been freed

      array_name = 'wcp: data%C_freed'
      CALL SPACE_resize_array( n_depen, data%C_freed, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

!  Free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y_l( j ) = zero ; prob%Y_u( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        data%dims_save_freed = data%dims
        a_ne = prob%A%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before removal of dependecies: ', /, &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

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

        IF ( data%QPP_inform%status /= 0 ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" )   &
             data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          GO TO 800
        END IF

!  Record revised array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions after removal of dependencies: ', /, &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

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

      array_name = 'wcp: data%Y'
      CALL SPACE_resize_array( prob%m, data%Y, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%HX'
      CALL SPACE_resize_array( data%dims%v_e, data%HX, inform%status,          &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%GRAD_L'
      CALL SPACE_resize_array( data%dims%c_e, data%GRAD_L, inform%status,      &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_X_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%DIST_X_l, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_Z_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%DIST_Z_l, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_X_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%PERTURB_X_l, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_Z_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%PERTURB_Z_l, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_X_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%DIST_X_u, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_Z_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%DIST_Z_u, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_X_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%PERTURB_X_u, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_Z_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%PERTURB_Z_u, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%BARRIER_X'
      CALL SPACE_resize_array( data%dims%x_free + 1, prob%n,                   &
             data%BARRIER_X, inform%status,                                    &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%MU_X_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%MU_X_l, inform%status,                                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%MU_X_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%MU_X_u, inform%status,                                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DY_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DY_l, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_C_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DIST_C_l, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_Y_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%DIST_Y_l, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_C_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%PERTURB_C_l, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_Y_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%PERTURB_Y_l, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DY_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DY_u, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_C_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DIST_C_u, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DIST_Y_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%DIST_Y_u, inform%status,                                     &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_C_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%PERTURB_C_u, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%PERTURB_Y_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%PERTURB_Y_u, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%C, inform%status,                                            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%BARRIER_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%BARRIER_C, inform%status,                                    &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%MU_C_l'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,         &
             data%MU_C_l, inform%status,                                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%MU_C_u'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,         &
             data%MU_C_u, inform%status,                                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%SCALE_C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%SCALE_C, inform%status,                                      &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DELTA'
      CALL SPACE_resize_array( data%dims%v_e, data%DELTA, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%RHS'
      CALL SPACE_resize_array( data%dims%v_e, data%RHS, inform%status,         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DZ_l'
      CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,        &
             data%DZ_l, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%DZ_u'
      CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                    &
             data%DZ_u, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

!  Allocate optional extra arrays

      nbnds = prob%n + data%dims%x_l_end - data%dims%x_free                    &
             - data%dims%x_u_start + data%dims%c_l_end - data%dims%c_l_start   &
             + data%dims%c_u_end - data%dims%c_u_start + 3

      IF ( nbnds > 0 ) THEN
        lbnds = nbnds
        IF ( control%use_corrector ) THEN
          lbreak = 4 * nbnds
        ELSE
          lbreak = 2 * nbnds
        END IF
      ELSE
        lbnds = 0
        lbreak = 0
      END IF

      array_name = 'wcp: data%MU'
      CALL SPACE_resize_array( lbnds, data%MU, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%COEF0'
      CALL SPACE_resize_array( lbnds, data%COEF0, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%COEF1'
      CALL SPACE_resize_array( lbnds, data%COEF1, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%COEF2'
      CALL SPACE_resize_array( lbnds, data%COEF2, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%COEF3'
      CALL SPACE_resize_array( lbnds, data%COEF3, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%COEF4'
      CALL SPACE_resize_array( lbnds, data%COEF4, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%BREAKP'
      CALL SPACE_resize_array( lbreak, data%BREAKP, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'wcp: data%IBREAK'
      CALL SPACE_resize_array( lbreak, data%IBREAK, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      IF ( control%use_corrector ) THEN

        array_name = 'wcp: data%DELTA_cor'
        CALL SPACE_resize_array( data%dims%v_e, data%DELTA_cor,                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        array_name = 'wcp: data%DY_cor_l'
        CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_l_end,       &
               data%DY_cor_l, inform%status,                                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dy_l_lower = data%dims%c_l_start
        dy_l_upper = data%dims%c_l_end

        array_name = 'wcp: data%DY_cor_u'
        CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_u_end,       &
               data%DY_cor_u, inform%status,                                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dy_u_lower = data%dims%c_u_start
        dy_u_upper = data%dims%c_u_end

        array_name = 'wcp: data%DZ_cor_l'
        CALL SPACE_resize_array( data%dims%x_free + 1, data%dims%x_l_end,      &
               data%DZ_cor_l, inform%status,                                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dz_l_lower = data%dims%x_free + 1
        dz_l_upper = data%dims%x_l_end

        array_name = 'wcp: data%DZ_cor_u'
        CALL SPACE_resize_array( data%dims%x_u_start, prob%n,                  &
               data%DZ_cor_u, inform%status,                                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dz_u_lower = data%dims%x_u_start
        dz_u_upper = prob%n

      ELSE

        array_name = 'wcp: data%DELTA_cor'
        CALL SPACE_resize_array( 0, data%DELTA_cor, inform%status,             &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        array_name = 'wcp: data%DY_cor_l'
        CALL SPACE_resize_array( 0, data%DY_cor_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dy_l_lower = 1
        dy_l_upper = 0

        array_name = 'wcp: data%DY_cor_u'
        CALL SPACE_resize_array( 0, data%DY_cor_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dy_u_lower = 1
        dy_u_upper = 0

        array_name = 'wcp: data%DZ_cor_l'
        CALL SPACE_resize_array( 0, data%DZ_cor_l, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dz_l_lower = 1
        dz_l_upper = 0

        array_name = 'wcp: data%DZ_cor_u'
        CALL SPACE_resize_array( 0, data%DZ_cor_u, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900

        dz_u_lower = 1
        dz_u_upper = 0

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
      data%A_sbls%ne = a_ne + data%dims%nc

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
      data%A_sbls%col( : a_ne ) = prob%A%col( : a_ne )
      data%A_sbls%val( : a_ne ) = prob%A%val( : a_ne )

!  ... and include the coodinates corresponding to the slack variables

      DO i = 1, data%dims%nc
        data%A_sbls%row( a_ne + i ) = data%dims%c_equality + i
        data%A_sbls%col( a_ne + i ) = prob%n + i
      END DO

!  the zero matrix C will be in zero form

      CALL SMT_put( data%C_sbls%type, 'ZERO', inform%alloc_status )

!  =================
!  Solve the problem
!  =================

      data%SBLS_control = control%SBLS_control
      data%SBLS_control%preconditioner = 2

      IF ( printi ) WRITE( control%out,                                        &
           "( /, ' <------ variable bounds ------>',                           &
        &        ' <----- constraint bounds ----->',                           &
        &     /, '    free   below    both   above',                           &
        &        '   equal   below    both   above',                           &
        &     /,  8I8 )" )                                                     &
            data%dims%x_free, data%dims%x_u_start - data%dims%x_free - 1,      &
            data%dims%x_l_end - data%dims%x_u_start + 1,                       &
            prob%n - data%dims%x_l_end, data%dims%c_equality,                  &
            data%dims%c_u_start - data%dims%c_equality - 1,                    &
            data%dims%c_l_end - data%dims%c_u_start + 1,                       &
            prob%m - data%dims%c_l_end

      IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
        CALL WCP_solve_main( data%dims, prob%n, prob%m,                        &
                             prob%A%val, prob%A%col, prob%A%ptr,               &
                             prob%C_l, prob%C_u, prob%X_l, prob%X_u,           &
                             prob%C, prob%X, prob%Y_l, prob%Y_u,               &
                             prob%Z_l, prob%Z_u, data%Y, data%HX,              &
                             data%GRAD_L, data%DIST_X_l, data%DIST_Z_l,        &
                             data%DIST_X_u, data%DIST_Z_u,                     &
                             data%BARRIER_X, data%MU_X_l, data%MU_X_u,         &
                             data%DY_l, data%DIST_C_l, data%DIST_Y_l,          &
                             data%DY_u, data%DIST_C_u, data%DIST_Y_u,          &
                             data%C, data%BARRIER_C,                           &
                             data%MU_C_l, data%MU_C_u,                         &
                             data%SCALE_C, data%DELTA, data%RHS,               &
                             data%DZ_l, data%DZ_u,                             &
                             data%PERTURB_X_l, data%PERTURB_X_u,               &
                             data%PERTURB_Y_l, data%PERTURB_Y_u,               &
                             data%PERTURB_Z_l, data%PERTURB_Z_u,               &
                             data%PERTURB_C_l, data%PERTURB_C_u,               &
                             prob%f,                                           &
                             data%MU, data%COEF0, data%COEF1, data%COEF2,      &
                             data%COEF3, data%COEF4, data%DELTA_cor,           &
                             data%H_sbls, data%A_sbls, data%C_sbls,            &
                             data%DY_cor_l, dy_l_lower, dy_l_upper,            &
                             data%DY_cor_u, dy_u_lower, dy_u_upper,            &
                             data%DZ_cor_l, dz_l_lower, dz_l_upper,            &
                             data%DZ_cor_u, dz_u_lower, dz_u_upper,            &
                             data%BREAKP, data%IBREAK, data%SBLS_data,         &
                             control, inform, data%SBLS_control,               &
                             prob%gradient_kind )
      ELSE
        CALL WCP_solve_main( data%dims, prob%n, prob%m,                        &
                             prob%A%val, prob%A%col, prob%A%ptr,               &
                             prob%C_l, prob%C_u, prob%X_l, prob%X_u,           &
                             prob%C, prob%X, prob%Y_l, prob%Y_u,               &
                             prob%Z_l, prob%Z_u, data%Y, data%HX,              &
                             data%GRAD_L, data%DIST_X_l, data%DIST_Z_l,        &
                             data%DIST_X_u, data%DIST_Z_u,                     &
                             data%BARRIER_X, data%MU_X_l, data%MU_X_u,         &
                             data%DY_l, data%DIST_C_l, data%DIST_Y_l,          &
                             data%DY_u, data%DIST_C_u, data%DIST_Y_u,          &
                             data%C, data%BARRIER_C,                           &
                             data%MU_C_l, data%MU_C_u,                         &
                             data%SCALE_C, data%DELTA, data%RHS,               &
                             data%DZ_l, data%DZ_u,                             &
                             data%PERTURB_X_l, data%PERTURB_X_u,               &
                             data%PERTURB_Y_l, data%PERTURB_Y_u,               &
                             data%PERTURB_Z_l, data%PERTURB_Z_u,               &
                             data%PERTURB_C_l, data%PERTURB_C_u,               &
                             prob%f,                                           &
                             data%MU, data%COEF0, data%COEF1, data%COEF2,      &
                             data%COEF3, data%COEF4, data%DELTA_cor,           &
                             data%H_sbls, data%A_sbls, data%C_sbls,            &
                             data%DY_cor_l, dy_l_lower, dy_l_upper,            &
                             data%DY_cor_u, dy_u_lower, dy_u_upper,            &
                             data%DZ_cor_l, dz_l_lower, dz_l_upper,            &
                             data%DZ_cor_u, dz_u_lower, dz_u_upper,            &
                             data%BREAKP, data%IBREAK, data%SBLS_data,         &
                             control, inform, data%SBLS_control,               &
                             prob%gradient_kind, G = prob%G )
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

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        IF ( control%record_x_status )                                         &
          CALL SORT_inverse_permute( data%QPP_map_freed%n,                     &
            data%QPP_map_freed%x_map,                                          &
            IX = inform%X_status( : data%QPP_map_freed%n ) )
        IF ( control%record_c_status ) THEN
          inform%C_status( prob%m + 1 : data%QPP_map_freed%m ) = 4
          CALL SORT_inverse_permute( data%QPP_map_freed%m,                     &
            data%QPP_map_freed%c_map,                                          &
            IX = inform%C_status( : data%QPP_map_freed%m ) )
        END IF
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform, prob,           &
                          get_all = .TRUE. )
        CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
                            data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed
        inform%c_implicit = inform%c_implicit + n_depen

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

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        IF ( control%record_x_status )                                         &
          CALL SORT_inverse_permute( data%QPP_map%n,                           &
            data%QPP_map%x_map, IX = inform%X_status( : data%QPP_map%n ) )
        IF ( control%record_c_status )                                         &
          CALL SORT_inverse_permute( data%QPP_map%m,                           &
            data%QPP_map%c_map, IX = inform%C_status( : data%QPP_map%m ) )

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
        CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

      IF ( control%print_level > 1 .AND. control%record_x_status ) THEN
        implicit = .FALSE.
        DO i = 1, prob%n
          IF ( inform%X_status( i ) == - 1 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2030 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'LOWER', prob%X( i ), prob%X_l( i ), &
              prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          ELSE IF ( inform%X_status( i ) == 1 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2030 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'UPPER', prob%X( i ), prob%X_l( i ), &
              prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          ELSE IF ( inform%X_status( i ) == - 2 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2030 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'LFREE', prob%X( i ), prob%X_l( i ), &
              prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          ELSE IF ( inform%X_status( i ) == 2 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2030 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'UFREE', prob%X( i ), prob%X_l( i ), &
              prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          ELSE IF ( inform%X_status( i ) == - 3 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2030 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'FREE ', prob%X( i ), prob%X_l( i ), &
              prob%X_u( i ), prob%Z_l( i ) + prob%Z_u( i )
          END IF
        END DO
      END IF

      IF ( control%print_level > 1 .AND. control%record_c_status ) THEN
        implicit = .FALSE.
        DO i = 1, prob%m
          IF ( inform%C_status( i ) == - 1 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'LOWER', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
          ELSE IF ( inform%C_status( i ) == 1 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'UPPER', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
          ELSE IF ( inform%C_status( i ) == - 2 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'LFREE', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
          ELSE IF ( inform%C_status( i ) == 2 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'UFREE', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
         ELSE IF ( inform%C_status( i ) == - 3 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'FREE ', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
          ELSE IF ( inform%C_status( i ) == 4 ) THEN
            IF ( .NOT. implicit ) THEN
              WRITE( control%out, 2040 ) ; implicit = .TRUE. ; END IF
            WRITE( control%out, 2050 ) i, 'DEPEN', prob%C( i ), prob%C_l( i ), &
              prob%C_u( i ), prob%Y_l( i ) + prob%Y_u( i )
          END IF
        END DO
      END IF

!  Compute total time

  800 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out, 2000 ) inform%time%total,              &
        inform%time%preprocess, inform%time%analyse, inform%time%factorize,    &
        inform%time%solve

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving WCP_solve ' )" )
      RETURN

!  Allocation error

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving WCP_solve ' )" )
      RETURN

!  Non-executable statements

 2000 FORMAT( /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',&
              /, 14X, ' =                  WCP total time                   =',&
              /, 14X, ' =', 16X, 0P, F12.2, 23x, '='                           &
              /, 14X, ' =    preprocess    analyse    factorize     solve   =',&
              /, 14X, ' =', 4F12.2, 3x, '=',                                   &
              /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=' )
 2010 FORMAT( ' ',  /, '   **  Error return ', I0, ' from WCP ' )
 2020 FORMAT( '   **  Error return ', I0, ' from ', A )
 2030 FORMAT( /, ' Implicit bounds: ', /, '                    ',              &
                 '        <------ Bounds ------> ', /                          &
                 '      #  state    value   ',                                 &
                 '    Lower       Upper       Dual ' )
 2040 FORMAT( /, ' Implicit constraints: ', /, '                   ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      #  state    value   ',                                 &
                 '    Lower       Upper     Multiplier ' )
 2050 FORMAT( I7, 1X, A6, 4ES12.4 )

!  End of WCP_solve

      END SUBROUTINE WCP_solve

!-*-*-*-*-*-   W C P _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE WCP_solve_main( dims, n, m, A_val, A_col, A_ptr,              &
                                 C_l, C_u, X_l, X_u, C_RES, X, Y_l, Y_u,       &
                                 Z_l, Z_u, Y, HX, GRAD_L, DIST_X_l,            &
                                 DIST_Z_l, DIST_X_u, DIST_Z_u,                 &
                                 BARRIER_X, MU_X_l, MU_X_u, DY_l,              &
                                 DIST_C_l, DIST_Y_l, DY_u, DIST_C_u,           &
                                 DIST_Y_u, C, BARRIER_C, MU_C_l, MU_C_u,       &
                                 SCALE_C, DELTA, RHS, DZ_l, DZ_u,              &
                                 PERTURB_X_l, PERTURB_X_u,                     &
                                 PERTURB_Y_l, PERTURB_Y_u,                     &
                                 PERTURB_Z_l, PERTURB_Z_u,                     &
                                 PERTURB_C_l, PERTURB_C_u,                     &
                                 f, MU, COEF0, COEF1, COEF2, COEF3,            &
                                 COEF4, DELTA_cor,                             &
                                 H_sbls, A_sbls, C_sbls,                       &
                                 DY_cor_l, dy_l_lower, dy_l_upper,             &
                                 DY_cor_u, dy_u_lower, dy_u_upper,             &
                                 DZ_cor_l, dz_l_lower, dz_l_upper,             &
                                 DZ_cor_u, dz_u_lower, dz_u_upper,             &
                                 BREAKP, IBREAK, SBLS_data, control, inform,   &
                                 SBLS_control, gradient_kind, G )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Finds a well-centered point within the polytope
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
!  dims is a structure of type WCP_data_type, whose components hold SCALAR
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
!  n, m, ..., Z_l, Z_u exactly as for prob% in WCP_solve
!
!  control and inform are exactly as for WCP_solve
!
!  The remaining arguments are used as internal workspace, and need not be
!  set on entry
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, gradient_kind
      INTEGER, INTENT( IN ) :: dy_l_lower, dy_l_upper, dy_u_lower, dy_u_upper
      INTEGER, INTENT( IN ) :: dz_l_lower, dz_l_upper, dz_u_lower, dz_u_upper
      REAL ( KIND = wp ), INTENT( IN ) :: f
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y_l, Y_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z_l, Z_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C_RES, Y
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%v_e ) :: DELTA, RHS
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%v_e ) :: HX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: GRAD_L
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: DZ_l,         &
                           DIST_X_l, DIST_Z_l, PERTURB_X_l, PERTURB_Z_l, MU_X_l
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%x_u_start : n ) ::    &
                     DZ_u, DIST_X_u, DIST_Z_u, PERTURB_X_u, PERTURB_Z_u, MU_X_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DY_l,               &
                           DIST_C_l, DIST_Y_l, PERTURB_C_l, PERTURB_Y_l, MU_C_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DY_u,               &
                           DIST_C_u, DIST_Y_u, PERTURB_C_u, PERTURB_Y_u, MU_C_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, BARRIER_C, SCALE_C

!  allocatable arrays and structures

      INTEGER, DIMENSION( : ) :: IBREAK
      REAL ( KIND = wp ), DIMENSION( : ) :: COEF0, COEF1, COEF2, COEF3, COEF4
      REAL ( KIND = wp ), DIMENSION( : ) :: MU, BREAKP
      REAL ( KIND = wp ), DIMENSION( dy_l_lower : dy_l_upper ) :: DY_cor_l
      REAL ( KIND = wp ), DIMENSION( dy_u_lower : dy_u_upper ) :: DY_cor_u
      REAL ( KIND = wp ), DIMENSION( dz_l_lower : dz_l_upper ) :: DZ_cor_l
      REAL ( KIND = wp ), DIMENSION( dz_u_lower : dz_u_upper ) :: DZ_cor_u
      REAL ( KIND = wp ), DIMENSION( : ) :: DELTA_cor

      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls, A_sbls
      TYPE ( SMT_type ), INTENT( IN ) :: C_sbls
      TYPE ( WCP_control_type ), INTENT( IN ) :: control
      TYPE ( WCP_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: SBLS_data
      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: SBLS_control

!  Parameters

      REAL ( KIND = wp ), PARAMETER :: eta = tenm4
      REAL ( KIND = wp ), PARAMETER :: sigma_max = point01
      REAL ( KIND = wp ), PARAMETER :: degen_tol = tenm5

!  Local variables

      INTEGER :: A_ne, i, l, start_print, stop_print, print_level
      INTEGER :: nbnds, nbnds_x, nbnds_c, cs_bad
      INTEGER :: out, error, it_best, infeas_max
      REAL :: time_record, time_start, time_now, time_solve
      REAL ( KIND = wp ) :: clock_record, clock_start, clock_now, clock_solve
      REAL ( KIND = wp ) :: pjgnrm, errorg, mu_target, pmax, amax, alpha
      REAL ( KIND = wp ) :: gi, merit, res_prim_dual, max_zr, max_yr, slkmax
      REAL ( KIND = wp ) :: res_prim, res_dual, slknes, slkmin, dist_bound
      REAL ( KIND = wp ) :: slknes_x, slknes_c, slkmax_x, slkmax_c, slknes_req
      REAL ( KIND = wp ) :: slkmin_x, slkmin_c, merit_best, too_close
      REAL ( KIND = wp ) :: required_infeas_reduction, relative_pivot_tol
      REAL ( KIND = wp ) :: prfeas, dufeas, p_min, p_max, d_min, d_max, balance
      REAL ( KIND = wp ) :: errorc, one_minus_alpha, pivot_tol, alpha_max, gmax
      REAL ( KIND = wp ) :: pmax_cor, omega_l, omega_u, perturb, max_r, mu_scale
      REAL ( KIND = wp ) :: reduce_perturb_factor, one_minus_red_pert_fac
      REAL ( KIND = wp ) :: bound_average, bound_length, perturb_max, alpha_est
      REAL ( KIND = wp ) :: perturb_x, perturb_c, perturb_y, perturb_z
      REAL ( KIND = wp ) :: mu_l, mu_u, max_xr, max_cr, perturbation_small
!     REAL ( KIND = wp ) :: old_merit, min_mu, nu
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: set_printp, printt, printi, printp, printe, printd, printw
      LOGICAL :: get_factors, refact, use_corrector, maxpiv, reset_mu
      LOGICAL :: now_interior, now_feasible, start_major
      LOGICAL :: use_scale_c = .FALSE.
!     LOGICAL :: use_scale_c = .TRUE.
      CHARACTER ( LEN = 1 ) :: re, co, al
      CHARACTER ( LEN = 2 ) :: coal
      INTEGER :: sif = 50
!     LOGICAL :: generate_sif = .TRUE.
      LOGICAL :: generate_sif = .FALSE.

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!     reset_mu = .TRUE.
      reset_mu = .FALSE.

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering WCP_solve_main ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( generate_sif .AND. PRESENT( G ) ) THEN
        WRITE( sif, "( 'NAME          WCPB_OUT', //, 'VARIABLES', / )" )
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
            WRITE( sif, "( '    RANGE    ', ' C', I8, ' ', ES12.5 )" )         &
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
      time_solve = 0.0 ; clock_solve = 0.0

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
          "( A, ' factor = ', I6, ' out of range [0,2]. Reset to 0' )" )       &
          prefix, SBLS_control%factorization
        SBLS_control%factorization = 0
      END IF

!  ==================
!  Input error checks
!  ==================

!  If there are no variables, exit

      IF ( n == 0 ) THEN
        i = COUNT( ABS( C_l( : dims%c_equality ) ) > control%stop_p ) +        &
            COUNT( C_l( dims%c_l_start : dims%c_l_end ) > control%stop_p ) +   &
            COUNT( C_u( dims%c_u_start : dims%c_u_end ) < - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = 0
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        C_RES = zero ; Y_l = zero ; Y_u = zero
        inform%obj = zero
        GO TO 810
      END IF

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

      DO i = dims%x_u_start, dims%x_l_end
        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN
          inform%status = GALAHAD_error_bad_bounds ; GO TO 700 ; END IF
      END DO

      DO i = dims%c_u_start, dims%c_l_end
        IF ( C_u( i ) - C_l( i ) <= epsmch ) THEN
          inform%status = GALAHAD_error_bad_bounds ; GO TO 700 ; END IF
      END DO

!  Set control parameters

      prfeas = MAX( control%prfeas, epsmch )
      dufeas = MAX( control%dufeas, epsmch )
      required_infeas_reduction = MAX( epsmch,                                 &
        MIN( control%required_infeas_reduction ** 2, one - epsmch ) )
      infeas_max = MAX( 0, control%infeas_max )
      relative_pivot_tol = SBLS_control%SLS_control%relative_pivot_tolerance

!  Record array size

      A_ne = A_ptr( m + 1 ) - 1

!  If required, write out the data matrix for the problem

      IF ( printd ) WRITE( out, 2150 ) ' a ', ( ( i, A_col( l ), A_val( l ),   &
                           l = A_ptr( i ), A_ptr( i + 1 ) - 1 ), i = 1, m )

      IF ( control%balance_initial_complementarity ) THEN
        IF ( control%mu_target <= zero ) THEN
          balance = ten
        ELSE
          balance = control%mu_target
        END IF
      END IF

!  Set the default simple bound perturbations, perturb

      IF ( control%perturbation_strategy <= 0 ) THEN
        perturb = zero
      ELSE
        IF ( control%perturb_start < zero ) THEN
          perturb = MIN( control%stop_p, control%stop_d, control%stop_c )
        ELSE
          perturb = control%perturb_start
        END IF
      END IF
      reduce_perturb_factor = control%reduce_perturb_factor
!     too_close = point01
      too_close = zero

!  =============================
!  Find a suitable initial point
!  =============================

      IF ( control%initial_point >= 1 ) THEN

!  Solve the problem of minimizing
!     1/2||x-x_m||^2_(X_D^-2) + 1/2||c-c_m||^2_(C_D^-2) + g^T x
!  subject to A x - c = 0, where x_m and c_m are "bound average"
!  values for x and c and X_d/C_d are "bound lengths";

!  for two-sided bounds x_l <= x <= x_u,
!    x_m = 1/2 (x_l + x_u ) and x_d = x_u - x_l
!  for one-sided bounds x_l <= x,
!    x_m = x_l + pr_feas and x_d = pr_feas
!  for free variables
!    x_m = 0 and x_d = infinite

!  Equivalently, solve the linear system

!   ( X_D^-2   0     A^T ) ( x )   ( X_D^-2 x_m )
!   (   0    C_D^-2  -I  ) ( c ) = ( C_D^-2 c_m )
!   (   A     -I      0  ) ( y )   (     0      )

! ::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Analyse the sparsity pattern of the required matrix
! ::::::::::::::::::::::::::::::::::::::::::::::::::::

        SCALE_C = one

!  Set up the bound averages and bound lengths; also set up the
!  weights (BARRIER) and right-hand sides (RHS) for the problem

!       dist_bound = prfeas
        dist_bound = ten

  200   CONTINUE
        RHS( : dims%x_free ) = zero
        DO i = dims%x_free + 1, dims%x_l_start - 1
          bound_average = dist_bound
          bound_length = dist_bound
          BARRIER_X( i ) =  one / bound_length ** 2
          RHS( i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%x_l_start, dims%x_u_start - 1
          bound_average = X_l( i ) + dist_bound
          bound_length = dist_bound
          BARRIER_X( i ) = one / bound_length ** 2
          RHS( i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%x_u_start, dims%x_l_end
          bound_average = half * ( X_l( i ) + X_u( i ) )
          bound_length = X_u( i ) - X_l( i )
          BARRIER_X( i ) =  one / bound_length ** 2
          RHS( i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%x_l_end + 1, dims%x_u_end
          bound_average = X_u( i ) - dist_bound
          bound_length = dist_bound
          BARRIER_X( i ) =  one / bound_length ** 2
          RHS( i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%x_u_end + 1, n
          bound_average = - dist_bound
          bound_length = dist_bound
          BARRIER_X( i ) =  one / bound_length ** 2
          RHS( i ) = bound_average / bound_length ** 2
        END DO

        IF ( gradient_kind == 1 ) THEN
          RHS( : n ) = RHS( : n ) - one
        ELSE IF ( gradient_kind /= 0 ) THEN
          RHS( : n ) = RHS( : n ) - G
        END IF

        DO i = dims%c_l_start, dims%c_u_start - 1
          bound_average = C_l( i ) + dist_bound
          bound_length = dist_bound
          BARRIER_C( i ) =  one / bound_length ** 2
          RHS( dims%c_b + i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          bound_average = half * ( C_l( i ) + C_u( i ) )
          bound_length = C_u( i ) - C_l( i )
          BARRIER_C( i ) = one / bound_length ** 2
          RHS( dims%c_b + i ) = bound_average / bound_length ** 2
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          bound_average = C_u( i ) - dist_bound
          bound_length = dist_bound
          BARRIER_C( i ) = one / bound_length ** 2
          RHS( dims%c_b + i ) = bound_average / bound_length ** 2
        END DO
        RHS( dims%y_s : dims%y_i - 1 ) = C_l( : dims%c_equality )
        RHS( dims%y_i : dims%y_e ) = zero

!  Factorize the required matrix

        get_factors = .TRUE. ; re = 'r'

!  complete A

        DO i = 1, dims%nc
          A_sbls%val( A_ne + i ) = - SCALE_C( dims%c_equality + i )
        END DO

!  Include the values of the barrier terms

        H_sbls%val( 1 : dims%x_free ) = zero
        H_sbls%val(dims%x_free + 1 : n ) = BARRIER_X
        H_sbls%val( dims%c_s : dims%c_e ) = BARRIER_C

! ::::::::::::::::::::::::::::::
!  Factorize the required matrix
! ::::::::::::::::::::::::::::::

        IF ( get_factors ) THEN
          CALL CLOCK_time( clock_record )
          IF ( printw ) WRITE( out, "( A,                                      &
         &  ' ......... factorization of KKT matrix ...............' )" ) prefix
          CALL SBLS_form_and_factorize( A_sbls%n, A_sbls%m, H_sbls, A_sbls,    &
            C_sbls, sbls_data, SBLS_control, inform%SBLS_inform )
          inform%time%analyse = inform%time%analyse +                          &
            inform%SBLS_inform%SLS_inform%time%analyse
          inform%time%clock_analyse = inform%time%clock_analyse +              &
            inform%SBLS_inform%SLS_inform%time%clock_analyse
          inform%time%factorize = inform%time%factorize +                      &
            inform%SBLS_inform%SLS_inform%time%factorize
          inform%time%clock_factorize = inform%time%clock_factorize +          &
            inform%SBLS_inform%SLS_inform%time%clock_factorize

          IF ( printw ) WRITE( out, "( A,                                      &
         &  ' ............... end of factorization ...............' )" ) prefix

!  Record the storage required

          inform%nfacts = inform%nfacts + 1
          inform%factorization_integer =                                       &
            inform%SBLS_inform%SLS_inform%integer_size_necessary
          inform%factorization_real =                                          &
            inform%SBLS_inform%SLS_inform%real_size_necessary

!  Test that the factorization succeeded

          inform%factorization_status = inform%SBLS_inform%status
          IF ( inform%factorization_status < 0 ) THEN
            IF ( printe ) WRITE( error, 2040 ) prefix,                         &
             inform%factorization_status, 'SBLS_form_and_factorize'

!  It didn't. We might have run out of options

            IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
              inform%status = GALAHAD_error_factorization ; GO TO 700

!  ... or we may change the method

            ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
              pivot_tol = relative_pivot_tol
              maxpiv = pivot_tol >= half
              SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
              SBLS_control%factorization = 2
              IF ( printi ) WRITE( out,                                        &
                "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

            ELSE
              pivot_tol = half
              maxpiv = .TRUE.
              SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
              SBLS_control%factorization = 2
              IF ( printi )                                                    &
                WRITE( out, "( A, ' Pivot tolerance increased ' )" ) prefix
            END IF
            alpha = zero
            GO TO 200

!  Record warning conditions

          ELSE
            IF (inform%factorization_status > 0 ) THEN
              IF ( printt ) THEN
                WRITE( out, 2050 ) prefix, inform%SBLS_inform%status,          &
                                   'SBLS_form_andfactorize'
              END IF
            END IF
          END IF

          SBLS_control%new_h = 1
          SBLS_control%new_a = 0
          SBLS_control%new_c = 0

          IF ( printt ) THEN
            CALL CLOCK_time( clock_now )
            WRITE( out, "( A, ' ** factorize time = ', F10.2 ) " ) prefix,     &
               clock_now - clock_record
            WRITE( out, 2060 ) prefix, inform%factorization_integer,           &
                               inform%factorization_real
          END IF
        ELSE
          inform%factorization_integer = 0
          inform%factorization_real = 0
        END IF

!  :::::::::::::::::::::::::::::::::::::::::::::::::
!  Solve the linear system to find the initial point
!  :::::::::::::::::::::::::::::::::::::::::::::::::

        IF ( printw ) WRITE( out, "( A,                                        &
       &  ' ............... compute initial point ............... ' )" ) prefix

!  Use a direct method

        DELTA = RHS

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls,                   &
                         sbls_data, SBLS_control, inform%SBLS_inform, DELTA )
        CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
        time_solve = time_solve + time_now - time_record
        clock_solve = clock_solve + clock_now - clock_record
        IF ( printt ) WRITE( out, "( A, ' ** solve time = ', F10.2 )" )        &
          prefix, clock_now - clock_record

        inform%status = inform%SBLS_inform%status
        IF ( inform%status /= 0 ) GO TO 700

        X = DELTA( dims%x_s : dims%x_e )
        C = DELTA( dims%c_s : dims%c_e )
        Y = DELTA( dims%y_s : dims%y_e )

!  Compute the residual of the linear system

        CALL WCP_residual( dims, n, m, dims%v_e, A_ne, A_val, A_col, A_ptr,    &
                           X, C, Y, RHS( dims%x_s : dims%x_e ),                &
                           RHS( dims%c_s : dims%c_e ),                         &
                           RHS( dims%y_s : dims%y_e ), HX( : dims%v_e ),       &
                           zero, BARRIER_X, BARRIER_C, SCALE_C,                &
                           errorg, errorc, print_level, control )

!  If the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely.

        IF ( SQRT( SUM( ( HX( : dims%v_e ) - RHS ) ** 2 ) ) > tenm8 ) THEN

!  It wasn't. We might have run out of options ...

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
          alpha = zero
          GO TO 200
        END IF

        IF ( printw ) WRITE( out,                                              &
             "( A, ' ........... initial point computed ...........' )" ) prefix

        IF ( printd ) THEN
          WRITE( out, 2120 ) ' DX ', DELTA( dims%x_s : dims%x_e )
          IF ( m > 0 ) WRITE( out, 2120 ) ' DY ', DELTA( dims%y_s : dims%y_e )
        END IF

        Y = - Y

!  Compute the gradient of the Lagrangian function.

        CALL WCP_Lagrangian_gradient( n, m, Y, A_ne, A_val, A_col, A_ptr,      &
                                      GRAD_L( dims%x_s : dims%x_e ),           &
                                      gradient_kind, G = G )

!  Compute individual shifts

        IF ( control%initial_point == 1 ) THEN

!  shifts for the problem variables

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( X( i ) >= prfeas ) THEN
              PERTURB_X_l( i ) = zero
            ELSE
              PERTURB_X_l( i ) = prfeas - X( i )
            END IF
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            IF ( X( i ) >= X_l( i ) + prfeas ) THEN
              PERTURB_X_l( i ) = zero
            ELSE
              PERTURB_X_l( i ) = X_l( i ) + prfeas - X( i )
            END IF
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            IF ( X( i ) <= X_u( i ) - prfeas ) THEN
              PERTURB_X_u( i ) = zero
            ELSE
              PERTURB_X_u( i ) = X( i ) + prfeas - X_u( i )
            END IF
          END DO
          DO i = dims%x_u_end + 1, n
            IF ( X( i ) <= - prfeas ) THEN
              PERTURB_X_u( i ) = zero
            ELSE
              PERTURB_X_u( i ) = X( i ) + prfeas
            END IF
          END DO

!  shifts for the slack variables

          DO i = dims%c_l_start, dims%c_l_end
            IF ( C( i ) >= C_l( i ) + prfeas ) THEN
              PERTURB_C_l( i ) = zero
            ELSE
              PERTURB_C_l( i ) = C_l( i ) + prfeas - C( i )
            END IF
          END DO
          DO i = dims%c_u_start,  dims%c_u_end
            IF ( C( i ) <= C_u( i ) - prfeas ) THEN
              PERTURB_C_u( i ) = zero
            ELSE
              PERTURB_C_u( i ) = C( i ) + prfeas - C_u( i )
            END IF
          END DO

!  values and shifts for the dual problem variables

          DO i = dims%x_free + 1, dims%x_u_start - 1
            Z_l( i ) = GRAD_L( i )
            IF ( Z_l( i ) >= dufeas ) THEN
              PERTURB_Z_l( i ) = zero
            ELSE
              PERTURB_Z_l( i ) = - GRAD_L( i ) + dufeas
            END IF
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( GRAD_L( i ) >= zero ) THEN
              Z_l( i ) = GRAD_L( i ) + dufeas
              Z_u( i ) = - dufeas
            ELSE
              Z_l( i ) = dufeas
              Z_u( i ) = GRAD_L( i ) - dufeas
            END IF
            IF ( Z_l( i ) >= dufeas ) THEN
              PERTURB_Z_l( i ) = zero
            ELSE
              PERTURB_Z_l( i ) = dufeas
            END IF
            IF ( Z_u( i ) <= - dufeas ) THEN
              PERTURB_Z_u( i ) = zero
            ELSE
              PERTURB_Z_u( i ) = dufeas
            END IF
          END DO
          DO i = dims%x_l_end + 1, n
            Z_u( i ) = GRAD_L( i )
            IF ( Z_u( i ) <= - dufeas ) THEN
              PERTURB_Z_u( i ) = zero
            ELSE
              PERTURB_Z_u( i ) = GRAD_L( i ) + dufeas
            END IF
          END DO

!  values and shifts for the dual slack variables

          DO i = dims%c_l_start, dims%c_u_start - 1
            Y_l( i ) = Y( i )
            IF ( Y_l( i ) >= dufeas ) THEN
              PERTURB_Y_l( i ) = zero
            ELSE
              PERTURB_Y_l( i ) = - Y( i ) + dufeas
            END IF
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( Y( i ) >= zero ) THEN
              Y_l( i ) = Y( i ) + dufeas
              Y_u( i ) = - dufeas
            ELSE
              Y_l( i ) = dufeas
              Y_u( i ) = Y( i ) - dufeas
            END IF
            IF ( Y_l( i ) >= dufeas ) THEN
              PERTURB_Y_l( i ) = zero
            ELSE
              PERTURB_Y_l( i ) = dufeas
            END IF
            IF ( Y_u( i ) <= - dufeas ) THEN
              PERTURB_Y_u( i ) = zero
            ELSE
              PERTURB_Y_u( i ) = dufeas
            END IF
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            Y_u( i ) = Y( i )
            IF ( Y_u( i ) <= - dufeas ) THEN
              PERTURB_Y_u( i ) = zero
            ELSE
              PERTURB_Y_u( i ) = Y( i ) + dufeas
            END IF
          END DO

!  Compute overall shifts

        ELSE

!  shifts for the problem variables

          perturb_x = zero
          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( X( i ) < prfeas )                                             &
              perturb_x = MAX( perturb_x, prfeas - X( i ) )
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            IF ( X( i ) < X_l( i ) + prfeas )                                  &
              perturb_x = MAX( perturb_x, X_l( i ) + prfeas - X( i ) )
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            IF ( X( i ) > X_u( i ) - prfeas )                                  &
              perturb_x = MAX( perturb_x, X( i ) + prfeas - X_u( i ) )
          END DO
          DO i = dims%x_u_end + 1, n
            IF ( X( i ) > - prfeas )                                           &
              perturb_x = MAX( perturb_x, X( i ) + prfeas )
          END DO

!  shifts for the slack variables

          perturb_c = zero
          DO i = dims%c_l_start, dims%c_l_end
            IF ( C( i ) < C_l( i ) + prfeas )                                  &
              perturb_c = MAX( perturb_c, C_l( i ) + prfeas - C( i ) )
          END DO
          DO i = dims%c_u_start,  dims%c_u_end
            IF ( C( i ) > C_u( i ) - prfeas )                                  &
              perturb_c = MAX( perturb_c, C( i ) + prfeas - C_u( i ) )
          END DO

!  values and shifts for the dual problem variables

          perturb_z = zero
          DO i = dims%x_free + 1, dims%x_u_start - 1
            Z_l( i ) = GRAD_L( i )
            IF ( GRAD_L( i ) < dufeas )                                        &
              perturb_z = MAX( perturb_z, dufeas - GRAD_L( i ) )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( GRAD_L( i ) >= zero ) THEN
              Z_l( i ) = GRAD_L( i ) + dufeas
              Z_u( i ) = - dufeas
            ELSE
              Z_l( i ) = dufeas
              Z_u( i ) = GRAD_L( i ) - dufeas
            END IF
          END DO
          DO i = dims%x_l_end + 1, n
            Z_u( i ) = GRAD_L( i )
            IF ( GRAD_L( i ) > - dufeas )                                      &
              perturb_z = MAX( perturb_z, GRAD_L( i ) + dufeas )
          END DO

!  values and shifts for the dual slack variables

          perturb_y = zero
          DO i = dims%c_l_start, dims%c_u_start - 1
            Y_l( i ) = Y( i )
            IF ( Y( i ) < dufeas )                                             &
              perturb_y = MAX( perturb_y, dufeas - Y( i ) )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( Y( i ) >= zero ) THEN
              Y_l( i ) = Y( i ) + dufeas
              Y_u( i ) = - dufeas
            ELSE
              Y_l( i ) = dufeas
              Y_u( i ) = Y( i ) - dufeas
            END IF
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            Y_u( i ) = Y( i )
            IF ( Y( i ) > - dufeas )                                           &
              perturb_y = MAX( perturb_y, Y( i ) + dufeas )
          END DO

!  Assign the shifts

!  Shift problem and slack variable bounds by the same amount

          IF ( control%initial_point == 2 ) THEN
            perturb = MAX( perturb_x, perturb_c )
            PERTURB_X_l = perturb ; PERTURB_X_u = perturb
            PERTURB_C_l = perturb ; PERTURB_C_u = perturb
            perturb = MAX( perturb_z, perturb_y )
            PERTURB_Z_l = perturb ; PERTURB_Z_u = perturb
            PERTURB_Y_l = perturb ; PERTURB_Y_u = perturb

!  Shift problem and slack variable bounds by their own overall shifts

          ELSE
            PERTURB_X_l = perturb_x ; PERTURB_X_u = perturb_x
            PERTURB_C_l = perturb_c ; PERTURB_C_u = perturb_c
            PERTURB_Z_l = perturb_z ; PERTURB_Z_u = perturb_z
            PERTURB_Y_l = perturb_y ; PERTURB_Y_u = perturb_y
          END IF
        END IF

      ELSE

!  Set the simple bound perturbations perturb

        PERTURB_X_l = perturb ; PERTURB_X_u = perturb
        PERTURB_C_l = perturb ; PERTURB_C_u = perturb
        PERTURB_Z_l = perturb ; PERTURB_Z_u = perturb
        PERTURB_Y_l = perturb ; PERTURB_Y_u = perturb

!  Move the input starting point away from any bounds,

!  The variable is a non-negativity

        DO i = dims%x_free + 1, dims%x_l_start - 1
          X( i ) = MAX( X( i ), prfeas )
          IF ( control%balance_initial_complementarity ) THEN
            Z_l( i ) = balance / ( X( i ) + PERTURB_X_l( i ) )
          ELSE
            Z_l( i ) = MAX( ABS( Z_l( i ) ), dufeas )
          END IF
          Z_u( i ) = zero
        END DO

!  The variable has just a lower bound

        DO i = dims%x_l_start, dims%x_u_start - 1
          X( i ) = MAX( X( i ), X_l( i ) + prfeas )
          IF ( control%balance_initial_complementarity ) THEN
            Z_l( i ) = balance / ( X( i ) - X_l( i ) + PERTURB_X_l( i ) )
          ELSE
            Z_l( i ) = MAX( ABS( Z_l( i ) ), dufeas )
          END IF
          Z_u( i ) = zero
        END DO

!  The variable has both lower and upper bounds

        DO i = dims%x_u_start, dims%x_l_end
          IF ( X_l( i ) + prfeas >= X_u( i ) - prfeas ) THEN
            X( i ) = half * ( X_l( i ) + X_u( i ) )
          ELSE
            X( i ) = MIN( MAX( X( i ), X_l( i ) + prfeas ), X_u( i ) - prfeas )
          END IF
          IF ( control%balance_initial_complementarity ) THEN
            Z_l( i ) = balance / ( X( i ) - X_l( i ) + PERTURB_X_l( i ) )
            Z_u( i ) = - balance / ( X_u( i ) + PERTURB_X_u( i ) - X( i ) )
          ELSE
            Z_l( i ) = MAX(   ABS( Z_l( i ) ),   dufeas )
            Z_u( i ) = MIN( - ABS( Z_u( i ) ), - dufeas )
          END IF
        END DO

!  The variable has just an upper bound

        DO i = dims%x_l_end + 1, dims%x_u_end
          X( i ) = MIN( X( i ), X_u( i ) - prfeas )
          IF ( control%balance_initial_complementarity ) THEN
            Z_u( i ) = - balance / ( X_u( i ) + PERTURB_X_u( i ) - X( i ) )
          ELSE
            Z_u( i ) = MIN( - ABS( Z_u( i ) ), - dufeas )
          END IF
          Z_l( i ) = zero
        END DO

!  The variable is a non-positivity

        DO i = dims%x_u_end + 1, n
          X( i ) = MIN( X( i ), - prfeas )
          IF ( control%balance_initial_complementarity ) THEN
            Z_u( i ) = balance / ( PERTURB_X_u( i ) - X( i ) )
          ELSE
            Z_u( i ) = MIN( - ABS( Z_u( i ) ), - dufeas )
          END IF
          Z_l( i ) = zero
        END DO

!  Compute the value of the constraint, and their residuals

        nbnds_c = 0
        IF ( m > 0 ) THEN
          C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
          C_RES( dims%c_l_start : dims%c_u_end ) = zero
          CALL WCP_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
          IF ( printd ) THEN
            WRITE( out,                                                        &
           "( /, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X, 'y_l', 9X, 'y_u')")
            DO i = 1, dims%c_l_start - 1
              WRITE( out, "( I6, 3ES12.4 )" ) i, C_RES( i ), C_l( i ), C_u( i )
            END DO
          END IF

!  The constraint has just a lower bound

          DO i = dims%c_l_start, dims%c_u_start - 1

!  Compute an appropriate scale factor

            IF ( use_scale_c ) THEN
              SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
            ELSE
              SCALE_C( i ) = one
            END IF

!  Scale the bounds

            C_l( i ) = C_l( i ) / SCALE_C( i )
            C( i ) = MAX( C_RES( i ) / SCALE_C( i ), C_l( i ) + prfeas )
            IF ( control%balance_initial_complementarity ) THEN
              Y_l( i ) = balance / ( C( i ) - C_l( i ) + PERTURB_C_l( i ) )
            ELSE
              Y_l( i ) = MAX( ABS( SCALE_C( i ) * Y_l( i ) ),  dufeas )
            END IF
          END DO

!  The constraint has both lower and upper bounds

          DO i = dims%c_u_start, dims%c_l_end

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
              C( i ) = MIN( MAX( C_RES( i ) / SCALE_C( i ),                    &
                                 C_l( i ) + prfeas), C_u( i ) - prfeas )
            END IF
            IF ( control%balance_initial_complementarity ) THEN
              Y_l( i ) = balance / ( C( i ) - C_l( i ) + PERTURB_C_l( i ) )
              Y_u( i ) = - balance / ( C_u( i ) + PERTURB_C_u( i ) - C( i ) )
            ELSE
              Y_l( i ) = MAX(   ABS( SCALE_C( i ) * Y_l( i ) ),   dufeas )
              Y_u( i ) = MIN( - ABS( SCALE_C( i ) * Y_u( i ) ), - dufeas )
            END IF
          END DO

!  The constraint has just an upper bound

          DO i = dims%c_l_end + 1, dims%c_u_end

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
            IF ( control%balance_initial_complementarity ) THEN
              Y_u( i ) = - balance / ( C_u( i ) + PERTURB_C_u( i ) - C( i ) )
            ELSE
              Y_u( i ) = MIN( - ABS( SCALE_C( i ) * Y_u( i ) ), - dufeas )
            END IF
          END DO
        END IF
      END IF

!  ==========================
!  End of initial point phase
!  ==========================

!  Now shift the bounds and dual variables appropriately

      nbnds_x = 0

!  The variable is free

      IF ( printd ) THEN
        WRITE( out,                                                            &
        "( /, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X, 'z_l', 9X, 'z_u')" )
        DO i = 1, dims%x_free
          WRITE( out, "( I6, ES12.4, 4( '      -     ') )" ) i, X( i )
        END DO
      END IF

!  The variable is a non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        nbnds_x = nbnds_x + 1
        DIST_X_l( i ) = X( i ) + PERTURB_X_l( i )
        IF ( DIST_X_l( i ) < too_close ) THEN
          DIST_X_l( i ) = DIST_X_l( i ) + too_close
          PERTURB_X_l( i ) = PERTURB_X_l( i ) + too_close
        END IF
        DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
        IF ( DIST_Z_l( i ) < too_close ) THEN
          DIST_Z_l( i ) = DIST_Z_l( i ) + too_close
          PERTURB_Z_l( i ) = PERTURB_Z_l( i ) + too_close
        END IF
        IF ( printd ) WRITE( out, "( I6, 2ES12.4, '      -     ', ES12.4,      &
       &  '      -     ' )" ) i, X( i ), zero, Z_l( i )
      END DO

!  The variable has just a lower bound

      DO i = dims%x_l_start, dims%x_u_start - 1
        nbnds_x = nbnds_x + 1
        DIST_X_l( i ) = X( i ) - X_l( i ) + PERTURB_X_l( i )
        IF ( DIST_X_l( i ) < too_close ) THEN
          DIST_X_l( i ) = DIST_X_l( i ) + too_close
          PERTURB_X_l( i ) = PERTURB_X_l( i ) + too_close
        END IF
        DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
        IF ( DIST_Z_l( i ) < too_close ) THEN
          DIST_Z_l( i ) = DIST_Z_l( i ) + too_close
          PERTURB_Z_l( i ) = PERTURB_Z_l( i ) + too_close
        END IF
        IF ( printd ) WRITE( out, "( I6, 2ES12.4, '      -     ', ES12.4,      &
       &  '      -     ' )" ) i, X( i ), X_l( i ), Z_l( i )
      END DO

!  The variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end
        nbnds_x = nbnds_x + 2
        DIST_X_l( i ) = X( i ) - X_l( i ) + PERTURB_X_l( i )
        IF ( DIST_X_l( i ) < too_close ) THEN
          DIST_X_l( i ) = DIST_X_l( i ) + too_close
          PERTURB_X_l( i ) = PERTURB_X_l( i ) + too_close
        END IF
        DIST_X_u( i ) = X_u( i ) + PERTURB_X_u( i ) - X( i )
        IF ( DIST_X_u( i ) < too_close ) THEN
          DIST_X_u( i ) = DIST_X_u( i ) + too_close
          PERTURB_X_u( i ) = PERTURB_X_u( i ) + too_close
        END IF
        DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
        IF ( DIST_Z_l( i ) < too_close ) THEN
          DIST_Z_l( i ) = DIST_Z_l( i ) + too_close
          PERTURB_Z_l( i ) = PERTURB_Z_l( i ) + too_close
        END IF
        DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
        IF ( DIST_Z_u( i ) < too_close ) THEN
          DIST_Z_u( i ) = DIST_Z_u( i ) + too_close
          PERTURB_Z_u( i ) = PERTURB_Z_u( i ) + too_close
        END IF
        IF ( printd ) WRITE( out, "( I6, 5ES12.4 )" )                          &
             i, X( i ), X_l( i ), X_u( i ), Z_l( i ), Z_u( i )
      END DO

!  The variable has just an upper bound

      DO i = dims%x_l_end + 1, dims%x_u_end
        nbnds_x = nbnds_x + 1
        DIST_X_u( i ) = X_u( i ) + PERTURB_X_u( i ) - X( i )
        IF ( DIST_X_u( i ) < too_close ) THEN
          DIST_X_u( i ) = DIST_X_u( i ) + too_close
          PERTURB_X_u( i ) = PERTURB_X_u( i ) + too_close
        END IF
        DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
        IF ( DIST_Z_u( i ) < too_close ) THEN
          DIST_Z_u( i ) = DIST_Z_u( i ) + too_close
          PERTURB_Z_u( i ) = PERTURB_Z_u( i ) + too_close
        END IF
        IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,       &
       &  '      -     ', ES12.4 )" ) i, X( i ), X_u( i ), Z_u( i )
      END DO

!  The variable is a non-positivity

      DO i = dims%x_u_end + 1, n
        nbnds_x = nbnds_x + 1
        DIST_X_u( i ) = PERTURB_X_u( i ) - X( i )
        IF ( DIST_X_u( i ) < too_close ) THEN
          DIST_X_u( i ) = DIST_X_u( i ) + too_close
          PERTURB_X_u( i ) = PERTURB_X_u( i ) + too_close
        END IF
        DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
        IF ( DIST_Z_u( i ) < too_close ) THEN
          DIST_Z_u( i ) = DIST_Z_u( i ) + too_close
          PERTURB_Z_u( i ) = PERTURB_Z_u( i ) + too_close
        END IF
        IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,       &
       &  '      -     ',  ES12.4 )" ) i, X( i ), zero, Z_u( i )
      END DO

!  Compute the value of the constraint, and their residuals

      nbnds_c = 0
      IF ( m > 0 ) THEN
        C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
        C_RES( dims%c_l_start : dims%c_u_end ) = zero
        CALL WCP_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
        IF ( printd ) THEN
          WRITE( out,                                                          &
          "( /, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X, 'y_l', 9X, 'y_u')")
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( I6, 3ES12.4 )" ) i, C_RES( i ), C_l( i ), C_u( i )
          END DO
        END IF

!  The constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          nbnds_c = nbnds_c + 1
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          DIST_C_l( i ) = C( i ) - C_l( i ) + PERTURB_C_l( i )
          IF ( DIST_C_l( i ) < too_close ) THEN
            DIST_C_l( i ) = DIST_C_l( i ) + too_close
            PERTURB_C_l( i ) = PERTURB_C_l( i ) + too_close
          END IF
          DIST_Y_l( i ) = Y_l( i ) + PERTURB_Y_l( i )
          IF ( DIST_Y_l( i ) < too_close ) THEN
            DIST_Y_l( i ) = DIST_Y_l( i ) + too_close
            PERTURB_Y_l( i ) = PERTURB_Y_l( i ) + too_close
          END IF
          IF ( printd ) WRITE( out,  "( I6, 2ES12.4, '      -     ', ES12.4,   &
         &  '      -     ' )" ) i, C_RES( i ), C_l( i ), Y_l( i )
        END DO

!  The constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          nbnds_c = nbnds_c + 2
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          DIST_C_l( i ) = C( i ) - C_l( i ) + PERTURB_C_l( i )
          IF ( DIST_C_l( i ) < too_close ) THEN
            DIST_C_l( i ) = DIST_C_l( i ) + too_close
            PERTURB_C_l( i ) = PERTURB_C_l( i ) + too_close
          END IF
          DIST_C_u( i ) = C_u( i ) + PERTURB_C_u( i ) - C( i )
          IF ( DIST_C_u( i ) < too_close ) THEN
            DIST_C_u( i ) = DIST_C_u( i ) + too_close
            PERTURB_C_u( i ) = PERTURB_C_u( i ) + too_close
          END IF
          DIST_Y_l( i ) = Y_l( i ) + PERTURB_Y_l( i )
          IF ( DIST_Y_l( i ) < too_close ) THEN
            DIST_Y_l( i ) = DIST_Y_l( i ) + too_close
            PERTURB_Y_l( i ) = PERTURB_Y_l( i ) + too_close
          END IF
          DIST_Y_u( i ) = - Y_u( i ) + PERTURB_Y_u( i )
          IF ( DIST_Y_u( i ) < too_close ) THEN
            DIST_Y_u( i ) = DIST_Y_u( i ) + too_close
            PERTURB_Y_u( i ) = PERTURB_Y_u( i ) + too_close
          END IF
          IF ( DIST_Y_u( i ) < too_close ) THEN
            DIST_Y_u( i ) = DIST_Y_u( i ) + too_close
            PERTURB_Y_u( i ) = PERTURB_Y_u( i ) + too_close
          END IF
          IF ( printd ) WRITE( out, "( I6, 5ES12.4 )" )                        &
            i, C_RES( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i )
        END DO

!  The constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          nbnds_c = nbnds_c + 1

!  Compute an appropriate initial value for the slack variable

          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          DIST_C_u( i ) = C_u( i ) + PERTURB_C_u( i ) - C( i )
          IF ( DIST_C_u( i ) < too_close ) THEN
            DIST_C_u( i ) = DIST_C_u( i ) + too_close
            PERTURB_C_u( i ) = PERTURB_C_u( i ) + too_close
          END IF
          DIST_Y_u( i ) = - Y_u( i ) + PERTURB_Y_u( i )
          IF ( DIST_Y_u( i ) < too_close ) THEN
            DIST_Y_u( i ) = DIST_Y_u( i ) + too_close
            PERTURB_Y_u( i ) = PERTURB_Y_u( i ) + too_close
          END IF
          IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,     &
         &  '      -     ', ES12.4 )" ) i, C_RES( i ), C_u( i ), Y_u( i )
        END DO
        res_prim = MAXVAL( ABS( C_RES ) )
      ELSE
        res_prim = zero
      END IF

!  Find the max-norm of the residual

      nbnds = nbnds_x + nbnds_c
      IF ( printi .AND. m > 0 .AND. dims%c_l_start <= dims%c_u_end )           &
        WRITE( out, "( A, ' largest/smallest scale factor ', 2ES12.4 )" )      &
          prefix, MAXVAL( SCALE_C ), MINVAL( SCALE_C )

!  Compute the complementary slackness

!     DO i = dims%x_free + 1, dims%x_l_start - 1
!       write(6,"(I6, ' x lower', 2ES12.4)" ) i, X( i ), Z_l( i )
!     END DO
!     DO i = dims%x_l_start, dims%x_l_end
!       write(6,"(I6, ' x lower', 2ES12.4)" ) i, DIST_X_l( i ), DIST_Z_l( i )
!     END DO
!     DO i = dims%x_u_start, dims%x_u_end
!       write(6,"(I6, ' x upper', 2ES12.4)" ) i, DIST_X_u( i ), DIST_Z_u( i )
!     END DO
!     DO i = dims%x_u_end + 1, n
!       write(6,"(I6, ' x upper', 2ES12.4)" ) i, X( i ), Z_u( i )
!     END DO

!     DO i = dims%c_l_start, dims%c_l_end
!       write(6,"(I6, ' c lower', 2ES12.4)" ) i, DIST_C_l( i ), DIST_Y_l( i )
!     END DO
!     DO i = dims%c_u_start, dims%c_u_end
!       write(6,"(I6, ' c upper', 2ES12.4)" ) i, DIST_C_u( i ), DIST_Y_u( i )
!     END DO

      slknes_x = DOT_PRODUCT( DIST_X_l( dims%x_free + 1 : dims%x_l_end ),      &
                              DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ) +   &
                 DOT_PRODUCT( DIST_X_u( dims%x_u_start : n ),                  &
                              DIST_Z_u( dims%x_u_start : n ) )
      slknes_c = DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),       &
                              DIST_Y_l( dims%c_l_start : dims%c_l_end ) ) +    &
                 DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),       &
                              DIST_Y_u( dims%c_u_start : dims%c_u_end ) )
      slknes = slknes_x + slknes_c

      slkmin_x = MIN( MINVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) *     &
                              DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ),    &
                      MINVAL( DIST_X_u( dims%x_u_start : n ) *                 &
                              DIST_Z_u( dims%x_u_start : n ) ) )
      slkmin_c = MIN( MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *      &
                              DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),     &
                      MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) *      &
                              DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )
      slkmin = MIN( slkmin_x, slkmin_c )

      slkmax_x = MAX( MAXVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) *     &
                              DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ),    &
                      MAXVAL( DIST_X_u( dims%x_u_start : n ) *                 &
                              DIST_Z_u( dims%x_u_start : n ) ) )
      slkmax_c = MAX( MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *      &
                              DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),     &
                      MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) *      &
                              DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )
      slkmax = MAX( slkmax_x, slkmax_c )

!     WRITE(6,2120) ' >0 ', X( dims%x_free + 1 : dims%x_l_start - 1 )
!     WRITE(6,2120) ' >l ', DIST_X_l( dims%x_l_start : dims%x_l_end )
!     WRITE(6,2120) ' <u ', DIST_X_u( dims%x_u_start : dims%x_u_end )
!     WRITE(6,2120) ' <0 ', - X( dims%x_u_end + 1 : n )

      p_min = MIN( MINVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                   MINVAL( DIST_X_u( dims%x_u_start : n ) ),                   &
                   MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

      p_max = MAX( MAXVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                   MAXVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) ),        &
                   MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

      d_min = MIN( MINVAL( DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                   MINVAL( DIST_Z_u( dims%x_u_start : n ) ),                   &
                   MINVAL( DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MINVAL( DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )

      d_max = MAX( MAXVAL( DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                   MAXVAL( DIST_Z_u( dims%x_u_start : n ) ),                   &
                   MAXVAL( DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),        &
                   MAXVAL( DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )

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
        slknes = slknes / nbnds
      ELSE
        slknes = zero
      END IF

      IF ( printt .AND. nbnds > 0 ) WRITE( out, 2130 )                         &
        prefix, slknes, prefix, slknes_x, prefix, slknes_c,                    &
        prefix, slkmin_x, slkmax_x, prefix, slkmin_c, slkmax_c,                &
        prefix, p_min, p_max, prefix, d_min, d_max

!  Compute the initial objective value

      inform%obj = f

      IF ( gradient_kind == 1 ) THEN
        inform%obj = inform%obj + SUM( X )
        gmax = one
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
        gmax = MAXVAL( ABS( G ) )
      ELSE
        gmax = zero
      END IF

!  Find the largest components of A

      IF ( A_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : A_ne ) ) )
      ELSE
        amax = zero
      END IF

      IF ( printi ) THEN
        WRITE( out, "( A, '  maximum element of A = ', ES12.4 )" ) prefix, amax
        WRITE( out, "( A, '  maximum element of g = ', ES12.4 )" ) prefix, gmax
      END IF

!  Set the target barrier parameters

      IF ( control%initial_point >= 1 ) THEN
!       mu_scale = 1.0_wp
!       mu_scale = 1.1_wp
        mu_scale = 10.0_wp
        DO i = dims%x_free + 1, dims%x_l_end
          MU_X_l( i ) = mu_scale * DIST_X_l( i ) * DIST_Z_l( i )
         END DO
        DO i = dims%x_u_start, n
          MU_X_u( i ) = mu_scale * DIST_X_u( i ) * DIST_Z_u( i )
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          MU_C_l( i ) = mu_scale * DIST_C_l( i ) * DIST_Y_l( i )
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          MU_C_u( i ) = mu_scale * DIST_C_u( i ) * DIST_Y_u( i )
        END DO
        mu_target = mu_scale * slkmax
      ELSE
        IF ( control%mu_target < zero ) THEN
          mu_target = sigma_max * slknes
        ELSE
          mu_target = control%mu_target
        END IF
        MU_X_l = mu_target ; MU_X_u = mu_target
        DO i = dims%x_u_start, dims%x_l_end
          mu_scale = MIN( mu_tol, MAX( one, X_u( i ) - X_l( i ) ) )
          MU_X_l( i ) = MU_X_l( i ) * mu_scale
          MU_X_u( i ) = MU_X_u( i ) * mu_scale
        END DO
        MU_C_l = mu_target ; MU_C_u = mu_target
        DO i = dims%c_u_start, dims%c_l_end
          mu_scale = MIN( mu_tol, MAX( one, C_u( i ) - C_l( i ) ) )
          MU_C_l( i ) = MU_C_l( i ) * mu_scale
          MU_C_u( i ) = MU_C_u( i ) * mu_scale
        END DO
      END IF

!  Compute the error in the complementry slackness

      slknes_req = SUM( ABS( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) *      &
                             DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) -      &
                             MU_X_l( dims%x_free + 1 : dims%x_l_end ) ) )      &
                 + SUM( ABS( DIST_X_u( dims%x_u_start : n ) *                  &
                             DIST_Z_u( dims%x_u_start : n ) -                  &
                             MU_X_u( dims%x_u_start : n ) ) )                  &
                 + SUM( ABS( DIST_C_l( dims%c_l_start : dims%c_l_end ) *       &
                             DIST_Y_l( dims%c_l_start : dims%c_l_end ) -       &
                             MU_C_l( dims%c_l_start : dims%c_l_end ) ) )       &
                 + SUM( ABS( DIST_C_u( dims%c_u_start : dims%c_u_end ) *       &
                             DIST_Y_u( dims%c_u_start : dims%c_u_end ) -       &
                             MU_C_u( dims%c_u_start : dims%c_u_end ) ) )

      IF ( nbnds > 0 ) THEN
        omega_l = MIN( tenm4, tenm4 * slkmin / mu_target )
!       omega_l = MIN( tenm10, tenm10 * slkmin / mu_target )
!       omega_l = MIN( point1, point1 * slkmin / mu_target )
        omega_u = ten ** 10
!       omega_u = one / omega_l
!       write(6,"( ' omega_l, omega_u ', 2ES12.4 )" ) omega_l, omega_u
      END IF

!  Compute the gradient of the Lagrangian function.

      Y = Y_l + Y_u
      CALL WCP_Lagrangian_gradient( n, m, Y, A_ne, A_val, A_col, A_ptr,        &
                                    GRAD_L( dims%x_s : dims%x_e ),             &
                                    gradient_kind, G = G )

!  Evaluate the merit function

      merit = WCP_merit_value( dims, n, m, Y, Y_l, DIST_Y_l, Y_u, DIST_Y_u,    &
                               Z_l, DIST_Z_l, Z_u, DIST_Z_u,                   &
                               DIST_X_l, DIST_X_u, DIST_C_l,                   &
                               DIST_C_u, GRAD_L( dims%x_s : dims%x_e ),        &
                               C_RES, res_dual,                                &
                               MU_X_l, MU_X_u, MU_C_l, MU_C_u )
      res_prim_dual = SUM( ABS( C_RES ) ) + res_dual
!     old_merit = merit

!  Test to see if we are feasible

      inform%feasible =                                                        &
        res_prim <= control%stop_p .AND.res_dual <= control%stop_d
      pjgnrm = infinity

      IF ( inform%feasible ) THEN
        IF ( printi ) WRITE( out, 2070 ) prefix
        inform%x_implicit =                                                    &
               COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < zero )       &
             + COUNT( X_l( dims%x_l_start : dims%x_l_end ) >                   &
                      X( dims%x_l_start : dims%x_l_end ) )                     &
             + COUNT( X( dims%x_u_start : dims%x_u_end ) >                     &
                      X_u( dims%x_u_start : dims%x_u_end ) )                   &
             + COUNT( X( dims%x_u_end + 1: n ) > zero )
        inform%z_implicit =                                                    &
               COUNT( Z_l( dims%x_free + 1 : dims%x_l_end ) < zero )           &
             + COUNT( Z_u( dims%x_u_start : n ) > zero )
        inform%c_implicit =                                                    &
               COUNT( C_l( dims%c_l_start : dims%c_l_end ) >                   &
                      C_RES( dims%c_l_start : dims%c_l_end ) )                 &
             + COUNT( C_RES( dims%c_u_start : dims%c_u_end ) >                 &
                      C_u( dims%c_u_start : dims%c_u_end ) )
        inform%y_implicit =                                                    &
               COUNT( Y_l( dims%c_l_start : dims%c_l_end ) < zero )            &
             + COUNT( Y_u( dims%c_u_start : dims%c_u_end ) > zero )

        IF ( MAX( inform%x_implicit, inform%c_implicit,                        &
             inform%y_implicit, inform%z_implicit ) == 0 .AND.                 &
             control%just_feasible ) THEN
          inform%status = 0
          GO TO 500
        END IF
      END IF

!  Prepare for the major iteration

      inform%iter = 0 ; inform%nfacts = 0
      IF ( printt ) WRITE( out, "( ' ', /, A, ' merit function value = ',      &
     &     ES12.4 )" ) prefix, merit

      IF ( n == 0 ) THEN
        inform%status = 0 ; GO TO 600
      END IF
      merit_best = merit ; it_best = 0

!  Test for convergence

!     IF ( res_prim <= control%stop_p .AND. res_dual <= control%stop_d .AND.   &
!          slknes_req <= control%stop_c ) THEN
!       inform%status = 0 ; GO TO 600
!     END IF

!  ===================================================
!  Analyse the sparsity pattern of the required matrix
!  ===================================================

      refact = .TRUE. ; re = ' '  ;  co = ' ' ; al = ' ' ;  cs_bad = - 1


      pivot_tol = relative_pivot_tol
      maxpiv = pivot_tol >= half
      now_feasible = .FALSE.
      IF ( control%mu_accept_fraction <= zero .OR.                             &
           control%mu_accept_fraction > one ) THEN
        mu_l = one
      ELSE
        mu_l = control%mu_accept_fraction
      END IF
      mu_u = one / mu_l

      IF ( printi ) WRITE( out,                                                &
       "(  /, A, '  Primal    convergence tolerance = ', ES12.4,               &
     &    /, A, '  Dual      convergence tolerance = ', ES12.4,                &
     &    /, A, '  Slackness convergence tolerance = ', ES12.4 )" )            &
         prefix, control%stop_p, prefix, control%stop_d, prefix, control%stop_c

      IF (  control%perturbation_small > zero ) THEN
        perturbation_small = control%perturbation_small
      ELSE
        perturbation_small = MIN( control%stop_p, control%stop_d )
      END IF

!  ---------------------------------------------------------------------
!  ------------ Start of Perturbation Reduction Loop ------------------
!  ---------------------------------------------------------------------

      DO

         perturb_max =                                                         &
           MAX( MAXVAL( ABS( PERTURB_X_l( dims%x_free + 1 : dims%x_l_end ) ) ),&
                MAXVAL( ABS( PERTURB_X_u( dims%x_u_start : n ) ) ),            &
                MAXVAL( ABS( PERTURB_Z_l( dims%x_free + 1 : dims%x_l_end ) ) ),&
                MAXVAL( ABS( PERTURB_Z_u( dims%x_u_start : n ) ) ),            &
                MAXVAL( ABS( PERTURB_C_l( dims%c_l_start : dims%c_l_end ) ) ), &
                MAXVAL( ABS( PERTURB_C_u( dims%c_u_start : dims%c_u_end ) ) ), &
                MAXVAL( ABS( PERTURB_Y_l( dims%c_l_start : dims%c_l_end ) ) ), &
                MAXVAL( ABS( PERTURB_Y_u( dims%c_u_start : dims%c_u_end ) ) ) )
         now_interior = perturb_max <= perturbation_small

         IF ( printi )                                                         &
           WRITE( out, "( /, A, '  -*-*-*- perturbations (#>0,max) = ', I0,    &
        &       ', ', ES10.4, ' -*-*-*-' )" ) prefix,                          &
           COUNT( PERTURB_X_l( dims%x_free + 1 : dims%x_l_end ) > zero ) +     &
           COUNT( PERTURB_X_u( dims%x_u_start : n ) > zero ) +                 &
           COUNT( PERTURB_Z_l( dims%x_free + 1 : dims%x_l_end ) > zero ) +     &
           COUNT( PERTURB_Z_u( dims%x_u_start : n ) > zero ) +                 &
           COUNT( PERTURB_C_l( dims%c_l_start : dims%c_l_end ) > zero ) +      &
           COUNT( PERTURB_C_u( dims%c_u_start : dims%c_u_end ) > zero ) +      &
           COUNT( PERTURB_Y_l( dims%c_l_start : dims%c_l_end ) > zero ) +      &
           COUNT( PERTURB_Y_u( dims%c_u_start : dims%c_u_end ) > zero ),       &
!          MIN( MINVAL( ABS( PERTURB_X_l( dims%x_free + 1 : dims%x_l_end ) ) ),&
!               MINVAL( ABS( PERTURB_X_u( dims%x_u_start : n ) ) ),            &
!               MINVAL( ABS( PERTURB_C_l( dims%c_l_start : dims%c_l_end ) ) ), &
!               MINVAL( ABS( PERTURB_C_u( dims%c_u_start : dims%c_u_end ) ) )),&
           perturb_max

           start_print = inform%iter
           start_major = .TRUE.

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

            IF ( start_major ) THEN
              WRITE( out, 2000 ) prefix
              WRITE( out, 2020 ) prefix, inform%iter, re, res_prim, res_dual,  &
                                 slknes_req, merit, mu_target, clock_now
              start_major = .FALSE.
            ELSE
              IF ( printt .OR. ( printi .AND. inform%iter == start_print ) )   &
                WRITE( out, 2000 ) prefix
              coal = '  ' ; coal = TRIM( co ) // TRIM( al )
              WRITE( out, 2030 ) prefix, inform%iter, re, res_prim, res_dual,  &
                                 slknes_req, merit, alpha, coal,               &
                                 mu_target, clock_now
            END IF

            IF ( printd ) THEN
              WRITE( out, 2120 ) ' X ', X
              WRITE( out, 2120 ) ' perturb_x_l ', perturb_x_l
!             WRITE( out, 2120 ) ' DIST_x_l ', DIST_x_l
              WRITE( out, 2120 ) ' perturb_x_u ', perturb_x_u
!             WRITE( out, 2120 ) ' DIST_x_u ', DIST_x_u
              WRITE( out, 2120 ) ' Z_l ', Z_l
              WRITE( out, 2120 ) ' perturb_z_l ', perturb_z_l
!             WRITE( out, 2120 ) ' DIST_z_l ', DIST_z_l
              WRITE( out, 2120 ) ' Z_u ', Z_u
              WRITE( out, 2120 ) ' perturb_z_u ', perturb_z_u
!             WRITE( out, 2120 ) ' DIST_z_u ', DIST_z_u
            END IF
          END IF

!  Test for optimality

          IF ( res_prim <= control%stop_p .AND.                                &
               res_dual <= control%stop_d ) THEN
            IF ( slknes_req <= control%stop_c ) THEN
              inform%status = 0 ; GO TO 490
            END IF

!  Test for quasi-optimality

            cs_bad = 0
            DO i = dims%x_free + 1, dims%x_l_end
              gi = DIST_X_l( i ) * DIST_Z_l( i )
              IF ( gi < mu_l * MU_X_l( i ) .OR. gi > mu_u * MU_X_l( i ) )      &
                cs_bad = cs_bad + 1
            END DO
            DO i = dims%x_u_start, n
              gi = DIST_X_u( i ) * DIST_Z_u( i )
              IF ( gi < mu_l * MU_X_u( i ) .OR. gi > mu_u * MU_X_u( i ) )      &
                cs_bad = cs_bad + 1
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              gi = DIST_C_l( i ) * DIST_Y_l( i )
              IF ( gi < mu_l * MU_C_l( i ) .OR. gi > mu_u * MU_C_l( i ) )      &
                cs_bad = cs_bad + 1
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              gi = DIST_C_u( i ) * DIST_Y_u( i )
              IF ( gi < mu_l * MU_C_u( i ) .OR. gi > mu_u * MU_C_u( i ) )      &
                cs_bad = cs_bad + 1
            END DO
            IF ( cs_bad == 0 ) THEN
              WRITE( out, "( /, A, ' Quasi-optimal point ' )" ) prefix
              inform%status = 0 ; GO TO 490
            END IF
          END IF

!  Test to see if more than maxit iterations have been performed

          inform%iter = inform%iter + 1
          IF ( inform%iter > control%maxit ) THEN
            inform%status = GALAHAD_error_max_iterations ; GO TO 600
          END IF

!  Check that the CPU time limit has not been reached

          CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now )
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - time_start > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
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

!         WRITE( 6, "( ' start, stop print, iter ', 3I8 )" )                   &
!           start_print, stop_print, inform%iter

!  Test to see whether the method has stalled

          IF ( merit <= required_infeas_reduction * merit_best ) THEN
            merit_best = merit
            it_best = 0
          ELSE
            it_best = it_best + 1
            IF ( it_best > infeas_max ) THEN
              IF ( printi ) WRITE( out, "( A, /, ' ============ the problem',  &
             &  ' appears to be infeasible  =============' )" ) prefix
              inform%status = GALAHAD_error_tiny_step ; GO TO 600
            END IF
          END IF

!  Compute the (Hessian) barrier terms -

!  problem variables:

          DO i = dims%x_free + 1, dims%x_u_start - 1
            IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )            &
                i, DIST_X_l( i ), DIST_Z_l( i )
            BARRIER_X( i ) = DIST_Z_l( i ) / DIST_X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )            &
                i, DIST_X_l( i ), DIST_Z_l( i )
            IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )            &
                i, DIST_X_u( i ), DIST_Z_u( i )
            BARRIER_X( i ) = DIST_Z_l( i ) / DIST_X_l( i ) +                   &
                             DIST_Z_u( i ) / DIST_X_u( i )
          END DO
          DO i = dims%x_l_end + 1, n
            IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )            &
                i, DIST_X_u( i ), DIST_Z_u( i )
            BARRIER_X( i ) = DIST_Z_u( i ) / DIST_X_u( i )
          END DO

!  slack variables:

          BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero
          DO i = dims%c_l_start, dims%c_u_start - 1
            IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )            &
                i, DIST_C_l( i ), DIST_Y_l( i )
            BARRIER_C( i ) = DIST_Y_l( i ) / DIST_C_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )            &
                i, DIST_C_l( i ), DIST_Y_l( i )
            IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )            &
                i, DIST_C_u( i ), DIST_Y_u( i )
            BARRIER_C( i ) = DIST_Y_l( i ) / DIST_C_l( i ) +                   &
                             DIST_Y_u( i ) / DIST_C_u( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printw )              &
              WRITE( 6, "( ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )            &
                i, DIST_C_u( i ), DIST_Y_u( i )
            BARRIER_C( i ) = DIST_Y_u( i ) / DIST_C_u( i )
          END DO

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-*-      Factorization      -*-*-*-*-*-*-*-*-*-
!  =====================================================================

          IF ( refact ) THEN

!  Only refactorize if B has changed

            re = 'r'
            get_factors = .TRUE.

!  complete A

            DO i = 1, dims%nc
              A_sbls%val( A_ne + i ) = - SCALE_C( dims%c_equality + i )
            END DO
            SBLS_control%new_a = 1

!  Include the values of the barrier terms

            H_sbls%val( 1 : dims%x_free ) = zero
            H_sbls%val(dims%x_free + 1 : n ) = BARRIER_X
            H_sbls%val( dims%c_s : dims%c_e ) = BARRIER_C

! ::::::::::::::::::::::::::::::
!  Factorize the required matrix
! ::::::::::::::::::::::::::::::

            IF ( get_factors ) THEN
              CALL CLOCK_time( clock_record )
              IF ( printw ) WRITE( out, "( A, ' .........',                    &
             &  ' factorization of KKT matrix ...............' )" ) prefix
              CALL SBLS_form_and_factorize( A_sbls%n, A_sbls%m, H_sbls,        &
                A_sbls, C_sbls, sbls_data, SBLS_control, inform%SBLS_inform )
              inform%time%analyse = inform%time%analyse +                      &
                inform%SBLS_inform%SLS_inform%time%analyse
              inform%time%clock_analyse = inform%time%clock_analyse +          &
                inform%SBLS_inform%SLS_inform%time%clock_analyse
              inform%time%factorize = inform%time%factorize +                  &
                inform%SBLS_inform%SLS_inform%time%factorize
              inform%time%clock_factorize = inform%time%clock_factorize +      &
                inform%SBLS_inform%SLS_inform%time%clock_factorize
              IF ( printw ) WRITE( out, "( A,                                  &
             & ' ............... end of factorization ...............')") prefix

!  Record the storage required

              inform%nfacts = inform%nfacts + 1
              inform%factorization_integer =                                   &
                inform%SBLS_inform%SLS_inform%integer_size_necessary
              inform%factorization_real =                                      &
                inform%SBLS_inform%SLS_inform%real_size_necessary

!  Test that the factorization succeeded

              inform%factorization_status = inform%SBLS_inform%status
              IF ( inform%factorization_status < 0 ) THEN
                IF ( printe ) WRITE( error, 2040 ) prefix,                     &
                  inform%factorization_status, 'SBLS_form_and_factorize'

!  It didn't. We might have run out of options

                IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
                  inform%status = GALAHAD_error_factorization ; GO TO 700

!  ... or we may change the method

                ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
                  pivot_tol = relative_pivot_tol
                  maxpiv = pivot_tol >= half
                  SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
                  SBLS_control%factorization = 2
                  IF ( printi ) WRITE( out,                                    &
                    "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

                ELSE
                  pivot_tol = half
                  maxpiv = .TRUE.
                  SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
                  SBLS_control%factorization = 2
                  IF ( printi )                                                &
                    WRITE( out, "( A, ' Pivot tolerance increased' )" ) prefix
                END IF
                alpha = zero
                IF ( printi ) WRITE( out, 2000 ) prefix
                CYCLE

!  Record warning conditions

              ELSE
                IF (inform%factorization_status > 0 ) THEN
                  IF ( printt ) THEN
                    WRITE( out, 2050 ) prefix, inform%SBLS_inform%status,      &
                                       'SBLS_form_andfactorize'
                  END IF
                END IF
              END IF
              SBLS_control%new_h = 1
              SBLS_control%new_a = 0
              SBLS_control%new_c = 0
            ELSE
              inform%factorization_integer = 0
              inform%factorization_real = 0
            END IF
            CALL CLOCK_time( clock_now )

            IF ( printt ) THEN
              WRITE( out, "( A, ' ** factorize time = ', F10.2 ) " ) prefix,   &
                clock_now - clock_record
              WRITE( out, 2060 ) prefix, inform%factorization_integer,         &
                                 inform%factorization_real
            END IF
          ELSE
            re = ' '
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

          IF ( printd ) WRITE( out, 2120 ) ' GRAD_L ',                         &
            GRAD_L( dims%x_s : dims%x_e )


!       A^T u + z - g
!       u - y
! 0 =   Ax - c
!      (x + perturb_x).(z + perturb_z) - mu_x
!      (c + perturb_c).(y + perturb_y) - mu_c

! DX = diag(x + perturb_x) (etc)

! Newton

!  (      I  A^T         ) ( dx )   ( g - A^T u - z  )
!  (          I   -I     ) ( dz )   ( y - u          )
!  ( A                -I ) ( du ) = ( c - Ax         )
!  ( DZ  DX              ) ( dy )   ( mu_x e - DX DZ e )
!  (              DC  DY ) ( dc )   ( mu_c e - DC DY e )

!  remove dz and dy

! DX dz = mu_x e - DZ DX e -    DZ dx
! dz = DX(inv) ( mu_x e- DZ DX e - DZ dx )
! dy = DC(inv) ( mu_c e - DY DC e - DY dc )

! =>

! A^T du + DX(inv) ( mu_x e - DZ DX e - DZ dx ) = g - A^T u - z
! =>
! - A^T du + DX(inv) DZ dx = - ( g - A^T u - mu_x DX(inv) e + perturb_z )

! du - DC(inv) ( mu_c e - DY DC e - DY dc ) = y - u
! =>
!  du + DC(inv) DY dc = - ( u + mu_c DC(inv) e + perturb_y )


! ( DX(inv) DZ            A^T ) ( dx )     (g-A^T u - mu_x DX(inv)e + perturb_z)
! (            DC(inv) DY  -I ) ( dc ) = - (    u + mu_c DC(inv)e + perturb_y  )
! (      A        -I          ) (-du )     (          A x - c                  )

!  Problem variables:

          RHS( : dims%x_free ) = - GRAD_L( : dims%x_free )
          DO i = dims%x_free + 1, dims%x_u_start - 1
            RHS( i ) = - GRAD_L( i ) + MU_X_l( i ) / DIST_X_l( i )             &
                       - PERTURB_Z_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            RHS( i ) = - GRAD_L( i ) + MU_X_l( i ) / DIST_X_l( i )             &
                       - MU_X_u( i ) / DIST_X_u( i )                           &
                       - PERTURB_Z_l( i ) + PERTURB_Z_u( i )
          END DO
          DO i = dims%x_l_end + 1, n
            RHS( i ) = - GRAD_L( i ) - MU_X_u( i ) / DIST_X_u( i )             &
                       + PERTURB_Z_u( i )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            RHS( dims%c_b + i ) = - Y( i ) + MU_C_l( i ) / DIST_C_l( i )       &
                                  - PERTURB_Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            RHS( dims%c_b + i ) = - Y( i ) + MU_C_l( i ) / DIST_C_l( i )       &
              - MU_C_u( i ) / DIST_C_u( i )                                    &
              - PERTURB_Y_l( i )  + PERTURB_Y_u( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            RHS( dims%c_b + i ) = - Y( i ) - MU_C_u( i ) / DIST_C_u( i )       &
                                           + PERTURB_Y_u( i )
          END DO

!  Include the constraint infeasibilities

          RHS( dims%y_s : dims%y_e ) = - C_RES
          DELTA = RHS
!IF ( printt ) WRITE( out, "( '  c_res ', ES12.4 )" )  MAXVAL( ABS( C_RES ) )

          IF ( printd ) THEN
            WRITE( out, 2120 ) ' RHS_x ', RHS( dims%x_s : dims%c_e )
            IF ( m > 0 ) WRITE( out, 2120 ) ' RHS_y ', RHS( dims%y_s : dims%y_e)
          END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the primal-dual direction for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Solve  ( H  A^T ) ( Dx^pd ) = - ( grad b )
!         ( A   0  ) ( Dy^pd )     (   r    )

          IF ( printw ) WRITE( out,                                            &
             "( A, ' ............... compute step  ...............' )" ) prefix

!  Use a direct method

          DELTA( : A_sbls%n + A_sbls%m ) = RHS( : A_sbls%n + A_sbls%m )

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls,                 &
                           sbls_data, SBLS_control, inform%SBLS_inform, DELTA )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          time_solve = time_solve + time_now - time_record
          clock_solve = clock_solve + clock_now - clock_record

          inform%status = inform%SBLS_inform%status
          IF ( inform%status /= GALAHAD_ok ) GO TO 700

          IF ( printt ) WRITE( out, "( A, ' ** solve time = ', F10.2 ) " )     &
            prefix, clock_now - clock_record
          IF ( printd ) THEN
            WRITE( out, 2120 ) ' SOL_x ', DELTA( dims%x_s : dims%c_e )
            IF ( m > 0 ) WRITE( out, 2120 ) ' SOL_y ', DELTA( dims%y_s:dims%y_e)
          END IF

!C_RES = zero
!CALL WCP_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n,  &
!             DELTA( dims%x_s : dims%x_e ), '+ ' )
!IF ( printt ) WRITE( out, "( '  A dx ', ES12.4 )" )  MAXVAL( ABS( C_RES ) )

!  Compute the residual of the linear system

          IF ( m > 0 ) THEN
            CALL WCP_residual( dims, n, m, dims%v_e,                           &
                               A_ne, A_val, A_col, A_ptr,                      &
                               DELTA( dims%x_s : dims%x_e ),                   &
                               DELTA( dims%c_s : dims%c_e ),                   &
                               DELTA( dims%y_s : dims%y_e ),                   &
                               RHS( dims%x_s : dims%x_e ),                     &
                               RHS( dims%c_s : dims%c_e ),                     &
                               RHS( dims%y_s : dims%y_e ),                     &
                               HX( : dims%v_e ),                               &
                               zero, BARRIER_X, BARRIER_C, SCALE_C,            &
                               errorg, errorc, print_level, control )
          ELSE
            CALL WCP_residual_unconstrained( dims, n, dims%v_e,                &
                                             DELTA( dims%x_s : dims%x_e ),     &
                                             RHS( dims%x_s : dims%x_e ),       &
                                             HX( : dims%v_e ),                 &
                                             zero, BARRIER_X,                  &
                                             errorg, print_level, control )
            errorc = zero
          END IF

!  If the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely.

          IF ( SQRT( SUM( ( HX( : dims%v_e ) - RHS ) ** 2 ) ) > merit ) THEN

!  It wasn't. We might have run out of options ...

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
              IF ( printi )                                                    &
                WRITE( out, "( A, ' Pivot tolerance increased' )" ) prefix
            END IF
            alpha = zero
            CYCLE
          END IF

          IF ( printw ) WRITE( out,                                            &
               "( A, ' ............... step computed ...............')" ) prefix

          IF ( printd ) THEN
            WRITE( out, 2120 ) ' DX ', DELTA( dims%x_s : dims%x_e )
            IF ( m > 0 ) WRITE( out, 2120 ) ' DY ', DELTA( dims%y_s : dims%y_e )
          END IF

!  =======
!  STEP 2:
!  =======

! ::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the search directions for the dual variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Problem variables:

!         l = 0
          DO i = dims%x_free + 1, dims%x_l_end
            DZ_l( i ) =   ( MU_X_l( i ) - DIST_Z_l( i ) *                      &
                             ( DIST_X_l( i ) + DELTA( i ) ) ) / DIST_X_l( i )
!           IF ( ABS( one + DELTA( i ) / DIST_X_l( i ) ) < 0.001 ) l = l + 1
          END DO

          DO i = dims%x_u_start, n
            DZ_u( i ) = - ( MU_X_u( i ) - DIST_Z_u( i ) *                      &
                             ( DIST_X_u( i ) - DELTA( i ) ) ) / DIST_X_u( i )
!           IF ( ABS( one - DELTA( i ) / DIST_X_u( i ) ) < 0.001 ) l = l + 1
          END DO

!         write(6,*) l, ' degenerate variable(s)'

!  Slack variables:

          DO i = dims%c_l_start, dims%c_l_end
            DY_l( i ) =   ( MU_C_l( i ) - DIST_Y_l( i ) *                      &
                     ( DIST_C_l( i ) + DELTA( dims%c_b + i ) ) ) / DIST_C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            DY_u( i ) = - ( MU_C_u( i ) - DIST_Y_u( i ) *                      &
                     ( DIST_C_u( i ) - DELTA( dims%c_b + i ) ) ) / DIST_C_u( i )
          END DO

          IF ( printd ) THEN
            WRITE( out, 2120 ) ' DZ_l ', DZ_l( dims%x_free + 1 : dims%x_l_end )
            WRITE( out, 2120 ) ' DZ_u ', DZ_u( dims%x_u_start : n )
          END IF

!  Calculate the norm of the search direction

          pmax = MAX( MAXVAL( ABS( DELTA( dims%x_s : dims%x_e ) ) ),           &
                      MAXVAL( ABS( DELTA( dims%c_s : dims%c_e ) ) ),           &
                      MAXVAL( ABS( DELTA( dims%y_s : dims%y_e ) ) ),           &
                      MAXVAL( ABS( DZ_l( dims%x_free + 1 : dims%x_l_end ) ) ), &
                      MAXVAL( ABS( DZ_u( dims%x_u_start  : n ) ) ),            &
                      MAXVAL( ABS( DY_l( dims%c_l_start  : dims%c_l_end ) ) ), &
                      MAXVAL( ABS( DY_u( dims%c_u_start  : dims%c_u_end ) ) ) )

          IF ( printp ) WRITE( out, 2140 ) pmax

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
            DO i = dims%x_free + 1, dims%x_u_start - 1
              RHS( i ) = - DELTA( i ) * DZ_l( i ) / DIST_X_l( i )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              RHS( i ) = - DELTA( i ) * DZ_l( i ) / DIST_X_l( i )              &
                         + DELTA( i ) * DZ_u( i ) / DIST_X_u( i )
            END DO
            DO i = dims%x_l_end + 1, n
              RHS( i ) =   DELTA( i ) * DZ_u( i ) / DIST_X_u( i )
            END DO

!  Slack variables:

            DO i = dims%c_l_start, dims%c_u_start - 1
              RHS( dims%c_b + i ) =                                            &
                - DELTA( dims%c_b + i ) * DY_l( i ) / DIST_C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              RHS( dims%c_b + i ) =                                            &
                - DELTA( dims%c_b + i ) * DY_l( i ) / DIST_C_l( i )            &
                + DELTA( dims%c_b + i ) * DY_u( i ) / DIST_C_u( i )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              RHS( dims%c_b + i ) =                                            &
                  DELTA( dims%c_b + i ) * DY_u( i ) / DIST_C_u( i )
            END DO

!  Include the constraint infeasibilities

            RHS( dims%y_s : dims%y_e ) = zero
            DELTA_cor = RHS

            IF ( printd ) THEN
              WRITE( out, 2120 ) ' RHS_cor_x ', RHS( dims%x_s : dims%x_e )
              IF ( m > 0 )                                                     &
                WRITE( out, 2120 ) ' RHS_cor_y ', RHS( dims%y_s : dims%y_e )
            END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the corrector direction for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

            IF ( printw ) WRITE( out,                                          &
              "( A, ' ............... compute step  ...............' )" ) prefix

!  Use a direct method

            CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SBLS_solve( A_sbls%n, A_sbls%m, A_sbls, C_sbls,               &
                             sbls_data, SBLS_control, inform%SBLS_inform,      &
                             DELTA_cor )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            time_solve = time_solve + time_now - time_record
            clock_solve = clock_solve + clock_now - clock_record

            inform%status = inform%SBLS_inform%status
            IF ( inform%status /= GALAHAD_ok ) GO TO 700

            IF ( printt ) WRITE( out, "( A, ' ** solve time = ', F10.2 ) " )   &
              prefix, clock_now - clock_record

!  Compute the residual of the linear system

            CALL WCP_residual( dims, n, m, dims%v_e, A_ne, A_val, A_col, A_ptr,&
                               DELTA_cor( dims%x_s : dims%x_e ),               &
                               DELTA_cor( dims%c_s : dims%c_e ),               &
                               DELTA_cor( dims%y_s : dims%y_e ),               &
                               RHS( dims%x_s : dims%x_e ),                     &
                               RHS( dims%c_s : dims%c_e ),                     &
                               RHS( dims%y_s : dims%y_e ),                     &
                               HX( : dims%v_e ), zero, BARRIER_X, BARRIER_C,   &
                               SCALE_C, errorg, errorc, print_level, control )

!  If the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely. Exit

            IF ( SQRT( SUM( ( HX( : dims%v_e ) - RHS ) ** 2 ) ) > merit ) THEN

!  It didn't. We might have run out of options

              IF ( SBLS_control%factorization == 2 .AND. maxpiv ) THEN
                inform%status = GALAHAD_error_ill_conditioned ; GO TO 600

!  ... or we may change the method

              ELSE IF ( SBLS_control%factorization < 2 .AND. maxpiv ) THEN
                pivot_tol = relative_pivot_tol
                maxpiv = pivot_tol >= half
                SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
                SBLS_control%factorization = 2
                IF ( printi ) WRITE( out,                                      &
                  "( A, ' Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

              ELSE
                pivot_tol = half
                maxpiv = .TRUE.
                SBLS_control%sls_control%relative_pivot_tolerance = pivot_tol
                SBLS_control%factorization = 2
                IF ( printi )                                                  &
                  WRITE( out, "( A, ' Pivot tolerance increased ' )" ) prefix
              END IF
              alpha = zero
              CYCLE
            END IF

            IF ( printw ) WRITE( out,                                          &
              "( A, ' ............... step computed ...............' )" ) prefix

            IF ( printd ) THEN
              WRITE( out, 2120 ) ' DX_cor ', DELTA_cor( dims%x_s : dims%x_e )
              IF ( m > 0 )                                                     &
                WRITE( out, 2120 ) ' DY_cor ', DELTA_cor( dims%y_s : dims%y_e )
            END IF

!  ========
!  STEP 2b:
!  ========

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Obtain the corrector search directions for the dual variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  Problem variables:

            DO i = dims%x_free + 1, dims%x_l_end
              DZ_cor_l( i ) = - ( DZ_l( i ) * DELTA( i ) +                     &
                              DIST_Z_l( i ) * DELTA_cor( i ) ) / DIST_X_l( i )
            END DO

            DO i = dims%x_u_start, n
              DZ_cor_u( i ) =   ( DZ_u( i ) * DELTA( i ) -                     &
                              DIST_Z_u( i ) * DELTA_cor( i ) ) / DIST_X_u( i )
            END DO

!  Slack variables:

            DO i = dims%c_l_start, dims%c_l_end
              DY_cor_l( i ) = - ( DY_l( i ) * DELTA( dims%c_b + i ) +          &
                DIST_Y_l( i ) * DELTA_cor( dims%c_b + i ) ) / DIST_C_l( i )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              DY_cor_u( i ) =   ( DY_u( i ) * DELTA( dims%c_b + i ) -          &
                DIST_Y_u( i ) * DELTA_cor( dims%c_b + i ) ) / DIST_C_u( i )
            END DO

            IF ( printd ) THEN
              WRITE( out, 2120 ) ' DZ_cor_l ',                                 &
                DZ_cor_l( dims%x_free + 1 : dims%x_l_end )
              WRITE( out, 2120 ) ' DZ_cor_u ', DZ_cor_u( dims%x_u_start : n )
            END IF

!  Calculate the norm of the search direction

            pmax_cor = MAX( MAXVAL( ABS( DELTA_cor( dims%x_s : dims%x_e ) ) ), &
              MAXVAL( ABS( DELTA_cor( dims%c_s : dims%c_e ) ) ),               &
              MAXVAL( ABS( DELTA_cor( dims%y_s : dims%y_e ) ) ),               &
              MAXVAL( ABS( DZ_cor_l( dims%x_free + 1 : dims%x_l_end ) ) ),     &
              MAXVAL( ABS( DZ_cor_u( dims%x_u_start  : n ) ) ),                &
              MAXVAL( ABS( DY_cor_l( dims%c_l_start  : dims%c_l_end ) ) ),     &
              MAXVAL( ABS( DY_cor_u( dims%c_u_start  : dims%c_u_end ) ) ) )

            IF ( printp ) WRITE( out, 2160 ) pmax_cor
          END IF

!  Check to see whether to use a corrector step based on the relative
!  sizes of the predictor and corrector

          IF ( control%use_corrector ) THEN
!           IF ( pmax_cor < ten * pmax ) THEN
              use_corrector = .TRUE.
!           ELSE
!             use_corrector = .FALSE.
!           END IF
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

          IF ( printw ) WRITE( out,                                            &
            "( A, ' .............. get steplength  ..............' )" ) prefix

!  Form the vector H dx + A(trans) dy

          HX( dims%x_s : dims%x_e ) = zero

          IF ( m > 0 ) CALL WCP_AX( n, HX( : n ), m, A_ne, A_val, A_col,       &
                                  A_ptr, m, DELTA( dims%y_s : dims%y_e ), '+T' )

          IF ( use_corrector ) THEN

!  If a fixed point on the central path is required, perform a safeguarded
!  global linesearch

            CALL WCP_min_piecewise_quartic( dims, n, m, nbnds, DIST_Z_l,       &
                                            DIST_Z_u, DZ_l, DZ_u, DZ_cor_l,    &
                                            DZ_cor_u, DIST_X_l, DIST_X_u,      &
                                            DELTA( dims%x_s : dims%x_e ),      &
                                            DELTA_cor( dims%x_s : dims%x_e ),  &
                                            DIST_Y_l, DIST_Y_u, DY_l, DY_u,    &
                                            DY_cor_l,                          &
                                            DY_cor_u, DIST_C_l, DIST_C_u,      &
                                            DELTA( dims%c_s : dims%c_e ),      &
                                            DELTA_cor( dims%c_s : dims%c_e ),  &
                                            res_prim_dual,                     &
                                            MU_X_l, MU_X_u, MU_C_l, MU_C_u,    &
                                            MU, omega_l, omega_u,              &
                                            alpha, alpha_max,                  &
                                            COEF4, COEF3, COEF1, COEF0,        &
                                            BREAKP, IBREAK, print_level,       &
                                            control )
            one_minus_alpha = one - alpha

!  Calculate the distances to the bounds and the dual variables at the
!  new point

            X = X + alpha * ( DELTA( dims%x_s : dims%x_e ) +                   &
                              alpha * DELTA_cor( dims%x_s : dims%x_e ) )
            Y = Y - alpha * ( DELTA( dims%y_s : dims%y_e ) +                   &
                              alpha * DELTA_cor( dims%y_s : dims%y_e ) )

            DO i = dims%x_free + 1, dims%x_l_end
              DIST_X_l( i ) = DIST_X_l( i ) +                                  &
                               alpha * ( DELTA( i ) + alpha * DELTA_cor( i ) )
              Z_l( i ) = Z_l( i ) + alpha * ( DZ_l( i ) + alpha * DZ_cor_l( i ))
              DIST_Z_l( i ) = DIST_Z_l( i ) +                                  &
                               alpha * ( DZ_l( i ) + alpha * DZ_cor_l( i ) )
            END DO

            DO i = dims%x_u_start, n
              DIST_X_u( i ) = DIST_X_u( i ) -                                  &
                                alpha * ( DELTA( i ) + alpha * DELTA_cor( i ) )
              Z_u( i ) = Z_u( i ) + alpha * ( DZ_u( i ) + alpha * DZ_cor_u( i ))
              DIST_Z_u( i ) = DIST_Z_u( i ) -                                  &
                                alpha * ( DZ_u( i ) + alpha * DZ_cor_u( i ) )
            END DO

!  Do the same for the slacks and their duals

            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l( i ) = DIST_C_l( i ) + alpha * ( DELTA( dims%c_b + i )  &
                               + alpha * DELTA_cor( dims%c_b + i ) )
              Y_l( i ) = Y_l( i ) + alpha * ( DY_l( i ) + alpha * DY_cor_l( i ))
              DIST_Y_l( i ) = DIST_Y_l( i ) +                                  &
                                 alpha * ( DY_l( i ) + alpha * DY_cor_l( i ) )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              DIST_C_u( i ) = DIST_C_u( i ) - alpha * ( DELTA( dims%c_b + i )  &
                              + alpha * DELTA_cor( dims%c_b + i ) )
              Y_u( i ) = Y_u( i ) + alpha * ( DY_u( i ) + alpha * DY_cor_u( i ))
              DIST_Y_u( i ) = DIST_Y_u( i ) -                                  &
                                alpha * ( DY_u( i ) + alpha * DY_cor_u( i ) )
            END DO
          ELSE

!  Perform a safeguarded global linesearch to find the step length

            IF ( m > 0 ) THEN
              CALL WCP_min_piecewise_quadratic( dims, n, m, nbnds, DIST_Z_l,   &
                                                DIST_Z_u, DZ_l, DZ_u, DIST_X_l,&
                                                DIST_X_u,                      &
                                                DELTA( dims%x_s : dims%x_e ),  &
                                                DIST_Y_l, DIST_Y_u, DY_l, DY_u,&
                                                DIST_C_l, DIST_C_u,            &
                                                DELTA( dims%c_s : dims%c_e ),  &
                                                res_prim_dual,                 &
                                                MU_X_l, MU_X_u, MU_C_l, MU_C_u,&
                                                MU, omega_l, omega_u,          &
                                                alpha, alpha_est, alpha_max,   &
                                                COEF2, COEF1, COEF0, BREAKP,   &
                                                IBREAK, print_level, control )
            ELSE
              CALL WCP_min_piecewise_quadratic( dims, n, m, nbnds, DIST_Z_l,   &
                                                DIST_Z_u, DZ_l, DZ_u, DIST_X_l,&
                                                DIST_X_u,                      &
                                                DELTA( dims%x_s : dims%x_e ),  &
                                                DIST_Y_l, DIST_Y_u, DY_l, DY_u,&
                                                DIST_C_l, DIST_C_u,            &
                                                DELTA( 1 : 0 ),                &
                                                res_prim_dual,                 &
                                                MU_X_l, MU_X_u, MU_C_l, MU_C_u,&
                                                MU, omega_l, omega_u,          &
                                                alpha, alpha_est, alpha_max,   &
                                                COEF2, COEF1, COEF0, BREAKP,   &
                                                IBREAK, print_level, control )
            END IF
            one_minus_alpha = one - alpha

!  Calculate the distances to the bounds and the dual variables at the
!  new point

            X = X + alpha * DELTA( dims%x_s : dims%x_e )
            Y = Y - alpha * DELTA( dims%y_s : dims%y_e )

            DO i = dims%x_free + 1, dims%x_l_end
              DIST_X_l( i ) = DIST_X_l( i ) + alpha * DELTA( i )
              Z_l( i ) = Z_l( i ) + alpha * DZ_l( i )
              DIST_Z_l( i ) = DIST_Z_l( i ) + alpha * DZ_l( i )
            END DO

            DO i = dims%x_u_start, n
              DIST_X_u( i ) = DIST_X_u( i ) - alpha * DELTA( i )
              Z_u( i ) = Z_u( i ) + alpha * DZ_u( i )
              DIST_Z_u( i ) = DIST_Z_u( i ) - alpha * DZ_u( i )
            END DO

!  Do the same for the slacks and their duals

            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l( i ) = DIST_C_l( i ) + alpha * DELTA( dims%c_b + i )
              Y_l( i ) = Y_l( i ) + alpha * DY_l( i )
              DIST_Y_l( i ) = DIST_Y_l( i ) + alpha * DY_l( i )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              DIST_C_u( i ) = DIST_C_u( i ) - alpha * DELTA( dims%c_b + i )
              Y_u( i ) = Y_u( i ) + alpha * DY_u( i )
              DIST_Y_u( i ) = DIST_Y_u( i ) - alpha * DY_u( i )
            END DO
          END IF

          IF ( alpha == alpha_max ) THEN
            al = 'b'
          ELSE
            al = ' '
          END IF

!  Update the slack variables

          IF ( use_corrector ) THEN
            C = C + alpha * ( DELTA( dims%c_s : dims%c_e ) +                   &
                  alpha * DELTA_cor( dims%c_s : dims%c_e ) )
          ELSE
            C = C + alpha * DELTA( dims%c_s : dims%c_e )
          END IF

!  Update the values of the merit function, the gradient of the Lagrangian,
!  and the constraint residuals

          GRAD_L( dims%x_s : dims%x_e ) = GRAD_L( dims%x_s : dims%x_e ) +      &
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

!         WRITE( 6, "(' updated  grad_l ', ES12.4 )" )                         &
!           MAXVAL( ABS( GRAD_L( dims%x_s : dims%x_e ) ) )
!         CALL WCP_Lagrangian_gradient( n, m, Y, A_ne, A_val, A_col,           &
!                                       A_ptr, GRAD_L( dims%x_s : dims%x_e ),  &
!                                       gradient_kind, G = G )
!          WRITE( 6, "(' computed grad_l ', ES12.4 )" )                        &
!           MAXVAL( ABS( GRAD_L( dims%x_s : dims%x_e ) ) )

!  Update the norm of the constraint residual

          res_prim = one_minus_alpha * res_prim
!         nu = one_minus_alpha * nu

!  Evaluate the merit function if not already done

          merit = WCP_merit_value( dims, n, m, Y, Y_l, DIST_Y_l, Y_u,          &
                                   DIST_Y_u, Z_l, DIST_Z_l, Z_u, DIST_Z_u,     &
                                   DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,     &
                                   GRAD_L( dims%x_s : dims%x_e ),              &
                                   C_RES, res_dual,                            &
                                   MU_X_l, MU_X_u, MU_C_l, MU_C_u )

!         write(6,"(' res_pd ', ES12.4 )") res_prim_dual * one_minus_alpha
          res_prim_dual = SUM( ABS( C_RES ) ) + res_dual

!         write(6,"( ' res_prim , ABS(C) ', 2ES12.4 )" )                       &
!           res_prim, SUM( ABS( C_RES ) )
!         write(6,"(' res_pd,, res_cs ', ES12.4 )") res_prim_dual
          IF ( printw ) write( out,"( ' step, merit = ', 2ES12.4 )" )          &
            alpha, merit

!         min_mu = MIN( MINVAL( MU_X_l( dims%x_free + 1 : dims%x_l_end ) ),    &
!                       MINVAL( MU_X_u( dims%x_u_start : n ) ),                &
!                       MINVAL( MU_C_l( dims%c_l_start : dims%c_l_end ) ),     &
!                       MINVAL( MU_C_u( dims%c_u_start : dims%c_u_end ) ) )
!         WRITE( 6, "( ' dm, bound ', 2ES12.4 )" ) old_merit - merit,          &
!           half * omega_l * min_mu * MIN( half,                               &
!             ( one - omega_l ) * min_mu / old_merit )
!         old_merit = merit

!  Compute the objective function value

          inform%obj = f

          IF ( gradient_kind == 1 ) THEN
            inform%obj = inform%obj + SUM( X )
          ELSE IF ( gradient_kind /= 0 ) THEN
            inform%obj = inform%obj + DOT_PRODUCT( G, X )
          END IF

!         write(6,"( 5ES12.4 )" ) C_RES( 1 : 5 )
          IF ( m > 0 ) THEN

            C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
            C_RES( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
            CALL WCP_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
            IF ( printt ) WRITE( out, "( '  Constraint residual ', ES12.4 )" ) &
              MAXVAL( ABS( C_RES ) )
!!           WRITE( 6, "( ' rec, cal cres = ', 2ES12.4 )" )                    &
!!             res_prim, MAXVAL( ABS( C_RES ) )
!            IF ( res_prim < MAXVAL( ABS( C_RES ) ) ) THEN
              res_prim = MAXVAL( ABS( C_RES ) )
!           END IF
          END IF
!         write(6,"( 5ES12.4 )" ) C_RES( 1 : 5 )

!         DO i = dims%x_free + 1, dims%x_l_end
!           write(6,"(I6, ' x lower', 2ES12.4)" ) i, DIST_X_l( i ), DIST_Z_l( i)
!         END DO
!         DO i = dims%x_u_start, n
!           write(6,"(I6, ' x upper', 2ES12.4)" ) i, DIST_X_u( i ), DIST_Z_u( i)
!         END DO
!         DO i = dims%c_l_start, dims%c_l_end
!           write(6,"(I6, ' c lower', 2ES12.4)" ) i, DIST_C_l( i ), DIST_Y_l( i)
!         END DO
!         DO i = dims%c_u_start, dims%c_u_end
!           write(6,"(I6, ' c upper', 2ES12.4)" ) i, DIST_C_u( i ), DIST_Y_u( i)
!         END DO

!  Compute the complementary slackness, and the min/max components
!  of the primal/dual infeasibilities

!         IF ( res_prim <= control%stop_p .AND. res_dual <= control%stop_d     &
          IF ( res_prim <= tenm4 .AND. res_dual <= tenm3                       &
               .AND. reset_mu .AND. cs_bad == 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' resetting mu ')" ) prefix
            reset_mu = .FALSE.
            DO i = dims%x_free + 1, dims%x_l_end
              MU_X_l( i ) = DIST_X_l( i ) * DIST_Z_l( i )
            END DO
            DO i = dims%x_u_start, n
              MU_X_u( i ) = DIST_X_u( i ) * DIST_Z_u( i )
            END DO
            DO i = dims%c_l_start, dims%c_l_end
              MU_C_l( i ) = DIST_C_l( i ) * DIST_Y_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              MU_C_u( i ) = DIST_C_u( i ) * DIST_Y_u( i )
            END DO
          END IF

          IF ( printt .AND. nbnds > 0 ) THEN
            slknes_x = DOT_PRODUCT( DIST_X_l( dims%x_free + 1 : dims%x_l_end ),&
                                    DIST_Z_l( dims%x_free + 1 : dims%x_l_end ))&
                     + DOT_PRODUCT( DIST_X_u( dims%x_u_start : n ),            &
                                    DIST_Z_u( dims%x_u_start : n ) )
            slknes_c = DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ), &
                                    DIST_Y_l( dims%c_l_start : dims%c_l_end ) )&
                     + DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ), &
                                    DIST_Y_u( dims%c_u_start : dims%c_u_end ) )
            slknes = slknes_x + slknes_c

            slkmin_x = MIN( MINVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end )*&
                                    DIST_Z_l( dims%x_free + 1 : dims%x_l_end)),&
                            MINVAL( DIST_X_u( dims%x_u_start : n ) *           &
                                    DIST_Z_u( dims%x_u_start : n ) ) )
            slkmin_c = MIN( MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *&
                                    DIST_Y_l( dims%c_l_start : dims%c_l_end )),&
                            MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end)*  &
                                    DIST_Y_u( dims%c_u_start : dims%c_u_end ) ))
            slkmin = MIN( slkmin_x, slkmin_c )

            slkmax_x = MAX( MAXVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end )*&
                                    DIST_Z_l( dims%x_free + 1 : dims%x_l_end)),&
                            MAXVAL( DIST_X_u( dims%x_u_start : n ) *           &
                                    DIST_Z_u( dims%x_u_start : n ) ) )
            slkmax_c = MAX( MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *&
                                    DIST_Y_l( dims%c_l_start : dims%c_l_end )),&
                            MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end)*  &
                                    DIST_Y_u( dims%c_u_start : dims%c_u_end ) ))

            p_min = MIN( MINVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) ), &
                         MINVAL( DIST_X_u( dims%x_u_start : n ) ),             &
                         MINVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),  &
                         MINVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

            p_max = MAX( MAXVAL( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) ), &
                         MAXVAL( DIST_X_u( dims%x_u_start : n ) ),             &
                         MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) ),  &
                         MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) ) )

            d_min = MIN( MINVAL( DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ), &
                         MINVAL( DIST_Z_u( dims%x_u_start : n ) ),             &
                         MINVAL( DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),  &
                         MINVAL( DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )

            d_max = MAX( MAXVAL( DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) ), &
                         MAXVAL( DIST_Z_u( dims%x_u_start : n ) ),             &
                         MAXVAL( DIST_Y_l( dims%c_l_start : dims%c_l_end ) ),  &
                         MAXVAL( DIST_Y_u( dims%c_u_start : dims%c_u_end ) ) )

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
            ELSE
              slknes = zero
            END IF

            WRITE( out, 2130 ) prefix, slknes, prefix, slknes_x, prefix,       &
              slknes_c, prefix, slkmin_x, slkmax_x, prefix,                    &
              slkmin_c, slkmax_c, prefix, p_min, p_max, prefix, d_min, d_max
          END IF

          slknes_req = SUM( ABS( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) *  &
                                 DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) -  &
                                 MU_X_l( dims%x_free + 1 : dims%x_l_end ) ) )  &
                     + SUM( ABS( DIST_X_u( dims%x_u_start : n ) *              &
                                 DIST_Z_u( dims%x_u_start : n ) -              &
                                 MU_X_u( dims%x_u_start : n ) ) )              &
                     + SUM( ABS( DIST_C_l( dims%c_l_start : dims%c_l_end ) *   &
                                 DIST_Y_l( dims%c_l_start : dims%c_l_end ) -   &
                                 MU_C_l( dims%c_l_start : dims%c_l_end ) ) )   &
                     + SUM( ABS( DIST_C_u( dims%c_u_start : dims%c_u_end ) *   &
                                 DIST_Y_u( dims%c_u_start : dims%c_u_end ) -   &
                                 MU_C_u( dims%c_u_start : dims%c_u_end ) ) )

          IF ( printd ) THEN
            WRITE( out, "( ' primal-dual -vs- primal dual variables ' )" )
            WRITE( out, "( ' lower ', /, ( 2( I6, 2ES12.4 ) ) )" )             &
              ( i, DIST_Z_l( i ), MU_X_l( i ) / DIST_X_l( i ),                 &
                i =  dims%x_free + 1, dims%x_l_end )
            WRITE( out, "( ' upper ', /, ( 2( I6, 2ES12.4 ) ) )" )             &
              ( i, DIST_Z_u( i ), - MU_X_u( i ) / DIST_X_u( i ),               &
                i = dims%x_u_start, n )
          END IF

!  Test to see if we are feasible

          IF ( res_prim <= control%stop_p ) THEN
            IF ( control%just_feasible ) THEN
              inform%status = 0
!             inform%feasible = res_dual <= control%stop_d
              IF ( printi ) THEN
                CALL CLOCK_TIME( clock_now )
                clock_now = clock_now - clock_start
                WRITE( out, 2070 ) prefix
                coal = '  ' ; coal = TRIM( co ) // TRIM( al )
                WRITE( out, 2030 ) prefix, inform%iter, re, res_prim,          &
                   res_dual, slknes_req, zero, alpha, coal, mu_target, clock_now
                IF ( printt ) WRITE( out, 2000 ) prefix
              END IF
              GO TO 490
            END IF

            IF ( .NOT. inform%feasible .AND. res_dual <= control%stop_d ) THEN
              IF ( printi ) WRITE( out, 2070 ) prefix
              inform%feasible = .TRUE.
            END IF

!  Check to see if we are feasible

          END IF
          IF ( control%just_feasible .AND. inform%feasible .AND.               &
            MAX( inform%x_implicit, inform%c_implicit,                         &
                 inform%y_implicit, inform%z_implicit ) == 0  ) THEN
            inform%status = 0
            GO TO 500
          END IF

!  =======
!  STEP 5:
!  =======

!  Compute the projected gradient of the Lagrangian function

          pjgnrm = zero

          DO i = 1, dims%x_free
            pjgnrm = MAX( pjgnrm, ABS(  GRAD_L( i ) ) )
          END DO

          DO i = dims%x_free + 1, dims%x_u_start - 1
            gi = GRAD_L( i )
            IF ( gi > zero )                                                   &
              gi = MIN( ABS( X_l( i ) - PERTURB_X_l( i ) - X( i ) ), gi )
            pjgnrm = MAX( pjgnrm, ABS( gi ) )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            gi = GRAD_L( i )
            IF ( gi < zero ) THEN
              gi = - MIN( ABS( X_u( i ) + PERTURB_X_u( i ) - X( i ) ), - gi )
            ELSE
              gi = MIN( ABS( X_l( i ) - PERTURB_X_l( i ) - X( i ) ), gi )
            END IF
            pjgnrm = MAX( pjgnrm, ABS( gi ) )
          END DO

          DO i = dims%x_l_end + 1, n
            gi = GRAD_L( i )
            IF ( gi < zero )                                                   &
              gi = - MIN( ABS( X_u( i ) + PERTURB_X_u( i ) - X( i ) ), - gi )
            pjgnrm = MAX( pjgnrm, ABS( gi ) )
          END DO

          IF ( printd ) THEN
            WRITE( out, 2120 ) ' DIST_X_l ', DIST_X_l
            WRITE( out, 2120 ) ' DIST_X_u ', DIST_X_u
            WRITE( out, "( ' ' )" )
          END IF

          IF ( printd ) WRITE( out, 2110 ) prefix, pjgnrm, prefix, res_prim

        END DO

!  ---------------------------------------------------------------------
!  ---------------------- End of Major Iteration -----------------------
!  ---------------------------------------------------------------------

  490   CONTINUE

!  Compute the values of the constraints

        C_RES( : m ) = zero
        CALL WCP_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )

!  Compute the number of, and largest, constraint violations

        inform%x_implicit =                                                    &
               COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < zero )       &
             + COUNT( X_l( dims%x_l_start : dims%x_l_end ) >                   &
                      X( dims%x_l_start : dims%x_l_end ) )                     &
             + COUNT( X( dims%x_u_start : dims%x_u_end ) >                     &
                      X_u( dims%x_u_start : dims%x_u_end ) )                   &
             + COUNT( X( dims%x_u_end + 1: n ) > zero )
        inform%z_implicit =                                                    &
               COUNT( Z_l( dims%x_free + 1 : dims%x_l_end ) < zero )           &
             + COUNT( Z_u( dims%x_u_start : n ) > zero )
        inform%c_implicit =                                                    &
               COUNT( C_l( dims%c_l_start : dims%c_l_end ) >                   &
                      C_RES( dims%c_l_start : dims%c_l_end ) )                 &
             + COUNT( C_RES( dims%c_u_start : dims%c_u_end ) >                 &
                      C_u( dims%c_u_start : dims%c_u_end ) )
        inform%y_implicit =                                                    &
               COUNT( Y_l( dims%c_l_start : dims%c_l_end ) < zero )            &
             + COUNT( Y_u( dims%c_u_start : dims%c_u_end ) > zero )

        max_xr = MAX( zero,                                                    &
                      MAXVAL( - X( dims%x_free + 1 : dims%x_l_start - 1 ) ),   &
                      MAXVAL( X_l( dims%x_l_start : dims%x_l_end ) -           &
                              X( dims%x_l_start : dims%x_l_end ) ),            &
                      MAXVAL( X( dims%x_u_start : dims%x_u_end ) -             &
                              X_u( dims%x_u_start : dims%x_u_end ) ),          &
                      MAXVAL( X( dims%x_u_end + 1 : n ) ) )
        max_zr = MAX( zero,                                                    &
                      MAXVAL( - Z_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                      MAXVAL( Z_u( dims%x_u_start : n ) ) )
        max_cr = MAX( zero, MAXVAL( C_l( dims%c_l_start : dims%c_l_end ) -     &
                                    C_RES( dims%c_l_start : dims%c_l_end ) ),  &
                            MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -   &
                                    C_u( dims%c_u_start : dims%c_u_end ) ) )
        max_yr = MAX( zero, MAXVAL( - Y_l( dims%c_l_start : dims%c_l_end ) ),  &
                            MAXVAL( Y_u( dims%c_u_start : dims%c_u_end ) ) )

!       DO i = dims%x_free + 1, dims%x_l_end
!         IF ( - Z_l( i ) == max_zr ) WRITE( out, "( I0, ' lower: dual = ',    &
!        &   ES16.8, ' perturb = ', ES16.8 )" ) i, Z_l( i ), PERTURB_Z_l( i )
!         IF ( - Z_l( i ) == max_zr ) WRITE( out, "( I0, ' lower: prim = ',    &
!        &   ES16.8, ' perturb = ', ES16.8 )" ) i, X( i ) - X_l( i ),          &
!          PERTURB_X_l( i )
!       END DO
!       DO i = dims%x_u_start, n
!         IF ( Z_u( i ) == max_zr ) WRITE( out, "( I0, ' upper: dual = ',      &
!        &   ES16.8, ' perturb = ', ES16.8 )" ) i, Z_u( i ), PERTURB_Z_u( i )
!       END DO

        max_r = MAX( max_xr, max_zr, max_cr, max_yr )
!       IF ( MAX( max_xr, max_cr ) <= control%stop_p .AND.                     &
!            MAX( max_zr, max_yr ) <= control%stop_d ) THEN

        IF ( max_r == zero ) THEN
          IF ( now_interior ) THEN
            IF ( printi ) WRITE( out,                                          &
              "( /, A, ' =============== well-centered interior point found',  &
           &        ' ==============' )" ) prefix
            EXIT
          ELSE
            IF ( .NOT. now_feasible ) THEN
              now_feasible = .TRUE.
              IF ( printi ) WRITE( out,                                        &
                "( /, A, ' ====================== interior point found',       &
             &        ' =====================' )" ) prefix
            END IF
          END IF
        ELSE IF ( max_r <= control%implicit_tol .AND.                          &
                  perturb_max <= perturbation_small ) THEN
           IF ( printi ) WRITE( out,                                           &
              "( /, A, ' ============= feasible but not interior point found', &
           &        ' =============' )" ) prefix
            EXIT
        END IF

        IF ( printi ) THEN
          IF ( max_xr > zero .OR. max_zr > zero .OR.                           &
               max_yr > zero .OR. max_yr > zero ) WRITE( out, "( '' )" )
          IF ( max_xr > zero )                                                 &
            WRITE( out, "( A, '  # infeasible variables = ', I0,               &
           &  ', infeasibility =', ES11.4 )" ) prefix, inform%x_implicit, max_xr
          IF ( max_zr > zero )                                                 &
            WRITE( out, "( A, '  # infeasible duals = ', I0,                   &
           &  ', infeasibility =', ES11.4 )" ) prefix, inform%z_implicit, max_zr
          IF ( max_cr > 0 )                                                    &
            WRITE( out, "( A, '  # infeasible constraints = ', I0,             &
           &  ', infeasibility =', ES11.4 )" ) prefix, inform%c_implicit, max_cr
          IF ( max_yr > 0 )                                                    &
            WRITE( out, "( A, '  # infeasible multipliers = ', I0,             &
           &  ', infeasibility =', ES11.4 )" ) prefix, inform%y_implicit, max_yr
        END IF

        one_minus_red_pert_fac = one - reduce_perturb_factor

        IF ( control%perturbation_strategy <= 0 ) THEN
          EXIT

!  Adjust perturb by reducing each relaxtion by the same amount

        ELSE IF ( control%perturbation_strategy == 1 .OR.                      &
                  control%perturbation_strategy == 3 ) THEN
          perturb = reduce_perturb_factor * perturb +                          &
            one_minus_red_pert_fac * max_r
!         write( 6, "( ' max r = ', ES12.4 )" ) max_r
!         write( 6, "( ' new perturb = ', ES12.4 )" ) perturb
!         write( 6, "( ' X = ', / ( 6ES12.4 ) )" ) X( : n )
!         write( 6, "( ' Z = ', / ( 6ES12.4 ) )" ) Z_l( : n )
          PERTURB_X_l = perturb ; PERTURB_X_u = perturb
          PERTURB_C_l = perturb ; PERTURB_C_u = perturb
          PERTURB_Z_l = perturb ; PERTURB_Z_u = perturb
          PERTURB_Y_l = perturb ; PERTURB_Y_u = perturb

!  The variable is a non-negativity

          DO i = dims%x_free + 1, dims%x_l_start - 1
            DIST_X_l( i ) = X( i ) + PERTURB_X_l( i )
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has a lower bound

          DO i = dims%x_l_start, dims%x_l_end
            DIST_X_l( i ) = X( i ) - X_l( i ) + PERTURB_X_l( i )
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has an upper bound

          DO i = dims%x_u_start, dims%x_u_end
            DIST_X_u( i ) = X_u( i ) + PERTURB_X_u( i ) - X( i )
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The variable is a non-positivity

          DO i = dims%x_u_end + 1, n
            DIST_X_u( i ) = PERTURB_X_u( i ) - X( i )
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The constraint has a lower bound

          DO i = dims%c_l_start, dims%c_l_end
            DIST_C_l( i ) = C( i ) - C_l( i ) + PERTURB_C_l( i )
            DIST_Y_l( i ) = Y_l( i ) + PERTURB_Y_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_l( i ), DIST_C_l( i )
          END DO

!  The constraint has an upper bound

          DO i = dims%c_u_start, dims%c_u_end
            DIST_C_u( i ) = C_u( i ) + PERTURB_C_u( i ) - C( i )
            DIST_Y_u( i ) = - Y_u( i ) + PERTURB_Y_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_u( i ), DIST_C_u( i )
          END DO

!  Adjust perturb by reducing each relaxtion as much as possible

        ELSE IF ( control%perturbation_strategy == 5 ) THEN

!  The variable is a non-negativity

          DO i = dims%x_free + 1, dims%x_l_start - 1
            PERTURB_X_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_X_l( i ) -            &
                   one_minus_red_pert_fac * X( i ) )
            DIST_X_l( i ) = X( i ) + PERTURB_X_l( i )
            PERTURB_Z_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Z_l( i ) -            &
                   one_minus_red_pert_fac * Z_l( i ) )
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has a lower bound

          DO i = dims%x_l_start, dims%x_l_end
            PERTURB_X_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_X_l( i ) -            &
                   one_minus_red_pert_fac * ( X( i ) - X_l( i ) ) )
            DIST_X_l( i ) = X( i ) - X_l( i ) + PERTURB_X_l( i )
            PERTURB_Z_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Z_l( i ) -            &
                   one_minus_red_pert_fac * Z_l( i ) )
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has an upper bound

          DO i = dims%x_u_start, dims%x_u_end
            PERTURB_X_u( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_X_u( i ) -            &
                   one_minus_red_pert_fac * ( X_u( i ) - X( i ) ) )
            DIST_X_u( i ) = X_u( i ) + PERTURB_X_u( i ) - X( i )
            PERTURB_Z_u( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Z_u( i ) +            &
                   one_minus_red_pert_fac * Z_u( i ) )
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The variable is a non-positivity

          DO i = dims%x_u_end + 1, n
            PERTURB_X_u( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_X_u( i ) -            &
                   one_minus_red_pert_fac * ( - X( i ) ) )
            DIST_X_u( i ) = PERTURB_X_u( i ) - X( i )
            PERTURB_Z_u( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Z_u( i ) +            &
                                  one_minus_red_pert_fac * Z_u( i ) )
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The constraint has a lower bound

          DO i = dims%c_l_start, dims%c_l_end
            PERTURB_C_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_C_l( i ) -            &
                   one_minus_red_pert_fac * ( C( i ) - C_l( i ) ) )
            DIST_C_l( i ) = C( i ) - C_l( i ) + PERTURB_C_l( i )
            PERTURB_Y_l( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Y_l( i ) -            &
                   one_minus_red_pert_fac * Y_l( i ) )
            DIST_Y_l( i ) = Y_l( i ) + PERTURB_Y_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_l( i ), DIST_C_l( i )
          END DO

!  The constraint has an upper bound

          DO i = dims%c_u_start, dims%c_u_end
            PERTURB_C_u( i ) =                                                 &
               MAX( zero, reduce_perturb_factor * PERTURB_C_u( i ) -           &
                    one_minus_red_pert_fac * ( C_u( i ) - C( i ) ) )
            DIST_C_u( i ) = C_u( i ) + PERTURB_C_u( i ) - C( i )
            PERTURB_Y_u( i ) =                                                 &
              MAX( zero, reduce_perturb_factor * PERTURB_Y_u( i ) +            &
                   one_minus_red_pert_fac * Y_u( i ) )
            DIST_Y_u( i ) = - Y_u( i ) + PERTURB_Y_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_u( i ), DIST_C_u( i )
          END DO

        ELSE

!  The variable is a non-negativity

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( X( i ) <= zero ) THEN
              PERTURB_X_l( i ) = reduce_perturb_factor * PERTURB_X_l( i ) -    &
                one_minus_red_pert_fac * X( i )
            ELSE
              IF ( X( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_X_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_X_l( i )
              ELSE
                PERTURB_X_l( i ) = zero
              END IF
            END IF
            DIST_X_l( i ) = X( i ) + PERTURB_X_l( i )
            IF ( Z_l( i ) <= zero ) THEN
              PERTURB_Z_l( i ) = reduce_perturb_factor * PERTURB_Z_l( i )      &
                - one_minus_red_pert_fac * Z_l( i )
            ELSE
              IF ( Z_l( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_Z_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Z_l( i )
              ELSE
                PERTURB_Z_l( i ) = zero
              END IF
            END IF
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has a lower bound

          DO i = dims%x_l_start, dims%x_l_end
            IF ( X( i ) <= X_l( i ) ) THEN
              PERTURB_X_l( i ) = reduce_perturb_factor * PERTURB_X_l( i ) -    &
                one_minus_red_pert_fac * ( X( i ) - X_l( i ) )
            ELSE
              IF ( X( i ) - X_l( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_X_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_X_l( i )
              ELSE
                PERTURB_X_l( i ) = zero
              END IF
            END IF
            DIST_X_l( i ) = X( i ) - X_l( i ) + PERTURB_X_l( i )
            IF ( Z_l( i ) <= zero ) THEN
              PERTURB_Z_l( i ) = reduce_perturb_factor * PERTURB_Z_l( i )      &
                - one_minus_red_pert_fac * Z_l( i )
            ELSE
              IF ( Z_l( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_Z_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Z_l( i )
              ELSE
                PERTURB_Z_l( i ) = zero
              END IF
            END IF
            DIST_Z_l( i ) = Z_l( i ) + PERTURB_Z_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_l( i ), DIST_X_l( i )
          END DO

!  The variable has an upper bound

          DO i = dims%x_u_start, dims%x_u_end
            IF ( X( i ) >= X_u( i ) ) THEN
              PERTURB_X_u( i ) = reduce_perturb_factor * PERTURB_X_u( i ) -    &
                one_minus_red_pert_fac * ( X_u( i ) - X( i ) )
            ELSE
              IF ( X_u( i ) - X( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_X_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_X_u( i )
              ELSE
                PERTURB_X_u( i ) = zero
              END IF
            END IF
            DIST_X_u( i ) = X_u( i ) + PERTURB_X_u( i ) - X( i )
            IF ( Z_u( i ) >= zero ) THEN
              PERTURB_Z_u( i ) = reduce_perturb_factor * PERTURB_Z_u( i )      &
                + one_minus_red_pert_fac * Z_u( i )
            ELSE
              IF ( Z_u( i ) >= - control%insufficiently_feasible ) THEN
                PERTURB_Z_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Z_u( i )
              ELSE
                PERTURB_Z_u( i ) = zero
              END IF
            END IF
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The variable is a non-positivity

          DO i = dims%x_u_end + 1, n
            IF ( X( i ) >= zero ) THEN
              PERTURB_X_u( i ) = reduce_perturb_factor * PERTURB_X_u( i ) +    &
                one_minus_red_pert_fac * X( i )
            ELSE
              IF ( X( i ) >= - control%insufficiently_feasible ) THEN
                PERTURB_X_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_X_u( i )
              ELSE
                PERTURB_X_u( i ) = zero
              END IF
            END IF
            DIST_X_u( i ) = PERTURB_X_u( i ) - X( i )
            IF ( Z_u( i ) >= zero ) THEN
              PERTURB_Z_u( i ) = reduce_perturb_factor * PERTURB_Z_u( i )      &
                + one_minus_red_pert_fac * Z_u( i )
            ELSE
              IF ( Z_u( i ) >= - control%insufficiently_feasible ) THEN
                PERTURB_Z_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Z_u( i )
              ELSE
                PERTURB_Z_u( i ) = zero
              END IF
            END IF
            DIST_Z_u( i ) = - Z_u( i ) + PERTURB_Z_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_X_u( i ), DIST_X_u( i )
          END DO

!  The constraint has a lower bound

          DO i = dims%c_l_start, dims%c_l_end
            IF ( C( i ) <= C_l( i ) ) THEN
              PERTURB_C_l( i ) = reduce_perturb_factor * PERTURB_C_l( i ) -    &
                one_minus_red_pert_fac * ( C( i ) - C_l( i ) )
            ELSE
              IF ( C( i ) - C_l( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_C_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_C_l( i )
              ELSE
                PERTURB_C_l( i ) = zero
              END IF
            END IF
            DIST_C_l( i ) = C( i ) - C_l( i ) + PERTURB_C_l( i )
            IF ( Y_l( i ) <= zero ) THEN
              PERTURB_Y_l( i ) = reduce_perturb_factor * PERTURB_Y_l( i )      &
                - one_minus_red_pert_fac * Y_l( i )
            ELSE
              IF ( Y_l( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_Y_l( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Y_l( i )
              ELSE
                PERTURB_Y_l( i ) = zero
              END IF
            END IF
            DIST_Y_l( i ) = Y_l( i ) + PERTURB_Y_l( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_l( i ), DIST_C_l( i )
          END DO

!  The constraint has an upper bound

          DO i = dims%c_u_start, dims%c_u_end
            IF ( C( i ) >= C_u( i ) ) THEN
              PERTURB_C_u( i ) = reduce_perturb_factor * PERTURB_C_u( i ) -    &
                one_minus_red_pert_fac * ( C_u( i ) - C( i ) )
            ELSE
              IF ( C_u( i ) - C( i ) <= control%insufficiently_feasible ) THEN
                PERTURB_C_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_C_u( i )
              ELSE
                PERTURB_C_u( i ) = zero
              END IF
            END IF
            DIST_C_u( i ) = C_u( i ) + PERTURB_C_u( i ) - C( i )
            IF ( Y_u( i ) >= zero ) THEN
              PERTURB_Y_u( i ) = reduce_perturb_factor * PERTURB_Y_u( i )      &
                + one_minus_red_pert_fac * Y_u( i )
            ELSE
              IF ( Y_u( i ) >= - control%insufficiently_feasible ) THEN
                PERTURB_Y_u( i ) =                                             &
                  control%reduce_perturb_multiplier * PERTURB_Y_u( i )
              ELSE
                PERTURB_Y_u( i ) = zero
              END IF
            END IF
            DIST_Y_u( i ) = - Y_u( i ) + PERTURB_Y_u( i )
!           write(48,"( i8, 2es12.4 )" ) i, PERTURB_C_u( i ), DIST_C_u( i )
          END DO
        END IF

!  Adjust mu to try to reduce infeasibilities

!        IF ( control%mu_increase_factor > one ) THEN
!          mu_target = control%mu_increase_factor * mu_target
!          DO i = dims%x_free + 1, dims%x_l_start - 1
!           IF ( X( i ) <= zero .OR. Z_l( i ) <= zero )                        &
!             MU_X_l( i ) = control%mu_increase_factor * MU_X_l( i )
!          END DO
!          DO i = dims%x_l_start, dims%x_l_end
!           IF ( X( i ) <= X_l( i ) .OR. Z_l( i ) <= zero )                    &
!             MU_X_l( i ) = control%mu_increase_factor * MU_X_l( i )
!          END DO
!          DO i = dims%x_u_start, dims%x_u_end
!           IF ( X( i ) >= X_u( i ) .OR. Z_u( i ) >= zero )                    &
!             MU_X_u( i ) = control%mu_increase_factor * MU_X_u( i )
!          END DO
!          DO i = dims%x_u_end + 1, n
!           IF ( X( i ) >= zero .OR. Z_u( i ) >= zero )                        &
!             MU_X_u( i ) = control%mu_increase_factor * MU_X_u( i )
!          END DO
!          DO i = dims%c_l_start, dims%c_l_end
!           IF ( C( i ) <= C_l( i ) .OR. Y_l( i ) <= zero )                    &
!             MU_C_l( i ) = control%mu_increase_factor * MU_C_l( i )
!          END DO
!          DO i = dims%c_u_start, dims%c_u_end
!           IF ( C( i ) >= C_u( i ) .OR. Y_u( i ) >= zero )                    &
!             MU_C_u( i ) = control%mu_increase_factor * MU_C_u( i )
!          END DO
!        END IF

!       IF ( .TRUE. ) THEN
!         DO i = dims%x_free + 1, dims%x_l_end
!           MU_X_l( i ) = mu_target * MAX( tenm2, MIN( ten2,                   &
!                               one / MIN( DIST_X_l( i ), DIST_Z_l( i )  ) ) )
!         END DO
!         DO i = dims%x_u_start, n
!           MU_X_u( i ) = mu_target * MAX( tenm2, MIN( ten2,                   &
!                               one / MIN( DIST_X_u( i ), DIST_Z_u( i )  ) ) )
!         END DO
!         DO i = dims%c_l_start, dims%c_l_end
!           MU_C_l( i ) = mu_target * MAX( tenm2, MIN( ten2,                   &
!                               one / MIN( DIST_C_l( i ), DIST_Y_l( i )  ) ) )
!         END DO
!         DO i = dims%c_u_start, dims%c_u_end
!           MU_C_u( i ) = mu_target * MAX( tenm2, MIN( ten2,                   &
!                               one / MIN( DIST_C_u( i ), DIST_Y_u( i )  ) ) )
!         END DO
!       END IF

        IF ( control%mu_increase_factor > one ) THEN
          DO i = dims%x_free + 1, dims%x_l_end
            IF ( PERTURB_X_l( i ) > zero .OR. PERTURB_Z_l( i ) > zero )        &
              MU_X_l( i ) = control%mu_increase_factor * MU_X_l( i )
          END DO
          DO i = dims%x_u_start, n
            IF ( PERTURB_X_u( i ) > zero .OR. PERTURB_Z_u( i ) > zero )        &
              MU_X_u( i ) = control%mu_increase_factor * MU_X_u( i )
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            IF ( PERTURB_C_l( i ) > zero .OR. PERTURB_C_l( i ) > zero )        &
              MU_C_l( i ) = control%mu_increase_factor * MU_C_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            IF ( PERTURB_C_u( i ) > zero .OR. PERTURB_C_u( i ) > zero )        &
              MU_C_u( i ) = control%mu_increase_factor * MU_C_u( i )
          END DO
        END IF

!  Calculate the complementarity

        slknes_req = SUM( ABS( DIST_X_l( dims%x_free + 1 : dims%x_l_end ) *    &
                               DIST_Z_l( dims%x_free + 1 : dims%x_l_end ) -    &
                               MU_X_l( dims%x_free + 1 : dims%x_l_end ) ) )    &
                   + SUM( ABS( DIST_X_u( dims%x_u_start : n ) *                &
                               DIST_Z_u( dims%x_u_start : n ) -                &
                               MU_X_u( dims%x_u_start : n ) ) )                &
                   + SUM( ABS( DIST_C_l( dims%c_l_start : dims%c_l_end ) *     &
                               DIST_Y_l( dims%c_l_start : dims%c_l_end ) -     &
                               MU_C_l( dims%c_l_start : dims%c_l_end ) ) )     &
                   + SUM( ABS( DIST_C_u( dims%c_u_start : dims%c_u_end ) *     &
                               DIST_Y_u( dims%c_u_start : dims%c_u_end ) -     &
                               MU_C_u( dims%c_u_start : dims%c_u_end ) ) )

!  Compute the constraint residuals

        C_RES( : dims%c_equality ) =                                           &
          C_RES( : dims%c_equality ) - C_l( : dims%c_equality )
        C_RES( dims%c_l_start : dims%c_u_end ) =                               &
          C_RES( dims%c_l_start : dims%c_u_end ) - SCALE_C * C

!  Evaluate the merit function if not already done

        merit = WCP_merit_value( dims, n, m, Y, Y_l, DIST_Y_l, Y_u,            &
                                 DIST_Y_u, Z_l, DIST_Z_l, Z_u, DIST_Z_u,       &
                                 DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,       &
                                 GRAD_L( dims%x_s : dims%x_e ),                &
                                 C_RES, res_dual,                              &
                                 MU_X_l, MU_X_u, MU_C_l, MU_C_u )
!       old_merit = merit
        IF ( control%perturbation_strategy > 2 )                               &
          reduce_perturb_factor = 0.25_wp * reduce_perturb_factor

!  ---------------------------------------------------------------------
!  -------------- End of Perturbation Reduction Loop -------------------
!  ---------------------------------------------------------------------

      END DO

  500 CONTINUE

!  Print details of the solution obtained

  600 CONTINUE

!  Compute the final objective function value

      inform%obj = f
      IF ( gradient_kind == 1 ) THEN
        inform%obj = inform%obj + SUM( X )
      ELSE IF ( gradient_kind /= 0 ) THEN
        inform%obj = inform%obj + DOT_PRODUCT( G, X )
      END IF

      IF ( printi ) THEN
        WRITE( out, 2010 ) prefix, inform%obj, prefix, inform%iter
        WRITE( out, 2110 ) prefix, pjgnrm, prefix, res_prim

        IF ( SBLS_control%factorization == 0 .OR.                              &
             SBLS_control%factorization == 1 ) THEN
          WRITE( out, "( /, A, '  Schur-complement method used ' )" ) prefix
        ELSE
          WRITE( out, "( /, A, '  Augmented system method used ' )" ) prefix
        END IF
      END IF

!  Exit

 700  CONTINUE

!  Unscale the constraint bounds

      DO i = dims%c_l_start, dims%c_l_end
        C_l( i ) = C_l( i ) * SCALE_C( i )
      END DO

      DO i = dims%c_u_start, dims%c_u_end
        C_u( i ) = C_u( i ) * SCALE_C( i )
      END DO

!  Compute the values of the constraints

      C_RES( : m ) = zero
      CALL WCP_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )

!  Compute the number of constraint violations

!     inform%x_implicit =                                                      &
!            COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 )                  &
!                   < control%implicit_tol )                                   &
!          + COUNT( X( dims%x_l_start : dims%x_l_end ) -                       &
!                   X_l( dims%x_l_start : dims%x_l_end )                       &
!                   < control%implicit_tol )                                   &
!          + COUNT( X( dims%x_u_start : dims%x_u_end ) -                       &
!                   X_u( dims%x_u_start : dims%x_u_end )                       &
!                   > - control%implicit_tol )                                 &
!          + COUNT( X( dims%x_u_end + 1: n ) > - control%implicit_tol )
!     inform%z_implicit =                                                      &
!            COUNT( Z_l( dims%x_free + 1 : dims%x_l_end )                      &
!                   < control%implicit_tol )                                   &
!          + COUNT( Z_u( dims%x_u_start : n ) > - control%implicit_tol )
!     inform%c_implicit =                                                      &
!            COUNT( C_RES( dims%c_l_start : dims%c_l_end ) -                   &
!                   C_l( dims%c_l_start : dims%c_l_end )                       &
!                   < control%implicit_tol )                                   &
!          + COUNT( C_RES( dims%c_u_start : dims%c_u_end ) -                   &
!                   C_u( dims%c_u_start : dims%c_u_end )                       &
!                   > - control%implicit_tol )
!     inform%y_implicit =                                                      &
!       COUNT( Y_l( dims%c_l_start : dims%c_l_end ) < control%implicit_tol )   &
!       + COUNT( Y_u( dims%c_u_start : dims%c_u_end ) > - control%implicit_tol )

      inform%x_implicit = 0 ; inform%z_implicit = 0
      inform%c_implicit = 0 ; inform%y_implicit = 0

      DO i = 1, n
        IF ( ABS( X_l( i ) - X_u( i ) ) < control%implicit_tol ) THEN
          IF ( control%record_x_status ) inform%X_status( i ) = 3
        ELSE IF ( X( i ) - X_l( i ) < control%implicit_tol ) THEN
          inform%x_implicit = inform%x_implicit + 1
          IF ( control%record_x_status ) inform%X_status( i ) = - 1
        ELSE IF ( X_u( i ) - X( i ) < control%implicit_tol ) THEN
          inform%x_implicit = inform%x_implicit + 1
          IF ( control%record_x_status ) inform%X_status( i ) = 1
        ELSE IF ( i < dims%x_u_start .AND.                                     &
                  ABS( Z_l( i ) ) < control%implicit_tol ) THEN
          inform%z_implicit = inform%z_implicit + 1
          IF ( control%record_x_status ) inform%X_status( i ) = - 2
        ELSE IF ( i > dims%x_l_end .AND.                                       &
                  ABS( Z_u( i ) ) < control%implicit_tol ) THEN
          inform%z_implicit = inform%z_implicit + 1
          IF ( control%record_x_status ) inform%X_status( i ) = 2
        ELSE IF ( ( i >= dims%x_u_start .AND. i <= dims%x_l_end ) .AND.        &
                  ( ABS( Z_l( i ) ) < control%implicit_tol .OR.                &
                    ABS( Z_u( i ) ) < control%implicit_tol ) ) THEN
          inform%z_implicit = inform%z_implicit + 1
          IF ( ABS( Z_l( i ) ) < control%implicit_tol .AND.                    &
               ABS( Z_u( i ) ) < control%implicit_tol )                        &
            inform%z_implicit = inform%z_implicit + 1
          IF ( control%record_x_status ) THEN
            IF ( ABS( Z_l( i ) ) < control%implicit_tol .AND.                  &
                 ABS( Z_u( i ) ) < control%implicit_tol ) THEN
              inform%X_status( i ) = - 3
            ELSE IF ( ABS( Z_l( i ) ) < control%implicit_tol ) THEN
              inform%X_status( i ) = - 2
            ELSE
              inform%X_status( i ) = 2
            END IF
          END IF
        ELSE
          IF ( control%record_x_status ) inform%X_status( i ) = 0
        END IF
      END DO

      IF ( m > 0 ) THEN
        DO i = 1, m
          IF ( ABS( C_l( i ) - C_u( i ) ) < control%implicit_tol ) THEN
            IF ( control%record_c_status ) inform%C_status( i ) = 3
          ELSE IF ( C( I ) - C_l( i ) < control%implicit_tol ) THEN
            inform%c_implicit = inform%c_implicit + 1
            IF ( control%record_c_status ) inform%C_status( i ) = - 1
          ELSE IF ( C_u( I ) - C( i ) < control%implicit_tol ) THEN
            inform%c_implicit = inform%c_implicit + 1
            IF ( control%record_c_status ) inform%C_status( i ) = 1
          ELSE IF ( i < dims%c_u_start .AND.                                   &
                    ABS( Y_l( i ) ) < control%implicit_tol ) THEN
            inform%z_implicit = inform%y_implicit + 1
            IF ( control%record_c_status ) inform%C_status( i ) = - 2
          ELSE IF ( i > dims%c_l_end .AND.                                     &
                    ABS( Y_u( i ) ) < control%implicit_tol ) THEN
            inform%y_implicit = inform%y_implicit + 1
            IF ( control%record_c_status ) inform%C_status( i ) = 2
          ELSE IF ( ( i >= dims%c_u_start .AND. i <= dims%c_l_end ) .AND.      &
                    ( ABS( Y_l( i ) ) < control%implicit_tol .OR.              &
                      ABS( Y_u( i ) ) < control%implicit_tol ) ) THEN
            inform%y_implicit = inform%y_implicit + 1
            IF ( ABS( Y_l( i ) ) < control%implicit_tol .AND.                  &
                 ABS( Y_u( i ) ) < control%implicit_tol )                      &
              inform%y_implicit = inform%y_implicit + 1
            IF ( control%record_c_status ) THEN
              IF ( ABS( Y_l( i ) ) < control%implicit_tol .AND.                &
                   ABS( Y_u( i ) ) < control%implicit_tol ) THEN
                inform%C_status( i ) = - 3
              ELSE IF ( ABS( Y_l( i ) ) < control%implicit_tol ) THEN
                inform%C_status( i ) = - 2
              ELSE
                inform%C_status( i ) = 2
              END IF
            END IF
          ELSE
            IF ( control%record_c_status ) inform%C_status( i ) = 0
          END IF
        END DO
      END IF

      inform%feasible =                                                        &
        res_prim <= control%stop_p .AND. res_dual <= control%stop_d .AND.      &
        MAX( inform%x_implicit, inform%c_implicit,                             &
             inform%y_implicit, inform%z_implicit ) == 0

!  If required, print the numbers and maximum sizes of violations

      IF ( printi ) THEN
        max_xr = MAX( zero,                                                    &
                      MAXVAL( - X( dims%x_free + 1 : dims%x_l_start - 1 ) ),   &
                      MAXVAL( X_l( dims%x_l_start : dims%x_l_end ) -           &
                              X( dims%x_l_start : dims%x_l_end ) ),            &
                      MAXVAL( X( dims%x_u_start : dims%x_u_end ) -             &
                              X_u( dims%x_u_start : dims%x_u_end ) ),          &
                      MAXVAL( X( dims%x_u_end + 1 : n ) ) )
        max_zr = MAX( zero,                                                    &
                      MAXVAL( - Z_l( dims%x_free + 1 : dims%x_l_end ) ),       &
                      MAXVAL( Z_u( dims%x_u_start : n ) ) )
        max_cr = MAX( zero, MAXVAL( C_l( dims%c_l_start : dims%c_l_end ) -     &
                                    C_RES( dims%c_l_start : dims%c_l_end ) ),  &
                            MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -   &
                                    C_u( dims%c_u_start : dims%c_u_end ) ) )
        max_yr = MAX( zero, MAXVAL( - Y_l( dims%c_l_start : dims%c_l_end ) ),  &
                            MAXVAL( Y_u( dims%c_u_start : dims%c_u_end ) ) )

        IF ( max_xr > zero .OR. max_zr > zero .OR.                             &
             max_yr > zero .OR. max_yr > zero ) WRITE( out, "( '' )" )
        IF ( max_xr > zero )                                                   &
          WRITE( out, "( A, '  # infeasible variables = ', I0,                 &
         &   ', infeasibility =', ES11.4 )" ) prefix, inform%x_implicit, max_xr
        IF ( max_zr > zero )                                                   &
          WRITE( out, "( A, '  # infeasible duals = ', I0,                     &
         &   ', infeasibility =', ES11.4 )" ) prefix, inform%z_implicit, max_zr
        IF ( max_cr > 0 )                                                      &
          WRITE( out, "( A, '  # infeasible constraints = ', I0,               &
         &   ', infeasibility =', ES11.4 )" ) prefix, inform%c_implicit, max_cr
        IF ( max_yr > 0 )                                                      &
          WRITE( out, "( A, '  # infeasible multipliers = ', I0,               &
         &   ', infeasibility =', ES11.4 )" ) prefix, inform%y_implicit, max_yr
      END IF

!     WRITE( 79, "( ' variables ', /, '       i  violation ' )" )
!     DO i = dims%x_free + 1, dims%x_l_start - 1
!       WRITE( 79, 2991 ) i, MAX( zero, - X( i ) )
!     END DO
!     DO i = dims%x_l_start, dims%x_l_end
!       WRITE( 79, 2991 ) i, MAX( zero, X_l( i ) - X( i ) )
!     END DO
!     DO i = dims%x_u_start, dims%x_u_end
!       WRITE( 79, 2991 ) i, MAX( zero, X( i ) - X_u( i ) )
!     END DO
!     DO i = dims%x_u_end + 1, n
!       WRITE( 79, 2991 ) i, MAX( zero, X( i ) )
!     END DO
!     WRITE( 79, "( /, ' constraints ', /, '       i  violation ' )" )
!     DO i = dims%c_l_start, dims%c_l_end
!       WRITE( 79, 2991 ) i, MAX( zero, C_l( i ) - C_RES( i ) )
!     END DO
!     DO i = dims%c_u_start, dims%c_u_end
!       WRITE( 79, 2991 ) i, MAX( zero, C_RES( i ) - C_u( i ) )
!     END DO
!2991 FORMAT( I8, ES12.4 )

!  If necessary, print warning messages

  810 CONTINUE
      IF ( printi ) then
        SELECT CASE( inform%status )
          CASE( - 1 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - input paramters incorrect ' )" ) prefix
          CASE( - 4 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - iteration bound exceeded ' ) " ) prefix
          CASE( - 5 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - the constraints are inconsistent ' )" ) prefix
          CASE( - 6 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - the constraints appear to be inconsistent')")&
                prefix
          CASE( - 7 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - factorization failure ' )" ) prefix
          CASE( - 8 ) ; WRITE( out,                                            &
            "( /, A, '  Warning - residuals too large for further progress' )")&
               prefix
        END SELECT
      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving WCP_solve_main ' )" ) prefix

      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Iter   p-feas  d-feas com-slk   merit  ',                &
                '  step        mu       time' )
 2010 FORMAT( //, A, '  Final objective function value ', ES22.14,             &
              /,  A, '  Total number of iterations = ', I0 )
 2020 FORMAT( A, I5, A1, 3ES8.1, ES9.1, '     -      ', ES7.1, 0P, F9.2 )
 2030 FORMAT( A, I5, A1, 3ES8.1, ES9.1, ES9.2, A2, 1X, ES7.1, 0P, F9.2 )
 2040 FORMAT( A, '   **  Error return ', I0, ' from ', A )
 2050 FORMAT( A, '   **  Warning ', I0, ' from ', A )
 2060 FORMAT( A, I8, ' integer and ', I8,                                      &
              ' real words needed for factorization' )
 2070 FORMAT( /, A, ' ================ point satisfying equations found',      &
                 ' =============== ', / )
 2120 FORMAT( A10, 7ES10.2, /, ( 10X, 7ES10.2 ) )
 2110 FORMAT( /, A, '  Norm of projected gradient is ', ES12.4, /, A,          &
                '  Norm of infeasibility is      ', ES12.4 )
 2130 FORMAT( A, 2X, ' == >  mu estimated   = ', ES9.1, /,                     &
              A, 2X, '       mu_x estimated = ', ES9.1, /,                     &
              A, 2X, '       mu_c estimated = ', ES9.1, /,                     &
              A, 2X, '  min/max slackness_x = ', 2ES12.4, /,                   &
              A, 2X, '  min/max slackness_c = ', 2ES12.4, /,                   &
              A, 2X, '  min/max primal feasibility = ', 2ES12.4, /,            &
              A, 2X, '  min/max dual   feasibility = ', 2ES12.4 )
 2140 FORMAT( /, '  Norm of (predictor) search direction = ', ES12.4 )
 2150 FORMAT( A6, /, ( 4( 2I5, ES10.2 ) ) )
 2160 FORMAT( /, '  Norm of (corrector) search direction = ', ES12.4 )

!  End of WCP_solve_main

      END SUBROUTINE WCP_solve_main

!-*-*-*-*-*-*-   W C P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE WCP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine WCP_initialize
!   control see Subroutine WCP_initialize
!   inform  see Subroutine WCP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( WCP_data_type ), INTENT( INOUT ) :: data
      TYPE ( WCP_control_type ), INTENT( IN ) :: control
      TYPE ( WCP_inform_type ), INTENT( INOUT ) :: inform

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

!  Deallocate all arrays allocated for the preprocessing stage

      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      IF ( data%QPP_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = ''
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'wcp: data%C_freed'
      CALL SPACE_dealloc_array( data%C_freed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%HX'
      CALL SPACE_dealloc_array( data%HX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%GRAD_L'
      CALL SPACE_dealloc_array( data%GRAD_L,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_X_l'
      CALL SPACE_dealloc_array( data%DIST_X_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_Z_l'
      CALL SPACE_dealloc_array( data%DIST_Z_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%Z_l'
      CALL SPACE_dealloc_array( data%Z_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_X_l'
      CALL SPACE_dealloc_array( data%PERTURB_X_l,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_X_u'
      CALL SPACE_dealloc_array( data%DIST_X_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_Z_u'
      CALL SPACE_dealloc_array( data%DIST_Z_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_X_u'
      CALL SPACE_dealloc_array( data%PERTURB_X_u,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_Z_u'
      CALL SPACE_dealloc_array( data%PERTURB_Z_u,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%BARRIER_X'
      CALL SPACE_dealloc_array( data%BARRIER_X,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DY_l'
      CALL SPACE_dealloc_array( data%DY_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_C_l'
      CALL SPACE_dealloc_array( data%DIST_C_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_Y_l'
      CALL SPACE_dealloc_array( data%DIST_Y_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_C_l'
      CALL SPACE_dealloc_array( data%PERTURB_C_l,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_Y_l'
      CALL SPACE_dealloc_array( data%PERTURB_Y_l,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DY_u'
      CALL SPACE_dealloc_array( data%DY_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_C_u'
      CALL SPACE_dealloc_array( data%DIST_C_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIST_Y_u'
      CALL SPACE_dealloc_array( data%DIST_Y_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_C_u'
      CALL SPACE_dealloc_array( data%PERTURB_C_u,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%PERTURB_Y_u'
      CALL SPACE_dealloc_array( data%PERTURB_Y_u,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%BARRIER_C'
      CALL SPACE_dealloc_array( data%BARRIER_C,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%SCALE_C'
      CALL SPACE_dealloc_array( data%SCALE_C,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DELTA'
      CALL SPACE_dealloc_array( data%DELTA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DZ_l'
      CALL SPACE_dealloc_array( data%DZ_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DZ_u'
      CALL SPACE_dealloc_array( data%DZ_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%MU_X_l'
      CALL SPACE_dealloc_array( data%MU_X_l,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%MU_X_u'
      CALL SPACE_dealloc_array( data%MU_X_u,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%MU_C_l'
      CALL SPACE_dealloc_array( data%MU_C_l,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%MU_C_u'
      CALL SPACE_dealloc_array( data%MU_C_u,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%MU'
      CALL SPACE_dealloc_array( data%MU,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%COEF0'
      CALL SPACE_dealloc_array( data%COEF0,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%COEF1'
      CALL SPACE_dealloc_array( data%COEF1,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%COEF2'
      CALL SPACE_dealloc_array( data%COEF2,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%COEF3'
      CALL SPACE_dealloc_array( data%COEF3,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%COEF4'
      CALL SPACE_dealloc_array( data%COEF4,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%BREAKP'
      CALL SPACE_dealloc_array( data%BREAKP,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%IBREAK'
      CALL SPACE_dealloc_array( data%IBREAK,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DELTA_cor'
      CALL SPACE_dealloc_array( data%DELTA_cor,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DY_cor_l'
      CALL SPACE_dealloc_array( data%DY_cor_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DY_cor_u'
      CALL SPACE_dealloc_array( data%DY_cor_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DZ_cor_l'
      CALL SPACE_dealloc_array( data%DZ_cor_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DZ_cor_u'
      CALL SPACE_dealloc_array( data%DZ_cor_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIAG_X'
      CALL SPACE_dealloc_array( data%DIAG_X,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%DIAG_C'
      CALL SPACE_dealloc_array( data%DIAG_C,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%Index_C_freed'
      CALL SPACE_dealloc_array( data%Index_C_freed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: inform%X_status'
      CALL SPACE_dealloc_array( inform%X_status,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: inform%C_status'
      CALL SPACE_dealloc_array( inform%C_status,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%A_sbls%row'
      CALL SPACE_dealloc_array( data%A_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%A_sbls%col'
      CALL SPACE_dealloc_array( data%A_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%A_sbls%val'
      CALL SPACE_dealloc_array( data%A_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'wcp: data%H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine WCP_terminate

      END SUBROUTINE WCP_terminate

!-*-  W C P _ L A G R A N G I A N _ G R A D I E N T   S U B R O U T I N E  -*-

      SUBROUTINE WCP_Lagrangian_gradient( n, m, Y, A_ne, A_val, A_col, A_ptr,  &
                                          GRAD_l, gradient_kind, G )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the gradient of the Lagrangian function
!
!  GRAD_L = g - A(transpose) y
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, A_ne, gradient_kind
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD_L
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: G

!  Add the product A( transpose ) y to the gradient of the quadratic

      GRAD_L = zero
      IF ( gradient_kind == 1 ) THEN
        GRAD_L = GRAD_L + one
      ELSE IF ( gradient_kind /= 0 ) THEN
        GRAD_L = GRAD_L + G
      END IF

      CALL WCP_AX( n, GRAD_L, m, A_ne, A_val, A_col, A_ptr, m, Y, '-T' )

      RETURN

!  End of WCP_Lagrangian_gradient

      END SUBROUTINE WCP_Lagrangian_gradient

!-*-*-*-*-*-   W C P _ M E R I T _ V A L U E   F U N C T I O N   -*-*-*-*-*-*-

      FUNCTION WCP_merit_value( dims, n, m, Y, Y_l, DIST_Y_l, Y_u, DIST_Y_u,   &
                                Z_l, DIST_Z_l, Z_u, DIST_Z_u,                  &
                                DIST_X_l, DIST_X_u, DIST_C_l,                  &
                                DIST_C_u, GRAD_L, C_RES, res_dual,             &
                                MU_X_l, MU_X_u, MU_C_l, MU_C_u )
      REAL ( KIND = wp ) WCP_merit_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the merit function
!
!     | < dz_l . dx_l > - mu_x_l | + | < dz_u . dx_u > - mu_x_u | +
!     | < dy_l . dc_l > - mu_c_l | + | < dy_u . dc_u > - mu_c_u | + res_dual
!
!  where
!
!               || GRAD_L - z_l - z_u ||
!   res_dual =  ||   y - y_l - y_u    ||
!               ||      C_RES         ||
!
!   GRAD_L = g - A(transpose) y = gradient of the Lagrangian
!   C_RES =  A x - SCALE_c * c
!   dx_l = x - x_l + omega_x_l
!   dx_u = x_u - x + omega_x_u
!   dc_l = c - c_l + omega_c_l
!   dc_u = c_u - c + omega_c_u
!   dz_l = z_l + omega_z_l
!   dz_u = - z_u + omega_z_u
!   dy_l = y_l + omega_y_l
!   dy_u = - y_u + omega_y_u
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( OUT ) :: res_dual
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: GRAD_L, Z_l, Z_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y_l, Y_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: DIST_X_l,          &
                                                            DIST_Z_l, MU_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : n ) :: DIST_X_u, DIST_Z_u, MU_X_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y, C_RES
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l,           &
                                                           DIST_Y_l, MU_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u,           &
                                                           DIST_Y_u, MU_C_u

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: res_cs

!  Compute in the l_1-norm

!     DO i = 1, dims%x_free
!       write(6,*) ' f ', ABS( GRAD_L( : dims%x_free ) )
!     END DO
      res_dual = SUM( ABS( GRAD_L( : dims%x_free ) ) ) ; res_cs = zero

!  Problem variables:

      DO i = dims%x_free + 1, dims%x_u_start - 1
!       write(6,*) ' l ', ABS( GRAD_L( i ) - Z_l( i ) )
        res_dual = res_dual + ABS( GRAD_L( i ) - Z_l( i ) )
        res_cs = res_cs + ABS( DIST_Z_l( i ) * DIST_X_l( i ) - MU_X_l( i ) )
      END DO
      DO i = dims%x_u_start, dims%x_l_end
!       write(6,*) 'lu', ABS( GRAD_L( i ) - Z_l( i ) - Z_u( i ) )
        res_dual = res_dual + ABS( GRAD_L( i ) - Z_l( i ) - Z_u( i ) )
        res_cs = res_cs + ABS( DIST_Z_l( i ) * DIST_X_l( i ) - MU_X_l( i ) )   &
                        + ABS( DIST_Z_u( i ) * DIST_X_u( i ) - MU_X_u( i ) )
      END DO
      DO i = dims%x_l_end + 1, n
!       write(6,*) ' u ', ABS( GRAD_L( i ) - Z_u( i ) )
        res_dual = res_dual + ABS( GRAD_L( i ) - Z_u( i ) )
        res_cs = res_cs + ABS( DIST_Z_u( i ) * DIST_X_u( i ) - MU_X_u( i ) )
      END DO

!  Slack variables:

      DO i = dims%c_l_start, dims%c_u_start - 1
!       write(6,*) 'dl', ABS( Y( i ) - Y_l( i ) )
        res_dual = res_dual + ABS( Y( i ) - Y_l( i ) )
        res_cs = res_cs + ABS( DIST_Y_l( i ) * DIST_C_l( i ) - MU_C_l( i ) )
      END DO
      DO i = dims%c_u_start, dims%c_l_end
!       write(6,*) 'dlu', ABS( Y( i ) - Y_l( i ) - Y_u( i ) )
        res_dual = res_dual + ABS( Y( i ) - Y_l( i ) - Y_u( i ) )
        res_cs = res_cs + ABS( DIST_Y_l( i ) * DIST_C_l( i ) - MU_C_l( i ) )   &
                        + ABS( DIST_Y_u( i ) * DIST_C_u( i ) - MU_C_u( i ) )
      END DO
      DO i = dims%c_l_end + 1, dims%c_u_end
!       write(6,*) 'du', ABS( Y( i ) - Y_u( i ) )
        res_dual = res_dual + ABS( Y( i ) - Y_u( i ) )
        res_cs = res_cs + ABS( DIST_Y_u( i ) * DIST_C_u( i ) - MU_C_u( i ) )
      END DO

      WCP_merit_value = SUM( ABS( C_RES ) ) + res_dual + res_cs
!     write(6,"(' res_cs ', ES12.4 )") res_cs

      RETURN

!  End of function WCP_merit_value

      END FUNCTION WCP_merit_value

!-*-*-*-*-*-*-   W C P _ R E S I D U A L   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE WCP_residual( dims, n, m, l_res, A_ne, A_val, A_col, A_ptr,   &
                               DX, DC, DY, RHS_x, RHS_c, RHS_y, RES,           &
                               barrier_free, BARRIER_X, BARRIER_C, SCALE_C,    &
                               errorg, errorc, print_level, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Compute the residual of the linear system

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, A_ne, l_res, print_level
      REAL ( KIND = wp ), INTENT( IN ) :: barrier_free
      REAL(  KIND = wp ), INTENT( OUT ) :: errorg, errorc
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( l_res ) :: RES
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RHS_x, DX
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: RHS_y, DY
      REAL ( KIND = wp ), INTENT( IN ),                                        &
           DIMENSION( dims%c_l_start : m ) :: RHS_c, DC, BARRIER_C, SCALE_C
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      TYPE ( WCP_control_type ), INTENT( IN ) :: control

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: res_tol

      res_tol = epsmch ** 0.5

!  Initalize RES as the zero vector

      RES( dims%y_s : dims%y_e ) = zero

!  Remember the barrier terms, and any diagonal perturbations

      IF ( barrier_free == zero ) THEN
        RES( : dims%x_free ) = zero
      ELSE
        RES( : dims%x_free ) = barrier_free * DX( : dims%x_free )
      END IF
      RES( dims%x_free + 1 : dims%x_e ) = BARRIER_X * DX( dims%x_free + 1 : )
      RES( dims%c_s : dims%c_e ) = BARRIER_C * DC

!  Include the contribution from A and A^T

      CALL WCP_AX( n, RES( dims%x_s : dims%x_e ), m,                           &
                   A_ne, A_val, A_col, A_ptr, m, DY, '+T' )
      CALL WCP_AX( m, RES( dims%y_s : dims%y_e ), m,                           &
                   A_ne, A_val, A_col, A_ptr, n, DX, '+ ' )

!  Include the contribution from the slack variables

      RES( dims%c_s : dims%c_e ) =                                             &
        RES( dims%c_s : dims%c_e ) - SCALE_C * DY( dims%c_l_start : m )
      RES( dims%y_i : dims%y_e ) =                                             &
        RES( dims%y_i : dims%y_e ) - SCALE_C * DC

!  Find the largest residual and component of the search direction

      IF ( control%out > 0 .AND. print_level >= 2 ) THEN
        errorg = MAX( MAXVAL( ABS( RES( dims%x_s : dims%x_e ) - RHS_x ) ),     &
                      MAXVAL( ABS( RES( dims%c_s : dims%c_e ) - RHS_c ) ) )
        IF ( print_level >= 4 ) THEN
          DO i = 1, n
            IF ( ABS( RES( i ) - RHS_x( i ) ) > res_tol )                      &
              WRITE( control%out, 2010 ) 'X', i, RES( i ), RHS_x( i )
          END DO
          DO i = dims%c_l_start, dims%c_u_end
            IF ( ABS( RES( dims%c_b + i ) - RHS_c( i ) ) > res_tol )           &
              WRITE( control%out, 2010 ) 'C', i, RES( dims%c_b + i ), RHS_c( i )
          END DO
        END IF
        IF ( m > 0 ) THEN
          errorc = MAXVAL( ABS( RES( dims%y_s : dims%y_e ) - RHS_y ) )
        ELSE
          errorc = zero
        END IF
        IF ( print_level >= 4 ) THEN
          DO i = 1, m
            IF ( ABS( RES( dims%y_s + i - 1 ) - RHS_y( i ) ) > res_tol )       &
              WRITE( control%out, 2010 )                                       &
                'C', I, RES( dims%y_s + i - 1 ), RHS_y( i )
          END DO
        END IF
        WRITE( control%out, 2000 ) errorg, errorc, MAX( MAXVAL( ABS( DX ) ),   &
                                                        MAXVAL( ABS( DC ) ),   &
                                                        MAXVAL( ABS( DY ) ) )
      END IF
      RETURN

!  Non-executable statements

 2000 FORMAT( ' ',                                                             &
          /, '         ***  Max component of gradient  residuals = ', ES12.4,  &
          /, '         ***  Max component of contraint residuals = ', ES12.4,  &
          /, '         ***  Max component of search direction    = ', ES12.4 )
 2010 FORMAT( ' ', A1, '-residual', I6, ' lhs = ', ES12.4,' rhs = ', ES12.4 )

!  End of subroutine WCP_residual

      END SUBROUTINE WCP_residual

!-   W C P _ R E S I D U A L _ U N C O N S T R A I N E D  S U B R O U T I N E  -

      SUBROUTINE WCP_residual_unconstrained( dims, n, l_res, DX, RHS_x, RES,   &
                         barrier_free, BARRIER_X, errorg, print_level, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Compute the residual of the linear system

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, l_res, print_level
      REAL ( KIND = wp ), INTENT( IN ) :: barrier_free
      REAL(  KIND = wp ), INTENT( OUT ) :: errorg
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( l_res ) :: RES
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RHS_x, DX
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      TYPE ( WCP_control_type ), INTENT( IN ) :: control

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: res_tol

      res_tol = epsmch ** 0.5

!  Remember the barrier terms, and any diagonal perturbations

      IF ( barrier_free == zero ) THEN
        RES( : dims%x_free ) = zero
      ELSE
        RES( : dims%x_free ) = barrier_free * DX( : dims%x_free )
      END IF
      RES( dims%x_free + 1 : dims%x_e ) = BARRIER_X * DX( dims%x_free + 1 : )

!  Find the largest residual and component of the search direction

      IF ( control%out > 0 .AND. print_level >= 2 ) THEN
        errorg = MAXVAL( ABS( RES( dims%x_s : dims%x_e ) - RHS_x ) )
        IF ( print_level >= 4 ) THEN
          DO i = 1, n
            IF ( ABS( RES( i ) - RHS_x( i ) ) > res_tol )                      &
              WRITE( control%out, 2010 ) 'X', i, RES( i ), RHS_x( i )
          END DO
        END IF
        WRITE( control%out, 2000 ) errorg, MAXVAL( ABS( DX ) )
      END IF
      RETURN

!  Non-executable statements

 2000 FORMAT( ' ',                                                             &
          /, '         ***  Max component of gradient  residuals = ', ES12.4,  &
          /, '         ***  Max component of search direction    = ', ES12.4 )
 2010 FORMAT( ' ', A1, '-residual', I6, ' lhs = ', ES12.4,' rhs = ', ES12.4 )

!  End of subroutine WCP_residual_unconstrained

      END SUBROUTINE WCP_residual_unconstrained

!- W C P _ m i n _ p i e c e w i s e _ q u a d r a t i c  S U B R O U T I N E -

      SUBROUTINE WCP_min_piecewise_quadratic( dims, n, m, nbnds, DIST_Z_l,     &
                                              DIST_Z_u, DZ_l, DZ_u, DIST_X_l,  &
                                              DIST_X_u, DX, DIST_Y_l,          &
                                              DIST_Y_u, DY_l,                  &
                                              DY_u, DIST_C_l, DIST_C_u, DC,    &
                                              res_prim_dual, MU_X_l, MU_X_u,   &
                                              MU_C_l, MU_C_u, MU,              &
                                              omega_l, omega_u,                &
                                              alpha, alpha_est, alpha_max,     &
                                              COEF2, COEF1, COEF0, BREAKP,     &
                                              IBREAK, print_level, control )
!                                             inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a global minimizer of the piecewise quadratic function
!
!  q(alpha) =
!    sum_i=1^n,j=1,2 | q_ij alpha^2 + l_ij alpha + c_ij - mu_i | +
!    sum_i=1^m,j=3,4 | q_ij alpha^2 + l_ij alpha + c_ij - mu_i | +
!    (1 - alpha) res_prim_dual
!
! for all alpha in [0,1] for which additionally
!
!  q_ij alpha^2 + l_ij alpha + c_ij >= omega_l mu_i  (i=1,n,j=1,2 & i=1,m,j=3,4)
!
! and
!
!  q_ij alpha^2 + l_ij alpha + c_ij <= omega_u mu_i  (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for omega_l in (0,1), omega_u > 1
!
!  Here:
!    j      q_ij                     lij                          cij
!    1  dx_i dz^L_i   (x_i - x^L_i) dz^L_i + dx_i z^L_i    (x_i - x^L_i) z^L_i
!    2  dx_i dz^U_i   (x_i - x^U_i) dz^U_i + dx_i z^U_i    (x_i - x^U_i) z^U_i
!    3  dc_i dy^L_i   (c_i - c^L_i) dy^L_i + dc_i y^L_i    (c_i - c^L_i) y^L_i
!    4  dc_i dy^U_i   (c_i - c^U_i) dy^U_i + dc_i y^U_i    (c_i - c^U_i) y^U_i

!  and res_prim_dual = || residual primal/dual optimality ||
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, nbnds, print_level
      REAL ( KIND = wp ), INTENT( IN ) :: res_prim_dual, omega_l, omega_u
      REAL ( KIND = wp ), INTENT( OUT ) :: alpha, alpha_est, alpha_max
      INTEGER, INTENT( OUT ), DIMENSION( 2 * nbnds ) :: IBREAK
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: DX
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: DIST_Z_l, DZ_l,         &
                                                       DIST_X_l, MU_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_u_start : n ) :: DIST_Z_u, DZ_u, DIST_X_u, MU_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_Y_l, DY_l,          &
                                                      DIST_C_l, MU_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_Y_u, DY_u,          &
                                                      DIST_C_u, MU_C_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_l_start : m ) :: DC
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( nbnds ) :: COEF2, COEF1, COEF0, MU
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2 * nbnds ) :: BREAKP
      TYPE ( WCP_control_type ), INTENT( IN ) :: control
!     TYPE ( WCP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, l, nroots, inheap, nbreak, cluster
      REAL ( KIND = wp ) :: c0, c1, c2, c0mu, root( 2 )
      REAL ( KIND = wp ) :: tol, close_tol, res_cs, alpha_min, q_alpha_est
      REAL ( KIND = wp ) :: alpha_l, alpha_u, alpha_m, q_l, q_u, q_m, q_min
      LOGICAL :: details, debug

      details = print_level >= 4
      debug = print_level >= 5
      tol = epsmch ** 0.75
      close_tol = tol

!  Compute the coefficients c2_ij, c1_ij and c0_ij

      l = 0
      DO i = dims%x_free + 1, dims%x_l_end
        l = l + 1
        COEF2( l ) = DX( i ) * DZ_l( i )
        COEF1( l ) = DIST_X_l( i ) * DZ_l( i ) + DX( i ) * DIST_Z_l( i )
        COEF0( l ) = DIST_X_l( i ) * DIST_Z_l( i )
        MU( l ) = MU_X_l( i )
      END DO
      DO i = dims%x_u_start, n
        l = l + 1
        COEF2( l ) = DX( i ) * DZ_u( i )
        COEF1( l ) = - DIST_X_u( i ) * DZ_u( i ) - DX( i ) * DIST_Z_u( i )
        COEF0( l ) = DIST_X_u( i ) * DIST_Z_u( i )
        MU( l ) = MU_X_u( i )
      END DO
      DO i = dims%c_l_start, dims%c_l_end
        l = l + 1
        COEF2( l ) = DC( i ) * DY_l( i )
        COEF1( l ) = DIST_C_l( i ) * DY_l( i ) + DC( i ) * DIST_Y_l( i )
        COEF0( l ) = DIST_C_l( i ) * DIST_Y_l( i )
        MU( l ) = MU_C_l( i )
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        l = l + 1
        COEF2( l ) = DC( i ) * DY_u( i )
        COEF1( l ) = - DIST_C_u( i ) * DY_u( i ) - DC( i ) * DIST_Y_u( i )
        COEF0( l ) = DIST_C_u( i ) * DIST_Y_u( i )
        MU( l ) = MU_C_u( i )
      END DO

!  Given omega_l in (0,1), find alpha_max, the largest value in [0,1] for which
!
!  c2_ij alpha^2 + c1_ij alpha + c0_ij >= omega_l mu_i
!     (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for all alpha <= alpha_max

!     WRITE( control%out, "( ' low ' )" )
      alpha_max = one
      DO l = 1, nbnds
        c2 = COEF2( l ) ; c1 = COEF1( l ) ; c0 = COEF0( l ) - omega_l * MU( l )
        CALL ROOTS_quadratic( c0, c1, c2, tol, nroots, root( 1 ), root( 2 ),   &
                              roots_debug )
        DO i = 1, nroots
          IF ( root( i ) > tenm7 ) alpha_max = MIN( alpha_max, root( i ) )
        END DO
!       WRITE( control%out, "( ' l, c0, c1, c2 a ', I6, 4ES12.4 )" )           &
!         l, c0, c1, c2, alpha_max
      END DO
!     WRITE( control%out, "( ' alpha_max ', ES12.4 )" ) alpha_max

!  Given omega_u > 1), find alpha_max, the largest value in [0,1] for which
!
!  c2_ij alpha^2 + c1_ij alpha + c0_ij >= omega_u mu_i
!     (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for all alpha <= alpha_max

!     WRITE( control%out, "( ' high ' )" )
      DO l = 1, nbnds
        c2 = COEF2( l ) ; c1 = COEF1( l ) ; c0 = COEF0( l ) - omega_u * MU( l )
        CALL ROOTS_quadratic( c0, c1, c2, tol, nroots, root( 1 ), root( 2 ),   &
                              roots_debug )
        DO i = 1, nroots
          IF ( root( i ) > tenm7 ) alpha_max = MIN( alpha_max, root( i ) )
        END DO
!       WRITE( control%out, "( ' l, c0, c1, c2 a ', I6, 4ES12.4 )" )           &
!         l, c0, c1, c2, alpha_max
      END DO

!  Compute alpha_est, the global minimizer of the overestimator
!
!  o(alpha) =
!    alpha^2 [ sum_i=1^n,j=1,2 | q_ij | + sum_i=1^m,j=3,4 | q_ij | ] +
!    (1-alpha) [ sum_i=1^n,j=1,2 | c_ij - mu_i | +
!                sum_i=1^m,j=3,4 | c_ij - mu_i | ] +
!    (1 - alpha) res_prim_dual
!
! for all alpha in [0,alpha_max]

      c0 = res_prim_dual ; c2 = zero
      DO l = 1, nbnds
        c0 = c0 + ABS( COEF0( l ) - MU( l ) )
        c2 = c2 + ABS( COEF2( l ) )
      END DO

      IF ( c2 /= zero ) THEN
        alpha_est = MIN( alpha_max, half * c0 / c2 )
      ELSE
        alpha_est = alpha_max
      END IF

      q_alpha_est = ( one - alpha_est ) * res_prim_dual
      DO l = 1, nbnds
        q_alpha_est = q_alpha_est + ABS( COEF0( l ) - MU( l ) +               &
          alpha_est * ( COEF1( l ) + alpha_est * COEF2( l ) ) )
      END DO

!  Next find all of the roots (the breakpoints) of
!     c2_ij alpha^2 + c1_ij alpha + c0_ij - mu_i =0
!  (i=1,n,j=1,2 & i=1,m,j=3,4) that lie in [0,alpha_max].

      nbreak = 0
      DO l = 1, nbnds
        c2 = COEF2( l ) ; c1 = COEF1( l ) ; c0 = COEF0( l ) - MU( l )
        CALL ROOTS_quadratic( c0, c1, c2, tol, nroots, root( 1 ), root( 2 ),   &
                              roots_debug )
        IF ( nroots >= 1 ) THEN
          IF ( root( 1 ) > zero .AND. root( 1 ) < alpha_max ) THEN
            nbreak = nbreak + 1
            BREAKP( nbreak ) = root( 1 )
            IBREAK( nbreak ) = l
          END IF
        END IF
        IF ( nroots == 2 ) THEN
          IF ( root( 2 ) > zero .AND. root( 2 ) < alpha_max ) THEN
            nbreak = nbreak + 1
            BREAKP( nbreak ) = root( 2 )
            IBREAK( nbreak ) = l
          END IF
        END IF
      END DO

!  Now use heapsort to arrange the breakpoints in increasing order.

      CALL SORT_heapsort_build( nbreak, BREAKP, inheap, ix = IBREAK )
      IF ( details ) WRITE( control%out,                                       &
       "(  I8, ' breakpoints, interval is [0,', ES12.4, ']' )" )               &
          nbreak, alpha_max

!  Build the coefficients of the quadratic q(alpha) at the start of
!  the initial interval

      c0 = res_prim_dual ; c1 = - res_prim_dual ; c2 = zero
      DO l = 1, nbnds
        c0mu = COEF0( l ) - MU( l )
        IF ( c0mu < zero .OR.                                                  &
             ( c0mu == zero .AND.( COEF1( l ) < zero .OR.                      &
               ( COEF1( l ) == zero .AND. COEF2( l ) < zero ) ) ) ) THEN
          COEF0( l ) = - c0mu
          COEF1( l ) = - COEF1( l ) ; COEF2( l ) = - COEF2( l )
        ELSE
          COEF0( l ) = c0mu
        END IF
        c0 = c0 + COEF0( l ) ; c1 = c1 + COEF1( l ) ; c2 = c2 + COEF2( l )
      END DO
!     WRITE(6,"( ' c0, c1, c2 ', 3ES12.4 )" ) c0, c1, c2

!  At each stage, consider the piecewise quadratic function between the
!  current breakpoint and the next.

      alpha_l = zero
      q_l = c0
      q_min = q_l
      alpha_min = alpha_l

      cluster = 0
      IF ( details ) WRITE( control%out, 2000 )
      DO
        IF ( nbreak > 0 ) THEN
          alpha_u = BREAKP( 1 )
          l = IBREAK( 1 )

!  Check that the interval upper bound should not really be considered
!  as a cluster of nearly identical points. If so, simply add the point
!  to the cluster, adjust the coefficient of the quadratic function and
!  find the next break point.

          IF ( alpha_u - alpha_l < close_tol ) THEN
            COEF0( l ) = - COEF0( l ) ; COEF1( l ) = - COEF1( l )
            COEF2( l ) = - COEF2( l )
            c0 = c0 + two * COEF0( l ) ; c1 = c1 + two * COEF1( l )
            c2 = c2 + two * COEF2( l )
            CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, ix = IBREAK )
            nbreak = nbreak - 1
            CYCLE
          END IF
        ELSE
          alpha_u = alpha_max
        END IF
        cluster = cluster + 1

!  Compute the minimizer of q(alpha) in [alpha_l,alpha_u], and
!  see if this is the new candidate incumbent global minimizer

        q_u = c0 + alpha_u * ( c1 + alpha_u * c2 )

        IF ( debug ) THEN
          WRITE(  control%out, "( ' c0, c1, c2 ', 3ES12.4 )" ) c0, c1, c2
          res_cs = ( 1 - alpha_u ) * res_prim_dual
          DO i = dims%x_free + 1, dims%x_u_start - 1
            res_cs = res_cs + ABS( ( DIST_Z_l( i ) + alpha_u * DZ_l( i ) ) *   &
                         ( DIST_X_l( i ) + alpha_u * DX( i ) ) - MU_X_l( i ) )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            res_cs = res_cs + ABS( ( DIST_Z_l( i ) + alpha_u * DZ_l( i ) ) *   &
                         ( DIST_X_l( i ) + alpha_u * DX( i ) ) - MU_X_l( i ) ) &
                            + ABS( ( DIST_Z_u( i ) - alpha_u * DZ_u( i ) ) *   &
                         ( DIST_X_u( i ) - alpha_u * DX( i ) ) - MU_X_u( i ) )
          END DO
          DO i = dims%x_l_end + 1, n
            res_cs = res_cs + ABS( ( DIST_Z_u( i ) - alpha_u * DZ_u( i ) ) *   &
                         ( DIST_X_u( i ) - alpha_u * DX( i ) ) - MU_X_u( i ) )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            res_cs = res_cs + ABS( ( DIST_Y_l( i ) + alpha_u * DY_l( i ) ) *   &
                         ( DIST_C_l( i ) + alpha_u * DC( i ) ) - MU_C_l( i ) )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            res_cs = res_cs + ABS( ( DIST_Y_l( i ) + alpha_u * DY_l( i ) ) *   &
                         ( DIST_C_l( i ) + alpha_u * DC( i ) ) - MU_C_l( i ) ) &
                            + ABS( ( DIST_Y_u( i ) - alpha_u * DY_u( i ) ) *   &
                         ( DIST_C_u( i ) - alpha_u * DC( i ) ) - MU_C_u( i ) )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            res_cs = res_cs + ABS( ( DIST_Y_u( i ) - alpha_u * DY_u( i ) ) *   &
                         ( DIST_C_u( i ) - alpha_u * DC( i ) ) - MU_C_u( i ) )
          END DO
          WRITE( control%out, "( ' q_upper = ', 56X, ES12.4 )" ) res_cs
        END IF

        IF ( c2 > zero ) THEN
          alpha_m = - half * c1 / c2
          IF ( alpha_m > alpha_l .AND. alpha_m < alpha_u ) THEN
            q_m = c0 + alpha_m * ( c1 + alpha_m * c2 )
            IF ( q_m < q_min ) THEN
              q_min = q_m
              alpha_min = alpha_m
            END IF
!           WRITE( control%out, 2010 )                                         &
            IF ( details ) WRITE( control%out, 2010 )                          &
              cluster, alpha_l, alpha_m, alpha_u, q_l, q_m, q_u
          ELSE
            IF ( q_u < q_min ) THEN
              q_min = q_u
              alpha_min = alpha_u
            END IF
!           WRITE( control%out, 2020 )                                         &
            IF ( details ) WRITE( control%out, 2020 )                          &
              cluster, alpha_l, alpha_u, q_l, q_u
          END IF
        ELSE
          IF ( q_u < q_min ) THEN
            q_min = q_u
            alpha_min = alpha_u
          END IF
!         WRITE( control%out, 2020 )                                           &
          IF ( details ) WRITE( control%out, 2020 )                            &
            cluster, alpha_l, alpha_u, q_l, q_u
        END IF

!  Move to the end of the interval

        IF ( nbreak == 0 ) EXIT
        alpha_l = alpha_u
        q_l = q_u

!  Adjust the coefficients of the quadratic to reflect the values in the
!  next interval

        COEF0( l ) = - COEF0( l ) ; COEF1( l ) = - COEF1( l )
        COEF2( l ) = - COEF2( l )
        c0 = c0 + two * COEF0( l ) ; c1 = c1 + two * COEF1( l )
        c2 = c2 + two * COEF2( l )
        CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, ix = IBREAK )
        nbreak = nbreak - 1
      END DO

      alpha = alpha_min
      IF ( details ) THEN
        WRITE( control%out, "( /, '   alpha_max       alpha       alpha_est ', &
       &  '      q(alpha)     q(alpha_est) ', /, 3ES14.7, 2ES15.7 )" )         &
          alpha_max, alpha, alpha_est, q_min, q_alpha_est
      END IF

      RETURN

! Non-executable statements

 2000 FORMAT( ' cluster    alpha_l     alpha_m     alpha_u     ',              &
              '    q_l         q_m         q_u')
 2010 FORMAT( I7, 6ES12.4 )
 2020 FORMAT( I7, ES12.4, '      -     ', 2ES12.4, '      -     ', ES12.4  )

!  End of subroutine WCP_min_piecewise_quadratic

      END SUBROUTINE WCP_min_piecewise_quadratic

!-*- W C P _ m i n _ p i e c e w i s e _ q u a r t i c  S U B R O U T I N E -*-

      SUBROUTINE WCP_min_piecewise_quartic( dims, n, m, nbnds, DIST_Z_l,       &
                                            DIST_Z_u, DZ_l, DZ_u, DZ_c_l,      &
                                            DZ_c_u, DIST_X_l, DIST_X_u, DX,    &
                                            DX_c, DIST_Y_l, DIST_Y_u, DY_l,    &
                                            DY_u, DY_c_l, DY_c_u, DIST_C_l,    &
                                            DIST_C_u, DC, DC_c, res_prim_dual, &
                                            MU_X_l, MU_X_u, MU_C_l, MU_C_u,    &
                                            MU, omega_l, omega_u,              &
                                            alpha, alpha_max,                  &
                                            COEF4, COEF3, COEF1, COEF0,        &
                                            BREAKP, IBREAK,                    &
                                            print_level, control )
!                                           inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a global minimizer of the piecewise quadratic function
!
!  q(alpha) =
!  sum_i=1^n,j=1,2 | c4_ij alpha^4 + c3_ij alpha^3
!                    + c1_ij alpha + c0_ij - mu_i | +
!  sum_i=1^m,j=3,4 | c4_ij alpha^4 + c3_ij alpha^3
!                   + c1_ij alpha + c0_ij - mu_i | +
!    (1 - alpha) res_prim_dual
!
! for all alpha in [0,1] for which additionally
!
!  c4_ij alpha^4 + c3_ij alpha^3 + c1_ij alpha + c0_ij >= omega_l mu_i
!
!  and
!
!  c4_ij alpha^4 + c3_ij alpha^3 + c1_ij alpha + c0_ij <= omega_u mu_i
!
!    (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for omega_l in (0,1), omega_u > 1
!
!  Here:
!    j      c4__ij                       c3_ij
!    1  dx_c_i dz_c^L_i     dx_c_i dz^L_i + dx_i dz_c^L_i
!    2  dx_c_i dz_c^U_i     dx_c_i dz^U_i + dx_i dz_c^U_i
!    3  dc_c_i dy_c^L_i     dc_c_i dy^L_i + dc_i dy_c^L_i
!    4  dc_c_i dy_c^U_i     dc_c_i dy^U_i + dc_i dy_c^U_i

!    j                   c1_ij                       c0_ij
!    1  (x_i - x^L_i) dz^L_i + dx_i z^L_i    (x_i - x^L_i) z^L_i
!    2  (x_i - x^U_i) dz^U_i + dx_i z^U_i    (x_i - x^U_i) z^U_i
!    3  (c_i - c^L_i) dy^L_i + dc_i y^L_i    (c_i - c^L_i) y^L_i
!    4  (c_i - c^U_i) dy^U_i + dc_i y^U_i    (c_i - c^U_i) y^U_i

!  and res_prim_dual = || residual primal/dual optimality ||
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( WCP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, nbnds, print_level
      REAL ( KIND = wp ), INTENT( IN ) :: res_prim_dual, omega_l, omega_u
      REAL ( KIND = wp ), INTENT( OUT ) :: alpha, alpha_max
      INTEGER, INTENT( OUT ), DIMENSION( 4 * nbnds ) :: IBREAK
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: DX, DX_c
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: DIST_Z_l, DZ_l, DZ_c_l, &
                                                       DIST_X_l, MU_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%x_u_start : n ) :: DIST_Z_u, DZ_u, DZ_c_u,             &
                                           DIST_X_u, MU_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_Y_l, DY_l, DY_c_l,  &
                                                      DIST_C_l, MU_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
        DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_Y_u, DY_u, DY_c_u,  &
                                                      DIST_C_u, MU_C_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : m ) :: DC, DC_c
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( nbnds ) :: COEF4, COEF3, COEF1, COEF0, MU
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 * nbnds ) :: BREAKP
      TYPE ( WCP_control_type ), INTENT( IN ) :: control
!     TYPE ( WCP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, l, nroots, inheap, nbreak, cluster
      REAL ( KIND = wp ) :: c0, c1, c2, c3, c4, c0mu, root( 4 ), tiny4
      REAL ( KIND = wp ) :: tol, close_tol, res_cs, alpha_min
      REAL ( KIND = wp ) :: alpha_l, alpha_u, alpha_m, q_l, q_u, q_m, q_min
      LOGICAL :: debug, details

      details = print_level >= 4
      debug = print_level >= 5
      tol = epsmch ** 0.75
      close_tol = tol
!     tiny4 = ten ** ( - 20 )
      tiny4 = ten ** ( - 16 )

!  Compute the coefficients c4_ij, c3_ij, c1_ij and c0_ij

      l = 0
      DO i = dims%x_free + 1, dims%x_l_end
        l = l + 1
        COEF4( l ) = DX_c( i ) * DZ_c_l( i )
        COEF3( l ) = DX_c( i ) * DZ_l( i ) + DX( i ) * DZ_c_l( i )
        IF ( ABS( COEF4( l ) ) < tiny4 * ABS( COEF3( l ) ) ) COEF4( l ) = zero
!       WRITE(6,*) DX_c( i ) * DIST_Z_l( i ) +  DX( i ) * DZ_l( i )            &
!                  + DIST_X_L( i ) * DZ_c_l( i )
        COEF1( l ) = DIST_X_l( i ) * DZ_l( i ) + DX( i ) * DIST_Z_l( i )
        COEF0( l ) = DIST_X_l( i ) * DIST_Z_l( i )
        MU( l ) = MU_X_l( i )
      END DO
      DO i = dims%x_u_start, n
        l = l + 1
        COEF4( l ) = DX_c( i ) * DZ_c_u( i )
        COEF3( l ) = DX_c( i ) * DZ_u( i ) + DX( i ) * DZ_c_u( i )
        IF ( ABS( COEF4( l ) ) < tiny4 * ABS( COEF3( l ) ) ) COEF4( l ) = zero
!      WRITE(6,*) - DX_c( i ) * DIST_Z_u( i ) +  DX( i ) * DZ_u( i )           &
!                 - DIST_X_u( i ) * DZ_c_u( i )
        COEF1( l ) = - DIST_X_u( i ) * DZ_u( i ) - DX( i ) * DIST_Z_u( i )
        COEF0( l ) = DIST_X_u( i ) * DIST_Z_u( i )
        MU( l ) = MU_X_u( i )
      END DO
      DO i = dims%c_l_start, dims%c_l_end
        l = l + 1
        COEF4( l ) = DC_c( i ) * DY_c_l( i )
        COEF3( l ) = DC_c( i ) * DY_l( i ) + DC( i ) * DY_c_l( i )
        IF ( ABS( COEF4( l ) ) < tiny4 * ABS( COEF3( l ) ) ) COEF4( l ) = zero
!       WRITE(6,*) DC_c( i ) * Y_l( i ) +  DC( i ) * DY_l( i )                 &
!                  + DIST_C_L( i ) * DY_c_l( i )
        COEF1( l ) = DIST_C_l( i ) * DY_l( i ) + DC( i ) * DIST_Y_l( i )
        COEF0( l ) = DIST_C_l( i ) * DIST_Y_l( i )
        MU( l ) = MU_C_l( i )
      END DO
      DO i = dims%c_u_start, dims%c_u_end
        l = l + 1
        COEF4( l ) = DC_c( i ) * DY_c_u( i )
        COEF3( l ) = DC_c( i ) * DY_u( i ) + DC( i ) * DY_c_u( i )
        IF ( ABS( COEF4( l ) ) < tiny4 * ABS( COEF3( l ) ) ) COEF4( l ) = zero
!        WRITE(6,*) DC_c( i ) * Y_u( i ) +  DC( i ) * DY_u( i )                &
!                   - DIST_C_u( i ) * DY_c_u( i )
        COEF1( l ) = - DIST_C_u( i ) * DY_u( i ) - DC( i ) * DIST_Y_u( i )
        COEF0( l ) = DIST_C_u( i ) * DIST_Y_u( i )
        MU( l ) = MU_C_u( i )
      END DO

!  Given omega_l in (0,1), find alpha_max, the largest value in [0,1] for which
!
!  c4_ij alpha^4 + c3_ij alpha^3 + c1_ij alpha + c0_ij
!    >= omega_l mu_i    (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for all alpha <= alpha_max

      c2 = zero
      alpha_max = one
      DO l = 1, nbnds
        c4 = COEF4( l ) ; c3 = COEF3( l )
        c1 = COEF1( l ) ; c0 = COEF0( l ) - omega_l * MU( l )
        CALL ROOTS_quartic( c0, c1, c2, c3, c4, tol, nroots, root( 1 ),        &
                            root( 2 ), root( 3 ), root( 4 ), roots_debug )
        DO i = 1, nroots
          IF ( root( i ) > tenm10 ) alpha_max = MIN( alpha_max, root( i ) )
        END DO
      END DO

!  Given omega_u > 1, find alpha_max, the largest value in [0,1] for which
!
!  c4_ij alpha^4 + c3_ij alpha^3 + c1_ij alpha + c0_ij
!    <= omega_u mu_i    (i=1,n,j=1,2 & i=1,m,j=3,4)
!
!  for all alpha <= alpha_max

      DO l = 1, nbnds
        c4 = COEF4( l ) ; c3 = COEF3( l )
        c1 = COEF1( l ) ; c0 = COEF0( l ) - omega_u * MU( l )
        CALL ROOTS_quartic( c0, c1, c2, c3, c4, tol, nroots, root( 1 ),        &
                            root( 2 ), root( 3 ), root( 4 ), roots_debug )
        DO i = 1, nroots
          IF ( root( i ) > tenm10 ) alpha_max = MIN( alpha_max, root( i ) )
        END DO
      END DO

!  Next find all of the roots (the breakpoints) of
!     c4_ij alpha^4 + c3_ij alpha^3 + c1_ij alpha + c0_ij - mu_i = 0
!  (i=1,n,j=1,2 & i=1,m,j=3,4) that lie in [0,alpha_max].

      nbreak = 0
      DO l = 1, nbnds
        c4 = COEF4( l ) ; c3 = COEF3( l )
        c1 = COEF1( l ) ; c0 = COEF0( l ) - MU( l )
        CALL ROOTS_quartic( c0, c1, c2, c3, c4, tol, nroots, root( 1 ),        &
                            root( 2 ), root( 3 ), root( 4 ), roots_debug )
        DO i = 1, nroots
          IF ( root( i ) > zero .AND. root( i ) < alpha_max ) THEN
            nbreak = nbreak + 1
            BREAKP( nbreak ) = root( i )
            IBREAK( nbreak ) = l
          END IF
        END DO
      END DO

!  Now use heapsort to arrange the breakpoints in increasing order.

      CALL SORT_heapsort_build( nbreak, BREAKP, inheap, ix = IBREAK )
      IF ( details ) WRITE( control%out,                                       &
       "(  I8, ' breakpoints, interval is [0,', ES12.4, ']' )" )               &
          nbreak, alpha_max

!  Build the coefficients of the quadratic q(alpha) at the start of
!  the initial interval

      c0 = res_prim_dual ; c1 = - res_prim_dual ; c3 = zero ; c4 = zero
      DO l = 1, nbnds
        c0mu = COEF0( l ) - MU( l )
        IF ( c0mu < zero .OR.                                                  &
             ( c0mu == zero .AND.( COEF1( l ) < zero .OR.                      &
               ( COEF1( l ) == zero .AND. ( COEF3( l ) < zero .OR.             &
                 ( COEF3( l ) == zero .AND. COEF4( l ) < zero ) ) ) ) ) ) THEN
          COEF0( l ) = - c0mu ; COEF1( l ) = - COEF1( l )
          COEF3( l ) = - COEF3( l ) ; COEF4( l ) = - COEF4( l )
        ELSE
          COEF0( l ) = c0mu
        END IF
        c0 = c0 + COEF0( l ) ; c1 = c1 + COEF1( l )
        c3 = c3 + COEF3( l ) ; c4 = c4 + COEF4( l )
      END DO
!     WRITE(6,"( ' c0, c1, c3, c4 ', 4ES12.4 )" ) c0, c1, c3, c4

!  At each stage, consider the piecewise quadratic function between the
!  current breakpoint and the next.

      alpha_l = zero
      q_l = c0
      q_min = q_l
      alpha_min = alpha_l

      cluster = 0
      IF ( details ) WRITE( control%out, 2000 )
      DO
        IF ( nbreak > 0 ) THEN
          alpha_u = BREAKP( 1 )
          l = IBREAK( 1 )

!  Check that the interval upper bound should not really be considered
!  as a cluster of nearly identical points. If so, simply add the point
!  to the cluster, adjust the coefficient of the quadratic function and
!  find the next break point.

          IF ( alpha_u - alpha_l < close_tol ) THEN
            COEF0( l ) = - COEF0( l ) ; COEF1( l ) = - COEF1( l )
            COEF3( l ) = - COEF3( l ) ; COEF4( l ) = - COEF4( l ) ;
            c0 = c0 + two * COEF0( l ) ; c1 = c1 + two * COEF1( l )
            c3 = c3 + two * COEF3( l ) ; c4 = c4 + two * COEF4( l )
            CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, ix = IBREAK )
            nbreak = nbreak - 1
            CYCLE
          END IF
        ELSE
          alpha_u = alpha_max
        END IF
        cluster = cluster + 1

!  Compute the minimizer of q(alpha) in [alpha_l,alpha_u], and
!  see if this is the new candidate incumbent global minimizer

!       q_u = c0 + alpha_l * ( c1 + alpha_l ** 2 * ( c3 + alpha_l * c4 ) )
!       write(6,"( ' q at start ', ES12.4 )" ) q_u
        q_u = c0 + alpha_u * ( c1 + alpha_u ** 2 * ( c3 + alpha_u * c4 ) )
!       write(6,"( ' res_cs ', ES12.4 )" ) q_u - res_prim_dual * ( 1 - alpha_u )

        IF ( q_u < q_min ) THEN
          q_min = q_u
          alpha_min = alpha_u
        END IF

        IF ( debug ) THEN
          WRITE(  control%out, "( ' c0, c1, c3, c4 ', 4ES12.4 )" ) c0, c1, c3, c4
          res_cs = ( 1 - alpha_u ) * res_prim_dual
          DO i = dims%x_free + 1, dims%x_u_start - 1
            res_cs = res_cs + ABS(                                             &
          ( DIST_Z_l( i ) + alpha_u * ( DZ_l( i ) + alpha_u * DZ_c_l( i ) ) ) *&
          ( DIST_X_l( i ) + alpha_u * ( DX( i ) + alpha_u * DX_c( i ) ) )      &
                - MU_X_l( i ) )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            res_cs = res_cs + ABS(                                             &
          ( DIST_Z_l( i ) + alpha_u * ( DZ_l( i ) + alpha_u * DZ_c_l( i ) ) ) *&
          ( DIST_X_l( i ) + alpha_u * ( DX( i ) + alpha_u * DX_c( i ) ) )      &
                - MU_X_l( i ) ) + ABS(                                         &
          ( DIST_Z_u( i ) - alpha_u * ( DZ_u( i ) + alpha_u * DZ_c_u( i ) ) ) *&
          ( DIST_X_u( i ) - alpha_u * ( DX( i ) + alpha_u * DX_c( i ) ) )      &
               - MU_X_u( i ) )
          END DO
          DO i = dims%x_l_end + 1, n
            res_cs = res_cs + ABS(                                             &
          ( DIST_Z_u( i ) - alpha_u * ( DZ_u( i ) + alpha_u * DZ_c_u( i ) ) ) *&
          ( DIST_X_u( i ) - alpha_u * ( DX( i ) + alpha_u * DX_c( i ) ) )      &
               - MU_X_u( i ) )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            res_cs = res_cs + ABS(                                             &
          ( DIST_Y_l( i ) + alpha_u * ( DY_l( i ) + alpha_u * DY_c_l( i ) ) ) *&
          ( DIST_C_l( i ) + alpha_u * ( DC( i ) + alpha_u * DC_c( i ) ) )      &
                - MU_C_l( i) )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            res_cs = res_cs + ABS(                                             &
          ( DIST_Y_l( i ) + alpha_u * ( DY_l( i ) + alpha_u * DY_c_l( i ) ) ) *&
          ( DIST_C_l( i ) + alpha_u * ( DC( i ) + alpha_u * DC_c( i ) ) )      &
                - MU_C_l( i ) ) + ABS(                                         &
          ( DIST_Y_u( i ) - alpha_u * ( DY_u( i ) + alpha_u * DY_c_u( i ) ) ) *&
          ( DIST_C_u( i ) - alpha_u * ( DC( i ) + alpha_u * DC_c( i ) ) )      &
               - MU_C_u( i ) )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            res_cs = res_cs + ABS(                                             &
          ( DIST_Y_u( i ) - alpha_u * ( DY_u( i ) + alpha_u * DY_c_u( i ) ) ) *&
          ( DIST_C_u( i ) - alpha_u * ( DC( i ) + alpha_u * DC_c( i ) ) )      &
               - MU_C_u( i ) )
          END DO
          WRITE( control%out, "( ' q_upper = ', 56X, ES12.4 )" ) res_cs
        END IF

!  Find the stationary points of q(alpha) (if any)

        CALL ROOTS_cubic( c1, two * c2, three * c3, four * c4, tol, nroots,    &
                          root( 1 ), root( 2 ), root( 3 ), roots_debug )

!  If the stationary points lie within the interval, check to see whether
!  any of them gives a lower function value

        DO i = 1, nroots
          alpha_m = root( i )
          IF ( alpha_m > alpha_l .AND. alpha_m < alpha_u ) THEN
            q_m = c0 + alpha_m * ( c1 + alpha_m ** 2 * ( c3 + alpha_m * c4 ) )
            IF ( q_m < q_min ) THEN
              q_min = q_m
              alpha_min = alpha_m
            END IF
          END IF
        END DO

        IF ( alpha_min > alpha_l .AND. alpha_min < alpha_u ) THEN
          IF ( details ) WRITE( control%out, 2010 )                            &
            cluster, alpha_l, alpha_min, alpha_u, q_l, q_min, q_u
        ELSE
          IF ( details ) WRITE( control%out, 2020   )                          &
            cluster, alpha_l, alpha_u, q_l, q_u
        END IF

!  Move to the end of the interval

        IF ( nbreak == 0 ) EXIT
        alpha_l = alpha_u
        q_l = q_u

!  Adjust the coefficients of the quadratic to reflect the values in the
!  next interval

        COEF0( l ) = - COEF0( l ) ; COEF1( l ) = - COEF1( l )
        COEF3( l ) = - COEF3( l ) ; COEF4( l ) = - COEF4( l )
        c0 = c0 + two * COEF0( l ) ; c1 = c1 + two * COEF1( l )
        c3 = c3 + two * COEF3( l ) ; c4 = c4 + two * COEF4( l )
        CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, ix = IBREAK )
        nbreak = nbreak - 1

      END DO
      alpha = alpha_min
      IF ( details )                                                           &
        WRITE( control%out, "( ' final step, value ', 2ES12.4 )" ) alpha, q_min

      RETURN

! Non-executable statements

 2000 FORMAT( ' cluster    alpha_l     alpha_m     alpha_u     ',              &
              '    q_l         q_m         q_u')
 2010 FORMAT( I7, 6ES12.4 )
 2020 FORMAT( I7, ES12.4, '      -     ', 2ES12.4, '      -     ', ES12.4  )

!  End of subroutine WCP_min_piecewise_quartic

      END SUBROUTINE WCP_min_piecewise_quartic

!!-*-*-*-*-*-*-   W C P _ A _ B Y _ C O L S   S U B R O U T I N E   -*-*-*-*-*-
!
!      SUBROUTINE WCP_A_by_cols( n, m, A_ne, A_val, A_col, A_ptr, B_val,       &
!                                B_row, B_colptr )
!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!!  Takes a matrix A stored by co-ordinates, and returns the same matrix
!!  as B stored by columns
!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!!  Dummy arguments
!
!      INTEGER, INTENT( IN ) :: n, m, A_ne
!      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
!      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
!      INTEGER, INTENT( OUT ), DIMENSION( A_ne ) :: B_row
!      INTEGER, INTENT( OUT ), DIMENSION( n + 1 ) :: B_colptr
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
!      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( A_ne ) :: B_val
!
!!  Local variables
!
!      INTEGER :: i, j, k, l
!
!!  count the number of nonzeros in each column
!
!      B_colptr( : n ) = 0
!      DO l = 1, A_ne
!        B_colptr( A_col( l ) ) = B_colptr( A_col( l ) ) + 1
!      END DO
!
!!  set the starting addresses for each column
!
!      j = 1
!      DO i = 1, n
!        l = j
!        j = j + B_colptr( i )
!        B_colptr( i ) = l
!      END DO
!
!!  move the entries from A to B
!
!      DO i = 1, m
!        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
!          j = A_col( l )
!          k = B_colptr( j )
!          B_val( k ) = A_val( l )
!          B_row( k ) = i
!          B_colptr( j ) = B_colptr( j ) + 1
!        END DO
!      END DO
!
!!  reset the starting addresses
!
!      DO i = n, 1, - 1
!        B_colptr( i + 1 ) = B_colptr( i )
!      END DO
!      B_colptr( 1 ) = 1
!
!      RETURN
!
!!  End of WCP_A_by_cols
!
!      END SUBROUTINE WCP_A_by_cols

!  End of module WCP

   END MODULE GALAHAD_WCP_double
