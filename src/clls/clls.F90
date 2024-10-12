! THIS VERSION: GALAHAD 5.1 - 2024-10-04 AT 14:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ C L L S    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1, July 20th 2022
!   as a modified version of ccqp

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CLLS_precision

!      -----------------------------------------------
!     | Minimize the least-squares objective function |
!     |                                               |
!     |  1/2 || A_o x - b ||^2 + 1/2 weight || x ||^2 |
!     |                                               |
!     | subject to the linear constraints and bounds  |
!     |                                               |
!     |             c_l <= A x <= c_u                 |
!     |             x_l <=  x <= x_u                  |
!     |                                               |
!     | using an infeasible-point primal-dual method  |
!      -----------------------------------------------

      USE GALAHAD_KINDS_precision
!$    USE omp_lib
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural,           &
                                STRING_ies, STRING_are, STRING_ordinal
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SMT_precision
      USE GALAHAD_QPT_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_LSP_precision, CLLS_dims_type => QPT_dimensions_type
      USE GALAHAD_QPD_precision, CLLS_data_type => QPD_data_type,              &
                                 CLLS_AX => QPD_AX, CLLS_abs_AX => QPD_abs_AX, &
                                 CLLS_AoX => QPD_A_by_col_X,                   &
                                 CLLS_abs_AoX => QPD_abs_A_by_col_X
      USE GALAHAD_CQP_precision, CLLS_merit_value => CQP_merit_value,          &
                              CLLS_compute_stepsize => CQP_compute_stepsize,   &
                              CLLS_compute_v_alpha => CQP_compute_v_alpha,     &
                              CLLS_compute_lmaxstep => CQP_compute_lmaxstep,   &
                              CLLS_compute_pmaxstep => CQP_compute_pmaxstep,   &
                              CLLS_indicators => CQP_indicators
      USE GALAHAD_ROOTS_precision
      USE GALAHAD_SORT_precision, ONLY: SORT_inverse_permute
      USE GALAHAD_FDC_precision
      USE GALAHAD_SLS_precision
      USE GALAHAD_CRO_precision
      USE GALAHAD_FIT_precision
      USE GALAHAD_CHECKPOINT_precision
      USE GALAHAD_RPD_precision, ONLY: RPD_inform_type,                        &
                                       RPD_write_qp_problem_data

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CLLS_initialize, CLLS_read_specfile, CLLS_solve,               &
                CLLS_solve_main, CLLS_terminate,                               &
                QPT_problem_type, SMT_type, SMT_put, SMT_get,                  &
                CLLS_AX, CLLS_data_type, CLLS_dims_type, CLLS_indicators,      &
                CLLS_full_initialize, CLLS_full_terminate,                     &
                CLLS_import, CLLS_solve_clls,                                  &
                CLLS_reset_control, CLLS_information

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE CLLS_initialize
       MODULE PROCEDURE CLLS_initialize, CLLS_full_initialize
     END INTERFACE CLLS_initialize

     INTERFACE CLLS_terminate
       MODULE PROCEDURE CLLS_terminate, CLLS_full_terminate
     END INTERFACE CLLS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER ( KIND = ip_ ), PARAMETER :: max_sc = 200
      INTEGER ( KIND = ip_ ), PARAMETER :: no_last = - 1000
      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point1 = 0.1_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point01 = 0.01_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: four = 4.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: eight = 8.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: sixteen = 16.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: hundred = 100.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: thousand = 1000.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = rp_ ), PARAMETER :: tenm5 = ten ** ( - 5 )
      REAL ( KIND = rp_ ), PARAMETER :: tenm7 = ten ** ( - 7 )
      REAL ( KIND = rp_ ), PARAMETER :: tenm10 = ten ** ( - 10 )
      REAL ( KIND = rp_ ), PARAMETER :: ten4 = ten ** 4
      REAL ( KIND = rp_ ), PARAMETER :: ten5 = ten ** 5
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = rp_ ), PARAMETER :: onemeps = one - epsmch
      REAL ( KIND = rp_ ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = rp_ ), PARAMETER :: rminvr_zero = epsmch
      REAL ( KIND = rp_ ), PARAMETER :: twentyeps = two * teneps
#ifdef REAL_32
      REAL ( KIND = rp_ ), PARAMETER :: stop_alpha = ten ** ( - 7 )
#else
      REAL ( KIND = rp_ ), PARAMETER :: stop_alpha = ten ** ( - 15 )
#endif
      REAL ( KIND = rp_ ), PARAMETER :: relative_pivot_default = 0.01_rp_

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CLLS_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   any printing will start on this iteration

        INTEGER ( KIND = ip_ ) :: start_print = - 1

!   any printing will stop on this iteration

        INTEGER ( KIND = ip_ ) :: stop_print = - 1

!   at most maxit inner iterations are allowed

        INTEGER ( KIND = ip_ ) :: maxit = 1000

!   the number of iterations for which the overall infeasibility
!     of the problem is not reduced by at least a factor %reduce_infeas
!     before the problem is flagged as infeasible (see reduce_infeas)

        INTEGER ( KIND = ip_ ) :: infeas_max = 10

!   the initial value of the barrier parameter will not be changed for the
!     first muzero_fixed iterations
!
        INTEGER ( KIND = ip_ ) :: muzero_fixed = 0

!   indicate whether and how much of the input problem
!    should be restored on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER ( KIND = ip_ ) :: restore_problem = 2

!   specifies the type of indicator function used. Pssible values are

!     1 primal indicator: constraint active <=> distance to nearest bound
!         <= %indicator_p_tol
!     2 primal-dual indicator: constraint active <=> distance to nearest bound
!        <= %indicator_tol_pd * size of corresponding multiplier
!     3 primal-dual indicator: constraint active <=> distance to nearest bound
!        <= %indicator_tol_tapia * distance to same bound at previous iteration

        INTEGER ( KIND = ip_ ) :: indicator_type = 2

!   which residual trajectory should be used to aim from the current iterate
!   to the solution

!     1 the Zhang linear residual trajectory
!     2 the Zhao-Sun quadratic residual trajectory
!     3 the Zhang arc ultimately switching to the Zhao-Sun residual trajectory
!     4 the mixed linear-quadratic residual trajectory
!     5 the Zhang arc ultimately switching to the mixed linear-quadratic
!       residual trajectory

        INTEGER ( KIND = ip_ ) :: arc = 1

!    the order of (Taylor/Puiseux) series to fit to the path data

        INTEGER ( KIND = ip_ ) :: series_order = 2

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER ( KIND = ip_ ) :: sif_file_device = 52

!    specifies the unit number to write generated QPLIB file describing the
!     current problem

        INTEGER ( KIND = ip_ ) :: qplib_file_device = 53

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = rp_ ) :: infinity = ten ** 19

!   the required absolute and relative accuracies for the primal infeasibility

        REAL ( KIND = rp_ ) :: stop_abs_p = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

        REAL ( KIND = rp_ ) :: stop_abs_d = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

        REAL ( KIND = rp_ ) :: stop_abs_c = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_c = epsmch

!   initial primal variables will not be closer than prfeas from their bounds

        REAL ( KIND = rp_ ) :: prfeas = ten4

!   initial dual variables will not be closer than dufeas from their bounds
!
        REAL ( KIND = rp_ ) :: dufeas = ten4

!   the initial value of the barrier parameter. If muzero is not positive,
!    it will be reset to an appropriate value

        REAL ( KIND = rp_ ) :: muzero = - one

!   the weight attached to primal-dual infeasibility compared to complementarity
!    when assessing step acceptance

        REAL ( KIND = rp_ ) :: tau = one

!   individual complementarities will not be allowed to be smaller than
!    gamma_c times the average value

        REAL ( KIND = rp_ ) :: gamma_c = tenm5

!   the average complementarity will not be allowed to be smaller than
!    gamma_f times the primal/dual infeasibility

        REAL ( KIND = rp_ ) :: gamma_f = tenm5

!   if the overall infeasibility of the problem is not reduced by at least a
!    factor reduce_infeas over %infeas_max iterations, the problem is flagged
!    as infeasible (see infeas_max)

        REAL ( KIND = rp_ ) :: reduce_infeas = one - point01

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = rp_ ) :: identical_bounds_tol = epsmch

!  start terminal extrapolation when mu reaches mu_pounce

        REAL ( KIND = rp_ ) :: mu_pounce = ten ** ( - 5 )

!   if %indicator_type = 1, a constraint/bound will be
!    deemed to be active <=> distance to nearest bound <= %indicator_p_tol

        REAL ( KIND = rp_ ) :: indicator_tol_p = epsmch

!   if %indicator_type = 2, a constraint/bound will be deemed to be active
!     <=> distance to nearest bound
!        <= %indicator_tol_pd * size of corresponding multiplier

        REAL ( KIND = rp_ ) :: indicator_tol_pd = 1.0_rp_

!   if %indicator_type = 3, a constraint/bound will be deemed to be active
!     <=> distance to nearest bound
!        <= %indicator_tol_tapia * distance to same bound at previous iteration

        REAL ( KIND = rp_ ) :: indicator_tol_tapia = 0.9_rp_

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = rp_ ) :: clock_time_limit = - one

!   the equality constraints will be preprocessed to remove any linear
!    dependencies if true

        LOGICAL :: remove_dependencies = .TRUE.

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if %just_feasible is true, the algorithm will stop as soon as a feasible
!     point is found. Otherwise, the optimal solution to the problem will be
!     found

        LOGICAL :: treat_separable_as_general = .FALSE.

!   if %treat_separable_as_general, is true, any separability in the
!    problem structure will be ignored

        LOGICAL :: just_feasible  = .FALSE.

!   if %getdua, is true, advanced initial values are obtained for the
!    dual variables

        LOGICAL :: getdua = .FALSE.

!  decide between Puiseux and Taylor series approximations to the arc

        LOGICAL :: puiseux = .TRUE.

!    try every order of series up to series_order?

        LOGICAL :: every_order = .TRUE.

!   if %feasol is true, the final solution obtained will be perturbed so that
!    variables close to their bounds are moved onto these bounds

        LOGICAL :: feasol = .FALSE.

!   if %balance_initial_complentarity is true, the initial complemetarity
!    is required to be balanced
!
        LOGICAL :: balance_initial_complentarity = .FALSE.
!
!  if %crossover is true, cross over the solution to one defined by
!   linearly-independent constraints if possible
!
        LOGICAL :: crossover = .TRUE.

!  if %reduced_pounce_system is true, eliminate fixed variables when
!  solving the linear system required by the attempted pounce to the solution

        LOGICAL :: reduced_pounce_system = .TRUE.

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

!  symmetric (indefinite) linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = "ssids" //         &
                                                             REPEAT( ' ', 25 )

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "CLLSPROB.SIF"  // REPEAT( ' ', 18 )

!  name of generated QPLIB file containing input problem

        CHARACTER ( LEN = 30 ) :: qplib_file_name =                            &
         "CLLSPROB.qplib"  // REPEAT( ' ', 16 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for SLS used by CLLS_pounce

        TYPE ( SLS_control_type ) :: SLS_pounce_control

!  control parameters for FIT

        TYPE ( FIT_control_type ) :: FIT_control

!  control parameters for ROOTS

        TYPE ( ROOTS_control_type ) :: ROOTS_control

!  control parameters for CRO

        TYPE ( CRO_control_type ) :: CRO_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CLLS_time_type

!  the total CPU time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = rp_ ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = rp_ ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = rp_ ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = rp_ ):: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = rp_ ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = rp_ ) :: clock_preprocess = 0.0

!  the clock time spent detecting linear dependencies

        REAL ( KIND = rp_ ) :: clock_find_dependent = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = rp_ ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CLLS_inform_type

!  return status. See CLLS_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the return status from the factorization

        INTEGER ( KIND = ip_ ) :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long_ ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long_ ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER ( KIND = ip_ ) :: nfacts = - 1

!  the total number of "wasted" function evaluations during the linesearch

        INTEGER ( KIND = ip_ ) :: nbacts = - 1

!  the number of threads used

        INTEGER ( KIND = ip_ ) :: threads = 1

!  the value of the objective function at the best estimate of the solution
!   determined by CLLS_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = rp_ ) :: complementary_slackness = HUGE( one )

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = rp_ ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  checkpoints(i) records the iteration at which the criticality measures
!   first fall below 10**-i, i = 1, ..., 16 (-1 means not achieved)

        INTEGER ( KIND = ip_ ), DIMENSION( 16 ) :: checkpointsIter = - 1
        REAL ( KIND = rp_ ), DIMENSION( 16 ) :: checkpointsTime = - one

!  timings (see above)

        TYPE ( CLLS_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

!  inform parameters for SLS_pounce

        TYPE ( SLS_inform_type ) :: SLS_pounce_inform

!  return information from FIT

        TYPE ( FIT_inform_type ) :: FIT_inform

!  return information from ROOTS

        TYPE ( ROOTS_inform_type ) :: ROOTS_inform

!  inform parameters for CRO

        TYPE ( CRO_inform_type ) :: CRO_inform

!  inform parameters for RPD

        TYPE ( RPD_inform_type ) :: RPD_inform
      END TYPE

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: CLLS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( CLLS_data_type ) :: CLLS_data
        TYPE ( CLLS_control_type ) :: CLLS_control
        TYPE ( CLLS_inform_type ) :: CLLS_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE CLLS_full_data_type

   CONTAINS

!-*-*-*-*-*-   C L L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CLLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for CLLS. This routine should be called before
!  CLLS_solve
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

      TYPE ( CLLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( CLLS_control_type ), INTENT( OUT ) :: control
      TYPE ( CLLS_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set real control parameters

      control%stop_abs_p = epsmch ** 0.33
!     control%stop_rel_p = epsmch ** 0.33
      control%stop_abs_c = epsmch ** 0.33
!     control%stop_rel_c = epsmch ** 0.33
      control%stop_abs_d = epsmch ** 0.33
!     control%stop_rel_d = epsmch ** 0.33
      control%indicator_tol_p = control%stop_abs_p

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%max_infeas = control%stop_abs_p
      control%FDC_control%prefix = '" - FDC:"                     '

!  Initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )
      control%SLS_control%prefix = '" - SLS:"                     '
      control%symmetric_linear_solver = inform%SLS_inform%solver

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%SLS_pounce_data,                               &
                           control%SLS_pounce_control,                         &
                           inform%SLS_pounce_inform )
      control%SLS_pounce_control%prefix = '" - SLS:"                     '

!  Set FIT control parameters

      CALL FIT_initialize( data%FIT_data, control%FIT_control,                 &
                           inform%FIT_inform )
      control%FIT_control%prefix = '" - FIT:"                     '

!  Set ROOTS control parameters

      CALL ROOTS_initialize( data%ROOTS_data, control%ROOTS_control,           &
                             inform%ROOTS_inform )
      control%ROOTS_control%tol = epsmch ** 0.75
      control%ROOTS_control%prefix = '" - ROOTS:"                   '

!  Set CRO control parameters

      CALL CRO_initialize( data%CRO_data, control%CRO_control,                 &
                           inform%CRO_inform )
      control%CRO_control%prefix = '" - CRO:"                     '

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of CLLS_initialize

      END SUBROUTINE CLLS_initialize

!- G A L A H A D -  C L L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE CLLS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for CLLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( CLLS_control_type ), INTENT( OUT ) :: control
     TYPE ( CLLS_inform_type ), INTENT( OUT ) :: inform

     CALL CLLS_initialize( data%clls_data, control, inform )

     RETURN

!  End of subroutine CLLS_full_initialize

     END SUBROUTINE CLLS_full_initialize

!-*-*-*-*-   C L L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE CLLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by CLLS_initialize could (roughly)
!  have been set as:

! BEGIN CLLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  maximum-number-of-iterations                      1000
!  maximum-number-of-pcg-iterations                  1000
!  maximum-poor-iterations-before-infeasible         200
!  barrier-fixed-until-iteration                     1
!  indicator-type-used                               3
!  arc-used                                          1
!  series-order                                      5
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  infinity-value                                    1.0D+19
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-5
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-5
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-5
!  mininum-initial-primal-feasibility                1000.0
!  mininum-initial-dual-feasibility                  1000.0
!  initial-barrier-parameter                         -1.0
!  feasibility-vs-complementarity-weight             1.0
!  balance-complentarity-factor                      1.0D-5
!  balance-feasibility-factor                        1.0D-5
!  poor-iteration-tolerance                          0.98
!  identical-bounds-tolerance                        1.0D-15
!  required-barrier-value-before-pounce              1.0D-5
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  treat-separable-as-general                        F
!  just-find-feasible-point                          F
!  balance-initial-complentarity                     F
!  get-advanced-dual-variables                       F
!  puiseux-series                                    T
!  try-every-order-of-series                         T
!  move-final-solution-onto-bound                    F
!  cross-over-solution                               T
!  solve-reduced-pounce-system                       T
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  symmetric-linear-equation-solver                  ssids
!  sif-file-name                                     CLLSPROB.SIF
!  qplib-file-name                                   CLLSPROB.qplib
!  output-line-prefix                                ""
! END CLLS SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( CLLS_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_print = start_print + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxit = stop_print + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infeas_max = maxit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: muzero_fixed = infeas_max + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: restore_problem = muzero_fixed + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_type = restore_problem + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: arc = indicator_type + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: series_order = arc + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_device = series_order + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qplib_file_device                   &
                                             = sif_file_device + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinity = qplib_file_device + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_p = infinity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_p = stop_abs_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_d = stop_rel_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_d = stop_abs_d + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_c = stop_rel_d + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_c = stop_abs_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prfeas = stop_rel_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dufeas = prfeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: muzero = dufeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tau = muzero + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: gamma_c = tau + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: gamma_f = gamma_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reduce_infeas = gamma_f + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: identical_bounds_tol                &
                                             = reduce_infeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: mu_pounce = identical_bounds_tol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_p = mu_pounce + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_pd                    &
                                            = indicator_tol_p + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: indicator_tol_tapia                 &
                                            = indicator_tol_pd + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit                      &
                                            = indicator_tol_tapia + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: remove_dependencies &
                                            = clock_time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: treat_zero_bounds_as_general        &
                                             = remove_dependencies + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: treat_separable_as_general          &
                                             = treat_zero_bounds_as_general + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: just_feasible                       &
                                             = treat_separable_as_general + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: getdua = just_feasible + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: puiseux = getdua + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: every_order = puiseux + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: feasol = every_order + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: balance_initial_complentarity       &
                                             = feasol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: crossover                           &
                                             = balance_initial_complentarity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = crossover + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: generate_sif_file                   &
                                             = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: generate_qplib_file                 &
                                             = generate_sif_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: symmetric_linear_solver             &
                                             = generate_qplib_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_name                       &
                                             = symmetric_linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = qplib_file_name + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'CLLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( infeas_max )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( muzero_fixed )%keyword = 'barrier-fixed-until-iteration'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( indicator_type )%keyword = 'indicator-type-used'
      spec( arc )%keyword = 'arc-used'
      spec( series_order )%keyword = 'series-order'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
      spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
      spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
      spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
      spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
      spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
      spec( prfeas )%keyword = 'mininum-initial-primal-feasibility'
      spec( dufeas )%keyword = 'mininum-initial-dual-feasibility'
      spec( muzero )%keyword = 'initial-barrier-parameter'
      spec( tau )%keyword = 'feasibility-vs-complementarity-weight'
      spec( gamma_c )%keyword = 'balance-complentarity-factor'
      spec( gamma_f )%keyword = 'balance-feasibility-factor'
      spec( reduce_infeas )%keyword = 'poor-iteration-tolerance'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( mu_pounce )%keyword = 'required-barrier-value-before-pounce'
      spec( indicator_tol_p )%keyword = 'primal-indicator-tolerance'
      spec( indicator_tol_pd )%keyword = 'primal-dual-indicator-tolerance'
      spec( indicator_tol_tapia )%keyword = 'tapia-indicator-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( treat_separable_as_general )%keyword = 'treat-separable-as-general'
      spec( just_feasible )%keyword = 'just-find-feasible-point'
      spec( getdua )%keyword = 'get-advanced-dual-variables'
      spec( puiseux )%keyword = 'puiseux-series'
      spec( every_order )%keyword = 'try-every-order-of-series'
      spec( feasol )%keyword = 'move-final-solution-onto-bound'
      spec( balance_initial_complentarity )%keyword =                          &
        'balance-initial-complentarity'
      spec( crossover )%keyword = 'cross-over-solution'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' clls: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( arc ),                                  &
                                 control%arc,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( series_order ),                         &
                                 control%series_order,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_device ),                    &
                                 control%qplib_file_device,                    &
                                 control%error )

!  Set real values

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
     CALL SPECFILE_assign_value( spec( prfeas ),                               &
                                 control%prfeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dufeas ),                               &
                                 control%dufeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero ),                               &
                                 control%muzero,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tau ),                                  &
                                 control%tau,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_c ),                              &
                                 control%gamma_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_f ),                              &
                                 control%gamma_f,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduce_infeas ),                        &
                                 control%reduce_infeas,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_pounce ),                            &
                                 control%mu_pounce,                            &
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
     CALL SPECFILE_assign_value( spec( treat_separable_as_general ),           &
                                 control%treat_separable_as_general,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( getdua ),                               &
                                 control%getdua,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( puiseux ),                              &
                                 control%puiseux,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( every_order ),                          &
                                 control%every_order,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasol ),                               &
                                 control%feasol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( balance_initial_complentarity ),        &
                                 control%balance_initial_complentarity,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crossover ),                            &
                                 control%crossover,                            &
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

!  Read the specfiles for SLS and SLS-POUNCE

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                       alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL SLS_read_specfile( control%SLS_pounce_control, device,            &
                       alt_specname = TRIM( alt_specname ) // '-SLS-POUNCE' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL SLS_read_specfile( control%SLS_pounce_control, device,            &
                       alt_specname = 'SLS-POUNCE' )
      END IF

!  Read the specfile for FIT

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FIT_read_specfile( control%FIT_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FIT' )
      ELSE
        CALL FIT_read_specfile( control%FIT_control, device )
      END IF

!  Read the specfile for CRO

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CRO_read_specfile( control%CRO_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CRO' )
      ELSE
        CALL CRO_read_specfile( control%CRO_control, device )
      END IF

!  Read the specfile for ROOTS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL ROOTS_read_specfile( control%ROOTS_control, device,               &
                              alt_specname = TRIM( alt_specname ) // '-ROOTS' )
      ELSE
        CALL ROOTS_read_specfile( control%ROOTS_control, device )
      END IF

      RETURN

      END SUBROUTINE CLLS_read_specfile

!-*-*-*-*-*-*-*-*-*-   C L L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE CLLS_solve( prob, data, control, inform,                      &
                             regularization_weight, W )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear least-squares objective function
!
!         1/2 || A_o x - b ||_W^2 + 1/2 weight || x ||^2
!
!  where
!
!             (c_l)_i <= (A x)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  x is a vector of n components ( x_1, .... , x_n ),  A_o and A are o by n
!  and m by n matrices, any of the bounds (c_l)_i, (c_u)_i, (x_l)_i, (x_u)_i
!  may be infinite, and the weighted norm ||v||_W = sqrt( sum_i=1^o w_i v_i^2 ),
!  using a primal-dual method. The subroutine is particularly appropriate
!  when A_0 and A are sparse.
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
!    to be solved since the last call to CLLS_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %o is an INTEGER variable, which must be set by the user to the
!    number of observations, o.  RESTRICTION: o >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Ao is a structure of type SMT_type used to hold the design matrix A_o.
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 10 ) = TRANSFER( 'COORDINATE', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o
!       %Ao%row( : ) the row indices of the components of A_o
!       %Ao%col( : ) the column indices of the components of A_o
!       %Ao%ne       the number of nonzeros used to store A_o
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored row by row
!       %Ao%col( : ) the column indices of the components of A_o
!       %Ao%ptr( : ) pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored column
!                    by column
!       %Ao%row( : ) the row indices of the components of A_o
!       %Ao%ptr( : ) pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 5 ) = TRANSFER( 'DENSE', %Ao%type )
!         [ or %Ao%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', %Ao%type ) ]
!       %Ao%val( : ) the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       %Ao%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', %Ao%type )
!       %Ao%val( : ) the values of the components of A_o, stored column
!                    by column with each the entries in each column in order
!                    of increasing row indicies.
!
!   %B is a REAL array of length o, which must be set by the user to the value
!    of the observations, b. The i-th component of B, i = 1, ...., o should
!    contain the value of b_i.
!
!   %A is a structure of type SMT_type used to hold the constraint matrix A.
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', %A%type )
!       %A%val( : ) the values of the components of A
!       %A%row( : ) the row indices of the components of A
!       %A%col( : ) the column indices of the components of A
!       %A%ne       the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', %A%type )
!       %A%val( : ) the values of the components of A, stored row by row
!       %A%col( : ) the column indices of the components of A
!       %A%ptr( : ) pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', %A%type )
!       %A%val( : ) the values of the components of A, stored column
!                    by column
!       %A%row( : ) the row indices of the components of A
!       %A%ptr( : ) pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 5 ) = TRANSFER( 'DENSE', %A%type )
!         [ or %A%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', %A%type ) ]
!       %A%val( : ) the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       %A%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', %A%type )
!       %A%val( : ) the values of the components of A, stored column
!                    by column with each the entries in each column in order
!                    of increasing row indicies.
!
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %R is a REAL array of length %o, which is used to store the values of
!    the residuals A_o x - b. It need not be set on entry. On exit, it will
!    have been filled with appropriate values.
!
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
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
!   %C_status is an INTEGER array of length %m, which will be set on exit to
!    indicate the likely ultimate status of the constraints. Possible values are
!    C_status( i ) < 0, the i-th constraint is likely in the active set,
!                       on its lower bound,
!                  > 0, the i-th constraint is likely in the active set
!                       on its upper bound, and
!                  = 0, the i-th constraint is likely not in the active set
!    It need not be set on entry.
!
!   %X_status is an INTEGER array of length %n, which will be set on exit to
!    indicate the likely ultimate status of the simple bound constraints.
!    Possible values are
!    X_status( i ) < 0, the i-th bound constraint is likely in the active set,
!                       on its lower bound,
!                  > 0, the i-th bound constraint is likely in the active set
!                       on its upper bound, and
!                  = 0, the i-th bound constraint is likely not in the active
!                       set
!    It need not be set on entry.
!
!  data is a structure of type CLLS_data_type which holds private internal data
!
!  control is a structure of type CLLS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to CLLS_initialize. See the preamble
!   for details
!
!  inform is a structure of type CLLS_inform_type that provides
!    information on exit from CLLS_solve. The component status
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
!        prob%m     >=  1
!        prob%m     >=  0
!        prob%Ao%type, prob%A%type, in { 'DENSE', 'DENSE_BY_COLUMNS',
!              'SPARSE_BY_ROWS', SPARSE_BY_COLUMNS','COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
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
!  On exit from CLLS_solve, other components of inform are given in the preamble
!
!  regularization_weight is an OPTIONAL REAL, that may be set by the user
!   to the value of the non-negative regularization weight. If it is absent,
!   the regularization weight will be zero.
!
!  W is an OPTIONAL REAL array of length prob%o, that may be set by the user
!   to the values of the components of the weights W. If it is absent,
!   the weights will all be taken to be 1.0.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( CLLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( CLLS_control_type ), INTENT( IN ) :: control
      TYPE ( CLLS_inform_type ), INTENT( OUT ) :: inform

!  optional dummy argument

      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: regularization_weight
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( prob%o ) :: W

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, n_depen, nzc
      REAL :: time_start, time_record, time_now
      REAL ( KIND = rp_ ) :: time_analyse, time_factorize
      REAL ( KIND = rp_ ) :: clock_start, clock_record, clock_now
      REAL ( KIND = rp_ ) :: clock_analyse, clock_factorize, cro_clock_matrix
      REAL ( KIND = rp_ ) :: av_bnd, weight
!     REAL ( KIND = rp_ ) :: fixed_sum, xi
      LOGICAL :: printi, remap_freed, reset_bnd
!     LOGICAL :: printa
      CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$    INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering CLLS_solve ' )" ) prefix

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

!  QPLIB file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  initialize counts

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%factorization_status = 0
      inform%iter = - 1 ; inform%nfacts = - 1 ; inform%nbacts = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%obj = - one
      inform%non_negligible_pivot = zero
      inform%feasible = .FALSE.
!$    inform%threads = OMP_GET_MAX_THREADS( )
      cro_clock_matrix = 0.0_rp_

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1
!     printa = control%out > 0 .AND. control%print_level >= 101

      IF ( PRESENT( regularization_weight ) ) THEN
        weight = regularization_weight
      ELSE
        weight = zero
      END IF

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%o < 0 .OR. prob%m < 0 .OR. weight < zero .OR.  &
           .NOT. QPT_keyword_A( prob%Ao%type ) .OR.                            &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status
        GO TO 800
      END IF

      IF ( PRESENT( W ) ) THEN
        IF ( MINVAL( W ) <= zero ) THEN
          inform%status = GALAHAD_error_restrictions
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        END IF
      END IF

!  if required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 )                   &
        CALL CLLS_summarize_problem( control%out, prob )

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

      IF ( prob%new_problem_structure ) THEN

!  store the problem dimensions

        SELECT CASE ( SMT_get( prob%Ao%type ) )
        CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
          data%ao_ne = prob%o * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%ao_ne = prob%AO%ptr( prob%m + 1 ) - 1
        CASE ( 'SPARSE_BY_COLUMNS' )
          data%ao_ne = prob%AO%ptr( prob%n + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%ao_ne = prob%Ao%ne
        END SELECT

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = prob%A%ne
        END SELECT

!  make sure that X_status, C_status, C and R have been allocated and are
!  large enough

        array_name = 'clls: prob%X_status'
        CALL SPACE_resize_array( prob%n, prob%X_status,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: prob%C_status'
        CALL SPACE_resize_array( prob%m, prob%C_status,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: prob%C'
        CALL SPACE_resize_array( prob%m, prob%C,                               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: prob%R'
        CALL SPACE_resize_array( prob%o, prob%R,                               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  perform the preprocessing

      IF ( prob%new_problem_structure ) THEN
        IF ( printi ) THEN
          IF ( prob%m > 0 ) THEN
            WRITE( control%out,                                                &
               "(  A, ' problem dimensions before preprocessing: ', /,  A,     &
           &   ' n = ', I0, ' m = ', I0, ' ao_ne = ', I0, ' a_ne = ', I0 )" )  &
               prefix, prefix, prob%n, prob%m, data%ao_ne, data%a_ne
          ELSE
            WRITE( control%out,                                                &
               "(  A, ' problem dimensions before preprocessing:',             &
           &   ' n = ', I0, ' ao_ne = ', I0 )" ) prefix, prob%n, data%ao_ne
          END IF
        END IF

        CALL LSP_initialize( data%LSP_map, data%LSP_control )
        data%LSP_control%infinity = control%infinity
        data%LSP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL LSP_reorder( data%LSP_map, data%LSP_control,                      &
                          data%LSP_inform, data%LSP_dims, prob,                &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

        IF ( data%LSP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%LSP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after LSP_reorder')" ) &
             prefix, data%LSP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL LSP_terminate( data%LSP_map, data%LSP_control, data%LSP_inform )
          GO TO 800
        END IF

!  record array lengths

        SELECT CASE ( SMT_get( prob%Ao%type ) )
        CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
          data%ao_ne = prob%o * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%ao_ne = prob%AO%ptr( prob%m + 1 ) - 1
        CASE ( 'SPARSE_BY_COLUMNS' )
          data%ao_ne = prob%AO%ptr( prob%n + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%ao_ne = prob%Ao%ne
        END SELECT

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = prob%A%ne
        END SELECT

        IF ( printi ) THEN
          IF ( prob%m > 0 ) THEN
            WRITE( control%out,                                                &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
           &   ' n = ', I0, ' m = ', I0, ' ao_ne = ', I0, ' a_ne = ', I0 )" )  &
               prefix, prefix, prob%n, prob%m, data%ao_ne, data%a_ne
          ELSE
            WRITE( control%out,                                                &
               "(  A, ' problem dimensions after  preprocessing:',             &
           &   ' n = ', I0, ' ao_ne = ', I0 )" ) prefix, prob%n, data%ao_ne
          END IF
        END IF

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL LSP_apply( data%LSP_map, data%LSP_inform,                       &
                          prob, get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

          IF ( data%LSP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%LSP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( A, ' status ', I0, ' after LSP_apply')" ) &
               prefix, data%LSP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) prefix, inform%status
            CALL LSP_terminate( data%LSP_map, data%LSP_control, data%LSP_inform)
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
              prefix, data%LSP_dims%c_equality,                                &
              TRIM( STRING_ies( data%LSP_dims%c_equality ) ),                  &
              prob%m, TRIM( STRING_pleural( prob%m ) )

!  set control parameters

        data%FDC_control = control%FDC_control
        data%FDC_control%max_infeas = control%stop_abs_p

!  find any dependent rows

        nzc = prob%A%ptr( data%LSP_dims%c_equality + 1 ) - 1
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL FDC_find_dependent( prob%n, data%LSP_dims%c_equality,             &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%LSP_dims%c_equality + 1 ), &
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
             REAL( time_now - time_start, rp_ ) > control%cpu_time_limit ) .OR.&
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
              prefix, n_depen, TRIM(STRING_pleural( n_depen ) ),               &
              TRIM( STRING_verb_pleural( n_depen ) ), data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y( : prob%m ) = zero
          prob%Z( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL CLLS_AX( prob%m, prob%C( : prob%m ), prob%m,                    &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
                       prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ', A,   &
         & ' dependent and will be temporarily removed' )" ) prefix, n_depen,  &
           TRIM( STRING_pleural( n_depen ) ), TRIM( STRING_are( n_depen ) )

!  allocate arrays to indicate which constraints have been freed

          array_name = 'clls: data%C_freed'
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

        CALL LSP_initialize( data%LSP_map_freed, data%LSP_control )
        data%LSP_control%infinity = control%infinity
        data%LSP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  store the problem dimensions

        data%LSP_dims_save_freed = data%LSP_dims
        data%a_ne = prob%A%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before removal of dependecies: ', &
              &   /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL LSP_reorder( data%LSP_map_freed, data%LSP_control,                &
                          data%LSP_inform, data%LSP_dims, prob,                &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

        data%LSP_dims%nc = data%LSP_dims%c_u_end - data%LSP_dims%c_l_start + 1
        data%LSP_dims%x_s = 1 ; data%LSP_dims%x_e = prob%n
        data%LSP_dims%c_s = data%LSP_dims%x_e + 1
        data%LSP_dims%c_e = data%LSP_dims%x_e + data%LSP_dims%nc
        data%LSP_dims%c_b = data%LSP_dims%c_e - prob%m
        data%LSP_dims%y_s = data%LSP_dims%c_e + 1
        data%LSP_dims%y_e = data%LSP_dims%c_e + prob%m
        data%LSP_dims%y_i = data%LSP_dims%c_s + prob%m
        data%LSP_dims%r_s = data%LSP_dims%y_e + 1
        data%LSP_dims%r_e = data%LSP_dims%y_e + prob%o
        data%LSP_dims%v_e = data%LSP_dims%r_e

!  test for satisfactory termination

        IF ( data%LSP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%LSP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after LSP_reorder')" ) &
             prefix, data%LSP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL LSP_terminate( data%LSP_map_freed, data%LSP_control,            &
                              data%LSP_inform )
          CALL LSP_terminate( data%LSP_map, data%LSP_control, data%LSP_inform )
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

      data%LSP_dims%nc = data%LSP_dims%c_u_end - data%LSP_dims%c_l_start + 1

!  arrays containing data relating to the composite vector ( x  c  y  r )
!  are partitioned as follows:

!  <---------- n --------->  <---- nc ------>  <-------- m ---------> <-- o ->
!                            <-------- m --------->
!                       <-------- m --------->
!  ----------------------------------------------------------------------------
!  |                   |    |                 |    |                 |        |
!  |         x              |       c         |          y           |    r   |
!  |                   |    |                 |    |                 |        |
!  ----------------------------------------------------------------------------
!   ^                 ^    ^ ^               ^ ^    ^               ^ ^       ^
!   |                 |    | |               | |    |               | |       |
!  x_s                |    |c_s              |y_s  y_i              |r_s     r_e
!                     |    |                 |                      |         =
!                    c_b  x_e               c_e                    y_e       v_e

      data%LSP_dims%x_s = 1 ; data%LSP_dims%x_e = prob%n
      data%LSP_dims%c_s = data%LSP_dims%x_e + 1
      data%LSP_dims%c_e = data%LSP_dims%x_e + data%LSP_dims%nc
      data%LSP_dims%c_b = data%LSP_dims%c_e - prob%m
      data%LSP_dims%y_s = data%LSP_dims%c_e + 1
      data%LSP_dims%y_e = data%LSP_dims%c_e + prob%m
      data%LSP_dims%y_i = data%LSP_dims%c_s + prob%m
      data%LSP_dims%r_s = data%LSP_dims%y_e + 1
      data%LSP_dims%r_e = data%LSP_dims%y_e + prob%o
      data%LSP_dims%v_e = data%LSP_dims%r_e

!  ----------------
!  set up workspace
!  ----------------

      data%ao_ne = prob%Ao%ptr( prob%n + 1 ) - 1
      data%a_ne = prob%A%ptr( prob%m + 1 ) - 1

      CALL CLLS_workspace( prob%n, prob%o, prob%m, data%LSP_dims, data%ao_ne,  &
                           data%a_ne, data%order, data%GRAD_L, data%DIST_X_l,  &
                           data%DIST_X_u, data%Z_l, data%Z_u,                  &
                           data%BARRIER_X, data%Y_l, data%DIST_C_l,            &
                           data%Y_u, data%DIST_C_u, data%C, data%BARRIER_C,    &
                           data%SCALE_C, data%RHS, data%OPT_alpha,             &
                           data%OPT_merit, data%BINOMIAL, data%CS_coef,        &
                           data%COEF, data%ROOTS, data%DX_zh, data%DY_zh,      &
                           data%DC_zh, data%DY_l_zh, data%DY_u_zh,             &
                           data%DZ_l_zh, data%DZ_u_zh, data%X_coef,            &
                           data%C_coef, data%Y_coef, data%Y_l_coef,            &
                           data%Y_u_coef, data%Z_l_coef, data%Z_u_coef,        &
                           data%R_last, data%c_last, data%X_last,              &
                           data%Y_last, data%Z_last,                           &
                           data%K_sls, control%error, control%series_order,    &
                           control%deallocate_error_fatal,                     &
                           control%space_critical, inform%status,              &
                           inform%alloc_status, inform%bad_alloc )

      array_name = 'clls: data%X_free'
      CALL SPACE_resize_array( prob%n, data%X_free,                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  =================
!  Solve the problem
!  =================

      CALL CLLS_solve_main( data%LSP_dims, prob%n, prob%o, prob%m, weight,     &
                            prob%Ao%val, prob%Ao%row, prob%Ao%ptr, prob%B,     &
                            prob%A%val, prob%A%col, prob%A%ptr,                &
                            prob%C_l, prob%C_u, prob%X_l, prob%X_u,            &
                            prob%R, prob%C, prob%X, prob%Y, prob%Z,            &
                            prob%C_status, prob%X_status,                      &
                            data%GRAD_L, data%DIST_X_l, data%DIST_X_u,         &
                            data%Z_l, data%Z_u, data%BARRIER_X,                &
                            data%Y_l, data%DIST_C_l, data%Y_u,                 &
                            data%DIST_C_u, data%C, data%BARRIER_C,             &
                            data%SCALE_C, data%RHS, data%R_last, data%C_last,  &
                            data%X_last, data%Y_last, data%Z_last,             &
                            data%K_sls, data%X_free,                           &
                            data%order, data%X_coef, data%C_coef,              &
                            data%Y_coef, data%Y_l_coef, data%Y_u_coef,         &
                            data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,       &
                            data%CS_coef, data%COEF,                           &
                            data%ROOTS, data%ROOTS_data,                       &
                            data%DX_zh, data%DC_zh, data%DY_zh, data%DY_l_zh,  &
                            data%DY_u_zh, data%DZ_l_zh, data%DZ_u_zh,          &
                            data%OPT_alpha, data%OPT_merit,                    &
                            data%SLS_data, data%SLS_pounce_data,               &
                            prefix, control, inform, data%K_sls_pounce, W )

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
        prob%C_status( prob%m + 1 : data%LSP_map_freed%m ) = 0
        CALL SORT_inverse_permute( data%LSP_map_freed%m,                       &
                                   data%LSP_map_freed%c_map,                   &
                                   IX = prob%C_status( : data%LSP_map_freed%m ))
        prob%X_status( prob%n + 1 : data%LSP_map_freed%n ) = - 1
        CALL SORT_inverse_permute( data%LSP_map_freed%n,                       &
                                   data%LSP_map_freed%x_map,                   &
                                   IX = prob%X_status( : data%LSP_map_freed%n ))
        CALL LSP_restore( data%LSP_map_freed, data%LSP_inform, prob,           &
                          get_all = .TRUE.)
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%LSP_dims = data%LSP_dims_save_freed

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
        prob%C_status( prob%m + 1 : data%LSP_map%m ) = 0
        CALL SORT_inverse_permute( data%LSP_map%m, data%LSP_map%c_map,         &
                                   IX = prob%C_status( : data%LSP_map%m ) )
        prob%X_status( prob%n + 1 : data%LSP_map%n ) = - 1
        CALL SORT_inverse_permute( data%LSP_map%n, data%LSP_map%x_map,         &
                                   IX = prob%X_status( : data%LSP_map%n ) )

!  full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL LSP_restore( data%LSP_map, data%LSP_inform, prob,               &
                            get_all = .TRUE. )

!  restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL LSP_restore( data%LSP_map, data%LSP_inform, prob,               &
                            get_b = .TRUE.,                                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  recover solution

        ELSE
          CALL LSP_restore( data%LSP_map, data%LSP_inform, prob,               &
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
      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, rp_ )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=',                                            &
    &   /, A, 3X, ' =                            total time              ',    &
    &             '             =',                                            &
    &   /, A, 3X, ' =', 24X, 0P, F12.2, 29x, '='                               &
    &   /, A, 3X, ' =    preprocess    analyse    factorize     solve    ',    &
    &             ' crossover   =',                                            &
    &   /, A, 3X, ' =', 5F12.2, 5x, '=',                                       &
    &   /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=') ")                                         &
        prefix, prefix, prefix, inform%time%clock_total, prefix, prefix,       &
        inform%time%clock_preprocess, inform%time%clock_analyse,               &
        inform%time%clock_factorize, inform%time%clock_solve,                  &
        inform%CRO_inform%time%clock_total - cro_clock_matrix, prefix

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving CLLS_solve ' )" ) prefix
      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, rp_ )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -CLLS_solve-', /,  A,                          &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving CLLS_solve ' )" ) prefix
      RETURN

!  non-executable statements

 2010 FORMAT( ' ', /, A, '    ** Error return ', I0, ' from CLLS ' )

!  End of CLLS_solve

      END SUBROUTINE CLLS_solve

!-*-*-*-*-*-   C L L S _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CLLS_solve_main( dims, n, o, m, weight, Ao_val, Ao_row,       &
                                  Ao_ptr, B, A_val, A_col, A_ptr,              &
                                  C_l, C_u, X_l, X_u, R, C_RES,                &
                                  X, Y, Z, C_stat, X_Stat, GRAD_L,             &
                                  DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X,     &
                                  Y_l, DIST_C_l, Y_u, DIST_C_u, C, BARRIER_C,  &
                                  SCALE_C, RHS, &
                                  R_last, C_last, X_last, Y_last, Z_last,      &
                                  K_sls, X_free,   &
                                  order, X_coef, C_coef, Y_coef, Y_l_coef,     &
                                  Y_u_coef, Z_l_coef, Z_u_coef, BINOMIAL,      &
                                  CS_coef, COEF, ROOTS, ROOTS_data,            &
                                  DX_zh, DC_zh, DY_zh, DY_l_zh,                &
                                  DY_u_zh, DZ_l_zh, DZ_u_zh,                   &
                                  OPT_alpha, OPT_merit, SLS_data,              &
                                  SLS_pounce_data, prefix, control, inform,    &
                                  K_sls_pounce, W )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic objective function
!
!         1/2 || A_o x - b ||^2 + 1/2 weight || x ||^2
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
!  GALAHAD_LSP. The same module may subsequently used to recover the solution.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type CLLS_data_type, whose components hold SCALAR
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
!    value dims%y_e + o
!
!  n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: n >= 1
!
!  o is an INTEGER variable, which must be set by the user to the
!    number of observations, o.  RESTRICTION: o >= 1
!
!  m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m.  RESTRICTION: m >= 0
!
!  weight is a REAL variable, that must be set to the regularization weight.
!    RESTRICTION: weight >= 0

!  Ao_row//ptr/val is used to hold the matrix A_o by columns. In particular:
!      Ao_row( : )   the row indices of the components of A_o
!      Ao_ptr( : )   pointers to the start of each column, and past the end of
!                    the last column.
!      Ao_val( : )   the values of the components of A_o
!
!  B is a REAL array of length o, which must be set by the user to the values
!   of the observations b.
!
!  A_col/ptr/val is used to hold the matrix L by rows. In particular:
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
!  RES is a REAL array of length o, which need not be set on entry. On exit,
!   the i-th component of RES will contain (Ao x - b)_i, for i = 1, .... , o.
!
!  C_RES is a REAL array of length m, which need not be set on entry. On exit,
!   the i-th component of C_RES will contain (A x)_i, for i = 1, .... , m.
!
!  X is a REAL array of length n, which must be set by
!   the user on entry to CLLS_solve to give an initial estimate of the
!   optimization parameters, x. The i-th component of X should contain
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need
!   not satisfy the simple bound constraints and may be perturbed by
!   CLLS_solve prior to the start of the minimization.  Any estimate which is
!   closer to one of its bounds than control%prfeas may be reset to try to
!   ensure that it is at least control%prfeas from its bounds. On exit from
!   CLLS_solve, X will contain the best estimate of the optimization
!   parameters found
!
!  Y is a REAL array of length m, which must be set by the user
!   on entry to CLLS_solve to give an initial estimates of the
!   optimal Lagrange multipiers, y. The i-th component of Y
!   should contain the initial estimate of y_i, for i = 1, .... , m.
!   Any estimate which is smaller than control%dufeas may be
!   reset to control%dufeas. The dual variable for any variable with both
!   On exit from CLLS_solve, Y will contain the best estimate of
!   the Lagrange multipliers found
!
!  Z, is a REAL array of length n, which must be set by
!   on entry to CLLS_solve to hold the values of the the dual variables
!   associated with the simple bound constraints.
!   Any estimate which is smaller than control%dufeas may be
!   reset to control%dufeas. The dual variable for any variable with both
!   infinite lower and upper bounds need not be set. On exit from
!   CLLS_solve, Z will contain the best estimates obtained
!
!  control and inform are exactly as for CLLS_solve
!
!  W is an OPTIONAL REAL array of length o, that, if PRESENT, must be set
!   by the user to the values of the vector of weights W. If W is absent,
!   weights of 1.0 will be used.
!
!  The remaining arguments are used as internal workspace, and need not be
!  set on entry
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( CLLS_dims_type ), INTENT( IN ) :: dims
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m, order
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ),                                    &
                              DIMENSION( Ao_ptr( n + 1 ) - 1 ) :: Ao_row
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
                           DIMENSION( Ao_ptr( n + 1 ) - 1 ) :: Ao_val
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: B
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ),                                    &
                              DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
                           DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( m ) :: C_stat
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: X_stat
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: X_free
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, X_last
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R, R_last
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( m ) :: Y, Y_last
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: Z, Z_last
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( m ) :: C_RES, C_last
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( dims%v_e ) :: RHS
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( dims%c_e ) :: GRAD_L
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
                          DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
                          DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) ::  Z_l
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
                          DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DIST_C_l
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DIST_C_u
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, BARRIER_C, SCALE_C
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( n, 0 : order ) :: X_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_l_start : dims%c_u_end, 0 : order ) :: C_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( m, 0 : order ) :: Y_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_l_start : dims%c_l_end, 0 : order ) ::  Y_l_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_u_start : dims%c_u_end, 0 : order ) ::  Y_u_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION(   dims%x_free + 1 : dims%x_l_end, 0 : order ) :: Z_l_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%x_u_start : n, 0 : order ) :: Z_u_coef
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( 0 : order - 1 , order ) :: BINOMIAL
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 0 : 2 * order ) :: CS_coef
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 0 : 2 * order ) :: COEF
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 2 * order ) :: ROOTS
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( n ) :: DX_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_l_start : dims%c_u_end ) :: DC_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( m ) :: DY_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_l_start : dims%c_l_end ) ::  DY_l_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%c_u_start : dims%c_u_end ) ::  DY_u_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION(   dims%x_free + 1 : dims%x_l_end ) :: DZ_l_zh
      REAL ( KIND = rp_ ), INTENT( OUT ),                                      &
        DIMENSION( dims%x_u_start : n ) :: DZ_u_zh
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( order ) :: OPT_alpha
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( order ) :: OPT_merit
      TYPE ( SMT_type ), INTENT( INOUT ) :: K_sls
      TYPE ( SMT_type ), INTENT( INOUT ) :: K_sls_pounce
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      TYPE ( CLLS_control_type ), INTENT( IN ) :: control
      TYPE ( CLLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_pounce_data
      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: ROOTS_data

!  optional dummy argument

      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: W

!  Parameters

      REAL ( KIND = rp_ ), PARAMETER :: eta = tenm4
      REAL ( KIND = rp_ ), PARAMETER :: sigma_max = point01
      REAL ( KIND = rp_ ), PARAMETER :: degen_tol = tenm5

!  Local variables

      INTEGER ( KIND = ip_ ) :: Ao_ne, A_ne, i, ii, j, k, l, b_change, c_change
      INTEGER ( KIND = ip_ ) :: start_print, stop_print, print_level
      INTEGER ( KIND = ip_ ) :: nbnds, nbnds_x, nbnds_c, muzero_fixed, nbact
      INTEGER ( KIND = ip_ ) :: out, error, it_best, infeas_max, iorder, sorder
      INTEGER ( KIND = ip_ ) :: primal_nonopt, dual_nonopt, cs_nonopt
      INTEGER ( KIND = ip_ ) :: npnc, npncpm, d_start, e_start, s_start
      INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: iorder_array
      REAL :: time, time_record, time_start, time_now, time_solve
      REAL ( KIND = rp_ ) :: time_analyse, time_factorize
      REAL ( KIND = rp_ ) :: clock_record, clock_start, clock_now, clock_solve
      REAL ( KIND = rp_ ) :: clock_analyse, clock_factorize
      REAL ( KIND = rp_ ) :: pjgnrm, mu, aomax, amax, gamma_f, bik, slope
      REAL ( KIND = rp_ ) :: cs, slknes, slkmin, reduce_infeas, tau, comp
      REAL ( KIND = rp_ ) :: slknes_x, slknes_c, slkmax_x, slkmax_c, res_cs
      REAL ( KIND = rp_ ) :: slkmin_x, slkmin_c, res_primal, res_primal_dual
      REAL ( KIND = rp_ ) :: merit, merit_trial, merit_best, merit_model
      REAL ( KIND = rp_ ) :: prfeas, dufeas, p_min, p_max, d_min, d_max
      REAL ( KIND = rp_ ) :: pivot_tol, min_pivot_tol
      REAL ( KIND = rp_ ) :: alpha, alpha_l, alpha_u, alpha_max, one_minus_alpha
      REAL ( KIND = rp_ ) :: sigma, gamma_c, gi, co, sigma_mu, sigma_mu2
      REAL ( KIND = rp_ ) :: one_plus_sigma_mu, two_plus_sigma_mu, balance
      REAL ( KIND = rp_ ) :: one_plus_2_sigma_mu, two_sigma_mu2, two_sigma_mu
      REAL ( KIND = rp_ ) :: opt_alpha_guarantee, opt_merit_guarantee
      REAL ( KIND = rp_ ) :: stop_p, stop_d, stop_c, two_mu
      REAL ( KIND = rp_ ) :: rnbnds, rnbnds_x, rnbnds_c
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: printt, printi, printe, printd, printw, set_printp, printp
      LOGICAL :: maxpiv, guarantee, optimal, present_weight
!     LOGICAL :: root_arc
      LOGICAL :: puiseux, get_stat, stat_known
      LOGICAL :: use_scale_c = .FALSE.
      CHARACTER ( LEN = 1 ) :: re, pui
      CHARACTER ( LEN = 2 ) :: arc
      CHARACTER ( len = 10 ) :: char_x, char_c, char_y
      CHARACTER ( len = 10 ) :: char_z_l, char_z_u, char_y_l, char_y_u
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: DX, WORK_n
      INTEGER ( KIND = ip_ ), DIMENSION( m ) :: C_stat_old
      INTEGER ( KIND = ip_ ), DIMENSION( n ) :: X_stat_old

      TYPE ( SLS_control_type ) :: SLS_control

      INTEGER ( KIND = ip_ ) :: sif = 50
!     LOGICAL :: generate_sif = .TRUE.
      LOGICAL :: generate_sif = .FALSE.

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering CLLS_solve_main ' )" ) prefix

!  move to argument list

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
!     IF ( control%out > 0 .AND. control%print_level >= 1 ) THEN
        WRITE( control%out, "( /, A, ' n = ', I0, ', o = ', I0,                &
       &                      ', m = ', I0 )" ) prefix, n, o, m
        WRITE( control%out, "( A, ' A0 (column-wise) =' )" ) prefix
        DO j = 1, n
         IF ( Ao_ptr( j ) <= Ao_ptr( j + 1 ) - 1 )                             &
           WRITE( control%out, "( ( 2( 2I8, ES24.16 ) ) )" )                   &
           ( Ao_row( i ), j, Ao_val( i ), i = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1 )
        END DO
        WRITE( control%out, "( A, ' B =', /, ( 5X, 3ES24.16 ) )" )             &
          prefix, B( : o )
        WRITE( control%out, "( A,                                              &
       &  ' X_l =', /, ( 5X, 3ES24.16 ) )" ) prefix, X_l( : n )
        WRITE( control%out, "( A,                                              &
       &  ' X_u =', /, ( 5X, 3ES24.16 ) )" ) prefix, X_u( : n )
        IF ( m > 0 ) THEN
          WRITE( control%out, "( A, ' A (row-wise) =' )" ) prefix
          DO i = 1, m
           IF ( A_ptr( i ) <= A_ptr( i + 1 ) - 1 )                             &
              WRITE( control%out, "( ( 2( 2I8, ES24.16 ) ) )" )                &
              ( i, A_col( j ), A_val( j ), j = A_ptr( i ), A_ptr( i + 1 ) - 1 )
          END DO
          WRITE( control%out, "( A,                                            &
         &  ' C_l =', /, ( 5X, 3ES24.16 ) )" ) prefix, C_l( : m )
          WRITE( control%out, "( A,                                            &
         &  ' C_u =', /, ( 5X, 3ES24.16 ) )" ) prefix, C_u( : m )
        END IF
      END IF

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( generate_sif ) THEN
        WRITE( sif, "( 'NAME          CLLS_OUT', //, 'VARIABLES', / )" )
        DO i = 1, n
          WRITE( sif, "( '    X', I8 )" ) i
        END DO

        WRITE( sif, "( /, 'GROUPS', / )" )
        DO j = 1, n
          DO l = Ao_ptr( i ), Ao_ptr( i + 1 ) - 1
            WRITE( sif, "( ' N  R', I8, ' X', I8, ' ', ES12.5 )" )             &
              Ao_row( l ), j, Ao_val( l )
          END DO
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
        DO i = 1, o
          IF ( B( i ) /= zero )                                                &
          WRITE( sif, "( '    RHS      ', ' R', I8, ' ', ES12.5 )" ) i, B( i )
        END DO
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

        WRITE( sif, "( /, 'GROUP TYPE', //,                                    &
       &                  ' GV LS        GVAR', //,                            &
       &                  'GROUP USES', // )" )
        DO i = 1, o
          DO l = Ao_ptr( i ), Ao_ptr( i + 1 ) - 1
            WRITE( sif, "( ' XT  R', I8, ' L2' )" ) i
          END DO
        END DO

        WRITE( sif, "( /, 'ENDATA', //,                                        &
       &                  'GROUPS        CLLS_OUT', //,                        &
       &                  'INDIVIDUALS', //,                                   &
       &                  ' T  LS', /,                                         &
       &                  ' F                      0.5 * GVAR * GVAR', /,      &
       &                  ' G                      GVAR', /,                   &
       &                  ' H                      1.0', /,                    &
       &                  'ENDATA' )" )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

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

      SLS_control = control%SLS_control

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
        END IF
        C_RES = zero ; Y = zero
        inform%obj = zero
        GO TO 800
      END IF

!  set control parameters

      present_weight = PRESENT( W )
      muzero_fixed = control%muzero_fixed
      prfeas = MAX( control%prfeas, epsmch )
      dufeas = MAX( control%dufeas, epsmch )
      reduce_infeas = MAX( epsmch,                                             &
                           MIN( control%reduce_infeas ** 2, one - epsmch ) )
      infeas_max = MAX( 0, control%infeas_max )
      get_stat = .FALSE.
      stat_known = .FALSE.
      iorder = 0
      optimal = .FALSE.

!  find the largest components of Ao and A

      Ao_ne = Ao_ptr( n + 1 ) - 1
      IF ( Ao_ne > 0 ) THEN
        aomax = MAXVAL( ABS( Ao_val( : Ao_ne ) ) )
      ELSE
        aomax = zero
      END IF

      A_ne = A_ptr( m + 1 ) - 1
      IF ( A_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : A_ne ) ) )
      ELSE
        amax = zero
      END IF

      IF ( printi ) WRITE( out,                                                &
        "( /, A, '  maximum element of Ao =', ES11.4 )" )  prefix, aomax
      IF ( printi .AND. m > 0 ) WRITE( out,                                    &
         "( A, '  maximum element of A  =', ES11.4 )" ) prefix, amax

!  set up structure for the matrix K (whose lower triangle is)

!       ( weight I + D     A    Ao^T  )   ( n  dimensional )
!       (               E -S          )   ( nc dimensional )
!       (     A        -S  0          )   ( m  dimensional )
!       (     Ao              -W^{-1} )   ( o  dimensional )

!  where D = (X-X_l)^-1 Z_l - (X_u-X)^-1 Z_u
!        E = (C-C_l)^-1 Y_l - (C_u-C)^-1 Y_u
!  and S is a diagonal constraint scaling matrix (by default I)

!  K will be stored in coordinate form

      npnc = n + dims%nc
      npncpm = npnc + m

!  input the 1,1 block, weight I + D (structure only)

      d_start = 0
      DO l = 1, n
        K_sls%row( l ) = l ; K_sls%col( l ) = l
      END DO

!  input the 2,2 block, E (structure only)

      e_start = n
      l = e_start
      DO j = n + 1, npnc
        l = l + 1
        K_sls%row( l ) = j ; K_sls%col( l ) = j
      END DO

!  input the 3,1 block, A

     DO i = 1, m
       ii = npnc + i
       DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
          l = l + 1
          K_sls%row( l ) = ii ; K_sls%col( l ) = A_col( k )
          K_sls%val( l ) = A_val( k )
       END DO
     END DO

!  input the 3,2 block, -S (structure only)

      s_start = l
      DO i = n + 1, npnc
        l = l + 1
        K_sls%row( l ) = m + i ; K_sls%col( l ) = i
      END DO

!  input the 4,1 block, Ao

      DO j = 1, n
        DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
          l = l + 1
          K_sls%row( l ) = npncpm + Ao_row( k ) ; K_sls%col( l ) = j
          K_sls%val( l ) = Ao_val( k )
        END DO
      END DO

!  input the 4,4 block, -W^{-1}

      IF ( present_weight ) THEN
        DO i = npncpm + 1, npncpm + o
          l = l + 1
          K_sls%row( l ) = i ; K_sls%col( l ) = i
          K_sls%val( l ) = - one / W( i - npncpm )
        END DO
      ELSE
        DO i = npncpm + 1, npncpm + o
          l = l + 1
          K_sls%row( l ) = i ; K_sls%col( l ) = i
          K_sls%val( l ) = - one
        END DO
      END IF

!  record the dimensions of K

      K_sls%n = npncpm + o ; K_sls%ne = l

!  record the required linear solver, and ammend if necessary

      CALL SLS_initialize_solver( control%symmetric_linear_solver,             &
                                  SLS_data, SLS_control%error,                 &
                                  inform%SLS_inform, check = .TRUE. )
      IF ( inform%SLS_inform%status < 0 ) THEN
        inform%status = inform%SLS_inform%status ; GO TO 600 ; END IF
      IF ( inform%SLS_inform%status == GALAHAD_error_unknown_solver ) THEN
        inform%status = GALAHAD_error_unknown_solver ; GO TO 600
      ELSE
        CALL SLS_initialize_solver( inform%SLS_inform%solver, SLS_pounce_data, &
                                    SLS_control%error,                         &
                                    inform%SLS_pounce_inform )
      END IF

!  analyse the sparsity pattern of K to build a tentaive ordering
!  for the sparse factorization

      time_analyse = inform%SLS_inform%time%analyse
      clock_analyse = inform%SLS_inform%time%clock_analyse
      CALL SLS_analyse( K_sls, SLS_data, SLS_control, inform%SLS_inform )

      inform%time%analyse = inform%time%analyse +                              &
        inform%SLS_inform%time%analyse - time_analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%SLS_inform%time%clock_analyse - clock_analyse

!  initialize status indicators

      X_stat = 0
      C_stat( : dims%c_equality ) = - 1
      C_stat( dims%c_equality + 1 : ) = 0

!  if required, write out the problem

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

!  record the initial point, move the starting point away from any bounds,
!  and move that for dual variables away from zero

      nbnds_x = 0

!  the variable is free

      IF ( printd ) THEN
        WRITE( out, "( /, A, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X,      &
       &       'z_l', 9X, 'z_u')") prefix
        DO i = 1, dims%x_free
          WRITE( out, "( A, I6, ES12.4, 4( '      -     '))" ) prefix, i, X( i )
        END DO
      END IF

!  the variable is a non-negativity

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

!  the variable has just a lower bound

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

!  the variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end

!  check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN
          inform%status = GALAHAD_error_bad_bounds
          GO TO 600
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

!  the variable has just an upper bound

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

!  the variable is a non-positivity

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

!  compute the value of the constraint, and their residuals

      nbnds_c = 0
      IF ( m > 0 ) THEN
        C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
        C_RES( dims%c_l_start : dims%c_u_end ) = zero
        CALL CLLS_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
        IF ( printd ) THEN
          WRITE( out, "( /, A, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X,     &
         &     'y_l', 9X, 'y_u' )") prefix
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( A, I6, 3ES12.4 )" )                                 &
              prefix, i, C_RES( i ), C_l( i ), C_u( i )
          END DO
        END IF

!  the constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          nbnds_c = nbnds_c + 1

!  compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )

!  compute an appropriate initial value for the slack variable

          C( i ) = MAX( C_RES( i ) / SCALE_C( i ), C_l( i ) + prfeas )
          DIST_C_l( i ) = C( i ) - C_l( i )
          C_RES( i ) = C_RES( i ) - SCALE_C( i ) * C( i )
          IF ( control%balance_initial_complentarity ) THEN
            Y_l( i ) = balance / DIST_C_l( i )
          ELSE
            Y_l( i ) = MAX( ABS( SCALE_C( i ) * Y( i ) ),  dufeas )
          END IF
          IF ( printd ) WRITE( out,  "( A, I6, 2ES12.4, '      -     ',        &
         &  ES12.4, '      -    ' )" ) prefix, i, C_RES( i ), C_l( i ), Y_l( i )
        END DO

!  the constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end

!  check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

          IF ( C_u( i ) - C_l( i ) <= epsmch ) THEN
            inform%status = GALAHAD_error_bad_bounds
            GO TO 600
          END IF
          nbnds_c = nbnds_c + 2

!  compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )
          C_u( i ) = C_u( i ) / SCALE_C( i )

!  compute an appropriate initial value for the slack variable

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

!  the constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          nbnds_c = nbnds_c + 1

!  compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( C_RES( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  scale the bounds

          C_u( i ) = C_u( i ) / SCALE_C( i )

!  compute an appropriate initial value for the slack variable

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
        inform%primal_infeasibility = MAXVAL( ABS( C_RES( : dims%c_u_end ) ) )
      ELSE
        inform%primal_infeasibility = zero
      END IF

!  record the starting vector

      C_last( dims%c_l_start : dims%c_u_end )                                  &
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

!  compute the objective residual

      R = - B
      CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )
!write(6,*) ' c ', K_sls%val( K_sls%ne )
!write(6,*) ' r ', R
      R_last = R
!write(6,*) ' d ', K_sls%val( K_sls%ne )

!  compute the objective function

      IF ( present_weight ) THEN
        inform%obj = half * DOT_PRODUCT( R, W * R )
      ELSE
        inform%obj = half * DOT_PRODUCT( R, R )
      END IF
      IF ( weight > zero )                                                     &
        inform%obj = inform%obj + half * weight * DOT_PRODUCT( X, X )

!  test to see if we are feasible

      inform%feasible = inform%primal_infeasibility <= control%stop_abs_p
      pjgnrm = infinity

      IF ( inform%feasible ) THEN
        IF ( printi ) WRITE( out, 2070 ) prefix
        IF ( control%just_feasible ) THEN
          inform%status = GALAHAD_ok
          GO TO 600
        END IF
      END IF

!  compute the gradient of the Lagrangian function

      CALL CLLS_Lagrangian_gradient( dims, n, o, m, weight,                    &
                                     X, R, Y, Y_l, Y_u, Z_l, Z_u,              &
                                     Ao_ne, Ao_val, Ao_row, Ao_ptr,            &
                                     A_ne, A_val, A_col, A_ptr,                &
                                     DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,   &
                                     GRAD_L( dims%x_s : dims%x_e ),            &
                                     control%getdua, dufeas, W )

!  evaluate the merit function

      tau = MAX( control%tau, zero )
      merit = CLLS_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,          &
                                DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,        &
                                GRAD_L( dims%x_s : dims%x_e ), C_RES,          &
                                tau, res_primal, inform%dual_infeasibility,    &
                                res_primal_dual, res_cs )

!  find the max-norm of the residual

      nbnds = nbnds_x + nbnds_c
      rnbnds_x = REAL( nbnds_x, KIND = rp_ )
      rnbnds_c = REAL( nbnds_c, KIND = rp_ )
      rnbnds = REAL( nbnds, KIND = rp_ )
      IF ( printi .AND. use_scale_c .AND. m > 0 .AND.                          &
           dims%c_l_start <= dims%c_u_end )                                    &
        WRITE( out, "( A, '  largest/smallest scale factor', 2ES11.4 )" )      &
          prefix, MAXVAL( SCALE_C ), MINVAL( SCALE_C )

!  compute the complementary slackness

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

!  record the slackness and the deviation from the central path

      IF ( nbnds_x > 0 ) THEN
        slknes_x = slknes_x / rnbnds_x
      ELSE
        slknes_x = zero
      END IF

      IF ( nbnds_c > 0 ) THEN
        slknes_c = slknes_c / rnbnds_c
      ELSE
        slknes_c = zero
      END IF

      IF ( nbnds > 0 ) THEN
        IF (  res_primal_dual > zero ) THEN
          gamma_f = control%gamma_f * slknes / res_primal_dual
        ELSE
          gamma_f = one
        END IF
        slknes = slknes / rnbnds
        gamma_c = control%gamma_c * slkmin / slknes
      ELSE
        gamma_f = zero ; slknes = zero ; gamma_c = zero
      END IF

      IF ( printw .AND. nbnds > 0 ) THEN
        WRITE( out, 2130 )                         &
          prefix, slknes, prefix, slknes_x, prefix, slknes_c, prefix, slkmin_x,&
          slkmax_x, prefix, slkmin_c, slkmax_c, prefix, p_min, p_max, prefix,  &
          d_min, d_max
        WRITE( out, "( A, 31X, ' min x gap = ', ES12.4, /,                     &
       &               A, 31X, ' min c gap = ', ES12.4 )" )                    &
          prefix, MINVAL( X_u( dims%x_u_start : dims%x_l_end ) -               &
                          X_l( dims%x_u_start : dims%x_l_end ) ),              &
          prefix, MINVAL( C_u( dims%c_u_start : dims%c_l_end ) -               &
                          C_l( dims%c_u_start : dims%c_l_end ) )
        WRITE( out, "( A, 31X, ' gamma_c,f = ', 2ES12.4 )" )                   &
          prefix, gamma_c, gamma_f
      END IF

!  set the initial barrier parameter

      sigma = sigma_max
      IF ( control%muzero < zero ) THEN
        IF ( control%arc == 2 ) THEN
          mu = slknes
          sigma = one
        ELSE
          mu = sigma * slknes
        END IF
      ELSE
        mu = control%muzero
      END IF
      inform%complementary_slackness = slknes

!  compute the binomial coefficients b_i^k = b_i^{k-1} + b_{i-1}^{k-1}

      IF ( order > 1 ) THEN
        BINOMIAL( 0, 1 ) = one
        DO j = 2, order
          BINOMIAL( j - 1, j - 1 ) = one
          BINOMIAL( 0, j ) = one
          DO i = 1, j - 1
            BINOMIAL( i, j ) = BINOMIAL( i, j - 1 ) + BINOMIAL( i - 1, j - 1 )
          END DO
        END DO
      END IF

!  prepare for the major iteration

      inform%iter = 0 ; inform%nfacts = 0
      IF ( printw ) WRITE( out, "( /, A, ' merit function value = ',           &
     &     ES12.4 )" ) prefix, merit

      IF ( n == 0 ) THEN
        inform%status = GALAHAD_ok ; GO TO 600
      END IF
      merit_best = merit ; it_best = 0

!  compute stopping tolerances

      stop_p = MAX( control%stop_abs_p,                                        &
                    control%stop_rel_p * inform%primal_infeasibility )
      stop_d = MAX( control%stop_abs_d,                                        &
                    control%stop_rel_d * inform%dual_infeasibility )
      stop_c = MAX( control%stop_abs_c,                                        &
                    control%stop_rel_c * inform%complementary_slackness )

!  test for convergence

      CALL CPU_TIME( time_record )
      CALL CHECKPOINT( inform%iter, time_record - time_start,                  &
         MAX( inform%primal_infeasibility,                                     &
         inform%dual_infeasibility, inform%complementary_slackness ),          &
         inform%checkpointsIter, inform%checkpointsTime, 1_ip_, 16_ip_ )
      IF ( inform%primal_infeasibility <= stop_p .AND.                         &
           inform%dual_infeasibility <= stop_d .AND.                           &
           inform%complementary_slackness <= stop_c ) THEN
        inform%status = GALAHAD_ok ; GO TO 600
      END IF

!  ===================================================
!  Analyse the sparsity pattern of the required matrix
!  ===================================================

      re = ' ' ; nbact = 0
      pivot_tol = SLS_control%relative_pivot_tolerance
      min_pivot_tol = SLS_control%minimum_pivot_tolerance
      maxpiv = pivot_tol >= half

      IF ( printi ) WRITE( out,                                                &
          "(  /, A, '  Primal    convergence tolerance =', ES11.4,             &
         &    /, A, '  Dual      convergence tolerance =', ES11.4,             &
         &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )         &
              prefix, stop_p, prefix, stop_d, prefix, stop_c

!  complete A

      DO i = 1, dims%nc
        K_sls%val( s_start + i ) = - SCALE_C( dims%c_equality + i )
      END DO

!  ---------------------------------------------------------------------
!  ---------------------- Start of Major Iteration ---------------------
!  ---------------------------------------------------------------------

      puiseux = control%puiseux
      IF ( puiseux ) THEN
        pui = 'P'
      ELSE
        pui = 'T'
      END IF

      DO

!  =======
!  STEP 1:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Test for Optimality   -*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  print a summary of the iteration

        CALL CLOCK_TIME( clock_now ) ; clock_now = clock_now - clock_start
        IF ( printi ) THEN
          IF ( inform%iter > 0 ) THEN
            IF ( printt .OR. ( printi .AND.                                    &
               inform%iter == start_print ) ) WRITE( out, 2000 ) prefix
            WRITE( out, 2030 ) prefix, inform%iter, re,                        &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness, inform%obj, alpha, mu,            &
             iorder, pui, arc, nbact, clock_now
          ELSE
            WRITE( out, 2000 ) prefix
            WRITE( out, 2020 ) prefix, inform%iter, re,                        &
              inform%primal_infeasibility, inform%dual_infeasibility,          &
              inform%complementary_slackness, inform%obj, mu, clock_now
          END IF

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' X ', X
            IF ( dims%c_l_start <= dims%c_l_end ) WRITE( out, 2100 ) prefix,   &
                ' C_l ', DIST_C_l( dims%c_l_start : dims%c_l_end )
            IF ( dims%c_u_start <= dims%c_u_end ) WRITE( out, 2100 ) prefix,   &
                ' C_u ', DIST_C_u( dims%c_u_start : dims%c_u_end )
            IF ( dims%x_free + 1 <= dims%x_l_end ) WRITE( out, 2100 )          &
              prefix,  ' Z_l ', Z_l( dims%x_free + 1 : dims%x_l_end )
            IF (  dims%x_u_start <= n ) WRITE( out, 2100 )                     &
              prefix, ' Z_u ', Z_u( dims%x_u_start :  n )
          END IF
        END IF

        IF ( control%arc == 2 .OR.                                             &
             ( control%arc == 3 .AND. mu <= tenm4 ) ) THEN
          arc = 'ZS'
        ELSE IF ( control%arc == 4 .OR.                                        &
             ( control%arc == 5 .AND. mu <= tenm10 ) ) THEN
          arc = 'ZP'
          puiseux = .TRUE.
        ELSE
          arc = 'Zh'
        END IF

!  test for optimality

!  find how many primal optimality conditions are violated in the
!  sense that we require (componentwise)
!   | primal optimality | <= MAX( stop_rel * | typical value |, stop_abs )

        IF ( m > 0 ) THEN
          RHS( dims%y_s : dims%c_e + dims%c_equality ) =                       &
            ABS( C_l( : dims%c_equality ) )
          RHS( dims%c_e + dims%c_l_start : dims%c_e +  dims%c_u_end ) =        &
            ABS( SCALE_C * C )
          CALL CLLS_abs_AX( m, RHS( dims%y_s : dims%y_e ), m, A_ne,            &
                           A_val, A_col, A_ptr, n, X, ' ' )
          IF ( printw ) WRITE( out, "( A, '  abs(primal) ', ES12.4 )" )        &
            prefix, MAXVAL( RHS( dims%y_s : dims%y_e ) )
          primal_nonopt = COUNT( ABS( C_RES ) > MAX( control%stop_abs_p,       &
            RHS( dims%y_s : dims%y_e ) * control%stop_rel_p ) )
        ELSE
          primal_nonopt = 0
        END IF

!  now find how many dual optimality conditions are violated in the
!  sense that we require (componentwise)
!   | dual optimality | <= MAX( stop_rel * | typical value |, stop_abs )

!  evaluate abs(dual)

        R_last = ABS( B )
        CALL CLLS_abs_AoX( o, R_last, n, Ao_ne, Ao_val, Ao_row, Ao_ptr,        &
                           n, X, ' ' )
        IF ( weight > zero ) THEN
          RHS( : n ) = weight * ABS( X )
        ELSE
          RHS( : n ) = zero
        END IF
        CALL CLLS_abs_AoX( n, RHS, n, Ao_ne, Ao_val, Ao_row, Ao_ptr,           &
                           o, R_last, 'T' )
        CALL CLLS_abs_AX( n, RHS( : n ), m, A_ne, A_val, A_col, A_ptr, m, Y,   &
                          'T' )
        dual_nonopt = 0
        DO i = 1, dims%x_free
          IF ( ABS( GRAD_L( i ) ) >                                            &
            MAX( control%stop_abs_d, RHS( i ) * control%stop_rel_d ) )         &
              dual_nonopt = dual_nonopt + 1
        END DO
        DO i = dims%x_free + 1, dims%x_u_start - 1
          RHS( i ) = RHS( i ) + ABS( Z_l( i ) )
          IF ( ABS( GRAD_L( i ) - Z_l( i ) ) >                                 &
            MAX( control%stop_abs_d, RHS( i ) * control%stop_rel_d ) )         &
              dual_nonopt = dual_nonopt + 1
        END DO
        DO i = dims%x_u_start, dims%x_l_end
          RHS( i ) = RHS( i ) + ABS( Z_l( i ) ) + ABS( Z_u( i ) )
          IF ( ABS( GRAD_L( i ) - Z_l( i ) - Z_u( i ) ) >                      &
            MAX( control%stop_abs_d, RHS( i ) * control%stop_rel_d ) )         &
              dual_nonopt = dual_nonopt + 1
        END DO
        DO i = dims%x_l_end + 1, n
          RHS( i ) = RHS( i ) + ABS( Z_u( i ) )
          IF ( ABS( GRAD_L( i ) - Z_u( i ) ) >                                 &
            MAX( control%stop_abs_d, RHS( i ) * control%stop_rel_d ) )         &
              dual_nonopt = dual_nonopt + 1
        END DO

        DO i = dims%c_l_start, dims%c_u_start - 1
          RHS( dims%c_b + i ) = ABS( Y( i ) ) + ABS( Y_l( i ) )
          IF ( ABS( Y( i ) - Y_l( i ) ) >  MAX( control%stop_abs_d,            &
            RHS( dims%c_b + i ) * control%stop_rel_d ) )                       &
              dual_nonopt = dual_nonopt + 1
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          RHS( dims%c_b + i ) = ABS( Y( i ) ) + ABS( Y_l( i ) ) + ABS( Y_u( i ))
          IF ( ABS( Y( i ) - Y_l( i ) - Y_u( i ) ) > MAX( control%stop_abs_d,  &
            RHS( dims%c_b + i ) * control%stop_rel_d ) )                       &
              dual_nonopt = dual_nonopt + 1
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          RHS( dims%c_b + i ) = ABS( Y( i ) ) + ABS( Y_u( i ) )
          IF ( ABS( Y( i ) - Y_u( i ) ) >  MAX( control%stop_abs_d,            &
            RHS( dims%c_b + i ) * control%stop_rel_d ) )                       &
              dual_nonopt = dual_nonopt + 1
        END DO

        IF ( printw ) WRITE( out, "( A, '  abs(dual) ', ES12.4 )" )            &
            prefix, MAXVAL( RHS( dims%x_s : dims%c_e ) )

!  finally find how many complementarity conditions are violated in the
!  sense that we require (componentwise)
!   | complementarity | <= MAX( stop_rel * | typical value |, stop_abs )

        cs_nonopt = 0
        cs = zero
        DO i = dims%x_free + 1, dims%x_l_start - 1
          cs = MAX( cs, ABS( Z_l( i ) ), ABS( X( i ) ) )
          IF ( ABS( Z_l( i ) * X( i ) ) > MAX( control%stop_abs_c,             &
                 MAX( ABS( Z_l( i ) ), ABS( X( i ) ) ) * control%stop_rel_c ) )&
                   cs_nonopt = cs_nonopt + 1
        END DO
        DO i = dims%x_l_start, dims%x_l_end
          cs = MAX( cs, ABS( Z_l( i ) ), ABS( DIST_X_l( i ) ) )
          IF ( ABS( Z_l( i ) * DIST_X_l( i ) ) > MAX( control%stop_abs_c,      &
                 MAX( ABS( Z_l( i ) ), ABS( DIST_X_l( i ) ) ) *                &
                   control%stop_rel_c ) ) cs_nonopt = cs_nonopt + 1
        END DO
        DO i = dims%x_u_start, dims%x_u_end
          cs =  MAX( cs, ABS( Z_u( i ) ), ABS( DIST_X_u( i ) ) )
          IF ( ABS( Z_u( i ) * DIST_X_u( i ) ) > MAX( control%stop_abs_c,      &
                 MAX( ABS( Z_u( i ) ), ABS( DIST_X_u( i ) ) ) *                &
                   control%stop_rel_c ) ) cs_nonopt = cs_nonopt + 1
        END DO
        DO i = dims%x_u_end + 1, n
          cs =  MAX( cs,  ABS( Z_u( i ) ), ABS( X( i ) ) )
          IF ( ABS( Z_u( i ) * X( i ) ) > MAX( control%stop_abs_c,             &
                 MAX( ABS( Z_u( i ) ), ABS( X( i ) ) ) * control%stop_rel_c ) )&
                   cs_nonopt = cs_nonopt + 1
        END DO

        DO i = dims%c_l_start, dims%c_l_end
          cs =  MAX( cs,  ABS( Y_l( i ) ), ABS( DIST_C_l( i ) ) )
          IF ( ABS( Y_l( i ) * DIST_C_l( i ) ) > MAX( control%stop_abs_c,      &
                 MAX( ABS( Y_l( i ) ), ABS( DIST_C_l( i ) ) ) *                &
                   control%stop_rel_c ) ) cs_nonopt = cs_nonopt + 1
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          cs =  MAX( cs, ABS( Y_u( i ) ), ABS( DIST_C_u( i ) ) )
          IF ( ABS( Y_u( i ) * DIST_C_u( i ) ) > MAX( control%stop_abs_c,      &
                 MAX( ABS( Y_u( i ) ), ABS( DIST_C_u( i ) ) ) *                &
                   control%stop_rel_c ) ) cs_nonopt = cs_nonopt + 1
          IF ( ABS( Y_u( i ) * DIST_C_u( i ) ) > MAX( control%stop_abs_c,      &
                 MAX( ABS( Y_u( i ) ), ABS( DIST_C_u( i ) ) ) *                &
                   control%stop_rel_c ) ) THEN
          END IF
        END DO

        IF ( printw ) WRITE( out, "( A, '  abs(comp) ', ES12.4 )" )            &
            prefix, cs

        IF ( printw ) WRITE( out, "( A, '  # primal, dual, complementarity',   &
       &                     ' violations ' , I0, 1X, I0, 1X, I0 )" )          &
                                 prefix, primal_nonopt, dual_nonopt, cs_nonopt

!  test for optimality

        CALL CPU_TIME( time_record )
        CALL CHECKPOINT( inform%iter, time_record - time_start,                &
           MAX( inform%primal_infeasibility,                                   &
           inform%dual_infeasibility, inform%complementary_slackness ),        &
           inform%checkpointsIter, inform%checkpointsTime, 1_ip_, 16_ip_ )
        IF ( primal_nonopt + dual_nonopt + cs_nonopt == 0 ) THEN
          inform%status = GALAHAD_ok ; GO TO 600
        END IF

!  test to see if more than maxit iterations have been performed

        inform%iter = inform%iter + 1
        IF ( inform%iter > control%maxit ) THEN
          inform%status = GALAHAD_error_max_iterations ; GO TO 600
        END IF

!  check that the CPU time limit has not been reached

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
             REAL( time_now - time_start, rp_ ) > control%cpu_time_limit ) .OR.&
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

!  Test to see whether the method has stalled

        IF ( merit <= reduce_infeas * merit_best ) THEN
          merit_best = merit
          it_best = 0
        ELSE
          it_best = it_best + 1
          IF ( it_best > infeas_max ) THEN
            IF ( inform%feasible ) THEN
              inform%status = GALAHAD_error_unbounded ; GO TO 600
            ELSE
              IF ( printi ) WRITE( out, "( /, A, ' ================= the ',    &
             &  'problem appears to be infeasible ================= ', / )" )  &
               prefix
              inform%status = GALAHAD_error_primal_infeasible ; GO TO 600
            END IF
          END IF
        END IF

!  compute the barrier terms

!  problem variables:

        DO i = dims%x_free + 1, dims%x_l_start - 1
          IF ( ABS( X( i ) ) <= degen_tol .AND. printd )                       &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, X( i ), Z_l( i )
          BARRIER_X( i ) = Z_l( i ) / X( i )
        END DO
        DO i = dims%x_l_start, dims%x_u_start - 1
          IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, DIST_X_l( i ), Z_l( i )
          BARRIER_X( i ) = Z_l( i ) / DIST_X_l( i )
        END DO
        DO i = dims%x_u_start, dims%x_l_end
          IF ( ABS( DIST_X_l( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, DIST_X_l( i ), Z_l( i )
          IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, DIST_X_u( i ), Z_u( i )
          BARRIER_X( i ) = Z_l( i ) / DIST_X_l( i ) - Z_u( i ) / DIST_X_u( i )
        END DO
        DO i = dims%x_l_end + 1, dims%x_u_end
          IF ( ABS( DIST_X_u( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, DIST_X_u( i ), Z_u( i )
          BARRIER_X( i ) = - Z_u( i ) / DIST_X_u( i )
        END DO
        DO i = dims%x_u_end + 1, n
          IF ( ABS( X( i ) ) <= degen_tol .AND. printd )                       &
            WRITE( out, "( A, ' i = ', i6, ' DIST X, Z ', 2ES12.4 )" )         &
              prefix, i, X( i ), Z_u( i )
          BARRIER_X( i ) = Z_u( i ) / X( i )
        END DO

!  slack variables:

        BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero
        DO i = dims%c_l_start, dims%c_u_start - 1
          IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
              prefix, i, DIST_C_l( i ), Y_l( i )
          BARRIER_C( i ) = Y_l( i ) / DIST_C_l( i )
        END DO
        DO i = dims%c_u_start, dims%c_l_end
          IF ( ABS( DIST_C_l( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
              prefix, i, DIST_C_l( i ), Y_l( i )
          IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
              prefix, i, DIST_C_u( i ), Y_u( i )
          BARRIER_C( i ) = Y_l( i ) / DIST_C_l( i ) - Y_u( i ) / DIST_C_u( i )
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          IF ( ABS( DIST_C_u( i ) ) <= degen_tol .AND. printd )                &
            WRITE( out, "( A, ' i = ', i6, ' DIST C, Y ', 2ES12.4 )" )         &
              prefix, i, DIST_C_u( i ), Y_u( i )
          BARRIER_C( i ) = - Y_u( i ) / DIST_C_u( i )
        END DO

!  =======
!  STEP 2:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-*-      Factorization      -*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  only refactorize if B has changed

        re = 'r'
        CALL CPU_TIME( time )

!  include the values of the barrier terms

        IF ( weight > zero ) THEN
          K_sls%val( d_start + 1 : d_start + dims%x_free ) = weight
          K_sls%val( d_start + dims%x_free + 1 : d_start + n )                 &
            = BARRIER_X + weight
        ELSE
          K_sls%val( d_start + 1 : d_start + dims%x_free ) = zero
          K_sls%val( d_start + dims%x_free + 1 : d_start + n ) = BARRIER_X
        END IF
        K_sls%val( e_start + 1 : e_start + dims%nc ) = BARRIER_C

! ::::::::::::::::::::::::::::::
!  Factorize the required matrix
! ::::::::::::::::::::::::::::::

   200  CONTINUE

!  factorize

!   ( weight I + D     A    Ao^T  )   ( n  dimensional )
!   (               E -S          )   ( nc dimensional )
!   (     A        -S  0          )   ( m  dimensional )
!   (     Ao               W^{-1} )   ( o  dimensional )

!   where D = (X-X_l)^-1 Z_l - (X_u-X)^-1 Z_u
!         E = (C-C_l)^-1 Y_l - (C_u-C)^-1 Y_u
!   and S is a diagonal constraint scaling matrix (by default I)

        IF ( printw ) WRITE( out, "( A,                                        &
       &  ' ......... factorization of KKT matrix ............... ' )" ) prefix

        time_factorize = inform%SLS_inform%time%factorize
        clock_factorize = inform%SLS_inform%time%clock_factorize

        CALL SLS_factorize( K_sls, SLS_data, SLS_control, inform%SLS_inform )
        inform%nfacts = inform%nfacts + 1

        inform%time%factorize = inform%time%factorize +                        &
          inform%SLS_inform%time%factorize - time_factorize
        inform%time%clock_factorize = inform%time%clock_factorize +            &
          inform%SLS_inform%time%clock_factorize - clock_factorize
        time_solve = 0.0 ; clock_solve = 0.0

        IF ( printw ) WRITE( out, "( A,                                        &
       &  ' ............... end of factorization ............... ' )" ) prefix

!  test that the factorization succeeded

        inform%factorization_status = inform%status
        IF ( inform%factorization_status < 0 ) THEN
          IF ( printe ) WRITE( error, "( A, '    ** Error return ', I0,        &
         &  ' from ', A )" ) prefix, inform%factorization_status,              &
            'SLS_factorize'

!  it didn't. We might have run out of options

          IF ( maxpiv ) THEN
            inform%status = GALAHAD_error_factorization ; GO TO 600

!  ... or we can increase the pivot tolerance

          ELSE IF ( SLS_control%relative_pivot_tolerance                       &
                      < relative_pivot_default ) THEN
            pivot_tol = relative_pivot_default
            min_pivot_tol = relative_pivot_default
            maxpiv = .FALSE.
            SLS_control%relative_pivot_tolerance = pivot_tol
            SLS_control%minimum_pivot_tolerance = min_pivot_tol
            IF ( printi ) WRITE( out,                                          &
              "( A, '    ** Pivot tolerance increased to', ES11.4 )" )         &
              prefix, pivot_tol
          ELSE
            pivot_tol = half
            min_pivot_tol = half
            maxpiv = .TRUE.
            SLS_control%relative_pivot_tolerance = pivot_tol
            SLS_control%minimum_pivot_tolerance = min_pivot_tol
            IF ( printi ) WRITE( out,                                          &
              "( A, '    ** Pivot tolerance increased to', ES11.4 )" )         &
              prefix, pivot_tol
          END IF
          alpha = zero ; nbact = 0
          inform%factorization_integer = - 1
          inform%factorization_real = - 1
          CYCLE

!  record warning conditions

        ELSE IF (inform%factorization_status > 0 ) THEN
           IF ( printt ) THEN
              WRITE( out, "( A, '   **  Warning ', I0, ' from ', A )" )        &
              prefix, inform%status, 'SLS_factorize'
            END IF
        END IF

!  Record the storage required

        inform%factorization_integer =                                         &
          inform%SLS_inform%integer_size_necessary
        inform%factorization_real =                                            &
          inform%SLS_inform%real_size_necessary

        IF ( printt ) THEN
          WRITE( out, "( A, ' factorization time = ', F0.2 )" ) prefix,        &
            inform%SLS_inform%time%factorize - time_factorize +                &
            inform%SLS_inform%time%clock_factorize - clock_factorize
          WRITE( out, "( A, 1X, I0, ' integer and ', I0, ' real words needed', &
         &    ' for factorization' )" ) prefix, inform%factorization_integer,  &
                                        inform%factorization_real
        END IF

!       IF ( printw ) WRITE( out, "( A,                                        &
!      &  ' ............... end of factorization ............... ' )" ) prefix

!  =======
!  STEP 3:
!  =======

        IF ( arc == 'ZS' .OR. arc == 'ZP' ) THEN
          two_mu = two * mu
          sigma_mu = sigma * mu
          sigma_mu2 = sigma_mu * mu
          IF ( puiseux ) THEN
            two_sigma_mu = two * sigma_mu
            two_sigma_mu2 = two * sigma_mu2
            two_plus_sigma_mu = two + sigma_mu
            one_plus_2_sigma_mu = one + two_sigma_mu
          ELSE
            one_plus_sigma_mu = one + sigma_mu
          END IF
        END IF

!  =======================================================================
!  -*-*-*-*-*-*-*-*-   Obtain the Primal-Dual Search Arc -*-*-*-*-*-*-*-*-
!  =======================================================================

!  we consider the search arc

!     v_l(alpha) = v + sum_k=1^l [ (-1)^k v^k / k! ] alpha^k

!  as alpha inceases from 0 to 1 and where v_l(alpha) is the l-th-order Taylor
!  series approximation of the arc v(1-alpha)) about alpha = 0 (equiv theta
!  = 1 - alpha about theta = 1) and for which v(theta) satisfies the conditions

!  ( Ao^T W Ao + weight I ) x(theta) - A^T y(theta) - z_l(theta) - z_u(theta)
!       - Ao^T W b         = dual(theta)
!  A x(theta) - S c(theta) = prim(theta)
!  X(theta) z(theta)       = comp(theta)

!  for suitable
!      prim(theta) = theta ( A x - S c )
!      dual(theta) = theta ( Ao^T W A_o x - A^T y - z - A_o^T W b )
!  (Taylor or Taylor-Puisuex) or
!      prim(theta) = theta^2 ( A x - S c )
!      dual(theta) = theta^2 ( Ao^T W Ao x - A^T y - z - Ao^T W b )
!  (Puiseux) and various possible comp(theta)

!  Let r = W ( Ao x - b ), g_l = Ao^T r + weight x - A^T y and r_c = A x - S c

!  To find the coefficients v^k = ( x^k, c^k, y^k, z_l^k, z_u^k, y_l^k, y_u^k ),
!  solve the equations

!   (  Ao^T W Ao + weight I     A^T -I   -I             ) (  x^k  )   (  h^k  )
!   (                           -S            -I  -I    ) (  c^k  )   (  d^k  )
!   (  A                    -S                          ) ( -y^k  )   (  a^k  )
!   (  Z_l                         X-X_l                ) ( z_l^k ) = ( r_l^k )
!   ( -Z_u                              X_u-X           ) ( z_u^k )   ( r_u^k )
!   (                       Y_l              C-C_l      ) ( y_l^k )   ( s_l^k )
!   (                      -Y_u                   C_u-C ) ( y_u^k )   ( s_u^k )

!  for k > 0 for which

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  arc 1: for the Zhang arc,
!    comp(theta) =   theta Xz + (1-theta) sigma mu e      (lower)
!            or    - theta Xz - (1-theta) sigma mu e      (upper)
!  (Taylor) or
!    comp(theta) =   theta^2 Xz + (1-theta^2) sigma mu e  (lower)
!            or    - theta^2 Xz - (1-theta^2) sigma mu e  (upper)
!  (Taylor-Puisuex or Puiseux)
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     h^1 = g_l - z_l - z_u
!     d^i = y - y_l - y_u
!     a^1 = r_c
!     r_l^1 = - mu e + (X-X_l)z_l                    (store in z_l^1)
!     r_u^1 =   mu e + (X_u-X)z_u                    (store in z_u^1)
!     s_l^1 = - mu e + (C-C_l)y_l                    (store in y_l^1)
!   & s_u^1 =   mu e + (C_u-C)y_u                    (store in y_u^1)

!  (k=1) and

!     h^k = 0
!     d^k = 0
!     a^k = 0
!     r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}      (store in z_l^k)
!     r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}      (store in z_u^k)
!     s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}      (store in y_l^k)
!   & s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}      (store in y_u^k)

!  (k>1) for the Taylor arc,

!     h^1 = g_l - z_l - z_u
!     d^i = y - y_l - y_u
!     a^1 = r_c
!     r_l^1 = 2 ( - mu e + (X-X_l)z_l )              (store in z_l^1)
!     r_u^1 = 2 (   mu e + (X_u-X)z_u )              (store in z_u^1)
!     s_l^1 = 2 ( - mu e + (C-C_l)y_l )              (store in y_l^1)
!   & s_u^1 = 2 (   mu e + (C_u-C)y_u )              (store in y_u^1)

!  (k=1),

!     h^2 = 0
!     d^2 = 0
!     a^2 = 0
!     r_l^2 = 2 ( - mu e + (X-X_l)z_l - X^1 z_l^1 )  (store in z_l^2)
!     r_u^2 = 2 (   mu e + (X_u-X)z_u + X^1 z_u^1 )  (store in z_u^2)
!     s_l^2 = 2 ( - mu e + (C-C_l)y_l - C^1 y_l^1 )  (store in y_l^2)
!   & s_u^2 = 2 (   mu e + (C_u-C)y_u + C^1 y_u^1 )  (store in y_u^2)

!  (k=2) and

!     h^k = 0
!     d^k = 0
!     a^k = 0
!     r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}      (store in z_l^k)
!     r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}      (store in z_u^k)
!     s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}      (store in y_l^k)
!   & s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}      (store in y_u^k)

!  (k>2) for the Zhang-Puiseux Taylor arc, or

!     h^1 = 2 ( g_l - z_l - z_u )
!     d^i = 2 ( y - y_l - y_u )
!     a^1 = 2 r_c
!     r_l^1 = 2 ( - mu e + (X-X_l)z_l )              (store in z_l^1)
!     r_u^1 = 2 (   mu e + (X_u-X)z_u )              (store in z_u^1)
!     s_l^1 = 2 ( - mu e + (C-C_l)y_l )              (store in y_l^1)
!   & s_u^1 = 2 (   mu e + (C_u-C)y_u )              (store in y_u^1)

!  (k=1),

!     h^1 = 2 ( g_l - z_l - z_u )
!     d^2 = 2 ( y - y_l - y_u )
!     a^2 = 2 r_c
!     r_l^2 = 2 ( - mu e + (X-X_l)z_l - X^1 z_l^1 )  (store in z_l^2)
!     r_u^2 = 2 (   mu e + (X_u-X)z_u + X^1 z_u^1 )  (store in z_u^2)
!     s_l^2 = 2 ( - mu e + (C-C_l)y_l - C^1 y_l^1 )  (store in y_l^2)
!   & s_u^2 = 2 (   mu e + (C_u-C)y_u + C^1 y_u^1 )  (store in y_u^2)

!  (k=2) and

!     h^k = 0
!     d^k = 0
!     a^k = 0
!     r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}      (store in z_l^k)
!     r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}      (store in z_u^k)
!     s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}      (store in y_l^k)
!   & s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}      (store in y_u^k)

!  (k>2) for the Puiseux arc, where b_i^k is the binomial coefficient
!     "k choose i"

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  arc 2: for the Zhao-Sun arc, comp(theta) =
!       theta Xz + sigma mu theta ( 1 - theta ) ( mu e - X z ) (lower) or
!     - theta Xz - sigma mu theta ( 1 - theta ) ( mu e - X z ) (upper)
!   or, in the Puiseux case,
!       theta^2 Xz + sigma mu theta^2 ( 1 - theta ) ( mu e - X z ) (lower) or
!     - theta^2 Xz - sigma mu theta^2 ( 1 - theta ) ( mu e - X z ) (upper)
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     h^1 = g_l - z_l - z_u
!     d^i = y - y_l - y_u
!     a^1 =  r_c
!     r_l^1 = - sigma mu [ mu e - (X-X_l)z_l ] + (X-X_l)z_l  (store in z_l^1)
!     r_u^1 =   sigma mu [ mu e + (X_u-X)z_u ] + (X_u-X)z_u  (store in z_u^1)
!     s_l^1 = - sigma mu [ mu e - (C-C_l)y_l ] + (C-C_l)y_l  (store in y_l^1)
!   & s_u^1 =   sigma mu [ mu e + (C_u-C)y_u ] + (C_u-C)y_u  (store in y_u^1)

!  (k=1),

!     h^2 = 0
!     d^2 = 0
!     a^2 = 0
!     r_l^2 = 2 ( - sigma mu [mu e - (X-X_l)z_l] - X^1 z_l^1 ) (store in z_l^2)
!     r_u^2 = 2 (   sigma mu [mu e + (X_u-X)z_u] + X^1 z_u^1 ) (store in z_u^2)
!     s_l^2 = 2 ( - sigma mu [mu e - (C-C_l)y_l] - C^1 y_l^1 ) (store in y_l^2)
!   & s_u^2 = 2 (   sigma mu [mu e + (C_u-C)y_u] + C^1 y_u^1 ) (store in y_u^2)

!  (k=2) and

!     h^k = 0
!     d^k = 0
!     a^k = 0
!     r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}      (store in z_l^k)
!     r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}      (store in z_u^k)
!     s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}      (store in y_l^k)
!   & s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}      (store in y_u^k)

!  (k>2) for the Taylor arc, or

!     h^1 = 2 ( g - A^Ty - z_l - z_u )
!     d^i = 2 ( y - y_l - y_u )
!     a^1 = 2 r_c
!     r_l^1 = - sigma mu [ mu e - (X-X_l)z_l ] + 2(X-X_l)z_l  (store in z_l^1)
!     r_u^1 =   sigma mu [ mu e + (X_u-X)z_u ] + 2(X_u-X)z_u  (store in z_u^1)
!     s_l^1 = - sigma mu [ mu e - (C-C_l)y_l ] + 2(C-C_l)y_l  (store in y_l^1)
!   & s_u^1 =   sigma mu [ mu e + (C_u-C)y_u ] + 2(C_u-C)y_u  (store in y_u^1)

!  (k=1),

!     h^2 = 2 ( g - A^Ty - z_l - z_u )
!     d^2 = 2 ( y - y_l - y_u )
!     a^2 = 2 r_c
!     r_l^2 = - 4 sigma mu [ mu e - (X-X_l)z_l ] + 2(X-X_l)z_l - 2 X^1 z_l^1
!                                                              (store in z_l^2)
!     r_u^2 =   4 sigma mu [ mu e + (X_u-X)z_u ] + 2(X_u-X)z_u + 2 X^1 z_u^1
!                                                              (store in z_u^2)
!     s_l^2 = - 4 sigma mu [ mu e - (C-C_l)y_l ] + 2(C-C_l)y_l - 2 C^1 y_l^1
!                                                              (store in y_l^2)
!   & s_u^2 =   4 sigma mu [ mu e + (C_u-C)y_u ] + 2(C_u-C)y_u + 2 C^1 y_u^1
!                                                              (store in y_u^2)

!  (k=2) and

!     h^3 = 0
!     d^3 = 0
!     a^3 = 0
!     r_l^3 = - 6 sigma mu [ mu e - (X-X_l)z_l ] - 3 [ X^1 z_l^2 + X^2 z_l^1 ]
!                                                              (store in z_l^3)
!     r_u^3 =   6 sigma mu [ mu e + (X_u-X)z_u ] + 3 [ X^1 z_u^2 + X^2 z_u^1 ]
!                                                              (store in z_u^3)
!     s_l^3 = - 6 sigma mu [ mu e - (C-C_l)y_l ] - 3 [ C^1 c_l^2 + C^2 c_l^1 ]
!                                                              (store in y_l^3)
!   & s_u^3 =   6 sigma mu [ mu e + (C_u-C)y_u ] + 3 [ C^1 c_u^2 + C^2 c_u^1 ]
!                                                              (store in y_u^3)

!  (k=3) and

!     h^k = 0
!     d^k = 0
!     a^k = 0
!     r_l^k = - sum_i=1^k-1 b_i^k X^i z_l^{k-i}      (store in z_l^k)
!     r_u^k =   sum_i=1^k-1 b_i^k X^i z_u^{k-i}      (store in z_u^k)
!     s_l^k = - sum_i=1^k-1 b_i^k C^i y_l^{k-i}      (store in y_l^k)
!   & s_u^k =   sum_i=1^k-1 b_i^k C^i y_u^{k-i}      (store in y_u^k)

!  (k>3) for the Puiseux arc

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Observing that
!
!        z_l^k = (X-X_l)^-1 [ r_l^k - Z_l x^k ]
!        z_u^k = (X_u-X)^-1 [ r_u^k + Z_u x^k ]
!        y_l^k = (C-C_l)^-1 [ s_l^k - Y_l c^k ]
!      & y_u^k = (C_u-C)^-1 [ s_u^k + Y_u c^k ]
!
!  and introducing r^k = W Ao x^k, we find on substitution that
!
!   ( weight I + D     A     Ao^T  ) ( x^k )
!   (               E -S           ) ( c^k ) = rhs =
!   (     A        -S  0           ) (-y^k )
!   (     Ao               -W^{-1} ) ( r^k )
!
!      ( h^k + (X-X_l)^-1 r_l^k + (X_u-X)^-1 r_u^k )
!      ( d^k + (C-C_l)^-1 s_l^k + (C_u-C)^-1 s_u^k )
!      (                 a^k                       )
!      (                  0                        )
!
!  where we recall that
!      D = (X-X_l)^-1 Z_l - (X_u-X)^-1 Z_u
!  and E = (C-C_l)^-1 Y_l - (C_u-C)^-1 Y_u
!
!  record the 0-th order coefficients

        X_coef( : , 0 ) = X
        C_coef( : , 0 ) = C
        Y_coef( : , 0 ) = Y
        Z_l_coef( : , 0 ) = Z_l
        Z_u_coef( : , 0 ) = Z_u
        Y_l_coef( : , 0 ) = Y_l
        Y_u_coef( : , 0 ) = Y_u

!  compute the k-th order coefficients

        DO k = 1, order

!  :::::::::::::::::::::::::::::::::::::
!  3a. Set up the right-hand-side vector
!  :::::::::::::::::::::::::::::::::::::

!  record rhs = ( h^k + (X-X_l)^-1 r_l^k + (X_u-X)^-1 r_u^k )
!               ( d^k + (C-C_l)^-1 s_l^k + (C_u-C)^-1 s_u^k )
!               (                   a^k                     )
!               (                    0                      )

          IF ( printd ) WRITE( out, 2100 )                                     &
            prefix, ' GRAD_L', GRAD_L( dims%x_s : dims%x_e )

!  for the 1st order Taylor and 1st and 2nd order Puiseux coefficients
!  and the Zhang arc or the 1st and 2nd order Zhang-Puiseux arc

          IF ( ( arc == 'Zh' .AND. ( k == 1 .OR. ( k == 2 .AND. puiseux ) ) )  &
                 .OR. ( arc == 'ZP' .AND. k <= 2 ) ) THEN

!  compute and store ( r_l^1, r_u^1, s_l^1, s_u^1 )

!  for the 1st order coefficients

            IF ( k == 1 ) THEN
              DO i = dims%x_free + 1, dims%x_l_end
                Z_l_coef( i, 1 ) = - mu + ( X( i ) - X_l( i ) ) * Z_l( i )
              END DO
              DO i = dims%x_u_start, n
                Z_u_coef( i, 1 ) =   mu + ( X_u( i ) - X( i ) ) * Z_u( i )
              END DO
              DO i = dims%c_l_start, dims%c_l_end
                Y_l_coef( i, 1 ) = - mu + ( C( i ) - C_l( i ) ) * Y_l( i )
              END DO
              DO i = dims%c_u_start, dims%c_u_end
                Y_u_coef( i, 1 ) =   mu + ( C_u( i ) - C( i ) ) * Y_u( i )
              END DO

!  for the 2nd order Puiseux coefficients

            ELSE
              DO i = dims%x_free + 1, dims%x_l_end
                Z_l_coef( i, 2 ) = - mu + ( X( i ) - X_l( i ) ) * Z_l( i ) -   &
                  X_coef( i, 1 ) * Z_l_coef( i, 1 )
              END DO
              DO i = dims%x_u_start, n
                Z_u_coef( i, 2 ) =   mu + ( X_u( i ) - X( i ) ) * Z_u( i ) +   &
                  X_coef( i, 1 ) * Z_u_coef( i, 1 )
              END DO
              DO i = dims%c_l_start, dims%c_l_end
                Y_l_coef( i, 2 ) = - mu + ( C( i ) - C_l( i ) ) * Y_l( i ) -   &
                  C_coef( i, 1 ) * Y_l_coef( i, 1 )
              END DO
              DO i = dims%c_u_start, dims%c_u_end
                Y_u_coef( i, 2 ) =   mu + ( C_u( i ) - C( i ) ) * Y_u( i ) +   &
                  C_coef( i, 1 ) * Y_u_coef( i, 1 )
              END DO
            END IF

!  double the Puiseux coefficients

            IF ( puiseux ) THEN
              Z_l_coef( dims%x_free + 1 : dims%x_l_end, k ) =                  &
                two * Z_l_coef( dims%x_free + 1 : dims%x_l_end, k )
              Z_u_coef( dims%x_u_start : n, k ) =                              &
                two * Z_u_coef( dims%x_u_start : n, k )
              Y_l_coef( dims%c_l_start : dims%c_l_end, k ) =                   &
                two * Y_l_coef( dims%c_l_start : dims%c_l_end, k )
              Y_u_coef( dims%c_u_start : dims%c_u_end, k ) =                   &
                two * Y_u_coef( dims%c_u_start : dims%c_u_end, k )
            END IF

!  for the Zhang arc

            IF ( arc == 'Zh' ) THEN

!  for the 1-st order rhs

              IF ( k == 1 ) THEN

!  rhs for problem variables: g + Hx - A^Ty - mu (X-X_l)^-1 e + mu (X_u-X)^-1 e

                RHS( : dims%x_free ) = GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = GRAD_L( i ) - mu / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = GRAD_L( i ) - mu / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = GRAD_L( i ) - mu / DIST_X_l( i )                  &
                                         + mu / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = GRAD_L( i ) + mu / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = GRAD_L( i ) - mu / X( i )
                END DO

!  rhs for slack variables: y - mu (C-C_l)^-1 e + mu (C_u-C)^-1 e

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = Y( i ) - mu / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = Y( i ) - mu / DIST_C_l( i )            &
                                               + mu / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = Y( i ) + mu / DIST_C_u( i )
                END DO

!  for the 2nd order rhs

              ELSE

!  rhs for problem variables: g + Hx - A^Ty - (X-X_l)^-1 ( mu e + X^1 z_l^1 )
!    + (X_u-X)^-1 ( mu e + X^1 z_u^1 )

                RHS( : dims%x_free ) = GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = GRAD_L( i )                                       &
                    - ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = GRAD_L( i )                                       &
                    - ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = GRAD_L( i )                                       &
                   - ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / DIST_X_l( i )&
                   + ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = GRAD_L( i )                                       &
                    + ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = GRAD_L( i )                                       &
                    - ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / X( i )
                END DO

!  rhs for slack variables: y - (C-C_l)^-1 ( mu e + C^1 y_l^1 )
!    + (C_u-C)^-1 ( mu e + C^1 y_u^1 )

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = Y( i )                                 &
                    - ( mu + C_coef( i, 1 ) * Y_l_coef( i, 1 ) ) / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = Y( i )                                 &
                   - ( mu + C_coef( i, 1 ) * Y_l_coef( i, 1 ) ) / DIST_C_l( i )&
                  + ( mu + C_coef( i, 1 ) * Y_u_coef( i, 1 ) ) / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = Y( i )                                 &
                    + ( mu + C_coef( i, 1 ) * Y_u_coef( i, 1 ) ) / DIST_C_u( i )
                END DO
              END IF

!  rhs for constraint infeasibilities: A x - S c

              RHS( dims%y_s : dims%y_e ) = C_RES( : dims%c_u_end )

!  double the Puiseux rhs

              IF ( puiseux ) RHS( : dims%y_e ) = two * RHS( : dims%y_e )

!  for the Zhang-Puiseux arc

            ELSE

!  for the 1-st order rhs

              IF ( k == 1 ) THEN

!  rhs for problem variables: g + Hx - A^Ty + z_l + z_u -
!                             2 mu (X-X_l)^-1 e + 2 mu (X_u-X)^-1 e )

                RHS( : dims%x_free ) = GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = GRAD_L( i ) + Z_l( i ) - two_mu / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = GRAD_L( i ) + Z_l( i ) - two_mu / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = GRAD_L( i ) + Z_l( i ) + Z_u( i )                 &
                                         - two_mu / DIST_X_l( i )              &
                                         + two_mu / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = GRAD_L( i ) + Z_u( i ) + two_mu / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = GRAD_L( i ) + Z_u( i ) - two_mu / X( i )
                END DO

!  rhs for slack variables:
!       y + y_l + y_u - 2 mu (C-C_l)^-1 e + 2 mu (C_u-C)^-1 e

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = Y( i ) + Y_l( i )                      &
                                               - two_mu / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = Y( i ) + Y_l( i ) + Y_u( i )           &
                                               - two_mu / DIST_C_l( i )        &
                                               + two_mu / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = Y( i ) + Y_u( i )                      &
                                               + two_mu / DIST_C_u( i )
                END DO

!  rhs for constraint infeasibilities: A x - S c

                RHS( dims%y_s : dims%y_e ) = C_RES( : dims%c_u_end )

!  for the 2nd order rhs

              ELSE

!  rhs for problem variables: 2 z_l + 2 z_u
!                  - 2 (X-X_l)^-1 ( mu e + X^1 z_l^1 )
!                  + 2 (X_u-X)^-1 ( mu e + X^1 z_u^1 )

                RHS( : dims%x_free ) = zero
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = two * ( Z_l( i ) -                                &
                    ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / X( i ) )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = two * ( Z_l( i ) -                                &
                    ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / DIST_X_l( i ) )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = two * ( Z_l( i ) + Z_u( i ) -                     &
                   ( mu + X_coef( i, 1 ) * Z_l_coef( i, 1 ) ) / DIST_X_l( i ) +&
                   ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / DIST_X_u( i ) )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = two * ( Z_u( i ) +                                &
                    ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / DIST_X_u( i ) )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = two * ( Z_u( i ) -                                &
                    ( mu + X_coef( i, 1 ) * Z_u_coef( i, 1 ) ) / X( i ) )
                END DO

!  rhs for slack variables: 2 y_l + 2 y_u
!             - 2 (C-C_l)^-1 ( mu e + C^1 y_l^1 )
!             + 2 (C_u-C)^-1 ( mu e + C^1 y_u^1 )

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = two * ( Y_l( i ) -                     &
                    ( mu + C_coef( i, 1 ) * Y_l_coef( i, 1 ) ) / DIST_C_l( i ) )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = two * ( Y_l( i ) + Y_u( i ) -          &
                   ( mu + C_coef( i, 1 ) * Y_l_coef( i, 1 ) ) / DIST_C_l( i ) +&
                  ( mu + C_coef( i, 1 ) * Y_u_coef( i, 1 ) ) / DIST_C_u( i ) )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = two * ( Y_u( i ) +                     &
                    ( mu + C_coef( i, 1 ) * Y_u_coef( i, 1 ) ) / DIST_C_u( i ) )
                END DO

!  rhs for constraint infeasibilities: 0

                RHS( dims%y_s : dims%y_e ) = zero
              END IF
            END IF

!  for the 1st and 2nd order Taylor and 1st to 3rd order Puiseux coefficients
!  and the Zhao-Sun arc

          ELSE IF ( arc == 'ZS' .AND.                                          &
            ( k <= 2 .OR. ( k <= 3 .AND. puiseux ) ) ) THEN

!  compute and store ( r_l^1, r_u^1, s_l^1, s_u^1 )

!  for the 1-st order coefficients

            IF ( k == 1 ) THEN

!  Puiseux case

              IF ( puiseux ) THEN
                DO i = dims%x_free + 1, dims%x_l_end
                  Z_l_coef( i, 1 ) = - sigma_mu2                               &
                    + two_plus_sigma_mu * ( X( i ) - X_l( i ) ) * Z_l( i )
                END DO
                DO i = dims%x_u_start, n
                  Z_u_coef( i, 1 ) =   sigma_mu2                               &
                    + two_plus_sigma_mu * ( X_u( i ) - X( i ) ) * Z_u( i )
                END DO
                DO i = dims%c_l_start, dims%c_l_end
                  Y_l_coef( i, 1 ) = - sigma_mu2                               &
                    + two_plus_sigma_mu * ( C( i ) - C_l( i ) ) * Y_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_u_end
                  Y_u_coef( i, 1 ) =   sigma_mu2                               &
                    + two_plus_sigma_mu * ( C_u( i ) - C( i ) ) * Y_u( i )
                END DO

!  Taylor case

              ELSE
                DO i = dims%x_free + 1, dims%x_l_end
                  Z_l_coef( i, 1 ) = - sigma_mu2                               &
                    + one_plus_sigma_mu * ( X( i ) - X_l( i ) ) * Z_l( i )
                END DO
                DO i = dims%x_u_start, n
                  Z_u_coef( i, 1 ) =   sigma_mu2                               &
                    + one_plus_sigma_mu * ( X_u( i ) - X( i ) ) * Z_u( i )
                END DO
                DO i = dims%c_l_start, dims%c_l_end
                  Y_l_coef( i, 1 ) = - sigma_mu2                               &
                    + one_plus_sigma_mu * ( C( i ) - C_l( i ) ) * Y_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_u_end
                  Y_u_coef( i, 1 ) =   sigma_mu2                               &
                    + one_plus_sigma_mu * ( C_u( i ) - C( i ) ) * Y_u( i )
                END DO
              END IF

!  for the 2nd order coefficients

            ELSE IF ( k == 2 ) THEN

!  Puiseux case

              IF ( puiseux ) THEN
                DO i = dims%x_free + 1, dims%x_l_end
                  Z_l_coef( i, 2 ) = two * ( - two * sigma_mu2                 &
                    + one_plus_2_sigma_mu * ( X( i ) - X_l( i ) ) * Z_l( i )   &
                      - X_coef( i, 1 ) * Z_l_coef( i, 1 ) )
                END DO
                DO i = dims%x_u_start, n
                  Z_u_coef( i, 2 ) = two * (   two * sigma_mu2                 &
                    + one_plus_2_sigma_mu * ( X_u( i ) - X( i ) ) * Z_u( i )   &
                      + X_coef( i, 1 ) * Z_u_coef( i, 1 ) )
                END DO
                DO i = dims%c_l_start, dims%c_l_end
                  Y_l_coef( i, 2 ) = two * ( - two * sigma_mu2                 &
                    + one_plus_2_sigma_mu * ( C( i ) - C_l( i ) ) * Y_l( i )   &
                      - C_coef( i, 1 ) * Y_l_coef( i, 1 ) )
                END DO
                DO i = dims%c_u_start, dims%c_u_end
                  Y_u_coef( i, 2 ) = two * (   two * sigma_mu2                 &
                    + one_plus_2_sigma_mu * ( C_u( i ) - C( i ) ) * Y_u( i )   &
                      + C_coef( i, 1 ) * Y_u_coef( i, 1 ) )
                END DO

!  Taylor case

              ELSE
                DO i = dims%x_free + 1, dims%x_l_end
                  Z_l_coef( i, 2 ) = two * (                                   &
                    - sigma_mu * ( mu - ( X( i ) - X_l( i ) ) * Z_l( i ) )     &
                      - X_coef( i, 1 ) * Z_l_coef( i, 1 ) )
                END DO
                DO i = dims%x_u_start, n
                  Z_u_coef( i, 2 ) = two * (                                   &
                      sigma_mu * ( mu + ( X_u( i ) - X( i ) ) * Z_u( i ) )     &
                      + X_coef( i, 1 ) * Z_u_coef( i, 1 ) )
                END DO
                DO i = dims%c_l_start, dims%c_l_end
                  Y_l_coef( i, 2 ) = two * (                                   &
                    - sigma_mu * ( mu - ( C( i ) - C_l( i ) ) * Y_l( i ) )     &
                      - C_coef( i, 1 ) * Y_l_coef( i, 1 ) )
                END DO
                DO i = dims%c_u_start, dims%c_u_end
                  Y_u_coef( i, 2 ) = two * (                                   &
                      sigma_mu * ( mu + ( C_u( i ) - C( i ) ) * Y_u( i ) )     &
                      + C_coef( i, 1 ) * Y_u_coef( i, 1 ) )
                END DO
              END IF

!  for the 3rd order Puiseux coefficients

            ELSE
              DO i = dims%x_free + 1, dims%x_l_end
                Z_l_coef( i, 3 ) = three * ( - two_sigma_mu2                   &
                  + two_sigma_mu * ( X( i ) - X_l( i ) ) * Z_l( i )            &
                    - X_coef( i, 1 ) * Z_l_coef( i, 2 )                        &
                    - X_coef( i, 2 ) * Z_l_coef( i, 1 ) )
              END DO
              DO i = dims%x_u_start, n
                Z_u_coef( i, 3 ) = three * (   two_sigma_mu2                   &
                  + two_sigma_mu * ( X_u( i ) - X( i ) ) * Z_u( i )            &
                    + X_coef( i, 1 ) * Z_u_coef( i, 2 )                        &
                    + X_coef( i, 2 ) * Z_u_coef( i, 1 ) )
              END DO
              DO i = dims%c_l_start, dims%c_l_end
                Y_l_coef( i, 3 ) = three * ( - two_sigma_mu2                   &
                  + two_sigma_mu * ( C( i ) - C_l( i ) ) * Y_l( i )            &
                    - C_coef( i, 1 ) * Y_l_coef( i, 2 )                        &
                    - C_coef( i, 2 ) * Y_l_coef( i, 1 ) )
              END DO
              DO i = dims%c_u_start, dims%c_u_end
                Y_u_coef( i, 3 ) = three * (   two_sigma_mu2                   &
                  + two_sigma_mu * ( C_u( i ) - C( i ) ) * Y_u( i )            &
                    + C_coef( i, 1 ) * Y_u_coef( i, 2 )                        &
                    + C_coef( i, 2 ) * Y_u_coef( i, 1 ) )
              END DO
            END IF

!  record rhs = h^k + (X-X_l)^-1 r_l^k + (X_u-X)^-1 r_u^k

!  for the 1st order rhs

            IF ( k == 1 ) THEN

!  Puiseux case

              IF ( puiseux ) THEN

!  rhs for problem variables:
!   2 ( g + Hx - A^Ty - z_l - z_u ) + (X-X_l)^-1 r_l^1 + (X_u-X)^-1 r_u^1 )

                RHS( : dims%x_free ) = two * GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) ) +                &
                    Z_l_coef( i, 1 ) / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) ) +                &
                    Z_l_coef( i, 1 ) / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) - Z_u( i ) ) +     &
                    Z_l_coef( i, 1 ) / DIST_X_l( i ) +                         &
                    Z_u_coef( i, 1 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = two * ( GRAD_L( i ) - Z_u( i ) ) +                &
                    Z_u_coef( i, 1 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = two * ( GRAD_L( i ) - Z_u( i ) ) -                &
                    Z_u_coef( i, 1 ) / X( i )
                END DO

!  rhs for slack variables:
!    2 ( y - y_l - y_u ) + (C-C_l)^-1 s_l^k + (C_u-C)^-1 s_u^k )

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_l( i ) ) +          &
                    Y_l_coef( i, 1 ) / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_l( i ) - Y_u( i )) +&
                    Y_l_coef( i, 1 ) / DIST_C_l( i ) +                         &
                    Y_u_coef( i, 1 ) / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_u( i ) ) +          &
                    Y_u_coef( i, 1 ) / DIST_C_u( i )
                END DO

!  rhs for constraint infeasibilities: 2 ( A x - S c )

                RHS( dims%y_s : dims%y_e ) = two * C_RES( : dims%c_u_end )

!  Taylor case

              ELSE

!  rhs for problem variables:
!   g + Hx - A^Ty - z_l - z_u + (X-X_l)^-1 r_l^1 + (X_u-X)^-1 r_u^1

                RHS( : dims%x_free ) = GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = GRAD_L( i ) - Z_l( i ) +                          &
                    Z_l_coef( i, 1 ) / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = GRAD_L( i ) - Z_l( i ) +                          &
                    Z_l_coef( i, 1 ) / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = GRAD_L( i ) - Z_l( i ) - Z_u( i ) +               &
                    Z_l_coef( i, 1 ) / DIST_X_l( i ) +                         &
                    Z_u_coef( i, 1 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = GRAD_L( i ) - Z_u( i ) +                          &
                    Z_u_coef( i, 1 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = GRAD_L( i ) - Z_u( i ) -                          &
                    Z_u_coef( i, 1 ) / X( i )
                END DO

!  rhs for slack variables:
!    y - y_l - y_u + (C-C_l)^-1 s_l^k + (C_u-C)^-1 s_u^k

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = Y( i ) - Y_l( i ) +                    &
                    Y_l_coef( i, 1 ) / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = Y( i ) - Y_l( i ) - Y_u( i ) +         &
                    Y_l_coef( i, 1 ) / DIST_C_l( i ) +                         &
                    Y_u_coef( i, 1 ) / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = Y( i ) - Y_u( i ) +                    &
                    Y_u_coef( i, 1 ) / DIST_C_u( i )
                END DO

!  rhs for constraint infeasibilities: A x - S c

                RHS( dims%y_s : dims%y_e ) = C_RES( : dims%c_u_end )
              END IF

!  for the 2nd order rhs

            ELSE IF ( k == 2 ) THEN

!  Puiseux case

              IF ( puiseux ) THEN

!  rhs for problem variables:
!   2 ( g + Hx - A^Ty - z_l - z_u ) + (X-X_l)^-1 r_l^2 + (X_u-X)^-1 r_u^2

                RHS( : dims%x_free ) = two * GRAD_L( : dims%x_free )
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) ) +                &
                    Z_l_coef( i, 2 ) / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) ) +                &
                    Z_l_coef( i, 2 ) / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = two * ( GRAD_L( i ) - Z_l( i ) - Z_u( i ) ) +     &
                    Z_l_coef( i, 2 ) / DIST_X_l( i ) +                         &
                    Z_u_coef( i, 2 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = two * ( GRAD_L( i ) - Z_u( i ) ) +                &
                    Z_u_coef( i, 2 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = two * ( GRAD_L( i ) - Z_u( i ) ) -                &
                    Z_u_coef( i, 2 ) / X( i )
                END DO

!  rhs for slack variables:
!    2 ( y - y_l - y_u ) + (C-C_l)^-1 s_l^2 + (C_u-C)^-1 s_u^2

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_l( i ) ) +          &
                    Y_l_coef( i, 2 ) / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_l( i ) - Y_u( i )) +&
                    Y_l_coef( i, 2 ) / DIST_C_l( i ) +                         &
                    Y_u_coef( i, 2 ) / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = two * ( Y( i ) - Y_u( i ) ) +          &
                    Y_u_coef( i, 2 ) / DIST_C_u( i )
                END DO

!  rhs for constraint infeasibilities: 2 ( A x - S c )

                RHS( dims%y_s : dims%y_e ) = two * C_RES( : dims%c_u_end )

!  Taylor case

              ELSE

!  rhs for problem variables: (X-X_l)^-1 r_l^2 + (X_u-X)^-1 r_u^2 )

                RHS( : dims%x_free ) = zero
                DO i = dims%x_free + 1, dims%x_l_start - 1
                  RHS( i ) = Z_l_coef( i, 2 ) / X( i )
                END DO
                DO i = dims%x_l_start, dims%x_u_start - 1
                  RHS( i ) = Z_l_coef( i, 2 ) / DIST_X_l( i )
                END DO
                DO i = dims%x_u_start, dims%x_l_end
                  RHS( i ) = Z_l_coef( i, 2 ) / DIST_X_l( i ) +                &
                             Z_u_coef( i, 2 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_l_end + 1, dims%x_u_end
                  RHS( i ) = Z_u_coef( i, 2 ) / DIST_X_u( i )
                END DO
                DO i = dims%x_u_end + 1, n
                  RHS( i ) = - Z_u_coef( i, 2 ) / X( i )
                END DO

!  rhs for slack variables: (C-C_l)^-1 s_l^2 + (C_u-C)^-1 s_u^2 )

                DO i = dims%c_l_start, dims%c_u_start - 1
                  RHS( dims%c_b + i ) = Y_l_coef( i, 2 ) / DIST_C_l( i )
                END DO
                DO i = dims%c_u_start, dims%c_l_end
                  RHS( dims%c_b + i ) = Y_l_coef( i, 2 ) / DIST_C_l( i ) +     &
                                        Y_u_coef( i, 2 ) / DIST_C_u( i )
                END DO
                DO i = dims%c_l_end + 1, dims%c_u_end
                  RHS( dims%c_b + i ) = Y_u_coef( i, 2 ) / DIST_C_u( i )
                END DO

!  rhs for constraint infeasibilities: A x - c

                RHS( dims%y_s : dims%y_e ) = zero
              END IF

!  for the 3rd order rhs

            ELSE

!  rhs for problem variables: (X-X_l)^-1 r_l^3 + (X_u-X)^-1 r_u^3

              RHS( : dims%x_free ) = zero
              DO i = dims%x_free + 1, dims%x_l_start - 1
                RHS( i ) = Z_l_coef( i, 3 ) / X( i )
              END DO
              DO i = dims%x_l_start, dims%x_u_start - 1
                RHS( i ) = Z_l_coef( i, 3 ) / DIST_X_l( i )
              END DO
              DO i = dims%x_u_start, dims%x_l_end
                RHS( i ) = Z_l_coef( i, 3 ) / DIST_X_l( i ) +                  &
                           Z_u_coef( i, 3 ) / DIST_X_u( i )
              END DO
              DO i = dims%x_l_end + 1, dims%x_u_end
                RHS( i ) = Z_u_coef( i, 3 ) / DIST_X_u( i )
              END DO
              DO i = dims%x_u_end + 1, n
                RHS( i ) = - Z_u_coef( i, 3 ) / X( i )
              END DO

!  rhs for slack variables: (C-C_l)^-1 s_l^3 + (C_u-C)^-1 s_u^3

                DO i = dims%c_l_start, dims%c_u_start - 1
                RHS( dims%c_b + i ) = Y_l_coef( i, 3 ) / DIST_C_l( i )
              END DO
              DO i = dims%c_u_start, dims%c_l_end
                RHS( dims%c_b + i ) = Y_l_coef( i, 3 ) / DIST_C_l( i ) +       &
                                      Y_u_coef( i, 3 ) / DIST_C_u( i )
              END DO
              DO i = dims%c_l_end + 1, dims%c_u_end
                RHS( dims%c_b + i ) = Y_u_coef( i, 3 ) / DIST_C_u( i )
              END DO

!  rhs for constraint infeasibilities: 0

              RHS( dims%y_s : dims%y_e ) = zero
            END IF


!  for the kth order coefficients

          ELSE

!  compute and store ( r_l^k, r_u^k, s_l^k, s_u^k )

            bik = BINOMIAL( 1, k )
            Z_l_coef( dims%x_free + 1 : dims%x_l_end, k ) =                    &
              - bik * X_coef( dims%x_free + 1 : dims%x_l_end, 1 )              &
                    * Z_l_coef( dims%x_free + 1 : dims%x_l_end, k - 1 )
            Z_u_coef( dims%x_u_start : n, k ) =                                &
                bik * X_coef( dims%x_u_start : n, 1 )                          &
                    * Z_u_coef( dims%x_u_start : n, k - 1 )
            Y_l_coef( dims%c_l_start : dims%c_l_end, k ) =                     &
              - bik * C_coef( dims%c_l_start : dims%c_l_end, 1 )               &
                    * Y_l_coef( dims%c_l_start : dims%c_l_end, k - 1 )
            Y_u_coef( dims%c_u_start : dims%c_u_end, k ) =                     &
                bik * C_coef( dims%c_u_start : dims%c_u_end, 1 )               &
                    * Y_u_coef( dims%c_u_start : dims%c_u_end, k - 1 )

            DO i = 2, k - 1
              bik = BINOMIAL( i, k )
              Z_l_coef( dims%x_free + 1 : dims%x_l_end, k ) =                  &
                Z_l_coef( dims%x_free + 1 : dims%x_l_end, k )                  &
                  - bik * X_coef( dims%x_free + 1 : dims%x_l_end, i )          &
                        * Z_l_coef( dims%x_free + 1 : dims%x_l_end, k - i )
              Z_u_coef( dims%x_u_start : n, k ) =                              &
                Z_u_coef( dims%x_u_start : n, k )                              &
                  + bik * X_coef( dims%x_u_start : n, i )                      &
                        * Z_u_coef( dims%x_u_start : n, k - i )
              Y_l_coef( dims%c_l_start : dims%c_l_end, k ) =                   &
                Y_l_coef( dims%c_l_start : dims%c_l_end, k )                   &
                  - bik * C_coef( dims%c_l_start : dims%c_l_end, i )           &
                        * Y_l_coef( dims%c_l_start : dims%c_l_end, k - i )
              Y_u_coef( dims%c_u_start : dims%c_u_end, k ) =                   &
                Y_u_coef( dims%c_u_start : dims%c_u_end, k )                   &
                  + bik * C_coef( dims%c_u_start : dims%c_u_end, i )           &
                        * Y_u_coef( dims%c_u_start : dims%c_u_end, k - i )
            END DO

!  rhs for problem variables:  (X-X_l)^-1 r_l^k ) + (X_u-X)^-1 r_u^k )

            RHS( : dims%x_free ) = zero
            DO i = dims%x_free + 1, dims%x_l_start - 1
              RHS( i ) = Z_l_coef( i, k ) / X( i )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              RHS( i ) = Z_l_coef( i, k ) / DIST_X_l( i )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              RHS( i ) = Z_l_coef( i, k ) / DIST_X_l( i )                      &
                       + Z_u_coef( i, k ) / DIST_X_u( i )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              RHS( i ) = Z_u_coef( i, k ) / DIST_X_u( i )
            END DO
            DO i = dims%x_u_end + 1, n
              RHS( i ) = - Z_u_coef( i, k ) / X( i )
            END DO

!  rhs for slack variables:  (C-C_l)^-1 s_l^k ) + (C_u-C)^-1 s_u^k )

            DO i = dims%c_l_start, dims%c_u_start - 1
              RHS( dims%c_b + i ) = Y_l_coef( i, k ) / DIST_C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              RHS( dims%c_b + i ) = Y_l_coef( i, k ) / DIST_C_l( i )           &
                                  + Y_u_coef( i, k ) / DIST_C_u( i )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              RHS( dims%c_b + i ) = Y_u_coef( i, k ) / DIST_C_u( i )
            END DO

!  rhs for constraint infeasibilities: 0

            RHS( dims%y_s : dims%y_e ) = zero
          END IF

!  rhs for Ao x^k - r_0 = 0: 0

          RHS( dims%r_s : dims%r_e  ) = zero

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' RHS_x ', RHS( dims%x_s : dims%x_e )
            IF ( m > 0 )                                                     &
              WRITE( out, 2100 ) prefix, ' RHS_y ', RHS( dims%y_s : dims%y_e )
          END IF

! :::::::::::::::::::::::::::::::::::
! 3b. Compute the series coefficients
! :::::::::::::::::::::::::::::::::::

!   solve ( weight I + D     A    Ao^T  ) ( x^k )
!         (               E -S          ) ( c^k ) = rhs
!         (     A        -S  0          ) (-y^k )
!         (     Ao              -W^{-1} ) ( r^k )

          IF ( printw ) THEN
            IF ( puiseux ) THEN
              WRITE( out, "( A, ' ........... compute ', I0, A2,               &
             &           ' order Puiseux coefficients  ........... ' )" )      &
                prefix, k, STRING_ordinal( k )
            ELSE
              WRITE( out, "( A, ' ............ compute ', I0, A2,              &
             &           ' order Taylor coefficients  ............ ' )" )      &
                prefix, k, STRING_ordinal( k )
            END IF
          END IF

!  use a direct method

          IF ( printd ) THEN
            WRITE( out, 2120 ) prefix, ' row ', K_sls%row( : K_sls%ne )
            WRITE( out, 2120 ) prefix, ' col ', K_sls%col( : K_sls%ne )
            WRITE( out, 2100 ) prefix, ' K ', K_sls%val( : K_sls%ne )
            WRITE( out, 2100 ) prefix, ' RHS ', RHS( : K_sls%n )
          END IF

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SLS_solve( K_sls, RHS, SLS_data, SLS_control, inform%SLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          time_solve = time_solve + time_now - time_record
          clock_solve = clock_solve + clock_now - clock_record

          inform%status = inform%status
          IF ( inform%status /= GALAHAD_ok ) GO TO 600

          IF ( printd ) THEN
!           WRITE( out, 2100 ) prefix, ' SOL ', RHS( : K_sls%n )
            WRITE( out, 2100 ) prefix, ' SOL_x ', RHS( dims%x_s : dims%x_e )
            IF ( m > 0 )                                                       &
              WRITE( out, 2100 ) prefix, ' SOL_y ', RHS( dims%y_s : dims%y_e )
          END IF

!  if the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely

          IF ( inform%SLS_inform%backward_error_1 > merit ) THEN

!  it didn't. We might have run out of options ...

            IF ( maxpiv ) THEN
              inform%status = GALAHAD_error_ill_conditioned ; GO TO 600

!  ... or we can increase the pivot tolerance

            ELSE IF ( SLS_control%relative_pivot_tolerance                     &
                      < relative_pivot_default ) THEN
              pivot_tol = relative_pivot_default
              min_pivot_tol = relative_pivot_default
              maxpiv = .FALSE.
              SLS_control%relative_pivot_tolerance = pivot_tol
              SLS_control%minimum_pivot_tolerance = min_pivot_tol
              IF ( printi ) WRITE( out,                                        &
                "( A, '    ** Pivot tolerance increased to', ES11.4 )" )       &
                prefix, pivot_tol
            ELSE
              pivot_tol = half
              min_pivot_tol = half
              maxpiv = .TRUE.
              SLS_control%relative_pivot_tolerance = pivot_tol
              SLS_control%minimum_pivot_tolerance = min_pivot_tol
              IF ( printi ) WRITE( out,                                        &
                "( A, '    ** Pivot tolerance increased to', ES11.4 )" )       &
                prefix, pivot_tol
            END IF
            alpha = zero ; nbact = 0

            GO TO 200
          END IF

!  record ( x^k, c^k, y^k )

          X_coef( : n, k ) = RHS( dims%x_s : dims%x_e )
          C_coef( dims%c_l_start : dims%c_u_end, k ) =                         &
            RHS( dims%c_s : dims%c_e )
          Y_coef( : m, k ) = - RHS( dims%y_s : dims%y_e )

!  compute ( z_l^k, z_u^k, y_l^k, y_u^k ) via

!     z_l^k = (X-X_l)^-1 [ r_l^k - Z_l x^k ]
!     z_u^k = (X_u-X)^-1 [ r_u^k + Z_u x^k ]
!     y_l^k = (C-C_l)^-1 [ s_l^k - Y_l c^k ]
!   & y_u^k = (C_u-C)^-1 [ s_u^k + Y_u c^k ]

          DO i = dims%x_free + 1, dims%x_l_start - 1
            Z_l_coef( i, k ) =                                                 &
             ( Z_l_coef( i, k ) - Z_l( i ) * X_coef( i, k ) ) / X( i )
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            Z_l_coef( i, k ) =                                                 &
             ( Z_l_coef( i, k ) - Z_l( i ) * X_coef( i, k ) ) / DIST_X_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            Z_u_coef( i, k ) =                                                 &
              ( Z_u_coef( i, k ) + Z_u( i ) * X_coef( i, k ) ) / DIST_X_u( i )
          END DO

          DO i = dims%x_u_end + 1, n
            Z_u_coef( i, k ) =                                                 &
              - ( Z_u_coef( i, k ) + Z_u( i ) * X_coef( i, k ) ) / X( i )
          END DO

          DO i = dims%c_l_start, dims%c_l_end
            Y_l_coef( i, k ) =                                                 &
              ( Y_l_coef( i, k ) - Y_l( i ) * C_coef( i, k ) ) / DIST_C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            Y_u_coef( i, k ) =                                                 &
              ( Y_u_coef( i, k ) + Y_u( i ) * C_coef( i, k ) ) / DIST_C_u( i )
          END DO
        END DO

!  finally, scale the coefficients v^k <- (-1)^k v^k / k!

        co = one
        DO k = 1, order
          co = - co / REAL( k, KIND = rp_ )
          X_coef( : , k ) = co * X_coef( : , k )
          C_coef( : , k ) = co * C_coef( : , k )
          Y_coef( : , k ) = co * Y_coef( : , k )
          Z_l_coef( : , k ) = co * Z_l_coef( : , k )
          Z_u_coef( : , k ) = co * Z_u_coef( : , k )
          Y_l_coef( : , k ) = co * Y_l_coef( : , k )
          Y_u_coef( : , k ) = co * Y_u_coef( : , k )
        END DO

        IF ( printw ) THEN
          WRITE( out, "( A, '   k   ||X^k||   ||C^k||   ||Y^k||',              &
        &               ' ||Z_l^k|| ||Z_u^k|| ||Y_l^k|| ||Y_u^k||' )" ) prefix
          DO k = 0, order
            char_x = MAXVAL_ABS( X_coef( : , k ) )
            char_c = MAXVAL_ABS( C_coef( : , k ) )
            char_y = MAXVAL_ABS( Y_coef( : , k ) )
            char_z_l = MAXVAL_ABS( Z_l_coef( : , k ) )
            char_z_u = MAXVAL_ABS( Z_u_coef( : , k ) )
            char_y_l = MAXVAL_ABS( Y_l_coef( : , k ) )
            char_y_u = MAXVAL_ABS( Y_u_coef( : , k ) )
            WRITE( out, "( A, 1X, A, I1, 7A10 )" ) prefix, arc, k,             &
              char_x, char_c, char_y, char_z_l, char_z_u, char_y_l, char_y_u
          END DO
        END IF

!  Additionally, if the Taylor Zhang arc is not being used, we need to include
!  this as a precaution to guarantee convergence

        guarantee = arc /= 'Zh' .OR. puiseux .OR.                              &
                    ( order > 1 .AND. .NOT. control%every_order )

        IF ( guarantee ) THEN

!  Set up the right-hand-side vector

!  record rhs = ( h^k + (X-X_l)^-1 r_l^k + (X_u-X)^-1 r_u^k )
!               ( d^k + (C-C_l)^-1 s_l^k + (C_u-C)^-1 s_u^k )
!               (                   a^k                     )
!               (                    0                      )

          IF ( printd ) WRITE( out, 2100 )                                     &
            prefix, ' GRAD_L', GRAD_L( dims%x_s : dims%x_e )

!  compute and store ( r_l^1, r_u^1, s_l^1, s_u^1 )

          DO i = dims%x_free + 1, dims%x_l_end
            DZ_l_zh( i ) = - mu + ( X( i ) - X_l( i ) ) * Z_l( i )
          END DO
          DO i = dims%x_u_start, n
            DZ_u_zh( i ) =   mu + ( X_u( i ) - X( i ) ) * Z_u( i )
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            DY_l_zh( i ) = - mu + ( C( i ) - C_l( i ) ) * Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            DY_u_zh( i ) =   mu + ( C_u( i ) - C( i ) ) * Y_u( i )
          END DO

!  rhs for problem variables: g + Hx - A^Ty - mu (X-X_l)^-1 e + mu (X_u-X)^-1 e

          RHS( : dims%x_free ) = GRAD_L( : dims%x_free )
          DO i = dims%x_free + 1, dims%x_l_start - 1
            RHS( i ) = GRAD_L( i ) - mu / X( i )
          END DO
          DO i = dims%x_l_start, dims%x_u_start - 1
            RHS( i ) = GRAD_L( i ) - mu / DIST_X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            RHS( i ) = GRAD_L( i ) - mu / DIST_X_l( i ) + mu / DIST_X_u( i )
          END DO
          DO i = dims%x_l_end + 1, dims%x_u_end
            RHS( i ) = GRAD_L( i ) + mu / DIST_X_u( i )
          END DO
          DO i = dims%x_u_end + 1, n
            RHS( i ) = GRAD_L( i ) - mu / X( i )
          END DO

!  rhs for slack variables: y - mu (C-C_l)^-1 e + mu (C_u-C)^-1 e

          DO i = dims%c_l_start, dims%c_u_start - 1
            RHS( dims%c_b + i ) = Y( i ) - mu / DIST_C_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            RHS( dims%c_b + i ) = Y( i ) - mu / DIST_C_l( i )                  &
                                         + mu / DIST_C_u( i )
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            RHS( dims%c_b + i ) = Y( i ) + mu / DIST_C_u( i )
          END DO

!  rhs for constraint infeasibilities: A x - S c

          IF ( m > 0 ) THEN
            RHS( dims%y_s : dims%y_e ) = C_RES( : dims%c_u_end )

            C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
            C_RES( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
            CALL CLLS_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr,              &
                          n, X, '+ ' )
            inform%primal_infeasibility = MAXVAL( ABS( C_RES ) )
          END IF

!  rhs for Ao x^k - r_0 = 0: 0

          RHS( dims%r_s : dims%r_e  ) = zero

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' RHS_x ', RHS( dims%x_s : dims%x_e )
            IF ( m > 0 )                                                       &
              WRITE( out, 2100 ) prefix, ' RHS_y ', RHS( dims%y_s : dims%y_e )
          END IF

! Compute the coefficients

!   solve ( weight I + D     A    Ao^T  ) ( x^k )
!         (               E -S          ) ( c^k ) = rhs
!         (     A        -S  0          ) (-y^k )
!         (     Ao              -W^{-1} ) ( r^k )

          IF ( printw ) WRITE( out, "( A, ' ............... compute',          &
         &        ' Zhang-Taylor coefficients  ............... ' )" )  prefix

!  use a direct method

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SLS_solve( K_sls, RHS, SLS_data, SLS_control,                   &
                          inform%SLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          time_solve = time_solve + time_now - time_record
          clock_solve = clock_solve + clock_now - clock_record

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' SOL_x ', RHS( dims%x_s : dims%x_e )
            IF ( m > 0 )                                                       &
              WRITE( out, 2100 ) prefix, ' SOL_y ', RHS( dims%y_s : dims%y_e )
          END IF

          inform%status = inform%status
          IF ( inform%status /= GALAHAD_ok ) GO TO 600


!  if the residual of the linear system is larger than the current
!  optimality residual, no further progress is likely

          IF ( inform%SLS_inform%backward_error_1 > merit ) THEN

!  it didn't. We might have run out of options ...

            IF ( maxpiv ) THEN
              inform%status = GALAHAD_error_ill_conditioned ; GO TO 600

!  ... or we can increase the pivot tolerance

            ELSE IF ( SLS_control%relative_pivot_tolerance                     &
                      < relative_pivot_default ) THEN
              pivot_tol = relative_pivot_default
              min_pivot_tol = relative_pivot_default
              maxpiv = .FALSE.
              SLS_control%relative_pivot_tolerance = pivot_tol
              SLS_control%minimum_pivot_tolerance = min_pivot_tol
              IF ( printi ) WRITE( out,                                        &
                "( A, '    ** Pivot tolerance increased to', ES11.4 )" )       &
                prefix, pivot_tol
            ELSE
              pivot_tol = half
              min_pivot_tol = half
              maxpiv = .TRUE.
              SLS_control%relative_pivot_tolerance = pivot_tol
              SLS_control%minimum_pivot_tolerance = min_pivot_tol
              IF ( printi ) WRITE( out,                                        &
                "( A, '    ** Pivot tolerance increased to', ES11.4 )" )       &
                prefix, pivot_tol
            END IF
            alpha = zero ; nbact = 0

            GO TO 200
          END IF

!  record ( x^k, c^k, y^k )

          DX_zh( : n ) = - RHS( dims%x_s : dims%x_e )
          DC_zh( dims%c_l_start : dims%c_u_end ) =                             &
            - RHS( dims%c_s : dims%c_e )
          DY_zh( : m ) = RHS( dims%y_s : dims%y_e )

!  compute ( z_l^k, z_u^k, y_l^k, y_u^k ) via

!     z_l^k = (X-X_l)^-1 [ r_l^k - Z_l x^k ]
!     z_u^k = (X_u-X)^-1 [ r_u^k + Z_u x^k ]
!     y_l^k = (C-C_l)^-1 [ s_l^k - Y_l c^k ]
!   & y_u^k = (C_u-C)^-1 [ s_u^k + Y_u c^k ]


          DO i = dims%x_free + 1, dims%x_l_start - 1
            DZ_l_zh( i ) =                                                     &
             - ( DZ_l_zh( i ) + Z_l( i ) * DX_zh( i ) ) / X( i )
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            DZ_l_zh( i ) =                                                     &
             - ( DZ_l_zh( i ) + Z_l( i ) * DX_zh( i ) ) / DIST_X_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            DZ_u_zh( i ) =                                                     &
              - ( DZ_u_zh( i ) - Z_u( i ) * DX_zh( i ) ) / DIST_X_u( i )
          END DO

          DO i = dims%x_u_end + 1, n
            DZ_u_zh( i ) =                                                     &
                ( DZ_u_zh( i ) - Z_u( i ) * DX_zh( i ) ) / X( i )
          END DO

          DO i = dims%c_l_start, dims%c_l_end
            DY_l_zh( i ) =                                                     &
              - ( DY_l_zh( i ) + Y_l( i ) * DC_zh( i ) ) / DIST_C_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            DY_u_zh( i ) =                                                     &
              - ( DY_u_zh( i ) - Y_u( i ) * DC_zh( i ) ) / DIST_C_u( i )
          END DO

          IF ( printw ) THEN
            char_x = MAXVAL_ABS( DX_zh( : ) )
            char_c = MAXVAL_ABS( DC_zh( : ) )
            char_y = MAXVAL_ABS( DY_zh( : ) )
            char_z_l = MAXVAL_ABS( DZ_l_zh( : ) )
            char_z_u = MAXVAL_ABS( DZ_u_zh( : ) )
            char_y_l = MAXVAL_ABS( DY_l_zh( : ) )
            char_y_u = MAXVAL_ABS( DY_u_zh( : ) )
            WRITE( out, "( A, ' ZT', I1, 7A10 )" ) prefix, 1,                  &
              char_x, char_c, char_y, char_z_l, char_z_u, char_y_l, char_y_u
          END IF
        END IF

        IF ( printt ) WRITE( out,                                              &
           "( A, ' time for solves = ', F0.2 ) " ) prefix, clock_solve
        inform%time%solve = inform%time%solve + REAL( time_solve, rp_ )
        inform%time%clock_solve = inform%time%clock_solve + clock_solve

        IF ( printw ) WRITE( out,                                              &
             "( A, ' ............... arc computed ............... ' )" ) prefix

!  =======
!  STEP 4:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Line search   -*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

        IF ( printw ) WRITE( out,                                              &
             "( A, ' .............. get steplength  .............. ' )" ) prefix

!  if a convergence guarantee is required, try the Taylor Zhang arc

        IF ( guarantee ) THEN

!  find the largest alpha in [0,1] for which

!     v_1(alpha) = v + alpha v^1

!  lies in a given wide neighbourhood of the central path

          CALL CLLS_compute_lmaxstep( dims, n, m, nbnds, X, X_l, X_u, DX_zh,   &
                                     C, C_l, C_u, DC_zh, Y_l, Y_u, DY_l_zh,    &
                                     DY_u_zh, Z_l, Z_u, DZ_l_zh, DZ_u_zh,      &
                                     gamma_c, gamma_f, res_primal_dual,        &
                                     alpha_max, inform%status )

!  check that resulting alpha is not too small

          IF ( inform%status == GALAHAD_error_tiny_step ) GO TO 600

! :::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Use a safeguarded arc-search, starting from alpha_max
! :::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  record the initial slope along the search arc

          slope = - ( merit - mu * rnbnds )

!  define an interval [alhpa_l,alpha_u] containing the required stepsize

          alpha_max = MIN( alpha_max, one )
          alpha_l = zero ; alpha_u = alpha_max ; alpha = alpha_u

          IF ( printw ) WRITE( out, "( /, A, ' ***  Linesearch       ',        &
         &  ' step       trial value     model value ', /, A, 16X, 3ES16.8 )" )&
              prefix, prefix, zero, merit, merit

!  backtracking loop

          nbact = 0
          DO

!  once the interval is small enough, accept the lower bound as the required
!  step so long_ as this step is not zero

            IF ( alpha_u - alpha_l <= stop_alpha .AND. alpha_l > zero ) THEN
              alpha = alpha_l
              EXIT
            END IF
            IF ( alpha_u <= epsmch ) THEN
              IF ( inform%iter - 1 > muzero_fixed ) THEN
                inform%status = GALAHAD_error_tiny_step ; GO TO 600
              ELSE
                muzero_fixed = inform%iter - 2
                EXIT
              END IF
            END IF

!  the merit value of an acceptable point must be smaller than a linear model

            merit_model = merit + alpha * eta * slope

!  compute the complementarity at the new point on the arc

            comp = zero
            DO i = dims%x_free + 1, dims%x_l_end
              comp = comp + ( Z_l( i ) + alpha * DZ_l_zh( i ) ) *              &
                            ( X( i ) + alpha * DX_zh( i ) - X_l( i ) )
            END DO
            DO i = dims%x_u_start, n
              comp = comp - ( Z_u( i ) + alpha * DZ_u_zh( i ) ) *              &
                            ( X_u( i ) - X( i ) - alpha * DX_zh( i ) )
            END DO

            DO i = dims%c_l_start, dims%c_l_end
              comp = comp + ( Y_l( i ) + alpha * DY_l_zh( i ) ) *              &
                            ( C( i ) + alpha * DC_zh( i ) - C_l( i ) )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              comp = comp - ( Y_u( i ) + alpha * DY_u_zh( i ) ) *              &
                            ( C_u( i ) - C( i ) - alpha * DC_zh( i ) )
            END DO

!  evaluate the merit function at the new point

            one_minus_alpha = one - alpha
            merit_trial = comp + one_minus_alpha * tau * res_primal_dual
            IF ( printw ) WRITE( out, "( A, 16X, 3ES16.8 )" )                  &
              prefix, alpha, merit_trial, merit_model

!  check to see if the Amijo criterion is satisfied.

            IF ( merit_trial <= merit_model ) THEN

!  if the current arc length is alpha_max, accept this as the required step

              IF ( alpha == alpha_max ) EXIT

!  increase the lower bound

              alpha_l = alpha
              alpha = half * ( alpha + alpha_u )

!  the current alpha is unacceptable ; reduce the upper bound

            ELSE
              alpha_u = alpha
              alpha = half * ( alpha + alpha_l )
            END IF
            nbact = nbact + 1
          END DO
          opt_alpha_guarantee = alpha
          opt_merit_guarantee = merit_trial
          IF ( printp ) WRITE( out, "( A, '      Zhang step, merit =',         &
         &            2ES24.16 )" ) prefix, alpha, merit_trial
        END IF

!  record the initial slope along the search arc

        IF ( arc == 'ZP' ) THEN
          slope = - two * ( merit - mu * rnbnds ) + tau * res_primal_dual
        ELSE IF ( puiseux ) THEN
          slope = - two * ( merit - mu * rnbnds )
        ELSE
          slope = - ( merit - mu * rnbnds )
        END IF
        IF ( printw ) WRITE( out, "( A, '  value and slope = ', 1P, 2D12.4)")  &
          prefix, merit, slope

!  loop over arcs of increasing order

        IF ( control%every_order .OR. order <= 0 ) THEN
          sorder = 1
        ELSE
          sorder = order
        END IF

  step: DO iorder = sorder, order

          CALL CLLS_compute_v_alpha( dims, n, m, iorder, X_coef, C_coef,       &
                           Y_coef, Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,     &
                           X, X_l, X_u, Z_l, Z_u, Y, Y_l, Y_u, C,              &
                           C_l, C_u, one, comp )

!  find the largest alpha in [0,1] for which

!     v_l(alpha) = v + sum_k=1^l [ (-1)^k v^k / k! ] alpha^k

!  lies in a given wide neighbourhood of the central path

!         IF ( .TRUE. ) THEN
          IF ( .FALSE. ) THEN    ! serial step
          CALL CLLS_compute_stepsize( dims, n, m, nbnds, iorder,               &
                                      puiseux .AND. arc /= 'ZP',               &
                                      X_coef, C_coef, Y_coef, Y_l_coef,        &
                                      Y_u_coef, Z_l_coef, Z_u_coef,            &
                                      X, X_l, X_u, Z_l, Z_u,                   &
                                      Y, Y_l, Y_u, C, C_l, C_u,                &
                                      gamma_c, gamma_f, res_primal_dual,       &
                                      alpha_max, slknes, control%prefix,       &
                                      control%out, print_level, inform%status )
          ELSE
          CALL CLLS_compute_pmaxstep( dims, n, m, nbnds, iorder,               &
                                      puiseux .AND. arc /= 'ZP',               &
                                      X_coef, C_coef, Y_l_coef, Y_u_coef,      &
                                      Z_l_coef, Z_u_coef, X_l, X_u, C_l, C_u,  &
                                      CS_coef, COEF, ROOTS, gamma_c, gamma_f,  &
                                      res_primal_dual, alpha_max,              &
                                      control%ROOTS_control, inform%threads,   &
                                      inform%status, inform%ROOTS_inform,      &
                                      ROOTS_data )

!  compute the best point on the arc and its complementarity

          CALL CLLS_compute_v_alpha( dims, n, m, iorder, X_coef, C_coef,Y_coef,&
                                     Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,   &
                                     X, X_l, X_u, Z_l, Z_u, Y, Y_l, Y_u, C,    &
                                     C_l, C_u, alpha_max, slknes )
          END IF

!  check that resulting alpha is not too small

          IF ( inform%status == GALAHAD_error_tiny_step ) THEN
            OPT_alpha( iorder ) = zero
            OPT_merit( iorder ) = merit
            IF ( printp ) WRITE( out, "( A, '  order ', I3, ' step, merit =',  &
           &                 2ES24.16 )" ) prefix, iorder, zero, merit
            CYCLE
          END IF

! :::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Use a safeguarded arc-search, starting from alpha_max
! :::::::::::::::::::::::::::::::::::::::::::::::::::::::

!  define an interval [alhpa_l,alpha_u] containing the required stepsize

          alpha_l = zero ; alpha_u = alpha_max ; alpha = alpha_max

          IF ( printw ) WRITE( out, "( /, A, ' ***  Linesearch       ',        &
         &  ' step       trial value     model value ', /, A, 16X, 3ES16.8 )" )&
              prefix, prefix, zero, merit, merit

!  backtracking loop

          nbact = 0
          DO

!  once the interval is small enough, accept the lower bound as the required
!  step so long_ as this step is not zero

            IF ( alpha_u - alpha_l <= stop_alpha .AND. alpha_l > zero ) THEN
              alpha = alpha_l
              EXIT
            END IF
            IF ( alpha_u <= epsmch ) THEN
              IF ( inform%iter - 1 > muzero_fixed ) THEN
                inform%status = GALAHAD_error_tiny_step
                OPT_alpha( iorder ) = zero
                OPT_merit( iorder ) = merit
                CYCLE step
              ELSE
                muzero_fixed = inform%iter - 2
                EXIT
              END IF
            END IF

!  the merit value of an acceptable point must be smaller than a linear model

            merit_model = merit + alpha * eta * slope

!  compute the complementarity at the new point on the arc

            CALL CLLS_compute_v_alpha( dims, n, m, iorder, X_coef, C_coef,     &
                             Y_coef, Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,   &
                             X, X_l, X_u, Z_l, Z_u, Y, Y_l, Y_u, C,            &
                             C_l, C_u, alpha, comp )

!  evaluate the merit function at the new point

            one_minus_alpha = one - alpha
            one_minus_alpha = one - alpha
            IF ( puiseux .AND. arc /= 'ZP' ) THEN
              merit_trial = comp + one_minus_alpha ** 2 * tau * res_primal_dual
            ELSE
              merit_trial = comp + one_minus_alpha * tau * res_primal_dual
            END IF
            IF ( printw ) WRITE( out, "( A, 16X, 3ES16.8 )" )                  &
              prefix, alpha, merit_trial, merit_model

!  check to see if the Amijo criterion is satisfied.

            IF ( merit_trial <= merit_model ) THEN

!  if the current arc length is alpha_max, accept this as the required step

              IF ( alpha == alpha_max ) EXIT

!  increase the lower bound

              alpha_l = alpha
              alpha = half * ( alpha + alpha_u )

!  the current alpha is unacceptable ; reduce the upper bound

            ELSE
              alpha_u = alpha
              alpha = half * ( alpha + alpha_l )
            END IF
            nbact = nbact + 1
          END DO
          OPT_alpha( iorder ) = alpha
          OPT_merit( iorder ) = merit_trial
          IF ( printp ) WRITE( out, "( A, '  order ', I3, ' step, merit =',    &
         &                     2ES24.16 )" ) prefix, iorder, alpha, merit_trial
        END DO step


!  if the complementarity is small enough, try a pounce to the solution

        IF ( mu <= control%mu_pounce .AND. alpha < one ) THEN

!  evaluate the pounce

          CALL CLLS_compute_v_alpha( dims, n, m, order, X_coef, C_coef,        &
                            Y_coef, Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,    &
                            X, X_l, X_u, Z_l, Z_u, Y, Y_l, Y_u, C,             &
                            C_l, C_u, one, comp )

!  project the pounce into the feasible region

          DO i = dims%x_free + 1, dims%x_l_end
            X( i ) = MAX( X( i ), X_l( i ) )
            Z_l( i ) = MAX( Z_l( i ), zero )
          END DO

          DO i = dims%x_u_start, n
            X( i ) = MIN( X( i ), X_u( i ) )
            Z_u( i ) = MIN( Z_u( i ), zero )
          END DO

          DO i = dims%c_l_start, dims%c_l_end
            C( i ) = MAX( C( i ), C_l( i ) )
            Y_l( i ) = MAX( Y_l( i ), zero )
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            C( i ) = MIN( C( i ), C_u( i ) )
            Y_u( i ) = MIN( Y_u( i ), zero )
          END DO

!  update the distances to the bounds

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

!  compute the constraint residuals

          IF ( m > 0 ) THEN
            C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
            C_RES( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
            CALL CLLS_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr,              &
                          n, X, '+ ' )
            inform%primal_infeasibility = MAXVAL( ABS( C_RES ) )
            IF ( printw ) WRITE( out, "( A, '  constraint residual ', ES12.4)")&
              prefix, inform%primal_infeasibility
          END IF

!  compute the residual

          R( : o ) = - B( : o )
          CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )

!  compute the gradient of the Lagrangian function

          CALL CLLS_Lagrangian_gradient( dims, n, o, m, weight,                &
                                         X, R, Y, Y_l, Y_u, Z_l, Z_u,          &
                                         Ao_ne, Ao_val, Ao_row, Ao_ptr,        &
                                         A_ne, A_val, A_col, A_ptr,            &
                                         DIST_X_l, DIST_X_u,                   &
                                         DIST_C_l, DIST_C_u,                   &
                                         GRAD_L( dims%x_s : dims%x_e ),        &
                                         control%getdua, dufeas, W )

!  evaluate the primal and dual infeasibility and merit function

          merit = CLLS_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,      &
                                    DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,    &
                                    GRAD_L( dims%x_s : dims%x_e ), C_RES,      &
                                    tau, res_primal, inform%dual_infeasibility,&
                                    res_primal_dual, res_cs )

!  compute the complementary slackness, and the min/max components
!  of the primal/dual infeasibilities

          slknes = DOT_PRODUCT( X( dims%x_free + 1 : dims%x_l_start - 1 ),     &
                                Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ) +&
                   DOT_PRODUCT( DIST_X_l( dims%x_l_start : dims%x_l_end ),     &
                                Z_l( dims%x_l_start : dims%x_l_end ) ) -       &
                   DOT_PRODUCT( DIST_X_u( dims%x_u_start : dims%x_u_end ),     &
                                Z_u( dims%x_u_start : dims%x_u_end ) ) +       &
                   DOT_PRODUCT( X( dims%x_u_end + 1 : n ),                     &
                              Z_u( dims%x_u_end + 1 : n ) ) +                  &
                   DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),     &
                                Y_l( dims%c_l_start : dims%c_l_end ) ) -       &
                   DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),     &
                                Y_u( dims%c_u_start : dims%c_u_end ) )

          IF ( nbnds > 0 ) THEN
            slknes = slknes / rnbnds
          ELSE
            slknes = zero
          END IF

!  test for optimality

!write(6,*) inform%primal_infeasibility, stop_p
!write(6,*) inform%dual_infeasibility, stop_d
!write(6,*) slknes, stop_c

          IF ( inform%primal_infeasibility <= stop_p .AND.                     &
               inform%dual_infeasibility <= stop_d .AND.                       &
               slknes <= stop_c ) THEN

!  checkpoint

            CALL CPU_TIME( time_record )
            CALL CHECKPOINT( inform%iter, time_record - time_start,            &
                             MAX( inform%primal_infeasibility,                 &
                                 inform%dual_infeasibility, slknes ),          &
                             inform%checkpointsIter, inform%checkpointsTime,   &
                             1_ip_, 16_ip_ )

!  if optimal, compute the objective residual ...

            R = - B
            CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )

!  ... and the objective function

            IF ( present_weight ) THEN
              inform%obj = half * DOT_PRODUCT( R, W * R )
            ELSE
              inform%obj = half * DOT_PRODUCT( R, R )
            END IF
            IF ( weight > zero )                                               &
              inform%obj = inform%obj + half * weight * DOT_PRODUCT( X, X )

            IF ( .NOT. inform%feasible ) THEN
              IF ( printi ) WRITE( out, 2070 ) prefix
              inform%feasible = .TRUE.
            END IF

!  print a summary of the final iteration

            CALL CLOCK_TIME( clock_now )
            IF ( printi ) THEN
              IF ( printt .OR. ( printi .AND.                                  &
                 inform%iter == start_print ) ) WRITE( out, 2000 ) prefix
              WRITE( out, 2030 ) prefix, inform%iter, re,                      &
               inform%primal_infeasibility, inform%dual_infeasibility,         &
               slknes, inform%obj, one, mu, order, pui, arc, nbact,            &
               clock_now - clock_start
            END IF

            IF ( printd ) THEN
              WRITE( out, 2100 ) prefix, ' X ', X
              IF ( dims%x_free + 1 <= dims%x_l_end ) WRITE( out, 2100 )        &
                prefix,  ' Z_l ', Z_l( dims%x_free + 1 : dims%x_l_end )
              IF (  dims%x_u_start <= n ) WRITE( out, 2100 )                   &
                prefix, ' Z_u ', Z_u( dims%x_u_start :  n )
            END IF
            inform%status = GALAHAD_ok ; GO TO 600
          END IF

!  if the pounce failed, revert to the best point found in the linesearch

        END IF

!  accept the point that gives the largest merit function decrease

        IF ( control%every_order .AND. order > 0 ) THEN
          iorder_array = MINLOC( OPT_merit( : order ) )
          iorder = iorder_array( 1 )
          alpha = OPT_alpha( iorder )
          merit_trial = OPT_merit( order )
        ELSE IF ( .NOT. control%every_order .AND. order > 0 ) THEN
          iorder = order
          alpha = OPT_alpha( iorder )
          merit_trial = OPT_merit( order )
        ELSE
          iorder = 1
        END IF

!  ensure that if guaranteed convergence is required, the merit function
!  decrease is at least as good as that provided by the Zhang-Taylor step

        IF ( puiseux ) THEN
          pui = 'P'
        ELSE
          pui = 'T'
        END IF

        IF ( guarantee ) THEN
          IF ( order <= 0 .OR. opt_merit_guarantee < merit_trial ) THEN
            iorder = 0
            pui = 'T'
            arc = 'Zh'
            alpha = opt_alpha_guarantee
            merit_trial = opt_merit_guarantee
          END IF
        END IF

!  recover the point that gives the largest merit function decrease

        IF ( iorder > 0 ) THEN
          CALL CLLS_compute_v_alpha( dims, n, m, iorder, X_coef, C_coef,       &
                           Y_coef, Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,     &
                           X, X_l, X_u, Z_l, Z_u, Y, Y_l, Y_u, C,              &
                           C_l, C_u, alpha, comp )

        ELSE
          DO i = 1, n
            X( i ) = X_coef( i, 0 ) + alpha * DX_zh( i )
          END DO
          DO i = dims%x_free + 1, dims%x_l_end
            Z_l( i ) = Z_l_coef( i, 0 ) + alpha * DZ_l_zh( i )
          END DO
          DO i = dims%x_u_start, n
            Z_u( i ) = Z_u_coef( i, 0 ) + alpha * DZ_u_zh( i )
          END DO
          DO i = 1, m
            Y( i ) = Y_coef( i, 0 ) + alpha * DY_zh( i )
          END DO
          DO i = dims%c_l_start, dims%c_u_end
            C( i ) = C_coef( i, 0 ) + alpha * DC_zh( i )
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            Y_l( i ) = Y_l_coef( i, 0 ) + alpha * DY_l_zh( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            Y_u( i ) = Y_u_coef( i, 0 ) + alpha * DY_u_zh( i )
          END DO

          comp = zero
          DO i = dims%x_free + 1, dims%x_l_end
            comp = comp + ( X( i ) - X_l( i ) ) * Z_l( i )
          END DO
          DO i = dims%x_u_start, n
            comp = comp + ( X( i ) - X_u( i ) ) * Z_u( i )
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            comp = comp + ( C( i ) - C_l( i ) ) * Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            comp = comp + ( C( i ) - C_u( i ) ) * Y_u( i )
          END DO

        END IF

        inform%nbacts = inform%nbacts + nbact

!  update the distances to the bounds with some precaution against exterme
!  roundoff

        DO i = dims%x_l_start, dims%x_l_end
          IF ( X( i ) <= X_l( i ) ) X( i ) = X( i ) + ABS( X( i ) ) * epsmch
          DIST_X_l( i ) = X( i ) - X_l( i )
        END DO

        DO i = dims%x_u_start, dims%x_u_end
          IF ( X( i ) >= X_u( i ) ) X( i ) = X( i ) - ABS( X( i ) ) * epsmch
          DIST_X_u( i ) = X_u( i ) - X( i )
        END DO

        DO i = dims%c_l_start, dims%c_l_end
          IF ( C( i ) <= C_l( i ) ) C( i ) = C( i ) + ABS( C( i ) ) * epsmch
          DIST_C_l( i ) = C( i ) - C_l( i )
        END DO

        DO i = dims%c_u_start, dims%c_u_end
          IF ( C( i ) >= C_u( i ) ) C( i ) = C( i ) - ABS( C( i ) ) * epsmch
          DIST_C_u( i ) = C_u( i ) - C( i )
        END DO

!  =======
!  STEP 5:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*-*-*-*-   Book keeping  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  compute the constraint residuals

        IF ( m > 0 ) THEN
          C_RES( : dims%c_equality ) = - C_l( : dims%c_equality )
          C_RES( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
          CALL CLLS_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr,                &
                        n, X, '+ ' )
          inform%primal_infeasibility = MAXVAL( ABS( C_RES ) )
          IF ( printw ) WRITE( out, "( A, '  constraint residual ', ES12.4 )" )&
            prefix, inform%primal_infeasibility
!         WRITE( 6, "( ' rec, cal cres = ', 2ES12.4 )" )                       &
!           inform%primal_infeasibility, MAXVAL( ABS( C_RES ) )
        END IF

!  compute the residual

        R( : o ) = - B( : o )
        CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )

!  compute the gradient of the Lagrangian function

        CALL CLLS_Lagrangian_gradient( dims, n, o, m, weight,                  &
                                       X, R, Y, Y_l, Y_u, Z_l, Z_u,            &
                                       Ao_ne, Ao_val, Ao_row, Ao_ptr,          &
                                       A_ne, A_val, A_col, A_ptr,              &
                                       DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u, &
                                       GRAD_L( dims%x_s : dims%x_e ),          &
                                       control%getdua, dufeas, W )

!  update the values of the merit function, the gradient of the Lagrangian,
!  and the constraint residuals

!       GRAD_L( dims%x_s : dims%x_e ) = GRAD_L( dims%x_s : dims%x_e ) +        &
!         alpha * HX( dims%x_s : dims%x_e )

!       C_RES = one_minus_alpha * C_RES

!  update the norm of the constraint residual

!       inform%primal_infeasibility =one_minus_alpha*inform%primal_infeasibility

!  compute the objective residual ...

        R = - B
        CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )

!  ... and the objective function

        IF ( present_weight ) THEN
          inform%obj = half * DOT_PRODUCT( R, W * R )
        ELSE
          inform%obj = half * DOT_PRODUCT( R, R )
        END IF
        IF ( weight > zero )                                                   &
          inform%obj = inform%obj + half * weight * DOT_PRODUCT( X, X )

!  evaluate the merit function

        merit = CLLS_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,        &
                                  DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,      &
                                  GRAD_L( dims%x_s : dims%x_e ), C_RES,        &
                                  tau, res_primal, inform%dual_infeasibility,  &
                                  res_primal_dual, res_cs )

!  compute the complementary slackness, and the min/max components
!  of the primal/dual infeasibilities

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
          slknes_x = slknes_x / rnbnds_x
        ELSE
          slknes_x = zero
        END IF

        IF ( nbnds_c > 0 ) THEN
          slknes_c = slknes_c / rnbnds_c
        ELSE
          slknes_c = zero
        END IF
        IF ( nbnds > 0 ) THEN
          slknes = slknes / rnbnds
          inform%complementary_slackness = slknes
        ELSE
          slknes = zero
        END IF

!  checkpoint

        CALL CPU_TIME( time_record )
        CALL CHECKPOINT( inform%iter, time_record - time_start,                &
                         MAX( inform%primal_infeasibility,                     &
                              inform%dual_infeasibility, slknes ),             &
                         inform%checkpointsIter, inform%checkpointsTime,       &
                         1_ip_, 16_ip_ )

!  test for optimality

        IF ( inform%primal_infeasibility <= stop_p .AND.                       &
             inform%dual_infeasibility <= stop_d .AND.                         &
             slknes <= stop_c ) THEN

!write(6,*) inform%primal_infeasibility, stop_p
!write(6,*) inform%dual_infeasibility, stop_d
!write(6,*) slknes, stop_c

          IF ( .NOT. inform%feasible ) THEN
            IF ( printi ) WRITE( out, 2070 ) prefix
            inform%feasible = .TRUE.
          END IF

!  print a summary of the final iteration

          CALL CLOCK_TIME( clock_now )
          IF ( printi ) THEN
            IF ( printt .OR. ( printi .AND.                                    &
               inform%iter == start_print ) ) WRITE( out, 2000 ) prefix
            WRITE( out, 2030 ) prefix, inform%iter, re,                        &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             slknes, inform%obj, one, mu, order, pui, arc, nbact,              &
             clock_now - clock_start
          END IF

          IF ( printd ) THEN
            WRITE( out, 2100 ) prefix, ' X ', X
            IF ( dims%x_free + 1 <= dims%x_l_end ) WRITE( out, 2100 )          &
              prefix,  ' Z_l ', Z_l( dims%x_free + 1 : dims%x_l_end )
            IF (  dims%x_u_start <= n ) WRITE( out, 2100 )                     &
              prefix, ' Z_u ', Z_u( dims%x_u_start :  n )
          END IF
          inform%status = GALAHAD_ok ; GO TO 600
        END IF

        IF ( printw .AND. nbnds > 0 ) WRITE( out, 2130 )                       &
          prefix, slknes, prefix, slknes_x, prefix, slknes_c, prefix, slkmin_x,&
          slkmax_x, prefix, slkmin_c, slkmax_c, prefix, p_min, p_max, prefix,  &
          d_min, d_max

!  test to see if we are feasible

        IF ( inform%primal_infeasibility <= stop_p ) THEN
          IF ( control%just_feasible ) THEN
            inform%status = GALAHAD_ok
            inform%feasible = .TRUE.
            IF ( printi ) THEN
              CALL CLOCK_TIME( clock_now )
              WRITE( out, 2070 ) prefix
              WRITE( out, 2030 ) prefix, inform%iter, re,                      &
                inform%primal_infeasibility, inform%dual_infeasibility,        &
                inform%complementary_slackness, zero, alpha, mu, nbact,        &
                clock_now - clock_start
              IF ( printt ) WRITE( out, 2000 ) prefix
            END IF
            GO TO 600
          END IF

          IF ( .NOT. inform%feasible ) THEN
            IF ( printi ) WRITE( out, 2070 ) prefix
            inform%feasible = .TRUE.
          END IF
        END IF

!  =======
!  STEP 6:
!  =======

!  =====================================================================
!  -*-*-*-*-*-*-*-*- Penalty and Indicator Updates -*-*-*-*-*-*-*-*-*-*-
!  =====================================================================

!  compute the new penalty parameter

        sigma = sigma_max
        IF ( arc == 'ZS' ) THEN
          mu = slknes
        ELSE
          IF ( inform%iter > muzero_fixed )                                    &
            mu = MIN( SQRT( ABS( slknes ) ), sigma ) * ABS( slknes )
        END IF

!  estimate the variable and constraint exit status

        IF ( get_stat ) THEN
          X_stat_old = X_stat ; C_stat_old = C_stat
!         DO i = 1, dims%c_l_start-1
!           write(6,"(' cl ', I7,' c,y =', 2ES9.1)") i, C_res(i), Y( i )
!         END DO
          CALL CLLS_indicators( dims, n, m, C_l, C_u, C_last, C,               &
                                DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,       &
                                DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,        &
                                Y_last, Z_last, control%indicator_type,        &
                                control%indicator_tol_p,                       &
                                control%indicator_tol_pd,                      &
                                control%indicator_tol_tapia, C_stat, X_stat )

!  count the number of active constraints/bounds

          IF ( printw )                                                        &
            WRITE( out, "( A, ' indicators: n_active/n, m_active/m ', 4I7 )" ) &
               prefix, COUNT( X_stat /= 0 ), n, COUNT( C_stat /= 0 ), m

          IF ( printd ) THEN
            write(6,"( ' X_stat ', /, ( 10I7 ) )" ) X_stat
            write(6,"( ' C_stat ', /, ( 10I7 ) )" ) C_stat
          END IF

          b_change = COUNT( X_stat_old - X_stat /= 0 )
          c_change = COUNT( C_stat_old - C_stat /= 0 )

!  if desired, see if the predicted variable and constraint exit status
!  provides an optimal solution; only test if the predicted set has changed

          IF ( b_change + c_change /= 0 ) THEN
            IF ( printi ) WRITE( out,                                          &
              "( A, '  changes in predicted X/C_stat = ',                      &
            &    I0, ', ', I0, ', pouncing for solution ...' )" )              &
               prefix, b_change, c_change
            CALL CLLS_pounce( n, o, m, weight, Ao_val, Ao_row, Ao_ptr,         &
                              B, A_val, A_col, A_ptr, C_l, C_u, X_l, X_u,      &
                              X_last, R_last, C_last, Y_last, Z_last,          &
                              C_stat, X_Stat, X_free, RHS, K_sls_pounce,       &
                              SLS_pounce_data, control, inform, optimal, W )
            IF ( printd ) THEN
              WRITE( out, "( ' X before ', /, ( 5ES12.4 ) )" ) X
              WRITE( out, "( ' X ', /, ( 5ES12.4 ) )" ) X_last
              WRITE( out, "( ' R before ', /, ( 5ES12.4 ) )" ) R
              WRITE( out, "( ' R ', /, ( 5ES12.4 ) )" ) R_last
              WRITE( out, "( ' C before ', /, ( 5ES12.4 ) )" ) C_res
              WRITE( out, "( ' C ', /, ( 5ES12.4 ) )" ) C_last
              WRITE( out, "( ' Y before ', /, ( 5ES12.4 ) )" ) Y
              WRITE( out, "( ' Y ', /, ( 5ES12.4 ) )" ) Y_last
              WRITE( out, "( ' Z before ', /, ( 5ES12.4 ) )" ) Z
              WRITE( out, "( ' Z ', /, ( 5ES12.4 ) )" ) Z_last
            END IF

            IF ( optimal ) THEN
              IF ( printi ) WRITE( out, "( A,                                  &
             &   '  pounce successful, optimal solution found' )" ) prefix
              X = X_last ; R = R_last ; C_res = C_last ; Y = Y_last ; Z = Z_last
              stat_known = .TRUE.
              GO TO 600
            ELSE
              IF ( printi ) WRITE( out, "( A,                                  &
             &   '  pounce unsuccessful, continuing' )" ) prefix
            END IF
          END IF
        END IF

        IF ( mu < control%mu_pounce ) THEN
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

!  compute the projected gradient of the Lagrangian function

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

        IF ( printd ) WRITE( out, 2110 ) prefix, pjgnrm, prefix,               &
          inform%primal_infeasibility
      END DO

!  ---------------------------------------------------------------------
!  ---------------------- End of Major Iteration -----------------------
!  ---------------------------------------------------------------------

  600 CONTINUE
      IF ( optimal ) GO TO 800

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
      CALL CLLS_AX( m, C_RES( : m ), m, A_ne, A_val, A_col,                    &
                    A_ptr, n, X, '+ ')
      IF ( printi .AND. m > 0 ) THEN
        WRITE( out, "( A, '  Computed constraint residual is', ES11.4 )" )     &
             prefix,                                                           &
             MAX( zero, MAXVAL( ABS( C_l( : dims%c_equality ) -                &
                                     C_RES(: dims%c_equality ) ) ),            &
                        MAXVAL( C_l(  dims%c_l_start : dims%c_l_end ) -        &
                                C_RES(  dims%c_l_start : dims%c_l_end ) ),     &
                        MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -       &
                                C_u( dims%c_u_start : dims%c_u_end ) ) )
      END IF

!  estimate the variable and constraint exit status

      IF ( .NOT. stat_known .AND. inform%status >= 0 ) THEN
!     IF ( .NOT. stat_known ) THEN
        X_stat_old = X_stat ; C_stat_old = C_stat
        CALL CLLS_indicators( dims, n, m, C_l, C_u, C_last, C,                 &
                              DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,         &
                              DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,          &
                              Y_last, Z_last, control%indicator_type,          &
                              control%indicator_tol_p,                         &
                              control%indicator_tol_pd,                        &
                              control%indicator_tol_tapia, C_stat, X_stat )

!  count the number of active constraints/bounds

        IF ( printi ) WRITE( out, "( A, '  Indicators: n_active/n,',           &
       &   ' m_active/m = ', 2( I0, '/', I0, : ', ' ) )" )                     &
             prefix, COUNT( X_stat /= 0 ), n, COUNT( C_stat /= 0 ), m

        b_change = COUNT( X_stat_old - X_stat /= 0 )
        c_change = COUNT( C_stat_old - C_stat /= 0 )

!      WRITE( 6, * ) ' before pounce'
!      WRITE( 6, "( ' b ', 5ES12.4 )" ) B
!      WRITE( 6, "( ' x ', 5ES12.4 )" ) X
!      WRITE( 6, "( ' y ', 5ES12.4 )" ) Y
!      WRITE( 6, "( ' z ', 5ES12.4 )" ) Z
!      WRITE( 6, "( ' c ', 5ES12.4 )" ) C
!      WRITE( 6, "( ' r ', 5ES12.4 )" ) R

!  if desired, see if the predicted variable and constraint exit status
!  provides an optimal solution

        IF ( b_change + c_change /= 0 ) THEN
          IF ( printi ) WRITE( out,                                            &
            "( A,  '  changes in predicted X/C_stat = ',                       &
          &    I0, ', ', I0, ', pouncing for solution ...' )" )                &
             prefix, b_change, c_change
          CALL CLLS_pounce( n, o, m, weight, Ao_val, Ao_row, Ao_ptr,           &
                            B, A_val, A_col, A_ptr, C_l, C_u, X_l, X_u,        &
                            X_last, R_last, C_last, Y_last, Z_last,            &
                            C_stat, X_Stat, X_free, RHS, K_sls_pounce,         &
                            SLS_pounce_data, control, inform, optimal, W )
          IF ( optimal ) THEN
            IF ( printi ) WRITE( out, "( A,                                    &
           &   '  pounce successful, optimal solution found' )" ) prefix
            X = X_last ; R = R_last ; C_res = C_last ; Y = Y_last ; Z = Z_last
            stat_known = .TRUE.
          ELSE
            IF ( printi ) WRITE( out, "( A,                                    &
           &   '  pounce unsuccessful, status = ', I0 )" ) prefix, inform%status
          END IF
        END IF
      END IF

!  compute the final objective residual ...

      R = - B
      CALL CLLS_AoX( o, R, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, n, X, '+ ' )

!  ... the objective function ...

      IF ( present_weight ) THEN
        inform%obj = half * DOT_PRODUCT( R, W * R )
      ELSE
        inform%obj = half * DOT_PRODUCT( R, R )
      END IF
      IF ( weight > zero )                                                     &
        inform%obj = inform%obj + half * weight * DOT_PRODUCT( X, X )

!  ... the distances to the bounds ...

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

!  ... the dual variables and Lagrange multipliers ...

      DO i = dims%x_free + 1, dims%x_l_end
        Z_l( i ) = MAX( Z( i ), zero )
      END DO

      DO i = dims%x_u_start, n
        Z_u( i ) = MIN( Z( i ), zero )
      END DO

      DO i = dims%c_l_start, dims%c_l_end
        Y_l( i ) = MAX( Y( i ), zero )
      END DO

      DO i = dims%c_u_start, dims%c_u_end
        Y_u( i ) = MIN( Y( i ), zero )
      END DO

!  ... the gradient of the Lagrangian function ..

      CALL CLLS_Lagrangian_gradient( dims, n, o, m, weight,                    &
                                     X, R, Y, Y_l, Y_u, Z_l, Z_u,              &
                                     Ao_ne, Ao_val, Ao_row, Ao_ptr,            &
                                     A_ne, A_val, A_col, A_ptr,                &
                                     DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,   &
                                     GRAD_L( dims%x_s : dims%x_e ),            &
                                     .FALSE., dufeas, W )

!  ... the norm of the projected gradient ...

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

!  ... and the primal and dual infeasibility

      merit = CLLS_merit_value( dims, n, m, X, Y, Y_l, Y_u, Z_l, Z_u,          &
                                DIST_X_l, DIST_X_u, DIST_C_l, DIST_C_u,        &
                                GRAD_L( dims%x_s : dims%x_e ), C_RES,          &
                                tau, res_primal, inform%dual_infeasibility,    &
                                res_primal_dual, res_cs )

!  if required, make the solution exactly complementary

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

!  print statistics

  800 CONTINUE
      IF ( printi ) THEN
        WRITE( out, "( /, A, '  Final objective function value is', ES21.14,   &
      &       /, A, '  Total number of iterations = ', I0,                     &
      &       /, A, '  Total number of backtracks = ', I0 )" )                 &
          prefix, inform%obj, prefix, inform%iter, prefix, inform%nbacts
        WRITE( out, 2110 ) prefix, pjgnrm, prefix, inform%primal_infeasibility
        IF ( control%getdua ) WRITE( out,                                      &
         "( /, A, ' Advanced starting point is used for dual variables' )" )   &
           prefix
        WRITE( out, "( A, '  gamma_c,f are', 2ES11.4 )" )                      &
          prefix, gamma_c, gamma_f
        IF ( puiseux ) THEN
          IF ( control%every_order ) THEN
            IF ( control%arc == 1 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Puiseux',    &
           &   ' fit to the Zhang arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 2 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Puiseux',    &
           &   ' fit to the Zhao-Sun arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 4 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Puiseux',    &
           &   ' fit to the Zhang-Puiseux arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 5 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Puiseux',    &
           &   ' fit to the Zhang-Puiseux arc is used' )" ) prefix, order
            ELSE
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Puiseux',    &
           &   ' fit to the Zhang-Zhao-Sun arc is used' )" ) prefix, order
            END IF
          ELSE
            IF ( control%arc == 1 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Puiseux',            &
           &   ' fit to the Zhang arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 2 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Puiseux',            &
           &   ' fit to the Zhao-Sun arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 4 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Puiseux',            &
           &   ' fit to the Zhang-Puiseux arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 5 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Puiseux',            &
           &   ' fit to the Zhang-Puiseux arc is used' )" ) prefix, order
            ELSE
              WRITE( control%out, "( A, '  Order ', I0, ' Puiseux',            &
           &   ' fit to the Zhang-Zhao-Sun arc is used' )" ) prefix, order
            END IF
          END IF
        ELSE
          IF ( control%every_order ) THEN
            IF ( control%arc == 1 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Taylor',     &
           &   ' fit to the Zhang arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 2 ) THEN
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Taylor',     &
           &   ' fit to the Zhao-Sun arc is used' )" ) prefix, order
!           ELSE IF ( control%arc == 4 ) THEN
!             WRITE( control%out, "( A, '  Maximum order ', I0, ' Taylor',     &
!          &   ' fit to the Zhang-Puiseux arc is used' )" ) prefix, order
            ELSE
              WRITE( control%out, "( A, '  Maximum order ', I0, ' Taylor',     &
           &   ' fit to the Zhang-Zhao-Sun arc is used' )" ) prefix, order
            END IF
          ELSE
            IF ( control%arc == 1 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Taylor',             &
           &   ' fit to the Zhang arc is used' )" ) prefix, order
            ELSE IF ( control%arc == 2 ) THEN
              WRITE( control%out, "( A, '  Order ', I0, ' Taylor',             &
           &   ' fit to the Zhao-Sun arc is used' )" ) prefix, order
!           ELSE IF ( control%arc == 4 ) THEN
!             WRITE( control%out, "( A, '  Order ', I0, ' Taylor',             &
!          &   ' fit to the Zhang Puiseux arc is used' )" ) prefix, order
            ELSE
              WRITE( control%out, "( A, '  Order ', I0, ' Taylor',             &
           &   ' fit to the Zhang-Zhao-Sun arc is used' )" ) prefix, order
            END IF
          END IF
        END IF
      END IF

!  If necessary, print warning messages

      IF ( printi ) then
        SELECT CASE( inform%status )
          CASE( GALAHAD_error_restrictions  ) ; WRITE( out, "( /, A,           &
         & '  Warning - input paramters incorrect' )" ) prefix
          CASE( GALAHAD_error_no_center ) ; WRITE( out, "( /, A,               &
         & '  Warning - the analytic center appears to be unbounded' )" ) prefix
          CASE( GALAHAD_error_bad_bounds ) ; WRITE( out, "( /, A,              &
         &  '  Warning - the constraints are inconsistent' )" ) prefix
          CASE( GALAHAD_error_primal_infeasible ) ; WRITE( out, "( /, A,       &
         &  '  Warning - the constraints appear to be inconsistent' )" ) prefix
          CASE( GALAHAD_error_factorization ) ; WRITE( out, "( /, A,           &
         &   '  Warning - factorization failure' )" ) prefix
          CASE( GALAHAD_error_ill_conditioned ) ; WRITE( out, "( /, A,         &
         &   '  Warning - no further progress possible' )"  ) prefix
          CASE( GALAHAD_error_tiny_step ) ; WRITE( out, "( /, A,               &
         &   '  Warning - step too small to make progress,',                   &
         &   ' problem maybe infeasible' )" ) prefix
          CASE( GALAHAD_error_max_iterations ) ; WRITE( out, "( /, A,          &
         &   '  Warning - iteration bound exceeded' )" ) prefix
          CASE( GALAHAD_error_unbounded ) ; WRITE( out, "( /, A,               &
         &   '  Warning - problem appears to be unbounded from below' )") prefix
        END SELECT
        WRITE( out, "( A, '  Linear system solver ', A, ' is used' )" )        &
            prefix, TRIM( control%symmetric_linear_solver )
        SELECT CASE ( control%indicator_type )
        CASE( 1 )
          WRITE( out, "( A, '  Primal indicators used' )" ) prefix
        CASE( 2 )
          WRITE( out, "( A, '  Primal-dual indicators used' )" ) prefix
        CASE( 3 )
          WRITE( out, "( A, '  Tapia indicators used' )" ) prefix
        END SELECT
      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving CLLS_solve_main ' )" ) prefix

      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Iter   p-feas  d-feas com-slk    obj   ',                &
                '  step   target   arc bt     time' )
 2020 FORMAT( A, I5, A1, 3ES8.1, ES9.1, '     -   ', ES7.1,                    &
            '    -   -', 0P, F9.2 )
 2030 FORMAT( A, I5, A1, 3ES8.1, ES9.1, ES8.1, 1X, ES7.1, I3, A1, A2, I3,      &
              0P, F9.2 )
 2070 FORMAT( /, A, ' ========================= feasible point found',         &
                    ' =========================', / )
 2100 FORMAT( A, A, /, ( 10X, 7ES10.2 ) )
 2120 FORMAT( A, A, /, ( 10X, 7I7 ) )
 2110 FORMAT( /, A, '  Norm of projected gradient is', ES11.4,                 &
              /, A, '  Norm of infeasibility is', ES11.4 )
 2130 FORMAT( A, 21X, ' == >  mu estimated   = ', ES10.2, /,                   &
              A, 21X, '       mu_x estimated = ', ES10.2, /,                   &
              A, 21X, '       mu_c estimated = ', ES10.2, /,                   &
              A, 21X, ' min/max slackness_x = ', 2ES12.4, /,                   &
              A, 21X, ' min/max slackness_c = ', 2ES12.4, /,                   &
              A, 14X, ' min/max primal feasibility = ', 2ES12.4, /,            &
              A, 14X, ' min/max dual   feasibility = ', 2ES12.4 )

      CONTAINS

        FUNCTION MAXVAL_ABS( VECT )
        CHARACTER ( len = 10 ) :: MAXVAL_ABS
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VECT
        IF ( SIZE( VECT ) > 0 ) THEN
          WRITE( MAXVAL_ABS, "( ES10.2 )" ) MAXVAL( ABS( VECT ) )
        ELSE
          MAXVAL_ABS = '     -    '
        END IF
        RETURN
        END FUNCTION MAXVAL_ABS

!  End of CLLS_solve_main

      END SUBROUTINE CLLS_solve_main

!-*-*-*-*-*-*-   C L L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CLLS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine CLLS_initialize
!   control see Subroutine CLLS_initialize
!   inform  see Subroutine CLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( CLLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( CLLS_control_type ), INTENT( IN ) :: control
      TYPE ( CLLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by FDC

      CALL FDC_terminate( data%FDC_data, data%FDC_control,                     &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FDC_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated by CRO

      CALL CRO_terminate( data%CRO_data, control%CRO_control,                  &
                          inform%CRO_inform )
      IF ( inform%CRO_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%CRO_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      inform%status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'clls: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated within SLS_pounce

      CALL SLS_terminate( data%SLS_pounce_data,                                &
                          control%SLS_pounce_control,                          &
                          inform%SLS_pounce_inform )
      inform%status = inform%SLS_pounce_inform%status
      IF ( inform%SLS_pounce_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'clls: data%SLS_pounce'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate FIT internal arrays

      CALL FIT_terminate( data%FIT_data, control%FIT_control,                  &
                          inform%FIT_inform )
      IF ( inform%FIT_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FIT_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate ROOTS internal arrays

      CALL ROOTS_terminate( data%ROOTS_data, control%ROOTS_control,            &
                            inform%ROOTS_inform )
      IF ( inform%ROOTS_inform%status /= GALAHAD_ok )                          &
        inform%status = inform%ROOTS_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate LSP internal arrays

      CALL LSP_terminate( data%LSP_map, data%LSP_control,                      &
                          data%LSP_inform )
      IF ( data%LSP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%LSP_inform%alloc_status
        inform%bad_alloc = data%LSP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL LSP_terminate( data%LSP_map_freed, data%LSP_control,                &
                          data%LSP_inform)
      IF ( data%LSP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%LSP_inform%alloc_status
        inform%bad_alloc = data%LSP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated for the preprocessing stage

      CALL LSP_terminate( data%LSP_map, data%LSP_control, data%LSP_inform )
      IF ( data%LSP_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%LSP_inform%alloc_status
        inform%bad_alloc = 'clls: data%LSP'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'clls: data%INDEX_C_freed'
      CALL SPACE_dealloc_array( data%INDEX_C_freed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%GRAD_L'
      CALL SPACE_dealloc_array( data%GRAD_L,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DIST_X_l'
      CALL SPACE_dealloc_array( data%DIST_X_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DIST_X_u'
      CALL SPACE_dealloc_array( data%DIST_X_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Z_l'
      CALL SPACE_dealloc_array( data%Z_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Z_u'
      CALL SPACE_dealloc_array( data%Z_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%BARRIER_X'
      CALL SPACE_dealloc_array( data%BARRIER_X,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_l'
      CALL SPACE_dealloc_array( data%Y_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DY_l'
      CALL SPACE_dealloc_array( data%DY_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DIST_C_l'
      CALL SPACE_dealloc_array( data%DIST_C_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_u'
      CALL SPACE_dealloc_array( data%Y_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DY_u'
      CALL SPACE_dealloc_array( data%DY_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DIST_C_u'
      CALL SPACE_dealloc_array( data%DIST_C_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%BARRIER_C'
      CALL SPACE_dealloc_array( data%BARRIER_C,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%SCALE_C'
      CALL SPACE_dealloc_array( data%SCALE_C,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%R_last'
      CALL SPACE_dealloc_array( data%R_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%X_last'
      CALL SPACE_dealloc_array( data%C_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%X_last'
      CALL SPACE_dealloc_array( data%X_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_last'
      CALL SPACE_dealloc_array( data%Y_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Z_last'
      CALL SPACE_dealloc_array( data%Z_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%OPT_alpha'
      CALL SPACE_dealloc_array( data%OPT_alpha,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%OPT_merit'
      CALL SPACE_dealloc_array( data%OPT_merit,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%X_coef'
      CALL SPACE_dealloc_array( data%X_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%C_coef'
      CALL SPACE_dealloc_array( data%C_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_coef'
      CALL SPACE_dealloc_array( data%Y_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_l_coef'
      CALL SPACE_dealloc_array( data%Y_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Y_u_coef'
      CALL SPACE_dealloc_array( data%Y_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Z_l_coef'
      CALL SPACE_dealloc_array( data%Z_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%Z_u_coef'
      CALL SPACE_dealloc_array( data%Z_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%BINOMIAL'
      CALL SPACE_dealloc_array( data%BINOMIAL,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%COEF'
      CALL SPACE_dealloc_array( data%COEF,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%CS_COEF'
      CALL SPACE_dealloc_array( data%CS_COEF,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%ROOTS'
      CALL SPACE_dealloc_array( data%ROOTS,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DX_zh'
      CALL SPACE_dealloc_array( data%DX_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DC_zh'
      CALL SPACE_dealloc_array( data%DC_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DY_zh'
      CALL SPACE_dealloc_array( data%DY_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DY_l_zh'
      CALL SPACE_dealloc_array( data%DY_l_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DY_u_zh'
      CALL SPACE_dealloc_array( data%DY_u_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DZ_l_zh'
      CALL SPACE_dealloc_array( data%DZ_l_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%DZ_u_zh'
      CALL SPACE_dealloc_array( data%DZ_u_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'clls: data%X_free'
      CALL SPACE_dealloc_array( data%X_free,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

     array_name = 'clls: data%K_sls%row'
     CALL SPACE_dealloc_array( data%K_sls%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%K_sls%col'
     CALL SPACE_dealloc_array( data%K_sls%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%K_sls%val'
     CALL SPACE_dealloc_array( data%K_sls%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%K_sls%type'
     CALL SPACE_dealloc_array( data%K_sls%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'clls: data%K_sls_pounce%row'
      CALL SPACE_dealloc_array( data%K_sls_pounce%row,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'clls: data%K_sls_pounce%col'
      CALL SPACE_dealloc_array( data%K_sls_pounce%col,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'clls: data%K_sls_pounce%val'
      CALL SPACE_dealloc_array( data%K_sls_pounce%val,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'clls: data%K_sls_pounce%type'
      CALL SPACE_dealloc_array( data%K_sls_pounce%type,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine CLLS_terminate

      END SUBROUTINE CLLS_terminate

! - G A L A H A D -  C L L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE CLLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( CLLS_control_type ), INTENT( IN ) :: control
     TYPE ( CLLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL CLLS_terminate( data%clls_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'clls: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%R'
     CALL SPACE_dealloc_array( data%prob%R,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Ao%ptr'
     CALL SPACE_dealloc_array( data%prob%Ao%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Ao%row'
     CALL SPACE_dealloc_array( data%prob%Ao%row,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Ao%col'
     CALL SPACE_dealloc_array( data%prob%Ao%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Ao%val'
     CALL SPACE_dealloc_array( data%prob%Ao%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Ao%type'
     CALL SPACE_dealloc_array( data%prob%Ao%type,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%Atype'
     CALL SPACE_dealloc_array( data%prob%A%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%C_status'
     CALL SPACE_dealloc_array( data%prob%C_status,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'clls: data%prob%X_status'
     CALL SPACE_dealloc_array( data%prob%X_status,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine CLLS_full_terminate

     END SUBROUTINE CLLS_full_terminate

!-*-*-*-*-*-*-*-   C L L S _ P O U N C E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE CLLS_pounce( n, o, m, weight, Ao_val, Ao_row, Ao_ptr,         &
                              B, A_val, A_col, A_ptr, C_l, C_u, X_l, X_u, X,   &
                              R, C, Y, Z, C_stat, X_stat, X_free, SOL, K_sls,  &
                              SLS_data, control, inform, optimal, W )

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ),                                    &
                              DIMENSION( Ao_ptr( n + 1 ) - 1 ) :: Ao_row
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
                           DIMENSION( Ao_ptr( n + 1 ) - 1 ) :: Ao_val
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: B
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ),                                    &
                              DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
                           DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( m ) :: C
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( m ) :: Y
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: Z
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m ) :: C_stat
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: X_stat
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: X_free
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + o + m ) :: SOL
      TYPE ( SMT_type ), INTENT( INOUT ) :: K_sls
      TYPE ( CLLS_control_type ), INTENT( IN ) :: control
      TYPE ( CLLS_inform_type ), INTENT( INOUT ) :: inform
      LOGICAL, INTENT( OUT ) :: optimal
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: sls_data

!  optional dummy argument

      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: W

!  construct the equality-constrained linear least-squares problem according
!   to the variables and constraints that are predicted to be active via

!  C_stat is an INTEGER array of length m, which must have been be set to
!   indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set

!  X_stat is an INTEGER array of length n, which must have been set to
!   indicate the likely ultimate status of the simple bound constraints
!   Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set

!  Report whether the solution to this equality-constrained linear
!  least-squares problem is optimal for the original problem

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, jj, k, l, alloc_status
      INTEGER ( KIND = ip_ ) :: n_free, n_fixed, m_active, nfpo, npo, next
      INTEGER ( KIND = ip_ ) :: nz_a_active, nz_ao_free, nz_ao, nz_k_sls
      REAL ( KIND = rp_ ) :: time_analyse, time_factorize, time_solve
      REAL ( KIND = rp_ ) :: clock_analyse, clock_factorize, clock_solve
      REAL ( KIND = rp_ ) :: ci, ei
      REAL ( KIND = rp_ ) :: feas = epsmch * 100.0_rp_
      LOGICAL :: x_feas, c_feas, y_feas, z_feas, present_weight
      CHARACTER ( LEN = 80 ) :: array_name

!  Using the sets B = { i | X_stat( i ) = 0 }, F = { i | X_stat( i ) /= 0 }
!  and A = { i | C_stat( i ) = 0 }, the corresponding "optimality" system

!     ( weight I_B      0      Ao_B^T  A_AB^T  I_B^T ) (   x_B )   (  0  )
!     (     0      weight I_F  Ao_F^T  A_AF^T    0   ) (   x_F )   (  0  )
!     (    Ao_B        Ao_F     - I      0       0   ) (    r  ) = (  b  )  (1)
!     (    A_AB        A_AF              0       0   ) ( - y_A )   ( c_A )
!     (    I_B          0                0       0   ) ( - z_B )   ( b_B )

!  may either be solved "as is", or by eliminating x_B = b_B,
!  solving the "reduced" system

!     ( weight I_F  Ao_F^T  A_AF^T ) (   x_F )   (        0       )
!     (    Ao_F      - I     0     ) (    r  ) = (   b - Ao_B x_b )         (2)
!     (    A_AF              0     ) ( - y_A )   ( c_A - A_AB x_b )

!  and then recovering

!     z_B = weight x_B + Ao_B^T r - A_AB^T y_A                              (3)

      present_weight = PRESENT( W )

!  1. Set up the matrices and right-hand sides involved

!  --------------------------------
!  1a. solve via the reduced system
!  --------------------------------

      IF ( control%reduced_pounce_system ) THEN

!  count the number of free variables (variables in F), and flag their
!  indices in X_free (a 0 value indicates a fixed variable). Also set
!  the active components of x to their appropriate bounds and the
!  inactive dual variables z to zero, and initialize the active dual
!  varaibles to g_B

        n_free = 0
        DO j = 1, n
          IF ( X_stat( j ) == 0 ) THEN ! free variable
            n_free = n_free + 1
            X_free( j ) = n_free
            Z( j ) = zero ! set z_F = 0
          ELSE ! fixed variable, set x_B = b_B accordingly
            X_free( j ) = 0
            IF ( X_stat( j ) < 0 ) THEN
              X( j ) = X_l( j )
            ELSE
              X( j ) = X_u( j )
            END IF
            Z( j ) = weight * X( j ) ! initialize z_B = weight x_B
          END IF
        END DO

!  form the (lower triangle of the) reduced matrix

!          ( weight I_F  Ao_F^T  A_AF^T )  ( n_free   dimensional )
!    K_r = (    Ao_F      - I     0     )  ( o        dimensional )
!          (    A_AF       0      0     )  ( m_active dimensional )

!  first, count the number of nonzeros in the reduced Jacobian, Ao_F ...

        nz_ao_free = 0
        DO j = 1, n
          IF ( X_stat( j ) == 0 )                                              &
            nz_ao_free = nz_ao_free + Ao_ptr( j + 1 ) - Ao_ptr( j )
        END DO

!  ... and the number of nonzeros in the reduced active Jacobian, A_AF

        m_active = 0 ; nz_a_active = 0
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN  ! active constraint
            m_active = m_active + 1
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( X_free( A_col( l ) ) > 0 ) nz_a_active = nz_a_active + 1
            END DO
          END IF
        END DO

!  record the total number of nonzeros in the lower triangle of K

        nz_k_sls = nz_ao_free + nz_a_active + o
        IF ( weight /= zero ) nz_k_sls = nz_k_sls + n_free
        nfpo = n_free + o

!  allocate space for K

        K_sls%n = nfpo + m_active ; K_sls%ne = nz_k_sls
        CALL SMT_put( K_sls%type, 'COORDINATE', alloc_status )

        array_name = 'clls: data%K_sls_pounce%row'
        CALL SPACE_resize_array( nz_k_sls, K_sls%row,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: data%K_sls_pounce%col'
        CALL SPACE_resize_array( nz_k_sls, K_sls%col,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: data%K_sls_pounce%val'
        CALL SPACE_resize_array( nz_k_sls, K_sls%val,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize the first terms in the right hand sides
!    (       0        )
!    (  b - Ao_B x_b  )
!    ( c_A - A_AB x_b )

        SOL( 1 : n_free ) = zero
        SOL( n_free + 1 : nfpo ) = B( 1 : o )


!     ( weight I_F    Ao_F^T  A_AF^T ) (   x_F )   (        0       )
!     (    Ao_F      - W^{-1}   0    ) (    r  ) = (   b - Ao_B x_b )        (2)
!     (    A_AF          0      0    ) ( - y_A )   ( c_A - A_AB x_b )

!  fill in the values of K: the 1,1 block weight I_F

        IF ( weight /= zero ) THEN
          DO j = 1, n_free
            K_sls%row( j ) = j ;  K_sls%col( j ) = j ; K_sls%val( j ) = weight
          END DO
          l = n_free
        ELSE
          l = 0
        END IF

!  the 2,1 block Ao_F

        DO j = 1, n
          jj = X_free( j )
          IF ( jj > 0 ) THEN ! free variable
            DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
              l = l + 1
              K_sls%row( l ) = n_free + Ao_row( k )
              K_sls%col( l ) = jj
              K_sls%val( l ) = Ao_val( k )
            END DO
          ELSE ! fixed variable, update the right-hand side b - Ao_B x_b
            DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
              i = n_free + Ao_row( k )
              SOL( i ) = SOL( i ) - Ao_val( k ) * X( j )
            END DO
          END IF
        END DO

!  the 2,2 block - W^{-1}

        IF ( present_weight ) THEN
          DO j = n_free + 1, nfpo
            l = l + 1
            K_sls%row( l ) = j ;  K_sls%col( l ) = j
            K_sls%val( l ) = - one / W( j - n_free )
          END DO
        ELSE
          DO j = n_free + 1, nfpo
            l = l + 1
            K_sls%row( l ) = j ;  K_sls%col( l ) = j ; K_sls%val( l ) = - one
          END DO
        END IF

!  the 3,1 block A_AF

        next = nfpo
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN  ! active constraint
            next = next + 1
            IF ( C_stat( i ) > 0 ) THEN ! initialize the right-hand side c_A
              SOL( next ) = C_u( i )
            ELSE
              SOL( next ) = C_l( i )
            END IF
            DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
              j = A_col( k )
              jj = X_free( j )
              IF ( jj > 0 ) THEN ! free variable
                l = l + 1
                K_sls%row( l ) = next
                K_sls%col( l ) = jj
                K_sls%val( l ) = A_val( k )
              ELSE ! fixed variable, update the right-hand side c_A - A_AB x_b
                SOL( next ) = SOL( next ) - A_val( k ) * X( j )
              END IF
            END DO
          END IF
        END DO

!  -----------------------------
!  1b. solve via the full system
!  -----------------------------

      ELSE

!  count the number of fixed variables (variables in B)

        n_fixed = COUNT( X_stat( 1 : n ) /= 0 )

!  form the (lower triangle of the) matrix

!     ( weight I    Ao^T   A_A^T  I_B^T )
!     (   Ao      - W^{-1}   0      0   )
!     (   A_A        0       0      0   )
!     (   I_B        0       0      0   )

!  first, count the number of nonzeros in the Jacobian, Ao ...

        nz_ao = Ao_ptr( n + 1 ) - 1

!  ... and the number of nonzeros in the active Jacobian, A_A

        m_active = 0 ; nz_a_active = 0
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN ! active constraint
            m_active = m_active + 1
            nz_a_active = nz_a_active + A_ptr( i + 1 ) - A_ptr( i )
          END IF
        END DO

!  record the total number of nonzeros in the lower triangle of K

        nz_k_sls = nz_ao + nz_a_active + o + n_fixed
        IF ( weight /= zero ) nz_k_sls = nz_k_sls + n
        npo = n + o

!  allocate space for K

        K_sls%n = npo + m_active + n_fixed ; K_sls%ne = nz_k_sls
        CALL SMT_put( K_sls%type, 'COORDINATE', alloc_status )

        array_name = 'clls: data%K_sls_pounce%row'
        CALL SPACE_resize_array( nz_k_sls, K_sls%row,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: data%K_sls_pounce%col'
        CALL SPACE_resize_array( nz_k_sls, K_sls%col,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'clls: data%K_sls_pounce%val'
        CALL SPACE_resize_array( nz_k_sls, K_sls%val,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the right hand sides
!     (  0  )
!     (  b  )

        SOL( 1 : n ) = zero
        SOL( n + 1 : npo ) = B( 1 : o )

!  fill in the values of K: the 1,1 block weight I

        IF ( weight /= zero ) THEN
          DO j = 1, n
            K_sls%row( j ) = j ;  K_sls%col( j ) = j ; K_sls%val( j ) = weight
          END DO
          l = n
        ELSE
          l = 0
        END IF

!  the 2,1 block Ao

        DO j = 1, n
          DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
            l = l + 1
            K_sls%row( l ) = n + Ao_row( k )
            K_sls%col( l ) = j
            K_sls%val( l ) = Ao_val( k )
            END DO
        END DO

!  the 2,2 block - W^{-1}

        IF ( present_weight ) THEN
          DO j = n + 1, npo
            l = l + 1
            K_sls%row( l ) = j ;  K_sls%col( l ) = j
            K_sls%val( l ) = - one / W( j - n )
          END DO
        ELSE
          DO j = n + 1, npo
            l = l + 1
            K_sls%row( l ) = j ;  K_sls%col( l ) = j ; K_sls%val( l ) = - one
          END DO
        END IF

!  the 3,1 block A_AF

        next = npo
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN ! active constraint
            next = next + 1
            DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
              l = l + 1
              K_sls%row( l ) = next
              K_sls%col( l ) = A_col( k )
              K_sls%val( l ) = A_val( k )
            END DO
            IF ( C_stat( i ) > 0 ) THEN ! set the right-hand side c_A
              SOL( next ) = C_u( i )
            ELSE
              SOL( next ) = C_l( i )
            END IF
          END IF
        END DO

!  the 4,1 block I_B

        next = npo
        DO j = 1, n
          IF ( X_stat( j ) /= 0 ) THEN ! fixed variable
            l = l + 1
            next = next + 1
            K_sls%row( l ) = next
            K_sls%col( l ) = j
            K_sls%val( l ) = one
            IF ( X_stat( j ) < 0 ) THEN ! set the right hand side b_B
              SOL( next ) = X_l( j )
            ELSE
              SOL( next ) = X_u( j )
            END IF
          END IF
        END DO
      END IF

!  2. Analyse and factorize the system matrix K, and solve the resulting system

!  analyse K

      time_analyse = inform%SLS_inform%time%analyse
      clock_analyse = inform%SLS_inform%time%clock_analyse

      CALL SLS_analyse( K_sls, SLS_data, control%SLS_control,                  &
                        inform%SLS_inform )

      inform%time%analyse = inform%time%analyse +                              &
        inform%SLS_inform%time%analyse - time_analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%SLS_inform%time%clock_analyse - clock_analyse

!  check that the analysis succeeded

      IF ( inform%sls_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_factorization
        GO TO 900
      END IF

!  factorize K

      time_factorize = inform%SLS_inform%time%factorize
      clock_factorize = inform%SLS_inform%time%clock_factorize

      CALL SLS_factorize( K_sls, SLS_data, control%SLS_control,                &
                          inform%SLS_inform )
      inform%nfacts = inform%nfacts + 1

      inform%time%factorize = inform%time%factorize +                          &
        inform%SLS_inform%time%factorize - time_factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%SLS_inform%time%clock_factorize - clock_factorize

!  check that the factorization succeeded

      IF ( inform%sls_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_factorization
        GO TO 900
      END IF

!  solve the required optimality system

      time_solve = inform%SLS_inform%time%solve
      clock_solve = inform%SLS_inform%time%clock_solve

      CALL SLS_solve( K_sls, SOL, SLS_data, control%SLS_control,               &
                      inform%SLS_inform )
      inform%nfacts = inform%nfacts + 1

      inform%time%solve = inform%time%solve +                                  &
        inform%SLS_inform%time%solve - time_solve
      inform%time%clock_solve = inform%time%clock_solve +                      &
        inform%SLS_inform%time%clock_solve - clock_solve

!  check that the solve succeeded

      IF ( inform%sls_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_factorization
        GO TO 900
      END IF

!  --------------------------------------------------------
!  3a. recover the solution from that of the reduced system
!  --------------------------------------------------------

      IF ( control%reduced_pounce_system ) THEN

!  recover x

        X( : n_free ) = SOL( : n_free )

!  recover r

        R( : o ) = SOL( n_free + 1 : nfpo )

!  recover y

        next = nfpo
        DO i = 1, m
         IF ( C_stat( i ) == 1 .OR. C_stat( i ) == - 1 ) THEN
            next = next + 1
            Y( i ) = - SOL( next )
          ELSE
            Y( i ) = zero
          END IF
        END DO

!  recover z_B <- z_B + Ao_B^T r ...

        DO j = 1, n
          IF ( X_free( j ) == 0 ) THEN ! fixed variable
            DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
              i = Ao_row( k )
              Z( j ) = Z( j ) + Ao_val( k ) * R( i )
            END DO
          END IF
        END DO

!  and then z_B <- z_B - A_AB^T y_A

        next = nfpo
        DO i = 1, m
          IF ( C_stat( i ) /= 0 ) THEN  ! active constraint
            next = next + 1
            DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
              j = A_col( k )
              IF ( X_free( j ) == 0 ) THEN ! fixed variable
                Z( j ) = Z( j ) - A_val( k ) * Y( i )
              END IF
            END DO
          END IF
        END DO

!  -----------------------------------------------------
!  3b. recover the solution from that of the full system
!  -----------------------------------------------------

      ELSE

!  recover x

        X( : n ) = SOL( : n )

!  recover r

        R( : o ) = SOL( n + 1 : npo )

!  recover y

        next = npo
        DO i = 1, m
         IF ( C_stat( i ) == 1 .OR. C_stat( i ) == - 1 ) THEN
            next = next + 1
            Y( i ) = - SOL( next )
          ELSE
            Y( i ) = zero
          END IF
        END DO

!  recover z

        DO j = 1, n
          IF ( X_stat( j ) == 1 .OR. X_stat( j ) == - 1 ) THEN
            next = next + 1
            Z( j ) = - SOL( next )
          ELSE
            Z( j ) = zero
          END IF
        END DO
      END IF

! 4. Check if the pounced solution is feasibile

!  compute c and an estimate e of the error in c (stored in SOL(:m))

      DO i = 1, m
        ci = zero ; ei = zero
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          ci = ci + A_val( l ) * X( A_col( l ) )
          ei = ei + ABS( A_val( l ) ) * ABS( X( A_col( l ) ) )
        END DO
        C( i ) = ci ; SOL( i ) = ei * epsmch
      END DO

      x_feas = .TRUE. ; c_feas = .TRUE. ; y_feas = .TRUE. ; z_feas = .TRUE.

!  check for feasibility of the general constraints

      DO i = 1, m

!  equality constraint

        ei = SOL( i )
        IF ( C_l( i ) == C_u( i ) ) THEN

!  infeasible inequality constraint

        ELSE IF ( C( i ) < C_l( i ) - ei ) THEN
          c_feas = .FALSE. ; EXIT
        ELSE IF ( C( i ) > C_u( i ) + ei ) THEN
          c_feas = .FALSE. ; EXIT

!  incorrect Lagrange multiplier of active constraint at lower bound

        ELSE IF ( C( i ) < C_l( i ) + ei ) THEN
          IF ( Y( i ) < - feas ) THEN
            y_feas = .FALSE. ; EXIT
          END IF

!  incorrect Lagrange multiplier of active counstraint at upper bound

        ELSE IF ( C( i ) > C_u( i ) - ei ) THEN
          IF ( Y( i ) > feas ) THEN
            y_feas = .FALSE. ; EXIT
          END IF

!  incorrect Lagrange multiplier of inactive constraint

        ELSE
          IF ( ABS( Y( i ) ) > feas ) THEN
            y_feas = .FALSE. ; EXIT
          END IF
        END IF
      END DO

!  check for feasibility of the simple bound constraints

      DO j = 1, n

!  fixed variable

        IF ( X_l( j ) == X_u( j ) ) THEN

!  infeasible simple bound

        ELSE IF ( X( j ) < X_l( j ) - feas ) THEN
          x_feas = .FALSE. ; EXIT
        ELSE IF ( X( j ) > X_u( j ) + feas ) THEN
          x_feas = .FALSE. ; EXIT

!  incorrect dual variable of variable at lower bound

        ELSE IF ( X( j ) < X_l( j ) + feas ) THEN
          IF ( Z( j ) < - feas ) THEN
            z_feas = .FALSE. ; EXIT
          END IF

!  incorrect dual variable of variable at upper bound

        ELSE IF ( X( j ) > X_u( j ) - feas ) THEN
          IF ( Z( j ) > feas ) THEN
            z_feas = .FALSE. ; EXIT
          END IF

!  incorrect dual variable of inactive simple bound

        ELSE
          IF ( ABS( Z( j ) ) > feas ) THEN
            z_feas = .FALSE. ; EXIT
          END IF
        END IF
      END DO

      optimal = x_feas .AND. c_feas .AND. y_feas .AND. z_feas
      RETURN

!  error return

  900 CONTINUE
      optimal = .FALSE.

      RETURN

!  End of CLLS_pounce

      END SUBROUTINE CLLS_pounce

!-*-  C L L S _ L A G R A N G I A N _ G R A D I E N T   S U B R O U T I N E  -*-

      SUBROUTINE CLLS_Lagrangian_gradient( dims, n, o, m, weight,              &
                                           X, R, Y, Y_l, Y_u, Z_l, Z_u,        &
                                           Ao_ne, Ao_val, Ao_row, Ao_ptr,      &
                                           A_ne, A_val, A_col, A_ptr,          &
                                           DIST_X_l, DIST_X_u, DIST_C_l,       &
                                           DIST_C_u, GRAD_L,                   &
                                           getdua, dufeas, W )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the gradient of the Lagrangian function
!
!  GRAD_L = Ao^T W r + weight x - A^T y, where r = Ao x - b
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( CLLS_dims_type ), INTENT( IN ) :: dims
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m, Ao_ne, A_ne
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( m ) :: Y
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( INOUT ),                                    &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l
      REAL ( KIND = rp_ ), INTENT( INOUT ),                                    &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u
      REAL ( KIND = rp_ ), INTENT( INOUT ),                                    &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) :: Z_l
      REAL ( KIND = rp_ ), INTENT( INOUT ),                                    &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( Ao_ne ) :: Ao_row
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( Ao_ne ) :: Ao_val
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: GRAD_L
      REAL ( KIND = rp_ ), INTENT( IN ) :: dufeas
      LOGICAL, INTENT( IN ) :: getdua
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l
      REAL ( KIND = rp_ ), INTENT( IN ),                                       &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u

!  optional dummy argument

      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: W

!  Local variables

      INTEGER ( KIND = ip_ ) :: i
      REAL ( KIND = rp_ ) :: gi

!  compute the gradient of the Lagrangian. Start with the weight x term

      IF ( weight == zero ) THEN
        GRAD_L = zero
      ELSE
        GRAD_L = weight * X
      END IF

!  add the Ao^T W r term

      IF ( PRESENT( W ) ) THEN
        CALL CLLS_AoX( n, GRAD_L, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, o, W * R,  &
                       '+T' )
      ELSE
        CALL CLLS_AoX( n, GRAD_L, n, Ao_ne, Ao_val, Ao_row, Ao_ptr, o, R, '+T' )
      END IF

!  subtract the A^T y term

      CALL CLLS_AX( n, GRAD_L, m, A_ne, A_val, A_col, A_ptr, m, Y, '-T' )

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

!  End of CLLS_Lagrangian_gradient

      END SUBROUTINE CLLS_Lagrangian_gradient

!-*-*-*-*-*-*-   C L L S _ w o r k s p a c e   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE CLLS_workspace( n, o, m, dims, ao_ne, a_ne, order, GRAD_L,    &
                                 DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X,      &
                                 Y_l, DIST_C_l, Y_u, DIST_C_u, C, BARRIER_C,   &
                                 SCALE_C, RHS, OPT_alpha, OPT_merit,           &
                                 BINOMIAL, CS_coef, COEF, ROOTS, DX_zh,        &
                                 DY_zh, DC_zh, DY_l_zh, DY_u_zh, DZ_l_zh,      &
                                 DZ_u_zh, X_coef, C_coef, Y_coef, Y_l_coef,    &
                                 Y_u_coef, Z_l_coef, Z_u_coef, R_last,         &
                                 C_last, X_last, Y_last, Z_last, K_sls,        &
                                 error, series_order, deallocate_error_fatal,  &
                                 space_critical, status, alloc_status,         &
                                 bad_alloc )

!  allocate workspace arrays for use in CLLS_solve_main

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, o, n, ao_ne, a_ne
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: error, series_order
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: order
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, alloc_status
      LOGICAL, INTENT( IN ) :: deallocate_error_fatal, space_critical
      TYPE ( CLLS_dims_type ), INTENT( IN ) :: dims
      CHARACTER ( LEN = 80 ), INTENT( INOUT ) :: bad_alloc
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( : ) ::     &
           GRAD_L, DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X, Y_l, DIST_C_l,     &
           Y_u, DIST_C_u, C, BARRIER_C, SCALE_C, RHS, OPT_alpha, OPT_merit,    &
           CS_coef, COEF, ROOTS, DX_zh, DY_zh, DC_zh, DY_l_zh,                 &
           DY_u_zh, DZ_l_zh, DZ_u_zh, R_last, C_last, X_last, Y_last, Z_last
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( INOUT ), DIMENSION( :, : ) ::  &
           X_coef, C_coef, Y_coef, Y_l_coef, Y_u_coef, Z_l_coef, Z_u_coef,     &
           BINOMIAL
      TYPE ( SMT_type ), INTENT( INOUT ) :: K_sls

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  allocate workspace arrays

      array_name = 'clls: GRAD_L'
      CALL SPACE_resize_array( dims%c_e, GRAD_L,                               &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DIST_X_l'
      CALL SPACE_resize_array( dims%x_l_start, dims%x_l_end, DIST_X_l,         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DIST_X_u'
      CALL SPACE_resize_array( dims%x_u_start, dims%x_u_end, DIST_X_u,         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Z_l'
      CALL SPACE_resize_array( dims%x_free + 1, dims%x_l_end, Z_l,             &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Z_u'
      CALL SPACE_resize_array( dims%x_u_start, n, Z_u,                         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: BARRIER_X'
      CALL SPACE_resize_array( dims%x_free + 1, n, BARRIER_X,                  &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_l'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_l_end, Y_l,              &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DIST_C_l'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_l_end, DIST_C_l,         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_u'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, Y_u,              &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DIST_C_u'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, DIST_C_u,         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: C'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_u_end, C,                &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: BARRIER_C'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_u_end, BARRIER_C,        &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: SCALE_C'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_u_end, SCALE_C,          &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: RHS'
      CALL SPACE_resize_array( dims%v_e, RHS,                                  &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      order = series_order

      array_name = 'clls: OPT_alpha'
      CALL SPACE_resize_array( order, OPT_alpha,                               &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: OPT_merit'
      CALL SPACE_resize_array( order, OPT_merit,                               &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: X_coef'
      CALL SPACE_resize_array( 1_ip_, n, 0_ip_, order, X_coef,                 &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: C_coef'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_u_end, 0_ip_,            &
             order, C_coef, status, alloc_status, array_name = array_name,     &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_coef'
      CALL SPACE_resize_array( 1_ip_, m, 0_ip_, order, Y_coef,                 &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_l_coef'
      CALL SPACE_resize_array(                                                 &
             dims%c_l_start, dims%c_l_end, 0_ip_, order, Y_l_coef,             &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_u_coef'
      CALL SPACE_resize_array(                                                 &
             dims%c_u_start, dims%c_u_end, 0_ip_, order, Y_u_coef,             &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Z_l_coef'
      CALL SPACE_resize_array(                                                 &
             dims%x_free + 1, dims%x_l_end, 0_ip_, order, Z_l_coef,            &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Z_u_coef'
      CALL SPACE_resize_array(                                                 &
             dims%x_u_start, n, 0_ip_, order, Z_u_coef,                        &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: BINOMIAL'
      CALL SPACE_resize_array( 0_ip_, order - 1_ip_, order, BINOMIAL,          &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: CS_coef'
      CALL SPACE_resize_array( 0_ip_, 2 * order, CS_coef,                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: COEF'
      CALL SPACE_resize_array( 0_ip_, 2 * order, COEF,                         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: ROOTS'
      CALL SPACE_resize_array( 2 * order, ROOTS,                               &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DX_zh'
      CALL SPACE_resize_array( 1_ip_, n, DX_zh,                                &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DC_zh'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_u_end, DC_zh,            &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DY_zh'
      CALL SPACE_resize_array( 1_ip_, m, DY_zh,                                &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DY_l_zh'
      CALL SPACE_resize_array( dims%c_l_start, dims%c_l_end, DY_l_zh,          &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DY_u_zh'
      CALL SPACE_resize_array( dims%c_u_start, dims%c_u_end, DY_u_zh,          &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DZ_l_zh'
      CALL SPACE_resize_array( dims%x_free + 1, dims%x_l_end, DZ_l_zh,         &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: DZ_u_zh'
      CALL SPACE_resize_array( dims%x_u_start, n, DZ_u_zh,                     &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: R_last'
      CALL SPACE_resize_array( o, R_last,                                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: C_last'
      CALL SPACE_resize_array( m, C_last,                                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: X_last'
      CALL SPACE_resize_array( n, X_last,                                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Y_last'
      CALL SPACE_resize_array( m, Y_last,                                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: Z_last'
      CALL SPACE_resize_array( n, Z_last,                                      &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

!  K will be in coordinate form

      CALL SMT_put( K_sls%type, 'COORDINATE', alloc_status )
      K_sls%n = n + dims%nc + m + o
      K_sls%ne = n + dims%nc + a_ne + dims%nc + ao_ne + o

!  allocate space for K

      array_name = 'clls: K_sls%row'
      CALL SPACE_resize_array( K_sls%ne, K_sls%row,                            &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: K_sls%col'
      CALL SPACE_resize_array( K_sls%ne, K_sls%col,                            &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      array_name = 'clls: K_sls%val'
      CALL SPACE_resize_array( K_sls%ne, K_sls%val,                            &
             status, alloc_status, array_name = array_name,                    &
             deallocate_error_fatal = deallocate_error_fatal,                  &
             exact_size = space_critical,                                      &
             bad_alloc = bad_alloc, out = error )
      IF ( status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine CLLS_workspace

      END SUBROUTINE CLLS_workspace

!-*-*-  C L L S _ s u m m a r i z e _ p r o b l e m   S U B R O U T I N E  -*-*-

     SUBROUTINE CLLS_summarize_problem( out, prob )

!  Summarizes the problem prob on output device out

!  Nick Gould, December 23rd 2014

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j

      WRITE( out, "( ' n, o, m = ', I0, 1X, I0, 1X, I0 )" )                    &
        prob%n, prob%o, prob%m

!  objective function

      WRITE( out, "( ' B = ', /, ( 5ES12.4 ) )" ) prob%B( : prob%o )
      IF ( prob%o > 0 ) THEN
        IF ( SMT_get( prob%Ao%type ) == 'DENSE' .OR.                           &
             SMT_get( prob%Ao%type ) == 'DENSE_BY_COLUMNS' ) THEN
          WRITE( out, "( ' Ao (dense) = ', /, ( 5ES12.4 ) )" )                 &
            prob%Ao%val( : prob%n * prob%o )
        ELSE IF ( SMT_get( prob%Ao%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( out, "( ' Ao (row-wise) = ' )" )
          DO i = 1, prob%o
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
              ( i, prob%Ao%col( j ), prob%Ao%val( j ),                         &
                j = prob%Ao%ptr( i ), prob%Ao%ptr( i + 1 ) - 1 )
          END DO
        ELSE IF ( SMT_get( prob%Ao%type ) == 'SPARSE_BY_COLUMNS' ) THEN
          WRITE( out, "( ' Ao (column-wise) = ' )" )
          DO j = 1, prob%n
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
              ( j, prob%Ao%row( i ), prob%Ao%val( i ),                         &
                i = prob%Ao%ptr( j ), prob%Ao%ptr( j + 1 ) - 1 )
          END DO
        ELSE
          WRITE( out, "( ' Ao (co-ordinate) = ' )" )
          WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                             &
          ( prob%Ao%row( i ), prob%Ao%col( i ), prob%Ao%val( i ),              &
            i = 1, prob%Ao%ne)
        END IF
      END IF

!  simple bounds

      WRITE( out, "( ' X_l = ', /, ( 5ES12.4 ) )" ) prob%X_l( : prob%n )
      WRITE( out, "( ' X_u = ', /, ( 5ES12.4 ) )" ) prob%X_u( : prob%n )

!  general constraints

      IF ( prob%m > 0 ) THEN
        IF ( SMT_get( prob%A%type ) == 'DENSE' .OR.                            &
             SMT_get( prob%A%type ) == 'DENSE_BY_COLUMNS' ) THEN
          WRITE( out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )                  &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_COLUMNS' ) THEN
          WRITE( out, "( ' A (column-wise) = ' )" )
          DO j = 1, prob%n
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
              ( j, prob%A%row( i ), prob%A%val( i ),                           &
                i = prob%A%ptr( j ), prob%A%ptr( j + 1 ) - 1 )
          END DO
        ELSE
          WRITE( out, "( ' A (co-ordinate) = ' )" )
          WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                             &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
        END IF
        WRITE( out, "( ' C_l = ', /, ( 5ES12.4 ) )" ) prob%C_l( : prob%m )
        WRITE( out, "( ' C_u = ', /, ( 5ES12.4 ) )" ) prob%C_u( : prob%m )
      END IF

      RETURN

!  end of subroutine CLLS_summarize_problem

      END SUBROUTINE CLLS_summarize_problem

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  C L L S _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE CLLS_import( control, data, status, n, o, m,                   &
                             Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr,           &
                             A_type, A_ne, A_row, A_col, A_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to CLLS_solve
!
!  data is a scalar variable of type CLLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, o >= 0, m >= 0 or requirement that type contains
!       its relevant string 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
!       'COORDINATE', 'SPARSE_BY_ROWS' or 'SPARSE_BY_COLUMNS',
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  o is a scalar variable of type default integer, that holds the number of
!   observations
!
!  m is a scalar variable of type default integer, that holds the number of
!   residuals
!
!  Ao_type is a character string that specifies the objective design matrix
!   storage scheme used. It should be one of 'coordinate', 'sparse_by_rows',
!   'sparse_by_columns', 'dense', 'dense_by_rows' or 'dense_by_columns';
!   lower or upper case variants are allowed
!
!  Ao_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  Ao_row is a rank-one array of type default integer, that holds the row
!   indices J in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  Ao_col is a rank-one array of type default integer, that holds the column
!   indices of J in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense schemes are used, and
!   in this case can be of length 0
!
!  Ao_ptr is a rank-one array of dimension n+1 and type default integer,
!   that holds the starting position of each row of J, as well as the total
!   number of entries plus one, in the sparse row-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!  A_type is a character string that specifies the constraint Jacobian storage
!   scheme used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   or 'absent', the latter if m = 0; lower or upper case variants are allowed
!
!  A_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  A_row is a rank-one array of type default integer, that holds the row
!   indices J in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  A_col is a rank-one array of type default integer, that holds the column
!   indices of J in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense scheme is used, and
!   in this case can be of length 0
!
!  A_ptr is a rank-one array of dimension n+1 and type default integer,
!   that holds the starting position of each row of J, as well as the total
!   number of entries plus one, in the sparse row-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CLLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m, Ao_ne, A_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Ao_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_ptr
     CHARACTER ( LEN = * ), INTENT( IN ) :: A_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%clls_control = control

     error = data%clls_control%error
     space_critical = data%clls_control%space_critical
     deallocate_error_fatal = data%clls_control%space_critical

!  allocate vector space if required

     array_name = 'clls: data%prob%B'
     CALL SPACE_resize_array( o, data%prob%B,                                  &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal =                                           &
              data%clls_control%deallocate_error_fatal,                        &
            exact_size = data%clls_control%space_critical,                     &
            bad_alloc = data%clls_inform%bad_alloc,                            &
            out = data%clls_control%error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X,                                  &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%X_l'
     CALL SPACE_resize_array( n, data%prob%X_l,                                &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%X_u'
     CALL SPACE_resize_array( n, data%prob%X_u,                                &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%Z'
     CALL SPACE_resize_array( n, data%prob%Z,                                  &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%C'
     CALL SPACE_resize_array( m, data%prob%C,                                  &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%C_l'
     CALL SPACE_resize_array( m, data%prob%C_l,                                &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%C_u'
     CALL SPACE_resize_array( m, data%prob%C_u,                                &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%Y'
     CALL SPACE_resize_array( m, data%prob%Y,                                  &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%C_status'
     CALL SPACE_resize_array( m, data%prob%C_status,                           &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

     array_name = 'clls: data%prob%X_status'
     CALL SPACE_resize_array( n, data%prob%X_status,                           &
            data%clls_inform%status, data%clls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%clls_inform%bad_alloc, out = error )
     IF ( data%clls_inform%status /= 0 ) GO TO 900

!  put data into the required components of the qpt storage type

     data%prob%n = n ; data%prob%o = o ; data%prob%m = m

!  set Ao appropriately in the qpt storage type

     SELECT CASE ( Ao_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( Ao_row ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'COORDINATE',                          &
                     data%clls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = Ao_ne

       array_name = 'clls: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900
       IF ( data%f_indexing ) THEN
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_ROWS',                      &
                     data%clls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( o + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( o + 1 )
       END IF
       array_name = 'clls: data%prob%Ao%ptr'
       CALL SPACE_resize_array( o + 1, data%prob%Ao%ptr,                       &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_COLUMNS',                   &
                     data%clls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( n + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( n + 1 )
       END IF
       array_name = 'clls: data%prob%Ao%ptr'
       CALL SPACE_resize_array( n + 1, data%prob%Ao%ptr,                       &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 )
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 ) + 1
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE', 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE',                               &
                     data%clls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'clls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE_BY_COLUMNS',                    &
                     data%clls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'clls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%clls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

!  set A appropriately in the qpt storage type

     SELECT CASE ( A_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( A_row ) .AND. PRESENT( A_col ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%A%type, 'COORDINATE',                           &
                     data%clls_inform%alloc_status )
       data%prob%A%n = n ; data%prob%A%m = m
       data%prob%A%ne = A_ne

       array_name = 'clls: data%prob%A%row'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%row,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%col'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%col,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%val'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%val,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%A%row( : data%prob%A%ne ) = A_row( : data%prob%A%ne )
         data%prob%A%col( : data%prob%A%ne ) = A_col( : data%prob%A%ne )
       ELSE
         data%prob%A%row( : data%prob%A%ne ) = A_row( : data%prob%A%ne ) + 1
         data%prob%A%col( : data%prob%A%ne ) = A_col( : data%prob%A%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( A_ptr ) .AND. PRESENT( A_col ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%A%type, 'SPARSE_BY_ROWS',                       &
                     data%clls_inform%alloc_status )
       data%prob%A%n = n ; data%prob%A%m = m
       IF ( data%f_indexing ) THEN
         data%prob%A%ne = A_ptr( m + 1 ) - 1
       ELSE
         data%prob%A%ne = A_ptr( m + 1 )
       END IF
       array_name = 'clls: data%prob%A%ptr'
       CALL SPACE_resize_array( m + 1, data%prob%A%ptr,                        &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%col'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%col,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%val'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%val,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%A%ptr( : m + 1 ) = A_ptr( : m + 1 )
         data%prob%A%col( : data%prob%A%ne ) = A_col( : data%prob%A%ne )
       ELSE
         data%prob%A%ptr( : m + 1 ) = A_ptr( : m + 1 ) + 1
         data%prob%A%col( : data%prob%A%ne ) = A_col( : data%prob%A%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( A_ptr ) .AND. PRESENT( A_row ) ) ) THEN
         data%clls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%A%type, 'SPARSE_BY_COLUMNS',                   &
                     data%clls_inform%alloc_status )
       data%prob%A%n = n ; data%prob%A%m = m
       IF ( data%f_indexing ) THEN
         data%prob%A%ne = A_ptr( n + 1 ) - 1
       ELSE
         data%prob%A%ne = A_ptr( n + 1 )
       END IF
       array_name = 'clls: data%prob%A%ptr'
       CALL SPACE_resize_array( n + 1, data%prob%A%ptr,                        &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%row'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%row,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       array_name = 'clls: data%prob%A%val'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%val,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%A%ptr( : n + 1 ) = A_ptr( : n + 1 )
         data%prob%A%row( : data%prob%A%ne ) = A_row( : data%prob%A%ne )
       ELSE
         data%prob%A%ptr( : n + 1 ) = A_ptr( : n + 1 ) + 1
         data%prob%A%row( : data%prob%A%ne ) = A_row( : data%prob%A%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE', 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%prob%A%type, 'DENSE',                                &
                     data%clls_inform%alloc_status )
       data%prob%A%n = n ; data%prob%A%m = m
       data%prob%A%ne = m * n

       array_name = 'clls: data%prob%A%val'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%val,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%prob%A%type, 'DENSE_BY_COLUMNS',                     &
                     data%clls_inform%alloc_status )
       data%prob%A%n = n ; data%prob%A%m = m
       data%prob%A%ne = m * n

       array_name = 'clls: data%prob%A%val'
       CALL SPACE_resize_array( data%prob%A%ne, data%prob%A%val,               &
              data%clls_inform%status, data%clls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%clls_inform%bad_alloc, out = error )
       IF ( data%clls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%clls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%clls_inform%status
     RETURN

!  End of subroutine CLLS_import

     END SUBROUTINE CLLS_import

!-  G A L A H A D -  C L L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE CLLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See CLLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CLLS_control_type ), INTENT( IN ) :: control
     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%clls_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine CLLS_reset_control

     END SUBROUTINE CLLS_reset_control

!-*-*-  G A L A H A D -  C L L S _ s o l v e _ c l l s  S U B R O U T I N E  -*-

     SUBROUTINE CLLS_solve_clls( data, status, Ao_val, B, A_val, C_l, C_u,     &
                                 X_l, X_u, X, R, C, Y, Z, X_stat, C_stat,      &
                                 regularization_weight, W )

!  solve the constrained linear least-squares problem whose structure was
!  previously imported. See CLLS_solve for a description of the required
!  arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type CLLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, clls_solve above.
!
!  Ao_val is a rank-one array of type default real, that holds the values
!   of the design matrix Ao in the storage scheme specified in clls_import.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of linear terms of the observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  A_val is a rank-one array of type default real, that holds the values of
!   the constraint Jacobian A in the storage scheme specified in clls_import.
!
!  C_l, C_u are rank-one arrays of dimension m, that hold the values of
!   the lower and upper bounds, c_l and c_u, on the general linear constraints.
!   Any bound c_l(i) or c_u(i) larger than or equal to control%infinity in
!   absolute value will be regarded as being infinite (see the entry
!   control%infinity). Thus, an infinite lower bound may be specified by
!   setting the appropriate component of C_l to a value smaller than
!   -control%infinity, while an infinite upper bound can be specified by
!   setting the appropriate element of C_u to a value larger than
!   control%infinity.
!
!  X_l, X_u are rank-one arrays of dimension n, that hold the values of
!   the lower and upper bounds, c_l and c_u, on the variables x.
!   Any bound x_l(i) or x_u(i) larger than or equal to control%infinity in
!   absolute value will be regarded as being infinite (see the entry
!   control%infinity). Thus, an infinite lower bound may be specified by
!   setting the appropriate component of X_l to a value smaller than
!   -control%infinity, while an infinite upper bound can be specified by
!   setting the appropriate element of X_u to a value larger than
!   control%infinity.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  R is a rank-one array of dimension m and type default
!   real, that holds the vector of residuals Ao x - b.
!   The i-th component of R, i = 1, ... , m, contains (Ao x - b)_i.
!
!  C is a rank-one array of dimension m and type default
!   real, that holds the vector of the constraints A x.
!   The i-th component of C, i = 1, ... , m, contains (A x)_i.
!
!  Y is a rank-one array of dimension m and type default
!   real, that holds the vector of the Lagrange multipliers, y.
!   The i-th component of Y, i = 1, ... , m, contains (y)_i.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  X_stat is a rank-one array of dimension n and type default integer,
!   that mwill be set on exit to indicate which constraints are in the final
!   working set. Possible exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  C_stat is a rank-one array of dimension m and type default integer,
!   that will be set on exit to indicate which constraints are in the final
!   working set. Possible exit values are
!   C_stat( i ) < 0, the i-th constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  regularization_weight is an optional scalar of type default real that
!   may be set to the value of the non-negative regularization weight.
!   If it is absent, the regularization weight will be zero.
!
!  W is an optional rank-one array of type default real that may be
!   set to the values of the components of the weights W.
!   The i-th component of W, i = 1, ... , o, contains (w)_i.
!   If it is absent, the weights will all be taken to be 1.0.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: Ao_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: C_l, C_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X_l, X_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Y, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R, C
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: C_stat, X_stat
     REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: regularization_weight
     REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( data%prob%o ) :: W

!  local variables

     INTEGER ( KIND = ip_ ) :: m, o, n

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o ; m = data%prob%m

!  save the observations

     data%prob%B( : o ) = B( : o )

!  save the lower and upper simple bounds

     data%prob%X_l( : n ) = X_l( : n )
     data%prob%X_u( : n ) = X_u( : n )

!  save the lower and upper constraint bounds

     data%prob%C_l( : m ) = C_l( : m )
     data%prob%C_u( : m ) = C_u( : m )

!  save the initial primal and dual variables and Lagrange multipliers

     data%prob%X( : n ) = X( : n )
     data%prob%Z( : n ) = Z( : n )
     data%prob%Y( : m ) = Y( : m )

!  save the objective design matrix Ao entries

     IF ( data%prob%Ao%ne > 0 )                                                &
       data%prob%Ao%val( : data%prob%Ao%ne ) = Ao_val( : data%prob%Ao%ne )

!  save the constraint Jacobian A entries

     IF ( data%prob%A%ne > 0 )                                                 &
       data%prob%A%val( : data%prob%A%ne ) = A_val( : data%prob%A%ne )

!  call the solver

     CALL CLLS_solve( data%prob, data%clls_data, data%clls_control,            &
                      data%clls_inform, regularization_weight, W )

!  recover the optimal primal and dual variables, Lagrange multipliers,
!  constraint values and status values for constraints and simple bounds

     X( : n ) = data%prob%X( : n )
     Z( : n ) = data%prob%Z( : n )
     Y( : m ) = data%prob%Y( : m )
     IF ( ALLOCATED( data%prob%C ) ) THEN
       C( : m ) = data%prob%C( : m )
     ELSE
       C( : m ) = infinity
     END IF
     IF ( ALLOCATED( data%prob%R ) ) THEN
       R( : o ) = data%prob%R( : o )
     ELSE
       R( : o ) = infinity
     END IF
     IF ( ALLOCATED( data%prob%C_status ) ) THEN
       C_stat( : m ) = data%prob%C_status( : m )
     ELSE
       C_stat( : m ) = 0
     END IF
     IF ( ALLOCATED( data%prob%X_status ) ) THEN
       X_stat( : n ) = data%prob%X_status( : n )
     ELSE
       X_stat( : n ) = 0
     END IF

     status = data%clls_inform%status
     RETURN

!  End of subroutine CLLS_solve_clls

     END SUBROUTINE CLLS_solve_clls

!-  G A L A H A D -  C L L S _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE CLLS_information( data, inform, status )

!  return solver information during or after solution by CLLS
!  See CLLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( CLLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%clls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine CLLS_information

     END SUBROUTINE CLLS_information

!  End of module CLLS

    END MODULE GALAHAD_CLLS_precision
