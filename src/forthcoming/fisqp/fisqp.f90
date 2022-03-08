! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ F I S Q P   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould, Yueling Loh and Daniel P. Robinson

!  History -
!   originally written in Matlab by Yueling Loh and Daniel P. Robinson
!   initial Fortran translation, GALAHAD Version 2.6, November 23th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FISQP_double

!     ----------------------------------------------------------
!    |                                                          |
!    | FiSQP, a Filter SQP method for nonlinear optimization    |
!    |                                                          |
!    | Aim: to find a (local) minimizer of the nonlinear        |
!    | programming problem                                      |
!    |                                                          |
!    |  minimize               f (x)                            |
!    |  subject to          a_i^T x   = b_i      i in E_l       !
!    |             b_i^l <= a_i^T x  <= b_i^u    i in I_l       |
!    |                       c_i (x)  =  0       i in E_g       |
!    |             c_i^l <=  c_i (x) <= c_i^u    i in I_g       |
!    |  and          x^l <=       x  <= x^u                     |
!    |                                                          |
!     ----------------------------------------------------------

!$    USE omp_lib
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPACE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type
     USE GALAHAD_USERDATA_double
     USE GALAHAD_FILTER_double
     USE GALAHAD_L1QP_double
     USE GALAHAD_EQP_double
     USE GALAHAD_SLS_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM
     USE GALAHAD_ROOTS_double, ONLY: ROOTS_quadratic
     USE GALAHAD_STRING
     USE GALAHAD_OPT_double
     USE GALAHAD_MOP_double
     USE GALAHAD_LMS_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: FISQP_initialize, FISQP_read_specfile, FISQP_solve,             &
               FISQP_terminate, NLPT_problem_type, GALAHAD_userdata_type,      &
               SMT_type, SMT_put

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER :: fixed = 0
     INTEGER, PARAMETER :: equality = 0
     INTEGER, PARAMETER :: lower = 1
     INTEGER, PARAMETER :: upper = 2
     INTEGER, PARAMETER :: both = 3
     INTEGER, PARAMETER :: free = 4
     INTEGER, PARAMETER :: history_max = 100

     INTEGER, PARAMETER :: s_accel = 1
     INTEGER, PARAMETER :: s_normal = 2

     INTEGER, PARAMETER :: identity_predictor_hessian = 0
     INTEGER, PARAMETER :: se_modified_predictor_hessian = 1
     INTEGER, PARAMETER :: dd_modified_predictor_hessian = 2
     INTEGER, PARAMETER :: l_bfgs_predictor_hessian = 3
     INTEGER, PARAMETER :: powell_l_bfgs_predictor_hessian = 4
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: five = 5.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point8 = 0.8_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: minus_infinity = - ( HUGE( one ) / two )
     REAL ( KIND = wp ), PARAMETER :: ten5 = ten ** 5
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: y_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: z_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: mu_tol = point1

!    LOGICAL, PARAMETER :: print_debug = .TRUE.
     LOGICAL, PARAMETER :: print_debug = .FALSE.
     LOGICAL :: alt_accel = .FALSE.    !  attempt to solve EQP in one stage

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FISQP_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!   any printing will start on this iteration

       INTEGER :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER :: stop_print = - 1

!   the number of iterations between printing

       INTEGER :: print_gap = 1

!   removal of the file alive_file from unit alive_unit causes execution
!    to cease

       INTEGER :: alive_unit = 60
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d                       '

!   maximum number of iterations

       INTEGER :: maxit = 100

!   number of fails allowed before a monotone step is required

       INTEGER :: max_fails = 0

!   predictor Hessian B_k update strategy
!    = 0 if scaled identity
!    = 1 if Schmabel_Eskow modified exact Hessian used
!    = 2 if diagonally_dominant modified exact Hessian used
!    = 3 if L-BFGS with indefinite update skipping used
!    = 4 if L-BFGS with Powell corrections used

       INTEGER :: predictor_hessian = se_modified_predictor_hessian

!   scale the constraints
!    = 0 unscaled
!    = 1 scale by the infinity norms of the Jacobian rows at the initial point
!    = 2 scale as in 1 but rescale relative to the largest

       INTEGER :: scale_constraints = 0

!  indefinite linear equation solver for use with predictor = 2

       CHARACTER ( LEN = 30 ) :: linear_solver_for_modifications =             &
          "sils" // REPEAT( ' ', 26 )

!   any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = wp ) :: infinity = ten ** 19

!   overall convergence tolerances. The iteration will terminate when the norm
!    of violation of the constraints (the "primal infeasibility") is smaller
!    than stop_p, the norm of the gradient of the Lagrangian function (the
!    "dual infeasibility") is smaller than stop_d, and the norm of the
!    complementary slackness is smaller than stop_c

!   the required absolute and relative accuracies for the primal infeasibility

       REAL ( KIND = wp ) :: stop_abs_p = ten ** ( - 5 )
       REAL ( KIND = wp ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

       REAL ( KIND = wp ) :: stop_abs_d = ten ** ( - 5 )
       REAL ( KIND = wp ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

       REAL ( KIND = wp ) :: stop_abs_c = ten ** ( - 5 )
       REAL ( KIND = wp ) :: stop_rel_c = epsmch

!   the required absolute and relative accuracies for the infeasibility
!    The iteration will stop at a minimizer of the infeasibility if the
!    gradient of the infeasibility (J^T c) is smaller in norm than
!    control%stop_abs_i times the norm of c

       REAL ( KIND = wp ) :: stop_abs_i = ten ** ( - 5 )
       REAL ( KIND = wp ) :: stop_rel_i = epsmch

!   the minimum useful predictor decrease allowed when approximately feasible

       REAL ( KIND = wp ) :: stop_predictor = ten ** ( - 12 )

!   the maximum infeasibility tolerated will be the larger of
!    max_abs_i and max_rel_i times the initial infeasibility

       REAL ( KIND = wp ) :: max_abs_i = ten
       REAL ( KIND = wp ) :: max_rel_i = ten

!   the minimum and maximum constraint scaling factors allowed with
!    scale_constraints > 0

       REAL ( KIND = wp ) :: min_constraint_scaling = ten ** ( - 5 )
       REAL ( KIND = wp ) :: max_constraint_scaling = ten ** 5

!   the minimum perturbation when building the predictor Hessian

       REAL ( KIND = wp ) :: min_hessian_perturbation = ten ** ( - 5 )

!   initial trust-region radius for steering subproblem

       REAL ( KIND = wp ) :: radius_steering = ten ** 2

!   initial trust-region radius for accelerator subproblem

       REAL ( KIND = wp ) :: radius_accelerator = ten ** 2

!   step reduction factor when back-tracking to balance the step steering and
!    predictor steps

       REAL ( KIND = wp ) :: tau_reduce = 0.5_wp

!   lower bound on the back-tracking balancing step

       REAL ( KIND = wp ) :: tau_min = ten ** ( - 14 )

!   initial penalty parameter

       REAL ( KIND = wp ) :: sigma_0 = 10.0_wp

!   minimum penalty parameter increase

       REAL ( KIND = wp ) :: sigma_inc = 5.0_wp

!   use the accelerator step?

       LOGICAL :: use_accelerator = .TRUE.

!   lower bound on step size for cauchy-f

       REAL ( KIND = wp ) :: alpha_f_min = ten ** ( - 8 )

!   lower bound on step size for cauchy-phi

       REAL ( KIND = wp ) :: alpha_phi_min = ten ** ( - 8 )

!   Filter margin reduction parameter; require an improvement by at least
!    beta * violation in one filter dimension

       REAL ( KIND = wp ) :: beta = 0.98_wp

!   Filter margin reduction parameter; require an improvement in violation
!    by at least eta_v * linearized violation reduction

       REAL ( KIND = wp ) :: eta_v = ten ** ( - 3 )

!   Filter margin reduction parameter; require an improvement in objective
!    by at least gamma * new violation

       REAL ( KIND = wp ) :: gamma = ten ** ( - 3 )

!   violation reduction parameter; require an improvement in linearized
!    violation by at least gamma_v * linearized objective reduction to
!    be a v-pair

       REAL ( KIND = wp ) :: gamma_v = ten ** ( - 3 )

!   objective reduction parameter; require an improvement in objective value
!    by at least gamma_f * predicted objective reduction to be an o-pair

       REAL ( KIND = wp ) :: gamma_f = ten ** ( - 4 )

!   penalty function reduction parameter; require an improvement in penalty
!    value by at least gamma_phi * predicted penalty reduction to be an b-
!    or p-pair

       REAL ( KIND = wp ) :: gamma_phi = ten ** ( - 4 )

!   penalty parameter increase parameter; increase sigma if linearized
!    penalty function is less that eta_sigma * linearized violation

       REAL ( KIND = wp ) :: eta_sigma = ten ** ( - 6 )

!   The penalty parameter will be updated if the predicted penalty function
!    decrease at s_k relative to that at s_k^p is smaller than eta_phi

       REAL ( KIND = wp ) :: eta_phi = ten ** ( - 3 )

!   backtracking parameter

       REAL ( KIND = wp ) :: alpha_reduce = 0.5_wp

!   a lower bound on permitted step size

       REAL ( KIND = wp ) :: s_tiny = epsmch

!  zero Jacobian entry tolerance

       REAL ( KIND = wp ) :: jacobian_zero_tolerance = epsmch

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   full_solution specifies whether the full solution or only highlights
!    will be printed

       LOGICAL :: full_solution = .TRUE.

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!   just use the penalty function?

       LOGICAL :: just_penalty = .FALSE.

!   just use the filter function?

       LOGICAL :: just_filter = .FALSE.

!  use local information from steering step for the filter envelope?

       LOGICAL :: filter_uses_steering = .TRUE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FILTER

       TYPE ( FILTER_control_type ) :: FILTER_control

!  control parameters for QP

       TYPE ( L1QP_control_type ) :: QP_steer_control
       TYPE ( L1QP_control_type ) :: QP_pred_control

!  control parameters for EQP

       TYPE ( EQP_control_type ) :: QP_accel_control

!  control parameters for LMS

       TYPE ( LMS_control_type ) :: LMS_control

!  control parameters for SLS

       TYPE ( SLS_control_type ) :: SLS_control

     END TYPE FISQP_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FISQP_time_type

!  the total CPU time spent in the package

       REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

       REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

       REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

       REAL ( KIND = wp ):: factorize = 0.0

!  the CPU time spent computing the search direction

       REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

       REAL ( KIND = wp ) :: clock_preprocess = 0.0

!  the clock time spent analysing the required matrices prior to factorization

       REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

       REAL ( KIND = wp ) :: clock_solve = 0.0
     END TYPE FISQP_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FISQP_inform_type

!  return status. See FISQP_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the name of the user-supplied evaluation routine for which an error ocurred

       CHARACTER ( LEN = 12 ) :: bad_eval = REPEAT( ' ', 12 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the value of the objective function at the best estimate of the solution
!   determined by FISQP_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

       REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

       REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

       REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  the number of times that penalty mode was entered

       INTEGER :: entered_penalty = 0

!  the number of iterations in penalty mode.

       INTEGER :: iter_in_penalty = 0

!  the number of accepted v-pairs

       INTEGER :: num_v = 0

!  the number of accepted o-pairs

       INTEGER :: num_o = 0

!  the number of accepted b-pairs

       INTEGER :: num_b = 0

!  the number of accepted p-pairs

       INTEGER :: num_p = 0

!  the number of nonmonotone steps taken

       INTEGER :: num_nm = 0

!  the number of objective and constraint function evaluations

       INTEGER :: fc_eval = 0

!  the number of gradient and Jacobian evaluations

       INTEGER :: gj_eval = 0

!  the number of Hessian evaluations

       INTEGER :: h_eval = 0

!  the number of factorizations that modified the original matrix

       INTEGER :: modifications = 0

!  the number of threads used

       INTEGER :: threads = 1

!  was the last whether Hessian modified?

!      LOGICAL :: B_modified = .FALSE.

!  timings (see above)

       TYPE ( FISQP_time_type ) :: time

!  inform parameters for FILTER

       TYPE ( FILTER_inform_type ) :: FILTER_inform

!  inform parameters for QP

       TYPE ( L1QP_inform_type ) :: QP_steer_inform
       TYPE ( L1QP_inform_type ) :: QP_pred_inform

!  inform parameters for EQP

       TYPE ( EQP_inform_type ) :: QP_accel_inform

!  inform parameters for LMS

       TYPE ( LMS_inform_type ) :: LMS_inform

!  inform parameters for SLS

       TYPE ( SLS_inform_type ) :: SLS_inform

     END TYPE FISQP_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: FISQP_data_type
       INTEGER :: branch, eval_status, H_ne, J_ne, fails, success_iter
       INTEGER :: out, error, print_level, start_print, stop_print
       INTEGER :: print_level_eqp, print_level_eqp_sbls, print_level_eqp_gltr
       REAL :: time_start, time_now
       REAL ( KIND = wp ) :: clock_start, clock_now
       REAL ( KIND = wp ) :: alpha, tau, stop_p, stop_d, stop_c, stop_i
       REAL ( KIND = wp ) :: del_ellf, del_ellf_ref, del_ellphi, del_ellv
       REAL ( KIND = wp ) :: del_ellv_ref, del_ellv_steer_ref, del_qf, del_qphi
       REAL ( KIND = wp ) :: del_qphi_pred, ellv, f_current, f_ref, f_trial
       REAL ( KIND = wp ) :: rho_f_ref, rho_phi_ref, sigma, sigma_new, s_norm
       REAL ( KIND = wp ) :: sigma_new_ref, accel_norm
       REAL ( KIND = wp ) :: phi, phi_ref, primal_viol, primal_viol_ref
       REAL ( KIND = wp ) :: comp_viol, comp_viol_ref, viol_trial,phi_trial
       REAL ( KIND = wp ) :: stop_p_inner, stop_d_inner, stop_c_inner, h_norm
       REAL ( KIND = wp ) :: radius_accelerator
       LOGICAL :: set_printt, set_printi, set_printw, set_printd
       LOGICAL :: set_printm, printe, printi, printt, printm, printw, printd
       LOGICAL :: reverse_fc, reverse_gj, reverse_hl, reverse_hlprod
       LOGICAL :: filter_acceptable, p_mode, check_filter, accepted, accel_found
       LOGICAL :: exit_small_s, check_b_pair
       LOGICAL :: print_iteration_header, print_1st_header
       CHARACTER ( LEN = 1 ) :: pair_type = REPEAT( ' ', 1 )
       CHARACTER ( LEN = 1 ) :: it_type = REPEAT( ' ', 1 )
       CHARACTER ( LEN = 1 ) :: d_type = REPEAT( ' ', 1 )
       CHARACTER ( LEN = 8 ) :: step_used = REPEAT( ' ', 8 )
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DG
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_ref
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_steer
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_pred
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_accel
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_accel_ref
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_ref
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_scale
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_accel
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_accel
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK_n
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK_m

!  copy of controls

       TYPE ( FISQP_control_type ) :: control

!  data for FILTER

       TYPE ( FILTER_data_type ) :: FILTER_data

!  data for steering QP

       TYPE ( QPT_problem_type ) :: QP_steer
       TYPE ( L1QP_data_type ) :: QP_steer_data

!  data for predictor QP

       TYPE ( QPT_problem_type ) :: QP_pred
       TYPE ( L1QP_data_type ) :: QP_pred_data
       TYPE ( SLS_data_type ) :: SLS_data

!  data for accelerator QP

       TYPE ( QPT_problem_type ) :: QP_accel
       TYPE ( EQP_data_type ) :: QP_accel_data
       TYPE ( EQP_control_type ) :: QP_accel_control

     END TYPE FISQP_data_type

   CONTAINS

!-*  G A L A H A D -  F I S Q P _ I N I T I A L I Z E  S U B R O U T I N E  *-

     SUBROUTINE FISQP_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for FISQP controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FISQP_data_type ), INTENT( OUT ) :: data
     TYPE ( FISQP_control_type ), INTENT( OUT ) :: control
     TYPE ( FISQP_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  Initalize L1QP components

     CALL L1QP_initialize( data%QP_steer_data, control%QP_steer_control,       &
                           inform%QP_steer_inform )
     control%QP_steer_control%prefix =  '" - StQP:"                   '
     control%QP_steer_control%refine = .FALSE.
     CALL L1QP_initialize( data%QP_pred_data, control%QP_pred_control,         &
                           inform%QP_pred_inform )
     control%QP_pred_control%prefix =  '" - PrQP:"                    '
     control%QP_pred_control%refine = .FALSE.

!  Intialize EQP data

     CALL EQP_initialize( data%QP_accel_data, control%QP_accel_control,        &
                          inform%QP_accel_inform )
     control%QP_accel_control%prefix = '" - AcQP:"                   '
!    control%QP_accel_control%SBLS_control%prefix = '" -- SBLS:"               '

!  initalize LMS components

     CALL LMS_initialize( data%QP_pred%H_lm, control%LMS_control,              &
                          inform%LMS_inform )
     control%LMS_control%prefix = '" - LMS:"                     '

     RETURN

!  End of subroutine FISQP_initialize

     END SUBROUTINE FISQP_initialize

!-*-*-   F I S Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE FISQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by FISQP_initialize could (roughly)
!  have been set as:

! BEGIN FISQP SPECIFICATIONS (DEFAULT)
! error-printout-device                             6
! printout-device                                   6
! alive-device                                      60
! print-level                                       1
! start-print                                       -1
! stop-print                                        -1
! iterations-between-printing                       1
! maximum-number-of-iterations                      50
! predictor-hessian                                 2
! scale-constraints                                 0
! max-fails-before-monotone                         0
! infinity-value                                    1.0D+19
! absolute-primal-accuracy                          6.0D-6
! relative-primal-accuracy                          2.0D-16
! absolute-dual-accuracy                            6.0D-6
! relative-dual-accuracy                            2.0D-16
! absolute-complementary-slackness-accuracy         6.0D-6
! relative-complementary-slackness-accuracy         2.0D-16
! absolute-infeasiblity-tolerated                   6.0D-6
! relative-infeasiblity-tolerated                   2.0D-16
! minimum-useful-predictor-decrease                 1.0D-12
! maximum-absolute-infeasibility                    10.0
! maximum-relative-infeasibility                    10.0
! minimum-constraint-scaling-factor                 1.0D-5
! maximum-constraint-scaling-factor                 1.0D+5
! minimum-predictor-hessian-perturbation            1.0D-5
! initial-steering-model-radius                     1.0D+2
! use-accelerator                                   yes
! initial-accelerator-model-radius                  1.0D+2
! step-balance-reduction-factor                     0.5
! step-balance-minimum                              1.0D-14
! tiny-step                                         2.0D-16
! required-filter-margin-reduction                  0.98
! required-v-filter-reduction-vs-dlv                1.0D-3
! required-f-filter-reduction-vs-dlv                1.0D-3
! required-v-reduction-vs-df                        1.0D-3
! required-f-reduction-vs-df                        1.0D-4
! required-p-reduction-vs-dp                        1.0D-4
! required-dlphi-reduction-vs-dlv                   1.0D-6
! required-dqphi-reduction-s-vs-pred                1.0D-3
! backtracking-linesearch-reduction-factor          0.5
! cauchy-f-stepsize-minimum                         1.0D-8
! cauchy-phi-stepsize-minimum                       1.0D-8
! jacobian-zero-tolerance                           2.0D-16
! maximum-cpu-time-limit                            -1.0
! maximum-clock-time-limit                          -1.0
! print-full-solution                               no
! space-critical                                    no
! deallocate-error-fatal                            no
! just-use-the-penalty-function                     no
! just-use-the-filter                               no
! use-accelerator-step                              yes
! filter-uses-steering-information                  yes
! alive-filename                                    ALIVE.d
! linear-equation-solver-for-modifications          sils
! output-line-prefix                                ""
! END FISQP SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FISQP_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: alive_unit = out + 1
     INTEGER, PARAMETER :: print_level = alive_unit + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print  = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: maxit = print_gap + 1
     INTEGER, PARAMETER :: infinity = maxit + 1
     INTEGER, PARAMETER :: stop_abs_p = infinity + 1
     INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
     INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
     INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
     INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
     INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
     INTEGER, PARAMETER :: stop_abs_i = stop_rel_c + 1
     INTEGER, PARAMETER :: stop_rel_i = stop_abs_i + 1
     INTEGER, PARAMETER :: stop_predictor = stop_rel_i + 1
     INTEGER, PARAMETER :: max_abs_i = stop_predictor + 1
     INTEGER, PARAMETER :: max_rel_i = max_abs_i + 1
     INTEGER, PARAMETER :: min_constraint_scaling = max_abs_i + 1
     INTEGER, PARAMETER :: max_constraint_scaling                              &
                             = min_constraint_scaling + 1
     INTEGER, PARAMETER :: min_hessian_perturbation                            &
                             = max_constraint_scaling + 1
     INTEGER, PARAMETER :: just_penalty = min_hessian_perturbation + 1
     INTEGER, PARAMETER :: just_filter = just_penalty + 1
     INTEGER, PARAMETER :: max_fails = just_filter + 1
     INTEGER, PARAMETER :: predictor_hessian = max_fails + 1
     INTEGER, PARAMETER :: scale_constraints = predictor_hessian + 1
     INTEGER, PARAMETER :: radius_steering = scale_constraints + 1
     INTEGER, PARAMETER :: eta_v = radius_steering + 1
     INTEGER, PARAMETER :: tau_reduce = eta_v + 1
     INTEGER, PARAMETER :: tau_min = tau_reduce + 1
     INTEGER, PARAMETER :: eta_sigma = tau_min + 1
     INTEGER, PARAMETER :: eta_phi = eta_sigma + 1
     INTEGER, PARAMETER :: sigma_0 = eta_phi + 1
     INTEGER, PARAMETER :: sigma_inc = sigma_0 + 1
     INTEGER, PARAMETER :: use_accelerator = sigma_inc + 1
     INTEGER, PARAMETER :: radius_accelerator = use_accelerator + 1
     INTEGER, PARAMETER :: alpha_f_min = radius_accelerator + 1
     INTEGER, PARAMETER :: alpha_phi_min = alpha_f_min + 1
     INTEGER, PARAMETER :: beta = alpha_phi_min + 1
     INTEGER, PARAMETER :: gamma = beta + 1
     INTEGER, PARAMETER :: gamma_v = gamma + 1
     INTEGER, PARAMETER :: gamma_f = gamma_v + 1
     INTEGER, PARAMETER :: gamma_phi = gamma_f + 1
     INTEGER, PARAMETER :: alpha_reduce = gamma_phi + 1
     INTEGER, PARAMETER :: s_tiny = alpha_reduce + 1
     INTEGER, PARAMETER :: jacobian_zero_tolerance = s_tiny + 1
     INTEGER, PARAMETER :: cpu_time_limit = jacobian_zero_tolerance + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: filter_uses_steering = clock_time_limit + 1
     INTEGER, PARAMETER :: full_solution = filter_uses_steering + 1
     INTEGER, PARAMETER :: space_critical = full_solution + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: linear_solver_for_modifications = alive_file + 1
     INTEGER, PARAMETER :: prefix = linear_solver_for_modifications + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'FISQP          '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( alive_unit )%keyword = 'alive-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print  )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( predictor_hessian )%keyword = 'predictor-hessian'
     spec( scale_constraints )%keyword = 'scale-constraints'
     spec( max_fails )%keyword = 'max-fails-before-monotone'

!  real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
     spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
     spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
     spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
     spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
     spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
     spec( stop_abs_i )%keyword = 'absolute-infeasiblity-tolerated'
     spec( stop_rel_i )%keyword = 'relative-infeasiblity-tolerated'
     spec( stop_predictor )%keyword = 'minimum-useful-predictor-decrease'
     spec( max_abs_i )%keyword = 'maximum-absolute-infeasibility'
     spec( max_rel_i )%keyword = 'maximum-relative-infeasibility'
     spec( min_constraint_scaling )%keyword                                    &
       = 'minimum-constraint-scaling-factor'
     spec( max_constraint_scaling )%keyword                                    &
       = 'maximum-constraint-scaling-factor'
     spec( min_hessian_perturbation )%keyword                                  &
       = 'minimum-predictor-hessian-perturbation'
     spec( sigma_0 )%keyword = 'initial-penalty-parameter'
     spec( sigma_inc )%keyword = 'minimum-penalty-parameter-increment'
     spec( radius_steering )%keyword = 'initial-steering-model-radius'
     spec( radius_accelerator )%keyword = 'initial-accelerator-model-radius'
     spec( tau_reduce )%keyword = 'step-balance-reduction-factor'
     spec( tau_min )%keyword = 'step-balance-minimum'
     spec( s_tiny )%keyword = 'tiny-step'
     spec( beta )%keyword = 'required-filter-margin-reduction'
     spec( gamma )%keyword = 'required-f-filter-reduction-vs-dlv'
     spec( eta_v )%keyword = 'required-v-filter-reduction-vs-dlv'
     spec( gamma_v )%keyword = 'required-v-reduction-vs-df'
     spec( gamma_f )%keyword = 'required-f-reduction-vs-df'
     spec( gamma_phi )%keyword = 'required-p-reduction-vs-dp'
     spec( eta_sigma )%keyword = 'required-dlphi-reduction-vs-dlv'
     spec( eta_phi )%keyword = 'required-dqphi-reduction-s-vs-pred'
     spec( alpha_reduce )%keyword = ' backtracking-linesearch-reduction-factor'
     spec( alpha_f_min )%keyword = 'cauchy-f-stepsize-minimum '
     spec( alpha_phi_min )%keyword = 'cauchy-phi-stepsize-minimum '
     spec( jacobian_zero_tolerance )%keyword = 'jacobian-zero-tolerance'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  logical key-words

     spec( full_solution )%keyword = 'print-full-solution'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
     spec( just_penalty )%keyword = 'just-use-the-penalty-function'
     spec( just_filter )%keyword = 'just-use-the-filter'
     spec( use_accelerator )%keyword = 'use-accelerator-step'
     spec( filter_uses_steering )%keyword = 'filter-uses-steering-information'

!  character key-words

     spec( alive_file )%keyword = 'alive-filename'
     spec( linear_solver_for_modifications )%keyword =                         &
        'linear-equation-solver-for-modifications'
     spec( prefix )%keyword = 'output-line-prefix'

!  read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  interpret the result

!  set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print  ),                          &
                                 control%stop_print ,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_fails ),                            &
                                 control%max_fails,                            &
                                 control%error )

!  set real values

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
     CALL SPECFILE_assign_value( spec( stop_abs_i ),                           &
                                 control%stop_abs_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_i ),                           &
                                 control%stop_rel_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_predictor ),                       &
                                 control%stop_predictor,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_abs_i ),                            &
                                 control%max_abs_i,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_rel_i ),                            &
                                 control%max_rel_i,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_constraint_scaling ),               &
                                 control%min_constraint_scaling,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_constraint_scaling ),               &
                                 control%max_constraint_scaling,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_hessian_perturbation ),             &
                                 control%min_hessian_perturbation,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( predictor_hessian ),                    &
                                 control%predictor_hessian,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_constraints ),                    &
                                 control%scale_constraints,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_steering ),                      &
                                 control%radius_steering,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_v ),                                &
                                 control%eta_v,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tau_reduce ),                           &
                                 control%tau_reduce,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tau_min ),                              &
                                 control%tau_min,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_sigma ),                            &
                                 control%eta_sigma,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_phi ),                              &
                                 control%eta_phi,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sigma_0 ),                              &
                                 control%sigma_0,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sigma_inc ),                            &
                                 control%sigma_inc,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( use_accelerator ),                      &
                                 control%use_accelerator,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_accelerator ),                   &
                                 control%radius_accelerator,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_f_min ),                          &
                                 control%alpha_f_min,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_phi_min ),                        &
                                 control%alpha_phi_min,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( beta ),                                 &
                                 control%beta,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma ),                                &
                                 control%gamma,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_v ),                              &
                                 control%gamma_v,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_f ),                              &
                                 control%gamma_f,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_phi ),                            &
                                 control%gamma_phi,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_reduce ),                         &
                                 control%alpha_reduce,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( s_tiny ),                               &
                                 control%s_tiny,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( filter_uses_steering ),                 &
                                 control%filter_uses_steering,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( jacobian_zero_tolerance ),              &
                                 control%jacobian_zero_tolerance,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  set logical values

     CALL SPECFILE_assign_value( spec( full_solution ),                        &
                                 control%full_solution,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( just_penalty ),                         &
                                 control%just_penalty,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( just_filter ),                          &
                                 control%just_filter,                          &
                                 control%error )

!  set character values

     CALL SPECFILE_assign_string( spec( alive_file ), control%alive_file,      &
                                  control%error )
     CALL SPECFILE_assign_value( spec( linear_solver_for_modifications ),      &
                                 control%linear_solver_for_modifications,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  read the controls for the sub-problem solvers and preconditioner

     IF ( PRESENT( alt_specname ) ) THEN

!  set FILTER control values

       CALL FILTER_read_specfile( control%FILTER_control, device,              &
             alt_specname = TRIM( alt_specname ) // '-FILTER' )

!  set QP control values

       CALL L1QP_read_specfile( control%QP_steer_control, device,              &
             alt_specname = TRIM( alt_specname ) // '-STEERING-QP' )
       CALL L1QP_read_specfile( control%QP_pred_control, device,               &
             alt_specname = TRIM( alt_specname ) // '-PREDICTOR-QP' )

!  set EQP control values

       CALL EQP_read_specfile( control%QP_accel_control, device,               &
             alt_specname = TRIM( alt_specname ) // '-ACCELERATOR-EQP' )

!  set LMS control values

       CALL LMS_read_specfile( control%LMS_control, device,                    &
             alt_specname = TRIM( alt_specname ) // '-LMS' )

!  set SLS control values

       CALL SLS_read_specfile( control%SLS_control, device,                    &
             alt_specname = TRIM( alt_specname ) // '-SLS' )

     ELSE

!  set FILTER control values

       CALL FILTER_read_specfile( control%FILTER_control, device )

!  set QP control values

       CALL L1QP_read_specfile( control%QP_steer_control, device,              &
             alt_specname = 'STEERING-QP' )
       CALL L1QP_read_specfile( control%QP_pred_control, device,               &
             alt_specname = 'PREDICTOR-QP' )

!  set EQP control values

       CALL EQP_read_specfile( control%QP_accel_control, device,               &
             alt_specname = 'ACCELERATOR-EQP' )

!  set LMS control values

       CALL LMS_read_specfile( control%LMS_control, device )

!  set SLS control values

       CALL SLS_read_specfile( control%SLS_control, device )
     END IF

     RETURN

!  End of subroutine FISQP_read_specfile

     END SUBROUTINE FISQP_read_specfile

!-*-*-*-  G A L A H A D -  F I S Q P _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE FISQP_solve( nlp, control, inform, data, userdata,             &
                             eval_FC, eval_GJ, eval_HL )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  FISQP_solve, a method for finding a local minimizer of a function subject
!  to general constraints and simple bounds on the sizes of the variables

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_FISQP.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_FISQP_double
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m is a scalar variable of type default integer, that holds the number of
!   constraints
!
!  H is scalar variable of type SMT_TYPE that holds the Hessian matrix H. The
!   following components are used here:
!
!   H%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of H%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of H%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme, the
!    first fourteen components of H%type must contain the string SPARSE_BY_ROWS,
!    and for the diagonal storage scheme, the first eight components of H%type
!    must contain the string DIAGONAL.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into H%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%H%type, 'COORDINATE' )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   H%ne is a scalar variable of type default integer, that holds the number of
!    entries in the lower triangular part of H in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the lower triangular part of H in the sparse
!    co-ordinate storage scheme. It need not be allocated for any of the other
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the lower triangular part of H in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!    be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of  each row of the lower
!    triangular part of H, as well as the total number of entries plus one,
!    in the sparse row-wise storage scheme. It need not be allocated when the
!    other schemes are used.
!
!  J is scalar variable of type SMT_TYPE that holds the Jacobian matrix J. The
!   following components are used here:
!
!   J%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of J%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of J%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme, and
!    the first fourteen components of J%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into J%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%J%type, 'COORDINATE' )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   J%ne is a scalar variable of type default integer, that holds the number of
!    entries in J in the sparse co-ordinate storage scheme. It need not be set
!    for any of the other two schemes.
!
!   J%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the Jacobian matrix J in any of the available
!    storage schemes.
!
!   J%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of J in the sparse co-ordinate storage scheme. It need not
!    be allocated for any of the other two schemes.
!
!   J%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of J in either the sparse co-ordinate,
!    or the sparse row-wise storage scheme. It need not be allocated when the
!    dense storage scheme is used.
!
!   J%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of each row of J, as well as
!    the total number of entries plus one, in the sparse row-wise storage
!    scheme. It need not be allocated when the other schemes are used.
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
!
!  C is a rank-one allocatable array of dimension n and type default real,
!   that holds the constraint value c. The i-th component of C, i = 1, ... , m,
!   contains g_j.
!
!  X is a rank-one allocatable array of dimension n and type default real, that
!   holds the values x of the optimization variables. The j-th component of
!   X, j = 1, ... , n, contains x_j.
!
!  pname is a scalar variable of type default character and length 10, which
!   contains the ``name'' of the problem for printing. The default ``empty''
!   string is provided.
!
!  VNAMES is a rank-one allocatable array of dimension n and type default
!   character and length 10, whose j-th entry contains the ``name'' of the j-th
!   variable for printing. This is only used  if ``debug''printing
!   control%print_level > 4) is requested, and will be ignored if the array is
!   not allocated.
!
! control is a scalar variable of type TRU_control_type. See TRU_initialize
!  for details
!
! inform is a scalar variable of type TRU_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will
!  have been set:
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
!
!     0. The run was succesful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending
!        array is written on unit control%error and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction nlp%n > 0 or requirement that nlp%H_type contains
!        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
!          or 'DIAGONAL' has been violated.
!    -5. The problem appears (locally) to have no feasible point.
!    -7. The objective function appears to be unbounded from below
!    -9. The analysis phase of the factorization failed; the return status
!        from the factorization package is given in the component
!        inform%factor_status
!   -10. The factorization failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -11. The solution of a set of linear equations using factors from the
!        factorization package failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -16. The problem is so ill-conditioned that further progress is impossible.
!   -18. Too many iterations have been performed. This may happen if
!        control%maxit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -19. Too much time has passed. This may happen if control%cpu_time_limit or
!        control%clock_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of the solver by removing the file
!        named control%alive_file from unit control%alive_unit.
!   -78. A problem evaluation error occurred.
!
!  -102. The predictor step could not improve the penalty function (but the
!        current iterate is feasible).
!  -103. A step could not be found in backtracking (pmode).
!  -104. A step could not be found in backtracking (fmode).
!  -105. A step could not be found in the steering subproblem.
!  -106. A step could not be found in the predictor subproblem.
!  -108. The steering step did not achieve its purpose.
!  -109. tau linesearch computation failed.
!  -111. There are no general constraints.
!
!     2. The user should compute the objective function value f(x) and the
!        constraint function values c(x) at the point x indicated in nlp%X
!        and then re-enter the subroutine. The required values should be set in
!        nlp%f and nlp%C respectively, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) and/or c(x)  - for instance, if
!        any of the functions is undefined at x - the user need not set nlp%f
!        or nlp%C, but should then set data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and the Jacobian of
!        the constraints nabla_x c(x) and then re-enter the subroutine. The
!        value of the i-th component of the gradient should be set in nlp%G(i),
!        for i = 1, ..., n, while the nonzeros of the Jacobian should be set
!        in nlp%J%val in the same order as in the storage scheme already
!        established in nlp%J,, and data%eval_status should be set to 0. If the
!        user is unable to evaluate any of the components of the gradient or
!        Jacobian - for instance if a component of the gradient or Jacobian
!        is undefined at x - the user need not set nlp%G or nlp%J%val, but
!        should then set data%eval_status to a non-zero value.
!     4. The user should compute the Hessian of the Lagrangian function
!        nabla_xx f(x) - sum_i=1^m y_i c_i(x) at the point x indicated in nlp%X
!        and y in nlp%Y and then re-enter the subroutine. The nonzeros of the
!        Hessian should be set in nlp%H%val in the same order as in the storage
!        scheme already established in nlp%H, and data%eval_status should be
!        set to 0. If the user is unable to evaluate a component of the Hessian
!        - for instance, if a component of the Hessian is undefined at x - the
!        user need not set nlp%H%val, but should then set data%eval_status to
!        a non-zero value.
!     5. The user should compute each of the gradient, Jacobian and Hessian
!        as described in 3 and 4 above, and then re-enter the subroutine
!        with data%eval_status set to 0. If the user is unable to evaluate
!        any of this data, nlp%G, nlp%J%val and nlp%H%val need not be set but
!        then data%eval_status should be set to a non-zero value.
!     6. The user should compute the product
!        ( nabla_xx f(x) - sum_i=1^m y_i c_i(x) ) v of the Hessian of the
!        Lagrangian function nabla_xx f(x) - sum_i=1^m y_i c_i(x) at the point
!        x indicated in nlp%X with the vector v, and add the result to the
!        vector u and then re-enter the subroutine. The vectors u and v are
!        given in data%U and data%V respectively, the resulting vector u +
!        nabla_xx f(x)v should be set in data%U and  data%eval_status should
!        be set to 0. If the user is unable to evaluate the product - for
!        instance, if a component of the Hessian is undefined at x - the user
!        need not alter data%U, but should then set data%eval_status to a
!        non-zero value.
!
!  alloc_status is a scalar variable of type default integer, that gives
!   the status of the last attempted array allocation or deallocation.
!   This will be 0 if status = 0.
!
!  bad_alloc is a scalar variable of type default character
!   and length 80, that  gives the name of the last internal array
!   for which there were allocation or deallocation errors.
!   This will be the null string if status = 0.
!
!  iter is a scalar variable of type default integer, that holds the
!   number of iterations performed.
!
!  cg_iter is a scalar variable of type default integer, that gives the
!   total number of conjugate-gradient iterations required.
!
!  fc_eval is a scalar variable of type default integer, that gives the
!   total number of objective and constraint function evaluations performed.
!
!  gj_eval is a scalar variable of type default integer, that gives the
!   total number of objective gradient and constraint Jacobian evaluations
!   performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of Lagrangian Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type TRU_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    analyse is a scalar variable of type default real, that gives
!      the CPU time spent analysing required matrices prior to factorization.
!
!    factorize is a scalar variable of type default real, that gives
!      the CPU time spent factorizing the required matrices.
!
!    solve is a scalar variable of type default real, that gives
!     the CPU time spent using the factors to solve relevant linear equations.
!
!    clock_total is a scalar variable of type default real, that gives
!     the total clock time spent in the package.
!
!    clock_analyse is a scalar variable of type default real, that gives
!      the clock time spent analysing required matrices prior to factorization.
!
!    clock_factorize is a scalar variable of type default real, that gives
!      the clock time spent factorizing the required matrices.
!
!    clock_solve is a scalar variable of type default real, that gives
!     the clock time spent using the factors to solve relevant linear equations.
!
!  data is a scalar variable of type TRU_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be 
!   used to pass user data to and from the eval_* subroutines (see below)
!   Available coomponents which may be allocated as required are:
!
!    integer is a rank-one allocatable array of type default integer.
!    real is a rank-one allocatable array of type default real
!    complex is a rank-one allocatable array of type default comple.
!    character is a rank-one allocatable array of type default character.
!    logical is a rank-one allocatable array of type default logical.
!    integer_pointer is a rank-one pointer array of type default integer.
!    real_pointer is a rank-one pointer array of type default  real
!    complex_pointer is a rank-one pointer array of type default complex.
!    character_pointer is a rank-one pointer array of type default character.
!    logical_pointer is a rank-one pointer array of type default logical.
!
!  eval_FC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If C is present, the values of the constraint functions
!   c(x) evaluated at x=X must be returned in C, and the status variable set
!   to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_FC is not present, FISQP_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_GJ is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). If G is present, the components of
!   the gradient nabla_x f(x) of the objective function evaluated at x=X
!   must be returned in G. If GJ is present, the nonzeros of the Jacobian
!   nabla_x c(x) evaluated at x=X must be returned in J_val in the same
!   order as presented in nlp%J, and the status variable set to 0.
!   If the evaluation is impossible at x=X, status should be set to a
!   nonzero value. If eval_GJ is not present, FISQP_solve will return to the
!   user with inform%status = 3 or 5 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HL is not present,
!   FISQP_solve will return to the user with inform%status = 4 or 5 each time
!   an evaluation is required.
!
!  eval_HLPROD is an optional subroutine which if present must have
!   the arguments given below (see the interface blocks). The sum
!   u + nabla_xx ( f(x) - sum_i=1^m y_i c_i(x) ) v of the product of the Hessian
!   nabla_xx f(x) + sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y with the vector v=V and the vector u=U must be returned in U,
!   and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_HPROD is not present,
!   FISQP_solve will return to the user with inform%status = 6 each time an
!   evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( FISQP_control_type ), INTENT( INOUT ) :: control
     TYPE ( FISQP_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( FISQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_GJ, eval_HL

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_FC( status, X, userdata, f, C )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ) :: f
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_GJ( status, X, userdata, G, Jval )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G, Jval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HL( status, X, Y, userdata, Hval, no_f )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL
     END INTERFACE

!  ==========================================================================
!
!  Problem -
!
!    minimize   f(x) subject to  c_l <= c(x) <= c_u and  x_l <= x <= x_u
!
!  where c_l and c_u are vectors of lower and upper bounds on the constraint
!  vector-valued function c( x ), and x_l and x_u are vectors of lower and
!  upper bounds on the primal variables x.
!
!  Main algorithm -
!
!  Filter SQP algorithm described by Algorithm 1 in
!
!  Gould, Loh and Robinson
!   "A nonmonotne Filter SQP method: local convergence and numerical results"
!   SIAM J. Optimization 25(3) 1885-1911 (2015)
!
!  Corresponding step n in the algorithm marked as A1:n in comments below
!
!  ==========================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ic, ii, ir, j, l, ne_eqp, s_start, s_type, active
!    INTEGER :: Rk
     REAL ( KIND = wp ) :: complementary_slackness, multiplier_norm
     REAL ( KIND = wp ) :: ellv_pred, ellv_steer, del_ellv_pred, del_ellv_steer
     REAL ( KIND = wp ) :: fil_f, fil_viol, gts, sths, rho_f, rho_phi
     REAL ( KIND = wp ) :: max_viol, new_sigma, new_sigma_denom, relaxed_viol
     REAL ( KIND = wp ) :: alpha, alpha_cf, del_qf_pred, del_qf_cf, si, val
     REAL ( KIND = wp ) :: delta, dgtdg, dxtdg, dual_infeasibility, radius

!    REAL ( KIND = wp ) :: num_c_act, num_c_free, num_x_free
!    REAL ( KIND = wp ) :: C_l_act, C_u_act, c_act, c_free
!    REAL ( KIND = wp ) :: X_l_act, X_u_act, x_act, x_free
!    REAL ( KIND = wp ) :: K, rhs, sol, g_old, Jx_old, step
!    REAL ( KIND = wp ) :: B_k, B_k_scalar, Bk_D, Hess, Hess_D, Heps
!    REAL ( KIND = wp ) :: normg, normHess, muLS, accel_bd_viol
     LOGICAL :: kkt_accel, kkt_pred, lin_feas, acceptable, names
     CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$   INTEGER :: OMP_GET_MAX_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 900
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective and constraint evaluation
       GO TO 20
     CASE ( 130 ) ! gradient and Jacobian evaluation
       GO TO 130
     CASE ( 140 ) ! Hessian evaluation
       GO TO 140
!    CASE ( 150 ) ! Hessian-vector product
!      GO TO 150
     CASE ( 210 ) ! objective and constraint evaluation
       GO TO 210
     CASE ( 320 ) ! objective and constraint evaluation
       GO TO 320
!    CASE ( 570 ) ! gradient and Jacobian evaluation
!      GO TO 570
!    CASE ( 580 ) ! Hessian evaluation
!      GO TO 580
!    CASE ( 590 ) ! objective and constraint evaluation
!      GO TO 590
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE

! ------------------------------------------------------------------------
!                            INITIALIZATION (A1:1-2)
! ------------------------------------------------------------------------

     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
!$   inform%threads = OMP_GET_MAX_THREADS( )
     inform%status = GALAHAD_ok
     inform%alloc_status = 0 ; inform%bad_alloc = ''
     inform%iter = 0
     inform%fc_eval = 0 ; inform%gj_eval = 0 ; inform%h_eval = 0

     inform%obj = HUGE( one )

!  copy control parameters so that the package may alter values if necessary

     data%control = control

!  ensure that control parameters are sensible

     IF ( data%control%predictor_hessian == powell_l_bfgs_predictor_hessian )  &
       data%control%predictor_hessian = l_bfgs_predictor_hessian
     IF ( data%control%predictor_hessian /= se_modified_predictor_hessian      &
         .AND. data%control%predictor_hessian /= dd_modified_predictor_hessian &
         .AND. data%control%predictor_hessian /= l_bfgs_predictor_hessian )    &
       data%control%predictor_hessian = identity_predictor_hessian

!  decide how much reverse communication is required

     data%reverse_fc = .NOT. PRESENT( eval_FC )
     data%reverse_gj = .NOT. PRESENT( eval_GJ )
     data%reverse_hl = .NOT. PRESENT( eval_HL )
!    data%reverse_hlprod = .NOT. PRESENT( eval_HLPROD )

!  control the output printing

     data%out = data%control%out ; data%error = data%control%error
     data%print_level_eqp = data%control%QP_accel_control%print_level
     data%print_level_eqp_sbls                                                 &
       = data%control%QP_accel_control%SBLS_control%print_level
     data%print_level_eqp_gltr                                                 &
       = data%control%QP_accel_control%GLTR_control%print_level

!  error output

     data%printe = data%error > 0 .AND. data%control%print_level >= 1

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printm, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. data%control%print_level >= 5

!  print level shorthands

     IF ( data%control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = data%control%start_print
     END IF

     IF ( data%control%stop_print < 0 ) THEN
       data%stop_print = data%control%maxit + 1
     ELSE
       data%stop_print = data%control%stop_print
     END IF

!  set print agenda for the first iteration

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE.
       data%printd = .FALSE. ; data%print_level = 0
     END IF

     data%print_iteration_header = data%print_level > 0
     data%print_1st_header = .TRUE.

     IF ( data%printd ) WRITE( data%out, "( ' (A1:1-2)' )" )

!  initialize the filter

     CALL FILTER_initialize_filter( data%FILTER_data,                          &
                                    data%control%FILTER_control,               &
                                    inform%FILTER_inform )
     IF ( inform%FILTER_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_inform%status
       inform%alloc_status = inform%FILTER_inform%alloc_status
       GO TO 900
     END IF

!    IF ( nlp%m == 0 ) THEN
!      inform%status = - 111
!      GO TO 900
!    END IF

!  initialize counters

!    inform%B_modified = .FALSE.

!  initialize parameters

!    muLS = 1.0e-8  ;  ! regularization parameter for LS multipliers
     data%sigma = data%control%sigma_0
     data%sigma_new_ref = data%control%sigma_0
!    B_k = eye_n

!  set up successful set

     data%success_iter = 0
     data%fails = 0
     data%radius_accelerator = data%control%radius_accelerator

!  do not allow "neither penalty nor filter" mode

     IF ( data%control%just_penalty .AND. data%control%just_filter ) THEN
       WRITE( data%out, "( 'Please pick a mode (filter/penalty).' )" )
       inform%status = - 112
       RETURN
     END IF

!  only in penalty mode

     IF ( data%control%just_penalty ) THEN
       data%p_mode = .TRUE.
       data%check_filter = .FALSE.
       inform%entered_penalty = 0
     ELSE
       data%p_mode = .FALSE.
       data%check_filter = .TRUE.
     END IF

!  only in filter mode

     IF ( data%control%just_filter ) data%p_mode = .FALSE.

! -----------------------------------------------------------------------------
!                            SET UP THE PROBLEM
! -----------------------------------------------------------------------------

!  project x to ensure feasibility

     nlp%X = MAX( nlp%X_l, MIN( nlp%X, nlp%X_u ) )

!  evaluate the objective and general constraint function values

     IF ( data%reverse_fc ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_FC'
         inform%status = GALAHAD_error_evaluation ; GO TO 900
       END IF
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%obj = nlp%f
     inform%fc_eval = inform%fc_eval + 1

!  print problem name and header, if requested

     IF ( data%printi ) WRITE( data%out,                                       &
         "( A, ' +', 76( '-' ), '+', /,                                        &
      &     A, 18X, 'Filter Sequential Quadratic Programming', /,              &
      &     A, ' +', 76( '-' ), '+', /, A,                                     &
      &     6X, 'mode (F=filter,P=penalty) pair (v,o,b,p) step ',              &
      &     '(p=pred,a=accel)' )" ) prefix, prefix, prefix, prefix

!  determine the number of nonzeros in the Hessian

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%H_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%H_ne = nlp%H%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%H_ne = ( nlp%n * ( nlp%n+ 1 ) ) / 2
     END SELECT

!  determine the number of nonzeros in the constraint Jacobian

     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%J_ne = nlp%J%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%J_ne = nlp%J%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%J_ne = nlp%m * nlp%n
     END SELECT

!  set up space for general vectors used

     array_name = 'fisqp: nlp%gL'
     CALL SPACE_resize_array( nlp%n, nlp%gL,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: nlp%X_status'
     CALL SPACE_resize_array( nlp%n, nlp%X_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: nlp%C_status'
     CALL SPACE_resize_array( nlp%m, nlp%C_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%S'
     CALL SPACE_resize_array( nlp%n, data%S,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%S_ref'
     CALL SPACE_resize_array( nlp%n, data%S_ref,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%S_pred'
     CALL SPACE_resize_array( nlp%n, data%S_pred,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     IF ( data%control%use_accelerator ) THEN
       array_name = 'fisqp: data%S_accel'
       CALL SPACE_resize_array( nlp%n, data%S_accel,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%S_accel_ref'
       CALL SPACE_resize_array( nlp%n, data%S_accel_ref,                       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

     array_name = 'fisqp: data%X_ref'
     CALL SPACE_resize_array( nlp%n, data%X_ref,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%X_trial'
     CALL SPACE_resize_array( nlp%n, data%X_trial,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%C_trial'
     CALL SPACE_resize_array( nlp%m, data%C_trial,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%Y_accel'
     CALL SPACE_resize_array( nlp%m, data%Y_accel,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%Z_accel'
     CALL SPACE_resize_array( nlp%n, data%Z_accel,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%WORK_m'
     CALL SPACE_resize_array( nlp%m, data%WORK_m,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%WORK_n'
     CALL SPACE_resize_array( nlp%n, data%WORK_n,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     IF ( nlp%m > 0 ) THEN
       array_name = 'fisqp: data%S_steer'
       CALL SPACE_resize_array( nlp%n, data%S_steer,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

!  initialize values at R(k) when k = 0

     data%X_ref = nlp%X
     data%f_ref = nlp%f
     data%accepted = .TRUE.

!  set up space for the steering LP

     IF ( nlp%m > 0 ) THEN
       array_name = 'fisqp: data%QP_steer%G'
       CALL SPACE_resize_array( nlp%n, data%QP_steer%G,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%H%row'
       CALL SPACE_resize_array( 0, data%QP_steer%H%row,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%H%col'
       CALL SPACE_resize_array( 0, data%QP_steer%H%col,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%H%val'
       CALL SPACE_resize_array( 0, data%QP_steer%H%val,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%A%row'
       CALL SPACE_resize_array( data%J_ne, data%QP_steer%A%row,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%A%col'
       CALL SPACE_resize_array( data%J_ne, data%QP_steer%A%col,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%A%val'
       CALL SPACE_resize_array( data%J_ne, data%QP_steer%A%val,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%C_l'
       CALL SPACE_resize_array( nlp%m, data%QP_steer%C_l, inform%status,       &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%C_u'
       CALL SPACE_resize_array( nlp%m, data%QP_steer%C_u, inform%status,       &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%X_l'
       CALL SPACE_resize_array( nlp%n, data%QP_steer%X_l,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%X_u'
       CALL SPACE_resize_array( nlp%n, data%QP_steer%X_u,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%X'
       CALL SPACE_resize_array( nlp%n, data%QP_steer%X,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%C'
       CALL SPACE_resize_array( nlp%m, data%QP_steer%C, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

!** next arrays only needed for current lpqp

!      array_name = 'fisqp: data%QP_steer%H%ptr'
!      CALL SPACE_resize_array( nlp%n + 1, data%QP_steer%H%ptr,                &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = data%control%deallocate_error_fatal,    &
!             exact_size = data%control%space_critical,                        &
!             bad_alloc = inform%bad_alloc, out = data%error )
!      IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%A%ptr'
       CALL SPACE_resize_array( nlp%n + 1, data%QP_steer%A%ptr,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%Y'
       CALL SPACE_resize_array( nlp%m, data%QP_steer%Y,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_steer%Z'
       CALL SPACE_resize_array( nlp%n, data%QP_steer%Z,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set up the structural data for the steering LP

       data%QP_steer%new_problem_structure = .TRUE.
       data%QP_steer%n = nlp%n ; data%QP_steer%m = nlp%m
       data%QP_steer%A%n = nlp%n ; data%QP_steer%A%m = nlp%m

!  ** to do - update lpqp to allow other Hessian/gradient types **
!    data%QP_steer%gradient_kind = 0
!      CALL SMT_put( data%QP_steer%H%type, 'NONE', inform%alloc_status )
!      CALL SMT_put( data%QP_steer%H%type, 'COORDINATE', inform%alloc_status )
!      data%QP_steer%H%n = nlp%n ; data%QP_steer%H%m = nlp%n
!      data%QP_steer%H%ne = 0

       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'COORDINATE' )
         data%QP_steer%A%ne = nlp%J%ne
         IF ( data%QP_steer%A%ne > 0 ) THEN
           data%QP_steer%A%row( : data%QP_steer%A%ne )                         &
             = nlp%J%row( : data%QP_steer%A%ne )
           data%QP_steer%A%col( : data%QP_steer%A%ne )                         &
             = nlp%J%col( : data%QP_steer%A%ne )
         END IF
       CASE ( 'SPARSE_BY_ROWS' )
         data%QP_steer%A%ne = nlp%J%ptr( nlp%m + 1 ) - 1
         IF ( data%QP_steer%A%ne > 0 ) THEN
           DO i = 1, nlp%m
             data%QP_steer%A%row( nlp%J%ptr( i ) : nlp%J%ptr( i + 1 ) - 1 ) = i
           END DO
           data%QP_steer%A%col( : data%QP_steer%A%ne )                         &
             = nlp%J%col( : data%QP_steer%A%ne )
         END IF
       CASE ( 'DENSE' )
         data%QP_steer%A%ne = 0
         DO i = 1, nlp%m
           DO j = 1, nlp%n
             data%QP_steer%A%ne = data%QP_steer%A%ne + 1
             data%QP_steer%A%row( data%QP_steer%A%ne ) = i
             data%QP_steer%A%col( data%QP_steer%A%ne ) = j
           END DO
         END DO
       END SELECT
       CALL SMT_put( data%QP_steer%A%type, 'COORDINATE', inform%alloc_status )
     END IF

!  set up space for the predictor QP

     array_name = 'fisqp: data%QP_pred%G'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%G,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%A%row'
     CALL SPACE_resize_array(  data%J_ne, data%QP_pred%A%row,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%A%col'
     CALL SPACE_resize_array( data%J_ne, data%QP_pred%A%col,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%A%val'
     CALL SPACE_resize_array( data%J_ne, data%QP_pred%A%val,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%C_l'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%C_l, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%C_u'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%C_u, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%X_l'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%X_l,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%X_u'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%X_u,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%X'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%X,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%Y_l'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%Y_l,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%Y_u'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%Y_u,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%Z_l'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%Z_l,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%Z_u'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%Z_u,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%C'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%C, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     IF ( data%control%predictor_hessian == l_bfgs_predictor_hessian ) THEN
       array_name = 'fisqp: data%DX'
       CALL SPACE_resize_array( nlp%n, data%DX,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%DG'
       CALL SPACE_resize_array( nlp%n, data%DG,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%DY'
       CALL SPACE_resize_array( nlp%m, data%DY,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%DZ'
       CALL SPACE_resize_array( nlp%n, data%DZ,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

     IF ( data%control%scale_constraints > 0 ) THEN
       array_name = 'fisqp: data%C_scale'
       CALL SPACE_resize_array( nlp%m, data%C_scale,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
     END IF

!** next arrays only needed for current lpqp

!    array_name = 'fisqp: data%QP_pred%H%ptr'
!    CALL SPACE_resize_array( nlp%n + 1, data%QP_pred%H%ptr,                   &
!           inform%status, inform%alloc_status, array_name = array_name,       &
!           deallocate_error_fatal = data%control%deallocate_error_fatal,      &
!           exact_size = data%control%space_critical,                          &
!           bad_alloc = inform%bad_alloc, out = data%error )
!    IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%A%ptr'
     CALL SPACE_resize_array( nlp%n + 1, data%QP_pred%A%ptr,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%u'
     CALL SPACE_resize_array( nlp%m, data%QP_pred%Y,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fisqp: data%QP_pred%Z'
     CALL SPACE_resize_array( nlp%n, data%QP_pred%Z,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set up the structural data for the predictor QP

     data%QP_pred%new_problem_structure = .TRUE.
     data%QP_pred%n = nlp%n ; data%QP_pred%m = nlp%m

!  set up the structural Jacobian data for the predictor QP

     data%QP_pred%A%n = nlp%n ; data%QP_pred%A%m = nlp%m
     IF ( nlp%m > 0 ) THEN
       data%QP_pred%A%ne = data%QP_steer%A%ne
       data%QP_pred%A%row( : data%QP_pred%A%ne )                               &
         = data%QP_steer%A%row( : data%QP_steer%A%ne )
       data%QP_pred%A%col( : data%QP_pred%A%ne )                               &
         = data%QP_steer%A%col( : data%QP_steer%A%ne )
       CALL SMT_put( data%QP_pred%A%type, 'COORDINATE', inform%alloc_status )
     ELSE
       data%QP_pred%A%ne = 0
       CALL SMT_put( data%QP_pred%A%type, 'COORDINATE', inform%alloc_status )
     END IF

!  set up the structural Hessian data for the predictor QP

     SELECT CASE( data%control%predictor_hessian )

!  if a scaled-identity predictor Hessian is required, used the
!  scaled-identity matrix format

     CASE( identity_predictor_hessian )
       array_name = 'fisqp: data%QP_pred%H%val'
       CALL SPACE_resize_array( 1, data%QP_pred%H%val,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
       CALL SMT_put( data%QP_pred%H%type, 'SCALED_IDENTITY',                   &
                     inform%alloc_status )

!  if a modified-exact-Hessian predictor Hessian is required, simply use the
!  storage required for the exact Hessian

     CASE( se_modified_predictor_hessian, dd_modified_predictor_hessian )
       data%QP_pred%H%ne = data%H_ne + nlp%n
       array_name = 'fisqp: data%QP_pred%H%row'
       CALL SPACE_resize_array( data%QP_pred%H%ne, data%QP_pred%H%row,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_pred%H%col'
       CALL SPACE_resize_array( data%QP_pred%H%ne, data%QP_pred%H%col,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_pred%H%val'
       CALL SPACE_resize_array( data%QP_pred%H%ne, data%QP_pred%H%val,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

!  copy the row and column indices from the true Hessian

       data%QP_pred%H%n = nlp%n ; data%QP_pred%H%ne = data%H_ne
       SELECT CASE ( SMT_get( nlp%H%type ) )
       CASE ( 'COORDINATE' )
         data%QP_pred%H%ne = nlp%H%ne
         data%QP_pred%H%row( : data%QP_pred%H%ne ) = nlp%H%row
         data%QP_pred%H%col( : data%QP_pred%H%ne ) = nlp%H%col
       CASE ( 'SPARSE_BY_ROWS' )
         data%QP_pred%H%ne = nlp%H%ptr( nlp%m + 1 ) - 1
         DO i = 1, nlp%n
           data%QP_pred%H%row( nlp%H%ptr( i ) : nlp%H%ptr( i + 1 ) - 1 ) = i
         END DO
         data%QP_pred%H%col( : data%QP_pred%H%ne ) = nlp%H%col
       CASE ( 'DENSE' )
         data%QP_pred%H%ne = 0
         DO i = 1, nlp%n
           DO j = 1, i
             data%QP_pred%H%ne = data%QP_pred%H%ne + 1
             data%QP_pred%H%row( data%QP_pred%H%ne ) = i
             data%QP_pred%H%col( data%QP_pred%H%ne ) = j
           END DO
         END DO
       END SELECT

!  provide space for the diagonal modifications

       DO i = 1, nlp%n
         data%QP_pred%H%row( data%QP_pred%H%ne + i ) = i
         data%QP_pred%H%col( data%QP_pred%H%ne + i ) = i
       END DO
       CALL SMT_put( data%QP_pred%H%type, 'COORDINATE', inform%alloc_status )
!write(6,*) data%QP_pred%H%ne + nlp%n
!do i = 1,  data%QP_pred%H%ne + nlp%n
!  write(6,*) data%QP_pred%H%row(i), data%QP_pred%H%col(i)
!end do

!  set the data structures to accommodate the factors of the modfified H

       data%control%SLS_control%prefix = '" - SLS:"                     '
       CALL SLS_initialize( data%control%linear_solver_for_modifications,      &
          data%SLS_data, data%control%SLS_control, inform%SLS_inform )
       IF ( inform%SLS_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_analysis ; GO TO 900
       END IF

       data%control%SLS_control%pivot_control = 2
       IF ( data%control%predictor_hessian == se_modified_predictor_hessian )  &
         data%control%SLS_control%pivot_control = 4

       CALL SLS_analyse( data%QP_pred%H, data%SLS_data,                        &
                         data%control%SLS_control, inform%SLS_inform )
       IF ( inform%SLS_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_analysis ; GO TO 900
       END IF

!  if a limited memory-based secant approximation of the Hessian is required,
!  set up the storage required

     CASE( l_bfgs_predictor_hessian )
       CALL SMT_put( data%QP_pred%H%type, 'LBFGS', inform%alloc_status )
       data%control%LMS_control%method = 1
       CALL LMS_setup( nlp%n, data%QP_pred%H_lm, data%control%LMS_control,     &
                       inform%LMS_inform )
     END SELECT

!  if necessary, set up space for the accelerator EQP

     IF ( data%control%use_accelerator ) THEN

       array_name = 'fisqp: data%QP_accel%H%row'
       CALL SPACE_resize_array( data%H_ne, data%QP_accel%H%row,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%H%col'
       CALL SPACE_resize_array( data%H_ne, data%QP_accel%H%col,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%H%val'
       CALL SPACE_resize_array( data%H_ne, data%QP_accel%H%val,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%G'
       CALL SPACE_resize_array( nlp%n, data%QP_accel%G,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       Ne_eqp = data%J_ne + nlp%n
       array_name = 'fisqp: data%QP_accel%A%row'
       CALL SPACE_resize_array( ne_eqp, data%QP_accel%A%row,                   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%A%col'
       CALL SPACE_resize_array( ne_eqp, data%QP_accel%A%col,                   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%A%val'
       CALL SPACE_resize_array( ne_eqp, data%QP_accel%A%val,                   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%X'
       CALL SPACE_resize_array( nlp%n, data%QP_accel%X,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%C'
       CALL SPACE_resize_array( nlp%n + nlp%m, data%QP_accel%C,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'fisqp: data%QP_accel%Y'
       CALL SPACE_resize_array( nlp%n + nlp%m, data%QP_accel%Y,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set up the structural data for the accelerator EQP

       data%QP_accel%n = nlp%n
       data%QP_accel%H%n = nlp%n
       data%QP_accel%H%ne = data%H_ne
       SELECT CASE ( SMT_get( nlp%H%type ) )
       CASE ( 'COORDINATE' )
         data%QP_accel%H%ne = nlp%H%ne
         data%QP_accel%H%row( : data%QP_accel%H%ne ) = nlp%H%row
         data%QP_accel%H%col( : data%QP_accel%H%ne ) = nlp%H%col
       CASE ( 'SPARSE_BY_ROWS' )
         data%QP_accel%H%ne = nlp%H%ptr( nlp%m + 1 ) - 1
         DO i = 1, nlp%n
           data%QP_accel%H%row( nlp%H%ptr( i ) : nlp%H%ptr( i + 1 ) - 1 ) = i
         END DO
         data%QP_accel%H%col( : data%QP_accel%H%ne ) = nlp%H%col
       CASE ( 'DENSE' )
         data%QP_accel%H%ne = 0
         DO i = 1, nlp%n
           DO j = 1, i
             data%QP_accel%H%ne = data%QP_accel%H%ne + 1
             data%QP_accel%H%row( data%QP_accel%H%ne ) = i
             data%QP_accel%H%col( data%QP_accel%H%ne ) = j
           END DO
         END DO
       END SELECT
       CALL SMT_put( data%QP_accel%H%type, 'COORDINATE', inform%alloc_status )
       data%QP_accel%A%n = nlp%n
       CALL SMT_put( data%QP_accel%A%type, 'COORDINATE', inform%alloc_status )
     END IF

!  ----------------------------------------------------------------------------
!                           START OF MAIN LOOP (A1:3)
!  ----------------------------------------------------------------------------

     IF ( data%printd ) WRITE( data%out, "( A, ' (A1:3)' )" ) prefix
     data%exit_small_s = .FALSE.

!    DO
  50   CONTINUE
!      IF ( inform%status == GALAHAD_ok ) GO TO 900

       IF ( data%printd ) THEN
         WRITE( data%out, "( A, ' X ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%X( : nlp%n )
         WRITE( data%out, "( A, ' C ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%C( : nlp%m )
       END IF

       IF ( data%p_mode ) THEN
         data%it_type = 'P'
       ELSE
         data%it_type = 'F'
       END IF

!  revert to the last successful point if too many unsuccessful iterations
!  have occurred (A1:4-5)

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:4-5)' )" ) prefix
       IF ( data%fails > data%control%max_fails ) THEN
         nlp%f = data%f_ref
         nlp%X = data%X_ref
         data%S = data%S_ref
         IF ( data%control%use_accelerator ) data%S_accel = data%S_accel_ref
         data%primal_viol = data%primal_viol_ref
!write(6,"( 'viol reset a', ES12.4 )") data%primal_viol
!        data%comp_viol = data%comp_viol_ref

         IF ( data%printi ) THEN
           IF ( data%print_iteration_header .OR.                               &
                data%print_1st_header) WRITE( data%out, 2000 ) prefix
           data%print_1st_header = .FALSE.
           data%print_iteration_header = .FALSE.
           CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
           data%time_now = data%time_now - data%time_start
           data%clock_now = data%clock_now - data%clock_start
           IF ( inform%iter > 0 ) THEN
             WRITE( data%out, 2010 )                                           &
               prefix, inform%iter, data%it_type, data%pair_type,              &
               data%step_used( 1 : 1 ), inform%obj,                            &
               inform%primal_infeasibility, inform%dual_infeasibility,         &
               inform%complementary_slackness,                                 &
               ADJUSTR( STRING_integer_6( data%QP_accel%A%m ) ),               &
               data%s_norm, data%d_type,                                       &
               inform%FILTER_inform%filter_size, data%sigma, data%clock_now
           ELSE
             WRITE( data%out, 2020 ) prefix, inform%iter,                      &
               inform%obj, inform%primal_infeasibility,                        &
               inform%dual_infeasibility, inform%complementary_slackness,      &
               inform%FILTER_inform%filter_size, data%sigma, data%clock_now
           END IF

!          IF ( mod( inform%iter, 50 ) == 0 ) WRITE( data%out, 2100 )
!          WRITE( data%out, 2110 ) inform%iter, data%p_mode, nlp%f,            &
!            data%primal_viol, data%comp_viol, data%sigma
         END IF
         data%pair_type = ' '
         GO TO 170
       END IF

!  if required, record gL - J^T dy - dz for the limited-memory predictor Hessian

       IF ( data%accepted ) THEN
         IF ( data%control%predictor_hessian == l_bfgs_predictor_hessian .AND. &
              inform%iter > 0 ) THEN
           data%DG = nlp%gL - data%DZ
           CALL mop_AX( - one, nlp%J, data%DY, one, nlp%gL,                    &
                    transpose = .TRUE.,  m_matrix = nlp%m, n_matrix = nlp%n )
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
         END IF

!  obtain the gradient of the objective function and the Jacobian
!  of the constraints. The data is stored in a sparse format

         inform%gj_eval = inform%gj_eval + 1
         IF ( data%reverse_gj ) THEN
           data%branch = 130 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_GJ( data%eval_status, nlp%X, userdata, nlp%G, nlp%J%val )
           IF ( data%eval_status /= 0 ) THEN
             inform%bad_eval = 'eval_GJ'
             inform%status = GALAHAD_error_evaluation ; GO TO 900
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the gradient and Jacobian

  130  CONTINUE

!  scale the constraints if required

       IF ( inform%iter == 0 .AND. data%control%scale_constraints > 0 ) THEN

!  compute the largest entry in each row of the Jacobian

         SELECT CASE ( SMT_get( nlp%J%type ) )
         CASE ( 'COORDINATE' )
           data%C_scale = zero
           DO l = 1, nlp%J%ne
             data%C_scale( nlp%J%row( l ) ) =                                  &
               MAX( data%C_scale( nlp%J%row( l ) ), ABS( nlp%J%val( l ) ) )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, nlp%m
             data%C_scale( i ) = MAXVAL(                                       &
               ABS( nlp%J%val( nlp%J%ptr( i ) : nlp%J%ptr( i + 1 ) - 1 ) ) )
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, nlp%m
             data%C_scale( i ) = MAXVAL( ABS( nlp%J%val( l + 1 : l + nlp%n ) ) )
             l = l + nlp%n
           END DO
         END SELECT

!  scale by the largest entries, constrained to a safe interval

         data%C_scale( : nlp%m ) = MIN( data%control% max_constraint_scaling,  &
           MAX( data%C_scale( : nlp%m ), data%control% min_constraint_scaling ))

         IF ( data%control%scale_constraints > 1 ) THEN
           val = MAXVAL( data%C_scale( : nlp%m ) )
           data%C_scale( : nlp%m ) = data%C_scale( : nlp%m ) / val
         END IF

         write(6,"( ' c_scale ', /, ( 5ES12.4 ) )" ) data%C_scale( : nlp%m )

!  scale c and its bounds

         nlp%C( : nlp%m ) = nlp%C( : nlp%m ) / data%C_scale( : nlp%m )
         WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )                &
           nlp%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) / data%C_scale( : nlp%m )
         WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )                  &
           nlp%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) / data%C_scale( : nlp%m )
       END IF

!  scale the Jacobian if required

!         CALL mop_row_infinity_norms(  nlp%J, data%WORK_m, .FALSE., 6, 6, 0 )
!write(6,"( ' J_row_norms', /, ( 5ES12.4 ) )" )   data%WORK_m( : nlp%m )

       IF ( data%accepted .AND. data%control%scale_constraints > 0 ) THEN
!write(6,"( ' c_scale ', /, ( 5ES12.4 ) )" ) data%C_scale( : nlp%m )

         SELECT CASE ( SMT_get( nlp%J%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, nlp%J%ne
             nlp%J%val( l ) = nlp%J%val( l ) / data%C_scale( nlp%J%row( l ) )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           data%QP_accel%A%ne = 0
           DO i = 1, nlp%m
             DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
               nlp%J%val( l ) = nlp%J%val( l ) / data%C_scale( i )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, nlp%m
             DO j = 1, nlp%n
               l = l + 1
               nlp%J%val( l ) = nlp%J%val( l ) / data%C_scale( i )
             END DO
           END DO
         END SELECT
       END IF

!write(6,"( ' c ', /, ( 5ES12.4 ) )" )  nlp%C( : nlp%m )
!write(6,"( ' J ', /, ( 5ES12.4 ) )" )  nlp%J%val( : nlp%J%ne )

!  compute the gradient of the Lagrangian

       nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) - nlp%Z( : nlp%n )
       CALL mop_AX( - one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,      &
                    m_matrix = nlp%m, n_matrix = nlp%n )
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &

!      WRITE(6,*) ' gl, y ', maxval( nlp%gL ), maxval( nlp%Y )
!      WRITE( data%out, "(A, /, ( 4ES20.12 ) )" ) ' gl_after ',  nlp%gl

       inform%obj = nlp%f

!  compute norms of the primal and dual feasibility and the complemntary
!  slackness

       IF ( data%control%scale_constraints > 0 ) THEN
         multiplier_norm =                                                     &
           MAX( one, OPT_multiplier_norm( nlp%n, nlp%Z( : nlp%n ),           &
                nlp%m, nlp%Y( : nlp%m ) / data%C_scale( : nlp%m ) ) )
         inform%primal_infeasibility =                                         &
           OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                  &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ),   &
                                     SCALE = data%C_scale( : nlp%m ) )
       ELSE
         multiplier_norm =                                                     &
           MAX( one, OPT_multiplier_norm( nlp%n, nlp%Z( : nlp%n ),             &
                                          nlp%m, nlp%Y( : nlp%m ) ) )
         inform%primal_infeasibility =                                         &
           OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                  &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
       END IF
       inform%dual_infeasibility =                                             &
         OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) ) / multiplier_norm
       inform%complementary_slackness =                                        &
         OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                 &
            nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),          &
            nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                       &
            nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) ) / multiplier_norm

!  ---------------------
!  first-iteration tasks
!  ---------------------

       IF ( inform%iter == 0 ) THEN

!  compute the stopping tolerances

         data%stop_p = MAX( data%control%stop_abs_p,                           &
            data%control%stop_rel_p * inform%primal_infeasibility )
         data%stop_d = MAX( data%control%stop_abs_d,                           &
           data%control%stop_rel_d * inform%dual_infeasibility )
         data%stop_c = MAX( data%control%stop_abs_c,                           &
           data%control%stop_rel_c * inform%complementary_slackness )
         data%stop_i = MAX( data%control%stop_abs_i, data%control%stop_rel_i )
         IF ( data%printi ) WRITE( data%out,                                   &
             "(  /, A, '  Primal    convergence tolerance =', ES11.4,          &
            &    /, A, '  Dual      convergence tolerance =', ES11.4,          &
            &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )      &
                 prefix, data%stop_p, prefix, data%stop_d, prefix, data%stop_c

!  compute the one-norm of the violation

         data%primal_viol = OPT_primal_infeasibility( nlp%n, nlp%X,            &
                nlp%X_l, nlp%X_u, nlp%m, nlp%C, nlp%C_l, nlp%C_u, norm = 1 )
!write(6,"( 'viol reset b', ES12.4 )") data%primal_viol

         data%primal_viol_ref = data%primal_viol
!        data%comp_viol = inform%complementary_slackness
!        data%comp_viol_ref = data%comp_viol

!  record the maximum infeasibility allowed, and set up the initial filter

         max_viol = MAX( data%control%max_abs_i,                               &
                         data%control%max_rel_i * data%primal_viol )

!  set up the initial filter

         CALL FILTER_update_filter( minus_infinity, max_viol,                  &
           data%FILTER_data, data%control%FILTER_control, inform%FILTER_inform )

       END IF

!  ---------------------
!  check for termination
!  ---------------------

!  exit if an approximate KKT point has been found

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%dual_infeasibility <= data%stop_d .AND.                     &
            inform%complementary_slackness <= data%stop_c ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
                "( /, A, ' Termination criteria satisfied ' )" ) prefix
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  print details of the current iteration

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out, 2000 ) prefix
         data%print_1st_header = .FALSE.
         data%print_iteration_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2010 )                                             &
             prefix, inform%iter, data%it_type, data%pair_type,                &
             data%step_used( 1 : 1 ), inform%obj,                              &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness,                                   &
             ADJUSTR( STRING_integer_6( data%QP_accel%A%m ) ),                 &
             data%s_norm, data%d_type,                                         &
             inform%FILTER_inform%filter_size, data%sigma, data%clock_now
         ELSE
           WRITE( data%out, 2020 ) prefix, inform%iter,                        &
             inform%obj, inform%primal_infeasibility,                          &
             inform%dual_infeasibility, inform%complementary_slackness,        &
             inform%FILTER_inform%filter_size, data%sigma, data%clock_now
         END IF

!        IF ( mod( inform%iter, 50 ) == 0 ) WRITE( data%out, 2100 )
!        WRITE( data%out, 2110 ) inform%iter, data%p_mode, nlp%f,              &
!          data%primal_viol, data%comp_viol, data%sigma
       END IF
       data%pair_type = ' '

!  ----------------------------------
!  check for unsuccessful termination
!  ----------------------------------

!  exit if the iteration limit has been exceeded

       inform%iter = inform%iter + 1

       IF ( inform%iter > data%control%maxit ) THEN
         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Iteration limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  exit if the elapsed-time limit has been exceeded

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
             REAL( data%time_now - data%time_start, wp )                       &
               > data%control%cpu_time_limit ) .OR.                            &
             ( data%control%clock_time_limit >= zero .AND.                     &
               data%clock_now - data%clock_start                               &
              > data%control%clock_time_limit ) ) THEN
         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Time limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_time_limit ; GO TO 900
       END IF

!  compute the Hessian

       inform%h_eval = inform%h_eval + 1
       IF ( data%accepted ) THEN
         data%WORK_m( : nlp%m ) = nlp%Y( : nlp%m )  ! temporary copy
         IF ( data%control%scale_constraints > 0 )                             &
           nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) / data%C_scale( : nlp%m )

         IF ( data%reverse_hl ) THEN
            data%branch = 140 ; inform%status = 4 ; RETURN
         ELSE
            CALL eval_HL( data%eval_status, nlp%X, nlp%Y, userdata, nlp%H%val )
            IF ( data%eval_status /= 0 ) THEN
              inform%bad_eval = 'eval_HL'
              inform%status = GALAHAD_error_evaluation ; GO TO 900
            END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian of the Lagrangian

  140  CONTINUE
       IF ( data%accepted ) nlp%Y( : nlp%m ) = data%WORK_m( : nlp%m )

!write(6,"( ' H ', /, ( 5ES12.4 ) )" )  nlp%H%val( : nlp%H%ne )

!  compute an estimate of the norm of the Hessian

       data%h_norm = MAXVAL( ABS( nlp%H%val( : data%H_ne ) ) )
!write(6,*) nlp%H%row( : data%H_ne )
!write(6,*) nlp%H%col( : data%H_ne )
!write(6,*) nlp%H%val( : data%H_ne )

!write(6,*) ' accepted ', data%accepted, data%control%predictor_hessian
       IF ( data%accepted ) THEN

         SELECT CASE( data%control%predictor_hessian )

!  if required, modify the exact Hessian

         CASE( se_modified_predictor_hessian, dd_modified_predictor_hessian )

!  compute the modifications

           data%QP_pred%H%n = nlp%n ; data%QP_pred%H%ne = data%H_ne
           data%QP_pred%H%val( : data%H_ne ) = nlp%H%val( : data%H_ne )
!write(6,*)  'h ', data%QP_pred%H%val( : data%H_ne )
!write(6,*)  'det ', data%QP_pred%H%val( 1 ) * data%QP_pred%H%val( 3 ) - data%QP_pred%H%val( 2 ) ** 2
!write(6,*)' pivot_control ',   data%control%SLS_control%pivot_control
           CALL SLS_factorize( data%QP_pred%H, data%SLS_data,                  &
                               data%control%SLS_control, inform%SLS_inform )

!write(6,*) inform%SLS_inform%first_modified_pivot, &
!  inform%SLS_inform%largest_modified_pivot
!  storee the modifications

!  use the modifications provided by the factorization, if any

           IF ( data%control%predictor_hessian ==                              &
                  se_modified_predictor_hessian ) THEN
             IF ( inform%SLS_inform%status < 0 ) THEN
!write(6,*) ' factorize SLS_inform%status ', inform%SLS_inform%status
               inform%status = GALAHAD_error_factorization ; GO TO 900
             END IF

             IF ( inform%SLS_inform%largest_modified_pivot > zero ) THEN
               inform%modifications = inform%modifications + 1
               CALL SLS_enquire( data%SLS_data, inform%SLS_inform,             &
                 PERTURBATION = data%QP_pred%H%val( data%QP_pred%H%ne + 1 :    &
                                                    data%QP_pred%H%ne + nlp%n ))
               IF ( inform%SLS_inform%status < 0 ) THEN
!write(6,*) ' enquire SLS_inform%status ', inform%SLS_inform%status
                 inform%status = GALAHAD_error_factorization ; GO TO 900
               END IF

!  ensure that the modifications are not too small

               DO i = data%QP_pred%H%ne + 1, data%QP_pred%H%ne + nlp%n
                 data%QP_pred%H%val( i ) = MAX( data%QP_pred%H%val( i ),       &
                   data%control%min_hessian_perturbation )
               END DO

               IF ( data%printt ) WRITE( data%out, "( A, ' H perturbed by',    &
              &   ES23.16, ' using ', A )" ) prefix,                           &
                inform%SLS_inform%largest_modified_pivot,                      &
                TRIM( data%control%linear_solver_for_modifications )
!write(6,*) ' pert = ', data%QP_pred%H%val( data%QP_pred%H%ne + 1 :        &
!                                           data%QP_pred%H%ne + nlp%n )

               data%QP_pred%H%ne = data%H_ne + nlp%n
             END IF

!  if the matrix is indefinite, compute diagonal-dominant modifications

           ELSE ! dd_modified_predictor_hessian ) THEN
             IF ( inform%SLS_inform%status >= 0 ) THEN ! no modification
             ELSE IF ( inform%SLS_inform%status /= - 20 ) THEN ! error
!write(6,*) ' factorize SLS_inform%status ', inform%SLS_inform%status
               inform%status = GALAHAD_error_factorization ; GO TO 900
             ELSE ! not positive definite

!  temporarily store the diagonal of the Hessian in QP_pred%H%val and the
!  sum of absolute values of its off-diagonals (+1) in WORK_n

               data%QP_pred%H%val( data%H_ne + 1 : data%H_ne + nlp%n ) = zero
               data%WORK_n( : nlp%n ) = one
               DO l = 1, data%H_ne
                 i = data%QP_pred%H%row( l ) ; j = data%QP_pred%H%col( l )
                 val = data%QP_pred%H%val( l )
                 IF ( i == j ) THEN
                   data%QP_pred%H%val( data%H_ne + i ) =                       &
                     data%QP_pred%H%val( data%H_ne + i ) + val
                 ELSE
                   data%WORK_n( i ) = data%WORK_n( i ) + ABS( val )
                   data%WORK_n( j ) = data%WORK_n( j ) + ABS( val )
                 END IF
               END DO

!  ensure that the model Hessian is diagonally dominant

               val = zero
               DO i = 1, nlp%n
                 IF ( data%WORK_n( i )                                         &
                      > data%QP_pred%H%val( data%H_ne + i ) ) THEN
                   data%QP_pred%H%val( data%H_ne + i )                         &
                     = data%WORK_n( i ) - data%QP_pred%H%val( data%H_ne + i )
                   val = MAX( val, data%QP_pred%H%val( data%H_ne + i ) )
                 ELSE
                   data%QP_pred%H%val( data%H_ne + i ) = zero
                 END IF
               END DO
               IF ( data%printt ) WRITE( data%out, "( A, ' H pertubed by',     &
              &   ES23.16, ' using diagonal-dominance modifications' )" )      &
                    prefix, val
               data%QP_pred%H%ne = data%H_ne + nlp%n
             END IF
           END IF
!do i = 1, data%QP_pred%H%ne
! write(6,*) data%QP_pred%H%row(i), data%QP_pred%H%col(i), data%QP_pred%H%val(i)
!end do
!  if required, update the limited-memory predictor Hessian, B_k

         CASE( l_bfgs_predictor_hessian )
           IF ( inform%iter > 1 ) THEN
             data%DG = nlp%gL - data%DG
             dxtdg = DOT_PRODUCT( data%DX, data%DG )

!  ensure that the limited-memory formula is well defined

             IF ( dxtdg > zero ) THEN
               CALL SMT_put( data%QP_pred%H%type, 'LBFGS', inform%alloc_status )
               dgtdg = DOT_PRODUCT( data%DG, data%DG )
!              delta = dgtdg / dxtdg
               delta =  data%h_norm
!write(6,*) ' delta', delta, ' H_ne ', data%H_ne

               CALL LMS_form( data%DX, data%DG, delta,                         &
                              data%QP_pred%H_lm, data%control%LMS_control,     &
                              inform%LMS_inform )
               IF ( data%printt ) WRITE( data%out,                             &
                 "( A, ' L-BFGS update with ', I0, ' vector', A )" ) prefix,   &
                   inform%LMS_inform%length,                                   &
                   STRING_pleural( inform%LMS_inform%length )

               IF ( data%printd ) THEN
                 data%DX = zero
                 DO i = 1, nlp%n
                   data%DX( i ) = one
                   CALL LMS_apply( data%DX, data%WORK_n, data%QP_pred%H_lm,    &
                                   data%control%LMS_control, inform%LMS_inform )
                   WRITE(data%out,"( A, ' H_lm column ', I0, /,                &
                  &  ( 4ES18.10 ) )" ) prefix, i, data%WORK_n
                   data%DX( i ) = zero
                 END DO
               END IF
               SELECT CASE ( data%control%QP_pred_control%CQP_control%&
                               &SBLS_control%preconditioner )
               CASE ( : 0, 2, 3, 5, 8 : )
                 data%control%QP_pred_control%CQP_control%&
                   &SBLS_control%preconditioner = 7
               END SELECT
             ELSE
               IF ( data%printt ) WRITE( data%out,                             &
                                    "( A, ' L-BFGS updated skipped' )" ) prefix
             END IF
           ELSE
             CALL SMT_put( data%QP_pred%H%type, 'IDENTITY', inform%alloc_status)
           END IF
         END SELECT
       END IF

!  Update B_k as positive definite matrix
!  ----------------------------------------------------------------
!  = 0 if scaled identity
!  = 1 if use modified Hessian
!  = 2 if use BFGS                 ! not implemented
!  = 3 if use BFGS with evals modified  ! not implemented
!  = 4 if alternately scale identity    ! not implemented

!       normg = TWO_NORM( nlp%G )
!
!       IF ( data%control%predictor_hessian == 0 ) THEN
!         B_k_scalar = MIN( data%control%B_norm_max,                           &
!                           MAX( data%control%B_norm_min,normg ) )
!         B_k = B_k_scalar * eye_n
!       ELSE IF ( data%control%predictor_hessian == 1 ) THEN
!         Hess = cutest_hess( nlp%X, - nlp%Y )
!         [Hess_U, Hess_D] = eig( Hess )
!         normHess = norm( Hess,2 )
!         IF ( normHess == zero ) THEN
!             Heps = MAX( tenm5, MIN( normg, ten5 ) )
!         ELSE
!             Heps = MAX( tenm5, norm( Hess,2 ) / data%control%Bk_cond_bd )
!         END IF
!         Bk_D = Hess_D
!         DO i = 1, n
!           IF ( Bk_D( i,i ) >= Heps ) THEN
!           ELSE IF ( Bk_D( i,i ) <= - Heps ) THEN
!             Bk_D( i,i ) = - Bk_D( i,i )
!             inform%B_modified = .TRUE.
!           ELSE
!             Bk_D( i,i ) = Heps
!             inform%B_modified = .TRUE.
!           END IF
!         END DO
!         B_k = Hess_U * Bk_D * Hess_U^T
!         fprintf( 'Why am I in this try-catch in line 421?\n' )
!         B_k_scalar = MIN( data%control%B_norm_max,                           &
!                           MAX( data%control%B_norm_min, normg ) )
!         B_k = B_k_scalar * eye_n
!       END IF

!  track if in penalty mode

       IF ( data%p_mode ) inform%iter_in_penalty = inform%iter_in_penalty + 1

!  ----------------------------------------------------------------------------
!                     COMPUTE THE STEERING STEP (A1:7)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:7)' )" ) prefix
       IF ( data%printw ) WRITE( data%out, "( /, A, 1X, 30('-'),               &
      &  ' Steering step ', 30('-'), / )" ) prefix

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  The steering subproblem may be written as
!
!   min  || min( c_k + J_k s - c_l, c_u - c_k - J_k s, 0 )||_1
!   s.t.  x_l <= x + s <= x_u
!         inf_norm( s ) <= radius
!
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  steering is not required if we are feasible

       IF ( data%primal_viol /= zero ) THEN

!  update the trust region radius (infinity norm)

         radius = data%control%radius_steering

!  set up the steering LP data

         data%QP_steer%f = zero
!        data%QP_steer%G( : nlp%n ) = zero
         data%QP_steer%gradient_kind = 0
         data%QP_steer%Hessian_kind = 0
         data%QP_steer%target_kind = 0
         data%QP_steer%A%val( : data%QP_steer%A%ne )                           &
           = nlp%J%val( : data%QP_steer%A%ne )
         data%QP_steer%X_l( : nlp%n )                                          &
           = MAX( nlp%X_l( : nlp%n ) - nlp%X( : nlp%n ), - radius )
         data%QP_steer%X_u( : nlp%n )                                          &
           = MIN( nlp%X_u( : nlp%n ) - nlp%X( : nlp%n ), radius )
         IF ( nlp%m > 0 ) THEN
           WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )
             data%QP_steer%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) - nlp%C( : nlp%m)
           ELSE WHERE
             data%QP_steer%C_l( : nlp%m ) = nlp%C_l( : nlp%m )
           END WHERE
           WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )
             data%QP_steer%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) - nlp%C( : nlp%m)
           ELSE WHERE
             data%QP_steer%C_u( : nlp%m ) = nlp%C_u( : nlp%m )
           END WHERE
         END IF
         data%QP_steer%X( : nlp%n ) = zero
         data%QP_steer%Y( : nlp%m ) = zero
         data%QP_steer%Z( : nlp%n ) = zero
         data%control%QP_steer_control%rho = one

!  solve the problem

         IF ( data%printt ) WRITE( data%out, "( A, ' linear violation before', &
        &  ' steering =', ES11.4 )") prefix, data%primal_viol
         CALL L1QP_solve( data%QP_steer, data%QP_steer_data,                   &
                          data%control%QP_steer_control, inform%QP_steer_inform)
!                         C_stat = nlp%C_status, X_stat = nlp%X_status )
         IF ( inform%QP_steer_inform%status /= GALAHAD_ok                      &
          .AND. inform%QP_steer_inform%status /= GALAHAD_error_max_iterations  &
          .AND. inform%QP_steer_inform%status /= GALAHAD_error_ill_conditioned &
          .AND. inform%QP_steer_inform%status /= GALAHAD_error_tiny_step ) THEN
           IF ( data%printe )                                                  &
             WRITE( data%error, "( ' On exit from steering QP_solve status',   &
            &  ' = ', I0 )" ) inform%QP_steer_inform%status
           inform%status = GALAHAD_error_qp_solve ; GO TO 900
         END IF
!        WRITE(6,*) ' x ', data%QP_steer%X( : data%QP_steer%n )

!  restore the solution

         data%S_steer( : nlp%n ) = data%QP_steer%X( : nlp%n )

!  calculate linearized feasiblity improvement

         CALL mop_AX( one, nlp%J, data%S_steer, zero, data%WORK_m,             &
!                     out = data%out, error = data%error,                      &
!                     print_level = data%print_level,                          &
                      m_matrix = nlp%m, n_matrix = nlp%n )

         ellv_steer                                                            &
           = OPT_primal_infeasibility( nlp%n, nlp%X + data%S_steer, nlp%X_l,   &
               nlp%X_u, nlp%m, nlp%C + data%WORK_m, nlp%C_l, nlp%C_u, norm = 1 )
         IF ( data%printt ) WRITE( data%out, "( A, ' linear violation after',  &
        &  ' steering =', ES11.4 )") prefix, ellv_steer
!write(6,*) data%primal_viol, ellv_steer
         del_ellv_steer = data%primal_viol - ellv_steer
         IF ( data%printt ) WRITE( data%out, "( A, ' steering objective ',     &
        &  'decrease =', ES12.4 )") prefix, del_ellv_steer

!  we are feasible so steering is unnecessary

       ELSE
         ellv_steer = zero ; del_ellv_steer = zero
         IF ( data%printt )                                                    &
           WRITE( data%out, "( A, ' no steering necessary' )" ) prefix
       END IF

!  update values at R(k)

       IF ( data%fails == 0 ) data%del_ellv_steer_ref = del_ellv_steer

!  check if an infeasible stationary point has been detected (A1:8-9)

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:8-9)' )" ) prefix

!write(6,*) del_ellv_steer, data%stop_d
!write(6,*) data%primal_viol, data%stop_p

       IF ( del_ellv_steer <= data%stop_i .AND.                                &
            data%primal_viol > data%stop_p ) THEN
         inform%status = GALAHAD_error_primal_infeasible
         GO TO 900
       END IF

!  check if the steering step achieved its purpose.

       IF ( del_ellv_steer <= zero .AND.                                       &
            data%primal_viol >= data%stop_p / ten ) THEN
         inform%status = - 108
         GO TO 900
       END IF

!  check if we are linearized feasible

       lin_feas = ellv_steer <= data%stop_p / ten
       IF ( data%printt ) WRITE( data%out, "( A, ' linear feasible? ', L1 )" ) &
                            prefix, lin_feas

!  ----------------------------------------------------------------------------
!                   COMPUTE THE PREDICTOR STEP (A1:10)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:10)' )" ) prefix
       IF ( data%printw ) WRITE( data%out, "( /, A, 1X, 30('-'),               &
      &  ' Predictor step ',    30('-') )" ) prefix

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  if linearly feasible:
!
!   min   f_k + g_k^T s + 0.5 s^T B_k s
!   s.t.  c_l <= c_k + J_k s <= c_u
!         x_l <= x + s <= x_u

!  else:

!   min   f_k + g_k^T s + 0.5 s^T B_k s
!             + sigma || min( c_k + J_k s - c_l, c_u - c_k - J_k s, 0 )||_1
!   s.t.  x_l <= x + s <= x_u

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set up the predictor QP data

       data%QP_pred%f = nlp%f
       data%QP_pred%G( : nlp%n ) = nlp%G( : nlp%n )
       data%QP_pred%A%val( : data%QP_pred%A%ne )                               &
         = nlp%J%val( : data%QP_pred%A%ne )
       IF ( data%control%predictor_hessian == identity_predictor_hessian ) THEN
!        data%QP_pred%H%val( 1 ) = one
!        data%QP_pred%H%val( 1 ) = ten ** 4
         data%QP_pred%H%val( 1 ) = MAX( data%h_norm, one )
       END IF
       IF ( .TRUE. ) THEN
         data%QP_pred%X_l( : nlp%n ) = nlp%X_l( : nlp%n ) - nlp%X( : nlp%n )
         data%QP_pred%X_u( : nlp%n ) = nlp%X_u( : nlp%n ) - nlp%X( : nlp%n )
       ELSE
         radius = data%control%radius_steering
         data%QP_pred%X_l( : nlp%n )                                           &
           = MAX( nlp%X_l( : nlp%n ) - nlp%X( : nlp%n ), - radius )
         data%QP_pred%X_u( : nlp%n )                                           &
           = MIN( nlp%X_u( : nlp%n ) - nlp%X( : nlp%n ), radius )
       END IF
       IF ( nlp%m > 0 ) THEN
         WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )
           data%QP_pred%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) - nlp%C( : nlp%m )
         ELSE WHERE
           data%QP_steer%C_l( : nlp%m ) = nlp%C_l( : nlp%m )
         END WHERE
         WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )
           data%QP_pred%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) - nlp%C( : nlp%m )
         ELSE WHERE
           data%QP_pred%C_u( : nlp%m ) = nlp%C_u( : nlp%m )
         END WHERE
       END IF
       data%QP_pred%X( : nlp%n ) = zero
       data%QP_pred%Y( : nlp%m ) = zero
       data%QP_pred%Z( : nlp%n ) = zero

!  convert the predictor QP problem into an l_1 QP if required

!write(6,*) ' lin_feas', lin_feas

  160  CONTINUE
       IF ( lin_feas ) THEN
         data%control%QP_pred_control%rho = zero
       ELSE
         data%control%QP_pred_control%rho = data%sigma
       END IF

!  solve the predictor QP problem

!write(6,*) ' before l1qp pred'
       CALL L1QP_solve( data%QP_pred, data%QP_pred_data,                       &
                        data%control%QP_pred_control, inform%QP_pred_inform,   &
                        C_stat = nlp%C_status, X_stat = nlp%X_status )
!write(6,*) ' after l1qp pred'
       IF ( inform%QP_pred_inform%status /= GALAHAD_ok .AND.                   &
            inform%QP_pred_inform%status /= GALAHAD_error_max_iterations .AND. &
            inform%QP_pred_inform%status /= GALAHAD_error_ill_conditioned .AND.&
            inform%QP_pred_inform%status /= GALAHAD_error_tiny_step ) THEN
         IF ( data%printm )                                                    &
           WRITE( data%error, "( A, ' On exit from predictor QP_solve status', &
          & ' = ', I0 )" ) prefix, inform%QP_pred_inform%status
         IF ( lin_feas ) THEN
           IF ( data%printt ) WRITE( data%out, "( /, A, ' Should be linearly', &
          &   ' feasible, let''s have another go ...' )" ) prefix
           lin_feas = .FALSE. ; GO TO 160
         END IF
         inform%status = GALAHAD_error_qp_solve ; GO TO 900
       END IF

       data%S_pred( : nlp%n ) = data%QP_pred%X( : nlp%n )
!write(6,*) 'S_pred', data%S_pred(:nlp%n)

!write(6,*) 'x_s', nlp%X_status( : nlp%n )
!write(6,*) 'c_s', nlp%C_status( : nlp%m  )
       active = COUNT( ABS(  nlp%X_status( : nlp%n ) ) /= 0 ) +                &
               COUNT( ABS(  nlp%C_status( : nlp%m ) ) /= 0 )
       IF ( data%printt ) WRITE( data%out, "( A, 1X, I0, ' active constraint', &
      &                     A )" ) prefix, active, STRING_pleural( active )

!  if we are primal feasible and the step is tiny, we are likely optimal. Check

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            TWO_NORM(  data%S_pred( : nlp%n ) ) <= data%control%s_tiny ) THEN

!  use the predictor QP multipliers and dual variables

         nlp%Y( : nlp%m ) = data%QP_pred%Y( : nlp%m )
         nlp%Z( : nlp%n ) = data%QP_pred%Z( : nlp%n )
         nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) - nlp%Z( : nlp%n )
         CALL mop_AX( - one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,    &
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
                    m_matrix = nlp%m, n_matrix = nlp%n )
         IF ( data%printd ) THEN
           WRITE( data%out, "( /, A, ' predictor' )" ) prefix
           IF ( nlp%m > 0 ) WRITE( data%out, "( A, ' Y ', /, ( 5ES12.4 ) )" )  &
             prefix, nlp%Y( : nlp%m )
           WRITE( data%out, "( A, ' Z ', /, ( 5ES12.4 ) )" )                   &
             prefix, nlp%Z( : nlp%n )
           WRITE( data%out, "( A, ' Gl ', /, ( 5ES12.4 ) )" )                  &
             prefix, nlp%Gl( : nlp%n )
         END IF

!  compute the infeasibility measures

         multiplier_norm                                                       &
           = MAX( one, OPT_multiplier_norm( nlp%n, nlp%Z( : nlp%n ),           &
                                            nlp%m, nlp%Y( : nlp%m ) ) )
         inform%dual_infeasibility =                                           &
           OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) ) / multiplier_norm
         inform%complementary_slackness =                                      &
           OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),               &
              nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),        &
              nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                     &
              nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) ) / multiplier_norm

!  check if an approximate KKT point has been found

         kkt_pred = inform%primal_infeasibility <= data%stop_p .AND.           &
                    inform%dual_infeasibility <= data%stop_d .AND.             &
                    inform%complementary_slackness <= data%stop_c
         IF ( data%printt ) WRITE( data%out, "( A, ' KKT measures ',           &
        &  '(predictor  ):', 3ES11.4 )" ) prefix, inform%primal_infeasibility, &
               inform%dual_infeasibility, inform%complementary_slackness

         IF ( kkt_pred ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
                  "( /, A, ' Termination criteria satisfied ' )" ) prefix
           inform%status = GALAHAD_ok ; GO TO 900
         END IF
       END IF

!  calculate the linearized feasiblity improvement

       CALL mop_AX( one, nlp%J, data%S_pred, zero, data%WORK_m,                &
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
                    m_matrix = nlp%m, n_matrix = nlp%n )

       ellv_pred = OPT_primal_infeasibility( nlp%n, nlp%X + data%S_pred,       &
          nlp%X_l, nlp%X_u, nlp%m, nlp%C + data%WORK_m, nlp%C_l, nlp%C_u,      &
          norm = 1 )
       IF ( data%printt ) WRITE( data%out, "( A, ' linear violation after',    &
          &  ' predictor step =', ES11.4 )") prefix, ellv_pred
       del_ellv_pred = data%primal_viol - ellv_pred

!  calculate the penalty function improvement

       IF ( SMT_get( data%QP_pred%H%type ) == 'LBFGS' ) THEN
         CALL LMS_apply( data%S_pred, data%WORK_n, data%QP_pred%H_lm,          &
                         data%control%LMS_control, inform%LMS_inform )
       ELSE
         CALL mop_AX( one, data%QP_pred%H, data%S_pred, zero, data%WORK_n,     &
                      symmetric = .TRUE.,                                      &
!                     out = data%out, error = data%error,                      &
!                     print_level = data%print_level,                          &
                      m_matrix = nlp%n, n_matrix = nlp%n )
       END IF

       del_qf_pred = - DOT_PRODUCT( data%S_pred( : nlp%n ), nlp%G( : nlp%n )   &
                                      + half * data%WORK_n( : nlp%n ) )
       data%del_qphi_pred = del_qf_pred + data%sigma * del_ellv_pred
       IF ( data%printt )                                                      &
         WRITE( data%out, "( A, ' predictor objective decrease =',             &
      &   ES12.4 )" ) prefix, data%del_qphi_pred

 ! check if the predictor is linearly feasible

       lin_feas = ellv_pred <= data%stop_p / five
!write(6,*) ' lin_feas', lin_feas

!  -----------------------------------------------------------------------------
!                      COMPUTE THE SEARCH DIRECTION (A1:13)
!  -----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:13)' )" ) prefix

!  simple backtracking line search to find s_k

       data%tau = one

       IF ( lin_feas ) THEN
         data%S = data%S_pred
!write(6,*) ' a'
!        data%ellv = ellv_pred
         data%del_ellv = del_ellv_pred
       ELSE
!write(6,*) ' b'
         IF ( del_ellv_steer > zero ) THEN
           DO
             IF ( data%tau < data%control%tau_min ) EXIT
             data%S                                                            &
               = data%tau * data%S_pred + ( one - data%tau ) * data%S_steer

! write(6,*) 'S_pred', data%S_pred(:nlp%n)
! write(6,*) 'S_steer', data%S_steer(:nlp%n)


             CALL mop_AX( one, nlp%J, data%S, zero, data%WORK_m,               &
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
                    m_matrix = nlp%m, n_matrix = nlp%n )

             data%ellv = OPT_primal_infeasibility( nlp%n, nlp%X + data%S,      &
               nlp%X_l, nlp%X_u, nlp%m, nlp%C + data%WORK_m,                   &
               nlp%C_l, nlp%C_u, norm = 1 )
             data%del_ellv = data%primal_viol - data%ellv
             IF ( data%del_ellv >= data%control%eta_v * del_ellv_steer ) EXIT
             data%tau = data%tau * data%control%tau_reduce
           END DO

           IF ( data%tau < data%control%tau_min ) THEN
             inform%status = - 109
             GO TO 900
           END IF
           IF ( data%printt ) WRITE( data%out, "( A, ' tau steplength = ',     &
          &   ES11.4 )" ) prefix, data%tau

!  data%primal_viol < data%stop_p / ten (see conditions on s_steer)

         ELSE
! write(6,*) 'S_steer', data%S_steer(:nlp%n)
           data%S = data%S_steer
!          data%ellv = ellv_steer
           data%del_ellv = del_ellv_steer
         END IF
       END IF

! Update values at R(k)

       IF ( data%fails == 0 ) THEN
         data%S_ref = data%S
         data%del_ellv_ref = data%del_ellv
       END IF

!  ----------------------------------------------------------------------------
!     UPDATE THE PENALTY PARAMETER TO ENSURE PENALTY-FUNCTION DESCENT (A1:14)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:14)' )" ) prefix

!  compute B s

       IF ( SMT_get( data%QP_pred%H%type ) == 'LBFGS' ) THEN
         CALL LMS_apply( data%S, data%WORK_n, data%QP_pred%H_lm,              &
                         data%control%LMS_control, inform%LMS_inform )
       ELSE
! write(6,*) 'H', ( data%QP_pred%H%row(i), data%QP_pred%H%col(i), &
!  data%QP_pred%H%val(i), i = 1,  data%QP_pred%H%ne )
!write(6,*) 'S', data%S(:nlp%n)
! write(6,*) 'S', data%S(:data%QP_pred%H%n)
         CALL mop_AX( one, data%QP_pred%H, data%S, zero, data%WORK_n,          &
                      symmetric = .TRUE.,                                      &
!                     out = data%out, error = data%error,                      &
!                     print_level = data%print_level,                          &
                      m_matrix = nlp%n, n_matrix = nlp%n )
       END IF

!  calculate linearized penalty

       data%del_ellf = - DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )
       data%del_qf = data%del_ellf                                             &
         - half * DOT_PRODUCT( data%S( : nlp%n ), data%WORK_n( : nlp%n ) )
       data%del_ellphi = data%del_ellf + data%sigma * data%del_ellv

!  update the penalty parameter

       IF ( data%del_ellphi <                                                  &
            data%sigma * data%control%eta_sigma * del_ellv_steer ) THEN
         new_sigma_denom                                                       &
           = data%del_ellv - data%control%eta_sigma * del_ellv_steer
         IF ( new_sigma_denom <= zero ) THEN
           new_sigma = zero
         ELSE
           new_sigma = - data%del_ellf / new_sigma_denom
         END IF
         data%sigma_new                                                        &
           = MAX( data%sigma + data%control%sigma_inc, new_sigma )
         IF ( data%printt ) WRITE( data%out, "( A, ' penalty parameter',       &
        &   ' updated to', ES11.4 )" ) prefix, data%sigma_new
       ELSE
         data%sigma_new = data%sigma
       END IF

!  calculate changes in the models of penalty function

       data%del_ellphi = data%del_ellf + data%sigma_new * data%del_ellv
       data%del_qphi = data%del_qf + data%sigma_new * data%del_ellv
       data%del_qphi_pred = del_qf_pred + data%sigma_new * del_ellv_pred
       data%phi = nlp%f + data%sigma_new * data%primal_viol

!  update values at R(k)

       IF ( data%fails == 0 ) THEN
         data%phi_ref = data%phi
         data%sigma_new_ref = data%sigma_new
         data%del_ellf_ref = data%del_ellf
       END IF

!  ----------------------------------------------------------------------------
!                      COMPUTE THE ACCELERATOR STEP (A1:15)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:15)' )" ) prefix
       IF ( data%printw ) WRITE( data%out, "( /, A, 1X, 30('-'),               &
      &  ' Accelerator step ',    30('-'), / )" ) prefix

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Solve the accelerator subproblem
!
!    min   g_k^T( pred_k + s) + 0.5 ( pred_k + s )^T H_k ( pred_k + s )
!      = ( g_k + H_k pred_k )^T s + 0.5 s^T H_k s
!    s.t. [J s]_pred_active = 0, [s]_pred_active = 0, ||s||_2 <= radius_accel

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

       data%accel_found = .FALSE.

       IF ( data%control%use_accelerator ) THEN
!        IF ( inform%iter == 10 ) alt_accel = .TRUE.

!  set the objective details

         IF ( alt_accel ) THEN
           data%QP_accel%f = nlp%f
           data%QP_accel%G( : nlp%n ) = nlp%G( : nlp%n )
         ELSE
           CALL mop_AX( one, nlp%H, data%S_pred, zero, data%WORK_n,            &
                        symmetric = .TRUE.,                                    &
!                       out = data%out,  error = data%error,                   &
!                       print_level = data%print_level,                        &
                        m_matrix = nlp%n, n_matrix = nlp%n )
           CALL mop_AX( one, nlp%J, data%S_pred, zero, data%WORK_m,            &
!                       out = data%out, error = data%error,                    &
!                       print_level = data%print_level,                        &
                        m_matrix = nlp%m, n_matrix = nlp%n )

           data%QP_accel%f = nlp%f + DOT_PRODUCT( data%S_pred( : nlp%n ),      &
             nlp%G( : nlp%n ) + half * data%WORK_n( : nlp%n ) )
           data%QP_accel%G( : nlp%n )                                          &
             = nlp%G( : nlp%n ) + data%WORK_n( : nlp%n )
         END IF
         data%QP_accel%H%val( : data%H_ne ) = nlp%H%val( : data%H_ne )
         data%QP_accel%X( : nlp%n ) = data%S_pred( : nlp%n )

!write(6,*) ' --- g ', data%QP_accel%g
!write(6,*) ' --- f ', data%QP_accel%f
!write(6,*) ' --- H ', data%QP_accel%H%val

!     IF ( data%printi ) WRITE( data%out, "(                                   &
!     &   /, A, ' - # active counstraints: ', I0, ' from ', I0, ', bounds: ',  &
!     &  I0, ' from ', I0 )" ) prefix, m_working, nlp%m, n_working, nlp%n

!  set the right-hand sides for the active constraints

!write(6,*) 'C_status', nlp%C_status(:nlp%m)

!write(6,*) ' alt_accel ', alt_accel
         data%QP_accel%m = 0
         DO i = 1, nlp%m
           IF ( nlp%C_status( i ) < 0 ) THEN
!write(6,*) ' constraint ', i, ' lower bound'
             data%QP_accel%m = data%QP_accel%m + 1
             nlp%C_status( i ) = - data%QP_accel%m
             IF ( alt_accel ) THEN
               data%QP_accel%C( data%QP_accel%m ) = nlp%C( i )
             ELSE
!              data%QP_accel%C( data%QP_accel%m ) = zero
               data%QP_accel%C( data%QP_accel%m )                              &
                 = nlp%C( i ) + data%WORK_m( i ) - nlp%C_l( i )
             END IF
             data%QP_accel%Y( data%QP_accel%m ) = data%QP_pred%Y( i )
           ELSE IF ( nlp%C_status( i ) > 0 ) THEN
!write(6,*) ' constraint ', i, ' upper bound'
             data%QP_accel%m = data%QP_accel%m + 1
             nlp%C_status( i ) = data%QP_accel%m
             IF ( alt_accel ) THEN
               data%QP_accel%C( data%QP_accel%m ) = nlp%C( i )
             ELSE
!              data%QP_accel%C( data%QP_accel%m ) = zero
               data%QP_accel%C( data%QP_accel%m )                              &
                 = nlp%C( i ) + data%WORK_m( i ) - nlp%C_u( i )
             END IF
             data%QP_accel%Y( data%QP_accel%m ) = data%QP_pred%Y( i )
           END IF
         END DO
         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) < 0 ) THEN
!write(6,*) ' variable ', i, ' lower bound'
             data%QP_accel%m = data%QP_accel%m + 1
             nlp%X_status( i ) = - data%QP_accel%m
             IF ( alt_accel ) THEN
               data%QP_accel%C( data%QP_accel%m ) = nlp%X_l( i )
             ELSE
!              data%QP_accel%C( data%QP_accel%m ) = zero
               data%QP_accel%C( data%QP_accel%m )                              &
                 = nlp%X( i ) + data%S_pred( i ) - nlp%X_l( i )
             END IF
             data%QP_accel%Y( data%QP_accel%m ) = data%QP_pred%Z( i )
           ELSE IF ( nlp%X_status( i ) > 0 ) THEN
!write(6,*) ' variable ', i, ' upper bound'
             data%QP_accel%m = data%QP_accel%m + 1
             nlp%X_status( i ) = data%QP_accel%m
             IF ( alt_accel ) THEN
               data%QP_accel%C( data%QP_accel%m ) = nlp%X_u( i )
             ELSE
!              data%QP_accel%C( data%QP_accel%m ) = zero
               data%QP_accel%C( data%QP_accel%m )                              &
                 = nlp%X( i ) + data%S_pred( i ) - nlp%X_u( i )
             END IF
             data%QP_accel%Y( data%QP_accel%m ) = data%QP_pred%Z( i )
           END IF
         END DO
         data%QP_accel%A%m = data%QP_accel%m

!  place the entries in the Jacobian of working constraints

         data%QP_accel%A%ne = 0
         SELECT CASE ( SMT_get( nlp%J%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, nlp%J%ne
             i = nlp%J%row( l )
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    data%control%jacobian_zero_tolerance ) THEN
               data%QP_accel%A%ne = data%QP_accel%A%ne + 1
               data%QP_accel%A%row( data%QP_accel%A%ne ) = ii
               data%QP_accel%A%col( data%QP_accel%A%ne ) = nlp%J%col( l )
               data%QP_accel%A%val( data%QP_accel%A%ne ) = nlp%J%val( l )
             END IF
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           data%QP_accel%A%ne = 0
           DO i = 1, nlp%m
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    data%control%jacobian_zero_tolerance ) THEN
               DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                 data%QP_accel%A%ne = data%QP_accel%A%ne + 1
                 data%QP_accel%A%row( data%QP_accel%A%ne ) = ii
                 data%QP_accel%A%col( data%QP_accel%A%ne ) = nlp%J%col( l )
                 data%QP_accel%A%val( data%QP_accel%A%ne ) = nlp%J%val( l )
               END DO
             END IF
           END DO
         CASE ( 'DENSE' )
           data%QP_accel%A%ne = 0 ; l = 0
           DO i = 1, nlp%m
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    data%control%jacobian_zero_tolerance ) THEN
               DO j = 1, nlp%n
                 data%QP_accel%A%ne = data%QP_accel%A%ne + 1 ; l = l + 1
                 data%QP_accel%A%row( data%QP_accel%A%ne ) = ii
                 data%QP_accel%A%col( data%QP_accel%A%ne ) = j
                 data%QP_accel%A%val( data%QP_accel%A%ne ) = nlp%J%val( l )
               END DO
             ELSE
               l = l + nlp%n
             END IF
           END DO
         END SELECT
         DO j = 1, nlp%n
           i = ABS( nlp%X_status( j ) )
           IF ( i > 0 ) THEN
             data%QP_accel%A%ne = data%QP_accel%A%ne + 1
             data%QP_accel%A%row( data%QP_accel%A%ne ) = i
             data%QP_accel%A%col( data%QP_accel%A%ne ) = j
             data%QP_accel%A%val( data%QP_accel%A%ne ) = one
           END IF
         END DO

!  get rid of tiny entries

!        DO i = 1, data%QP_accel%A%ne
!          IF( ABS( data%QP_accel%A%val( i )) <= 1.0D-14 ) THEN
!          write(6,*) ' setting a(',i,')= ', data%QP_accel%A%val( i ),'to zero'
!           data%QP_accel%A%val( i ) = zero
!          END IF
!        END DO

!  find the step s (stored in data%QP_accel%X) and Lagrange multiplier
!  y (stored in data%QP_accel%Y)

         IF ( data%printt ) WRITE( data%out, "( A, ' find the accelerator',    &
        &  ' step - entering EQP: n = ', I0, ', m_working = ', I0 )" )         &
             prefix, nlp%n, data%QP_accel%m
         data%accel_found = .TRUE.
         data%control%QP_accel_control%radius = data%radius_accelerator
         CALL EQP_solve( data%QP_accel, data%QP_accel_data,                    &
                         data%control%QP_accel_control, inform%QP_accel_inform )

         IF ( data%printd ) THEN
           WRITE( control%out, "( A, ' x_accel =', 3ES24.16, /,                &
          & ( 10X, 3ES24.16 ) )" ) prefix, data%QP_accel%X( : data%QP_accel%n )
           WRITE( control%out, "( A, ' y_accel =', 3ES24.16, /,                &
          & ( 10X, 3ES24.16 ) )" ) prefix, data%QP_accel%Y( : data%QP_accel%m )
           WRITE( control%out, "( A, ' s_pred  =', 3ES24.16, /,                &
          & ( 10X, 3ES24.16 ) )" ) prefix, data%S_pred( : nlp%n )
         END IF

!  check to see if the subproblem was infeasible

         IF ( inform%QP_accel_inform%status ==                                 &
              GALAHAD_error_primal_infeasible ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' EQP accelerator infeasible ... skipping' )" ) prefix
           data%accel_found = .FALSE.
           data%S_accel( : nlp%n ) = data%S_pred( : nlp%n )
!          GO TO 200

!  check to see if another error occured

         ELSE IF ( inform%QP_accel_inform%status /= GALAHAD_ok ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' EQP accelerator failure, status = ', I0,               &
          &      ' ... skipping' )" ) prefix, inform%QP_accel_inform%status
           data%accel_found = .FALSE.
           data%S_accel( : nlp%n ) = data%S_pred( : nlp%n )
!          inform%status = GALAHAD_error_qp_solve ; GO TO 990

!  spread the solution into the vectors (s_a,y_a,z_a,c_a)

         ELSE

!  truncate the solution so that x_l <= x + s_pred + alpha s_accel <= x_u

           alpha = one
           DO i = 1, nlp%n
             si = data%QP_accel%X( i )
             IF ( si > zero ) THEN
               alpha = MIN( alpha,                                             &
                 ( nlp%X_u( i ) - nlp%X( i ) - data%S_pred( i ) ) / si )
             ELSE IF ( si < zero ) THEN
               alpha = MIN( alpha,                                             &
                 ( nlp%X_l( i ) - nlp%X( i ) - data%S_pred( i ) ) / si )
             END IF
           END DO

           IF ( alpha < one ) THEN
             data%QP_accel%X( : nlp%n ) = alpha * data%QP_accel%X( : nlp%n )
             IF ( data%printt )                                                &
               WRITE( data%out, "( A, ' truncated accelerator steplength = ',  &
              &  ES11.4 )" ) prefix, alpha
           END IF

!  spread the solution into the vectors (s_a,y_a,z_a,c_a)

           IF ( alt_accel ) THEN
             data%S_accel = data%QP_accel%X( : nlp%n )
           ELSE
             data%S_accel = data%S_pred( : nlp%n ) + data%QP_accel%X( : nlp%n )
           END IF
           DO i = 1, nlp%m
             IF ( nlp%C_status( i ) == 0 ) THEN
               data%Y_accel( i ) = zero
             ELSE
               data%Y_accel( i ) = data%QP_accel%Y( ABS( nlp%C_status( i ) ) )
             END IF
           END DO
           DO i = 1, nlp%n
             IF ( nlp%X_status( i ) == 0 ) THEN
               data%Z_accel( i ) = zero
             ELSE
               data%Z_accel( i ) = data%QP_accel%Y( ABS( nlp%X_status( i ) ) )
             END IF
           END DO
!          data%C_p( : nlp%m ) = zero
!          DO l = 1, data%CQP_prob%A%ne
!            i = data%CQP_prob%A%row( l )
!            data%C_accel( i ) = data%C_accel( i )                             &
!              + data%CQP_prob%A%val( l ) * data%S_accel(data%CQP_prob%A%col(l))
!          END DO
         END IF

! Check if bound constraints satisfied
!    accel_bd_viol = OPT_primal_infeasibility_bounds( nlp%n, nlp%X +           &
!    data%S_accel, nlp%X_l, nlp%X_u, norm = 1 )

!  update values at R(k)

         IF ( data%fails == 0 )                                                &
           data%S_accel_ref( : nlp%n ) = data%S_accel( : nlp%n )
       END IF

!  ----------------------------------------------------------------------------
!                      COMPUTE THE CAUCHY STEPS (A1:16)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:16)' )" ) prefix

!  compute H * s, g^T s and s^T H s

       CALL mop_AX( one, nlp%H, data%S, zero, data%WORK_n, symmetric = .TRUE., &
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
                    m_matrix = nlp%n, n_matrix = nlp%n )
       gts = DOT_PRODUCT( nlp%G( : nlp%n ), data%S( : nlp%n ) )
       sths = DOT_PRODUCT( data%WORK_n( : nlp%n ), data%S( : nlp%n ) )

!  compute the Cauchy-f step length

       IF ( sths <= zero ) THEN
         alpha_cf = one
       ELSE
         alpha_cf = MIN( one, - gts / sths )
       END IF
       del_qf_cf = MAX(  - alpha_cf * ( gts + half * alpha_cf * sths ), zero )
       rho_f = MAX( MIN( data%del_ellf, del_qf_cf ), zero )

! Cauchy-phi step (dpr: below was done in shoddy way. made simple)
!      CALL FISQP_calc_cauchy_qphi( data%S, nlp%X, nlp%C, Jx, nlp%f, nlp%G,    &
!                                   Hess, data%sigma_new, constraints,         &
!                                   cauchy_phi_val )
!      del_qphi_cphi                                                           &
!        = ( nlp%f + data%sigma_new * data%primal_viol ) - qphi_cauchy_val
!      rho_phi = MAX( MIN( data%del_ellphi, del_qphi_cphi ), zero )

       rho_phi = MAX( data%del_ellphi, zero )

!  update values at R(k)

       IF ( data%fails == 0 ) THEN
         data%rho_f_ref = rho_f
         data%rho_phi_ref = rho_phi
       END IF

!  ----------------------------------------------------------------------------
!                             PENALTY MODE (A1:17)
!  ----------------------------------------------------------------------------

  170  CONTINUE
       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:17)' )" ) prefix
       data%step_used = '***    '

       IF ( .NOT. data%p_mode ) GO TO 270
!      IF ( data%p_mode ) THEN
         IF ( data%printt )                                                    &
           WRITE( data%out, "( A, ' in penalty mode' )" ) prefix
         data%alpha = one

!  loop to find acceptable step (A1:18)

         IF ( data%printd ) WRITE( data%out, "( A, ' (A1:18)' )" ) prefix
!        DO
  180      CONTINUE
           data%s_norm = TWO_norm( data%alpha * data%S )
           IF ( data%accel_found ) THEN
             data%accel_norm = TWO_NORM( data%alpha * data%S_accel )
             IF ( MAX( data%s_norm, data%accel_norm )                          &
                    < data%control%s_tiny ) THEN
               inform%status = GALAHAD_error_tiny_step ; GO TO 900
             END IF
           ELSE
             IF ( data%s_norm < data%control%s_tiny ) THEN
               inform%status = GALAHAD_error_tiny_step ; GO TO 900
             END IF
           END IF
!          ELSE
!            data%accel_norm = inf
!          END IF
!          data%s_norm = MIN( data%s_norm, data%accel_norm )


           IF ( data%accel_found ) THEN
             s_start = s_accel
           ELSE
             s_start = s_normal
           END IF
           s_type = s_start

           data%p_mode = .TRUE.
           data%filter_acceptable = .FALSE.
           data%pair_type = 'n'

!  loop over possible steps (A1:20)

           IF ( data%printd ) WRITE( data%out, "( A, ' (A1:20)' )" ) prefix
!          DO s_type = s_start, s_normal
  200        CONTINUE

!  set the trial point

             IF ( s_type == s_accel ) THEN
               data%X_trial = nlp%X + data%alpha * data%S_accel
             ELSE
               data%X_trial = nlp%X + data%alpha * data%S
             END IF
             data%X_trial = MAX( nlp%X_l, MIN( data%X_trial, nlp%X_u ) )

!  evaluate the objective and constraints at the trial point

             IF ( data%reverse_fc ) THEN
               data%WORK_n( : nlp%n ) = nlp%X( : nlp%n )  ! temporary copy
               data%WORK_m( : nlp%m ) = nlp%C( : nlp%m )  ! temporary copy
               data%f_current = nlp%f                     ! temporary copy
               nlp%X = data%X_trial
               data%branch = 210 ; inform%status = 2 ; RETURN
             ELSE
               CALL eval_FC( data%eval_status, data%X_trial, userdata,         &
                             data%f_trial, data%C_trial )
               IF ( data%eval_status /= 0 ) THEN
                 inform%bad_eval = 'eval_FC'
                 inform%status = GALAHAD_error_evaluation ; GO TO 900
               END IF
             END IF

!  return from reverse communication to obtain the objective value

  210        CONTINUE
             IF ( data%reverse_fc ) THEN
               data%f_trial = nlp%f
               IF ( data%control%scale_constraints > 0 ) THEN
                 data%C_trial = nlp%C / data%C_scale
               ELSE
                 data%C_trial = nlp%C
               END IF
               nlp%X( : nlp%n ) = data%WORK_n( : nlp%n )  ! restore copy
               nlp%C( : nlp%m ) = data%WORK_m( : nlp%m )  ! restore copy
               nlp%f = data%f_current                     ! restore copy
             ELSE
               IF ( data%control%scale_constraints > 0 )                       &
                 data%C_trial = data%C_trial / data%C_scale
             END IF
             inform%fc_eval = inform%fc_eval + 1

!  evaluate the violation and penalty function at the trial point

             data%viol_trial                                                   &
               = OPT_primal_infeasibility( nlp%n, data%X_trial,                &
                   nlp%X_l, nlp%X_u, nlp%m, data%C_trial, nlp%C_l, nlp%C_u,    &
                   norm = 1 )
             data%phi_trial                                                    &
               = data%f_trial + data%sigma_new_ref * data%viol_trial

             IF ( data%printt ) THEN
               IF ( s_type == s_accel ) THEN
                 WRITE( data%out,"( A, ' accelerator s, trial f and c',        &
                &  ES9.2, ES21.13, ES20.13 )" )                                &
                    prefix, data%accel_norm, data%f_trial, data%viol_trial
               ELSE
                 WRITE( data%out,"( A, ' predictor   s, trial f and c',        &
                &  ES9.2, ES21.13, ES20.13 )" )                                &
                    prefix, data%s_norm, data%f_trial, data%viol_trial
               END IF
             END IF

!  ------------- check for a p-pair --------------

!  check if the new point gives a p-pair (A1:21-22)

             IF ( data%phi_trial <= data%phi_ref - data%control%gamma_phi      &
                    * data%alpha * data%rho_phi_ref ) THEN
               IF ( data%printd ) WRITE( data%out, "( A, ' (A1:21-22)')") prefix
               data%pair_type = 'p'
               IF ( s_type == s_accel ) THEN
                 data%step_used = 'accel  '
               ELSE
                 data%step_used = 'pred   '
               END IF
               inform%num_p = inform%num_p + 1
               data%accepted = .TRUE.
               IF ( data%printt ) THEN
                 IF ( s_type == s_accel ) THEN
                   WRITE( data%out, "( A, ' accelerator step with stepsize',   &
                  &   ES11.4, ' gives a p-pair' )" ) prefix, data%alpha
                 ELSE
                   WRITE( data%out, "( A, ' predictor step with stepsize',     &
                  &   ES11.4, ' gives a p-pair' )" ) prefix, data%alpha
                 END IF
               END IF
               GO TO 250

!  exit loop if the more than max_fails failures have occurred (A1:23-24)

             ELSE IF ( data%fails <= data%control%max_fails .AND.              &
                       data%control%max_fails > 0 ) THEN
               IF ( data%printd ) WRITE( data%out, "( A, ' (A1:23-24)')") prefix
               IF ( s_type == s_accel ) THEN
                 data%step_used = 'accel_nm'
               ELSE
                 data%step_used = 'pred_nm '
               END IF
               inform%num_nm = inform%num_nm + 1
               data%fails = data%fails + 1
               data%accepted = .FALSE.
               IF ( data%printt ) THEN
                 IF ( s_type == s_accel ) THEN
                   WRITE( data%out, "( A, ' accelerator step with stepsize',   &
                  &   ES11.4, ' accepted as a failure' )" ) prefix, data%alpha
                 ELSE
                   WRITE( data%out, "( A, ' predictor step with stepsize',     &
                  &   ES11.4, ' accepted as a failure' )" ) prefix, data%alpha
                 END IF
               END IF
               GO TO 450
             END IF

!  end of loop over possible steps (A1:20)

             IF ( data%printd ) WRITE( data%out, "( A, ' (A1:20)' )" ) prefix
             s_type = s_type + 1
             IF ( s_type <= s_normal ) GO TO 200
!          END DO

!  reduce step size (A1:19)

           IF ( data%printd ) WRITE( data%out, "( A, ' (A1:19)' )" ) prefix
           data%alpha = data%alpha * data%control%alpha_reduce
!          data%radius_accelerator
!             = data%radius_accelerator * data%control%alpha_reduce

!  end of loop to find acceptable step (started A1:18)

           GO TO 180
!        END DO
  250    CONTINUE
!        IF ( data%alpha == one ) data%radius_accelerator
!               = data%radius_accelerator * two

!  check if the new point is acceptable to filter (A1:25-26)

         IF ( data%printd ) WRITE( data%out, "( A, ' (A1:25-26)' )" ) prefix
         IF ( data%check_filter ) THEN
           CALL FILTER_acceptable( data%f_trial, data%viol_trial,              &
                                   data%FILTER_data,                           &
                                   data%control%FILTER_control, acceptable )
           IF ( acceptable ) THEN
             data%p_mode = .FALSE.
             data%filter_acceptable = .TRUE.
             data%accepted = .TRUE.
             IF ( data%printt ) THEN
               IF ( s_type == s_accel ) THEN
                 WRITE( data%out, "( A, ' accelerator step with stepsize',     &
                &   ES11.4, ' acceptable to filter' )" ) prefix, data%alpha
               ELSE
                 WRITE( data%out, "( A, ' predictor step with stepsize',       &
                &   ES11.4, ' acceptable to filter' )" ) prefix, data%alpha
               END IF
             END IF
           END IF
         END IF
         GO TO 450

!  ----------------------------------------------------------------------------
!                             FILTER MODE (A1:27)
!  ----------------------------------------------------------------------------

!      ELSE
  270    CONTINUE
         IF ( data%printd ) WRITE( data%out, "( A, ' (A1:27)' )" ) prefix
         IF ( data%printt )                                                    &
           WRITE( data%out, "( A, ' in filter mode' )" ) prefix

!  loop to find acceptable step (A1:29)

         IF ( data%printd ) WRITE( data%out, "( A, ' (A1:29)' )" ) prefix
         data%alpha = one
!        DO
  290      CONTINUE
           data%s_norm = TWO_norm( data%alpha * data%S )
           IF ( data%accel_found ) THEN
             data%accel_norm = TWO_NORM( data%alpha * data%S_accel )
             IF ( MAX( data%s_norm, data%accel_norm )                          &
                    < data%control%s_tiny ) THEN
               inform%status = GALAHAD_error_tiny_step ; GO TO 900
             END IF
           ELSE
             IF ( data%s_norm < data%control%s_tiny ) THEN
               inform%status = GALAHAD_error_tiny_step ; GO TO 900
             END IF
           END IF

           IF ( data%accel_found ) THEN
             s_start = s_accel
           ELSE
             s_start = s_normal
           END IF
           s_type = s_start

           data%p_mode = .FALSE.
           data%accepted = .FALSE.
           data%filter_acceptable = .FALSE.
           data%pair_type = 'n'

!  loop over possible steps (A1:31)

!          DO s_type = s_start, s_normal
  310        CONTINUE
             IF ( data%printd ) WRITE( data%out, "( A, ' (A1:31)' )" ) prefix

!  set the trial point

             IF ( s_type == s_accel ) THEN
               data%check_b_pair = data%fails <= data%control%max_fails .AND.  &
                 data%control%max_fails > 0
               data%X_trial = nlp%X + data%alpha * data%S_accel
             ELSE
               data%check_b_pair = .NOT. data%control%just_filter
               data%X_trial = nlp%X + data%alpha * data%S
             END IF
!write(6,*) 'x',  nlp%X( : nlp%n )
!write(6,*) 's',  data%S_accel( : nlp%n )
!write(6,*) 'x+', data%X_trial( : nlp%n )
            data%X_trial = MAX( nlp%X_l, MIN( data%X_trial, nlp%X_u ) )
!write(6,*) 'x+', data%X_trial( : nlp%n )

!  evaluate the objective and constraints at the trial point

             IF ( data%reverse_fc ) THEN
               data%WORK_n( : nlp%n ) = nlp%X( : nlp%n )  ! temporary copy
               data%WORK_m( : nlp%m ) = nlp%C( : nlp%m )  ! temporary copy
               data%f_current = nlp%f                     ! temporary copy
               nlp%X = data%X_trial
               data%branch = 320 ; inform%status = 2 ; RETURN
             ELSE
               CALL eval_FC( data%eval_status, data%X_trial, userdata,         &
                             data%f_trial, data%C_trial )
               IF ( data%eval_status /= 0 ) THEN
                 inform%bad_eval = 'eval_FC'
                 inform%status = GALAHAD_error_evaluation ; GO TO 900
               END IF
             END IF

!  return from reverse communication to obtain the objective value

  320        CONTINUE
             IF ( data%reverse_fc ) THEN
               data%f_trial = nlp%f
               IF ( data%control%scale_constraints > 0 ) THEN
                 data%C_trial = nlp%C / data%C_scale
               ELSE
                 data%C_trial = nlp%C
               END IF
               nlp%X( : nlp%n ) = data%WORK_n( : nlp%n )  ! restore copy
               nlp%C( : nlp%m ) = data%WORK_m( : nlp%m )  ! restore copy
               nlp%f = data%f_current                     ! restore copy
             ELSE
               IF ( data%control%scale_constraints > 0 )                       &
                 data%C_trial = data%C_trial / data%C_scale
             END IF
             inform%fc_eval = inform%fc_eval + 1

!  evaluate the violation and penalty function at the trial point

             data%viol_trial                                                   &
               = OPT_primal_infeasibility( nlp%n, data%X_trial, nlp%X_l,       &
                   nlp%X_u, nlp%m, data%C_trial, nlp%C_l, nlp%C_u, norm = 1 )
             data%phi_trial                                                    &
              = data%f_trial + data%sigma_new_ref * data%viol_trial

             IF ( data%printt ) THEN
               IF ( s_type == s_accel ) THEN
                 WRITE( data%out,"( A, ' accelerator s, trial f and c',        &
                &  ES9.2, ES21.13, ES20.13 )" )                                &
                  prefix, data%accel_norm, data%f_trial, data%viol_trial
               ELSE
                 WRITE( data%out,"( A, ' predictor   s, trial f and c',        &
                &  ES9.2, ES21.13, ES20.13 )" )                                &
                  prefix, data%s_norm, data%f_trial, data%viol_trial
               END IF
             END IF

!  check if the trial point is acceptable to filter

             CALL FILTER_acceptable( data%f_trial, data%viol_trial,            &
                                     data%FILTER_data,                         &
                                     data%control%FILTER_control, acceptable )
!  the trial point is acceptable

             IF ( acceptable ) THEN
               data%filter_acceptable = .TRUE.

!  -------------- check for a v-pair --------------

!  check if the new point gives a v-pair (A1:32-33)

!   WRITE(6,"( ' del_ellf_ref, gamma_v * del_ellv_ref ', 2ES12.4 )" )          &
!     data%del_ellf_ref,                                                       &
!     data%control%gamma_v * data%del_ellv_ref

               IF ( data%del_ellf_ref                                          &
                      < data%control%gamma_v * data%del_ellv_ref ) THEN
                 IF ( data%printd )                                            &
                   WRITE( data%out, "( A, ' (A1:32-33)' )" ) prefix

!  compute the current iterate's filter elements

                 IF ( data%control%filter_uses_steering ) THEN
                   relaxed_viol = data%primal_viol_ref                         &
                     - data%alpha * data%control%eta_v * data%del_ellv_steer_ref
                   fil_viol = MAX( relaxed_viol,                               &
                                     data%control%beta * data%primal_viol_ref )
                   fil_f = data%f_ref - data%control%gamma                     &
                     * MIN( relaxed_viol,                                      &
                            data%control%beta * data%primal_viol_ref )
                 ELSE
                   fil_viol = data%control%beta * data%primal_viol_ref
                   fil_f = data%f_ref - data%control%gamma                     &
                               * data%control%beta * data%primal_viol_ref
                 END IF

!  check if trial step is acceptable to current iterate

                 IF ( data%viol_trial <= fil_viol .OR.                         &
                      data%f_trial <= fil_f ) THEN
                   data%pair_type = 'v'
                   data%accepted = .TRUE.

!  update the filter to add [ fil_f, fil_viol ]

                   CALL FILTER_update_filter( fil_f, fil_viol,                 &
                                              data%FILTER_data,                &
                                              data%control%FILTER_control,     &
                                              inform%FILTER_inform )
                   IF ( data%printt ) THEN
                     IF ( s_type == s_accel ) THEN
                       WRITE( data%out, "( A, ' accelerator step with step',   &
                      & 'size', ES11.4, ' gives a v-pair' )") prefix, data%alpha
                     ELSE
                       WRITE( data%out, "( A, ' predictor step with stepsize', &
                      &   ES11.4, ' gives a v-pair' )" ) prefix, data%alpha
                     END IF
                   END IF
                   inform%num_v = inform%num_v + 1
                   GO TO 440
                 END IF

!  -------------- check for an o-pair --------------

!  check if the new point gives an o-pair (A1:34-35)

               ELSE
                 IF ( data%printd )                                            &
                   WRITE( data%out, "( A, ' (A1:34-35)' )" ) prefix

!  check if sufficient decrease in actual objective

!   WRITE(6,"( ' f_trial, f_ref, gamma_f * alpha * rho_f_ref', 3ES12.4 )" )    &
!      data%f_trial, data%f_ref,                                               &
!      data%control%gamma_f * data%alpha * data%rho_f_ref

                 IF ( data%f_trial <= data%f_ref - data%control%gamma_f *      &
                        data%alpha * data%rho_f_ref ) THEN
                   data%pair_type = 'o'
                   IF ( data%printt ) THEN
                     IF ( s_type == s_accel ) THEN
                       WRITE( data%out, "( A, ' accelerator step with step',   &
                      & 'size', ES11.4, ' gives an o-pair')") prefix, data%alpha
                     ELSE
                       WRITE( data%out, "( A, ' predictor step with stepsize', &
                      &   ES11.4, ' gives an o-pair' )" ) prefix, data%alpha
                     END IF
                   END IF
                   inform%num_o = inform%num_o + 1
                   data%accepted = .TRUE.
                   GO TO 440
                 END IF
               END IF
             END IF

!  -------------- check for a b-pair --------------

             IF ( data%check_b_pair ) THEN
               IF ( data%printd ) WRITE( data%out,                             &
             &    "( A, ' (A1:37-39) or (A1:42-43)')") prefix

!  the new point gives a b-pair (A1:37-39) & (A1:42-43)

               IF ( data%phi_trial < data%phi_ref                              &
                      - data%control%gamma_phi * data%alpha * data%rho_phi_ref &
                     .AND. data%viol_trial < data%primal_viol_ref ) THEN
                 data%pair_type = 'b'
                 data%accepted = .TRUE.

!  compute current filter elements

                 relaxed_viol = data%primal_viol_ref                           &
                   - data%alpha * data%control%eta_v * data%del_ellv_steer_ref
                 fil_viol = MAX( relaxed_viol,                                 &
                              data%control%beta * data%primal_viol_ref )
                 fil_f = data%f_ref - data%control%gamma *                     &
                   MIN( relaxed_viol, data%control%beta * data%primal_viol_ref )

!  update the filter to add [ fil_f, fil_viol ]

                 CALL FILTER_update_filter( fil_f, fil_viol,                   &
                                            data%FILTER_data,                  &
                                            data%control%FILTER_control,       &
                                            inform%FILTER_inform )
                 IF ( data%printt ) THEN
                   IF ( s_type == s_accel ) THEN
                     WRITE( data%out, "( A, ' accelerator step with step',     &
                    & 'size', ES11.4, ' gives a b-pair' )") prefix, data%alpha
                   ELSE
                     WRITE( data%out, "( A, ' predictor step with stepsize',   &
                    &   ES11.4, ' gives a b-pair' )" ) prefix, data%alpha
                   END IF
                 END IF
                 inform%num_b = inform%num_b + 1

!  change to penalty mode

                 data%p_mode = .TRUE.
                 GO TO 440

!  exit loop if more than max_fails failures have occurred (A1:40-41)

               ELSE IF ( data%fails <= data%control%max_fails .AND.            &
                         data%control%max_fails > 0 ) THEN
                 IF ( s_type == s_accel ) THEN
                   data%step_used = 'accel_nm'
                 ELSE
                   data%step_used = 'pred_nm '
                 END IF
                 inform%num_nm = inform%num_nm + 1
                 data%fails = data%fails + 1
                 data%accepted = .FALSE.
                 IF ( data%printt ) THEN
                   IF ( s_type == s_accel ) THEN
                     WRITE( data%out, "( A, ' accelerator step with stepsize', &
                    &   ES11.4, ' accepted as a failure' )" ) prefix, data%alpha
                   ELSE
                     WRITE( data%out, "( A, ' predictor step with stepsize',   &
                    &   ES11.4, ' accepted as a failure' )" ) prefix, data%alpha
                   END IF
                 END IF
                 GO TO 450
               END IF
             END IF

!  end of loop over possible steps (A1:31)

             IF ( data%printd ) WRITE( data%out, "( A, ' (A1:31)' )" ) prefix
             s_type = s_type + 1
             IF ( s_type <= s_normal ) GO TO 310
!          END DO

!  reduce step size (A1:30)

           IF ( data%printd ) WRITE( data%out, "( A, ' (A1:30)' )" ) prefix
           data%alpha = data%alpha * data%control%alpha_reduce
!          data%radius_accelerator
!            = data%radius_accelerator * data%control%alpha_reduce

!  end of loop to find acceptable step (A1:29)

           GO TO 290
!        END DO

  440    CONTINUE
         IF ( s_type == s_accel ) THEN
           data%step_used = 'accel  '
           ELSE
           data%step_used = 'pred   '
         END IF

!  count if entered p_mode

         IF ( data%p_mode ) inform%entered_penalty = inform%entered_penalty + 1
!      END IF

!  ----------------------------------------------------------------------------
!                  UPDATE THE PENALTY PARAMETER (A1:45-46)
!  ----------------------------------------------------------------------------

  450  CONTINUE
       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:45-46)' )" ) prefix
       IF ( data%del_qphi_pred <= zero .OR.                                    &
            data%del_qphi < data%control%eta_phi * data%del_qphi_pred ) THEN
         data%sigma_new = MAX( two * data%sigma_new,                           &
                               data%sigma_new + data%control%sigma_inc )
         IF ( data%printt ) WRITE( data%out, "( A, ' penalty parameter',       &
        &   ' updated to', ES11.4 )" ) prefix, data%sigma_new
       END IF

!  print a one-line summary of the iteration

!       IF ( data%control%print_level > 0 ) THEN
!         WRITE( data%out, "( '   |   ', I3, ES8.2, 2ES12.2, 4ES9.2, I5 )" )   &
!           inform%B_modified, del_ellv_steer, data%del_qphi_pred,             &
!           data%phi, data%tau, TWO_NORM( data%S ), data%alpha,data%pair_type, &
!           inform%FILTER_inform%filter_size, data%step_used
!       END IF

!  ----------------------------------------------------------------------------
!                        UPDATE THE ITERATES (A1:47)
!  ----------------------------------------------------------------------------

       IF ( data%printd ) WRITE( data%out, "( A, ' (A1:47)' )" ) prefix
!      g_old = nlp%G
!      Jx_old = Jx
!      step = data%X_trial - nlp%X

       IF ( data%control%predictor_hessian == l_bfgs_predictor_hessian )      &
         data%DX = data%X_trial - nlp%X
       nlp%X = data%X_trial
       nlp%f = data%f_trial ; inform%obj = nlp%f
       nlp%C = data%C_trial
       data%sigma = data%sigma_new

       IF ( data%accepted ) THEN
         data%fails = 0
!        Rk = inform%iter
         data%success_iter = data%success_iter + 1
         data%X_ref = nlp%X
         data%f_ref = nlp%f
       END IF

!  ----------------------------------------------------------------------------
!      CALCULATE LAGRANGE MULTIPLIER ESTIMATES AND CHECK KKT CONDITIONS
!  ----------------------------------------------------------------------------

!  compute the gradient of the Lagrangian using the predictor multipliers

       IF ( data%control%predictor_hessian == l_bfgs_predictor_hessian )       &
         data%DY( : nlp%m ) = data%QP_pred%Y( : nlp%m ) - nlp%Y( : nlp%m )
       nlp%Y( : nlp%m )                                                        &
         = data%QP_pred%Y( : nlp%m )
!        = data%QP_pred%Y_l( : nlp%m ) - data%QP_pred%Y_u( : nlp%m )
       IF ( data%control%predictor_hessian == l_bfgs_predictor_hessian )       &
         data%DZ( : nlp%n ) = data%QP_pred%Z( : nlp%n ) - nlp%Z( : nlp%n )
       nlp%Z( : nlp%n )                                                        &
         = data%QP_pred%Z( : nlp%n )
!        = data%QP_pred%Z_l( : nlp%n ) - data%QP_pred%Z_u( : nlp%n )
       nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) - nlp%Z( : nlp%n )
       CALL mop_AX( - one, nlp%J, nlp%Y, one, nlp%Gl, transpose = .TRUE.,      &
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &
                    m_matrix = nlp%m, n_matrix = nlp%n )

       IF ( data%printd ) THEN
         WRITE( data%out, "( /, A, ' predictor' )" ) prefix
         IF ( nlp%m > 0 ) WRITE( data%out, "( A, ' Y ', /, ( 5ES12.4 ) )" )    &
           prefix, nlp%Y( : nlp%m )
         WRITE( data%out, "( A, ' Z ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%Z( : nlp%n )
         WRITE( data%out, "( A, ' Gl ', /, ( 5ES12.4 ) )" )                    &
           prefix, nlp%Gl( : nlp%n )
       END IF

!  compute the infeasibility measures

       IF ( data%control%scale_constraints > 0 ) THEN
         multiplier_norm =                                                     &
           MAX( one, OPT_multiplier_norm( nlp%n, nlp%Z( : nlp%n ),             &
                nlp%m, nlp%Y( : nlp%m ) / data%C_scale( : nlp%m ) ) )
         inform%primal_infeasibility =                                         &
           OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                  &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ),   &
                                     SCALE = data%C_scale( : nlp%m ) )
       ELSE
         multiplier_norm =                                                     &
           MAX( one, OPT_multiplier_norm( nlp%n, nlp%Z( : nlp%n ),             &
                                            nlp%m, nlp%Y( : nlp%m ) ) )
         inform%primal_infeasibility =                                         &
           OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                  &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
       END IF
       inform%dual_infeasibility =                                             &
         OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) ) / multiplier_norm
       inform%complementary_slackness =                                        &
         OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                 &
            nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),          &
            nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                       &
            nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) ) / multiplier_norm

!  record if an approximate KKT point has been found

       kkt_pred = inform%primal_infeasibility <= data%stop_p .AND.             &
                  inform%dual_infeasibility <= data%stop_d .AND.               &
                  inform%complementary_slackness <= data%stop_c
       IF ( data%printt ) WRITE( data%out, "( A, ' KKT measures ',             &
      &  '(predictor  ):', 3ES11.4 )" ) prefix, inform%primal_infeasibility,   &
           inform%dual_infeasibility, inform%complementary_slackness

       IF ( data%control%use_accelerator ) THEN

!  compute the gradient of the Lagrangian using the accelerator multipliers

!write(6,"( ' y ', /, ( 6ES12.4) )" ) nlp%Y( : nlp%m )
!write(6,"( ' y_accel ', /, ( 6ES12.4) )" ) data%Y_accel( : nlp%m )
!write(6,"( ' z ', /, ( 6ES12.4) )" ) nlp%Z( : nlp%n )
!write(6,"( ' z_accel ', /, ( 6ES12.4) )" ) data%Z_accel( : nlp%n )

         data%WORK_n( : nlp%n ) = nlp%G( : nlp%n ) - data%Z_accel( : nlp%n )
         CALL mop_AX( - one, nlp%J, data%Y_accel, one, data%WORK_n,            &
                      transpose = .TRUE.,                                      &
!                     out = data%out, error = data%error,                      &
!                     print_level = data%print_level,                          &
                      m_matrix = nlp%m, n_matrix = nlp%n )

         IF ( data%printd ) THEN
           WRITE( data%out, "( /, A, ' accelerator' )" ) prefix
           IF ( nlp%m > 0 ) WRITE( data%out, "( A, ' Y ',                      &
          &     /, ( 5ES12.4 ) )" ) prefix, data%Y_accel( : nlp%m )
           WRITE( data%out, "( A, ' Z ', /, ( 5ES12.4 ) )" )                   &
             prefix, data%Z_accel( : nlp%n )
           WRITE( data%out, "( A, ' Gl ', /, ( 5ES12.4 ) )" )                  &
             prefix, data%WORK_n( : nlp%n )
         END IF

!  compute the infeasibility measures using the accelerator's multipliers

         IF ( data%control%scale_constraints > 0 ) THEN
           multiplier_norm =                                                   &
             MAX( one, OPT_multiplier_norm( nlp%n, data%Z_accel( : nlp%n ),    &
                  nlp%m, data%Y_accel( : nlp%m ) / data%C_scale( : nlp%m ) ) )
         ELSE
           multiplier_norm =                                                   &
             MAX( one, OPT_multiplier_norm( nlp%n, data%Z_accel( : nlp%n ),    &
                  nlp%m, data%Y_accel( : nlp%m ) ) )
         END IF
         dual_infeasibility =                                                  &
           OPT_dual_infeasibility( nlp%n, data%WORK_n( : nlp%n ) )             &
             / multiplier_norm
         complementary_slackness =                                             &
           OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),               &
              nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), data%Z_accel( : nlp%n ), &
              nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                     &
              nlp%C_u( : nlp%m ), data%Y_accel( : nlp%m ) ) / multiplier_norm

!  record if an approximate KKT point has been found

         kkt_accel = inform%primal_infeasibility <= data%stop_p .AND.          &
                     dual_infeasibility <= data%stop_d .AND.                   &
                     complementary_slackness <= data%stop_c
         IF ( data%printt ) WRITE( data%out, "( A, ' KKT measures ',           &
      &  '(accelerator):', 3ES11.4 )" ) prefix, inform%primal_infeasibility,   &
             dual_infeasibility, complementary_slackness

!  select the better of the (possibly) two measures

!write(6,*) 'y_accel',  data%Y_accel( : nlp%m )
!write(6,*) 'z_accel',  data%Z_accel( : nlp%n )
!        IF ( .TRUE. ) THEN
!        IF ( .FALSE. ) THEN
         IF ( kkt_accel .OR.                                                   &
              complementary_slackness < inform%complementary_slackness ) THEN
!        IF ( kkt_accel .OR.                                                   &
!          complementary_slackness < inform%complementary_slackness .OR.       &
!            inform%iter > 8 ) THEN
           nlp%Y( : nlp%m ) = data%Y_accel( : nlp%m )
           nlp%Z( : nlp%n ) = data%Z_accel( : nlp%n )
           nlp%gL( : nlp%n ) = data%WORK_n( : nlp%n )
           inform%dual_infeasibility = dual_infeasibility
           inform%complementary_slackness = complementary_slackness

           IF ( kkt_accel ) THEN
             IF ( data%printt ) WRITE( data%out,                               &
                    "( /, A, ' Termination criteria satisfied ' )" ) prefix
             inform%status = GALAHAD_ok ; GO TO 900
           END IF
         ELSE
           IF ( kkt_pred ) THEN
             IF ( data%printt ) WRITE( data%out,                               &
                    "( /, A, ' Termination criteria satisfied ' )" ) prefix
             inform%status = GALAHAD_ok ; GO TO 900
           END IF
         END IF
       ELSE

!  exit if an approximate KKT point has been found

         IF ( kkt_pred ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
                  "( /, A, ' Termination criteria satisfied ' )" ) prefix
           inform%status = GALAHAD_ok ; GO TO 900
         END IF
       END IF
       data%primal_viol = OPT_primal_infeasibility( nlp%n, nlp%X,              &
              nlp%X_l, nlp%X_u, nlp%m, nlp%C, nlp%C_l, nlp%C_u, norm = 1 )

       IF ( data%printd ) THEN
         WRITE( control%out, "( A, ' x = ', 3ES24.16, /,                       &
        & ( 5X, 3ES24.16 ) )" ) prefix, nlp%X( : nlp%n )
         WRITE( control%out, "( A, ' y = ', 3ES24.16, /,                       &
        & ( 5X, 3ES24.16 ) )" ) prefix, nlp%Y( : nlp%m )
         WRITE( control%out, "( A, ' z = ', 3ES24.16, /,                       &
        & ( 5X, 3ES24.16 ) )" ) prefix, nlp%Z( : nlp%n )
       END IF

!  check for optimality based on the predictor step

       IF ( ABS( data%del_qphi_pred ) <= data%control%stop_predictor           &
            .AND. data%primal_viol <= data%stop_p ) THEN
         inform%status = - 102
         GO TO 900
       END IF

!  update values at R(k)

       IF ( data%accepted ) THEN
         data%primal_viol_ref = data%primal_viol
!        data%comp_viol_ref = data%comp_viol
       END IF

!  set print agenda for the next iteration

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE.
         data%printd = .FALSE. ; data%print_level = 0
       END IF

       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%QP_steer_control%print_level > 0 .OR.                    &
         data%control%QP_pred_control%print_level > 0 .OR.                     &
         data%control%QP_accel_control%print_level > 0 .OR.                    &
         data%control%FILTER_control%print_level > 0

       IF ( data%printd ) THEN
         WRITE( data%out,"( '     X_l           X         X_u' )" )
         DO i = 1, nlp%n
         WRITE( data%out,"( 3ES12.4 )" ) nlp%X_l(i), nlp%X(i), nlp%X_u(i)
         END DO
       END IF

!  ----------------------------------------------------------------------------
!                      END OF MAIN LOOP (STARTED A1:3)
!  ----------------------------------------------------------------------------

       GO TO 50
!    END DO

!  summarize the final results

 900 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, wp )
     inform%time%clock_total = data%clock_now - data%clock_start

!  restore scaled-cnstraint data

     IF ( data%control%scale_constraints > 0 ) THEN
       nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) / data%C_scale( : nlp%m )
       nlp%C( : nlp%m ) = nlp%C( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )                  &
         nlp%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )                    &
         nlp%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) * data%C_scale( : nlp%m )
     END IF

!  print details of the final iteration

     IF ( data%printi ) THEN
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%QP_steer_control%print_level > 0 .OR.                    &
         data%control%QP_pred_control%print_level > 0 .OR.                     &
         data%control%QP_accel_control%print_level > 0 .OR.                    &
         data%control%FILTER_control%print_level > 0
       IF ( inform%status == GALAHAD_ok .OR.                                   &
            inform%status == GALAHAD_error_primal_infeasible ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out, 2000 ) prefix
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2010 )                                             &
             prefix, inform%iter, data%it_type, data%pair_type,                &
             data%step_used( 1 : 1 ), inform%obj,                              &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness,                                   &
             ADJUSTR( STRING_integer_6( data%QP_accel%A%m ) ), data%s_norm,    &
             data%d_type, inform%FILTER_inform%filter_size, data%sigma,        &
             inform%time%clock_total
         ELSE
           WRITE( data%out, 2020 ) prefix, inform%iter, inform%obj,            &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness, inform%FILTER_inform%filter_size, &
             data%sigma, inform%time%clock_total
         END IF
!        WRITE( data%out, 2110 )                                               &
!          inform%iter, data%p_mode, nlp%f, data%primal_viol, data%comp_viol,  &
!          data%sigma
!      ELSE IF ( inform%status == - 1 .OR. inform%status == - 6 ) THEN
!        WRITE( data%out, "( '   |   ', I3, ES8.2 )" )                         &
!          inform%B_modified, del_ellv_steer
!      ELSE IF ( ( inform%status == - 2 ) ) THEN
!        WRITE( data%out, "( '   |   ', I3, ES8.2, ES12.2 )" )                 &
!          inform%B_modified, del_ellv_steer, data%del_qphi_pred
!      ELSE IF ( inform%status == - 4 .OR. inform%status == - 3 ) THEN
!        WRITE( data%out, "( '   |', 77X, ES8.2 )" ) data%alpha

         IF ( inform%status == GALAHAD_ok ) THEN
           WRITE(  data%out,                                                   &
             "( /, A, ' Approximate locally-optimal solution found' )" ) prefix
         ELSE
           WRITE(  data%out,                                                   &
             "( /, A, ' Approximate infeasible critical point found' )" ) prefix
         END IF
         SELECT CASE ( data%control%predictor_hessian )
         CASE( se_modified_predictor_hessian )
           WRITE( data%out, "( A,                                              &
          &   ' Modified-exact predictor Hessian used' )" )  prefix
         CASE( l_bfgs_predictor_hessian )
           WRITE( data%out, "( A,                                              &
          &  ' L-BFGS predictor Hessian with skipping used' )" ) prefix
         CASE( powell_l_bfgs_predictor_hessian )
           WRITE( data%out, "( A,                                              &
          &  ' L-BFGS predictor Hessian with Powell corrections used' )") prefix
         CASE DEFAULT
           WRITE( data%out, "( A,                                              &
          &   ' Scaled-identity predictor Hessian used' )" )  prefix
         END SELECT
       END IF

!      WRITE( data%out, "( A, ' +', 76( '-' ), '+' )" ) prefix
!      WRITE( data%out, "( A, ' Status: ', I0 )" ) prefix, inform%status
     END IF

!  print details of the solution

     IF ( data%control%print_level > 0 .AND. data%out > 0 ) THEN
       l = 2
       IF ( data%control%full_solution ) l = nlp%n
       IF ( data%control%print_level >= 10 ) l = nlp%n

       names = ALLOCATED( nlp%VNAMES )
       IF ( names ) THEN
         WRITE( data%out, "( /, A, ' Solution: ', /, A, '                   ', &
        &         '             <------ Bounds ------> ', /, A,                &
        &         '      # name          value   ',                            &
        &         '    Lower       Upper       Dual' )" ) prefix, prefix, prefix
       ELSE
         WRITE( data%out, "( /, A, ' Solution: ', /, A, '        ',            &
        &         '           <------ Bounds ------> ', /, A,                  &
        &         '      #    value   ',                                       &
        &         '    Lower       Upper       Dual' )" ) prefix, prefix, prefix
       END IF
       DO j = 1, 2
         IF ( j == 1 ) THEN
           ir = 1 ; ic = MIN( l, nlp%n )
         ELSE
           IF ( names ) THEN
             IF ( ic < nlp%n - l ) WRITE( data%out, 2040 ) prefix
           ELSE
             IF ( ic < nlp%n - l ) WRITE( data%out, 2060 ) prefix
           END IF
           ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
         END IF
         IF ( names ) THEN
           DO i = ir, ic
             WRITE( data%out, 2030 ) prefix, i, nlp%VNAMES( i ), nlp%X( i ),   &
               nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
           END DO
         ELSE
           DO i = ir, ic
             WRITE( data%out, 2050 ) prefix, i, nlp%X( i ),                    &
               nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
           END DO
         END IF
       END DO

       IF ( nlp%m > 0 ) THEN
         l = 2
         IF ( data%control%full_solution ) l = nlp%m
         IF ( data%control%print_level >= 10 ) l = nlp%m

         names = ALLOCATED( nlp%CNAMES )
         IF ( names ) THEN
           WRITE( data%out, "( /, A, ' Constraints:', /, A, '              ',  &
          &       '                    <------ Bounds ------> ', /, A,         &
          &       '      # name           value       ',                       &
          &       'Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
         ELSE
           WRITE( data%out, "( /, A, ' Constraints:', /, A, '              ',  &
          &       '         <------ Bounds ------> ', /, A,                    &
          &       '      #     value       ',                                  &
          &       'Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
         END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1 ; ic = MIN( l, nlp%m )
           ELSE
             IF ( names ) THEN
               IF ( ic < nlp%m - l ) WRITE( data%out, 2040 ) prefix
             ELSE
               IF ( ic < nlp%m - l ) WRITE( data%out, 2060 ) prefix
             END IF
             ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
           END IF
           IF ( names ) THEN
             DO i = ir, ic
               WRITE( data%out, 2030 ) prefix, i, nlp%CNAMES( i ), nlp%C( i ), &
                 nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
             END DO
           ELSE
             DO i = ir, ic
               WRITE( data%out, 2050 ) prefix, i, nlp%C( i ),                  &
                 nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
             END DO
           END IF
         END DO
       END IF

       IF ( nlp%m > 0 ) THEN
         multiplier_norm = MAXVAL( ABS( nlp%Y( : nlp%m ) ) )
       ELSE
         multiplier_norm = zero
       END IF

       WRITE( data%out, "( /, A, ' Problem: ', A10, 17X,                       &
      &          ' Solver: FiSQP', /, A,                                       &
      &  ' n              =     ', bn, I11,                                    &
      &  '     m               =', bn, I11, /,                                 &
      & A, ' Objective      = ', ES15.8,                                       &
      &    '     Complementarity =', ES11.4, /,                                &
      & A, ' Violation      =     ', ES11.4,                                   &
      &    '     Dual infeas.    =', ES11.4, /,                                &
      & A, ' Max multiplier =     ', ES11.4,                                   &
      &    '     Max dual var.   =', ES11.4, /,                                &
      & A, ' Iterations     =     ', bn, I11,                                  &
      &    '     Time            =', F11.2 , /,                                &
      & A, ' Function evals =     ', bn, I11 )" )                              &
        prefix, nlp%pname, prefix, nlp%n, nlp%m, prefix, inform%obj,           &
        inform%complementary_slackness, prefix, inform%primal_infeasibility,   &
        inform%dual_infeasibility, prefix, multiplier_norm,                    &
        MAXVAL( ABS( nlp%Z( : nlp%n ) ) ), prefix, inform%iter,                &
        inform%time%clock_total, prefix, inform%fc_eval
     END IF

     IF ( data%control%error > 0 .AND. data%control%print_level > 0 ) THEN
       SELECT CASE ( inform%status )
       CASE( - 103 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   A step could not be found in backtracking (pmode)' )" )    &
          prefix, 'FiSQP_solve', prefix
       CASE( - 104 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   A step could not be found in backtracking (fmode)' )" )    &
          prefix, 'FiSQP_solve', prefix
       CASE( - 105 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   A step could not be found in the steering subproblem')" )  &
          prefix, 'FiSQP_solve', prefix
       CASE( - 106 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   A step could not be found in the predictor subproblem')" ) &
          prefix, 'FiSQP_solve', prefix
       CASE( - 108 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   The steering step did not reduce the linearized ',         &
        &    'infeasibility' )" ) prefix, 'FiSQP_solve', prefix
       CASE( - 109 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '   tau linesearch computation failed' )" )                    &
          prefix, 'FiSQP_solve', prefix
       CASE( - 111 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '  There are no general constraints' )" )                      &
          prefix, 'FiSQP_solve', prefix
       CASE DEFAULT
         CALL SYMBOLS_status( inform%status, data%control%error, prefix,       &
                              'FiSQP_solve' )
       END SELECT
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, wp )
     inform%time%clock_total = data%clock_now - data%clock_start
     RETURN

!  non-executable statements

 2000 FORMAT( /, A, ' iter       obj fun  pr_feas du_feas cmp_slk actve ',     &
                    '  step  #fil   sigma    time' )
 2010 FORMAT( A, I5, 3A1, ES12.4, 3ES8.1, A6, ES8.1, A1, I4, ES8.1, F8.1 )
 2020 FORMAT( A, I5, '   ', ES12.4, 3ES8.1, '     -    -    ', I4, ES8.1, F8.1 )
 2030 FORMAT( A, I7, 1X, A10, 4ES12.4 )
 2040 FORMAT( A, 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2050 FORMAT( A, I7, 4ES12.4 )
 2060 FORMAT( A, 6X, '.', 4( 2X, 10( '.' ) ) )

!2100 FORMAT( '  Iter   p_mode       f           v       comp_v    sigma   ',  &
!       '   |   B_modified  d_lv      d_qphi_pred   ',                         &
!       ' phi         tau      alpha      pair  fil_size  step' )
!2110 FORMAT( I6, 6X, L1, ES10.2, 3ES11.2 )

!  end of subroutine FISQP_solve

     END SUBROUTINE FISQP_solve

!-*-  G A L A H A D -  F I S Q P _ t e r m i n a t e  S U B R O U T I N E -*-

     SUBROUTINE FISQP_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FISQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( FISQP_control_type ), INTENT( IN ) :: control
     TYPE ( FISQP_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate arrays set for FILTER

     CALL FILTER_terminate( data%FILTER_data, control%FILTER_control,          &
                            inform%FILTER_inform )
     IF ( inform%FILTER_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_inform%status
       inform%alloc_status = inform%FILTER_inform%alloc_status
       inform%bad_alloc = inform%FILTER_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate arrays set for QP

     CALL L1QP_terminate( data%QP_steer_data, control%QP_steer_control,        &
                          inform%QP_steer_inform )
     IF ( inform%QP_steer_inform%status /= 0 ) THEN
       inform%status = inform%QP_steer_inform%status
       inform%alloc_status = inform%QP_steer_inform%alloc_status
       inform%bad_alloc = inform%QP_steer_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL L1QP_terminate( data%QP_pred_data, control%QP_pred_control,          &
                          inform%QP_pred_inform )
     IF ( inform%QP_pred_inform%status /= 0 ) THEN
       inform%status = inform%QP_pred_inform%status
       inform%alloc_status = inform%QP_pred_inform%alloc_status
       inform%bad_alloc = inform%QP_pred_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate arrays set for EQP

     CALL EQP_terminate( data%QP_accel_data, control%QP_accel_control,         &
                         inform%QP_accel_inform )
     IF ( inform%QP_accel_inform%status /= 0 ) THEN
       inform%status = inform%QP_accel_inform%status
       inform%alloc_status = inform%QP_accel_inform%alloc_status
       inform%bad_alloc = inform%QP_accel_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arrays allocated within LMS

     CALL LMS_terminate( data%QP_pred%H_lm, data%control%LMS_control,          &
                         inform%LMS_inform )
     inform%status = inform%LMS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%LMS_inform%alloc_status
       inform%bad_alloc = inform%LMS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate all remaining allocated arrays

     array_name = 'fisqp: data%QP_steer%G'
     CALL SPACE_dealloc_array( data%QP_steer%G,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%H%row'
     CALL SPACE_dealloc_array( data%QP_steer%H%row,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%H%col'
     CALL SPACE_dealloc_array( data%QP_steer%H%col,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%H%val'
     CALL SPACE_dealloc_array( data%QP_steer%H%val,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%A%row'
     CALL SPACE_dealloc_array( data%QP_steer%A%row,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%A%col'
     CALL SPACE_dealloc_array( data%QP_steer%A%col,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%A%val'
     CALL SPACE_dealloc_array( data%QP_steer%A%val,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%C_l'
     CALL SPACE_dealloc_array( data%QP_steer%C_l, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%C_u'
     CALL SPACE_dealloc_array( data%QP_steer%C_u, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%X_l'
     CALL SPACE_dealloc_array( data%QP_steer%X_l,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%X_u'
     CALL SPACE_dealloc_array( data%QP_steer%X_u,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%X'
     CALL SPACE_dealloc_array( data%QP_steer%X,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_steer%C'
     CALL SPACE_dealloc_array( data%QP_steer%C, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%DG'
     CALL SPACE_dealloc_array( data%DG,                                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%DX'
     CALL SPACE_dealloc_array( data%DX,                                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S_ref'
     CALL SPACE_dealloc_array( data%S_ref,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S_steer'
     CALL SPACE_dealloc_array( data%S_steer,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S_pred'
     CALL SPACE_dealloc_array( data%S_pred,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S_accel'
     CALL SPACE_dealloc_array( data%S_accel,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%S_accel_ref'
     CALL SPACE_dealloc_array( data%S_accel_ref,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%X_ref'
     CALL SPACE_dealloc_array( data%X_ref,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%X_trial'
     CALL SPACE_dealloc_array( data%X_trial,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%C_trial'
     CALL SPACE_dealloc_array( data%C_trial,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%Y_accel'
     CALL SPACE_dealloc_array( data%Y_accel,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%Z_accel'
     CALL SPACE_dealloc_array( data%Z_accel,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%WORK_m'
     CALL SPACE_dealloc_array( data%WORK_m,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%WORK_n'
     CALL SPACE_dealloc_array( data%WORK_n,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

    array_name = 'fisqp: data%QP_pred%H%row'
    CALL SPACE_dealloc_array( data%QP_pred%H%row, inform%status,               &
           inform%alloc_status, array_name = array_name,                       &
           bad_alloc = inform%bad_alloc, out = data%error )
    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

    array_name = 'fisqp: data%QP_pred%H%col'
    CALL SPACE_dealloc_array( data%QP_pred%H%col, inform%status,               &
           inform%alloc_status, array_name = array_name,                       &
           bad_alloc = inform%bad_alloc, out = data%error )
    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%H%val'
     CALL SPACE_dealloc_array( data%QP_pred%H%val,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%G'
     CALL SPACE_dealloc_array( data%QP_pred%G,                                 &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%A%row'
     CALL SPACE_dealloc_array( data%QP_pred%A%row,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%A%col'
     CALL SPACE_dealloc_array( data%QP_pred%A%col,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%A%val'
     CALL SPACE_dealloc_array( data%QP_pred%A%val,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%C_l'
     CALL SPACE_dealloc_array( data%QP_pred%C_l, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%C_u'
     CALL SPACE_dealloc_array( data%QP_pred%C_u, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%X_l'
     CALL SPACE_dealloc_array( data%QP_pred%X_l,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%X_u'
     CALL SPACE_dealloc_array( data%QP_pred%X_u,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%X'
     CALL SPACE_dealloc_array( data%QP_pred%X,                                 &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%Y_l'
     CALL SPACE_dealloc_array( data%QP_pred%Y_l,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%Y_u'
     CALL SPACE_dealloc_array( data%QP_pred%Y_u,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%Z_l'
     CALL SPACE_dealloc_array( data%QP_pred%Z_l,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%Z_u'
     CALL SPACE_dealloc_array( data%QP_pred%Z_u,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_pred%C'
     CALL SPACE_dealloc_array( data%QP_pred%C, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%H%row'
     CALL SPACE_dealloc_array( data%QP_accel%H%row,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%H%col'
     CALL SPACE_dealloc_array( data%QP_accel%H%col,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%H%val'
     CALL SPACE_dealloc_array( data%QP_accel%H%val,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%G'
     CALL SPACE_dealloc_array( data%QP_accel%G,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%A%row'
     CALL SPACE_dealloc_array( data%QP_accel%A%row,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%A%col'
     CALL SPACE_dealloc_array( data%QP_accel%A%col,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%A%val'
     CALL SPACE_dealloc_array( data%QP_accel%A%val,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%X'
     CALL SPACE_dealloc_array( data%QP_accel%X,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%C'
     CALL SPACE_dealloc_array( data%QP_accel%C,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%QP_accel%Y'
     CALL SPACE_dealloc_array( data%QP_accel%Y,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fisqp: data%C_scale'
     CALL SPACE_dealloc_array( data%C_scale,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine FISQP_terminate

     END SUBROUTINE FISQP_terminate

!  End of module GALAHAD_FISQP

   END MODULE GALAHAD_FISQP_double
