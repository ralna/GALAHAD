! THIS VERSION: GALAHAD 3.3 - 05/05/2021 AT 14:15 GMT

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ N L S   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.7. October 27th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_NLS_double

!     --------------------------------------------------------------------
!    |                                                                    |
!    |   NLS, an algorithm for nonlinear least-squares                    |
!    |                                                                    |
!    |   Aim: find a (local) minimizer of the objective function          |
!    |                                                                    |
!    |            1/2 ||c(x)||_W^2 = 1/2 sum_i=1^m w_i c_i^2(x)           |
!    |                                                                    |
!    |   where the residual c(x) is a smooth vector-valued function       |
!    |   and the weights w_i are positive                                 |
!    |                                                                    |
!     --------------------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type
     USE GALAHAD_USERDATA_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_PSLS_double
     USE GALAHAD_GLRT_double
     USE GALAHAD_RQS_double
     USE GALAHAD_BSC_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_ROOTS_double
     USE GALAHAD_MOP_double, ONLY: mop_Ax, mop_column_2_norms
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM
     USE GALAHAD_STRING, ONLY: STRING_integer_6

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: NLS_initialize, NLS_read_specfile, NLS_solve, NLS_terminate,    &
               NLS_subproblem_initialize, NLS_subproblem_read_specfile,        &
               NLS_subproblem_solve, NLS_subproblem_terminate,                 &
               NLS_full_initialize, NLS_full_terminate, NLS_import,            &
               NLS_information, NLS_solve_with_mat, NLS_solve_without_mat,     &
               NLS_solve_reverse_with_mat, NLS_solve_reverse_without_mat,      &
               NLS_reset_control, NLPT_problem_type, GALAHAD_userdata_type,    &
               SMT_type, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE NLS_initialize
       MODULE PROCEDURE NLS_initialize, NLS_full_initialize
     END INTERFACE NLS_initialize

     INTERFACE NLS_terminate
       MODULE PROCEDURE NLS_terminate, NLS_full_terminate
     END INTERFACE NLS_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER  :: nskip_prec_max = 0
     INTEGER, PARAMETER  :: history_max = 100
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: quarter = 0.25_wp
     REAL ( KIND = wp ), PARAMETER :: eighth = 0.125_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: sixteenth = 0.0625_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: sixteen = 16.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm6 = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

     REAL ( KIND = wp ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = wp ), PARAMETER :: gamma_2 = half
     REAL ( KIND = wp ), PARAMETER :: gamma_3 = two
     REAL ( KIND = wp ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = wp ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: mu_2 = point1
!    REAL ( KIND = wp ), PARAMETER :: theta = half
     REAL ( KIND = wp ), PARAMETER :: theta = point1
     REAL ( KIND = wp ), PARAMETER :: weight_zero = epsmch

!  models

     INTEGER, PARAMETER  :: dynamic_model = 0
     INTEGER, PARAMETER  :: first_order_model = 1
     INTEGER, PARAMETER  :: diagonal_hessian_model = 2
     INTEGER, PARAMETER  :: gauss_newton_model = 3
     INTEGER, PARAMETER  :: newton_model = 4
     INTEGER, PARAMETER  :: gauss_to_newton_model = 5
     INTEGER, PARAMETER  :: tensor_gauss_newton_model = 6
     INTEGER, PARAMETER  :: tensor_newton_model = 7
     INTEGER, PARAMETER  :: tensor_gauss_to_newton_model = 8

!  regularization norms

     INTEGER, PARAMETER  :: user_regularization = - 3
     INTEGER, PARAMETER  :: lmbfgs_regularization = - 2
     INTEGER, PARAMETER  :: euclidean_regularization = - 1
     INTEGER, PARAMETER  :: automatic_regularization = 0
     INTEGER, PARAMETER  :: diagonal_jtj_regularization = 1
     INTEGER, PARAMETER  :: diagonal_hessian_regularization = 2
     INTEGER, PARAMETER  :: band_regularization = 3
     INTEGER, PARAMETER  :: reordered_band_regularization = 4
     INTEGER, PARAMETER  :: schnabel_eskow_regularization = 5
     INTEGER, PARAMETER  :: gmps_regularization = 6
     INTEGER, PARAMETER  :: lin_more_regularization = 7
     INTEGER, PARAMETER  :: mi28_regularization = 8
     INTEGER, PARAMETER  :: munksgaard_regularization = 9
     INTEGER, PARAMETER  :: expanding_band_regularization = 10

!  weight update strategies

     INTEGER, PARAMETER  :: weight_update_basic = 1
     INTEGER, PARAMETER  :: weight_update_zero_reset = 2
     INTEGER, PARAMETER  :: weight_update_imitate_tr = 3
     INTEGER, PARAMETER  :: weight_update_increase = 4
     INTEGER, PARAMETER  :: weight_update_gpt = 5

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NLS_subproblem_control_type

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

!   the maximum number of iterations performed

       INTEGER :: maxit = 100

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   is the Jacobian matrix of first derivatives available (>= 2), is access
!    only via matrix-vector products (=1) or is it not available (<=0) ?

       INTEGER :: jacobian_available = 1

!   is the Hessian matrix of second derivatives available (>= 2), is access
!    only via matrix-vector products (=1) or is it not available (<=0) ?

       INTEGER :: hessian_available = 0

!   specify the model used. Possible values are
!
!      0  dynamic (*not yet implemented*)
!      1  first-order (no Hessian)
!      2  barely second-order (identity Hessian)
!      3  Gauss-Newton (J^T J Hessian)
!      4  second-order (exact Hessian)
!      5  Gauss-Newton to Newton transition
!      6  tensor Gauss-Newton treated as a least-squares model
!      7  tensor Gauss-Newton treated as a general model
!      8  tensor Gauss-Newton transition from a least-squares to a general model

       INTEGER :: model = gauss_newton_model

!   specify the norm used when regularizing the model problem. The norm is
!    defined via ||v||^2 = v^T S v,  and will also define the preconditioner
!    used for iterative methods. Possible values for S are
!
!     -3  user's own regularization norm
!     -2  S = limited-memory BFGS matrix (with
!          %PSLS_control%lbfgs_vectors history) (*not yet implemented*)
!     -1  identity (= Euclidan two-norm)
!      0  automatic (*not yet implemented*)
!      1  diagonal, S = diag( max( JTJ Hessian, %PSLS_contro%min_diagonal ) )
!      2  diagonal, S = diag( max( Hessian, %PSLS_contro%min_diagonal ) )
!      3  banded, S = band( Hessian ) with semi-bandwidth
!           %PSLS_control%semi_bandwidth
!      4  re-ordered band, P=band(order(A)) with semi-bandwidth
!           %PSLS_control%semi_bandwidth
!      5  full factorization, S = Hessian, Schnabel-Eskow modification
!      6  full factorization, S = Hessian, GMPS modification (*not yet *)
!      7  incomplete factorization of Hessian, Lin-More'
!      8  incomplete factorization of Hessian, HSL_MI28
!      9  incomplete factorization of Hessian, Munskgaard (*not yet *)
!     10  expanding band of Hessian (*not yet implemented*)

       INTEGER :: norm = euclidean_regularization

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!     strategy with this history length used

       INTEGER :: non_monotone = 1

!   define the weight-update strategy:
!        1 (basic), 2 (reset to zero when very successful),
!        3 (imitate TR), 4 (increase lower bound), 5 (GPT)

       INTEGER :: weight_update_strategy = weight_update_basic

!   overall convergence tolerances. The iteration will terminate when
!      ||c||_2 <= MAX( %stop_c_absolute, %stop_c_relative * ||c_initial||_2
!     or when the norm of the gradient g = J^T(x) c(x) / ||c(x)||_2 of ||c||_2,
!      ||g||_2 <= MAX( %stop_g_absolute, %stop_g_relative * ||g_initial||_2,
!     or if the step is less than %stop_s

       REAL ( KIND = wp ) :: stop_c_absolute = tenm6
       REAL ( KIND = wp ) :: stop_c_relative = - one
       REAL ( KIND = wp ) :: stop_g_absolute = tenm6
       REAL ( KIND = wp ) :: stop_g_relative = - one
       REAL ( KIND = wp ) :: stop_s = epsmch

!   the regularization power (<2 => chosen according to the model)

       REAL ( KIND = wp ) :: power = - one

!   initial value for the regularization weight  (-ve => 1/||g_0||)

       REAL ( KIND = wp ) :: initial_weight = hundred

!   minimum permitted regularization weight

       REAL ( KIND = wp ) :: minimum_weight = tenm8

!   initial value for the inner regularization weight for tensor GN (-ve => 0)

       REAL ( KIND = wp ) :: initial_inner_weight = 0.0_wp
!      REAL ( KIND = wp ) :: initial_inner_weight = 0.0001_wp

!   a potential iterate will only be accepted if the actual decrease
!    f - f(x_new) is larger than %eta_successful times that predicted
!    by a quadratic model of the decrease. The regularization weight will be
!    decreaed if this relative decrease is greater than %eta_very_successful
!    but smaller than %eta_too_successful

       REAL ( KIND = wp ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = wp ) :: eta_very_successful = point9
       REAL ( KIND = wp ) :: eta_too_successful = two

!   on very successful iterations, the regularization weight will be reduced
!    by the factor %weight_decrease but no more than %weight_decrease_min
!    while if the iteration is unsucceful, the weight will be increased by a
!    factor %weight_increase but no more than %weight_increase_max
!    (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
!    and Toint, 2011)

       REAL ( KIND = wp ) :: weight_decrease_min = point1
!      REAL ( KIND = wp ) :: weight_decrease = half
       REAL ( KIND = wp ) :: weight_decrease = point1
!      REAL ( KIND = wp ) :: weight_increase = two
       REAL ( KIND = wp ) :: weight_increase = ten
       REAL ( KIND = wp ) :: weight_increase_max = hundred

!  expert parameters as suggested in Gould, Porcelli and Toint, "Updating the
!   regularization parameter in the adaptive cubic regularization algorithm",
!   RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011),
!      http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
!  (these are denoted beta, epsilon_chi and alpha_max in the paper)

       REAL ( KIND = wp ) :: reduce_gap = point01
       REAL ( KIND = wp ) :: tiny_gap = tenm8
       REAL ( KIND = wp ) :: large_root = two

!  if the Gauss-Newto to Newton model is specified, switch to Newton as
!   soon as the norm of the gradient g is smaller than switch_to_newton

       REAL ( KIND = wp ) :: switch_to_newton = 0.1_wp

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   use a direct (factorization) or (preconditioned) iterative method to
!    find the search direction

       LOGICAL :: subproblem_direct = .FALSE.

!   should the weight be renormalized to account for a change in scaling?

       LOGICAL :: renormalize_weight = .FALSE.

!   allow the user to perform a "magic" step to improve the objective

       LOGICAL :: magic_step = .FALSE.

!   print values of the objective/gradient rather than ||c|| and its gradient

       LOGICAL :: print_obj = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for RQS

       TYPE ( RQS_control_type ) :: RQS_control

!  control parameters for GLRT

       TYPE ( GLRT_control_type ) :: GLRT_control

!  control parameters for PSLS

       TYPE ( PSLS_control_type ) :: PSLS_control

!  control parameters for BSC

       TYPE ( BSC_control_type ) :: BSC_control

!  control parameters for ROOTS

       TYPE ( ROOTS_control_type ) :: ROOTS_control

     END TYPE NLS_subproblem_control_type

!  extend newton_control_type so that it contains a copy of itself so that the
!  Newton solve may be called as an internal subroutine to solve subproblems

     TYPE, PUBLIC, EXTENDS( NLS_subproblem_control_type ) :: NLS_control_type
       TYPE ( NLS_subproblem_control_type ) :: subproblem_control
     END TYPE NLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived types with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NLS_time_type

!  the total CPU time spent in the package

       REAL :: total = 0.0

!  the CPU time spent preprocessing the problem

       REAL :: preprocess = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

       REAL :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

       REAL :: factorize = 0.0

!  the CPU time spent computing the search direction

       REAL :: solve = 0.0

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

     END TYPE NLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived types with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NLS_subproblem_inform_type

!  return status. See NLS_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the name of the user-supplied evaluation routine for which an error ocurred

       CHARACTER ( LEN = 12 ) :: bad_eval = REPEAT( ' ', 12 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of CG iterations performed

       INTEGER :: cg_iter = 0

!  the total number of evaluations of the residual function c(x)

       INTEGER :: c_eval = 0

!  the total number of evaluations of the Jacobian J(x) of c(x)

       INTEGER :: j_eval = 0

!  the total number of evaluations of the scaled Hessian H(x,y) of c(x)

       INTEGER :: h_eval = 0

!  the maximum number of factorizations in a sub-problem solve

       INTEGER :: factorization_max = 0

!  the return status from the factorization

       INTEGER :: factorization_status = 0

!   the maximum number of entries in the factors

       INTEGER ( KIND = long ) :: max_entries_factors = 0

!  the total integer workspace required for the factorization

       INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

       INTEGER :: factorization_real = - 1

!  the average number of factorizations per sub-problem solve

       REAL ( KIND = wp ) :: factorization_average = zero

!  the value of the objective function 1/2||c(x)||^2_W at the best estimate of
!   the solution, x, determined by NLS_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the residual ||c(x)||_W at the best estimate of the solution,
!   x, determined by NLS_solve

       REAL ( KIND = wp ) :: norm_c = HUGE( one )

!  the norm of the gradient of ||c(x)||_W of the objective function
!   at the best estimate, x, of the solution determined by NLS_solve

       REAL ( KIND = wp ) :: norm_g = HUGE( one )

!  the final regularization weight used

       REAL ( KIND = wp ) :: weight = one

!  timings (see above)

       TYPE ( NLS_time_type ) :: time

!  inform parameters for RQS

       TYPE ( RQS_inform_type ) :: RQS_inform

!  inform parameters for GLRT

       TYPE ( GLRT_inform_type ) :: GLRT_inform

!  inform parameters for PSLS

       TYPE ( PSLS_inform_type ) :: PSLS_inform

!  inform parameters for BSC

       TYPE ( BSC_inform_type ) :: BSC_inform

!  inform parameters for ROOTS

       TYPE ( ROOTS_inform_type ) :: ROOTS_inform

     END TYPE NLS_subproblem_inform_type

!  extend newton_inform_type so that it contains a copy of itself so that the
!  Newton solve may be called as an internal subroutine to solve subproblems

     TYPE, PUBLIC, EXTENDS( NLS_subproblem_inform_type ) :: NLS_inform_type
       TYPE ( NLS_subproblem_inform_type ) :: subproblem_inform
     END TYPE NLS_inform_type

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   regularization derived type, sigma ||.||_S^p / p, where ||x||_S^2 = x^T S x
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NLS_regularization_data_type

! the stabilisation weight, sigma (if any)

        REAL ( KIND = wp ) :: weight = 0.0_wp

! the stabilisation power, p

        REAL ( KIND = wp ) :: power = 2.0_wp

!  the stabilisation scaling matrix, S

       TYPE ( SMT_type ) :: matrix

     END TYPE NLS_regularization_data_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: NLS_subproblem_data_type
       INTEGER :: branch = 1
       INTEGER :: branch_newton = 1
       INTEGER :: eval_status, out, start_print, stop_print, regularization_type
       INTEGER :: print_level, print_level_glrt, print_level_rqs, ref( 1 )
       INTEGER :: len_history, ibound, ipoint, icp, lbfgs_mem, max_hist, jtj_ne
       INTEGER :: nskip_lbfgs, nskip_prec, non_monotone_history
       INTEGER :: print_gap, max_diffs, latest_diff, total_diffs, model_used
       INTEGER :: total_facts, h_ne, s_ne
       REAL :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now, delta, power
       REAL ( KIND = wp ) :: f_ref, f_trial, f_best, m_best, model, ratio, rp
       REAL ( KIND = wp ) :: weight, old_weight, weight_trial, etat, ometat
       REAL ( KIND = wp ) :: df, stg, hstbs, s_norm, weight_max, xtsx_current
       REAL ( KIND = wp ) :: stop_c, stop_g, s_new_norm, norm_c_trial
       REAL ( KIND = wp ) :: a0, a1, a2, a3, a4, steplength, g_norm, phi
       REAL ( KIND = wp ) :: inner_weight, s_norm_successful, final_weight, xtsx
       REAL ( KIND = wp ) :: minimum_weight, obj_current, norm_c_current
       LOGICAL :: printi, printt, printd, printw, printm, gauss_to_newton_model
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm, set_printw
       LOGICAL :: monotone, new_point, got_j, got_h, poor_model, reduce, n_or_gn
       LOGICAL :: reverse_c, reverse_j, reverse_h, reverse_jprod, reverse_hprod
       LOGICAL :: reverse_scale, reverse_hprods, non_trivial_regularization
       LOGICAL :: successful, transpose, form_regularization, f_is_nan, g_is_nan
       LOGICAL :: stabilised, hessian_available, jacobian_available, re_entry
       LOGICAL :: w_eq_identity, step_accepted, hessian_computed, map_h_to_jtj
       CHARACTER ( LEN = 1 ) :: negcur = ' '
       CHARACTER ( LEN = 1 ) :: perturb = ' '
       CHARACTER ( LEN = 1 ) :: hard = ' '
       CHARACTER ( LEN = 1 ) :: accept = ' '
       TYPE ( RQS_history_type ), DIMENSION( history_max ) :: history
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PAST
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PTR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ORDER
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_map
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Hs_map
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: S_map
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: F_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HSHS

       TYPE ( SMT_type ) :: JT
       TYPE ( SMT_type ) :: H

       TYPE ( NLS_regularization_data_type ) :: regularization

!  copys of controls

       TYPE ( NLS_control_type ) :: control
       TYPE ( NLS_subproblem_control_type ) :: subproblem_control
       TYPE ( NLS_inform_type ) :: inform

!  data for RQS

       TYPE ( RQS_data_type ) :: RQS_data

!  data for GLRT

       TYPE ( GLRT_data_type ) :: GLRT_data

!  data for PSLS

       TYPE ( PSLS_data_type ) :: PSLS_data

!  data for BSC

       TYPE ( BSC_data_type ) :: BSC_data
       TYPE ( BSC_control_type ) :: BSC_control_tensor_model

!  data for tensor-model minimization

       TYPE ( NLPT_problem_type ):: tensor_model
       TYPE ( GALAHAD_userdata_type ) :: subproblem_userdata

!  data for ROOTS

       TYPE ( ROOTS_data_type ) :: ROOTS_data

     END TYPE NLS_subproblem_data_type

!  extend newton_data_type so that it contains a copy of itself so that the
!  Newton solve may be called as an internal subroutine to solve subproblems

     TYPE, PUBLIC, EXTENDS( NLS_subproblem_data_type ) :: NLS_data_type
       TYPE ( NLS_subproblem_data_type ) :: subproblem_data
     END TYPE NLS_data_type

     TYPE, PUBLIC :: NLS_full_data_type
       LOGICAL :: f_indexing
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       TYPE ( NLS_data_type ) :: NLS_data
       TYPE ( NLS_control_type ) :: NLS_control
       TYPE ( NLS_inform_type ) :: NLS_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( GALAHAD_userdata_type ) :: userdata
     END TYPE NLS_full_data_type

   CONTAINS

! G A L A H A D - N E W T O N _ N L S _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE NLS_subproblem_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for NLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_subproblem_control_type ), INTENT( OUT ) :: control
     TYPE ( NLS_subproblem_inform_type ), INTENT( OUT ) :: inform
     TYPE ( NLS_subproblem_data_type ), INTENT( INOUT ) :: data

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize RQS components

     CALL RQS_initialize( data%RQS_data, control%RQS_control,                  &
                          inform%RQS_inform )
     control%RQS_control%prefix = '" - RQS:"                     '

!  initalize GLRT components

     CALL GLRT_initialize( data%GLRT_data, control%GLRT_control,               &
                           inform%GLRT_inform )
     control%GLRT_control%prefix = '" - GLRT:"                    '

!  initalize PSLS components

     CALL PSLS_initialize( data%PSLS_data, control%PSLS_control,               &
                           inform%PSLS_inform )
     control%PSLS_control%prefix = '" - PSLS:"                    '

!  initalize BSC components

     CALL BSC_initialize( data%BSC_data, control%BSC_control,                  &
                          inform%BSC_inform )
     control%BSC_control%prefix = '" - BSC:"                     '

!  initalize ROOTS components

     CALL ROOTS_initialize( data%ROOTS_data, control%ROOTS_control,            &
                            inform%ROOTS_inform )
     control%ROOTS_control%tol = epsmch ** 0.75
     control%ROOTS_control%prefix = '" - ROOTS:"                   '

!  initial private data. Set branch for initial entry

     data%branch = 1 ; data%branch_newton = 1

     RETURN

!  End of subroutine NLS_subproblem_initialize

     END SUBROUTINE NLS_subproblem_initialize

!-*-*-  G A L A H A D -  N L S _ I N I T I A L I Z E  S U B R O U T I N E  -*-*-

     SUBROUTINE NLS_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for NLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_control_type ), INTENT( OUT ) :: control
     TYPE ( NLS_inform_type ), INTENT( OUT ) :: inform
     TYPE ( NLS_data_type ), INTENT( INOUT ) :: data

!  control parameters for the main iteration

     CALL NLS_subproblem_initialize( data%NLS_subproblem_data_type,            &
                                     control%NLS_subproblem_control_type,      &
                                     inform%NLS_subproblem_inform_type )

!  control parameters for the tensor subproblem solves

     CALL NLS_subproblem_initialize( data%subproblem_data,                     &
                                     control%subproblem_control,               &
                                     inform%subproblem_inform )
     control%subproblem_control%model = gauss_newton_model
     control%subproblem_control%maxit = 50
     control%subproblem_control%print_obj = .TRUE.

     RETURN

!  End of subroutine NLS_initialize

     END SUBROUTINE NLS_initialize

!- G A L A H A D -  N L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE NLS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for NLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLS_control_type ), INTENT( OUT ) :: control
     TYPE ( NLS_inform_type ), INTENT( OUT ) :: inform

     CALL NLS_initialize( data%nls_data, control, inform )

     RETURN

!  End of subroutine NLS_full_initialize

     END SUBROUTINE NLS_full_initialize

!-*-  N L S _ N E W T O N _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-

     SUBROUTINE NLS_subproblem_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by NLS_initialize could (roughly)
!  have been set as:

! BEGIN NLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  jacobian-available                              2
!  hessian-available                               2
!  history-length-for-non-monotone-descent         0
!  model-used                                      2
!  norm-used                                       1
!  weight-update-strategy                          0
!  absolute-residual-accuracy-required             1.0D-8
!  relative-residual-reduction-required            2.0D-16
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-8
!  minimum-step-allowed                            2.0D-16
!  regularization-power                            -1.0D+0
!  initial-regularization-weight                   1.0D+0
!  minimum-regularization-weight                   1.0D-9
!  initial-inner-regularization-weight             0.0D+0
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  too-successful-iteration-tolerance              2.0
!  regularization-weight-minimum-decrease-factor   0.1
!  regularization-weight-decrease-factor           0.5
!  regularization-weight-increase-factor           2.0
!  regularization-weight-maximum-increase-factor   100.0
!  reduce-gap                                      0.01
!  large-root                                      2.0
!  tiny-gap                                        1.0D-8
!  switch-to-newton-tolerance                      0.1
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  sub-problem-direct                              no
!  choose-magic-step                               no
!  print-objective                                 no
!  renormalize-weight                              no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END NLS SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_subproblem_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: maxit = print_gap + 1
     INTEGER, PARAMETER :: alive_unit = maxit + 1
     INTEGER, PARAMETER :: jacobian_available = alive_unit + 1
     INTEGER, PARAMETER :: hessian_available = jacobian_available + 1
     INTEGER, PARAMETER :: model = hessian_available + 1
     INTEGER, PARAMETER :: norm = model + 1
     INTEGER, PARAMETER :: non_monotone = norm + 1
     INTEGER, PARAMETER :: weight_update_strategy = non_monotone + 1
     INTEGER, PARAMETER :: stop_c_absolute = weight_update_strategy + 1
     INTEGER, PARAMETER :: stop_c_relative = stop_c_absolute + 1
     INTEGER, PARAMETER :: stop_g_absolute = stop_c_relative + 1
     INTEGER, PARAMETER :: stop_g_relative = stop_g_absolute + 1
     INTEGER, PARAMETER :: stop_s = stop_g_relative + 1
     INTEGER, PARAMETER :: power = stop_s + 1
     INTEGER, PARAMETER :: initial_weight = power + 1
     INTEGER, PARAMETER :: minimum_weight = initial_weight + 1
     INTEGER, PARAMETER :: initial_inner_weight = minimum_weight + 1
     INTEGER, PARAMETER :: eta_successful = initial_inner_weight + 1
     INTEGER, PARAMETER :: eta_very_successful = eta_successful + 1
     INTEGER, PARAMETER :: eta_too_successful = eta_very_successful + 1
     INTEGER, PARAMETER :: weight_decrease_min = eta_too_successful + 1
     INTEGER, PARAMETER :: weight_decrease = weight_decrease_min + 1
     INTEGER, PARAMETER :: weight_increase = weight_decrease + 1
     INTEGER, PARAMETER :: weight_increase_max = weight_increase + 1
     INTEGER, PARAMETER :: reduce_gap = weight_increase_max + 1
     INTEGER, PARAMETER :: tiny_gap = reduce_gap + 1
     INTEGER, PARAMETER :: large_root = tiny_gap + 1
     INTEGER, PARAMETER :: switch_to_newton = large_root + 1
     INTEGER, PARAMETER :: cpu_time_limit = switch_to_newton + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: subproblem_direct = clock_time_limit + 1
     INTEGER, PARAMETER :: renormalize_weight = subproblem_direct + 1
     INTEGER, PARAMETER :: magic_step = renormalize_weight + 1
     INTEGER, PARAMETER :: print_obj = magic_step + 1
     INTEGER, PARAMETER :: space_critical = print_obj + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 6 ), PARAMETER :: specname = 'NLS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( alive_unit )%keyword = 'alive-device'
     spec( jacobian_available )%keyword = 'jacobian-available'
     spec( hessian_available )%keyword = 'hessian-available'
     spec( model )%keyword = 'model-used'
     spec( norm )%keyword = 'norm-used'
     spec( non_monotone )%keyword = 'history-length-for-non-monotone-descent'
     spec( weight_update_strategy )%keyword = 'weight-update-strategy'

!  Real key-words

     spec( stop_c_absolute )%keyword = 'absolute-residual-accuracy-required'
     spec( stop_c_relative )%keyword = 'relative-residual-reduction-required'
     spec( stop_g_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_g_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( power )%keyword = 'regularization-power'
     spec( initial_weight )%keyword = 'initial-regularization-weight'
     spec( minimum_weight )%keyword = 'minimum-regularization-weight'
     spec( initial_inner_weight )%keyword =                                    &
         'initial-inner-regularization-weight'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( eta_too_successful )%keyword = 'too-successful-iteration-tolerance'
     spec( weight_decrease_min )%keyword =                                     &
       'regularization-weight-minimum-decrease-factor'
     spec( weight_decrease )%keyword = 'regularization-weight-decrease-factor'
     spec( weight_increase )%keyword = 'regularization-weight-increase-factor'
     spec( weight_increase_max )%keyword =                                     &
       'regularization-weight-maximum-increase-factor'
     spec( reduce_gap )%keyword = 'reduce-gap'
     spec( tiny_gap )%keyword = 'tiny-gap'
     spec( large_root )%keyword = 'large-root'
     spec( switch_to_newton )%keyword = 'switch-to-newton-tolerance'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( subproblem_direct )%keyword = 'sub-problem-direct'
     spec( magic_step )%keyword = 'choose-magic-step'
     spec( print_obj )%keyword = 'print-objective'
     spec( renormalize_weight )%keyword = 'renormalize-weight'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( alive_file )%keyword = 'alive-filename'
     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec,                  &
                           control%error )
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
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( jacobian_available ),                   &
                                 control%jacobian_available,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( model ),                                &
                                 control%model,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( norm ),                                 &
                                 control%norm,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( non_monotone ),                         &
                                 control%non_monotone,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_update_strategy ),               &
                                 control%weight_update_strategy,               &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( stop_c_absolute ),                      &
                                 control%stop_c_absolute,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_c_relative ),                      &
                                 control%stop_c_relative,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_g_absolute ),                      &
                                 control%stop_g_absolute,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_g_relative ),                      &
                                 control%stop_g_relative,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( power ),                                &
                                 control%power,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_weight ),                       &
                                 control%initial_weight,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minimum_weight ),                       &
                                 control%minimum_weight,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_inner_weight ),                 &
                                 control%initial_inner_weight,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_successful ),                       &
                                 control%eta_successful,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_very_successful ),                  &
                                 control%eta_very_successful,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_too_successful ),                   &
                                 control%eta_too_successful,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_decrease_min ),                  &
                                 control%weight_decrease_min,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_decrease ),                      &
                                 control%weight_decrease,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_increase ),                      &
                                 control%weight_increase,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_increase_max ),                  &
                                 control%weight_increase_max,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduce_gap ),                           &
                                 control%reduce_gap,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tiny_gap ),                             &
                                 control%tiny_gap,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( large_root ),                           &
                                 control%large_root,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( switch_to_newton ),                     &
                                 control%switch_to_newton,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( subproblem_direct ),                    &
                                 control%subproblem_direct,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( magic_step ),                           &
                                 control%magic_step,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_obj ),                            &
                                 control%print_obj,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( renormalize_weight ),                   &
                                 control%renormalize_weight,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( alive_file ),                           &
                                 control%alive_file,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  read the controls for the sub-problem solvers and regularizations

     IF ( PRESENT( alt_specname ) ) THEN
       CALL RQS_read_specfile( control%RQS_control, device,                    &
              alt_specname = TRIM( alt_specname ) // '-RQS' )
       CALL GLRT_read_specfile( control%GLRT_control, device,                  &
              alt_specname = TRIM( alt_specname ) // '-GLRT' )
       CALL PSLS_read_specfile( control%PSLS_control, device,                  &
              alt_specname = TRIM( alt_specname ) // '-PSLS'  )
       CALL ROOTS_read_specfile( control%ROOTS_control, device,                &
              alt_specname = TRIM( alt_specname ) // '-ROOTS' )
     ELSE
       CALL RQS_read_specfile( control%RQS_control, device )
       CALL GLRT_read_specfile( control%GLRT_control, device )
       CALL PSLS_read_specfile( control%PSLS_control, device )
       CALL ROOTS_read_specfile( control%ROOTS_control, device )
     END IF

     RETURN

!  End of subroutine NLS_subproblem_read_specfile

     END SUBROUTINE NLS_subproblem_read_specfile

!-*-*-*-*-   N L S _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE NLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by NLS_initialize could (roughly)
!  have been set as:

! BEGIN NLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  jacobian-available                              2
!  hessian-available                               2
!  history-length-for-non-monotone-descent         0
!  model-used                                      2
!  norm-used                                       1
!  weight-update-strategy                          0
!  absolute-residual-accuracy-required             1.0D-8
!  relative-residual-reduction-required            2.0D-16
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-8
!  minimum-step-allowed                            2.0D-16
!  regularization-power                            -1.0D+0
!  initial-regularization-weight                   1.0D+0
!  minimum-regularization-weight                   1.0D-9
!  initial-inner-regularization-weight             0.0D+0
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  too-successful-iteration-tolerance              2.0
!  regularization-weight-minimum-decrease-factor   0.1
!  regularization-weight-decrease-factor           0.5
!  regularization-weight-increase-factor           2.0
!  regularization-weight-maximum-increase-factor   100.0
!  reduce-gap                                      0.01
!  large-root                                      2.0
!  tiny-gap                                        1.0D-8
!  switch-to-newton-tolerance                      0.1
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  sub-problem-direct                              no
!  renormalize-weight                              no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END NLS SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  reset control parameters for the main iteration

     CALL NLS_subproblem_read_specfile( control%NLS_subproblem_control_type,   &
                                        device, alt_specname )

!  reset control parameters for the tensor subproblem solves

     IF ( PRESENT( alt_specname ) ) THEN
       CALL NLS_subproblem_read_specfile( control%subproblem_control, device,  &
            alt_specname = TRIM( alt_specname ) // '-NLS-INNER' )
     ELSE
       CALL NLS_subproblem_read_specfile( control%subproblem_control, device,  &
            alt_specname = 'NLS-INNER' )
     END IF

     RETURN

!  End of subroutine NLS_read_specfile

     END SUBROUTINE NLS_read_specfile

!-*-*-*-*-  G A L A H A D -  N L S _ s o l v e  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE NLS_solve( nlp, control, inform, data, userdata,               &
                           W, eval_C, eval_J, eval_H, eval_JPROD,              &
                           eval_HPROD, eval_HPRODS, eval_SCALE )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  NLS_solve, a regularization method for finding a local unconstrained
!    minimizer of a nonlinear least-squares objective, 1/2 ||c(x)||_W^2
!    = 1/2 sum_i w_i c_i^2(x)

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_NLS.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_NLS_double
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m is a scalar variable of type default integer, that holds the number of
!   residuals
!
!  C is a rank-one allocatable array of dimension m and type default real,
!   that holds the residuals c(x). The i-th component of C, i = 1,  ... ,  m,
!   contains c_i(x).
!
!  J is scalar variable of type SMT_TYPE that holds the Jacobian matrix
!   J(x) = nabla r(x), i.e., J_i,j(x) = d r_i(x) / d x_j. The following
!   components are used here:
!
!   J%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of J%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of J%type must
!    contain the string COORDINATE, and for the sparse row-wise storage scheme,
!    the first fourteen components of J%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into J%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Jacobian we wish
!    to store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%J%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   J%ne is a scalar variable of type default integer, that holds the number
!    of entries in the Jacobian J(x) in the sparse co-ordinate storage scheme.
!    It need not be set for any of the other two schemes.
!
!   J%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries in the Jacobian J(x) in any of the available
!    storage schemes.
!
!   J%row is a rank-one allocatable array of type default integer, that holds
!    the row indices in the Jacobian J(x) in the sparse co-ordinate storage
!    scheme. It need not be allocated for any of the other two schemes.
!
!   J%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the Jacobian J(x) in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme.
!    It need not be allocated when the dense scheme is used.
!
!   J%ptr is a rank-one allocatable array of dimension m+1 and type default
!    integer, that holds the starting position of each row of Jacobian J(x),
!    as well as J%ptr(m+1) = the total number of entries plus one, in the
!    sparse row-wise storage scheme. It need not be allocated when the other
!    schemes are used.
!
!  H is scalar variable of type SMT_TYPE that holds the scaled Hessian matrix
!   H(x) = sum_{i=1}^m c_i(x) H_i(x), where H_i(x) is the Hessian of c_i(x).
!   The following components are used here:
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
!         CALL SMT_put( nlp%H%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   H%ne is a scalar variable of type default integer, that holds the number of
!    entries in the lower triangular part of H(x) in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the lower triangular part of H(x) in the sparse
!    co-ordinate storage scheme. It need not be allocated for any of the other
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the  lower triangular part of H(x) in
!    either the sparse co-ordinate, or the sparse row-wise storage scheme. It
!    need not be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of each row of the lower
!    triangular part of H(x), as well as H%ptr(n+1) = the total number of
!    entries plus one, in the sparse row-wise storage scheme. It need not be
!    allocated when the other schemes are used.
!
!  P is scalar variable of type SMT_TYPE that holds the matrix of
!   residual-Hessian-vector products P(x,v) = (H_1(x)v,...,H_m(x)v) for
!   a given v. The following components are used here:
!
!   P%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first sixteen components of P%type must contain the string
!    DENSE_BY_COLUMNS. For the sparse co-ordinate scheme, the first ten
!    components of P%type must contain the string COORDINATE, and for the
!    sparse column-wise storage scheme, the first seventeen components of
!    P%type must contain the string SPARSE_BY_COLUMNS.
!    ** NB ** COORDINATE not yet implemented
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into H%type. For example, if nlp is
!    of derived type packagename_problem_type and involves product matrix
!    that we wish to store using the sparse column-wise scheme, we may simply
!
!         CALL SMT_put( nlp%P%type, 'SPARSE_BY_COLUMNS', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   P%ne is a scalar variable of type default integer, that holds the number of
!    entries in P(x,v) in the sparse co-ordinate storage scheme. It need not
!    be set for any of the other permitted schemes.
!
!   P%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the product matrix P in any of the available
!    storage schemes. For the dense scheme P is stored by columns.
!
!   P%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of P(x,v) in either the sparse co-ordinate storage
!    scheme or the sparse column-wise storage scheme. It  need not be
!    allocated when the dense scheme is used.
!
!   P%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of P(x,v) in the sparse co-ordinate scheme
!    It need not be allocated when the other storage schemes are used.
!
!   P%ptr is a rank-one allocatable array of dimension m+1 and type default
!    integer, that holds the starting position of each column of P(x,v),
!    as well as P%ptr(m+1) = the total number of entries plus one, in the
!    sparse column-wise storage scheme. It need not be allocated when the
!    other schemes are used.
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
!  CNAMES is a rank-one allocatable array of dimension m and type default
!   character and length 10, whose i-th entry contains the ``name'' of the i-th
!   residual for printing. This is only used  if ``debug''printing
!   control%print_level > 4) is requested, and will be ignored if the array is
!   not allocated.
!
! control is a scalar variable of type NLS_control_type. See
!  NLS_initialize for details
!
! inform is a scallar variable of type NLS_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will
!  have been set in inform(1):
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
!
!     0. The run was successful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending
!        array is written on unit control%error and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains
!        its relevant string 'DENSE', 'COORDINATE' or 'SPARSE_BY_ROWS'
!          has been violated.
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
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named
!        control%alive_file from unit unit control%alive_unit.
!
!     2. The user should compute the residual function value c(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The value of
!        the i-th component of the residual should be set in nlp%C(i), for i =
!        1, ..., m and data%eval_status should be set to 0. If the user is
!        unable to evaluate a component of c(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%C, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the Jacobian of the residual function J(x) =
!        nabla_x c(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value l-th component of the Jacobian stored according
!        to the scheme input in the remainder of nlp%J should be set in
!        nlp%J%val(l), for l = 1, ..., nlp%J%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of J(x) -
!        for instance if a component of the Jacobian is undefined at x - the
!        user need not set nlp%J%val, but should then set data%eval_status
!        to a non-zero value.
!     4. The user should compute the weighted Hessian of the residual function
!        H(x,y) = sum_{i=1}^m y_i nabla_xx y_i(x) at the point x indicated
!        in nlp%X with weights y given by data%Y, and then re-enter the
!        subroutine. The value l-th component of H(x,y) stored according to
!        the scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of H(x,y) -
!        for instance, if a component of the Hessian is undefined at (x,y) - the
!        user need not set nlp%H%val, but should then set data%eval_status
!        to a non-zero value.
!     5. The user should compute the product J(x)v (when transpose = .FALSE.)
!        or J^T(x)v (when transpose = .TRUE.) of the Jacobian of the residual
!        function J(x) (or its traspose) at the point x indicated in nlp%X
!        with the vector v, and add the result to the vector u and then re-enter
!        the subroutine. The logical transpose and vectors u and v are given
!        in data%transpose, data%U and data%V respectively, the
!        resulting vector u + J(x) or u + J^T(x)v as appropriate should be set
!        in data%U and data%eval_status should be set to 0. If the user
!        is unable to evaluate the product - for instance, if a component of
!        J(x) is undefined at x - the user need not alter data%U, but
!        should then set data%eval_status to a non-zero value.
!     6. The user should compute the product H(x,y)v of the Hessian of
!        the residual function H(x,y) at the point (x,y) indicated in nlp%X
!        and data%Y with the vector v and add the result to the vector u
!        and then re-enter the subroutine. The vectors u and v are given in
!        data%U and data%V respectively, the resulting vector
!        u + H(x,y)v should be set in data%U and  data%eval_status
!        should be set to 0. If the user is unable to evaluate the product -
!        for instance, if a component of H(x,y) is undefined at (x,y) - the
!        user need not alter data%U, but should then set
!        data%eval_status to a non-zero value.
!     7. The user should compute the matrix whose columns are the products
!        H_i(x)v between the HessianH_i(x) of the ith residual function at
!        the point x indicated in nlp%X a given vector v held in data%V.
!        The nonzeros for column i must be stored in nlp%P%val(l), for
!        l = nlp%P%ptr(i), ...,  nlp%P%ptr(i+1) for each i = 1,...,m,
!        in the same order as the row indices were assigned on input in
!        nlp%P%row(l). If the user is unable to evaluate the products -
!        for instance, if a component of H_i(x) is undefined at x - the
!        user need not assign nlp%P%val, but should then set
!        data%eval_status to a non-zero value.
!     8. The user should compute the product u = S(x)v of their preconditioner
!        S(x) at the point x indicated in nlp%X with the vector v. The vectors
!        v is given in data%V, the resulting vector u = S(x)v should be set
!        in data%U and data%eval status should be set to 0. If the user is
!        unable to evaluate the product—for instance, if a component of the
!        preconditioner is undefined at x—the user need not set data%U, but
!        should then set data%eval statusto a non-zero value.
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
!  factorization_status is a scalar variable of type default integer, that
!   gives the return status from the matrix factorization.
!
!  factorization_integer is a scalar variable of type default integer,
!   that gives the amount of integer storage used for the matrix factorization.
!
!  factorization_real is a scalar variable of type default integer,
!   that gives the amount of real storage used for the matrix factorization.
!
!  c_eval is a scalar variable of type default integer, that gives the
!   total number of residual function evaluations performed.
!
!  j_eval is a scalar variable of type default integer, that gives the
!   total number of residual Jacobian evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of scaled Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function 1/2 ||c(x)||_2^2 at the best estimate
!   of the solution found.
!
!  norm_c is a scalar variable of type default real, that holds the value of
!   the norm of the residual function ||c(x)||_2 at the best estimate of the
!   solution found.
!
!  norm_g is a scalar variable of type default real, that holds the value of
!   the norm of the residual function gradient ||J^T(x)c(x)||_2/||c(x)||_2
!   at the best estimate of the solution found.
!
!  time is a scalar variable of type NLS_time_type whose components are
!   used to hold elapsed CPU and clock times for the various parts of the
!   calculation.
!
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    preprocess is a scalar variable of type default real, that gives the
!      CPU time spent reordering the problem to standard form prior to solution.
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
!    clock_preprocess is a scalar variable of type default real, that gives
!      the clock time spent reordering the problem to standard form prior
!      to solution.
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
!  data is a scalar variable of type NLS_data_type used for internal data.
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
!  W is an optional rank-one array of type default real that if present
!   must be of length nlp%m and filled with the weights w_i > 0. If W is
!   absent, weights of one will be used.
!
!  eval_C is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the residual
!   function c(x) evaluated at x=X must be returned in C, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_C is not present, NLS_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_J is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Jacobian
!   nabla_x c(x) of the residual function evaluated at x=X must be returned in
!   J_val in the same order as presented in nlp%J,, and the status variable set
!   to 0. If the evaluation is impossible at X, status should be set to a
!   nonzero value. If eval_J is not present, NLS_solve will return to the
!   user with inform%status = 3 each time an evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the weighted Hessian
!   H(x,y) = sum_i y_i nabla_xx c_i(x) of the residual function evaluated at
!   x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_H is not present,
!   NLS_solve will return to the user with inform%status = 4 each time an
!   evaluation is required.
!
!  eval_JPROD is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum u + J(x) v,
!   (when transpose=.FALSE.) or u + J^T(x) v (when transpose=.TRUE.)
!   of the Jacobian (or its transpose) evaluated  at x=X with the vector v=V
!   and the vector u=U must be returned in U, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_JPROD is not present, NLS_solve will return to the user
!   with inform%status = 5 each time an evaluation is required. The Jacobian
!   has already been evaluated or used at x=X if got_j is .TRUE.
!
!  eval_HPROD is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum u + H(x,y) v,
!   where H(x,y) = sum_i y_i nabla_xx c_i(x), of u=U and the product of the
!   weighted Hessian HC(x,y) evaluated at x=X and y=Y with the vector v=V,
!   and the vector u=U must be returned in U, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_HPROD is not present, NLS_solve will return to the user
!   with inform%status = 6 each time an evaluation is required. The Hessian
!   has already been evaluated or used at x=X if got_h is .TRUE.
!
!  eval_HPRODS is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The nonzeros of
!   the matrix whose ith column is the product nabla_xx c_i(x) v between
!   the Hessian of the ith residual function evaluated at x=X and the
!   vector v=V must be returned in P_val in the same order as presented in
!   nlp%P, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HPRODS is not
!   present, NLS_solve will return to the user with inform%status = 7
!   each time an evaluation is required. The Hessians have already been
!   evaluated or used at x=X if got_h is .TRUE.
!
!  eval_SCALE is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = S(x) v of the
!   user's regularization scaling matrix S(x) evaluated at x=X with the vector
!   v=V, the result u must be retured in U, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_SCALE is not present, NLS_solve will return to the user
!   with inform%status = 8 each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( NLS_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( NLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: W
     OPTIONAL :: eval_C, eval_J, eval_H, eval_JPROD, eval_HPROD, eval_HPRODS,  &
                 eval_SCALE

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_C( status, X, userdata, C )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_C
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_J( status, X, userdata, J_val )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_J
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_H( status, X, Y, userdata, H_val )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_JPROD( status, X, userdata, transpose, U, V, got_j )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_j
       END SUBROUTINE eval_JPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HPROD( status, X, Y, userdata, U, V, got_h )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HPRODS( status, X, V, userdata, P_val, got_h )
       USE GALAHAD_USERDATA_double, ONLY: GALAHAD_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HPRODS
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_SCALE( status, X, userdata, U, V )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_SCALE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, ic, ir, l, ll, nroots
     REAL ( KIND = wp ) :: ared, prered, rounding, root1, root2, root3, alpha
     REAL ( KIND = wp ) :: c0, c1, c2, c3, sths, reg, val

     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_facts
     CHARACTER ( LEN = 80 ) :: array_name
!    REAL ( KIND = wp ), DIMENSION( nlp%n ) :: V
!    REAL ( KIND = wp ) :: H_dense( nlp%n, nlp%n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 ) prefix =                         &
       control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF

!  initial input

     IF ( inform%status == 1 ) THEN

       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  ensure that input parameters are within allowed ranges

       IF ( nlp%n <= 0 .OR. nlp%m <= 0 ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input m, n = ', I0, ', ', I0, ' not permitted' )" )  &
           prefix, nlp%m , nlp%n
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END IF

!  check that the Jacobian is available in some form (may change in future)

       IF ( control%jacobian_available <= 0 ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: jacobian must be available in some form' )" ) prefix
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END IF

!  see if W = I
       data%w_eq_identity = .NOT. PRESENT( W )
       IF ( .NOT. data%w_eq_identity ) THEN
         IF ( COUNT( W( : nlp%m ) <= zero ) > 0 ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: input entries of W must be strictly positive' )" ) &
             prefix
           inform%status = GALAHAD_error_restrictions
           GO TO 990
         END IF
       END IF

!  branch to special code for Newton or Gauss-Newton variants

       IF (  control%model == first_order_model .OR.                           &
             control%model == diagonal_hessian_model .OR.                      &
             control%model == gauss_newton_model .OR.                          &
             control%model == newton_model .OR.                                &
             control%model == gauss_to_newton_model ) THEN
         data%subproblem_control = control%NLS_subproblem_control_type
         data%n_or_gn = .TRUE.
         data%branch_newton = 10 ; GO TO 800
       ELSE
         data%n_or_gn = .FALSE.
       END IF
       data%branch = 10
     ELSE IF ( inform%status == 11 ) THEN
       IF (  control%model == first_order_model .OR.                           &
             control%model == diagonal_hessian_model .OR.                      &
             control%model == gauss_newton_model .OR.                          &
             control%model == newton_model .OR.                                &
             control%model == gauss_to_newton_model ) THEN
         data%subproblem_control = control%NLS_subproblem_control_type
         data%n_or_gn = .TRUE.
         data%branch_newton = 20 ; GO TO 800
       ELSE
         data%n_or_gn = .FALSE.
       END IF
       data%branch = 20
     ELSE
       data%n_or_gn = .FALSE.
     END IF
!    WRITE( 6, * ) ' branch ', data%branch
     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! re-entry without initialization
       GO TO 20
     CASE ( 30 )  ! initial residual evaluation
       GO TO 30
     CASE ( 110 ) ! initial Jacobian evaluation or Jacobian transpose vect prod
       GO TO 110
     CASE ( 120 ) ! Hessian evaluation
       GO TO 120
     CASE ( 220 ) ! Jacobian vector product
       GO TO 220
     CASE ( 230 ) ! Hessians-vector product
       GO TO 230
     CASE ( 280 ) ! Jacobian vector product
       GO TO 280
     CASE ( 290 ) ! Hessians-vector product
       GO TO 290
     CASE ( 320 ) ! residual evaluation
       GO TO 320
     CASE ( 820 ) ! Newton/Gauss-Newton variants
       GO TO 820
     END SELECT

!  ============================================================
!  0. Initialization - a tensor Gauss-Newton model will be used
!  ============================================================

  10 CONTINUE

!  check that the Hessian is specified in a permitted format

!    data%hessian_available = control%hessian_available >= 2 .OR.              &
!      PRESENT( eval_H )
     data%hessian_available = control%hessian_available >= 2
     IF ( data%hessian_available ) THEN
       SELECT CASE (  SMT_get( nlp%H%type ) )
       CASE ( 'DIAGONAL', 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE',             &
              'IDENTITY', 'SCALE_IDENTITY', 'NONE', 'ZERO' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input H%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%H%type )
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END SELECT

!  record the problem dimensions

       nlp%H%n = nlp%n ; nlp%H%m = nlp%n
       SELECT CASE ( SMT_get( nlp%H%type ) )
       CASE( 'DENSE' )
         IF ( MOD(  nlp%n, 2 ) == 0 ) THEN
           nlp%H%ne = ( nlp%n / 2 ) * ( nlp%n + 1 )
         ELSE
           nlp%H%ne = nlp%n * ( ( nlp%n + 1 ) / 2 )
         END IF
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%H%ne = nlp%H%ptr( nlp%n + 1 ) - 1
       CASE ( 'DIAGONAL' )
         nlp%H%ne = nlp%n
       CASE ( 'NONE' )
         nlp%H%ne = 0
       CASE ( 'COORDINATE' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input H%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%H%type )
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END SELECT
     END IF

!  record controls and ensure that data is consistent

     data%control = control
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     data%control%initial_inner_weight                                         &
       = MAX( data%control%initial_inner_weight, zero )
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%successful = .TRUE.
     data%negcur = ' '
     data%ratio = - one
     data%total_facts = 0
     data%nskip_prec = nskip_prec_max
     data%re_entry = .FALSE.

     inform%iter = 0 ; inform%cg_iter = 0
     inform%c_eval = 0 ; inform%j_eval = 0 ; inform%h_eval = 0
     inform%factorization_max = 0 ; inform%factorization_status = 0
     inform%max_entries_factors = 0 ; inform%factorization_average = zero
     inform%factorization_integer = - 1 ; inform%factorization_integer = - 1

!  decide how much reverse communication is required

     data%reverse_c = .NOT. PRESENT( eval_C )

!  check to see if the Jacobian is available explicitly or only via its
!  action on a vector, and whether reverse communication will be required

     data%jacobian_available = data%control%jacobian_available >= 2 .OR.       &
       PRESENT( eval_J )
     IF ( data%jacobian_available ) THEN
       nlp%J%n = nlp%n ; nlp%J%m = nlp%m
       data%reverse_j = .NOT. PRESENT( eval_J )

!  if the Jacobian is not available explicitly, revert to a Gauss-Newton model

     ELSE
       data%subproblem_control = control%NLS_subproblem_control_type
       IF ( data%hessian_available ) THEN
         data%subproblem_control%model = gauss_to_newton_model
       ELSE
         data%subproblem_control%model = gauss_newton_model
       END IF
       data%branch_newton = 10 ; GO TO 800
     END IF
     data%reverse_jprod = .NOT. PRESENT( eval_JPROD )

!  check to see if the Hessian is available explicitly, available via its
!  action on a vector, or is unavailable, and whether reverse communication
!  will be required

     IF ( data%hessian_available ) THEN
       data%reverse_h = .NOT. PRESENT( eval_H )
     ELSE IF ( data%control%hessian_available == 1 ) THEN
       data%reverse_h = .FALSE.
       data%control%subproblem_direct = .FALSE.
     ELSE
       data%control%model = tensor_gauss_newton_model
     END IF
     data%reverse_hprod = .NOT. PRESENT( eval_HPROD )
     data%reverse_hprods = .NOT. PRESENT( eval_HPRODS )

!  initialize the model to Gauss-Newton if the Gauss-Newton to Newton
!  strategy has been specified

     data%gauss_to_newton_model =                                              &
       data%control%model == tensor_gauss_to_newton_model
     data%map_h_to_jtj = data%hessian_available .AND.                          &
                         ( data%gauss_to_newton_model .OR.                     &
                           data%control%model == tensor_newton_model )
     data%model_used = data%control%model
     IF ( data%gauss_to_newton_model )                                         &
       data%control%model = tensor_gauss_newton_model
     data%hessian_computed = data%hessian_available .AND.                      &
       data%control%model == tensor_newton_model

!  decide whether to form the regularization scaling matrix and make
!  model-specific choices

     IF ( data%control%model == tensor_newton_model ) THEN
       data%form_regularization                                                &
         = data%jacobian_available .AND. data%hessian_available
     ELSE IF ( data%control%model == tensor_gauss_newton_model ) THEN
       data%form_regularization = data%jacobian_available
     ELSE
       IF ( data%control%norm >= 0 )                                           &
         data%control%norm = euclidean_regularization
       data%form_regularization = .FALSE.
     END IF
     data%inner_weight = data%control%initial_inner_weight
     data%reverse_scale = .NOT. PRESENT( eval_SCALE )

!  set the power for the regularization

     IF ( control%power >= two ) THEN
       data%power = control%power
     ELSE
       data%power = two
     END IF

!  make sure that P%type is set if a tensor Newton model is required

     IF ( data%control%model == tensor_newton_model .OR.                       &
          data%control%model == tensor_gauss_newton_model .OR.                 &
          data%control%model == tensor_gauss_to_newton_model ) THEN
       IF ( .NOT. ALLOCATED( nlp%P%type ) ) THEN
         CALL SMT_put( nlp%P%type, 'SPARSE_BY_COLUMNS', inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF
       ELSE
         SELECT CASE ( SMT_get( nlp%P%type ) )
         CASE ( 'SPARSE_BY_COLUMNS', 'DENSE_BY_COLUMNS' )
         CASE DEFAULT
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: input P%type ', A, ' not permitted' )" )           &
               prefix, SMT_get( nlp%P%type )
           inform%status = GALAHAD_error_restrictions
           GO TO 990
         END SELECT
       END IF

!  compute the length of nlp%P%val

       nlp%P%m = nlp%n ; nlp%P%n = nlp%m
       SELECT CASE ( SMT_get( nlp%P%type ) )
       CASE ( 'DENSE_BY_COLUMNS' )
         nlp%P%ne = nlp%m * nlp%n
       CASE ( 'COORDINATE' ) ! will already be set
       CASE DEFAULT
         nlp%P%ne = nlp%P%ptr( nlp%m + 1 ) - 1
       END SELECT
     END IF

!  set specific controls for the sub-problem solvers

     data%regularization_type = data%control%norm
     IF ( data%regularization_type < user_regularization .OR.                  &
          data%regularization_type == lmbfgs_regularization .OR.               &
          data%regularization_type == automatic_regularization .OR.            &
          data%regularization_type > expanding_band_regularization )           &
        data%regularization_type = diagonal_jtj_regularization
     data%subproblem_data%regularization_type = data%regularization_type
!write(6,*) ' * subproblem scaling type ', &
! data%subproblem_data%regularization_type
     data%control%GLRT_control%unitm                                           &
       = data%regularization_type == euclidean_regularization
     SELECT CASE ( data%regularization_type )
     CASE ( diagonal_hessian_regularization )
       data%control%PSLS_control%preconditioner = 1
     CASE ( band_regularization )
       data%control%PSLS_control%preconditioner = 2
     CASE ( reordered_band_regularization )
       data%control%PSLS_control%preconditioner = 3
     CASE ( schnabel_eskow_regularization )
       data%control%PSLS_control%preconditioner = 4
     CASE ( gmps_regularization )
       data%control%PSLS_control%preconditioner = 5
     CASE ( lin_more_regularization )
       data%control%PSLS_control%preconditioner = 6
     CASE ( mi28_regularization )
       data%control%PSLS_control%preconditioner = 7
     CASE ( munksgaard_regularization )
       data%control%PSLS_control%preconditioner = 8
     CASE ( expanding_band_regularization )
       data%control%PSLS_control%preconditioner = 9
     CASE DEFAULT
       data%control%PSLS_control%preconditioner = - 1
     END SELECT
     data%control%PSLS_control%new_structure = .TRUE.
     data%control%RQS_control%initial_multiplier = zero

!  insist on iterative subproblem solution if user scaling is provided

     IF ( data%regularization_type == user_regularization )                    &
       data%control%subproblem_control%subproblem_direct = .FALSE.

!  check that the Jacobian is specified in a permitted format

     IF ( data%jacobian_available ) THEN
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input J%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%J%type )
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END SELECT
     END IF

!  create a file which the user may subsequently remove to cause
!  immediate termination of a run

     IF ( control%alive_unit > 0 ) THEN
      INQUIRE( FILE = control%alive_file, EXIST = alive )
      IF ( .NOT. alive ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'NEW' )
         REWIND control%alive_unit
         WRITE( control%alive_unit, "( ' GALAHAD rampages onwards ' )" )
         CLOSE( control%alive_unit )
       END IF
     END IF

!  allocate basic space to solve the problem

     array_name = 'nls: nlp%G'
     CALL SPACE_resize_array( nlp%n, nlp%G, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%C_current'
     CALL SPACE_resize_array( nlp%m, data%C_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%G_current'
     CALL SPACE_resize_array( nlp%n, data%G_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%S'
     CALL SPACE_resize_array( nlp%n, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%U'
     CALL SPACE_resize_array( MAX( nlp%n, nlp%m ), data%U, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%V'
     CALL SPACE_resize_array( MAX( nlp%n, nlp%m ), data%V, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%W'
     CALL SPACE_resize_array( nlp%n, data%W, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( .NOT. data%monotone ) THEN
       array_name = 'nls: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     array_name = 'nls: data%Y'
     CALL SPACE_resize_array( nlp%m, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  compute the number of nonzeros in J

     IF ( data%regularization_type > diagonal_jtj_regularization .OR.          &
          data%control%subproblem_control%subproblem_direct ) THEN
       nlp%J%n = nlp%n ; nlp%J%m = nlp%m
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         nlp%J%ne = nlp%J%m * nlp%J%n
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%J%ne = nlp%J%ptr( nlp%m + 1 ) - 1
       END SELECT

!  an assembled Hessian approximation is required to compute the scaling matrix,
!  so provide J(transpose) = JT

       data%JT%n = nlp%m ; data%JT%m = nlp%n ; data%JT%ne = nlp%J%ne
       CALL SMT_put( data%JT%type, 'COORDINATE', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: data%JT%row'
       CALL SPACE_resize_array( data%JT%ne, data%JT%row, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%col'
       CALL SPACE_resize_array( data%JT%ne, data%JT%col, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%val'
       CALL SPACE_resize_array( data%JT%ne, data%JT%val, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  assign the row and column indices of JT

       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, nlp%m
           DO j = 1, nlp%n
             l = l + 1
             data%JT%row( l ) = j ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
             data%JT%row( l ) = nlp%J%col( l ) ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           data%JT%row( l ) = nlp%J%col( l )
           data%JT%col( l ) = nlp%J%row( l )
         END DO
       END SELECT
     END IF

!    IF ( data%regularization_type > diagonal_jtj_regularization ) THEN
     IF ( data%regularization_type > diagonal_jtj_regularization .OR.          &
          data%control%subproblem_control%subproblem_direct ) THEN
!         data%control%subproblem_direct ) THEN

!  record the sparsity pattern of J^T J in data%H

       data%control%BSC_control%new_a = 3
       data%control%BSC_control%extra_space_s = 0
       data%control%BSC_control%s_also_by_column = data%map_h_to_jtj
       CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,            &
                      data%control%BSC_control, inform%BSC_inform )
       data%control%BSC_control%new_a = 1

!   if required, find a mapping for the entries of H(x,c) into the existing
!   structure in data%H for J^T J; the sparsity pattern of H(x,c) lies
!   within that of J^T J

       IF ( data%map_h_to_jtj ) THEN
         CALL NLS_set_map( data%H, nlp%H, data%IW, data%PTR, data%ROW,         &
                           data%ORDER, .TRUE.,                                 &
                           data%control%deallocate_error_fatal,                &
                           data%control%space_critical,  data%control%error,   &
                           data%H_map, inform%status, inform%alloc_status,     &
                           inform%bad_alloc )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
     END IF

!  initialize additional space for the tensor module mimimization, in which
!  the residuals c(x+s) for given x are approximated by r(s) = c(x) + J(x) s
!  + 1/2 ( s^T H_i s )_i=1^m, where (w_i)_i=1^m denotes the vector whose
!  ith component is w_i. The value of s is sought to minimize phi(s) =
!  1/2||r(s)||^2_2 using a adaptive regularized Newton method

     data%tensor_model%n = nlp%n ; data%tensor_model%m = nlp%m

!  s is stored in tm%X

     array_name = 'nls: data%tensor_model%X'
     CALL SPACE_resize_array( data%tensor_model%n, data%tensor_model%X,        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )

!  r(s) is stored in tm%C

     array_name = 'nls: data%tensor_model%C'
     CALL SPACE_resize_array( nlp%m, data%tensor_model%C,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )

!  J(x)s is stored in JS

     array_name = 'nls: data%JS'
     CALL SPACE_resize_array( nlp%m, data%JS, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )

!  1/2 ( s^T H_i s )_i=1^m is stored in HSHS

     array_name = 'nls: data%HSHS'
     CALL SPACE_resize_array( nlp%m, data%HSHS, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )

!  store the matrix whose rows are (g_i + H_i s)^T, i = 1,...,m
!  in tensor_model%J

     IF ( data%control%subproblem_control%subproblem_direct ) THEN
       data%tensor_model%J%m = nlp%m ; data%tensor_model%J%n = nlp%n
       data%tensor_model%J%ne = data%JT%ne
       CALL SMT_put( data%tensor_model%J%type, 'COORDINATE',                   &
                     inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: data%tensor_model%J%row'
       CALL SPACE_resize_array( data%tensor_model%J%ne,                        &
              data%tensor_model%J%row, inform%status,                          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%tensor_model%J%col'
       CALL SPACE_resize_array( data%tensor_model%J%ne,                        &
              data%tensor_model%J%col, inform%status,                          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%tensor_model%J%val'
       CALL SPACE_resize_array( data%tensor_model%J%ne,                        &
              data%tensor_model%J%val, inform%status,                          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       data%tensor_model%J%col( : data%tensor_model%J%ne )                     &
         = data%JT%row( : data%tensor_model%J%ne )
       data%tensor_model%J%row( : data%tensor_model%J%ne )                     &
         = data%JT%col( : data%tensor_model%J%ne )
     END IF

!  if required, set up space for the Hessian of the tensor model

!  the weighted Hessian of the tensor model is required; set up space and
!  record its row and column indices

     IF ( data%map_h_to_jtj ) THEN
       CALL SMT_put( data%tensor_model%H%type,                                 &
                     TRIM( SMT_get( nlp%H%type ) ), inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       data%tensor_model%H%n = nlp%H%n
       data%tensor_model%H%m = nlp%H%m
       data%tensor_model%H%ne = nlp%H%ne

       array_name = 'nls: data%tensor_model%H%val'
       CALL SPACE_resize_array( nlp%H%ne, data%tensor_model%H%val,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       SELECT CASE ( SMT_get( nlp%H%type ) )
       CASE ( 'SPARSE_BY_ROWS' )
         array_name = 'nls: data%tensor_model%H%ptr'
         CALL SPACE_resize_array( nlp%H%n + 1, data%tensor_model%H%ptr,        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'nls: data%tensor_model%H%col'
         CALL SPACE_resize_array( nlp%H%ne, data%tensor_model%H%col,           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         data%tensor_model%H%ptr( : nlp%H%n + 1 )                              &
           = nlp%H%ptr( : nlp%H%n + 1 )
         data%tensor_model%H%col( : nlp%H%ne ) = nlp%H%col( : nlp%H%ne )
       CASE ( 'COORDINATE' )
         array_name = 'nls: data%tensor_model%H%row'
         CALL SPACE_resize_array( nlp%H%ne, data%tensor_model%H%row,           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'nls: data%tensor_model%H%col'
         CALL SPACE_resize_array( nlp%H%ne, data%tensor_model%H%col,           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         data%tensor_model%H%row( : nlp%H%ne ) = nlp%H%row( : nlp%H%ne )
         data%tensor_model%H%col( : nlp%H%ne ) = nlp%H%col( : nlp%H%ne )
       END SELECT

!  no tensor-model Hessian is required

     ELSE
       CALL SMT_put( data%tensor_model%H%type, 'ZERO', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF
     END IF

!  given the column-wise storage scheme for the matrix HS, whose columns are
!  the vectors H_i s, i = 1,...,n, create a mapping, HS_map, of the entries
!  of HS into J and JT. Firstly, order JT by columns

     IF ( data%control%subproblem_control%subproblem_direct ) THEN
       CALL NLS_set_map( nlp%P, data%JT, data%IW, data%PTR, data%ROW,          &
                         data%ORDER, .FALSE.,                                  &
                         data%control%deallocate_error_fatal,                  &
                         data%control%space_critical,  data%control%error,     &
                         data%Hs_map, inform%status, inform%alloc_status,      &
                         inform%bad_alloc )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  provide space for the regularization

!write(6,*) 'scaling type ', data%regularization_type
     IF ( data%regularization_type == euclidean_regularization ) THEN
       data%regularization%matrix%m = nlp%n
       data%regularization%matrix%n = nlp%n
       data%regularization%matrix%ne = nlp%n
       CALL SMT_put( data%regularization%matrix%type, 'IDENTITY',              &
                     inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF
     ELSE IF ( data%regularization_type == diagonal_jtj_regularization ) THEN
       array_name = 'nls: data%regularization%matrix%val'
       CALL SPACE_resize_array( nlp%n, data%regularization%matrix%val,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

     ELSE IF ( data%regularization_type > diagonal_jtj_regularization ) THEN
       array_name = 'nls: data%regularization%matrix%ptr'
       CALL SPACE_resize_array( nlp%n + 1, data%regularization%matrix%ptr,     &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: IW'
       CALL SPACE_resize_array( nlp%n + 1, data%IW, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  ===============================
!  re-entry without initialization
!  ===============================

  20 CONTINUE
!    CALL CLOCK_time( data%clock_now )
!    write(6,*) ' 20 elapsed', data%clock_now - data%clock_start

!  control the output printing

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

     IF ( data%control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = data%control%print_gap
     END IF

     data%out = data%control%out
     data%print_level_glrt = data%control%GLRT_control%print_level
     data%print_level_rqs = data%control%RQS_control%print_level
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
     END IF

!  set the initial weight

     IF ( data%control%weight_update_strategy == weight_update_zero_reset ) THEN
       inform%weight = weight_zero
     ELSE
       inform%weight = data%control%initial_weight
     END IF
!    inform%weight = data%control%initial_weight
     data%step_accepted = .FALSE.
     data%poor_model = .FALSE.
     data%s_norm_successful = one
     data%minimum_weight = data%control%minimum_weight

! evaluate the residual function c(x) at the initial point

     IF ( data%reverse_c ) THEN
       data%branch = 30 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_C( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%C( : nlp%m ) )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_C'
         inform%status = GALAHAD_error_evaluation ; GO TO 900
       END IF
     END IF

!  return from reverse communication with the residual function value c(x)

  30 CONTINUE
!    CALL CLOCK_time( data%clock_now )
!    write(6,*) ' 30 elapsed', data%clock_now - data%clock_start
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 30' )" ) prefix
     inform%c_eval = inform%c_eval + 1
     IF ( data%w_eq_identity ) THEN
       data%Y( : nlp%m ) = nlp%C( : nlp%m )
       inform%norm_c = TWO_NORM( nlp%C( : nlp%m ) )
       inform%obj = half * inform%norm_c ** 2
     ELSE
       data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
       val = DOT_PRODUCT( data%Y( : nlp%m ), nlp%C( : nlp%m ) )
       inform%norm_c = SQRT( val )
       inform%obj = half * val
     END IF

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj
!    write(6,*) ' objective is NaN? ', data%f_is_nan

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%bad_eval = 'NaN'
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  compute the residual stopping tolerance

     data%stop_c = MAX( MAX( data%control%stop_c_absolute, zero ),             &
       MAX( data%control%stop_c_relative, zero ) * inform%norm_c, epsmch )

!  stop in the unlikely event that the initial residual is already small

     IF ( inform%norm_c <= data%stop_c ) THEN
       inform%status = GALAHAD_ok ; GO TO 910
     END IF

!  initialize the history of objective values

     data%f_ref = inform%obj
     IF ( .NOT. data%monotone ) THEN
        data%F_hist = data%f_ref ; data%D_hist = zero ; data%max_hist = 1
     END IF

!  ============================================================================
!  Start of main iteration
!  ============================================================================

 100 CONTINUE
!      CALL CLOCK_time( data%clock_now )
!      write(6,*) ' 100 elapsed', data%clock_now - data%clock_start
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 100' )" ) prefix

!  evaluate the Jacobian J(x) of c(x)

       IF ( .NOT. data%poor_model ) THEN
         IF ( data%jacobian_available ) THEN
           IF ( data%reverse_j ) THEN
             data%branch = 110 ; inform%status = 3 ; RETURN
           ELSE
             CALL eval_J( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%J%val )
           END IF

!  otherwise evaluate the product g = J^T(x) W c(x)

         ELSE
           data%transpose = .TRUE.
           IF ( data%reverse_jprod ) THEN
             data%U( : nlp%n ) = zero
             IF ( data%w_eq_identity ) THEN
               data%V( : nlp%m ) = nlp%C( : nlp%m )
             ELSE
               data%V( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
             END IF
             data%branch = 110 ; inform%status = 5 ; RETURN
           ELSE
             nlp%G( : nlp%n ) = zero
             IF ( data%w_eq_identity ) THEN
               CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ), userdata,  &
                                data%transpose, nlp%G( : nlp%n ),              &
                                nlp%C( : nlp%m ), .FALSE. )
             ELSE
               CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ), userdata,  &
                                data%transpose, nlp%G( : nlp%n ),              &
                                W( : nlp%m ) * nlp%C( : nlp%m ), .FALSE. )
             END IF
           END IF
         END IF
       END IF

!  return from reverse communication with the Jacobian-residual product

 110   CONTINUE
!      CALL CLOCK_time( data%clock_now )
!      write(6,*) ' 110 elapsed', data%clock_now - data%clock_start
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 110' )" ) prefix

       IF ( .NOT. data%poor_model ) THEN
         inform%j_eval = inform%j_eval + 1

!        write(6,*) nlp%J%m, nlp%J%n, nlp%J%ne
!        write(6,*) ( nlp%J%row(i), nlp%J%col(i), nlp%J%val(i), i = 1, nlp%J%ne)

!  compute the product g = J^T(x) W c(x) from J(x) if necessary

         IF ( data%jacobian_available ) THEN
           IF ( data%w_eq_identity ) THEN
             CALL mop_Ax( one, nlp%J, nlp%C( : nlp%m ),                        &
                        zero, nlp%G( : nlp%n ),                                &
                        out = data%out, error = data%control%error,            &
                        print_level = 0, transpose = .TRUE. )
!                       print_level = 1, transpose = .TRUE. )
           ELSE
             CALL mop_Ax( one, nlp%J, W( : nlp%m ) * nlp%C( : nlp%m ),         &
                        zero, nlp%G( : nlp%n ),                                &
                        out = data%out, error = data%control%error,            &
                        print_level = 0, transpose = .TRUE. )
!                       print_level = 1, transpose = .TRUE. )
           END IF
         ELSE
           IF ( data%reverse_jprod ) nlp%G( : nlp%n ) = data%U( : nlp%n )
         END IF

!  compute the gradient of ||c(x)||

         data%g_norm = TWO_NORM( nlp%G( : nlp%n ) )
         IF ( inform%norm_c > zero ) THEN
           inform%norm_g = data%g_norm / inform%norm_c
         ELSE
           inform%norm_g = zero
         END IF
         data%new_point = .TRUE.

!  deal with NaN gradient values
!  -----------------------------

         data%g_is_nan = inform%norm_g /= inform%norm_g
         IF ( data%g_is_nan ) THEN
           IF ( inform%iter > 0 ) THEN
             data%poor_model = .FALSE.
             data%accept = 'r'
             nlp%X( : nlp%n ) = data%X_current( : nlp%n )
             nlp%C( : nlp%m ) = data%C_current( : nlp%m )

!  control printing for the NaN case

             IF ( inform%iter >= data%start_print .AND.                        &
                  inform%iter < data%stop_print .AND.                          &
                  MOD( inform%iter + 1 - data%start_print, data%print_gap )    &
                    == 0 ) THEN
               data%printi = data%set_printi ; data%printt = data%set_printt
               data%printm = data%set_printm ; data%printw = data%set_printw
               data%printd = data%set_printd
               data%print_level = data%control%print_level
               data%control%GLRT_control%print_level = data%print_level_glrt
               data%control%RQS_control%print_level = data%print_level_rqs
             ELSE
               data%printi = .FALSE. ; data%printt = .FALSE.
               data%printm = .FALSE. ; data%printw = .FALSE.
               data%printd = .FALSE.
               data%print_level = 0
               data%control%GLRT_control%print_level = 0
               data%control%RQS_control%print_level = 0
             END IF
             data%print_iteration_header = data%print_level > 1 .OR.           &
               ( data%control%GLRT_control%print_level > 0 .AND. .NOT.         &
                 data%control%subproblem_direct ) .OR.                         &
               ( data%control%RQS_control%print_level > 0 .AND.                &
                 data%control%subproblem_direct )

!  print one-line summary

             IF ( data%printi ) THEN
                IF ( data%print_iteration_header .OR.                          &
                     data%print_1st_header ) THEN
                 WRITE( data%out, 2090 ) prefix
                 IF ( data%subproblem_control%subproblem_direct ) THEN
                   IF ( data%control%print_obj ) THEN
                     WRITE( data%out, 2170 ) prefix
                   ELSE
                     WRITE( data%out, 2160 ) prefix
                   END IF
                 ELSE
                   IF ( data%control%print_obj ) THEN
                     WRITE( data%out, 2180 ) prefix
                   ELSE
                     WRITE( data%out, 2190 ) prefix
                   END IF
                 END IF
               END IF
               data%print_1st_header = .FALSE.
               char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
               IF ( data%subproblem_control%subproblem_direct ) THEN
                 char_facts =                                                  &
                   ADJUSTR( STRING_integer_6( data%total_facts ) )
               ELSE
                 char_facts =                                                  &
                 ADJUSTR( STRING_integer_6( inform%subproblem_inform%cg_iter ) )
               END IF
               WRITE( data%out, "( A, A6, 1X, 3A1, ES11.4, '    NaN    ',      &
              &                    ES9.1,  2ES8.1, 1X, A6, F8.2 )" )           &
                  prefix, char_iter, data%accept, data%negcur, data%hard,      &
                  inform%norm_c, data%ratio, data%old_weight, data%s_norm,     &
                  char_facts, data%clock_now
             END IF
             inform%obj = data%obj_current
             inform%norm_c = data%norm_c_current

!  check to see if we are still "alive"

             IF ( data%control%alive_unit > 0 ) THEN
               INQUIRE( FILE = data%control%alive_file, EXIST = alive )
               IF ( .NOT. alive ) THEN
                 inform%status = GALAHAD_error_alive
                 RETURN
               END IF
             END IF

!  check to see if the iteration limit has been exceeded

             inform%iter = inform%iter + 1
             IF ( inform%iter > data%control%maxit .AND.                       &
                  data%step_accepted ) THEN
               inform%status = GALAHAD_error_max_iterations ; GO TO 900
             END IF

!  increase the regularization weight and try again

             inform%weight = data%control%weight_increase * data%old_weight
             GO TO 100
           ELSE
             IF ( data%printi ) WRITE( data%out,                               &
                "( A, ' initial gradient value is a NaN' )" ) prefix
             inform%bad_eval = 'NaN'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
         END IF

!  reset the initial weight to ||g|| if no sensible value is given

         IF ( inform%iter == 0 ) THEN
           IF ( data%control%initial_weight <= zero )                          &
              inform%weight = one / inform%norm_g

!  compute the gradient stopping tolerance

           data%stop_g = MAX( MAX( data%control%stop_g_absolute, zero ),       &
             MAX( data%control%stop_g_relative, zero ) * inform%norm_g, epsmch )

           IF ( data%printi )                                                  &
             WRITE( data%out, "( A, '  Problem: ', A, ' (n = ', I0, ', m = ',  &
          &   I0, ')', /, A, '  NLS stopping tolerances (c,J''c/c) =',         &
          &   2ES9.2, / )" ) prefix, TRIM( nlp%pname ), nlp%n, nlp%m,          &
             prefix, data%stop_c, data%stop_g
         END IF
       END IF

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%GLRT_control%print_level = data%print_level_glrt
         data%control%RQS_control%print_level = data%print_level_rqs
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%GLRT_control%print_level = 0
         data%control%RQS_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         ( data%control%subproblem_control%GLRT_control%print_level > 0 .AND.  &
           .NOT. data%control%subproblem_control%subproblem_direct ) .OR.      &
         data%control%subproblem_control%RQS_control%print_level > 0 .OR.      &
         ( ( data%control%model == tensor_gauss_newton_model .OR.              &
             data%control%model == tensor_newton_model .OR.                    &
             data%control%model == tensor_gauss_to_newton_model ) .AND.        &
           data%control%subproblem_control%print_level > 0 )

!  print one-line summary

       IF ( data%printi ) THEN
          IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2090 ) prefix
           IF ( data%subproblem_control%subproblem_direct ) THEN
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2170 ) prefix
             ELSE
               WRITE( data%out, 2160 ) prefix
             END IF
           ELSE
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2180 ) prefix
             ELSE
               WRITE( data%out, 2190 ) prefix
             END IF
           END IF
         END IF

         data%print_1st_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 0 ) THEN
           IF ( data%subproblem_control%subproblem_direct ) THEN
             char_facts =                                                      &
               ADJUSTR( STRING_integer_6( data%total_facts ) )
           ELSE
             char_facts =                                                      &
               ADJUSTR( STRING_integer_6( inform%subproblem_inform%cg_iter ) )
           END IF
           IF ( data%control%print_obj ) THEN
             WRITE( data%out, 2120 ) prefix, char_iter, data%accept,           &
                data%negcur, data%hard, inform%obj,                            &
                inform%norm_g, data%ratio, data%old_weight,                    &
                data%s_norm, char_facts, data%clock_now
           ELSE
             WRITE( data%out, 2120 ) prefix, char_iter, data%accept,           &
                data%negcur, data%hard, inform%norm_c,                         &
                inform%norm_g, data%ratio, data%old_weight,                    &
                data%s_norm, char_facts, data%clock_now
           END IF
         ELSE
           IF ( data%control%print_obj ) THEN
             WRITE( data%out, 2140 ) prefix,                                   &
                char_iter, inform%obj, inform%norm_g
           ELSE
             WRITE( data%out, 2140 ) prefix,                                   &
                char_iter, inform%norm_c, inform%norm_g
           END IF
         END IF
       END IF

!  ============================================================================
!  1. Test for convergence
!  ============================================================================

!  stop if the gradient is small enough

       IF ( inform%norm_c <= data%stop_c .OR.                                  &
            inform%norm_g <= data%stop_g ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  stop if the gradient is swamped by the Hessian

!       IF ( data%control%hessian_available .AND. inform%iter > 0 ) THEN
!         IF ( inform%norm_g <= MIN( one,                                      &
!               MAXVAL( ABS( data%H%val( : data%H%ne ) ) ) * epsmch ) ) THEN
!         write(6,*) ' stopping as g is too ill-conditioned to make ',         &
!        &   'further progress!'
!         write(6,*) inform%norm_g,                                            &
!           MAXVAL( ABS( nlp%H%val( : nlp%H%ne ) ) ) * epsmch
!           inform%status = GALAHAD_error_ill_conditioned ; GO TO 900
!         END IF
!       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive
           RETURN
         END IF
       END IF

!  check to see if the iteration limit has been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit .AND. data%step_accepted ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF
!      write(6,*) inform%norm_c, data%stop_c, inform%norm_g, data%stop_g

!  check to see if the Gauss-Newton model should be exchanged for a Newton one

       IF ( data%gauss_to_newton_model ) THEN
         IF ( inform%norm_g < data%control%switch_to_newton ) THEN
           IF ( data%control%model == tensor_gauss_newton_model ) THEN
!            IF ( control%power < two ) data%power = three
             data%control%model = tensor_newton_model
             data%subproblem_control%model = newton_model
             data%form_regularization                                          &
               = data%jacobian_available .AND. data%hessian_available
             data%re_entry = .FALSE.
             data%print_1st_header = .TRUE.
             IF ( data%printi ) WRITE( data%out,                               &
               "( /, A, '  ... switching to Newton model', / )" ) prefix
           END IF
         END IF
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%c_eval, prefix, inform%j_eval,&
           prefix, inform%h_eval, prefix, inform%iter, prefix, inform%cg_iter, &
           prefix, inform%obj, prefix, inform%norm_g
         WRITE ( data%out, 2010 ) prefix
!        l = nlp%n
         l = 2
         DO j = 1, 2
            IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, nlp%n )
            ELSE
               IF ( ic < nlp%n - l ) WRITE( data%out, 2050 ) prefix
               ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
            END IF
            IF ( ALLOCATED( nlp%vnames ) ) THEN
              DO i = ir, ic
                 WRITE( data%out, 2020 ) prefix, nlp%vnames( i ), nlp%X( i ),  &
                  nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2030 ) prefix, i, nlp%X( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

!  recompute the scaled Hessian if it has changed

       data%perturb = ' '
       IF ( data%new_point ) THEN
         data%nskip_prec = data%nskip_prec + 1
         data%got_h = .FALSE.
         data%hessian_computed = data%hessian_available .AND.                  &
           data%control%model == tensor_newton_model .AND.                     &
           data%nskip_prec > nskip_prec_max

!  form the scaled Hessian or a scaling matrix based on the scaled Hessian

         IF ( data%hessian_computed ) THEN
           IF ( data%reverse_h ) THEN
             data%branch = 120 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          data%Y( : nlp%m ), userdata,                         &
                          nlp%H%val( : nlp%H%ne ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_H'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         END IF
       END IF

!  return from reverse communication with the scaled Hessian

 120   CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 120' )" ) prefix

!  the Hessian has changed

       IF ( data%new_point ) THEN
         IF ( data%hessian_computed ) THEN
           inform%h_eval = inform%h_eval + 1  ; data%got_h = .TRUE.

!  debug printing for H

           IF ( data%printd ) THEN
             WRITE( data%out, "( A, ' Scaled Hessian' )" ) prefix
             DO l = 1, nlp%H%ne
               WRITE( data%out, "( A, 2I7, ES24.16 )" ) prefix,                &
                 nlp%H%row( l ), nlp%H%col( l ), nlp%H%val( l )
             END DO
           END IF
         END IF

!  if required, form the Hessian to provide a scaling matrix

         IF ( data%regularization_type > diagonal_jtj_regularization .OR.      &
              data%control%subproblem_control%subproblem_direct ) THEN

!  form the transpose of the Jacobian

           data%JT%val( : data%JT%ne ) = nlp%J%val( : data%JT%ne )

!  insert the values of J^T W J into H

           IF ( data%w_eq_identity ) THEN
             CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,      &
                            data%control%BSC_control, inform%BSC_inform )
           ELSE
             CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,      &
                            data%control%BSC_control, inform%BSC_inform,       &
                            D = W( : nlp%m ) )
           END IF

!  append the values of H(x,Wc) if they are required

           IF ( data%hessian_computed ) THEN
             DO l = 1, nlp%H%ne
               j =  data%H_map( l )
               data%H%val( j ) = data%H%val( j ) + nlp%H%val( l )
             END DO
           END IF
         END IF

!        write(6,"( 5ES12.4 )" ) ( nlp%G( l ), l = 1, nlp%n )
!        write(6,"( ( 2I8, ES12.4 ) )" ) ( data%H%row( l ), data%H%col( l ),   &
!                                          data%H%val( l ), l = 1, data%h_ne )

!  recompute the scaling matrix

!  build the scaling matrix from H

!write(6,*) 'scaling type ', data%regularization_type
         IF ( data%regularization_type > diagonal_jtj_regularization .AND.     &
              data%form_regularization ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
                 "( A, ' Computing scaling matrix' )" ) prefix
           CALL PSLS_build( data%H, data%regularization%matrix,                &
             data%PSLS_data, data%control%PSLS_control, inform%PSLS_inform )

!  check for error returns

           IF ( inform%PSLS_inform%status /= 0 ) THEN
             inform%status = inform%PSLS_inform%status ; GO TO 900
           END IF
           IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'

!  build the scaling matrix as the diagonal matrix whose entries are
!  the squares of the W-norms of the columns of J

         ELSE IF ( data%regularization_type ==                                 &
            diagonal_jtj_regularization .AND. data%form_regularization ) THEN
           data%regularization%matrix%n = nlp%n
           data%regularization%matrix%m = nlp%n
           data%regularization%matrix%ne = nlp%n
           CALL SMT_put( data%regularization%matrix%type, 'DIAGONAL',          &
                         inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF
           CALL mop_column_2_norms( nlp%J,                                     &
                  data%regularization%matrix%val( : nlp%n ), W = W )
           data%regularization%matrix%val( : nlp%n )                           &
             = data%regularization%matrix%val( : nlp%n ) ** 2
         END IF
         data%control%PSLS_control%new_structure = .FALSE.
       END IF

   190 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 190' )" ) prefix

!  ============================================================================
!  2. Calculate the search direction, s
!  ============================================================================

!  solve the tensor model problem - the residuals c(x+s) are approximated by
!  r(s) = c(x) + J(x) s + 1/2 ( s^T H_i s )_i=1^m, where (w_i)_i=1^m denotes
!  the vector whose ith component is w_i. The value of s sought, s_k, is a
!  local minimizer of phi(s) = 1/2||r(s)||^2_2 + 1/p weight ||s||_S^p

       IF ( .NOT. data%successful ) THEN
         IF ( data%inner_weight == zero ) THEN
           data%inner_weight = 0.0001_wp
         ELSE
           data%inner_weight = data%inner_weight * 10.0_wp
         END IF
       END IF
       data%tensor_model%pname = 'tensor    '

       IF ( .TRUE. ) THEN
         data%tensor_model%X( : nlp%n ) = zero
         GO TO 260
       END IF

!  ------------------------------- ignore this part --------------------------

!  first, find a starting guess by minimizing phi(- alpha g)

       data%tensor_model%X( : nlp%n ) = - nlp%G( : nlp%n ) / data%g_norm

!  compute J(x) g

       IF ( data%jacobian_available ) THEN
         CALL mop_Ax( one, nlp%J, data%tensor_model%X( : nlp%n ), zero,        &
                      data%JS( : nlp%m ), out = data%out,                      &
                      error = data%control%error, print_level = 0,             &
                      transpose = .FALSE. )

!  if the Jacobian is unavailable, obtain a matrix-free product

       ELSE
         data%transpose = .FALSE.
         data%U( : nlp%m ) = zero
         data%V( : nlp%n ) = data%tensor_model%X( : nlp%n )
         IF ( data%reverse_jprod ) THEN
           data%branch = 220 ; inform%status = 5 ; RETURN
         ELSE
           data%JS( : nlp%m ) = zero
           CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, data%transpose, data%JS( : nlp%m ),      &
                            data%tensor_model%X( : nlp%n ), got_j = data%got_j )
         END IF
       END IF

!  return from reverse communication with the Jacobian-vector product

   220 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 220' )" ) prefix
       IF ( .NOT. data%jacobian_available .AND. data%reverse_jprod )           &
         data%JS( : nlp%m ) = data%U( : nlp%m )

!  evaluate H_i(x) g

       IF ( data%reverse_hprods ) THEN
         data%V( : nlp%n ) = data%tensor_model%X( : nlp%n )
         data%branch = 230 ; inform%status = 7 ; RETURN
       ELSE
         CALL eval_HPRODS( data%eval_status, nlp%X( : nlp%n ),                 &
                           data%tensor_model%X( : nlp%n ), userdata,           &
                           nlp%P%val, got_h = .FALSE. )
         IF ( data%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_HPRODS'
           inform%status = GALAHAD_error_evaluation ; GO TO 900
         END IF
       END IF

!  return from reverse communication with the Hessians-vector product

   230 CONTINUE

!  compute 1/2 ( g^T H_j g )_j

       SELECT CASE ( SMT_get( nlp%P%type ) )
       CASE ( 'DENSE_BY_COLUMNS' )
         l = 0
         DO j = 1, nlp%m
           sths = zero
           DO i = 1, nlp%n
             l = l + 1
             sths = sths + nlp%P%val( l ) * data%tensor_model%X( i )
           END DO
           data%HSHS( j ) = half * sths
         END DO
       CASE ( 'COORDINATE' )
         data%HSHS( : nlp%m ) = zero
         DO l = 1, nlp%P%ne
           j = nlp%P%col( l )
           data%HSHS( j ) = data%HSHS( j )                                     &
             + nlp%P%val( l ) * data%tensor_model%X( nlp%P%row( l ) )
         END DO
         data%HSHS( : nlp%m ) = half * data%HSHS( : nlp%m )
       CASE DEFAULT
         DO j = 1, nlp%m
           sths = zero
           DO l = nlp%P%ptr( j ), nlp%P%ptr( j + 1 ) - 1
             sths = sths                                                       &
              + nlp%P%val( l ) * data%tensor_model%X( nlp%P%row( l ) )
           END DO
           data%HSHS( j ) = half * sths
         END DO
       END SELECT

!  compute the coefficients of the tensor model along the steepest
!  descent direction, i.e., the quartic function phi(-alpha g) =
!   a4 * alpha ** 4 + a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0

!      data%a0 = half * DOT_PRODUCT( nlp%C(  : nlp%m ), nlp%C(  : nlp%m ) )
       data%a1 = DOT_PRODUCT( nlp%C(  : nlp%m ), data%JS(  : nlp%m ) )
       data%a2 = DOT_PRODUCT( nlp%C(  : nlp%m ), data%HSHS(  : nlp%m ) ) +     &
                 half * DOT_PRODUCT( data%JS(  : nlp%m ), data%JS(  : nlp%m ) )
       data%a3 = DOT_PRODUCT( data%JS(  : nlp%m ), data%HSHS(  : nlp%m ) )
       data%a4 =                                                               &
         half * DOT_PRODUCT( data%HSHS(  : nlp%m ), data%HSHS(  : nlp%m ) )

!  since regularization is required, add 1/p weight ||s||_S^p to ap term

       IF ( data%power == two ) THEN
         data%a2 = data%a2 + half * data%inner_weight                          &
           * SUM( data%tensor_model%X( : nlp%n ) ** 2 )
       ELSE
         data%a4 = data%a4 + quarter * data%inner_weight                       &
           * SUM( data%tensor_model%X( : nlp%n ) ** 4 )
       END IF

!  find the roots of the cubic
!    phi'(-alpha g) = 4 a4 * alpha**3 + 3 a3 * alpha**2 + 2 a2 * alpha + a1 = 0

       CALL ROOTS_cubic( data%a1, two * data%a2, three * data%a3,              &
                         four * data%a4, data%control%ROOTS_control%tol,       &
                         nroots, root1, root2, root3, .FALSE. )
!      write(6,*) data%a1, two * data%a2, three * data%a3,                     &
!        four * data%a4, nroots, root1, root2, root3

!  pick the smallest positive root

       IF ( nroots == 3 .AND. root1 <= zero ) THEN
         data%steplength = root3
       ELSE
         data%steplength = root1
       END IF
!      data%tensor_model%X( : nlp%n )                                          &
!        = data%steplength * data%tensor_model%X( : nlp%n )

!  --------------------------- end of ignored part -----------------------

!  a stabilised (Gauss-)Newton method is applied to phi(s) to find s_k

  260  CONTINUE

!  mock do loop to allow reverse communication

       data%regularization%weight = inform%weight
       data%regularization%power = data%power

       IF ( data%re_entry ) THEN
         inform%subproblem_inform%status = 11
       ELSE
         inform%subproblem_inform%status = 1
         data%subproblem_control = data%control%subproblem_control

         IF ( data%control%model == tensor_gauss_newton_model ) THEN
           data%subproblem_control%model = gauss_newton_model
         ELSE IF ( data%control%model == tensor_newton_model ) THEN
           data%subproblem_control%model = newton_model
         ELSE IF ( data%control%model == tensor_gauss_to_newton_model ) THEN
           data%subproblem_control%model = gauss_to_newton_model
         END IF

         data%re_entry = .TRUE.
!        data%subproblem_control%subproblem_direct = .TRUE.
!        data%subproblem_control%hessian_available = 1
         IF ( .NOT. data%control%subproblem_control%subproblem_direct ) THEN
           data%subproblem_control%jacobian_available = 1
           IF ( data%subproblem_control%norm /= user_regularization )          &
             data%subproblem_control%norm = euclidean_regularization
           IF ( data%subproblem_data%regularization_type /=                    &
                user_regularization )                                          &
             data%subproblem_data%regularization_type = euclidean_regularization
         END IF
         data%subproblem_control%prefix = "'-" // data%control%prefix( 2 : 29 )
       END IF
       inform%subproblem_inform%iter = 0
       inform%subproblem_inform%cg_iter = 0
       inform%subproblem_inform%RQS_inform%factorizations = 0

       data%subproblem_control%stop_g_relative = MIN( half,                    &
         MAX( control%subproblem_control%stop_g_relative, zero ) )
       data%subproblem_control%stop_g_absolute = MIN( half * inform%norm_g,    &
         MAX( control%subproblem_control%stop_g_absolute, zero ) )
       data%control%subproblem_control%norm = data%control%norm

!  main loop to minimize the tensor model; the current estimate of the
!  solution is s = tensor_model%X

!data%subproblem_control%print_level = 1
   270 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 270, status = ',  &
      &  I0  )" ) prefix, inform%subproblem_inform%status

!  minimize the tensor model; the current estimate of the solution s = tm%X

!write(6,*) ' into NLS_subproblem_solve'
         CALL NLS_subproblem_solve( data%tensor_model,                         &
                                    data%subproblem_control,                   &
                                    inform%subproblem_inform,                  &
                                    data%subproblem_data,                      &
                                    data%subproblem_userdata, W = W,           &
                                    stabilisation = data%regularization )
!write(6,*) ' out of NLS_subproblem_solve, status = ', &
! inform%subproblem_inform%status

         SELECT CASE ( inform%subproblem_inform%status )

!  obtain the residual function r(s) = c(x) + J(x) s + 1/2 ( s^T H_i(x) s )_i,
!  i=1^m, or, componentwise, r_i(s) = c_i(s) + g^T_i(x) s + 1/2 s^T H_i(x) s,
!  where g_i(x) is the gradient of c_i(x). Firstly compute J(x) s

         CASE ( 2 )
           IF ( data%jacobian_available ) THEN
             CALL mop_Ax( one, nlp%J, data%tensor_model%X( : nlp%n ), zero,    &
                          data%JS( : nlp%m ), out = data%out,                  &
                          error = data%control%error, print_level = 0,         &
                          transpose = .FALSE. )

!  if the Jacobian is unavailable, obtain a matrix-free product

           ELSE
             data%transpose = .FALSE.
             IF ( data%reverse_jprod ) THEN
               data%U( : nlp%m ) = zero
               data%V( : nlp%n ) = data%tensor_model%X( : nlp%n )
               data%branch = 280 ; inform%status = 5 ; RETURN
             ELSE
               data%JS( : nlp%m ) = zero
               CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),            &
                                userdata, data%transpose, data%JS( : nlp%m ),  &
                                data%tensor_model%X( : nlp%n ),                &
                                got_j = data%got_j )
             END IF
           END IF

!  evaluate the scaled Hessian H(x,y) = sum_i y_i H_i(x)

         CASE ( 4 )
           IF ( data%reverse_h ) THEN
             data%Y( : nlp%m ) = data%subproblem_data%Y( : nlp%m )
             data%branch = 280 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          data%subproblem_data%Y( : nlp%m ), userdata,         &
                          data%tensor_model%H%val( : nlp%H%ne ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_H'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF

!  evaluate the sum u = u + J(s) v or u = u + J^T(s) v, where
!  J(s) = J(x) + P(x,s) and (P^T(x,s))_i = H_i s _i=1,...,m.

         CASE ( 5 )

!  firstly compute u = u + J^T(x) v

           IF ( data%subproblem_data%transpose ) THEN
             IF ( data%jacobian_available ) THEN
               CALL mop_Ax( one, nlp%J, data%subproblem_data%V( : nlp%m ), one,&
                            data%subproblem_data%U( : nlp%n ), out = data%out, &
                            error = data%control%error, print_level = 0,       &
                            transpose = .TRUE. )
             ELSE
               data%transpose = .TRUE.
               IF ( data%reverse_jprod ) THEN
                 data%U( : nlp%n ) = data%subproblem_data%U( : nlp%n )
                 data%V( : nlp%m ) = data%subproblem_data%V( : nlp%m )
                 data%branch = 280 ; inform%status = 5 ; RETURN
               ELSE
                 CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),          &
                                  userdata, data%transpose,                    &
                                  data%subproblem_data%U( : nlp%n ),           &
                                  data%subproblem_data%V( : nlp%m ),           &
                                  got_j = data%got_j )
               END IF
             END IF

!  otherwise compute u = u + J(x) v

           ELSE
             IF ( data%jacobian_available ) THEN
               CALL mop_Ax( one, nlp%J, data%subproblem_data%V( : nlp%n ), one,&
                            data%subproblem_data%U( : nlp%m ), out = data%out, &
                            error = data%control%error, print_level = 0,       &
                            transpose = .FALSE. )
             ELSE
               data%transpose = .FALSE.
               IF ( data%reverse_jprod ) THEN
                 data%U( : nlp%m ) = data%subproblem_data%U( : nlp%m )
                 data%V( : nlp%n ) = data%subproblem_data%V( : nlp%n )
                 data%branch = 280 ; inform%status = 5 ; RETURN
               ELSE
                 CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),          &
                                  userdata, data%transpose,                    &
                                  data%subproblem_data%U( : nlp%m ),           &
                                  data%subproblem_data%V( : nlp%n ),           &
                                  got_j = data%got_j )
               END IF
             END IF
           END IF

!  evaluate the sum u = u + H(x,y) v

         CASE ( 6 )
           CALL mop_Ax( one, data%tensor_model%H,                              &
                        data%subproblem_data%V( : nlp%m ), one,                &
                        data%subproblem_data%U( : nlp%n ), out = data%out,     &
                        error = data%control%error, print_level = 0,           &
                        symmetric = .TRUE. )

!  evaluate the scaled vector u = S(x) v

         CASE ( 8 )
           IF ( data%reverse_scale ) THEN
             data%V( : nlp%n ) = data%subproblem_data%V( : nlp%n )
             data%branch = 280 ; inform%status = 8 ; RETURN
           ELSE
             CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%subproblem_data%U( : nlp%n ),               &
                              data%subproblem_data%V( : nlp%n ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_SCALE'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF

!  compute a "magic" step. Find the stepsize alpha for which phi(alpha s)
!  is smallest

         CASE ( 9 )

!  for given alpha,

!    phi(alpha s) = a0 + a1 alpha + a2 alpha^2 + a3 alpha^3 + a4 alpha^4,

!  where

!    a0 = 1/2 ||c(x)||_2^2,
!    a1 = c^T(x) (J(x) s),
!    a2 = 1/2 s^T (J^T(x) J(x) + H(x,c))s,
!    a3 = sum_i g(x)_i^T s ) ( 1/2 s^T h_i(x) s ), and
!    a4 = 1/2 sum_i (1/2 s^T h_i(x) s)^2.

!  Compute the corefficients a0, ..., a4

           IF ( data%power == two ) THEN
             data%a0 = half * DOT_PRODUCT( nlp%C( : nlp%m ), nlp%C( : nlp%m ) )
             data%a1 = DOT_PRODUCT( nlp%C( : nlp%m ), data%JS( : nlp%m ) )
             data%a2 = DOT_PRODUCT( nlp%C( : nlp%m ), data%HSHS( : nlp%m ) ) + &
                    half * DOT_PRODUCT( data%JS( : nlp%m ), data%JS( : nlp%m ) )
             data%a3 = DOT_PRODUCT( data%JS( : nlp%m ), data%HSHS( : nlp%m ) )
             data%a4 =                                                         &
               half * DOT_PRODUCT( data%HSHS( : nlp%m ), data%HSHS( : nlp%m ) )

!  add the regularization term 1/2 weight ||s||_S^2 to a2

             CALL mop_Ax( one, data%regularization%matrix,                     &
                          data%tensor_model%X( : nlp%n ),                      &
                          zero, data%subproblem_data%SX( : nlp%n ),            &
                          symmetric = .TRUE., out = data%out,                  &
                          error = data%control%error, print_level = 0 )
             reg = half * inform%weight *                        &
                         DOT_PRODUCT( data%tensor_model%X( : nlp%n ),          &
                                      data%subproblem_data%SX( : nlp%n ) )
             data%a2 = data%a2 + reg

!  find the roots of the cubic
!    phi'(alpha s) = a1 + 2 a2 * alpha + 3 a3 * alpha^2 + 4 a4 * alpha^3 = 0

             CALL ROOTS_cubic( data%a1, two * data%a2, three * data%a3,        &
                               four * data%a4, data%control%ROOTS_control%tol, &
                               nroots, root1, root2, root3, .FALSE. )
!      write(6,*) data%a1, two * data%a2, three * data%a3,                     &
!        four * data%a4, nroots, root1, root2, root3

             c0 = data%a0 + data%a1 + data%a2 - reg + data%a3 + data%a4
             c0 = SQRT( two * c0 )
             c1 = data%a0 + root1 * ( data%a1 + root1 *                        &
                 ( data%a2 - reg + root1 * ( data%a3 + data%a4 * root1 ) ) )
             c1 = SQRT( two * c1 )
             IF ( nroots == 3 ) THEN
               c2 = data%a0 + root2 * ( data%a1 + root2 *                      &
                   ( data%a2 - reg + root2 * ( data%a3 + data%a4 * root2 ) ) )
               c2 = SQRT( two * c2 )
               c3 = data%a0 + root3 * ( data%a1 + root3 *                      &
                   ( data%a2 - reg + root3 * ( data%a3 + data%a4 * root3 ) ) )
               c3 = SQRT( two * c3 )
             END IF

             IF ( .FALSE. ) THEN
!            IF ( .TRUE. ) THEN
               write(6,"( ' weight ', ES12.4 )" ) inform%weight
               write(6,"( 's,a', 6ES12.4 )" ) TWO_NORM( nlp%X( : nlp%n ) ),    &
                 data%a0, data%a1, data%a2, data%a3, data%a4
               write(6,"( ' alpha =', ES18.10, ' c,phi''(alpha) =',2ES18.10 )")&
                 one, c0, data%a1 + two * data%a2 + three * data%a3 +          &
                            four * data%a4
               write(6,"( ' alpha =', ES18.10, ' c,phi''(alpha) =',2ES18.10 )")&
                 root1, c1, data%a1 + root1 * ( two * data%a2 + root1 *        &
                              ( three * data%a3 + four * data%a4 * root1 ) )
               IF ( nroots == 3 ) THEN
                 write(6,"( ' alpha =', ES18.10,' c,phi''(alpha) =',2ES18.10)")&
                   root2, c2, data%a1 + root2 * ( two * data%a2 + root2 *      &
                                ( three * data%a3 + four * data%a4 * root2 ) )
                 write(6,"( ' alpha =', ES18.10,' c,phi''(alpha) =',2ES18.10)")&
                   root3, c3, data%a1 + root3 * ( two * data%a2 + root3 *      &
                                ( three * data%a3 + four * data%a4 * root3 ) )
               END IF
             END IF

!  pick the best root, alpha

             IF ( nroots == 3 ) THEN
               IF ( c1 <= c3 ) THEN
                 alpha = root1
               ELSE
                 alpha = root3
               END IF
             ELSE
               alpha = root1
             END IF

!  replace s by alpha s

             data%tensor_model%X( : nlp%n )                                    &
               = alpha * data%tensor_model%X( : nlp%n )

!  replace r(s) by r(alpha s)

             data%JS( : nlp%m ) = alpha * data%JS( : nlp%m )
             data%HSHS( : nlp%m ) = ( alpha ** 2 ) * data%HSHS( : nlp%m )
             data%tensor_model%C( : nlp%m )                                    &
               = nlp%C( : nlp%m ) + data%JS( : nlp%m ) + data%HSHS( : nlp%m )
             nlp%P%val( : nlp%P%ne ) = alpha * nlp%P%val( : nlp%P%ne )
           END IF
         END SELECT

!  return from reverse communication with the Jacobian-vector product

   280   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 280, status = ',&
        &  I0  )" ) prefix, inform%subproblem_inform%status

!  continue with the computation of r(s). Recover reverse-communication data

         SELECT CASE ( inform%subproblem_inform%status )
         CASE ( 2 )
           IF ( .NOT. data%jacobian_available .AND. data%reverse_jprod )       &
             data%JS( : nlp%m ) = data%U( : nlp%m )
         CASE ( 4 )
           IF ( data%reverse_h ) data%tensor_model%H%val( : nlp%H%ne )         &
             = nlp%H%val( : nlp%H%ne )
         CASE ( 5 )
           IF ( data%subproblem_data%transpose ) THEN
             IF ( .NOT. data%jacobian_available .AND. data%reverse_jprod )     &
               data%subproblem_data%U( : nlp%n ) = data%U( : nlp%n )
           ELSE
             IF ( .NOT. data%jacobian_available .AND. data%reverse_jprod )     &
               data%subproblem_data%U( : nlp%m ) = data%U( : nlp%m )
           END IF
         CASE ( 8 )
           IF ( data%reverse_scale )                                           &
             data%subproblem_data%U( : nlp%n ) = data%U( : nlp%n )
         END SELECT

!  evaluate H_i(x) s

         SELECT CASE ( inform%subproblem_inform%status )
         CASE ( 2, 3, 5 )
           IF ( data%reverse_hprods ) THEN
             data%V( : nlp%n ) = data%tensor_model%X( : nlp%n )
             data%branch = 290 ; inform%status = 7 ; RETURN
           ELSE
             CALL eval_HPRODS( data%eval_status, nlp%X( : nlp%n ),             &
                               data%tensor_model%X( : nlp%n ), userdata,       &
                               nlp%P%val, got_h = .FALSE. )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_HPRODS'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         END SELECT

!  return from reverse communication with the Hessains-vector product

   290   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 290, status = ',&
        &  I0  )" ) prefix, inform%subproblem_inform%status

!  continue with the computation of r(s)

        SELECT CASE ( inform%subproblem_inform%status )

!  compute 1/2 ( s^T H_i s )_j

         CASE ( 2 )
           SELECT CASE ( SMT_get( nlp%P%type ) )
           CASE ( 'DENSE_BY_COLUMNS' )
             l = 0
             DO j = 1, nlp%m
               sths = zero
               DO i = 1, nlp%n
                 l = l + 1
                 sths = sths + nlp%P%val( l ) * data%tensor_model%X( i )
               END DO
               data%HSHS( j ) = half * sths
             END DO
           CASE ( 'COORDINATE' )
             data%HSHS( : nlp%m ) = zero
             DO l = 1, nlp%P%ne
               j = nlp%P%col( l )
               data%HSHS( j ) = data%HSHS( j )                                 &
                 + nlp%P%val( l ) * data%tensor_model%X( nlp%P%row( l ) )
             END DO
             data%HSHS( : nlp%m ) = half * data%HSHS( : nlp%m )
           CASE DEFAULT
             DO j = 1, nlp%m
               sths = zero
               DO l = nlp%P%ptr( j ), nlp%P%ptr( j + 1 ) - 1
                 sths = sths                                                   &
                  + nlp%P%val( l ) * data%tensor_model%X( nlp%P%row( l ) )
               END DO
               data%HSHS( j ) = half * sths
             END DO
           END SELECT

!  compute r(s) = c(x) + J(x) s + 1/2 ( s^T H_i s )_i

           data%tensor_model%C( : nlp%m ) =                                    &
             nlp%C( : nlp%m ) + data%JS( : nlp%m ) + data%HSHS( : nlp%m )

!  obtain the Jacobian J(s), where J^T(s) = ( g_i + H_i s )_i=1,...,m

         CASE ( 3 )
           data%tensor_model%J%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne )
           SELECT CASE ( SMT_get( nlp%P%type ) )
           CASE ( 'DENSE_BY_COLUMNS' )
             l = 0
             DO i = 1, nlp%m
               DO j = 1, nlp%n
                 l = l + 1
                 ll = data%Hs_map( l )
                 IF ( ll > 0 ) data%tensor_model%J%val( ll )                   &
                   = data%tensor_model%J%val( ll ) + nlp%P%val( l )
               END DO
             END DO
           CASE ( 'COORDINATE' )
             DO l = 1, nlp%P%ne
               ll = data%Hs_map( l )
               IF ( ll > 0 ) data%tensor_model%J%val( ll )                     &
                 = data%tensor_model%J%val( ll ) + nlp%P%val( l )
             END DO
           CASE DEFAULT
             DO i = 1, nlp%m
               DO l = nlp%P%ptr( i ), nlp%P%ptr( i + 1 ) - 1
                 ll = data%Hs_map( l )
                 IF ( ll > 0 ) data%tensor_model%J%val( ll )                   &
                   = data%tensor_model%J%val( ll ) + nlp%P%val( l )
               END DO
             END DO
           END SELECT

!  continue with the sums u = u + J(s) v or u = u + J^T(s) v by including the
!  terms P^T(x,s) v or P(x,s) v respectively

         CASE ( 5 )

!  compute u = u + P(x,s) v

           IF ( data%subproblem_data%transpose ) THEN
             CALL mop_Ax( one, nlp%P, data%subproblem_data%V( : nlp%m ), one,  &
                          data%subproblem_data%U( : nlp%n ), out = data%out,   &
                          error = data%control%error, print_level = 0,         &
                          transpose = .FALSE. )

!  compute u = u + P^T(x,s) v

           ELSE
             CALL mop_Ax( one, nlp%P, data%subproblem_data%V( : nlp%n ), one,  &
                          data%subproblem_data%U( : nlp%m ), out = data%out,   &
                          error = data%control%error, print_level = 0,         &
                          transpose = .TRUE. )
           END IF

         END SELECT

!  continue the computation of the search direction

       IF ( inform%subproblem_inform%status > 0 ) GO TO 270

!  search direction found

       data%S( : nlp%n ) = data%tensor_model%X( : nlp%n )
       CALL mop_Ax( one, data%regularization%matrix,                           &
                    data%S( : nlp%n ), zero,                                   &
                    data%subproblem_data%SX( : nlp%n ),                        &
                    symmetric = .TRUE., out = data%out,                        &
                    error = data%control%error, print_level = 0 )
       data%subproblem_data%xtsx                                               &
         = DOT_PRODUCT( data%S( : nlp%n ), data%subproblem_data%SX( : nlp%n ) )
       data%s_norm = SQRT( data%subproblem_data%xtsx )
       data%model = inform%subproblem_inform%obj - inform%obj                  &
         - ( data%regularization%weight / data%regularization%power )          &
            * data%s_norm ** data%regularization%power

       data%final_weight = inform%subproblem_inform%weight

!  update the factorization or Krylov iteration counts

       IF ( data%control%subproblem_control%subproblem_direct ) THEN
         data%total_facts                                                      &
           = data%total_facts + data%subproblem_data%total_facts
         inform%factorization_max = MAX( inform%factorization_max,             &
           inform%subproblem_inform%RQS_inform%factorizations )
         inform%factorization_average =  data%total_facts / inform%iter
       END IF

!  ============================================================================
!  3. check for acceptance of the new point
!  ============================================================================

!  see if the correction will make any difference

       IF ( MAXVAL( ABS( data%S( : nlp%n ) ) / MAX( one, nlp%X( : nlp%n ) ) )  &
            <= data%control%stop_s ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 900
       END IF

!  compute the slope and curvature along the step

       data%stg = DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )
       data%hstbs = data%model - data%stg
!      write(6,*) ' stg = ', data%stg

!  record the current point

       data%obj_current = inform%obj
       data%norm_c_current = inform%norm_c
       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%C_current( : nlp%m ) = nlp%C( : nlp%m )

!  form the trial point

       nlp%X( : nlp%n ) = data%X_current( : nlp%n ) + data%S( : nlp%n )

!  evaluate the objective function at the trial point

       IF ( data%reverse_c ) THEN
         data%branch = 320 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_C( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%C( : nlp%m ) )
         IF ( data%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_C'
           inform%status = GALAHAD_error_evaluation ; GO TO 900
         END IF
       END IF

!  return from reverse communication with the objective value

   320 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 320' )" ) prefix
       inform%c_eval = inform%c_eval + 1
       IF ( data%w_eq_identity ) THEN
         data%Y( : nlp%m ) = nlp%C( : nlp%m )
         data%norm_c_trial = TWO_NORM( nlp%C( : nlp%m ) )
         data%f_trial = half * data%norm_c_trial ** 2
       ELSE
         data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
         val = DOT_PRODUCT( data%Y( : nlp%m ), nlp%C( : nlp%m ) )
         data%norm_c_trial = SQRT( val )
         data%f_trial = half * val
       END IF

!      if(data%printi) write(6,*) ' f_trial ', data%f_trial

!  deal with NaN trial objective values
!  ------------------------------------

!      data%f_is_nan = IEEE_IS_NAN( data%f_trial )
       data%f_is_nan = data%f_trial /= data%f_trial
       IF ( data%f_is_nan ) THEN
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%C( : nlp%m ) = data%C_current( : nlp%m )

!  control printing for the NaN case

         IF ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print .AND.                              &
              MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) &
             THEN
           data%printi = data%set_printi ; data%printt = data%set_printt
           data%printm = data%set_printm ; data%printw = data%set_printw
           data%printd = data%set_printd
           data%print_level = data%control%print_level
           data%control%GLRT_control%print_level = data%print_level_glrt
           data%control%RQS_control%print_level = data%print_level_rqs
         ELSE
           data%printi = .FALSE. ; data%printt = .FALSE.
           data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
           data%print_level = 0
           data%control%GLRT_control%print_level = 0
           data%control%RQS_control%print_level = 0
         END IF
         data%print_iteration_header = data%print_level > 1 .OR.               &
           ( data%control%GLRT_control%print_level > 0 .AND. .NOT.             &
             data%control%subproblem_control%subproblem_direct ) .OR.          &
!            data%control%subproblem_direct ) .OR.                             &
           ( data%control%RQS_control%print_level > 0 .AND.                    &
             data%control%subproblem_control%subproblem_direct )
!            data%control%subproblem_direct )

!  print one-line summary

         IF ( data%printi ) THEN
            IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
             WRITE( data%out, 2090 ) prefix
             IF ( data%subproblem_control%subproblem_direct ) THEN
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2170 ) prefix
               ELSE
                 WRITE( data%out, 2160 ) prefix
               END IF
             ELSE
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2180 ) prefix
               ELSE
                 WRITE( data%out, 2190 ) prefix
               END IF
             END IF
           END IF
           data%print_1st_header = .FALSE.
           char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
           IF ( data%subproblem_control%subproblem_direct ) THEN
             char_facts =                                                      &
               ADJUSTR( STRING_integer_6( data%total_facts ) )
           ELSE
             char_facts =                                                      &
                 ADJUSTR( STRING_integer_6( inform%subproblem_inform%cg_iter ) )
           END IF
           WRITE( data%out,  "( A, A6, 1X, 3A1, '    NaN           -    ',     &
          &  '    - Inf ',  2ES8.1, 1X, A6, F8.2 )" )                          &
              prefix, char_iter, data%accept, data%negcur, data%hard,          &
              inform%weight, data%s_norm, char_facts, data%clock_now
         END IF

!  check to see if we are still "alive"

         IF ( data%control%alive_unit > 0 ) THEN
           INQUIRE( FILE = data%control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             inform%status = GALAHAD_error_alive
             RETURN
           END IF
         END IF

!  check to see if the iteration limit has been exceeded

         inform%iter = inform%iter + 1
         IF ( inform%iter > data%control%maxit .AND. data%step_accepted ) THEN
           inform%status = GALAHAD_error_max_iterations ; GO TO 900
         END IF

!  increase the regularization weight and try again

         IF ( inform%weight == zero ) THEN
           inform%weight = 0.0001_wp
         ELSE
           inform%weight = inform%weight * 10.0_wp
           inform%weight = data%control%weight_increase * inform%weight
         END IF
         GO TO 190
       END IF

!  compute the change in objective and the slope

       data%df = inform%obj - data%f_trial
!      if (data%printi) write(6,*) ' dm, df ', - data%model, data%df

!  compute the ratio of actual to predicted reduction over the current iteration

       rounding =                                                              &
         MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = wp ) * epsmch

       ared = data%df + rounding
       prered = - data%model + rounding
       IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )            &
         ared = prered
       data%ratio = ared / prered
!      write(6,*) ' ratio ', data%ratio, ared, prered
       IF ( data%printm ) WRITE( data%out, "( /, A, ' actual, predicted',      &
      &   ' reductions = ', 2ES12.4 )" ) prefix, ared, prered

!  compute the ratio of actual to predicted reduction over the recent history

       IF ( .NOT. data%monotone ) THEN

!  compute the largest f in the history

         data%ref = MAXLOC( data%F_hist( data%non_monotone_history + 2         &
                            - data%max_hist : data%non_monotone_history + 1 ) )
         data%f_ref = data%F_hist( data%ref( 1 ) )

!  use the larger of these two ratios to assess progress

         data%ratio = MAX( data%ratio, ( data%f_trial - data%f_ref ) /         &
           ( SUM( data%D_hist( data%ref( 1 ) + 1 :                             &
             data%non_monotone_history + 1 ) ) + data%model ) )
       END IF

!  the new point is acceptable

       IF ( data%ratio >= data%control%eta_successful ) THEN
         data%poor_model = .FALSE.
         data%accept = 'a'
         data%step_accepted = .TRUE.
         inform%norm_c = data%norm_c_trial
         inform%obj = data%f_trial
         data%s_norm_successful = data%s_norm

!  stop if the residual is sufficiently small

         IF ( inform%norm_c <= data%stop_c ) THEN

!  print one-line summary

           IF ( data%printi ) THEN
             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
             data%time_now = data%time_now - data%time_start
             data%clock_now = data%clock_now - data%clock_start
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2090 ) prefix
               IF ( data%subproblem_control%subproblem_direct ) THEN
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2170 ) prefix
                 ELSE
                   WRITE( data%out, 2160 ) prefix
                 END IF
               ELSE
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2180 ) prefix
                 ELSE
                   WRITE( data%out, 2190 ) prefix
                 END IF
               END IF
             END IF
             data%print_1st_header = .FALSE.
             char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
             IF ( inform%iter > 0 ) THEN
               IF ( data%subproblem_control%subproblem_direct ) THEN
                 char_facts =                                                  &
                   ADJUSTR( STRING_integer_6( data%total_facts ) )
               ELSE
                 char_facts =                                                  &
                 ADJUSTR( STRING_integer_6( inform%subproblem_inform%cg_iter ) )
               END IF
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2120 ) prefix, char_iter, data%accept,       &
                    data%negcur, data%hard, inform%obj,                        &
                    inform%norm_g, data%ratio, data%old_weight,                &
                    data%s_norm, char_facts, data%clock_now
               ELSE
                 WRITE( data%out, 2120 ) prefix, char_iter, data%accept,       &
                    data%negcur, data%hard, inform%norm_c,                     &
                    inform%norm_g, data%ratio, data%old_weight,                &
                    data%s_norm, char_facts, data%clock_now
               END IF
             ELSE
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2140 ) prefix,                               &
                    char_iter, inform%obj, inform%norm_g
               ELSE
                 WRITE( data%out, 2140 ) prefix,                               &
                    char_iter, inform%norm_c, inform%norm_g
               END IF
             END IF
           END IF
           inform%status = GALAHAD_ok ; GO TO 900
         END IF

!  update the history

         IF ( data%monotone ) THEN
           data%f_ref = inform%obj

!  shift history of function and model values

         ELSE
           DO i = 1, data%non_monotone_history
             data%F_hist( i ) = data%F_hist( i + 1 )
             data%D_hist( i ) = data%D_hist( i + 1 )
           END DO

!  replace the oldest

           data%F_hist( data%non_monotone_history + 1 ) = inform%obj
           data%D_hist( data%non_monotone_history + 1 ) = data%model

!  find how much past history is allowed

           data%max_hist = MIN( data%max_hist + 1, data%non_monotone_history )
         END IF

!  the new point is not acceptable

       ELSE
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%C( : nlp%m ) = data%C_current( : nlp%m )
         IF ( data%w_eq_identity ) THEN
           data%Y( : nlp%m ) = nlp%C( : nlp%m )
         ELSE
           data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
         END IF
         data%new_point = .FALSE.
       END IF

!  ==========================================================
!  4. Update the regularization weight and other book-keeping
!  ==========================================================

       data%old_weight = inform%weight
       data%successful = data%ratio >= data%control%eta_successful

!  update the weight

       SELECT CASE ( data%control%weight_update_strategy )
       CASE ( weight_update_zero_reset )
         IF ( data%ratio < data%control%eta_successful ) THEN
           inform%weight = MAX( data%control%weight_increase * inform%weight,  &
                                data%minimum_weight )
         ELSE IF ( data%ratio >= data%control%eta_very_successful .AND.        &
                  data%ratio <= data%control%eta_too_successful ) THEN
           inform%weight = weight_zero
         END IF
!      CASE ( weight_update_gpt )
!         CALL ARC_adjust_weight( inform%weight, data%model, data%stg,         &
!                                 data%hstbs, data%s_norm, data%ratio,         &
!                                 data%ARC_control )
!         inform%weight = MAX( data%minimum_weight, inform%weight )

!         IF ( data%ratio < control%eta_successful ) THEN
!           IF ( data%control%subproblem_direct ) THEN
!             val = two * inform%RQS_inform%pole / data%s_norm_successful
!           ELSE
!             val = - two * inform%GLRT_inform%leftmost /data%s_norm_successful
!           END IF
!           inform%weight = MAX( inform%weight, val )
!         END IF
       CASE DEFAULT
         IF ( data%ratio < data%control%eta_successful ) THEN
           inform%weight = data%control%weight_increase * inform%weight
         ELSE IF ( data%ratio >= data%control%eta_very_successful .AND.        &
                  data%ratio <= data%control%eta_too_successful ) THEN
           inform%weight = MAX( data%minimum_weight,                           &
                                  data%control%weight_decrease * inform%weight )
         END IF
       END SELECT

       IF ( data%ratio >= data%control%eta_successful ) THEN
         IF ( data%control%model == tensor_newton_model .OR.                   &
              data%control%model == tensor_gauss_to_newton_model .OR.          &
              data%control%model == tensor_gauss_newton_model ) THEN
           data%inner_weight = data%control%initial_inner_weight
         END IF

!  compute ||s||_S

         IF ( data%control%norm /= euclidean_regularization ) THEN
           IF ( data%control%renormalize_weight ) THEN
             data%s_new_norm = data%s_norm
             IF ( data%printt )                                                &
               WRITE( data%out, "( A, ' ratio new, old norms = ', ES12.4 )" )  &
                 prefix, data%s_new_norm / data%s_norm
           ELSE
             data%s_new_norm = data%s_norm
           END IF

!  if the norm has changed, adjust the weight accordingly

           inform%weight = inform%weight * ( data%s_new_norm / data%s_norm )
           data%s_norm = data%s_new_norm
         END IF
       END IF
!      write(6,*) 'weight', inform%weight

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start
       IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,     &
      &    F12.2,  ' seconds' )" ) prefix, data%clock_now
       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
              data%time_now > data%control%cpu_time_limit ) .OR.               &
            ( data%control%clock_time_limit >= zero .AND.                      &
              data%clock_now > data%control%clock_time_limit ) ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

!  =========================================
!  End of the main (Tensor-Newton) iteration
!  =========================================

       GO TO 100

!  ==================================
!  Newton or Gauss-Newton solver used
!  ==================================

 800 CONTINUE
     IF ( control%out > 0 .AND. control%print_level >= 4 )                     &
       WRITE( control%out, "( A, ' statement 800' )" ) prefix
     data%NLS_subproblem_data_type%regularization_type = control%norm

 810 CONTINUE
     data%branch = data%branch_newton
!    WRITE(6,*) ' in data%branch ', data%branch
     CALL NLS_subproblem_solve( nlp, data%subproblem_control,                  &
                            inform%NLS_subproblem_inform_type,                 &
                            data%NLS_subproblem_data_type, userdata, W = W,    &
                            eval_C = eval_C, eval_J = eval_J, eval_H = eval_H, &
                            eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,  &
                            eval_SCALE = eval_SCALE )
!    WRITE( 6, * ) ' out data%branch ', &
!      data%branch, inform%NLS_subproblem_inform_type%status
     data%branch_newton = data%branch
     SELECT CASE ( inform%NLS_subproblem_inform_type%status )
     CASE ( 2 : 6, 8 )
       inform%status = inform%NLS_subproblem_inform_type%status
       data%branch = 820 ; RETURN
     CASE( : - 1 )
       inform%status = inform%NLS_subproblem_inform_type%status
       GO TO 990
     CASE DEFAULT
       inform%status = GALAHAD_ok
       GO TO 900
     END SELECT

 820 CONTINUE
     GO TO 810

!  ================
!  Terminal returns
!  ================

 900 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 900' )" ) prefix

!  print details of solution

     IF ( inform%norm_c > zero ) THEN
       inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) ) / inform%norm_c
     ELSE
       inform%norm_g = zero
     END IF
!    write(6,*) ' final weight = ', inform%weight

 910 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 910' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi .AND. .NOT. data%n_or_gn ) THEN

!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%c_eval, inform%j_eval, inform%h_eval,   &
!         inform%iter, inform%cg_iter, inform%obj, inform%norm_g
!      WRITE ( data%out, 2010 )
!      IF ( data%print_level > 3 ) THEN
!         l = nlp%n
!      ELSE
!         l = 2
!      END IF
!      DO j = 1, 2
!         IF ( j == 1 ) THEN
!            ir = 1 ; ic = MIN( l, nlp%n )
!         ELSE
!            IF ( ic < nlp%n - l ) WRITE( data%out, 2050 )
!            ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
!         END IF
!         DO i = ir, ic
!            WRITE ( data%out, 2020 ) nlp%vnames( i ), nlp%X( i ), nlp%G( i )
!         END DO
!      END DO

       WRITE( data%out, "( /, A, '  Problem: ', A, ' (n = ', I0, ', m = ', I0, &
    &   ')', /, A, '  NLS stopping tolerances (c,J''c/c) =', 2ES9.2 )" )     &
       prefix, TRIM( nlp%pname ), nlp%n, nlp%m, prefix, data%stop_c, data%stop_g
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
         data%non_monotone_history
       SELECT CASE( data%model_used )
       CASE ( tensor_gauss_newton_model )
         WRITE( data%out, "( A, '  Regularized tensor Gauss-Newton',           &
       &   ' model used'  )" ) prefix
       CASE ( tensor_newton_model )
         WRITE( data%out, "( A, '  Regularized tensor Newton',                 &
       &   ' model used'  )" ) prefix
       CASE ( tensor_gauss_to_newton_model )
         WRITE( data%out, "( A, '  Regularized tensor Gauus-Newton then',      &
       &   ' Newton model used'  )" ) prefix
       END SELECT
       WRITE( data%out, "( A, '  Regularization power =', F4.1 )" )            &
          prefix, data%power
       IF ( data%control%subproblem_control%subproblem_direct ) THEN
!      IF ( data%control%subproblem_direct ) THEN
         IF ( inform%RQS_inform%dense_factorization ) THEN
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (eigen solver SYSV',                       &
          &      ') of the regularization sub-problem' )" ) prefix
         ELSE
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (solver ', A,                              &
          &      ') of the regularization sub-problem' )" )                    &
              prefix, TRIM( data%control%RQS_control%definite_linear_solver )
         END IF
         SELECT CASE ( data%regularization_type )
         CASE ( user_regularization )
           WRITE( data%out, "( A, '  User-defined regularization used' )" )    &
             prefix
         CASE ( euclidean_regularization )
           WRITE( data%out, "( A, '  Euclidean regularization used' )" ) prefix
         CASE ( diagonal_jtj_regularization )
           WRITE( data%out, "( A, '  Diagonal (JTJ) regularization used' )" )  &
             prefix
         CASE ( diagonal_hessian_regularization )
           WRITE( data%out, "( A, '  Diagonal (H) regularization used' )" )    &
             prefix
         CASE ( band_regularization )
           WRITE( data%out, "( A, '  Band regularization (semi-bandwidth ',    &
          &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
         CASE ( reordered_band_regularization )
           WRITE( data%out, "( A, ' Reordered band regularization',            &
          &  ' (semi-bandwidth ',  I0, ') used' )" ) prefix,                   &
             inform%PSLS_inform%semi_bandwidth_used
         CASE ( schnabel_eskow_regularization, gmps_regularization,            &
                lin_more_regularization, mi28_regularization )
           WRITE( data%out, "( A, '  Modified full matrix regularization',     &
          &  ' used' )" ) prefix
         END SELECT
         WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
        &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
           inform%RQS_inform%factorizations,                                   &
           inform%RQS_inform%time%clock_factorize
         IF ( TRIM( data%control%RQS_control%definite_linear_solver ) ==       &
              'pbtr' ) THEN
           WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
          & ', semi-bandwidth = ', I0  )" ) prefix, inform%max_entries_factors,&
              inform%RQS_inform%SLS_inform%semi_bandwidth
         ELSE
           WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
             prefix, inform%max_entries_factors
         END IF
       ELSE
         WRITE( data%out,                                                      &
           "( A, '  Iterative solution of the regularization sub-problem' )" ) &
              prefix
         IF ( data%regularization_type > 0 )                                   &
           WRITE( data%out, "( A, '  Hessian semi-bandwidth (original,',       &
          &     ' re-ordered) = ', I0, ', ', I0 )" ) prefix,                   &
             inform%PSLS_inform%semi_bandwidth,                                &
             inform%PSLS_inform%reordered_semi_bandwidth
         SELECT CASE ( data%regularization_type )
         CASE ( user_regularization )
           WRITE( data%out, "( A, '  User-defined regularization used' )" )    &
             prefix
         CASE ( euclidean_regularization )
           WRITE( data%out, "( A, '  Euclidean regularization used' )" ) prefix
         CASE ( diagonal_jtj_regularization )
           WRITE( data%out, "( A, '  Diagonal (JTJ) regularization used' )" )  &
             prefix
         CASE ( diagonal_hessian_regularization )
           WRITE( data%out, "( A, '  Diagonal (H) regularization used' )" )    &
             prefix
         CASE ( band_regularization )
           WRITE( data%out, "( A, '  Band regularization (semi-bandwidth ',    &
          &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
         CASE ( reordered_band_regularization )
           WRITE( data%out, "( A, ' Reordered band regularization',            &
          &  ' (semi-bandwidth ', I0, ') used' )" ) prefix,                    &
             inform%PSLS_inform%semi_bandwidth_used
         CASE ( schnabel_eskow_regularization )
           WRITE( data%out, "( A, '  SE (solver ', A, ') full regularization', &
          & ' used' )" ) prefix,                                               &
             TRIM( data%control%PSLS_control%definite_linear_solver )
         CASE ( gmps_regularization )
          WRITE( data%out, "( A, '  GMPS (solver ', A, ') full',               &
         & ' regularization used')" ) prefix,                                  &
            TRIM( data%control%PSLS_control%definite_linear_solver )
         CASE ( lin_more_regularization )
           WRITE( data%out, "( A, '  Lin-More''(', I0, ') incomplete',         &
          &  ' Cholesky factorization regularization used ' )" )               &
            prefix, data%control%PSLS_control%icfs_vectors
         CASE ( mi28_regularization )
           WRITE( data%out, "( A, '  HSL_MI28(', I0, ',', I0,                  &
             & ') incomplete Cholesky factorization regularization used ' )" ) &
             prefix, data%control%PSLS_control%mi28_lsize,                     &
            data%control%PSLS_control%mi28_rsize
         END SELECT
         IF ( data%control%renormalize_weight ) WRITE( data%out,               &
            "( A, '  Weight renormalized' )" ) prefix
       END IF
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 980' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 990' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'NLS_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # Hessian evaluations   = ', I10,                        &
              /, A, ' # major  iterations     = ', I10,                        &
              /, A, ' # minor (cg) iterations = ', I10,                        &
             //, A, ' objective value         = ', ES22.14,                    &
              /, A, ' gradient norm           = ', ES12.4 )
 2010 FORMAT( /, A, ' name                  X                   G ' )
 2020 FORMAT(  A, 1X, A10, 2ES22.14 )
 2030 FORMAT(  A, 1X, I10, 2ES22.14 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2090 FORMAT( A, '        (a=accept r=reject b=TR boundary',                   &
                 ' n=-ve curvature h=hard case)' )
 2120 FORMAT( A, A6, 1X, 3A1, 2ES11.4, ES9.1, 2ES8.1, 1X, A6, F8.2 )
 2140 FORMAT( A, A6, 4X, 2ES11.4 )
 2160 FORMAT( A, '    It         c        J''c/c     ',                        &
             ' ratio   weight   step  # fact    time' )
 2170 FORMAT( A, '    It         f           g      ',                         &
             ' ratio   weight   step  # fact    time' )
 2180 FORMAT( A, '    It         c        J''c/c     ',                        &
             ' ratio   weight   step    # cg    time' )
 2190 FORMAT( A, '    It         f           g      ',                         &
             ' ratio   weight   step    # cg    time' )

 !  End of subroutine NLS_solve

     END SUBROUTINE NLS_solve

!-  G A L A H A D - N L S _ s u b p r o b l e m _ s o l v e  S U B R O U T I N E

     SUBROUTINE NLS_subproblem_solve( nlp, control, inform, data, userdata,    &
                                      W, stabilisation, eval_C, eval_J,        &
                                      eval_H, eval_JPROD, eval_HPROD,          &
                                      eval_SCALE )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  NLS_solve, a regularization method for finding a local unconstrained
!    minimizer of a stabilised nonlinear least-squares objective,
!      f(x) = 1/2 ||c(x)||_W^2 + sigma/p ||x||_S^p,
!    where W and S are positive-definite matrices, ||x||_S^2 = x^T S x
!    and the weight sigma is non-negative

!  This variant implements the Newton or Gauss-Newton method

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_NLS.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_NLS_double
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m is a scalar variable of type default integer, that holds the number of
!   residuals
!
!  C is a rank-one allocatable array of dimension m and type default real,
!   that holds the residuals c(x). The i-th component of C, i = 1,  ... ,  m,
!   contains c_i(x).
!
!  J is scalar variable of type SMT_TYPE that holds the Jacobian matrix
!   J(x) = nabla r(x), i.e., J_i,j(x) = d r_i(x) / d x_j. The following
!   components are used here:
!
!   J%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of J%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of J%type must
!    contain the string COORDINATE, and for the sparse row-wise storage scheme,
!    the first fourteen components of J%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into J%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Jacobian we wish
!    to store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%J%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   J%ne is a scalar variable of type default integer, that holds the number
!    of entries in the Jacobian J(x) in the sparse co-ordinate storage scheme.
!    It need not be set for any of the other two schemes.
!
!   J%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries in the Jacobian J(x) in any of the available
!    storage schemes.
!
!   J%row is a rank-one allocatable array of type default integer, that holds
!    the row indices in the Jacobian J(x) in the sparse co-ordinate storage
!    scheme. It need not be allocated for any of the other two schemes.
!
!   J%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the Jacobian J(x) in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme.
!    It need not be allocated when the dense scheme is used.
!
!   J%ptr is a rank-one allocatable array of dimension m+1 and type default
!    integer, that holds the starting position of each row of Jacobian J(x),
!    as well as J%ptr(m+1) = the total number of entries plus one, in the
!    sparse row-wise storage scheme. It need not be allocated when the other
!    schemes are used.
!
!  H(x,y) is scalar variable of type SMT_TYPE that holds the scaled Hessian
!   matrix H(x,y) = sum_{i=1}^m y_i(x) H_i(x), where H_i(x) is the Hessian
!   of c_i(x) and y is given. The following components are used here:
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
!         CALL SMT_put( nlp%H%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   H%ne is a scalar variable of type default integer, that holds the number of
!    entries in the lower triangular part of H(x) in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the lower triangular part of H(x) in the sparse
!    co-ordinate storage scheme. It need not be allocated for any of the other
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the  lower triangular part of H(x) in
!    either the sparse co-ordinate, or the sparse row-wise storage scheme. It
!    need not be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of each row of the lower
!    triangular part of H(x), as well as H%ptr(n+1) = the total number of
!    entries plus one, in the sparse row-wise storage scheme. It need not be
!    allocated when the other schemes are used.
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
!  CNAMES is a rank-one allocatable array of dimension m and type default
!   character and length 10, whose i-th entry contains the ``name'' of the i-th
!   residual for printing. This is only used  if ``debug''printing
!   control%print_level > 4) is requested, and will be ignored if the array is
!   not allocated.
!
! controls is a scalar variable of type NLS_control_type. See
!  NLS_initialize for details
!
! inform is a scalar variable of type NLS_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will
!  have been set in inform:
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
!
!     0. The run was successful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending
!        array is written on unit control%error and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains
!        its relevant string 'DENSE', 'COORDINATE' or 'SPARSE_BY_ROWS'
!          has been violated.
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
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named
!        control%alive_file from unit unit control%alive_unit.
!
!     2. The user should compute the residual function value c(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The value of
!        the i-th component of the residual should be set in nlp%C(i), for i =
!        1, ..., m and data%eval_status should be set to 0. If the user is
!        unable to evaluate a component of c(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%C, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the Jacobian of the residual function J(x) =
!        nabla_x c(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value l-th component of the Jacobian stored according
!        to the scheme input in the remainder of nlp%J should be set in
!        nlp%J%val(l), for l = 1, ..., nlp%J%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of J(x) -
!        for instance if a component of the Jacobian is undefined at x - the
!        user need not set nlp%J%val, but should then set data%eval_status
!        to a non-zero value.
!     4. The user should compute the weighted Hessian of the residual function
!        H(x,y) = sum_{i=1}^m y_i nabla_xx y_i(x) at the point x indicated
!        in nlp%X with weights y given by data%Y, and then re-enter the
!        subroutine. The value l-th component of H(x,y) stored according to
!        the scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of H(x,y) -
!        for instance, if a component of the Hessian is undefined at (x,y) - the
!        user need not set nlp%H%val, but should then set data%eval_status
!        to a non-zero value.
!     5. The user should compute the product J(x)v (when transpose = .FALSE.)
!        or J^T(x)v (when transpose = .TRUE.) of the Jacobian of the residual
!        function J(x) (or its traspose) at the point x indicated in nlp%X
!        with the vector v, and add the result to the vector u and then re-enter
!        the subroutine. The logical transpose and vectors u and v are given
!        in data%transpose, data%U and data%V respectively, the
!        resulting vector u + J(x) or u + J^T(x)v as appropriate should be set
!        in data%U and data%eval_status should be set to 0. If the user
!        is unable to evaluate the product - for instance, if a component of
!        J(x) is undefined at x - the user need not alter data%U, but
!        should then set data%eval_status to a non-zero value.
!     6. The user should compute the product H(x,y)v of the Hessian of
!        the residual function H(x,y) at the point (x,y) indicated in nlp%X
!        and data%Y with the vector v and add the result to the vector u
!        and then re-enter the subroutine. The vectors u and v are given in
!        data%U and data%V respectively, the resulting vector
!        u + H(x,y)v should be set in data%U and  data%eval_status
!        should be set to 0. If the user is unable to evaluate the product -
!        for instance, if a component of H(x,y) is undefined at (x,y) - the
!        user need not alter data%U, but should then set
!        data%eval_status to a non-zero value.
!     7. Not used by this subroutine.
!     8. The user should compute the product u = S(x)v of their preconditioner
!        S(x) at the point x indicated in nlp%X with the vector v. The vectors
!        v is given in data%V, the resulting vector u = S(x)v should be set
!        in data%U and data%eval status should be set to 0. If the user is
!        unable to evaluate the product—for instance, if a component of the
!        preconditioner is undefined at x—the user need not set data%U, but
!        should then set data%eval statusto a non-zero value.
!     9. The user has the opportunity to replace the estimate x in nlp%X
!        by a value x_better for which f(x_better) <= f(x). If the user
!        choses to do so, she should replace nlp%X by x_better and also
!        record c(x_better) in nlp%C.
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
!  factorization_status is a scalar variable of type default integer, that
!   gives the return status from the matrix factorization.
!
!  factorization_integer is a scalar variable of type default integer,
!   that gives the amount of integer storage used for the matrix factorization.
!
!  factorization_real is a scalar variable of type default integer,
!   that gives the amount of real storage used for the matrix factorization.
!
!  c_eval is a scalar variable of type default integer, that gives the
!   total number of residual function evaluations performed.
!
!  j_eval is a scalar variable of type default integer, that gives the
!   total number of residual Jacobian evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of scaled Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function 1/2 ||c(x)||_2^2 at the best estimate
!   of the solution found.
!
!  norm_c is a scalar variable of type default real, that holds the value of
!   the norm of the residual function ||c(x)||_2 at the best estimate of the
!   solution found.
!
!  norm_g is a scalar variable of type default real, that holds the value of
!   the norm of the residual function gradient ||J^T(x)c(x)||_2/||c(x)||_2
!   at the best estimate of the solution found.
!
!  time is a scalar variable of type NLS_time_type whose components are
!   used to hold elapsed CPU and clock times for the various parts of the
!   calculation.
!
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    preprocess is a scalar variable of type default real, that gives the
!      CPU time spent reordering the problem to standard form prior to solution.
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
!    clock_preprocess is a scalar variable of type default real, that gives
!      the clock time spent reordering the problem to standard form prior
!      to solution.
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
!  data is an a scalar variable of type NLS_data_type used for
!   internal data.
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
!  W is an optional rank-one array of type default real that if present
!   must be of length nlp%m and filled with the weights w_i > 0. If W is
!   absent, weights of one will be used.
!
!  stabilisation is an optional scalar variable of type NLS_regularization_type
!   that contains the data for the stablilisation term, weight/p ||x||_S^p. If
!   absent, no stabilisation term is added, while if present, the following
!   components should be set:
!
!    weight is a scalar variable of type default real, that holds the value
!      sigma in the stabilisation sigma/p ||x||_S^p. The default is %weight =
!      0.0. If %weight = 0.0, the remaining values below need not be set.
!
!    power is a scalar variable of type default real, that holds the power
!      p of the stabilisation sigma/p ||x||_S^p. The default is %power = 2.0.
!
!    matrix is scalar variable of type SMT_TYPE that holds the
!     regularization scaling matrix S. The following components are used here:
!
!     matrix%type is an allocatable array of rank one and type
!      default character, that is used to indicate the storage scheme used.
!      If the dense storage scheme is used, the first five components of
!      %matrix%type must contain the string DENSE, for the sparse
!      co-ordinate scheme, the first ten components must contain the string
!      COORDINATE, for the sparse row-wise storage scheme, the first fourteen
!      components must contain the string SPARSE_BY_ROWS, for the diagonal
!      storage scheme, the first eight components must contain the string
!      DIAGONAL, for the scaled identity scheme, the first fourteen characters
!      must contain the string SCALED_IDENTITY, and for the identity scheme
!      the first eight characters must contain the string IDENTITY.
!
!      For convenience, the procedure SMT_put may be used to allocate sufficient
!      space and insert the required keyword into %matrix%type.
!      For example, if S is to be input using the co-ordinate scheme, we may
!      simply
!
!           CALL SMT_put( data%matrix%type, 'COORDINATE', stat )
!
!      See the documentation for the galahad package SMT for further details
!      on the use of SMT_put.
!
!     matrix%ne is a scalar variable of type default integer, that holds
!      the number of entries in the lower triangular part of S in the sparse
!      co-ordinate storage scheme. It need not be set for any of the other
!      three schemes.
!
!     matrix%val is a rank-one allocatable array of type default
!      real, that holds the values of the entries of the lower triangular part
!      of S in any of the available non-identity storage schemes. If the scaled
!      identity scheme is used, only the value %matrix%val(1) need be specified,
!      while for the identity scheme, %matrix%val is not accessed.
!
!     matrix%row is a rank-one allocatable array of type default
!      integer, that holds the row indices of the lower triangular part of S
!      in the sparse co-ordinate storage scheme. It need not be allocated for
!      any of the other three schemes.
!
!     matrix%col is a rank-one allocatable array variable of
!      type default integer, that holds the column indices of the lower
!      triangular part of S in either the sparse co-ordinate, or the
!      sparse row-wise storage scheme. It need not be allocated when the
!      dense or diagonal storage schemes are used.
!
!     matrix%ptr is a rank-one allocatable array of dimension
!      n+1 and type default integer, that holds the starting position of
!      each row of the lower triangular part of S, as well as
!      %matrix%ptr(n+1) = the total number of entries plus one,
!      in the sparse row-wise storage scheme. It need not be allocated when
!      the other schemes are used.
!
!  eval_C is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the residual
!   function c(x) evaluated at x=X must be returned in C, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_C is not present, NLS_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_J is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Jacobian
!   nabla_x c(x) of the residual function evaluated at x=X must be returned in
!   J_val in the same order as presented in nlp%J,, and the status variable set
!   to 0. If the evaluation is impossible at X, status should be set to a
!   nonzero value. If eval_J is not present, NLS_solve will return to the
!   user with inform%status = 3 each time an evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the weighted Hessian
!   H(x,y) = sum_i y_i nabla_xx c_i(x) of the residual function evaluated at
!   x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_H is not present,
!   NLS_solve will return to the user with inform%status = 4 each time an
!   evaluation is required.
!
!  eval_JPROD is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum u + J(x) v,
!   (when transpose=.FALSE.) or u + J^T(x) v (when transpose=.TRUE.)
!   of the Jacobian (or its transpose) evaluated  at x=X with the vector v=V
!   and the vector u=U must be returned in U, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_JPROD is not present, NLS_solve will return to the user
!   with inform%status = 5 each time an evaluation is required. The Jacobian
!   has already been evaluated or used at x=X if got_j is .TRUE.
!
!  eval_HPROD is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum u + H(x,y) v,
!   where H(x,y) = sum_i y_i nabla_xx c_i(x), of u=U and the product of the
!   weighted Hessian HC(x,y) evaluated at x=X and y=Y with the vector v=V,
!   and the vector u=U must be returned in U, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_HPROD is not present, NLS_solve will return to the user
!   with inform%status = 6 each time an evaluation is required. The Hessian
!   has already been evaluated or used at x=X if got_h is .TRUE.
!
!  eval_SCALE is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = S(x) v of the
!   user's scaling matrix S(x) evaluated at x=X with the vector v=V, the result
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_SCALE
!   is not present, NLS_solve will return to the user with inform%status = 8
!   each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( NLS_subproblem_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_subproblem_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( NLS_subproblem_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: W
     TYPE ( NLS_regularization_data_type ), INTENT( IN ),                      &
                                            OPTIONAL :: stabilisation
     OPTIONAL :: eval_C, eval_J, eval_H, eval_JPROD, eval_HPROD, eval_SCALE

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_C( status, X, userdata, C )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_C
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_J( status, X, userdata, J_val )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_J
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_H( status, X, Y, userdata, H_val )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_JPROD( status, X, userdata, transpose, U, V, got_j )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_j
       END SUBROUTINE eval_JPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HPROD( status, X, Y, userdata, U, V, got_h )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_SCALE( status, X, userdata, U, V )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_SCALE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, ic, ir, l, facts_this_solve
     REAL ( KIND = wp ) :: ared, prered, rounding, val

     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_facts, char_sit, char_sit2
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 ) prefix =                         &
       control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) THEN
       data%branch = 10
     ELSE IF ( inform%status == 11 ) THEN
       data%branch = 20
     END IF

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! re-entry without initialization
       IF ( PRESENT( stabilisation ) ) THEN
         data%stabilised = stabilisation%weight > zero
       ELSE
         data%stabilised = .FALSE.
       END IF
       GO TO 20
     CASE ( 30 )  ! initial residual evaluation
       GO TO 30
     CASE ( 40 )  ! initial norm scaling
       GO TO 40
     CASE ( 110 ) ! initial Jacobian evaluation or Jacobian transpose vect prod
       GO TO 110
     CASE ( 120 ) ! Hessian evaluation
       GO TO 120
     CASE ( 220 ) ! norm scaling
       GO TO 220
     CASE ( 230 ) ! Jacobian vector product
       GO TO 230
     CASE ( 240 ) ! Jacobian transpose vector product
       GO TO 240
     CASE ( 250 ) ! Hessian-vector or scaling-matrix product
       GO TO 250
     CASE ( 320 ) ! residual evaluation
       GO TO 320
     CASE ( 330 ) ! norm scaling
       GO TO 330
     CASE ( 380 ) ! allow the user to compute a "magic" step
       GO TO 380
     CASE ( 390 ) ! norm scaling after a "magic" step
       GO TO 390
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
     data%set_printw = control%out > 0 .AND. control%print_level >= 4
     IF ( data%set_printw )                                                    &
       WRITE( control%out, "( A, ' statement 10' )" ) prefix

!  record controls and ensure that data is consistent

     data%control%NLS_subproblem_control_type = control

     data%w_eq_identity = .NOT. PRESENT( W )
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     data%control%initial_inner_weight                                         &
       = MAX( data%control%initial_inner_weight, zero )
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%successful = .TRUE.
     data%negcur = ' '
     data%ratio = - one
     data%total_facts = 0
     data%nskip_prec = nskip_prec_max
     data%reduce = .FALSE.

     inform%iter = 0 ; inform%cg_iter = 0
     inform%c_eval = 0 ; inform%j_eval = 0 ; inform%h_eval = 0
     inform%factorization_max = 0 ; inform%factorization_status = 0
     inform%max_entries_factors = 0 ; inform%factorization_average = zero
     inform%factorization_integer = - 1 ; inform%factorization_integer = - 1

!  check to see if stabilisation is required

     IF ( PRESENT( stabilisation ) ) THEN
       data%stabilised = stabilisation%weight > zero
     ELSE
       data%stabilised = .FALSE.
     END IF

!write(6,*) stabilisation%weight, stabilisation%power, data%stabilised

!  decide how much reverse communication is required

     data%reverse_c = .NOT. PRESENT( eval_C )

!  check to see if the Jacobian is available explicitly or only via its
!  action on a vector, and whether reverse communication will be required

!    write(6,*) ' data%control%jacobian_available ', data%control%jacobian_available
     data%jacobian_available = data%control%jacobian_available >= 2 .OR.       &
       PRESENT( eval_J )
     IF ( data%jacobian_available ) THEN
       data%reverse_j = .NOT. PRESENT( eval_J )
     ELSE
       data%reverse_j = .FALSE.
       data%control%subproblem_direct = .FALSE.
       IF ( data%control%norm >= 0 )                                           &
         data%control%norm = euclidean_regularization
     END IF
     data%reverse_jprod = .NOT. PRESENT( eval_JPROD )

!  check to see if the Hessian is available explicitly, available via its
!  action on a vector, or is unavailable, and whether reverse communication
!  will be required

!    data%hessian_available = data%control%hessian_available >= 2 .OR.         &
!      PRESENT( eval_H )
     data%hessian_available = data%control%hessian_available >= 2
     IF ( data%hessian_available ) THEN
       data%reverse_h = .NOT. PRESENT( eval_H )
     ELSE IF ( data%control%hessian_available == 1 ) THEN
       data%reverse_h = .FALSE.
       IF ( data%control%model /= first_order_model .AND.                      &
            data%control%model /= diagonal_hessian_model .AND.                 &
            data%control%model /= gauss_newton_model )                         &
         data%control%subproblem_direct = .FALSE.
     ELSE
       IF ( data%control%model /= first_order_model .AND.                      &
            data%control%model /= diagonal_hessian_model .AND.                 &
            data%control%model /= gauss_newton_model )                         &
         data%control%model = gauss_newton_model
     END IF
     data%reverse_hprod = .NOT. PRESENT( eval_HPROD )

!  initialize the model to Gauss-Newton if the Gauss-Newton to Newton
!  strategy has been specified

     data%gauss_to_newton_model =                                              &
       data%control%model == gauss_to_newton_model
     data%map_h_to_jtj = data%hessian_available .AND.                          &
                         ( data%gauss_to_newton_model .OR.                     &
                           data%control%model == newton_model )
     data%model_used = data%control%model
     IF ( data%gauss_to_newton_model )                                         &
       data%control%model = gauss_newton_model
     data%hessian_computed = data%hessian_available .AND.                      &
       data%control%model == newton_model

!  record the problem dimensions

     IF ( data%jacobian_available ) THEN
        nlp%J%m = nlp%m ; nlp%J%n = nlp%n
     END IF

!  check that the Hessian is specified in a permitted format

!inform%status = -1
!write(6,*) ' **************** deliberate quit', data%hessian_available
!GO TO 990
     IF ( data%hessian_available ) THEN
       SELECT CASE ( SMT_get( nlp%H%type ) )
       CASE ( 'DIAGONAL', 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE',             &
              'IDENTITY', 'SCALE_IDENTITY', 'NONE', 'ZERO' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input H%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%H%type )
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END SELECT
     END IF

!  find the number of nonzeros in the Hessian

     IF ( data%map_h_to_jtj ) THEN
       nlp%H%n = nlp%n ; nlp%H%m = nlp%n
       SELECT CASE (  SMT_get( nlp%H%type ) )
       CASE ( 'DIAGONAL' )
         nlp%H%ne = nlp%n
       CASE ( 'DENSE' )
         IF ( MOD(  nlp%n, 2 ) == 0 ) THEN
           nlp%H%ne = ( nlp%n / 2 ) * ( nlp%n + 1 )
         ELSE
           nlp%H%ne = nlp%n * ( ( nlp%n + 1 ) / 2 )
         END IF
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%H%ne = nlp%H%ptr( nlp%n + 1 ) - 1
       CASE ( 'NONE' )
         nlp%H%ne = 0
       END SELECT
     END IF

!  decide whether to form the scaling matrix and make model-specific choices

     IF ( data%control%model == newton_model ) THEN
       data%form_regularization                                                &
         = data%jacobian_available .AND. data%hessian_available
     ELSE IF ( data%control%model == gauss_newton_model ) THEN
       data%form_regularization = data%jacobian_available
     ELSE
       IF ( data%control%norm >= 0 )                                           &
         data%control%norm = euclidean_regularization
       data%form_regularization = .FALSE.
     END IF
     data%reverse_scale = .NOT. PRESENT( eval_SCALE )
     data%regularization_type = data%control%norm

!  set the power for the regularization

     IF ( control%power >= two ) THEN
       data%power = control%power
     ELSE
       SELECT CASE ( data%control%model )
       CASE ( first_order_model, diagonal_hessian_model,                       &
              gauss_newton_model, gauss_to_newton_model )
         data%power = two
       CASE DEFAULT
         data%power = three
       END SELECT
     END IF

!write(6,*) data%control%model, data%control%hessian_available

!  set specific controls for the sub-problem solvers

!write(6,*) ' ** subproblem scaling type ', data%regularization_type
!     data%regularization_type = data%control%norm
!write(6,*) ' ** subproblem scaling type ', data%regularization_type
!write(6,*) ' ** subproblem regularization type ', data%regularization_type
     data%control%GLRT_control%unitm                                           &
       = data%regularization_type == euclidean_regularization
     SELECT CASE ( data%regularization_type )
     CASE ( diagonal_hessian_regularization )
       data%control%PSLS_control%preconditioner = 1
     CASE ( band_regularization )
       data%control%PSLS_control%preconditioner = 2
     CASE ( reordered_band_regularization )
       data%control%PSLS_control%preconditioner = 3
     CASE ( schnabel_eskow_regularization )
       data%control%PSLS_control%preconditioner = 4
     CASE ( gmps_regularization )
       data%control%PSLS_control%preconditioner = 5
     CASE ( lin_more_regularization )
       data%control%PSLS_control%preconditioner = 6
     CASE ( mi28_regularization )
       data%control%PSLS_control%preconditioner = 7
     CASE ( munksgaard_regularization )
       data%control%PSLS_control%preconditioner = 8
     CASE ( expanding_band_regularization )
       data%control%PSLS_control%preconditioner = 9
     CASE DEFAULT
       data%control%PSLS_control%preconditioner = - 1
     END SELECT
     data%control%PSLS_control%new_structure = .TRUE.
     data%control%RQS_control%initial_multiplier = zero

!  check that the Jacobian is specified in a permitted format

     IF ( data%jacobian_available ) THEN
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input J%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%J%type )
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END SELECT
     END IF

!  create a file which the user may subsequently remove to cause
!  immediate termination of a run

     IF ( control%alive_unit > 0 ) THEN
      INQUIRE( FILE = control%alive_file, EXIST = alive )
      IF ( .NOT. alive ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'NEW' )
         REWIND control%alive_unit
         WRITE( control%alive_unit, "( ' GALAHAD rampages onwards ' )" )
         CLOSE( control%alive_unit )
       END IF
     END IF

!  allocate sufficient space to solve the problem

     array_name = 'nls: nlp%G'
     CALL SPACE_resize_array( nlp%n, nlp%G, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%C_current'
     CALL SPACE_resize_array( nlp%m, data%C_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%G_current'
     CALL SPACE_resize_array( nlp%n, data%G_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%S'
     CALL SPACE_resize_array( nlp%n, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%U'
     CALL SPACE_resize_array( MAX( nlp%n, nlp%m ), data%U, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%V'
     CALL SPACE_resize_array( MAX( nlp%n, nlp%m ), data%V, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%W'
     CALL SPACE_resize_array( nlp%n, data%W, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'nls: data%Y'
     CALL SPACE_resize_array( nlp%m, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( .NOT. data%monotone ) THEN
       array_name = 'nls: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  ensure that parameters are set correctly for the diagonal-Hessian model case

     IF ( data%control%subproblem_direct .AND.                                 &
          data%control%model == diagonal_hessian_model ) THEN
       data%H%n = nlp%n ; data%H%m = nlp%n ; data%H%ne = nlp%n
       CALL SMT_put( data%H%type, 'DIAGONAL', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: H%val'
       CALL SPACE_resize_array( nlp%n, data%H%val, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
       data%H%val = one
     END IF

!  provide space for the regularization

     IF ( data%regularization_type == diagonal_jtj_regularization ) THEN
       data%regularization%matrix%m = nlp%n
       data%regularization%matrix%n = nlp%n
       data%regularization%matrix%ne = nlp%n
       CALL SMT_put( data%regularization%matrix%type, 'DIAGONAL',              &
                     inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: data%regularization%matrix%val'
       CALL SPACE_resize_array( nlp%n, data%regularization%matrix%val,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  compute the number of nonzeros in J

     IF ( data%regularization_type > diagonal_jtj_regularization .OR.          &
          ( data%control%subproblem_direct .AND.                               &
            ( data%control%model == gauss_newton_model .OR.                    &
              data%control%model == newton_model ) ) ) THEN
       nlp%J%n = nlp%n ; nlp%J%m = nlp%m
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         nlp%J%ne = nlp%J%m * nlp%J%n
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%J%ne = nlp%J%ptr( nlp%m + 1 ) - 1
       END SELECT

!  an assembled Hessian approximation is required to compute the scaling matrix,
!  so provide J(transpose) = JT

       data%JT%n = nlp%m ; data%JT%m = nlp%n ; data%JT%ne = nlp%J%ne
       CALL SMT_put( data%JT%type, 'COORDINATE', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: data%JT%row'
       CALL SPACE_resize_array( data%JT%ne, data%JT%row, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%col'
       CALL SPACE_resize_array( data%JT%ne, data%JT%col, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%val'
       CALL SPACE_resize_array( data%JT%ne, data%JT%val, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  assign the row and column indices of JT

       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, nlp%m
           DO j = 1, nlp%n
             l = l + 1
             data%JT%row( l ) = j ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
             data%JT%row( l ) = nlp%J%col( l ) ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           data%JT%row( l ) = nlp%J%col( l )
           data%JT%col( l ) = nlp%J%row( l )
         END DO
       END SELECT
     END IF

     IF ( data%regularization_type > diagonal_jtj_regularization .OR.          &
          ( data%control%subproblem_direct .AND.                               &
            ( data%control%model == gauss_newton_model .OR.                    &
              data%control%model == newton_model ) ) ) THEN

!  record the sparsity pattern of J^T J in data%H

       data%control%BSC_control%new_a = 3
       data%control%BSC_control%extra_space_s = 0
       data%control%BSC_control%s_also_by_column                               &
          = data%map_h_to_jtj .OR. data%stabilised
       CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,            &
                      data%control%BSC_control, inform%BSC_inform )
       data%control%BSC_control%new_a = 1

!   if required, find a mapping for the entries of H(x,c) into the existing
!   structure in data%H for J^T J; the sparsity pattern of H(x,c) lies
!   within that of J^T J

       IF ( data%map_h_to_jtj ) THEN
         CALL NLS_set_map( data%H, nlp%H, data%IW, data%PTR, data%ROW,         &
                           data%ORDER, .TRUE.,                                 &
                           data%control%deallocate_error_fatal,                &
                           data%control%space_critical,  data%control%error,   &
                           data%H_map, inform%status, inform%alloc_status,     &
                           inform%bad_alloc )
         IF ( inform%status /= 0 ) GO TO 980
       END IF

!   if required, find a mapping for the entries of S into the existing
!   structure in data%H for J^T J; the sparsity pattern of S lies
!   within that of J^T J

       IF ( data%stabilised ) THEN
         CALL NLS_set_map( data%H, stabilisation%matrix, data%IW, data%PTR,    &
                           data%ROW, data%ORDER, .TRUE.,                       &
                           data%control%deallocate_error_fatal,                &
                           data%control%space_critical,  data%control%error,   &
                           data%S_map, inform%status, inform%alloc_status,     &
                           inform%bad_alloc )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
     END IF

!  provide workspace to allow for the stabilization term if necessary

     IF ( data%stabilised ) THEN
       array_name = 'nls: data%SX'
       CALL SPACE_resize_array( nlp%n, data%SX, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%SV'
       CALL SPACE_resize_array( nlp%n, data%SV, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  determine how much space was required to hold S

       IF ( data%regularization_type == euclidean_regularization ) THEN
         data%s_ne = nlp%n
       ELSE IF ( data%regularization_type /= user_regularization ) THEN
         SELECT CASE ( SMT_get( stabilisation%matrix%type ) )
         CASE ( 'DENSE' )
           IF ( MOD(  nlp%n, 2 ) == 0 ) THEN
             data%s_ne = ( nlp%n / 2 ) * ( nlp%n + 1 )
           ELSE
             data%s_ne = nlp%n * ( ( nlp%n + 1 ) / 2 )
           END IF
         CASE ( 'SPARSE_BY_ROWS' )
           data%s_ne = stabilisation%matrix%ptr( nlp%n + 1 ) - 1
         CASE ( 'COORDINATE' )
           data%s_ne = stabilisation%matrix%ne
         CASE DEFAULT
           data%s_ne = nlp%n
         END SELECT
       END IF
     END IF

!  re-entry without initialization

  20 CONTINUE
!    CALL CLOCK_time( data%clock_now )
!    write(6,*) ' 20 elapsed', data%clock_now - data%clock_start
     data%set_printw = control%out > 0 .AND. control%print_level >= 4
     IF ( data%set_printw )                                                    &
       WRITE( control%out, "( A, ' statement 20' )" ) prefix

!  control the output printing

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

     IF ( control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = control%print_gap
     END IF

     data%out = data%control%out
     data%print_level_glrt = data%control%GLRT_control%print_level
     data%print_level_rqs = data%control%RQS_control%print_level
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
     END IF

!  set the initial weight

     IF ( data%control%weight_update_strategy == weight_update_zero_reset ) THEN
       inform%weight = weight_zero
     ELSE
       inform%weight = control%initial_weight
     END IF
     data%step_accepted = .FALSE.
     data%poor_model = .FALSE.
     data%s_norm_successful = one
     data%minimum_weight = data%control%minimum_weight

! evaluate the residual function c(x) at the initial point

     IF ( data%reverse_c ) THEN
       data%branch = 30 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_C( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%C( : nlp%m ) )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_C'
         inform%status = GALAHAD_error_evaluation ; GO TO 900
       END IF
     END IF

!  return from reverse communication with the residual function value c(x)

  30 CONTINUE
!    CALL CLOCK_time( data%clock_now )
!    write(6,*) ' 30 elapsed', data%clock_now - data%clock_start
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 30' )" ) prefix
     inform%c_eval = inform%c_eval + 1
     IF ( data%w_eq_identity ) THEN
       data%Y( : nlp%m ) = nlp%C( : nlp%m )
       inform%norm_c = TWO_NORM( nlp%C( : nlp%m ) )
       inform%obj = half * inform%norm_c ** 2
     ELSE
!write(6,*) ' w ', W( : nlp%m )
       data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
       val = DOT_PRODUCT( data%Y( : nlp%m ), nlp%C( : nlp%m ) )
       inform%norm_c = SQRT( val )
       inform%obj = half * val
     END IF

!  account for the stabilization term if necessary

     IF ( data%stabilised ) THEN
       IF ( data%regularization_type == user_regularization ) THEN
         IF ( data%reverse_scale ) THEN
           data%V( : nlp%n ) = nlp%X( : nlp%n )
           data%branch = 40 ; inform%status = 8 ; RETURN
         ELSE
           CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,      &
                            data%SX( : nlp%n ), nlp%X( : nlp%n ) )
           IF ( data%eval_status /= 0 ) THEN
             inform%bad_eval = 'eval_SCALE'
             inform%status = GALAHAD_error_evaluation ; GO TO 900
           END IF
         END IF
       ELSE
         CALL mop_Ax( one, stabilisation%matrix,                               &
                      nlp%X( : nlp%n ), zero, data%SX( : nlp%n ),              &
                      symmetric = .TRUE., out = data%out,                      &
                      error = data%control%error, print_level = 0 )
       END IF
     END IF

!  return from reverse communication with the scaled vector u = S(x) x

  40 CONTINUE
     IF ( data%stabilised ) THEN
       IF ( data%regularization_type ==                                        &
              user_regularization .AND. data%reverse_scale )                   &
         data%SX( : nlp%n ) = data%U( : nlp%n )
       data%xtsx = DOT_PRODUCT( nlp%X( : nlp%n ), data%SX( : nlp%n ) )
       inform%obj = inform%obj +                                               &
         ( stabilisation%weight / stabilisation%power )                        &
            * data%xtsx ** ( stabilisation%power / two )
     END IF

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj
!write(6,*) ' objective is NaN? ', data%f_is_nan

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%bad_eval = 'NaN'
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  compute the residual stopping tolerance

     data%stop_c = MAX( MAX( control%stop_c_absolute, zero ),                  &
                        MAX( control%stop_c_relative, zero ) * inform%norm_c,  &
                        epsmch )

!  stop in the unlikely event that the initial residual is already small

     IF ( inform%norm_c <= data%stop_c ) THEN
       inform%status = GALAHAD_ok ; GO TO 910
     END IF

!  initialize the history of objective values

     data%f_ref = inform%obj
     IF ( .NOT. data%monotone ) THEN
        data%F_hist = data%f_ref ; data%D_hist = zero ; data%max_hist = 1
     END IF

!  ============================================================================
!  Start of main iteration
!  ============================================================================

 100 CONTINUE
!      CALL CLOCK_time( data%clock_now )
!      write(6,*) ' 100 elapsed', data%clock_now - data%clock_start
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 100' )" ) prefix

!  evaluate the Jacobian J(x) of c(x)

!write(6,*) ' data%poor_model ', data%poor_model
       IF ( .NOT. data%poor_model ) THEN
!write(6,*) ' data%jacobian_available ', data%jacobian_available
         IF ( data%jacobian_available ) THEN
           IF ( data%reverse_j ) THEN
             data%branch = 110 ; inform%status = 3 ; RETURN
           ELSE
             CALL eval_J( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%J%val )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_J'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF

!  otherwise evaluate the product g = J^T(x) W c(x)

         ELSE
           data%transpose = .TRUE.
           IF ( data%reverse_jprod ) THEN
             data%U( : nlp%n ) = zero ; data%V( : nlp%m ) = data%Y( : nlp%m )
!write(6,*) ' c ',  nlp%C( : nlp%m )
!write(6,*) ' v ',  data%V( : nlp%m )
             data%branch = 110 ; inform%status = 5 ; RETURN
           ELSE
             nlp%G( : nlp%n ) = zero
             CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%transpose, nlp%G( : nlp%n ),                &
                              data%Y( : nlp%m ), .FALSE. )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_JPROD'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         END IF
       END IF

!  return from reverse communication with the Jacobian-residual product

 110   CONTINUE

!      CALL CLOCK_time( data%clock_now )
!      write(6,*) ' 110 elapsed', data%clock_now - data%clock_start
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 110' )" ) prefix

       IF ( .NOT. data%poor_model ) THEN
         inform%j_eval = inform%j_eval + 1

!  compute the product g = J^T(x) W c(x) from J(x) if necessary

         IF ( data%jacobian_available ) THEN
           CALL mop_Ax( one, nlp%J, data%Y( : nlp%m ), zero, nlp%G( : nlp%n ), &
                        out = data%out, error = data%control%error,            &
                        print_level = 0, transpose = .TRUE. )
         ELSE
           IF ( data%reverse_jprod ) nlp%G( : nlp%n ) = data%U( : nlp%n )
         END IF
!write(6,*) ' g ',  nlp%G( : nlp%n )

!  account for the gradient of the stabilization term if necessary

         IF ( data%stabilised ) THEN
           IF ( stabilisation%power == two ) THEN
             nlp%G( : nlp%n ) = nlp%G( : nlp%n ) +                             &
               stabilisation%weight * data%SX( : nlp%n )
           ELSE
             nlp%G( : nlp%n ) = nlp%G( : nlp%n ) + stabilisation%weight        &
               * ( data%xtsx ** ( ( stabilisation%power - two ) / two ) )      &
               *  data%SX( : nlp%n )
           END IF
         END IF
!write(6,*) ' x ',  nlp%X( : nlp%n )
!write(6,*) ' g ',  nlp%G( : nlp%n )

!  compute the gradient of ||g(x)||

         data%g_norm = TWO_NORM( nlp%G( : nlp%n ) )
         IF ( inform%norm_c > zero ) THEN
           inform%norm_g = data%g_norm / inform%norm_c
         ELSE
           inform%norm_g = zero
         END IF
         data%new_point = .TRUE.

!  deal with NaN gradient values
!  -----------------------------

         data%g_is_nan = inform%norm_g /= inform%norm_g
         IF ( data%g_is_nan ) THEN
           IF ( inform%iter > 0 ) THEN
             data%poor_model = .FALSE.
             data%accept = 'r'
             nlp%X( : nlp%n ) = data%X_current( : nlp%n )
             nlp%C( : nlp%m ) = data%C_current( : nlp%m )
             IF ( data%stabilised ) THEN
               data%SX( : nlp%n ) = data%SV( : nlp%n )
               data%xtsx = data%xtsx_current
             END IF

!  control printing for the NaN case

             IF ( inform%iter >= data%start_print .AND.                        &
                  inform%iter < data%stop_print .AND.                          &
                  MOD( inform%iter + 1 - data%start_print, data%print_gap )    &
                    == 0 ) THEN
               data%printi = data%set_printi ; data%printt = data%set_printt
               data%printm = data%set_printm ; data%printw = data%set_printw
               data%printd = data%set_printd
               data%print_level = data%control%print_level
               data%control%GLRT_control%print_level = data%print_level_glrt
               data%control%RQS_control%print_level = data%print_level_rqs
             ELSE
               data%printi = .FALSE. ; data%printt = .FALSE.
               data%printm = .FALSE. ; data%printw = .FALSE.
               data%printd = .FALSE.
               data%print_level = 0
               data%control%GLRT_control%print_level = 0
               data%control%RQS_control%print_level = 0
             END IF
             data%print_iteration_header = data%print_level > 1 .OR.           &
               ( data%control%GLRT_control%print_level > 0 .AND. .NOT.         &
                 data%control%subproblem_direct ) .OR.                         &
               ( data%control%RQS_control%print_level > 0 .AND.                &
                 data%control%subproblem_direct )

!  print one-line summary

             IF ( data%printi ) THEN
                IF ( data%print_iteration_header .OR.                          &
                     data%print_1st_header ) THEN
                 WRITE( data%out, 2090 ) prefix
                 IF ( data%control%subproblem_direct ) THEN
                   IF ( data%control%print_obj ) THEN
                     WRITE( data%out, 2170 ) prefix
                   ELSE
                     WRITE( data%out, 2160 ) prefix
                   END IF
                 ELSE
                   IF ( data%control%print_obj ) THEN
                     WRITE( data%out, 2110 ) prefix
                   ELSE
                     WRITE( data%out, 2100 ) prefix
                   END IF
                 END IF
               END IF
               data%print_1st_header = .FALSE.
               char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
               IF ( data%control%subproblem_direct ) THEN
                 char_facts = ADJUSTR( STRING_integer_6( data%total_facts ) )
                 WRITE( data%out,  "( A, A6, 1X, 3A1, ES11.4, '    NaN    ',   &
                &  ES9.1,  2ES8.1, 1X, A6, F8.2 )" )                           &
                    prefix, char_iter, data%accept, data%negcur, data%hard,    &
                    inform%norm_c, data%ratio, data%old_weight, data%s_norm,   &
                    char_facts, data%clock_now
               ELSE
                 char_sit =                                                    &
                    ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter ) )
                 char_sit2 =                                                   &
                    ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter_pass2 ) )
                 WRITE( data%out, "( A, A6, 1X, 3A1, ES11.4, '    NaN    ',    &
                &  ES9.1, 2ES8.1, 1X, 2A6, F8.2 )" ) prefix,                   &
                    char_iter, data%accept, data%negcur, data%perturb,         &
                    inform%norm_c, data%ratio, data%old_weight, data%s_norm,   &
                    char_sit, char_sit2, data%clock_now
               END IF
             END IF
             inform%obj = data%obj_current
             inform%norm_c = data%norm_c_current

!  check to see if we are still "alive"

             IF ( data%control%alive_unit > 0 ) THEN
               INQUIRE( FILE = data%control%alive_file, EXIST = alive )
               IF ( .NOT. alive ) THEN
                 inform%status = GALAHAD_error_alive
                 RETURN
               END IF
             END IF

!  check to see if the iteration limit has been exceeded

             inform%iter = inform%iter + 1
             IF ( inform%iter > data%control%maxit .AND.                       &
                  data%step_accepted ) THEN
               inform%status = GALAHAD_error_max_iterations ; GO TO 900
             END IF

!  increase the regularization weight and try again

             inform%weight = data%control%weight_increase * data%old_weight
             GO TO 100
           ELSE
             IF ( data%printi ) WRITE( data%out,                               &
                "( A, ' initial gradient value is a NaN' )" ) prefix
             inform%bad_eval = 'eval_JPROD'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
         END IF

!  reset the initial weight to ||g|| if no sensible value is given

         IF ( inform%iter == 0 ) THEN
           IF ( data%control%initial_weight <= zero )                          &
              inform%weight = one / inform%norm_g

!  compute the gradient stopping tolerance

           data%stop_g = MAX( MAX( control%stop_g_absolute, zero ),            &
             MAX( control%stop_g_relative, zero ) * inform%norm_g, epsmch )

           IF ( data%printi ) THEN
             IF ( data%stabilised ) THEN
               WRITE( data%out, "( A, '  Problem: ', A, ' (n = ', I0,          &
            &  ', m = ', I0, ', weight = ', ES8.2, ', p = ', ES8.2, ')', /, A, &
            &   '  NLS stopping tolerances (c,J''c/c) =', 2ES9.2, / )" )       &
               prefix, TRIM( nlp%pname ), nlp%n, nlp%m,                        &
               MAX( stabilisation%weight, zero ), stabilisation%power,         &
               prefix, data%stop_c, data%stop_g
             ELSE
               WRITE( data%out, "( A, '  Problem: ', A, ' (n = ', I0,          &
            &  ', m = ',I0, ')', /, A,                                         &
            &   '  NLS stopping tolerances (c,J''c/c) =', 2ES9.2, / )" )       &
               prefix, TRIM( nlp%pname ), nlp%n, nlp%m,                        &
               prefix, data%stop_c, data%stop_g
             END IF
           END IF
         END IF
       END IF

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%GLRT_control%print_level = data%print_level_glrt
         data%control%RQS_control%print_level = data%print_level_rqs
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%GLRT_control%print_level = 0
         data%control%RQS_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         ( data%control%GLRT_control%print_level > 0 .AND. .NOT.               &
           data%control%subproblem_direct ) .OR.                               &
         ( data%control%RQS_control%print_level > 0 .AND.                      &
           data%control%subproblem_direct )

!  print one-line summary

       IF ( data%printi ) THEN
          IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2090 ) prefix
           IF ( data%control%subproblem_direct ) THEN
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2170 ) prefix
             ELSE
               WRITE( data%out, 2160 ) prefix
             END IF
           ELSE
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2110 ) prefix
             ELSE
               WRITE( data%out, 2100 ) prefix
             END IF
           END IF
         END IF

         data%print_1st_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 0 ) THEN
           IF ( data%control%subproblem_direct ) THEN
             char_facts =                                                      &
               ADJUSTR( STRING_integer_6( data%total_facts ) )
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2120 ) prefix, char_iter, data%accept,         &
                  data%negcur, data%hard, inform%obj,                          &
                  inform%norm_g, data%ratio, data%old_weight,                  &
                  data%s_norm, char_facts, data%clock_now
             ELSE
               WRITE( data%out, 2120 ) prefix, char_iter, data%accept,         &
                  data%negcur, data%hard, inform%norm_c,                       &
                  inform%norm_g, data%ratio, data%old_weight,                  &
                  data%s_norm, char_facts, data%clock_now
             END IF
           ELSE
             char_sit = ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter ) )
             char_sit2 =                                                       &
                ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter_pass2 ) )
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2130 ) prefix, char_iter, data%accept,         &
                  data%negcur, data%perturb, inform%obj,                       &
                  inform%norm_g, data%ratio, data%old_weight, data%s_norm,     &
                  char_sit, char_sit2, data%clock_now
             ELSE
               WRITE( data%out, 2130 ) prefix, char_iter, data%accept,         &
                  data%negcur, data%perturb, inform%norm_c,                    &
                  inform%norm_g, data%ratio, data%old_weight, data%s_norm,     &
                  char_sit, char_sit2, data%clock_now
             END IF
           END IF
         ELSE
           IF ( data%control%subproblem_direct ) THEN
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2140 ) prefix,                                 &
                  char_iter, inform%obj, inform%norm_g
             ELSE
               WRITE( data%out, 2140 ) prefix,                                 &
                  char_iter, inform%norm_c, inform%norm_g
             END IF
           ELSE
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2150 ) prefix,                                 &
                  char_iter, inform%obj, inform%norm_g
             ELSE
               WRITE( data%out, 2150 ) prefix,                                 &
                  char_iter, inform%norm_c, inform%norm_g
             END IF
           END IF
         END IF
       END IF

!  ============================================================================
!  1. Test for convergence
!  ============================================================================

!  stop if the gradient is small enough

       IF ( inform%norm_c <= data%stop_c .OR.                                  &
            inform%norm_g <= data%stop_g ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  stop if the gradient is swamped by the Hessian

!      IF ( data%control%hessian_available .AND. inform%iter > 0 ) THEN
!        IF ( inform%norm_g <= MIN( one,                                       &
!             MAXVAL( ABS( data%H%val( : data%H%ne ) ) ) * epsmch ) ) THEN
!         write(6,*) ' stopping as g is too ill-conditioned to make ',         &
!        &   'further progress!'
!         write(6,*) inform%norm_g,                                            &
!           MAXVAL( ABS( nlp%H%val( : nlp%H%ne ) ) ) * epsmch
!         inform%status = GALAHAD_error_ill_conditioned ; GO TO 900
!         END IF
!       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive
           RETURN
         END IF
       END IF

!  check to see if the iteration limit has been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit .AND. data%step_accepted ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF
!      write(6,*) inform%norm_c, data%stop_c, inform%norm_g, data%stop_g

!  check to see if the Gauss-Newton model should be exchanged for a Newton one

       IF ( data%gauss_to_newton_model ) THEN
         IF ( inform%norm_g < data%control%switch_to_newton ) THEN
           IF ( data%control%model == gauss_newton_model ) THEN
             IF ( control%power < two ) data%power = three
             data%control%model = newton_model
             data%control%BSC_control%new_a = 2
             data%control%BSC_control%extra_space_s = 0
             data%BSC_control_tensor_model%new_a = 2
             data%BSC_control_tensor_model%extra_space_s = 0
             data%form_regularization                                          &
               = data%jacobian_available .AND. data%hessian_available
             data%print_1st_header = .TRUE.
             IF ( data%printi ) WRITE( data%out,                               &
               "( /, A, '  ... switching to Newton model', / )" ) prefix
           END IF
         END IF
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%c_eval, prefix, inform%j_eval,&
           prefix, inform%h_eval, prefix, inform%iter, prefix, inform%cg_iter, &
           prefix, inform%obj, prefix, inform%norm_g
         WRITE ( data%out, 2010 ) prefix
!        l = nlp%n
         l = 2
         DO j = 1, 2
            IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, nlp%n )
            ELSE
               IF ( ic < nlp%n - l ) WRITE( data%out, 2050 ) prefix
               ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
            END IF
            IF ( ALLOCATED( nlp%vnames ) ) THEN
              DO i = ir, ic
                 WRITE( data%out, 2020 ) prefix, nlp%vnames( i ), nlp%X( i ),  &
                  nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2030 ) prefix, i, nlp%X( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

!  recompute the scaled Hessian if it has changed

       data%perturb = ' '
       IF ( data%new_point ) THEN
         data%nskip_prec = data%nskip_prec + 1
         data%got_h = .FALSE.
         data%hessian_computed = data%hessian_available .AND.                  &
           data%control%model == newton_model .AND.                            &
           data%nskip_prec > nskip_prec_max

!  form the scaled Hessian or a scaling matrix based on the scaled Hessian

         IF ( data%hessian_computed ) THEN
           IF ( data%reverse_h ) THEN
             data%branch = 120 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          data%Y( : nlp%m ), userdata,                         &
                          nlp%H%val( : nlp%H%ne ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_H'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         END IF
       END IF

!  return from reverse communication with the scaled Hessian

 120   CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 120' )" ) prefix

!  the Hessian has changed

       IF ( data%new_point ) THEN
         IF ( data%hessian_computed ) THEN
           inform%h_eval = inform%h_eval + 1  ; data%got_h = .TRUE.

!  debug printing for H

           IF ( data%printd ) THEN
             WRITE( data%out, "( A, ' Scaled Hessian' )" ) prefix
             DO l = 1, nlp%H%ne
               WRITE( data%out, "( A, 2I7, ES24.16 )" ) prefix,                &
                 nlp%H%row( l ), nlp%H%col( l ), nlp%H%val( l )
             END DO
           END IF
         END IF

!  if required, form the Hessian to provide a scaling matrix

         IF ( data%regularization_type > diagonal_jtj_regularization .OR.      &
              ( data%control%subproblem_direct .AND.                           &
                ( data%control%model == gauss_newton_model .OR.                &
                  data%control%model == newton_model ) ) ) THEN

!  form the transpose of the Jacobian

           data%JT%val( : data%JT%ne ) = nlp%J%val( : data%JT%ne )

!  form J^T W J in H, ensuring that there is sufficent additional
!  space to store the stabilisation Hessian if used

           IF ( data%w_eq_identity ) THEN
             CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,      &
                            data%control%BSC_control, inform%BSC_inform )
           ELSE
             CALL BSC_form( nlp%n, nlp%m, data%JT, data%H, data%BSC_data,      &
                            data%control%BSC_control, inform%BSC_inform,       &
                            D = W( : nlp%m ) )
           END IF

!  append the values of H(x,Wc) if they are required

           IF ( data%hessian_computed ) THEN
             DO l = 1, nlp%H%ne
               j =  data%H_map( l )
               data%H%val( j ) = data%H%val( j ) + nlp%H%val( l )
             END DO
           END IF

!  append the values of the stablisation Hessian if they are required

           IF ( data%stabilised ) THEN
             SELECT CASE ( SMT_get( stabilisation%matrix%type ) )
             CASE ( 'IDENTITY' )
               IF ( stabilisation%power == two ) THEN
                 val = stabilisation%weight
               ELSE
                 val = stabilisation%weight *                                  &
                   ( data%xtsx ** ( ( stabilisation%power - two ) / two ) )
               END IF
               DO l = 1, stabilisation%matrix%ne
                 j = data%S_map( l )
                 data%H%val( j ) = data%H%val( j ) + val
               END DO
             CASE ( 'SCALED_IDENTITY' )
               IF ( stabilisation%power == two ) THEN
                 val = stabilisation%weight * stabilisation%matrix%val( 1 )
               ELSE
                 val = stabilisation%weight *                                  &
                 ( data%xtsx ** ( ( stabilisation%power - two ) / two ) )      &
                   * stabilisation%matrix%val( 1 )
               END IF
               DO l = 1, stabilisation%matrix%ne
                 j = data%S_map( l )
                 data%H%val( j ) = data%H%val( j ) + val
               END DO
             CASE DEFAULT
               IF ( stabilisation%power == two ) THEN
                 val = stabilisation%weight
               ELSE
                 val = stabilisation%weight *                                  &
                 ( data%xtsx ** ( ( stabilisation%power - two ) / two ) )
               END IF
               DO l = 1, stabilisation%matrix%ne
                 j = data%S_map( l )
                 data%H%val( j ) = data%H%val( j ) +                           &
                   val * stabilisation%matrix%val( l )
               END DO
             END SELECT
           END IF
         END IF

!        write(6,"( 5ES12.4 )" ) ( nlp%G( l ), l = 1, nlp%n )
!        write(6,"( ( 2I8, ES12.4 ) )" ) ( data%H%row( l ), data%H%col( l ),   &
!                                          data%H%val( l ), l = 1,  data%H%ne )

!  if the Hessian has changed, recompute the scaling matrix

!  recompute the scaling matrix

!  the search-direction subproblem will be solved directly

         IF ( data%control%subproblem_direct ) THEN

!  build the scaling matrix

           IF ( data%regularization_type > diagonal_jtj_regularization .AND.   &
                data%form_regularization ) THEN
             IF ( data%printt ) WRITE( data%out,                               &
                   "( A, ' Computing scaling matrix' )" ) prefix
             CALL PSLS_build( data%H, data%regularization%matrix,              &
               data%PSLS_data, data%control%PSLS_control, inform%PSLS_inform )

!  check for error returns

             IF ( inform%PSLS_inform%status /= 0 ) THEN
               inform%status = inform%PSLS_inform%status ; GO TO 900
             END IF

             data%non_trivial_regularization  = .TRUE.
             IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'

!  build the scaling matrix as the diagonal matrix whose entries are
!  the squares of the two-norms of the columns of J

           ELSE IF ( data%regularization_type == diagonal_jtj_regularization   &
                     .AND. data%form_regularization ) THEN
             CALL mop_column_2_norms( nlp%J,                                   &
                    data%regularization%matrix%val( : nlp%n ), W = W )
             data%regularization%matrix%val( : nlp%n )                         &
               = data%regularization%matrix%val( : nlp%n ) ** 2
             data%non_trivial_regularization = .TRUE.
!            write(6,"( ' scaling ', /, ( 5ES12.4 ) )" )                       &
!              data%regularization%matrix%val( : nlp%n )
           ELSE
             data%non_trivial_regularization = .FALSE.
           END IF
           data%control%PSLS_control%new_structure = .FALSE.

!  the search-direction subproblem will be solved iteratively

         ELSE
           IF ( data%nskip_prec > nskip_prec_max ) THEN
             IF ( data%regularization_type > diagonal_jtj_regularization .AND. &
                  data%form_regularization ) THEN

!  form and factorize the scaling matrix obtained from H

               IF ( data%printt ) WRITE( data%out,                             &
                     "( A, ' Computing scaling matrix' )" ) prefix
               CALL PSLS_form_and_factorize( data%H, data%PSLS_data,           &
                 data%control%PSLS_control, inform%PSLS_inform )

!  check for error returns

               IF ( inform%PSLS_inform%status /= 0 ) THEN
                 inform%status = inform%PSLS_inform%status ; GO TO 900
               END IF
               IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'
               data%control%PSLS_control%new_structure = .FALSE.

!  build the scaling matrix as the diagonal matrix whose entries are
!  the squares of the two-norms of the columns of J

             ELSE IF ( data%regularization_type ==                             &
               diagonal_jtj_regularization .AND. data%form_regularization ) THEN
               CALL mop_column_2_norms( nlp%J,                                 &
                      data%regularization%matrix%val( : nlp%n ), W = W )
               data%regularization%matrix%val( : nlp%n )                       &
                 = data%regularization%matrix%val( : nlp%n ) ** 2
             END IF
             data%nskip_prec = 0
           END IF
         END IF
       END IF

   190 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 190' )" ) prefix
!      write(6,*) ' f ', inform%obj

!  ============================================================================
!  2. Calculate the search direction, s
!  ============================================================================
!  .............................................................................
!                        Linear or quadratic model
!  .............................................................................

!  2a. Direct solution
!  -------------------

!      write(6,*) ' direct? ', data%control%subproblem_direct
      IF ( data%control%subproblem_direct ) THEN

!  estimate Lagrange multipler for the next regularization subproblem

         IF ( inform%iter > 1 ) THEN

!  only the weight for the next problem differs from the current one

!          write(6,*) ' poor model ', data%poor_model
           IF ( data%poor_model ) THEN

!  if there is a history of points with smaller norms, record them

             IF ( inform%RQS_inform%len_history > 0 ) THEN
               data%len_history = inform%RQS_inform%len_history
               data%history( : data%len_history )                              &
                 = inform%RQS_inform%history( : data%len_history )
             ELSE
               data%len_history = 0
             END IF

!  set the lower bound and estimate of the next multiplier to the current
!  values, as Newton will converge rapidly from here

             data%control%RQS_control%lower = inform%RQS_inform%multiplier
             data%control%RQS_control%initial_multiplier =                     &
               data%control%RQS_control%lower
             data%control%RQS_control%use_initial_multiplier = .TRUE.

!  if the hard case was possible, slightly perturb the multiplier

             IF ( inform%RQS_inform%pole > zero )                              &
               data%control%RQS_control%initial_multiplier =                   &
                 data%control%RQS_control%initial_multiplier                   &
                   + MAX( inform%RQS_inform%pole, one ) * epsmch ** half

!  look through the history to see if a better starting value is available

             DO i = data%len_history, 1, - 1
               IF ( data%history( i )%x_norm > inform%weight ) THEN
                 data%control%RQS_control%initial_multiplier =                 &
                   data%history( i )%lambda
               ELSE
                 EXIT
               END IF
             END DO
             data%control%RQS_control%initialize_approx_eigenvector = .FALSE.
!            data%control%RQS_control%initialize_approx_eigenvector = .TRUE.

!  the next problem is likley different - try to guess a good initial
!  value for the next multiplier

           ELSE
             data%control%RQS_control%lower = zero
             data%control%RQS_control%use_initial_multiplier = .TRUE.
             IF ( inform%RQS_inform%multiplier == zero ) THEN
               data%control%RQS_control%initial_multiplier = zero
             ELSE
               data%control%RQS_control%initial_multiplier =                   &
                 inform%RQS_inform%multiplier *                                &
                   ( data%old_weight / inform%weight ) +                       &
                 inform%RQS_inform%pole *                                      &
                 ( one - ( data%old_weight / inform%weight ) )
               IF ( inform%RQS_inform%pole > zero )                            &
                 data%control%RQS_control%initial_multiplier =                 &
                   data%control%RQS_control%initial_multiplier                 &
                     + MAX( inform%RQS_inform%pole, one ) * epsmch ** half
             END IF
!            data%control%RQS_control%initialize_approx_eigenvector = .TRUE.
           END IF
         END IF

!  refactorize the Hessian if it has changed

         IF ( data%new_point ) THEN
           IF ( data%nskip_prec > nskip_prec_max ) THEN
             IF ( inform%iter <= 1 )THEN
               data%control%RQS_control%new_h = 2
             ELSE
               data%control%RQS_control%new_m = 1
               data%control%RQS_control%new_h = 1
             END IF
             data%nskip_prec = 0
           ELSE
             data%control%RQS_control%new_m = 0
             data%control%RQS_control%new_h = 0
           END IF
         END IF

!  Solve the regularization subproblem
!  ...................................

  200    CONTINUE
         data%model = zero
         facts_this_solve = inform%RQS_inform%factorizations
         IF ( data%printw ) WRITE( data%out, "( A, ' enter rqs_solve')" ) prefix
!write(6,*) 'x', nlp%X( : nlp%n ), inform%weight
!write(6,*) 'f', data%model
!write(6,*) 'g', nlp%G( : nlp%n )
!write(6,*) 'h', data%H%val( : nlp%n )
!write(6,*) 'power', data%power
         IF ( data%non_trivial_regularization ) THEN
           CALL RQS_solve( nlp%n, data%power, inform%weight, data%model,       &
                           nlp%G, data%H, data%S, data%RQS_data,               &
                           data%control%RQS_control, inform%RQS_inform,        &
                           M = data%regularization%matrix )
         ELSE
           CALL RQS_solve( nlp%n, data%power, inform%weight, data%model,       &
                           nlp%G, data%H, data%S, data%RQS_data,               &
                           data%control%RQS_control, inform%RQS_inform )
         END IF
!write(6,*) 's', data%S( : nlp%n )
!        write(6,*) ' ||s|| ', TWO_NORM( data%S( : nlp%n ) )
         IF ( data%printw ) WRITE( data%out, "( A, ' exit rqs_solve')" ) prefix

!  check for successful convergence

         IF ( inform%RQS_inform%status == GALAHAD_error_ill_conditioned ) THEN
           inform%weight = inform%weight * 10.0_wp
           GO TO 200
         ELSE IF ( inform%RQS_inform%status < 0 ) THEN
           IF ( data%printt ) WRITE( data%out, "( /,                           &
          &    A, ' Error return from RQS, status = ', I0 )" ) prefix,         &
             inform%RQS_inform%status
           inform%status = inform%RQS_inform%status
           GO TO 900
         END IF
         data%model = inform%RQS_inform%obj
!write(6,*) ' obj, reg',inform%RQS_inform%obj, inform%RQS_inform%obj_regularized

!        CALL mop_Ax( one, data%H, data%S( : nlp%n ), zero, &
!                     data%U( : nlp%n ), data%out, data%control%error, &
!                     0, symmetric = .TRUE. )
!        write(6,*) ' model est, calc ', &
!         data%model, DOT_PRODUCT( nlp%G( : nlp%n ), data%S( : nlp%n ) ) + &
!         half * DOT_PRODUCT( data%U( : nlp%n ), data%S( : nlp%n ) )
!        write(6,*) ' f_old, new ', inform%obj, data%f_trial
!        write(6,*) ' ||x+s|| ', TWO_NORM( nlp%X( : nlp%n ) + data%S( : nlp%n ))
!        write(6,"(A, 3ES12.4 )" ) ' f, model, s = ', inform%obj, &
!        inform%obj + data%model, TWO_NORM( data%S( : nlp%n ) )

         IF ( inform%RQS_inform%hard_case ) data%hard = 'h'
         facts_this_solve = inform%RQS_inform%factorizations - facts_this_solve
         inform%factorization_average =                                        &
           inform%RQS_inform%factorizations / inform%iter
         inform%factorization_max =                                            &
           MAX( inform%factorization_max, facts_this_solve )
         inform%max_entries_factors = MAX( inform%max_entries_factors,         &
                                         inform%RQS_inform%max_entries_factors )
         data%s_norm = inform%RQS_inform%x_norm

         data%total_facts = inform%RQS_inform%factorizations
         IF ( inform%RQS_inform%pole > zero ) THEN
           data%negcur = 'n'
         ELSE
           data%negcur = ' '
         END IF

         IF ( inform%RQS_inform%hard_case ) THEN
           data%hard = 'h'
         ELSE
           data%hard = ' '
         END IF

!write(6,*) ' go to 300 '
         GO TO 300
       END IF

!  2b. Iterative solution
!  ----------------------

       data%control%GLRT_control%stop_relative                                 &
         = MIN( data%control%GLRT_control%stop_relative, inform%norm_g ** 0.1 )

!      data%model = zero
       data%model = inform%obj
       data%control%GLRT_control%f_0 = inform%obj
       data%S( : nlp%n ) = zero
       data%G_current( : nlp%n ) = nlp%G( : nlp%n )
       IF ( data%new_point ) THEN
         inform%GLRT_inform%status = 1
       ELSE
!        inform%GLRT_inform%status = 6
         inform%GLRT_inform%status = 1
       END IF
!      data%control%GLRT_control%print_level = 1

!  Start of the generalized Lanczos iteration
!  ..........................................

  210  CONTINUE

!  perform a generalized Lanczos iteration

         IF ( data%printw ) WRITE( data%out, "( A, ' statement 210, GLRT_',    &
        &  'inform%status = ', I0 )" ) prefix, inform%GLRT_inform%status
         CALL GLRT_solve( nlp%n, data%power, inform%weight, data%S,            &
                          data%G_current, data%V, data%GLRT_data,              &
                          data%control%GLRT_control, inform%GLRT_inform )
         IF ( data%printw ) WRITE( data%out, "( A, ' statement > 210, GLRT_',  &
        &  'inform%status = ', I0 )" ) prefix, inform%GLRT_inform%status


!  compute sv = S(x) v for the stabilization term if necessary

       IF ( inform%GLRT_inform%status == 3 .AND. data%stabilised ) THEN
         IF ( data%regularization_type == user_regularization ) THEN
           IF ( data%reverse_scale ) THEN
             data%branch = 220 ; inform%status = 8 ; RETURN
           ELSE
             CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%SV( : nlp%n ), data%V( : nlp%n ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_SCALE'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         ELSE
           CALL mop_Ax( one, stabilisation%matrix,                             &
                        data%V( : nlp%n ), zero, data%SV( : nlp%n ),           &
                        symmetric = .TRUE., out = data%out,                    &
                        error = data%control%error, print_level = 0 )
         END IF
       END IF

!  return from reverse communication with the scaled vector u = S(x) x

  220  CONTINUE
       IF ( inform%GLRT_inform%status == 3 .AND. data%stabilised .AND.         &
              data%regularization_type == user_regularization .AND.            &
              data%reverse_scale ) data%SV( : nlp%n ) = data%U( : nlp%n )

!  branch to perform required computation before re-entering GLRT

         SELECT CASE( inform%GLRT_inform%status )

!  form the preconditioned gradient

         CASE ( 2 )

!  use the factors obtained from PSLS

           IF ( data%regularization_type > diagonal_jtj_regularization ) THEN
             CALL PSLS_solve( data%V, data%PSLS_data,                          &
                              data%control%PSLS_control, inform%PSLS_inform )

!  use the column scaling factors from J

           ELSE IF ( data%regularization_type ==                               &
                     diagonal_jtj_regularization ) THEN
             data%V( : nlp%n )                                                 &
               = data%V( : nlp%n ) / data%regularization%matrix%val( : nlp%n )

!  apply the user's scaling matrix

           ELSE IF ( data%regularization_type == user_regularization ) THEN
             IF ( data%reverse_scale ) THEN
               data%branch = 250 ; inform%status = 8 ; RETURN
             ELSE
               CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,  &
                                data%U( : nlp%n ), data%V( : nlp%n ) )
               IF ( data%eval_status /= 0 ) THEN
                 inform%bad_eval = 'eval_SCALE'
                 inform%status = GALAHAD_error_evaluation ; GO TO 900
               END IF
               data%V( : nlp%n ) = data%U( : nlp%n )
             END IF
           END IF

!  form the Hessian-vector product v <- u = H v

         CASE ( 3 )
!write(6,*) ' model = ', data%control%model
           SELECT CASE( data%control%model )

!  linear model

           CASE ( first_order_model )
             data%V( : nlp%n ) = zero

!  quadratic model with diagonal Hessian

           CASE ( diagonal_hessian_model )
             IF ( .NOT. data%w_eq_identity )                                   &
               data%V( : nlp%n ) = W( : nlp%n ) * data%V( : nlp%n )

!  quadratic model with true Hessian

           CASE ( newton_model, gauss_newton_model )
             data%W( : nlp%n ) = data%V( : nlp%n )

!  if the Jacobian has been calculated, form the product v <- J v directly

!write(6,*) ' available ', data%jacobian_available
!write(6,*) ' w  ', data%W
             IF ( data%jacobian_available ) THEN
               CALL mop_Ax( one, nlp%J, data%W( : nlp%n ), zero,               &
                            data%V( : nlp%m ), out = data%out,                 &
                            error = data%control%error, print_level = 0,       &
                            transpose = .FALSE. )

!  if the Jacobian is unavailable, obtain a matrix-free product

             ELSE
               data%transpose = .FALSE.
               IF ( data%reverse_jprod ) THEN
!write(6,*) ' via reverse'
                 data%V( : nlp%n ) = data%W( : nlp%n )
                 data%U( : nlp%m ) = zero
                 data%branch = 230 ; inform%status = 5 ; RETURN
               ELSE
!write(6,*) ' via jprod'
                 data%V( : nlp%m ) = zero
                 CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),          &
                                  userdata, data%transpose,                    &
                                  data%V( : nlp%m ), data%W( : nlp%n ),        &
                                  got_j = data%got_j )
!write(6,*) ' trans  ', data%transpose
!write(6,*) ' w  ', data%W
!write(6,*) ' v  ', data%V
                 IF ( data%eval_status /= 0 ) THEN
                   inform%bad_eval = 'eval_JPROD'
                   inform%status = GALAHAD_error_evaluation ; GO TO 900
                 END IF
               END IF
             END IF

           END SELECT

!  restore the gradient

         CASE ( 4 )
           data%G_current( : nlp%n ) = nlp%G( : nlp%n )

!  successful return

         CASE ( GALAHAD_ok, GALAHAD_warning_on_boundary,                       &
                GALAHAD_error_max_iterations )
           data%model = inform%GLRT_inform%obj
           GO TO 260

!  error returns

         CASE DEFAULT
           IF ( data%printt ) WRITE( data%out, "( /,                           &
           &  A, ' Error return from GLRT, status = ', I0 )" ) prefix,         &
             inform%GLRT_inform%status
           inform%status = inform%GLRT_inform%status
           GO TO 900
         END SELECT

!  return from reverse communication with the Jacobian-vector product

   230   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 230, GLRT_',    &
        &  'inform%status = ', I0 )" ) prefix, inform%GLRT_inform%status
         IF ( inform%GLRT_inform%status == 3 .AND.                             &
              ( data%control%model == newton_model .OR.                        &
                data%control%model == gauss_newton_model ) ) THEN

!  if the Jacobian has been calculated, form the product u = J^T W v directly

!write(6,*) ' v  ', data%V
           IF ( data%jacobian_available ) THEN
             IF ( data%w_eq_identity ) THEN
               CALL mop_Ax( one, nlp%J, data%V( : nlp%m ), zero,               &
                            data%U( : nlp%n ), out = data%out,                 &
                            error = data%control%error, print_level = 0,       &
                            transpose = .TRUE. )
             ELSE
               CALL mop_Ax( one, nlp%J, W( : nlp%m ) * data%V( : nlp%m ),      &
                            zero, data%U( : nlp%n ), out = data%out,           &
                            error = data%control%error, print_level = 0,       &
                            transpose = .TRUE. )
             END IF

!  if the Jacobian is unavailable, obtain a matrix-free product

           ELSE
             data%transpose = .TRUE.
             IF ( data%reverse_jprod ) THEN
               IF ( data%w_eq_identity ) THEN
                  data%V( : nlp%m ) = data%U( : nlp%m )
               ELSE
                  data%V( : nlp%m ) = W( : nlp%m ) * data%U( : nlp%m )
               END IF
!write(6,*) ' v  ', data%V
               data%U( : nlp%n ) = zero
               data%branch = 240 ; inform%status = 5 ; RETURN
             ELSE
               data%U( : nlp%n ) = zero
               IF ( data%w_eq_identity ) THEN
                 CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),          &
                                  userdata, data%transpose, data%U( : nlp%n ), &
                                  data%V( : nlp%m ), got_j = data%got_j )
                 IF ( data%eval_status /= 0 ) THEN
                   inform%bad_eval = 'eval_JPROD'
                   inform%status = GALAHAD_error_evaluation ; GO TO 900
                 END IF
               ELSE
                 CALL eval_JPROD( data%eval_status, nlp%X( : nlp%n ),          &
                                  userdata, data%transpose, data%U( : nlp%n ), &
                                  W( : nlp%m ) * data%V( : nlp%m ),            &
                                  got_j = data%got_j )
                 IF ( data%eval_status /= 0 ) THEN
                   inform%bad_eval = 'eval_JPROD'
                   inform%status = GALAHAD_error_evaluation ; GO TO 900
                 END IF
               END IF
             END IF
           END IF
         END IF

!  return from reverse communication with the Jacobian transpose-vector
!  product

   240   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 240, GLRT_',    &
        &  'inform%status = ', I0 )" ) prefix, inform%GLRT_inform%status
         IF ( inform%GLRT_inform%status == 3 .AND.                             &
              data%control%model == newton_model ) THEN
           IF ( .NOT. data%hessian_available .AND.                             &
                .NOT. data%got_h ) inform%h_eval = inform%h_eval + 1

!  if the Hessian has been calculated, form the product directly

           IF ( data%hessian_available ) THEN
             CALL mop_Ax( one, nlp%H, data%W( : nlp%n ), one,                  &
                          data%U( : nlp%n ), data%out, data%control%error,     &
                          0, symmetric = .TRUE. )

!  if the Hessian is unavailable, obtain a matrix-free product

           ELSE
             IF ( data%reverse_hprod ) THEN
               data%V( : nlp%n ) = data%W( : nlp%n )
               data%branch = 250 ; inform%status = 6 ; RETURN
             ELSE
               CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),            &
                                data%Y( : nlp%m ), userdata,                   &
                                data%U( : nlp%n ), data%W( : nlp%n ),          &
                                got_h = data%got_h )
               IF ( data%eval_status /= 0 ) THEN
                 inform%bad_eval = 'eval_HPROD'
                 inform%status = GALAHAD_error_evaluation ; GO TO 900
               END IF
               data%got_h = .TRUE.
             END IF
           END IF
         END IF

!  return from reverse communication with the Hessian-vector product
!  or preconditioned vector

   250   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 250, GLRT_',    &
        &  'inform%status = ', I0 )" ) prefix, inform%GLRT_inform%status
         IF ( inform%GLRT_inform%status == 3 ) THEN
           IF ( data%control%model == newton_model .OR.                        &
                data%control%model == gauss_newton_model ) THEN
             data%V( : nlp%n ) = data%U( : nlp%n )
           END IF

!  include the product with the Hessian of the regularization term, if needed.
!  Hessian = weight*(power-2)*(x^TSx)^(power-4)/2 * Sx (Sx)^T +
!            weight*(x^TSx)^(power-2)/2 * S when p>2 or
!          = weight * S when p=2

!  i.e., Hessian v = weight*(power-2)*(x^TSx)^(power-4)/2 * ( v^T Sx ) * Sx +
!                    weight*(x^TSx)^(power-2)/2 * Sv when p>2 or
!                  = weight * Sv when p=2

           IF ( data%stabilised ) THEN
             IF ( stabilisation%power == two ) THEN
               data%V( : nlp%n ) =                                             &
                 data%V( : nlp%n ) + stabilisation%weight * data%SV( : nlp%n )
             ELSE
               IF ( data%xtsx > zero ) THEN
                 val = stabilisation%weight *                                  &
                   ( data%xtsx ** ( ( stabilisation%power - two ) / two ) )
                 data%V( : nlp%n ) =                                           &
                   data%V( : nlp%n ) + val * data%SV( : nlp%n )
                 IF ( stabilisation%power == four ) THEN
                   val = stabilisation%weight * two                            &
                         * DOT_PRODUCT( data%W( : nlp%n ), data%SX( : nlp%n ) )
                 ELSE
                   val = stabilisation%weight *                                &
                         ( stabilisation%power - two ) *                       &
                   ( data%xtsx ** ( ( stabilisation%power - four ) / two ) )   &
                         * DOT_PRODUCT( data%W( : nlp%n ), data%SX( : nlp%n ) )
                 END IF
                 data%V( : nlp%n ) =                                           &
                   data%V( : nlp%n ) + val * data%SX( : nlp%n )
               END IF
             END IF
           END IF
         END IF

         IF ( inform%GLRT_inform%status == 2 .AND. data%reverse_scale .AND.    &
              data%regularization_type == user_regularization ) THEN
           data%V( : nlp%n ) = data%U( : nlp%n )
         END IF
       GO TO 210

!  End of the generalized Lanczos iteration
!  ........................................

   260 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 260' )" ) prefix
!write(6,*) ' model ', data%model
       data%model = data%model - inform%obj

!  Record whether there is negative curvature or if the boundary is encountered

       IF ( inform%GLRT_inform%negative_curvature ) THEN
         data%negcur = 'n'
       ELSE
         data%negcur = ' '
       END IF
       data%s_norm = inform%GLRT_inform%xpo_norm
!      write(6,*) ' s_norm ', data%s_norm

!      Record the total number of Lanczos iterations

       inform%cg_iter = inform%cg_iter +                                       &
         inform%GLRT_inform%iter + inform%GLRT_inform%iter_pass2
       IF ( data%printt ) WRITE( data%out,                                     &
          "( /, A, ' CG iterations required = ', I8 )" )                       &
            prefix, inform%GLRT_inform%iter

!  ============================================================================
!  3. check for acceptance of the new point
!  ============================================================================

   300 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 300' )" ) prefix
!      IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
!        WRITE( data%out, "( /, A, ' name                  S' )" ) prefix
!        DO i = 1, nlp%n
!          WRITE( data%out, "(  A, 1X, I10, ES22.14 )" )  prefix, i, data%S( i )
!        END DO
!      END IF

!  see if the correction will make any difference

       IF ( MAXVAL( ABS( data%S( : nlp%n ) ) / MAX( one, nlp%X( : nlp%n ) ) )  &
            <= data%control%stop_s ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 900
       END IF

!   CALL mop_Ax( one, nlp%J,  data%S( : nlp%n ), zero,                         &
!                data%V( : nlp%m ), out = data%out,                            &
!                error = data%control%error, print_level = 0,                  &
!                transpose = .FALSE. )
!   data%U( : nlp%n ) = zero
!   CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),                       &
!                    nlp%C( : nlp%m ), userdata,                               &
!                    data%U( : nlp%n ), data%S( : nlp%n ),                     &
!                    got_h = data%got_h )
!    WRITE(6,*) ' ||Js||^2/||s||^2, ||c|| ',                                   &
!    ( TWO_NORM( data%V( : nlp%m ) ) / TWO_NORM( data%S( : nlp%m ) ) ) **2,    &
!    DOT_PRODUCT( data%S( : nlp%m ), data%U( : nlp%n ) ) /                     &
!      TWO_NORM( data%S( : nlp%m ) ) **2
!    TWO_NORM( nlp%C( : nlp%m ) )

!  compute the slope and curvature along the step

       data%stg = DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )
       data%hstbs = data%model - data%stg
!      write(6,*) ' stg = ', data%stg

!  record the current point

       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%C_current( : nlp%m ) = nlp%C( : nlp%m )
       data%obj_current = inform%obj
       data%norm_c_current = inform%norm_c

!  form the trial point

!write(6,"( ' s = ', 2ES12.4 )" ) data%S( : nlp%n )
       nlp%X( : nlp%n ) = data%X_current( : nlp%n ) + data%S( : nlp%n )

!  evaluate the objective function at the trial point

       IF ( data%reverse_c ) THEN
         data%branch = 320 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_C( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%C( : nlp%m ) )
         IF ( data%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_C'
           inform%status = GALAHAD_error_evaluation ; GO TO 900
         END IF
       END IF

!  return from reverse communication with the objective value

   320 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 320' )" ) prefix
       inform%c_eval = inform%c_eval + 1
       IF ( data%w_eq_identity ) THEN
         data%Y( : nlp%m ) = nlp%C( : nlp%m )
         data%norm_c_trial = TWO_NORM( nlp%C( : nlp%m ) )
         data%f_trial = half * data%norm_c_trial ** 2
       ELSE
         data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
         val = DOT_PRODUCT( data%Y( : nlp%m ), nlp%C( : nlp%m ) )
         data%norm_c_trial = SQRT( val )
         data%f_trial = half * val
       END IF

!      IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
!        WRITE( data%out, "( /, A, ' name                  C' )" ) prefix
!        DO i = 1, nlp%m
!          WRITE( data%out, "(  A, 1X, I10, ES22.14 )" )  prefix, i, nlp%C( i )
!        END DO
!      END IF

!      write(6,*) ' reg, wei ft', data%stabilised, stabilisation%weight
!      write(6,*) ' stabilised ? ', data%stabilised
!      write(6,*) ' ftrial before, ||s|| ', data%f_trial,TWO_NORM(nlp%X(:nlp%n))

!  account for the stabilization term if necessary

       IF ( data%stabilised ) THEN
         data%SV( : nlp%n ) = data%SX( : nlp%n )
         data%xtsx_current = data%xtsx
         IF ( data%regularization_type == user_regularization ) THEN
           IF ( data%reverse_scale ) THEN
             data%V( : nlp%n ) = nlp%X( : nlp%n )
             data%branch = 330 ; inform%status = 8 ; RETURN
           ELSE
             CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%SX( : nlp%n ), nlp%X( : nlp%n ) )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_SCALE'
               inform%status = GALAHAD_error_evaluation ; GO TO 900
             END IF
           END IF
         ELSE
           CALL mop_Ax( one, stabilisation%matrix,                             &
                        nlp%X( : nlp%n ), zero, data%SX( : nlp%n ),            &
                        symmetric = .TRUE., out = data%out,                    &
                        error = data%control%error, print_level = 0 )
         END IF
       END IF
!      write(6,*) ' ftrial after ', data%f_trial

!  return from reverse communication with the scaled vector u = S(x) x

  330  CONTINUE
       IF ( data%stabilised ) THEN
         IF ( data%regularization_type == user_regularization .AND.            &
                data%reverse_scale ) data%SX( : nlp%n ) = data%U( : nlp%n )
         data%xtsx = DOT_PRODUCT( nlp%X( : nlp%n ), data%SX( : nlp%n ) )
         data%f_trial = data%f_trial +                                         &
           ( stabilisation%weight / stabilisation%power )                      &
              * data%xtsx ** ( stabilisation%power / two )
       END IF

!  deal with NaN trial objective values
!  ------------------------------------

!      data%f_is_nan = IEEE_IS_NAN( data%f_trial )
       data%f_is_nan = data%f_trial /= data%f_trial
       IF ( data%f_is_nan ) THEN
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%C( : nlp%m ) = data%C_current( : nlp%m )
         IF ( data%stabilised ) THEN
           data%SX( : nlp%n ) = data%SV( : nlp%n )
           data%xtsx = data%xtsx_current
         END IF

!  control printing for the NaN case

         IF ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print .AND.                              &
              MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) &
             THEN
           data%printi = data%set_printi ; data%printt = data%set_printt
           data%printm = data%set_printm ; data%printw = data%set_printw
           data%printd = data%set_printd
           data%print_level = data%control%print_level
           data%control%GLRT_control%print_level = data%print_level_glrt
           data%control%RQS_control%print_level = data%print_level_rqs
         ELSE
           data%printi = .FALSE. ; data%printt = .FALSE.
           data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
           data%print_level = 0
           data%control%GLRT_control%print_level = 0
           data%control%RQS_control%print_level = 0
         END IF
         data%print_iteration_header = data%print_level > 1 .OR.               &
           ( data%control%GLRT_control%print_level > 0 .AND. .NOT.             &
             data%control%subproblem_direct ) .OR.                             &
           ( data%control%RQS_control%print_level > 0 .AND.                    &
             data%control%subproblem_direct )

!  print one-line summary

         IF ( data%printi ) THEN
            IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
             WRITE( data%out, 2090 ) prefix
             IF ( data%control%subproblem_direct ) THEN
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2170 ) prefix
               ELSE
                 WRITE( data%out, 2160 ) prefix
               END IF
             ELSE
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2110 ) prefix
               ELSE
                 WRITE( data%out, 2100 ) prefix
               END IF
             END IF
           END IF
           data%print_1st_header = .FALSE.
           char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
           IF ( data%control%subproblem_direct ) THEN
             char_facts = ADJUSTR( STRING_integer_6( data%total_facts ) )
             WRITE( data%out,  "( A, A6, 1X, 3A1, '    NaN           -    ',   &
            &  '    - Inf ',  2ES8.1, 1X, A6, F8.2 )" )                        &
                prefix, char_iter, data%accept, data%negcur, data%hard,        &
                inform%weight, data%s_norm, char_facts, data%clock_now
           ELSE
             char_sit = ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter ) )
             char_sit2 =                                                       &
                ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter_pass2 ) )
             WRITE( data%out, "( A, A6, 1X, 3A1, '    NaN           -    ',    &
            &  '    - Inf ', 2ES8.1, 1X, 2A6, F8.2 )" ) prefix, char_iter,     &
                data%accept, data%negcur, data%perturb,                        &
                inform%weight, data%s_norm, char_sit, char_sit2, data%clock_now
           END IF
         END IF

!  check to see if we are still "alive"

         IF ( data%control%alive_unit > 0 ) THEN
           INQUIRE( FILE = data%control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             inform%status = GALAHAD_error_alive
             RETURN
           END IF
         END IF

!  check to see if the iteration limit has been exceeded

         inform%iter = inform%iter + 1
         IF ( inform%iter > data%control%maxit .AND. data%step_accepted ) THEN
           inform%status = GALAHAD_error_max_iterations ; GO TO 900
         END IF

!  increase the regularization weight and try again

         inform%weight = data%control%weight_increase * inform%weight
         GO TO 190
       END IF

!  compute the change in objective

       data%df = inform%obj - data%f_trial
!      if (data%printi) write(6,*) ' dm, df ', - data%model, data%df

!  compute the ratio of actual to predicted reduction over the current iteration

!      rounding = MAX( one, ABS( inform%obj ) ) * teneps
       rounding =                                                              &
         MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = wp ) * epsmch

       ared = data%df + rounding
       prered = - data%model + rounding
       IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )            &
         ared = prered
!write(6,*) ' ared, pred ', ared, prered
       data%ratio = ared / prered
!      write(6,*) ' ratio ', data%ratio, ared, prered
       IF ( data%printm ) WRITE( data%out, "( /, A, ' actual, predicted',      &
      &   ' reductions = ', 2ES12.4 )" ) prefix, ared, prered

!  compute the ratio of actual to predicted reduction over the recent history

       IF ( .NOT. data%monotone ) THEN

!  compute the largest f in the history

         data%ref = MAXLOC( data%F_hist( data%non_monotone_history + 2         &
                            - data%max_hist : data%non_monotone_history + 1 ) )
         data%f_ref = data%F_hist( data%ref( 1 ) )

!  use the larger of these two ratios to assess progress

         data%ratio = MAX( data%ratio, ( data%f_trial - data%f_ref ) /         &
           ( SUM( data%D_hist( data%ref( 1 ) + 1 :                             &
             data%non_monotone_history + 1 ) ) + data%model ) )
       END IF

!  the new point is acceptable

       IF ( data%ratio >= data%control%eta_successful ) THEN
         data%poor_model = .FALSE.
         data%accept = 'a'
         data%step_accepted = .TRUE.
         inform%norm_c = data%norm_c_trial
         inform%obj = data%f_trial
         data%s_norm_successful = data%s_norm

!  stop if the residual is sufficiently small

         IF ( inform%norm_c <= data%stop_c ) THEN

!  print one-line summary

           IF ( data%printi ) THEN
             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
             data%time_now = data%time_now - data%time_start
             data%clock_now = data%clock_now - data%clock_start
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2090 ) prefix
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2170 ) prefix
               ELSE
                 WRITE( data%out, 2160 ) prefix
               END IF
             ELSE
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2110 ) prefix
               ELSE
                 WRITE( data%out, 2100 ) prefix
               END IF
             END IF
             data%print_1st_header = .FALSE.
             char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
             IF ( inform%iter > 0 ) THEN
               IF ( data%control%subproblem_direct ) THEN
                 char_facts =                                                  &
                   ADJUSTR( STRING_integer_6( data%total_facts ) )
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2120 ) prefix, char_iter, data%accept,     &
                      data%negcur, data%hard, inform%obj,                      &
                      inform%norm_g, data%ratio, data%old_weight,              &
                      data%s_norm, char_facts, data%clock_now
                 ELSE
                   WRITE( data%out, 2120 ) prefix, char_iter, data%accept,     &
                      data%negcur, data%hard, inform%norm_c,                   &
                      inform%norm_g, data%ratio, data%old_weight,              &
                      data%s_norm, char_facts, data%clock_now
                 END IF
               ELSE
                 char_sit = ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter))
                 char_sit2 =                                                   &
                    ADJUSTR( STRING_integer_6( inform%GLRT_inform%iter_pass2 ) )
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2130 ) prefix, char_iter, data%accept,     &
                      data%negcur, data%perturb, inform%obj,                   &
                      inform%norm_g, data%ratio, data%old_weight, data%s_norm, &
                      char_sit, char_sit2, data%clock_now
                 ELSE
                   WRITE( data%out, 2130 ) prefix, char_iter, data%accept,     &
                      data%negcur, data%perturb, inform%norm_c,                &
                      inform%norm_g, data%ratio, data%old_weight, data%s_norm, &
                      char_sit, char_sit2, data%clock_now
                 END IF
               END IF
             ELSE
               IF ( data%control%subproblem_direct ) THEN
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2140 ) prefix,                             &
                      char_iter, inform%obj, inform%norm_g
                 ELSE
                   WRITE( data%out, 2140 ) prefix,                             &
                      char_iter, inform%norm_c, inform%norm_g
                 END IF
               ELSE
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2150 ) prefix,                             &
                        char_iter, inform%obj, inform%norm_g
                 ELSE
                   WRITE( data%out, 2150 ) prefix,                             &
                        char_iter, inform%norm_c, inform%norm_g
                 END IF
               END IF
             END IF
           END IF
           inform%status = GALAHAD_ok ; GO TO 900
         END IF

!  if a "magic" step is permitted, return to the user to allow for this
!  opportunity

         IF ( data%control%magic_step ) THEN
           data%branch = 380 ; inform%status = 9 ; RETURN
         END IF

!  the new point is not acceptable

       ELSE
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%C( : nlp%m ) = data%C_current( : nlp%m )
         IF ( data%w_eq_identity ) THEN
           data%Y( : nlp%m ) = nlp%C( : nlp%m )
         ELSE
           data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
         END IF
         IF ( data%stabilised ) THEN
           data%SX( : nlp%n ) = data%SV( : nlp%n )
           data%xtsx = data%xtsx_current
         END IF
         data%new_point = .FALSE.
       END IF

!  return after possible magic step

  380  CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 380' )" ) prefix

!  update the objective function value to account for the magic step

       IF ( data%ratio >= data%control%eta_successful .AND.                    &
            data%control%magic_step ) THEN
         inform%c_eval = inform%c_eval + 1
         IF ( data%w_eq_identity ) THEN
           data%Y( : nlp%m ) = nlp%C( : nlp%m )
           inform%norm_c = TWO_NORM( nlp%C( : nlp%m ) )
           inform%obj = half * inform%norm_c ** 2
         ELSE
           data%Y( : nlp%m ) = W( : nlp%m ) * nlp%C( : nlp%m )
           val = DOT_PRODUCT( data%Y( : nlp%m ), nlp%C( : nlp%m ) )
           inform%norm_c = SQRT( val )
           inform%obj = half * val
         END IF

!  account for the stabilization term if necessary

         IF ( data%stabilised ) THEN
           IF ( data%regularization_type == user_regularization ) THEN
             IF ( data%reverse_scale ) THEN
               data%V( : nlp%n ) = nlp%X( : nlp%n )
               data%branch = 390 ; inform%status = 8 ; RETURN
             ELSE
               CALL eval_SCALE( data%eval_status, nlp%X( : nlp%n ), userdata,  &
                                data%SX( : nlp%n ), nlp%X( : nlp%n ) )
               IF ( data%eval_status /= 0 ) THEN
                 inform%bad_eval = 'eval_SCALE'
                 inform%status = GALAHAD_error_evaluation ; GO TO 900
               END IF
             END IF
           ELSE
             CALL mop_Ax( one, stabilisation%matrix,                           &
                          nlp%X( : nlp%n ), zero, data%SX( : nlp%n ),          &
                          symmetric = .TRUE., out = data%out,                  &
                          error = data%control%error, print_level = 0 )
           END IF
         END IF
       END IF

!  return from reverse communication with the scaled vector u = S(x) x

  390  CONTINUE
       IF ( data%ratio >= data%control%eta_successful ) THEN
         IF ( data%control%magic_step .AND. data%stabilised ) THEN
           IF ( data%regularization_type == user_regularization .AND.          &
                data%reverse_scale ) data%SX( : nlp%n ) = data%U( : nlp%n )
           data%xtsx = DOT_PRODUCT( nlp%X( : nlp%n ), data%SX( : nlp%n ) )
             inform%obj = inform%obj +                                         &
               ( stabilisation%weight / stabilisation%power )                  &
                  * data%xtsx ** ( stabilisation%power / two )
         END IF

!  update the history

         IF ( data%monotone ) THEN
           data%f_ref = inform%obj

!  shift history of function and model values

         ELSE
           DO i = 1, data%non_monotone_history
             data%F_hist( i ) = data%F_hist( i + 1 )
             data%D_hist( i ) = data%D_hist( i + 1 )
           END DO

!  replace the oldest

           data%F_hist( data%non_monotone_history + 1 ) = inform%obj
           data%D_hist( data%non_monotone_history + 1 ) = data%model

!  find how much past history is allowed

           data%max_hist = MIN( data%max_hist + 1, data%non_monotone_history )
         END IF
       END IF

!  ==========================================================
!  4. Update the regularization weight and other book-keeping
!  ==========================================================

       data%old_weight = inform%weight
       data%successful = data%ratio >= data%control%eta_successful

       SELECT CASE ( data%control%weight_update_strategy )
       CASE ( weight_update_zero_reset )
         IF ( data%ratio < data%control%eta_successful ) THEN
           inform%weight = MAX( data%control%weight_increase * inform%weight,  &
                                data%minimum_weight )
         ELSE IF ( data%ratio >= data%control%eta_very_successful .AND.        &
                  data%ratio <= data%control%eta_too_successful ) THEN
           inform%weight = weight_zero
         END IF
       CASE ( weight_update_imitate_tr )
         IF ( data%ratio < data%control%eta_successful ) THEN
           IF ( .NOT. data%reduce ) THEN
             data%delta = data%s_norm
             data%reduce = .TRUE.
           ELSE
             data%delta = data%delta / data%control%weight_increase
           END IF
           inform%weight = inform%weight + ( data%g_norm / data%delta ) *      &
             ( data%control%weight_increase - one )
         ELSE
           data%reduce = .FALSE.
           IF ( data%ratio >= data%control%eta_very_successful .AND.           &
                  data%ratio <= data%control%eta_too_successful ) THEN
             inform%weight = MAX( data%minimum_weight,                         &
               data%control%weight_decrease * inform%weight )
           END IF
         END IF
!      CASE ( weight_update_gpt )
!        CALL ARC_adjust_weight( inform%weight, data%model, data%stg,          &
!                                data%hstbs, data%s_norm, data%ratio,          &
!                                data%ARC_control )
!        inform%weight = MAX( data%minimum_weight, inform%weight )

!        IF ( data%ratio < control%eta_successful ) THEN
!          IF ( data%control%subproblem_direct ) THEN
!            val = two * inform%RQS_inform%pole / data%s_norm_successful
!          ELSE
!            val = - two * inform%GLRT_inform%leftmost /data%s_norm_successful
!          END IF
!          inform%weight = MAX( inform%weight, val )
!        END IF
       CASE DEFAULT
         IF ( data%ratio < data%control%eta_successful ) THEN
           inform%weight = data%control%weight_increase * inform%weight
           IF ( data%control%weight_update_strategy ==                         &
                  weight_update_increase .AND.                                 &
                data%control%model == gauss_newton_model ) THEN
             IF ( data%s_norm <= ten ** ( - 4 ) .AND.                          &
                  inform%norm_g < inform%norm_c ) THEN
               data%minimum_weight = MAX( data%minimum_weight, inform%weight )
               IF ( data%printi ) WRITE( data%out, "( A, ' increasing min ',   &
              &  'weight to', ES9.2 )" ) prefix, data%minimum_weight
             END IF
           END IF
         ELSE IF ( data%ratio >= data%control%eta_very_successful .AND.        &
                  data%ratio <= data%control%eta_too_successful ) THEN
           inform%weight = MAX( data%minimum_weight,                           &
                                data%control%weight_decrease * inform%weight )
         END IF
       END SELECT

       IF ( data%ratio >= data%control%eta_successful ) THEN
         IF ( data%control%norm /= euclidean_regularization ) THEN
           IF ( data%control%renormalize_weight ) THEN
             IF ( data%control%subproblem_direct ) THEN
               data%s_new_norm = data%s_norm
             ELSE
               IF ( data%regularization_type >                                 &
                    diagonal_jtj_regularization ) THEN
                 data%s_new_norm = PSLS_norm( data%H, data%S, data%PSLS_data,  &
                     data%control%PSLS_control, inform%PSLS_inform )
                 IF ( inform%PSLS_inform%status == GALAHAD_norm_unknown ) THEN
                   data%s_new_norm = data%s_norm
                 ELSE IF ( inform%PSLS_inform%status /= 0 ) THEN
                   GO TO 980
                 END IF
               ELSE IF ( data%regularization_type ==                           &
                         diagonal_jtj_regularization ) THEN
                 data%s_new_norm = SQRT( DOT_PRODUCT( data%S( : nlp%n ),       &
                   data%regularization%matrix%val( : nlp%n )                   &
                     * data%S( : nlp%n ) ) )
               END IF
             END IF
             IF ( data%printt )                                                &
               WRITE( data%out, "( A, ' ratio new, old norms = ', ES12.4 )" )  &
                 prefix, data%s_new_norm / data%s_norm
           ELSE
             data%s_new_norm = data%s_norm
           END IF

!  if the norm has changed, adjust the weight accordingly

           inform%weight = inform%weight * ( data%s_new_norm / data%s_norm )
           data%s_norm = data%s_new_norm
         END IF
       END IF
!      write(6,*) ' weight update ', inform%weight

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start
       IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,     &
      &    F12.2,  ' seconds' )" ) prefix, data%clock_now
       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
              data%time_now > data%control%cpu_time_limit ) .OR.               &
            ( data%control%clock_time_limit >= zero .AND.                      &
              data%clock_now > data%control%clock_time_limit ) ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

!write(6,*) ' f ', data%f_trial, data%ratio
!stop
     GO TO 100

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 900' )" ) prefix

!  print details of solution

     IF ( inform%norm_c > zero ) THEN
       inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) ) / inform%norm_c
     ELSE
       inform%norm_g = zero
     END IF
!    write(6,*) ' final weight = ', inform%weight

 910 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 910' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN

!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%c_eval, inform%j_eval, inform%h_eval,   &
!         inform%iter, inform%cg_iter, inform%obj, inform%norm_g
!      WRITE ( data%out, 2010 )
!      IF ( data%print_level > 3 ) THEN
!         l = nlp%n
!      ELSE
!         l = 2
!      END IF
!      DO j = 1, 2
!         IF ( j == 1 ) THEN
!            ir = 1 ; ic = MIN( l, nlp%n )
!         ELSE
!            IF ( ic < nlp%n - l ) WRITE( data%out, 2050 )
!            ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
!         END IF
!         DO i = ir, ic
!            WRITE ( data%out, 2020 ) nlp%vnames( i ), nlp%X( i ), nlp%G( i )
!         END DO
!      END DO

       WRITE( data%out, "( /, A, '  Problem: ', A, ' (n = ', I0, ', m = ', I0, &
    &   ')', /, A, '  NLS stopping tolerances (c,J''c/c) =', 2ES9.2 )" )       &
          prefix, TRIM( nlp%pname ), nlp%n, nlp%m, prefix,                     &
          data%stop_c, data%stop_g
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" )        &
         prefix, data%non_monotone_history
       IF ( data%gauss_to_newton_model .AND. data%control%model ==             &
            newton_model ) data%control%model = gauss_to_newton_model
       SELECT CASE( data%model_used )
       CASE ( first_order_model )
         WRITE( data%out, "( A, '  First-order model used' )" ) prefix
       CASE ( diagonal_hessian_model )
         WRITE( data%out, "( A, '  Second-order model with identity',          &
        &  ' Hessian used' )" ) prefix
       CASE ( gauss_newton_model )
         WRITE( data%out, "( A, '  Gauss-Newton model used' )" ) prefix
       CASE ( newton_model )
         WRITE( data%out, "( A, '  Second-order (Newton) model used' )" ) prefix
       CASE ( gauss_to_newton_model )
         WRITE( data%out, "( A, '  Gauss-Newton-to-Newton model used' )") prefix
       END SELECT
       WRITE( data%out, "( A, '  Regularization power =', F4.1 )" )            &
          prefix, data%power
       IF ( data%control%magic_step )                                          &
         WRITE( data%out, "( A, '  Magic step used' )" ) prefix
       IF ( data%control%subproblem_direct ) THEN
         IF ( inform%RQS_inform%dense_factorization ) THEN
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (eigen solver SYSV',                       &
          &      ') of the regularization sub-problem' )" ) prefix
         ELSE
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (solver ', A,                              &
          &      ') of the regularization sub-problem' )" )                    &
              prefix, TRIM( data%control%RQS_control%definite_linear_solver )
         END IF
         SELECT CASE ( data%regularization_type )
         CASE ( user_regularization )
           WRITE( data%out, "( A, '  User-defined regularization used' )" )    &
             prefix
         CASE ( euclidean_regularization )
           WRITE( data%out, "( A, '  Euclidean regularization used' )" ) prefix
         CASE ( diagonal_jtj_regularization )
           WRITE( data%out, "( A, '  Diagonal (JTJ) regularization used' )" )  &
             prefix
         CASE ( diagonal_hessian_regularization )
           WRITE( data%out, "( A, '  Diagonal (H) regularization used' )" )    &
             prefix
         CASE ( band_regularization )
           WRITE( data%out, "( A, '  Band regularization (semi-bandwidth ',    &
          &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
         CASE ( reordered_band_regularization )
           WRITE( data%out, "( A, ' Reordered band regularization',            &
          &   ' (semi-bandwidth ', I0, ') used' )" ) prefix,                   &
             inform%PSLS_inform%semi_bandwidth_used
         CASE ( schnabel_eskow_regularization, gmps_regularization,            &
                lin_more_regularization, mi28_regularization )
           WRITE( data%out, "( A, '  Modified full matrix regularization',     &
          & ' used' )" ) prefix
         END SELECT
         WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
        &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
           inform%RQS_inform%factorizations,                                   &
           inform%RQS_inform%time%clock_factorize
         IF ( TRIM( data%control%RQS_control%definite_linear_solver ) ==       &
              'pbtr' ) THEN
           WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
          & ', semi-bandwidth = ', I0  )" ) prefix, inform%max_entries_factors,&
              inform%RQS_inform%SLS_inform%semi_bandwidth
         ELSE
           WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
             prefix, inform%max_entries_factors
         END IF
       ELSE
         WRITE( data%out,                                                      &
           "( A, '  Iterative solution of the regularization sub-problem' )" ) &
              prefix
         IF ( data%regularization_type > 0 )                                   &
           WRITE( data%out, "( A, '  Hessian semi-bandwidth (original,',       &
          &     ' re-ordered) = ', I0, ', ', I0 )" ) prefix,                   &
             inform%PSLS_inform%semi_bandwidth,                                &
             inform%PSLS_inform%reordered_semi_bandwidth
         SELECT CASE ( data%regularization_type )
         CASE ( user_regularization )
           WRITE( data%out, "( A, '  User-defined regularization used' )" )    &
             prefix
         CASE ( euclidean_regularization )
           WRITE( data%out, "( A, '  Euclidean regularization used' )" ) prefix
         CASE ( diagonal_jtj_regularization )
           WRITE( data%out, "( A, '  Diagonal (JTJ) regularization used' )" )  &
             prefix
         CASE ( diagonal_hessian_regularization )
           WRITE( data%out, "( A, '  Diagonal (H) regularization used' )" )    &
             prefix
         CASE ( band_regularization )
           WRITE( data%out, "( A, '  Band regularization (semi-bandwidth ',    &
          &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
         CASE ( reordered_band_regularization )
           WRITE( data%out, "( A, ' Reordered band regularization',            &
          &    ' (semi-bandwidth ', I0, ') used' )" ) prefix,                  &
               inform%PSLS_inform%semi_bandwidth_used
         CASE ( schnabel_eskow_regularization )
           WRITE( data%out, "( A, '  SE (solver ', A, ') full',                &
          &  ' regularization used' )" ) prefix,                               &
           TRIM( data%control%PSLS_control%definite_linear_solver )
         CASE ( gmps_regularization )
           WRITE( data%out, "( A, '  GMPS (solver ', A, ') full',              &
          &    ' regularization used' )" ) prefix,                             &
               TRIM( data%control%PSLS_control%definite_linear_solver )
         CASE ( lin_more_regularization )
           WRITE( data%out, "( A, '  Lin-More''(', I0, ') incomplete',         &
          & ' Cholesky factorization regularization used ' )" )                &
            prefix, data%control%PSLS_control%icfs_vectors
         CASE ( mi28_regularization )
           WRITE( data%out, "( A, '  HSL_MI28(', I0, ',', I0, ') incomplete',  &
           & ' Cholesky factorization regularization used ' )" ) prefix,       &
            data%control%PSLS_control%mi28_lsize,                              &
            data%control%PSLS_control%mi28_rsize
         END SELECT
         IF ( data%control%renormalize_weight ) WRITE( data%out,               &
            "( A, '  Weight renormalized' )" ) prefix
       END IF
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 980' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 990' )" ) prefix
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'NLS_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # Hessian evaluations   = ', I10,                        &
              /, A, ' # major  iterations     = ', I10,                        &
              /, A, ' # minor (cg) iterations = ', I10,                        &
             //, A, ' objective value         = ', ES22.14,                    &
              /, A, ' gradient norm           = ', ES12.4 )
 2010 FORMAT( /, A, ' name                  X                   G ' )
 2020 FORMAT(  A, 1X, A10, 2ES22.14 )
 2030 FORMAT(  A, 1X, I10, 2ES22.14 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2090 FORMAT( A, '        (a=accept r=reject n=-ve curvature h=hard case)' )
 2100 FORMAT( A, '    It         c        J''c/c     ',                        &
             ' ratio   weight    step pass 1 pass 2   time' )
 2110 FORMAT( A, '    It         f          g      ',                          &
             ' ratio   weight    step pass 1 pass 2   time' )
 2120 FORMAT( A, A6, 1X, 3A1, 2ES11.4, ES9.1, 2ES8.1, 1X, A6, F8.2 )
 2130 FORMAT( A, A6, 1X, 3A1, 2ES11.4, ES9.1, 2ES8.1, 1X, 2A6, F8.2 )
 2140 FORMAT( A, A6, 4X, 2ES11.4 )
 2150 FORMAT( A, A6, 4X, 2ES11.4 )
 2160 FORMAT( A, '    It         c        J''c/c     ',                        &
             ' ratio   weight   step  # fact    time' )
 2170 FORMAT( A, '    It         f           g      ',                         &
             ' ratio   weight   step  # fact    time' )

 !  End of subroutine NLS_subproblem_solve

     END SUBROUTINE NLS_subproblem_solve

!! G A L A H A D - N L S _ u p d a t e _ h i s t o r y  S U B R O U T I N E
!
!     SUBROUTINE NLS_update_history( history, max_hist, F_hist, F_ref, f )
!
!!-----------------------------------------------
!!   D u m m y   A r g u m e n t s
!!-----------------------------------------------
!
!     INTEGER, INTENT( IN ) :: history
!     INTEGER, INTENT( INOUT ) :: max_hist
!     REAL ( KIND = wp ), INTENT( OUT ) :: F_ref
!     REAL ( KIND = wp ), INTENT( IN ) :: f
!     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( history + 1 ) :: F_hist
!
!!-----------------------------------------------
!!   L o c a l   V a r i a b l e s
!!-----------------------------------------------
!
!     INTEGER :: i
!
!!  Shift history of function values
!
!     DO i = 1, history
!       F_hist( i ) = F_hist( i + 1 )
!     END DO
!
!!  Replace the oldest
!
!     F_hist( history + 1 ) = f
!
!!  Find how much past history is allowed
!
!     max_hist = MIN( max_hist + 1, history )
!
!!  Compute the largest f in the history
!
!     f_ref = MAXVAL( F_hist( history + 2 - max_hist : history + 1 ) )
!
!     RETURN
!
!     END SUBROUTINE NLS_update_history

!-*-*-*-  G A L A H A D -  N L S _ s e t _ m a p  S U B R O U T I N E  -*-*-*-

     SUBROUTINE NLS_set_map( A, B, IW, PTR, ROW, ORDER, b_in_a,                &
                             deallocate_error_fatal, space_critical, out,      &
                             MAP, status, alloc_status, bad_alloc )

!  find a mapping of the entries of the matrix B into A, or vice versa - the
!  sparsity pattern of the relevant one is presumed to be a subset of the other,
!  and b_in_a should be set true iff it is B in A that is required. A should
!  be stored by columns (either as a sparse or dense matrix) while B can
!  be in any supported GALAHAD format.

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: out
     LOGICAL, INTENT( IN ) :: b_in_a, deallocate_error_fatal, space_critical
     INTEGER, INTENT( OUT ) :: status
     INTEGER, INTENT( INOUT ) :: alloc_status
     INTEGER, INTENT( INOUT ), ALLOCATABLE, DIMENSION( : ) :: IW, PTR, ROW
     INTEGER, INTENT( INOUT ), ALLOCATABLE, DIMENSION( : ) :: ORDER, MAP
     CHARACTER ( LEN = 80 ), INTENT( INOUT ) :: bad_alloc
     TYPE ( SMT_type ), INTENT( IN ) :: A, B

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, l, ll
     LOGICAL :: b_dense
     CHARACTER ( LEN = 80 ) :: array_name

!    write(6,*) ' B in A? ', b_in_a
!    write(6,*) ' type A, B ', SMT_get( A%type ), ' ', SMT_get( B%type )
!  First order B by columns. Assign workspace as well as space for the required
!  mapping, MAP

     array_name = 'nls: IW'
     CALL SPACE_resize_array( B%m, IW, status, alloc_status,                   &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical, bad_alloc = bad_alloc, out = out )
     IF ( status /= 0 ) RETURN

     array_name = 'nls: PTR'
     CALL SPACE_resize_array( B%n + 1, PTR, status, alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical, bad_alloc = bad_alloc, out = out )

     array_name = 'nls: ROW'
     CALL SPACE_resize_array( B%ne, ROW, status, alloc_status,                 &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical, bad_alloc = bad_alloc, out = out )

     array_name = 'nls: ORDER'
     CALL SPACE_resize_array( B%ne, ORDER, status, alloc_status,               &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical, bad_alloc = bad_alloc, out = out )

     array_name = 'nls: MAP'
     CALL SPACE_resize_array( B%ne, MAP, status, alloc_status,                 &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical, bad_alloc = bad_alloc, out = out )

!  count the numbers of entries in each column of B

     SELECT CASE ( SMT_get( B%type ) )
     CASE ( 'DENSE' )
       DO j = 1, B%n
         PTR( j + 1 ) = B%n + 1 - j
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       PTR( 2 : B%n + 1 ) = 0
       DO i = 1, B%n
         DO l = B%ptr( i ), B%ptr( i + 1 ) - 1
           j = B%col( l ) + 1
           PTR( j ) = PTR( j ) + 1
         END DO
       END DO
     CASE ( 'COORDINATE' )
       PTR( 2 : B%n + 1 ) = 0
       DO l = 1, B%ne
         j = B%col( l ) + 1
         PTR( j ) = PTR( j ) + 1
       END DO
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
       PTR( 2 : B%n + 1 ) = 1
     END SELECT

!  set the starting addresses for each column of B

     PTR( 1 ) = 1
     DO i = 2, B%n + 1
       PTR( i ) = PTR( i ) + PTR( i - 1 )
     END DO

!  compute the column ordering of B

     SELECT CASE ( SMT_get( B%type ) )
     CASE ( 'DENSE' )
       l = 0
       DO i = 1, B%n
         DO j = 1, i
           l = l + 1
           ll = PTR( j )
           ROW( ll ) = i
           ORDER( ll ) = l
           PTR( j ) = ll + 1
         END DO
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO i = 1, B%n
         DO l = B%ptr( i ), B%ptr( i + 1 ) - 1
           j = B%col( l )
           ll = PTR( j )
           ROW( ll ) = i
           ORDER( ll ) = l
           PTR( j ) = ll + 1
         END DO
       END DO
     CASE ( 'COORDINATE' )
       DO l = 1, B%ne
         j = B%col( l )
         ll = PTR( j )
         ROW( ll ) = B%row( l )
         ORDER( ll ) = l
         PTR( j ) = ll + 1
       END DO
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
       DO l = 1, B%n
         j = l
         ll = PTR( j )
         ROW( ll ) = j
         ORDER( ll ) = l
         PTR( j ) = ll + 1
       END DO
     END SELECT

!  reset the starting addresses for each column of B

     DO i = B%n, 1, - 1
       PTR( i + 1 ) = PTR( i )
     END DO
     PTR( 1 ) = 1

!  for each column in turn, find the position in A of each entry of B

     IW( : B%m ) = 0

     IF ( b_in_a ) THEN
       IF ( SMT_get( A%type ) == 'SPARSE_BY_COLUMNS' .OR.                      &
            SMT_get( A%type ) == 'COORDINATE' ) THEN
         DO j = 1, A%n
           DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
             IW( A%row( l ) ) = l
           END DO
           DO l = PTR( j ), PTR( j + 1 ) - 1
             MAP( ORDER( l ) ) = IW( ROW( l ) )
           END DO
           DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
             IW( A%row( l ) ) = 0
           END DO
         END DO
       ELSE ! dense A
         ll = 0
         DO j = 1, A%n
           DO i = 1, A%m
             ll = ll + 1
             IW( i ) = ll
           END DO
           DO l = PTR( j ), PTR( j + 1 ) - 1
             MAP( ORDER( l ) ) = IW( ROW( l ) )
           END DO
           IW( : A%m ) = 0
         END DO
       END IF

!  for each column in turn, find the position in B of each entry of A

     ELSE
       b_dense = SMT_get( B%type ) == 'DENSE'
       ll = 0
       IF ( SMT_get( A%type ) == 'SPARSE_BY_COLUMNS' .OR.                      &
            SMT_get( A%type ) == 'COORDINATE' ) THEN
         DO j = 1, B%n
           DO l = PTR( j ), PTR( j + 1 ) - 1
             IW( ROW( l ) ) = ORDER( l )
           END DO
           IF ( b_dense ) THEN
             DO i = 1, B%m
               ll = ll + 1
               MAP( ll ) = IW( i )
             END DO
           ELSE
             DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
               MAP( l ) = IW( A%row( l ) )
             END DO
           END IF
           DO l = PTR( j ), PTR( j + 1 ) - 1
             IW( ROW( l ) ) = 0
           END DO
         END DO
       ELSE ! dense A by columns
         DO j = 1, B%n
           DO l = PTR( j ), PTR( j + 1 ) - 1
             IW( ROW( l ) ) = ORDER( l )
           END DO
           DO i = 1, B%m
             ll = ll + 1
             MAP( ll ) = IW( i )
           END DO
           DO l = PTR( j ), PTR( j + 1 ) - 1
             IW( ROW( l ) ) = 0
           END DO
         END DO
       END IF
     END IF

!  discard the workspace

     array_name = 'nls: PTR'
     CALL SPACE_dealloc_array( PTR, status, alloc_status,                      &
        array_name = array_name, bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'nls: ROW'
     CALL SPACE_dealloc_array( ROW, status, alloc_status,                      &
        array_name = array_name, bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'nls: ORDER'
     CALL SPACE_dealloc_array( ORDER, status, alloc_status,                    &
        array_name = array_name, bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'nls: IW'
     CALL SPACE_dealloc_array( IW, status, alloc_status,                       &
        array_name = array_name, bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     status = GALAHAD_ok
     RETURN

!  end of subroutine NLS_set_map

     END SUBROUTINE NLS_set_map

!-*-*-  G A L A H A D -  N L S _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE NLS_subproblem_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_subproblem_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLS_subproblem_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_subproblem_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

!  integer arrays

     array_name = 'nls: data%IW'
     CALL SPACE_dealloc_array( data%IW,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%PAST'
     CALL SPACE_dealloc_array( data%PAST,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%ROW'
     CALL SPACE_dealloc_array( data%ROW,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%PTR'
     CALL SPACE_dealloc_array( data%PTR,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%ORDER'
     CALL SPACE_dealloc_array( data%ORDER,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%H_map'
     CALL SPACE_dealloc_array( data%H_map,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%Hs_map'
     CALL SPACE_dealloc_array( data%Hs_map,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  real arrays

     array_name = 'nls: data%X_current'
     CALL SPACE_dealloc_array( data%X_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%C_current'
     CALL SPACE_dealloc_array( data%C_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%G_current'
     CALL SPACE_dealloc_array( data%G_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%W'
     CALL SPACE_dealloc_array( data%W,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%SX'
     CALL SPACE_dealloc_array( data%SX,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%SV'
     CALL SPACE_dealloc_array( data%SV,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%D_hist'
     CALL SPACE_dealloc_array( data%D_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%F_hist'
     CALL SPACE_dealloc_array( data%F_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%JS'
     CALL SPACE_dealloc_array( data%JS,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%HSHS'
     CALL SPACE_dealloc_array( data%HSHS,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%JT%row'
     CALL SPACE_dealloc_array( data%JT%row,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%JT%col'
     CALL SPACE_dealloc_array( data%JT%col,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%JT%val'
     CALL SPACE_dealloc_array( data%JT%val,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%H%row'
     CALL SPACE_dealloc_array( data%H%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%X'
     CALL SPACE_dealloc_array( data%tensor_model%X,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%G'
     CALL SPACE_dealloc_array( data%tensor_model%G,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%H%ptr'
     CALL SPACE_dealloc_array( data%tensor_model%H%ptr,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%H%row'
     CALL SPACE_dealloc_array( data%tensor_model%H%row,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%H%col'
     CALL SPACE_dealloc_array( data%tensor_model%H%col,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%tensor_model%H%val'
     CALL SPACE_dealloc_array( data%tensor_model%H%val,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%regularization%matrix%row'
     CALL SPACE_dealloc_array( data%regularization%matrix%row,                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%regularization%matrix%col'
     CALL SPACE_dealloc_array( data%regularization%matrix%col,                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%regularization%matrix%val'
     CALL SPACE_dealloc_array( data%regularization%matrix%val,                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within PSLS

     CALL PSLS_terminate( data%PSLS_data, data%control%PSLS_control,           &
                          inform%PSLS_inform )
     inform%status = inform%PSLS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%PSLS_inform%alloc_status
       inform%bad_alloc = inform%PSLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arrays allocated within BSC

     CALL BSC_terminate( data%BSC_data, control%BSC_control, inform%BSC_inform )
     inform%status = inform%BSC_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%BSC_inform%alloc_status
       inform%bad_alloc = inform%BSC_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arrays allocated within GLRT

     CALL GLRT_terminate( data%GLRT_data, data%control%GLRT_control,           &
                          inform%GLRT_inform )
     inform%status = inform%GLRT_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%GLRT_inform%alloc_status
       inform%bad_alloc = inform%GLRT_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arraysn allocated within RQS

     CALL RQS_terminate( data%RQS_data, data%control%RQS_control,              &
                          inform%RQS_inform )
     inform%status = inform%RQS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%RQS_inform%alloc_status
       inform%bad_alloc = inform%RQS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Close and delete 'alive' file

     IF ( control%alive_unit > 0 ) THEN
       INQUIRE( FILE = control%alive_file, EXIST = alive )
       IF ( alive .AND. control%alive_unit > 0 ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
         REWIND control%alive_unit
         CLOSE( control%alive_unit, STATUS = 'DELETE' )
       END IF
     END IF

     RETURN

!  End of subroutine NLS_terminate

     END SUBROUTINE NLS_subproblem_terminate

!-*-*-  G A L A H A D -  N L S _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE NLS_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLS_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_inform_type ), INTENT( INOUT ) :: inform

!  Deallocate all arrays allocated within calls to the main algorithm

     CALL NLS_subproblem_terminate( data%NLS_subproblem_data_type,             &
                                control%NLS_subproblem_control_type,           &
                                inform%NLS_subproblem_inform_type )
     IF ( inform%NLS_subproblem_inform_type%status /= GALAHAD_ok ) THEN
       inform%status = inform%NLS_subproblem_inform_type%status
       RETURN
     END IF

!  Deallocate all arrays allocated within calls to subproblem solutions

     CALL NLS_subproblem_terminate( data%subproblem_data,                      &
                                    control%subproblem_control,                &
                                    inform%subproblem_inform )
     RETURN

!  End of subroutine NLS_terminate

     END SUBROUTINE NLS_terminate

! -  G A L A H A D -  N L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE NLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLS_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL NLS_terminate( data%nls_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'nls: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%C'
     CALL SPACE_dealloc_array( data%nlp%C,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%W'
     CALL SPACE_dealloc_array( data%W,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%J%row'
     CALL SPACE_dealloc_array( data%nlp%J%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%J%col'
     CALL SPACE_dealloc_array( data%nlp%J%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%J%ptr'
     CALL SPACE_dealloc_array( data%nlp%J%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%J%val'
     CALL SPACE_dealloc_array( data%nlp%J%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%J%type'
     CALL SPACE_dealloc_array( data%nlp%J%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%H%row'
     CALL SPACE_dealloc_array( data%nlp%H%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%H%col'
     CALL SPACE_dealloc_array( data%nlp%H%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%H%ptr'
     CALL SPACE_dealloc_array( data%nlp%H%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%H%val'
     CALL SPACE_dealloc_array( data%nlp%H%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%H%type'
     CALL SPACE_dealloc_array( data%nlp%H%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%P%row'
     CALL SPACE_dealloc_array( data%nlp%P%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%P%col'
     CALL SPACE_dealloc_array( data%nlp%P%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%P%ptr'
     CALL SPACE_dealloc_array( data%nlp%P%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%P%val'
     CALL SPACE_dealloc_array( data%nlp%P%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'nls: data%nlp%P%type'
     CALL SPACE_dealloc_array( data%nlp%P%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine NLS_full_terminate

     END SUBROUTINE NLS_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  N L S _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE NLS_import( control, data, status, n, m,                       &
                            J_type, J_ne, J_row, J_col, J_ptr,                 &
                            H_type, H_ne, H_row, H_col, H_ptr,                 &
                            P_type, P_ne, P_row, P_col, P_ptr, W )

!  import fixed problem data into internal storage prior to solution. 
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading 
!   comments to NLS_solve
!
!  data is a scalar variable of type NLS_full_data_type used for internal data
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
!   -3. The restriction n > 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'ABSENT' has been violated.
!  -79. An optional array required by storage type H_type or P_type is missing
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m is a scalar variable of type default integer, that holds the number of
!   residuals
!
!  J_type is a character string that specifies the Jacobian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   or 'absent', the latter if access to the Jacobian is via matrix-vector 
!   products; lower or upper case variants are allowed
!
!  J_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set 
!  for any of the other schemes.
!
!  J_row is a rank-one array of type default integer, that holds the row 
!   indices J in the sparse co-ordinate storage scheme. It need not be set 
!   for any of the other schemes, and in this case can be of length 0
!
!  J_col is a rank-one array of type default integer, that holds the column 
!   indices of J in either the sparse co-ordinate, or the sparse row-wise 
!   storage scheme. It need not be set when the dense scheme is used, and 
!   in this case can be of length 0
!
!  J_ptr is a rank-one array of dimension n+1 and type default integer, 
!   that holds the starting position of each row of J, as well as the total 
!   number of entries plus one, in the sparse row-wise storage scheme. 
!   It need not be set when the other schemes are used, and in this case 
!   can be of length 0
!
!   ******************************************************************
!   ** NB The following H_ arguments are optional and need only be  **
!   **    supplied when the Newton or tensor-Newton method is used  **
!   ******************************************************************
!
!  H_type is a character string that specifies the Hessian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   'diagonal' or 'absent', the latter if access to the Hessian is via
!   matrix-vector products; lower or upper case variants are allowed.
!
!  H_ne is a scalar variable of type default integer, that holds the number of
!   entries in the  lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other three schemes.
!
!  H_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  H_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense or diagonal storage schemes are used, and in this 
!   case can be of length 0
!
!  H_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the  lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!   **************************************************************
!   ** NB The following P_ arguments are optional and need only **
!   **    be supplied when the tensor-Newton method is used     **
!   **************************************************************
!
!  P_type is a character string that specifies the residual-Hessians-vector
!   product matrix storage scheme used. It should be one of 'dense_by_columns, 
!   'coordinate', 'sparse_by_columns', or 'absent', the latter if access to 
!   the Jacobian is via matrix-vector products; lower or upper case variants 
!   are allowed **NB 'coordinate' has not yet been implemented **
!
!  P_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set 
!  for any of the other schemes.
!
!  P_row is a rank-one array of type default integer, that holds the row 
!   indices of P in either the sparse co-ordinate, or the sparse column-wise 
!   storage scheme. It need not be set when the dense scheme is used, and 
!   in this case can be of length 0
!
!  P_col is a rank-one array of type default integer, that holds the column
!   indices P in the sparse co-ordinate storage scheme. It need not be set 
!   for any of the other schemes, and in this case can be of length 0
!
!  P_ptr is a rank-one array of dimension n+1 and type default integer, 
!   that holds the starting position of each column of P, as well as the total 
!   number of entries plus one, in the sparse column-wise storage scheme. 
!   It need not be set when the other schemes are used, and in this case 
!   can be of length 0
!
!  W is an optional rank-one array of dimension m and type default
!   real, that holds the vector of weights w attached to the residuals
!   in the least-squares objective function.
!   If W is present, the i-th component of W, i = 1, ... , m, contains (w)i.
!   If W is not present, weights of one will be used

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: n, m, J_ne
     INTEGER, INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: J_type
     INTEGER, DIMENSION( : ), INTENT( IN ) :: J_row, J_col, J_ptr
     INTEGER, INTENT( IN ), OPTIONAL :: H_ne, P_ne
     CHARACTER ( LEN = * ), INTENT( IN ), OPTIONAL :: H_type
     INTEGER, DIMENSION( : ), INTENT( IN ), OPTIONAL :: H_row, H_col, H_ptr
     CHARACTER ( LEN = * ), INTENT( IN ), OPTIONAL :: P_type
     INTEGER, DIMENSION( : ), INTENT( IN ), OPTIONAL :: P_row, P_col, P_ptr
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( m ), OPTIONAL :: W

!  local variables

     INTEGER :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     LOGICAL :: newton, tensor_newton
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     data%nls_control = control

!  check for the expected matrices

     IF ( PRESENT( H_type ) ) THEN
       newton = .TRUE.
       IF ( PRESENT( P_type ) ) THEN
         tensor_newton = .TRUE.
       ELSE
         tensor_newton = .FALSE.
       END IF
     ELSE
       newton = .FALSE.
       tensor_newton = .FALSE.
     END IF

     error = data%nls_control%error
     space_critical = data%nls_control%space_critical
     deallocate_error_fatal = data%nls_control%deallocate_error_fatal

!  allocate space if required

     array_name = 'nls: data%nlp%X'
     CALL SPACE_resize_array( n, data%nlp%X,                                   &
            data%nls_inform%status, data%nls_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%nls_inform%bad_alloc, out = error )
     IF ( data%nls_inform%status /= 0 ) GO TO 900

     array_name = 'nls: data%nlp%C'
     CALL SPACE_resize_array( m, data%nlp%C,                                   &
            data%nls_inform%status, data%nls_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%nls_inform%bad_alloc, out = error )
     IF ( data%nls_inform%status /= 0 ) GO TO 900

     array_name = 'nls: data%nlp%G'
     CALL SPACE_resize_array( n, data%nlp%G,                                   &
            data%nls_inform%status, data%nls_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%nls_inform%bad_alloc, out = error )
     IF ( data%nls_inform%status /= 0 ) GO TO 900

!  put data into the required components of the nlpt storage type

     data%nlp%n = n ; data%nlp%m = m

!  set J appropriately in the nlpt storage type

     SELECT CASE ( J_type )
     CASE ( 'coordinate', 'COORDINATE' )
       CALL SMT_put( data%nlp%J%type, 'COORDINATE',                            &
                     data%nls_inform%alloc_status )
       data%nlp%J%n = n ; data%nlp%J%m = m
       data%nlp%J%ne = J_ne

       array_name = 'nls: data%nlp%J%row'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%row,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       array_name = 'nls: data%nlp%J%col'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%col,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       array_name = 'nls: data%nlp%J%val'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%val,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       data%nlp%J%row( : data%nlp%J%ne ) = J_row( : data%nlp%J%ne )
       data%nlp%J%col( : data%nlp%J%ne ) = J_col( : data%nlp%J%ne )

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       CALL SMT_put( data%nlp%J%type, 'SPARSE_BY_ROWS',                        &
                     data%nls_inform%alloc_status )
       data%nlp%J%n = n ; data%nlp%J%m = m
       data%nlp%J%ne = J_ptr( m + 1 ) - 1
       array_name = 'nls: data%nlp%J%ptr'
       CALL SPACE_resize_array( m + 1, data%nlp%J%ptr,                         &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       array_name = 'nls: data%nlp%J%col'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%col,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       array_name = 'nls: data%nlp%J%val'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%val,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

       data%nlp%J%ptr( : m + 1 ) = J_ptr( : m + 1 )
       data%nlp%J%col( : data%nlp%J%ne ) = J_col( : data%nlp%J%ne )

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%nlp%J%type, 'DENSE',                                 &
                     data%nls_inform%alloc_status )
       data%nlp%J%n = n ; data%nlp%J%m = m
       data%nlp%J%ne = m * n

       array_name = 'nls: data%nlp%J%val'
       CALL SPACE_resize_array( data%nlp%J%ne, data%nlp%J%val,                 &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900

     CASE ( 'absent', 'ABSENT' )
       data%nls_control%jacobian_available = 1
     CASE DEFAULT
       data%nls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT       

!  if present, set H appropriately in the nlpt storage type

     IF ( newton ) THEN
       SELECT CASE ( H_type )
       CASE ( 'coordinate', 'COORDINATE' )
         IF ( .NOT. ( PRESENT( H_row ) .AND. PRESENT( H_col ) ) ) THEN
           data%nls_inform%status = GALAHAD_error_optional
           GO TO 900
         END IF
         CALL SMT_put( data%nlp%H%type, 'COORDINATE',                          &
                       data%nls_inform%alloc_status )
         data%nlp%H%n = n
         data%nlp%H%ne = H_ne

         array_name = 'nls: data%nlp%H%row'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%row,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%H%col'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%H%val'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         data%nlp%H%row( : data%nlp%H%ne ) = H_row( : data%nlp%H%ne )
         data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )

       CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
         IF ( .NOT. ( PRESENT( H_ptr ) .AND. PRESENT( H_col ) ) ) THEN
           data%nls_inform%status = GALAHAD_error_optional
           GO TO 900
         END IF
         CALL SMT_put( data%nlp%H%type, 'SPARSE_BY_ROWS',                      &
                       data%nls_inform%alloc_status )
         data%nlp%H%n = n
         data%nlp%H%ne = H_ptr( n + 1 ) - 1

         array_name = 'nls: data%nlp%H%ptr'
         CALL SPACE_resize_array( n + 1, data%nlp%H%ptr,                       &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%H%col'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%H%val'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         data%nlp%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
         data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )

       CASE ( 'dense', 'DENSE' )
         CALL SMT_put( data%nlp%H%type, 'DENSE',                               &
                       data%nls_inform%alloc_status )
         data%nlp%H%n = n
         data%nlp%H%ne = ( n * ( n + 1 ) ) / 2

         array_name = 'nls: data%nlp%H%val'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

       CASE ( 'diagonal', 'DIAGONAL' )
         CALL SMT_put( data%nlp%H%type, 'DIAGONAL',                            &
                       data%nls_inform%alloc_status )
         data%nlp%H%n = n
         data%nlp%H%ne = n

         array_name = 'nls: data%nlp%H%val'
         CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

       CASE ( 'absent', 'ABSENT' )
         data%nls_control%hessian_available = 1
       CASE DEFAULT
         data%nls_inform%status = GALAHAD_error_unknown_storage
         GO TO 900
       END SELECT       
     END IF

!  if present, set P appropriately in the nlpt storage type

     IF ( tensor_newton ) THEN
       SELECT CASE ( P_type )
       CASE ( 'coordinate', 'COORDINATE' )
         IF ( .NOT. ( PRESENT( P_row ) .AND. PRESENT( P_col ) ) ) THEN
           data%nls_inform%status = GALAHAD_error_optional
           GO TO 900
         END IF
         CALL SMT_put( data%nlp%J%type, 'COORDINATE',                          &
                       data%nls_inform%alloc_status )
         data%nlp%P%n = m ; data%nlp%P%m = n
         data%nlp%P%ne = P_ne

         array_name = 'nls: data%nlp%P%row'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%row,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%P%col'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%col,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%P%val'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         data%nlp%P%row( : data%nlp%P%ne ) = P_row( : data%nlp%P%ne )
         data%nlp%P%col( : data%nlp%P%ne ) = P_col( : data%nlp%P%ne )

       CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
         IF ( .NOT. ( PRESENT( P_ptr ) .AND. PRESENT( P_row ) ) ) THEN
           data%nls_inform%status = GALAHAD_error_optional
           GO TO 900
         END IF
         CALL SMT_put( data%nlp%P%type, 'SPARSE_BY_COLUMNS',                   &
                       data%nls_inform%alloc_status )
         data%nlp%P%n = m ; data%nlp%P%m = n
         data%nlp%P%ne = P_ptr( m + 1 ) - 1

         array_name = 'nls: data%nlp%P%ptr'
         CALL SPACE_resize_array( m + 1, data%nlp%P%ptr,                       &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%P%row'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%row,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                  &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         array_name = 'nls: data%nlp%P%val'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

         data%nlp%P%ptr( : m + 1 ) = P_ptr( : m + 1 )
         data%nlp%P%row( : data%nlp%P%ne ) = P_row( : data%nlp%P%ne )

       CASE ( 'dense', 'DENSE', 'dense_by_columns', 'DENSE_BY_COLUMNS' )
         CALL SMT_put( data%nlp%P%type, 'DENSE_BY_COLUMNS',                    &
                       data%nls_inform%alloc_status )
         data%nlp%P%n = m ; data%nlp%P%m = n
         data%nlp%P%ne = m * n

         array_name = 'nls: data%nlp%P%val'
         CALL SPACE_resize_array( data%nlp%P%ne, data%nlp%P%val,               &
                data%nls_inform%status, data%nls_inform%alloc_status,          &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
                bad_alloc = data%nls_inform%bad_alloc, out = error )
         IF ( data%nls_inform%status /= 0 ) GO TO 900

       CASE ( 'absent', 'ABSENT' )
       CASE DEFAULT
         data%nls_inform%status = GALAHAD_error_unknown_storage
         GO TO 900
       END SELECT
     END IF

!  save non-trivial weights

     IF ( PRESENT( W ) ) THEN
       array_name = 'nls: data%W'
       CALL SPACE_resize_array( m, data%W,                                     &
              data%nls_inform%status, data%nls_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nls_inform%bad_alloc, out = error )
       IF ( data%nls_inform%status /= 0 ) GO TO 900
       data%W( : m ) = W( : m )
     END IF

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%nls_inform%status
     RETURN

!  End of subroutine NLS_import

     END SUBROUTINE NLS_import

!-  G A L A H A D -  N L S _ r e s e t _ c o n t r o l   S U B R O U T I N E  -

     SUBROUTINE NLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See NLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_control_type ), INTENT( IN ) :: control
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: status

!  set control in internal data

     data%nls_control = control
     
!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine NLS_reset_control

     END SUBROUTINE NLS_reset_control

!-  G A L A H A D -  N L S _ s o l v e _ w i t h _ m a t  S U B R O U T I N E  -

     SUBROUTINE NLS_solve_with_mat( data, userdata, status, X, C, G,           &
                                    eval_C, eval_J, eval_H, eval_HPRODS )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual, Jacobian, Hessian and residual-Hessians vector product 
!  operations are available via subroutine calls. See NLS_solve for a 
!  description of the required arguments. The variable status is a proxy 
!  for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: C
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_C, eval_J, eval_H, eval_HPRODS
     OPTIONAL :: eval_H, eval_HPRODS

     data%nls_inform%status = status
     IF ( data%nls_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     IF ( ALLOCATED( data%W ) ) THEN
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, userdata, W = data%W,                    &
                       eval_C = eval_C, eval_J = eval_J,                       &
                       eval_H = eval_H, eval_HPRODS = eval_HPRODS )
     ELSE
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, userdata,                                &
                       eval_C = eval_C, eval_J = eval_J,                       &
                       eval_H = eval_H, eval_HPRODS = eval_HPRODS )
     END IF

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     C( : data%nlp%m ) = data%nlp%C( : data%nlp%m )
     G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     status = data%nls_inform%status

     RETURN

!  end of subroutine NLS_solve_with_mat

     END SUBROUTINE NLS_solve_with_mat

! - G A L A H A D -  N L S _ s o l v e _ without _ m a t  S U B R O U T I N E -

     SUBROUTINE NLS_solve_without_mat( data, userdata, status, X, C, G,        &
                                       eval_C, eval_JPROD, eval_HPROD,         &
                                       eval_HPRODS )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual, Jacobian, Hessian-vector and residual-Hessians vector product 
!  operations are available via subroutine calls. See NLS_solve for a 
!  description of the required arguments. The variable status is a proxy 
!  for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: C
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_C, eval_JPROD, eval_HPROD, eval_HPRODS
     OPTIONAL :: eval_HPROD, eval_HPRODS

     data%nls_inform%status = status
     IF ( data%nls_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     IF ( ALLOCATED( data%W ) ) THEN
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, userdata, W = data%W,                    &
                       eval_C = eval_C, eval_JPROD = eval_JPROD,               &
                       eval_HPROD = eval_HPROD, eval_HPRODS = eval_HPRODS )
     ELSE
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, userdata,                                &
                       eval_C = eval_C, eval_JPROD = eval_JPROD,               &
                       eval_HPROD = eval_HPROD, eval_HPRODS = eval_HPRODS )
     END IF

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     C( : data%nlp%m ) = data%nlp%C( : data%nlp%m )
     G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     status = data%nls_inform%status

     RETURN

!  end of subroutine NLS_solve_without_mat

     END SUBROUTINE NLS_solve_without_mat

!-  G A L A H A D -  N L S _ s o l v e _ reverse _ M A T  S U B R O U T I N E -

     SUBROUTINE NLS_solve_reverse_with_mat( data, status, eval_status,         &
                                            X, C, G, J_val, Y, H_val, V, P_val )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual, Jacobian, Hessian and residual-Hessians vector product 
!  operations are available via reverse communications. See NLS_solve for a 
!  description of the required arguments. The variable status is a proxy 
!  for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( INOUT ) :: eval_status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: C
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: J_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ), OPTIONAL :: Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ), OPTIONAL :: H_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ), OPTIONAL :: V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ), OPTIONAL :: P_val

!  recover data from reverse communication

     data%nls_inform%status = status
     data%nls_data%eval_status = eval_status
     SELECT CASE ( data%nls_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 ) data%nlp%C( : data%nlp%m ) = C( : data%nlp%m )
     CASE( 3 ) 
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%J%val( : data%nlp%J%ne ) = J_val( : data%nlp%J%ne )
     CASE( 4 ) 
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%H%val( : data%nlp%H%ne ) = H_val( : data%nlp%H%ne )
     CASE( 7 ) 
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%P%val( : data%nlp%P%ne ) = P_val( : data%nlp%P%ne )
     END SELECT

!  call the solver

     IF ( ALLOCATED( data%W ) ) THEN
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, data%userdata, W = data%W )
     ELSE
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, data%userdata )
     END IF

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%nls_inform%status )
     CASE( 0 )
       C( : data%nlp%m ) = data%nlp%C( : data%nlp%m )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 4 )
       Y( : data%nlp%m ) = data%nls_data%Y( : data%nlp%m )
     CASE( 7 )
       V( : data%nlp%n ) = data%nls_data%V( : data%nlp%n )
     CASE( 5, 6 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%nls_inform%status
     END SELECT
     status = data%nls_inform%status

     RETURN

!  end of subroutine NLS_solve_reverse_with_mat

     END SUBROUTINE NLS_solve_reverse_with_mat

!-  G A L A H A D -  N L S _ s o l v e _ reverse _ no _ mat  S U B R O U T I N E

     SUBROUTINE NLS_solve_reverse_without_mat( data, status, eval_status,      &
                                               X, C, G, transpose, U, V,       &
                                               Y, P_val )
                                               

!  solve the nonlinear least-squares problem previously imported when access
!  to residual, Jacobian, Hessian-vector and residual-Hessians vector product 
!  operations are available via reverse communications. See NLS_solve for a 
!  description of the required arguments. The variable status is a proxy 
!  for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( INOUT ) :: eval_status
     LOGICAL, INTENT( INOUT ) :: transpose
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: C
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ), OPTIONAL :: Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ), OPTIONAL :: P_val

!  recover data from reverse communication

     data%nls_inform%status = status
     data%nls_data%eval_status = eval_status
     SELECT CASE ( data%nls_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 ) data%nlp%C( : data%nlp%m ) = C( : data%nlp%m )
     CASE( 5 ) 
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         IF ( data%nls_data%transpose ) THEN
           data%nls_data%U( : data%nlp%n ) = U( : data%nlp%n )
         ELSE
           data%nls_data%U( : data%nlp%m ) = U( : data%nlp%m )
         END IF
       END IF
     CASE( 6 )
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nls_data%U( : data%nlp%n ) = U( : data%nlp%n )
     CASE( 7 ) 
       data%nls_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%P%val( : data%nlp%P%ne ) = P_val( : data%nlp%P%ne )
     END SELECT

!  call the solver

     IF ( ALLOCATED( data%W ) ) THEN
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, data%userdata, W = data%W )
     ELSE
       CALL NLS_solve( data%nlp, data%nls_control, data%nls_inform,            &
                       data%nls_data, data%userdata )
     END IF

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%nls_inform%status )
     CASE( 0 )
       C( : data%nlp%m ) = data%nlp%C( : data%nlp%m )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 2 ) 
     CASE( 3, 4 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%nls_inform%status
     CASE( 5 )
       transpose = data%nls_data%transpose
       IF ( transpose ) THEN
         U( : data%nlp%n ) = data%nls_data%U( : data%nlp%n )
         V( : data%nlp%m ) = data%nls_data%V( : data%nlp%m )
       ELSE
         U( : data%nlp%m ) = data%nls_data%U( : data%nlp%m )
         V( : data%nlp%n ) = data%nls_data%V( : data%nlp%n )
       END IF
     CASE( 6 )
       Y( : data%nlp%m ) = data%nls_data%Y( : data%nlp%m )
       V( : data%nlp%n ) = data%nls_data%V( : data%nlp%n )
     CASE( 7 )
       V( : data%nlp%n ) = data%nls_data%V( : data%nlp%n )
     END SELECT
     status = data%nls_inform%status

     RETURN

!  end of subroutine NLS_solve_reverse_without_mat

     END SUBROUTINE NLS_solve_reverse_without_mat

!-  G A L A H A D -  N L S _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE NLS_information( data, inform, status )

!  return solver information during or after solution by NLS
!  See NLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%nls_inform
     
!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine NLS_information

     END SUBROUTINE NLS_information

!  End of module GALAHAD_NLS

   END MODULE GALAHAD_NLS_double
