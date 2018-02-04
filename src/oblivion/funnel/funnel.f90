! THIS VERSION: GALAHAD 2.6 - 13/02/2014 AT 09:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ F U N N E L   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.1, October 17th 2007 for equalities
!   version for inequalities  GALAHAD Version 2.6, July 13th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FUNNEL_double

!     ----------------------------------------------------------
!    |                                                          |
!    | FUNNEL, a trust-funnel method for nonlinear optimization |
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
     USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_are
     USE GALAHAD_SPACE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SBLS_double
     USE GALAHAD_LLS_double
     USE GALAHAD_LLST_double
     USE GALAHAD_EQP_double
     USE GALAHAD_TRS_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM
     USE GALAHAD_ROOTS_double, ONLY: ROOTS_quadratic

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: FUNNEL_initialize, FUNNEL_read_specfile, FUNNEL_solve,          &
               FUNNEL_terminate, NLPT_problem_type

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

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point8 = 0.8_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: y_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: z_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: mu_tol = point1

!    LOGICAL, PARAMETER :: print_debug = .TRUE.
     LOGICAL, PARAMETER :: print_debug = .FALSE.

!    LOGICAL, PARAMETER :: nop = .TRUE.
     LOGICAL, PARAMETER :: nop = .FALSE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FUNNEL_control_type

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

!   variant of algorithm used:
!     1 = squared violation (RAL-TR-2014-001)
!     2 = submitted (RAL-P-2014-001)
!     3 = nopi_2 (NAXYS-02-2014)

       INTEGER :: algorithm_variant = 3

!   any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = wp ) :: infinity = ten ** 19

!   overall convergence tolerances. The iteration will terminate when the norm
!    of violation of the constraints (the "primal infeasibility") is smaller
!    than stop_p, the norm of the gradient of the Lagrangian function (the
!    "dual infeasibility") is smaller than stop_d, and the norm of the
!    complementary slackness is smaller than stop_c

!   the required absolute and relative accuracies for the primal infeasibility

       REAL ( KIND = wp ) :: stop_abs_p = epsmch
       REAL ( KIND = wp ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

       REAL ( KIND = wp ) :: stop_abs_d = epsmch
       REAL ( KIND = wp ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

       REAL ( KIND = wp ) :: stop_abs_c = epsmch
       REAL ( KIND = wp ) :: stop_rel_c = epsmch

!   the required absolute and relative accuracies for the infeasibility
!   The iteration will stop at a minimizer of the infeasibility if the
!   gradient of the infeasibility (J^T c) is smaller in norm than
!   control%stop_abs_i times the norm of c

       REAL ( KIND = wp ) :: stop_abs_i = epsmch
       REAL ( KIND = wp ) :: stop_rel_i = epsmch

!   the initial value of the barrier parameter. If mu_initial is not positive,
!    it will be reset to an appropriate value

       REAL ( KIND = wp ) :: mu_initial = - one

!  initial values for the trust-region radiii for the objective and contraint
!  models

       REAL ( KIND = wp ) :: initial_t_model_radius = ten
       REAL ( KIND = wp ) :: initial_n_model_radius = ten

!   initial variables will not be closer than min_feas_p from their bounds

        REAL ( KIND = wp ) :: min_feas_p = one

!   initial dual variables will not be closer than min_feas_d from their bounds
!
        REAL ( KIND = wp ) :: min_feas_d = one

!   the smallest value of the barrier parameter allowed. If mu_smallest
!    is not positive, the barrier parameter will not be restricted

       REAL ( KIND = wp ) :: mu_smallest = - one

!   a potential point whose linear decrease predicted by the model will only
!    be accepted if the actual decrease f - f(x_new) is larger than
!    eta_successful times that predicted by a quadratic model of the decrease

       REAL ( KIND = wp ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = wp ) :: eta_very_successful = point9

!   amount by which the radii are increased and decreased following a
!    very successful and unsuccessful iteration, respectively

       REAL ( KIND = wp ) :: radius_increase = two
       REAL ( KIND = wp ) :: radius_decrease = half

!  choose 0 < eta_1 <= eta_2 < 1, 0 < gamma_1 <= gamma_2 < 1, ...

       REAL ( KIND = wp ) :: eta_1 =  ten ** ( - 8 )
       REAL ( KIND = wp ) :: eta_2 = point9
       REAL ( KIND = wp ) :: gamma_1 = half
       REAL ( KIND = wp ) :: gamma_2 = point9

!  ... { kappa_ca, kappa_n, kappa_y, kappa_nr, kappa_D } in (0,infty), ...

       REAL ( KIND = wp ) :: kappa_ca = 1000.0_wp
       REAL ( KIND = wp ) :: kappa_y = HUGE( one )
       REAL ( KIND = wp ) :: kappa_nr = one
       REAL ( KIND = wp ) :: kappa_D = ten ** 10

!  ... { kappa_vv, kappa_fbn, kappa_B, kappa_fbt, kappa_tt, kappa_tg, kappa_chi,
!        kappa_deltavv, kappa_delta, kappa_t1, kappa_t2} in (0,1), ...

       REAL ( KIND = wp ) :: kappa_vv = point1
       REAL ( KIND = wp ) :: kappa_fbn = point01
       REAL ( KIND = wp ) :: kappa_fbt = point01
       REAL ( KIND = wp ) :: kappa_delta = point1
       REAL ( KIND = wp ) :: kappa_B = point9
       REAL ( KIND = wp ) :: kappa_tt = point9
       REAL ( KIND = wp ) :: kappa_tg = point1
       REAL ( KIND = wp ) :: kappa_chi = point1
       REAL ( KIND = wp ) :: kappa_deltavv = half
       REAL ( KIND = wp ) :: kappa_t1 = point9
       REAL ( KIND = wp ) :: kappa_t2 = half

!  ... kappa_cd in (0,1-kappa_tg), ...

       REAL ( KIND = wp ) :: kappa_cd = point8

!  ... kappa_n >= 1 (>0 for algorithm_variant = 3) ...

       REAL ( KIND = wp ) :: kappa_n = one

!   ... and { kappa_cr, kappa_v, kappa_VS } in (1,infty)

       REAL ( KIND = wp ) :: kappa_cr = two
       REAL ( KIND = wp ) :: kappa_v = two
       REAL ( KIND = wp ) :: kappa_VS = two

!  solve the normal model using factorization

       LOGICAL :: direct_solution_of_normal_model = .FALSE.

!  solve the tangential model using factorization

       LOGICAL :: direct_solution_of_tangential_model = .FALSE.

!  allow extrapolation

       LOGICAL :: allow_extrapolatiion = .FALSE.

!  use a second-order correction if necessary

       LOGICAL :: use_second_order_correction = .FALSE.

!   fulsol specifies whether the full solution or only highlights will be
!    printed

       LOGICAL :: fulsol = .TRUE.

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!   is a normal step almost always worth computing?

       LOGICAL :: n_deemed_desirable  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for LLST

       TYPE ( LLST_control_type ) :: LLST_control

!  control parameters for LLS

       TYPE ( LLS_control_type ) :: LLS_control
       TYPE ( LLS_control_type ) :: LLS_control_n
       TYPE ( LLS_control_type ) :: LLS_control_y
       TYPE ( LLS_control_type ) :: LLS_control_s

!  control parameters for EQP

       TYPE ( EQP_control_type ) :: EQP_control

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: TRS_control

!  control parameters for SBLS

       TYPE ( SBLS_control_type ) :: SBLS_control

     END TYPE FUNNEL_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FUNNEL_time_type

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
     END TYPE FUNNEL_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FUNNEL_inform_type

!  return status. See FUNNEL_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of CG iterations performed

       INTEGER :: cg_iter = 0

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the number of factorizations used

       INTEGER :: factorizations_normal = 0
       INTEGER :: factorizations_tangential = 0

!  the number of factorizations that modified the original matrix

       INTEGER :: modifications = 0

!  the return status from the factorization

       INTEGER :: factorization_status = 0

!   the maximum number of entries in the factors

       INTEGER ( KIND = long ) :: max_entries_factors_normal = 0
       INTEGER ( KIND = long ) :: max_entries_factors_multipliers = 0
       INTEGER ( KIND = long ) :: max_entries_factors_tangential = 0

!  the total integer workspace required for the factorization

       INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

       INTEGER :: factorization_real = - 1

!  the number of threads used

       INTEGER :: threads = 1

!  the value of the objective function at the best estimate of the solution
!   determined by FUNNEL_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

       REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

       REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

       REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  timings (see above)

       TYPE ( FUNNEL_time_type ) :: time

!  inform parameters for LLST

       TYPE ( LLST_inform_type ) :: LLST_inform

!  inform parameters for LLS

       TYPE ( LLS_inform_type ) :: LLS_inform
       TYPE ( LLS_inform_type ) :: LLS_inform_n
       TYPE ( LLS_inform_type ) :: LLS_inform_y
       TYPE ( LLS_inform_type ) :: LLS_inform_s

!  inform parameters for SBLS

       TYPE ( SBLS_inform_type ) :: SBLS_inform

!  inform parameters for TRS

       TYPE ( TRS_inform_type ) :: TRS_inform

!  inform parameters for EQP

       TYPE ( EQP_inform_type ) :: EQP_inform

     END TYPE FUNNEL_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: FUNNEL_data_type
       INTEGER :: m_in, m_eq, n_fr, n_total, branch, eval_status
       INTEGER :: out, print_level, start_print, stop_print
       INTEGER :: print_level_lls, print_level_lls_sbls, print_level_lls_gltr
       INTEGER :: print_level_eqp, print_level_eqp_sbls, print_level_eqp_gltr
       INTEGER :: print_level_trs, print_level_sbls
       INTEGER :: print_level_llst, print_level_llst_sbls
       REAL ( KIND = wp ) :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
       REAL ( KIND = wp ) :: violation, violation_plus, violation_max
       REAL ( KIND = wp ) :: pi_f, pi_v, radius, radius_n, radius_t, f_current
       REAL ( KIND = wp ) :: xi_v, eta_successful, eta_very_successful
       REAL ( KIND = wp ) :: gamma_1, gamma_2
       REAL ( KIND = wp ) :: radius_t_0, radius_n_0, kappa_ca, kappa_n, kappa_y
       REAL ( KIND = wp ) :: kappa_nr, kappa_delta, kappa_vv
       REAL ( KIND = wp ) :: omkappa_fbn, omkappa_fbt, kappa_B, kappa_tt
       REAL ( KIND = wp ) :: kappa_deltavv, kappa_t1, kappa_t2, kappa_cd
       REAL ( KIND = wp ) :: kappa_cr, kappa_v, kappa_VS, kappa_tg, kappa_chi
       REAL ( KIND = wp ) :: radius_increase, radius_decrease
       REAL ( KIND = wp ) :: mu, barrier, barrier_new, rho, barrier_terms
       REAL ( KIND = wp ) :: mv0, mvn, mvnc, mvd, mf0, mfn, mfd, mfdc
       REAL ( KIND = wp ) :: delta_mvnc, delta_mvn, delta_mvd
       REAL ( KIND = wp ) :: delta_mftc, delta_mfn, delta_mft, delta_mfd
       REAL ( KIND = wp ) :: norm_c, norm_n, norm_t, norm_c_plus, mu_smallest
       REAL ( KIND = wp ) :: stop_p, stop_d, stop_c, max_y
       REAL ( KIND = wp ) :: stop_p_inner, stop_d_inner, stop_c_inner
       LOGICAL :: set_printt, set_printi, set_printw, set_printd
       LOGICAL :: set_printm, printe, printi, printt, printm, printw, printd
       LOGICAL :: print_iteration_header, print_1st_header, start_inner
       LOGICAL :: reverse_fc, reverse_gj, reverse_hl, reverse_hlprod, acc
       LOGICAL :: new_gradient, new_hessian, n_eq_0, t_eq_0, fixed_variables
       LOGICAL :: use_violation_squared, tau_flag_raised
       CHARACTER ( LEN = 1 ) :: it_type, success, n_end, t_end, s_end
       TYPE ( TRS_history_type ), DIMENSION( history_max ) :: history
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_in, C_eq
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_bound_type, C_bound_type
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_order, X_fr
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_diag
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_n
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_plus
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CAUCHY
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: T
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: N
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_plus
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_soc
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_plus
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_soc
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_mod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Cv
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_mf
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_barrier
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       TYPE ( SMT_type ) :: C0, HESS_barrier, Jv, Pinv2
       TYPE ( LLST_data_type ) :: LLST_data
       TYPE ( LLS_data_type ) :: LLS_data
       TYPE ( EQP_data_type ) :: EQP_data
       TYPE ( TRS_data_type ) :: TRS_data
       TYPE ( SBLS_data_type ) :: SBLS_data
       TYPE ( LLST_control_type ) :: LLST_control
       TYPE ( LLS_control_type ) :: LLS_control
       TYPE ( EQP_control_type ) :: EQP_control
       TYPE ( TRS_control_type ) :: TRS_control
       TYPE ( SBLS_control_type ) :: SBLS_control

     END TYPE FUNNEL_data_type

   CONTAINS

!-*  G A L A H A D -  F U N N E L _ I N I T I A L I Z E  S U B R O U T I N E  *-

     SUBROUTINE FUNNEL_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for FUNNEL controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FUNNEL_data_type ), INTENT( OUT ) :: data
     TYPE ( FUNNEL_control_type ), INTENT( OUT ) :: control
     TYPE ( FUNNEL_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  redefine absolute stopping tolerances

     control%stop_abs_p = tenm5
     control%stop_abs_d = tenm5
     control%stop_abs_c = tenm5
     control%stop_abs_i = tenm5

!  Intialize LLST data

     CALL LLST_initialize( data%LLST_data, control%LLST_control,               &
                           inform%LLST_inform )
     control%LLST_control%prefix = '" - LLST:"                    '
     control%LLST_control%SBLS_control%prefix = '" -- SBLS:"                   '

!  Intialize LLS data

     CALL LLS_initialize( data%LLS_data, control%LLS_control,                  &
                          inform%LLS_inform )
     control%LLS_control%prefix = '" - LLS:"                     '
     control%LLS_control%SBLS_control%prefix = '" -- SBLS:"                   '

!  Intialize EQP data

     CALL EQP_initialize( data%EQP_data, control%EQP_control,                  &
                          inform%EQP_inform )
     control%EQP_control%prefix = '" - EQP:"                     '
     control%EQP_control%SBLS_control%prefix = '" -- SBLS:"                   '

!  initalize TRS components

     CALL TRS_initialize( data%TRS_data, control%TRS_control,                  &
                          inform%TRS_inform )
     control%TRS_control%prefix = '" - TRS:"                     '

!  Initalize SBLS components

     CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
     control%SBLS_control%prefix =  '" - SBLS:"                    '

     RETURN

!  End of subroutine FUNNEL_initialize

     END SUBROUTINE FUNNEL_initialize

!-*-*-   F U N N E L _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE FUNNEL_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by FUNNEL_initialize could (roughly)
!  have been set as:

! BEGIN FUNNEL SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    1
!  start-print                                    -1
!  stop-print                                     -1
!  iterations-between-printing                    1
!  maximum-number-of-iterations                   50
!  algorithm-variant-used                         3
!  infinity-value                                 1.0D+19
!  absolute-primal-accuracy                       6.0D-6
!  relative-primal-accuracy                       2.0D-16
!  absolute-dual-accuracy                         6.0D-6
!  relative-dual-accuracy                         2.0D-16
!  absolute-complementary-slackness-accuracy      6.0D-6
!  relative-complementary-slackness-accuracy      2.0D-16
!  absolute-infeasiblity-tolerated                6.0D-6
!  relative-infeasiblity-tolerated                2.0D-16
!  initial-barrier-parameter                      -1.0
!  mininum-initial-primal-feasibility             1.0
!  mininum-initial-dual-feasibility               1.0
!  initial-n-model-radius                         1.0D+1
!  initial-t-model-radius                         1.0D+1
!  successful-iteration-tolerance                 0.01
!  very-successful-iteration-tolerance            0.9
!  smallest-barrier-parameter                     -1.0
!  direct-solution-of-normal-model                no
!  direct-solution-of-tangential-model            no
!  allow-extrapolation                            no
!  use-second-order-correction                    no
!  print-full-solution                            no
!  space-critical                                 no
!  deallocate-error-fatal                         no
!  alive-filename                                 ALIVE.d
!  output-line-prefix                             ""
! END FUNNEL SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FUNNEL_control_type ), INTENT( INOUT ) :: control
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
     INTEGER, PARAMETER :: algorithm_variant = maxit + 1
     INTEGER, PARAMETER :: infinity = algorithm_variant + 1
     INTEGER, PARAMETER :: stop_abs_p = infinity + 1
     INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
     INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
     INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
     INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
     INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
     INTEGER, PARAMETER :: stop_abs_i = stop_rel_c + 1
     INTEGER, PARAMETER :: stop_rel_i = stop_abs_i + 1
     INTEGER, PARAMETER :: min_feas_p = stop_rel_i + 1
     INTEGER, PARAMETER :: min_feas_d = min_feas_p + 1
     INTEGER, PARAMETER :: mu_initial = min_feas_d + 1
     INTEGER, PARAMETER :: initial_n_model_radius = mu_initial + 1
     INTEGER, PARAMETER :: initial_t_model_radius = initial_n_model_radius + 1
     INTEGER, PARAMETER :: eta_successful = initial_t_model_radius + 1
     INTEGER, PARAMETER :: eta_very_successful = eta_successful + 1
     INTEGER, PARAMETER :: mu_smallest = eta_very_successful + 1

     INTEGER, PARAMETER :: direct_solution_of_normal_model = mu_smallest + 1
     INTEGER, PARAMETER :: direct_solution_of_tangential_model                 &
                             = direct_solution_of_normal_model + 1
     INTEGER, PARAMETER :: allow_extrapolatiion                                &
                             = direct_solution_of_tangential_model + 1
     INTEGER, PARAMETER :: use_second_order_correction                         &
                             = allow_extrapolatiion + 1
     INTEGER, PARAMETER :: fulsol = use_second_order_correction + 1
     INTEGER, PARAMETER :: space_critical = fulsol + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'FUNNEL          '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( alive_unit )%keyword = 'alive-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print  )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( algorithm_variant )%keyword = 'algorithm-variant-used'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
     spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
     spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
     spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
     spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
     spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
     spec( stop_abs_i )%keyword = 'absolute-infeasiblity-tolerated'
     spec( stop_rel_i )%keyword = 'relative-infeasiblity-tolerated'
     spec( mu_initial )%keyword = 'initial-barrier-parameter'
     spec( initial_n_model_radius )%keyword = 'initial-n-model-radius'
     spec( initial_t_model_radius )%keyword = 'initial-t-model-radius'
     spec( min_feas_p )%keyword = 'mininum-initial-primal-feasibility'
     spec( min_feas_d )%keyword = 'mininum-initial-dual-feasibility'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( mu_smallest )%keyword = 'smallest-barrier-parameter'

!  Logical key-words

     spec( direct_solution_of_normal_model )%keyword =                         &
       'direct-solution-of-normal-model'
     spec( direct_solution_of_tangential_model )%keyword =                     &
       'direct-solution-of-tangential-model'
     spec( allow_extrapolatiion )%keyword = 'allow-extrapolation'
     spec( use_second_order_correction )%keyword = 'use-second-order-correction'
     spec( fulsol )%keyword = 'print-full-solution'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( alive_file )%keyword = 'alive-filename'
     spec( prefix )%keyword = 'output-line-prefix'

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
     CALL SPECFILE_assign_value( spec( algorithm_variant ),                    &
                                 control%algorithm_variant,                    &
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
     CALL SPECFILE_assign_value( spec( stop_abs_i ),                           &
                                 control%stop_abs_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_i ),                           &
                                 control%stop_rel_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_initial ),                           &
                                 control%mu_initial,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_feas_p ),                           &
                                 control%min_feas_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_feas_d ),                           &
                                 control%min_feas_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_n_model_radius ),               &
                                 control%initial_n_model_radius,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_t_model_radius ),               &
                                 control%initial_t_model_radius,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_successful ),                       &
                                 control%eta_successful,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_very_successful ),                  &
                                 control%eta_very_successful,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_smallest ),                          &
                                 control%mu_smallest,                          &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( direct_solution_of_normal_model ),      &
                                 control%direct_solution_of_normal_model,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( direct_solution_of_tangential_model ),  &
                                 control%direct_solution_of_tangential_model,  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( allow_extrapolatiion ),                 &
                                 control%allow_extrapolatiion,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( use_second_order_correction ),          &
                                 control%use_second_order_correction,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( fulsol ),                               &
                                 control%fulsol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_string( spec( alive_file ), control%alive_file,      &
                                  control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )
!  Set LLST control values

     CALL LLST_read_specfile( control%LLST_control, device )

!  Set LLS control values

     CALL LLS_read_specfile( control%LLS_control, device )

!  Set EQP control values

     CALL EQP_read_specfile( control%EQP_control, device )

!  Set TRS control values

     CALL TRS_read_specfile( control%TRS_control, device )

!  Set SBLS control values

     CALL SBLS_read_specfile( control%SBLS_control, device )

     RETURN

     END SUBROUTINE FUNNEL_read_specfile

!-*-  G A L A H A D -  F U N N E L _ t e r m i n a t e  S U B R O U T I N E -*-

     SUBROUTINE FUNNEL_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FUNNEL_data_type ), INTENT( INOUT ) :: data
     TYPE ( FUNNEL_control_type ), INTENT( IN ) :: control
     TYPE ( FUNNEL_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate arrays set for SBLS

     CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,                &
                         inform%SBLS_inform )
     IF ( inform%SBLS_inform%status /= 0 ) THEN
       inform%status = inform%SBLS_inform%status
       inform%alloc_status = inform%SBLS_inform%alloc_status
       inform%bad_alloc = inform%SBLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate arrays set for LLST

     CALL LLST_terminate( data%LLST_data, control%LLST_control,                &
                          inform%LLST_inform )
     IF ( inform%LLST_inform%status /= 0 ) THEN
       inform%status = inform%LLST_inform%status
       inform%alloc_status = inform%LLST_inform%alloc_status
       inform%bad_alloc = inform%LLST_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate arrays set for LLS

     CALL LLS_terminate( data%LLS_data, control%LLS_control,                   &
                         inform%LLS_inform )
     IF ( inform%LLS_inform%status /= 0 ) THEN
       inform%status = inform%LLS_inform%status
       inform%alloc_status = inform%LLS_inform%alloc_status
       inform%bad_alloc = inform%LLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate arrays set for EQP

     CALL EQP_terminate( data%EQP_data, control%EQP_control,                   &
                         inform%EQP_inform )
     IF ( inform%EQP_inform%status /= 0 ) THEN
       inform%status = inform%EQP_inform%status
       inform%alloc_status = inform%EQP_inform%alloc_status
       inform%bad_alloc = inform%EQP_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arraysn allocated within TRS

     CALL TRS_terminate( data%TRS_data, control%TRS_control,                   &
                         inform%TRS_inform )
     inform%status = inform%TRS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%TRS_inform%alloc_status
       inform%bad_alloc = inform%TRS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all remaining allocated arrays

     array_name = 'funnel: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'funnel: data%H_diag'
     CALL SPACE_dealloc_array( data%H_diag,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine FUNNEL_terminate

     END SUBROUTINE FUNNEL_terminate

!-*-*-*-  G A L A H A D -  F U N N E L _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE FUNNEL_solve( nlp, control, inform, data, userdata,            &
                              eval_FC, eval_GJ, eval_HL )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  FUNNEL_solve, a method for finding a local minimizer of a function subject
!  to general constraints and simple bounds on the sizes of the variables

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_FUNNEL.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_FUNNEL_double
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
!  factorization_status is a scalar variable of type default integer, that
!   gives the return status from the matrix factorization.
!
!  factorization_integer is a scalar variable of type default integer,
!   that gives the amount of integer storage used for the matrix factorization.
!
!  factorization_real is a scalar variable of type default integer,
!   that gives the amount of real storage used for the matrix factorization.
!
!  f_eval is a scalar variable of type default integer, that gives the
!   total number of objective function evaluations performed.
!
!  g_eval is a scalar variable of type default integer, that gives the
!   total number of objective function gradient evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of objective function Hessian evaluations performed.
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
!  userdata is a scalar variable of type NLPT_userdata_type which may be used
!   to pass user data to and from the eval_* subroutines (see below)
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
!   be set to a nonzero value. If eval_FC is not present, FUNNEL_solve will
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
!   nonzero value. If eval_GJ is not present, FUNNEL_solve will return to the
!   user with inform%status = 3 or 5 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HL is not present,
!   FUNNEL_solve will return to the user with inform%status = 4 or 5 each time
!   an evaluation is required.
!
!  eval_HLPROD is an optional subroutine which if present must have
!   the arguments given below (see the interface blocks). The sum
!   u + nabla_xx ( f(x) - sum_i=1^m y_i c_i(x) ) v of the product of the Hessian
!   nabla_xx f(x) + sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y with the vector v=V and the vector u=U must be returned in U,
!   and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_HPROD is not present,
!   FUNNEL_solve will return to the user with inform%status = 6 each time an
!   evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( FUNNEL_control_type ), INTENT( INOUT ) :: control
     TYPE ( FUNNEL_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( FUNNEL_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_GJ, eval_HL

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_FC( status, X, userdata, f, C )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ) :: f
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_GJ( status, X, userdata, G, Jval )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G, Jval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HL( status, X, Y, userdata, Hval, no_f )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL
     END INTERFACE

!  ==========================================================================
!  Main algorithm
!
!  Trust-funnel algorithm for minimizing the barrier subproblem
!  minimize_{x in Re^n, s in Re^m}  f(x,c;mu)
!  subject to c(x,c) = 0, c^L < c < c^U  &  x^L < x < x^U.
!
!  [1:step] indicates the step in Algorithm 1 in the paper
!   Curtis, Gould, Robinson, Toint,
!   "An interior-point trust-funnel algorithm for nonlinear programming
!    using a squared violation infeasibility measure",
!   Technical Report RAL-TR-2014-001, STFC-Rutherford Appleton Labs, England
!  and corresponds to algorithm_variant=1
!
!  [2:step] indicates the step in Algorithm 1 in the paper
!   Curtis, Gould, Robinson, Toint,
!   "An interior-point trust-funnel algorithm for nonlinear programming",
!   Preprint RAL-P-2014-001, STFC-Rutherford Appleton Labs, England
!  and corresponds to algorithm_variant=2
!
!  [3:step] indicates the step in Algorithm 1 in the paper
!   Curtis, Gould, Robinson, Toint,
!   "An interior-point trust-funnel algorithm for nonlinear programming",
!   Report NAXYS-02-2014, U. Namur, Belgium
!  and corresponds to algorithm_variant=3
!
!  ==========================================================================
!
!  Notation
!
!  Problem - minimize f(x) : c^E(x) = 0, c^L <= c^I(x) <= c^U, x^L <= x <= x^U
!
!  f(x,s;mu) := f(x) - mu sum_{i=1}^M log (c-c^L)(c^U-c)
!                    - mu sum_{i=1}^N log (x-x^L)(x^U-x)
!
!  v     := ( x ),  v^L = ( x^L ), and v^U = ( x^U )
!           ( c )         ( c^L )            ( c^U )
!  c(v)  := (   c^E(x)   ),
!           ( c^I(x) - c )
!  y     := (    y^E    ), with y^L > 0 and y^U < 0
!           ( y^L + y^U )
!  z     := ( z^L + z^U ), with z^L > 0 and z^U < 0
!  J(v)  :=  grad c(x,c) = ( J^E(x)   0 ),
!                          ( J^I(x)  -I )
!  P(v)  := ( diag(min(x-x^L,x^U-x))              0          ),
!           (            0             diag(min(c-c^L,c^U-c)))
!  vi(v) := ||c(x,c)||_2^2  [1] or := ||c(x,c)||_2  [2,3]
!
!  Forcing functions omega_n, omega_y and omega_t such that
!  omega_t(omega_n(tau)) <= kappa_omega tau
!  for all tau >= 0 and for some kappa_omega in (0,1)}
!
!  ==========================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ii, ic, ir, j, l, ne
     REAL ( KIND = wp ) :: a0, a1, a2, av, val, mu_old, d_scale, radius
     REAL ( KIND = wp ) :: alpha, alpha_r, alpha_u, alpha_b, space, stopr
     CHARACTER ( LEN = 6 ) :: state
     CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$   INTEGER :: OMP_GET_MAX_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 910
     END IF
     IF ( inform%status == 1 ) data%branch = 1

     SELECT CASE ( data%branch )
     CASE ( 1 )  ! initialization
       GO TO 10
     CASE ( 2 )  ! initial objective and constraint evaluation
       GO TO 20
     CASE ( 3 )  ! gradient and Jacobian evaluation
       GO TO 230
     CASE ( 4 )  ! Hessian evaluation
       GO TO 240
!    CASE ( 5 )  ! Hessian-vector product
!      GO TO 150
     CASE ( 6 )  ! objective and constraint evaluation
       GO TO 260
     CASE ( 7 )  ! gradient and Jacobian evaluation
       GO TO 570
     CASE ( 8 )  ! Hessian evaluation
       GO TO 580
     CASE ( 9 )  ! objective and constraint evaluation
       GO TO 590
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE

     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
!$   inform%threads = OMP_GET_MAX_THREADS( )
     inform%status = GALAHAD_ok
     inform%alloc_status = 0 ; inform%bad_alloc = ''

     inform%f_eval = 0 ; inform%g_eval = 0

     inform%obj = HUGE( one )
!    inform%primal_infeasibility = HUGE( one )
!    inform%dual_infeasibility = HUGE( one )

     data%new_gradient = .TRUE.
     data%new_hessian = .TRUE.

     data%LLST_control = control%LLST_control
     data%LLS_control = control%LLS_control
     data%EQP_control = control%EQP_control

!  decide how much reverse communication is required

     data%reverse_fc = .NOT. PRESENT( eval_FC )
     data%reverse_gj = .NOT. PRESENT( eval_GJ )
     data%reverse_hl = .NOT. PRESENT( eval_HL )
!    data%reverse_hlprod = .NOT. PRESENT( eval_HLPROD )

!  ===========================
!  Control the output printing
!  ===========================

     data%out = control%out
     data%print_level_llst = control%LLS_control%print_level
     data%print_level_llst_sbls = control%LLST_control%SBLS_control%print_level
     data%print_level_lls = control%LLS_control%print_level
     data%print_level_lls_sbls = control%LLS_control%SBLS_control%print_level
     data%print_level_lls_gltr = control%LLS_control%GLTR_control%print_level
     data%print_level_eqp = control%EQP_control%print_level
     data%print_level_eqp_sbls = control%EQP_control%SBLS_control%print_level
     data%print_level_eqp_gltr = control%EQP_control%GLTR_control%print_level
     data%print_level_trs = control%TRS_control%print_level
     data%print_level_sbls = control%SBLS_control%print_level

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. control%print_level >= 2

!  as per printm, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. control%print_level >= 5

!  print level shorthands

     IF ( control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       data%stop_print = control%maxit + 1
     ELSE
       data%stop_print = control%stop_print
     END IF

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd ; data%print_level = control%print_level
       data%print_level = control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE.
       data%printd = .FALSE. ; data%print_level = 0
     END IF

     data%print_iteration_header = data%print_level > 0
     data%print_1st_header = .TRUE.

!  how many free variables are there?

     data%n_fr = 0
     DO i = 1, nlp%n
       IF ( nlp%X_l( i ) > nlp%X_u( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910
       ELSE IF ( nlp%X_l( i ) /= nlp%X_u( i ) ) THEN
         data%n_fr = data%n_fr + 1
       END IF
     END DO
     data%fixed_variables = data%n_fr < nlp%n

!  ** temporary ... stop if there are fixed variables

     IF ( data%fixed_variables ) THEN
       i = nlp%n - data%n_fr
       IF ( data%printi ) WRITE( data%out, "( A, ' ** warning: there ', A, 1X, &
      & I0, ' fixed variable', A )" ) prefix, TRIM( STRING_are( i ) ), i,      &
         TRIM( STRING_pleural( i ) )
!      inform%status = GALAHAD_error_bad_bounds
!      GO TO 910
     END IF

!  how many inequalities are there?

     data%m_in = 0
     DO i = 1, nlp%m
       IF ( nlp%C_l( i ) > nlp%C_u( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910
       ELSE IF ( nlp%C_l( i ) /= nlp%C_u( i ) ) THEN
         data%m_in = data%m_in + 1
       END IF
     END DO

!  allocate space for the variables v

!    data%n_total = nlp%n + data%m_in
     data%n_total = data%n_fr + data%m_in

     array_name = 'funnel: data%V'
     CALL SPACE_resize_array( data%n_total, data%V, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  type of variable is indicated in X_bound_type:
!    0=fixed,1=lower bounded,2=upper bounded,3=two-sided bounded,4=free
!  select the starting point satisfying x^l < x < x^u. If, additionally,
!  there are fixed variables, record the list of free ones (in X_fr)
!  and whether a variable is fixed (X_order() = 0) or free (X_order() > 0)

     array_name = 'funnel: data%X_bound_type'
     CALL SPACE_resize_array( nlp%n, data%X_bound_type, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  there are fixed variables

     IF ( data%fixed_variables ) THEN
       array_name = 'funnel: data%X_fr'
       CALL SPACE_resize_array( data%n_fr, data%X_fr, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'funnel: data%X_order'
       CALL SPACE_resize_array( nlp%n, data%X_order, inform%status,            &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       data%n_fr = 0
       DO i = 1, nlp%n
         IF ( nlp%X_l( i ) /= nlp%X_u( i ) ) THEN
           data%n_fr = data%n_fr + 1
           data%X_order( i ) = data%n_fr
           data%X_fr( data%n_fr ) = i
           IF ( nlp%X_l( i ) >= - control%infinity ) THEN
             IF ( nlp%X_u( i ) <= control%infinity ) THEN
               data%X_bound_type( i ) = both
               space = MIN( half * ( nlp%X_u( i ) - nlp%X_l( i ) ),            &
                            control%min_feas_p )
               data%V( data%n_fr )                                             &
                 = MIN( MAX( nlp%X( i ), nlp%X_l( i ) + space ),               &
                             nlp%X_u( i ) - space )
             ELSE
               data%X_bound_type( i ) = lower
               data%V( data%n_fr )                                             &
                 = MAX( nlp%X( i ), nlp%X_l( i ) + control%min_feas_p )
             END IF
           ELSE
             IF ( nlp%X_u( i ) <= control%infinity ) THEN
               data%X_bound_type( i ) = upper
               data%V( data%n_fr )                                             &
                 = MIN( nlp%X( i ), nlp%X_u( i ) - control%min_feas_p )
             ELSE
               data%X_bound_type( i ) = free
               data%V( data%n_fr ) = nlp%X( i )
             END IF
           END IF
         ELSE
           nlp%X( i ) = nlp%X_l( i )
           data%X_order( i ) = 0
           data%X_bound_type( i ) = fixed
         END IF
       END DO
       nlp%X( data%X_fr( : data%n_fr ) ) = data%V( : data%n_fr )

!  there are no fixed variables

     ELSE
       DO i = 1, nlp%n
         IF ( nlp%X_l( i ) /= nlp%X_u( i ) ) THEN
           IF ( nlp%X_l( i ) >= - control%infinity ) THEN
             IF ( nlp%X_u( i ) <= control%infinity ) THEN
               data%X_bound_type( i ) = both
               space = MIN( half * ( nlp%X_u( i ) - nlp%X_l( i ) ),            &
                            control%min_feas_p )
               data%V( i ) = MIN( MAX( nlp%X( i ), nlp%X_l( i ) + space ),     &
                                  nlp%X_u( i ) - space )
             ELSE
               data%X_bound_type( i ) = lower
               data%V( i )                                                     &
                 = MAX( nlp%X( i ), nlp%X_l( i ) + control%min_feas_p )
             END IF
           ELSE
             IF ( nlp%X_u( i ) <= control%infinity ) THEN
               data%X_bound_type( i ) = upper
               data%V( i )                                                     &
                 = MIN( nlp%X( i ), nlp%X_u( i ) - control%min_feas_p )
             ELSE
               data%X_bound_type( i ) = free
               data%V( i ) = nlp%X( i )
             END IF
           END IF
         ELSE
           data%X_bound_type( i ) = fixed
           data%V( i ) = nlp%X_l( i )
         END IF
       END DO
       nlp%X( : nlp%n ) = data%V( : nlp%n )
     END IF

!  ------------------------  COMPUTE FUNCTION VALUES --------------------------

!  evaluate the objective and general constraint function values

     IF ( data%reverse_fc ) THEN
       data%branch = 2 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
     END IF

!  return from reverse communication with the objective and constraint values

  20 CONTINUE
     inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1

!  constraints C_in(i), i = 1, .., m_i will be inequalities
!  type of constraint is indicated in C_bound_type:
!    0=equality,1=lower bounded,2=upper bounded,3=two-sided bounded,4=free
!  select the slack starting point satisfying c^l < c < c^u

     data%m_in = COUNT( nlp%C_l( : nlp%m ) /= nlp%C_u( : nlp%m ) )
     data%m_eq = nlp%m - data%m_in

     array_name = 'funnel: data%C_in'
     CALL SPACE_resize_array( data%m_in, data%C_in, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_eq'
     CALL SPACE_resize_array( data%m_eq, data%C_eq, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_bound_type'
     CALL SPACE_resize_array( nlp%m, data%C_bound_type, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     data%m_in = 0 ; data%m_eq = 0
     DO i = 1, nlp%m
       IF ( nlp%C_l( i ) /= nlp%C_u( i ) ) THEN
         data%m_in = data%m_in + 1
         data%C_in( data%m_in ) = i
         IF ( nlp%C_l( i ) >= - control%infinity ) THEN
           IF ( nlp%C_u( i ) <= control%infinity ) THEN
             data%C_bound_type( i ) = both
             space = MIN( half * ( nlp%C_u( i ) - nlp%C_l( i ) ),              &
                          control%min_feas_p )
             data%V( data%n_fr + data%m_in )                                 &
                = MIN( MAX( nlp%C( i ), nlp%C_l( i ) + space ),                &
                       nlp%C_u( i ) - space )
           ELSE
             data%C_bound_type( i ) = lower
             data%V( data%n_fr + data%m_in )                                 &
               = MAX( nlp%C( i ), nlp%C_l( i ) + control%min_feas_p )
           END IF
         ELSE
           IF ( nlp%C_u( i ) <= control%infinity ) THEN
             data%C_bound_type( i ) = upper
             data%V( data%n_fr + data%m_in )                                 &
               = MIN( nlp%C( i ), nlp%C_u( i ) - control%min_feas_p )
           ELSE
             data%C_bound_type( i ) = free
             data%V( data%n_fr + data%m_in ) = nlp%C( i )
           END IF
         END IF
       ELSE
         data%m_eq = data%m_eq + 1
         data%C_eq( data%m_eq ) = i
         data%C_bound_type( i ) = equality
       END IF
     END DO

!  select the dual starting point satisfying  y^L_{-1} > 0, y^U_{-1} < 0 ...

     array_name = 'funnel: data%Y_l'
     CALL SPACE_resize_array( nlp%m, data%Y_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Y_u'
     CALL SPACE_resize_array( nlp%m, data%Y_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  record the dimensons of the barrier Hessian

     data%HESS_barrier%m = data%n_total
     data%HESS_barrier%n = data%n_total
     IF ( data%fixed_variables ) THEN
       data%HESS_barrier%ne = 0
       DO l = 1, nlp%H%ne
         i = data%X_order( nlp%H%row( l ) ) ; j = data%X_order( nlp%H%col( l ) )
         IF ( i > 0 .AND. j > 0 )                                              &
           data%HESS_barrier%ne = data%HESS_barrier%ne + 1
       END DO
     ELSE
       data%HESS_barrier%ne = nlp%H%ne
     END IF
     CALL SMT_put( data%HESS_barrier%type, 'COORDINATE', inform%alloc_status )
     data%barrier_terms = zero
     DO i = 1, nlp%m
       SELECT CASE ( data%C_bound_type( i ) )
       CASE ( equality )
         data%Y_l( i ) = nlp%Y( i )
       CASE ( lower )
         data%Y_l( i ) = MAX( nlp%Y( i ), control%min_feas_d )
         nlp%Y( i ) = data%Y_l( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + one
       CASE ( upper )
         data%Y_u( i ) = MIN( nlp%Y( i ), - control%min_feas_d )
         nlp%Y( i ) = data%Y_u( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + one
       CASE ( both )
         data%Y_l( i ) = MAX( nlp%Y( i ), control%min_feas_d )
         data%Y_u( i ) = MIN( nlp%Y( i ), - control%min_feas_d )
         nlp%Y( i ) = data%Y_l( i ) + data%Y_u( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + two
       CASE ( free )
         data%Y_l( i ) = zero
         nlp%Y( i ) = zero
       END SELECT
     END DO

!  ... and z^L_{-1} > 0, z^U_{-1} < 0

     array_name = 'funnel: data%Z_l'
     CALL SPACE_resize_array( nlp%n, data%Z_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Z_u'
     CALL SPACE_resize_array( nlp%n, data%Z_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     DO i = 1, nlp%n
       SELECT CASE ( data%X_bound_type( i ) )
       CASE ( fixed )
         data%Z_l( i ) = nlp%Z( i )
       CASE ( lower )
         data%Z_l( i ) = MAX( nlp%Z( i ), control%min_feas_d )
         nlp%Z( i ) = data%Z_l( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + one
       CASE ( upper )
         data%Z_u( i ) = MIN( nlp%Z( i ), - control%min_feas_d )
         nlp%Z( i ) = data%Z_u( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + one
       CASE ( both )
         data%Z_l( i ) = MAX( nlp%Z( i ), control%min_feas_d )
         data%Z_u( i ) = MIN( nlp%Z( i ), - control%min_feas_d )
         nlp%Z( i ) = data%Z_l( i ) + data%Z_u( i )
         data%HESS_barrier%ne = data%HESS_barrier%ne + 1
         data%barrier_terms = data%barrier_terms + two
       CASE ( free )
         data%Z_l( i ) = zero
         nlp%Z( i ) = zero
       END SELECT
     END DO

!  record the norm of the multipliers

     IF ( data%n_fr > 0 ) THEN
       IF ( data%fixed_variables ) THEN
         data%max_y = MAXVAL( ABS( nlp%Z( data%X_fr( : data%n_fr ) ) ) )
       ELSE
         data%max_y = MAXVAL( ABS( nlp%Z( : nlp%n ) ) )
       END IF
     ELSE
       data%max_y = zero
     END IF

     IF ( nlp%m > 0 ) THEN
       data%max_y = MAX( data%max_y, MAXVAL( ABS( nlp%Y( : nlp%m ) ) ) )
     END IF
     data%max_y = MAX( data%max_y, one )

!  assign the space for J(v)

     data%Jv%m = nlp%m
     data%Jv%n = data%n_fr + data%m_in
     IF ( data%fixed_variables ) THEN
       data%Jv%ne = data%m_in
       DO l = 1, nlp%J%ne
         j = data%X_order( nlp%J%col( l ) )
         IF ( j > 0 ) data%Jv%ne = data%Jv%ne + 1
       END DO
     ELSE
       data%Jv%ne = nlp%J%ne + data%m_in
     END IF
     CALL SMT_put( data%Jv%type, 'COORDINATE', inform%alloc_status )

     array_name = 'funnel: data%Jv%row'
     CALL SPACE_resize_array( data%Jv%ne, data%Jv%row, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Jv%col'
     CALL SPACE_resize_array( data%Jv%ne, data%Jv%col, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Jv%val'
     CALL SPACE_resize_array( data%Jv%ne, data%Jv%val, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  assign the row and column indices of J(v)

     IF ( data%fixed_variables ) THEN
       ne = 0
       DO l = 1, nlp%J%ne
         i = nlp%J%row( l ) ; j = data%X_order( nlp%J%col( l ) )
         IF ( j > 0 ) THEN
           ne = ne + 1
           data%Jv%row( ne ) = i ; data%Jv%col( ne ) = j
         END IF
       END DO
     ELSE
       data%Jv%row( : nlp%J%ne ) = nlp%J%row( : nlp%J%ne )
       data%Jv%col( : nlp%J%ne ) = nlp%J%col( : nlp%J%ne )
       ne = nlp%J%ne
     END IF

     DO ii = 1, data%m_in
       i = data%C_in( ii )
       ne = ne + 1
       data%Jv%row( ne ) = i
       data%Jv%col( ne ) = data%n_fr + ii
       data%Jv%val( ne ) = - one
     END DO

!  assign the space for the barrier gradient and Hessian

     data%HESS_barrier%m = data%n_total
     data%HESS_barrier%n = data%n_total

     array_name = 'funnel: data%GRAD_barrier'
     CALL SPACE_resize_array( data%n_total, data%GRAD_barrier,                 &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%HESS_barrier%row'
     CALL SPACE_resize_array( data%HESS_barrier%ne, data%HESS_barrier%row,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%HESS_barrier%col'
     CALL SPACE_resize_array( data%HESS_barrier%ne, data%HESS_barrier%col,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%HESS_barrier%val'
     CALL SPACE_resize_array( data%HESS_barrier%ne, data%HESS_barrier%val,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  assign the row and column indices of the barrier Hessian

     IF ( data%fixed_variables ) THEN
       ne = 0
       DO l = 1, nlp%H%ne
         i = data%X_order( nlp%H%row( l ) ) ; j = data%X_order( nlp%H%col( l ) )
         IF ( i > 0 .AND. j > 0 ) THEN
           ne = ne + 1
           data%HESS_barrier%row( ne ) = i ; data%HESS_barrier%col( ne ) = j
         END IF
       END DO
     ELSE
       data%HESS_barrier%row( : nlp%H%ne ) = nlp%H%row( : nlp%H%ne )
       data%HESS_barrier%col( : nlp%H%ne ) = nlp%H%col( : nlp%H%ne )
       ne = nlp%H%ne
     END IF
     IF ( data%fixed_variables ) THEN
       DO ii = 1, data%n_fr
         i = data%X_fr( ii )
         SELECT CASE ( data%X_bound_type( i ) )
         CASE ( lower, upper, both )
           ne = ne + 1
           data%HESS_barrier%row( ne ) = ii
           data%HESS_barrier%col( ne ) = ii
         END SELECT
       END DO
     ELSE
       DO i = 1, nlp%n
         SELECT CASE ( data%X_bound_type( i ) )
         CASE ( lower, upper, both )
           ne = ne + 1
           data%HESS_barrier%row( ne ) = i
           data%HESS_barrier%col( ne ) = i
         END SELECT
       END DO
     END IF

     DO ii = 1, data%m_in
       i = data%C_in( ii )
       j = data%n_fr + ii
       SELECT CASE ( data%C_bound_type( i ) )
       CASE ( lower, upper, both )
         ne = ne + 1
         data%HESS_barrier%row( ne ) = j
         data%HESS_barrier%col( ne ) = j
       END SELECT
     END DO

!  assign the space for P_k^{-2}

     data%Pinv2%m = data%n_total
     data%Pinv2%n = data%n_total
     data%Pinv2%ne = data%n_total
     CALL SMT_put( data%Pinv2%type, 'DIAGONAL', inform%alloc_status )
     array_name = 'lls: data%Pinv2%val'
     CALL SPACE_resize_array( data%Pinv2%ne, data%Pinv2%val,                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  further array allocations

     array_name = 'funnel: data%Cv'
     CALL SPACE_resize_array( nlp%m, data%Cv, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Y'
     CALL SPACE_resize_array( nlp%m, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%N'
     CALL SPACE_resize_array( data%n_total, data%N, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%T'
     CALL SPACE_resize_array( data%n_total, data%T, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%V_plus'
     CALL SPACE_resize_array( data%n_total, data%V_plus, inform%status,        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%W'
     CALL SPACE_resize_array( data%n_total + nlp%m, data%W,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_current'
     CALL SPACE_resize_array( nlp%m, data%C_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%P'
     CALL SPACE_resize_array( data%n_total, data%P, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%R'
     CALL SPACE_resize_array( MAX( data%n_total, nlp%m ), data%R,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%CAUCHY'
     CALL SPACE_resize_array( data%n_total, data%CAUCHY, inform%status,        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%GRAD_mf'
     CALL SPACE_resize_array( data%n_total, data%GRAD_mf,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  assign the zero matrix C0

      CALL SMT_put( data%C0%type, 'ZERO', inform%alloc_status )

! [1,2,3:2] assign control parameters
!        0 < eta_1 <= eta_2 < 1, 0 < gamma_1 <= gamma_2 < 1,
!        { delta^n_0, delta^t_0, kappa_ca,
!          kappa_n, kappa_y, kappa_nr, kappa_delta } in (0,infty),
!        { kappa_vv, kappa_fbn, kappa_B,
!          kappa_fbt, kappa_tt, kappa_tg, kappa_deltavv,
!          kappa_t1, kappa_t2} in (0,1),
!        kappa_cd in (0,1-kappa_tg) and
!        { kappa_cr, kappa_v, kappa_VS } in (1,infty)

     data%kappa_ca = control%kappa_ca
     data%kappa_n = control%kappa_n
     data%kappa_y = control%kappa_y
     data%kappa_nr = control%kappa_nr
     data%kappa_delta = control%kappa_delta
     data%kappa_vv = control%kappa_vv
     data%omkappa_fbn = one - control%kappa_fbn
     data%omkappa_fbt = one - control%kappa_fbt
     data%kappa_B = control%kappa_B
     data%kappa_tt = control%kappa_tt
     data%kappa_tg = control%kappa_tg
     data%kappa_deltavv = control%kappa_deltavv
     data%kappa_chi = control%kappa_chi
     data%kappa_t1 = control%kappa_t1
     data%kappa_t2 = control%kappa_t2
     data%kappa_cd = control%kappa_cd
     data%kappa_cr = control%kappa_cr
     data%kappa_v = control%kappa_v
     data%kappa_VS = control%kappa_VS
     data%use_violation_squared = control%algorithm_variant == 1
     data%tau_flag_raised = .TRUE.

     data%SBLS_control = control%SBLS_control
     IF ( data%SBLS_control%factorization < 0 .OR.                             &
          data%SBLS_control%factorization > 3 ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
         "( A,' factor = ', I0, ' out of range [0,3]. Reset to 0' )" )         &
         prefix, data%SBLS_control%factorization
       data%SBLS_control%factorization = 0
     END IF

!  set the initial trust-region radii

     data%radius_n = control%initial_n_model_radius
     data%radius_t = control%initial_t_model_radius

!  radius increase and decrease factors

     data%radius_increase = MAX( one, control%radius_increase )
     data%radius_decrease = MAX( control%gamma_1,                              &
                              MIN( control%gamma_2, control%radius_decrease ) )

!  successful and very_successful step tolerances

     data%eta_very_successful = MIN( MAX( control%eta_very_successful,         &
                                          control%eta_1 ), control%eta_2 )
     data%eta_successful = MIN( MAX( control%eta_successful,                   &
                                     control%eta_1 ), data%eta_very_successful)

!  [1,2:3] perform the slack reset, compute c(v_0) and the violation
!  vi(v_0) :=  m^v_0(0) := 1/2 ||c(v_0)||_2^2

     CALL FUNNEL_slack_reset( nlp%m, data%m_eq, data%m_in, data%C_eq,          &
                        data%C_in, data%n_fr, data%C_bound_type, nlp%C,        &
                        nlp%C_l, nlp%C_u, data%V, data%Cv, data%norm_c,        &
                        data%violation, data%use_violation_squared,            &
                        data%mv0 )

!  [1,2:4] set k<- 0, pi^f_{-1} = 0 and
!      vi^max_0 = max[kappa_ca, kappa_cr vi(x_0,c_0)].

     inform%iter = 0
     data%it_type = ' '
     data%pi_f = zero
     data%violation_max = MAX( data%kappa_ca, data%kappa_cr * data%violation )

       IF ( data%printi )                                                      &
         WRITE( data%out, "( /, A, ' Problem: ', A )" ) prefix, nlp%pname

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!              S T A R T    O F    B A R R I E R  I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     IF ( control%mu_initial <= zero ) THEN
       data%mu = one
     ELSE
       data%mu = control%mu_initial
     END IF

     IF ( control%mu_smallest <= zero ) THEN
       data%mu_smallest = epsmch
     ELSE
       data%mu_smallest = control%mu_smallest
     END IF

!100 CONTINUE

!  compute the new barrier function

       IF ( data%fixed_variables ) THEN
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,            &
                    n_fr = data%n_fr, X_fr = data%X_fr )
       ELSE
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
       END IF
       data%mf0 = data%barrier
       data%start_inner = .TRUE.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  [1,2:5] S T A R T  O F  I N N E R  M I N I M I Z A T I O N  I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

 200   CONTINUE
!write(6,*) ' max y ', data%max_y

         IF ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print ) THEN
           data%printi = data%set_printi ; data%printt = data%set_printt
           data%printm = data%set_printm ; data%printw = data%set_printw
           data%printd = data%set_printd
           data%print_level = control%print_level

           data%LLST_control%print_level = data%print_level_lls
           data%LLST_control%SBLS_control%print_level                          &
             = data%print_level_llst_sbls
           data%LLS_control%print_level = data%print_level_lls
           data%LLS_control%SBLS_control%print_level                           &
             = data%print_level_lls_sbls
           data%LLS_control%GLTR_control%print_level                           &
             = data%print_level_lls_gltr
           data%EQP_control%print_level = data%print_level_eqp
           data%EQP_control%SBLS_control%print_level                           &
             = data%print_level_eqp_sbls
           data%EQP_control%GLTR_control%print_level                           &
             = data%print_level_eqp_gltr
           data%TRS_control%print_level = data%print_level_trs
           data%SBLS_control%print_level = data%print_level_sbls
         ELSE
           data%printi = .FALSE. ; data%printt = .FALSE.
           data%printm = .FALSE. ; data%printw = .FALSE.
           data%printd = .FALSE. ; data%print_level = 0
           data%LLST_control%print_level = 0
           data%LLST_control%SBLS_control%print_level = 0
           data%LLS_control%print_level = 0
           data%LLS_control%SBLS_control%print_level = 0
           data%LLS_control%GLTR_control%print_level = 0
           data%EQP_control%print_level = 0
           data%EQP_control%SBLS_control%print_level = 0
           data%EQP_control%GLTR_control%print_level = 0
           data%TRS_control%print_level = 0
           data%SBLS_control%print_level = 0
         END IF

         data%print_iteration_header = data%print_level > 1 .OR.               &
           data%LLST_control%print_level > 0 .OR.                              &
           data%LLS_control%print_level > 0 .OR.                               &
           data%TRS_control%print_level > 0 .OR.                               &
           data%EQP_control%print_level > 0

!  set the overall trust-region radius

!        data%radius = MIN( data%radius_t, data%radius_n )
         data%radius = MIN( data%radius_t, two * data%radius_n )

!  [1,2:6] compute P_k = P(v_k) ...

         data%P( :  data%n_fr ) = one
!        IF ( data%fixed_variables ) THEN
!          DO ii = 1, data%n_fr
!            i = data%X_fr( ii )
!            SELECT CASE ( data%X_bound_type( i ) )
!            CASE ( fixed )
!              data%P( ii ) = one
!            CASE ( lower )
!              data%P( ii ) = data%V( ii ) - nlp%X_l( i )
!            CASE ( upper )
!              data%P( ii ) = nlp%X_u( i ) - data%V( ii )
!            CASE ( both )
!              data%P( ii ) = MIN( data%V( ii ) - nlp%X_l( i ),                &
!                                 nlp%X_u( i ) - data%V( ii ) )
!            CASE ( free )
!              data%P( ii ) = one
!            END SELECT
!          END DO
!        ELSE
!          DO i = 1, nlp%n
!            SELECT CASE ( data%X_bound_type( i ) )
!            CASE ( fixed )
!              data%P( i ) = one
!            CASE ( lower )
!              data%P( i ) = data%V( i ) - nlp%X_l( i )
!            CASE ( upper )
!              data%P( i ) = nlp%X_u( i ) - data%V( i )
!            CASE ( both )
!              data%P( i ) = MIN( data%V( i ) - nlp%X_l( i ),                  &
!                                 nlp%X_u( i ) - data%V( i ) )
!            CASE ( free )
!              data%P( i ) = one
!            END SELECT
!          END DO
!        END IF

         DO ii = 1, data%m_in
           i = data%C_in( ii )
           j = data%n_fr + ii
           SELECT CASE ( data%C_bound_type( i ) )
           CASE ( equality )
             data%P( j ) = one
           CASE ( lower )
             data%P( j ) = data%V( j ) - nlp%C_l( i )
           CASE ( upper )
             data%P( j ) = nlp%C_u( i ) -  data%V( j )
           CASE ( both )
             data%P( j ) = MIN( data%V( j ) - nlp%C_l( i ),                    &
                                nlp%C_u( i ) -  data%V( j ) )
           CASE ( free )
             data%P( j ) = one
           END SELECT
         END DO

!  ... P_k^{-2}

         data%Pinv2%val( : data%n_total ) = one / data%P( : data%n_total ) ** 2

!  -------------------  COMPUTE FIRST DERIVATIVE VALUES -----------------------

!  evaluate the Jacobian of the general constraint functions, stored in
!  "co-ordinate" format, and the gradient of the objective

         IF ( data%new_gradient ) THEN
           IF ( data%reverse_gj ) THEN
             data%branch = 3 ; inform%status = 3 ; RETURN
           ELSE
             CALL eval_GJ( data%eval_status, nlp%X, userdata, nlp%G, nlp%J%val )
           END IF
         END IF

!  return from reverse communication with the gradient and Jacobian

    230  CONTINUE
         IF ( data%new_gradient ) THEN
           inform%g_eval = inform%g_eval + 1

!  compute J(v_k) ...

           IF ( data%fixed_variables ) THEN
             ne = 0
             DO l = 1, nlp%J%ne
               IF ( data%X_order( nlp%J%col( l ) ) > 0 ) THEN
                 ne = ne + 1
                 data%Jv%val( ne ) = nlp%J%val( l )
               END IF
             END DO
           ELSE
             data%Jv%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne )
           END IF
         END IF

!  ... and P_k J(v_k)^T c(v_k) (stored in v_plus) ...

         data%V_plus( : data%n_total ) = zero
         DO l = 1, data%Jv%ne
           j = data%Jv%col( l )
           data%V_plus( j ) = data%V_plus( j )                                 &
             + data%Jv%val( l ) * data%Cv( data%Jv%row( l ) )
         END DO
         data%V_plus( : data%n_total )                                         &
           = data%P( : data%n_total ) * data%V_plus( : data%n_total )

!   ...  pi^v_k:= pi^v(x_k,c_k) := ||P_k J(v_k)^T c(v_k)||_2

         data%pi_v = TWO_NORM( data%V_plus( : data%n_total ) )

!   ... and xi^v_k:= pi^v_k / v_k (or 0 if v_k = 0 ) if feasibilty uses the
!   violation rather than the violation squared

         IF ( data%use_violation_squared ) THEN
           data%xi_v = data%pi_v
         ELSE
           IF ( data%violation > 0 ) THEN
             data%xi_v = data%pi_v / data%violation
           ELSE
             data%xi_v = zero
           END IF
         END IF

!  compute the primal infeasibility and the complementary slackness

         inform%primal_infeasibility = zero
         inform%complementary_slackness = zero
         DO i = 1, nlp%n
           SELECT CASE ( data%X_bound_type( i ) )
           CASE ( lower )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility, nlp%X_l( i ) - nlp%X( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%X_l( i ) - nlp%X( i ) ) * data%Z_l( i ) )
           CASE ( upper )
             inform%primal_infeasibility =                                     &
               MAX( inform%primal_infeasibility, nlp%X( i ) - nlp%X_u( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%X_u( i ) - nlp%X( i ) ) * data%Z_u( i ) )
           CASE ( both )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility,                             &
                      nlp%X_l( i ) - nlp%X( i ), nlp%X( i ) - nlp%X_u( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%X_l( i ) - nlp%X( i ) ) * data%Z_l( i ) )         &
                + ABS( ( nlp%X_u( i ) - nlp%X( i ) ) * data%Z_u( i ) )
           END SELECT
         END DO
         DO i = 1, nlp%m
           SELECT CASE ( data%C_bound_type( i ) )
           CASE ( lower )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility, nlp%C_l( i ) - nlp%C( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%C_l( i ) - nlp%C( i ) ) * data%Y_l( i ) )
           CASE ( upper )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility, nlp%C( i ) - nlp%C_u( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%C_u( i ) - nlp%C( i ) ) * data%Y_u( i ) )
           CASE ( both )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility,                             &
                      nlp%C_l( i ) - nlp%C( i ), nlp%C( i ) - nlp%C_u( i ) )
             inform%complementary_slackness = inform%complementary_slackness   &
                + ABS( ( nlp%C_l( i ) - nlp%C( i ) ) * data%Y_l( i ) )         &
                + ABS( ( nlp%C_u( i ) - nlp%C( i ) ) * data%Y_u( i ) )
           CASE ( equality )
             inform%primal_infeasibility                                       &
               = MAX( inform%primal_infeasibility,                             &
                      nlp%C_l( i ) - nlp%C( i ), nlp%C( i ) - nlp%C_u( i ) )
           END SELECT
         END DO
!write(6,*) ' slackness ', inform%complementary_slackness
         IF ( data%barrier_terms > zero ) inform%complementary_slackness       &
           = inform%complementary_slackness / data%barrier_terms

!  compute the dual infeasibility

         data%W( : nlp%n ) = nlp%G( : nlp%n )
!write(6,*) ' g ', data%W( : nlp%n )
         DO i = 1, nlp%n
           SELECT CASE ( data%X_bound_type( i ) )
           CASE ( lower )
             data%W( i ) = data%W( i ) - data%Z_l( i )
           CASE ( upper )
             data%W( i ) = data%W( i ) - data%Z_u( i )
           CASE ( both )
!write(6,*) i, data%W( i ), data%Z_l( i ), data%Z_u( i )
             data%W( i ) = data%W( i ) - data%Z_l( i ) - data%Z_u( i )
           END SELECT
         END DO

         DO l = 1, nlp%J%ne
           i = nlp%J%row( l ) ; j = nlp%J%col( l ) ; val = nlp%J%val( l )
           SELECT CASE ( data%C_bound_type( i ) )
           CASE ( equality )
             data%W( j ) = data%W( j ) - val * nlp%Y( i )
           CASE ( lower )
!write(6,*) j, data%W( j ), i, val, data%Y_l( i ), data%W( j ) - val * data%Y_l( i ), nlp%Y( i )
             data%W( j ) = data%W( j ) - val * data%Y_l( i )
           CASE ( upper )
             data%W( j ) = data%W( j ) - val * data%Y_u( i )
           CASE ( both )
             data%W( j ) = data%W( j ) - val * ( data%Y_l( i ) + data%Y_u( i ) )
           CASE ( free )
           END SELECT
         END DO
!write(6,*) ' infeas ', data%W( : nlp%n )
         IF ( data%fixed_variables ) THEN
           inform%dual_infeasibility                                           &
             = MAXVAL( ABS( data%W( data%X_fr( : data%n_fr ) ) ) )
!write(6,*) ' dual ', data%W( data%X_fr( : data%n_fr ) )
         ELSE
           inform%dual_infeasibility = MAXVAL( ABS( data%W( : nlp%n ) ) )
         END IF
         inform%dual_infeasibility = inform%dual_infeasibility / data%max_y

!write(6,*)  ' y ', nlp%Y( 1 )
!do i = 1, nlp%n
!  write(6,"(I6, 4ES12.4)" ) i, nlp%G( i ), data%Z_l( i ), data%Z_u( i ), nlp%Z( i )
!end do
         IF ( data%printt ) WRITE( data%out, "( A, '   p_infeas = ', ES10.4,   &
         &  ', d_infeas = ', ES10.4, ', comp_slack = ', ES10.4 )" ) prefix,    &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness

!  compute the overall termination tolerances

         IF (  inform%iter == 0 ) THEN
           data%stop_p = MAX( control%stop_abs_p, control%stop_rel_p *         &
                                inform%primal_infeasibility )
           data%stop_d = MAX( control%stop_abs_d, control%stop_rel_d *         &
                                inform%dual_infeasibility )
           data%stop_c = MAX( control%stop_abs_c, control%stop_rel_c *         &
                                inform%complementary_slackness )

!  compute the inner-iteration termination tolerances

           IF ( data%barrier_terms > zero ) THEN
             data%stop_p_inner = MAX( data%stop_p, data%mu )
             data%stop_d_inner = MAX( data%stop_d, data%mu )
             data%stop_c_inner = mu_tol * data%mu

!  special case when there are no inequalities

           ELSE
             data%stop_p_inner = data%stop_p
             data%stop_d_inner = data%stop_d
             data%stop_c_inner = data%mu
           END IF
           IF ( data%printi ) WRITE( data%out, 2050 )                          &
             prefix, data%mu, data%stop_p_inner, data%stop_d_inner
         END IF

!  print a summary of the iteration if required

         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         inform%time%total = data%time_now - data%time_start
         inform%time%clock_total = data%clock_now - data%clock_start

         IF ( data%printi ) THEN
           IF ( data%print_iteration_header .OR. data%print_1st_header )       &
             WRITE( data%out, 2020 ) prefix, prefix
           data%print_1st_header = .FALSE.
           IF ( .NOT. data%start_inner ) THEN
             WRITE( data%out, 2030 ) prefix,                                   &
                inform%iter, data%it_type, data%success, data%n_end,           &
                data%t_end, data%s_end, inform%obj,                            &
                inform%primal_infeasibility, inform%dual_infeasibility,        &
                data%rho, data%radius_n, data%radius_t, data%violation_max,    &
                inform%time%clock_total
           ELSE
             WRITE( data%out, 2040 ) prefix, inform%iter, data%it_type,        &
                inform%obj, inform%primal_infeasibility,                       &
                inform%dual_infeasibility, data%radius_t, data%radius_n,       &
                data%violation_max, inform%time%clock_total
           END IF
         END IF
         data%start_inner = .FALSE.

!  ---------------------------  TEST FOR OPTIMALITY ---------------------------

!write(6,*) ABS( inform%complementary_slackness - data%mu ), data%stop_c_inner

         IF ( inform%primal_infeasibility <= data%stop_p_inner .AND.           &
              inform%dual_infeasibility <= data%stop_d_inner .AND.             &
              ABS( inform%complementary_slackness - data%mu )                  &
                <= data%stop_c_inner ) THEN

!write(6,*) ' optimality test satisfied '

           IF ( inform%primal_infeasibility <= data%stop_p .AND.               &
                inform%dual_infeasibility <= data%stop_d .AND.                 &
                inform%complementary_slackness <= data%stop_c ) THEN
             inform%status = GALAHAD_ok
             GO TO 610
           ELSE
             GO TO 510
           END IF
         END IF

!  test that the iteration limit has not been reached

         IF ( inform%iter > control%maxit ) THEN
           inform%status = GALAHAD_error_max_iterations
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( /, A, ' - the iteration limit has been reached' )" ) prefix
           GO TO 610
         END IF

!  compute primal dual variables (y^L,y^U,z^L,z^U) and
!   grad f(v;mu) := grad f(x)
!    - mu sum_{i=1}^N e_i/(x-x^L) - mu sum_{i=1}^N e_i/(x-x^U)
!    - mu sum_{i=1}^M e_N+i/(c-c^L) - mu sum_{i=1}^M e_N+i/(c-c^U)

         IF ( data%fixed_variables ) THEN
           data%GRAD_barrier( : data%n_fr ) = nlp%G( data%X_fr( : data%n_fr) )
         ELSE
           data%GRAD_barrier( : nlp%n ) = nlp%G( : nlp%n )
         END IF
         data%GRAD_barrier( data%n_fr + 1 : data%n_total ) = zero

         IF ( data%fixed_variables ) THEN
           DO ii = 1, data%n_fr
             i = data%X_fr( ii )
             IF ( data%X_bound_type( i ) == lower .OR.                         &
                  data%X_bound_type( i ) == both ) THEN
               data%Z_l( i ) = data%mu / ( data%V( ii ) - nlp%X_l( i ) )
               data%GRAD_barrier( ii ) = data%GRAD_barrier( ii ) - data%Z_l( i )
             END IF
             IF ( data%X_bound_type( i ) == upper .OR.                         &
                  data%X_bound_type( i ) == both )  THEN
               data%Z_u( i ) = data%mu / ( data%V( ii ) - nlp%X_u( i ) )
               data%GRAD_barrier( ii ) = data%GRAD_barrier( ii ) - data%Z_u( i )
             END IF
           END DO
         ELSE
           DO i = 1, nlp%n
             IF ( data%X_bound_type( i ) == lower .OR.                         &
                  data%X_bound_type( i ) == both ) THEN
               data%Z_l( i ) = data%mu / ( data%V( i ) - nlp%X_l( i ) )
               data%GRAD_barrier( i ) = data%GRAD_barrier( i ) - data%Z_l( i )
             END IF
             IF ( data%X_bound_type( i ) == upper .OR.                         &
                  data%X_bound_type( i ) == both )  THEN
               data%Z_u( i ) = data%mu / ( data%V( i ) - nlp%X_u( i ) )
               data%GRAD_barrier( i ) = data%GRAD_barrier( i ) - data%Z_u( i )
             END IF
           END DO
         END IF
!write(6,*) 'A z(1) ', data%Z_l( 1 )
!write(6,*) 'A z(1) ', data%Z_l( 1 ), data%Z_u( 1 )
!write(6,*) ' x - x_l ', data%V(1) - nlp%X_l(1)
!write(6,*) ' x - x_u ', data%V(1) - nlp%X_u(1)

!write(6,*) ' y_l = mu / c '
         DO ii = 1, data%m_in
           i = data%C_in( ii )
           j =  data%n_fr + ii
           IF ( data%C_bound_type( i ) == lower .OR.                           &
                data%C_bound_type( i ) == both ) THEN
             data%Y_l( i ) = data%mu / ( data%V( j ) - nlp%C_l( i ) )
!write(6,*) ' y_l ', data%Y_l( i )
!write(6,*) ' mu, c ', data%mu, ( data%V( j ) - nlp%C_l( i ) )
             data%GRAD_barrier( j ) = data%GRAD_barrier( j ) - data%Y_l( i )
           END IF
           IF ( data%C_bound_type( i ) == upper .OR.                           &
                data%C_bound_type( i ) == both ) THEN
             data%Y_u( i ) = data%mu / ( data%V( j ) - nlp%C_u( i ) )
             data%GRAD_barrier( j ) = data%GRAD_barrier( j ) - data%Y_u( i )
           END IF
         END DO

!  compute y_k

!        DO i = 1, nlp%m
!          SELECT CASE ( data%C_bound_type( i ) )
!          CASE ( lower )
!            nlp%Y( i ) = data%Y_l( i )
!          CASE ( upper )
!            nlp%Y( i ) = data%Y_u( i )
!          CASE ( both )
!            nlp%Y( i ) = data%Y_l( i ) + data%Y_u( i )
!          CASE ( free )
!            nlp%Y( i ) = zero
!          END SELECT
!        END DO

!  compute z_k

         DO i = 1, nlp%n
           SELECT CASE ( data%X_bound_type( i ) )
           CASE ( lower )
             nlp%Z( i ) = data%Z_l( i )
           CASE ( upper )
             nlp%Z( i ) = data%Z_u( i )
           CASE ( both )
             nlp%Z( i ) = data%Z_l( i ) + data%Z_u( i )
           CASE ( free )
             nlp%Z( i ) = zero
           END SELECT
         END DO

!  [1,2,3:9] compute a normal step if pi^v_k > 0 and either
!    ||c(x_k,c_k)||_2 > omega_n( pi^f_{k-1} ) or
!    vi(x_k,c_k) >= kappa_vv vi^max_k or it is "deemed desirable"

         IF ( data%xi_v > epsmch  ) THEN
!write(6,*) data%pi_v, FORCING( 'n', data%pi_f )
!write(6,*) data%pi_v > FORCING( 'n', data%pi_f ), &
! data%violation >= data% kappa_vv * data%violation_max, &
! control%n_deemed_desirable
           IF ( data%pi_v > FORCING( 'n', data%pi_f ) .OR.                     &
                data%violation >= data% kappa_vv * data%violation_max .OR.     &
                control%n_deemed_desirable ) THEN

!  [3:10] compute the unconstrained Cauchy point n^U_k := n^C_k(alpha_U),
!    n^C_k(alpha) := (n^Cx_k(alpha)) := -alpha P_k^2 J(v_k)^T c(v_k),
!                   (n^Cc_k(alpha))
!  alpha_U is the solution to
!    minimize_{alpha>= 0} m^v_k (n^C_k(alpha) )
!  and m^v_k(n) := 1/2 ||c(v_k) + J(v_k) n||_2^2

!  compute -  P_k^2 J(v_k)^T c(v_k) (store in n^C_k)

             data%CAUCHY( : data%n_total )                                     &
               = - data%P( : data%n_total ) * data%V_plus( : data%n_total )

!  compute J(v_k) n^C_k

             data%W( : nlp%m ) = zero
             DO l = 1, data%Jv%ne
               i = data%Jv%row( l )
               j = data%Jv%col( l )
               data%W( i ) = data%W( i ) + data%Jv%val( l ) * data%CAUCHY( j )
             END DO

!  compute the line minimizer of ||c(v_k) + alpha J(v_k) n^C_k ||_2^2

             alpha_u = data%pi_v ** 2 /                                        &
                         DOT_PRODUCT( data%W( : nlp%m ), data%W( : nlp%m ) )

!  [3:11] if the tau flag is raised, ensure that
!  delta_k^v >= kappa_n ||P_k^{-1}n^U_k||_2 and lower the flag

             IF ( control%algorithm_variant == 3 ) THEN
               IF ( data%tau_flag_raised ) THEN
                 data%radius_n = MAX( data%radius_n, data%kappa_n *            &
                   alpha_u * TWO_NORM( data%CAUCHY( : data%n_total ) ) )
                 data%tau_flag_raised = .FALSE.
               END IF
             END IF

!  ============================= NORMAL SUBPROBLEM ===========================
!
!  [1,2:10,3:12] compute the normal step n_k as an approximate solution to
!
!    minimize_{n} m^v_k(n)
!    subject to n in N_k := {n: ||P_k^{-1} n||_2 <= radius^n_k} &            (A)
!      v^L + kappa_fbn (v_k - v^L) <= v_k + n <= v^U + kappa_fbn (v_k - v^U)
!
!  that satisfies (A)'s constraints
!
!  For [1], m^v_k(n) := ||c(v_k) + J(v_k) n||_2^2, while for [2,3],
!  m^v_k(n) := ||c(v_k) + J(v_k) n||_2. For [3] radius := radius^n_k, while for
!  [1,2] radius := min( radius^n_k, kappa_n ||c(x_k,c_k)||_2 )
!  ============================================================================

!  ........................... Normal Cauchy step .............................

!  compute the normal Cauchy point n^C_k := n^C_k(alpha_N),
!   n^C_k(alpha) := (n^Cx_k(alpha)) := -alpha P_k^2 J(v_k)^T c(v_k),
!                   (n^Cc_k(alpha))
!   alpha_N is the solution to
!    minimize_{alpha>= 0} m^v_k (n^C_k(alpha) )
!    subject to n^C_k(alpha) in N_k &
!      v^L + kappa_fbn (v_k - v^L) <= v_k + n^C_k(alpha)
!                                  <= v^U + kappa_fbn (v_k - v^U),

!  compute the step alpha for which alpha * || P_k J(v_k)^T c(v_k) ||_2 = radius

             SELECT CASE ( control%algorithm_variant )
             CASE ( 1, 2 )
               data%LLS_control%radius                                         &
                 = MIN( data%radius_n, data%kappa_n * data%norm_c )
             CASE ( 3 )
               data%LLS_control%radius = data%radius_n
             END SELECT
             alpha_r = data%LLS_control%radius /                               &
                         TWO_NORM( data%V_plus( : data%n_total ) )

!  find the largest alpha for which (A) is satisfied for alpha n^C_k

             IF ( data%fixed_variables ) THEN
               alpha_b = STEP_fraction_to_boundary(                            &
                 nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%CAUCHY,   &
                 data%omkappa_fbn, data%X_bound_type, nlp%X_l, nlp%X_u,        &
                 data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,               &
                 n_fr = data%n_fr, X_fr = data%X_fr )
             ELSE
               alpha_b = STEP_fraction_to_boundary(                            &
                 nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%CAUCHY,   &
                 data%omkappa_fbn, data%X_bound_type, nlp%X_l, nlp%X_u,        &
                 data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
             END IF

!  compute the Cauchy step n^C_k

!write(6,*) ' n alpha ', alpha_r, alpha_u, alpha_b

             alpha = MIN( alpha_r, alpha_u, alpha_b )
             data%CAUCHY( : data%n_total )                                     &
               = alpha * data%CAUCHY( : data%n_total )

!  compute m^v_k(n) := 1/2 ||c(v_k) + J(v_k) n||_2^2 at n = n^C_k and
!  Delta q^{v,n^C}_k := m^v_k(0) - m^v_k(n^C_k)

             data%mvnc = MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv,       &
                                  data%CAUCHY, data%W,                         &
                                  data%use_violation_squared )
             data%delta_mvnc = data%mv0 - data%mvnc

             IF ( data%printw ) WRITE( data%out,                               &
               "( /, A, ' * model at initial and Cauchy points = ',            &
              &   2ES12.4 )" ) prefix, data%mv0, data%mvnc

!  ...................... Normal Cauchy step found ............................

!  now compute the normal step n_k as an approximate solution to
!
!    minimize_{n} m^v_k(n) := 1/2 ||c(v_k) + J(v_k) n||_2^2
!    subject to n in N_k := {n: ||P_k^{-1} n||_2 <= radius^n_k}
!  and
!   ||P_k^{-1} n||_2 <= kappa_n ||c(x_k,c_k)||_2                             (C)

!  direct solution ...

             IF ( control%direct_solution_of_normal_model ) THEN
               IF ( data%printw ) WRITE( data%out,                             &
               "( /, A, ' * entering LLST for normal step: radius = ',         &
              &  ES9.2 )" ) prefix, data%LLS_control%radius

               CALL LLST_solve( nlp%m, data%n_total, data%LLS_control%radius,  &
                   data%Jv, - data%Cv, data%N, data%LLST_data,                 &
                   data%LLST_control, inform%LLST_inform, S = data%Pinv2 )

               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' * returned from LLST (normal step)' )" ) prefix

               inform%max_entries_factors_normal                               &
                 = MAX( inform%max_entries_factors_normal,                     &
                   inform%LLST_inform%SBLS_inform%SLS_inform%entries_in_factors)
               inform%factorizations_normal                                    &
                 = inform%factorizations_normal +                              &
                    inform%LLST_inform%factorizations

!  ... or iterative one

             ELSE
               IF ( data%printw ) WRITE( data%out,                             &
               "( /, A, ' * entering LLS for normal step: radius = ',          &
              &      ES9.2 )" )  prefix, data%LLS_control%radius

               CALL LLS_solve_main( data%n_total, nlp%m, data%Jv, data%Cv,     &
                                    data%mvn, data%N, data%LLS_data,           &
                                    data%LLS_control, inform%LLS_inform,       &
                                    S = data%P )

               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' * returned from LLS (normal step)' )" ) prefix

               inform%max_entries_factors_normal                               &
                 = MAX( inform%max_entries_factors_normal,                     &
                   inform%LLS_inform%SBLS_inform%SLS_inform%entries_in_factors )
             END IF

!  if required, summarize the normal-step iteration

             IF ( data%printt ) THEN
               IF ( control%direct_solution_of_normal_model ) THEN
                 IF ( data%LLST_control%out > 0 .AND.                          &
                      data%LLST_control%print_level > 0 )                      &
                   WRITE( data%out, "( '' )" )
                 WRITE( data%out, "( A, ' - on exit from LLST: status = ', I0, &
                & ', facts = ', I0, ', time = ', F0.2, /, A,                   &
                & ' - LLST residual =', ES12.4, ', || n || = ', ES10.4 )" )    &
                      prefix, inform%LLST_inform%status,                       &
                      inform%LLST_inform%factorizations,                       &
                      inform%LLST_inform%time%clock_total, prefix,             &
                      inform%LLST_inform%r_norm, inform%LLST_inform%x_norm

               ELSE
                 IF ( data%LLS_control%out > 0 .AND.                           &
                      data%LLS_control%print_level > 0 )                       &
                   WRITE( data%out, "( '' )" )
                 WRITE( data%out, "( A, ' - on exit from LLS: status = ', I0,  &
                & ', cg iter = ', I0, ', time = ', F0.2, /, A,                 &
                & ' - LLS objective value =', ES12.4,                          &
                & ', || n || = ', ES10.4 )" ) prefix,                          &
                      inform%LLS_inform%status, inform%LLS_inform%cg_iter,     &
                      inform%LLS_inform%time%clock_total, prefix,              &
                      inform%LLS_inform%obj, inform%LLS_inform%norm_x
               END IF
             END IF

!  find the largest alpha for which
!    v^L + kappa_fbn (v_k - v^L) <= v_k + alpha n_k
!      <= v^U + kappa_fbn (v_k - v^U)                                       (A2)
!  is satisfied along the normal step n_k

             IF ( data%fixed_variables ) THEN
               alpha_b = STEP_fraction_to_boundary(                            &
                 nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%N,        &
                 data%omkappa_fbn, data%X_bound_type, nlp%X_l, nlp%X_u,        &
                 data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,               &
                 n_fr = data%n_fr, X_fr = data%X_fr )
             ELSE
               alpha_b = STEP_fraction_to_boundary(                            &
                 nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%N,        &
                 data%omkappa_fbn, data%X_bound_type, nlp%X_l, nlp%X_u,        &
                 data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
             END IF

!  step back along the normal step if necessary to satisfy A2

!write(6,*) ' n alpha ', alpha_r, alpha_u, alpha_b

             alpha = MIN( one, alpha_b )
             IF ( data%printw .AND. alpha < one ) WRITE( data%out,             &
               "( A, '   normal step cut back by', ES11.4  )" ) prefix, alpha
             data%N( : data%n_total ) = alpha * data%N( : data%n_total )

!  compute m^v_k(n) := 1/2 ||c(v_k) + J(v_k) n||_2^2 at n = n_k and
!  Delta q^{v,n}_k := m^v_k(0) - m^v_k(n_k)

             data%mvn =                                                        &
               MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv, data%N, data%W, &
                        data%use_violation_squared )
             data%delta_mvn = data%mv0 - data%mvn

!  check that the approximation also satisfies
!    Delta q^{v,n}_k >= m^v_k(0) - m^v_k(n^C_k)                              (B)

             IF ( data%delta_mvn >= data%delta_mvnc ) THEN
               data%n_end = 'n'
             ELSE
               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, '   normal Cauchy step preferred ' )" ) prefix
!write(6,*) ' normal, Cauchy models',  data%delta_mvn, data%delta_mvnc
               data%N( : data%n_total ) = data%CAUCHY( : data%n_total )
               data%mvn = data%mvnc
               data%delta_mvn = data%delta_mvnc
               data%n_end = 'c'
             END IF

             data%n_eq_0 = .FALSE.

!  compute m^v_k(n) := 1/2 ||c(v_k) + J(v_k) n||_2^2 at n = n_k and
!  Delta q^{v,n}_k := m^v_k(0) - m^v_k(n_k)

             data%mvn =                                                        &
               MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv, data%N, data%W, &
                        data%use_violation_squared )
             data%delta_mvn = data%mv0 - data%mvn

!  [1,2:11-12] otherwise there is little benefit to be gained from computing
!  a normal step so set n_k = 0

           ELSE
             data%n_eq_0 = .TRUE.
             data%n_end = '0'
             data%mvn = data%mv0
             data%delta_mvn = zero
           END IF

!  we are already feasible, so set n_k = 0

         ELSE
           data%n_eq_0 = .TRUE.
           data%n_end = '0'
           data%mvn = data%mv0
           data%delta_mvn = zero
         END IF

!  -------------------  COMPUTE SECOND DERIVATIVE VALUES ----------------------

         IF ( data%new_hessian ) THEN

!  select a vector yhat_k satisfying ||yhat_k||_2 <= kappa_y (store in nlp%y)

!  recompute the Hessian of the Lagrangian at (x_k,yhat_k)

           IF ( data%reverse_hl ) THEN
             data%branch = 4 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_HL( data%eval_status, nlp%X, nlp%Y, userdata, nlp%H%val )
           END IF
         END IF

!  return from reverse communication with the Hessian of the Lagrangian

 240     CONTINUE

!  [1:14,2:13] compute the barrier Hessian ( Hess Lagrangian(x,y)  0 ) +
!                                   (            0          0 )
!      ( Z^L (X - X^L)^{-1} + Z^U (X - X^U)^{-1}           0          )
!      (          0           Y^L (C - C^L)^{-1} + Y^U (C - C^U)^{-1} )

         IF ( data%new_hessian ) THEN
           IF ( data%fixed_variables ) THEN
             ne = 0
             DO l = 1, nlp%H%ne
               i = data%X_order( nlp%H%row( l ) )
               j = data%X_order( nlp%H%col( l ) )
               IF ( i > 0 .AND. j > 0 ) THEN
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) = nlp%H%val( l )
               END IF
             END DO
           ELSE
             data%HESS_barrier%val( : nlp%H%ne ) = nlp%H%val( : nlp%H%ne )
             ne = nlp%H%ne
           END IF

           IF ( data%fixed_variables ) THEN
             DO ii = 1, data%n_fr
               i = data%X_fr( ii )
               SELECT CASE ( data%X_bound_type( i ) )
               CASE ( lower )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_l( i ) / ( data%V( ii ) - nlp%X_l( i ) )
               CASE ( upper )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_u( i ) / ( data%V( ii ) - nlp%X_u( i ) )
               CASE ( both )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_l( i ) / ( data%V( ii ) - nlp%X_l( i ) ) +           &
                   data%Z_u( i ) / ( data%V( ii ) - nlp%X_u( i ) )
               END SELECT
             END DO
           ELSE
             DO i = 1, nlp%n
               SELECT CASE ( data%X_bound_type( i ) )
               CASE ( lower )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_l( i ) / ( data%V( i ) - nlp%X_l( i ) )
               CASE ( upper )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_u( i ) / ( data%V( i ) - nlp%X_u( i ) )
               CASE ( both )
                 ne = ne + 1
                 data%HESS_barrier%val( ne ) =                                 &
                   data%Z_l( i ) / ( data%V( i ) - nlp%X_l( i ) ) +            &
                   data%Z_u( i ) / ( data%V( i ) - nlp%X_u( i ) )
               END SELECT
             END DO
           END IF

           DO ii = 1, data%m_in
             i = data%C_in( ii )
             j = data%n_fr + ii
             SELECT CASE ( data%C_bound_type( i ) )
             CASE ( lower )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Y_l( i ) / ( data%V( j ) - nlp%C_l( i ) )
             CASE ( upper )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Y_u( i ) / ( data%V( j ) - nlp%C_u( i ) )
             CASE ( both )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Y_l( i ) / ( data%V( j ) - nlp%C_l( i ) ) +              &
                 data%Y_u( i ) / ( data%V( j ) - nlp%C_u( i ) )
             END SELECT
           END DO
         END IF

         IF ( .NOT. data%n_eq_0 ) THEN

!  compute m^f_k(n) := f(v_k;mu) + grad f(v_k;mu)^T n + 1/2 n^T G_k n at n = n_k

           data%mfn = MODEL_f( data%n_total, data%barrier, data%GRAD_barrier,  &
                               data%HESS_barrier, data%N, data%W )

!  compute grad m^f_k(n_k)

           data%GRAD_mf( : data%n_total ) = GRAD_model_f( data%n_total,        &
             data%GRAD_barrier, data%HESS_barrier, data%N )

!  compute ||P_k^{-1} n_k||_2

           data%norm_n                                                         &
             = TWO_norm( data%N( : data%n_total ) / data%P( : data%n_total ) )

!  do the same if n_k = 0

         ELSE
           data%mfn = data%barrier
           data%GRAD_mf( : data%n_total ) = data%GRAD_barrier( : data%n_total )
           data%norm_n = zero
         END IF

!  set Delta m^fn_k := m^f_k(0) - m^f_k(n_k)

         data%delta_mfn = data%mf0 - data%mfn

!  if ||P_k^{-1} n_k||_2 <= kappa_B radius_k, consider finding a tangental step

!write(6,*) ' n ',  data%norm_n, data%kappa_B * data%radius

!  [1:15,2:14] decide whether to compute new Lagrange multiplier estimates

         IF ( data%norm_n <= data%kappa_B * data%radius ) THEN

!  =========================== MULTIPLIER SUBPROBLEM ==========================
!
!  [1:16,2:15] find appropriate Lagrange multipliers by approximately
!      minimizing{y in R^M} 1/2 ||P_k(grad m^f_k(n_k) - J^T(v_k) y)||^2_2
!
!  ============================================================================

!  ** to do ... approximate solve **

!  exact minimizer satisfies
!    ( P_k^-2   J^T(v_k) ) ( r_k ) = ( grad m^f_k(n_k) )                    (S)
!    ( J(v_k)       0    ) ( y_k )   (       0         )

!  factorize the matrix from (S)

           IF ( data%printw ) WRITE( data%out,                                 &
               "( /, A, ' * entering SBLS to compute multipliers' )" ) prefix
           CALL SBLS_form_and_factorize( data%n_total, nlp%m, data%Pinv2,      &
                                         data%Jv, data%C0, data%SBLS_data,     &
                                         data%SBLS_control,                    &
                                         inform%SBLS_inform )
           IF ( inform%SBLS_inform%status < 0 ) THEN
             inform%status = GALAHAD_error_factorization
             GO TO 910
           END IF
           inform%max_entries_factors_multipliers                              &
             = MAX( inform%max_entries_factors_multipliers,                    &
                    inform%SBLS_inform%SLS_inform%entries_in_factors )

!  set up the righ-hand side of (S)

           data%W( : data%n_total ) = data%GRAD_mf( : data%n_total )
           data%W( data%n_total + 1 : data%n_total + nlp%m ) = zero

!  solve (S)

           data%SBLS_control%affine = .TRUE.
           CALL SBLS_solve( data%n_total, nlp%m, data%Jv, data%C0,             &
                            data%SBLS_data, data%SBLS_control,                 &
                            inform%SBLS_inform, data%W )

!  find an approximate minimizer of ||P_k(grad m^f_k(n_k) - J(v_k)^T y)||^2

           data%Y( : nlp%m ) = data%W( data%n_total + 1 : data%n_total + nlp%m )

!  record Largrange multiplier and dual variable estimates y and z

!write(6,*) ' nlp y = from LS problem'
           nlp%Y( : nlp%m ) = data%Y( : nlp%m )
           IF ( data%fixed_variables ) THEN
             nlp%Z( data%X_fr( : data%n_fr ) )                                 &
               = nlp%Z( data%X_fr( : data%n_fr ) )                             &
                 + data%Pinv2%val( : data%n_fr ) * data%W( : data%n_fr )
           ELSE
             nlp%Z( : nlp%n ) = nlp%Z( : nlp%n )                               &
               + data%Pinv2%val( : nlp%n ) * data%W( : nlp%n )
           END IF
           IF ( data%printw ) WRITE( data%out,                                 &
               "( A, ' * multipliers computed' )" ) prefix

           IF (.FALSE. ) THEN ! ************************ FALSE **************
!          IF (.TRUE. ) THEN ! ************************** TRUE ***************

!  split z = z_l + z_u with z_l >= 0, z_u <= 0 so as to minimize
!    || (x-x_l) z_l - mu ||^2 + || (x-x_u) z_u - mu ||^2

           IF ( data%fixed_variables ) THEN
             DO ii = 1, data%n_fr
               i = data%X_fr( ii )
               SELECT CASE ( data%X_bound_type( i ) )
               CASE ( lower )
                 nlp%Z( i ) = MAX( nlp%Z( i ), zero )
                 data%Z_l( i ) = nlp%Z( i )
               CASE ( upper )
                 nlp%Z( i ) = MIN( nlp%Z( i ), zero )
                 data%Z_u( i ) = nlp%Z( i )
               CASE ( both )
                 data%Z_l( i ) = MAX( nlp%Z( i ), zero,                        &
                     ( nlp%Z( i ) * ( data%V( ii ) - nlp%X_u( i ) ) ** 2       &
                       + ( nlp%X_u( i ) - nlp%X_l( i ) ) * data%mu )           &
                       / ( ( data%V( ii ) - nlp%X_l( i ) ) ** 2                &
                           + ( data%V( ii ) - nlp%X_u( i ) ) ** 2 ) )
                 data%Z_u( i ) = nlp%Z( i ) - data%Z_l( i )
               END SELECT
             END DO
           ELSE
             DO i = 1, nlp%n
               SELECT CASE ( data%X_bound_type( i ) )
               CASE ( lower )
                 nlp%Z( i ) = MAX( nlp%Z( i ), zero )
                 data%Z_l( i ) = nlp%Z( i )
               CASE ( upper )
                 nlp%Z( i ) = MIN( nlp%Z( i ), zero )
                 data%Z_u( i ) = nlp%Z( i )
               CASE ( both )
                 data%Z_l( i ) = MAX( nlp%Z( i ), zero,                        &
                     ( nlp%Z( i ) * ( data%V( i ) - nlp%X_u( i ) ) ** 2        &
                       + ( nlp%X_u( i ) - nlp%X_l( i ) ) * data%mu )           &
                       / ( ( data%V( i ) - nlp%X_l( i ) ) ** 2                 &
                           + ( data%V( i ) - nlp%X_u( i ) ) ** 2 ) )
                 data%Z_u( i ) = nlp%Z( i ) - data%Z_l( i )
               END SELECT
             END DO
           END IF

!  do the same for y

!write(6,*) ' y_l = y+'
           DO ii = 1, data%m_in
             i = data%C_in( ii )
             j = data%n_fr + ii
             SELECT CASE ( data%C_bound_type( i ) )
             CASE ( lower )
               nlp%Y( i ) = MAX( nlp%Y( i ), zero )
               data%Y_l( i ) = nlp%Y( i )
!write(6,*) ' y_l ',  data%Y_l( i )
             CASE ( upper )
               nlp%Y( i ) = MIN( nlp%Y( i ), zero )
               data%Y_u( i ) = nlp%Y( i )
             CASE ( both )
               data%Y_l( i ) = MAX( nlp%Y( i ), zero,                          &
                   ( nlp%Y( i ) * ( data%V( j ) - nlp%C_u( i ) ) ** 2          &
                     + ( nlp%C_u( i ) - nlp%C_l( i ) ) * data%mu )             &
                     / ( ( data%V( j ) - nlp%C_l( i ) ) ** 2                   &
                         + ( data%V( j ) - nlp%C_u( i ) ) ** 2 ) )
               data%Y_u( i ) = nlp%Y( i ) - data%Y_l( i )
             END SELECT
           END DO
           END IF  !  ************************ END ***********************

!  record the norm of the multipliers

           IF ( data%n_fr > 0 ) THEN
             IF ( data%fixed_variables ) THEN
               data%max_y = MAXVAL( ABS( nlp%Z( data%X_fr( : data%n_fr ) ) ) )
             ELSE
               data%max_y = MAXVAL( ABS( nlp%Z( : nlp%n ) ) )
             END IF
           ELSE
             data%max_y = zero
           END IF

           IF ( nlp%m > 0 ) THEN
             data%max_y = MAX( data%max_y, MAXVAL( ABS( nlp%Y( : nlp%m ) ) ) )
           END IF

           data%max_y = MAX( data%max_y, one )

!write(6,*) 'E z(1) ', data%Z_l( 1 ), data%Z_u( 1 )
!write(6,*) ' x - x_l ', data%V(1) - nlp%X_l(1)
!write(6,*) ' x - x_u ', data%V(1) - nlp%X_u(1)

!        IF ( data%norm_n <= data%kappa_B * data%radius ) THEN

!  compute y_k and r_k satisfying
!             r_k :=  (r^x_k) := P_k^2 (grad m^f_k(n_k) - J(v_k)^T y_k )
!                     (r^c_k)
!  and
!   ||y_k - y^Lc_k||_2} <= omega_y( ||c(v_k)||_2),
!   <P_k grad m^f_k(n_k><P_k(grad m^f_k(n_k) - J(v_k)^T y_k)>
!        == grad m^f_k(n_k)^T r_k >= 0,
!  and
!   ||P_k (grad m^f_k(n_k) - J(v_k)^T y_k )||_2
!     <= kappa_nr ||P_k gradm^f_k(n_k)||_2

!  compute P_k (grad m^f_k(n_k) - J(v_k)^T y_k )

           data%V_plus( : data%n_total ) = data%GRAD_mf( : data%n_total )
           DO l = 1, data%Jv%ne
             j = data%Jv%col( l )
             data%V_plus( j ) = data%V_plus( j )                               &
               - data%Jv%val( l ) * data%Y( data%Jv%row( l ) )
           END DO
           data%V_plus( : data%n_total )                                       &
             = data%P( : data%n_total ) * data%V_plus( : data%n_total )

!do i = 1, nlp%n
!write(6,*) ' i, x, g ', i, nlp%X( i ), data%GRAD_mf( i )
!end do

!write(6,*) ' ||Pr|| ', TWO_NORM( data%V_plus( : data%n_total ) )
!  compute r_k := P_k^2 (grad m^f_k(n_k) - J(v_k)^T y_k )

           data%R( : data%n_total )                                            &
              = data%P( : data%n_total ) * data%V_plus( : data%n_total )

!  compute the criticality measure
!    pi_k^f := gradm^f_k(n_k)^T r_k} /
!               ||P_k (grad m^f_k(n_k) - J(v_k)^T y_k )||_2} if r_k /= 0     (P)
!              0                                             otherwise

           val = TWO_NORM( data%V_plus( : data%n_total ) )

           IF ( .TRUE. ) THEN
             data%pi_f = val
           ELSE
             IF ( val /= zero ) THEN
               data%pi_f = DOT_PRODUCT( data%GRAD_mf( : data%n_total ),        &
                                        data%R( : data%n_total ) ) / val
               IF ( data%pi_f < zero ) data%pi_f = val
             ELSE
               data%pi_f = zero
             END IF
           END IF

!  =========================== TANGENTIAL SUBPROBLEM ==========================
!
!  if pi^f_k > omega_t ( ||c(v_k)||_2 ) compute the tangential step t_k
!  approximately

!    minimizing{T} m^f_k(n_k + t) such that J(v_k) t = 0,
!     n_k + t_k in B_k := { d :  ||P_k^{-1} d ||_2} <= radius },             (R)
!
!    and
!
!     v^L + kappa_fbt (v_k + n_k - v^L) <= v_k + n_k + t                     (F)
!      <= v^U + kappa_fbt (v_k + n_k - v^U)
!
!  for some appropriate radius
!  ============================================================================
!

!  [1:17-18,2:16-17] check for termination

!          IF ( data%pi_f <= epsmch .AND.                                      &
           IF ( data%pi_f <= data%stop_d_inner .AND.                           &
               inform%primal_infeasibility <= data%stop_p_inner ) THEN
!write(6,*) ' optimality test 2 satisfied '

!!!! only for exmeremtation !!!

!data%Y_l( 1 ) = nlp%Y( 1 )

             GO TO 510

!  [1:19-20,2:18-19] if the criticality measure is too small realtive to the
!  infeasibility, skip the tangential step

           ELSE IF ( data%pi_f <= FORCING( 't', data%pi_v ) ) THEN
             data%t_eq_0 = .TRUE.
             data%t_end = '0'
             data%delta_mft = zero
             data%delta_mfd = data%delta_mfn

!  [1:21,2:20;3:22] compute a tangential step

           ELSE

!  ........................... Tangential Cauchy step .........................

!  [1:22-25,2:21-24,3:23-26] compute the tangential Cauchy step
!   t^C_k = t^C_k(alpha_T), where
!     t^C_k(alpha) := (t^Cx_k(alpha)) := -alpha (r^x_k) = -alpha r_k,
!                     (t^Cc_k(alpha))           (r^c_k)
!  and alpha_T is the minimizer of
!      minimize_{alpha >= 0}
!      m^f_k (n_k + t^C_k(alpha) )
!      subject to ||P_k^{-1} (n_k + t^C_k(alpha) )||_2 <= radius_k,
!      v^L + kappa_fbt (v_k + n_k - v^L) <= v_k + n_k + t^C_k(alpha) <=
!        v^U + kappa_fbt (v_k + n_k - v^U),

!  compute - r_k (store in t^C_k)

             data%CAUCHY( : data%n_total ) = - data%R( : data%n_total )

!  compute the step alpha so that
!  ||P_k^{-1} (n_k + alpha t^C_k) )||_2 = radius for appropriate radius ...

             IF ( .NOT. data%n_eq_0 ) THEN
               radius = data%radius
               data%V_plus( : data%n_total ) = data%CAUCHY( : data%n_total ) / &
                                                 data%P( : data%n_total )
               data%W( : data%n_total ) = data%N( : data%n_total ) /           &
                                            data%P( : data%n_total )
               a2 = ( TWO_NORM( data%V_plus( : data%n_total ) ) ) ** 2
               a1 = two * DOT_PRODUCT( data%V_plus( : data%n_total ),          &
                                       data%W( : data%n_total ) )
               a0 = ( TWO_NORM( data%W( : data%n_total ) ) ) ** 2 - radius ** 2
               CALL ROOTS_quadratic( a0, a1, a2, epsmch, i, av, alpha_r,       &
                                     .FALSE. )

!  ...or alpha ||P_k^{-1} alpha t^C_k )||_2 = radius (if n_k = 0)

             ELSE
               IF ( data%use_violation_squared ) THEN
                 radius = MIN( data%radius,                                    &
                               SQRT( data%kappa_v * data%violation_max ) )
               ELSE
                 radius = MIN( data%radius, data%kappa_v * data%violation_max )
               END IF
               alpha_r = data%radius /                                         &
                           TWO_NORM( data%CAUCHY( : data%n_total ) /           &
                                     data%P( : data%n_total ) )
             END IF

!  compute the line minimizer of m^f_k (n_k + t^C_k(alpha) ). First compute
!  H t^C_k and ( t^C_k )^T H t^C_k

             data%W( : data%n_total ) = zero
             DO l = 1, data%HESS_barrier%ne
               i = data%HESS_barrier%row( l ) ; j = data%HESS_barrier%col( l )
               val = data%HESS_barrier%val( l )
               data%W( i ) = data%W( i ) + val * data%CAUCHY( j )
               IF ( i /= j ) data%W( j ) = data%W( j ) + val * data%CAUCHY( i )
             END DO
             av = DOT_PRODUCT( data%W( : data%n_total ),                       &
                               data%CAUCHY( : data%n_total ) )

!  if ( t^C_k )^T H t^C_k is positive the minimizer is
!   -  ( t^C_k )^T grad m^f(n_k) / ( t^C_k )^T H t^C_k

             IF ( av > zero ) THEN
               alpha_u = - DOT_PRODUCT( data%GRAD_mf( : data%n_total ),        &
                                        data%CAUCHY( : data%n_total ) ) / av

!  otherwise the minimizer is at - infinity

             ELSE
               alpha_u = HUGE( one )
             END IF

!  finally find the largest alpha for which
!      v^L + kappa_fbt (v_k + n_k - v^L) <= v_k + n_k + alpha t^C_k
!        <= v^U + kappa_fbt (v_k + n_k - v^U) is satisfied for alpha t^C_k

             IF ( data%fixed_variables ) THEN
               IF ( data%n_eq_0 ) THEN
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%CAUCHY, &
                   data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,      &
                   data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,             &
                   n_fr = data%n_fr, X_fr = data%X_fr )
               ELSE
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V + data%N,     &
                   data%CAUCHY, data%omkappa_fbt, data%X_bound_type,           &
                   nlp%X_l, nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l,    &
                   nlp%C_u, n_fr = data%n_fr, X_fr = data%X_fr )
               END IF
             ELSE
               IF ( data%n_eq_0 ) THEN
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%CAUCHY, &
                   data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,      &
                   data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
               ELSE
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V + data%N,     &
                   data%CAUCHY, data%omkappa_fbt, data%X_bound_type,           &
                   nlp%X_l, nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l,    &
                   nlp%C_u )
               END IF
             END IF

!  compute the Cauchy step

!write(6,*) ' t alpha ', alpha_r, alpha_u, alpha_b

             alpha = MIN( alpha_r, alpha_u, alpha_b )
             data%CAUCHY( : data%n_total )                                     &
               = alpha * data%CAUCHY( : data%n_total )

!  compute m^f_k(d) := f(v_k;mu) + grad f(v_k;mu)^T d + 1/2 d^T G_k d at
!  d = n_k + t^C_k and Delta m^ftC_k := m^f_k(n_k) - m^f_k(n_k + t^C_k)

             IF ( .NOT. data%n_eq_0 ) THEN
               data%mfdc = MODEL_f( data%n_total, data%barrier,                &
                 data%GRAD_barrier, data%HESS_barrier, data%N + data%CAUCHY,   &
                 data%W )
             ELSE
               data%mfdc = MODEL_f( data%n_total, data%barrier,                &
                 data%GRAD_barrier, data%HESS_barrier, data%CAUCHY, data%W )
             END IF
             data%delta_mftc = data%mfn - data%mfdc

!  ....................... Tangential Cauchy step found .......................

!  now compute the tangential step t_k as an approximate solution to (R)

             data%T( : data%n_total ) = zero

!  debug if necessary

             IF ( data%printd ) THEN
               WRITE( data%out, "( /, A, ' EQP subproblem ' )" ) prefix
               WRITE( data%out, "( ( A, ' i, X, G ', I6, 2ES12.4 ) )" )        &
                 ( prefix, i, data%T( i ), data%GRAD_mf( i ),                  &
                   i = 1, data%n_total )
               IF ( nlp%m > 0 ) THEN
!                WRITE( data%out, "( ( A, ' i, C, Y ', I6, 2ES12.4 ) )" )      &
!                  ( prefix, i, data%C_in( i ), data%Y( i ), i = 1, nlp%m )
                 WRITE( data%out, "( ' J: row, col, val', /, 3( 2I6, ES12.4))")&
                   ( data%Jv%row( i ), data%Jv%col( i ), data%Jv%val( i ),     &
                     i = 1, data%Jv%ne )
               END IF
               WRITE( data%out, "( ' H: row, col, val', /, 3( 2I6, ES12.4 ) )")&
                 ( data%HESS_barrier%row( i ), data%HESS_barrier%col( i ),     &
                   data%HESS_barrier%val( i ), i = 1, data%HESS_barrier%ne )
             END IF

!  ensure (R) is feasible by choosing t_i in
!   { t : ||P_k^{-1} t ||_2} <= radius := radius_k - ||P_k^{-1} s_n ||_2 }  (R2)

             IF ( data%n_eq_0 ) THEN
               data%EQP_control%radius = radius
             ELSE
               data%EQP_control%radius = radius - data%norm_n
             END IF

!write(6,*) ' Pinv2 ', data%Pinv2%val

!  approximately solve (R)

             IF ( control%direct_solution_of_tangential_model ) THEN
               IF ( data%printw ) WRITE( data%out,                             &
                 "( /, A, ' * entering TRS for tangential step: radius = ',    &
              &     ES9.2 )" ) prefix, data%EQP_control%radius
               CALL TRS_solve( data%n_total, data%EQP_control%radius,          &
                               data%barrier, data%GRAD_mf, data%HESS_barrier,  &
                               data%T, data%TRS_data, data%TRS_control,        &
                               inform%TRS_inform, M = data%Pinv2, A = data%Jv, &
                               Y = data%Y )
               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' * returned from TRS (tangential step)' )" ) prefix
               data%mfd = inform%TRS_inform%obj
               inform%max_entries_factors_tangential                           &
                 = MAX( inform%max_entries_factors_tangential,                 &
                   inform%TRS_inform%SLS_inform%entries_in_factors )
               inform%factorizations_tangential                                &
                 = inform%TRS_inform%factorizations
             ELSE
               data%EQP_control%preconditioner = 5
               IF ( data%printw ) WRITE( data%out,                             &
                 "( /, A, ' * entering EQP for tangential step: radius = ',    &
              &     ES9.2 )" ) prefix, data%EQP_control%radius
               CALL EQP_solve_main( data%n_total, nlp%m, data%HESS_barrier,    &
                                    data%GRAD_mf, data%barrier, data%Jv,       &
                                    data%mfd, data%T, data%Y, data%EQP_data,   &
                                    data%EQP_control, inform%EQP_inform,       &
                                    D = data%Pinv2%val )
               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' * returned from EQP (tangential step)' )" ) prefix
               inform%max_entries_factors_tangential                           &
                 = MAX( inform%max_entries_factors_tangential,                 &
                   inform%EQP_inform%SBLS_inform%SLS_inform%entries_in_factors )
               inform%factorizations_tangential                                &
                 = inform%factorizations_tangential + 1
             END IF

!  find the largest alpha for which
!      v^L + kappa_fbt (v_k + n_k - v^L) <= v_k + n_k + alpha t_k           (F2)
!        <= v^U + kappa_fbt (v_k + n_k - v^U) is satisfied for alpha t_k

             IF ( data%fixed_variables ) THEN
               IF ( data%n_eq_0 ) THEN
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%T,      &
                   data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,      &
                   data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,             &
                   n_fr = data%n_fr, X_fr = data%X_fr )
               ELSE
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V + data%N,     &
                   data%T, data%omkappa_fbt, data%X_bound_type, nlp%X_l,       &
                   nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,    &
                   n_fr = data%n_fr, X_fr = data%X_fr )
               END IF
             ELSE
               IF ( data%n_eq_0 ) THEN
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%T,      &
                   data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,      &
                   data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
               ELSE
                 alpha_b = STEP_fraction_to_boundary(                          &
                   nlp%n, nlp%m, data%m_in, data%n_total, data%V + data%N,     &
                   data%T, data%omkappa_fbt, data%X_bound_type, nlp%X_l,       &
                   nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
               END IF
             END IF

!  step back along the step t_k -> alpha t_k if necessary to satisfy (F2)

             alpha = MIN( one, alpha_b )
!write(6,*) ' step ', alpha
             data%T( : data%n_total ) = alpha * data%T( : data%n_total )

!  compute m^f_k(n) := f(v_k;mu) + grad f(v_k;mu)^T n + 1/2 n^T G_k n at n = d_k
!  and Delta m^ft_k := m^f_k(n_k) - m^f_k(n_k + t_k)

             IF ( .NOT. data%n_eq_0 ) THEN
               data%mfd = MODEL_f( data%n_total, data%barrier,                 &
                 data%GRAD_barrier, data%HESS_barrier, data%N + data%T, data%W )
             ELSE
               data%mfd = MODEL_f( data%n_total, data%barrier,                 &
                 data%GRAD_barrier, data%HESS_barrier, data%T, data%W )
!write(6,*) ' slope ', DOT_PRODUCT( data%GRAD_barrier( : data%n_total ),       &
!                                   data%T( : data%n_total ) )
             END IF
             data%delta_mft = data%mfn - data%mfd
!write(6,*) ' mf0, mfn, mfd ', data%mf0, data%mfn, data%mfd

!  check that the approximation also satisfies
!    Delta m^ft_k) >= m^f_k(n_k) - m^f_k(n_k + t^C_k)                       (B2)
!
             IF ( data%delta_mft >= data%delta_mftc ) THEN
               data%t_end = 't'
             ELSE
               data%T( : data%n_total ) = data%CAUCHY( : data%n_total )
               data%mfd = data%mfdc
               data%delta_mft = data%delta_mftc
               data%t_end = 'c'
             END IF

!  ** to do ... inexact solves **

! [ensure also that either
!  (1a) ||c(v_k) + J(v_k)(n_k + t_k)||_2^2 <=
!          kappa_tg ||c(v_k)||_2^2 + (1-kappa_tg) ||c(v_k)+J(v_k)n_k||_2^2
!       if n_k /= 0
!  or
!  (2a) 1/2 ||c(v_k) + J(v_k)(n_k + t_k)||_2^2 <= kappa_tt vi^max_k
!       if n_k = 0
!  for [1], or either
!  (1b) ||c(v_k) + J(v_k)(n_k + t_k)||_2 <=
!          kappa_tg ||c(v_k)||_2 + (1-kappa_tg) ||c(v_k)+J(v_k)n_k||_2
!       if n_k /= 0
!  or
!  (2b) ||c(v_k) + J(v_k)(n_k + t_k)||_2 <= kappa_tt vi^max_k
!       if n_k = 0
!  for [2,3], holds - (1) and (2) are guaranteed for EQP]

             data%t_eq_0 = .FALSE.

!  compute Delta q^{f,d}_k := Delta q^{f,t}_k + Delta q^{f,n}_k

             data%delta_mfd = data%delta_mfn + data%delta_mft

!  compute ||P_k^{-1} t_k||_2

             data%norm_t =                                                     &
               TWO_norm( data%T( : data%n_total ) / data%P( : data%n_total ) )

!  [1:26-28,2:25-27,3:27-29] skip the tangential step if the proposed
!  tangential step is too large relative to the decrease in the model of the
!  barrier function,  i.e., ||P_k^{-1} t_k||_2 > kappa_VS ||P_k^{-1} n_k||_2
!  and Delta q^{f,d}_k < kappa_delta Delta q^{f,t}_k, ...

             IF ( data%norm_t > data%kappa_VS * data%norm_n .AND.              &
                  data%delta_mfd < data%kappa_delta * data%delta_mft ) THEN
               data%t_eq_0 = .TRUE.
               data%t_end = '0'
               data%delta_mft = zero
               data%delta_mfd = data%delta_mfn
             END IF
           END IF

!  ... or the normal step already almost lies on the trust-region boundary

         ELSE
           IF ( data%printw ) WRITE( data%out,                                 &
             "( /, A, ' * skipping tangential step as normal step =', ES12.4,  &
            &   /, A, '   is larger than max fraction of radius   =', ES12.4)")&
              prefix, data%norm_n, prefix, data%kappa_B * data%radius
           data%t_eq_0 = .TRUE.
           data%t_end = '0'
           data%delta_mft = zero
           data%delta_mfd = data%delta_mfn

!  in the latter case, set y_k = 0, define r_k := P_k^2 grad m^f_k(n_k),
!  and compute pi^f_k from (P)

           data%V_plus( : data%n_total )                                       &
             = data%P( : data%n_total ) * data%GRAD_mf( : data%n_total )
           data%R( : data%n_total )                                            &
              = data%P( : data%n_total ) * data%V_plus( : data%n_total )
           val = TWO_NORM( data%V_plus( : data%n_total ) )
           IF ( val /= zero ) THEN
             data%pi_f = DOT_PRODUCT( data%GRAD_mf( : data%n_total ),          &
                                      data%R( : data%n_total ) ) / val
             IF ( data%pi_f < zero ) data%pi_f = val
           ELSE
             data%pi_f = zero
           END IF
         END IF

!  set the trial step d_k = n_k + t_k and trial iterate v_k^+ = v_k + d_k

         IF ( .NOT. ( data%n_eq_0 .AND. data%t_eq_0 ) ) THEN
           data%f_current = nlp%f
           data%C_current( : nlp%m ) = nlp%C( : nlp%m )
           IF ( data%n_eq_0 ) THEN
             data%V_plus( : data%n_total ) = data%V( : data%n_total ) +        &
               data%T( : data%n_total )
           ELSE IF ( data%t_eq_0 ) THEN
             data%V_plus( : data%n_total ) = data%V( : data%n_total ) +        &
               data%N( : data%n_total )
           ELSE
             data%V_plus( : data%n_total ) = data%V( : data%n_total ) +        &
               data%N( : data%n_total ) + data%T( : data%n_total )
           END IF

!  ------------------------  COMPUTE FUNCTION VALUES --------------------------

!  evaluate the function and general constraint values

           IF ( data%fixed_variables ) THEN
             nlp%X( data%X_fr( : data%n_fr ) ) = data%V_plus( : data%n_fr )
           ELSE
             nlp%X = data%V_plus( : nlp%n )
           END IF
           IF ( data%reverse_fc ) THEN
             data%branch = 6 ; inform%status = 2 ; RETURN
           ELSE
             CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
           END IF
         END IF

!  return from reverse communication with the function and constraint values

 260     CONTINUE

!   --------------------------------------------------
!   [1:37,2:36,3:38] if d_k = 0, this is a Y-ITERATION
!   --------------------------------------------------

         IF ( data%n_eq_0 .AND. data%t_eq_0 ) THEN
           IF ( data%printw ) WRITE( data%out, "( /, A, ' * Y-iteration' )" )  &
             prefix
           data%it_type = 'Y'
           data%success = ' '

!  [1:38,2:37,3:39] perform the y-iteration updates given by
!    v_{k+1} <- v_k,
!    delta^n_{k+1} <- delta^n_k,
!    delta^t_{k+1} <- delta^t_k, and
!    v_{k+1}^{max} <-  vi^max_k,
!  i.e., leave things as they are

           data%new_gradient = .FALSE.
           data%new_hessian = .FALSE.

!  compute the new barrier function

         ELSE
           inform%f_eval = inform%f_eval + 1
           IF ( data%fixed_variables ) THEN
             data%barrier_new =                                                &
               BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f,          &
                        data%V_plus, data%mu, data%X_bound_type, nlp%X_l,      &
                        nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l,        &
                        nlp%C_u, n_fr = data%n_fr, X_fr = data%X_fr )
           ELSE
             data%barrier_new =                                                &
               BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f,          &
                        data%V_plus, data%mu, data%X_bound_type, nlp%X_l,      &
                        nlp%X_u, data%C_bound_type, data%C_in, nlp%C_l,        &
                        nlp%C_u )
           END IF
!write(6,*) ' barrier new, old ', data%barrier_new, data%barrier

!  compute c(v_k^+) (store in r)...

           data%R( : nlp%m ) = nlp%C( : nlp%m )
           DO ii = 1, data%m_eq
             i = data%C_eq( ii )
             data%R( i ) = data%R( i ) - nlp%C_l( i )
           END DO
           DO ii = 1, data%m_in
             i = data%C_in( ii )
             data%R( i ) = data%R( i ) - data%V_plus( data%n_fr + ii )
           END DO

!  ... and the new violation vi(v_k^+)

           data%norm_c_plus = TWO_NORM( data%R( : nlp%m ) )
           IF ( data%use_violation_squared ) THEN
             data%violation_plus = half * data%norm_c_plus ** 2
           ELSE
             data%violation_plus = data%norm_c_plus
           END IF

!  compute Delta q^{f,d}_k := Delta q^{f,t}_k + Delta q^{f,n}_k

           data%delta_mfd = data%delta_mfn + data%delta_mft

!  ---------------------------------------------------
!  [1:39,2:38,3:40] otherwise if t_k /= 0,
!   Delta q^{f,d}_k >= kappa_delta Delta q^{f,t}_k and
!   vi(v_k^+) <= vi^max_k, this is an F-ITERATION
!  ---------------------------------------------------

           IF ( .NOT. data%t_eq_0 .AND.                                        &
                 data%delta_mfd >= data%kappa_delta * data%delta_mft .AND.     &
                 data%violation_plus <= data%violation_max ) THEN
             data%it_type = 'F'

!  compute rho^f_k := [ f(w_k;mu) - f(w_k^+;mu) ] / Delta q^{f,d}_k

             data%rho = ( data%barrier - data%barrier_new ) / data%delta_mfd
!            write(6,*) ' rho ', data%rho

!  [1:40,2:39,3:41] perform the f-iteration updates:
!  if rho_k^f >= eta_1 is satisfied, set
!    v_{k+1} = v_k^+
!    perform the slack reset, compute c(v_k+1) and the violation
!    vi(v_k+1) :=  m^v_k+1(0) := 1/2 ||c(v_k+1)||_2^2, set
!    delta^t_{k+1} in [delta^t_k, infty)  if rho_k^f >= eta_2,
!                    [gamma_2 delta^t_k, delta^t_k ] otherwise
!    delta^n_{k+1} >= max( kappa_deltavv pi^vi(v_{k+1}), delta^n_k )
!                     for [1,2] or
!                  >= delta^n_k for [3]
!    vi^max_{k+1} = vi^max_k (nothing changes)
!    and raise the tau flag

             IF ( data%rho >= data%eta_successful ) THEN
               inform%obj = nlp%f
               data%barrier = data%barrier_new
               data%mf0 = data%barrier
               data%V( : data%n_total ) = data%V_plus( : data%n_total )
!              data%Cv( : nlp%m ) = data%R( : nlp%m )
!              data%norm_c = data%norm_c_plus
!              data%violation = data%violation_plus
               CALL FUNNEL_slack_reset( nlp%m, data%m_eq, data%m_in, data%C_eq,&
                            data%C_in, data%n_fr, data%C_bound_type, nlp%C,    &
                            nlp%C_l, nlp%C_u, data%V, data%Cv, data%norm_c,    &
                            data%violation,                                    &
                            data%use_violation_squared, data%mv0 )
               IF (  data%rho >= data%eta_very_successful ) THEN
                 IF ( data%printw ) WRITE( data%out, "( /, A, ' * very ',      &
                &  'successful F-iteration, rho =', ES12.4 )" ) prefix, data%rho
                 data%success = 'V'
                 data%radius_t = data%radius_increase * data%radius_t
                 IF ( control%algorithm_variant < 3 ) THEN
                   data%radius_n = MAX( data%radius_increase * data%radius_n,  &
                                        data%kappa_deltavv * data%violation )
                 ELSE
                   data%radius_n = data%radius_increase * data%radius_n
                   data%tau_flag_raised = .TRUE.
                 END IF
               ELSE
                 IF ( data%printw ) WRITE( data%out, "( /, A, ' * ',           &
                &  'successful F-iteration, rho =', ES12.4 )" ) prefix, data%rho
                 data%success = 'S'
                 IF ( control%algorithm_variant < 3 ) THEN
                   data%radius_n = MAX( data%radius_n,                         &
                                        data%kappa_deltavv * data%violation )
                 ELSE
                   data%tau_flag_raised = .TRUE.
                 END IF
               END IF
               data%new_gradient = .TRUE.
               data%new_hessian = .TRUE.

!  otherwise set
!    v_{k+1} = v_k
!    delta^t_{k+1} in [gamma_1 delta^t_k, gamma_2 delta^t_k ]
!    delta^n_{k+1} = delta^n_k (nothing changes) and
!    vi^max_{k+1} = vi^max_k (nothing changes)

             ELSE
               IF ( data%printw ) WRITE( data%out, "( /, A, ' * unsuccessful', &
              &  ' F-iteration, rho =', ES12.4 )" ) prefix, data%rho
               data%success = 'U'
               nlp%f = data%f_current
               nlp%C( : nlp%m ) = data%C_current( : nlp%m )
               data%radius_t = data%radius_decrease * data%radius_t
               data%new_gradient = .FALSE.
               data%new_hessian = .FALSE.
             END IF

!   ------------------------------------------------
!   [1:41,2:40,3:42] otherwise this is a V-ITERATION
!   ------------------------------------------------

           ELSE
             data%it_type = 'V'

!  compute m^v_k(d_k) and Delta q^{v,d}_k := m^v_k(0) - m^v_k(d_k)

             IF ( data%n_eq_0 ) THEN
               data%mvd = MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv,      &
                                   data%T, data%W, data%use_violation_squared )

             ELSE IF ( data%t_eq_0 ) THEN
               data%mvd = MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv,      &
                                   data%N, data%W, data%use_violation_squared )
             ELSE
               data%mvd = MODEL_v( nlp%m, data%n_total, data%Cv, data%Jv,      &
                                   data%N + data%T, data%W,                    &
                                   data%use_violation_squared )
             END IF
             data%delta_mvd = data%mv0 - data%mvd

!  compute rho_k^v := [ vi(v_k) - vi(v_k^+) ] / Delta q^{v,d}_k

             data%rho                                                          &
               = ( data%violation - data%violation_plus ) / data%delta_mvd
!write(6,*) data%violation, data%violation_plus, data%mv0, data%mvd
!write(6,*) ' rho ', data%rho

!  compute the acceptability test (ACC)
!    n_k /= 0 and Delta q^{v,d}_k >= kappa_cd Delta q^{v,n}_k

             data%acc = .NOT. data%n_eq_0 .AND.                                &
                data%delta_mvd >= data%kappa_cd * data%delta_mvn

!  [1:42,2:41,3:43] perform the v-iteration updates:
!  if ACC and  rho_k^v >= eta_1 is satisfied, set
!    v_{k+1} = v_k^+
!    perform the slack reset, compute c(v_k+1) and the violation
!    vi(v_k+1) :=  m^v_k+1(0) := 1/2 ||c(v_k+1)||_2^2, set
!    delta^n_{k+1} >= max [kappa_deltavv pi^vi(v_k+1), radius^n_k ]
!                       if rho_k^v >= eta_2 is satisfied,
!                   = max [kappa_deltavv pi^vi(v_k+1), radius^n_k ]
!                       otherwise
!                     for [1,2] or
!    delta^n_{k+1} >= radius^n_k if rho_k^v >= eta_2 is satisfied,
!                   = radius^n_k otherwise (nothing changes)
!                     for [3]
!    vi^max_{k+1} = max[ kappa_t1 vi^max_k,
!                        vi(v_k+1) + kappa_t2( vi(v_k) - vi(v_k+1)) ] and
!    radius^t_{k+1} = radius^t_k (nothing changes)

             IF ( data%acc .AND. data%rho >= data%eta_successful ) THEN
               inform%obj = nlp%f
               data%barrier = data%barrier_new
               data%mf0 = data%barrier
               data%V( : data%n_total ) = data%V_plus( : data%n_total )
!              data%Cv( : nlp%m ) = data%R( : nlp%m )
!              data%norm_c = data%norm_c_plus
               CALL FUNNEL_slack_reset( nlp%m, data%m_eq, data%m_in, data%C_eq,&
                            data%C_in, data%n_fr, data%C_bound_type, nlp%C,    &
                            nlp%C_l, nlp%C_u, data%V, data%Cv, data%norm_c,    &
                            data%violation_plus,                               &
                            data%use_violation_squared, data%mv0 )
               IF ( data%rho >= data%eta_very_successful ) THEN
                 IF ( data%printw ) WRITE( data%out, "( /, A, ' * very ',      &
                &  'successful V-iteration, rho =', ES12.4 )" ) prefix, data%rho
                 data%success = 'V'
                 IF ( control%algorithm_variant < 3 ) THEN
                   data%radius_n = MAX( data%radius_increase * data%radius_n,  &
                                      data%kappa_deltavv * data%violation_plus )
                 ELSE
                   data%radius_n = data%radius_increase * data%radius_n
                 END IF
               ELSE
                 IF ( data%printw ) WRITE( data%out, "( /, A, ' * ',           &
                &  'successful V-iteration, rho =', ES12.4 )" ) prefix, data%rho
                 data%success = 'S'
                 IF ( control%algorithm_variant < 3 ) THEN
                   data%radius_n = MAX( data%radius_n,                         &
                                      data%kappa_deltavv * data%violation_plus )
                 END IF
               END IF
               data%violation_max                                              &
                 = MAX( data%kappa_t1 * data%violation_max,                    &
                        data%violation_plus + data%kappa_t2 *                  &
                          ( data%violation - data%violation_plus ) )
               data%violation = data%violation_plus
               data%new_gradient = .TRUE.
               data%new_hessian = .TRUE.

!  otherwise, set
!    v_{k+1} = v_k
!    delta^n_{k+1} in [gamma_1 radius^n_k, gamma_2 radius^n_k ]
!    vi^max_{k+1} = vi^max_k and
!    radius^t_{k+1} = radius^t_k (nothing changes)

             ELSE
               IF ( data%printw ) WRITE( data%out, "( /, A, ' * unsuccessful', &
              &  ' V-iteration, rho =', ES12.4 )" ) prefix, data%rho
               data%success = 'U'
               nlp%f = data%f_current
               nlp%C( : nlp%m ) = data%C_current( : nlp%m )
               DO
                 data%radius_n = data%radius_decrease * data%radius_n
                 IF ( data%radius_n < data%norm_n ) EXIT
               END DO
               data%new_gradient = .FALSE.
               data%new_hessian = .FALSE.
             END IF
           END IF
         END IF

!  second-order correction?

         data%s_end = ' '

!   k <- k + 1

         inform%iter = inform%iter + 1
         GO TO 200

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!      E N D   O F   I N N E R   M I N I M I Z A T I O N   I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!500   CONTINUE

!  print a summary of the iteration if required

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       inform%time%total = data%time_now - data%time_start
       inform%time%clock_total = data%clock_now - data%clock_start

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header )         &
           WRITE( data%out, 2020 ) prefix, prefix
         data%print_1st_header = .FALSE.
         WRITE( data%out, 2030 ) prefix,                                       &
            inform%iter, data%it_type, data%success, data%n_end,               &
            data%t_end, data%s_end, inform%obj,                                &
            inform%primal_infeasibility, inform%dual_infeasibility,            &
            data%rho, data%radius_n, data%radius_t, data%violation_max,        &
            inform%time%clock_total
       END IF

!  compute the dual variables

       IF ( data%fixed_variables ) THEN
         DO ii = 1, data%n_fr
           i = data%X_fr( ii )
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             data%Z_l( i ) = data%mu / ( data%V( ii ) - nlp%X_l( i ) )
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             data%Z_u( i ) = data%mu / ( data%V( ii ) - nlp%X_u( i ) )
           END IF
         END DO
       ELSE
         DO i = 1, nlp%n
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             data%Z_l( i ) = data%mu / ( data%V( i ) - nlp%X_l( i ) )
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             data%Z_u( i ) = data%mu / ( data%V( i ) - nlp%X_u( i ) )
           END IF
         END DO
       END IF
!write(6,*) 'D z(1) ', data%Z_l( 1 )
!write(6,*) 'D z(1) ', data%Z_l( 1 ), data%Z_u( 1 )
!write(6,*) ' x - x_l ', data%V(1) - nlp%X_l(1)
!write(6,*) ' x - x_u ', data%V(1) - nlp%X_u(1)

!write(6,*) ' end y_l = mu / c '
       DO ii = 1, data%m_in
         i = data%C_in( ii )
         j =  data%n_fr + ii
         IF ( data%C_bound_type( i ) == lower .OR.                             &
              data%C_bound_type( i ) == both ) THEN
           data%Y_l( i ) = data%mu / ( data%V( j ) - nlp%C_l( i ) )
!write(6,*) ' mu, c ', data%mu, ( data%V( j ) - nlp%C_l( i ) )
!write(6,*) ' y_l ', data%Y_l( i )
         END IF
         IF ( data%C_bound_type( i ) == upper .OR.                             &
              data%C_bound_type( i ) == both ) THEN
           data%Y_u( i ) = data%mu / ( data%V( j ) - nlp%C_u( i ) )
         END IF
       END DO

 510   CONTINUE

!      data%Y_l(2) = SQRT(one/ten) * data%Y_l(2)
!write(6,*) ' y_2 ', data%Y_l(2),  data%mu / ( data%V( data%n_fr+1 ) - nlp%C_l( 2 ) )

!  reduce the barrier parameter

       mu_old = data%mu
!      DO
         data%mu = point1 * data%mu
         IF ( data%mu <= data%mu_smallest ) GO TO 600
!        IF ( data%pi_v > data%mu .OR. data%pi_f > data%mu ) EXIT
!      END DO
       data%print_iteration_header = .TRUE.

!  compute the inner-iteration termination tolerances

       data%stop_p_inner = MAX( data%stop_p, data%mu )
       data%stop_d_inner = MAX( data%stop_d, data%mu )
       data%stop_c_inner = mu_tol * data%mu

       IF ( data%printi ) THEN
         WRITE( data%out, 2050 ) prefix, data%mu, data%stop_p_inner,           &
                                 data%stop_d_inner
         WRITE( data%out, 2020 ) prefix, prefix
       END IF

       data%new_gradient = .TRUE.
       data%new_Hessian = .TRUE.

!  rescale any potentially degenerate dual variables

       d_scale = SQRT( data%mu / mu_old ) ; val = mu_old ** 0.45 ; l = 0
       IF ( data%fixed_variables ) THEN
         DO ii = 1, data%n_fr
           i = data%X_fr( ii )
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             IF ( data%Z_l( i ) <= val .AND.                                   &
                  data%V( ii ) - nlp%X_l( i ) <= val ) THEN
               data%Z_l( i ) = data%Z_l( i ) * d_scale
               l = l + 1
             END IF
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             IF ( - data%Z_u( i ) <= val .AND.                                 &
                  nlp%X_u( i ) - data%V( ii ) <= val ) THEN
               data%Z_u( i ) = data%Z_u( i ) * d_scale
               l = l + 1
             END IF
           END IF
         END DO
       ELSE
         DO i = 1, nlp%n
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             IF ( data%Z_l( i ) <= val .AND.                                   &
                  data%V( i ) - nlp%X_l( i ) <= val ) THEN
               data%Z_l( i ) = data%Z_l( i ) * d_scale
               l = l + 1
             END IF
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             IF ( - data%Z_u( i ) <= val .AND.                                 &
                  nlp%X_u( i ) - data%V( i ) <= val ) THEN
               data%Z_u( i ) = data%Z_u( i ) * d_scale
               l = l + 1
             END IF
           END IF
         END DO
       END IF
!write(6,*) ' rescale z_l ', data%Z_l( 1 )
!write(6,*) ' x - x_l ', data%V(1) - nlp%X_l(1)

!  do the same for Lagrange multipliers

       DO ii = 1, data%m_in
         i = data%C_in( ii )
         j =  data%n_fr + ii
         IF ( data%C_bound_type( i ) == lower .OR.                             &
              data%C_bound_type( i ) == both ) THEN
           IF ( data%Y_l( i ) <= val .AND.                                     &
                data%V( j ) - nlp%C_l( i ) <= val ) THEN
             data%Y_l( i ) = data%Y_l( i ) * d_scale
!write(6,*) ' y_l ', i
             l = l + 1
           END IF
         END IF
         IF ( data%C_bound_type( i ) == upper .OR.                             &
              data%C_bound_type( i ) == both )  THEN
           IF ( - data%Y_u( i ) <= val .AND.                                   &
                nlp%C_u( i ) - data%V( j ) <= val ) THEN
             data%Y_u( i ) = data%Y_u( i ) * d_scale
!write(6,*) ' y_u ', i
             l = l + 1
           END IF
         END IF
       END DO
       IF ( l > 0 .AND. data%printt ) WRITE( data%out, "( A, I0,               &
      &   ' potentially degenerate constraints recorded ' )" ) prefix, l

!  ========================================================================
!                    E X T R A P O L A T I O N
!  ========================================================================

!  extrapolate to find a new starting point - use the perturbed primal-dual
!  system

!        ( g(x) - J^T(x) y - z_l - z_u )
!        (       y - y_l - y_u         )
!        (           c(x) - c          )
!  rhs = (   Y_l ( c - c_l ) - mu e    ) = 0
!        (   Y_u ( c - c_u ) - mu e    )
!        (   Z_l ( x - x_l ) - mu e    )
!        (   Z_u ( x - x_u ) - mu e    )

!  particularly use (x^+,c^+,y^+,y_l^+,y_u^+,z_l^+,z_u^+) =
!    (x,c,y,y_l,y_u,z_l,z_u) + (dx,dc,dy,dy_l,dy_u,dz_l,dz_u), where

!   ( H(x,y)     -J^T(x)                 -I     -I   ) ( dx )
!   (               I     -I      -I                 ) ( dc )
!   (  J(x)   -I                                     ) ( dy )
!   (        Y_l          C-C_l                      ) (dy_l) = - rhs
!   (        Y_u                 C-C_u               ) (dy_u)
!   (  Z_l                              X-X_l        ) (dz_l)
!   (  Z_u                                     X-X_u ) (dz_u)

!  or (more succinctly)

!  ( H(x,y) + (X-X_l)^-1 Z_l                    J^T(x) ) (   dx  )
!  (        + (X-X_u)^-1 Z_u                           ) (       )
!  (                          (C-C_l)^-1 Y_l +   -I    ) (   dc  )
!  (                          (C-C_u)^-1 Y_u           ) (       )
!  (       J(x)                     -I                 ) ( -y-dy )

!        ( g(x) - mu (X-X_l)^-1 e - mu (X-X_u)^-1 e )
!    = - (      - mu (C-C_l)^-1 e - mu (C-C_u)^-1 e )                        (E)
!        (                 c(x) - c                 )

!  followed by y_l + dy_l = [C-C_l]^-1 [ mu e - Y_l dc ]
!              y_u + dy_u = [C-C_u]^-1 [ mu e - Y_u dc ]
!              z_l + dz_l = [X-X_l]^-1 [ mu e - Z_l dx ]
!              z_u + dz_u = [X-X_u]^-1 [ mu e - Z_u dx ]

!  ========================================================================

       IF ( .NOT. control%allow_extrapolatiion ) GO TO 590

!  compute the new barrier function

       IF ( data%fixed_variables ) THEN
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,            &
                    n_fr = data%n_fr, X_fr = data%X_fr )
       ELSE
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
       END IF

!   compute P_k = P(v_k) ...

       data%P( :  data%n_fr ) = one
!      IF ( data%fixed_variables ) THEN
!        DO ii = 1, data%n_fr
!          i = data%X_fr( ii )
!          SELECT CASE ( data%X_bound_type( i ) )
!          CASE ( fixed )
!            data%P( ii ) = one
!          CASE ( lower )
!            data%P( ii ) = data%V( ii ) - nlp%X_l( i )
!          CASE ( upper )
!            data%P( ii ) = nlp%X_u( i ) - data%V( ii )
!          CASE ( both )
!            data%P( ii ) = MIN( data%V( ii ) - nlp%X_l( i ),                  &
!                               nlp%X_u( i ) - data%V( ii ) )
!          CASE ( free )
!            data%P( ii ) = one
!          END SELECT
!        END DO
!      ELSE
!        DO i = 1, nlp%n
!          SELECT CASE ( data%X_bound_type( i ) )
!          CASE ( fixed )
!            data%P( i ) = one
!          CASE ( lower )
!            data%P( i ) = data%V( i ) - nlp%X_l( i )
!          CASE ( upper )
!            data%P( i ) = nlp%X_u( i ) - data%V( i )
!          CASE ( both )
!            data%P( i ) = MIN( data%V( i ) - nlp%X_l( i ),                    &
!                               nlp%X_u( i ) - data%V( i ) )
!          CASE ( free )
!            data%P( i ) = one
!          END SELECT
!        END DO
!      END IF

       DO ii = 1, data%m_in
         i = data%C_in( ii )
         j = data%n_fr + ii
         SELECT CASE ( data%C_bound_type( i ) )
         CASE ( equality )
           data%P( j ) = one
         CASE ( lower )
           data%P( j ) = data%V( j ) - nlp%C_l( i )
         CASE ( upper )
           data%P( j ) = nlp%C_u( i ) -  data%V( j )
         CASE ( both )
           data%P( j ) = MIN( data%V( j ) - nlp%C_l( i ),                      &
                              nlp%C_u( i ) -  data%V( j ) )
         CASE ( free )
           data%P( j ) = one
         END SELECT
       END DO

!  and P_k^{-2}

       data%Pinv2%val( : data%n_total ) = one / data%P( : data%n_total ) ** 2

!  -------------------  COMPUTE FIRST DERIVATIVE VALUES -----------------------

!  evaluate the Jacobian of the general constraint functions, stored in
!  "co-ordinate" format, and the gradient of the objective

       IF ( data%new_gradient ) THEN
         IF ( data%reverse_gj ) THEN
           data%branch = 7 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_GJ( data%eval_status, nlp%X, userdata, nlp%G, nlp%J%val )
         END IF
       END IF

!  return from reverse communication with the gradient and Jacobian

  570  CONTINUE
       IF ( data%new_gradient ) THEN
         inform%g_eval = inform%g_eval + 1

!  compute J(v_k), c.f. the l"off-diagonal" block of the matrix in (E)

         IF ( data%fixed_variables ) THEN
           ne = 0
           DO l = 1, nlp%J%ne
             IF ( data%X_order( nlp%J%col( l ) ) > 0 ) THEN
               ne = ne + 1
               data%Jv%val( ne ) = nlp%J%val( l )
             END IF
           END DO
         ELSE
           data%Jv%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne )
         END IF

!  compute primal dual variables (y^L,y^U,z^L,z^U) and
!   grad f(v;mu) := grad f(x)
!    - mu sum_{i=1}^N e_i / (x-x^L) - mu sum_{i=1}^N e_i / (x-x^U)
!    - mu sum_{i=1}^M e_N+i / (c-c^L) - mu sum_{i=1}^M e_N+i / (c-c^U)

        IF ( data%fixed_variables ) THEN
           data%GRAD_barrier( : data%n_fr ) = nlp%G( data%X_fr( : data%n_fr ) )
         ELSE
           data%GRAD_barrier( : nlp%n ) = nlp%G( : nlp%n )
         END IF
         data%GRAD_barrier( data%n_fr + 1 : data%n_total ) = zero

         IF ( data%fixed_variables ) THEN
           DO ii = 1, data%n_fr
             i = data%X_fr( ii )
             IF ( data%X_bound_type( i ) == lower .OR.                         &
                  data%X_bound_type( i ) == both ) THEN
!              data%Z_l( i ) = data%mu / ( data%V( ii ) - nlp%X_l( i ) )
!              data%GRAD_barrier( ii ) = data%GRAD_barrier( ii ) - data%Z_l( i )
               data%GRAD_barrier( ii ) = data%GRAD_barrier( ii )               &
                 - data%mu / ( data%V( ii ) - nlp%X_l( i ) )
             END IF
             IF ( data%X_bound_type( i ) == upper .OR.                         &
                  data%X_bound_type( i ) == both )  THEN
!              data%Z_u( i ) = data%mu / ( data%V( ii ) - nlp%X_u( i ) )
!              data%GRAD_barrier( ii ) = data%GRAD_barrier( ii ) - data%Z_u( i )
               data%GRAD_barrier( ii ) = data%GRAD_barrier( ii )               &
                  - data%mu / ( data%V( ii ) - nlp%X_u( i ) )
             END IF
           END DO
         ELSE
           DO i = 1, nlp%n
             IF ( data%X_bound_type( i ) == lower .OR.                         &
                  data%X_bound_type( i ) == both ) THEN
!              data%Z_l( i ) = data%mu / ( data%V( i ) - nlp%X_l( i ) )
!              data%GRAD_barrier( i ) = data%GRAD_barrier( i ) - data%Z_l( i )
               data%GRAD_barrier( i ) = data%GRAD_barrier( i )                 &
                 - data%mu / ( data%V( i ) - nlp%X_l( i ) )
             END IF
             IF ( data%X_bound_type( i ) == upper .OR.                         &
                  data%X_bound_type( i ) == both )  THEN
!              data%Z_u( i ) = data%mu / ( data%V( i ) - nlp%X_u( i ) )
!              data%GRAD_barrier( i ) = data%GRAD_barrier( i ) - data%Z_u( i )
               data%GRAD_barrier( i ) = data%GRAD_barrier( i )                 &
                  - data%mu / ( data%V( i ) - nlp%X_u( i ) )
             END IF
           END DO
         END IF

!write(6,*) 'F z(1) ', data%Z_l( 1 ), data%V( 1 ) - nlp%X_l( 1 )

!       data%Z_l(1) = SQRT(one/ten) * data%Z_l(1)

!write(6,*) 'B z(1) ', data%Z_l( 1 )
!write(6,*) 'B z(1) ', data%Z_l( 1 ), data%Z_u( 1 )
!write(6,*) ' x - x_l ', data%V(1) - nlp%X_l(1)
!write(6,*) ' x - x_u ', data%V(1) - nlp%X_u(1)

         DO ii = 1, data%m_in
           i = data%C_in( ii )
           j =  data%n_fr + ii
           IF ( data%C_bound_type( i ) == lower .OR.                           &
                data%C_bound_type( i ) == both ) THEN
!            data%Y_l( i ) = data%mu / ( data%V( j ) - nlp%C_l( i ) )
!            data%GRAD_barrier( j ) = data%GRAD_barrier( j ) - data%Y_l( i )
             data%GRAD_barrier( j ) = data%GRAD_barrier( j )                   &
                                      - data%mu / ( data%V( j ) - nlp%C_l( i ) )
           END IF
           IF ( data%C_bound_type( i ) == upper .OR.                           &
                data%C_bound_type( i ) == both ) THEN
!            data%Y_u( i ) = data%mu / ( data%V( j ) - nlp%C_u( i ) )
!            data%GRAD_barrier( j ) = data%GRAD_barrier( j ) - data%Y_u( i )
             data%GRAD_barrier( j ) = data%GRAD_barrier( j )                   &
                                      - data%mu / ( data%V( j ) - nlp%C_u( i ) )
           END IF
         END DO
       END IF

!  -------------------  COMPUTE SECOND DERIVATIVE VALUES ----------------------

       IF ( data%new_hessian ) THEN

!  select a vector yhat_k satisfying ||yhat_k||_2 <= kappa_y (store in nlp%y)

!  recompute the Hessian of the Lagrangian at (x_k,yhat_k)

         IF ( data%reverse_hl ) THEN
           data%branch = 8 ; inform%status = 4 ; RETURN
         ELSE
           CALL eval_HL( data%eval_status, nlp%X, nlp%Y, userdata, nlp%H%val )
         END IF
       END IF

!  return from reverse communication with the Hessian of the Lagrangian

  580  CONTINUE

!  compute the barrier Hessian ( Hess Lagrangian(x,y)  0 ) +
!                              (            0          0 )
!      ( Z^L (X - X^L)^{-1} + Z^U (X - X^U)^{-1}           0          )
!      (          0           Y^L (C - C^L)^{-1} + Y^U (C - C^U)^{-1} )
!  c.f. the leading block of the matrix in (E)

       IF ( data%new_hessian ) THEN
         IF ( data%fixed_variables ) THEN
           ne = 0
           DO l = 1, nlp%H%ne
             i = data%X_order( nlp%H%row( l ) )
             j = data%X_order( nlp%H%col( l ) )
             IF ( i > 0 .AND. j > 0 ) THEN
               ne = ne + 1
               data%HESS_barrier%val( ne ) = nlp%H%val( l )
             END IF
           END DO
         ELSE
           data%HESS_barrier%val( : nlp%H%ne ) = nlp%H%val( : nlp%H%ne )
           ne = nlp%H%ne
         END IF

         IF ( data%fixed_variables ) THEN
           DO ii = 1, data%n_fr
             i = data%X_fr( ii )
             SELECT CASE ( data%X_bound_type( i ) )
             CASE ( lower )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_l( i ) / ( data%V( ii ) - nlp%X_l( i ) )
             CASE ( upper )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_u( i ) / ( data%V( ii ) - nlp%X_u( i ) )
             CASE ( both )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_l( i ) / ( data%V( ii ) - nlp%X_l( i ) ) +             &
                 data%Z_u( i ) / ( data%V( ii ) - nlp%X_u( i ) )
             END SELECT
           END DO
         ELSE
           DO i = 1, nlp%n
             SELECT CASE ( data%X_bound_type( i ) )
             CASE ( lower )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_l( i ) / ( data%V( i ) - nlp%X_l( i ) )
             CASE ( upper )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_u( i ) / ( data%V( i ) - nlp%X_u( i ) )
             CASE ( both )
               ne = ne + 1
               data%HESS_barrier%val( ne ) =                                   &
                 data%Z_l( i ) / ( data%V( i ) - nlp%X_l( i ) ) +              &
                 data%Z_u( i ) / ( data%V( i ) - nlp%X_u( i ) )
             END SELECT
           END DO
         END IF

         DO ii = 1, data%m_in
           i = data%C_in( ii )
           j = data%n_fr + ii
           SELECT CASE ( data%C_bound_type( i ) )
           CASE ( lower )
             ne = ne + 1
             data%HESS_barrier%val( ne ) =                                     &
               data%Y_l( i ) / ( data%V( j ) - nlp%C_l( i ) )
           CASE ( upper )
             ne = ne + 1
             data%HESS_barrier%val( ne ) =                                     &
               data%Y_u( i ) / ( data%V( j ) - nlp%C_u( i ) )
           CASE ( both )
             ne = ne + 1
             data%HESS_barrier%val( ne ) =                                     &
               data%Y_l( i ) / ( data%V( j ) - nlp%C_l( i ) ) +                &
               data%Y_u( i ) / ( data%V( j ) - nlp%C_u( i ) )
           END SELECT
         END DO
       END IF

!  set up the right-hand side of (E)

       data%T( : data%n_total ) = zero
       data%EQP_control%radius = data%omkappa_fbn
       data%EQP_control%preconditioner = 5

!  find the extrapolation step by solving (E)

       IF ( data%printw ) WRITE( data%out,                                     &
         "( /, A, ' * entering EQP for extrapolation step: radius = ',         &
        & ES9.2 )" ) prefix, data%EQP_control%radius
       CALL EQP_solve_main( data%n_total, nlp%m, data%HESS_barrier,            &
                            data%GRAD_barrier, data%barrier, data%Jv,          &
                            data%mfd, data%T, nlp%Y, data%EQP_data,            &
                            data%EQP_control, inform%EQP_inform,               &
                            C = data%CV( : nlp%m ), D = data%Pinv2%val )
       IF ( data%printw ) WRITE( data%out,                                     &
         "( A, ' * returned from EQP (extrapolation step)' )" ) prefix

!  update y

!write(6,*) ' nlp y = from EQP problem'
!      nlp%Y( : nlp%m ) = - nlp%Y( : nlp%m )
!if(nlp%m>0) write(6,*) ' y ', nlp%Y( 1 )

!  update z_l <- [X-X_l]^-1 [ mu e - Z_l dx ]
!         z_u <- [X-X_u]^-1 [ mu e - Z_u dx ]
!         y_l <- [C-C_l]^-1 [ mu e - Y_l dc ]
!         y_u <- [C-C_u]^-1 [ mu e - Y_u dc ]

       IF ( data%fixed_variables ) THEN
         DO ii = 1, data%n_fr
           i = data%X_fr( ii )
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             data%Z_l( i ) = ( data%mu - data%Z_l( i ) * data%T( ii ) )        &
                               / ( data%V( ii ) - nlp%X_l( i ) )
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             data%Z_u( i ) = ( data%mu - data%Z_u( i ) * data%T( ii ) )        &
                               / ( data%V( ii ) - nlp%X_u( i ) )
           END IF
         END DO
       ELSE
         DO i = 1, nlp%n
           IF ( data%X_bound_type( i ) == lower .OR.                           &
                data%X_bound_type( i ) == both ) THEN
             data%Z_l( i ) = ( data%mu - data%Z_l( i ) * data%T( i ) )         &
                               / ( data%V( i ) - nlp%X_l( i ) )
           END IF
           IF ( data%X_bound_type( i ) == upper .OR.                           &
                data%X_bound_type( i ) == both )  THEN
             data%Z_u( i ) = ( data%mu - data%Z_u( i ) * data%T( i ) )         &
                               / ( data%V( i ) - nlp%X_u( i ) )
           END IF
         END DO
       END IF
!write(6,*) 'C z(1) ', data%Z_l( 1 )
!write(6,*) 'C z(1) ', data%Z_l( 1 ), data%Z_u( 1 )
!write(6,*) ' x - x_u ', data%V(1) - nlp%X_u(1)

!write(6,*) ' y_l = extrapolated '
       DO ii = 1, data%m_in
         i = data%C_in( ii )
         j =  data%n_fr + ii
         IF ( data%C_bound_type( i ) == lower .OR.                             &
              data%C_bound_type( i ) == both ) THEN
           data%Y_l( i ) = ( data%mu - data%Y_l( i ) * data%T( j ) )           &
                             / ( data%V( j ) - nlp%C_l( i ) )
         END IF
         IF ( data%C_bound_type( i ) == upper .OR.                             &
              data%C_bound_type( i ) == both ) THEN
           data%Y_u( i ) = ( data%mu - data%Y_u( i ) * data%T( j ) )           &
                             / ( data%V( j ) - nlp%C_u( i ) )
         END IF
       END DO
!write(6,*) ' E y_2 ', data%Y_l(2)

!  find the largest alpha for which
!      v^L + kappa_fbt (v_k - v^L) <= v_k + alpha t_k
!        <= v^U + kappa_fbt (v_k - v^U) is satisfied for alpha t_k          (E2)

       IF ( data%fixed_variables ) THEN
         alpha_b = STEP_fraction_to_boundary(                                  &
           nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%T,              &
           data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,              &
           data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,                     &
           n_fr = data%n_fr, X_fr = data%X_fr )
       ELSE
         alpha_b = STEP_fraction_to_boundary(                                  &
           nlp%n, nlp%m, data%m_in, data%n_total, data%V, data%T,              &
           data%omkappa_fbt, data%X_bound_type, nlp%X_l, nlp%X_u,              &
           data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
       END IF

!write(6,*) ' alpha_b ', alpha_b

!  step back along the normal step if necessary to satisfy (E2)

       alpha = MIN( one, alpha_b )
       data%T( : data%n_total ) = alpha * data%T( : data%n_total )

!  set the new iterate v_k^+ = v_k + d_k

       data%V( : data%n_total )                                                &
         = data%V( : data%n_total ) + data%T( : data%n_total )

!write(6,*) ' new z_l ', data%Z_l( 1 )
!write(6,*) ' new x - x_l ', data%V(1) - nlp%X_l(1)

!  ------------------------  COMPUTE FUNCTION VALUES --------------------------

       IF ( data%fixed_variables ) THEN
         nlp%X( data%X_fr( : data%n_fr ) ) = data%V( : data%n_fr )
       ELSE
         nlp%X = data%V( : nlp%n )
       END IF

!write(6,*) ' comp = ', &
!           ( nlp%X_l( 1 ) - nlp%X( 1 ) ) * data%Z_l( 1 ),                     &
!           ( nlp%X_u( 1 ) - nlp%X( 1 ) ) * data%Z_u( 1 )

       IF ( data%reverse_fc ) THEN
         data%branch = 9 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
       END IF

!  return from reverse communication with the function and constraint values

 590   CONTINUE

!  compute the new barrier function

       inform%f_eval = inform%f_eval + 1
       inform%obj = nlp%f
       IF ( data%fixed_variables ) THEN
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u,            &
                    n_fr = data%n_fr, X_fr = data%X_fr )
       ELSE
         data%barrier =                                                        &
           BARRIER( nlp%n, nlp%m, data%m_in, data%n_total, nlp%f, data%V,      &
                    data%mu, data%X_bound_type, nlp%X_l, nlp%X_u,              &
                    data%C_bound_type, data%C_in, nlp%C_l, nlp%C_u )
       END IF

!  reset the trust-region radii to their initial values

       data%radius_n = control%initial_n_model_radius
       data%radius_t = control%initial_t_model_radius

!   compute z_k

       DO i = 1, nlp%n
         SELECT CASE ( data%X_bound_type( i ) )
         CASE ( lower )
           nlp%Z( i ) = data%Z_l( i )
         CASE ( upper )
           nlp%Z( i ) = data%Z_u( i )
         CASE ( both )
           nlp%Z( i ) = data%Z_l( i ) + data%Z_u( i )
         CASE ( free )
           nlp%Z( i ) = zero
         END SELECT
       END DO

       inform%iter = inform%iter + 1
       data%it_type = 'E'
       data%mf0 = data%barrier
       data%start_inner = .TRUE.
!      GO TO 100
       GO TO 200

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                   E N D    O F    B A R R I E R  I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  -------------
!  Normal return
!  -------------

 600 CONTINUE

!  print a summary of the iteration if required

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start

     IF ( data%printi ) THEN
       IF ( data%print_iteration_header .OR. data%print_1st_header )           &
         WRITE( data%out, 2020 ) prefix, prefix
       data%print_1st_header = .FALSE.
       WRITE( data%out, 2030 ) prefix,                                         &
          inform%iter, data%it_type, data%success, data%n_end,                 &
          data%t_end, data%s_end, inform%obj,                                  &
          inform%primal_infeasibility, inform%dual_infeasibility,              &
          data%rho, data%radius_n, data%radius_t, data%violation_max,          &
          inform%time%clock_total
     END IF

 610 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start

!  compute the dual variables

!    DO i = 1, nlp%n
!      SELECT CASE ( data%X_bound_type( i ) )
!      CASE ( fixed )
!        nlp%Z( i ) = data%Z_l( i )
!      CASE ( lower )
!        nlp%Z( i ) = data%Z_l( i )
!      CASE ( upper )
!        nlp%Z( i ) = data%Z_u( i )
!      CASE ( both )
!        nlp%Z( i ) = data%Z_l( i ) + data%Z_u( i )
!      CASE ( free )
!        nlp%Z( i ) = zero
!      END SELECT
!    END DO

!  compute the primal infeasibility and the complementary slackness

     inform%primal_infeasibility = zero ; inform%complementary_slackness = zero
     DO i = 1, nlp%n
       SELECT CASE ( data%X_bound_type( i ) )
       CASE ( lower )
         inform%primal_infeasibility =                                         &
           MAX( inform%primal_infeasibility, nlp%X_l( i ) - nlp%X( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           ABS( ( nlp%X_l( i ) - nlp%X( i ) ) * nlp%Z( i ) ) )
       CASE ( upper )
         inform%primal_infeasibility =                                         &
           MAX( inform%primal_infeasibility, nlp%X( i ) - nlp%X_u( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           ABS( ( nlp%X_u( i ) - nlp%X( i ) ) * nlp%Z( i ) ) )
       CASE ( both )
         inform%primal_infeasibility = MAX( inform%primal_infeasibility,       &
           nlp%X_l( i ) - nlp%X( i ), nlp%X( i ) - nlp%X_u( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           MIN( ABS( ( nlp%X_l( i ) - nlp%X( i ) ) * nlp%Z( i ) ),             &
                ABS( ( nlp%X_u( i ) - nlp%X( i ) ) * nlp%Z( i ) ) ) )
       END SELECT
     END DO
     DO i = 1, nlp%m
       SELECT CASE ( data%C_bound_type( i ) )
       CASE ( lower )
         inform%primal_infeasibility =                                         &
           MAX( inform%primal_infeasibility, nlp%C_l( i ) - nlp%C( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           ABS( ( nlp%C_l( i ) - nlp%C( i ) ) * nlp%Y( i ) ) )
       CASE ( upper )
         inform%primal_infeasibility =                                         &
           MAX( inform%primal_infeasibility, nlp%C( i ) - nlp%C_u( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           ABS( ( nlp%C_u( i ) - nlp%C( i ) ) * nlp%Y( i ) ) )
       CASE ( both, equality )
         inform%primal_infeasibility = MAX( inform%primal_infeasibility,       &
           nlp%C_l( i ) - nlp%C( i ), nlp%C( i ) - nlp%C_u( i ) )
         inform%complementary_slackness = MAX( inform%complementary_slackness, &
           MIN( ABS( ( nlp%C_l( i ) - nlp%C( i ) ) * nlp%Y( i ) ),             &
                ABS( ( nlp%C_u( i ) - nlp%C( i ) ) * nlp%Y( i ) ) ) )
       END SELECT
     END DO
!    IF ( data%barrier_terms > zero ) inform%complementary_slackness           &
!          = inform%complementary_slackness / data%barrier_terms

!  compute the dual infeasibility

     data%W( : nlp%n ) = nlp%G( : nlp%n )
     DO l = 1, nlp%J%ne
       j = nlp%J%col( l )
       data%W( j ) = data%W( j ) - nlp%J%val( l ) * nlp%Y( nlp%J%row( l ) )
     END DO
     IF ( data%fixed_variables ) THEN
       DO i = 1, nlp%n
         IF ( data%X_order( i ) > 0 ) THEN
           data%W( i ) = data%W( i ) - nlp%Z( i )
         ELSE
           nlp%Z( i ) = data%W( i )
           data%W( i ) = zero
         END IF
       END DO
     ELSE
       data%W( : nlp%n ) = data%W( : nlp%n ) - nlp%Z( : nlp%n )
     END IF
     inform%dual_infeasibility = MAXVAL( ABS( data%W( : nlp%n ) ) ) / data%max_y

!  print the solution

     IF ( control%print_level > 0 ) THEN
       l = 2
       IF ( control%fulsol ) l = nlp%n
       IF ( control%print_level >= 10 ) l = nlp%n

       stopr = control%stop_abs_d
       WRITE( data%out, "( /, A, ' Solution: ', /, A, '                    ',  &
      &         '              <------ Bounds ------> ', /, A,                 &
      &         '      # name      state     value   ',                        &
      &         '    Lower       Upper       Dual ' )" ) prefix, prefix, prefix
       DO j = 1, 2
         IF ( j == 1 ) THEN
           ir = 1 ; ic = MIN( l, nlp%n )
         ELSE
           IF ( ic < nlp%n - l ) WRITE( data%out, 2010 ) prefix
           ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
         END IF
         DO i = ir, ic
          state = ' FREE'
          IF ( ABS( nlp%X( i )   - nlp%X_l( i ) ) < ten * stopr )              &
            state = 'LOWER'
          IF ( ABS( nlp%X( i )   - nlp%X_u( i ) ) < ten * stopr )              &
            state = 'UPPER'
          IF ( ABS( nlp%X_l( I ) - nlp%X_u( I ) ) < stopr )                    &
            state = 'FIXED'
           WRITE( data%out, 2000 ) prefix, i, nlp%VNAMES( i ), state,          &
             nlp%X( i ), nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
         END DO
       END DO

!  print the constraint details

       IF ( nlp%m > 0 ) THEN
         l = 2
         IF ( control%fulsol ) l = nlp%m
         IF ( control%print_level >= 10 ) l = nlp%m

         WRITE( data%out, "( /, A, ' Constraints: ', /, A, '               ',  &
       &    '                       <------ Bounds ------> ', /, A,            &
       &    '      # name      state     value   ',                            &
       &    '    Lower       Upper     Multiplier ' )" ) prefix, prefix, prefix
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1 ; ic = MIN( l, nlp%m )
           ELSE
             IF ( ic < nlp%m - l ) WRITE( data%out, 2010 ) prefix
             ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
           END IF
           DO i = ir, ic
            state = ' FREE'
            IF ( ABS( nlp%C( I ) - nlp%C_l( i ) ) < ten * stopr )              &
              state = 'LOWER'
            IF ( ABS( nlp%C( I ) - nlp%C_u( i ) ) < ten * stopr )              &
              state = 'UPPER'
            IF ( ABS( nlp%C_l( i ) - nlp%C_u( i ) ) < stopr )                  &
              state = 'EQUAL'
             WRITE( data%out, 2000 ) prefix, i, nlp%CNAMES( i ), state,        &
               nlp%C( i ), nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
           END DO
         END DO
       END IF

!  print algorithmic details

       IF ( control%direct_solution_of_normal_model ) THEN
         WRITE( data%out,                                                      &
         "( /, A, ' Direct solution (solver ', A,                              &
        &      ') of the normal trust-region sub-problem' )" )                 &
            prefix, TRIM( data%TRS_control%symmetric_linear_solver )
         WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
        &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
           inform%factorizations_normal, inform%TRS_inform%time%clock_factorize
         IF ( TRIM( control%TRS_control%symmetric_linear_solver )              &
              == 'pbtr' ) THEN
           WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
          & ', semi-bandwidth = ', I0  )" ) prefix,                            &
             inform%max_entries_factors_normal,                                &
             inform%TRS_inform%SLS_inform%semi_bandwidth
         ELSE
           WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
             prefix, inform%max_entries_factors_normal
         END IF
       ELSE
         WRITE( data%out, "( /, A, ' Iterative solution (solver ', A,          &
          &    ', preconditioner = ', I0,') of the', /, A,                     &
          &    '  normal trust-region sub-problem' )" )                        &
              prefix, TRIM( control%SBLS_control%symmetric_linear_solver ),    &
              inform%LLS_inform%SBLS_inform%preconditioner, prefix
         WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )         &
            prefix, inform%max_entries_factors_normal
       END IF

       IF ( control%direct_solution_of_tangential_model ) THEN
         IF ( inform%TRS_inform%dense_factorization ) THEN
           WRITE( data%out,                                                    &
           "( /, A, ' Direct solution (eigen solver SYSV',                     &
          &      ') of the tangential trust-region sub-problem' )" ) prefix
         ELSE
           WRITE( data%out,                                                    &
           "( /, A, ' Direct solution (solver ', A,                            &
          &      ') of the tangential trust-region sub-problem' )" )           &
              prefix, TRIM( data%TRS_control%symmetric_linear_solver )
         END IF
         WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
        &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
           inform%factorizations_tangential,                                   &
           inform%TRS_inform%time%clock_factorize
         IF ( TRIM( control%TRS_control%symmetric_linear_solver )              &
              == 'pbtr' ) THEN
           WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
          & ', semi-bandwidth = ', I0  )" ) prefix,                            &
             inform%max_entries_factors_tangential,                            &
             inform%TRS_inform%SLS_inform%semi_bandwidth
         ELSE
           WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
             prefix, inform%max_entries_factors_tangential
         END IF
       ELSE
         WRITE( data%out, "( /, A, ' Iterative solution (solver ', A,          &
          &    ', preconditioner = ', I0,') of the', /, A,                     &
          &    '  tangential trust-region sub-problem' )" )                    &
              prefix, TRIM( control%SBLS_control%symmetric_linear_solver ),    &
              inform%EQP_inform%SBLS_inform%preconditioner, prefix
         WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )         &
            prefix, inform%max_entries_factors_tangential
       END IF
       WRITE( data%out, "( A, ' Number of threads = ', I0 )" )                 &
         prefix, inform%threads

       IF ( nlp%m > 0 ) THEN ; space = MAXVAL( ABS( nlp%Y ) ) ;
         ELSE ; space = zero ; END IF

       WRITE( data%out, "( /, A, ' Problem: ', 16X, A10,                       &
     &          '  Solver: ', 10X, '  Funnel [', I0, ']' /, A,                 &
     &          ' n              =     ',bn, I12,                              &
     &          '    m               = ', bn, I12, /, A,                       &
     &          ' Objective      = ', ES16.8,                                  &
     &          '    Violation       = ', ES12.4, /, A,                        &
     &          ' Dual infeas    =     ', ES12.4,                              &
     &          '    Complementarity = ', ES12.4, /, A,                        &
     &          ' Max multiplier =     ', ES12.4,                              &
     &          '    Max dual var.   = ', ES12.4, /, A,                        &
     &          ' Iterations     =     ', bn, I12,                             &
     &          '    Function evals  = ', bn, I12, /, A,                       &
     &          ' Gradient evals =     ',bn, I12,                              &
     &          '    Time            = ', F12.2 )" )                           &
         prefix, nlp%pname, control%algorithm_variant, prefix,                 &
         nlp%n, nlp%m, prefix, inform%obj, inform%primal_infeasibility,        &
         prefix, inform%dual_infeasibility, inform%complementary_slackness,    &
         prefix, space, MAXVAL( ABS( nlp%Z ) ), prefix, inform%iter,           &
         inform%f_eval, prefix, inform%g_eval, inform%time%clock_total
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

 910 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start
     IF ( control%print_level > 0 .AND. control%error > 0 )                    &
       CALL SYMBOLS_status( inform%status, control%error, prefix, 'FUNNEL' )

     RETURN

!  non-executable statements

 2000 FORMAT( A, I7, 1X, A10, A6, 4ES12.4 )
 2010 FORMAT( A, 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2020 FORMAT( A, '  it_type=F-V-Y-E  success=V-S-U-_  n=n-c-0  t=t-c-0  s=_',  &
              /, A, '  iter   nts      f       primal ',                       &
              '  dual     rho    rad_n   rad_t   v_max    time' )
 2030 FORMAT( A, I6, 2A1, 1X, 3A1, ES12.4, 2ES8.1, ES9.1, 3ES8.1, F7.1 )
 2040 FORMAT( A, I6, A1, 5X, ES12.4, 2ES8.1, '     -   ', 3ES8.1, F7.1 )
 2050 FORMAT( /, A, '  <==== mu = ', ES10.4, ' ---- stop_p = ', ES10.4,        &
              ' ---- stop_d = ', ES10.4, '  ====>' )

!  internal procedures

     CONTAINS

! -*-*-*-*-*-*-*- F O R C I N G   I N T E R N A L   F U N C T I O N -*-*-*-*-*-

       FUNCTION FORCING( char, argument )

!  Evaluates the forcing function min( |argument|, argument^2 )

       REAL ( KIND = wp ) :: FORCING

!  Dummy arguments

       CHARACTER ( LEN = 1 ), INTENT( IN ) :: char
       REAL ( KIND = wp ), INTENT( IN ) :: argument

       SELECT CASE ( char )
       CASE ( 'n' )
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       CASE ( 'y' )
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       CASE DEFAULT
!        FORCING = point01 * MIN( point01, MIN( ABS( argument ), argument ** 2))
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       END SELECT

!  End of function FORCING

       END FUNCTION FORCING

! -*-*-*-*-*-*-*- B A R R I E R   I N T E R N A L   F U N C T I O N -*-*-*-*-*-

       FUNCTION BARRIER( n, m, m_in, n_total, f, V, mu, X_bound_type, X_l,     &
                         X_u, C_bound_type, C_in, C_l, C_u, n_fr, X_fr )

! evaluate the barrier function
!  f(v;mu) := f(x) - mu sum_{i=1}^M log (c-c^L)(c^U-c) where v := ( x )
!                  - mu sum_{i=1}^N log (x-x^L)(x^U-x)            ( c )

       REAL ( KIND = wp ) :: BARRIER

!  Dummy arguments

       INTEGER, INTENT( IN ) :: n, m, m_in, n_total
       REAL ( KIND = wp ), INTENT( IN ) :: f, mu
       INTEGER, INTENT( IN ), DIMENSION( n ) :: X_bound_type
       INTEGER, INTENT( IN ), DIMENSION( m ) :: C_bound_type
       INTEGER, INTENT( IN ), DIMENSION( m_in ) :: C_in
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n_total ) :: V
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
       INTEGER, INTENT( IN ), OPTIONAL :: n_fr
       INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( : ) :: X_fr

!  Local variable

       INTEGER :: i, ii

       BARRIER = f

       IF ( PRESENT( n_fr ) .AND. PRESENT( X_fr ) ) THEN
         DO ii = 1, n_fr
           i = X_fr( ii )
           IF ( X_bound_type( i ) == lower .OR. X_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( V( ii ) - X_l( i ) )
           IF ( X_bound_type( i ) == upper .OR. X_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( X_u( i ) - V( ii ) )
         END DO
         DO ii = 1, m_in
           i = C_in( ii )
           IF ( C_bound_type( i ) == lower .OR. C_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( V( n_fr + ii ) - C_l( i ) )
           IF ( C_bound_type( i ) == upper .OR. C_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( C_u( i ) - V( n_fr + ii ) )
         END DO
       ELSE
         DO i = 1, n
           IF ( X_bound_type( i ) == lower .OR. X_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( V( i ) - X_l( i ) )
           IF ( X_bound_type( i ) == upper .OR. X_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( X_u( i ) - V( i ) )
         END DO
         DO ii = 1, m_in
           i = C_in( ii )
           IF ( C_bound_type( i ) == lower .OR. C_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( V( n + ii ) - C_l( i ) )
           IF ( C_bound_type( i ) == upper .OR. C_bound_type( i ) == both )    &
             BARRIER = BARRIER - mu * LOG( C_u( i ) - V( n + ii ) )
         END DO
       END IF

!  End of function BARRIER

       END FUNCTION BARRIER

! -*-*-*-*-*-*-*- M O D E L _ V   I N T E R N A L   F U N C T I O N -*-*-*-*-*-

       FUNCTION MODEL_v( m, n_total, Cv, Jv, D, R, use_violation_squared )

! evaluate the v-model function
!  m^v(d) := 1/2 ||c(v) + J(v) d||_2^2, where v := ( x )
!                                                  ( c )

       REAL ( KIND = wp ) :: MODEL_v

!  Dummy arguments

       INTEGER, INTENT( IN ) :: m, n_total
       LOGICAL, INTENT( IN ) :: use_violation_squared
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Cv
       REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: R
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n_total ) :: D
       TYPE ( SMT_type ), INTENT( IN ) :: Jv

!  Local variable

       INTEGER :: i, l
!      REAL ( KIND = wp ) :: val

       R( : m ) = Cv( : m )
       DO l = 1, Jv%ne
         i = Jv%row( l )
         R( i ) = R( i ) + Jv%val( l ) * D( Jv%col( l ) )
       END DO
       IF ( use_violation_squared ) THEN
         MODEL_v = half * SUM( R( : m ) ** 2 )
       ELSE
         MODEL_v = TWO_NORM( R( : m ) )
       END IF

!  End of function MODEL_v

       END FUNCTION MODEL_v

! -*-*-*-*-*-*-*- M O D E L _ F   I N T E R N A L   F U N C T I O N -*-*-*-*-*-

       FUNCTION MODEL_f( n_total, barrier, GRAD_barrier, HESS_barrier, D, HD )

! evaluate the f-model function
!  m^f(d) := f(v;mu) + grad f(v;mu)^T d + 1/2 d^T G d,  where v := ( x ),
!                                                                  ( c )
!  G = ( Hess Lagrangian(x,y)  0 ) +
!      (            0          0 )
!      ( Z^L (X - X^L)^{-1} + Z^U (X - X^U)^{-1}           0          )
!      (          0           Y^L (C - C^L)^{-1} + Y^U (C - C^U)^{-1} )
!
!  and f(v;mu) := f(x) - mu sum_{i=1}^M log (c-c^L)(c^U-c)
!                      - mu sum_{i=1}^N log (x-x^L)(x^U-x)
!
!  (all pre-computed)

       REAL ( KIND = wp ) :: MODEL_f

!  Dummy arguments

       INTEGER, INTENT( IN ) :: n_total
       REAL ( KIND = wp ), INTENT( IN ) :: barrier
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n_total ) :: GRAD_barrier, D
       REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n_total ) :: HD
       TYPE ( SMT_type ), INTENT( IN ) :: HESS_barrier

!  Local variable

       INTEGER :: i, j, l
       REAL ( KIND = wp ) :: val

       HD( : n_total ) = zero
       DO l = 1, HESS_barrier%ne
         i = HESS_barrier%row( l ) ; j = HESS_barrier%col( l )
         val = HESS_barrier%val( l )
         HD( i ) = HD( i ) + val * D( j )
         IF ( i /= j ) HD( j ) = HD( j ) + val * D( i )
       END DO
       MODEL_f = barrier                                                       &
                 + DOT_PRODUCT( GRAD_barrier( : n_total ), D( : n_total ) )    &
                 + half * DOT_PRODUCT( HD( : n_total ), D( : n_total ) )

!  End of function MODEL_f

       END FUNCTION MODEL_f

! -*-*-*-  G R A D _ M O D E L _ F   I N T E R N A L   F U N C T I O N  -*-*-*-

       FUNCTION GRAD_model_f( n_total, GRAD_barrier, HESS_barrier, D )

! evaluate the gradient of the f-model function
!  grad m^f(d) := grad f(v;mu) + G d,  where v := ( x ),
!                                                 ( c )
!  G = ( Hess Lagrangian(x,y)  0 ) +
!      (            0          0 )
!      ( Z^L (X - X^L)^{-1} + Z^U (X - X^U)^{-1}           0          )
!      (          0           Y^L (C - C^L)^{-1} + Y^U (C - C^U)^{-1} )
!
!  and f(v;mu) := f(x) - mu sum_{i=1}^M log (c-c^L)(c^U-c)
!                      - mu sum_{i=1}^N log (x-x^L)(x^U-x)
!
!  (all pre-computed)

       INTEGER :: n_total
       REAL ( KIND = wp ), DIMENSION( n_total ) :: GRAD_model_f

!  Dummy arguments

!      REAL ( KIND = wp ) :: barrier
       REAL ( KIND = wp ), DIMENSION( n_total ) :: GRAD_barrier, D
       TYPE ( SMT_type ) :: HESS_barrier

!  Local variable

       INTEGER :: i, j, l
       REAL ( KIND = wp ) :: val

       GRAD_model_f( : n_total ) = GRAD_barrier( : n_total )
       DO l = 1, HESS_barrier%ne
         i = HESS_barrier%row( l ) ; j = HESS_barrier%col( l )
         val = HESS_barrier%val( l )
         GRAD_model_f( i ) = GRAD_model_f( i ) + val * D( j )
         IF ( i /= j ) GRAD_model_f( j ) = GRAD_model_f( j ) + val * D( i )
       END DO

!  End of function GRAD_model_f

       END FUNCTION GRAD_model_f

! -*- S T E P _ f r a c t i o n _ t o _ b o u n d a r y   INTERNAL FUNCTION -*-

       FUNCTION STEP_fraction_to_boundary( n, m, m_in, n_total, V, S, omftb,   &
                         X_bound_type, X_l, X_u, C_bound_type, C_in, C_l, C_u, &
                         n_fr, X_fr )

!  find the largest alpha for which
!    v^L + ftb (v - v^L) <= v + alpha s <= v^U + ftb (v - v^U)
!  is satisfied along the step s from v. NB: omftb = 1 - ftb where 0 < ftb << 1

       REAL ( KIND = wp ) :: STEP_fraction_to_boundary

!  Dummy arguments

       INTEGER :: n, m, m_in, n_total
       REAL ( KIND = wp ) :: omftb
       INTEGER, DIMENSION( n ) :: X_bound_type
       INTEGER, DIMENSION( m ) :: C_bound_type
       INTEGER, DIMENSION( m_in ) :: C_in
       REAL ( KIND = wp ), DIMENSION( n_total ) :: V, S
       REAL ( KIND = wp ), DIMENSION( n ) :: X_l, X_u
       REAL ( KIND = wp ), DIMENSION( m ) :: C_l, C_u
       INTEGER, OPTIONAL :: n_fr
       INTEGER, OPTIONAL, DIMENSION( : ) :: X_fr

!  Local variable

       INTEGER :: i, ii, j
       STEP_fraction_to_boundary = HUGE( one )

       IF ( PRESENT( n_fr ) .AND. PRESENT( X_fr ) ) THEN
         DO ii = 1, n_fr
           i = X_fr( ii )
           SELECT CASE ( X_bound_type( i ) )
           CASE ( lower )
             IF ( S( ii ) < zero )                                             &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_l( i ) - V( ii ) ) / S( ii ) )
           CASE ( upper )
             IF ( S( ii ) > zero )                                             &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_u( i ) -  V( ii ) ) / S( ii ) )
           CASE ( both )
             IF ( S( ii ) > zero )                                             &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_u( i ) -  V( ii ) ) / S( ii ) )
             IF ( S( ii ) < zero )                                             &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_l( i ) - V( ii ) ) / S( ii ) )
           END SELECT
         END DO
         DO ii = 1, m_in
           i = C_in( ii )
           j = n_fr + ii
           SELECT CASE ( C_bound_type( i ) )
           CASE ( lower )
             IF ( S( j ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_l( i ) - V( j ) ) / S( j ) )
           CASE ( upper )
             IF ( S( j ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_u( i ) -  V( j ) ) / S( j ) )
           CASE ( both )
             IF ( S( j ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_l( i ) - V( j ) ) / S( j ) )
             IF ( S( j ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_u( i ) -  V( j ) ) / S( j ) )
           END SELECT
         END DO
       ELSE
         DO i = 1, n
           SELECT CASE ( X_bound_type( i ) )
           CASE ( lower )
             IF ( S( i ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_l( i ) - V( i ) ) / S( i ) )
           CASE ( upper )
             IF ( S( i ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_u( i ) -  V( i ) ) / S( i ) )
           CASE ( both )
             IF ( S( i ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_u( i ) -  V( i ) ) / S( i ) )
             IF ( S( i ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( X_l( i ) - V( i ) ) / S( i ) )
           END SELECT
         END DO
         DO ii = 1, m_in
           i = C_in( ii )
           j = n + ii
           SELECT CASE ( C_bound_type( i ) )
           CASE ( lower )
             IF ( S( j ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_l( i ) - V( j ) ) / S( j ) )
           CASE ( upper )
             IF ( S( j ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_u( i ) -  V( j ) ) / S( j ) )
           CASE ( both )
             IF ( S( j ) < zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_l( i ) - V( j ) ) / S( j ) )
             IF ( S( j ) > zero )                                              &
               STEP_fraction_to_boundary = MIN( STEP_fraction_to_boundary,     &
                 omftb * ( C_u( i ) -  V( j ) ) / S( j ) )
           END SELECT
         END DO
       END IF

!  End of function STEP_fraction_to_boundary

       END FUNCTION STEP_fraction_to_boundary

! -*-*- S L A C K _ R E S E T    I N T E R N A L   S U B R O U T I N E  -*-*-

       SUBROUTINE FUNNEL_slack_reset( m, m_eq, m_in, C_eq, C_in, n_fr,         &
                                      C_bound_type, C, C_l, C_u, V, Cv,        &
                                      norm_c, violation,                       &
                                      use_violation_squared, mv0 )

!   perform a slack reset to c_k as given by
!     for i: [c^L]_i <= [c_k]_i <= [c^A]_i
!       [c_k]_i <-       [c_k]_i                if [c(x_k)-c_k]_i <= 0,
!                   min( [c(x_k)]_i, [c^A]_i) ) otherwise
!     otherwise (i.e., [c^A]_i <= [c_k]_i <= [c^U]_i)
!       [c_k]_i <-       [c_k]_i                if [c(x_k)-c_k]_i >= 0,
!                   max( [c(x_k)]_i, [c^A]_i) ) otherwise
!   where c^A = 1/2 ( c^L + c^U ), and compute c(v_k) and the violation

!  Dummy arguments

       INTEGER, INTENT( IN ) :: m, m_eq, m_in, n_fr
       REAL ( KIND = wp ), INTENT( OUT ) :: norm_c, violation, mv0
       LOGICAL, INTENT( IN ) :: use_violation_squared
       INTEGER, INTENT( IN ), DIMENSION( m_eq ) :: C_eq
       INTEGER, INTENT( IN ), DIMENSION( m_in ) :: C_in
       INTEGER, INTENT( IN ), DIMENSION( m ) :: C_bound_type
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
       REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: Cv
       REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n_fr + m_in ) :: V

!  Local variable

       INTEGER :: i, ii, j
       REAL ( KIND = wp ) :: av

!  perform the reset

       DO ii = 1, m_in
         i = C_in( ii )
         j = n_fr + ii
         SELECT CASE ( C_bound_type( i ) )
         CASE ( lower )
!write(6,*) ' lower ', i, C( i ),  V( j )
           IF ( C( i ) > V( j ) ) write(6,*) ' slack reset'
           IF ( C( i ) > V( j ) ) V( j ) = C( i )
         CASE ( upper )
           IF ( C( i ) < V( j ) ) V( j ) = C( i )
         CASE ( both )
           av = half * ( C_l( i ) + C_u( i ) )
           IF ( V( j ) <= av ) THEN
             IF ( C( i ) > V( j ) ) V( j ) = MIN( C( i ), av )
           ELSE
             IF ( C( i ) < V( j ) ) V( j ) = MAX( C( i ), av )
           END IF
         END SELECT
       END DO

!  compute c(v_k) ...

       Cv( : m ) = C( : m )
       DO ii = 1, m_eq
         i = C_eq( ii )
         Cv( i ) = Cv( i ) - C_l( i )
       END DO
       DO ii = 1, m_in
         i = C_in( ii )
!write(6,*) ' cv ', i, Cv( i ),  V( n_fr + ii ), Cv( i ) - V( n_fr + ii )
         Cv( i ) = Cv( i ) - V( n_fr + ii )
       END DO

!  ... and vi(v_k) :=  m^v_k(0) := 1/2 ||c(v_k)||_2^2  (=violation)

       norm_c = TWO_NORM( Cv( : m ) )
       IF ( use_violation_squared ) THEN
         violation = half * norm_c ** 2
       ELSE
         violation = norm_c
       END IF
       mv0 = violation

       RETURN

!  End of subroutine FUNNEL_slack_reset

       END SUBROUTINE FUNNEL_slack_reset

!  End of subroutine FUNNEL_solve

     END SUBROUTINE FUNNEL_solve

!  End of module GALAHAD_FUNNEL

   END MODULE GALAHAD_FUNNEL_double

















