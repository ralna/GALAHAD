! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ T R I M S Q P   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!
!  Author: Nick Gould and Daniel Robinson


MODULE GALAHAD_TRIMSQP_double

!     ------------------------------------------------------------------
!    |                                                                  |
!    | trimSQP: a trust-region SQP method for solving nonlinear         |
!    |          optimization problems, in which descent is imposed      |
!    |          explicitely as an additional constraint.                |
!    |                                                                  |
!    | Aim: to find a (local) minimizer of the nonlinear                |
!    |      programming problem                                         |
!    |                                                                  |
!    |  minimize               f (x)                                    |
!    |  subject to        cl <= c(x) <= cu                              |
!    |                    Al <= A x  <= Au                              |
!    |                    xl <=   x  <= xu                              |
!    |                                                                  |
!     ------------------------------------------------------------------

  USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type, &
                                 NLPT_write_problem
  USE GALAHAD_LSQP_double
  USE GALAHAD_QPC_double
  USE GALAHAD_EQP_double
  USE GALAHAD_QPT_double
  USE GALAHAD_QPD_double, LSQP_data_type => QPD_data_type
  USE GALAHAD_SMT_double
  USE GALAHAD_SPECFILE_double
  USE GALAHAD_SPACE_double
  USE GALAHAD_COPYRIGHT
  USE GAlAHAD_SYMBOLS
  USE GALAHAD_NORMS_double
  USE GALAHAD_CHECK_double
  USE GALAHAD_SORT_double
  USE GALAHAD_mop_double

  IMPLICIT NONE

  PRIVATE
  PUBLIC :: TRIMSQP_initialize, TRIMSQP_read_specfile,   &
            TRIMSQP_solve, TRIMSQP_terminate

!  Set precision

  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

  REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
  REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
  REAL ( KIND = wp ), PARAMETER :: five = 5.0_wp
  REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
  REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
  REAL ( KIND = wp ), PARAMETER :: twentieth = 0.05_wp
  REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
  REAL ( KIND = wp ), PARAMETER :: tenm2 = 0.01_wp
  REAL ( KIND = wp ), PARAMETER :: tenm3 = 0.001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm4 = 0.0001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm6 = 0.000001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm7 = 0.0000001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm8 = 0.00000001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm9 = 0.000000001_wp
  REAL ( KIND = wp ), PARAMETER :: tenm10= 0.0000000001_wp
  REAL ( KIND = wp ), PARAMETER :: tenp2 = 100.0_wp
  REAL ( KIND = wp ), PARAMETER :: tenp3 = 1000.0_wp
  REAL ( KIND = wp ), PARAMETER :: tenp4 = 10000.0_wp
  REAL ( KIND = wp ), PARAMETER :: hundred = ten ** ( 2 )
  REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
  REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
  REAL ( KIND = wp ), PARAMETER :: point3 = 0.3_wp
  REAL ( KIND = wp ), PARAMETER :: point4 = 0.4_wp
  REAL ( KIND = wp ), PARAMETER :: point5 = 0.5_wp
  REAL ( KIND = wp ), PARAMETER :: point6 = 0.6_wp
  REAL ( KIND = wp ), PARAMETER :: point7 = 0.7_wp
  REAL ( KIND = wp ), PARAMETER :: point8 = 0.8_wp
  REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
  REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
  REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
  REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
  REAL ( KIND = wp ), PARAMETER :: tenp5  = ten ** 5
  REAL ( KIND = wp ), PARAMETER :: tenp6  = ten ** 6
  REAL ( KIND = wp ), PARAMETER :: tenp7  = ten ** 7
  REAL ( KIND = wp ), PARAMETER :: tenp8  = ten ** 8
  REAL ( KIND = wp ), PARAMETER :: tenp9  = ten ** 9
  REAL ( KIND = wp ), PARAMETER :: tiny   = ten ** ( - 7 )
  REAL ( KIND = wp ), PARAMETER :: very_tiny = ten ** ( - 9 )
  REAL ( KIND = wp ), PARAMETER :: mu_tiny = ten ** ( - 6 )
  REAL ( KIND = wp ), PARAMETER :: y_tiny = ten ** ( - 6 )
  REAL ( KIND = wp ), PARAMETER :: z_tiny = ten ** ( - 6 )
  REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 10 )
  REAL ( KIND = wp ), PARAMETER :: hzero = ten ** ( - 10 )
  REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( 1 )
!  REAL ( KIND = wp ), PARAMETER :: too_small = epsmch ** 0.5

  LOGICAL, PARAMETER :: exact_dual = .FALSE.
!  LOGICAL, PARAMETER :: exact_dual = .TRUE.


!  =====================================
!  The BFGS_data_type derived type
!  =====================================
  TYPE, PUBLIC :: TRIMSQP_bfgs_type
     integer :: mod_type
     REAL ( KIND = wp ) :: std, theta, damp_factor, stBs
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: d, Bs
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: gradLx_new, gradLx
  END TYPE TRIMSQP_bfgs_type

!  =====================================
!  The L_BFGS_data_type derived type
!  =====================================
  TYPE, PUBLIC :: TRIMSQP_l_bfgs_type
     integer, allocatable, dimension( : ) :: ind
     REAL ( KIND = wp ) :: eta
     REAL ( KIND = WP ), ALLOCATABLE, DIMENSION( :, : ) :: A, B, S, Y, SB_inner
     REAL ( KIND = WP ), ALLOCATABLE, DIMENSION( : ) :: gradLx_new, gradLx
     TYPE ( SMT_type ) :: A_smt, B_smt
  END TYPE TRIMSQP_l_bfgs_type

!  =====================================
!  The TRIMSQP_control_type derived type
!  =====================================
  TYPE, PUBLIC :: TRIMSQP_control_type
     INTEGER :: error, out, alive_unit, print_level, max_iterate
     INTEGER :: start_print, stop_print, print_gap
     INTEGER :: f_derivative_level, c_derivative_level
     INTEGER :: header_every, correction_type, NM_steps
     INTEGER :: B_type, L_BFGS_number, L_BFGS_curve_mod
     REAL ( KIND = wp ) :: stop_p, stop_c, stop_d
     REAL ( KIND = wp ) :: initial_penalty, max_penalty, penalty_expansion
     REAL ( KIND = wp ) :: initial_TRpred, max_TRpred
     REAL ( KIND = wp ) :: max_TRsqp, TRsqp_scale
     REAL ( KIND = wp ) :: max_infeas, infinity
     REAL ( KIND = wp ) :: eta_successful, eta_very_successful
     REAL ( KIND = wp ) :: eta_extremely_successful
     LOGICAL :: print_sol, fulsol, J_implicit, H_implicit
     LOGICAL :: space_critical, deallocate_error_fatal
     LOGICAL :: use_steering, use_seqp, use_siqp
     CHARACTER ( LEN = 30 ) :: alive_file
     TYPE ( QPC_control_type ) :: QPpred_control, QPsiqp_control
     TYPE ( QPC_control_type ) :: QPsteer_control
     TYPE ( EQP_control_type ) :: QPseqp_control
     TYPE ( LSQP_control_type ) :: QPfeas_control!, QPsteer_control
  END TYPE TRIMSQP_control_type

!  ================================
!  The TRIMSQP_time_type derived type
!  ================================

  TYPE, PUBLIC :: TRIMSQP_time_type
     REAL :: total, preprocess, analyse, factorize, solve
     REAL :: phase1_total, phase1_analyse, phase1_factorize, phase1_solve
  END TYPE TRIMSQP_time_type

!  ==================================
!  The TRIMSQP_inform_type derived type
!  ==================================

  TYPE, PUBLIC :: TRIMSQP_inform_type
     INTEGER :: status, alloc_status, iterate, cg_iter, itcgmx
     INTEGER :: num_f_eval, num_g_eval, num_c_eval, num_J_eval, num_H_eval
     INTEGER :: nvar, ngeval, iskip, ifixed, nfacts, nmods
     INTEGER :: factorization_status, num_descent_active
     INTEGER :: factorization_integer, factorization_real
     REAL ( KIND = wp ) :: obj, primal_vl, dual_vl, comp_vl
     LOGICAL :: newsol
     CHARACTER ( LEN = 80 ) :: bad_alloc
     TYPE ( TRIMSQP_time_type ) :: time
     TYPE ( QPC_inform_type )   :: QPpred_inform, QPsiqp_inform, QPsteer_inform
     TYPE ( EQP_inform_type )   :: QPseqp_inform
     TYPE ( LSQP_inform_type )  :: QPfeas_inform
  END TYPE TRIMSQP_inform_type

!  =====================================
!  The TRIMSQP_revert_type derived type
!  =====================================

  TYPE, PUBLIC :: TRIMSQP_revert_type
     REAL( KIND = wp ) :: f_revert, norm_c_revert, merit_revert
     !REAL( KIND = wp ) :: primal_vl_rev, dual_vl_rev, comp_vl_rev
     !REAL( KIND = wp ) :: sub_primal_vl_rev, sub_dual_vl_rev, sub_comp_vl_rev
     REAL( KIND = wp ) :: TRpred_revert
     REAL( KIND = wp ) :: primal_vl_revert, dual_vl_revert, comp_vl_revert
     REAL( KIND = wp ) :: penalty_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: g_revert, Jval_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_revert, Z_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_revert, Y_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ax_revert, Y_a_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Bval_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_RES_l_revert, C_RES_u_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_RES_l_revert, A_RES_u_revert
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_RES_l_revert, X_RES_u_revert
  END TYPE TRIMSQP_revert_type

!  =====================================
!  The TRIMSQP_nonmonotone_type derived type
!  =====================================

  TYPE, PUBLIC :: TRIMSQP_nonmonotone_type
     logical :: revert, active
     integer :: num_fail
     real ( KIND = wp ) :: decreaseH, merit_best
  END TYPE TRIMSQP_nonmonotone_type

!  =====================================
!  The TRIMSQP_data_type derived type
!  =====================================

  TYPE, PUBLIC :: TRIMSQP_data_type
     REAL ( KIND = wp ) :: penalty, penalty_new, merit, merit_new, ratio
     REAL ( KIND = wp ) :: penalty_pre_steer, merit_pre_steer
     REAL ( KIND = wp ) :: TRpred, TRsqp, alpha, alpha_c, alpha_feas, F_new
     REAL ( KIND = wp ) :: TRpred_expand, TRpred_contract, f_cauchy
     REAL ( KIND = wp ) :: inf_norm_Y_p, inf_norm_Y_c, inf_norm_Y_s
     REAL ( KIND = wp ) :: min_TRpred, min_TRsqp, TR_reset_value
     REAL ( KIND = wp ) :: primal_vl, dual_vl, comp_vl, inf_norm_Y_steer
     REAL ( KIND = wp ) :: primal_vl2, dual_vl2, comp_vl2
     REAL ( KIND = wp ) :: primal_vl3, dual_vl3, comp_vl3
     REAL ( KIND = wp ) :: sub_primal_vl, sub_dual_vl, eta, eta_contract
     REAL ( KIND = wp ) :: sub_comp_vl, sub_subgrad_vl
     REAL ( KIND = wp ) :: sub_stop_p, sub_stop_d, sub_stop_c
     REAL ( KIND = wp ) :: norm_c, norm_c_new, two_norm_s_p !norm_c_linearize
     REAL ( KIND = wp ) :: inf_norm_s_p, inf_norm_s_c, inf_norm_s_s, inf_norm_s_steer
     REAL ( KIND = wp ) :: inf_norm_s_f, H_norm_bound, inf_norm_Y, Y_bound
     REAL ( KIND = wp ) :: max_infeas
     REAL ( KIND = wp ) :: decreaseH_pred, decreaseH_cauchy, decreaseH_full
     REAL ( KIND = wp ) :: decreaseB, decreaseB_smooth
     REAL ( KIND = wp ) :: decrease_con_viol_s_p, decrease_con_viol_s_steer
     REAL ( KIND = wp ) :: gts_pred, stHs, vtBv, LBFGS_damping_factor
     REAL ( KIND = wp ) :: Sp_B_Sp, Sp_H_Sp, Sc_H_Sc, Ss_H_Ss, Sf_H_Sf
     REAL ( KIND = wp ) :: min_penalty
     REAL ( KIND = wp ) :: steer_L_factor, steer_Q_factor, con_viol_s_steer, ac_factor
     REAL ( KIND = wp ) :: opt_measure, opt_measure_cur, opt_measure_p, opt_measure_s
     REAL ( KIND = wp ) :: primal_vl_cur, primal_vl_p, primal_vl_s
     REAL ( KIND = wp ) :: dual_vl_cur, dual_vl_p, dual_vl_s
     REAL ( KIND = wp ) :: comp_vl_cur, comp_vl_p, comp_vl_s
     REAL ( KIND = wp ) :: dec_norm_c_pred, dec_norm_c_steer
     REAL ( KIND = wp ) :: norm_c_linearize_pred, norm_c_linearize_cauchy
     REAL ( KIND = wp ) :: norm_c_linearize_full, norm_c_linearize_steer
     LOGICAL :: converged, merit_first_order, blow_up, sqp_good_dec, TR_reset
     LOGICAL :: sqp_ratio_used, sqp_computed, step_accepted, LP_penalty_update
     LOGICAL :: seqp_computed, seqp_use_pred, penalty_steer_reset
     LOGICAL :: steering_good, computed_steering, mono_blow_up, check_suboptimal
     INTEGER :: iterate, lbreak, num_consec_blow_up, QPpred_fails
     INTEGER :: iterates_pred, iterates_sqp, num_consec_Y_free, consec_Y_free_needed
     INTEGER :: num_consec_Y_active, consec_Y_active_needed
     INTEGER :: num_sat, num_vl_l, num_vl_u, success, best_mults
     INTEGER :: nfr, nfx, nwA, nwA_comp, nwJ, nwJ_comp
     INTEGER :: nspos, nsneg, nJpos, nJneg, nApos, nAneg
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IBREAK
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: vl_l, vl_u, sat
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: fr, fx, wA, wA_comp, wJ, wJ_comp
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: spos, sneg, Apos, Aneg, Jpos, Jneg
     CHARACTER ( LEN = 7 ) :: success_str
     CHARACTER ( LEN = 4 ) :: descent_constraint_status
     CHARACTER ( LEN = 8 ) :: change_penalty
     CHARACTER ( LEN = 1 ) :: mults_used
     CHARACTER, ALLOCATABLE, DIMENSION( : )  :: approx_type
     CHARACTER ( LEN = 2 ), ALLOCATABLE, DIMENSION( : ) :: C_type, A_type,X_type
     TYPE ( TRIMSQP_control_type ) :: control
     TYPE ( QPT_problem_type ) :: QPpred, QPsiqp, QPseqp, QPfeas, QPsteer
     TYPE ( EQP_data_type ) :: QPseqp_data
     TYPE ( QPC_data_type ) :: QPpred_data, QPsiqp_data
     TYPE ( LSQP_data_type ) :: QPfeas_data, QPsteer_data
     TYPE ( TRIMSQP_l_bfgs_type ) :: L_BFGS
     TYPE ( TRIMSQP_bfgs_type ) :: BFGS
     TYPE ( TRIMSQP_revert_type ) :: revert
     TYPE ( TRIMSQP_nonmonotone_type ) :: NM
     type ( SMT_type ) :: B
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_new, C_cauchy
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: cauchy_Y, cauchy_Y_a
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CplusJsc, CplusJxSp, cauchy_Z
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: s_p, s_c, s_ac,  s_s, s_f, w, s_steer
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GplusHs, descent_con
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: J_norms, H_norms
     !REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Jv, Av, Atv, Atv2
     !REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Atv3, Jtv3
     !REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Atv_cur, Atv_p, Atv_s
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AxSp, AxSc, AxSac, AxSs, AXplusSc
     !REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Jtv, Jtv2, Hv, BxSp, HxSp, HxSc, HxSs, HxSf
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BxSp, HxSp, HxSc, HxSac, HxSs, HxSf
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JxSp, JxSc, JxSac, JxSs, JxSf, JxSsteer
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: u_in, v_in
     !REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Jtv_cur, Jtv_p, Jtv_s
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY, AtYa
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_cur, AtYa_cur
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_p, AtYa_p
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_s, AtYa_s
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: J_cauchy, g_prev, Jval_prev, Jval_dummy
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_seqp, Ya_seqp, Z_seqp
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAKP
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: cauchy_Zexact,sqp_Zexact, pred_Zexact
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Zexact
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_RES_l, C_RES_u
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_RES_l, A_RES_u
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_RES_l, X_RES_u
  END TYPE TRIMSQP_data_type

!  ==================================
!  Interfaces
!  ==================================

!  INTERFACE TWO_NORM
!
!     FUNCTION DNRM2( n, X, incx )
!       DOUBLE PRECISION :: DNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!     END FUNCTION DNRM2
!
!  END INTERFACE

CONTAINS

!-*-*-*-*  G A L A H A D -  TRIMSQP_initialize  S U B R O U T I N E -*-*-*-*

  SUBROUTINE TRIMSQP_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRIMSQP controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    TYPE ( TRIMSQP_data_type ), INTENT( OUT ) :: data
    TYPE ( TRIMSQP_control_type ), INTENT( OUT ) :: control
    TYPE ( TRIMSQP_inform_type ), INTENT( INOUT ) :: inform

!  Intialize LSQP data for feasible point subproblem.

    CALL LSQP_initialize( data%QPfeas_data, control%QPfeas_control,            &
                          inform%QPfeas_inform )
    !CALL LSQP_initialize( data%QPsteer_data, control%QPsteer_control )

!  Initialize QPC data for steering LP subproblem.

    CALL QPC_initialize( data%QPsteer_data, control%QPsteer_control,           &
                         inform%QPsteer_inform )
    control%QPsteer_control%prefix = '" - QPC-(steering):"         '
    control%QPsteer_control%QPA_control%prefix = '" -- QPA-(steering):"        '
    control%QPsteer_control%QPB_control%prefix = '" -- QPB-(steering):"        '

!  Intialize QPC data for predictor QP subproblem.

    CALL QPC_initialize( data%QPpred_data, control%QPpred_control,             &
                         inform%QPpred_inform )
    control%QPpred_control%prefix = '" - QPC-(predictor):"         '
    control%QPpred_control%QPA_control%prefix = '" -- QPA-(predictor):"        '
    control%QPpred_control%QPB_control%prefix = '" -- QPB-(predictor):"        '

!  Intialize QPC data for SIQP correction QP subproblem.

    CALL QPC_initialize( data%QPsiqp_data, control%QPsiqp_control,             &
                         inform%QPsiqp_inform )
    control%QPsiqp_control%prefix = '" - QPC-(siqp-corrector):"     '
    control%QPsiqp_control%QPA_control%prefix ='" -- QPA-(sqp-corrector):"     '
    control%QPsiqp_control%QPB_control%prefix ='" -- QPB-(sqp-corrector):"     '

!  Intialize EQP data for SEQP correction QP subproblem.

    CALL EQP_initialize( data%QPseqp_data, control%QPseqp_control,             &
                         inform% QPseqp_inform )
    control%QPsiqp_control%prefix = '" - QPC-(seqp-corrector):"     '

!  Error and ordinary output unit numbers

    control%error = 6
    control%out   = 6

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

    control%alive_unit = 60
    control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

    control%print_level = 1

!  Derivative level used.

    control%f_derivative_level = 2
    control%c_derivative_level = 2

!  How often major header is printed.

    control%header_every = 20

!  Control parameters for SQP step.  Correction_type determines the
!  type of constraints that are used in the SQP correction subproblem.
!  NM_steps determines the number of steps that are allowed in a
!  non-monotone framework before sufficient decrease in the merit
!  function is obtained.

    control%correction_type = 0
    control%NM_steps      = 1

!  Method for supplying Jacobian and Hessian of the Lagrangian.  These
!  may or may not be applicable based on the values of f_derivative_level
!  and c_derivative_level.

    control%J_implicit = .FALSE.
    control%H_implicit = .FALSE.

!  Maximum number of iterations

    control%max_iterate = 1000

!  Any printing will start on this iteration

    control%start_print = - 1

!  Any printing will stop on this iteration

    control%stop_print = - 1

!  Printing will only occur every print_gap iterations

    control%print_gap = 1

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than
!  control%stop_p and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stop_d

    control%stop_p = tenm5
    control%stop_c = tenm5
    control%stop_d = tenm5

!  Initial and maximum values for the trust-region radius for both predictor
!  and sqp correction subproblems.

    control%initial_TRpred     = one
    control%max_TRpred         = hundred
    control%max_TRsqp          = hundred
    control%TRsqp_scale = five

!  Type of positive definite approximation used for B. 0 = identity,
!  1 = wighted diagonal, 2 BFGS, 3 = limite-memory BFGS.  If LBFGS is used,
!  then L_BFGS_number tells teh number of vectors that will be used.

    control%B_type   = 2
    control%L_BFGS_number = 5
    control%L_BFGS_curve_mod = 1

!  The Maximum infeasibility tolerated will be the larger of max_infeas
!  and ten times the initial infeasibility

    control%max_infeas = 100.0_wp

!  User defined infinity

    control%infinity = ten ** 19

!  Initial and maximum value of penalty parameter and expansion factor

    control%initial_penalty   = one
    control%max_penalty   = tenp8
    control%penalty_expansion = two

!  Trust-region radius is adjusted based on whether f - f(x_new) is larger
!  than control%eta_successful(eta_very_successful) times that predicted
!  by a quadratic model of the decrease.

!    control%eta_successful = ten ** ( - 3 )
    control%eta_successful = ten ** ( - 1 )
    control%eta_very_successful = point5
    control%eta_extremely_successful = point9

!  Fulsol specifies whether the full solution or only highlights
!  will be printed

    control%print_sol = .FALSE.
    control%fulsol    = .TRUE.

!  If space_critical is true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

    control%space_critical = .FALSE.

!   Decide whether to use steering.

    control%use_steering = .TRUE.

!   Decide what time of SQP steps to try.

    control%use_seqp = .true.
    control%use_siqp = .true.

!   If deallocate_error_fatal is true, any array/pointer deallocation error
!   will terminate execution. Otherwise, computation will continue

    control%deallocate_error_fatal  = .FALSE.

    RETURN

!  End of subroutine TRIMSQP_initialize

  END SUBROUTINE TRIMSQP_initialize

!-*-*- T R I M S Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

  SUBROUTINE TRIMSQP_read_specfile( control, device, alt_specname_TRIMSQP,    &
                                    alt_specname_QPfeas, alt_specname_QPpred, &
                                    alt_specname_QPsiqp, alt_specname_QPseqp, &
                                    alt_specname_QPsteer )

!  Reads the content of a specification file, and performs the assignment
!  of values associated with given keywords to the corresponding control
!  parameters.

!  The default values as given by TRIMSQP_initialize could (roughly)
!  have been set as:

! BEGIN TRIMSQP SPECIFICATIONS (DEFAULT)
! error-printout-device                           6
! printout-device                                 6
! alive-device
! print-level                                     1
! print-header-every                              30
! start-print                                     11
! stop-print                                      66
! iterations-between-printing                     1
! print-solution                                  yes
! print-full-solution                             yes
! maximum-number-of-iterations                    1000
! correction-type                                 2
! corrector-trust-region-scale-factor             5.0D+0
! B-type                                          2
! number-limited-memory-vectors                   5
! non-monotone-steps                              2
! primal-accuracy-required                        1.0D-5
! dual-accuracy-required                          1.0D-5
! complementarity-accuracy-required               1.0D-5
! initial-radius-predictor                        1.0D+0
! maximimum-radius-predictor                      1.0D+2
! successful-iteration-tolerance                  0.01
! very-successful-iteration-tolerance             0.5
! extremely-successful-iteration-tolerance        0.9
!  maximum-infeasibility                          5.0D+3
! infinity                                        1.0D+19
! initial-penalty-parameter                       1.0D+0
! maximum-penalty-parameter                       1.0D+8
! penalty-parameter-expansion-factor              10.0D+0
! f-derivative-level                              2
! c-derivative-level                              2
! Jacobian-implicit                               no
! Hessian-implicit                                no
! space-critical                                  no
! deallocate-error-fatal                          no
! alive-filename
! END TRIMSQP SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    TYPE ( TRIMSQP_control_type ), INTENT( INOUT ) :: control
    INTEGER, INTENT( IN ) :: device
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_TRIMSQP
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_QPfeas
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_QPpred
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_QPsiqp
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_QPseqp
    CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_QPsteer

!  Programming: Nick Gould and Daniel Robinson, January 2008.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    INTEGER, PARAMETER :: lspec = 60
    CHARACTER( LEN = 16 ), PARAMETER :: specname_TRIMSQP = 'TRIMSQP         '
    CHARACTER( LEN = 16 ), PARAMETER :: specname_QPfeas  = 'QP_feasibility  '
    CHARACTER( LEN = 16 ), PARAMETER :: specname_QPpred  = 'QP_predictor    '
    CHARACTER( LEN = 16 ), PARAMETER :: specname_QPsiqp  = 'QP_siqp         '
    CHARACTER( LEN = 16 ), PARAMETER :: specname_QPseqp  = 'QP_seqp         '
    CHARACTER( LEN = 16 ), PARAMETER :: specname_QPsteer = 'QP_steering     '
    TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

    spec%keyword = ''

!  Integer key-words

    spec(  1 )%keyword = 'error-printout-device'
    spec(  2 )%keyword = 'printout-device'
    spec(  3 )%keyword = 'alive-device'
    spec(  4 )%keyword = 'print-level'
    spec(  5 )%keyword = 'maximum-number-of-iterations'
    spec(  6 )%keyword = 'start-print'
    spec(  7 )%keyword = 'stop-print'
    spec(  8 )%keyword = 'iterations-between-printing'
    spec(  9 )%keyword = 'f-derivative-level'
    spec( 10 )%keyword = 'c-derivative-level'
    spec( 11 )%keyword = 'print-header-every'
    spec( 12 )%keyword = 'correction-type'
    spec( 13 )%keyword = 'non-monotone-steps'
    spec( 14 )%keyword = 'B-type'
    spec( 15 )%keyword = 'number-limited-memory-vectors'
    spec( 16 )%keyword = 'L-BFGS-curvature-fix-type'

!  Real key-words

    spec( 17 )%keyword = 'primal-accuracy-required'
    spec( 18 )%keyword = 'dual-accuracy-required'
    spec( 19 )%keyword = 'complementarity-accuracy-required'
    spec( 20 )%keyword = 'successful-iteration-tolerance'
    spec( 21 )%keyword = 'very-successful-iteration-tolerance'
    spec( 22 )%keyword = 'extremely-successful-iteration-tolerance'
    spec( 23 )%keyword = 'initial-radius-predictor'
    spec( 24 )%keyword = 'maximum-radius-predictor'
    spec( 25 )%keyword = 'corrector-trust-region-scale-factor'

    spec( 27 )%keyword = 'maximum-infeasibility'
    spec( 28 )%keyword = 'infinity'
    spec( 29 )%keyword = 'initial-penalty-parameter'
    spec( 30 )%keyword = 'maximum-penalty-parameter'
    spec( 31 )%keyword = 'penalty-parameter-expansion-factor'

!  Logical key-words

    spec( 40 )%keyword = 'space-critical'
    spec( 41 )%keyword = 'deallocate-error-fatal'
    spec( 42 )%keyword = 'use-steering'
    spec( 43 )%keyword = 'use-seqp'
    spec( 44 )%keyword = 'use-siqp'

    spec( 45 )%keyword = 'Jacobian-implicit'
    spec( 46 )%keyword = 'Hessian-implicit'

    spec( 48 )%keyword = 'print-solution'
    spec( 49 )%keyword = 'print-full-solution'

!  Character key-words

    spec( 60 )%keyword = 'alive-filename'

!  Read the specfile

    IF ( PRESENT( alt_specname_TRIMSQP ) ) THEN
       CALL SPECFILE_read( device, alt_specname_TRIMSQP, spec, lspec, control%error )
    ELSE
       CALL SPECFILE_read( device, specname_TRIMSQP, spec, lspec, control%error )
    END IF

!  Interpret the result

!  Set integer values

    CALL SPECFILE_assign_integer( spec( 1 ), control%error,                   &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 2 ), control%out,                     &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 3 ), control%out,                     &
                                  control%alive_unit )
    CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,             &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 5 ), control%max_iterate,             &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,             &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,              &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,               &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 9 ), control%f_derivative_level,      &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 10 ), control%c_derivative_level,     &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 11 ), control%header_every,           &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 12 ), control%correction_type,       &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 13 ), control%NM_steps,             &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 14 ), control%B_type,                 &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 15 ), control%L_BFGS_number,           &
                                  control%error )
    CALL SPECFILE_assign_integer( spec( 16 ), control%L_BFGS_curve_mod,        &
                                  control%error )


!  Set real values

    CALL SPECFILE_assign_real( spec( 17 ), control%stop_p,                    &
                                control%error )
    CALL SPECFILE_assign_real( spec( 18 ), control%stop_d,                    &
                                control%error )
    CALL SPECFILE_assign_real( spec( 19 ), control%stop_c,                    &
                                control%error )
    CALL SPECFILE_assign_real( spec( 20 ), control%eta_successful,            &
                               control%error )
    CALL SPECFILE_assign_real( spec( 21 ), control%eta_very_successful,       &
                               control%error )
    CALL SPECFILE_assign_real( spec( 22 ), control%eta_extremely_successful,  &
                                control%error )
    CALL SPECFILE_assign_real( spec( 23 ), control%initial_TRpred,            &
                                control%error )
    CALL SPECFILE_assign_real( spec( 24 ), control%max_TRpred,                &
                                control%error )
    CALL SPECFILE_assign_real( spec( 25 ), control%TRsqp_scale,        &
                                control%error )
!    CALL SPECFILE_assign_real( spec( 26 ), control%max_TRsqp,                &
!                                control%error )
    CALL SPECFILE_assign_real( spec( 27 ), control%max_infeas,                &
                                control%error )
    CALL SPECFILE_assign_real( spec( 28 ), control%infinity,                  &
                                control%error )
    CALL SPECFILE_assign_real( spec( 29 ), control%initial_penalty,           &
                                control%error )
    CALL SPECFILE_assign_real( spec( 30 ), control%max_penalty,               &
                                control%error )
    CALL SPECFILE_assign_real( spec( 31 ), control%penalty_expansion,         &
                                control%error )


!  Set logical values

    CALL SPECFILE_assign_logical( spec( 40 ), control%space_critical,         &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 41 ),                                 &
                                  control%deallocate_error_fatal,             &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 42 ),                                 &
                                  control%use_steering,                       &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 43 ),                                 &
                                  control%use_seqp,                           &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 44 ),                                 &
                                  control%use_siqp,                           &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 45 ), control%J_implicit,             &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 46 ), control%H_implicit,             &
                                  control%error )
    CALL SPECFILE_assign_logical( spec( 48 ), control%print_sol,              &
                                   control%error )
    CALL SPECFILE_assign_logical( spec( 49 ), control%fulsol,                 &
                                   control%error )

!  Set character values

    CALL SPECFILE_assign_string( spec( 60 ), control%alive_file,              &
                                 control%error )

!  Set LSQP control values

    IF ( PRESENT( alt_specname_QPfeas ) ) THEN
      CALL LSQP_read_specfile(control%QPfeas_control,device,alt_specname_QPfeas)
    ELSE
      CALL LSQP_read_specfile(control%QPfeas_control,device,specname_QPfeas)
    END IF

    !IF ( PRESENT( alt_specname_QPsteer ) ) THEN
    !  CALL LSQP_read_specfile(control%QPsteer_control,device,alt_specname_QPsteer)
    !ELSE
    !  CALL LSQP_read_specfile(control%QPsteer_control,device,specname_QPsteer)
    !END IF


!  Set QPC control values

    IF ( PRESENT( alt_specname_QPpred ) ) THEN
       CALL QPC_read_specfile(control%QPpred_control,device,alt_specname_QPpred)
    ElSE
       CALL QPC_read_specfile( control%QPpred_control, device, specname_QPpred )
    END IF

    IF ( PRESENT( alt_specname_QPsiqp )  ) THEN
       CALL QPC_read_specfile( control%QPsiqp_control, device, alt_specname_QPsiqp )
    ElSE
       CALL QPC_read_specfile( control%QPsiqp_control, device, specname_QPsiqp )
    END IF

    IF ( PRESENT( alt_specname_QPseqp )  ) THEN
       CALL EQP_read_specfile( control%QPseqp_control, device, alt_specname_QPseqp )
    ElSE
       CALL EQP_read_specfile( control%QPseqp_control, device, specname_QPseqp )
    END IF

    IF ( PRESENT( alt_specname_QPsteer ) ) THEN
      CALL QPC_read_specfile(control%QPsteer_control,device,alt_specname_QPsteer)
    ELSE
      CALL QPC_read_specfile(control%QPsteer_control,device,specname_QPsteer)
    END IF

    RETURN

  END SUBROUTINE TRIMSQP_read_specfile

!-*-*-*  G A L A H A D -  T R I M S Q P _ s o l v e  S U B R O U T I N E  -*-*-*

  SUBROUTINE TRIMSQP_solve( nlp, control, inform, data, eval_FC, eval_G,       &
                            eval_J, eval_HL, userdata )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  TRIMSQP_solve : an SQP method for finding a local minimizer of an objective
!                  function f(x), subject to general constraints, linear
!                  constraints, and simple bounds on the variables.
!
! control%print_level = GALAHAD_SILENT  : nothing is printed
!                       GALAHAD_TRACE   : main user line is printed (one line)
!                       GALAHAD_ACTION  : addionally, main summary info.
!                       GALAHAD_DETAILS :
!                       GALAHAD_DEBUG   :
!                       GALAHAD_CRAZY   : more or less everything.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
    TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
    TYPE ( NLPT_userdata_type ), INTENT( INOUT ), OPTIONAL :: userdata
    TYPE ( TRIMSQP_control_type ), INTENT( INOUT ) :: control
    TYPE ( TRIMSQP_inform_type ), INTENT( INOUT ) :: inform
    TYPE ( TRIMSQP_data_type ), INTENT( INOUT ) :: data
    OPTIONAL eval_FC, eval_G, eval_J, eval_HL

    INTERFACE

       SUBROUTINE eval_FC( status, X, userdata, F, C )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
         REAL ( kind = wp ), OPTIONAL, INTENT( OUT ) :: F
         REAL ( kind = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC

       SUBROUTINE eval_G( status, X, userdata, G )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G

       SUBROUTINE eval_J( status, X, userdata, J_val )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: J_val
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_J

       SUBROUTINE eval_HL(status, X, Y, userdata, Hval, no_f )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
         LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL

    END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

  integer :: i, iores, out, max_iterate, m, m_a, n, nLMV
  real ( kind = wp ) :: dummy_real, too_small, Gi, vl
  real ( kind = wp ) :: stop_p!, stop_d, stop_c
  !real ( kind = wp ) :: opt_measure, opt_measure_cur, opt_measure_p, opt_measure_s
  logical :: filexx, print_1line, print_detail, print_debug, qpc_successful
  INTEGER :: sfiledevice = 62                           ! solution file number
!  INTEGER :: QPpred_sif_out = 66
  CHARACTER ( LEN = 30 ) :: sfilename = 'TRIMSQPSOL.d'  ! solution file name

!******************************************************************************
!******************************************************************************
write(*,*) ' -------************ (n,m,ma) = ', nlp%n, nlp%m, nlp%m_a

  nLMV = data%control%L_BFGS_number

! Copy control data structure into data.

  data%control = control

! Not sure where else to put this.

  nlp%infinity = data%control%infinity

! For convenience

  out         = data%control%out
  max_iterate = data%control%max_iterate
  m           = nlp%m
  m_a         = nlp%m_a
  n           = nlp%n
  stop_p      = data%control%stop_p
  !stop_d      = data%control%stop_d
  !stop_c      = data%control%stop_c

! Initalize some components of data.

  data%penalty = data%control%initial_penalty
  data%TRpred  = data%control%initial_TRpred
  data%TRsqp   = data%control%TRsqp_scale * data%control%initial_TRpred

  data%lbreak  = 2*nlp%m   ! generally an overestimate by far.

  data%penalty_new            = data%penalty
  data%min_penalty            = tenm5
  data%consec_Y_free_needed   = 10
  data%consec_Y_active_needed = 10
  !data%prev_Y_active          = 0
  data%iterate                = 0
  !data%prev_iterate           = 0

  !data%steer_L_factor = point9
  !data%steer_L_factor = point5
  data%steer_L_factor = point1
  !data%steer_Q_factor = point5
  !data%steer_Q_factor = tenm2
  !data%steer_Q_factor = tenm5
  data%steer_Q_factor = point1
  !data%steer_Q_factor = point5
  data%steering_good  = .false.

  data%ac_factor = tenm2

  data%mono_blow_up = .false.

  data%inf_norm_Y_p = zero
  data%inf_norm_Y_c = zero
  data%inf_norm_Y_s = zero

  !data%eta               = one
  !data%eta_contract      = half
  !data%sub_stop_p        = min( point1, half*data%eta )
  !data%sub_stop_d        = data%sub_stop_p
  !data%sub_stop_c        = data%sub_stop_p


  data%TRpred_expand     = two
  data%TRpred_contract   = half
  data%min_TRpred        = ten ** ( - 7 )
  data%min_TRsqp         = ten ** ( - 7 )
!  data%min_TRpred        = ten ** ( - 10 )
!  data%min_TRsqp         = ten ** ( - 10 )
  !data%TR_reset_value    = min ( point1, tenp5 * data%min_TRpred )
  !data%TR_reset_value    = 0.000001_wp
  data%TR_reset_value    = 0.000001_wp
  data%TR_reset_value    = max( data%TR_reset_value, data%min_TRpred )
  data%converged         = .false.
  data%step_accepted     = .true.
  data%LP_penalty_update = .false.
  data%change_penalty    = '    same'
  data%success_str       = '       '
  data%success           = 0
  data%H_norm_bound      = ten ** ( 5 )
  data%BFGS%damp_factor  = 0.2_wp
  data%NM%revert         = .false.
  data%NM%active         = .false.
  data%NM%num_fail       = 0

  data%penalty_steer_reset = .false.

  inform%iterate    = 0
  inform%primal_vl  = one
  inform%dual_vl    = one
  inform%comp_vl    = one
  inform%obj        = zero
  inform%time%total = zero
  inform%num_f_eval = 0
  inform%num_g_eval = 0
  inform%num_c_eval = 0
  inform%num_J_eval = 0
  inform%num_H_eval = 0
  inform%status     = 0
  data%num_sat      = 0
  data%num_vl_l     = 0
  data%num_vl_u     = 0

  inform%num_descent_active = 0
  data%num_consec_blow_up   = 0

! Set dimensions of matrices of type SMT_type
! (this probably could/should be done by user).

  if ( nlp%m_a > 0 ) then
     nlp%A%m = nlp%m_a
     nlp%A%n = nlp%n
  end if

  if ( nlp%m > 0 ) then
     nlp%J%m  = nlp%m
     nlp%J%n  = nlp%n
     nlp%J%ne = size( nlp%J%val )
  end if

  nlp%H%m = nlp%n
  nlp%H%n = nlp%n

! Allocate some components of data.

  CALL SPACE_resize_array( nlp%n, data%s_p, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%s_c, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%s_ac, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%s_s, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%s_f, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%s_steer, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Hv, inform%status, inform%alloc_status  )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%HxSp, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%HxSc, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%HxSac, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%HxSs, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%HxSf, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%BxSp, inform%status, inform%alloc_status  )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array(nlp%n, data%GplusHs, inform%status,      &
                                               inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%descent_con, inform%status,      &
                                                    inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%X_type, inform%status,      &
                                               inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%X_RES_l, inform%status,      &
                                                inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%X_RES_u, inform%status,      &
                                                inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%spos, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%sneg, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%cauchy_Zexact, inform%status,      &
                                                      inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%pred_Zexact, inform%status,      &
                                                      inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%cauchy_Z, inform%status,      &
                                                 inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%sqp_Zexact, inform%status,      &
                                                   inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%Zexact, inform%status,      &
                                                   inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%Z_seqp, inform%status,      &
                                                   inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%H_norms, inform%status,      &
                                                inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

 ! CALL SPACE_resize_array( nlp%m, data%Jv, inform%status, inform%alloc_status )
 ! IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSp, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSc, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSac, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSs, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSsteer, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%JxSf, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%Jpos, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%Jneg, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%u_in, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%v_in, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%JtY, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%JtY_cur, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%JtY_p, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%JtY_s, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv_cur, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv_p, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv_s, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv2, inform%status, inform%alloc_status)
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Jtv3, inform%status, inform%alloc_status)
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array(nlp%m, data%C_new, inform%status, inform%alloc_status)
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%C_cauchy, inform%status,      &
                                                 inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%cauchy_Y, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%C_type, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%CplusJSc, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%CplusJxSp, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

!  CALL SPACE_resize_array( nlp%m, data%CplusJxSs, inform%status, inform%alloc_status )
!  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%w, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%J_norms, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%Y_seqp, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%sat, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%vl_l, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%vl_u, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( data%lbreak, data%IBREAK, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( data%lbreak, data%BREAKP, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%C_RES_l, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m, data%C_RES_u, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%m_a, data%Av, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%AXplusSc, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%AxSp, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%AxSc, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%AxSac, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%AxSs, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%Apos, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%Aneg, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%Ya_seqp, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%G_prev, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%AtYa, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%AtYa_cur, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%AtYa_p, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%n, data%AtYa_s, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv_cur, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv_p, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv_s, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv2, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  !CALL SPACE_resize_array( nlp%n, data%Atv3, inform%status, inform%alloc_status )
  !IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%A_type, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%A_RES_l, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%A_RES_u, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  CALL SPACE_resize_array( nlp%m_a, data%cauchy_Y_a, inform%status, inform%alloc_status )
  IF ( inform%status /= GALAHAD_ok ) GO TO 990

  if ( data%control%NM_steps > 0 ) then

     CALL SPACE_resize_array( nlp%n, data%revert%G_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%n, data%revert%X_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%n, data%revert%Z_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m, data%revert%C_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m, data%revert%Y_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m_a, data%revert%Ax_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m_a, data%revert%Y_a_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m, data%revert%C_RES_l_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m, data%revert%C_RES_u_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m_a, data%revert%A_RES_l_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%m_a, data%revert%A_RES_u_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%n, data%revert%X_RES_l_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%n, data%revert%X_RES_u_revert, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

  end if

  ! Determine constraint type.
  !***************************

  ! General constraints.

  if ( nlp%m > 0 ) then
     do i = 1, nlp%m
        if ( nlp%C_l(i) > - data%control%infinity ) then
           if ( nlp%C_u(i) < data%control%infinity ) then
              if ( abs( nlp%C_u(i) - nlp%C_l(i) ) > two*data%control%stop_p ) then
                 data%C_type(i) = 'RB'
              else
                 data%C_type(i) = 'EQ'
                 nlp%C_u( i )   = ( nlp%C_l( i ) + nlp%C_u(i) ) / two
                 nlp%C_l( i )   = nlp%C_u( i )
              end if
           else
              data%C_type(i) = 'LB'
           end if
        elseif ( nlp%C_u(i) >= data%control%infinity ) then
           data%C_type(i) = 'FR'
        else
           data%C_type(i) = 'UB'
        end if
     end do
  end if

  ! Linear constraints.

  if ( nlp%m_a > 0 ) then
     do i = 1, nlp%m_a
        if ( nlp%A_l(i) > - data%control%infinity ) then
           if ( nlp%A_u(i) < data%control%infinity ) then
              if ( abs( nlp%A_u(i) - nlp%A_l(i) ) > two*data%control%stop_p ) then
                 data%A_type(i) = 'RB'
              else
                 data%A_type(i) = 'EQ'
                 nlp%A_l( i )   = ( nlp%A_l( i ) + nlp%A_u( i ) ) / two
                 nlp%A_u( i )   = nlp%A_l( i )
              end if
           else
              data%A_type(i) = 'LB'
           end if
        elseif ( nlp%A_u(i) >= data%control%infinity ) then
           data%A_type(i) = 'FR'
        else
           data%A_type(i) = 'UB'
        end if
     end do
  end if

  ! Bound constraints.

  do i = 1, nlp%n
     if ( nlp%X_l(i) > - data%control%infinity ) then
        if ( nlp%X_u(i) < data%control%infinity ) then
           if ( abs( nlp%X_u(i) - nlp%X_l(i) ) > two*data%control%stop_p ) then
              data%X_type(i) = 'RB'
           else
              data%X_type(i) = 'EQ'
              nlp%X_l( i )   = ( nlp%X_l( i ) + nlp%X_u( i ) ) / two
              nlp%X_u( i )   = nlp%X_l( i )
           end if
        else
           data%X_type(i) = 'LB'
        end if
     elseif ( nlp%X_u(i) >= data%control%infinity ) then
        data%X_type(i) = 'FR'
     else
        data%X_type(i) = 'UB'
     end if
  end do

  ! Allocate and set sparsity for predictor subproblem.
  ! ***************************************************
  call build_QPpred( nlp, data%QPpred, inform, data )
  if ( inform%status /= GALAHAD_ok ) then
     write(out,"('ERROR : trimsqp : subroutine build_QPpred')")
     GO TO 999
  end if

  ! Allocate and set sparsity for SQP subproblems.
  ! *********************************************

  if ( data%control%use_siqp ) then
     call build_QPsiqp( nlp, data%QPsiqp, inform, data )
     if ( inform%status /= GALAHAD_ok ) then
       write(out,"(' ERROR : trimsqp : subroutine build_QPsiqp')")
       GO TO 999
    end if
  end if
  if ( data%control%use_seqp ) then
     call build_QPseqp( nlp, data%QPseqp, inform, data )
     if ( inform%status /= GALAHAD_ok ) then
        write(out,"(' ERROR : trimsqp : subroutine build_QPseqp')")
        GO TO 999
     end if
  end if

  !if ( nlp%pname == 'HS99' ) then
     !nlp%X = zero
  !end if

  ! Make x feasible with respect to the bounds.
  ! *************************************************

  if ( data%control%print_level >= GALAHAD_DEBUG ) then
     write(data%control%out,3075) nlp%X
  end if

  nlp%X = min( max( nlp%X, nlp%X_l ), nlp%X_u )

  if ( data%control%print_level >= GALAHAD_DEBUG ) then
     write(data%control%out,3076) nlp%X
  end if

  ! Find "closest" x that is feasible with respect to the linear constraints.
  ! *************************************************************************

  if ( nlp%m_a > 0 ) then

     call build_QPfeas( nlp, data%QPfeas, inform, data%control)
     if ( inform%status /= GALAHAD_ok ) then
        write(out,"(' ERROR : trimsqp : subroutine build_QPfeas')")
        GO TO 999
     end if

     call fill_QPfeas( nlp, data%QPfeas, inform, data%control )
     if ( inform%status /= GALAHAD_ok ) then
        write(out,"(' ERROR : trimsqp : subroutine fill_QPfeas')")
        GO TO 999
     end if

     if ( data%control%print_level >= GALAHAD_DEBUG ) then
        write(out,*) ' ================ BEGIN : QPfeas ================='
        call QPT_write_problem( out, data%QPfeas, 1 )
        write(out,*) ' ================ END : QPfeas ===================='
     end if

     call LSQP_solve( data%QPfeas, data%QPfeas_data,  &
                      data%control%QPfeas_control, inform%QPfeas_inform )

     if ( inform%QPfeas_inform%status /= GALAHAD_OK ) then
        write(out,*) ' QPfeas : return status ', inform%QPfeas_inform%status
        GO TO 999
     end if

     ! get the solution.

     nlp%X   = data%QPfeas%X( : nlp%n )
     nlp%Y_a = data%QPfeas%Y

     nlp%Ax = zero
     call mop_Ax( one, nlp%A, nlp%X, one, nlp%Ax,      &
                  data%control%out, data%control%error )

     ! print the updated value of X.

     if ( data%control%print_level >= GALAHAD_DEBUG ) then
        write(data%control%out,3077) nlp%X
     end if

  end if

  ! Build QP used for steering subproblem.
  ! **************************************

  if ( nlp%m > 0 ) then
     call build_QPsteer( nlp, data%QPsteer, inform, data%control )
     if ( inform%status /= GALAHAD_OK ) then
        write(out,"(' ERROR : trimsqp : subroutine build_QPsteer')")
        GO TO 999
     end if
  end if

  ! Evaluate functions
  ! ******************

  call eval_FC( inform%status, nlp%X, userdata, nlp%f, nlp%c )
  if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_FC'

  inform%num_f_eval = inform%num_f_eval + 1

  call eval_G( inform%status, nlp%X, userdata, nlp%G )
  if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_G'

  inform%num_g_eval = inform%num_g_eval + 1

  if ( nlp%m > 0 ) then

     call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
     if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_J'

     CALL SPACE_resize_array( nlp%J%ne, data%J_cauchy, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%J%ne, data%Jval_prev, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     CALL SPACE_resize_array( nlp%J%ne, data%Jval_dummy, inform%status, inform%alloc_status )
     IF ( inform%status /= GALAHAD_ok ) GO TO 990

     inform%num_J_eval = inform%num_J_eval + 1
     inform%num_c_eval = inform%num_c_eval + 1  !From earlier.

     if ( data%control%NM_steps > 0 ) then
        CALL SPACE_resize_array( nlp%J%ne, data%revert%Jval_revert, inform%status, inform%alloc_status )
        IF ( inform%status /= GALAHAD_ok ) GO TO 990
     end if

  end if

  ! Constraint violation and maximum allowable infeasibility
  ! ********************************************************

  if ( nlp%m > 0 ) then
     call constraint_violation( nlp, nlp%C, data, data%norm_c, inform%status )
  else
     data%norm_c          = zero ;           data%norm_c_linearize_pred = zero
     data%penalty         = zero ;           data%norm_c_linearize_cauchy = zero
     data%penalty_new     = zero ;           data%norm_c_linearize_steer = zero
     data%change_penalty  = '      NA' ;     data%norm_c_linearize_full = zero
     data%C_new           = zero ;
     data%norm_c_new      = zero ;
  end if

  ! Compute merit function
  ! **********************

  data%merit = nlp%f + data%penalty * data%norm_c

  ! Print the problem.
  ! ******************

  if ( data%control%print_level >= GALAHAD_DETAILS ) then
     call NLPT_write_problem( nlp, out, 4 )
  end if

  ! Set up tolerances for merit function.

  data%eta           = one
  data%eta_contract  = half
  data%sub_stop_p    = min( point1, half*data%eta )
  data%sub_stop_d    = data%sub_stop_p
  data%sub_stop_c    = data%sub_stop_p

!============================================================================
! BEGIN: main do loop.
!============================================================================

  do while ( (.not. data%converged) .and. (data%iterate <= max_iterate) )

     ! Check for max iterations.
     ! *************************

     if ( data%iterate == max_iterate) then
        inform%status = GALAHAD_error_max_iterations
     end if

     ! If reverted, skip to print main summary line.
     ! *********************************************

     if ( data%NM%revert ) then
        data%NM%revert  = .false.
        data%mults_used = 'R'
        data%best_mults = 0
        goto 710
     end if

     ! If penalty was increased due to steering and step not acceppted, reset.
     ! ***********************************************************************

     if ( data%penalty_steer_reset ) then
        data%penalty             = data%penalty_pre_steer
        data%merit               = data%merit_pre_steer
        data%penalty_steer_reset = .false.
     end if

     ! Compute residual vectors if step has been accepted.
     !****************************************************

     if ( data%step_accepted .or. data%iterate == 0 ) then

        if ( nlp%m > 0 ) then
           call get_residuals( nlp%C_l, nlp%C, nlp%C_u,  &
                               data%C_type, data%C_RES_l, data%C_RES_u )
        end if
        if ( nlp%m_a > 0 ) then
           call get_residuals( nlp%A_l, nlp%Ax, nlp%A_u,  &
                               data%A_type, data%A_RES_l, data%A_RES_u )
        end if
        call get_residuals( nlp%X_l, nlp%X, nlp%X_u,  &
                            data%X_type, data%X_RES_l, data%X_RES_u )

     end if

     ! Check optimality using the values in nlp.
     !******************************************

     if ( nlp%m > 0 ) then
        data%JtY_cur = zero
        call mop_Ax( one, nlp%J, nlp%Y( : nlp%m ), one, data%JtY_cur,      &
                     out, data%control%error, transpose=.true. )
     end if
     if ( nlp%m_a > 0 ) then
        data%AtYa_cur = zero
        call mop_Ax( one, nlp%A, nlp%Y_a( : nlp%m_a ), one, data%AtYa_cur, &
                     out, data%control%error, transpose=.true. )
     end if

     call check_optimal(nlp, nlp%G, data%JtY_cur, data%AtYa_cur,              &
                     nlp%Z( : n), data%Zexact, nlp%Y( :m), nlp%Y_a( : m_a),   &
                     data%C_type, data%A_type, data%X_type,                   &
                     data%C_RES_l, data%C_RES_u,                              &
                     data%A_RES_l, data%A_RES_u,                              &
                     data%X_RES_l, data%X_RES_u,                              &
                     data%primal_vl_cur, data%dual_vl_cur,  data%comp_vl_cur, &
                     exact_dual, inform%status, data%control%out )

     data%opt_measure_cur = max( data%primal_vl_cur, data%dual_vl_cur )
     data%opt_measure_cur = max( data%opt_measure_cur, data%comp_vl_cur )

     if ( data%iterate == 0 ) then

        data%best_mults = 1
        data%mults_used = 'I'

        if ( exact_dual ) then
           nlp%Z = data%Zexact
        end if

        data%primal_vl = data%primal_vl_cur
        data%dual_vl   = data%dual_vl_cur
        data%comp_vl   = data%comp_vl_cur

        data%opt_measure = data%opt_measure_cur

        ! need these when checking optimality of merit function.

        data%JtY = data%JtY_cur ;     data%AtYa = data%AtYa_cur

        goto 700

     end if

     ! Check optimality for predictor multipliers.
     !********************************************

     if ( nlp%m > 0 ) then
        data%JtY_p = zero
        call mop_Ax( one, nlp%J, data%QPpred%Y(: m), one, data%JtY_p, &
                     out, data%control%error, transpose=.true. )
     end if

     if ( nlp%m_a > 0 ) then
        data%AtYa_p = zero
        call mop_Ax( one, nlp%A, data%QPpred%Y( m+1:m+m_a ), one, data%AtYa_p, &
                     out, data%control%error, transpose=.true. )
     end if

     call check_optimal( nlp, nlp%G, data%JtY_p, data%AtYa_p,                  &
                        data%QPpred%Z(:nlp%n), data%pred_Zexact,               &
                        data%QPpred%Y(:nlp%m),                                 &
                        data%QPpred%Y(nlp%m+1 :nlp%m+nlp%m_a),                 &
                        data%C_type, data%A_type, data%X_type,                 &
                        data%C_RES_l, data%C_RES_u,                            &
                        data%A_RES_l, data%A_RES_u,                            &
                        data%X_RES_l, data%X_RES_u,                            &
                        data%primal_vl_p, data%dual_vl_p, data%comp_vl_p,      &
                        exact_dual, inform%status, data%control%out )

     data%opt_measure_p = max( data%primal_vl_p, data%dual_vl_p, data%comp_vl_p)

     ! If computed, check optimality for SQP multipliers and select the best.
     !***********************************************************************

     data%best_mults = -1

     if ( data%sqp_computed ) then

        if ( data%seqp_computed ) then

           data%Y_seqp = zero
           data%Y_seqp( data%wJ( : data%nwJ) ) = data%QPseqp%Y( : data%nwJ )

           data%Ya_seqp = zero
           data%Ya_seqp( data%wA( : data%nwA) ) = data%QPseqp%Y( data%nwJ + 1 : data%nwJ + data%nwA )

           data%Z_seqp = zero
           data%Z_seqp( data%fx( : data%nfx) ) = &
                                data%QPseqp%Y( data%nwJ + data%nwA + 1 : data%nwJ + data%nwA + data%nfx )

           if ( nlp%m > 0 ) then
              data%JtY_s = zero
              call mop_Ax(one, nlp%J, data%Y_seqp, one, data%JtY_s, &
                          out, data%control%error, transpose=.true. )
           end if

           if ( nlp%m_a > 0 ) then
              data%AtYa_s = zero
              call mop_Ax( one, nlp%A, data%Ya_seqp, one, data%AtYa_s, &
                           out, data%control%error, transpose=.true.   )
           end if

           call check_optimal( nlp, nlp%G, data%JtY_s, data%AtYa_s,            &
                             data%Z_seqp, data%sqp_Zexact,                     &
                             data%Y_seqp, data%Ya_seqp,                        &
                             data%C_type, data%A_type, data%X_type,            &
                             data%C_RES_l, data%C_RES_u,                       &
                             data%A_RES_l, data%A_RES_u,                       &
                             data%X_RES_l, data%X_RES_u,                       &
                             data%primal_vl_s, data%dual_vl_s, data%comp_vl_s, &
                             exact_dual, inform%status, data%control%out )

        else ! siqp has been computed.

           if ( nlp%m > 0 ) then
              data%JtY_s = zero
              call mop_Ax(one, nlp%J, data%QPsiqp%Y( : m ), one, data%JtY_s, &
                          out, data%control%error, transpose=.true. )
           end if

           if ( nlp%m_a > 0 ) then
              data%AtYa_s = zero
              call mop_Ax(one, nlp%A, data%QPsiqp%Y( m+1:m+m_a ), one, &
                         data%AtYa_s, out, data%control%error, transpose=.true.)
           end if

           call check_optimal(nlp, nlp%G, data%JtY_s, data%AtYa_s,             &
                             data%QPsiqp%Z( : n), data%sqp_Zexact,             &
                             data%QPsiqp%Y( : m),  data%QPsiqp%Y(m+1:m+m_a),   &
                             data%C_type, data%A_type, data%X_type,            &
                             data%C_RES_l, data%C_RES_u,                       &
                             data%A_RES_l, data%A_RES_u,                       &
                             data%X_RES_l, data%X_RES_u,                       &
                             data%primal_vl_s, data%dual_vl_s, data%comp_vl_s, &
                             exact_dual, inform%status, data%control%out )

        end if

        data%opt_measure_s = max( data%primal_vl_s, data%dual_vl_s )
        data%opt_measure_s = max( data%opt_measure_s, data%comp_vl_s )

        dummy_real = min( data%opt_measure_cur, data%opt_measure_p )

        if ( data%opt_measure_s <= ten*dummy_real ) then ! DPR

           data%best_mults = 3
           data%mults_used = 'S'

           if ( data%seqp_computed ) then

              if ( nlp%m > 0 ) then
                 nlp%Y = data%Y_seqp
              end if
              if ( nlp%m_a > 0 ) then
                 nlp%Y_a = data%Ya_seqp
              end if
              if ( exact_dual ) then
                 nlp%Z = data%sqp_Zexact
              else
                 nlp%Z = data%Z_seqp
              end if

           else  ! siqp computed

              if ( nlp%m > 0 ) then
                 nlp%Y = data%QPsiqp%Y( 1 : nlp%m )
              end if
              if ( nlp%m_a > 0 ) then
                 nlp%Y_a = data%QPsiqp%Y( nlp%m + 1 : nlp%m + nlp%m_a )
              end if
              if ( exact_dual ) then
                 nlp%Z = data%sqp_Zexact
              else
                 nlp%Z = data%QPsiqp%Z( 1 : nlp%n )
              end if

           end if

           data%inf_norm_Y = data%inf_norm_Y_s

           data%primal_vl = data%primal_vl_s
           data%dual_vl   = data%dual_vl_s
           data%comp_vl   = data%comp_vl_s

           data%opt_measure = data%opt_measure_s

           ! need these when checking optimality of merit function.

           data%JtY = data%JtY_s ;     data%AtYa = data%AtYa_s

        end if

     end if

     if ( data%best_mults == -1 ) then  ! if sqp computed, mults were bad.

        if ( data%opt_measure_p <= ten * data%opt_measure_cur ) then

           data%best_mults = 2
           data%mults_used = 'P'

           if ( nlp%m > 0 ) then
              nlp%Y = data%QPpred%Y(: nlp%m)
           end if
           if ( nlp%m_a > 0 ) then
              nlp%Y_a = data%QPpred%Y( nlp%m + 1 : nlp%m + nlp%m_a )
           end if
           if ( exact_dual ) then
              nlp%Z = data%pred_Zexact
           else
              nlp%Z = data%QPpred%Z( : nlp%n)
           end if

           data%inf_norm_Y = data%inf_norm_Y_p

           data%primal_vl = data%primal_vl_p
           data%dual_vl   = data%dual_vl_p
           data%comp_vl   = data%comp_vl_p

           data%opt_measure = data%opt_measure_p

           ! need these when checking optimality of merit function.

           data%JtY = data%JtY_p ;     data%AtYa = data%AtYa_p

        else  ! keep the current mults.

           data%best_mults = 1
           data%mults_used = 'C'

           if ( exact_dual ) then
              nlp%Z = data%Zexact
           end if

           data%primal_vl = data%primal_vl_cur
           data%dual_vl   = data%dual_vl_cur
           data%comp_vl   = data%comp_vl_cur

           data%opt_measure = data%opt_measure_cur

           ! need these when checking optimality of merit function.

           data%JtY = data%JtY_cur ;     data%AtYa = data%AtYa_cur

        end if

     end if

700  continue

     ! Test for convergence and max iterations.
     ! *************************

     if ( data%iterate == max_iterate) then
        inform%status = GALAHAD_error_max_iterations
     end if

     if ( data%primal_vl <= data%control%stop_p ) then
        if ( data%dual_vl <= data%control%stop_d ) then
           if ( data%comp_vl <= data%control%stop_c ) then
              data%converged         = .true.
              data%merit_first_order = .true.   ! Assumes I coded correctly.
              inform%status          = GALAHAD_ok
           end if
        end if
     end if

     ! Print summary for all optimality stuff.
     ! ***************************************

     if ( data%control%print_level >= GALAHAD_DETAILS ) then
        call print_optimality_summary( data )
     end if

     ! If converged or max iterations reached, print main summary and exit.
     ! ********************************************************************

     if (data%converged .or. inform%status /= GALAHAD_ok) then
        goto 710
     end if

     ! Check first-order optimality of merit function.
     !************************************************

     data%change_penalty = '    same'

     data%check_suboptimal = .true.

     if ( data%check_suboptimal ) then

        call check_sub_optimal( nlp, nlp%Y, data%JtY, nlp%Y_a,  &
                                data%AtYa, nlp%Z,               &
                                data%C_RES_l, data%C_RES_u,     &
                                data%A_RES_l, data%A_RES_u,     &
                                data%X_RES_l, data%X_RES_u,     &
                                data, inform, out )

        data%merit_first_order  = .false.

        if ( data%sub_primal_vl <= data%sub_stop_p ) then
           if ( data%sub_dual_vl <= data%sub_stop_d ) then
              if ( data%sub_comp_vl <= data%sub_stop_c ) then
                 if ( data%sub_subgrad_vl <= teneps ) then

                    data%merit_first_order  = .true.

                    if ( data%NM%active ) then
                       data%NM%num_fail = 0
                       data%NM%active   = .false.
                    end if

                    if (data%norm_c <= max(data%eta, point9*stop_p) ) then
                       data%eta        = data%eta_contract * data%eta
                       data%sub_stop_p = max( half*data%eta, point9*stop_p )
                       data%sub_stop_d = data%sub_stop_p
                       data%sub_stop_c = data%sub_stop_p
                    else
                       data%sub_stop_p = max( half*data%sub_stop_p, point9*stop_p )
                       data%sub_stop_d = data%sub_stop_p
                       data%sub_stop_c = data%sub_stop_p
                       dummy_real      = data%control%penalty_expansion * data%penalty
                       data%penalty    = min( data%control%max_penalty, dummy_real )
                       data%merit      = nlp%f + data%penalty * data%norm_c
                       data%change_penalty = 'increase'
                    end if
                 end if
              end if
           end if
        end if

        ! Possibly print summary of sub-optimality check.

        if ( data%control%print_level >= GALAHAD_DETAILS ) then
           write(out, 1003) data%sub_stop_p, data%sub_stop_d, data%sub_stop_c, &
                            data%sub_primal_vl, data%sub_dual_vl,              &
                            data%sub_comp_vl, data%merit_first_order,          &
                            data%norm_c, data%eta, data%penalty, data%merit,   &
                            data%eta_contract
        end if

     end if

710  continue

     ! Print main summary line - optimality measures.
     !***********************************************

     if ( data%control%print_level >= GALAHAD_TRACE ) then
        if ( mod( data%iterate, data%control%header_every ) == 0 ) then
           write( out, 1000 )  ! header
        end if
        write( out, 1001 ) data%iterate, data%penalty, data%merit,     &
                           data%primal_vl, data%dual_vl, data%comp_vl, &
                           data%mults_used, data%merit_first_order
     end if

720  continue

     ! Exit if converged/max iterates, otherwise increase iterate and proceed.
     ! ***********************************************************************

     if ( data%converged .or. inform%status /= GALAHAD_ok ) then
        EXIT
     end if

     data%iterate = data%iterate + 1

     !write(*,*) 'Xl = ', nlp%X_l
     !write(*,*) 'X = ', nlp%X
     !write(*,*) 'Xu = ', nlp%X_u


     ! Decide if nonmonotone should be used.
     ! *************************************

     if ( data%control%NM_steps > 0 .and. (.not. data%NM%active) .and. m > 0 ) then
!        if ( data%primal_vl <= five*five*five ) then
        if ( data%primal_vl <= one ) then

           data%NM%active                = .true.
           data%revert%X_revert          = nlp%X
           data%revert%Y_revert          = nlp%Y
           data%revert%Y_a_revert        = nlp%Y_a
           data%revert%Z_revert          = nlp%Z
           data%revert%f_revert          = nlp%F
           data%revert%G_revert          = nlp%G
           data%revert%C_revert          = nlp%C
           data%revert%Jval_revert       = nlp%J%val
           data%revert%Ax_revert         = nlp%Ax
           data%revert%Bval_revert       = data%B%val
           data%revert%norm_c_revert     = data%norm_c
           data%revert%TRpred_revert     = data%TRpred
           data%revert%primal_vl_revert  = data%primal_vl
           data%revert%dual_vl_revert    = data%dual_vl
           data%revert%comp_vl_revert    = data%comp_vl
           data%revert%merit_revert      = data%merit
           data%revert%penalty_revert    = data%penalty
           data%revert%C_RES_l_revert    = data%C_RES_l
           data%revert%C_RES_u_revert    = data%C_RES_u
           data%revert%A_RES_l_revert    = data%A_RES_l
           data%revert%A_RES_u_revert    = data%A_RES_u
           data%revert%X_RES_l_revert    = data%X_RES_l
           data%revert%X_RES_u_revert    = data%X_RES_u

        end if
     end if

     ! If steering and monotone, then save penalty and merit value.
     ! ************************************************************

     if ( data%control%use_steering .and. m > 0 ) then
        if (.not. data%NM%active ) then
           data%penalty_pre_steer = data%penalty
           data%merit_pre_steer   = data%merit
        end if
     end if


     ! ***************************************!
     !  Starting computation of next iterate. !
     ! ***************************************!

     ! Possibly print the problem data.
     ! ********************************

     if ( data%control%print_level >= GALAHAD_CRAZY ) then
        call NLPT_write_problem( nlp, out, 1 )
     end if

     ! Compute the Predictor Step possibly using steering.
     ! ***************************************************

     ! Compute BFGS info if applicable.

     if ( data%control%B_type == 3 .and. data%success >= 1  ) then

        data%BFGS%gradLx     = data%G_prev
        data%BFGS%gradLx_new = nlp%G

        if ( nlp%m > 0 ) then
           data%Jval_dummy = nlp%J%val
           nlp%J%val       = data%Jval_prev
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%BFGS%gradLx, out, &
                         data%control%error, transpose=.true. )
           nlp%J%val = data%Jval_dummy
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%BFGS%gradLx_new, out, &
                         data%control%error, transpose=.true. )
        end if

     elseif ( data%control%B_type == 2 .and. data%success >= 1  ) then

        data%L_BFGS%gradLx     = data%G_prev
        data%L_BFGS%gradLx_new = nlp%G

        if ( nlp%m > 0 ) then
           data%Jval_dummy = nlp%J%val
           nlp%J%val       = data%Jval_prev
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%L_BFGS%gradLx, out, &
                         data%control%error, transpose=.true. )
           nlp%J%val = data%Jval_dummy
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%L_BFGS%gradLx_new, out, &
                         data%control%error, transpose=.true. )
        end if

     end if

     ! Fill predictor subproblem  -- QPpred.

     call fill_QPpred( nlp, data%QPpred, inform, data )
     if ( inform%status /= GALAHAD_ok ) then
        write(out,"('ERROR : trimsqp : subroutine fill_QPpred')")
        GO TO 999
     end if

     data%steering_good     = .false.
     data%computed_steering = .false.

     do while ( .not. data%steering_good )

        qpc_successful    = .false.
        data%QPpred_fails = 0

        do while ( .not. qpc_successful )
           !write(*,*) 'nlp%c = ', nlp%C
           !write(*,*) 'nlp%cl= ', nlp%C_l
           !write(*,*) 'nlp%cu = ', nlp%C_u
           !write(*,*) 'pred%x_l = ', data%QPpred%X_l
           !write(*,*) 'pred%x_u = ', data%QPpred%X_u

           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(out,*) ' =============== BEGIN : QPpred ================='
              call QPT_write_problem( out, data%QPpred, 1 )
              write(out,*) ' ================ END : QPpred ================='
           end if

           if ( data%iterate == 1 ) then
              data%control%QPpred_control%generate_sif_file = .true.
           end if

           data%u_in = data%QPpred%X(n+1:n+m)
           data%v_in = data%QPpred%X(n+m+1:n+2*m)

           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write( out, * ) 'QPpred%Xl = ', data%QPpred%X_l
              write( out, * ) 'QPpred%Xu = ', data%QPpred%X_u
              write( out, * ) 'QPpred%Cl = ', data%QPpred%C_l
              write( out, * ) 'QPpred%Cu = ', data%QPpred%C_u
              write( out, * ) 'Htype     = ', data%QPpred%H%type
              write( out, * ) 'Hrow      = ', data%QPpred%H%row
              write( out, * ) 'Hcol      =', data%QPpred%H%col
              write( out, * ) 'Hval      =', data%QPpred%H%val
              write( out, * ) 'Arow      = ', data%QPpred%A%row
              write( out, * ) 'Acol      = ', data%QPpred%A%col
              write( out, * ) 'Aval      = ', data%QPpred%A%val
              write( out, * ) 'Ane       = ', data%QPpred%A%ne
              write( out, * ) 'G         = ', data%QPpred%G
              write( out, * ) 'n         = ', data%QPpred%n
              write( out, * ) 'm         = ', data%QPpred%m
              write( out, * ) 'X         = ', data%QPpred%X
              write( out, * ) 'Y         = ', data%QPpred%Y
              write( out, * ) 'Z         = ', data%QPpred%Z
              write( out, * ) 'infinity  = ', data%control%QPpred_control%infinity
           end if

!           control%QPpred_control%QPB_control%muzero = 1.8e+14

           call QPC_solve( data%QPpred, data%QPpred%C_status,                 &
                           data%QPpred%X_status, data%QPpred_data,            &
                           data%control%QPpred_control, inform%QPpred_inform  )
           !write(*,*) 'return from QPC for predictor = ', inform%QPpred_inform%status
           !if ( data%iterate == 42 ) then
           !   return
           !end if
           ! The predictor step s_p and multiplier info.

           data%s_p          = data%QPpred%X( : nlp%n )
           data%inf_norm_s_p = MAXVAL( ABS(data%s_p) )

           if (nlp%m > 0 ) then
              data%inf_norm_Y_p = maxval( abs(data%QPpred%Y( : nlp%m)) )
              data%JxSp         = zero
              call mop_Ax( one, nlp%J, data%s_p, one, data%JxSp, &
                           out, control%error, transpose=.false. )
              data%CplusJxSp = nlp%C + data%JxSp
              call constraint_violation(nlp, data%CplusJxSp, data,                 &
                                        data%norm_c_linearize_pred, inform%status, &
                                        data%sat, data%vl_l, data%vl_u,            &
                                        data%num_sat, data%num_vl_l, data%num_vl_u )
           end if

           data%dec_norm_c_pred = data%norm_c - data%norm_c_linearize_pred

           ! Compute decrease in CONVEX model.
           data%BxSp = zero
           call mop_Ax( one, data%B, data%s_p, one, data%BxSp, &
                        out, control%error, symmetric=.true. )
           if ( data%control%B_type == 2 ) then
              do i = 1, nLMV
                 data%BxSp = data%BxSp - dot_product(data%s_p, data%L_BFGS%A( :, nLMV )) * data%L_BFGS%A( :, nLMV )
                 data%BxSp = data%BxSp + dot_product(data%s_p, data%L_BFGS%B( :, nLMV )) * data%L_BFGS%B( :, nLMV )
              end do
           end if
           data%Sp_B_Sp = DOT_PRODUCT( data%BxSp, data%s_p )  ! s_p^T B s_p

           data%decreaseB = data%penalty * data%dec_norm_c_pred
           data%decreaseB = data%decreaseB - DOT_PRODUCT( nlp%G, data%s_p )
           data%decreaseB = data%decreaseB - half * data%Sp_B_Sp

           ! Compute decrease in SMOOTH convex model.
           data%decreaseB_smooth = -DOT_PRODUCT(nlp%G,data%s_p) - half * data%Sp_B_Sp
           if ( nlp%m > 0 ) then
              dummy_real = zero
              do i = 1, m
                 dummy_real =  data%u_in(i) + data%v_in(i) - data%QPpred%X(n+i) - data%QPpred%X(n+m+i)
              end do
              data%decreaseB_smooth = data%decreaseB_smooth + data%penalty*dummy_real
           end if

           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(out, 3071 ) data%norm_c, data%norm_c_linearize_pred, &
                   data%dec_norm_c_pred, data%decreaseB
              write( out, * ) 'iterate   = ', data%iterate
              write( out, * ) 'S_p       = ', data%s_p
              write( out, * ) 'QPpred%X  = ', data%QPpred%X
              write( out, * ) 'JxSp      = ', data%JxSp
              write( out, * ) 'CplusJxSp = ', data%CplusJxSp
              write( out, * ) 'Y_p       = ', data%QPpred%Y( : m )
              write( out, * ) 'Ya_p      = ', data%QPpred%Y( m+1 : m+m_a )
              write( out, * ) 'Z_p       = ', data%QPpred%Z( : n )
              write( out, * ) 'X+S_p     = ', nlp%X + data%s_p
              write( out, * ) 'gts_p     = ', DOT_PRODUCT( nlp%G, data%s_p )
              write( out, * ) 'sp_B_sp   = ', data%Sp_B_sp
              write( out, * ) 'decreaseB =  ', data%decreaseB
              write( out, * ) 'decreaseBsmooth = ', data%decreaseB_smooth
           end if

           !if ( data%iterate == 1 ) then
           !   return
           !end if

           ! Do not need this....just doing it to have it.
           if ( nlp%m_a > 0 ) then
              data%AxSp = zero
              call mop_Ax( one, nlp%A, data%s_p, one, data%AxSp, &
                   out, control%error, transpose=.false. )
           end if

           if ( inform%QPpred_inform%status /= GALAHAD_OK .or. data%decreaseB < - tenm5 ) then
              write(*,*) 'WARNING : trimsqp : entered the --Catch-- bad predictor.'
              write(*,*) 'Predictor status = ', inform%QPpred_inform%status
              write(*,*) 'decreaseB        = ', data%decreaseB
              return
              data%QPpred_fails     = data%QPpred_fails + 1

              data%QPpred%X( 1:n )  = zero
              if ( m > 0 ) then
                 do i = 1, m
                    data%QPpred%X( n+i )   = max( zero, nlp%C_l(i) - nlp%C(i) )
                    data%QPpred%X( n+m+i ) = max( zero, nlp%C(i) - nlp%C_u(i) )
                 end do
              end if
              data%QPpred%Y( 1:m ) = zero !nlp%Y
              if ( m_a > 0 ) then
                 data%QPpred%Y( m+1:m+m_a ) = zero !nlp%Y_a
              end if

              if ( data%QPpred_fails == 1 ) then
                 data%control%QPpred_control%qpb_or_qpa = .true.
                 inform%QPpred_inform%status            = GALAHAD_OK
              elseif ( data%QPpred_fails == 2 ) then
                 data%control%QPpred_control%qpb_or_qpa = .false.
                 data%control%QPpred_control%no_qpb     = .true.
                 inform%QPpred_inform%status            = GALAHAD_OK
              else
                 data%control%QPpred_control%no_qpb = .false.
              end if
           else
              qpc_successful = .true.
           end if


           if ( inform%QPpred_inform%status /= GALAHAD_OK ) then
              write(out,*) ' QPpred : returned status ', inform%QPpred_inform%status
              GO TO 999
           end if

        end do

!!$        ! The predictor step s_p and multiplier info.
!!$
!!$        data%s_p          = data%QPpred%X( : nlp%n )
!!$        data%inf_norm_s_p = MAXVAL( ABS(data%s_p) )
!!$
!!$        if (nlp%m > 0 ) then
!!$           data%inf_norm_Y_p = maxval( abs(data%QPpred%Y( : nlp%m)) )
!!$           data%JxSp         = zero
!!$           call mop_Ax( one, nlp%J, data%s_p, one, data%JxSp, &
!!$                        out, control%error, transpose=.false. )
!!$           data%CplusJxSp = nlp%C + data%JxSp
!!$           call constraint_violation(nlp, data%CplusJxSp, data,                 &
!!$                                     data%norm_c_linearize_pred, inform%status, &
!!$                                     data%sat, data%vl_l, data%vl_u,            &
!!$                                     data%num_sat, data%num_vl_l, data%num_vl_u )
!!$
!!$        end if
!!$
!!$        data%dec_norm_c_pred = data%norm_c - data%norm_c_linearize_pred
!!$
!!$        ! Compute decrease in CONVEX model.
!!$        data%BxSp = zero
!!$        call mop_Ax( one, data%B, data%s_p, one, data%BxSp, &
!!$                     out, control%error, symmetric=.true. )
!!$        data%Sp_B_Sp = DOT_PRODUCT( data%BxSp, data%s_p )  ! s_p^T B s_p
!!$
!!$        data%decreaseB = data%penalty * data%dec_norm_c_pred
!!$        data%decreaseB = data%decreaseB - DOT_PRODUCT( nlp%G, data%s_p )
!!$        data%decreaseB = data%decreaseB - half * data%Sp_B_Sp
!!$
!!$        if ( data%control%print_level >= GALAHAD_DEBUG ) then
!!$           write(out, 3071 ) data%norm_c, data%norm_c_linearize_pred, &
!!$                             data%dec_norm_c_pred, data%decreaseB
!!$           write( out, * ) 'S_p       = ', data%s_p
!!$           write( out, * ) 'QPpred%X  = ', data%QPpred%X
!!$           write( out, * ) 'Y_p       = ', data%QPpred%Y( : m )
!!$           write( out, * ) 'Ya_p      = ', data%QPpred%Y( m+1 : m+m_a )
!!$           write( out, * ) 'Z_p       = ', data%QPpred%Z( : n )
!!$           write( out, * ) 'X+S_p     = ', nlp%X + data%s_p
!!$           write( out, * ) 'gts_p     = ', DOT_PRODUCT( nlp%G, data%s_p )
!!$           write( out, * ) 'sp_B_sp   = ', data%Sp_B_sp
!!$           write( out, * ) 'decreaseB =  ', data%decreaseB
!!$        end if
!!$

        ! Compute product of A with predictor step.
        if ( nlp%m_a > 0 ) then
           data%AxSp = zero
           call mop_Ax( one, nlp%A, data%s_p, one, data%AxSp, &
                        out, control%error, transpose=.false. )
        end if

        ! May not want to try steering.
        ! Note: if change the "if" part, then must change print_pred_steer.

        if ( m <= 0 .or. (.not. data%control%use_steering) &
                    .or. (data%NM%active .and. data%NM%num_fail > 0) ) then
           data%steering_good = .true.
           goto 711
        end if

        ! Possibly compute steering direction.

        if ( data%norm_c_linearize_pred < half * stop_p ) then
           data%steering_good = .true.
        else
           if ( .not. data%computed_steering ) then

              data%computed_steering = .true.

              call fill_QPsteer( nlp, data%QPsteer, inform, data%control, data )
              if ( inform%status /= GALAHAD_OK ) then
                 write(out,"('ERROR : trimsqp : subroutine fill_QPsteer')")
                 GO TO 999
              end if

              if ( data%control%print_level >= GALAHAD_DEBUG ) then
                 write(out,*) ' ============== BEGIN : QPsteer ================'
                 call QPT_write_problem( out, data%QPsteer, 1 )
                 write(out,*) ' =============== END : QPsteer ================='
              end if

              call QPC_solve(data%QPsteer, data%QPsteer%C_status,              &
                            data%QPsteer%X_status, data%QPsteer_data,          &
                            data%control%QPsteer_control, inform%QPsteer_inform)

              if ( inform%QPsteer_inform%status /= GALAHAD_ok ) then
                 write(out,*) ' QPsteer : returned status ', inform%QPsteer_inform%status
                 GO TO 999
              end if

              data%s_steer          = data%QPsteer%X( : nlp%n )
              data%inf_norm_s_steer = MAXVAL( ABS(data%s_steer) )

              data%inf_norm_Y_steer = maxval( abs(data%QPsteer%Y( : nlp%m)) )

              data%JxSsteer = zero
              call mop_Ax( one, nlp%J, data%s_steer, one, data%JxSsteer, &
                           out, control%error, transpose=.false. )
              call constraint_violation( nlp, nlp%C+data%JxSsteer, data,     &
                                         data%norm_c_linearize_steer,        &
                                         inform%status, data%sat, data%vl_l, &
                                         data%vl_u, data%num_sat,            &
                                         data%num_vl_l, data%num_vl_u)

              data%dec_norm_c_steer = data%norm_c - data%norm_c_linearize_steer

           end if

!           if ( data%norm_c_linearize_steer < min(tenm9, tenm3*stop_p) ) then
!           if ( data%norm_c_linearize_steer < tenm10 ) then
!              data%steering_good = .false.
!              data%penalty = min(data%control%penalty_expansion*data%penalty, data%control%max_penalty )
!           elseif ( data%dec_norm_c_pred >= data%steer_L_factor * data%dec_norm_c_steer ) then
           if ( data%dec_norm_c_pred >= data%steer_L_factor * data%dec_norm_c_steer ) then
              !if ( .true. ) then
                 data%steering_good = .true.
              !else
              !   data%penalty = min( data%control%penalty_expansion*data%penalty, data%control%max_penalty )
              !end if
           else
              data%steering_good = .false.
              !data%penalty = min( data%control%penalty_expansion*data%penalty, data%control%max_penalty )
           end if

        end if

        if ( data%steering_good ) then
           if ( data%decreaseB >= data%steer_Q_factor * data%penalty * data%dec_norm_c_pred  .or. &
                abs(data%decreaseB) <= tenm6 ) then
              ! relax...steering is already good.
           else
              data%steering_good = .false.
           end if
        end if

711     continue

        if (data%control%print_level >= GALAHAD_DETAILS ) then
           call print_pred_steer( data )  ! Details on predictor/steering.
        end if

        ! Possibly redefine data for predictor problem.

        if ( .not. data%steering_good ) then

           data%penalty = data%control%penalty_expansion*data%penalty

           data%QPpred%X( 1:n )  = zero
           if ( m > 0 ) then
              do i = 1, m
                 data%QPpred%X( n+i )   = max( zero, nlp%C_l(i) - nlp%C(i) )
                 data%QPpred%X( n+m+i ) = max( zero, nlp%C(i) - nlp%C_u(i) )
              end do
           end if

           data%QPpred%G( n+1:n+2*m )  = data%penalty

           data%QPpred%Y( 1:m ) = nlp%Y
           if ( m_a > 0 ) then
              data%QPpred%Y( m+1:m+m_a ) = nlp%Y_a
           end if
           data%QPpred%Z        = zero
           data%QPpred%Z( 1:n ) = nlp%Z

        end if

        if ( data%penalty > data%control%max_penalty ) then
           inform%status = -40 ! max penalty reached
           go to 720
        end if

     end do

     data%iterates_pred = inform%QPpred_inform%QPA_inform%major_iter
     data%iterates_pred = data%iterates_pred + inform%QPpred_inform%QPB_inform%iter

!!$     ! Compute decrease in CONVEX model.
!!$     ! *********************************
!!$
!!$     data%BxSp = zero
!!$     call mop_Ax( one, data%B, data%s_p, one, data%BxSp, &
!!$                  out, control%error, symmetric=.true. )
!!$     data%Sp_B_Sp = DOT_PRODUCT( data%BxSp, data%s_p )  ! s_p^T B s_p
!!$
!!$     data%decreaseB = data%penalty * data%dec_norm_c_pred
!!$     data%decreaseB = data%decreaseB - DOT_PRODUCT( nlp%G, data%s_p )
!!$     data%decreaseB = data%decreaseB - half * data%Sp_B_Sp
!!$
!!$     if ( data%control%print_level >= GALAHAD_DEBUG ) then
!!$        write(out, 3071 ) data%norm_c, data%norm_c_linearize_pred, &
!!$                          data%dec_norm_c_pred, data%decreaseB
!!$        if ( data%control%print_level >= GALAHAD_CRAZY ) then
!!$           write( out, * ) 'S_p   = ', data%s_p
!!$           write( out, * ) 'Y_p   = ', data%QPpred%Y( : m )
!!$           write( out, * ) 'Ya_p   = ', data%QPpred%Y( m+1 : m+m_a )
!!$           write( out, * ) 'Z_p   = ', data%QPpred%Z( : n )
!!$           write( out, * ) 'X+S_p = ', nlp%X + data%s_p
!!$        end if
!!$     end if

     ! Compute merit function.

     data%merit = nlp%f + data%penalty*data%norm_c

     ! Evaluate Hessian now with (possibly) new multiplier information
     !****************************************************************

     if ( m > 0 ) then
        data%JtY_p = zero
        call mop_Ax( one, nlp%J, data%QPpred%Y( : m), one, data%JtY_p,   &
                     out, data%control%error, transpose=.true. )
     end if
     if ( m_a > 0 ) then
        data%AtYa_p = zero
        call mop_Ax( one, nlp%A, data%QPpred%Y(m+1 : m+m_a), one, data%AtYa_p, &
                     out, data%control%error, transpose=.true. )
     end if

     ! Check if predictor multipliers are better than current multipliers.

     dummy_real = zero

     do i = 1, nlp%n

        Gi = nlp%G( i )
        vl = Gi

        if ( nlp%m > 0 ) then
           vl = vl - data%JtY_p( i )
        end if
        if ( nlp%m_a > 0 ) then
           vl = vl - data%AtYa_p( i )
        end if

        vl = vl - data%QPpred%Z( i )
        vl = abs(vl) / ( one + abs( Gi ) )

        dummy_real = max( dummy_real, vl )

     end do

     if ( dummy_real < data%dual_vl ) then
        call eval_HL( inform%status, nlp%X, data%QPpred%Y(:m), userdata,       &
                      nlp%H%val )
        if ( inform%status /= GALAHAD_OK ) write(out,1002) 'eval_HL'
        inform%num_H_eval = inform%num_H_eval + 1

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write( data%control%out, * ) ' -evaluted Hessian at (x,y_p).'
           call print_SMT( nlp%H, 'H', data%control%error, out, inform%status )
        end if
     else
        call eval_HL( inform%status, nlp%X, nlp%Y, userdata, nlp%H%val )
        if ( inform%status /= GALAHAD_OK ) write(out,1002) 'eval_HL'
        inform%num_H_eval = inform%num_H_eval + 1

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write( data%control%out, * ) ' -evaluted Hessian at current (x,y).'
           call print_SMT( nlp%H, 'H', data%control%error, out, inform%status )
        end if
     end if

     ! Solve for Cauchy step : s_c
     !****************************

     ! compute needed data for subroutine get_cauchy_step

     data%two_norm_s_p = NRM2( nlp%n, data%s_p, 1 )
     data%gts_pred = DOT_PRODUCT( data%s_p, nlp%G )

     data%Sp_H_Sp = zero
     data%HxSp    = zero
     call mop_Ax( one, nlp%H, data%s_p, one, data%HxSp,  &
                  out, control%error, symmetric = .TRUE. )
     data%Sp_H_Sp = DOT_PRODUCT( data%s_p, data%HxSp )

     ! Compute decrease in FAITHFUL model at the predictor step (SEQP step might need this).

     data%decreaseH_pred = data%penalty * data%dec_norm_c_pred
     data%decreaseH_pred = data%decreaseH_pred - DOT_PRODUCT( nlp%G, data%s_p )
     data%decreaseH_pred = data%decreaseH_pred - half * data%Sp_H_Sp

     if ( data%control%print_level >= GALAHAD_DEBUG ) then
        write( data%control%out, 3065 ) data%two_norm_s_p, data%gts_pred, data%Sp_H_Sp
     end if

     ! Call subroutine get_cauchy_step if needed.

     if ( data%inf_norm_s_p < tenm7 ) then

        data%alpha_c = one
        if ( data%control%print_level >= GALAHAD_DEBUG ) write( out, 3082 )

     elseif ( data%Sp_B_Sp >= data%Sp_H_Sp ) then

        data%alpha_c = one
        if ( data%control%print_level >= GALAHAD_DEBUG ) write( out, 3081 )

     else

        if ( nlp%m > 0 ) then
           call mop_row_2_norms(nlp%J, data%J_norms, symmetric=.false.,        &
                                out=data%control%out, error=data%control%error,&
                                print_level=0 )
        end if

        !too_small = epsmch ** 0.5
        too_small = epsmch ** point8

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           print_1line  = .true. ; print_detail = .true. ; print_debug  = .true.
           write(out,*) 'C    = ', nlp%C
           write(out,*) 'Js_p = ', data%JxSp
        else
           print_1line  = .false. ; print_detail = .false. ; print_debug  = .false.
        end if

        call get_cauchy_step( nlp%m, data%C_type,                             &
                        zero, data%gts_pred, data%Sp_H_Sp, data%two_norm_s_p, &
                        data%penalty, data%C_RES_l, data%C_RES_u, data%JxSp,  &
                        data%J_norms, data%lbreak, data%IBREAK,               &
                        data%BREAKP, .true., data%control%out, print_1line,   &
                        print_detail, print_debug, t_min=data%alpha_c,        &
                        too_small=too_small, inform=inform%status )
        if ( inform%status == -4 ) then
           write(*,*) 'Cauchy step returned status = -4 '
           exit
        end if

     end if

     ! the Cauchy step s_c.

     data%alpha_c      = min( one, data%alpha_c )  ! DPR: this should always be true.
     data%s_c          = data%alpha_c * data%s_p
     data%inf_norm_s_c = data%alpha_c * data%inf_norm_s_p

     if ( data%control%print_level >= GALAHAD_DEBUG ) then
        if ( data%control%print_level >= GALAHAD_CRAZY ) then
           write( out, * ) 'S_c   = ', data%s_c
           write( out, * ) 'X+S_c = ', nlp%X + data%s_c
        end if
        write( out, * )  ' ending Cauchy subproblem .....'
        write(out,*) ' ....start finding model decrease at Cauchy point'
     end if

     ! Compute change in "faithful" model function.
     !*********************************************

     if ( nlp%m > 0 ) then
        data%JxSc = zero
        call mop_Ax( one, nlp%J, data%s_c, one, data%JxSc, &
                     out, control%error, transpose=.false. )
        data%CplusJSc = nlp%C + data%JxSc
        call constraint_violation(nlp, data%CplusJSc, data,                    &
                                  data%norm_c_linearize_cauchy, inform%status, &
                                  data%sat, data%vl_l, data%vl_u,              &
                                  data%num_sat, data%num_vl_l, data%num_vl_u )
     end if

     data%HxSc = zero
     call mop_Ax( one, nlp%H, data%s_c, one, data%HxSc, &
                  out, control%error, symmetric=.true.  )

     data%Sc_H_Sc = DOT_PRODUCT( data%HxSc, data%s_c )  ! s_c^T H s_c

     data%decreaseH_cauchy = data%penalty * ( data%norm_c - data%norm_c_linearize_cauchy )
     data%decreaseH_cauchy = data%decreaseH_cauchy - DOT_PRODUCT( nlp%G, data%s_c )
     data%decreaseH_cauchy = data%decreaseH_cauchy - half * data%Sc_H_Sc

     if ( data%control%print_level >= GALAHAD_DEBUG ) then
        write(out, 3067) data%norm_c, data%norm_c_linearize_cauchy,                      &
                         data%decreaseH_cauchy, data%num_sat, data%num_vl_l, data%num_vl_u
        write(out,'(3(3x, I7, 13x) )') (data%sat(i), data%vl_l(i), data%vl_u(i), i = 1, m)
     end if

     ! Possibly solve for SQP step s_s.
     !*********************************

     ! Determine if we want to compute an SQP step.

     if ( data%NM%active ) then
        data%sqp_computed = .true.
     else
        if ( data%opt_measure <= point5 .or. data%inf_norm_s_f <= point1 ) then
           data%sqp_computed = .true.
        else
           data%sqp_computed = .true.
           !data%sqp_computed = .false.
        end if
     end if

     data%seqp_computed = .false.

     if (data%sqp_computed .and. data%norm_c_linearize_pred < tenm7 .and. data%control%use_seqp) then

        ! Compute SEQP step.
        ! *****************

        data%seqp_computed = .true.

        data%TRsqp = max( data%control%TRsqp_scale * data%TRpred, data%min_TRsqp )

        ! Compute approximate Cauchy point.
        ! *********************************

        if ( data%decreaseH_pred > data%ac_factor * data%decreaseH_cauchy ) then
           data%s_ac = data%s_p
           data%AxSac = data%AxSp
           data%JxSac = data%JxSp
           data%seqp_use_pred = .true.
        else
           data%s_ac  = data%s_c
           data%AxSac = data%alpha_c * data%AxSp
           data%JxSac = data%alpha_c * data%JxSp
           data%seqp_use_pred = .false.
        end if

        call fill_QPseqp( nlp, data%QPseqp, inform, data )
        if ( inform%status /= GALAHAD_OK ) then
           write(out,"(' ERROR : trimsqp : subroutine fill_QPseqp')")
           GO TO 999
        end if

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write(out,*) ' ============== BEGIN : QPseqp ================'
           call QPT_write_problem( out, data%QPseqp, 1 )
           write(out,*) ' =============== END : QPseqp ================='
        end if

       !  write(*,*) 'C = ', data%QPseqp%C
!         write(*,*) 'G = ', data%QPseqp%G
!         write(*,*) 'Hrow =  ', data%QPseqp%H%row
!         write(*,*) 'Hcol =', data%QPseqp%H%col
!         write(*,*) 'Hval =', data%QPseqp%H%val
!         write(*,*) 'radius = ', data%control%QPseqp_control%radius
!         write(*,*) 'Arow =  ', data%QPseqp%A%row
!         write(*,*) 'Acol =', data%QPseqp%A%col
!         write(*,*) 'Aval =', data%QPseqp%A%val
!         write(*,*) 'A_m =', data%QPseqp%A%m
!         write(*,*) 'A_n =', data%QPseqp%A%n

        call EQP_solve( data%QPseqp, data%QPseqp_data, data%control%QPseqp_control, inform%QPseqp_inform )

        if ( inform%QPseqp_inform%status /= GALAHAD_OK ) then
           write(out,*) ' TRIMSQP : qp_seqp status = ', inform%QPseqp_inform%status
           go to 999
        end if

        data%iterates_sqp = inform%QPseqp_inform%cg_iter   ! Are these really iterations? ???? DPR

        ! The FULL length seqp step and multipliers.

        data%s_s          = data%QPseqp%X
        data%inf_norm_s_s = MAXVAL( ABS(data%s_s) )

        data%Y_seqp = zero
        data%Y_seqp( data%wJ( : data%nwJ) ) = data%QPseqp%Y( : data%nwJ )

        data%Ya_seqp = zero
        data%Ya_seqp( data%wA( : data%nwA) ) = data%QPseqp%Y( data%nwJ + 1 : data%nwJ + data%nwA )

        data%Z_seqp = zero
        data%Z_seqp( data%fx( : data%nfx) ) = data%QPseqp%Y( data%nwJ + data%nwA + 1 : data%nwJ + data%nwA + data%nfx )

        if (nlp%m > 0 ) then
           data%inf_norm_Y_s = maxval( abs(data%QPseqp%Y( : data%nwJ )) )
        end if

        ! Get max step ensuring feasibility for inactive/free constraints/variables.

        if ( nlp%m > 0 ) then
           data%JxSs = zero
           call mop_Ax( one, nlp%J, data%s_s, one, data%JxSs, &
                        out, control%error, transpose=.false. )
        end if

        if ( nlp%m_a > 0 ) then
           data%AxSs = zero
           call mop_Ax( one, nlp%A, data%s_s, one, data%AxSs, &
                        out, data%control%error, transpose=.false. )
        end if

        ! Do not really need this, but compute it anyways.
        data%HxSs = zero
!        call mop_Ax( one, data%QPseqp%H, data%s_s, one, data%HxSs, &
        call mop_Ax( one, nlp%H, data%s_s, one, data%HxSs, &
                     out, data%control%error, symmetric=.true. )
        data%Ss_H_Ss = DOT_PRODUCT( data%HxSs, data%s_s )

        data%alpha_feas = max_feas_step( nlp, data, data%X_type, data%A_type, data%C_type )

        ! The full step s_f.

        !write(*,*) 'seqp_use_pred = ', data%seqp_use_pred
        !write(*,*) 'data%alpha_feas = ', data%alpha_feas

        !if ( data%seqp_use_pred ) then
        !   data%s_f = data%s_p + data%alpha_feas * data%s_s
           !write(*,*) 'first'
        !else
           data%s_f = data%s_ac + data%alpha_feas * data%s_s
           !write(*,*) 'second'
        !end if
        data%inf_norm_s_f = MAXVAL( ABS(data%s_f) )

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write( out, * ) 'iterate        = ', data%iterate
           write( out, * ) 'S_s            = ', data%s_s
           write( out, * ) 'num_free       = ', data%nfr
           write( out, * ) 'fr             = ', data%fr
           write( out, * ) 'num_fx         = ', data%nfx
           write( out, * ) 'fx             = ', data%fx
           write( out, * ) 'num_wJ         = ', data%nwJ
           write( out, * ) 'wJ             = ', data%wJ
           write( out, * ) 'num_wJ_comp    = ', data%nwJ_comp
           write( out, * ) 'wJ_comp        = ', data%wJ_comp
           write( out, * ) 'num_wA         = ', data%nwA
           write( out, * ) 'wA             = ', data%wA
           write( out, * ) 'num_wA_comp    = ', data%nwA_comp
           write( out, * ) 'wA_comp        = ', data%wA_comp
           write( out, * ) 'QPseqp%X       = ', data%QPseqp%X
           write( out, * ) 'Y_s            = ', data%Y_seqp
           write( out, * ) 'Ya_s           = ', data%Ya_seqp
           write( out, * ) 'Z_s            = ', data%Z_seqp
           write( out, * ) 'G              = ', nlp%G
           write( out, * ) 'decreaseH_pred = ', data%decreaseH_pred
           write( out, * ) 'decreaseH_cauc = ', data%decreaseH_cauchy
           write( out, * ) 'alpha_feas     = ', data%alpha_feas
           if ( data%seqp_use_pred ) then
              write( out, * ) 'HxS_p          = ', data%HxSp
              write( out, * ) 'X+S_p+alpha*S_s      = ', nlp%X + data%s_p + data%alpha_feas*data%s_s
              write( out, * ) 'g+HxSp         = ', nlp%G + data%HxSp
              dummy_real = DOT_PRODUCT( nlp%G + data%HxSp, data%s_s )
              write( out, * ) '(g+HxSp)^T s_s = ', dummy_real
           else
              write( out, * ) 'HxS_ac          = ', data%HxSac
              write( out, * ) 'X+S_ac+alpha*S_s      = ', nlp%X + data%s_ac + data%alpha_feas*data%s_s
              write( out, * ) 'g+HxSac         = ', nlp%G + data%HxSac
              dummy_real = DOT_PRODUCT( nlp%G + data%HxSac, data%s_s )
              write( out, * ) '(g+HxSac)^T s_s = ', dummy_real
           end if
           write( out, * ) 'HxSs           = ', data%HxSs
           write( out, * ) 'Ss_H_Ss        = ', data%Ss_H_Ss
           write( out, * ) 'QPseqp_obj     = ', dummy_real + half*data%Ss_H_Ss
           write( out, * ) 'QPseqp%G       = ', data%QPseqp%G
           write( out, * ) 'QPseqp%H%val   = ', data%QPseqp%H%val
           write( out, * ) 'JxSs           = ', data%JxSs
           write( out, * )' .......starting full step info'
        end if
!!$        if ( data%control%print_level >= GALAHAD_DEBUG ) then
!!$              write(out, 3071 ) data%norm_c, data%norm_c_linearize_pred, &
!!$                   data%dec_norm_c_pred, data%decreaseB
!!$              write( out, * ) 'iterate   = ', data%iterate
!!$              write( out, * ) 'S_p       = ', data%s_p
!!$              write( out, * ) 'QPpred%X  = ', data%QPpred%X
!!$              write( out, * ) 'JxSp      = ', data%JxSp
!!$              write( out, * ) 'CplusJxSp = ', data%CplusJxSp
!!$              write( out, * ) 'Y_p       = ', data%QPpred%Y( : m )
!!$              write( out, * ) 'Ya_p      = ', data%QPpred%Y( m+1 : m+m_a )
!!$              write( out, * ) 'Z_p       = ', data%QPpred%Z( : n )
!!$              write( out, * ) 'X+S_p     = ', nlp%X + data%s_p
!!$              write( out, * ) 'gts_p     = ', DOT_PRODUCT( nlp%G, data%s_p )
!!$              write( out, * ) 'sp_B_sp   = ', data%Sp_B_sp
!!$              write( out, * ) 'decreaseB =  ', data%decreaseB
!!$           end if
        data%descent_constraint_status = '  NA'

     elseif ( data%sqp_computed .and. data%control%use_siqp ) then

        data%alpha_feas = -one

        data%TRsqp = max( data%control%TRsqp_scale * data%TRpred, data%min_TRsqp )

        ! Get A*s_c

        if ( nlp%m_a > 0 ) then
           data%AxSc = zero
           call mop_Ax( one, nlp%A, data%s_c, one, data%AxSc, &
                        out, data%control%error, transpose=.false. )
           data%AXplusSc = nlp%Ax + data%AxSc
        end if

        ! Compute the descent constraint for the SQP subproblem.

        data%GplusHs     = nlp%G + data%HxSc
        data%descent_con = data%GplusHs

        if ( nlp%m > 0 ) then

           data%w = zero
           data%w( data%vl_l( : data%num_vl_l ) ) = - one
           data%w( data%vl_u( : data%num_vl_u ) ) =   one

           call mop_Ax( data%penalty, nlp%J, data%w, one, data%descent_con, &
                        out, data%control%error, transpose = .true.         )

        end if

        ! Compute quantities that depend on the correction used.

        if ( data%control%correction_type == 0 ) then
           ! relax
        else
           write(out,*) 'TRIMSQP : SQP corrections not yet implemented'
           GO TO 999
        end if

        ! Decide whether only want active set phase.

        if ( data%primal_vl <= sqrt(data%control%stop_p) .and. &
             data%dual_vl   <= sqrt(data%control%stop_d) .and. &
             data%comp_vl   <= sqrt(data%control%stop_c) ) then

           data%control%QPsiqp_control%no_qpb = .false.

        else

           data%control%QPsiqp_control%no_qpb = .false.

        end if

        qpc_successful = .FALSE.

        do while ( .not. qpc_successful )

           call fill_QPsiqp( nlp, data%QPsiqp, inform, data )
           if ( inform%status /= GALAHAD_ok ) then
              write(out,"(' ERROR : trimsqp : subroutine fill_QPsiqp')")
              GO TO 999
           end if

           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(out,*) ' ============== BEGIN : QPsiqp ================'
              call QPT_write_problem( out, data%QPsiqp, 1 )
              write(out,*) ' =============== END : QPsiqp ================='
              write(out, 3078 ) data%GplusHs
              write(out, 3079 ) data%descent_con
              write(out, 3080 ) dot_product(data%descent_con, data%QPsiqp%X(:n))
           end if

           ! solve for the SIQP step.

           call QPC_solve( data%QPsiqp, data%QPsiqp%C_status,                &
                           data%QPsiqp%X_status, data%QPsiqp_data,           &
                           data%control%QPsiqp_control, inform%QPsiqp_inform )

           if ( inform%QPsiqp_inform%status /= GALAHAD_OK ) then
              write(out,*) 'TRIMSQP : QPsiqp status = ', inform%QPsiqp_inform%status
              GO TO 999
           end if

           qpc_successful    = .true.
           data%iterates_sqp = inform%QPsiqp_inform%QPA_inform%major_iter
           data%iterates_sqp = data%iterates_sqp + inform%QPsiqp_inform%QPB_inform%iter

        end do

        ! Determine "activity" of the descent constraint.

        if ( data%QPsiqp%C_status( m + m_a + 1 ) == 0 ) then
           data%descent_constraint_status = '  FR'
        elseif ( data%QPsiqp%C_status( m + m_a + 1 ) > 0 ) then
           data%descent_constraint_status = 'FX-U'
           inform%num_descent_active = inform%num_descent_active + 1
        else
           data%descent_constraint_status = 'FX-L'  ! Should not happen.
           inform%num_descent_active = inform%num_descent_active + 1
        end if

        ! Based on descent-constraint activity, adjust the multipliers.

        if ( data%descent_constraint_status == '  FR' ) then
           ! relax
        elseif ( data%descent_constraint_status == 'FX-L' ) then
           write( out, * ) ' ERROR : descent_constraint lower fixed'
           GO TO 999
        else

           ! The multiplier for the descent constraint.

           dummy_real   = data%QPsiqp%Y( m + m_a + 1 )

           ! J multipliers

           data%QPsiqp%Y( data%vl_l(:data%num_vl_l) ) = data%QPsiqp%Y( data%vl_l(:data%num_vl_l) ) - data%penalty * dummy_real
           data%QPsiqp%Y( data%vl_u(:data%num_vl_u) ) = data%QPsiqp%Y( data%vl_u(:data%num_vl_u) ) + data%penalty * dummy_real
           data%QPsiqp%Y( : nlp%m )                   = data%QPsiqp%Y( : nlp%m ) / ( one - dummy_real )

           ! A and Z multipliers

           data%QPsiqp%Y(m+1:m+m_a) =  data%QPsiqp%Y(m+1:m+m_a) / (one - dummy_real)
           data%QPsiqp%Z            = data%QPsiqp%Z / (one - dummy_real)

        end if

        ! Store the SQP step, its infinity norm, and the infinity norm of Y_sqp.

        data%s_s          = data%QPsiqp%X( 1 : nlp%n )
        data%inf_norm_s_s = MAXVAL( ABS(data%s_s) )

        if (nlp%m > 0 ) then
           data%inf_norm_Y_s = maxval( abs(data%QPsiqp%Y(:m)) )
        end if

        ! The full step  : s_f = s_c + s_s.

        data%s_f          = data%s_c + data%s_s
        data%inf_norm_s_f = MAXVAL( ABS(data%s_f) )

        if ( data%control%print_level >= GALAHAD_CRAZY ) then
           write( out, * ) 'S_s   = ', data%s_s
           write( out, * ) 'Y_s   = ', data%QPsiqp%Y( : m )
           write( out, * ) 'Ya_s   = ', data%QPsiqp%Y( m+1 : m+m_a )
           write( out, * ) 'Z_s   = ', data%QPsiqp%Z( : n )
           write( out, * )' .......starting full step info'
        end if

     end if

     ! Compute full step info.

     if ( data%sqp_computed ) then

        if ( nlp%m > 0 ) then
           data%JxSf = zero
           call mop_Ax( one, nlp%J, data%s_f, one, data%JxSf, &
                        out, control%error, transpose=.false. )
           call constraint_violation(nlp, nlp%C + data%JxSf, data,             &
                                     data%norm_c_linearize_full, inform%status )
        end if

        data%HxSf = zero
        call mop_Ax( one, nlp%H, data%s_f, one, data%HxSf, &
                     out, control%error, symmetric=.true.  )

        data%Sf_H_Sf = DOT_PRODUCT( data%HxSf, data%s_f )  ! s_f^T H s_f

        data%decreaseH_full = data%penalty * (data%norm_c - data%norm_c_linearize_full)
        data%decreaseH_full = data%decreaseH_full - DOT_PRODUCT(nlp%G, data%s_f)
        data%decreaseH_full = data%decreaseH_full - half * data%Sf_H_sf

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write(out, 3068 ) data%norm_c, data%norm_c_linearize_full
           if ( data%control%print_level >= GALAHAD_CRAZY ) then
              write( out, * ) 'S_f   = ', data%s_f
              write( out, * ) 'inf_norm_s_f = ', data%inf_norm_s_f
              write( out, * ) 'X+S_f = ', nlp%X + data%s_f
           end if
        end if

        if (data%decreaseH_full >= point1 * data%decreaseH_cauchy) then
           data%sqp_good_dec   = .true.
           data%sqp_ratio_used = .true.
        else
           data%sqp_good_dec   = .false.
           if ( data%NM%active .and. data%NM%num_fail /= 0 ) then
              data%sqp_ratio_used = .true.
           else
              data%sqp_ratio_used = .false.
           end if
        end if

     else  ! sqp not computed.

        if ( data%control%print_level >= GALAHAD_DETAILS ) then
           write( out, * )' -- skipping computation of SQP step'
        end if

        data%iterates_sqp   = 0
        data%s_s            = zero
        data%inf_norm_s_s   = zero
        data%inf_norm_Y_s   = zero
        data%s_f            = data%s_c
        data%inf_norm_s_f   = data%inf_norm_s_c
        data%decreaseH_full = data%decreaseH_cauchy

        data%descent_constraint_status = '  NA'

        data%sqp_good_dec   = .false.
        data%sqp_ratio_used = .false.
        data%alpha_feas     = -one

     end if

     ! Evaluate functions and compute ratio
     ! ************************************

     if ( data%sqp_ratio_used ) then

        ! Evaluate functions and constraint violation at new point.

        call eval_FC( inform%status, nlp%X + data%s_f, userdata,               &
                      data%F_new, data%C_new )
                      
        if (inform%status /= GALAHAD_ok) write(out,1002) 'eval_FC'

        inform%num_f_eval = inform%num_f_eval + 1

        if ( nlp%m > 0 ) then

           inform%num_c_eval = inform%num_c_eval + 1  ! From above.

           call constraint_violation( nlp, data%C_new, data,         &
                                      data%norm_c_new, inform%status )

        end if

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write(out, 3070 ) data%F_new, data%norm_c_new,  &
                             data%decreaseH_cauchy, data%decreaseH_full
        end if

        data%merit_new = data%f_new + data%penalty * data%norm_c_new

       ! Compute ratio of actual to predicted.

        if ( data%NM%active ) then
           if ( data%NM%num_fail == 0 ) then
              data%NM%decreaseH  = data%decreaseH_full
              data%NM%merit_best = data%merit ! this may be different than data%revert%merit_revert. B/c penalt changes?
           end if
           data%ratio = data%NM%merit_best - data%merit_new
           data%ratio = data%ratio / data%NM%decreaseH
           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(out, *) 'NM%num_fail   = ', data%NM%num_fail
              write(out, *) 'NM%merit_best = ', data%NM%merit_best
              write(out, *) 'NM%decreaseH  = ', data%NM%decreaseH
           end if
        else
           data%ratio =  ( data%merit - data%merit_new ) / data%decreaseH_full
           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(out, *) 'decreaseH_full  = ', data%decreaseH_full
           end if
        end if

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write(out, *) 'merit      = ', data%merit
           write(out, *) 'merit_new  = ', data%merit_new
        end if

     else  ! sqp_ratio not used.

        ! Evaluate functions and constraint violatoin at new point.

        call eval_FC( inform%status, nlp%X + data%s_c, userdata,               &
                      data%F_new, data%C_new )
                      
        if (inform%status /= GALAHAD_ok) write(out,1002) 'eval_FC'

        inform%num_f_eval = inform%num_f_eval + 1

        if ( nlp%m > 0 ) then

           inform%num_c_eval = inform%num_c_eval + 1 ! From above

           call constraint_violation( nlp, data%C_new, data,         &
                                      data%norm_c_new, inform%status )

        end if

        data%merit_new = data%f_new + data%penalty * data%norm_c_new

        ! Compute ratio of actual to predicted.

        ! Either ~sqp_computed -> NM%active = .false.
        !                     or
        ! sqp_computed, which must imply NM%num_fail = 0 to be in this case.
        ! In either case we have the following definition for the ratio.

        if ( data%NM%active ) then
           if ( data%control%print_level >= GALAHAD_DEBUG ) then
              write(data%control%out, 3016 ) ! Saving for possible reversion
           end if
           data%NM%decreaseH  = data%decreaseH_cauchy
           data%NM%merit_best = data%merit  ! may be different than merit_revert
        end if

        data%ratio =  ( data%merit - data%merit_new ) / data%decreaseH_cauchy

        if ( data%control%print_level >= GALAHAD_DEBUG ) then
           write(out, *) 'merit     = ', data%merit
           write(out, *) 'merit_new = ', data%merit_new
           write(out,3070) data%f_new, data%norm_c_new,  &
                           data%decreaseH_cauchy, data%decreaseH_full
        end if

     end if

!!$     write(*,*) 'ratio = ', data%ratio
!!$     write(*,*) 'success = ', data%control%eta_successful
!!$     write(*,*) 'very = ', data%control%eta_very_successful
!!$     write(*,*) 'extrem = ', data%control%eta_extremely_successful


     ! Determine level of success.
     ! ***************************

     if ( data%ratio >= data%control%eta_extremely_successful .and. &
          data%ratio <= two-data%control%eta_extremely_successful ) then
        data%success_str = 'extreme'
        data%success     = 3
     elseif ( data%ratio >= data%control%eta_very_successful .and. &
              data%ratio <= two-data%control%eta_very_successful) then
        data%success_str = '   very'
        data%success     = 2
     elseif ( data%ratio >= data%control%eta_successful ) then
        data%success_str = 'success'
        data%success     = 1
     else
        data%success = 0
        data%success_str   = 'failure'
     end if

     ! Determine step acceptance.
     ! **************************

     if ( data%NM%active ) then
        if ( data%success >= 1 ) then
           data%NM%num_fail        = 0
           data%NM%active          = .false.
           data%step_accepted      = .true.
        else
           data%NM%num_fail  = data%NM%num_fail + 1
           if (data%NM%num_fail <= data%control%NM_steps ) then
              data%step_accepted = .true.
           else
              data%NM%revert          = .true.
              data%step_accepted      = .false.
           end if
        end if
     else ! must be monotone
        if ( data%success >= 1 ) then
           data%step_accepted = .true.
        else
           data%step_accepted = .false.
           if ( m > 0 .and. data%control%use_steering ) then
              data%penalty_steer_reset = .true.
           end if
        end if
     end if

     ! Print summary for Predictor, Cauchy, and SQP steps.
     !****************************************************

     if ( data%iterate >= 0 ) then
        if ( data%control%print_level >= GALAHAD_ACTION ) then
            write( out, 1005 ) data%iterate-1,                                 &
                  data%control%B_type, data%sqp_computed,  data%decreaseH_full,&
                  data%BFGS%mod_type, data%control%correction_type,            &
                  data%sqp_ratio_used, data%TRpred, data%TRsqp,                &
                  data%ratio, data%iterates_pred,                              &
                  data%iterates_sqp, data%success_str,  data%inf_norm_s_p,     &
                  data%inf_norm_s_s, data%alpha_c,                             &
                  data%descent_constraint_status, data%NM%active,              &
                  data%decreaseH_cauchy, data%inf_norm_s_f,                    &
                  data%control%NM_steps, data%inf_norm_Y_p,                    &
                  data%sqp_good_dec, data%step_accepted,                       &
                  data%inf_norm_Y_s, data%NM%num_fail, data%alpha_feas,        &
                  data%NM%revert, data%seqp_computed
        end if
     end if

     ! Revert to former data if non-monotone fails.  Update predictor TR.
     ! ******************************************************************

     if ( data%NM%revert ) then

        data%NM%active   = .false.
        data%NM%num_fail = 0

        nlp%X          = data%revert%X_revert
        nlp%Y          = data%revert%Y_revert
        nlp%Y_a        = data%revert%Y_a_revert
        nlp%Z          = data%revert%Z_revert
        nlp%f          = data%revert%f_revert
        nlp%G          = data%revert%G_revert
        nlp%C          = data%revert%C_revert
        nlp%J%val      = data%revert%Jval_revert
        nlp%Ax         = data%revert%Ax_revert
        data%B%val     = data%revert%Bval_revert
        data%merit     = data%revert%merit_revert
        data%penalty   = data%revert%penalty_revert
        data%norm_c    = data%revert%norm_c_revert
        data%TRpred    = data%revert%TRpred_revert
        data%primal_vl = data%revert%primal_vl_revert
        data%dual_vl   = data%revert%dual_vl_revert
        data%comp_vl   = data%revert%comp_vl_revert
        data%C_RES_l   = data%revert%C_RES_l_revert
        data%C_RES_u   = data%revert%C_RES_u_revert
        data%A_RES_l   = data%revert%A_RES_l_revert
        data%A_RES_u   = data%revert%A_RES_u_revert
        data%X_RES_l   = data%revert%X_RES_l_revert
        data%X_RES_u   = data%revert%X_RES_u_revert

        data%TRpred = max( tenth * data%TRpred, data%min_TRpred )

        goto 715

     end if

     ! Update trust-region, possibly compute data for BFGS update
     ! **********************************************************

     if ( .not. data%step_accepted ) then  ! monotone & success = 0

        ! Ensure that the predictor step is different next time.

        data%TRpred = max( data%min_TRpred, half * data%inf_norm_s_p )

     else ! step_accepted

        if ( data%sqp_ratio_used ) then
           nlp%X = nlp%X + data%s_f
        else
           nlp%X = nlp%X + data%s_c
        end if

        ! Compute BFGS info if applicable.

        if ( (data%control%B_type == 3 .or. data%control%B_type == 2 ) &
                          .and. data%success >= 1 ) then ! BFGS or LBFGS
           if ( m > 0 ) then
              data%Jval_prev = nlp%J%val
           end if
           data%G_prev = nlp%G
        end if

        ! Update problem functions and merit function.

        nlp%F = data%f_new

        if ( nlp%m > 0 ) then

           nlp%C = data%C_new ;    data%norm_c = data%norm_c_new

           call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
           if ( inform%status /= 0 ) write( out, 1002 ) 'eval_J'

           inform%num_J_eval = inform%num_J_eval + 1

        end if

        data%merit = data%merit_new

        if ( nlp%m_a > 0 ) then
           nlp%Ax = zero
           call mop_Ax( one, nlp%A, nlp%X, one, nlp%Ax,       &
                        out, control%error, transpose=.false. )
        end if

        call eval_G( inform%status, nlp%X, userdata, nlp%G  )
        if ( inform%status /= 0 ) write( out, 1002 ) 'eval_G'

        inform%num_g_eval = inform%num_g_eval + 1

        ! Update trust-region

        if ( data%NM%active ) then ! success = 0

!           data%TRpred = min( 100_wp * data%TRpred, data%control%max_TRpred
           data%TRpred = min( data%TRpred, data%control%max_TRpred )

        else ! sucess >= 1

           dummy_real = max( data%inf_norm_s_p, data%inf_norm_s_s )

           select case ( data%success )
           case ( 3 )
              dummy_real  = data%inf_norm_s_s / data%control%TRsqp_scale
              data%TRpred = five*max( data%inf_norm_s_p, dummy_real )
           case ( 2 )
              dummy_real  = data%inf_norm_s_s / data%control%TRsqp_scale
              data%TRpred = two*max( data%inf_norm_s_p, dummy_real )
           case ( 1 )
              dummy_real  = data%inf_norm_s_s / data%control%TRsqp_scale
              data%TRpred = max( data%inf_norm_s_p, dummy_real )
              !data%TRpred = data%TRpred
           case default
              write(out,*) ' This should not happend : trimsqp : frog'
           end select

           if ( data%TR_reset_value > data%TRpred ) then
              data%TRpred = data%TR_reset_value
              data%TR_reset = .true.
           else
              data%TR_reset = .false.
           end if

           data%TRpred = min( data%TRpred, data%control%max_TRpred )

        end if

     end if

715 continue

end do

! **********************************
! END: Main do loop.
! **********************************

! Check for max number of iterations

  !if ( data%iterate >= data%control%max_iterate .and. (.not. data%converged) ) then
  !   inform%status = GALAHAD_MAX_ITERATIONS_REACHED
  !end if

  ! Fill inform.

  !inform%status = ?  Should already be filled
  inform%iterate    = data%iterate
  inform%primal_vl  = data%primal_vl
  inform%dual_vl    = data%dual_vl
  inform%comp_vl    = data%comp_vl
  inform%obj        = nlp%f
  inform%time%total = zero  ! Add this later.

! *****************************************************************************

  IF ( data%control%print_level >= GALAHAD_TRACE ) then

     WRITE( data%control%out, "( /, ' PROBLEM NAME : ', A8 )" ) nlp%pname


     SELECT CASE ( inform%status )

     CASE ( GALAHAD_ok )
        WRITE( out, "(' RESULT       : success' )" )
     CASE ( GALAHAD_error_max_iterations )
        WRITE( out, "(' RESULT       : maximum iterations reached' )" )
     CASE ( -40 )
        WRITE( out, "(' RESULT       : maximum penalty parameter reached.')" )
     CASE DEFAULT
        WRITE(out,*) ' ERROR : trimsqp : inform%status unrecognized on exit. '
     END SELECT

     write( data%control%out, "(' EXIT STATUS  : ', I4, / )" ) inform%status
     IF ( data%control%print_sol ) THEN
        IF( data%control%fulsol ) then
           WRITE( data%control%out, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
           IF ( nlp%m  > 0 ) WRITE( data%control%out, "( ' Y  ', / ( 3ES24.16 ) )" ) nlp%Y
           IF ( nlp%m_a> 0 ) WRITE( data%control%out, "( ' Y_a', / ( 3ES24.16 ) )" ) nlp%Y_a
           WRITE( data%control%out, "( ' Z', / ( 3ES24.16 ) )" ) nlp%Z
        ELSE
           write(*,*) 'trimsqp : not yet implemented - 100'
           WRITE( data%control%out, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
           IF ( nlp%m > 0 ) WRITE( data%control%out, "( ' Y', / ( 3ES24.16 ) )" ) nlp%Y
           IF ( nlp%m_a> 0 ) WRITE( data%control%out, "( ' Y_a', / ( 3ES24.16 ) )" ) nlp%Y_a
           WRITE( data%control%out, "( ' Z', / ( 3ES24.16 ) )" ) nlp%Z
        END IF
     END IF
  end IF

 ! Deallocate problems

  call dealloc_QPfeas( data%QPfeas, inform, data%control )
  if ( inform%status /= GALAHAD_ok ) go to 992

  call dealloc_QPpred( data%QPpred, inform, data%control )
  if ( inform%status /= GALAHAD_ok ) go to 992

  call dealloc_QPsiqp( data%QPsiqp, inform, data%control )
  if ( inform%status /= GALAHAD_ok ) go to 992

  call dealloc_QPseqp( data%QPseqp, inform, data%control )
  if ( inform%status /= GALAHAD_ok ) go to 992

  !  Write out the solution to file "sfilename".
  INQUIRE( FILE = sfilename, EXIST = filexx )
  IF ( filexx ) THEN
     OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',               &
          STATUS = 'OLD', IOSTAT = iores )
  ELSE
     OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',               &
          STATUS = 'NEW', IOSTAT = iores )
  END IF
  IF ( iores /= 0 ) THEN
     WRITE( data%control%out, "( ' IOSTAT = ', I6, ' when opening file ', A9 )" )         &
          iores, sfilename
     RETURN
  END IF

  WRITE( sfiledevice, "( '*   TRIMSQP solution for problem name: ', A8 )" )     &
       nlp%pname
  WRITE( sfiledevice, "( /, '*   variables ', / )" )
  DO i = 1, nlp%n
     WRITE( sfiledevice, "( '    Solution  ', A10, ES12.5 )" )               &
          nlp%VNAMES( i ), nlp%X( i )
  END DO
  WRITE( sfiledevice, "( /, '*   Lagrange multipliers - general constraints ', / )" )
  DO i = 1, nlp%m
     WRITE( sfiledevice, "( ' M-gen  Solution  ', A10, ES12.5 )" )              &
          nlp%CNAMES( i ), nlp%Y( i )
  END DO
  WRITE( sfiledevice, "( /, '*   Lagrange multipliers - linear constraints ', / )" )
  DO i = 1, nlp%m_a
     WRITE( sfiledevice, "( ' M-lin  Solution  ', A10, ES12.5 )" )              &
          nlp%ANAMES( i ), nlp%Y_a( i )
  END DO
  WRITE( sfiledevice, "( /, ' XL Solution  ', 10X, ES12.5 )" ) nlp%f
  CLOSE( UNIT = sfiledevice, IOSTAT = iores )
  IF ( iores /= 0 ) GO TO 991

  RETURN ! stops normal execution.

! Error statements and exits.
 990 CONTINUE
     WRITE( data%control%error, 2000) inform%status, inform%alloc_status
     STOP

 991 CONTINUE
     WRITE( data%control%out, "( ' TRIMSQP: error closing file.  iores =  ', I0 )" ) iores
     STOP

 992 CONTINUE
     WRITE( data%control%error, 2001) inform%status, inform%alloc_status
     STOP

999  Continue
     STOP

! Formatting statements.

1000 FORMAT( /,  &
     ' Iterate   Penalty       Merit          Primal_vl         Dual_vl ', &
     '        Comp_Slack    Y   FOO  ' )
1001 FORMAT(3X, I5, 2X, ES8.2, 2X, ES15.8, 3(2X, ES15.9), &
            2X, A, 2X, L3 )
1002 format( 1x, 'stat /=0 on return from ', A )
1003 format( /, &
     1x, (93('=')), /, &
     T35, 'Sub-Optimality Results', /,                                     &
     T35, '----------------------', /,                                     &
     1x, 'sub_stop_p      = ', ES12.6, T37, 'sub_stop_d   = ', ES13.6,     &
                                       T67, 'sub_stop_c   = ', ES12.6, /,  &
     1x, 'sub_primal_vl   = ', ES12.6, T37, 'sub_dual_vl  = ', ES13.6,     &
                                       T67, 'sub_comp_vl  = ', ES12.6, /,  &
     1x, 'merit_1st_order = ', L1,     T37, 'norm_c       = ', ES13.6,     &
                                       T67, 'eta          = ', ES12.6, /,  &
     1x, 'penalty         = ', ES12.6, T37, 'merit        = ', ES13.6,     &
                                       T67, 'eta_contract = ', ES12.6, /,  &
     1x, (93('=')) )

!1002 FORMAT( /,  &
!     ' Iterate   Penalty          Merit            Primal_vl        Dual_vl  ',&
!     '       Comp_Slack     TRpred    TRsqp       |c(x)|_1         C_forcing',&
!     '     FOO   Penalty' )
!1003 FORMAT(3X, I5, 2X, ES8.2, 2X, ES19.12, 3(2X, ES15.9), 2(2X, ES8.2),       &
!            2( 2X, ES15.9), 2X, L3, 2x, A8 )
!1004 FORMAT( /,  &
!     '          approx_B   iter_p    inf_norm_s_p         alpha        ', &
!     'decrease_cauchy   sqp   iter_s       inf_norm_s_s   descent       decrease_full ', &
!     '     ratio   accept   success' )
!1005 FORMAT(T11, A, 2X, I7, 2X, ES13.6, 2X, ES12.6, 2X, ES19.12,      &
!            2X, A3, 2X, I7, 4X, ES13.6, 2X, A4, 5X, ES19.12,          &
!            2X, ES9.2, 2X, A3, 5X, A7, / )

1005 FORMAT( /, &
  2X, '****************************************************',                  &
      '***********************************************', /,                    &
  2X, '*****                        BEGIN SUMMARY (TRIMSQP) :',                &
      ' ITERATE S(', I7,')                     *****', /,                      &
  2X, '****************************************************',                  &
      '***********************************************', /,                    &
  2X, 'B-approx-type   = ', 12x, I1, T38, 'SQP-computed    = ', 12x, L1,       &
                                     T72, 'decrease-full   = ', ES13.6,   /,   &
  2X, 'BFGS-mod-type   = ', 12x, I1, T38, 'correction-type = ', 12x, I1,       &
                                     T72, 'sqp-ratio-used  = ', 12x, L1,  /,   &
  2X, 'predict-radius  = ', ES13.6,  T38, 'sqp-radius      = ', ES13.6,        &
                                     T72, 'ratio           = ', ES13.6,   /,   &
  2X, 'iterations-pred = ', 6x, I7,  T38, 'iterations-sqp  = ', 6x, I7,        &
                                     T72, 'step-sucess     = ', 6x, A7,   /,   &
  2X, 'inf-norm-pred   = ', ES13.6,  T38, 'inf-norm-sqp    = ', ES13.6,        &
                                     T72, 'blow-up         = ', 12x,    /,     &
  2X, 'alpha-cauchy    = ', ES13.6,  T38, 'descent-status  = ', 9x, A4,        &
                                     T72, 'non-mono-active = ', 12x, L1,  /,   &
  2X, 'decrease-cauchy = ', ES13.6,  T38, 'inf-norm-full   = ', ES13.6,        &
                                     T72, 'non-mono-steps  = ', 12x, I1,  /,   &
  2x, 'inf-norm-Y_p    = ', ES13.6,  T38, 'sqp-good-dec    = ', 12x, L1,       &
                                     T72, 'step-accepted   = ', 12x, L1,  /,   &
                                     T38, 'inf-norm-Y_s    = ', ES13.6,        &
                                     T72, 'non-mono-#-fail = ', 12x, I1,  /,   &
                                     T38, 'alpha-feasible  = ', ES13.6,        &
                                     T72, 'reverting       = ', 12x, L1,  /,   &
                                     T38, 'sEqp-computed   = ', 12x, L1,  /,   &
  2X, '****************************************************',                  &
      '***********************************************', /,                    &
  2X, '*****                                 END SUMMARY (TRIMSQP)',           &
      '                                   *****', /,                           &
  2X, '****************************************************',                  &
      '***********************************************' )
!1000 FORMAT(1X, I5, 2X, ES9.2, 2X, 4ES16.9 L4, '     -          -        -    -',         &
!             ES12.4, '  -     -     ', I4, '      -            -',            &
!             '              -                -         ')

!every line of output (except the first)
!1001 FORMAT( I5, 1X, 3ES19.12, L4, 2ES11.4, I4, I5, ES12.4, I3, ES11.4, I4,   &
 !            ES11.4, ES12.4, ES19.11, ES19.11, L3 )

!column headers for ouput
!1000 FORMAT( /, ' Iterate  Penalty   Merit      Primal_Feas       Dual_Feas     Comp_Slack' )


!!             '      sgn   dxMAX       yMAX     QP QPit      f      #b',       &
!             '   sigma     nv     TR       y_model          gmts',            &
!             '              merit        imp' )

2000 format(1x, '** ERROR : allocation error in subroutine TRIMSQP_solve.',   &
            ' error= ', I0, ' status= ', I0, '.')
2001 format(1x, '** ERROR : de-allocation error in subroutine TRIMSQP_solve.',   &
            ' error= ', I0, ' status= ', I0, '.')
!3000 format(1x, (65('*')), / )
!3001 format(1x, (65('*')),                                                         /, &
!     1x, '                  fill_QPfeas : ( iterate = ', I5,')                   ',/, &
!     1x, '                  ---------------------------------                    ',/, &
!     1x, '    QPfeas%X_l           QPfeas%X         QPfeas%X_u          QPfeas%X0',/, &
!   ( 2x, ES16.9, 3x, ES16.9, 3x, ES16.9, 3x, ES16.9) )
!3002 format( /, &
!     1x, '    QPfeas%C_l           QPfeas%C_u          QPfeas%Y     ',/, &
!     ( 2x, ES16.9, 3x, ES16.9, 3x, ES16.9) )
!3003 format( /, &
!     1x, '      QPfeas%Z        QPfeas%WEIGHT ',/, &
!     ( 2x, ES16.9, 3x, ES16.9) )
!3004 format( /, 2x, ' QPfeas%m = ', I5, '  QPfeas%n = ', I5,  &
!                2x, ' Gradient_kind = ', I1, '  Hessian_kind = ', I1 )
!!$3005 format(1x, (65('*')), /                                         &
!!$     1x, '                fill_QPpred : computing s_p(', I5,')',/, &
!!$     1x, '                ----------------------------------  ',/, &
!!$     1x, '    QPpred%X_l           QPpred%X         QPpred%X_u',/, &
!!$    (2x, ES16.9, 3x, ES16.9, 3x, ES16.9 ) )
!!$3006 format( /, &
!!$     1x, '    QPpred%C_l           QPpred%C_u ',/, &
!!$     ( 2x, ES16.9, 3x, ES16.9 ) )
!!$3007 format( /, &
!!$     1x, '    QPpred%Y           QPpred%C_status ', /, ( 2x, ES16.9, 12x, I2 ) )
!!$3008 format( /,  &
!!$     1x, '      QPpred%Z         QPpred%X_status        QPpred%G ', /,  &
!!$     ( 2x, ES16.9, 12x, I2, 11x, ES16.9 ) )
!!$3009 format( /, 2x, ' QPpred%m = ', I5, '   QPpred%n = ', I5, '   QPpred%f = ', ES16.9 )
!!$3010 format(1x, (65('*')), /                                      &
!!$     1x, '                fill_QPsiqp : computing s_s(', I5,')',/, &
!!$     1x, '                ---------------------------------  ',/, &
!!$     1x, '    QPsiqp%X_l           QPsiqp%X         QPsiqp%X_u  ',/, &
!!$    (2x, ES16.9, 3x, ES16.9, 3x, ES16.9 ) )
!!$
!!$3011 format( /, &
!!$     1x, '    QPsiqp%C_l           QPsiqp%C_u ',/, &
!!$     ( 2x, ES16.9, 3x, ES16.9 ) )
!!$3012 format( /, &
!!$     1x, '    QPsiqp%Y           QPsiqp%C_status ', /, ( 2x, ES16.9, 12x, I2 ) )
!!$3013 format( /,  &
!!$     1x, '      QPsiqp%Z         QPsiqp%X_status        QPsiqp%G ', /,  &
!!$     ( 2x, ES16.9, 12x, I2, 11x, ES16.9 ) )
!!$3014 format( /, 2x, ' QPsiqp%m = ', I5, '   QPsiqp%n = ', I5, '   QPsiqp%f = ', ES16.9 )
!3015 format( 1x, 'Merit function is first-order optimal', &
!                 ' - turning non-monotone off.', /)
3016 format( 1x, ' NM%active = T : saving model decrease for possible revert', / )
!3015 format( /, '---------------------------------------------------------------',/,&
!             1x,'BFGS is being used, the following data has been computed',/ &
!             10x, 'd', 15x, 'Bs', 11x, 'gradLxnew', 10x, 'gradLx', /          &
!             6x, '---------', 7x, '---------', 8x, '---------', 8x, '----------' )
!3016 format( /, 's^Td = ', ES16.9, 5x, 's^TBs = ', ES16.9, /, &
!                '--------------------------------------------------------------')
!3017 format( 1x, '(num_consec_Y_free,consec_Y_free_needed) = ', &
!             2(I4), '  New penalty parameter will be ', ES8.2 )
!!$3018 format(1x, (65('*')),                                                         /, &
!!$     1x, '                  fill_QPsteer : ( iterate = ', I5,')                   ',/, &
!!$     1x, '                  ---------------------------------                    ',/, &
!!$     1x, '    QPsteer%X_l           QPsteer%X         QPsteer%X_u',/, &
!!$   ( 2x, ES16.9, 3x, ES16.9, 3x, ES16.9) )
!!$3019 format( /, &
!!$     1x, '    QPsteer%C_l           QPsteer%C_u          QPsteer%Y     ',/, &
!!$     ( 2x, ES16.9, 3x, ES16.9, 3x, ES16.9) )
!!$3020 format( /, 1x, '      QPsteer%Z           QPsteer%G',/, ( 2x, ES16.9, 3x, ES16.9 ) )
!!$3021 format( /, 2x, ' QPsteer%m = ', I5, '  QPsteer%n = ', I5,  &
!!$                2x, ' Gradient_kind = ', I1, '  Hessian_kind = ', I1 )
!!$3038 format(1x, (65('*')),                                                         /, &
!!$     1x, '                  fill_QPseqp : ( iterate = ', I5,')                   ',/, &
!!$     1x, '                  ---------------------------------                    ',/, &
!!$     1x, '    QPseqp%X',/, &
!!$   ( 2x, ES16.9 ) )
!!$3039 format( /, &
!!$     1x, '    QPseqp%C (constant in constraint)     QPseqp%Y',/, &
!!$     ( 2x, ES16.9, 15x, ES16.9 ) )
!!$3040 format( /, 1x, '      QPseqp%G',/, ( 2x, ES16.9 ) )
!!$3041 format( /, 2x, ' QPseqp%m = ', I5, '  QPseqp%n = ', I5 )




!3050 format( 2x, '        X                 G                  Z', /, &
!            (2x, ES16.9, 3x, ES16.9, 3x, ES16.9 ) )
!3051 format( 2x, '        C          C_type          Y', /, &
!            (2x, ES16.9, 4x, A3, 4x, ES16.9 ) )
!3052 format( 2x, '        Ax         A_type          Y_a', /, &
!            (2x, ES16.9, 4x, A3, 4x, ES16.9 ) )
!3060 format(1x, '(optimal_measure, optimal_measure2, optimal_measure_3)  =  ', &
!            ES15.9, 3X, ES15.9, 3X, ES15.9 )
!3061 format(1x, 'optimal_measure  =  ', ES15.9 )
!3062 format(  &
!     1x, '(primal_vl, dual_vl, comp_vl)     =', 3(2X, ES15.9), /, &
!     1x, '(primal_tol, dual_tol, comp_tol)  =', 3(2X, ES15.9) )
!3063 format(  &
!     1x, '(sub_primal_vl, sub_dual_vl, sub_comp_vl)     =', 3(2X, ES15.9), /, &
!     1x, '(sub_primal_tol, sub_dual_tol, sub_comp_tol)  =', 3(2X, ES15.9) )
!3064 format( 1x, 'result of first-order check : penalty <--- ', A8 )
3065 format( 1x, '(two_norm_s, g^Ts, stHs) =', 3(2X, ES16.9) )
!3066 format( 1x, '-- entering get_cauchy_step.......with row-wise norms:', /, &
!             1x, 'J_row_norms = ', ( T16, ES16.9) )
3067 format( 1x, '(c_norm, c_linearize_norm) =', 2(2x, ES15.9 ), /,            &
      1x, 'decrease in model at cauchy point = ', ES16.9, /,                   &
      1x, 'num_sat = ', I7, 5x, 'num_vl_l = ', I7, 5x, 'num_vl_u = ', I7, /, &
      1x, '    sat                   vl_l                   vl_u', /,   &
      1x, '  -------                -------                -------'     )
3068 format( 1x, '-- full step : (c_norm, c_linearize_norm) =', 2(2x, ES15.9 ) )
!3069 format( 1x, '-- recomputing merit function at new penalty value.' )
3070 format( 1x, '-- full step computed and gives good decrease.  GOOD!',    /, &
             1x, '(fnew, cnew_vl) =', 2(2x, ES16.9 ), /, &
             1x, '(decrease cauchy, decrease full) =', 2(2x, ES16.9) )
3071 format( 1x, '(c_norm, c_linearize_norm, dec_con_vl_s_p ) =', 3(2x, ES16.9 ), /,          &
      1x, 'decrease in CONVEX model at predictor point = ', ES16.9         )
!3072 format( 1x, '(c_norm, con_viol_s_steer, dec_con_vl_s_steer ) =', 3(2x, ES15.9 ) )

!3071 format( 1x, '-- full step : change in model < tiny.  BAD!',    /, &
!             1x, '(decrease cauchy, decrease full) =', 2(2x, ES16.9) )
!3072 format( 1x, '-- full step not computed OR not good decrease in model.', /, &
!             1x, '(fnew, cnew_vl, max_infeas_bar) =', 3(2x, ES16.9 ), /, &
!             1x, '(decrease cauchy, decrease full) =', 2(2x, ES16.9) )
!3073 format( 1x, '...... starting subproblem optimality check with :', /, &
!          8x, 'JtY', 15x, 'AtY_a', 17x, 'Z', /         &
!          6x, 3('--------', 11x ) )
!3074 format( 3( 1x, ES16.9, 2x) )
3075 format(1x, 'X value input by user : X = ', /, (1x, ES16.9))
3076 format(1x, 'Result of making X feasible with respect to bounds : ', &
                'X = ', /, (1x, ES16.9) )
3077 format(1x, 'Result of making X feasible with respect to linear ', &
                'constraints : X = ', /, (1x, ES16.9) )
3078 format(1x, 'g + H s_c = ', (T14, ES16.9) )
3079 format(1x, 'descent_constraint = ', (T23, ES16.9) )
3080 format(1x, 'descen_constraint^T s(initial) = ', ES16.9 )
3081 format(1x, 'No need to call subroutine get_cauchy_step : s_c = s_p' )
3082 format(1x, 'Not calling get_cauchy_step : s_p is really small!' )

!  End of subroutine TRIMSQP_solve

     END SUBROUTINE TRIMSQP_solve


!-*-  G A L A H A D -  T R I M S Q P _ t e r m i n a t e  S U B R O U T I N E -*

     SUBROUTINE TRIMSQP_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TRIMSQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRIMSQP_control_type ), INTENT( IN ) :: control
     TYPE ( TRIMSQP_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate arrays set for feasibility LP and steering LP.

     CALL LSQP_terminate( data%QPfeas_data, control%QPfeas_control,              &
                         inform%QPfeas_inform )
     IF ( inform%QPfeas_inform%status /= GALAHAD_ok ) THEN
       inform%status       = GALAHAD_error_deallocate
       inform%alloc_status = inform%QPfeas_inform%alloc_status
       inform%bad_alloc    = inform%QPfeas_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!!$     CALL LSQP_terminate( data%QPsteer_data, control%QPsteer_control,              &
!!$                         inform%QPsteer_inform )
!!$     IF ( inform%QPsteer_inform%status /= GALAHAD_ok ) THEN
!!$       inform%status       = GALAHAD_error_deallocate
!!$       inform%alloc_status = inform%QPsteer_inform%alloc_status
!!$       inform%bad_alloc    = inform%QPsteer_inform%bad_alloc
!!$       IF ( control%deallocate_error_fatal ) RETURN
!!$     END IF

!  Deallocate arrays set for QPC

     CALL QPC_terminate( data%QPsteer_data, control%QPsteer_control,              &
                         inform%QPsteer_inform )
     IF ( inform%QPsteer_inform%status /= GALAHAD_ok ) THEN
       inform%status       = GALAHAD_error_deallocate
       inform%alloc_status = inform%QPsteer_inform%alloc_status
       inform%bad_alloc    = inform%QPsteer_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL QPC_terminate( data%QPpred_data, control%QPpred_control,              &
                         inform%QPpred_inform )
     IF ( inform%QPpred_inform%status /= GALAHAD_ok ) THEN
       inform%status       = GALAHAD_error_deallocate
       inform%alloc_status = inform%QPpred_inform%alloc_status
       inform%bad_alloc    = inform%QPpred_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL QPC_terminate( data%QPsiqp_data, control%QPsiqp_control,              &
                         inform%QPsiqp_inform )
     IF ( inform%QPsiqp_inform%status /= GALAHAD_ok  ) THEN
       inform%status       = GALAHAD_error_deallocate
       inform%alloc_status = inform%QPsiqp_inform%alloc_status
       inform%bad_alloc    = inform%QPsiqp_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate arrays set for EQP.

     CALL EQP_terminate( data%QPseqp_data, control%QPseqp_control,            &
                         inform%QPseqp_inform )
     IF ( inform%QPseqp_inform%status /= GALAHAD_ok ) THEN
       inform%status       = GALAHAD_error_deallocate
       inform%alloc_status = inform%QPseqp_inform%alloc_status
       inform%bad_alloc    = inform%QPseqp_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all remaining allocated arrays

     !array_name = 'trimSQP: data%Jv'
     !CALL SPACE_dealloc_array( data%Jv,                                        &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Jtv'
     !CALL SPACE_dealloc_array( data%Jtv,                                       &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Jtv2'
     !CALL SPACE_dealloc_array( data%Jtv2,                                      &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Av'
     !CALL SPACE_dealloc_array( data%Av,                                        &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Atv'
     !CALL SPACE_dealloc_array( data%Atv,                                       &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Atv2'
     !CALL SPACE_dealloc_array( data%Atv2,                                      &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     !array_name = 'trimSQP: data%Hv'
     !CALL SPACE_dealloc_array( data%Hv,                                        &
     !     inform%status, inform%alloc_status, array_name = array_name,         &
     !     bad_alloc = inform%bad_alloc, out = control%error )
     !IF ( control%deallocate_error_fatal .AND.  &
     !     inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%IBREAK'
     CALL SPACE_dealloc_array( data%IBREAK,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%BREAKP'
     CALL SPACE_dealloc_array( data%BREAKP,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%J_norms'
     CALL SPACE_dealloc_array( data%J_norms,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%H_norms'
     CALL SPACE_dealloc_array( data%H_norms,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%s_p'
     CALL SPACE_dealloc_array( data%s_p,                                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%s_c'
     CALL SPACE_dealloc_array( data%s_c,                                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%s_s'
     CALL SPACE_dealloc_array( data%s_s,                                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%s_f'
     CALL SPACE_dealloc_array( data%s_f,                                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%X_type'
     CALL SPACE_dealloc_array( data%X_type,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%A_type'
     CALL SPACE_dealloc_array( data%A_type,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%C_type'
     CALL SPACE_dealloc_array( data%C_type,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%approx_type'
     CALL SPACE_dealloc_array( data%approx_type,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%X_RES_l'
     CALL SPACE_dealloc_array( data%X_RES_l,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%X_RES_u'
     CALL SPACE_dealloc_array( data%X_RES_u,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%A_RES_l'
     CALL SPACE_dealloc_array( data%A_RES_l,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%A_RES_u'
     CALL SPACE_dealloc_array( data%A_RES_u,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%C_RES_l'
     CALL SPACE_dealloc_array( data%C_RES_l,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%C_RES_u'
     CALL SPACE_dealloc_array( data%C_RES_u,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%J_cauchy'
     CALL SPACE_dealloc_array( data%J_cauchy,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%C_cauchy'
     CALL SPACE_dealloc_array( data%C_cauchy,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%C_new'
     CALL SPACE_dealloc_array( data%C_new,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'trimSQP: data%gplusHs'
     CALL SPACE_dealloc_array( data%gplusHs,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.  &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine TRIMSQP_terminate

   END SUBROUTINE TRIMSQP_terminate


!-*-   B U I L D _ Q P f e a s   S U B R O U T I N E -*

  SUBROUTINE build_QPfeas( nlp, QPfeas, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Allocates components of variable LP of type
!   QPT_problem_type.  Storage type is determined
!   by the storage in the corresponding
!   components in problem nlp of NLPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( NLPT_problem_type ), intent( in ) :: nlp
    type( QPT_problem_type ), intent( inout ) :: QPfeas

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    integer :: len

    ! Define infinity

    control%QPfeas_control%infinity = control%infinity / ten

    ! Set problem dimensions

    QPfeas%m  = nlp%m_a
    QPfeas%n  = nlp%n

    QPfeas%Hessian_kind          = 2               ! General weights used.
    QPfeas%gradient_kind         = 0               ! G = 0.
    QPfeas%new_problem_structure = .true.

    ! set some dimensions

    QPfeas%A%m = nlp%m_a  ;  QPfeas%A%n = nlp%n

    ! Allocate components independent of storage type.

    !CALL SPACE_resize_array( QPfeas%n, QPfeas%G, inform%status, inform%alloc_status )
    !IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%X_l, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%X, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%X0, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%X_u, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%Z, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%n, QPfeas%WEIGHT, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%m, QPfeas%C_l, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%m, QPfeas%C, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%m, QPfeas%C_u, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPfeas%m, QPfeas%Y, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    ! Allocate A: since LSQP does not support all storage types that
    ! trimsqp will support, temporarily convert storage type to
    ! coordinate.  Thus, regardless of nlp%A%type, QPfeas%A%type will be
    ! coordinate.  This spaces is the space that is allocated below.

    SELECT CASE ( SMT_get( nlp%A%type ) )

    CASE ( 'DENSE' )

       len = nlp%m_a * nlp%n

       CALL SPACE_resize_array( len, QPfeas%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPfeas%A%ne = len

    CASE ( 'COORDINATE' )

       len =  nlp%A%ne

       CALL SPACE_resize_array( len, QPfeas%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPfeas%A%ne = len

    CASE ( 'SPARSE_BY_ROWS')

       len =  nlp%A%ptr( nlp%m_a + 1 ) - 1

       CALL SPACE_resize_array( len, QPfeas%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPfeas%A%ne = len

    CASE ( 'SPARSE_BY_COLUMNS' )

       len =  nlp%A%ptr( nlp%n + 1 ) - 1

       CALL SPACE_resize_array( len, QPfeas%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPfeas%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPfeas%A%ne = len

    CASE DEFAULT

       WRITE( control%error, 1000)

    END SELECT

    call SMT_put( QPfeas%A%type, 'COORDINATE', inform%status )

    inform%status = 0

    RETURN

! Abnormal returns

990 CONTINUE

    WRITE( control%error, 1001) inform%status, inform%alloc_status

    RETURN

! Format statements

1000 FORMAT(1X, '** ERROR : unrecognized storage type in subroutine ALLOC_LP.')
1001 FORMAT(1X, '** ERROR : allocation error in subroutine ALLOC_LP.',   &
            ' error= ', I0, ' status= ', I0, '.')

  END SUBROUTINE build_QPfeas

!-*-   B U I L D _ Q P s t e e r   S U B R O U T I N E -*

  SUBROUTINE build_QPsteer( nlp, QPsteer, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Allocates components of variable LP of type
!   QPT_problem_type.  Storage type is determined
!   by the storage in the corresponding
!   components in problem nlp of NLPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( NLPT_problem_type ), intent( in ) :: nlp
    type( QPT_problem_type ), intent( inout ) :: QPsteer

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    integer :: len

    ! Define infinity

    control%QPsteer_control%infinity = control%infinity / ten

    ! Set problem dimensions

    QPsteer%m  = nlp%m + nlp%m_a
    QPsteer%n  = nlp%n + 2*nlp%m

    QPsteer%Hessian_kind          = 0               ! Linear program.
    !QPsteer%gradient_kind         = 1               ! general G.
    QPsteer%new_problem_structure = .true.
    !QPsteer%Weights = zero

    control%QPsteer_control%QPA_control%cold_start = 0

    ! set some dimensions

    QPsteer%A%m = nlp%m + nlp%m_a  ;  QPsteer%A%n = nlp%n + 2*nlp%m

    ! Allocate components independent of storage type.

    CALL SPACE_resize_array( QPsteer%n, QPsteer%G, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%X_l, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%X, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%X0, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%X_u, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%X_status, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%n, QPsteer%Z, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    !CALL SPACE_resize_array( QPsteer%n, QPsteer%WEIGHT, inform%status, inform%alloc_status )
    !IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%m, QPsteer%C_l, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%m, QPsteer%C, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%m, QPsteer%C_status, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%m, QPsteer%C_u, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( QPsteer%m, QPsteer%Y, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    ! Allocate A: since LSQP does not support all storage types that
    ! trimsqp will support, temporarily convert storage type to
    ! coordinate.  Thus, regardless of nlp%A%type, QPsteer%A%type will be
    ! coordinate.  This spaces is the space that is allocated below.

    SELECT CASE ( SMT_get( nlp%A%type ) )

    CASE ( 'DENSE' )

       len = nlp%m_a * nlp%n

       CALL SPACE_resize_array( len, QPsteer%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPsteer%A%ne = len

    CASE ( 'COORDINATE' )

       len =  nlp%A%ne + nlp%J%ne + 2*nlp%m

       CALL SPACE_resize_array( len, QPsteer%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPsteer%A%ne = len

    CASE ( 'SPARSE_BY_ROWS')

       len =  nlp%A%ptr( nlp%m_a + 1 ) - 1

       CALL SPACE_resize_array( len, QPsteer%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPsteer%A%ne = len

    CASE ( 'SPARSE_BY_COLUMNS' )

       len =  nlp%A%ptr( nlp%n + 1 ) - 1

       CALL SPACE_resize_array( len, QPsteer%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990
       CALL SPACE_resize_array( len, QPsteer%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) GO TO 990

       QPsteer%A%ne = len

    CASE DEFAULT

       WRITE( control%error, 1000)

    END SELECT

    call SMT_put( QPsteer%A%type, 'COORDINATE', inform%status )
    call SMT_put( QPsteer%H%type, 'COORDINATE', inform%status )

    CALL SPACE_resize_array( 0, QPsteer%H%val, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990
    CALL SPACE_resize_array( 0, QPsteer%H%col, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990
    CALL SPACE_resize_array( 0, QPsteer%H%row, inform%status, inform%alloc_status )
    IF ( inform%status /= 0 ) GO TO 990

    QPsteer%H%ne = 0

    inform%status = 0

    RETURN

! Abnormal returns

990 CONTINUE

    WRITE( control%error, 1001) inform%status, inform%alloc_status

    RETURN

! Format statements

1000 FORMAT(1X, '** ERROR : unrecognized storage type in subroutine ALLOC_LP.')
1001 FORMAT(1X, '** ERROR : allocation error in subroutine ALLOC_LP.',   &
            ' error= ', I0, ' status= ', I0, '.')

  END SUBROUTINE build_QPsteer

!-*-   B U I L D _ Q P p r e d  S U B R O U T I N E -*

  SUBROUTINE build_QPpred( nlp, QPpred, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Allocates components of variable QPpred of type QPT_problem_type
!   and sets the sparsity patterns.  Storage formatc is determined by the
!   storage in the corresponding components in problem nlp of
!   NLPT_problem_type.  See TRIMSQP_3 for problem form, which is
!   is determined by the value of control%B_type.
!
!   B_type   0   identity
!            1   weighted identity
!            2   L-BFGS
!            3   Full BFGS
!            4   Exact
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout ) :: inform
    type( TRIMSQP_data_type ), intent( inout ) :: data
    type( NLPT_problem_type ), intent( in ) :: nlp
    type( QPT_problem_type ), intent( inout ) :: QPpred

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    integer :: len, Ane, tally, i, j, out, error, m, n, m_a, L!, Hne, Bne
    integer :: B_type, Bind, Aind, I_Bind, I_Aind, Ltn, np2m
    character, allocatable, dimension( : ) :: store_type

    ! For convenience

    error = data%control%error ;
    out  = data%control%out    ;     L      = data%control%L_BFGS_number
    m_a  = nlp%m_a             ;     B_type = data%control%B_type
    m    = nlp%m
    n    = nlp%n
    np2m = n+2*m
    Ltn  = L*n

    ! Set infinity

    data%control%QPpred_control%infinity = data%control%infinity / 100.0_wp

    ! Set dimensions

    if ( B_type == 2 )  then  ! L-BFGS
       QPpred%m   =  m + m_a + 2 * L
       QPpred%n   =  n + 2*m + 2 * L
       QPpred%A%m =  m + m_a + 2 * L
       QPpred%A%n =  n + 2*m + 2 * L
       QPpred%H%m =  n + 2*m + 2 * L
       QPpred%H%n =  n + 2*m + 2 * L
    elseif ( 0 <= B_type .and. B_type <= 4 )  then
       QPpred%m   = m + m_a ;    QPpred%n   = n + 2*m
       QPpred%A%m = m + m_a ;    QPpred%A%n = n + 2*m
       QPpred%H%m = n + 2*m ;    QPpred%H%n = n + 2*m
    else
       write(out,*) ' ERROR:TRIMSQP:illegal value for control%B_type'
    end if

    ! Signify new problem structure.

    QPpred%new_problem_structure = .true.

    ! Allocate components independent of storage type.

    QPpred%gradient_kind = 2

    CALL SPACE_resize_array( QPpred%n, QPpred%G, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%n, QPpred%X_l, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%n, QPpred%X, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%n, QPpred%X_u, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%n, QPpred%Z, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%n, QPpred%X_status, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%m, QPpred%C_l, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%m, QPpred%C, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%m, QPpred%C_u, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%m, QPpred%C_status, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPpred%m, QPpred%Y, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    if ( B_type == 2 ) then
       QPpred%G( np2m+1 : )   =  zero
       QPpred%X_l( np2m+1 : ) = -data%control%QPpred_control%infinity
       QPpred%X_u( np2m+1 : ) =  data%control%QPpred_control%infinity
       QPpred%C_l = zero
       QPpred%C_u = zero
    end if

    ! Set up for hot starts.

    QPpred%C_status = zero
    QPpred%X_status = zero
    !data%control%QPpred_control%QPA_control%cold_start = 0

    !*************************************************************
    ! Allocate storage for "A" : components dependent on storage !
    !                            type of nlp%A and nlp%J         !
    ! NB: nlp%J and nlp%A must use the same storage type.        !
    !************************************************************!

    if ( m > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%J%type ), inform%status )
    elseif ( m_a > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%A%type ), inform%status )
    else
       call SMT_put( store_type, 'COORDINATE', inform%status ) !maybe for L-BFGS
    end if

    SELECT CASE ( SMT_get( store_type ) )

    CASE ( 'DENSE' )
       write(out,*) ' TRIMSQP:build_QPpred: not yet implimented.'
    CASE ( 'COORDINATE' )

       if ( B_type == 2 ) then  ! L-BFGS
          allocate( data%L_BFGS%A( 1:n, 1:L ), data%L_BFGS%B( 1:n, 1:L ) )
          allocate( data%L_BFGS%S( 1:n, 1:L ), data%L_BFGS%Y( 1:n, 1:L ) )
          allocate( data%L_BFGS%ind( 1:L ), data%L_BFGS%SB_inner( 1:L, 1:L ) )

          data%L_BFGS%eta = point1

          ! CALL SPACE_resize_array( Ltn, data%L_BFGS%A_smt%row, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990
!           CALL SPACE_resize_array( Ltn, data%L_BFGS%A_smt%col, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990
!           CALL SPACE_resize_array( Ltn, data%L_BFGS%A_smt%val, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990

!           CALL SPACE_resize_array( Ltn, data%L_BFGS%B_smt%row, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990
!           CALL SPACE_resize_array( Ltn, data%L_BFGS%B_smt%col, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990
!           CALL SPACE_resize_array( Ltn, data%L_BFGS%B_smt%val, inform%status, inform%alloc_status )
!           IF ( inform%status /= GALAHAD_ok ) GO TO 990

!           data%L_BFGS%A_smt%m  = n
!           data%L_BFGS%A_smt%n  = L
!           data%L_BFGS%A_smt%ne = Ltn

!           data%L_BFGS%B_smt%m  = n
!           data%L_BFGS%B_smt%n  = L
!           data%L_BFGS%B_smt%ne = Ltn

!           tally = 0
!           do j = 1, L
!              do i = 1, n
!                 tally = tally + 1
!                 data%L_BFGS%A_smt%row( tally ) = i
!                 data%L_BFGS%A_smt%col( tally ) = j
!                 data%L_BFGS%B_smt%row( tally ) = i
!                 data%L_BFGS%B_smt%col( tally ) = j
!              end do
!           end do

       end if

       ! allocate needed vectors
       !************************
       len = 0
       if ( m   > 0 )     len = nlp%J%ne + 2*m
       if ( m_a > 0 )     len = len + nlp%A%ne
       if ( B_type == 2 ) len = len + 2*(n+1)*L

       CALL SPACE_resize_array( len, QPpred%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPpred%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPpred%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       QPpred%A%val = zero
       QPpred%A%ne  = len

       ! fill sparsity pattern and signify storage type
       !***********************************************
       call SMT_put( QPpred%A%type, 'COORDINATE', inform%status )

       Ane = 0

       ! Do the [ J I -I ] part.
       if ( m > 0 ) then
          QPpred%A%row( 1 : nlp%J%ne ) = nlp%J%row
          QPpred%A%col( 1 : nlp%J%ne ) = nlp%J%col
          Ane = nlp%J%ne
          do i = 1, m
             QPpred%A%row( Ane + i )     = i
             QPpred%A%col( Ane + i )     = n + i
             QPpred%A%val( Ane + i )     = one
             QPpred%A%row( Ane + i + m ) = i
             QPpred%A%col( Ane + i + m ) = n + m + i
             QPpred%A%val( Ane + i + m ) = -one
          end do
          Ane = Ane + 2*m
       end if

       ! Next the A part.
       if ( m_a > 0 ) then
          QPpred%A%row( Ane + 1 : Ane + nlp%A%ne ) = m + nlp%A%row
          QPpred%A%col( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%col
          Ane = Ane + nlp%A%ne
       end if

       ! Finally, the additional constraints from the limited memory vectors.
       if ( B_type == 2 ) then
          Aind   = Ane
          I_Aind = Ane + Ltn
          Bind   = I_Aind + L
          I_Bind = Bind + Ltn
          do i = 1, L
             do j = 1, n
                Aind = Aind + 1
                Bind = Bind + 1
                QPpred%A%row( Aind ) = m + m_a + i
                QPpred%A%col( Aind ) = j
                QPpred%A%row( Bind ) = m + m_a + L + i
                QPpred%A%col( Bind ) = j
             end do
             I_Aind = I_Aind + 1
             I_Bind = I_Bind + 1
             QPpred%A%row( I_Aind ) = m + m_a + i
             QPpred%A%col( I_Aind ) = n + 2*m + i
             QPpred%A%val( I_Aind ) = -one
             QPpred%A%row( I_Bind ) = m + m_a + L + i
             QPpred%A%col( I_Bind ) = n + 2*m + L + i
             QPpred%A%val( I_Bind ) = -one
          end do
       end if
    CASE ( 'SPARSE_BY_ROWS')
       write(out,*) ' TRIMSQP:build_QPpred: not yet implimented.'
    CASE ( 'SPARSE_BY_COLUMNS' )
       write(out,*) ' TRIMSQP:build_QPpred: not yet implimented.'
    CASE DEFAULT
       inform%status = GALAHAD_error_input_status
       go to 991
    END SELECT

    !********************************************************
    ! Allocate storage for the Hessian of the predictor QP. !
    !********************************************************

    if ( B_type == 3 ) then   ! BFGS

       len = n*(n+1)/2

       call SMT_put( QPpred%H%type, 'COORDINATE', inform%status )
       CALL SPACE_resize_array( len, QPpred%H%val, inform%status, inform%alloc_status )
       CALL SPACE_resize_array( len, QPpred%H%row, inform%status, inform%alloc_status )
       CALL SPACE_resize_array( len, QPpred%H%col, inform%status, inform%alloc_status )
       QPpred%H%m  = n + 2*m
       QPpred%H%n  = n + 2*m
       QPpred%H%ne = len

       tally = 0
       do i = 1, n
          do j = 1, i
             tally = tally + 1
             QPpred%H%row( tally ) = i
             QPpred%H%col( tally ) = j
          end do
       end do

       call SMT_put( data%B%type, 'COORDINATE', inform%status )
       call SPACE_resize_array( len, data%B%val, inform%status, inform%alloc_status )
       call SPACE_resize_array( len, data%B%row, inform%status, inform%alloc_status )
       call SPACE_resize_array( len, data%B%col, inform%status, inform%alloc_status )
       data%B%m   = n
       data%B%n   = n
       data%B%ne  = len

       data%B%row = QPpred%H%row
       data%B%col = QPpred%H%col
       data%B%val = QPpred%H%val

       CALL SPACE_resize_array( n, data%BFGS%d, inform%status, inform%alloc_status )
       CALL SPACE_resize_array( n, data%BFGS%Bs, inform%status, inform%alloc_status )
       CALL SPACE_resize_array( n, data%BFGS%gradLx, inform%status, inform%alloc_status )
       CALL SPACE_resize_array( n, data%BFGS%gradLx_new, inform%status, inform%alloc_status )

       if ( data%control%NM_steps > 0 ) then  ! non-monotone
          CALL SPACE_resize_array( len, data%revert%Bval_revert, inform%status, inform%alloc_status )
       end if

    elseif ( B_type == 4 ) then  ! EXACT

       ! use storage format used to store H.
       write(out,*) ' TRIMSQP:build_QPpred: not yet implimented.'

    else  ! Identity, weighted-diagonal, or L-BFGS.

       len = n + 2*m
       if ( B_type == 2 ) then
          len = len + 2*L
          CALL SPACE_resize_array( n, data%L_BFGS%gradLx, inform%status, inform%alloc_status )
          CALL SPACE_resize_array( n, data%L_BFGS%gradLx_new, inform%status, inform%alloc_status )
       end if

       call SMT_put( data%B%type,   'DIAGONAL', inform%status )
       call SPACE_resize_array( n, data%B%val, inform%status, inform%alloc_status )
       data%B%m  = n
       data%B%n  = n

       if ( data%control%NM_steps > 0 ) then  ! non-monotone
          CALL SPACE_resize_array( n, data%revert%Bval_revert, inform%status, inform%alloc_status )
       end if

       call SMT_put( QPpred%H%type, 'DIAGONAL', inform%status )
       call SPACE_resize_array( len, QPpred%H%val, inform%status, inform%alloc_status )
       QPpred%H%m   = len
       QPpred%H%n   = len
       QPpred%H%ne  = len

       if ( B_type == 0 ) then
          data%B%val = one
          do i = 1, n
             QPpred%H%val( i ) = one
             data%B%val(   i ) = one
          end do
       end if
       do i = n+1, np2m
          QPpred%H%val( i ) = zero
       end do
       if ( B_type == 2 ) then  ! L-BFGS
          do i = np2m + 1, np2m+L
             QPpred%H%val( i )   = -one
             QPpred%H%val( i+L ) =  one
          end do
       end if

    end if

    return

! Abnormal returns

990 continue
    write( error, 1001 ) inform%status, inform%alloc_status
    return

991 continue
    write( error, 1000 )
    return

! Format statements

1000 FORMAT(1X, '** ERROR trimsqp  : unrecognized storage type in subroutine build_QPpred.')
1001 FORMAT(1X, '** ERROR trimsqp : allocation error in subroutine', &
                ' build_QPpred : error= ', I0, ' status= ', I0, '.'  )

  END SUBROUTINE build_QPpred



!-*-   B U I L D _ Q P s q p  S U B R O U T I N E -*

  SUBROUTINE build_QPsiqp( nlp, QPsiqp, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!   Allocates components of variable QPsiqp of typ QPT_problem_type
!   and sets the sparsity patterns for the "H" and "A".
!   Storage type is determined by the storage in the corresponding
!   components in problem nlp of NLPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!---------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_data_type ), intent( inout )    :: data
    type( NLPT_problem_type ), intent( in )       :: nlp
    type( QPT_problem_type ), intent( out )       :: QPsiqp

!---------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------------------

    integer :: len, Ane, i, j, tally, descent_row
    character, allocatable, dimension( : ) :: store_type

    ! Define infinity

    data%control%QPsiqp_control%infinity = data%control%infinity / ten

    ! Set dimensions

    QPsiqp%m = nlp%m + nlp%m_a + 1
    QPsiqp%n = nlp%n + 2*nlp%m

    QPsiqp%A%m = nlp%m + nlp%m_a + 1 ;      QPsiqp%A%n = nlp%n + 2*nlp%m
    QPsiqp%H%m = nlp%n + 2*nlp%m ;          QPsiqp%H%n = nlp%n + 2*nlp%m

    ! Indicate new problem structure.

    QPsiqp%new_problem_structure = .true.

    ! Allocate components independent of storage type.

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%G, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_l, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_u, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%Z, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_status, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_l, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_u, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_status, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPsiqp%m, QPsiqp%Y, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    ! Prepare for hot starts.

    if ( QPsiqp%m > 0 ) then
       QPsiqp%C_status  = zero
    end if
    QPsiqp%X_status                                    = zero
    data%control%QPsiqp_control%QPA_control%cold_start = 0
    !data%control%QPsiqp_control%QPA_control%cold_start = 1

    !****************************************************************
    ! Allocate A: storage dependent on storage type in nlp%J.       !
    ! NB: nlp%J and nlp%A must be for the same storage type in nlp. !
    !****************************************************************

    if (nlp%m > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%J%type ), inform%status )
    elseif ( nlp%m_a > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%A%type ), inform%status )
    else
       call SMT_put( store_type, 'COORDINATE', inform%status ) ! dummy value DPR: maybe dense for descent constraint.
    end if

    SELECT CASE ( SMT_get( store_type ) )

    CASE ( 'DENSE' )

       ! allocate needed vectors
       !************************

       len = (nlp%m + nlp%m_a + 1) * nlp%n

       CALL SPACE_resize_array( len, QPsiqp%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%A%type, 'DENSE', inform%status )

    CASE ( 'COORDINATE' )

       ! allocate needed vectors
       !************************
       if ( nlp%m > 0 ) then
          if ( nlp%m_a > 0 ) then
             len =  nlp%J%ne + 2*nlp%m + nlp%A%ne + nlp%n
          else
             len =  nlp%J%ne + 2*nlp%m + nlp%n
          end if
       else
          if ( nlp%m_a > 0 ) then
             len =  nlp%A%ne + nlp%n
          else
             len =  nlp%n
          end if
       end if

       CALL SPACE_resize_array( len, QPsiqp%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       QPsiqp%A%ne = len

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%A%type, 'COORDINATE', inform%status  )

       Ane = 0

       ! first [ J  I  -I ].
       if ( nlp%m > 0 ) then
          QPsiqp%A%row( Ane + 1 : Ane + nlp%J%ne ) = nlp%J%row
          QPsiqp%A%col( Ane + 1 : Ane + nlp%J%ne ) = nlp%J%col
          Ane = nlp%J%ne
          do i = 1, nlp%m
             QPsiqp%A%row( Ane + i )         = i
             QPsiqp%A%col( Ane + i )         = nlp%n + i
             QPsiqp%A%row( Ane + nlp%m + i ) = i
             QPsiqp%A%col( Ane + nlp%m + i ) = nlp%n + nlp%m + i
          end do
          Ane = Ane + 2*nlp%m
       end if

       ! Now A.
       if ( nlp%m_a > 0 ) then
          QPsiqp%A%row( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%row + nlp%m
          QPsiqp%A%col( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%col
          Ane = Ane + nlp%A%ne
       end if

       ! the implied descent constraint.
       do i = 1, nlp%n
          QPsiqp%A%row( Ane + i ) = nlp%m + nlp%m_a + 1
          QPsiqp%A%col( Ane + i ) = i
       end do

    CASE ( 'SPARSE_BY_ROWS')

       ! allocate needed vectors
       !************************

       len =  nlp%J%ptr( nlp%m + 1 ) + nlp%A%ptr( nlp%m_a + 1 ) - 2 + nlp%n

       CALL SPACE_resize_array( len, QPsiqp%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( nlp%m + nlp%m_a + 2, QPsiqp%A%ptr, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%A%type, 'SPARSE_BY_ROWS', inform%status )

       tally = 1

       QPsiqp%A%ptr( 1 ) = 1

       ! the J part
       do i = 1, nlp%m
          do j = nlp%J%ptr( i ), nlp%J%ptr( i+1 ) - 1
             QPsiqp%A%col( tally ) = nlp%J%col( j )
             tally = tally + 1
          end do
          QPsiqp%A%ptr( i + 1 ) = tally
       end do

       ! the A part
       if ( nlp%m_a > 0 ) then
          do i = 1, nlp%m
             do j = nlp%A%ptr( i ), nlp%A%ptr( i+1 ) - 1
                QPsiqp%A%col( tally ) = nlp%A%col( j )
                tally = tally + 1
             end do
             QPsiqp%A%ptr( nlp%m + i + 1 ) = tally
          end do
       end if

       ! the imposed descent constraint.
       do i = 1, nlp%n
          QPsiqp%A%col( tally ) = i
       end do
       QPsiqp%A%ptr( nlp%m + nlp%m_a + 2 ) = tally + nlp%n

    CASE ( 'SPARSE_BY_COLUMNS' )

       ! allocate needed vectors
       !************************

       len =  nlp%J%ptr( nlp%n + 1 ) + nlp%A%ptr( nlp%n + 1 ) - 2 + nlp%n

       CALL SPACE_resize_array( len, QPsiqp%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( nlp%n + 1, QPsiqp%A%ptr, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%A%type, 'SPARSE_BY_COLUMNS', inform%status )

       descent_row = nlp%m + nlp%m_a + 1

       tally = 1

       QPsiqp%A%ptr( 1 ) = 1

       do j = 1, nlp%n

          ! J part
          do i = nlp%J%ptr( j ), nlp%J%ptr( j+1 ) - 1
             QPsiqp%A%row( tally ) = nlp%J%ptr( i )
             tally = tally + 1
          end do

          ! A part
          if ( nlp%m_a > 0 ) then
             do i = nlp%A%ptr( j ), nlp%A%ptr( j+1 ) - 1
                QPsiqp%A%row( tally ) = nlp%A%ptr( i )
                tally = tally + 1
             end do
          end if

          ! Imposed descent constraint
          QPsiqp%A%row ( tally ) = descent_row
          tally = tally + 1

          ! Set ptr.
          QPsiqp%A%ptr( j + 1 ) = tally

       end do

    CASE DEFAULT

       WRITE( data%control%error, 1000 )

    END SELECT

    !***************************************************************
    ! Allocate H: storage dependent on storage type in nlp%H.      !
    !***************************************************************

    SELECT CASE ( SMT_get( nlp%H%type ) )

    CASE ( 'DENSE' )

       ! allocate needed vectors
       !************************

       len = size( nlp%H%val )

       CALL SPACE_resize_array( len, QPsiqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%H%type, 'DENSE', inform%status )

    CASE ( 'COORDINATE' )

       ! allocate needed vectors
       !************************

       len =  nlp%H%ne

       CALL SPACE_resize_array( len, QPsiqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%H%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%H%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%H%type, 'COORDINATE', inform%status )

       QPsiqp%H%ne = len

       QPsiqp%H%row = nlp%H%row
       QPsiqp%H%col = nlp%H%col

    CASE ( 'SPARSE_BY_ROWS')

       ! allocate needed vectors
       !************************

       len =  nlp%H%ptr(nlp%n+1) - 1

       CALL SPACE_resize_array( len, QPsiqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%H%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( nlp%n + 1, QPsiqp%H%ptr, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%H%type, 'SPARSE_BY_ROWS', inform%status )

       QPsiqp%H%col = nlp%H%col
       QPsiqp%H%ptr = nlp%H%ptr

    CASE ( 'SPARSE_BY_COLUMNS' )

       ! allocate needed vectors
       !************************

       len =  nlp%H%ptr(nlp%n+1) - 1

       CALL SPACE_resize_array( len, QPsiqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPsiqp%H%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( nlp%n + 1, QPsiqp%H%ptr, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%H%type, 'SPARSE_BY_COLUMNS', inform%status )

       QPsiqp%H%row = nlp%H%row
       QPsiqp%H%ptr = nlp%H%ptr

    CASE ( 'DIAGONAL' )

       ! allocate needed vectors
       !************************

       len =  nlp%n

       CALL SPACE_resize_array( len, QPsiqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPsiqp%H%type, 'DIAGONAL', inform%status )

    CASE DEFAULT

       WRITE( data%control%error, 1000 )

    END SELECT

    inform%status = 0

    RETURN

! Abnormal returns

990 CONTINUE
    WRITE( data%control%error, 1001 ) inform%status, inform%alloc_status

    RETURN

! Format statements

1000 FORMAT(1X, '** ERROR trimsqp : unrecognized storage type in subroutine build_QPsiqp.')
1001 FORMAT(1X, '** ERROR trimsqp : allocation error in subroutine build_QPsiqp.',   &
            ' error= ', I0, ' status= ', I0, '.')

  END SUBROUTINE build_QPsiqp

!-*-   B U I L D _ Q P s e q p  S U B R O U T I N E -*

  SUBROUTINE build_QPseqp( nlp, QPseqp, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!   Allocates components of variable QPseqp of typ QPT_problem_type
!   and sets the sparsity patterns for the "H" and "A".
!   Storage type is determined by the storage in the corresponding
!   components in problem nlp of NLPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!---------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_data_type ), intent( inout )    :: data
    type( NLPT_problem_type ), intent( in )       :: nlp
    type( QPT_problem_type ), intent( out )       :: QPseqp

!---------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------------------

    integer :: len
    character, allocatable, dimension( : ) :: store_type

    !!!! Define infinity
    !!!! data%control%QPseqp_control%infinity = data%control%infinity / ten

    ! Set MAX dimensions.  Must reset these at fill time.
    QPseqp%m = nlp%m + nlp%m_a + nlp%n
    QPseqp%n = nlp%n

    !QPsiqp%A%m = nlp%m + nlp%m_a + 1 ;      QPsiqp%A%n = nlp%n + 2*nlp%m
    !QPsiqp%H%m = nlp%n + 2*nlp%m ;          QPsiqp%H%n = nlp%n + 2*nlp%m

    ! Indicate new problem structure.
    QPseqp%new_problem_structure = .true.

    QPseqp%gradient_kind = -1

    ! Allocate components independent of storage type.

    CALL SPACE_resize_array( QPseqp%n, QPseqp%G, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    !CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_l, inform%status, inform%alloc_status )
    !IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%n, QPseqp%X, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    !CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_u, inform%status, inform%alloc_status )
    !IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%n, QPseqp%Z, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%n, QPseqp%X_status, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    !CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_l, inform%status, inform%alloc_status )
    !IF ( inform%status /= GALAHAD_ok ) GO TO 990

    ! This is the constant term, NOT the constraints!
    CALL SPACE_resize_array( QPseqp%m, QPseqp%C, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    !CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_u, inform%status, inform%alloc_status )
    !IF ( inform%status /= GALAHAD_ok ) GO TO 990

    !CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_status, inform%status, inform%alloc_status )
    !IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%n, data%fr, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%n, data%fx, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( nlp%m_a, data%wA, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( nlp%m_a, data%wA_comp, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( nlp%m, data%wJ, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( nlp%m, data%wJ_comp, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    CALL SPACE_resize_array( QPseqp%m, QPseqp%Y, inform%status, inform%alloc_status )
    IF ( inform%status /= GALAHAD_ok ) GO TO 990

    ! Set constant in constraint to zero.

    QPseqp%C = zero


    !! Prepare for hot starts.
    !if ( QPsiqp%m > 0 ) then
    !   QPsiqp%C_status  = zero
    !!end if
    !QPsiqp%X_status                                    = zero
    !data%control%QPsiqp_control%QPA_control%cold_start = 0
    !data%control%QPsiqp_control%QPA_control%cold_start = 1

    !****************************************************************
    ! Allocate A: storage dependent on storage type in nlp%J.       !
    ! NB: nlp%J and nlp%A must be for the same storage type in nlp. !
    !****************************************************************

    if (nlp%m > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%J%type ), inform%status )
    elseif ( nlp%m_a > 0 ) then
       call SMT_put( store_type, SMT_get( nlp%A%type ), inform%status )
    else
       call SMT_put( store_type, 'COORDINATE', inform%status ) ! dummy value
    end if

    SELECT CASE ( SMT_get( store_type ) )

    CASE ( 'DENSE' )

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE ( 'COORDINATE' )

       ! allocate needed vectors - based on maximum possible.
       !*****************************************************

       if ( nlp%m > 0 ) then
          if ( nlp%m_a > 0 ) then
             len =  nlp%J%ne + nlp%A%ne + nlp%n
          else
             len =  nlp%J%ne + nlp%n
          end if
       else
          if ( nlp%m_a > 0 ) then
             len =  nlp%A%ne + nlp%n
          else
             len =  nlp%n
          end if
       end if

       CALL SPACE_resize_array( len, QPseqp%A%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPseqp%A%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPseqp%A%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       QPseqp%A%ne = len

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPseqp%A%type, 'COORDINATE', inform%status  )

!!$       !Do this in fill qpseqp
!!$       Ane = 0
!!$
!!$       ! first [ J  I  -I ].
!!$       if ( nlp%m > 0 ) then
!!$          QPsiqp%A%row( Ane + 1 : Ane + nlp%J%ne ) = nlp%J%row
!!$          QPsiqp%A%col( Ane + 1 : Ane + nlp%J%ne ) = nlp%J%col
!!$          Ane = nlp%J%ne
!!$          do i = 1, nlp%m
!!$             QPsiqp%A%row( Ane + i )         = i
!!$             QPsiqp%A%col( Ane + i )         = nlp%n + i
!!$             QPsiqp%A%row( Ane + nlp%m + i ) = i
!!$             QPsiqp%A%col( Ane + nlp%m + i ) = nlp%n + nlp%m + i
!!$          end do
!!$          Ane = Ane + 2*nlp%m
!!$       end if
!!$
!!$       ! Now A.
!!$       if ( nlp%m_a > 0 ) then
!!$          QPsiqp%A%row( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%row + nlp%m
!!$          QPsiqp%A%col( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%col
!!$          Ane = Ane + nlp%A%ne
!!$       end if
!!$
!!$       ! the implied descent constraint.
!!$       do i = 1, nlp%n
!!$          QPsiqp%A%row( Ane + i ) = nlp%m + nlp%m_a + 1
!!$          QPsiqp%A%col( Ane + i ) = i
!!$       end do

    CASE ( 'SPARSE_BY_ROWS')

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE ( 'SPARSE_BY_COLUMNS' )

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE DEFAULT

       WRITE( data%control%error, 1000 )

    END SELECT

    !***************************************************************
    ! Allocate H: storage dependent on storage type in nlp%H.      !
    !***************************************************************

    SELECT CASE ( SMT_get( nlp%H%type ) )

    CASE ( 'DENSE' )

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE ( 'COORDINATE' )

       ! allocate needed vectors
       !************************

       len =  nlp%H%ne

       CALL SPACE_resize_array( len, QPseqp%H%val, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPseqp%H%col, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990
       CALL SPACE_resize_array( len, QPseqp%H%row, inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 990

       ! fill needed componenents dependent on sparsity
       !***********************************************

       call SMT_put( QPseqp%H%type, 'COORDINATE', inform%status )

       QPseqp%H%ne = len

       ! These are not needed.
       QPseqp%H%m  = nlp%n
       QPseqp%H%n  = nlp%n

       QPseqp%H%row = nlp%H%row
       QPseqp%H%col = nlp%H%col

    CASE ( 'SPARSE_BY_ROWS')

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE ( 'SPARSE_BY_COLUMNS' )

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE ( 'DIAGONAL' )

       write(*,*) ' WARNING : TRIMSQP : build_QPseqp : not yet implemented!'

    CASE DEFAULT

       WRITE( data%control%error, 1000 )

    END SELECT

    inform%status = 0

    RETURN

! Abnormal returns

990 CONTINUE
    WRITE( data%control%error, 1001 ) inform%status, inform%alloc_status

    RETURN

! Format statements

1000 FORMAT(1X, '** ERROR trimsqp : unrecognized storage type in subroutine build_QPseqp.')
1001 FORMAT(1X, '** ERROR trimsqp : allocation error in subroutine build_QPseqp.',   &
            ' error= ', I0, ' status= ', I0, '.')

  END SUBROUTINE build_QPseqp



!-*-   F I L L _ Q P F E A S  S U B R O U T I N E -*

  SUBROUTINE fill_QPfeas( nlp, QPfeas, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Fills the components of LP with the correct data from nlp that finds
!   a "closest" point to our initial point that satisfies the linear
!   constraints.  See the documentation in LSQP to see the precise form
!   that is used.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( NLPT_problem_type ), intent( inout ) :: nlp
    type( QPT_problem_type ), intent( inout ) :: QPfeas

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    integer :: tally, i, j

    inform%status = -1

    QPfeas%f    = zero
    QPfeas%C_l  = nlp%A_l
    QPfeas%C_u  = nlp%A_u
    QPfeas%Y    = zero

    QPfeas%X_l  = nlp%X_l
    QPfeas%X_u  = nlp%X_u
    QPfeas%X    = nlp%X
    QPfeas%X0   = nlp%X
    QPfeas%Z    = zero

    do i = 1, QPfeas%n
       QPfeas%WEIGHT(i) = max( min( abs( QPfeas%X(i) ), 100.0_wp ), one )
    end do

    ! Load QPfeas%A with the portion from A.

    SELECT CASE ( SMT_get( nlp%A%type ) )

    CASE ('DENSE')

       do i = 1, nlp%m_a
          QPfeas%A%val( 1+(i-1)*nlp%n : i*nlp%n ) = nlp%A%val( 1+(i-1)*nlp%n : i*nlp%n )
          QPfeas%A%row( 1+(i-1)*nlp%n : i*nlp%n ) = i
          do j = 1, nlp%n
             QPfeas%A%col( (i-1)*nlp%n + j ) = j
          end do
       end do

    CASE ('SPARSE_BY_ROWS')

       tally = 1

       do i = 1, nlp%m_a
          do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
             QPfeas%A%row( tally ) = i
             QPfeas%A%col( tally ) = nlp%A%col( j )
             QPfeas%A%val( tally ) = nlp%A%val( j )
             tally = tally + 1
          end do
       end do

    CASE ('SPARSE_BY_COLUMNS')

       tally = 1

       do j = 1, nlp%n
          do i = nlp%A%ptr(j), nlp%A%ptr(j+1)-1
             QPfeas%A%row( tally ) = nlp%A%row( i )
             QPfeas%A%col( tally ) = j
             QPfeas%A%val( tally ) = nlp%A%val( i )
             tally = tally + 1
          end do
       end do

    CASE('COORDINATE')

       QPfeas%A%val( 1:nlp%A%ne ) = nlp%A%val
       QPfeas%A%row( 1:nlp%A%ne ) = nlp%A%row
       QPfeas%A%col( 1:nlp%A%ne ) = nlp%A%col

    CASE DEFAULT

       write( control%error, 1000 )

    END SELECT

    inform%status = 0

    return

! Format statements

1000 FORMAT(1X, '** ERROR : unrecognized storage type in subroutine fill_QPfeas.')

  END SUBROUTINE fill_QPfeas

!-*-   F I L L _ Q P s t e e r  S U B R O U T I N E -*

  SUBROUTINE fill_QPsteer( nlp, QPsteer, inform, control, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Fills the components of LP with the correct data from nlp that finds
!   a "closest" point to our initial point that satisfies the linear
!   constraints.  See the documentation in LSQP to see the precise form
!   that is used.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( TRIMSQP_data_type ), intent( in ) :: data
    type( NLPT_problem_type ), intent( inout ) :: nlp
    type( QPT_problem_type ), intent( inout ) :: QPsteer

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

    integer :: tally, i, j, m, m_a, n

    inform%status = -1

    m = nlp%m
    m_a = nlp%m_a
    n = nlp%n


    QPsteer%f = zero

    QPsteer%G = zero
    QPsteer%G( n+1 : ) = one

    QPsteer%C_l( : m )  = nlp%C_l-nlp%C;    QPsteer%C_l( m+1 : m + m_a )  = nlp%A_l-nlp%Ax
    QPsteer%C_u( : m )  = nlp%C_u-nlp%C;    QPsteer%C_u( m+1 : m + m_a )  = nlp%A_u-nlp%Ax

    QPsteer%C_status = 0

    QPsteer%Y = zero

    ! X variables.

    do i = 1, n
       QPsteer%X_l( i ) = max( nlp%X_l(i)-nlp%X(i), -data%TRpred )
       QPsteer%X_u( i ) = min( nlp%X_u(i)-nlp%X(i),  data%TRpred )
    end do

    ! U and V variables.

    QPsteer%X_l( n+1 : ) = zero
    QPsteer%X_u( n+1 : ) = data%control%infinity

    !QPsteer%X = zero
    QPsteer%X = zero
    do i = 1, m
       if ( nlp%C(i) < nlp%C_l(i) ) then
          QPsteer%X(n+i)   = nlp%C_l(i) - nlp%C(i)
          QPsteer%X(n+m+i) = zero
       elseif ( nlp%C(i) > nlp%C_u(i) ) then
          QPsteer%X(n+i)   = zero
          QPsteer%X(n+m+i) = nlp%C(i) - nlp%C_u(i)
       end if
    end do

    QPsteer%X_status = 0

!    QPsteer%X0   = nlp%X

    QPsteer%Z    = zero

 !   do i = 1, QPsteer%n
 !      QPsteer%WEIGHT(i) = max( min( abs( QPsteer%X(i) ), 100.0_wp ), one )
 !   end do

    ! Load QPsteer%A with the portions from | J I -I |
    !                                       | I 0  0 |

    ! First the J part.

    SELECT CASE ( SMT_get( nlp%J%type ) )

    CASE ('DENSE')

       do i = 1, nlp%m_a
          QPsteer%A%val( 1+(i-1)*nlp%n : i*nlp%n ) = nlp%A%val( 1+(i-1)*nlp%n : i*nlp%n )
          QPsteer%A%row( 1+(i-1)*nlp%n : i*nlp%n ) = i
          do j = 1, nlp%n
             QPsteer%A%col( (i-1)*nlp%n + j ) = j
          end do
       end do

    CASE ('SPARSE_BY_ROWS')

       tally = 1

       do i = 1, nlp%m_a
          do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
             QPsteer%A%row( tally ) = i
             QPsteer%A%col( tally ) = nlp%A%col( j )
             QPsteer%A%val( tally ) = nlp%A%val( j )
             tally = tally + 1
          end do
       end do

    CASE ('SPARSE_BY_COLUMNS')

       tally = 1

       do j = 1, nlp%n
          do i = nlp%A%ptr(j), nlp%A%ptr(j+1)-1
             QPsteer%A%row( tally ) = nlp%A%row( i )
             QPsteer%A%col( tally ) = j
             QPsteer%A%val( tally ) = nlp%A%val( i )
             tally = tally + 1
          end do
       end do

    CASE('COORDINATE')

       QPsteer%A%val( 1:nlp%J%ne ) = nlp%J%val
       QPsteer%A%row( 1:nlp%J%ne ) = nlp%J%row
       QPsteer%A%col( 1:nlp%J%ne ) = nlp%J%col

    CASE DEFAULT

       write( control%error, 1000 )

    END SELECT

    ! Now the I -I part.

    tally = nlp%J%ne

    do i = 1, m

       QPsteer%A%val( tally + i ) = one
       QPsteer%A%row( tally + i ) = i
       QPsteer%A%col( tally + i ) = n + i

       QPsteer%A%val( tally + m + i ) = -one
       QPsteer%A%row( tally + m + i ) = i
       QPsteer%A%col( tally + m + i ) = n + m + i

    end do

    ! Finally, the A part.

    SELECT CASE ( SMT_get( nlp%A%type ) )

    CASE ('DENSE')

       do i = 1, nlp%m_a
          QPsteer%A%val( 1+(i-1)*nlp%n : i*nlp%n ) = nlp%A%val( 1+(i-1)*nlp%n : i*nlp%n )
          QPsteer%A%row( 1+(i-1)*nlp%n : i*nlp%n ) = i
          do j = 1, nlp%n
             QPsteer%A%col( (i-1)*nlp%n + j ) = j
          end do
       end do

    CASE ('SPARSE_BY_ROWS')

       tally = 1

       do i = 1, nlp%m_a
          do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
             QPsteer%A%row( tally ) = i
             QPsteer%A%col( tally ) = nlp%A%col( j )
             QPsteer%A%val( tally ) = nlp%A%val( j )
             tally = tally + 1
          end do
       end do

    CASE ('SPARSE_BY_COLUMNS')

       tally = 1

       do j = 1, nlp%n
          do i = nlp%A%ptr(j), nlp%A%ptr(j+1)-1
             QPsteer%A%row( tally ) = nlp%A%row( i )
             QPsteer%A%col( tally ) = j
             QPsteer%A%val( tally ) = nlp%A%val( i )
             tally = tally + 1
          end do
       end do

    CASE('COORDINATE')

       tally = tally + 2*m

       QPsteer%A%val( tally + 1 : tally + nlp%A%ne ) = nlp%A%val
       QPsteer%A%row( tally + 1 : tally + nlp%A%ne ) = m + nlp%A%row
       QPsteer%A%col( tally + 1 : tally + nlp%A%ne ) = nlp%A%col

    CASE DEFAULT

       write( control%error, 1001 )

    END SELECT

    inform%status = 0

    return

! Format statements

1000 FORMAT(1X, '** ERROR : unrecognized storage type nlp%J%type in subroutine fill_QPsteer.')
1001 FORMAT(1X, '** ERROR : unrecognized storage type nlp%A%type in subroutine fill_QPsteer.')

  END SUBROUTINE fill_QPsteer

!-*-   F I L L _ Q P _ p r e d  S U B R O U T I N E -*

  SUBROUTINE fill_QPpred( nlp, QPpred, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Fills the components of QPpred with the correct data from nlp that
!   is needed to solve for the "predictor" step.  See the documentation
!   in TRIMSQP for more information.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout ) :: inform
    type( TRIMSQP_data_type ), intent( inout )      :: data
    type( NLPT_problem_type ), intent( inout )   :: nlp
    type( QPT_problem_type ), intent( inout )    :: QPpred

!-------------------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------

    integer :: tally, i, j, l, Ane, m, n, m_a, np2m, B_type, error, out
    integer :: Ltn, A_ind, B_ind, iterate
    real( kind = wp ) :: Bij, di, Bsi, theta, stBs, damp_factor
    real( kind = wp ) :: B_lammin, B_lammax, dummy_real

    B_lammin = tenm5
    B_lammax = tenp8

    ! For convenience

    n       = nlp%n
    m       = nlp%m
    m_a     = nlp%m_a
    np2m    = n + 2*m
    out     = data%control%out
    error   = data%control%error
    L       = data%control%L_BFGS_number
    B_type  = data%control%B_type
    Ltn     = L*n
    iterate = data%iterate

    ! Fill constants.
    !****************

    QPpred%f = zero

    ! Fill vectors.
    !**************

    ! G
    QPpred%G( 1 : n )        = nlp%G
    QPpred%G( n + 1 : np2m ) = data%penalty

    ! C_l, C_u, and Y
    if ( m > 0 ) then
       QPpred%C_l( 1 : m )  = nlp%C_l - nlp%C
       QPpred%C_u( 1 : m )  = nlp%C_u - nlp%C
       QPpred%Y  ( 1 : m )  = nlp%Y
    end if
    if ( m_a > 0 ) then
       QPpred%C_l( m + 1 : m + m_a )  = nlp%A_l - nlp%Ax
       QPpred%C_u( m + 1 : m + m_a )  = nlp%A_u - nlp%Ax
       QPpred%Y(   m + 1 : m + m_a )  = nlp%Y_a
    end if
    if ( B_type == 2 ) then  ! L-BFGS
       QPpred%Y( m + m_a + 1 : m + m_a + 2*L ) = zero
    end if

    ! X
    QPpred%X( 1 : n )  = zero
    if ( m > 0 ) then
       do i = 1, m
          QPpred%X( n + i )      = max( zero, nlp%C_l(i) - nlp%C(i) )
          QPpred%X( n + m + i )  = max( zero, nlp%C(i) - nlp%C_u(i) )
       end do
    end if
    if ( B_type == 2 ) then  ! L-BFGS
       QPpred%X( np2m + 1 : ) = zero
    end if

    ! C --- Js+u-v = u-v
    QPpred%C = zero
    QPpred%C( 1 : m ) = QPpred%X( n + 1 : n + m )
    QPpred%C( 1 : m ) = QPpred%C( 1 : m ) - QPpred%X( n + m + 1 : np2m )

    ! X_l and X_u
    do i = 1, n
       QPpred%X_l( i )  = max( nlp%X_l(i) - nlp%X(i), -data%TRpred )
       QPpred%X_l( i )  = min( zero, QPpred%X_l( i ) )  ! DPR: Do we want this?
       QPpred%X_u( i )  = min( nlp%X_u(i) - nlp%X(i),  data%TRpred )
       QPpred%X_u( i )  = max( zero, QPpred%X_u( i ) )  ! DPR: Do we want this?
    end do
    if ( m > 0 ) then
       QPpred%X_l( n + 1 : np2m )  = zero
       QPpred%X_u( n + 1 : np2m )  = data%control%QPpred_control%infinity
    end if

    ! Z
    QPpred%Z( 1 : n )         = nlp%Z
    QPpred%Z( n + 1 : np2m )  = zero
    if ( B_type == 2 ) then  ! L-BFGS
       QPpred%Z( np2m + 1 : ) = zero
    end if

    !--------------------------------------------------------------------
    ! Load QPpred%A with the portion from J, I, -I, A and maybe more if !
    ! limited memeory BFGS is being used.                               !
    ! NB: Assumes that nlp%A and nlp%J have the same storage format.    !
    !--------------------------------------------------------------------

    if ( QPpred%m > 0 ) then

       SELECT CASE ( SMT_get( QPpred%A%type ) )

       CASE ('DENSE')

          ! The first m rows.
          !**********************

          do i = 1, m

             tally = (i-1) * (n + 2*m) + 1

             ! The J part.

             do j = 1, n
                QPpred%A%val( tally ) = nlp%J%val( n*(i-1) + j )
                tally = tally + 1
             end do

             ! The I and -I part corresponding to (u,v) variables.

             do l = 1, i-1
                QPpred%A%val( tally )         = zero  ! u part
                QPpred%A%val( tally + m ) = zero  ! v part
                tally = tally + 1
             end do
             QPpred%A%val( tally )          =  one  ! u part
             QPpred%A%val( tally +  m ) = -one  ! v part
             do l = i+1, m
                QPpred%A%val( tally )         = zero  ! u part
                QPpred%A%val( tally + m ) = zero  ! v part
                tally = tally + 1
             end do

          end do

          ! The last m_a rows.
          !***********************

          if ( m_a > 0 ) then

             tally = m * (n + 2*m)

             QPpred%A%val( tally + 1 : tally + m_a * n ) = nlp%A%val

          end if

       CASE ('SPARSE_BY_ROWS')

          tally = 1

          QPpred%A%ptr( 1 ) = 1

          ! The first m rows.
          !**********************

          do i = 1, m

             ! ith row of J

             do j = nlp%J%ptr(i), nlp%J%ptr(i+1)-1
                QPpred%A%col( tally ) = nlp%J%col( j )
                QPpred%A%val( tally ) = nlp%J%val( j )
                tally = tally + 1
             end do

             ! ith row of  [ I -I]

             QPpred%A%col( tally ) = n + i
             QPpred%A%val( tally ) = one

             tally = tally + 1

             QPpred%A%col( tally ) = n + m + i
             QPpred%A%val( tally ) = -one

             tally = tally + 1

             ! Set ptr for row i.

             QPpred%A%ptr( i + 1 ) = tally + 1

          end do

          ! The last nlp%ma rows.
          !**********************

          if ( m_a > 0 ) then

             do i = 1, m_a

                do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
                   QPpred%A%col( tally ) = nlp%A%col( j )
                   QPpred%A%val( tally ) = nlp%A%val( j )
                   tally = tally + 1
                end do

                QPpred%A%ptr( m + 1 + i ) = tally + 1

             end do

          end if

       CASE ('SPARSE_BY_COLUMNS')

          tally = 1

          QPpred%A%ptr( 1 ) = 1

          ! The first n columns.
          !*************************

          do i = 1, n

             ! ith col of J

             do j = nlp%J%ptr(i), nlp%J%ptr(i+1)-1
                QPpred%A%row( tally ) = nlp%J%row( j )
                QPpred%A%val( tally ) = nlp%J%val( j )
                tally = tally + 1
             end do

             ! ith col of  A

             if ( m_a > 0 ) then

                do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
                   QPpred%A%row( tally ) = nlp%A%row( j )
                   QPpred%A%val( tally ) = nlp%A%val( j )
                   tally = tally + 1
                end do

             end if

             ! Set ptr for row i.

             QPpred%A%ptr( i + 1 ) = tally + 1

          end do

          ! The last 2*m columns.
          !**************************

          do i = 1, m

             QPpred%A%row( tally ) = i
             QPpred%A%val( tally ) = one
             tally = tally + 1

             QPpred%A%ptr( n + 1 + i ) = tally + 1

             QPpred%A%row( tally ) = i
             QPpred%A%val( tally ) = -one

             QPpred%A%ptr( n + 1 + i + m ) = tally + 1 + m

          end do

       CASE('COORDINATE')

          Ane = 0

          ! First do the [ J -I  I ] row.

          if ( m > 0 ) then
             QPpred%A%val( 1 : nlp%J%ne ) = nlp%J%val
             Ane = nlp%J%ne + 2*m
          end if

          ! Next, the A part.

          if ( m_a > 0 ) then
             QPpred%A%val( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%val
             Ane = Ane + nlp%A%ne
          end if

          ! Finally, the matrices for L-BFGS
          ! ********************************

          if ( data%control%B_type == 2 ) then

             ! Define diagonal B_0.

             if ( iterate == 1 ) then
                QPpred%H%val( : n ) = one
                data%B%val          = one
                go to 400
             else
                if ( data%sqp_computed ) then
                   dummy_real = max( B_lammin, data%Ss_H_Ss )
                   dummy_real = min( dummy_real, B_lammax )
                else
                   dummy_real = max( B_lammin, data%Sp_B_Sp )
                   dummy_real = min( dummy_real, B_lammax )
                end if
                QPpred%H%val( : n ) = dummy_real
                data%B%val          = dummy_real
             end if

             call get_L_BFGS( iterate, data%B, data%L_BFGS%A, data%L_BFGS%B,  &
                              data%L_BFGS%S, data%L_BFGS%Y, data%s_f,         &
                              data%L_BFGS%gradLx_new, data%L_BFGS%gradLx,     &
                              data%L_BFGS%SB_inner, data%L_BFGS%eta,          &
                              data%control%L_BFGS_curve_mod, data%L_BFGS%ind, &
                              data%control%L_BFGS_number, data%control%error, &
                              data%control%out )

!              ! Figure out which column is being added/replaced.

!              last = min( iterate - 1, L )

!              if ( iterate <= L+1 ) then
!                 ind( iterate - 1 ) = iterate - 1
!              else
!                 dummy_int     = ind( 1 )
!                 ind( 1: L-1 ) = ind( 2 : L )
!                 ind( L )      = dummy_int
!              end if

!              col = data%L_BFGS%ind( last )

!              ! Replace that column in S and Y.
!              S( :, col ) = s_f
!              Y( :, col ) = gradLx_new - gradLx

!              modified = .false.

!              do i = 1, last

!                 if ( i == last ) then
!                    B(:,ind(last)) = Y(:,ind(last)) / dot_product( Y(:,ind(last)), S(:,ind(last)) )**half
!                    do j = 1, last-1
!                       SBinner( ind(last), ind(j) ) = dot_product( S(:,ind(last)), B(:,ind(j)) )
!                    end do
!                 end if
!                 A(:,ind(i)) = zero
!                 call mop_Ax( one, Bzero, S(:,ind(i)), one, A(:,ind(i)), &
!                              out, control%error, transpose=.false. )
!                 do k = 1, i-1
!                    A(:,ind(i)) = A(:,ind(i)) - dot_product( A(:,ind(k)), S(:,ind(i)) ) * A(:,ind(k))
!                    A(:,ind(i)) = A(:,ind(i)) + SBinner(ind(i),ind(k)) * B(:,ind(k))
!                 end do
!                 Bs = A(:,ind(i))
!                 A(:,ind(i)) = Bs / ( dot_product( S(:,ind(i)), Bs ) )**half

!                 if ( i== last ) then

!                    ! Check if sufficiently positive definite.
!                    yts  = dot_product( Y(:,ind(last)), S(:,ind(last)) )
!                    stBs = dot_product( S(:,ind(last)), Bs )  ! s_k^T B_k s_k

!                    if ( yts < eta*stBs .and. .not. modified ) then
!                       if ( curve_mod == 0 ) then ! skip it
!                          theta = 0
!                       elseif ( curve_mod == 1) then ! % Powell
!                          theta          = (1-eta)*stBs/(stBs-yts)
!                          Y(:,ind(last)) = theta * Y(:,ind(last)) + (1-theta)*Bs
!                       else
!                          write(error, *)' ERROR:get_L_BFGS: disallowed value curve_mod.'
!                       end if

!                       B(:,ind(last)) = Y(:,ind(last)) / ( dot_product( Y(:,ind(last)), S(:,ind(last)) ) )**half
!                       do j= 1, last-1
!                          SBinner(ind(last),ind(j)) = dot_product( S(:,ind(last)), B(:,ind(j)) )
!                       end do

!                       ! Double check if sufficiently positive definite.
!                       yts = dot_product( Y(:,ind(last)), S(:,ind(last)) )
!                       sta = dot_product( S(:,ind(last)), A(:,ind(last)) )  ! s_k^T B_k s_k
!                       if ( yts < 0.5*eta*sta ) then
!                          write(error,*) 'ERROR:get_L_BFGS: curvature still wrong after modification!'
!                       end if
!                    end if

!                 end if

!              end do

!           do j = 1, L
!              do i = 1, n
!                 data%L_BFGS%A(i,j) = n*(j-1) + i
!              end do
!           end do
!           write(*,*) '4 col = ', data%L_BFGS%A( :, 4)
!           write(*,*) '5 row = ', data%L_BFGS%A( 5, : )
!           return
!          write(*,*) 'shape of A = ', shape( data%L_BFGS%A )
!          write(*,*) 'shape of B = ', shape( data%L_BFGS%B )
!          write(*,*) 'length QPred%A%val = ', shape( QPpred%A%val )

             ! Finally, define the rest of the constraints.

             A_ind = Ane
             B_ind = A_ind + Ltn + L
             do j = 1, min(iterate+1, L)
                QPpred%A%val( A_ind + 1 : A_ind + n ) = data%L_BFGS%A( :, j )
                QPpred%A%val( B_ind + 1 : B_ind + n ) = data%L_BFGS%B( :, j )
                A_ind = A_ind + n
                B_ind = B_ind + n
             end do

400          continue

          end if

       CASE DEFAULT

          write( error, 1000 )
          inform%status = GALAHAD_error_input_status
          return

       END SELECT

    end if

    !--------------------------------------------------
    ! Load QPpred%H with the portion from "B".        !
    ! B_type :  0 = identity, 1 = weighted diagonal,  !
    !           2 = L-BFGS, 3 = BFGS,                 !
    !           4 = exact H.                          !
    !--------------------------------------------------

    SELECT CASE ( B_type)

    CASE ( 0 )  ! "B" is Identity - diagonal storage.
    !************************************************
       ! Relax....all done in build_QPpred

    CASE ( 1 )  ! "B" is weighted diagonal - diagonal storage.
    !*********************************************************
       if ( data%iterate == 1 ) then
          QPpred%H%val( : n ) = one
          data%B%val          = one
       else
          if ( data%sqp_computed ) then
             dummy_real = max( B_lammin, data%Ss_H_Ss )
             dummy_real = min( dummy_real, B_lammax )
          else
             dummy_real = max( B_lammin, data%Sp_B_Sp )
             dummy_real = min( dummy_real, B_lammax )
          end if
          QPpred%H%val( : n ) = dummy_real
          data%B%val          = dummy_real
       end if

    CASE ( 2 )  ! L-BFGS (same as case 1)
    !************************************

       ! Relax .... B = B0 is set above in "jacobian"

!        if ( data%iterate == 1 ) then
!           QPpred%H%val( : n ) = one
!           data%B%val          = one
!        else
!           if ( data%sqp_computed ) then
!              dummy_real = max( B_lammin, data%Ss_H_Ss )
!              dummy_real = min( dummy_real, B_lammax )
!           else
!              dummy_real = max( B_lammin, data%Sp_B_Sp )
!              dummy_real = min( dummy_real, B_lammax )
!           end if
!           QPpred%H%val( : n ) = dummy_real
!           data%B%val          = dummy_real
!        end if

!        ! Load the matrices A and B of SMT type.

!        tally = 0
!        do j = 1, L
!           data%L_BFGS%A_smt%val( tally+1 : tally+n ) = data%L_BFGS%A( :, L )
!           data%L_BFGS%B_smt%val( tally+1 : tally+n ) = data%L_BFGS%B( :, L )
!           tally = tally + n
!        end do

    CASE ( 3 )  ! BFGS - coordinate storage.
    !***************************************

       if ( data%iterate == 1 ) then  ! Form the identity
          if ( data%control%print_level >= GALAHAD_DEBUG ) then
             write(out, "(/, 'BFGS - iterate = 1 - forming identity',/)" )
          end if
          data%B%val = zero
          tally = 0
          do i = 1, n
             tally = tally + i
             data%B%val(tally) = one
          end do
          data%QPpred%H%val( : data%B%ne ) = data%B%val

          data%BFGS%mod_type = 0

          goto 900
       end if

!       if ( data%step_accepted ) then  ! Changed Feb. 15, 2009
       if ( data%success >= 1 ) then

          ! Compute data needed to perform BFGS update.
          ! Note: all data used must be computed at end of previous iterate.

          data%BFGS%d = data%BFGS%gradLx_new - data%BFGS%gradLx

          if ( data%sqp_ratio_used ) then
             data%BFGS%std  = dot_product( data%s_f, data%BFGS%d )
             data%BFGS%Bs   = zero
             call mop_Ax( one, data%B, data%s_f, one, data%BFGS%Bs, &
                          out, error, symmetric=.true. )
             stBs = dot_product( data%BFGS%BS, data%s_f )
          else
             if ( data%sqp_computed .and. data%seqp_computed ) then
                data%BFGS%std  = dot_product( data%s_ac, data%BFGS%d )
                data%BFGS%Bs   = zero
                call mop_Ax( one, data%B, data%s_ac, one, data%BFGS%Bs, &
                             out, error, symmetric=.true. )
                stBs = dot_product( data%BFGS%BS, data%s_ac )
             else
                data%BFGS%std  = dot_product( data%s_c, data%BFGS%d )
                data%BFGS%Bs   = zero
                call mop_Ax( one, data%B, data%s_c, one, data%BFGS%Bs, &
                             out, error, symmetric=.true. )
                stBs = dot_product( data%BFGS%BS, data%s_c )
             end if
          end if

          ! Print the data

          if ( data%control%print_level >= GALAHAD_DEBUG ) then
             write( out, 1015 )
             write( out,  '(4(2x, ES15.8))') &
                  ( data%BFGS%d(i), data%BFGS%BS(i), data%BFGS%gradLx(i), data%BFGS%gradLx_new(i),  i = 1, n )
             write( out, 1016 ) data%BFGS%std, stBs
          end if

          ! Possibly skip the BFGS update if stBs is too small.

          if ( stBs <= tenm5 ) then

             if ( data%control%print_level >= GALAHAD_DEBUG ) then
                write( out, 1018 ) stBs
             end if

             data%QPpred%H%val( : data%B%ne ) = data%B%val ! Is this needed?
             goto 900

          end if

          ! Perform the BFGS update (currently only damping option available )

          damp_factor = data%BFGS%damp_factor

          if ( data%BFGS%std >= damp_factor * stBs ) then

             theta = one ;    data%BFGS%mod_type = 0

          else

             data%BFGS%mod_type = 1

             theta       = ( one - damp_factor )*stBs / ( stBs - data%BFGS%std )
             data%BFGS%d = theta*data%BFGS%d + (one-theta) * data%BFGS%Bs

             if ( data%sqp_ratio_used ) then
                data%BFGS%std  = dot_product( data%BFGS%d, data%s_f )
             else
                if ( data%sqp_computed .and. data%seqp_computed ) then
                   data%BFGS%std  = dot_product( data%BFGS%d, data%s_ac )
                else
                   data%BFGS%std  = dot_product( data%BFGS%d, data%s_c )
                end if
             end if

             if ( data%control%print_level >= GALAHAD_DEBUG ) then
                write( out, 1001 ) theta
             end if

             if ( data%control%print_level >= GALAHAD_DEBUG ) then
                write( out, 1017 )
                write( out,  '(4(2x, ES15.8))') &
                     ( data%BFGS%d(i), data%BFGS%BS(i), data%BFGS%gradLx(i), data%BFGS%gradLx_new(i),  i = 1, n )
                write( out, 1016 ) data%BFGS%std, stBs
             end if

          end if

          data%BFGS%theta = theta

          tally = 1
          do i = 1, n
             di  = data%BFGS%d(i)
             Bsi = data%BFGS%Bs(i)
             do j = 1, i
                Bij = data%B%val(tally)
                Bij = Bij - Bsi*data%BFGS%Bs(j) / stBs
                Bij = Bij + di*data%BFGS%d(j) / data%BFGS%std
                Bij = min( Bij, tenp9 )
                Bij = max( Bij, -tenp9 )
                data%B%val(tally) = Bij
                tally = tally + 1
             end do
          end do
          data%QPpred%H%val( : data%B%ne ) = data%B%val

       else ! Step not accepted - just refill.

          data%QPpred%H%val( : data%B%ne ) = data%B%val ! Is this needed?

       end if

    CASE ( 4 )  ! Exact
    !*******************
       write(*,*) 'not yet implemented - qppred_fill'

    CASE DEFAULT
    !************
       write( error, 1000  )
       inform%status = GALAHAD_error_input_status
       return

    END SELECT

900 continue

    ! Save Bval_revert if just beginning nonmonotone sequence.

    if ( data%NM%active .and. data%NM%num_fail == 0 ) then
       data%revert%Bval_revert = data%B%val
       ! Save L-BFGS vectors/matrix.
    end if

    return

! Format statements
1000 FORMAT(1X, '** ERROR : unrecognized storage type in subroutine fill_QPpred.')
1001 format(1x, 'BFGS : Damping used with theta = ', ES12.5 )
1015 format( /, '---------------------------------------------------------------',/,&
             1x,'Initial data needed for BFGS has been computed to be:',/ &
             10x, 'd', 15x, 'Bs', 11x, 'gradLxnew', 10x, 'gradLx', /          &
             6x, '---------', 7x, '---------', 8x, '---------', 8x, '----------' )
1016 format( /, 's^Td = ', ES16.9, 5x, 's^TBs = ', ES16.9, /, &
                '--------------------------------------------------------------')
1017 format( /, '---------------------------------------------------------------',/,&
             1x,'BFGS has been formed with the following data:',/ &
             10x, 'd', 15x, 'Bs', 11x, 'gradLxnew', 10x, 'gradLx', /          &
             6x, '---------', 7x, '---------', 8x, '---------', 8x, '----------' )
1018 format(1x, 'stBs = ', ES16.9, 3x, 'skipping BFGS update.' )

  END SUBROUTINE fill_QPpred



!-*-   F I L L _ Q P _ s q p   S U B R O U T I N E -*

  SUBROUTINE fill_QPsiqp( nlp, QPsiqp, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Fills the components of QPsiqp with the correct data from nlp that
!   is needed to solve for the "sqp correction" step.  See the
!   documentation in TRIMSQP for more information.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_data_type ), intent( in )       :: data
    type( NLPT_problem_type ), intent( inout )    :: nlp
    type( QPT_problem_type ), intent( inout )     :: QPsiqp
    !integer, intent( in ) :: correction_type


!-------------------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------

    integer :: tally, tally2, i, j, Ane, correction_type, m, m_a, n

    correction_type = data%control%correction_type

    m   = nlp%m
    m_a = nlp%m_a
    n   = nlp%n

    ! Fill constants.
    !****************

    ! F
    QPsiqp%f = zero !Irrelavant, but actually f_k + g_k^T s_c + half s_c ^T H_k s_c

    ! Fill vectors.
    !****************

    tally = 0

    select case (correction_type)

    case (0)
    !*******

       ! General constraints C and Y
       if ( m > 0 ) then
          !QPsiqp%C_l( : m ) = nlp%C_l( : m ) - ( nlp%C( : m ) + data%Jv )
          !QPsiqp%C_u( : m ) = nlp%C_u( : m ) - ( nlp%C( : m ) + data%Jv )
          QPsiqp%C_l( : m ) = nlp%C_l( : m ) - data%CplusJSc
          QPsiqp%C_u( : m ) = nlp%C_u( : m ) - data%CplusJSc
          QPsiqp%Y  ( : m ) = nlp%Y  ! DPR: should use cauchy mults?
          tally            = m
       end if

       ! Linear constraints C and Y
       if ( m_a > 0 ) then
          !QPsiqp%C_l( tally + 1 : tally + m_a )  = nlp%A_l - ( nlp%Ax + data%Av )
          !QPsiqp%C_u( tally + 1 : tally + m_a )  = nlp%A_u - ( nlp%Ax + data%Av )
          QPsiqp%C_l( tally + 1 : tally + m_a )  = nlp%A_l - data%AXplusSc
          QPsiqp%C_u( tally + 1 : tally + m_a )  = nlp%A_u - data%AXplusSc
          QPsiqp%Y(   tally + 1 : tally + m_a )  = nlp%Y_a  ! DPR: cauchy multipliers?
          tally                                 = m + m_a
       end if

       ! Descent condition C and Y.
       QPsiqp%C_l( tally + 1 )  = - data%control%infinity
       QPsiqp%C_u( tally + 1 )  = zero + tenm5 !two*epsmch**(0.1_wp) ! DPR: added for experiment
       QPsiqp%Y  ( tally + 1 )  = zero

       ! G
       QPsiqp%G( : n )             = data%GplusHs
!       QPsiqp%G( n + 1 : n + 2*m ) = data%SQP_penalty
       QPsiqp%G( n + 1 : n + 2*m ) = data%penalty

       ! X_l and X_u
       do i = 1, n

          QPsiqp%X_l( i )  = nlp%X_l(i) - ( nlp%X(i) + data%s_c(i) )   ! Suppose to be negative
          !QPsiqp%X_l( i )  = min( nlp%X_l(i), data%min_TRsqp )         ! ensure feasible region.
          QPsiqp%X_l( i )  = max( QPsiqp%X_l(i), - data%TRsqp )
          QPsiqp%X_l( i )  = min( QPsiqp%X_l(i), zero )  ! Do I want this?

          QPsiqp%X_u( i )  = nlp%X_u(i) - ( nlp%X(i) + data%s_c(i) ) ! Suppose to be positive.
          !QPsiqp%X_u( i )  = max( QPsiqp%X_u( i ), - data%min_TRsqp ) ! Ensure feasible region.
          QPsiqp%X_u( i )  = min( QPsiqp%X_u(i), data%TRsqp )
          QPsiqp%X_u( i )  = max( QPsiqp%X_u(i), zero )  ! Do I want this?

       end do

       QPsiqp%X_l( n + 1 : n + 2*m ) = zero                   ! (u,v)
       !QPsiqp%X_u( n + 1 : n + 2*m ) = data%control%infinity  ! (u,v)

       QPsiqp%X_u( n + 1 : n + 2*m ) = data%control%QPsiqp_control%infinity
       QPsiqp%X_u( n + data%sat(:data%num_sat) )     = zero   ! DPR : might want to do more here.
       QPsiqp%X_u( n + m + data%sat(:data%num_sat) ) = zero

       ! X
       QPsiqp%X(: n ) = zero

       !do i = 1, n
       !   QPsiqp%X( i ) = max( QPsiqp%X_l(i), zero )
       !   QPsiqp%X( i ) = min( QPsiqp%X_u(i), zero )
       !end do

       do i = 1, m
!!$          QPsiqp%X( n + i )     = max( zero, nlp%C_l(i) - data%CplusJSc(i) )
!!$          QPsiqp%X( n + m + i ) = max( zero, data%CplusJSc(i) - nlp%C_u(i) )

          QPsiqp%X( n + i )     = max( zero, nlp%C_l(i) - data%CplusJSc(i) )
          QPsiqp%X( n + m + i ) = max( zero, data%CplusJSc(i) - nlp%C_u(i) )

       end do

       ! Z
       QPsiqp%Z( : n )              = nlp%Z   ! DPR: cauchy multipliers?
       QPsiqp%Z ( n + 1 : n + 2*m ) = zero

    case( 1 )
    !*******

       ! General constraints C and Y
       if ( m > 0 ) then
          QPsiqp%C_l( : m )  = nlp%C_l( : m ) - data%C_cauchy
          QPsiqp%C_u( : m )  = nlp%C_u( : m ) - data%C_cauchy
          QPsiqp%Y  ( : m )  = nlp%Y  ! DPR: probably use mults from predictor.
          tally             = m
       end if

       ! Linear constraints C and Y
       if ( m_a > 0 ) then

          QPsiqp%C_l( tally + 1 : tally + m_a )  = nlp%A_l - ( nlp%Ax + data%AxSp )
          QPsiqp%C_u( tally + 1 : tally + m_a )  = nlp%A_u - ( nlp%Ax + data%AxSp )

          QPsiqp%Y( tally + 1 : tally + m_a )  = nlp%Y_a  ! DPR: again use predictor multipliers.

          tally = m + m_a

       end if

       ! Descent condition C and Y.
       QPsiqp%C_l( tally + 1 )  = - data%control%infinity
       QPsiqp%C_u( tally + 1 )  = zero + 0.0005_wp !two*epsmch**(0.1_wp) ! DPR: added for experiemnt
       QPsiqp%Y  ( tally + 1 )  = zero

       ! G
       QPsiqp%G( : n ) = data%GplusHs

       if ( m > 0 ) then
!          QPsiqp%G( n + 1 : n + 2*m )  = data%SQP_penalty
          QPsiqp%G( n + 1 : n + 2*m )  = data%penalty

       end if

       ! X
       QPsiqp%X  = zero   ! (x,u,v)

       if ( m > 0 ) then
          do i = 1, m
             QPsiqp%X( n + i )      = max( zero, nlp%C_l(i) - data%C_cauchy(i) )
             QPsiqp%X( n + m + i )  = max( zero, data%C_cauchy(i) - nlp%C_u(i) )
          end do
       end if

       ! X_l and X_u
       do i = 1, n

          QPsiqp%X_l( i )  = nlp%X_l(i) - ( nlp%X(i) + data%s_c(i) )
          QPsiqp%X_l( i )  = max( QPsiqp%X_l(i), - data%TRsqp )
          QPsiqp%X_l( i )  = min( QPsiqp%X_l(i), zero )  ! Do I want this?

          QPsiqp%X_u( i )  = nlp%X_u(i) - ( nlp%X(i) + data%s_c(i) )
          QPsiqp%X_u( i )  = min( QPsiqp%X_u(i), data%TRsqp )   ! DPR: probably want to ensure that the initial
                                                              !point is feasible by perturbing either
                                                              ! the bound or the intial point.
          QPsiqp%X_u( i )  = max( QPsiqp%X_u(i), zero )  ! Do I want this?

       end do

       if ( m > 0 ) then
          QPsiqp%X_l( n + 1 : n + 2*m ) = zero
          QPsiqp%X_u( n + 1 : n + 2*m ) = data%control%infinity
       end if

       ! Z
       QPsiqp%Z( 1 : n ) = nlp%Z   ! DPR: again from predictor multipliers

       if ( m > 0 ) then
          QPsiqp%Z( n + 1 : n + 2*m ) = zero
       end if

    case (2)
    !*******

       ! General constraints C and Y
       if ( m > 0 ) then
          QPsiqp%C_l( : m )  = nlp%C_l( : m ) - data%C_cauchy
          QPsiqp%C_u( : m )  = nlp%C_u( : m ) - data%C_cauchy
          QPsiqp%Y  ( : m )  = nlp%Y  ! DPR: probably use mults from predictor.
          tally             = m
       end if

       ! Linear constraints C and Y
       if ( m_a > 0 ) then

          QPsiqp%C_l( tally + 1 : tally + m_a )  = nlp%A_l - ( nlp%Ax + data%AxSp )
          QPsiqp%C_u( tally + 1 : tally + m_a )  = nlp%A_u - ( nlp%Ax + data%AxSp )

          QPsiqp%Y( tally + 1 : tally + m_a )  = nlp%Y_a  ! DPR: again use predictor multipliers.

          tally = m + m_a

       end if

       ! Descent condition C and Y.
       QPsiqp%C_l( tally + 1 )  = - data%control%infinity
       QPsiqp%C_u( tally + 1 )  = zero + 0.0005_wp !two*epsmch**(0.1_wp) ! DPR: added for experiemnt
       QPsiqp%Y  ( tally + 1 )  = zero

       ! G
       QPsiqp%G( : n ) = data%GplusHs

       if ( m > 0 ) then
!          QPsiqp%G( n + 1 : n + 2*m )  = data%SQP_penalty
          QPsiqp%G( n + 1 : n + 2*m )  = data%penalty
       end if

       ! X
       QPsiqp%X  = zero   ! (x,u,v)

       if ( m > 0 ) then
          do i = 1, m
             QPsiqp%X( n + i )      = max( zero, nlp%C_l(i) - data%C_cauchy(i) )
             QPsiqp%X( n + m + i )  = max( zero, data%C_cauchy(i) - nlp%C_u(i) )
          end do
       end if

       ! X_l and X_u
       do i = 1, n

          QPsiqp%X_l( i )  = nlp%X_l(i) - ( nlp%X(i) + data%s_c(i) )
          QPsiqp%X_l( i )  = max( QPsiqp%X_l(i), - data%TRsqp )
          QPsiqp%X_l( i )  = min( QPsiqp%X_l(i), zero )  ! Do I want this?

          QPsiqp%X_u( i )  = nlp%X_u(i) - ( nlp%X(i) + data%s_c(i) )
          QPsiqp%X_u( i )  = min( QPsiqp%X_u(i), data%TRsqp )   ! DPR: probably want to ensure that the initial
                                                              !point is feasible by perturbing either
                                                              ! the bound or the intial point.
          QPsiqp%X_u( i )  = max( QPsiqp%X_u(i), zero )  ! Do I want this?

       end do

       if ( m > 0 ) then
          QPsiqp%X_l( n + 1 : n + 2*m ) = zero
          QPsiqp%X_u( n + 1 : n + 2*m ) = data%control%infinity
!          QPsiqp%X_u( n + 1 : n + 2*m ) = 1000_wp * QPsiqp%X( n + 1 : n + 2*m )
       end if

       ! Z
       QPsiqp%Z( 1 : n ) = nlp%Z   ! DPR: again from predictor multipliers

       if ( m > 0 ) then
          QPsiqp%Z( n + 1 : n + 2*m ) = zero
       end if

    end select

    QPsiqp%C_status( 1 : nlp%m + nlp%m_a ) = data%QPpred%C_status
    QPsiqp%X_status(1:nlp%n)               = data%QPpred%X_status(1:nlp%n)

    !---------------------------------------------------------------------
    ! Load QPsiqp%A with the portion from J, I, -I, A, and the descent condition. !
    ! NB: Assumes that nlp%A and nlp%J have the same storage format.      !
    !----------------------------------------------------------------------

    SELECT CASE ( SMT_get( QPsiqp%A%type ) )

    CASE ('DENSE')

       tally  = m * n
       tally2 = m_a * m

       ! First the J part.

       QPsiqp%A%val( 1 : tally ) = nlp%J%val

       ! Next the A part.

       if ( m_a > 0 ) then
          QPsiqp%A%val( tally + 1 : tally + tally2 ) = nlp%A%val
          tally = tally + tally2
       end if

       ! Finally the imposed descent constrain.

       QPsiqp%A%val( tally + 1 : tally + n ) = nlp%G + data%descent_con

    CASE ('SPARSE_BY_ROWS')

       tally = 1

       QPsiqp%A%ptr( 1 ) = 1

       ! First do the J, which is the first m rows.

       do i = 1, m

          do j = nlp%J%ptr(i), nlp%J%ptr(i+1)-1
             QPsiqp%A%val( tally ) = nlp%J%val( j )
             tally = tally + 1
          end do

       end do

       ! Next do the A part, which is the next m_a rows.

       if ( m_a > 0 ) then

          do i = 1, m_a

             do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
                QPsiqp%A%val( tally ) = nlp%A%val( j )
                tally = tally + 1
             end do

          end do

       end if

       ! Finally do the descent constraint, which is the final constraint.

       QPsiqp%A%val( tally : tally + n ) = nlp%G + data%descent_con

    CASE ('SPARSE_BY_COLUMNS')

       tally = 1

       QPsiqp%A%ptr( 1 ) = 1

       ! Loop over the n columns.
       !*****************************

       do i = 1, n

          ! ith col of J

          do j = nlp%J%ptr(i), nlp%J%ptr(i+1)-1
             QPsiqp%A%val( tally ) = nlp%J%val( j )
             tally = tally + 1
          end do

          ! ith col of  A

          if ( m_a > 0 ) then

             do j = nlp%A%ptr(i), nlp%A%ptr(i+1)-1
                QPsiqp%A%val( tally ) = nlp%A%val( j )
                tally = tally + 1
             end do

          end if

          ! ith col of descent condition.

          QPsiqp%A%val( tally : tally + n ) = nlp%G + data%descent_con
          tally = tally + n

       end do

    CASE('COORDINATE')

       ! First do the [ J  I -I ] part

       Ane = 0

       if ( m > 0 ) then
          Ane = nlp%J%ne
          if ( correction_type == 2 ) then
             QPsiqp%A%val( 1 : Ane ) = data%J_cauchy
          else
             QPsiqp%A%val( 1 : Ane ) = nlp%J%val
          end if
          QPsiqp%A%val( Ane + 1 : Ane + m )       =   one
          QPsiqp%A%val( Ane + m + 1 : Ane + 2*m ) = - one
          Ane = Ane + 2*m
       end if

       ! Next the A part.

       if ( m_a > 0 ) then
          QPsiqp%A%val( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%val
          Ane = Ane + nlp%A%ne
       end if

       ! Finally the imposed descent condition

       !QPsiqp%A%val( Ane + 1 : Ane + n ) = data%GplusHs
       QPsiqp%A%val( Ane + 1 : Ane + n ) = data%descent_con


    CASE DEFAULT

       write( data%control%error, 1000 )

    END SELECT

    !-------------------------------------------
    ! Load QPsiqp%H with nlp%H                  !
    !-------------------------------------------

    select case ( SMT_get(data%QPsiqp%H%type) )

    case ('COORDINATE')

       QPsiqp%H%val = nlp%H%val

    case default

       write(*,*) 'fill_QPsiqp : not yet implemented'
    end select

    inform%status = 0

    return


! Format statements

1000 FORMAT(1X, '** ERROR : unrecognized storage type in subroutine fill_QPsiqp.')


  END SUBROUTINE fill_QPsiqp

!-*-   F I L L _ Q P _ s e q p   S U B R O U T I N E -*

  SUBROUTINE fill_QPseqp( nlp, QPseqp, inform, data )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Fills the components of QPseqp with the correct data from nlp that
!   is needed to solve for the "sqp correction" step.  See the
!   documentation in TRIMSQP for more information.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_data_type ), intent( inout )       :: data
    type( NLPT_problem_type ), intent( inout )    :: nlp
    type( QPT_problem_type ), intent( inout )     :: QPseqp


!-------------------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------

    integer :: nfr, nfx, nwA, nwA_comp, nwJ, nwJ_comp, i, j, tally, out, nvl
    integer, dimension( max(max(nlp%m, nlp%n), nlp%m_a ) ) :: vl

    out = data%control%out

    inform%status = -1

    ! Compute variables and constraints that are active and inactive.
    ! ***************************************************************

    ! First the variables.

    nfr = 0
    nfx = 0

!!$    write(*,*) 'QPpred_X_stat = ',data%QPpred%X_status
!!$    write(*,*) 'X = ', nlp%X
!!$    write(*,*) 's_p = ', data%s_p
!!$    write(*,*) 'c = ', nlp%C
!!$    write(*,*) 'JxSp = ', data%JxSp
!!$    write(*,*) 'QPpred_C_stat = ',data%QPpred%C_status
!!$    write(*,*) 'pred full sol = ', data%QPpred%X
!!$    write(*,*) 'A( x + sp ) = ', nlp%Ax + data%AxSp

 if ( data%seqp_use_pred ) then

    do i = 1, nlp%n
       if ( data%QPpred%X_status(i) /= 0 ) then
          select case ( data%X_type(i) )
          case ('LB')
             if ( abs(nlp%X(i) + data%s_p(i) - nlp%X_l(i))/(one + abs(nlp%X_l(i))) < data%control%stop_p ) then
                nfx            = nfx + 1
                data%fx( nfx ) = i
             else
                nfr            = nfr + 1
                data%fr( nfr ) = i
             end if
          case ('UB')
             if ( abs(nlp%X(i) + data%s_p(i) - nlp%X_u(i))/(one + abs(nlp%X_u(i))) < data%control%stop_p ) then
                nfx            = nfx + 1
                data%fx( nfx ) = i
             else
                nfr            = nfr + 1
                data%fr( nfr ) = i
             end if
          case ('FR')
             nfr            = nfr + 1
             data%fr( nfr ) = i
          case ('EQ')
             nfx            = nfx + 1
             data%fx( nfx ) = i
          case ('RB')
             if ( abs(nlp%X(i) + data%s_p(i) - nlp%X_u(i))/(one + abs(nlp%X_u(i))) < data%control%stop_p ) then
                nfx            = nfx + 1
                data%fx( nfx ) = i
             elseif ( abs(nlp%X(i) + data%s_p(i) - nlp%X_l(i))/(one + abs(nlp%X_l(i))) < data%control%stop_p ) then
                nfx            = nfx + 1
                data%fx( nfx ) = i
             else
                nfr            = nfr + 1
                data%fr( nfr ) = i
             end if
          case default
             write( out, 1000 )
             inform%status = -1
             return
          end select
       else
          nfr            = nfr + 1
          data%fr( nfr ) = i
       end if
    end do

 else

    call get_active( -data%X_RES_l, data%X_RES_u,                         &
                      data%s_ac + data%X_RES_l, data%X_RES_u - data%s_ac, &
                      data%X_type, nfx, data%fx, nfr, data%fr, nvl, vl,   &
                      point1*data%control%stop_p, out )

    data%fx( nfx+1 : nfx+nvl )  = vl( : nvl )
    nfx = nfx + nvl

 end if

 data%nfr = nfr
 data%nfx = nfx

 ! Next the linear constraints.

 nwA      = 0
 nwA_comp = 0

 if ( data%seqp_use_pred ) then
    do i = 1, nlp%m_a
       if ( data%QPpred%C_status( nlp%m + i ) /= 0 ) then
          nwA            = nwA + 1
          data%wA( nwA ) = i
       else
          nwA_comp                 = nwA_comp + 1
          data%wA_comp( nwA_comp ) = i
       end if
    end do
 else
    call get_active( -data%A_RES_l, data%A_RES_u,                               &
                     data%AxSac + data%A_RES_l, data%A_RES_u - data%AxSac,     &
                     data%A_type, nwA, data%wA, nwA_comp, data%wA_comp, nvl, vl,&
                     point1*data%control%stop_p, out )

    data%wA( nwA+1 : nwA+nvl )  = vl( : nvl )
    nwA = nwA + nvl
 end if

 data%nwA      = nwA
 data%nwA_comp = nwA_comp

 ! Finally, the general constraints.

 nwJ      = 0
 nwJ_comp = 0

 if ( data%seqp_use_pred ) then
    do i = 1, nlp%m
       if ( data%QPpred%C_status( i ) /= 0 ) then
          nwJ            = nwJ + 1
          data%wJ( nwJ ) = i
       else
          nwJ_comp                 = nwJ_comp + 1
          data%wJ_comp( nwJ_comp ) = i
       end if
    end do
 else
    call get_active( -data%C_RES_l, data%C_RES_u,                               &
                     data%JxSac + data%C_RES_l, data%C_RES_u - data%JxSac,     &
                     data%C_type, nwJ, data%wJ, nwJ_comp, data%wJ_comp, nvl, vl,&
                     point1*data%control%stop_p, out )

    data%wJ( nwJ+1 : nwJ+nvl )  = vl( : nvl )
    nwJ = nwJ + nvl
 end if

 data%nwJ      = nwJ
 data%nwJ_comp = nwJ_comp

 ! Now define the problem.
 ! ***********************

 ! The dimensions.

 QPseqp%n = nlp%n
 QPseqp%m = nwJ + nwA + nfx

 ! The QP Hessian, gradient, and constant.

 QPseqp%H%val = nlp%H%val

 if ( data%seqp_use_pred ) then
    QPseqp%G = nlp%G + data%HxSp
 else
    data%HxSac = zero
    call mop_Ax( one, nlp%H, data%s_ac, one, data%HxSac, &
                 out, data%control%error, symmetric=.true. )
    QPseqp%G = nlp%G + data%HxSac
 end if

 QPseqp%f = zero

 ! Constant in the constraints.

 !QPseqp%C = zero  ---- set in build_QPseqp

 ! Starting point

 QPseqp%X = zero

 ! The multipliers

 QPseqp%Y = zero

 ! Now the constraint matrix A.
 ! ****************************

 tally = 0

 ! First the predicted active nonlinear constraints.

 J_1: do j = 1, nlp%J%ne
    I_1: do i = 1, nwJ
       if ( nlp%J%row( j ) == data%wJ( i ) ) then
          tally = tally + 1
          QPseqp%A%row( tally ) = i
          QPseqp%A%col( tally ) = nlp%J%col( j )
          QPseqp%A%val( tally ) = nlp%J%val( j )
          exit I_1
       end if
    end do I_1
 end do J_1

 ! Now the active linear constraints.

 J_2: do j = 1, nlp%A%ne
    I_2: do i = 1, nwA
       if ( nlp%A%row( j ) == data%wA( i ) ) then
          tally = tally + 1
          QPseqp%A%row( tally ) = nwJ + i
          QPseqp%A%col( tally ) = nlp%A%col( j )
          QPseqp%A%val( tally ) = nlp%A%val( j )
          exit I_2
       end if
    end do I_2
 end do J_2

 ! Finally the fixed variables.

 do i = 1, nfx
    tally = tally + 1
    QPseqp%A%row( tally ) = nwJ + nwA + i
    QPseqp%A%col( tally ) = data%fx( i )
    QPseqp%A%val( tally ) = one
 end do

 ! The total

 QPseqp%A%ne = tally
 QPseqp%A%m = nwJ + nwA + nfx
 QPseqp%A%n = nlp%n
 !    write(*,*) 'tally = ', tally

 ! Set the trust-region.

 data%control%QPseqp_control%radius = data%TRsqp

 inform%status = 0

 return

1000 FORMAT(3X, '** ERROR : unrecognized X_type in subroutine fill_QPseqp.')

  end SUBROUTINE fill_QPseqp


!-*-   D E A L L O C _ Q P f e a s  S U B R O U T I N E -*

  SUBROUTINE dealloc_QPfeas( QPfeas, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Deallocates components of variable LP of type
!   QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( QPT_problem_type ), intent( inout )     :: QPfeas

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     character( len = 80 ) :: array_name


    ! Deallocate components independent of storage type.

    array_name = 'TRIMSQP : data%QPfeas%G'
    CALL SPACE_dealloc_array( QPfeas%G,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%X_l'
    CALL SPACE_dealloc_array( QPfeas%X_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%X'
    CALL SPACE_dealloc_array( QPfeas%X,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%X_u'
    CALL SPACE_dealloc_array( QPfeas%X_u,                                &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%X_status'
    CALL SPACE_dealloc_array( QPfeas%X_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%Z'
    CALL SPACE_dealloc_array( QPfeas%Z,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%C_l'
    CALL SPACE_dealloc_array( QPfeas%C_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%C'
    CALL SPACE_dealloc_array( QPfeas%C,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%C_u'
    CALL SPACE_dealloc_array( QPfeas%C_u,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%C_status'
    CALL SPACE_dealloc_array( QPfeas%C_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%Y'
    CALL SPACE_dealloc_array( QPfeas%Y,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%A%val'
    CALL SPACE_dealloc_array( QPfeas%A%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%A%row'
    CALL SPACE_dealloc_array( QPfeas%A%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%A%col'
    CALL SPACE_dealloc_array( QPfeas%A%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%A%ptr'
    CALL SPACE_dealloc_array( QPfeas%A%ptr,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPfeas%A%type'
    CALL SPACE_dealloc_array( QPfeas%A%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    inform%status = GALAHAD_ok

  END SUBROUTINE dealloc_QPfeas



!-*-   D E A L L O C _ Q P p r e d   S U B R O U T I N E -*

  SUBROUTINE dealloc_QPpred( QPpred, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-
!
!   Deallocates components of variable QPpred of type
!   QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( QPT_problem_type ), intent( inout )     :: QPpred

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     character( len = 80 ) :: array_name


    ! Deallocate components independent of storage type.

    array_name = 'TRIMSQP : data%QPpred%G'
    CALL SPACE_dealloc_array( QPpred%G,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%X_l'
    CALL SPACE_dealloc_array( QPpred%X_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%X'
    CALL SPACE_dealloc_array( QPpred%X,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%X_u'
    CALL SPACE_dealloc_array( QPpred%X_u,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%X_status'
    CALL SPACE_dealloc_array( QPpred%X_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%Z'
    CALL SPACE_dealloc_array( QPpred%Z,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%C_l'
    CALL SPACE_dealloc_array( QPpred%C_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%C'
    CALL SPACE_dealloc_array( QPpred%C,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%C_u'
    CALL SPACE_dealloc_array( QPpred%C_u,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%C_status'
    CALL SPACE_dealloc_array( QPpred%C_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%Y'
    CALL SPACE_dealloc_array( QPpred%Y,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%A%val'
    CALL SPACE_dealloc_array( QPpred%A%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%A%row'
    CALL SPACE_dealloc_array( QPpred%A%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%A%col'
    CALL SPACE_dealloc_array( QPpred%A%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%A%type'
    CALL SPACE_dealloc_array( QPpred%A%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%A%ptr'
    CALL SPACE_dealloc_array( QPpred%A%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%H%val'
    CALL SPACE_dealloc_array( QPpred%H%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%H%row'
    CALL SPACE_dealloc_array( QPpred%H%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%H%col'
    CALL SPACE_dealloc_array( QPpred%H%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%H%ptr'
    CALL SPACE_dealloc_array( QPpred%H%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPpred%H%type'
    CALL SPACE_dealloc_array( QPpred%H%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    inform%status = GALAHAD_ok

  END SUBROUTINE dealloc_QPpred



!-*-   D E A L L O C _ Q P s q p   S U B R O U T I N E -*

  SUBROUTINE dealloc_QPsiqp( QPsiqp, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-
!
!   Deallocates components of variable QPsiqp of type
!   QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( QPT_problem_type ), intent( inout )     :: QPsiqp

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     character( len = 80 ) :: array_name


    ! Deallocate components independent of storage type.

    array_name = 'TRIMSQP : data%QPsiqp%G'
    CALL SPACE_dealloc_array( QPsiqp%G,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%X_l'
    CALL SPACE_dealloc_array( QPsiqp%X_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%X'
    CALL SPACE_dealloc_array( QPsiqp%X,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%X_u'
    CALL SPACE_dealloc_array( QPsiqp%X_u,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%X_status'
    CALL SPACE_dealloc_array( QPsiqp%X_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%Z'
    CALL SPACE_dealloc_array( QPsiqp%Z,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%C_l'
    CALL SPACE_dealloc_array( QPsiqp%C_l,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%C'
    CALL SPACE_dealloc_array( QPsiqp%C,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%C_u'
    CALL SPACE_dealloc_array( QPsiqp%C_u,                               &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%C_status'
    CALL SPACE_dealloc_array( QPsiqp%C_status,                           &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%Y'
    CALL SPACE_dealloc_array( QPsiqp%Y,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%A%val'
    CALL SPACE_dealloc_array( QPsiqp%A%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%A%row'
    CALL SPACE_dealloc_array( QPsiqp%A%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%A%col'
    CALL SPACE_dealloc_array( QPsiqp%A%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%A%type'
    CALL SPACE_dealloc_array( QPsiqp%A%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%A%ptr'
    CALL SPACE_dealloc_array( QPsiqp%A%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%H%val'
    CALL SPACE_dealloc_array( QPsiqp%H%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%H%row'
    CALL SPACE_dealloc_array( QPsiqp%H%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%H%col'
    CALL SPACE_dealloc_array( QPsiqp%H%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%H%ptr'
    CALL SPACE_dealloc_array( QPsiqp%H%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPsiqp%H%type'
    CALL SPACE_dealloc_array( QPsiqp%H%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    inform%status = GALAHAD_ok

  END SUBROUTINE dealloc_QPsiqp

!-*-   D E A L L O C _ Q P s e q p   S U B R O U T I N E -*

  SUBROUTINE dealloc_QPseqp( QPseqp, inform, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-
!
!   Deallocates components of variable QPseqp of type
!   QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    type( TRIMSQP_inform_type ), intent( inout )  :: inform
    type( TRIMSQP_control_type ), intent( inout ) :: control
    type( QPT_problem_type ), intent( inout )     :: QPseqp

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     character( len = 80 ) :: array_name


    ! Deallocate components independent of storage type.

    array_name = 'TRIMSQP : data%QPseqp%G'
    CALL SPACE_dealloc_array( QPseqp%G,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%X'
    CALL SPACE_dealloc_array( QPseqp%X,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%C'
    CALL SPACE_dealloc_array( QPseqp%C,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%Y'
    CALL SPACE_dealloc_array( QPseqp%Y,                                 &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%A%val'
    CALL SPACE_dealloc_array( QPseqp%A%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%A%row'
    CALL SPACE_dealloc_array( QPseqp%A%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%A%col'
    CALL SPACE_dealloc_array( QPseqp%A%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%A%type'
    CALL SPACE_dealloc_array( QPseqp%A%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%A%ptr'
    CALL SPACE_dealloc_array( QPseqp%A%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%H%val'
    CALL SPACE_dealloc_array( QPseqp%H%val,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%H%row'
    CALL SPACE_dealloc_array( QPseqp%H%row,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%H%col'
    CALL SPACE_dealloc_array( QPseqp%H%col,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%H%ptr'
    CALL SPACE_dealloc_array( QPseqp%H%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    array_name = 'TRIMSQP : data%QPseqp%H%type'
    CALL SPACE_dealloc_array( QPseqp%H%type,                             &
         inform%status, inform%alloc_status, array_name = array_name,    &
         bad_alloc = inform%bad_alloc, out = control%error )
    IF ( control%deallocate_error_fatal .AND.  &
         inform%status /= GALAHAD_ok ) RETURN

    inform%status = GALAHAD_ok

  END SUBROUTINE dealloc_QPseqp


!-*-   p r i n t _ s m t   S U B R O U T I N E -*

  SUBROUTINE print_SMT( A, name, error, out, status )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Prints componenets of variable of type
!   smt_type.  Prints appropriate components which
!   is dependent on the storage type used.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!---------------------------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------------------------

    integer, intent( in ) :: error, out
    integer, intent( inout ) :: status
    character, intent( in ) :: name
    type( SMT_type ), intent( in )  :: A

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: i


    status = -1

    ! **********************
    !     Begin cases      !
    !***********************

    SELECT CASE ( SMT_get( A%type ) )


    CASE ('DENSE')

       ! print Ane, Aval, name, m, n,

       write( out, 2000 ) name
       write( out, 3000 ) A%m, A%type
       if ( allocated(A%id) ) then
          write( out, 3001 ) A%n, A%id
       else
          write( out, 3003 ) A%n
       end if
       write( out, 3002 ) A%ne

       write( out, '(/, T31, "val")')
       write( out, '(T22, ES19.10)')  ( A%val(i),  i = 1, A%ne )
!       write( out, '(T22, ES19.10)')  ( A%val(i),  i = 1, A%n*(A%n + 1)/2 )

    CASE ('SPARSE_BY_ROWS')

       write(*,*) 'not yet implemented : print_smt'


    CASE ('SPARSE_BY_COLUMNS')

       write(*,*) 'not yet implemented : print_smt'

    CASE('COORDINATE')

       ! print Arow, Acol, Aval

       write( out, 2000 ) name
       write( out, 3000 ) A%m, A%type
       if ( allocated(A%id) ) then
          write( out, 3001 ) A%n, A%id
       else
          write( out, 3003 ) A%n
       end if
       write( out, 3002 ) A%ne
       write( out, '(/, T17, "row", T28, "col", T45, "val")')
       write( out, '(T12, I8, T23, I8, T35, ES19.10)')  &
            ( A%row(i), A%col(i), A%val(i),  i = 1, A%ne )


    CASE ('DIAGONAL')

       ! print m, n, type, id, Aval

       write( out, 2000 ) name
       write( out, 3000 ) A%m, A%type
       if ( allocated( A%id ) ) then
          write( out, 3001 ) A%n, A%id
       else
          write( out, 3003 ) A%n
       end if
       write( out, '(/, T31, "val")')
       write( out, '(T22, ES19.10)')  ( A%val(i),  i = 1, A%n )


    CASE DEFAULT

       write( error, 1000 )

    END SELECT


    status = 0

    return

! Format statements

1000 FORMAT(3X, '** ERROR : unrecognized storage type in subroutine build_QPsiqp.')

2000 FORMAT( /,  &
     1X, '                    STATISTICS FOR MATRIX ', A, '                  ',/,  &
     1X, '                    -----------------------                        ' )

3000 FORMAT(/, T16, 'm  = ', I10, 7X, 'type = ', 60A )
3001 FORMAT(   T16, 'n  = ', I10, 7X, 'id   = ', 60A )
3002 FORMAT(   T16, 'ne = ', I10 )
3003 FORMAT(   T16, 'n  = ', I10 )


  END SUBROUTINE print_SMT


!-*-   c h e c k _ o p t i m a l   S U B R O U T I N E -*

  SUBROUTINE check_optimal( nlp, G, JtY, AtY_a, Z_in, Z_out, Y, Y_a,        &
                            C_type, A_type, X_type,                         &
                            C_RES_l, C_RES_u, A_RES_l, A_RES_u,             &
                            X_RES_l, X_RES_u, primal_vl, dual_vl, comp_vl,  &
                            exact, status, out )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Checks optimality for the given problem.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!---------------------------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------------------------

    type( NLPT_problem_type ), intent( inout )         :: nlp
    real( kind = wp ), dimension( : ), intent( in )    :: JtY, AtY_a, G
    real( kind = wp ), dimension( : ), intent( in )    :: Y, Y_a, Z_in
    real( kind = wp ), dimension( : ), intent( out )   :: Z_out
    character( len = 2 ), dimension( : ), intent( in ) :: C_type, A_type, X_type
    real( kind = wp ), dimension( : ), intent( in )    :: C_RES_l, C_RES_u
    real( kind = wp ), dimension( : ), intent( in )    :: A_RES_l, A_RES_u
    real( kind = wp ), dimension( : ), intent( in )    :: X_RES_l, X_RES_u
    real( kind = wp ), intent( out ) :: primal_vl, dual_vl, comp_vl
    logical, intent( in )  :: exact
    integer, intent( in )  :: out
    integer, intent( out ) :: status

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: i
    real( kind = wp ) :: Gi, vl, term1, term2, rel_res !sign, rel_res
    real( kind = wp ), dimension( size(Z_in) ) :: Z


    status = -1

    ! Primal violation.
    !******************

    primal_vl   = zero

    !write(*,*) 'Ctype = ', C_type
    !write(*,*) 'Cresl = ', C_RES_l
    !write(*,*) 'Cresu = ', C_RES_u

    ! First constraints nlp%C

    if ( nlp%m > 0 ) then
       do i = 1, nlp%m
          select case( C_type(i) )
          case('LB')
             rel_res   =  C_RES_l(i) / ( one + abs( nlp%C_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('UB')
             rel_res   =  C_RES_u(i) / ( one + abs( nlp%C_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('RB')
             rel_res   =  C_RES_l(i) / ( one + abs( nlp%C_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
             rel_res   =  C_RES_u(i) / ( one + abs( nlp%C_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('EQ')
             rel_res   =  abs( C_RES_l(i) ) / ( one + abs( nlp%C_l(i) ) )
             primal_vl =  min( primal_vl, - rel_res )
          case('FR')
             ! relax
          case default
             write( out, * ) 'ERROR: check_optimal : C_type = ? '
          end select
       end do
    end if

    ! Now linear constraints

    if ( nlp%m_a > 0 ) then
       do i = 1, nlp%m_a
          select case( A_type(i) )
          case('LB')
             rel_res   =  A_RES_l(i) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('UB')
             rel_res   =  A_RES_u(i) / ( one + abs( nlp%A_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('RB')
             rel_res   =  A_RES_l(i) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
             rel_res   =  A_RES_u(i) / ( one + abs( nlp%A_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('EQ')
             rel_res   =  abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, - rel_res )
          case('FR')
             ! relax
          case default
             write( out, * ) 'ERROR: check_optimal : A_type = ? '
          end select
       end do
    end if

    ! Finally the bounds.

    do i = 1, nlp%n
       select case( X_type(i) )
       case('LB')
          rel_res   =  X_RES_l(i) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('UB')
          rel_res   =  X_RES_u(i) / ( one + abs( nlp%X_u(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('RB')
          rel_res   =  X_RES_l(i) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, rel_res )
          rel_res   =  X_RES_u(i) / ( one + abs( nlp%X_u(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('EQ')
          rel_res   =  abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, - rel_res )
       case('FR')
          ! relax
       case default
          write( out, * ) 'ERROR: check_optimal : X_type = ? '
       end select
    end do

    primal_vl = abs( primal_vl )

    ! Dual violation
    !***************

    dual_vl = zero
    Z_out   = zero

    !write(*,*) 'G = ', G
    !write(*,*) 'JtY = ', JtY
    !write(*,*) 'AtY = ', AtY_a
    !write(*,*) 'Z = ', Z_in

    if ( exact ) then
       Z_out = G
       if ( nlp%m > 0 ) then
          Z_out = Z_out - JtY
       end if
       if ( nlp%m_a > 0 ) then
          Z_out = Z_out - AtY_a
       end if
       dual_vl = zero
       Z = Z_out
    else
       do i = 1, nlp%n

          Gi = G( i )
          vl = Gi

          if ( nlp%m > 0 ) then
             vl = vl - JtY( i )
          end if

          if ( nlp%m_a > 0 ) then
             vl = vl - AtY_a( i )
          end if

          vl = vl - Z_in( i )
          vl = abs(vl) / ( one + abs( Gi ) )
          !write(*,*) 'vl = ', vl
          dual_vl = max( dual_vl, vl )

       end do
       Z = Z_in
    end if
!write(*,*)
!!$    dual_vl = zero

!!$    ! Determine whether want "exact" dual = 0 or not.
!!$
!!$    if ( exact ) then
!!$       Z = G
!!$       if ( nlp%m > 0 ) then
!!$          Z = Z - JtY
!!$       end if
!!$       if ( nlp%m_a > 0 ) then
!!$          Z = Z - AtY_a
!!$       end if
!!$    else
!!$       Z = Z_in
!!$    end if
!!$
!!$    write(*,*) 'G = ', G
!!$    write(*,*) 'Jtv = ', JtY
!!$    write(*,*) 'Atv = ', AtY_a
!!$    write(*,*) 'Z_in  = ', Z_in
!!$    write(*,*) 'Z_out = ', Z
!!$
!!$
!!$    ! compute dual violation
!!$
!!$    do i = 1, nlp%n
!!$
!!$       Gi = G( i )
!!$       vl = Gi
!!$
!!$       if ( nlp%m > 0 ) then
!!$          vl = vl - JtY( i )
!!$       end if
!!$
!!$       if ( nlp%m_a > 0 ) then
!!$          vl = vl - AtY_a( i )
!!$       end if
!!$
!!$       vl = vl - Z( i )
!!$       vl = abs(vl) / ( one + abs( Gi ) )
!!$
!!$       dual_vl = max( dual_vl, vl )
!!$
!!$    end do

    ! Complementarity
    !****************

!!$    write(*,*) 'C_resl = ', C_RES_l
!!$    write(*,*) 'C_resu = ', C_RES_u
!!$    write(*,*) 'A_resl = ', A_RES_l
!!$    write(*,*) 'A_resu = ', A_RES_u
!!$    write(*,*) 'X_resl = ', X_RES_l
!!$    write(*,*) 'X_resu = ', X_RES_u
!!$    write(*,*) 'Y = ', Y
!!$    write(*,*) 'Y_a = ', Y_a
!!$    write(*,*) 'Z = ', Z


    comp_vl = zero

! First general constraints.

    do i = 1, nlp%m
       if ( Y(i) > zero ) then
          if ( C_type(i) == 'LB' .or. C_type(i) == 'RB'  ) then
             term1   = min( one, abs( C_RES_l(i) ) / ( one + abs( nlp%C_l(i))) )
             term2   = min( one, abs( Y(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( C_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y(i)) )
          end if
       elseif ( Y(i) < zero ) then
          if ( C_type(i) == 'UB' .or. C_type(i) == 'RB' ) then
             term1   = min( one, abs( C_RES_u(i) ) / ( one + abs( nlp%C_u(i))) )
             term2   = min( one, abs( Y(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( C_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y(i)) )
          end if
       end if
    end do


   ! Linear constraints nlp%A

    do i = 1, nlp%m_a
       if ( Y_a(i) > zero ) then
          if ( A_type(i) == 'LB' .or. A_type(i) == 'RB'  ) then
             term1   = min( one, abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i))) )
             term2   = min( one, abs( Y_a(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( A_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y_a(i)) )
          end if
       elseif ( Y_a(i) < zero ) then
          if ( A_type(i) == 'UB' .or. A_type(i) == 'RB' ) then
             term1   = min( one, abs( A_RES_u(i) ) / ( one + abs( nlp%A_u(i))) )
             term2   = min( one, abs( Y_a(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( A_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y_a(i)) )
          end if
       end if
    end do

! Bounds on variables.

    do i = 1, nlp%n
       if ( Z(i) > zero ) then
          if ( X_type(i) == 'LB' .or. X_type(i) == 'RB'  ) then
             term1   = min( one, abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i))) )
             term2   = min( one, abs( Z(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( X_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Z(i)) )
          end if
       elseif ( Z(i) < zero ) then
          if ( X_type(i) == 'UB' .or. X_type(i) == 'RB'  ) then
             term1   = min( one, abs( X_RES_u(i) ) / ( one + abs( nlp%X_u(i))) )
             term2   = min( one, abs( Z(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( X_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Z(i)) )
          end if
       end if
    end do


!!$    ! First general constraints.
!!$
!!$    if ( nlp%m > 0 ) then
!!$
!!$       do i = 1, nlp%m
!!$          select case( C_type(i) )
!!$          case ('LB')
!!$             term1   = min( one, abs( C_RES_l(i) ) / ( one + abs( nlp%C_l(i))) )
!!$             term2   = min( one, abs( Y(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, - Y(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          case ('UB')
!!$             term1   = min( one, abs( C_RES_u(i) ) / ( one + abs( nlp%C_u(i))) )
!!$             term2   = min( one, abs( Y(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, Y(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          case ('RB')
!!$             if ( nlp%C(i) <= (nlp%C_l(i) + nlp%C_u(i) ) / two ) then
!!$                term1   = min( one, abs(C_RES_l(i)) / ( one + abs( nlp%C_l(i))) )
!!$                term2   = min( one, abs( Y(i) ) )
!!$                comp_vl = max( comp_vl, term1 * term2 )
!!$                sign    = max( zero, - Y(i) )
!!$                comp_vl = max( comp_vl, sign )
!!$             else
!!$                term1   = min( one, abs(C_RES_u(i)) / ( one + abs( nlp%C_u(i))) )
!!$                term2   = min( one, abs( Y(i) ) )
!!$                comp_vl = max( comp_vl, term1 * term2 )
!!$                sign    = max( zero, Y(i) )
!!$                comp_vl = max( comp_vl, sign )
!!$             end if
!!$          case ('EQ')
!!$             ! relax
!!$          case ('FR')
!!$             comp_vl = max( comp_vl, abs( Y(i) ) )
!!$          case default
!!$             write( out, * ) 'ERROR: check_optimal : C_type = ? '
!!$          end select
!!$       end do
!!$    end if
!!$
!!$    ! Next linear constraints nlp%A
!!$
!!$    if ( nlp%m_a > 0 ) then
!!$
!!$       do i = 1, nlp%m_a
!!$          select case( A_type(i) )
!!$          case ('LB')
!!$             term1   = min( one, abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i))) )
!!$             term2   = min( one, abs( Y_a(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, - Y_a(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          case ('UB')
!!$             term1   = min( one, abs( A_RES_u(i) ) / ( one + abs( nlp%A_u(i))) )
!!$             term2   = min( one, abs( Y_a(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, Y_a(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          case ('RB')
!!$             if ( nlp%Ax(i) <= (nlp%A_l(i) + nlp%A_u(i) ) / two ) then
!!$                term1   = min( one, abs(A_RES_l(i)) / ( one + abs( nlp%A_l(i))) )
!!$                term2   = min( one, abs( Y_a(i) ) )
!!$                comp_vl = max( comp_vl, term1 * term2 )
!!$                sign    = max( zero, - Y_a(i) )
!!$                comp_vl = max( comp_vl, sign )
!!$             else
!!$                term1   = min( one, abs(A_RES_u(i)) / ( one + abs( nlp%A_u(i))) )
!!$                term2   = min( one, abs( Y_a(i) ) )
!!$                comp_vl = max( comp_vl, term1 * term2 )
!!$                sign    = max( zero, Y_a(i) )
!!$                comp_vl = max( comp_vl, sign )
!!$             end if
!!$          case ('EQ')
!!$             ! relax
!!$          case ('FR')
!!$             comp_vl = max( comp_vl, abs( Y_a(i) ) )
!!$          case default
!!$             write( out, * ) 'ERROR: check_optimal : A_type = ? '
!!$          end select
!!$       end do
!!$    end if
!!$
!!$    ! Finally bound constraints.
!!$
!!$    do i = 1, nlp%n
!!$       select case( X_type(i) )
!!$       case ('LB')
!!$          term1   = min( one, abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i) ) ) )
!!$          term2   = min( one, abs( Z(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, - Z(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('UB')
!!$          term1   = min( one, abs( X_RES_u(i) ) / ( one + abs( nlp%X_u(i) ) ) )
!!$          term2   = min( one, abs( Z(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, Z(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('RB')
!!$          if ( nlp%X(i) <= (nlp%X_l(i) + nlp%X_u(i) ) / two ) then
!!$             term1   = min( one, abs(X_RES_l(i)) / ( one + abs( nlp%X_l(i))) )
!!$             term2   = min( one, abs( Z(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, - Z(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          else
!!$             term1   = min( one, abs(X_RES_u(i)) / ( one + abs( nlp%X_u(i))) )
!!$             term2   = min( one, abs( Z(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, Z(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          end if
!!$       case ('EQ')
!!$          ! relax
!!$       case ('FR')
!!$          comp_vl = max( comp_vl, abs( Z(i) ) )
!!$       case default
!!$          write( out, * ) 'ERROR: check_optimal : X_type = ? '
!!$       end select
!!$    end do

    comp_vl = abs(comp_vl) ! to prevent -0.0000000000 from being printed.

    status = 0

    return

  END SUBROUTINE check_optimal



!-*-   G E T _ C A U C H Y _ S T E P   S U B R O U T I N E -*

  SUBROUTINE get_cauchy_step( m, C_type, f, g_s, s_hs, s_norm, rho_g,        &
                              RES_l, RES_u, A_s, A_norms, lbreak, IBREAK,    &
                              BREAKP, exact, out, print_1line, print_detail, &
                              print_debug, t_min, too_small, inform )

    ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    !
    !  Find the global minimizer of the function
    !
    !     1/2 x(T) H x + c(T) x
    !        + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
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
    !  RES_l is a REAL array of length m, that must be set by
    !   the user to the value of A x - c_l for all components which have
    !   lower bounds/are equalities
    !
    !  RES_u is a REAL array of length m, that must be
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
    !  IBREAK is an INTEGER workspace array of length lbreak
    !
    !  BREAKP is a REAL workspace array of length lbreak
    !
    !  lbreak is an INTEGER that must be at least 2m
    !
    !  t_min  is a REAL variable, which gives the required value of t on exit
    !
    !  inform is an INTEGER variable, which gives the exit status. Possible
    !   values are:
    !
    !    0     the minimizer given in t_min occurs between breakpoints after first
    !    1     the minimizer given in t_min occurs at the breakpoint indicated by
    !          the variable active
    !    2     the minimizer occurs at t_min = 1.
    !   -1     the minimizer given in t_min occurs before the first breakpoint
    !   -2     the function is unbounded from below. Ignore the value in t_min
    !   -3     the value m is negative. Ignore the value in t_min
    !
    ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    !  Dummy arguments

    INTEGER, INTENT( IN ) :: m, out, lbreak
    INTEGER, INTENT( OUT ) :: inform
    LOGICAL, INTENT( IN ) :: print_1line, print_detail, print_debug, exact
    REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, s_norm
    REAL ( KIND = wp ), INTENT( IN ) :: rho_g, too_small
    REAL ( KIND = wp ), INTENT( OUT ) :: t_min
    INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
    REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_norms
    !REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
    REAL ( KIND = wp ), INTENT( IN ),                                        &
         DIMENSION( m ) ::  RES_l
    REAL ( KIND = wp ), INTENT( IN ),                                        &
         DIMENSION( m ) ::  RES_u
    REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
    REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lbreak ) :: BREAKP
    CHARACTER ( Len = 2 ), INTENT( IN ), DIMENSION( m ) :: C_type

    !  Local variables

    INTEGER :: i, j, nbreak, inheap, iter, ibreakp, nbreak_total
    INTEGER :: cluster_start, cluster_end
    REAL ( KIND = wp ) :: as, res
    REAL ( KIND = wp ) :: val, slope, curv, slope_old, exact_val, exact_slope
    REAL ( KIND = wp ) :: t_break, t_star, feasep, epsqrt, infeas_g
    REAL ( KIND = wp ) :: fun, gradient, gradient_in, t_pert, t_old
    REAL ( KIND = wp ) :: pert_val, pert_eps, cosine, tiny_cosine, val_old
    REAL ( KIND = wp ) :: breakp_max, slope_infeas_g
    REAL ( KIND = wp ) :: fun_min

    LOGICAL :: beyond_first_breakpoint, recover
    CHARACTER ( LEN = 14 ) :: cluster

    t_min = zero
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
       t_pert = - epsmch  ! DPR: Probably should be positive.  Potential bug.
    END IF

    infeas_g   = zero ; slope_infeas_g = zero
    breakp_max = zero

    do i = 1, m

       select case ( C_type(i) )

       case ( 'EQ' )
       !**********************
       ! equality constraints
       !**********************

          res = RES_l( i )
          infeas_g = infeas_g + ABS( res )

          as = A_s( i )
          IF ( ABS( as ) < too_small ) CYCLE

          IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as
          IF ( res + t_pert * as > zero ) slope_infeas_g = slope_infeas_g + as

          !  Find if the step will change the status of the constraint

          t_break = - res / as

          IF ( ( as > zero .AND. res > zero ) .OR.                              &
               ( as < zero .AND. res < zero ) ) CYCLE
          cosine = ABS( as ) / ( s_norm * A_norms( i ) )
          IF ( print_debug .AND. cosine < tiny_cosine ) THEN
             WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                          &
                  i, res, as,  t_break, cosine
          END IF

          !  Find the breakpoint

          IF ( print_debug ) THEN
             WRITE( out, "( ' const EQ ', i5, 4ES18.10 )" )                         &
                  i, res, as,  t_break, cosine
          END IF

          if ( t_break <= one ) then
             nbreak = nbreak + 1
             IF ( res /= zero ) THEN
                IBREAK( nbreak ) = - i
             ELSE
                IBREAK( nbreak ) = i
             END IF
             BREAKP( nbreak ) = t_break
             breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
          end if

       case ( 'LB' )
       ! **********************************
       ! constraints with only lower bounds
       ! **********************************

          res = RES_l( i )
          IF ( res < zero ) infeas_g = infeas_g - res

          as = A_s( i )
          IF ( ABS( as ) < too_small ) CYCLE

          IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as

          !  Find if the step will change the status of the constraint

          t_break = - res / as

          IF ( ( as > zero .AND. res >= zero ) .OR.                              &
               ( as < zero .AND. res < zero ) ) CYCLE
          cosine = ABS( as ) / ( s_norm * A_norms( i ) )
          IF ( print_debug .AND. cosine < tiny_cosine ) THEN
             WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                           &
                  i, res, as,  t_break, cosine
          END IF

          !  Find the breakpoint

          IF ( print_debug ) THEN
             WRITE( out, "( ' const LB ', i5, 4ES18.10 )" )                         &
                  i, res, as,  t_break, cosine
          END IF

          if ( t_break <= one ) then
             nbreak           = nbreak + 1
             IBREAK( nbreak ) = i
             BREAKP( nbreak ) = t_break
             breakp_max       = MAX( breakp_max, BREAKP( nbreak ) )
          end if

       case ( 'UB' )
       ! **********************************
       ! constraints with only upper bounds
       ! **********************************

          res = RES_u( i )
          IF ( res < zero ) infeas_g = infeas_g - res

          as = - A_s( i )
          IF ( ABS( as ) < too_small ) CYCLE

          IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as
          ! IF ( res + t_pert * as < zero ) write(6,*) ' slope_term = u ', i

          !  Find if the step will change the status of the constraint

          t_break = - res / as

          IF ( ( as > zero .AND. res >= zero ) .OR.                              &
               ( as < zero .AND. res < zero ) ) CYCLE
          cosine = ABS( as ) / ( s_norm * A_norms( i ) )
          IF ( print_debug .AND. cosine < tiny_cosine ) THEN
             WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                           &
                  i, res, as,  t_break, cosine
          END IF

          !  Find the breakpoint

          IF ( print_debug ) THEN
             WRITE( out, "( ' const UB ', i5, 4ES18.10 )" )                         &
                  i, res, as,  t_break, cosine
          END IF

          if ( t_break <= one ) then
             nbreak = nbreak + 1
             IBREAK( nbreak ) = - i
             BREAKP( nbreak ) = t_break
             breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
          end if

       case ( 'RB' )
       ! **************************************************************
       ! constraints have BOTH upper and lower bounds (not an equality)
       ! **************************************************************

          do j = 1,2

             if ( j == 1 ) then
                res = RES_l( i )
                as = A_s(i)
             else
                res = RES_u( i )
                as = - A_s(i)
             end if
             IF ( res < zero ) infeas_g = infeas_g - res

!!$            if ( j == 1 ) then
!!$               as = A_s( i )
!!$            else
!!$               as = - A_s( i )
!!$            end if
             IF ( ABS( as ) < too_small ) CYCLE

             IF ( res + t_pert * as < zero ) slope_infeas_g = slope_infeas_g - as

             !  Find if the step will change the status of the constraint

             t_break = - res / as

             IF ( ( as > zero .AND. res >= zero ) .OR.                              &
                  ( as < zero .AND. res < zero ) ) CYCLE
             cosine = ABS( as ) / ( s_norm * A_norms( i ) )
             IF ( print_debug .AND. cosine < tiny_cosine ) THEN
                WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )                           &
                     i, res, as,  t_break, cosine
             END IF

             !  Find the breakpoint

             IF ( print_debug ) THEN
                WRITE( out, "( ' const RB ', i5, 4ES18.10 )" )                         &
                     i, res, as,  t_break, cosine
             END IF
             if ( t_break <= one ) then
                nbreak = nbreak + 1
                if ( j == 1 ) then
                   IBREAK( nbreak ) =   i
                else
                   IBREAK( nbreak ) = - i
                end if
                BREAKP( nbreak ) = t_break
                breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
             end if

          end do

       case ( 'FR' )
       ! ************************************
       ! constraints which are actually free.
       ! ************************************

          ! Relax....do nothing.

       case default
       ! ********************
       ! Error - invalid case
       ! ********************

          write( *, *) 'Error: unknown element of C_type'

       end select

    END DO

    nbreak_total = nbreak

    !  Record the initial function, slope and curvature

    val   = f + rho_g * infeas_g
    slope = g_s + rho_g * slope_infeas_g
    curv  = s_hs

    ! Give an intial gradient value coming IN (arbitrarily negative is fine).

    gradient_in = - one

    IF ( print_detail ) THEN
       CALL cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,   &
            RES_l, RES_u, A_s, zero,      &
            t_pert, too_small,                 &
            exact_val, exact_slope, exact )
       write( out, 2010 ) '  val', val, exact_val
       write( out, 2010 ) 'slope', slope, exact_slope
    END IF

    !  Record the function value and gradient at (just on the other side of)
    !  the initial point.

    fun = val ;    gradient = slope ;    recover = .FALSE. ;    fun_min = fun

    !  Order the breakpoints in increasing size using a heapsort.
    !  Build the heap.

    CALL SORT_heapsort_build( nbreak, BREAKP, inheap, ix = IBREAK )
    cluster_start = 1
    cluster_end = 0
    cluster = '      0      0'

    !  =======================================================================
    !  Start the main loop to find the global minimizer of the piecewise
    !  quadratic function for 0 <= t <= 1. Consider the problem over
    !  successive pieces.
    !  =======================================================================

    t_break = zero
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

       !  Check possible exit conditions.  If satisfied, exit.


       !  If the gradient of the unvariate function increases.

       IF ( gradient > gzero ) THEN
          if ( curv > - hzero ) then
             IF ( inform == 0 ) then
                t_min = min( t_break, one)
                inform = 1
                if ( print_debug ) then
                   write(out, "(' Minimum must occur at current break point.')")
                end if
             else
                if ( print_debug ) then
                   write(out, "(' Minimum occurs at 0.')")
                end if
             end IF
             EXIT
          else
             if ( gradient_in < - gzero ) then
                if ( print_detail .or. print_debug ) then
                   write( out, 2060 ) ! Possibly save it and then move on.
                   write( out, "('(fun_min,fun) = ', 2ES18.12)" )
                end if
                if ( fun < fun_min ) then
                   t_min   = t_break
                   fun_min = fun
                end if
             end if
          end if
       END IF

       !  If the gradient of the univariate function is small and its curvature
       !  is positive, exit.  Note: this forces exit with the minimizer closest
       !  to zero if the problem happens to be piecewise linear and have a
       !  flat piece, i.e. \_______/.

       IF ( ABS( gradient ) <= gzero ) THEN
          IF ( curv > - hzero ) THEN
             if ( inform == 0 ) then
                t_min  = min( t_break, one)  ! Changed this
                inform = 1
                if ( print_debug ) then
                   write( out, 2061 ) ! Solution found.
                end if
             else
                if ( print_debug ) then
                   write(out, "(' Minimum occurs at 0.')")
                end if
             end if
             EXIT
          END IF
       END IF

       !  Find the next breakpoint

       t_old = t_break
       IF ( nbreak > 0 ) THEN
          t_break = BREAKP( 1 )
          CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, ix = IBREAK )
          cluster_end = cluster_end + 1
          cluster_start = cluster_end
       ELSE
          t_break = biginf ! one
       END IF

       !  If curve > hzero, then we must have gradient < -gzero,
       !  so compute the line minimum.

       IF ( curv > hzero ) THEN
          IF ( print_detail ) WRITE( out, "( ' slope, curv ', 2ES12.4 )" )  &
                                                              slope,  curv
          t_star = - slope / curv

          !  If the line minimum occurs before the breakpoint, the line minimum
          !  gives the required minimizer, provided nbreak /= 0.  In this case,
          !  we must cut back to t = 1.  In either case, we then Exit.

          IF ( nbreak == 0 .OR. t_star < t_break ) THEN

             if ( t_star >= one ) then ! nbreak = 0

                if ( print_detail ) then
                   write( out, 2050 ) t_old, t_break, t_star
                   write( out, 2100 ) ! cutting back to t = 1.
                end if

                t_star = one

             else ! unique minimizer must occur at t_star.

                if ( print_detail ) then
                   write( out, 2050 ) t_old, t_break, t_star
                   write( out, 2062 )
                end if
             end if

             t_min = t_star

             !  Calculate the function value for the piecewise quadratic

             fun   = val + t_min * ( slope + half * t_min * curv )
             !val   = val + half * t_min * slope   ! DPR: ???

             if ( t_min < one ) then ! must be between break points.
                slope = slope + curv * t_min
                IF ( beyond_first_breakpoint ) inform = 0
             else
                inform = 2
                slope = slope + curv
             end if

             IF ( print_detail ) WRITE( out, 2000 )
             IF ( print_1line ) WRITE( out, &
                  "( 3X, I7, ES12.4, A14, 3ES12.4 )" ) &
                  iter, t_min, '      -      -', fun, slope, curv
             IF ( print_debug ) THEN
                exact_val = cauchy_get_val( m, C_type, f, g_s, s_hs, rho_g,     &
                     RES_l, RES_u, A_s, t_min, too_small )
                write( out, 2010 ) '  val', fun, exact_val
             END IF

             EXIT

          ELSE  ! minimum is past the break point, and nbreak /= 0.

             if ( print_detail ) then
                write( out, 2050 ) t_old, t_break, t_star
                write( out, "(' Proceeding on ...' )" )
             end IF

          END IF

       ELSE  ! curv <= hzero. Take full step to next break point.

          IF ( print_detail ) WRITE( out, 2040 ) t_old, t_break

          !  Possibly check the end point t = 1.

          if ( nbreak == 0 ) then
             if ( print_detail )  write( out, 2070 ) ! examining t = 1
             fun = val + slope + half * curv
             if ( fun < fun_min ) then
                t_min   = one
                fun_min = fun
                inform  = 2
                if ( print_detail ) write( out, 2080 )  ! global minimizer
             else
                inform = 1
                if ( print_detail ) then
                   write( out, 2090 ) t_min, fun_min ! not global minimizer
                end if
             end if
             if ( print_detail ) then
                write( out, "('Value at t = one is : ')" )
                write( out, 2000 )
                if ( print_1line ) then
                   write( out, "( 3X, I7, 5X, 'one', 22X, ES12.4)" ) iter, fun
                end if
             end if
             EXIT
          end if

       END IF

       ! Record the gradient coming INTO new breakpoint breakpoint.

       gradient_in = slope + curv * t_break

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

       feasep  = t_break + pert_val
       t_pert  = pert_val + pert_eps
       t_break = feasep

       IF ( t_break < breakp_max ) THEN

          DO
             ibreakp = IBREAK( nbreak )

             IF ( ibreakp < 0 ) THEN
                ibreakp = - ibreakp
                IF ( C_type( ibreakp ) == 'EQ' ) THEN
                   slope = slope + rho_g * ABS( A_s( ibreakp ) )
                END IF
             END IF

             !  Update the slope

             IF ( print_detail ) WRITE( out, 2020 )     &
                  'C', IBREAK( nbreak ), BREAKP( nbreak )
             slope = slope + rho_g * ABS( A_s( ibreakp ) )


             !  If the last breakpoint has been passed, exit

             nbreak = nbreak - 1
             IF( nbreak == 0 ) EXIT

             !  Determine if other terms become active at the breakpoint

             IF ( BREAKP( 1 ) >= feasep ) EXIT
             CALL SORT_heapsort_smallest( nbreak, BREAKP( : nbreak ), inheap,  &
                                          ix = IBREAK )
             cluster_end = cluster_end + 1

          END DO

          !  Compute the function value and gradient at (just on the other side
          !  of) the breakpoint

          fun = val + t_break * ( slope_old + half * t_break * curv )
          gradient = slope + t_break * curv
          WRITE( cluster, "( 2I7 )" ) cluster_start, cluster_end

          IF ( print_detail ) THEN

             call cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g, &
                                            RES_l, RES_u, A_s, t_break,     &
                                            t_pert, too_small,              &
                                            val, slope, exact )

             write( out, 2010 ) '  val', fun, exact_val
             write( out, 2010 ) 'slope', gradient, exact_slope
          END IF

          !  Fit the new quadratic so that it's value is fun at the breakpoint

          val = val + t_break * ( slope_old - slope )

          !  Check that the size of the line gradient has not shrunk significantly in
          !  the current segment of the piecewise arc. If it has, there may be a loss
          !  of accuracy, so the line derivative should be recomputed.

          IF ( ABS( slope ) < - epsqrt * slope_old ) THEN
             IF ( print_debug )                                                 &
                  WRITE( out, "( ' recompute line derivative ... ' )" )
             CALL cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,   &
                                            RES_l, RES_u, A_s, t_break,       &
                                            t_pert, too_small,                &
                                            val, slope, exact )

             gradient = slope + t_break * curv
             IF ( print_debug )                                                 &
                  WRITE( out, "( ' val, slope ', 2ES22.14 )" ) val, slope
          ENDIF

       ELSE  ! Essentially, all break points reached.

          !  Special case: all the remaining breakpoints are reached

          IF ( print_detail )  write( out, 2063 ) ! special case.

          !  Compute the function value and gradient at (just on the other side of)
          !  the breakpoint

          CALL cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,      &
                                         RES_l, RES_u, A_s, t_break,          &
                                         t_pert, too_small, val, slope, exact )

          gradient    = slope + t_break * curv
          nbreak      = 0
          cluster_end = nbreak_total
       END IF

       beyond_first_breakpoint = .TRUE.
       inform = 0  ! added by Daniel....mistake by Nick?

    !  ================
    !  End of main loop
    !  ================

    END DO

    !     IF ( inform == - 2 ) RETURN

    !  Check to ensure that rounding has not caused an increase in the objective
    !  value

    val = cauchy_get_val( m, C_type, f, g_s, s_hs, rho_g,     &
                          RES_l, RES_u, A_s, t_min, too_small )

    recover = f + rho_g * infeas_g + epsmch ** 0.33 <= val

    IF ( .NOT. recover ) RETURN

    write(*,*) 'get_cauchy : RECOVERY entered (not yet implemented)'
    inform = -4
    return

    IF ( print_detail ) WRITE( out,                                          &
         "( ' *** predicted vs actual function values =', /, 2ES22.14,       &
         &        /, ' .... being more careful ... ' )" )                             &
         f + rho_g * infeas_g, val

    !  ==========================================================================
    !  This part of the code is to cope with the possibility that rounding errors
    !  have so dominated the search that a descent point has not been found. A
    !  more cautious search will be performed.
    !  ==========================================================================

    iter = 0 ; t_break = zero ; t_min = zero

    !  Compute the initial function, slope and curvature
    call cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,   &
         RES_l, RES_u, A_s, zero,    &
         t_pert, too_small,          &
         val, slope, exact )
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
          if ( curv >= 0 ) then
             IF ( inform == 0 ) t_min = t_break
             EXIT
          else
             if ( gradient_in < -gzero ) then

             end if
          end if
       END IF


       !  If the gradient of the univariate function is small and its curvature
       !  is positive, exit

       IF ( ABS( gradient ) <= gzero ) THEN
          IF ( curv > - hzero ) THEN
             IF ( inform == 0 ) t_min = t_break
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
          t_break = one !biginf
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
             t_min = t_star
             IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star

             !  Calculate the function value for the piecewise quadratic

             val = cauchy_get_val( m, C_type, f, g_s, s_hs, rho_g,          &
                  RES_l, RES_u, A_s, t_break, too_small )


             !  If the function value has risen in the current interval, search the
             !  interval for a better value, and exit

             IF ( val_old < val ) THEN
!!$              CALL QPA_linesearch_interval( dims, n, m,                        &
!!$                                            f, g_s, s_hs, rho_g, rho_b,        &
!!$                                            X, X_l, X_u, RES_l, RES_u, S, A_s, &
!!$                                            t_old, val_old, t_min, val,        &
!!$                                            too_small, out, print_detail )
             END IF

             IF ( print_detail ) WRITE( out, 2000 )
             IF ( print_1line ) WRITE( out, &
                  "( 3X, I7, ES12.4, A14, 3ES12.4 )" ) &
                  iter, t_min, '      -      -', fun, zero, curv
             IF ( print_debug ) THEN
                exact_val = cauchy_get_val( m, C_type, f, g_s, s_hs, rho_g,          &
                     RES_l, RES_u, A_s, t_min, too_small )

                WRITE( out, 2010 ) '  val', fun, exact_val
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
             t_min = one !biginf
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
       call cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,   &
            RES_l, RES_u, A_s, t_break,    &
            t_pert, too_small,          &
            val, slope, exact )

       gradient = slope + t_break * curv

       IF ( print_debug )                                                     &
            WRITE( out, "( ' val_old, val ', 2ES22.14 )" ) val_old, val
       IF ( val_old < val ) THEN
          t_min = t_break
!!$          CALL QPA_linesearch_interval( dims, n, m,                            &
!!$                                        f, g_s, s_hs, rho_g, rho_b, X,         &
!!$                                        X_l, X_u, RES_l, RES_u, S, A_s,        &
!!$                                        t_old, val_old, t_min, val,            &
!!$                                        too_small, out, print_detail )
          EXIT
       END IF
       val_old = val

       beyond_first_breakpoint = .TRUE.



       !  =========================
       !  End of main recovery loop
       !  =========================

    END DO

    RETURN

    !  Non-executable statements

2000 FORMAT( /, '  **  iter break point      cluster      ', &
         ' fun       slope        curv ', / )
2010 FORMAT( 1X, A5, '(est,true) = ', 2ES22.14 )
2020 FORMAT( ' breakpoint for ', A1, '-term ', I7, ' reached, step = ', ES12.4)
    !2030 FORMAT( ' breakpoint for ', A1, '-term ', I7, ' is acceptable,',         &
    !             ' cosine = ', ES12.4 )
2040 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, ']' )
2050 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, &
         '], stationary point = ', ES12.4 )
2060 FORMAT( /, 'Indefinite : local minimum found. ', &
                'Documenting and moving on.', / )
2061 format(1x, 'Minimum (possibly closest minimum if problem is linear) ', &
                'must occur at current break point.' )
2062 format(1x, 'UNIQUE minimizer must occur at stationary point.' )
2063 format(1x, 'Entering special case : final break points are of order ', &
                'machine epsilon from current break point.' )
2070 FORMAT( /, 'Not positive definte, all remaining break points reached :', &
                ' checking end point t = 1.....' )
2080 FORMAT(    '.... is the global minimizer.', / )
2090 FORMAT(    '.... is NOT the global minimizer.  The minimizer occurs at', /&
                ' t_min = ', ES14.7, 3x, 'with a value of ', ES16.9, '.')
2100 FORMAT( /, 'All break points passed and positive definite : ', &
                'since stationary point is past t=1, the global ',  &
                'minimizer must occur at end point t = 1.' )

    ! End of subroutine get_cauchy_step

  END SUBROUTINE get_cauchy_step


!-*-*-*-*-*   C A U C H Y _ G E T _ V A L   F U N C T I O N   -*-*-*-*-*-*-*-

      FUNCTION cauchy_get_val( m, C_type, f, g_s, s_hs, rho_g,          &
                               RES_l, RES_u, A_s, t, too_small )
      REAL ( KIND = wp ) cauchy_get_val

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the penalty function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

!      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: m!, n
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho_g
      REAL ( KIND = wp ), INTENT( IN ) :: t, too_small
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
      !REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
      CHARACTER ( Len = 2 ), INTENT( IN ), DIMENSION( m ) :: C_type
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( m ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( m ) ::  RES_u

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: as, infeas_g

      infeas_g = zero

      do i = 1, m

         select case( C_type(i) )

         case ('EQ')
         ! equality constraints

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g + ABS( RES_l( i ) )
            ELSE
               infeas_g = infeas_g + ABS( RES_l( i ) + t * as )
            END IF

         case ('LB')
         ! constraints with lower bounds only

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_l( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
            END IF

         case ('UB')
         ! constraints with upper bounds only

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_u( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
            END IF

         case ('RB')
         ! constraints with both upper and lower bounds.  Not equality.

            as = A_s( i )

            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_l( i ), zero )
               infeas_g = infeas_g - MIN( RES_u( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
               infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
            END IF

         case ('FR')
         ! constraint is free.

            ! Relax

         case default

            write(*,*) 'error: unrecognizable C_type in get_val'

         end select

      end do

      cauchy_get_val = f + t * ( g_s + half * t * s_hs ) + rho_g * infeas_g

      RETURN

!  End of function cauchy_get_val

    END FUNCTION cauchy_get_val

!-*   C A U C H Y _ G  E T _ V A L _ A N D _ S L O P E   S U B R O U T I  N E -*-*

    SUBROUTINE cauchy_get_val_and_slope( m, C_type, f, g_s, s_hs, rho_g,   &
                                         RES_l, RES_u, A_s, t,    &
                                         t_pert, too_small,          &
                                         val, slope, exact )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value and slope (in the direction S) of the penalty function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

!      TYPE ( QPA_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: m!, n
      logical, intent( in ) :: exact
      CHARACTER ( Len = 2 ), DIMENSION( m ), INTENT( IN ) :: C_type
!      INTEGER, INTENT( IN ) :: m_link
!      INTEGER, INTENT( IN ), DIMENSION( m ) :: C_stat
!      INTEGER, INTENT( IN ), DIMENSION( m_link ) :: REF
!      INTEGER, INTENT( IN ), DIMENSION( n ) :: B_stat
      REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho_g
      REAL ( KIND = wp ), INTENT( IN ) :: t, t_pert, too_small
      REAL ( KIND = wp ), INTENT( OUT ) :: val, slope
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: A_s
!      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( m ) ::  RES_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( m ) ::  RES_u

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: tp, as, infeas_g, slope_g

      IF ( exact ) THEN
        tp = t
      ELSE
        tp = t + t_pert
      END IF
      infeas_g = zero ; slope_g = zero

      Do i = 1, m

         select case ( C_type(i) )

         case ('EQ')
         ! equality constraints

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g + ABS( RES_l( i ) )
            ELSE
               infeas_g = infeas_g + ABS( RES_l( i ) + t * as )
               IF ( RES_l( i ) + tp * as < zero ) THEN
                  slope_g = slope_g - as
               ELSE IF ( RES_l( i ) + tp * as > zero ) THEN
                  slope_g = slope_g + as
               END IF
            END IF

         case ('LB')
         ! constraints with only lower bounds

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_l( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
               IF ( RES_l( i ) + tp * as < zero ) THEN
                  slope_g = slope_g - as
               END IF
            END IF

         case ('UB')
         ! constraints with upper bounds

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_u( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
               IF ( RES_u( i ) - tp * as < zero ) THEN
                  slope_g = slope_g + as
               END IF
            END IF

         case ('RB')
         ! constraints bounded below and above.  Not equality.

            as = A_s( i )
            IF ( ABS( as ) < too_small ) THEN
               infeas_g = infeas_g - MIN( RES_l( i ), zero )
               infeas_g = infeas_g - MIN( RES_u( i ), zero )
            ELSE
               infeas_g = infeas_g - MIN( RES_l( i ) + t * as, zero )
               infeas_g = infeas_g - MIN( RES_u( i ) - t * as, zero )
               IF ( RES_l( i ) + tp * as < zero ) THEN
                  slope_g = slope_g - as
               ELSEIF ( RES_u( i ) - tp * as < zero ) THEN
                  slope_g = slope_g + as
               END IF
            END IF

         case ('FR')
         ! Free constraint

            ! Relax

         case default

            write(*,*) 'Error: cauchy_get_val_and_slope. C_type = ?'

         end select

      END DO

      val = f + t * ( g_s + half * t * s_hs ) + rho_g * infeas_g
      slope = ( g_s + t * s_hs ) + rho_g * slope_g

      RETURN

!  End of subroutine cauchy_get_val_and_slope

    END SUBROUTINE cauchy_get_val_and_slope



!-*   C H E C K _ S U B  P R O B L E M _ O P T I M A L   S U B R O U T I N E -*-*

    SUBROUTINE check_sub_optimal( nlp, Y, JtY, Y_a, AtY_a, Z,         &
                                  C_RES_l, C_RES_u,                   &
                                  A_RES_l, A_RES_u, X_RES_l, X_RES_u, &
                                  data, inform, out )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  tests optimality of the subproblem for the current penalty parameter.  If
!  the penalty parameter is large enough, then first-order points should be
!  first-order points of the orginal problem.
!r
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    type( TRIMSQP_inform_type ), intent( inout )    :: inform
    type( TRIMSQP_data_type ), intent( inout )      :: data
    type( NLPT_problem_type ), intent( inout )      :: nlp
    real( kind = wp ), intent( in ), dimension( : ) :: Y, Y_a
    real( kind = wp ), intent( in ), dimension( : ) :: JtY, AtY_a, Z
    real( kind = wp ), intent( in ), dimension( : ) :: C_RES_l, C_RES_u
    real( kind = wp ), intent( in ), dimension( : ) :: A_RES_l, A_RES_u
    real( kind = wp ), intent( in ), dimension( : ) :: X_RES_l, X_RES_u
    integer, intent( in ) :: out

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: i
    real( kind = wp ) :: Gi, vl !, zi, low, upp, abs_i, Yi, sign
    real( kind = wp ) :: primal_vl, dual_vl, comp_vl, rel_res
    real( kind = wp ) :: penalty, dummy_real, term1, term2 !, stop_c
    real( kind = wp ) :: sub_stop_p, sub_stop_c, subgrad_vl

    inform%status = -1

!!$    write(*,*) 'subroblem X = ', nlp%X
!!$    write(*,*) 'subroblem Z = ', nlp%Z
!!$    write(*,*) 'subproblem Y = ', data%QPpred%Y(1: nlp%m )
!!$    write(*,*) 'J = ', nlp%J%val
!!$    write(*,*) 'Jty = ', data%Jtv
!!$    write(*,*) 'C-type = ', data%C_type
!!$    write(*,*) 'penalty = ', data%penalty

    penalty  = data%penalty
    !stop_c =  data%sub_stop_c
    !stop_


    ! Primal violation.
    !******************

    primal_vl = zero

    ! First the linear constraints

    if ( nlp%m_a > 0 ) then
       do i = 1, nlp%m_a
          select case( data%A_type(i) )
          case('LB')
             rel_res   =  A_RES_l(i) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('UB')
             rel_res   =  A_RES_u(i) / ( one + abs( nlp%A_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('RB')
             rel_res   =  A_RES_l(i) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, rel_res )
             rel_res   =  A_RES_u(i) / ( one + abs( nlp%A_u(i) ) )
             primal_vl =  min( primal_vl, rel_res )
          case('EQ')
             rel_res   =  abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i) ) )
             primal_vl =  min( primal_vl, - rel_res )
          case('FR')
             ! relax
          case default
             write( data%control%out, * ) 'ERROR: check_optimal : A_type = ? '
          end select
       end do
    end if

    ! Finally the bounds.

    do i = 1, nlp%n
       select case( data%X_type(i) )
       case('LB')
          rel_res   =  X_RES_l(i) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('UB')
          rel_res   =  X_RES_u(i) / ( one + abs( nlp%X_u(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('RB')
          rel_res   =  X_RES_l(i) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, rel_res )
          rel_res   =  X_RES_u(i) / ( one + abs( nlp%X_u(i) ) )
          primal_vl =  min( primal_vl, rel_res )
       case('EQ')
          rel_res   =  abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i) ) )
          primal_vl =  min( primal_vl, - rel_res )
       case('FR')
          ! relax
       case default
          write( data%control%out, * ) 'ERROR: check_optimal : X_type = ? '
       end select
    end do

    primal_vl = abs( primal_vl )


    ! Dual violation
    !***************

    dual_vl = zero

    do i = 1, nlp%n

       Gi = nlp%G( i )
       vl = Gi

       if ( nlp%m > 0 ) then
          vl = vl - JtY( i )
       end if

       if ( nlp%m_a > 0 ) then
          vl = vl - AtY_a( i )
       end if

       vl = vl - Z( i )
       vl = abs(vl) / ( one + abs( Gi ) )

       dual_vl = max( dual_vl, vl )

    end do

    ! Subgradient violation.
    !***********************

    sub_stop_p = Data%sub_stop_p
    sub_stop_c = data%sub_stop_c

    subgrad_vl = zero

    do i = 1, nlp%m

       select case ( data%C_type( i ) )

       case ('EQ')

          if ( C_RES_l(i) < -sub_stop_p) then
             dummy_real = max( penalty*(one-sub_stop_c) - Y(i), zero )
             subgrad_vl = max( dummy_real, Y(i) - penalty*(one+sub_stop_c) )
          elseif ( C_RES_l(i) > sub_stop_p) then
             dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
             subgrad_vl = max( dummy_real, Y(i) - penalty*(-one+sub_stop_c) )
          else
             dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
             subgrad_vl = max( dummy_real, Y(i) - penalty*(one+sub_stop_c) )
          end if


!!$          if ( nlp%C(i) < nlp%C_l(i) - stop_c ) then
!!$             dummy_real = abs( Y(i) - penalty ) / ( one + penalty )
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          elseif ( nlp%C(i) > nlp%C_l(i) + stop_c ) then
!!$             dummy_real = abs( Y(i) + penalty ) / ( one + penalty )
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          else
!!$             comp_vl = max( comp_vl,  (Y(i) - penalty) / penalty )
!!$             comp_vl = max( comp_vl, -(Y(i) + penalty) / penalty )
!!$          end if

       case ('LB')

          if ( C_RES_l(i) > sub_stop_p ) then
             dummy_real = max( -sub_stop_c - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - sub_stop_c )
          elseif ( C_RES_l(i) < -sub_stop_p ) then
             dummy_real = max( penalty*(one-sub_stop_c) - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - penalty*(1+sub_stop_c) )
          else
             dummy_real = max( -sub_stop_c - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - penalty*(1+sub_stop_c) )
          end if

!!$          if ( nlp%C(i) < nlp%C_l(i) - stop_c ) then
!!$             dummy_real = abs( Y(i) - penalty ) / ( one + penalty )
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          elseif ( nlp%C(i) > nlp%C_l(i) + stop_c ) then
!!$             comp_vl = max( comp_vl, abs( Y(i) ) )
!!$          else
!!$             comp_vl = max( comp_vl, ( Y(i) - penalty ) / penalty )
!!$             comp_vl = max( comp_vl, - Y(i) )
!!$          end if

       case ('UB')

          if ( C_RES_u(i) > sub_stop_p ) then
             dummy_real = max( -sub_stop_c - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - sub_stop_c )
          elseif ( C_RES_u(i) < -sub_stop_p ) then
             dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) + penalty*(one-sub_stop_c) )
          else
             dummy_real = max( -penalty*(one + sub_stop_c) - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - sub_stop_c )
          end if

!!$          if ( nlp%C(i) > nlp%C_u(i) + stop_c ) then
!!$             dummy_real = abs( Y(i) + penalty ) / (one + penalty)
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          elseif ( nlp%C(i) < nlp%C_u(i) - stop_c ) then
!!$             comp_vl = max( comp_vl, abs( Y(i) ) )
!!$          else
!!$             comp_vl = max( comp_vl, -(Y(i) + penalty) / penalty )
!!$             comp_vl = max( comp_vl, Y(i) )
!!$          end if

       case ('RB')

          if ( C_RES_u(i) < -sub_stop_p ) then
             dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) + penalty*(one-sub_stop_c) )
          elseif ( C_RES_l(i) < -sub_stop_p ) then
             dummy_real = max( penalty*(one-sub_stop_c) - Y(i), zero )
             subgrad_vl = max(dummy_real, Y(i) - penalty*(1+sub_stop_c) )
          elseif ( abs(C_RES_u(i))  <= sub_stop_c ) then
             if ( abs(C_RES_l(i))  <= sub_stop_c ) then
                dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
                subgrad_vl = max(dummy_real, Y(i) - penalty*(one+sub_stop_c) )
             else
                dummy_real = max( -penalty*(one+sub_stop_c) - Y(i), zero )
                subgrad_vl = max(dummy_real, Y(i) - sub_stop_c )
             end if
          elseif ( abs(C_RES_l(i))  <= sub_stop_c ) then
             dummy_real = max( -sub_stop_c - Y(i), zero )
             subgrad_vl = max( dummy_real, Y(i) - penalty*(one+sub_stop_c) )
          else ! free
             subgrad_vl = max( subgrad_vl, abs( Y(i) ) - sub_stop_c )
          end if

!!$
!!$          if ( nlp%C(i) < nlp%C_l(i) - stop_c ) then
!!$             dummy_real = abs( Y(i) - penalty ) / ( one + penalty )
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          elseif ( nlp%C(i) > nlp%C_u(i) + stop_c ) then
!!$             dummy_real = abs( Y(i) + penalty ) / (one + penalty)
!!$             comp_vl    = max( comp_vl, dummy_real )
!!$          elseif ( ( nlp%C(i) > nlp%C_l(i) + stop_c ) .and. &
!!$                   ( nlp%C(i) < nlp%C_u(i) - stop_c ) ) then
!!$             comp_vl = max( comp_vl, abs( Y(i) ) )
!!$          elseif ( abs( nlp%C(i) - nlp%C_l(i) ) <= stop_c ) then
!!$             comp_vl = max( comp_vl,  (Y(i) - penalty) / penalty )
!!$             comp_vl = max( comp_vl, - Y(i) )
!!$          elseif ( abs( nlp%C(i) - nlp%C_u(i) ) <= stop_c ) then
!!$             comp_vl = max( comp_vl, -(Y(i) + penalty) / penalty )
!!$             comp_vl = max( comp_vl, Y(i) )
!!$          else
!!$             write( data%control%out, * ) ' check_subproblem_optimal :', &
!!$                  ' this never should happen!.'
!!$          end if

       case ('FR')

          subgrad_vl = max( subgrad_vl, abs( Y(i) ) - sub_stop_c )
!         comp_vl = max( comp_vl, abs( Y(i) ) )

       case default
          write( out, * ) 'ERROR: check_subproblem_optimal : unrecognized C_type value'

       end select

    end do

    ! Complementarity
    !****************

    comp_vl = zero

    ! Linear constraints nlp%A

    do i = 1, nlp%m_a
       if ( Y_a(i) > zero ) then
          if ( data%A_type(i) == 'LB' .or. data%A_type(i) == 'RB'  ) then
             term1   = min( one, abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i))) )
             term2   = min( one, abs( Y_a(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( data%A_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y_a(i)) )
          end if
       elseif ( Y_a(i) < zero ) then
          if ( data%A_type(i) == 'UB' .or. data%A_type(i) == 'RB' ) then
             term1   = min( one, abs( A_RES_u(i) ) / ( one + abs( nlp%A_u(i))) )
             term2   = min( one, abs( Y_a(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( data%A_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Y_a(i)) )
          end if
       end if
    end do

! Bounds on variables.

    do i = 1, nlp%n
       if ( Z(i) > zero ) then
          if ( data%X_type(i) == 'LB' .or. data%X_type(i) == 'RB'  ) then
             term1   = min( one, abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i))) )
             term2   = min( one, abs( Z(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( data%X_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Z(i)) )
          end if
       elseif ( Z(i) < zero ) then
          if ( data%X_type(i) == 'UB' .or. data%X_type(i) == 'RB'  ) then
             term1   = min( one, abs( X_RES_u(i) ) / ( one + abs( nlp%X_u(i))) )
             term2   = min( one, abs( Z(i) ) )
             comp_vl = max( comp_vl, term1 * term2 )
          elseif ( data%X_type(i) /= 'EQ' ) then
             comp_vl = max( comp_vl, abs( Z(i)) )
          end if
       end if
    end do


!!$    do i = 1, nlp%m_a
!!$       select case( data%A_type(i) )
!!$       case ('LB')
!!$          term1   = min( one, abs( A_RES_l(i) ) / ( one + abs( nlp%A_l(i))) )
!!$          term2   = min( one, abs( Y_a(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, - Y_a(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('UB')
!!$          term1   = min( one, abs( A_RES_u(i) ) / ( one + abs( nlp%A_u(i))) )
!!$          term2   = min( one, abs( Y_a(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, Y_a(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('RB')
!!$          if ( nlp%Ax(i) <= (nlp%A_l(i) + nlp%A_u(i) ) / two ) then
!!$             term1   = min( one, abs(A_RES_l(i)) / ( one + abs( nlp%A_l(i))) )
!!$             term2   = min( one, abs( Y_a(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, - Y_a(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          else
!!$             term1   = min( one, abs(A_RES_u(i)) / ( one + abs( nlp%A_u(i))) )
!!$             term2   = min( one, abs( Y_a(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, Y_a(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          end if
!!$       case ('EQ')
!!$          ! relax
!!$       case ('FR')
!!$          comp_vl = max( comp_vl, abs( Y_a(i) ) )
!!$       case default
!!$          write( data%control%out, * ) 'ERROR: check_optimal : A_type = ? '
!!$       end select
!!$    end do

!!$    ! Finally bound constraints.
!!$
!!$    do i = 1, nlp%n
!!$       select case( data%X_type(i) )
!!$       case ('LB')
!!$          term1   = min( one, abs( X_RES_l(i) ) / ( one + abs( nlp%X_l(i) ) ) )
!!$          term2   = min( one, abs( Z(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, - Z(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('UB')
!!$          term1   = min( one, abs( X_RES_u(i) ) / ( one + abs( nlp%X_u(i) ) ) )
!!$          term2   = min( one, abs( Z(i) ) )
!!$          comp_vl = max( comp_vl, term1 * term2 )
!!$          sign    = max( zero, Z(i) )
!!$          comp_vl = max( comp_vl, sign )
!!$       case ('RB')
!!$          if ( nlp%X(i) <= (nlp%X_l(i) + nlp%X_u(i) ) / two ) then
!!$             term1   = min( one, abs(X_RES_l(i)) / ( one + abs( nlp%X_l(i))) )
!!$             term2   = min( one, abs( Z(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, - Z(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          else
!!$             term1   = min( one, abs(X_RES_u(i)) / ( one + abs( nlp%X_u(i))) )
!!$             term2   = min( one, abs( Z(i) ) )
!!$             comp_vl = max( comp_vl, term1 * term2 )
!!$             sign    = max( zero, Z(i) )
!!$             comp_vl = max( comp_vl, sign )
!!$          end if
!!$       case ('EQ')
!!$          ! relax
!!$       case ('FR')
!!$          comp_vl = max( comp_vl, abs( Z(i) ) )
!!$       case default
!!$          write( out, * ) 'ERROR: check_optimal : X_type = ? '
!!$       end select
!!$    end do

    comp_vl = abs(comp_vl) ! to prevent -0.0000000000 from being printed.


    data%sub_primal_vl  = primal_vl
    data%sub_dual_vl    = dual_vl
    data%sub_comp_vl    = comp_vl
    data%sub_subgrad_vl = subgrad_vl

    inform%status = 0

    return

    END SUBROUTINE check_sub_optimal


!-*   C O N S T R A I N T_ V I O L A T I O N   S U B R O U T I N E -*-*

    SUBROUTINE constraint_violation( nlp, v, data, viol, status, &
                                   sat, vl_l, vl_u, num_sat, num_vl_l, num_vl_u )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
! Computes the "ell one" violation of the vector v. Specifically,
!
!        viol = sum( abs( min( v - C_l, C_u - v, 0 ) ) )
!
! Status of subroutine: well tested.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    type( NLPT_problem_type ), intent( in ) :: nlp
    type( TRIMSQP_data_type ), intent( in ) :: data
    integer, intent( out ) :: status
    integer, intent( inout ), optional :: num_sat, num_vl_l, num_vl_u
    integer, intent( inout ), dimension( nlp%m ), optional :: sat, vl_l, vl_u
    real( kind = wp ), intent( out ) :: viol
    real( kind = wp ), intent( in ), dimension( nlp%m ) :: v

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: i
    real( kind = wp ) :: res

    status = 0

    viol = zero

    if ( present(sat) ) then

       num_sat  = 0
       num_vl_l = 0
       num_vl_u = 0

       do i = 1, nlp%m
          select case( data%C_type( i ) )
          case('LB')
             res  = v( i ) - nlp%C_l( i )
             viol = viol + min( zero, res )
             if ( res < -tiny ) then
                num_vl_l       = num_vl_l + 1
                vl_l(num_vl_l) = i
             else
                num_sat        = num_sat + 1
                sat( num_sat ) = i
             end if
          case('UB')
             res  = nlp%C_u( i ) - v( i )
             viol = viol + min( zero, res )
             if ( res < -tiny ) then
                num_vl_u       = num_vl_u + 1
                vl_u(num_vl_u) = i
             else
                num_sat        = num_sat + 1
                sat( num_sat ) = i
             end if
          case('RB')
             res  = v( i ) - nlp%C_l( i )
             viol = viol + min( zero, res )
             if ( res < -tiny ) then
                num_vl_l       = num_vl_l + 1
                vl_l(num_vl_l) = i
             else
                res  = nlp%C_u( i ) - v( i )
                viol = viol + min( zero, res )
                if ( res < -tiny ) then
                   num_vl_u       = num_vl_u + 1
                   vl_u(num_vl_u) = i
                else
                   num_sat        = num_sat + 1
                   sat( num_sat ) = i
                end if
             end if
          case('EQ')
             res  = v( i ) - nlp%C_l( i )
             viol = viol - abs(res)
             if ( res > tiny ) then
                num_vl_u       = num_vl_u + 1
                vl_u(num_vl_u) = i
             elseif ( res < -tiny ) then
                num_vl_l       = num_vl_l + 1
                vl_l(num_vl_l) = i
             else
                num_sat        = num_sat + 1
                sat( num_sat ) = i
             end if
          case('FR')
             num_sat        = num_sat + 1
             sat( num_sat ) = i
          case default
             status = - 1
             write( data%control%out, * ) &
                  'ERROR: constraint_violation : unrecognized C_type value'
          end select
       end do

    else

       do i = 1, nlp%m
          select case( data%C_type( i ) )
          case('LB')
             res   = v( i ) - nlp%C_l( i )
             viol = viol + min( res, zero )
          case('UB')
             res    = nlp%C_u( i ) - v( i )
             viol = viol + min( res, zero )
          case('RB')
             res    = v( i ) - nlp%C_l( i )
             viol = viol + min( res, zero )
             res    = nlp%C_u( i ) - v( i )
             viol = viol + min( res, zero )
          case('EQ')
             res    = v( i ) - nlp%C_l( i )
             viol = viol - abs(res)
          case('FR')
             ! relax
          case default
             status = - 1
             write( data%control%out, * ) &
                  'ERROR: constraint_violation : unrecognized C_type value'
          end select
       end do

    end if

    viol = abs(viol)

    return

  END SUBROUTINE constraint_violation

!-*   q p c _ f a i l  S U B R O U T I N E -*-*

    SUBROUTINE qpc_fail( qp, status, data  )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!    qp  scalar variable of type integer. 1 = predictor, 2 = sqp-correction.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    type( TRIMSQP_data_type ), intent( inout ) :: data
    integer, intent( inout ) :: status
    integer, intent( in ) :: qp

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    select case( status )
    case ( - 8 )
       if ( data%control%print_level >= GALAHAD_DEBUG ) then
          write(data%control%out, * ) 'setting QPB_control%center to .FALSE'
       end if
       if ( qp == 1 ) then
          data%control%QPpred_control%QPB_control%center = .false.
       else
          data%control%QPsiqp_control%QPB_control%center = .false.
       end if
    case default
       write( data%control%out, * ) 'qpc_fail : not yet implemented'
    end select

    status = 0

    return

  END SUBROUTINE qpc_fail

!-*   p r i n t _ o p t i m a l i t y _ s u m m a r y S U B R O U T I N E -*-*

  SUBROUTINE print_optimality_summary( data )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!    print details of optimality check.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    type( TRIMSQP_data_type ), intent( in ) :: data

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------


    write(data%control%out, 3000) data%best_mults, data%mults_used,     &
                                  data%primal_vl_cur, data%dual_vl_cur, &
                                  data%comp_vl_cur, data%opt_measure_cur

    if ( data%iterate >= 1 ) then
       write(data%control%out, 3002) data%primal_vl_p, data%dual_vl_p, &
                                     data%comp_vl_p, data%opt_measure_p
    end if

    if ( data%sqp_computed ) then
       write(data%control%out, 3003) data%primal_vl_s, data%dual_vl_s, &
                                     data%comp_vl_s, data%opt_measure_s
    end if

    write(data%control%out, 3004 ) data%primal_vl, data%dual_vl, &
                                   data%comp_vl, data%opt_measure

    write(data%control%out, 3001)

    return

3000 format(/,                     &
     1x, (122('-')), /,             &
     T39, 'Optimality Results', /, &
     T39, '------------------', /, &
     1x, 'best_mults    = ', I1,     T34, 'mults_used  = ', A,      /, &
     1x, 'primal_vl_cur = ', ES12.6, T34, 'dual_vl_cur = ', ES12.6, &
     T64, 'comp_vl_cur = ', ES12.6,  T94, 'opt_measure_cur = ', ES12.6 )
3001 format(1x, (122('-')) )
3002 format( &
     1x, 'primal_vl_p   = ', ES12.6, T34, 'dual_vl_p   = ', ES12.6, &
     T64, 'comp_vl_p   = ', ES12.6, T94, 'opt_measure_p   = ', ES12.6 )
3003 format( &
     1x, 'primal_vl_s   = ', ES12.6, T34, 'dual_vl_s   = ', ES12.6, &
     T64, 'comp_vl_s   = ', ES12.6, T94, 'opt_measure_s   = ', ES12.6 )
3004 format( &
     1x, 'primal_vl     = ', ES12.6, T34, 'dual_vl     = ', ES12.6, &
     T64, 'comp_vl     = ', ES12.6, T94, 'opt_measure     = ', ES12.6 )

  END SUBROUTINE print_optimality_summary

!-*   p r i n t _ p r e d _ s t e e r  S U B R O U T I N E -*-*

  SUBROUTINE print_pred_steer( data )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!    print details of predictor and steering.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    type( TRIMSQP_data_type ), intent( in ) :: data
    !type( NLPT_problem_type ), intent( in ) :: nlp

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: out

    out = data%control%out

    write(out, 3000) data%control%use_steering, data%steering_good,   &
                     data%computed_steering, data%inf_norm_s_p,       &
                     data%inf_norm_Y_p, data%decreaseB, data%norm_c,  &
                     data%norm_c_linearize_pred, data%dec_norm_c_pred

    !if ( nlp%m <= 0 .or. (data%NM%active .and. data%NM%num_fail > 0)  &
    !                .or. (data%primal_vl < data%control%stop_p)      &
    !                .or. (.not. data%control%use_steering) ) then
    !   ! relax
    !elseif ( data%norm_c_linearize_pred < tenm6 ) then
    !   ! relax
    !else
    if ( data%computed_steering ) then
       write(out, 3002) data%norm_c, data%norm_c_linearize_steer,     &
                        data%dec_norm_c_steer, data%inf_norm_s_steer, &
                        data%inf_norm_Y_steer
    end if

    write(data%control%out, 3001) data%penalty, data%TRpred

    return

3000 format(/, &
     1x, (30('*-*')), /,                     &
     T31, 'Predictor - Steering Results', /, &
     T31, '----------------------------', /, &
     1x, 'use_steer  = ', L1,     T30, 'steer_good     = ', L1,         &
                                  T62, 'computed-steer   = ', L1, /,    &
     1x, 'norm_s_p   = ', ES11.5, T30, 'norm_Y_p       = ', ES11.5,     &
                                  T62, 'decreaseB_pred   = ', ES12.5 /, &
     1x, 'norm_c     = ', ES11.5, T30, 'norm_lin_pred  = ', ES11.5,     &
                                  T62, 'dec_lin_s_p      = ', ES12.5 )
3001 format( &
     1x, 'penaltynew = ', ES11.5, T30, 'predictor-TR   = ', ES11.5, /,  &
     1x, (30('*-*')) )
3002 format(1x, 'norm_c     = ', ES11.5, T30, 'norm_lin_steer = ', ES11.5,    &
                                         T62, 'dec_c_lin_steer  = ', ES12.5,/,&
            1x, 'norm_steer = ', ES11.5, T30, 'norm_Y_steer   = ', ES11.5 )

  END SUBROUTINE print_pred_steer


!-*   m a x _ f e a s _  s t e p  F U N C T I O N -*-*

  function max_feas_step( nlp, data, X_type, A_type, C_type )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!    compute maximum feasible step
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    real( kind = wp ) :: max_feas_step
    type( TRIMSQP_data_type ), intent( inout ) :: data
    type( NLPT_problem_type ), intent( in ) :: nlp
    character( len = 2 ), intent( in ), dimension(:) :: X_type, A_type, C_type

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: nspos, nsneg, nApos, nAneg, nJpos, nJneg
    integer :: i
    real( kind = wp ) :: alpha_x, alpha_A, alpha_C
    real( kind = wp ) :: dummy
!    real( kind = wp ), parameter :: very_tiny = 1

    nspos = 0
    nsneg = 0
    do i = 1, data%nfr
       if ( data%s_s( data%fr(i) ) > very_tiny ) then
          nspos = nspos + 1
          data%spos( nspos ) = data%fr(i)
       elseif ( data%s_s( data%fr(i) ) < - very_tiny ) then
          nsneg = nsneg + 1
          data%sneg( nsneg ) = data%fr(i)
       end if
    end do
    data%nspos = nspos
    data%nsneg = nsneg

    nApos = 0
    nAneg = 0
    do i = 1, data%nwA_comp
       if ( data%AxSs( data%wA_comp(i) ) > very_tiny ) then
          nApos = nApos + 1
          data%Apos( nApos ) = data%wA_comp(i)
       elseif ( data%AxSs( data%wA_comp(i) ) < - very_tiny ) then
          nAneg = nAneg + 1
          data%Aneg( nAneg ) = data%wA_comp(i)
       end if
    end do
    data%nApos = nApos
    data%nAneg = nAneg

    nJpos = 0
    nJneg = 0
    do i = 1, data%nwJ_comp
       if ( data%JxSs( data%wJ_comp(i) ) > very_tiny ) then
          nJpos = nJpos + 1
          data%Jpos(nJpos) = data%wJ_comp(i)
       elseif ( data%JxSs( data%wJ_comp(i) ) < - very_tiny ) then
          nJneg = nJneg + 1
          data%Jneg(nJneg) = data%wJ_comp(i)
       end if
    end do
    data%nJpos = nJpos
    data%nJneg = nJneg

    ! Now compute maximal feasible distance along ss.
    alpha_x = one
    alpha_A = one
    alpha_C = one


!!$    write(*,*) 'nfx = ', data%nfx
!!$    write(*,*) 'fx = ', data%fx
!!$    write(*,*) 'nfr = ', data%nfr
!!$    write(*,*) 'fr = ', data%fr
!!$
!!$    write(*,*) 'nspos = ', data%nspos
!!$    write(*,*) 'nsneg = ', data%nsneg
!!$    write(*,*) 'spos = ', data%spos
!!$    write(*,*) 'sneg = ', data%sneg
!!$    write(*,*) 'Xtype = ', X_type
!!$    write(*,*) 'X = ', nlp%X
!!$    write(*,*) 'sp = ', data%s_p
!!$    write(*,*) 'ss = ', data%S_s
!!$    write(*,*) 'nApos = ', data%nApos
!!$    write(*,*) 'nAneg = ', data%nAneg
!!$    write(*,*) 'Apos = ', data%Apos
!!$    write(*,*) 'Aneg = ', data%Aneg
!!$    write(*,*) 'Ax = ', nlp%Ax
!!$    write(*,*) 'Asp = ', data%AxSp
!!$    write(*,*) 'Ass = ', data%AxSs
!!$    write(*,*) 'C = ', nlp%C
!!$    write(*,*) 'Jsp = ', data%JxSp
!!$    write(*,*) 'Jss = ', data%JxSs

    do i = 1, nspos
       if ( X_type( data%spos(i) ) == 'RB' .or. X_type( data%spos(i) ) == 'UB' ) then
          dummy = nlp%X_u( data%spos(i) ) - ( nlp%X( data%spos(i) ) + data%s_ac( data%spos(i) ) )
          dummy = dummy / data%s_s( data%spos(i) )
          alpha_x = min( alpha_x,  dummy )
       end if
    end do
    do i = 1, nsneg
       if ( X_type( data%sneg(i) ) == 'RB' .or. X_type( data%sneg(i) ) == 'LB' ) then
          dummy = nlp%X_l( data%sneg(i) ) - ( nlp%X( data%sneg(i) ) + data%s_ac( data%sneg(i) ) );
          dummy = dummy / data%s_s( data%sneg(i) )
          alpha_x = min( alpha_x,  dummy )
       end if
    end do

    do i = 1, nApos
       if ( A_type( data%Apos(i) ) == 'RB' .or. A_type( data%Apos(i) ) == 'UB' ) then
          dummy = nlp%A_u( data%Apos(i) ) - ( nlp%Ax( data%Apos(i) ) + data%AxSac( data%Apos(i) ) )
          dummy = dummy / data%AxSs( data%Apos(i) )
          alpha_A = min( alpha_A,  dummy )
       end if
    end do
    do i = 1, nAneg
       if ( A_type( data%Aneg(i) ) == 'RB' .or. A_type( data%Aneg(i) ) == 'LB' ) then
          dummy = nlp%A_l( data%Aneg(i) ) - ( nlp%Ax( data%Aneg(i) ) + data%AxSac( data%Aneg(i) ) )
          dummy = dummy / data%AxSs( data%Aneg(i) )
          alpha_A = min( alpha_A,  dummy )
       end if
    end do

    do i = 1, nJpos
       if ( C_type( data%Jpos(i)) == 'RB' .or. C_type( data%Jpos(i)) == 'UB' ) then
          dummy = nlp%C_u( data%Jpos(i) ) - ( nlp%C( data%Jpos(i) ) + data%JxSac( data%Jpos(i) ) )
          dummy = dummy / data%JxSs( data%Jpos(i) )
          alpha_C = min( alpha_C,  dummy )
       end if
    end do
    do i = 1, nJneg
       if ( C_type( data%Jneg(i) ) == 'RB' .or. C_type( data%Jneg(i) ) == 'LB' ) then
          dummy = nlp%C_l( data%Jneg(i) ) - ( nlp%C( data%Jneg(i) ) + data%JxSac( data%Jneg(i) ) )
          dummy = dummy / data%JxSs( data%Jneg(i) )
          alpha_C = min( alpha_C,  dummy )
       end if
    end do

    max_feas_step = min( alpha_x, alpha_A, alpha_C )
!!$    write(*,*) 'alpha_x = ', alpha_x
!!$    write(*,*) 'alpha_C = ', alpha_C
!!$    write(*,*) 'alpha_A = ', alpha_A
!!$    write(*,*) 'alpha = ', max_feas_step

    return

!!$3000 format(/, &
!!$     1x, (30('*-*')), /,                     &
!!$     T31, 'Predictor - Steering Results', /, &
!!$     T31, '----------------------------', /, &
!!$     1x, 'use_steer  = ', L1,     T30, 'steer_good     = ', L1,         &
!!$                                  T62, 'need_best_linear = ', L1, /,    &
!!$     1x, 'norm_s_p   = ', ES11.5, T30, 'norm_Y_p       = ', ES11.5, /,  &
!!$     1x, 'norm_c     = ', ES11.5, T30, 'norm_lin_pred  = ', ES11.5,      &
!!$                                  T62, 'dec_lin_s_p      = ', ES12.5 )
!!$3001 format(1x, (30('*-*')) )
!!$3002 format(1x, 'norm_c     = ', ES11.5, T30, 'norm_lin_steer = ', ES11.5,      &
!!$                                         T62, 'dec_c_lin_steer  = ', ES12.5,/,&
!!$            1x, 'norm_steer = ', ES11.5, T30, 'norm_Y_steer   = ', ES11.5 )

  end function max_feas_step

!-*   g e t _ r e s i d u a l s  S U B R O U T I N E -*-*

  SUBROUTINE get_residuals( vl, v, vu, v_type, res_vl, res_vu )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

    character( len = 2 ), intent( in ), dimension(:) :: v_type
    real( kind = wp ), intent( in ), dimension(:) :: vl, v, vu
    real( kind = wp ), intent( out ), dimension(:) :: res_vl, res_vu
    !type( NLPT_problem_type ), intent( in ) :: nlp

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

    integer :: n, i

    n = size(vl)

    do i = 1, n
       if ( v_type( i ) == 'EQ' .or. v_type( i ) == 'LB' .or. v_type( i ) == 'RB'  ) then
          res_vl( i ) = v( i ) - vl( i )
       end if
       if ( v_type( i ) == 'UB' .or. v_type( i ) == 'RB' ) then
          res_vu( i ) = vu( i ) - v( i )
       end if
    end do

  END SUBROUTINE get_residuals

!-*   g e t _ a c t i v e  S U B R O U T I N E -*-*

  SUBROUTINE get_active( xl, xu, res_xl, res_xu, x_type, nfx, fx, nfr, fr, nvl, vl, tol, out )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Dummy arguments

  character( len = 2 ), intent( in ), dimension(:) :: x_type
  integer, intent( in ) :: out
  real( kind = wp ), intent( in ) :: tol
  real( kind = wp ), intent( in ), dimension(:) :: xl, xu
  integer, intent( inout ), dimension(:) :: fx, fr, vl
  real( kind = wp ), intent( in ), dimension(:) :: res_xl, res_xu
  integer, intent( out ) :: nfx, nfr, nvl
  !type( NLPT_problem_type ), intent( in ) :: nlp

!---------------------------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------------------------

  integer :: n, i
  real( kind = wp) :: r

  n = size( xl )

  nfx = 0
  nfr = 0
  nvl = 0

  do i = 1, n

     select case ( x_type(i) )
     case ('LB')
        r = res_xl(i) / (one + abs(xl(i)))
        if ( abs(r) < tol ) then
           nfx       = nfx + 1
           fx( nfx ) = i
        elseif ( r >= tol ) then
           nfr       = nfr + 1
           fr( nfr ) = i
        else
           nvl       = nvl + 1
           vl( nvl ) = i
        end if
     case ('UB')
        r = res_xu(i) / (one + abs(xu(i)))
        if ( abs(r) < tol ) then
           nfx       = nfx + 1
           fx( nfx ) = i
        elseif ( r >= tol ) then
           nfr       = nfr + 1
           fr( nfr ) = i
        else
           nvl       = nvl + 1
           vl( nvl ) = i
        end if
     case ('FR')
        nfr       = nfr + 1
        fr( nfr ) = i
     case ('EQ')
        r = res_xl(i) / (one + abs(xl(i)))
        if ( abs(r) < tol ) then
           nfx       = nfx + 1
           fx( nfx ) = i
        else
           nvl       = nvl + 1
           vl( nvl ) = i
        end if
     case ('RB')
        r = res_xu(i) / (one + abs(xu(i)))
        if ( abs(r) < tol ) then
           nfx       = nfx + 1
           fx( nfx ) = i
        elseif ( r <= -tol ) then
           nvl       = nvl + 1
           vl( nvl ) = i
        else
           r = res_xl(i) / (one + abs(xl(i)))
           if ( abs(r) < tol ) then
              nfx       = nfx + 1
              fx( nfx ) = i
           elseif ( r <= -tol ) then
              nvl       = nvl + 1
              vl( nvl ) = i
           else
              nfr       = nfr + 1
              fr( nfr ) = i
           end if
        end if
     case default
        write( out, * ) ' ERROR:trimsqp:get_active: Unrecognized value x_type(i).'
        return
     end select

  end do

  END SUBROUTINE get_active

!-*   g e t _ L _ B F G S   S U B R O U T I N E -*-*

  SUBROUTINE get_L_BFGS( iterate, Bzero, A, B, S, Y, s_f, gradLx_new, gradLx, &
                         SBinner, eta, curve_mod, ind, L, error, out )

  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  !
  !
  !
  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  ! Dummy arguments

  !character( len = 2 ), intent( in ), dimension(:) :: x_type
  integer, intent( in ) :: iterate, curve_mod, L, error, out
  real( kind = wp ), intent( in ) :: eta
  real( kind = wp ), intent( in ), dimension(:) :: gradLx_new, gradLx, s_f
  integer, intent( inout ), dimension(:) ::  ind
  real( kind = wp ), intent( inout ), dimension(:,:) :: A, B, S, Y, SBinner
  type( SMT_type ), intent( in ) :: Bzero

  !---------------------------------------------------
  !   L o c a l   V a r i a b l e s
  !---------------------------------------------------

  integer :: dummy_int, last, col, i, j, k
  real( kind = wp ) :: theta, sta, stBs, ytS
  real( kind = wp ), dimension(size(s_f)) :: Bs

 ! Figure out which column is being added/replaced.

  last = min( iterate - 1, L )

  if ( iterate <= L+1 ) then
     ind( iterate - 1 ) = iterate - 1
  else
     dummy_int     = ind( 1 )
     ind( 1: L-1 ) = ind( 2 : L )
     ind( L )      = dummy_int
  end if

  col = ind( last )

  ! Replace that column in S and Y.
  S( :, col ) = s_f
  Y( :, col ) = gradLx_new - gradLx

  do i = 1, last

     if ( i == last ) then
        B(:,ind(last)) = Y(:,ind(last)) / dot_product( Y(:,ind(last)), S(:,ind(last)) )**half
        do j = 1, last-1
           SBinner( ind(last), ind(j) ) = dot_product( S(:,ind(last)), B(:,ind(j)) )
        end do
     end if
     A(:,ind(i)) = zero
     call mop_Ax( one, Bzero, S(:,ind(i)), one, A(:,ind(i)), &
                  out, error, transpose=.false. )
     do k = 1, i-1
        A(:,ind(i)) = A(:,ind(i)) - dot_product( A(:,ind(k)), S(:,ind(i)) ) * A(:,ind(k))
        A(:,ind(i)) = A(:,ind(i)) + SBinner(ind(i),ind(k)) * B(:,ind(k))
     end do
     Bs = A(:,ind(i))
     A(:,ind(i)) = Bs / ( dot_product( S(:,ind(i)), Bs ) )**half

     if ( i== last ) then

        ! Check if sufficiently positive definite.
        yts  = dot_product( Y(:,ind(last)), S(:,ind(last)) )
        stBs = dot_product( S(:,ind(last)), Bs )  ! s_k^T B_k s_k

        if ( yts < eta*stBs ) then
           if ( curve_mod == 0 ) then ! skip it
              theta = 0
           elseif ( curve_mod == 1) then ! % Powell
              theta          = (1-eta)*stBs/(stBs-yts)
              Y(:,ind(last)) = theta * Y(:,ind(last)) + (1-theta)*Bs
           else
              write(error, *)' ERROR:get_L_BFGS: disallowed value curve_mod.'
           end if

           B(:,ind(last)) = Y(:,ind(last)) / ( dot_product( Y(:,ind(last)), S(:,ind(last)) ) )**half
           do j= 1, last-1
              SBinner(ind(last),ind(j)) = dot_product( S(:,ind(last)), B(:,ind(j)) )
           end do

           ! Double check if sufficiently positive definite.
           yts = dot_product( Y(:,ind(last)), S(:,ind(last)) )
           sta = dot_product( S(:,ind(last)), A(:,ind(last)) )  ! s_k^T B_k s_k
           if ( yts < 0.5*eta*sta ) then
              write(error,*) 'ERROR:get_L_BFGS: curvature MAY be wrong after modification!'
              if ( yts < tenm8 ) then
                 write(error,*) 'ERROR:get_L_BFGS: curvature PROBABLY wrong after modification!'
                 if ( yts <= zero ) then
                    write(error,*) 'ERROR:get_L_BFGS: curvature DEFINITELY wrong after modification!'
                 end if
              end if
           end if
        end if

     end if

  end do


end SUBROUTINE get_L_BFGS



!  End of module GALAHAD_TRIMSQP

   END MODULE GALAHAD_TRIMSQP_double
