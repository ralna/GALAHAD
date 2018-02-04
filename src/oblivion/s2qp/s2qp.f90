 ! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

 !-*-*-*-*-*-*-*-*  G A L A H A D _ S 2 Q P   M O D U L E  *-*-*-*-*-*-*-*-*-*-*

 ! Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 ! Author: Daniel Robinson

 MODULE GALAHAD_S2QP_double

 !------------------------------------------------------------------------------
 !                                                                             |
 ! MODULE NAME : GALAHAD_S2QP module.                                          |
 !                                                                             |
 ! CONTAINS :                                                                  |
 !                                                                             |
 !    Main Subroutines                                                         |
 !       S2QP_initialize, S2QP_read_specfile, S2QP_solve, S2QP_terminate       |
 !                                                                             |
 !    Quadratic problem related subroutines                                    |
 !       build_QPfeas, build_QPpred, build_QPsteer, build_QPseqp,              |
 !       build_QPsiqp, fill_QPfeas, fill_QPpred, fill_QPsteer,                 |
 !       fill_QPseqp, fill_QPsiqp, dealloc_QPpred, dealloc_QPfeas              |
 !       dealloc_QPseqp, dealloc_QPsiqp, dealloc_QPsteer, qpc_recover          |
 !                                                                             |
 !    Printing subroutines                                                     |
 !       print_predictor_problem, print_predictor_information,                 |
 !       print_seqp_problem, print_seqp_information,                           |
 !       print_siqp_problem, print_siqp_information,                           |
 !       print_fullstep_information, print_cauchy_information                  |
 !       print_optimality_summary                                              |
 !                                                                             |
 !   Optimality related subroutines                                            |
 !      get_best_opt, get_opt                                                  |
 !                                                                             |
 !   Cauchy step computation subroutines                                       |
 !      cauchy_step, cauchy_val, cauchy_val_and_slope                          |
 !                                                                             |
 !   Auxilliary subroutines and functions                                      |
 !      get_L_BFGS, get_active, max_feas_step, get_residuals,                  |
 !      constraint_violation                                                   |
 !                                                                             |
 ! USAGE :                                                                     |
 !   This module containts subroutines that may be used to solve constrained   |
 !   and unconstrained general nonlinear nonconvex optimization problems.      |
 !   Computes a local solution of problems in the following form:              |
 !                                                                             |
 !   minimize               f(x)                                               |
 !   subject to        cl <= c(x) <= cu      c is in R^m                       |
 !                     Al <= A x  <= Au      A is an m_a by n matrix           |
 !                     xl <=   x  <= xu      x is in R^n                       |
 !                                                                             |
 !------------------------------------------------------------------------------

 USE GALAHAD_NORMS_double
 USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type,         &
                                NLPT_write_problem
 USE GALAHAD_QPC_double
 USE GALAHAD_EQP_double
 USE GALAHAD_QPT_double
 USE GALAHAD_SMT_double
 USE GALAHAD_SILS_double
 USE GALAHAD_SPECFILE_double
 USE GALAHAD_SPACE_double
 USE GALAHAD_SYMBOLS
 USE GALAHAD_CHECK_double
 USE GALAHAD_SORT_double
 USE GALAHAD_mop_double

 IMPLICIT NONE

 PRIVATE
 PUBLIC :: S2QP_initialize, S2QP_read_specfile, S2QP_solve, S2QP_terminate

 !  Set precision

 INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

 !  Set other parameters

 REAL ( KIND = wp ), PARAMETER :: zero    = 0.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenm15  = 0.000000000000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm12  = 0.000000000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm10  = 0.0000000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm9   = 0.000000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm8   = 0.00000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm7   = 0.0000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm6   = 0.000001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm5   = 0.00001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm4   = 0.0001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm3   = 0.001_wp
 REAL ( KIND = wp ), PARAMETER :: tenm2   = 0.01_wp
 REAL ( KIND = wp ), PARAMETER :: point1  = 0.1_wp
 REAL ( KIND = wp ), PARAMETER :: point2  = 0.2_wp
 REAL ( KIND = wp ), PARAMETER :: point3  = 0.3_wp
 REAL ( KIND = wp ), PARAMETER :: point4  = 0.4_wp
 REAL ( KIND = wp ), PARAMETER :: half    = 0.5_wp
 REAL ( KIND = wp ), PARAMETER :: point6  = 0.6_wp
 REAL ( KIND = wp ), PARAMETER :: point7  = 0.7_wp
 REAL ( KIND = wp ), PARAMETER :: point8  = 0.8_wp
 REAL ( KIND = wp ), PARAMETER :: point9  = 0.9_wp
 REAL ( KIND = wp ), PARAMETER :: one     = 1.0_wp
 REAL ( KIND = wp ), PARAMETER :: two     = 2.0_wp
 REAL ( KIND = wp ), PARAMETER :: three   = 3.0_wp
 REAL ( KIND = wp ), PARAMETER :: four    = 4.0_wp
 REAL ( KIND = wp ), PARAMETER :: five    = 5.0_wp
 REAL ( KIND = wp ), PARAMETER :: ten     = 10.0_wp
 REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenp2   = 100.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenp3   = 1000.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenp4   = 10000.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenp5   = ten ** 5
 REAL ( KIND = wp ), PARAMETER :: tenp6   = ten ** 6
 REAL ( KIND = wp ), PARAMETER :: tenp7   = ten ** 7
 REAL ( KIND = wp ), PARAMETER :: tenp8   = ten ** 8
 REAL ( KIND = wp ), PARAMETER :: tenp9   = ten ** 9
 REAL ( KIND = wp ), PARAMETER :: biginf  = HUGE( one )
 REAL ( KIND = wp ), PARAMETER :: epsmch  = EPSILON( one )
 REAL ( KIND = wp ), PARAMETER :: teneps  = ten * epsmch
 REAL ( KIND = wp ), PARAMETER :: sqrteps = epsmch ** 0.5_wp
 REAL ( KIND = wp ), PARAMETER :: tiny    = ten ** ( - 7 )
 REAL ( KIND = wp ), PARAMETER :: mu_tiny = ten ** ( - 6 )
 REAL ( KIND = wp ), PARAMETER :: y_tiny  = ten ** ( - 6 )
 REAL ( KIND = wp ), PARAMETER :: z_tiny  = ten ** ( - 6 )
 REAL ( KIND = wp ), PARAMETER :: gzero   = ten ** ( - 10 )
 REAL ( KIND = wp ), PARAMETER :: hzero   = ten ** ( - 10 )

 LOGICAL, PARAMETER :: exact_dual = .FALSE.

 !==============================================================================
 ! The BFGS_data_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_bfgs_type
    INTEGER :: mod_type, update_number
    REAL ( KIND = wp ) :: std, theta, damp_factor, stBs
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: d, s, Bs
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: gradLx_new, gradLx
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: g_ref, Jval_ref, X_ref
 END TYPE S2QP_bfgs_type

 !==============================================================================
 ! The L_BFGS_data_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_l_bfgs_type
    INTEGER :: update_number, number_used
    REAL ( KIND = wp ) :: theta, damp_factor
    REAL ( KIND = WP ), ALLOCATABLE, DIMENSION( :, : ) :: A, B, S, BSinner
    REAL ( KIND = WP ), ALLOCATABLE, DIMENSION( : ) :: svec, y
    REAL ( KIND = WP ), ALLOCATABLE, DIMENSION( : ) :: gradLx_new, gradLx
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: g_ref, Jval_ref, X_ref
    TYPE ( SMT_type ) :: A_smt, B_smt
 END TYPE S2QP_l_bfgs_type

 !==============================================================================
 ! The S2QP_control_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_control_type
    INTEGER :: error, out, alive_unit, print_level, print_number, max_iterate
    INTEGER :: start_print, stop_print, print_gap
    INTEGER :: header_every, NM_steps
    INTEGER :: B_type, L_BFGS_number, L_BFGS_curve_mod
    REAL ( KIND = wp ) :: stop_p_abs, stop_c_abs, stop_d_abs
    REAL ( KIND = wp ) :: stop_p_rel, stop_c_rel, stop_d_rel
    REAL ( KIND = wp ) :: initial_penalty, max_penalty, penalty_expansion
    REAL ( KIND = wp ) :: initial_TRpred, max_TRpred
    REAL ( KIND = wp ) :: max_TRacc, TRacc_scale
    REAL ( KIND = wp ) :: infinity
    REAL ( KIND = wp ) :: eta_successful, eta_very_successful
    REAL ( KIND = wp ) :: eta_extremely_successful
    LOGICAL :: print_sol, fulsol
    LOGICAL :: space_critical, deallocate_error_fatal
    LOGICAL :: use_seqp, use_siqp, use_TRpred
    CHARACTER ( LEN = 30 ) :: alive_file
    TYPE ( QPC_control_type ) :: QPpred_control, QPsiqp_control
    TYPE ( QPC_control_type ) :: QPfeas_control, QPsteer_control
    TYPE ( EQP_control_type ) :: QPseqp_control
    TYPE ( SILS_control ) :: SILS_control
 END TYPE S2QP_control_type

 !==============================================================================
 ! The S2QP_time_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_time_type
    REAL( KIND = wp ) :: feas_preprocess, feas_depend, feas_analyse
    REAL( KIND = wp ) :: feas_factorize, feas_solve, feas_total
    REAL( KIND = wp ) :: pred_A_preprocess, pred_A_analyse
    REAL( KIND = wp ) :: pred_A_factorize, pred_A_solve, pred_A_total
    REAL( KIND = wp ) :: pred_B_preprocess, pred_B_analyse
    REAL( KIND = wp ) :: pred_B_factorize, pred_B_solve, pred_B_total
    REAL( KIND = wp ) :: pred_C_preprocess, pred_C_depend, pred_C_analyse
    REAL( KIND = wp ) :: pred_C_factorize, pred_C_solve, pred_C_total
    REAL( KIND = wp ) :: steer_A_preprocess, steer_A_analyse
    REAL( KIND = wp ) :: steer_A_factorize, steer_A_solve, steer_A_total
    REAL( KIND = wp ) :: steer_B_preprocess,steer_B_analyse
    REAL( KIND = wp ) :: steer_B_factorize, steer_B_solve, steer_B_total
    REAL( KIND = wp ) :: steer_C_preprocess, steer_C_depend, steer_C_analyse
    REAL( KIND = wp ) :: steer_C_factorize, steer_C_solve, steer_C_total
    REAL( KIND = wp ) :: cauchy_total
    REAL( KIND = wp ) :: seqp_factorize, seqp_solve, seqp_total
    REAL( KIND = wp ) :: siqp_A_preprocess, siqp_A_analyse
    REAL( KIND = wp ) :: siqp_A_factorize, siqp_A_solve, siqp_A_total
    REAL( KIND = wp ) :: siqp_B_preprocess, siqp_B_analyse
    REAL( KIND = wp ) :: siqp_B_factorize, siqp_B_solve, siqp_B_total
    REAL( KIND = wp ) :: siqp_C_preprocess, siqp_C_depend, siqp_C_analyse
    REAL( KIND = wp ) :: siqp_C_factorize, siqp_C_solve, siqp_C_total
    REAL( KIND = wp ) :: opt_test_total
    REAL( KIND = wp ) :: total_preprocess, total_depend, total_analyse
    REAL( KIND = wp ) :: total_factorize, total_solve, total_total
    REAL( KIND = wp ) :: in, out
    REAL( KIND = wp ) :: enter_s2qp, exit_s2qp, total
 END TYPE S2QP_time_type

 !==============================================================================
 ! The S2QP_inform_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_inform_type
    INTEGER :: status, alloc_status, iterate, cg_iter, itcgmx
    INTEGER :: num_f_eval, num_g_eval, num_c_eval, num_J_eval, num_H_eval
    INTEGER :: nvar, ngeval, iskip, ifixed, nfacts, nmods
    INTEGER :: factorization_status, num_predictors, num_descent_active
    INTEGER :: factorization_integer, factorization_real
    REAL ( KIND = wp ) :: obj, primal_vl, dual_vl, comp_vl
    LOGICAL :: newsol
    CHARACTER ( LEN = 80 ) :: bad_alloc
    TYPE ( S2QP_time_type ) :: time
    TYPE ( QPC_inform_type ) :: QPpred_inform, QPsiqp_inform
    TYPE ( QPC_inform_type ) :: QPfeas_inform, QPsteer_inform
    TYPE ( EQP_inform_type ) :: QPseqp_inform
 END TYPE S2QP_inform_type

 !==============================================================================
 ! The S2QP_revert_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_revert_type
    REAL( KIND = wp ) :: f, norm_c, merit
    REAL( KIND = wp ) :: TRpred, inf_norm_Y, min_TR
    REAL( KIND = wp ) :: primal_vl, dual_vl, comp_vl
    REAL( KIND = wp ) :: penalty, inf_norm_x, inf_norm_step1, inf_norm_s_p
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: g, Jval
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, Y
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ax, Y_a
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Bval
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_RES_l, C_RES_u
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_RES_l, A_RES_u
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_RES_l, X_RES_u
    CHARACTER ( LEN = 2 ) :: trial_step
 END TYPE S2QP_revert_type

 !==============================================================================
 ! The S2QP_nonmonotone_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_nonmonotone_type
    logical :: revert, active
    integer :: num_fail
    real ( KIND = wp ) :: decreaseH, delmod_ref, merit_ref
 END TYPE S2QP_nonmonotone_type

 !==============================================================================
 ! The S2QP_reverse_communication_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_reverse_communication_type
    real ( KIND = wp ) :: f
    real ( KIND = wp ), allocatable, dimension( : ) :: u, v, x, y, c, g
    real ( kind = wp ), allocatable, dimension( : ) :: Jval, Hval
    logical :: f_filled, c_filled, g_filled, J_filled, H_filled
 END TYPE S2QP_reverse_communication_type

 !==============================================================================
 ! The S2QP_data_type derived type
 !==============================================================================

 TYPE, PUBLIC :: S2QP_data_type
    REAL ( KIND = wp ) :: penalty, merit, merit_new, ratio
    REAL ( KIND = wp ) :: penalty_pre_steer, merit_pre_steer
    REAL ( KIND = wp ) :: TRpred, TRacc
    REAL ( KIND = wp ) :: alpha, alpha_end, alpha_c, alpha_feas, F_new
    REAL ( KIND = wp ) :: TRpred_expand, TRpred_contract
    REAL ( KIND = wp ) :: inf_norm_Y_p, inf_norm_Y_c, inf_norm_step1
    REAL ( KIND = wp ) :: inf_norm_Y_s, inf_norm_Y_steer
    REAL ( KIND = wp ) :: TR_reset_value, min_TR
    REAL ( KIND = wp ) :: primal_vl, dual_vl, comp_vl
    REAL ( KIND = wp ) :: primal_vl2, dual_vl2, comp_vl2
    REAL ( KIND = wp ) :: primal_vl3, dual_vl3, comp_vl3
    REAL ( KIND = wp ) :: norm_c, norm_c_new, two_norm_s_p
    REAL ( KIND = wp ) :: inf_norm_s_p, inf_norm_s_p_saved, two_norm_s_p_saved
    REAL ( KIND = wp ) :: inf_norm_s_c, inf_norm_s_ac, inf_norm_Y_p_saved
    REAL ( KIND = wp ) :: inf_norm_s_s, inf_norm_s_steer, inf_norm_s_f
    REAL ( KIND = wp ) :: inf_norm_Y, Y_bound
    REAL ( KIND = wp ) :: decreaseH_cauchy, decreaseH_full, delmod
    REAL ( KIND = wp ) :: decreaseB, decreaseB_saved
    REAL ( KIND = wp ) :: decreaseB_smooth, decreaseB_smooth_saved
    REAL ( KIND = wp ) :: stHs, vtBv, LBFGS_damping_factor
    REAL ( KIND = wp ) :: Sp_B_Sp, Sp_B_Sp_saved, Sp_H_Sp
    REAL ( KIND = wp ) :: Sc_H_Sc, Ss_H_Ss, Sf_H_Sf
    REAL ( KIND = wp ) :: Sp_H_Sp_saved
    REAL ( KIND = wp ) :: steer_L_factor, steer_Q_factor
    REAL ( KIND = wp ) :: con_viol_s_steer, ac_factor
    REAL ( KIND = wp ) :: opt_measure, opt_measure_cur
    REAL ( KIND = wp ) :: opt_measure_p, opt_measure_s
    REAL ( KIND = wp ) :: primal_vl_cur, primal_vl_p, primal_vl_s
    REAL ( KIND = wp ) :: dual_vl_cur, dual_vl_p, dual_vl_s
    REAL ( KIND = wp ) :: comp_vl_cur, comp_vl_p, comp_vl_s
    REAL ( KIND = wp ) :: dec_norm_c_pred, dec_norm_c_cauchy, dec_norm_c_steer
    REAL ( KIND = wp ) :: norm_c_linearize_pred, norm_c_linearize_pred_saved
    REAL ( KIND = wp ) :: norm_c_linearize_cauchy
    REAL ( KIND = wp ) :: norm_c_linearize_full, norm_c_linearize_steer
    REAL ( KIND = wp ) :: gtSp, gtSc, gtSf, gtSp_saved
    LOGICAL :: converged, blow_up, acc_good_dec, new_penalty
    LOGICAL :: acc_ratio_used, acc_computed, step_accepted!, LP_penalty_update
    LOGICAL :: siqp_computed, seqp_computed, seqp_try_pred!, penalty_steer_reset
    LOGICAL :: steering_good, computed_steering, check_suboptimal
    LOGICAL :: use_prev_pred
    INTEGER :: iterate, lbreak, QP_fails
    INTEGER :: iterates_pred, iterates_acc, num_consec_Y_free
    INTEGER :: num_consec_Y_active
    INTEGER :: num_sat, num_vl_l, num_vl_u, success, best_mults
    INTEGER :: nfr, nfx, nwA, nwA_comp, nwJ, nwJ_comp
    INTEGER :: nspos, nsneg, nJpos, nJneg, nApos, nAneg
    INTEGER :: nclb, ncub, ncrb, nce
    INTEGER :: branch
    INTEGER, ALLOCATABLE, DIMENSION( : ) :: clb, cub, crb, ce
    INTEGER, ALLOCATABLE, DIMENSION( : ) :: IBREAK
    INTEGER, ALLOCATABLE, DIMENSION( : ) :: vl_l, vl_u, sat
    INTEGER, ALLOCATABLE, DIMENSION( : ) :: fr, fx, wA, wA_comp, wJ, wJ_comp
    INTEGER, ALLOCATABLE, DIMENSION( : ) :: spos, sneg, Apos, Aneg, Jpos, Jneg
    CHARACTER ( LEN = 7 ) :: success_str
    CHARACTER ( LEN = 4 ) :: descent_constraint_status
    CHARACTER ( LEN = 1 ) :: mults_used, merit_first_order
    CHARACTER ( LEN = 2 ) :: trial_step
    CHARACTER, ALLOCATABLE, DIMENSION( : )  :: approx_type
    CHARACTER ( LEN = 2 ), ALLOCATABLE, DIMENSION( : ) :: C_type, A_type,X_type
    TYPE ( S2QP_control_type ) :: control
    TYPE ( QPT_problem_type ) :: QPpred, QPsiqp, QPseqp, QPfeas, QPsteer
    TYPE ( EQP_data_type ) :: QPseqp_data
    TYPE ( QPC_data_type ) :: QPpred_data, QPsiqp_data
    TYPE ( QPC_data_type ) :: QPsteer_data, QPfeas_data
    TYPE ( S2QP_l_bfgs_type ) :: L_BFGS
    TYPE ( S2QP_bfgs_type ) :: BFGS
    TYPE ( S2QP_revert_type ) :: revert
    TYPE ( S2QP_nonmonotone_type ) :: NM
    TYPE ( S2QP_reverse_communication_type ) :: RC
    type ( SMT_type ) :: B, B_full
    type ( SILS_factors ) :: SILS_factors
    type ( SILS_Ainfo ) :: SILS_Ainfo
    type ( SILS_Finfo ) :: SILS_Finfo
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_new
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CplusJsc
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: s_p, s_p_saved, w
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: s_c, s_s, s_f, s_steer
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GplusHs, descent_con
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Bval_saved
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: J_norms, H_norms
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Hval_saved, HxSp_saved
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AxSp, AxSp_saved, AxSc
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AxSac, AxSs, AXplusSc
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BxSp, HxSp, HxSc
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HxSs, HxSf
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JxSp, JxSp_saved,JxSsteer
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JxSc, JxSac, JxSs, JxSf
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: u_in, v_in, u_out
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: v_out, X_trial
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY, AtYa
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_cur, AtYa_cur
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_p, AtYa_p
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: JtY_s, AtYa_s
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_s, Ya_s, Z_s
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAKP
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_RES_l, C_RES_u
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_RES_l_new, C_RES_u_new
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_RES_l, A_RES_u
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_RES_l, X_RES_u
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: cauchyRESl, cauchyRESu
 END TYPE S2QP_data_type

 CONTAINS

 !***********  G A L A H A D -  S2QP_initialize  S U B R O U T I N E **********

 SUBROUTINE S2QP_initialize( data, control, inform )

 !-----------------------------------------------------------------------------
 ! Purpose: Provide default values for S2QP control parameters.  Also
 !          initializes control parameters for every quadratic subproblem
 !          that S2QP utilizes during its iterative process.
 !
 ! On Exit
 !
 !   inform%status    0  all control parameters have been successfully defined.
 !                  -31  one of the qpc_initialize routines returned an error.
 !                  -32  eqp_initialize failed.
 !-----------------------------------------------------------------------------

 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 TYPE ( S2QP_data_type ),    INTENT( INOUT ) :: data
 TYPE ( S2QP_control_type ), INTENT( OUT )   :: control
 TYPE ( S2QP_inform_type ),  INTENT( OUT )   :: inform

 !-----------------------------------------------------------------------------
 ! L o c a l  A r g u m e n t s
 !-----------------------------------------------------------------------------

 real( kind=wp ) :: infinity

 !-----------------------------------------------------------------------------

 ! Control parameters specifically for S2QP.
 ! -----------------------------------------

 ! Error and ordinary output unit numbers.

 control%error = 6
 control%out   = 6

 ! Removal of file alive_file from unit alive_unit causes execution to cease.

 control%alive_unit = 60
 control%alive_file = ''

 ! Level of output required. <= 0 gives no output, = 1 gives a one-line
 ! summary for every iteration, = 2 gives a summary of the inner iteration
 ! for each iteration, >= 3 gives increasingly verbose (debugging) output.

 control%print_level = 0

 ! Number of components of vectors to be printed.  If print_number is positive,
 ! then will print the first print_number components AND the last print_number
 ! components of each vector printed during the course of a run.

 control%print_number = 10

 ! How often major header is printed.

 control%header_every = 25

 ! Maximum number of nonmonotone steps allowed.

 control%NM_steps = 1

 ! Maximum number of iterations.

 control%max_iterate = 1000

 ! Any printing will start on this iteration.

 control%start_print = - 1

 ! Any printing will stop on this iteration.

 control%stop_print = - 1

 ! Printing will only occur every print_gap iterations.

 control%print_gap = 1

 ! Convergence tolerances. The iteration will terminate when either the norm
 ! of violation of the constraints (the "primal infeasibility") is smaller than
 ! control%stop_p and the norm of the gradient of the Lagrangian function (the
 ! "dual infeasibility") is smaller than control%stop_d, or when relevant
 ! relative primal and dual tolerances are satisfied.  See documentation for
 ! S2QP for more information.

 control%stop_p_abs = tenm6
 control%stop_c_abs = tenm6
 control%stop_d_abs = tenm6

 control%stop_p_rel = tenm7
 control%stop_c_rel = tenm7
 control%stop_d_rel = tenm7

 ! Initial and maximum values for the trust-region radius for both predictor
 ! and accelerator subproblems.

 control%initial_TRpred = ten
 control%max_TRpred     = tenp5
 control%use_TRpred     = .true.
 control%TRacc_scale    = three

 ! Type of positive definite approximation used for B. 0 = identity,
 ! 1 = wighted diagonal, 2 = LBFGS, 3 = BFGS.  If LBFGS is used,
 ! then L_BFGS_number tells the number of vectors that will be used.

 control%B_type           = 2
 control%L_BFGS_number    = 5
 control%L_BFGS_curve_mod = 1

 ! User defined infinity

 control%infinity = ten ** 19

 ! Initial and maximum value of penalty parameter and expansion factor.

 control%initial_penalty   = one
 control%max_penalty       = tenp8
 control%penalty_expansion = ten

 ! Trust-region radius is adjusted based on whether M - Mnew is larger
 ! than control%eta_successful(eta_very_successful) times that predicted
 ! by a quadratic model of the decrease.

 control%eta_successful           = point1
 control%eta_very_successful      = point7
 control%eta_extremely_successful = point9

 ! Fulsol specifies whether full solution or only highlights are printed.

 control%print_sol = .false.
 control%fulsol    = .false.

 ! Decide what type of accelerator step to try (when appropriate).

 control%use_seqp = .true.
 control%use_siqp = .true.

 ! If deallocate_error_fatal is true, any array/pointer deallocation error
 ! will terminate execution. Otherwise, computation will continue.

 control%deallocate_error_fatal = .false.

 ! Control parameters for various QP subproblems. These are the default values.
 ! Note: The values for infinity in QPA, QPB, LSQP, and QPC are overwritten.
 !-----------------------------------------------------------------------------

 ! Define the maximum infinity value to be used in QPA, QPB, and QPC.

 infinity  = control%infinity / hundred

 ! Intialize QPC data for feasible point subproblem.

 CALL QPC_initialize( data%QPfeas_data, control%QPfeas_control, &
                      inform%QPfeas_inform )

 if ( inform%QPfeas_inform%status /= GALAHAD_ok ) then
    write(control%error,1000) ;  go to 990
 end if

 control%QPfeas_control%treat_zero_bounds_as_general = .false.
 control%QPfeas_control%prefix   = ' -QPC'
 control%QPfeas_control%infinity = &
      min(infinity, control%QPfeas_control%infinity)

 control%QPfeas_control%QPA_control%prefix   = ' -QPA'
 control%QPfeas_control%QPA_control%infinity = &
      min(infinity, control%QPfeas_control%QPA_control%infinity)

 control%QPfeas_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPfeas_control%QPB_control%prefix   = ' -QPB'
 control%QPfeas_control%QPB_control%infinity = &
      min(infinity, control%QPfeas_control%QPB_control%infinity)

 control%QPfeas_control%QPB_control%LSQP_control%treat_zero_bounds_as_general= &
                                                                         .false.
 control%QPfeas_control%QPB_control%LSQP_control%prefix   = ' -LSQP'
 control%QPfeas_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPfeas_control%QPB_control%LSQP_control%infinity)

 ! Initialize QPC data for steering LP subproblem.

 CALL QPC_initialize( data%QPsteer_data, control%QPsteer_control, &
                      inform%QPsteer_inform )

 if ( inform%QPsteer_inform%status /= GALAHAD_ok ) then
    write(control%error,1000) ;  go to 990
 end if

 control%QPsteer_control%treat_zero_bounds_as_general = .false.
 control%QPsteer_control%prefix   = ' -QPC'
 control%QPsteer_control%infinity = &
      min(infinity, control%QPsteer_control%infinity)

 control%QPsteer_control%QPA_control%prefix   = ' -QPA'
 control%QPsteer_control%QPA_control%infinity = &
      min(infinity, control%QPsteer_control%QPA_control%infinity)

 control%QPsteer_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPsteer_control%QPB_control%prefix   = ' -QPB'
 control%QPsteer_control%QPB_control%infinity = &
      min(infinity, control%QPsteer_control%QPB_control%infinity)

 control%QPsteer_control%QPB_control%LSQP_control%treat_zero_bounds_as_general=&
                                                                         .false.
 control%QPsteer_control%QPB_control%LSQP_control%prefix   = ' -LSQP'
 control%QPsteer_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPsteer_control%QPB_control%LSQP_control%infinity)

 ! Intialize QPC data for predictor QP subproblem.

 CALL QPC_initialize( data%QPpred_data, control%QPpred_control, &
                      inform%QPpred_inform )

 if ( inform%QPpred_inform%status /= GALAHAD_ok ) then
    write(control%error,1000) ;  go to 990
 end if

 control%QPpred_control%treat_zero_bounds_as_general = .false.
 control%QPpred_control%prefix   = ' -QPC'
 control%QPpred_control%infinity = &
      min(infinity, control%QPpred_control%infinity)

 control%QPpred_control%QPA_control%prefix   = ' -QPA'
 control%QPpred_control%QPA_control%infinity = &
      min(infinity, control%QPpred_control%QPA_control%infinity)

 control%QPpred_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPpred_control%QPB_control%prefix   = ' -QPB'
 control%QPpred_control%QPB_control%infinity = &
      min(infinity, control%QPpred_control%QPB_control%infinity)

 control%QPpred_control%QPB_control%LSQP_control%treat_zero_bounds_as_general=&
                                                                         .false.
 control%QPpred_control%QPB_control%LSQP_control%prefix   = ' -LSQP'
 control%QPpred_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPpred_control%QPB_control%LSQP_control%infinity)

 ! Intialize QPC data for SIQP accelerator QP subproblem.

 CALL QPC_initialize( data%QPsiqp_data, control%QPsiqp_control, &
                      inform%QPsiqp_inform )

 if ( inform%QPsiqp_inform%status /= GALAHAD_ok ) then
    write(control%error,1000) ;  go to 990
 end if

 control%QPsiqp_control%treat_zero_bounds_as_general = .false.
 control%QPsiqp_control%prefix   = ' -QPC'
 control%QPsiqp_control%infinity = &
      min(infinity, control%QPsiqp_control%infinity)

 control%QPsiqp_control%QPA_control%prefix   = ' -QPA'
 control%QPsiqp_control%QPA_control%infinity = &
      min(infinity, control%QPsiqp_control%QPA_control%infinity)

 control%QPsiqp_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPsiqp_control%QPB_control%prefix   = ' -QPB'
 control%QPsiqp_control%QPB_control%infinity = &
      min(infinity, control%QPsiqp_control%QPB_control%infinity)

 control%QPsiqp_control%QPB_control%LSQP_control%treat_zero_bounds_as_general=&
                                                                         .false.
 control%QPsiqp_control%QPB_control%LSQP_control%prefix   = ' -LSQP'
 control%QPsiqp_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPsiqp_control%QPB_control%LSQP_control%infinity)

 ! Intialize EQP data for SEQP accelerator QP subproblem.

 CALL EQP_initialize( data%QPseqp_data, control%QPseqp_control, &
                      inform%QPseqp_inform )

 if ( inform%QPseqp_inform%status /= GALAHAD_ok ) then
    write(control%error,1001) ;  go to 991
 end if

 ! Different returns
 !------------------

 ! normal return

 inform%status = GALAHAD_ok
 return

 ! return because qpc_initialize failed.

 990 continue

 inform%status = -31
 return

 ! return because eqp_initialize failed.

 991 continue

 inform%status = -32
 return

 ! format statements

 1000 format(1x,'ERROR s2qp_initialize:qpc_initialize failed.')
 1001 format(1x,'ERROR s2qp_initialize:eqp_initialize failed.')

 END SUBROUTINE S2QP_initialize

 !*********  G A L A H A D S2QP_read_specfile  S U B R O U T I N E  ***********

 SUBROUTINE S2QP_read_specfile( control, device, alt_specname_S2QP,            &
                                alt_specname_QPfeas, alt_specname_QPpred,      &
                                alt_specname_QPsiqp, alt_specname_QPseqp,      &
                                alt_specname_QPsteer )

 !-----------------------------------------------------------------------------
 ! Purpose: Reads the content of a specification file, and performs the
 !          assignment of values associated with given keywords to the
 !          corresponding control parameters in S2QP, and all cascading
 !          control parameters contained within its QP subproblems.
 !
 ! Note: Once the S2QP control parameters are read in, it is followed by
 !       reading the control parameters for the QP subproblems and all
 !       cascading control parameter.  However, calls to QPC_modify_control
 !       and EQP_modify_control overwrites the following control parameters
 !       in all subproblem control parameters: error, out, prefix, infinity
 !       and deallocate_error_fatal.
 !
 ! The default values as given by S2QP_initialize could (roughly)
 ! have been set as:
 !
 ! BEGIN S2QP SPECIFICATIONS (DEFAULT)
 !  error-printout-device                           6
 !  printout-device                                 6
 !  alive-device                                    60
 !  print-level                                     0
 !  print-number                                    10
 !  print-header-every                              25
 !  start-print                                     -1
 !  stop-print                                      -1
 !  iterations-between-printing                     1
 !  print-solution                                  no
 !  print-full-solution                             no
 !  maximum-number-of-iterations                    1000
 !  ACC-trust-region-scale-factor                   3.0D+0
 !  B-type                                          2
 !  L-BFGS-curvature-fix-type                       1
 !  number-limited-memory-vectors                   5
 !  non-monotone-steps                              1
 !  absolute-primal-accuracy-required               1.0D-6
 !  absolute-dual-accuracy-required                 1.0D-6
 !  absolute-complementarity-accuracy-required      1.0D-6
 !  relative-primal-accuracy-required               1.0D-8
 !  relative-dual-accuracy-required                 1.0D-8
 !  relative-complementarity-accuracy-required      1.0D-8
 !  initial-radius-predictor                        1.0D+2
 !  maximum-radius-predictor                        1.0D+5
 !  use-predictor-trust-region                      yes
 !  successful-iteration-tolerance                  0.01
 !  very-successful-iteration-tolerance             0.7
 !  extremely-successful-iteration-tolerance        0.9
 !  infinity                                        1.0D+19
 !  initial-penalty-parameter                       1.0D+0
 !  maximum-penalty-parameter                       1.0D+8
 !  penalty-parameter-expansion-factor              10.0D+0
 !  use-seqp                                        yes
 !  use-siqp                                        yes
 !  deallocate-error-fatal                          no
 !  alive-filename
 ! END S2QP SPECIFICATIONS
 !
 !-----------------------------------------------------------------------------
 !   D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 TYPE ( S2QP_control_type ), INTENT( INOUT ) :: control
 INTEGER, INTENT( IN ) :: device
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_S2QP
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_QPfeas
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_QPpred
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_QPsiqp
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_QPseqp
 CHARACTER( LEN = 18 ), OPTIONAL :: alt_specname_QPsteer

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 INTEGER, PARAMETER :: lspec = 60
 CHARACTER( LEN = 18 ), PARAMETER :: specname_S2QP    = 'S2QP              '
 CHARACTER( LEN = 18 ), PARAMETER :: specname_QPfeas  = 'feasibility-QPC   '
 CHARACTER( LEN = 18 ), PARAMETER :: specname_QPpred  = 'predictor-QPC     '
 CHARACTER( LEN = 18 ), PARAMETER :: specname_QPsiqp  = 'accelerator(I)-QPC'
 CHARACTER( LEN = 18 ), PARAMETER :: specname_QPseqp  = 'accelerator(E)-EQP'
 CHARACTER( LEN = 18 ), PARAMETER :: specname_QPsteer = 'steering-QPC      '
 INTEGER :: error
 REAL( kind=wp ) :: infinity
 TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

 ! Control parameters for S2QP.
 !-----------------------------

 ! Define the keywords

 spec%keyword = ''

 ! Integer key-words

 spec(  1 )%keyword = 'error-printout-device'
 spec(  2 )%keyword = 'printout-device'
 spec(  3 )%keyword = 'alive-device'
 spec(  4 )%keyword = 'print-level'
 spec(  5 )%keyword = 'maximum-number-of-iterations'
 spec(  6 )%keyword = 'start-print'
 spec(  7 )%keyword = 'stop-print'
 spec(  8 )%keyword = 'iterations-between-printing'
 spec( 11 )%keyword = 'print-header-every'
 spec( 13 )%keyword = 'non-monotone-steps'
 spec( 14 )%keyword = 'B-type'
 spec( 15 )%keyword = 'number-limited-memory-vectors'
 spec( 16 )%keyword = 'L-BFGS-curvature-fix-type'

 ! Real key-words

 spec( 17 )%keyword = 'absolute-primal-accuracy-required'
 spec( 18 )%keyword = 'absolute-dual-accuracy-required'
 spec( 19 )%keyword = 'absolute-complementarity-accuracy-required'
 spec( 20 )%keyword = 'relative-primal-accuracy-required'
 spec( 21 )%keyword = 'relative-dual-accuracy-required'
 spec( 22 )%keyword = 'relative-complementarity-accuracy-required'

 spec( 23 )%keyword = 'successful-iteration-tolerance'
 spec( 24 )%keyword = 'very-successful-iteration-tolerance'
 spec( 25 )%keyword = 'extremely-successful-iteration-tolerance'
 spec( 26 )%keyword = 'initial-radius-predictor'
 spec( 27 )%keyword = 'maximum-radius-predictor'
 spec( 28 )%keyword = 'ACC-trust-region-scale-factor'

 spec( 29 )%keyword = 'infinity'
 spec( 30 )%keyword = 'initial-penalty-parameter'
 spec( 31 )%keyword = 'maximum-penalty-parameter'
 spec( 32 )%keyword = 'penalty-parameter-expansion-factor'

 ! Logical key-words

 spec( 41 )%keyword = 'deallocate-error-fatal'
 spec( 43 )%keyword = 'use-seqp'
 spec( 44 )%keyword = 'use-siqp'

 spec( 48 )%keyword = 'print-solution'
 spec( 49 )%keyword = 'print-full-solution'

 spec( 51 )%keyword = 'use-predictor-trust-region'

 ! More Integer key-words

 spec( 59 )%keyword = 'print-number'

 ! Character key-words

 spec( 60 )%keyword = 'alive-filename'

 ! Read the specfile

 if ( present( alt_specname_S2QP ) ) then
    CALL SPECFILE_read(device, alt_specname_S2QP, spec, lspec, control%error)
 else
    CALL SPECFILE_read(device, specname_S2QP, spec, lspec, control%error)
 end if

 ! Set integer values

 CALL SPECFILE_assign_integer( spec( 1 ), control%error, control%error )

 error = control%error

 CALL SPECFILE_assign_integer( spec( 2 ), control%out,               error )
 CALL SPECFILE_assign_integer( spec( 3 ), control%alive_unit,        error )
 CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,       error )
 CALL SPECFILE_assign_integer( spec( 5 ), control%max_iterate,       error )
 CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,       error )
 CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,        error )
 CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,         error )
 CALL SPECFILE_assign_integer( spec( 11 ), control%header_every,     error )
 CALL SPECFILE_assign_integer( spec( 13 ), control%NM_steps,         error )
 CALL SPECFILE_assign_integer( spec( 14 ), control%B_type,           error )
 CALL SPECFILE_assign_integer( spec( 15 ), control%L_BFGS_number,    error )
 CALL SPECFILE_assign_integer( spec( 16 ), control%L_BFGS_curve_mod, error )
 CALL SPECFILE_assign_integer( spec( 59 ), control%print_number,     error )

 ! Set real values

 CALL SPECFILE_assign_real( spec( 17 ), control%stop_p_abs,          error )
 CALL SPECFILE_assign_real( spec( 18 ), control%stop_d_abs,          error )
 CALL SPECFILE_assign_real( spec( 19 ), control%stop_c_abs,          error )
 CALL SPECFILE_assign_real( spec( 20 ), control%stop_p_rel,          error )
 CALL SPECFILE_assign_real( spec( 21 ), control%stop_d_rel,          error )
 CALL SPECFILE_assign_real( spec( 22 ), control%stop_c_rel,          error )
 CALL SPECFILE_assign_real( spec( 23 ), control%eta_successful,      error )
 CALL SPECFILE_assign_real( spec( 24 ), control%eta_very_successful, error )
 CALL SPECFILE_assign_real( spec( 26 ), control%initial_TRpred,      error )
 CALL SPECFILE_assign_real( spec( 27 ), control%max_TRpred,          error )
 CALL SPECFILE_assign_real( spec( 28 ), control%TRacc_scale,         error )
 CALL SPECFILE_assign_real( spec( 29 ), control%infinity,            error )
 CALL SPECFILE_assign_real( spec( 30 ), control%initial_penalty,     error )
 CALL SPECFILE_assign_real( spec( 31 ), control%max_penalty,         error )
 CALL SPECFILE_assign_real( spec( 32 ), control%penalty_expansion,   error )

 ! Set logical values

 CALL SPECFILE_assign_logical( spec( 41 ), control%deallocate_error_fatal,error)
 CALL SPECFILE_assign_logical( spec( 43 ), control%use_seqp,            error )
 CALL SPECFILE_assign_logical( spec( 44 ), control%use_siqp,            error )
 CALL SPECFILE_assign_logical( spec( 48 ), control%print_sol,           error )
 CALL SPECFILE_assign_logical( spec( 49 ), control%fulsol,              error )
 CALL SPECFILE_assign_logical( spec( 51 ), control%use_TRpred,          error )

 ! Set character values

 CALL SPECFILE_assign_string( spec( 60 ), control%alive_file, error )

 ! Control parameters for QP subproblems.
 ! Note: we modify the value of infinity in QPA, QPB, LSQP, and QPC to
 !       ensure that it is less than control%infinity/hundred.  This is
 !       also done in S2QP_initialize for the initialy infinity values.
 !---------------------------------------------------------------------

 ! Define the maximum infinity value to be used in QPA, QPB, and QPC.

 infinity  = control%infinity / hundred

 ! Feasibility QP control values

 if ( present( alt_specname_QPfeas ) ) then
    CALL QPC_read_specfile(control%QPfeas_control,device,alt_specname_QPfeas)
 else
    CALL QPC_read_specfile(control%QPfeas_control,device,specname_QPfeas)
 end if

 control%QPfeas_control%treat_zero_bounds_as_general = .false.
 control%QPfeas_control%infinity = &
      min(infinity, control%QPfeas_control%infinity)

 control%QPfeas_control%QPA_control%infinity = &
      min(infinity, control%QPfeas_control%QPA_control%infinity)

 control%QPfeas_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPfeas_control%QPB_control%infinity = &
      min(infinity, control%QPfeas_control%QPB_control%infinity)

 control%QPfeas_control%QPB_control%LSQP_control%treat_zero_bounds_as_general= &
                                                                         .false.
 control%QPfeas_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPfeas_control%QPB_control%LSQP_control%infinity)

 ! Predictor QP control values.

 if ( present( alt_specname_QPpred ) ) then
    CALL QPC_read_specfile(control%QPpred_control,device,alt_specname_QPpred)
 else
    CALL QPC_read_specfile(control%QPpred_control, device, specname_QPpred)
 end if

 control%QPpred_control%treat_zero_bounds_as_general = .false.
 control%QPpred_control%infinity = &
      min(infinity, control%QPpred_control%infinity)

 control%QPfeas_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPpred_control%QPA_control%infinity = &
      min(infinity, control%QPpred_control%QPA_control%infinity)

 control%QPpred_control%QPB_control%infinity = &
      min(infinity, control%QPpred_control%QPB_control%infinity)

 control%QPfeas_control%QPB_control%LSQP_control%treat_zero_bounds_as_general=&
                                                                         .false.
 control%QPpred_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPpred_control%QPB_control%LSQP_control%infinity)

 ! Steering QP control values.

 if ( present( alt_specname_QPsteer ) ) then
    CALL QPC_read_specfile(control%QPsteer_control,device,alt_specname_QPsteer)
 else
    CALL QPC_read_specfile(control%QPsteer_control,device,specname_QPsteer)
 end if

 control%QPsteer_control%treat_zero_bounds_as_general = .false.
 control%QPsteer_control%infinity = &
      min(infinity, control%QPsteer_control%infinity)

 control%QPsteer_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPsteer_control%QPA_control%infinity = &
      min(infinity, control%QPsteer_control%QPA_control%infinity)

 control%QPsteer_control%QPB_control%infinity = &
      min(infinity, control%QPsteer_control%QPB_control%infinity)

 control%QPsteer_control%QPB_control%LSQP_control%treat_zero_bounds_as_general=&
                                                                         .false.
 control%QPsteer_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPsteer_control%QPB_control%LSQP_control%infinity)

 ! SIQP accelerator QP control values.

 if ( present( alt_specname_QPsiqp )  ) then
    CALL QPC_read_specfile(control%QPsiqp_control, device, alt_specname_QPsiqp)
 else
    CALL QPC_read_specfile(control%QPsiqp_control, device, specname_QPsiqp)
 end if

 control%QPsiqp_control%treat_zero_bounds_as_general = .false.
 control%QPsiqp_control%infinity = &
      min(infinity, control%QPsiqp_control%infinity)

 control%QPsiqp_control%QPB_control%treat_zero_bounds_as_general = .false.
 control%QPsiqp_control%QPA_control%infinity = &
      min(infinity, control%QPsiqp_control%QPA_control%infinity)

 control%QPsiqp_control%QPB_control%infinity = &
      min(infinity, control%QPsiqp_control%QPB_control%infinity)

 control%QPsiqp_control%QPB_control%LSQP_control%treat_zero_bounds_as_general= &
                                                                         .false.
 control%QPsiqp_control%QPB_control%LSQP_control%infinity = &
      min(infinity, control%QPsiqp_control%QPB_control%LSQP_control%infinity)

 ! EQP accelerator control parameters.

 if ( present( alt_specname_QPseqp )  ) then
    CALL EQP_read_specfile(control%QPseqp_control, device, alt_specname_QPseqp)
 else
    CALL EQP_read_specfile(control%QPseqp_control, device, specname_QPseqp)
 end if

 return

 END SUBROUTINE S2QP_read_specfile

 !*************  G A L A H A D  S2QP_solve  S U B R O U T I N E  **************

 SUBROUTINE S2QP_solve( nlp, control, inform, data, userdata,                  &
                        eval_FC, eval_GJ, eval_HL )
 !-----------------------------------------------------------------------------
 !
 ! S2QP_solve : an SQP method for finding a local minimizer of a smooth
 !              objective function f(x) subject to general smooth constraints,
 !              linear constraints, and simple bounds on the variables.
 !
 ! control%print_level = GALAHAD_SILENT  : nothing is printed
 !                       GALAHAD_TRACE   : main line is printed (one line)
 !                       GALAHAD_ACTION  : additionally, main summary info.
 !                       GALAHAD_DETAILS :
 !                       GALAHAD_DEBUG   :
 !                       GALAHAD_CRAZY   : more or less everything.
 !
 ! inform%status : -31  maximum penalty parameter reached.
 !                 -32  converged to an infeasible stationary point.
 !                 -33  step too small, no further progress is possible.
 !                 -34  bounds and linear constraints appear to be infeasible.
 !                 -35  minimum predictor radius used and still unsuccessful.
 !
 !                 -50  error in subroutine build_QPfeas
 !                 -51  error in subroutine build_QPpred
 !                 -52  error in subroutine build_QPsteer
 !                 -53  error in subroutine build_QPseqp
 !                 -54  error in subroutine build_QPsiqp
 !                 -55  error in subroutine fill_QPfeas -- no longer exists.
 !                 -56  error in subroutine fill_QPpred
 !                 -57  error in subroutine fill_QPsteer -- no longer exists.
 !                 -58  error in subroutine fill_QPseqp
 !                 -59  error in subroutine fill_QPsiqp
 !
 !                 -60  not able to obtain a reasonable predictor step.
 !                 -61  not able to obtain a reasonable steering step.
 !                 -62  not able to obtain a reasonable Cauchy step.
 !                 -63  not able to obtain a reasonable SEQP step.
 !                 -64  not able to obtain a reasonable SIQP step.
 !
 !                 -70  error using subroutine get_L_BFGS
 !
 !                 -80  some problem function evaluation was undefined.
 !                 -81  inform%status /= 1 on entry.
 !
 !                 -98  something not yet implemented.
 !                 -99  part of the code has executed that should never occur.
 !
 ! use_seqp  use_siqp  action
 !    T         T      Compute seqp step if s_p is linearly feasible and
 !                     s_ac = s_p.  Otherwise, compute an siqp step.
 !    T         F      Compute seqp step at every iteration.
 !    F         T      Compute siqp step at every iteration.
 !    F         F      Use only predictor step.  Trust-region for predictor
 !                     step must be used.  Use change in convex model in the
 !                     ratio for determining step acceptance.
 !
 !-----------------------------------------------------------------------------

 !-----------------------------------------------------------------------------
 !   D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 TYPE ( NLPT_problem_type ),  INTENT( INOUT ) :: nlp
 TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
 TYPE ( S2QP_control_type ),  INTENT( INOUT ) :: control
 TYPE ( S2QP_inform_type ),   INTENT( INOUT ) :: inform
 TYPE ( S2QP_data_type ),     INTENT( INOUT ) :: data

 INTERFACE
    SUBROUTINE eval_FC(status, X, userdata, F, C)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
      REAL ( kind = wp ), OPTIONAL, INTENT( OUT ) :: F
      REAL ( kind = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_FC
    SUBROUTINE eval_GJ(status, X, userdata, G, J_val)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
      REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G, J_val
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_GJ
    SUBROUTINE eval_HL(status, X, Y, userdata, Hval,no_f)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
      LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
    END SUBROUTINE eval_HL
 END INTERFACE

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: i, iores, out, error, max_iterate, m, m_a, n, nLMV, NM_steps
 integer :: status, alloc_status, B_type, lenu, lenv, npupv
 integer :: print_level, print_number, dummy_int, j, l
 logical :: filexx, print_1line, print_debug, qpc_successful
 logical :: use_TRpred
 logical :: use_seqp, use_siqp, check_B, dummy_logic
 !  logical :: no_qpa_pred, no_qpb_pred, qpb_or_qpa_pred
 !  logical :: no_qpa_steer, no_qpb_steer, qpb_or_qpa_steer
 integer :: sfiledevice = 62                           ! solution file number
 character ( LEN = 30 ) :: sfilename = 'S2QPSOL.d'     ! solution file name
 ! integer :: QPpred_sif_out = 66
 real ( kind = wp ) :: dummy_real, dummy_real2, Bij, infinity, maxTR
 real ( kind = wp ) :: stop_p_abs, stop_d_abs, stop_c_abs
 real ( kind = wp ) :: L_factor, Q_factor
 !  real ( kind = wp ) :: muzero_pred, muzero_steer, Bij
 real ( kind = wp ), allocatable, dimension(:) :: ei, Bi
 real ( kind = wp ), allocatable, dimension(:,:) :: Bfull

 !-----------------------------------------------------------------------------

 ! For debugging purposes.

 check_B = .true. ! set true if want to call SILS to verify that B is SPD.

 ! The time that we have entered.

 call cpu_time( inform%time%enter_s2qp )

 ! Default informational variables.

 inform%iterate    = 0
 inform%primal_vl  = one
 inform%dual_vl    = one
 inform%comp_vl    = one
 inform%obj        = zero
 inform%num_f_eval = 0
 inform%num_g_eval = 0
 inform%num_c_eval = 0
 inform%num_J_eval = 0
 inform%num_H_eval = 0
 inform%num_predictors = 0
 inform%num_descent_active = 0

 inform%time%feas_preprocess = zero
 inform%time%feas_analyse    = zero
 inform%time%feas_factorize  = zero
 inform%time%feas_solve      = zero
 inform%time%feas_depend     = zero
 inform%time%feas_total      = zero

 inform%time%pred_A_preprocess = zero ;  inform%time%pred_B_preprocess = zero
 inform%time%pred_A_analyse    = zero ;  inform%time%pred_B_analyse    = zero
 inform%time%pred_A_factorize  = zero ;  inform%time%pred_B_factorize  = zero
 inform%time%pred_A_solve      = zero ;  inform%time%pred_B_solve      = zero
 inform%time%pred_A_total      = zero ;  inform%time%pred_B_total      = zero

 inform%time%pred_C_preprocess = zero
 inform%time%pred_C_depend     = zero
 inform%time%pred_C_analyse    = zero
 inform%time%pred_C_factorize  = zero
 inform%time%pred_C_solve      = zero
 inform%time%pred_C_total      = zero

 inform%time%steer_A_preprocess = zero ; inform%time%steer_B_preprocess = zero
 inform%time%steer_A_analyse    = zero ; inform%time%steer_B_analyse    = zero
 inform%time%steer_A_factorize  = zero ; inform%time%steer_B_factorize  = zero
 inform%time%steer_A_solve      = zero ; inform%time%steer_B_solve      = zero
 inform%time%steer_A_total      = zero ; inform%time%steer_B_total      = zero

 inform%time%steer_C_preprocess = zero
 inform%time%steer_C_depend     = zero
 inform%time%steer_C_analyse    = zero
 inform%time%steer_C_factorize  = zero
 inform%time%steer_C_solve      = zero
 inform%time%steer_C_total      = zero

 inform%time%cauchy_total = zero

 inform%time%seqp_factorize  = zero
 inform%time%seqp_solve      = zero
 inform%time%seqp_total      = zero

 inform%time%siqp_A_preprocess = zero ; inform%time%siqp_B_preprocess = zero
 inform%time%siqp_A_analyse    = zero ; inform%time%siqp_B_analyse    = zero
 inform%time%siqp_A_factorize  = zero ; inform%time%siqp_B_factorize  = zero
 inform%time%siqp_A_solve      = zero ; inform%time%siqp_B_solve      = zero
 inform%time%siqp_A_total      = zero ; inform%time%siqp_B_total      = zero

 inform%time%siqp_C_preprocess = zero
 inform%time%siqp_C_depend     = zero
 inform%time%siqp_C_analyse    = zero
 inform%time%siqp_C_factorize  = zero
 inform%time%siqp_C_solve      = zero
 inform%time%siqp_C_total      = zero

 inform%time%opt_test_total = zero

 inform%time%total = zero

 inform%time%total_preprocess = zero
 inform%time%total_depend     = zero
 inform%time%total_analyse    = zero
 inform%time%total_factorize  = zero
 inform%time%total_solve      = zero
 inform%time%total_total      = zero

 ! Exit immediately if inform%status /= 1.

 if ( inform%status /=  1 ) then ; inform%status = -81 ; go to 814 ; end if

 ! Copy control into data and make sure certain control parameters make sense.

 data%control = control

 if ( .not. (control%use_siqp .or. control%use_seqp) ) then
    data%control%use_TRpred = .true.
 end if

 ! To prevent printing errors (nlpt module should be updated).

 nlp%infinity = control%infinity

 ! trust-region related variables.

 data%TRpred_expand   = two
 data%TRpred_contract = half
 data%TR_reset_value  = tenm2

 ! do not use previous predictor step, because there isn't any!

 data%use_prev_pred = .false.

 ! initial penalty parameter.

 data%penalty = data%control%initial_penalty

 ! steering related variables.

 data%steer_L_factor = point8
 data%steer_Q_factor = point1

! For convenience.

 m            = max(0,nlp%m)
 m_a          = max(0,nlp%m_a)
 n            = nlp%n

 out          = data%control%out
 error        = data%control%error
 max_iterate  = data%control%max_iterate
 stop_p_abs   = data%control%stop_p_abs
 stop_d_abs   = data%control%stop_d_abs
 stop_c_abs   = data%control%stop_c_abs
 nLMV         = data%control%L_BFGS_number
 B_type       = data%control%B_type
 use_siqp     = data%control%use_siqp
 use_seqp     = data%control%use_seqp
 NM_steps     = data%control%NM_steps
 use_TRpred   = data%control%use_TRpred
 print_level  = data%control%print_level
 print_number = data%control%print_number
 infinity     = data%control%infinity
 maxTR        = data%control%max_TRpred

 L_factor     = data%steer_L_factor
 Q_factor     = data%steer_Q_factor

 ! iterate feasibility info for general constraints.

 data%num_sat = 0 ;  data%num_vl_l = 0 ; data%num_vl_u = 0

 ! Cauchy/approximate Cauchy step related variables.

 data%lbreak    = 2*m
 data%ac_factor = tenm2

 ! nonmonotone related variables.

 data%NM%active   = .false.
 data%NM%num_fail = 0

 ! algorithm progress related variables.

 data%iterate              = 1
 data%BFGS%damp_factor     = 0.2_wp
 data%BFGS%update_number   = 0
 data%L_BFGS%damp_factor   = 0.2_wp
 data%L_BFGS%update_number = 0
 data%seqp_try_pred        = .true.

 ! allocate a bunch of vectors.

 CALL SPACE_resize_array( n, data%s_p, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%s_p_saved, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%s_c, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%s_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%s_f, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%s_steer, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%HxSp, status, inform%alloc_status  )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%HxSp_saved, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%HxSc, status, inform%alloc_status  )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%HxSs, status, inform%alloc_status  )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%HxSf, status, inform%alloc_status  )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%BxSp, status, inform%alloc_status  )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array(n, data%GplusHs, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%descent_con, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%X_type, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%X_RES_l, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%X_RES_u, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%spos, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%sneg, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%Z_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%H_norms, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( nlp%H%ne, data%Hval_saved, status,inform%alloc_status)
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSp, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSp_saved, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSc, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSac, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSs, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSsteer, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%JxSf, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%Jpos, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%Jneg, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%JtY, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%JtY_cur, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%JtY_p, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%JtY_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array(m, data%C_new, status, inform%alloc_status)
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%C_type, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%CplusJSc, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%w, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%J_norms, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%Y_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%sat, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%vl_l, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%vl_u, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array(data%lbreak, data%IBREAK, status, inform%alloc_status)
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array(data%lbreak, data%BREAKP, status, inform%alloc_status)
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%C_RES_l, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%C_RES_u, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%C_RES_l_new, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%C_RES_u_new, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%cauchyRESl, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m, data%cauchyRESu, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AXplusSc, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AxSp, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AxSp_saved, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AxSc, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AxSac, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%AxSs, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%Apos, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%Aneg, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%Ya_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%AtYa, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%AtYa_cur, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%AtYa_p, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%AtYa_s, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%A_type, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%A_RES_l, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( m_a, data%A_RES_u, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( n, data%X_trial, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 ! Vectors used for either reverting values or for BFGS old values.

 if ( NM_steps > 0 .or. B_type == 2 .or. B_type == 3 ) then
   call SPACE_resize_array( n, data%revert%G, status, inform%alloc_status )
   if ( status /= GALAHAD_ok ) go to 990
   if  ( m > 0 ) then
     call SPACE_resize_array( nlp%J%ne, data%revert%Jval, &
                              status, inform%alloc_status)
     if ( status /= GALAHAD_ok ) go to 990
   end if
 end if
 if ( NM_steps > 0 .or. B_type == 3 ) then
   call SPACE_resize_array( n, data%revert%X, status, inform%alloc_status )
   if ( status /= GALAHAD_ok ) go to 990
 end if

 ! Vectors exclusively used for reverting to previous point.

 if ( NM_steps > 0 ) then
    CALL SPACE_resize_array( n, data%revert%Z, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( m, data%revert%C, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( m, data%revert%Y, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( m_a, data%revert%Ax, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( m_a, data%revert%Y_a, status, inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(m, data%revert%C_RES_l, status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(m, data%revert%C_RES_u, status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(m_a,data%revert%A_RES_l,status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(m_a,data%revert%A_RES_u,status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(n, data%revert%X_RES_l, status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array(n, data%revert%X_RES_u, status,inform%alloc_status)
    IF ( status /= GALAHAD_ok ) GO TO 990
 end if

 ! Compute general constraint types.

 data%nclb = 0 ;  data%ncub = 0 ;  data%ncrb = 0 ;  data%nce = 0

 if ( m > 0 ) then
    do i = 1, m
       if ( nlp%C_l(i) > -infinity ) then
          if ( nlp%C_u(i) < infinity ) then
             if ( abs( nlp%C_u(i) - nlp%C_l(i) ) > two*stop_p_abs ) then
                data%C_type(i) = 'RB'
                data%ncrb = data%ncrb + 1
             else
                data%C_type(i) = 'EQ'
                data%nce = data%nce + 1
                nlp%C_u( i )   = ( nlp%C_l( i ) + nlp%C_u(i) ) / two
                nlp%C_l( i )   = nlp%C_u( i )
             end if
          else
             data%C_type(i) = 'LB'
             data%nclb = data%nclb + 1
          end if
       elseif ( nlp%C_u(i) >= infinity ) then
          data%C_type(i) = 'FR'
       else
          data%C_type(i) = 'UB'
          data%ncub = data%ncub + 1
       end if
    end do

    CALL SPACE_resize_array( data%nclb, data%clb, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( data%ncub, data%cub, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( data%ncrb, data%crb, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( data%nce, data%ce, status, inform%alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990

    data%nclb = 0 ;  data%ncub = 0 ;  data%ncrb = 0 ;  data%nce = 0 ;

    do i = 1, m
       select case ( data%C_type(i) )
       case ( 'LB' )
          data%nclb = data%nclb + 1
          data%clb( data%nclb ) = i
       case ( 'UB' )
          data%ncub = data%ncub + 1
          data%cub( data%ncub ) = i
       case ( 'RB' )
          data%ncrb = data%ncrb + 1
          data%crb( data%ncrb ) = i
       case ( 'EQ' )
          data%nce = data%nce + 1
          data%ce( data%nce ) = i
       case default
          ! relax
       end select
    end do

 end if

 lenu  = data%ncrb + data%nce + data%nclb ! u-elastics in predictor/steering.
 lenv  = data%ncrb + data%nce + data%ncub ! v-elastics in predictor/steering.
 npupv = n + lenu + lenv

 CALL SPACE_resize_array( lenu, data%u_in, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( lenv, data%v_in, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( lenu, data%u_out, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( lenv, data%v_out, status, inform%alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 ! Linear constraint types.

 if ( m_a > 0 ) then
    do i = 1, m_a
       if ( nlp%A_l(i) > -infinity ) then
          if ( nlp%A_u(i) < infinity ) then
             if ( abs( nlp%A_u(i) - nlp%A_l(i) ) > two*stop_p_abs ) then
                data%A_type(i) = 'RB'
             else
                data%A_type(i) = 'EQ'
                nlp%A_l( i )   = ( nlp%A_l( i ) + nlp%A_u( i ) ) / two
                nlp%A_u( i )   = nlp%A_l( i )
             end if
          else
             data%A_type(i) = 'LB'
          end if
       elseif ( nlp%A_u(i) >= infinity ) then
          data%A_type(i) = 'FR'
       else
          data%A_type(i) = 'UB'
       end if
    end do
 end if

 ! Bound constraint types.

 do i = 1, n
    if ( nlp%X_l(i) > -infinity ) then
       if ( nlp%X_u(i) < infinity ) then
          if ( abs( nlp%X_u(i) - nlp%X_l(i) ) > two*stop_p_abs ) then
             data%X_type(i) = 'RB'
          else
             data%X_type(i) = 'EQ'
             nlp%X_l( i )   = ( nlp%X_l( i ) + nlp%X_u( i ) ) / two
             nlp%X_u( i )   = nlp%X_l( i )
          end if
       else
          data%X_type(i) = 'LB'
       end if
    elseif ( nlp%X_u(i) >= infinity ) then
       data%X_type(i) = 'FR'
    else
       data%X_type(i) = 'UB'
    end if
 end do

 ! Allocate and set sparsity for predictor subproblem.

 call build_QPpred( nlp, data%QPpred, status, data )
 if ( status /= GALAHAD_ok ) then ; inform%status = -51 ; go to 814; end if

 ! Allocate and set sparsity for steering subproblem.

 if ( m > 0 ) then
    call build_QPsteer( nlp, data%QPsteer, status, data%control, data )
    if ( status /= GALAHAD_ok ) then ; inform%status = -52 ; go to 814 ; end if
 end if

 ! Allocate and set sparsity for accelerator subproblems.

 if ( use_seqp ) then
    call build_QPseqp( nlp, data%QPseqp, status, data )
    if ( status /= GALAHAD_ok ) then ; status = -53 ; go to 814 ; end if
 end if

 if ( use_siqp ) then
    call build_QPsiqp( nlp, data%QPsiqp, status, control )
    if ( status /= GALAHAD_ok ) then ; status = -54 ; go to 814 ; end if
 end if

 ! Make x feasible with respect to the bounds.

 if ( print_level >= GALAHAD_debug ) then
    call print_real_vec( 'X-input', nlp%X, 5, print_number, out, error )
 end if

 nlp%X = min( max( nlp%X, nlp%X_l ), nlp%X_u )

 if ( print_level >= GALAHAD_debug ) then
    call print_real_vec('X-bound-feasible', nlp%X, 5, print_number, out, error)
 end if

 ! Find "closest" x that is feasible with respect to the linear constraints.

 if ( m_a > 0 ) then

    nlp%Ax = zero
    call mop_Ax( one, nlp%A, nlp%X, one, nlp%Ax, out, error )

    call get_residuals( m_a, nlp%A_l, nlp%Ax, nlp%A_u, &
                        data%A_type, data%A_RES_l, data%A_RES_u )

    call L1_viol(m_a, data%A_RES_l, data%A_RES_u, data%A_type, dummy_real, out)

    call build_QPfeas( nlp, data%QPfeas, status, data%control)

    if ( status /= GALAHAD_ok ) then
       inform%status = -50 ;  go to 813
    end if

    if ( dummy_real >= min(tenm7,stop_p_abs) ) then

       call fill_QPfeas( nlp, data%QPfeas )

       if ( print_level >= GALAHAD_crazy ) then
          write(out,*) ' ================ BEGIN : QPfeas ================='
          call QPT_write_problem( out, data%QPfeas, 1 )
          write(out,*) ' Ane = ', data%QPfeas%A%ne
          write(out,*) ' Hne = ', data%QPfeas%H%ne
          write(out,*) ' ================ END : QPfeas ===================='
       end if

       call QPC_solve( data%QPfeas, data%QPfeas%C_status,                &
                       data%QPfeas%X_status, data%QPfeas_data,           &
                       data%control%QPfeas_control, inform%QPfeas_inform )

       data%QPfeas%new_problem_structure = .false.

       inform%time%feas_total      = inform%time%feas_total &
                                   + inform%QPfeas_inform%time%total
       inform%time%feas_preprocess = inform%time%feas_preprocess &
                                   + inform%QPfeas_inform%time%preprocess
       inform%time%feas_depend     = inform%time%feas_depend &
                                   + inform%QPfeas_inform%time%find_dependent
       inform%time%feas_analyse    = inform%time%feas_analyse &
                                   + inform%QPfeas_inform%time%analyse
       inform%time%feas_factorize  = inform%time%feas_factorize &
                                   + inform%QPfeas_inform%time%factorize
       inform%time%feas_solve      = inform%time%feas_solve &
                                   + inform%QPfeas_inform%time%solve

       if ( inform%QPfeas_inform%status /= GALAHAD_ok ) then
          if ( inform%QPfeas_inform%status == GALAHAD_error_tiny_step ) then
             write(out,1007) ! Proceed on and hope for the best.
          else
             inform%status = GALAHAD_error_primal_infeasible ;  go to 814
          end if
       end if

       ! get the solution, recompute residuals, and print new value for x.

       nlp%X   = data%QPfeas%X( : n )
       nlp%Y_a = data%QPfeas%Y

       nlp%Ax = zero
       call mop_Ax( one, nlp%A, nlp%X, one, nlp%Ax, out, error )

       call get_residuals( m_a, nlp%A_l, nlp%Ax, nlp%A_u, &
                           data%A_type, data%A_RES_l, data%A_RES_u )

       call L1_viol(m_a,data%A_RES_l,data%A_RES_u,data%A_type,dummy_real,out)

       if ( dummy_real > min(tenm6,stop_p_abs) ) then
          inform%status = GALAHAD_error_primal_infeasible ;  go to 814
       end if

       if ( print_level >= GALAHAD_debug ) then
         call print_real_vec('X-linear-feasible',nlp%X,5,print_number,out,error)
          write(out,*) ' Violation of A_l <= A <= A_u is : ', dummy_real
       end if

    else
       if ( print_level >= GALAHAD_debug ) write(out, 3078)
    end if

 end if

 ! Now that we have our initial x, define the minimum TR radius and TRpred

 data%min_TR = max( teneps * max( one, maxval(nlp%X) ), tenm10 )
 data%TRpred = max( data%min_TR, data%control%initial_TRpred )

 ! Evaluate f(x) and c(x), compute residuals, and compute ||c(x)||_1.

 if ( m > 0 ) then

    call eval_FC( status, nlp%X, userdata, nlp%f, nlp%c )
    if ( status /= GALAHAD_ok ) then
       write( out, 1002 ) 'eval_FC' ;  go to 993
    end if

    inform%num_c_eval = inform%num_c_eval + 1

    data%C_RES_l = zero
    data%C_RES_u = zero
    call get_residuals( m, nlp%C_l, nlp%c, nlp%C_u, &
                        data%C_type, data%C_RES_l, data%C_RES_u )

    call L1_viol(m, data%C_RES_l, data%C_RES_u, data%C_type, data%norm_c, out )

 else

    call eval_FC( status, nlp%X, userdata, nlp%f )
    if ( status /= GALAHAD_ok ) then
       write( out, 1002 ) 'eval_FC' ;  go to 993
    end if

    data%norm_c       = zero ;  data%norm_c_linearize_pred   = zero
    data%penalty      = zero ;  data%norm_c_linearize_cauchy = zero
    data%C_new        = zero ;  data%norm_c_linearize_steer  = zero
    data%norm_c_new   = zero ;  data%norm_c_linearize_full   = zero
    data%inf_norm_Y   = zero ;  data%inf_norm_Y_s            = zero
    data%inf_norm_Y_p = zero ;  data%dec_norm_c_pred         = zero

 end if

 inform%num_f_eval = inform%num_f_eval + 1

 ! Intial value of the merit function.

 data%merit = nlp%f + data%penalty * data%norm_c

 ! Initial residuals for bound constraints

 data%X_RES_l = zero
 data%X_RES_u = zero
 call get_residuals( n, nlp%X_l, nlp%X, nlp%X_u,  &
                     data%X_type, data%X_RES_l, data%X_RES_u )

 ! Evaluate g(x) and J(x)

 if ( m > 0 ) then

    call eval_GJ( inform%status, nlp%X, userdata, nlp%G, nlp%J%val )
    if ( inform%status /= GALAHAD_ok ) then
       write( out, 1002 ) 'eval_GJ' ;  go to 993
    end if

    inform%num_J_eval = inform%num_J_eval + 1

 else

    call eval_GJ( inform%status, nlp%X, userdata, nlp%G )
    if ( inform%status /= GALAHAD_ok ) then
       write( out, 1002 ) 'eval_GJ' ;  go to 993
    end if

 end if

 inform%num_g_eval = inform%num_g_eval + 1

 ! Print the problem to be solved.

 if ( print_level >= GALAHAD_crazy ) call NLPT_write_problem( nlp, out, 4 )

 ! Compute initial optimality measures.

 if ( m > 0 ) then
   data%inf_norm_Y = maxval( abs( nlp%Y( : m ) ) )
   data%JtY = zero
   call mop_Ax( one, nlp%J, nlp%Y, one, data%JtY, out, error, transpose=.true. )
 end if

 if ( m_a > 0 ) then
   data%AtYa = zero
   call mop_Ax(one, nlp%A, nlp%Y_a, one, data%AtYa, out, error,transpose=.true.)
 end if

 call get_opt( .true., nlp, data, nlp%G, data%JtY, data%AtYa, nlp%Z, nlp%Y,   &
               nlp%Y_a, data%C_type, data%A_type, data%X_type,                &
               data%C_RES_l, data%C_RES_u, data%A_RES_l, data%A_RES_u,        &
               data%X_RES_l, data%X_RES_u,                                    &
               data%primal_vl, data%dual_vl, data%comp_vl, data%converged, out)

 data%opt_measure = max( data%primal_vl, data%dual_vl, data%comp_vl )

 ! Print initial main line summary.

 if ( print_level >= GALAHAD_TRACE ) then
    write( out, 1000 )  ! header
    write( out, 1004 ) 0, data%penalty, data%merit,                &
                       data%primal_vl, data%dual_vl, data%comp_vl, &
                       '-', data%inf_norm_Y, data%TRpred
 end if

 ! If optimal, do not even enter main loop.

 if ( data%converged ) then
    inform%status = 0 ;  go to 813
 end if

 !============================================================================
 ! BEGIN: main do loop
 !============================================================================

 do while ( data%iterate <= max_iterate )

!   if ( data%success >= 1 .and. data%iterate >= 0 ) then
!      use_TRpred = .false.
!   elseif (data%iterate == 0 ) then
!      use_TRpred = .true.
!   else
!      use_TRpred = .true.
!   end if



  ! Compute the predictor step; can we use a previous one?
  ! ******************************************************

  if ( data%use_prev_pred ) then

     ! note 0: using prev_pred -> no TR_pred -> ACC step computed
     ! note 1: no ACC step computed -> use B model for convergence -> TR_pred
     ! note 2: data%dec_norm_c_pred --- not needed
     !         data%BxSp            --- not needed

     if (print_level >= GALAHAD_details ) then
        write( out, *) ' --Using previous predictor information.'
     end if

     data%s_p                   = data%s_p_saved
     data%two_norm_s_p          = data%two_norm_s_p_saved
     data%inf_norm_s_p          = data%inf_norm_s_p_saved
     data%inf_norm_Y_p          = data%inf_norm_Y_p_saved
     data%gtSp                  = data%gtSp_saved
     data%norm_c_linearize_pred = data%norm_c_linearize_pred_saved
     data%Sp_B_Sp               = data%Sp_B_Sp_saved
     data%decreaseB             = data%decreaseB_saved
     data%AxSp                  = data%AxSp_saved
     data%JxSp                  = data%JxSp_saved
     data%B%val                 = data%Bval_saved
     data%HxSp                  = data%HxSp_saved
     data%Sp_H_Sp               = data%Sp_H_Sp_saved
     nlp%H%val                  = data%Hval_saved

     data%computed_steering = .false.
     data%steering_good     = .true.

     go to 806

  end if

  ! Fill predictor subproblem -- QPpred.

  call fill_QPpred( nlp, data%QPpred, status, data )
  if ( status /= GALAHAD_ok ) then ; inform%status = -56 ; go to 813 ; end if

  ! When debugging, make sure it is positive definite.

  if ( B_type == 3 .and. check_B ) then ! BFGS
     call SILS_initialize( data%SILS_factors, data%control%SILS_control )
     data%control%SILS_control%pivoting = 1
     call SILS_analyse( data%B, data%SILS_factors, data%control%SILS_control,  &
          data%SILS_Ainfo )
     if ( data%SILS_Ainfo%flag /= 0 ) then
        write(error, *) ' **ERROR:s2qp_solve:SILS_analyse flag = ',            &
          data%SILS_Ainfo%flag
        return
     end if
     call SILS_factorize( data%B, data%SILS_factors,                           &
                          data%control%SILS_control, data%SILS_Finfo )
     if ( data%SILS_Finfo%flag /= 0 ) then
        write(*,*) 'data%B%row = ', data%B%row
        write(*,*) 'data%B%col = ', data%B%col
        write(*,*) 'data%B%val = ', data%B%val
        write(error, *) ' **ERROR:s2qp:SILS_factorize returned flag = ',       &
          data%SILS_Finfo%flag
        write(error, *) ' **ERROR:s2qp:SILS_factorize num_neg_eigen = ',       &
          data%SILS_Finfo%neig
        write(*,*) 'n = ', nlp%n
        write(error, *) ' **ERROR:s2qp:SILS_factorize rank = ',                &
          data%SILS_Finfo%rank
        write(error, *) ' **ERROR:s2qp:SILS_factorize num_zero_eige = ',       &
          nlp%n - data%SILS_Finfo%rank
        return
     end if
     if ( print_level >= GALAHAD_details ) then
        allocate( ei(1:n), Bi(1:n) )
        ei = zero
        write(*,*)
        do i = 1, n
           ei(i) = one
           Bi = zero
           call mop_Ax( one, data%B, ei, one, Bi, out, error, symmetric=.true. )
           ei(i) = zero
           write(*, "('BFGS : B ', I2, ' column is')") i
           write(*,*) Bi
        end do
        write(*,*)
        deallocate( ei, Bi )
     end if
  end if

  ! When debugging, make sure the update is positive definite.

  if ( B_type == 2 .and. data%L_BFGS%number_used > 0 .and. check_B ) then
     allocate( Bfull(1:n,1:n), stat=status )
     do i = 1, n
        do j = 1, n
           Bij = zero
           if ( i == j ) Bij = Bij + data%B%val(i)
           do l = 1, data%L_BFGS%number_used
              Bij = Bij - data%L_BFGS%A(i,l) * data%L_BFGS%A(j,l)
              Bij = Bij + data%L_BFGS%B(i,l) * data%L_BFGS%B(j,l)
           end do
           Bfull(i,j) = Bij
        end do
     end do
     if ( print_level >= GALAHAD_details ) then
        write(*,*)
        do i = 1, n
           write(*, "('L-BFGS : B ', I2, ' column is')") i
           write(*,*) Bfull(:,i)
        end do
        write(*,*)
     end if
     deallocate( Bfull )
  end if

  ! Set some default values.

  data%steering_good     = .false.
  data%computed_steering = .false.
  data%new_penalty       = .false.

  ! Do loop for predictor step to ensure linearized infeasibility for predictor
  ! step is comparable to the best decrease given by the steering step.

  do while ( .not. data%steering_good )

     ! Compute the predictor step.

     qpc_successful = .false.
     data%QP_fails  = 0

     do while ( .not. qpc_successful )

        if ( print_level >= GALAHAD_crazy ) then
           call print_predictor_problem( data, out )
        end if

        ! Extract initial elastic variables and solve the QP.

        !data%u_in = data%QPpred%X( n+1 : n+lenu )
        !data%v_in = data%QPpred%X( n+lenu+1 : npupv )

        call QPC_solve( data%QPpred, data%QPpred%C_status,                     &
                        data%QPpred%X_status, data%QPpred_data,                &
                        data%control%QPpred_control, inform%QPpred_inform  )

        inform%num_predictors = inform%num_predictors + 1

        if ( B_type /= 2 ) then
           data%QPpred%new_problem_structure = .false.
        end if

        inform%time%pred_A_total      = inform%time%pred_A_total &
                              + inform%QPpred_inform%QPA_inform%time%total
        inform%time%pred_A_preprocess = inform%time%pred_A_preprocess &
                              + inform%QPpred_inform%QPA_inform%time%preprocess
        inform%time%pred_A_analyse    = inform%time%pred_A_analyse &
                              + inform%QPpred_inform%QPA_inform%time%analyse
        inform%time%pred_A_factorize  = inform%time%pred_A_factorize &
                              + inform%QPpred_inform%QPA_inform%time%factorize
        inform%time%pred_A_solve      = inform%time%pred_A_solve &
                              + inform%QPpred_inform%QPA_inform%time%solve

        inform%time%pred_B_total      = inform%time%pred_B_total &
                              + inform%QPpred_inform%QPB_inform%time%total
        inform%time%pred_B_preprocess = inform%time%pred_B_preprocess &
                              + inform%QPpred_inform%QPB_inform%time%preprocess
        inform%time%pred_B_analyse    = inform%time%pred_B_analyse &
                              + inform%QPpred_inform%QPB_inform%time%analyse
        inform%time%pred_B_factorize  = inform%time%pred_B_factorize &
                              + inform%QPpred_inform%QPB_inform%time%factorize
        inform%time%pred_B_solve      = inform%time%pred_B_solve &
                              + inform%QPpred_inform%QPB_inform%time%solve

        inform%time%pred_C_total      = inform%time%pred_C_total &
                                      + inform%QPpred_inform%time%total
        inform%time%pred_C_preprocess = inform%time%pred_C_preprocess &
                                      + inform%QPpred_inform%time%preprocess
        inform%time%pred_C_analyse    = inform%time%pred_C_analyse &
                                      + inform%QPpred_inform%time%analyse
        inform%time%pred_C_factorize  = inform%time%pred_C_factorize &
                                      + inform%QPpred_inform%time%factorize
        inform%time%pred_C_solve      = inform%time%pred_C_solve &
                                      + inform%QPpred_inform%time%solve

        ! compute the norm of the linearization---use elastic variables.

        data%u_out = data%QPpred%X( n+1 : n+lenu )
        data%v_out = data%QPpred%X( n+lenu+1 : npupv )

        data%norm_c_linearize_pred = sum(data%u_out) + sum(data%v_out)

        ! The predictor step s_p, multipliers, and decrease in SMOOTH model.

        data%s_p          = data%QPpred%X( : n )
        data%inf_norm_s_p = MAXVAL( ABS(data%s_p) )

        if ( m > 0 ) then
           data%inf_norm_Y_p    = maxval( abs( data%QPpred%Y(:m) ) )
           data%dec_norm_c_pred = data%norm_c - data%norm_c_linearize_pred
           data%JxSp = zero
           call mop_Ax( one, nlp%J, data%s_p, one, data%JxSp, &
                        out, error, transpose=.false. )
           data%decreaseB_smooth = data%penalty * data%dec_norm_c_pred
        else
           data%decreaseB_smooth = zero
           !data%dec_norm_c_pred = zero
           !data%inf_norm_Y_p    = zero
        end if

        data%BxSp = zero
        call mop_Ax( one, data%B, data%s_p, one, data%BxSp, &
                     out, error, symmetric=.true. )

        data%Sp_B_Sp = DOT_PRODUCT( data%BxSp, data%s_p )
        data%gtSp    = DOT_PRODUCT( nlp%G, data%s_p )

        data%decreaseB_smooth = data%decreaseB_smooth - data%gtSp
        data%decreaseB_smooth = data%decreaseB_smooth - half * data%Sp_B_Sp
        if ( B_type == 2 .and. data%L_BFGS%number_used > 0 ) then
           dummy_int  = min( nLMV, data%L_BFGS%number_used )
           dummy_real = &  ! w_a^T w_a
              dot_product( data%QPpred%X( npupv+1 : npupv+dummy_int ), &
                           data%QPpred%X( npupv+1 : npupv+dummy_int ) )
           dummy_real2 = & ! w_b^T w_b
              dot_product( data%QPpred%X(npupv+dummy_int+1:npupv+2*dummy_int), &
                           data%QPpred%X(npupv+dummy_int+1:npupv+2*dummy_int) )
           data%decreaseB_smooth = &
              data%decreaseB_smooth + half*( dummy_real-dummy_real2 )
        end if

        data%decreaseB = data%decreaseB_smooth

        ! Print summary of some stuff

        if ( print_level >= GALAHAD_debug ) call print_predictor_info(data, nlp)

        ! Check to make sure predictor step is good.  If not, try to recover.

        if ( data%decreaseB >= -tenm5 .and. &
             inform%QPpred_inform%status == GALAHAD_ok ) then

           qpc_successful = .true.

           ! reset control parameters to defaul values,
           ! if needed. DPR: fix later.

           !if ( data%QP_fails /= 0 ) then
           !   data%control%QPpred_control%qpb_or_qpa         = qpb_or_qpa_pred
           !   data%control%QPpred_control%no_qpa             = no_qpa_pred
           !   data%control%QPpred_control%no_qpb             = no_qpb_pred
           !   data%control%QPpred_control%QPB_control%muzero = muzero_pred
           !end if

        else

           qpc_successful = .false.

           write(out,*) 'WARNING s2qp:trying to recover from predictor.'
           write(out, "('QPpred_inform%status = ', I3)")                       &
             inform%QPpred_inform%status
           inform%status = -1
           return  ! DPR: remove later
           !call qpc_recover( data%QPpred, data%control%QPpred_control,  &
           !                  inform%QPpred_inform, nlp, data%QP_fails,  &
           !                  run_level, ' pred', error, B_type, status )
           if ( status /= 0 ) then
              inform%status = -60 ; go to 813
           end if

        end if

     end do

     ! Compute product of A with predictor step.

     if ( m_a > 0 ) then
        data%AxSp = zero
        call mop_Ax( one, nlp%A, data%s_p, one, data%AxSp, &
                     out, error, transpose=.false. )
     end if

     ! If we are in a nonmonotone phase, then we do not want to change
     ! the penalty parameter; do not compute a steering direction.

     if ( data%NM%active ) then
        data%steering_good = .true.
        go to 805
     end if

     ! Check if the predictor step is "good enough".

     dummy_real = data%norm_c_linearize_pred

     if ( (dummy_real < point1 * stop_p_abs) .or. &
          (data%norm_c >= point1 .and. dummy_real < point1*data%norm_c) ) then

        data%steering_good = .true.
        goto 805

     end if

     ! If currently feasible, then ssteer = 0 : do not compute.

     if ( data%norm_c <= tenm2 * stop_p_abs ) go to 805

     ! If steering already has been computed, skip to steering conditions.

     if ( data%computed_steering ) go to 804

     ! Compute steering direction.

     data%computed_steering = .true.

     ! Define the steering QP--actually an LP.

     call fill_QPsteer( nlp, data%QPsteer, data )

     ! Do loop for steering direction

     data%QP_fails  = 0
     qpc_successful = .false.

     do while ( .not. qpc_successful )

        if ( print_level >= GALAHAD_crazy ) then
           write(out,*) ' ============== BEGIN : QPsteer ================'
           call QPT_write_problem( out, data%QPsteer, 1 )
           write(out,*) ' =============== END : QPsteer ================='
        end if

        ! Elastic variables going in.

        data%u_in = data%QPsteer%X( n+1 : n+lenu )
        data%v_in = data%QPsteer%X( n+lenu+1 : npupv )

        ! Solve the steering subproblem.

        call QPC_solve(data%QPsteer, data%QPsteer%C_status,               &
                       data%QPsteer%X_status, data%QPsteer_data,          &
                       data%control%QPsteer_control, inform%QPsteer_inform)

        data%QPsteer%new_problem_structure = .false.

        inform%time%steer_A_total      = inform%time%steer_A_total &
                             + inform%QPsteer_inform%QPA_inform%time%total
        inform%time%steer_A_preprocess = inform%time%steer_A_preprocess &
                             + inform%QPsteer_inform%QPA_inform%time%preprocess
        inform%time%steer_A_analyse    = inform%time%steer_A_analyse &
                             + inform%QPsteer_inform%QPA_inform%time%analyse
        inform%time%steer_A_factorize  = inform%time%steer_A_factorize &
                             + inform%QPsteer_inform%QPA_inform%time%factorize
        inform%time%steer_A_solve      = inform%time%steer_A_solve &
                             + inform%QPsteer_inform%QPA_inform%time%solve

        inform%time%steer_B_total      = inform%time%steer_B_total &
                             + inform%QPsteer_inform%QPB_inform%time%total
        inform%time%steer_B_preprocess = inform%time%steer_B_preprocess &
                             + inform%QPsteer_inform%QPB_inform%time%preprocess
        inform%time%steer_B_analyse    = inform%time%steer_B_analyse &
                             + inform%QPsteer_inform%QPB_inform%time%analyse
        inform%time%steer_B_factorize  = inform%time%steer_B_factorize &
                             + inform%QPsteer_inform%QPB_inform%time%factorize
        inform%time%steer_B_solve      = inform%time%steer_B_solve &
                             + inform%QPsteer_inform%QPB_inform%time%solve

        inform%time%steer_C_total      = inform%time%steer_C_total &
                             + inform%QPsteer_inform%time%total
        inform%time%steer_C_preprocess = inform%time%steer_C_preprocess &
                             + inform%QPsteer_inform%time%preprocess
        inform%time%steer_C_depend     = inform%time%steer_c_depend &
                             + inform%QPsteer_inform%time%find_dependent
        inform%time%steer_C_analyse    = inform%time%steer_C_analyse &
                             + inform%QPsteer_inform%time%analyse
        inform%time%steer_C_factorize  = inform%time%steer_C_factorize &
                             + inform%QPsteer_inform%time%factorize
        inform%time%steer_C_solve      = inform%time%steer_C_solve &
                                       + inform%QPsteer_inform%time%solve

        ! Norm of linearization--- use elastic variables coming out.

        data%u_out = data%QPsteer%X( n+1 : n+lenu )
        data%v_out = data%QPsteer%X( n+lenu+1 : npupv )

        data%norm_c_linearize_steer = sum(data%u_out) + sum(data%v_out)

        ! The steering direction, multiplier, and decrease in c.

        data%s_steer          = data%QPsteer%X( : n )
        data%inf_norm_s_steer = maxval( abs( data%s_steer ) )
        data%inf_norm_Y_steer = maxval( abs( data%QPsteer%Y(:m) ) )

        data%JxSsteer = zero
        call mop_Ax( one, nlp%J, data%s_steer, one, data%JxSsteer, &
                     out, error, transpose=.false. )

        !call L1_viol( m, data%JxSsteer + data%C_RES_l, &  DPR: does this
        !              data%C_RES_u - data%JxSsteer,    &  make a difference?
        !              data%C_type, data%norm_c_linearize_steer, error)

        data%dec_norm_c_steer = data%norm_c - data%norm_c_linearize_steer

        ! Print steering results

        if ( print_level >= GALAHAD_debug ) call print_steering_info(data, nlp)

        ! Verify that the steering step is good.  If not, try to recover.

        dummy_real = data%dec_norm_c_steer
        status     = inform%QPsteer_inform%status

        if ( dummy_real >= tenm5 .or. &  ! DPR: is this okay?
             (dummy_real >= -tenm5 .and. status == GALAHAD_ok) ) then

           qpc_successful = .true.

           ! reset control parameters to default values, if needed.

           if ( data%QP_fails /= 0 ) then ! DPR: might want to change later.
               !data%control%QPsteer_control%qpb_or_qpa      = qpb_or_qpa_steer
               !data%control%QPsteer_control%no_qpa             = no_qpa_steer
               !data%control%QPsteer_control%no_qpb             = no_qpb_steer
               !data%control%QPsteer_control%QPB_control%muzero = muzero_steer
           end if

           ! Check for infeasible stationary point.

           if ( dummy_real <= tenm6 * data%norm_c ) then
              if ( data%dual_vl <= stop_d_abs .and. &
                   data%comp_vl <= stop_c_abs ) then
                 inform%status = -32
                 if ( print_level >= GALAHAD_DETAILS ) then
                    call print_pred_steer( data )
                 end if
                 go to 813
              end if
           end if

        else

           qpc_successful = .false.

           write(out,*) 'WARNING s2qp:trying to recover from steering step.'
           write(out,*) 'QPsteer statu = ', status
           inform%status = -1
           return  ! DPR: change later.
 !            call qpc_recover( data%QPsteer, data%control%QPsteer_control,  &
 !                              inform%QPsteer_inform, nlp, data%QP_fails,  &
 !                              run_level, 'steer', error, B_type, status )

           if ( status /= 0 ) then
              inform%status = -61 ; go to 813
           end if

        end if

     end do

     ! Go here instead of recomputing the steering direction everytime.
     ! ( once the steering direction has been computed )

 804  continue

     ! If steering direction is "very" feasible OR predictor step does not
     ! achieve a fixed factor of that achieved by the steering direction,
     ! then demand better from predictor.

     ! Check 1st-condition on the steering direction.

     if ( data%norm_c_linearize_steer <= tenm4 * stop_p_abs ) then
        data%steering_good = .false.
     elseif ( data%dec_norm_c_pred >= L_factor * data%dec_norm_c_steer ) then
        data%steering_good = .true.
     else
        data%steering_good = .false.
     end if

     ! Check 2nd-condition on the steering direction.

     if ( data%steering_good ) then
        if ( data%decreaseB >= Q_factor*data%penalty*data%dec_norm_c_pred .or. &
             abs(data%decreaseB) <= tenm6 ) then
           ! relax...steering is already good.
        else
           data%steering_good = .false.
        end if
     end if

     ! Go here if predictor step is sufficiently good by itself.
     ! This skips over the verification of the 1st and 2nd steering conditions.

 805  continue

     ! Print details on predictor/steering.

     if ( print_level >= GALAHAD_DETAILS ) call print_pred_steer( data )

     ! Possibly redefine data for "new" predictor problem.

     if ( .not. data%steering_good ) then

        if ( data%penalty >= data%control%max_penalty ) then
           inform%status = -31 ;  go to 813
        end if

        data%penalty = data%control%penalty_expansion * data%penalty
        data%new_penalty = .true.

        ! Reinitialize variables and multipliers.

        !data%QPpred%X = zero
        !data%QPpred%Y = zero
        !data%QPpred%Z = zero

        !if ( m > 0 ) then
        !   do i = 1, m
        !      data%QPpred%X( n+i )   = max( zero, nlp%C_l(i) - nlp%C(i) )
        !      data%QPpred%X( n+m+i ) = max( zero, nlp%C(i) - nlp%C_u(i) )
        !   end do
        !   data%QPpred%Y( 1:m ) = nlp%Y
        !end if

        !if ( m_a > 0 ) data%QPpred%Y( m+1:m+m_a ) = nlp%Y_a

        !data%QPpred%Z( 1:n ) = nlp%Z

        ! Adjust gradient for new penalty parameter.

        data%QPpred%G( n+1 : npupv ) = data%penalty

     end if

  end do

  ! Define number of iterates needed in final predictor problem.

  dummy_int = inform%QPpred_inform%QPA_inform%major_iter
  data%iterates_pred = dummy_int + inform%QPpred_inform%QPB_inform%iter

  ! Recompute a new value of the merit function, if needed.

  if ( data%new_penalty ) then
     data%merit = nlp%f + data%penalty*data%norm_c
  end if

  ! Get value for JtY_p.

  if ( m > 0 ) then
     data%JtY_p = zero
     call mop_Ax( one, nlp%J, data%QPpred%Y( : m ), one, data%JtY_p, &
                  out, error, transpose=.true. )
  end if

  ! Get value for AtYa_p

  if ( m_a > 0 ) then
     data%AtYa_p = zero
     call mop_Ax( one, nlp%A, data%QPpred%Y( m+1 : m+m_a ), one, data%AtYa_p, &
                  out, error, transpose=.true. )
  end if

  ! If no accelerator step is being used, then must be using only predictor step
  ! for convergence and must be using a trust-region radius on predictor step
  ! computation; skip Hessian related predictor info, Cauchy step computation,
  !  and the accelerator step computation.

  if ( .not. (use_seqp .or. use_siqp) ) then
     data%iterates_acc          = 0
     data%s_s                   = zero
     data%inf_norm_s_s          = zero
     data%inf_norm_Y_s          = zero
     data%s_f                   = data%s_p
     data%inf_norm_s_f          = data%inf_norm_s_p
     data%norm_c_linearize_full = data%norm_c_linearize_pred
     data%alpha_feas            = infinity
     data%decreaseH_full        = infinity
     data%descent_constraint_status = '  NA'

     data%seqp_computed = .false.
     data%siqp_computed = .false.

     go to 811
  end if

  ! Check if the predictor multipliers are better than the current multipliers.
  ! Then evaluate Hessian now with (possibly) new multiplier information.

  dummy_logic = .false.  ! get_opt will not check for optimality (abs/rel).
  call get_opt( dummy_logic, nlp, data, nlp%G, data%JtY_p, data%AtYa_p, &
                data%QPpred%Z( : n), data%QPpred%Y( : m),               &
                data%QPpred%Y(m+1 : m+m_a),                             &
                data%C_type, data%A_type, data%X_type,                  &
                data%C_RES_l, data%C_RES_u,                             &
                data%A_RES_l, data%A_RES_u,                             &
                data%X_RES_l, data%X_RES_u,                             &
                dual_vl=data%dual_vl_p, comp_vl=data%comp_vl_p,         &
                i_dev=out )

  if ( data%dual_vl_p <= data%dual_vl .and.                                    &
       data%comp_vl_p <= data%comp_vl ) then
     nlp%Y      = data%QPpred%Y( : m )
     nlp%Y_a    = data%QPpred%Y( m+1 : m+m_a )
     nlp%Z      = data%QPpred%Z( : n )
     data%JtY   = data%JtY_p
     data%AtYa  = data%AtYa_p
     if ( print_level >= GALAHAD_debug ) then
        write( out, "(/,' - Evaluating Hessian at (x,y_p).')")
     end if
  else
     if ( print_level >= GALAHAD_debug ) then
        write( out, "(/,' - Evaluating Hessian at (x,y).')")
     end if
  end if

  call eval_HL( inform%status, nlp%X, nlp%Y, userdata, nlp%H%val )
  if ( inform%status /= GALAHAD_ok ) write(out,1002) 'eval_HL'
  inform%num_H_eval = inform%num_H_eval + 1

  if ( print_level >= GALAHAD_crazy ) then
     call print_SMT( nlp%H, 'H', error, out, inform%status )
  end if

  ! Compute data needed for subroutine cauchy_step.

  data%two_norm_s_p = NRM2( n, data%s_p, 1 )

  data%HxSp    = zero
  call mop_Ax( one, nlp%H, data%s_p, one, data%HxSp,  &
               out, error, symmetric = .TRUE. )

  data%Sp_H_Sp = DOT_PRODUCT( data%s_p, data%HxSp )

  ! Go here if previous predictor step is used - this will skip
  ! all computation associated with the predictor step.  Note: if
  ! previous predictor is used, then there must be no trust-region
  ! on the predictor step, which means there must be an ACC step to
  ! be computed.  Thus, use_prev_pred --> ACC step must be computed.

 806 continue

  ! BEGIN : Computation of Cauchy step and related data
  ! ***************************************************

  ! Get alpha_end for computation of Cauchy step.  If possible, immediately
  ! define the Cauchy step and skip the unnecessary computation.

  if ( data%Sp_B_Sp >= data%Sp_H_Sp ) then
     if ( print_level >= GALAHAD_details ) then
        write(out,*)' --Sp_B_Sp >= Sp_H_Sp --> do not call cauchy_step : sc=sp.'
     end if
     if ( data%inf_norm_s_p - data%TRpred <= tenm6 * data%TRpred ) then
        data%alpha_end    = one
        data%alpha_c      = data%alpha_end
        data%s_c          = data%s_p
        data%inf_norm_s_c = data%inf_norm_s_p
     else
        data%alpha_end    = data%TRpred / data%inf_norm_s_p
        data%alpha_c      = data%alpha_end
        data%s_c          = data%alpha_c * data%s_p
        data%inf_norm_s_c = data%alpha_c * data%inf_norm_s_p
     end if
     go to 807
  else
     if ( data%inf_norm_s_p - data%TRpred <= tenm6 * data%TRpred ) then
        data%alpha_end = one
     else
        data%alpha_end = data%TRpred / data%inf_norm_s_p
     end if
  end if

  ! Compute relevant data required for call to subroutine cauchy_step.
  ! NOTE: Subroutine cauchy_step uses J_norms only for printing.

  if ( m > 0 ) then
     call mop_row_2_norms(nlp%J, data%J_norms, symmetric=.false., &
                          out=out, error=error, print_level=0 )
  end if

  if ( print_level >= GALAHAD_details ) then
     if ( print_level >= GALAHAD_debug ) then
        print_1line = .true.
        print_debug = .true.
     else
        print_1line = .true.
        print_debug = .false.
     end if
  else
     print_1line = .false.
     print_debug = .false.
  end if

  ! Compute the Cauchy step.

  if ( print_level >= GALAHAD_debug ) write( out, 9000 ) ! header

  call cpu_time( inform%time%in )

  call cauchy_step( m, data%alpha_end, data%C_type,                         &
                    zero, data%gtSp, data%Sp_H_Sp, data%two_norm_s_p,       &
                    data%penalty, data%C_RES_l, data%C_RES_u, data%JxSp,    &
                    data%J_norms, data%lbreak, data%IBREAK, data%BREAKP,    &
                    out, print_1line, print_debug, data%alpha_c, status )

  call cpu_time( inform%time%out )

  if ( print_level >= GALAHAD_debug ) write( out, 9001 ) ! footer

  inform%time%cauchy_total = &
                 inform%time%cauchy_total + ( inform%time%out - inform%time%in )

  if ( status == -4 ) then
     write(out,*) 'Cauchy step returned status = -4 '
     inform%status = -62 ;  go to 813
  end if

  ! Define the Cauchy step and compute its size.

  data%s_c          = data%alpha_c * data%s_p
  data%inf_norm_s_c = data%alpha_c * data%inf_norm_s_p

 807 continue

  ! Compute AxSc

  if ( m_a > 0 ) then
     data%AxSc = zero
     call mop_Ax( one, nlp%A, data%s_c, one, data%AxSc, &
                  out, error, transpose=.false. )
     data%AXplusSc = nlp%Ax + data%AxSc
  end if

  ! Compute JxSc

  if ( m > 0 ) then
     data%JxSc = zero
     call mop_Ax( one, nlp%J, data%s_c, one, data%JxSc, &
                  out, error, transpose=.false. )
  end if

  if ( m > 0 ) then
     data%CplusJSc = nlp%C + data%JxSc
     call get_residuals( m, nlp%C_l, data%CplusJSc, nlp%C_u, &
                         data%C_type, data%cauchyRESl,       &
                         data%cauchyRESu )
     if ( use_siqp ) then
        call L1_viol( m, data%cauchyRESl, data%cauchyRESu,           &
                      data%C_type, data%norm_c_linearize_cauchy,     &
                      error, tenm10, data%sat, data%vl_l, data%vl_u, &
                      data%num_sat, data%num_vl_l, data%num_vl_u )
     else
        call L1_viol( m, data%cauchyRESl, data%cauchyRESu,            &
                      data%C_type, data%norm_c_linearize_cauchy, error)
     end if
  end if

  ! compute HxSc

  data%HxSc = zero
  call mop_Ax( one, nlp%H, data%s_c, one, data%HxSc,  &
               out, error, symmetric = .TRUE. )

  ! Compute the change in the "faithful" NON-SMOOTH model function.

  data%gtSc    = DOT_PRODUCT( nlp%G, data%s_c )
  data%Sc_H_Sc = DOT_PRODUCT( data%HxSc, data%s_c )

  data%dec_norm_c_cauchy = data%norm_c - data%norm_c_linearize_cauchy
  data%decreaseH_cauchy = data%penalty * data%dec_norm_c_cauchy
  data%decreaseH_cauchy = data%decreaseH_cauchy - data%gtSc
  data%decreaseH_cauchy = data%decreaseH_cauchy - half * data%Sc_H_Sc

  if ( print_level >= GALAHAD_debug ) then
     call print_cauchy_info( data, nlp )
  end if

  ! *************************************************************************
  !          END : Computation of Cauchy step and related data.             *
  ! *************************************************************************

  ! *************************************************************************
  !        BEGIN : Computation of ACC step and resulting full step.         *
  ! *************************************************************************

  ! Define accelerator trust-region radius

  data%TRacc = max( data%control%TRacc_scale * data%TRpred, data%min_TR )

  ! Decide whether to compute an SEQP or SIQP step.
  ! Note: if we are here in the code, at least one of use_seqp/use_siqp is true.

  if ( .not. use_seqp ) then
     go to 809 ! compute an siqp step
  end if

  if ( data%norm_c_linearize_pred > point1 * stop_p_abs .and. use_siqp ) then
     go to 809 ! compute an siqp step
  end if

  ! --------------------- BEGIN: SEQP step ------------------------------

  ! Decide if compute from end of predictor or Cauchy step.

  if ( use_TRpred .and. data%seqp_try_pred ) then
     if ( data%TRpred - data%inf_norm_s_p  >= five*stop_p_abs*data%TRpred ) then
        ! predictor step inactive, so try seqp step from end of predictor step.
     else
        data%seqp_try_pred = .false.
     end if
  end if

  ! Define the SEQP subproblem.

  call fill_QPseqp( nlp, data%QPseqp, status, data )
  if ( status /= GALAHAD_ok ) then
     inform%status = -58 ; go to 813
  end if

  if ( print_level >= GALAHAD_crazy ) call print_seqp_problem( data, out )

  ! Compute the SEQP step.

  call EQP_solve( data%QPseqp, data%QPseqp_data,                    &
                  data%control%QPseqp_control, inform%QPseqp_inform )

  data%seqp_computed = .true.
  data%siqp_computed = .false.

  inform%time%seqp_total     = inform%time%seqp_total            &
                             + inform%QPseqp_inform%time%total
  inform%time%seqp_factorize = inform%time%seqp_factorize        &
                             + inform%QPseqp_inform%time%factorize
  inform%time%seqp_solve     = inform%time%seqp_solve            &
                             + inform%QPseqp_inform%time%solve

  data%iterates_acc = inform%QPseqp_inform%cg_iter

  ! The seqp multipliers

  if ( m > 0 ) then
     data%Y_s = zero
     if ( data%nwJ > 0 ) then
        data%Y_s( data%wJ( : data%nwJ) ) = data%QPseqp%Y( : data%nwJ )
        data%inf_norm_Y_s = maxval( abs(data%QPseqp%Y( : data%nwJ )) )
     else
        data%inf_norm_Y_s = zero
     end if
  end if

  if ( m_a > 0 ) then
     data%Ya_s = zero
     if ( data%nwA > 0 ) then
        data%Ya_s( data%wA( : data%nwA) ) = &
             data%QPseqp%Y( data%nwJ+1 : data%nwJ+data%nwA )
     end if
  end if

  data%Z_s = zero
  if ( data%nfx > 0 ) then
     data%Z_s( data%fx( : data%nfx) ) = &
          data%QPseqp%Y( data%nwJ+data%nwA+1 : data%nwJ+data%nwA+data%nfx )
  end if

  ! If call to EQP_solve failed, set s_s = 0 and define other related quantities

  if ( inform%QPseqp_inform%status /= GALAHAD_ok ) then

     data%s_s  = zero ;  data%inf_norm_s_s = zero ;  data%Ss_H_Ss = zero
     data%JxSs = zero ;  data%AxSs         = zero ;  data%HxSs    = zero

     data%alpha_feas = data%control%infinity

     if ( data%seqp_try_pred ) then
        data%s_f = data%s_p ;  data%inf_norm_s_f = data%inf_norm_s_p
     else
        data%s_f = data%s_c ;  data%inf_norm_s_f = data%inf_norm_s_c
     end if
     go to 808
  end if

  ! SEQP solution.

  data%s_s = data%QPseqp%X ;  data%inf_norm_s_s = MAXVAL( ABS(data%s_s) )

  ! Compute JxSs, AxSs, HxSs, and Ss_H_Ss - seqp

  if ( m > 0 ) then
     data%JxSs = zero
     call mop_Ax( one, nlp%J, data%s_s, one, data%JxSs, &
                  out, error, transpose=.false. )
  end if

  if ( m_a > 0 ) then
     data%AxSs = zero
     call mop_Ax( one, nlp%A, data%s_s, one, data%AxSs, &
                  out, error, transpose=.false. )
  end if

  data%HxSs = zero
  call mop_Ax( one, nlp%H, data%s_s, one, data%HxSs,  &
               out, error, symmetric = .TRUE. )

  data%Ss_H_Ss = DOT_PRODUCT( data%HxSs, data%s_s )

  ! Get max step ensuring feasibility for inactive/free constraints/variables
  ! and then define the full step s_f.

  if ( data%seqp_try_pred ) then
     data%alpha_feas = max_feas_step( nlp, data, nlp%X,                     &
                                      data%s_p, data%AxSp, data%JxSp,       &
                                      data%s_s, data%AxSs, data%JxSs,       &
                                      data%X_type, data%A_type, data%C_type )
     data%s_f = data%s_p + data%alpha_feas * data%s_s
  else
     data%alpha_feas = max_feas_step( nlp, data, nlp%X,                     &
                                      data%s_c, data%AxSc, data%JxSc,       &
                                      data%s_s, data%AxSs, data%JxSs,       &
                                      data%X_type, data%A_type, data%C_type )
     data%s_f = data%s_c + data%alpha_feas * data%s_s
  end if
  data%inf_norm_s_f = MAXVAL( ABS(data%s_f) )

  ! Go here if SEQP solve failed.

  808 continue

  if ( print_level >= GALAHAD_DEBUG ) call print_seqp_info(data,nlp)

  ! Dummy values for variables that do not apply to computing an SEQP step.

  data%descent_constraint_status = '  NA'

  go to 811

  ! --------------------- END : SEQP step ------------------------------

  809 continue

  ! -------------------- BEGIN : SIQP step -----------------------------

  ! Compute the descent constraint.

  data%GplusHs = nlp%G + data%HxSc ;  data%descent_con = data%GplusHs

  if ( m > 0 ) then
     data%w = zero
     data%w( data%vl_l( : data%num_vl_l ) ) = - one
     data%w( data%vl_u( : data%num_vl_u ) ) =   one
     call mop_Ax( data%penalty, nlp%J, data%w, one, data%descent_con, &
                  out, error, transpose=.true. )
  end if

  ! Define the SIQP problem.

  call fill_QPsiqp( nlp, data%QPsiqp, status, data )
  if ( status /= GALAHAD_ok ) then
     inform%status = -59 ; go to 813
  end if

  ! Print out the SIQP problem.

  if ( print_level >= GALAHAD_crazy ) call print_siqp_problem( data )

  ! Solve the SIQP quadratic program.

  call QPC_solve( data%QPsiqp, data%QPsiqp%C_status,                &
                  data%QPsiqp%X_status, data%QPsiqp_data,           &
                  data%control%QPsiqp_control, inform%QPsiqp_inform )

  data%siqp_computed = .true.
  data%seqp_computed = .false.

  inform%time%siqp_A_total      = inform%time%siqp_A_total &
                              + inform%QPsiqp_inform%QPA_inform%time%total
  inform%time%siqp_A_preprocess = inform%time%siqp_A_preprocess &
                              + inform%QPsiqp_inform%QPA_inform%time%preprocess
  inform%time%siqp_A_analyse    = inform%time%siqp_A_analyse &
                              + inform%QPsiqp_inform%QPA_inform%time%analyse
  inform%time%siqp_A_factorize  = inform%time%siqp_A_factorize &
                              + inform%QPsiqp_inform%QPA_inform%time%factorize
  inform%time%siqp_A_solve      = inform%time%siqp_A_solve &
                              + inform%QPsiqp_inform%QPA_inform%time%solve

  inform%time%siqp_B_total      = inform%time%siqp_B_total &
                              + inform%QPsiqp_inform%QPB_inform%time%total
  inform%time%siqp_B_preprocess = inform%time%siqp_B_preprocess &
                              + inform%QPsiqp_inform%QPB_inform%time%preprocess
  inform%time%siqp_B_analyse    = inform%time%siqp_B_analyse &
                              + inform%QPsiqp_inform%QPB_inform%time%analyse
  inform%time%siqp_B_factorize  = inform%time%siqp_B_factorize &
                              + inform%QPsiqp_inform%QPB_inform%time%factorize
  inform%time%siqp_B_solve      = inform%time%siqp_B_solve &
                              + inform%QPsiqp_inform%QPB_inform%time%solve

  inform%time%siqp_C_total      = inform%time%siqp_C_total &
                                + inform%QPsiqp_inform%time%total
  inform%time%siqp_C_preprocess = inform%time%siqp_C_preprocess &
                                + inform%QPsiqp_inform%time%preprocess
  inform%time%siqp_C_depend     = inform%time%siqp_C_depend &
                                + inform%QPsiqp_inform%time%find_dependent
  inform%time%siqp_C_analyse    = inform%time%siqp_C_analyse &
                                + inform%QPsiqp_inform%time%analyse
  inform%time%siqp_C_factorize  = inform%time%siqp_C_factorize &
                                + inform%QPsiqp_inform%time%factorize
  inform%time%siqp_C_solve      = inform%time%siqp_C_solve &
                                + inform%QPsiqp_inform%time%solve

  ! Number of ACC iterates needed.

  data%iterates_acc = inform%QPsiqp_inform%QPA_inform%major_iter
  data%iterates_acc = data%iterates_acc + inform%QPsiqp_inform%QPB_inform%iter

  ! The multipliers

  data%Y_s  = data%QPsiqp%Y( : m )
  data%Ya_s = data%QPsiqp%Y( m+1 : m+m_a )
  data%Z_s  = data%QPsiqp%Z ( : n )

  ! Determine "activity" of the descent constraint.

  if ( data%QPsiqp%C_status( m + m_a + 1 ) == 0 ) then
     data%descent_constraint_status = '  FR'
  elseif ( data%QPsiqp%C_status( m + m_a + 1 ) > 0 ) then
     data%descent_constraint_status = 'FX-U'
     inform%num_descent_active = inform%num_descent_active + 1
  else
     data%descent_constraint_status = 'FX-L'  ! Should not happen.
  end if

  ! Based on descent-constraint activity, adjust the multipliers.

  if ( data%descent_constraint_status == 'FX-U' ) then

     if ( print_level >= GALAHAD_DEBUG ) then
        write( out,*) ' Descent constraint active - adjusting mults.'
     end if

     ! The multiplier for the descent constraint.

     dummy_real = data%QPsiqp%Y( m + m_a + 1 )

     if ( dummy_real >= min(stop_d_abs, half) ) dummy_real = - dummy_real

     ! J multipliers

     dummy_int = data%num_vl_l
     data%Y_s( data%vl_l(:dummy_int) ) = &
               data%Y_s( data%vl_l(:dummy_int) ) - data%penalty * dummy_real

     dummy_int = data%num_vl_u
     data%Y_s( data%vl_u(:dummy_int) ) = &
               data%Y_s( data%vl_u(:dummy_int) ) + data%penalty * dummy_real

     data%Y_s = data%Y_s / ( one - dummy_real )

     ! A and Z multipliers

     data%Ya_s = data%Ya_s / (one - dummy_real)
     data%Z_s  = data%Z_s / (one - dummy_real)

  end if

  if ( m > 0 ) data%inf_norm_Y_s = maxval( abs(data%Y_s) )

  ! If call to qpc_solve failed or the qp solver thinks that the descent
  ! constraint is fixed on its lower bound, then set s_s = 0 and
  ! define other quantities related to s_s.

  if ( data%descent_constraint_status == 'FX-L' ) then
     data%s_s  = zero     ;  data%inf_norm_s_s = zero
     data%s_f  = data%s_c ;  data%inf_norm_s_f = data%inf_norm_s_c

     data%JxSs    = zero ;  data%AxSs = zero ;  data%HxSs = zero
     data%Ss_H_Ss = zero
     write(out,1006)
     go to 810
  end if

  if ( inform%QPsiqp_inform%status /= GALAHAD_ok ) then
     if ( inform%QPsiqp_inform%status == GALAHAD_error_tiny_step ) then
        ! relax
     elseif (inform%QPsiqp_inform%status == GALAHAD_error_max_iterations) then
        ! relax
     else
        data%s_s  = zero     ;  data%inf_norm_s_s = zero
        data%s_f  = data%s_c ;  data%inf_norm_s_f = data%inf_norm_s_c

        data%JxSs    = zero ;  data%AxSs = zero ;  data%HxSs = zero
        data%Ss_H_Ss = zero
        write(out,1005) inform%QPsiqp_inform%status
        go to 810
     end if
  end if

  ! The ACC step and related data.

  data%s_s = data%QPsiqp%X( 1:n ) ;  data%inf_norm_s_s = MAXVAL(ABS(data%s_s))
  data%s_f = data%s_c + data%s_s  ;  data%inf_norm_s_f = MAXVAL(ABS(data%s_f))

  ! Compute JxSs, AxSs, HxSs, Ss_H_Ss - siqp

  if ( m > 0 ) then
     data%JxSs = zero
     call mop_Ax( one, nlp%J, data%s_s, one, data%JxSs, &
                  out, error, transpose=.false. )
  end if

  if ( m_a > 0 ) then
     data%AxSs = zero
     call mop_Ax( one, nlp%A, data%s_s, one, data%AxSs, &
                  out, error, transpose=.false. )
  end if

  data%HxSs = zero
  call mop_Ax( one, nlp%H, data%s_s, one, data%HxSs,  &
               out, error, symmetric = .TRUE. )

  data%Ss_H_Ss = DOT_PRODUCT( data%HxSs, data%s_s )

  ! Go here if siqp step fails or descent constraint is active on lower bound.

  810 continue

  ! Print SIQP data.

  if (print_level >= GALAHAD_DEBUG) call print_siqp_info(data, nlp, dummy_real)

  ! Dummy values for variables that do not apply to computing an SIQP step.

  data%alpha_feas = data%control%infinity

  ! ----------------- END : SIQP step computation ------------------------

  ! *************************************************************************
  !     END : Computation of accelerator step and resulting full step.      *
  ! *************************************************************************

 811 continue

  ! Get trial point X_trial and predicted decrease delmod.
  ! ******************************************************

  if ( .not. (use_seqp .or. use_siqp ) ) then

     data%delmod  = data%decreaseB
     data%X_trial = nlp%X + data%s_p
     data%trial_step = 'p '
     data%inf_norm_step1 = data%inf_norm_s_p

  elseif ( data%seqp_computed ) then

     ! Take the step with better decrease between full and Cauchy step.

     if ( data%seqp_try_pred ) then

        if ( m > 0 ) then
           data%JxSf = data%JxSp + data%alpha_feas * data%JxSs
           call L1_viol( m, data%JxSf + data%C_RES_l, data%C_RES_u - data%JxSf,&
                         data%C_type, data%norm_c_linearize_full, error)
        end if
        data%HxSf    = data%HxSp + data%alpha_feas * data%HxSs
        data%Sf_H_Sf = DOT_PRODUCT( data%HxSf, data%s_f )
        data%gtSf    = DOT_PRODUCT(nlp%G, data%s_f)

        data%delmod  = data%penalty * (data%norm_c - data%norm_c_linearize_full)
        data%delmod  = data%delmod - data%gtSf
        data%delmod  = data%delmod - half * data%Sf_H_sf

        if ( data%delmod > data%decreaseH_cauchy .or. data%NM%active ) then
           data%X_trial = nlp%X + data%s_f
           data%trial_step = 'ps'
           data%inf_norm_step1 = data%inf_norm_s_p
        else
           data%X_trial = nlp%X + data%s_c
           data%delmod  = data%decreaseH_cauchy
           data%trial_step = 'c '
           data%inf_norm_step1 = data%inf_norm_s_c
        end if

     else

        if ( m > 0 ) then
           data%JxSf = data%JxSc + data%alpha_feas * data%JxSs
           call L1_viol(m, data%JxSf + data%C_RES_l, data%C_RES_u - data%JxSf,&
                        data%C_type, data%norm_c_linearize_full, error)
        end if
        data%HxSf    = data%HxSc + data%alpha_feas * data%HxSs
        data%Sf_H_Sf = DOT_PRODUCT( data%HxSf, data%s_f )
        data%gtSf    = DOT_PRODUCT(nlp%G, data%s_f)
        data%delmod  = data%penalty*(data%norm_c - data%norm_c_linearize_full)
        data%delmod  = data%delmod - data%gtSf
        data%delmod  = data%delmod - half * data%Sf_H_sf

        if ( data%delmod > data%decreaseH_cauchy .or. data%NM%active ) then
           data%X_trial = nlp%X + data%s_f
           data%trial_step = 'cs'
        else
           data%X_trial = nlp%X + data%s_c
           data%delmod  = data%decreaseH_cauchy
           data%trial_step = 'c '
        end if
        data%inf_norm_step1 = data%inf_norm_s_c
     end if

  else ! computed an siqp step

     ! Take the better of the full step and the Cauchy step.

     if ( m > 0 ) then
        data%JxSf = data%JxSc + data%JxSs
        call L1_viol( m, data%JxSf + data%C_RES_l, data%C_RES_u - data%JxSf,&
                      data%C_type, data%norm_c_linearize_full, error)
     end if
     data%HxSf    = data%HxSc + data%HxSs
     data%Sf_H_Sf = DOT_PRODUCT( data%HxSf, data%s_f )
     data%gtSf    = DOT_PRODUCT( nlp%G, data%s_f )

     data%delmod = data%penalty * (data%norm_c - data%norm_c_linearize_full)
     data%delmod = data%delmod - data%gtSf
     data%delmod = data%delmod - half * data%Sf_H_sf

     if ( data%delmod > data%decreaseH_cauchy .or. data%NM%active ) then
        data%X_trial = nlp%X + data%s_f
        data%trial_step = 'cs'
     else
        data%X_trial = nlp%X + data%s_c
        data%delmod  = data%decreaseH_cauchy
        data%trial_step = 'c '
     end if
     data%inf_norm_step1 = data%inf_norm_s_c

  end if

  ! Evaluate functions and compute merit function.
  ! **********************************************

  if ( m > 0 ) then

     call eval_FC(inform%status, data%X_trial, userdata, data%F_new, data%C_new)
     if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_FC'

     inform%num_c_eval = inform%num_c_eval + 1
     call get_residuals( m, nlp%C_l, data%C_new, nlp%C_u, &
                         data%C_type, data%C_RES_l_new, data%C_RES_u_new )
     call L1_viol( m, data%C_RES_l_new, data%C_RES_u_new, &
                   data%C_type, data%norm_c_new, out )
  else

     call eval_FC( inform%status, data%X_trial, userdata, data%F_new )
     if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_FC'

  end if

  inform%num_f_eval = inform%num_f_eval + 1

  data%merit_new = data%F_new + data%penalty * data%norm_c_new

  ! Compute the ratio of actual/predicted change in the merit function.
  ! *******************************************************************

  if ( data%NM%active ) then
     data%ratio = data%NM%merit_ref - data%merit_new + tenm15
     data%ratio = data%ratio / ( abs(data%NM%delmod_ref) + tenm15 )
  else
     data%ratio = data%merit - data%merit_new + tenm15
     data%ratio = data%ratio / ( abs(data%delmod) + tenm15 )
  end if

  ! Determine level of success.
  ! ***************************

  if ( data%ratio >= data%control%eta_extremely_successful ) then
     data%success_str = 'extreme'
     data%success     = 3
  elseif ( data%ratio >= data%control%eta_very_successful ) then
     data%success_str = '   very'
     data%success     = 2
  elseif ( data%ratio >= data%control%eta_successful ) then
     data%success_str = 'success'
     data%success     = 1
  else
     data%success_str = 'failure'
     data%success     = 0
  end if

  ! Summary of trial step info.
  ! ***************************

  if ( print_level >= GALAHAD_debug  ) call print_trial_info( data, nlp )

  ! Determine step acceptance and set new trust-region radius.
  ! Do we turn nonmonotone on?  Do we revert?
  ! **********************************************************

  data%NM%revert     = .false.
  data%use_prev_pred = .false.

  if ( data%success >= 1 ) then

     data%step_accepted = .true.

     if ( data%success == 3 ) then
        dummy_real = ten
     elseif ( data%success == 2 ) then
        dummy_real = five
     else
        dummy_real = two
     end if

     if ( data%NM%active ) then
        data%TRpred = data%revert%TRpred
        data%inf_norm_step1 = data%revert%inf_norm_step1
        data%trial_step     = data%revert%trial_step
     end if

     if ( data%trial_step == 'ps' ) then
        data%TRpred = dummy_real * max( data%TRpred, data%inf_norm_step1 )
     else
        data%TRpred = max( data%TRpred, dummy_real * data%inf_norm_step1 )
     end if
     data%TRpred = min( data%TRpred, maxTR )
     data%min_TR = max( teneps * max( one, maxval(nlp%X) ), tenm10 )

     if ( .not. data%NM%active ) then
        if ( B_type == 3 ) then
           data%BFGS%G_ref = nlp%G
           data%BFGS%X_ref = nlp%X
           if ( m > 0 ) data%BFGS%Jval_ref = nlp%J%val
        elseif ( B_type == 2 ) then
           data%L_BFGS%G_ref = nlp%G
           data%L_BFGS%X_ref = nlp%X
           if ( m > 0 ) data%L_BFGS%Jval_ref = nlp%J%val
        end if
     end if

     data%NM%active     = .false.
     data%NM%num_fail   = 0
     data%seqp_try_pred = .true.

  else ! unsuccessful

     if ( data%NM%active ) then
        if (data%NM%num_fail < NM_steps ) then
           data%step_accepted = .true.
           data%NM%num_fail   = data%NM%num_fail + 1
           data%seqp_try_pred = .true.
        else
           data%step_accepted = .false.
           data%NM%revert     = .true.
           data%NM%active     = .false.
           data%NM%num_fail   = 0
           data%seqp_try_pred = .false.
           if ( .not. use_TRpred ) data%use_prev_pred = .true.
        end if

     else ! currently in a monotone phase.

        ! Do we turn nonmonotone on?

        if ( NM_steps > 0 .and. data%primal_vl <= half ) then

           data%step_accepted = .true.
           data%NM%active     = .true.
           data%seqp_try_pred = .true.
           data%NM%num_fail   = 1

           data%NM%merit_ref  = data%merit
           data%NM%delmod_ref = data%delmod

           data%revert%X   = nlp%X   ;  data%revert%Z  = nlp%Z
           data%revert%f   = nlp%F   ;  data%revert%G  = nlp%G


           if ( m > 0 ) then
              data%revert%Jval       = nlp%J%val
              data%revert%norm_c     = data%norm_c
              data%revert%C          = nlp%C
              data%revert%C_RES_l    = data%C_RES_l
              data%revert%C_RES_u    = data%C_RES_u
              data%revert%Y          = nlp%Y
              data%revert%inf_norm_Y = data%inf_norm_Y
              data%revert%penalty    = data%penalty
           end if
           if ( m_a > 0 ) then
              data%revert%Ax      = nlp%Ax
              data%revert%A_RES_l = data%A_RES_l
              data%revert%A_RES_u = data%A_RES_u
              data%revert%Y_a     = nlp%Y_a
           end if

           data%revert%primal_vl    = data%primal_vl
           data%revert%dual_vl      = data%dual_vl
           data%revert%comp_vl      = data%comp_vl
           data%revert%merit        = data%merit
           data%revert%X_RES_l      = data%X_RES_l
           data%revert%X_RES_u      = data%X_RES_u
           data%revert%Bval         = data%B%val
           data%revert%TRpred       = data%TRpred
           data%revert%min_TR       = data%min_TR
           data%revert%trial_step   = data%trial_step
           data%revert%inf_norm_x   = maxval( abs(nlp%X) )
           data%revert%inf_norm_s_p = data%inf_norm_s_p

           if ( data%trial_step == 'p ' .or. data%trial_step == 'ps' ) then
              data%revert%inf_norm_step1 = data%inf_norm_s_p
           else
              data%revert%inf_norm_step1 = data%inf_norm_s_c
           end if

           if ( B_type == 3 ) then
              data%BFGS%G_ref = nlp%G
              data%BFGS%X_ref = nlp%X
              if ( m > 0 ) then
                 data%BFGS%Jval_ref = nlp%J%val
              end if
           elseif ( B_type == 2 ) then
              data%L_BFGS%G_ref = nlp%G
              data%L_BFGS%X_ref = nlp%X
              if ( m > 0 ) then
                 data%L_BFGS%Jval_ref = nlp%J%val
              end if
           end if

        else

           data%step_accepted = .false.
           data%seqp_try_pred = .false.
           if ( .not. use_TRpred ) data%use_prev_pred = .true.

        end if

        if ( .not. use_TRpred ) then

           ! note 1: using prev_pred -> no TR_pred -> ACC step computed.
           ! note 2: data%dec_norm_c_pred --- not needed
           !         data%BxSp            --- not needed

           data%s_p_saved                   = data%s_p
           data%two_norm_s_p_saved          = data%two_norm_s_p
           data%inf_norm_s_p_saved          = data%inf_norm_s_p
           data%inf_norm_Y_p_saved          = data%inf_norm_Y_p
           data%gtSp_saved                  = data%gtSp
           data%norm_c_linearize_pred_saved = data%norm_c_linearize_pred
           data%Sp_B_Sp_saved               = data%Sp_B_Sp
           data%decreaseB_saved             = data%decreaseB
           data%Bval_saved                  = data%B%val
           data%HxSp_saved                  = data%HxSp
           data%Sp_H_Sp_saved               = data%Sp_H_Sp
           data%Hval_saved                  = nlp%H%val
           if ( m_a > 0 ) data%AxSp_saved   = data%AxSp
           if ( m   > 0 ) data%JxSp_saved   = data%JxSp
        end if
     end if

     if ( data%step_accepted ) then  ! In or just entered NM sequence.

        data%TRpred = max( ten, data%TRpred )

     elseif ( .not. data%NM%revert ) then  ! Rejected step and monotone.

        if ( data%TRpred <= data%min_TR ) then
           inform%status = -35 ;  go to 813
        end if

        if ( data%trial_step == 'ps' ) then
           dummy_real = data%inf_norm_s_p - data%TRpred
           if ( dummy_real / max(one,data%TRpred) <= point1 ) then
              data%TRpred = half * min( data%TRpred, data%inf_norm_step1 )
           else
              data%TRpred = point9 * data%TRpred
           end if
        else
           data%TRpred = half * min( data%TRpred, data%inf_norm_step1 )
        end if
        data%TRpred = max( data%TRpred, data%min_TR )

     end if

  end if

  ! Print summary for Predictor, Cauchy, and ACC steps.
  ! ***************************************************

  if ( print_level >= GALAHAD_action ) call print_step_summary( data )

  ! If flagged to revert, do it now.
  ! ********************************

  if ( data%NM%revert ) then

     nlp%X = data%revert%X
     nlp%Z = data%revert%Z
     nlp%f = data%revert%f
     nlp%G = data%revert%G

     if ( m > 0 ) then
        data%norm_c     = data%revert%norm_c
        nlp%C           = data%revert%C
        nlp%J%val       = data%revert%Jval
        nlp%Y           = data%revert%Y
        data%inf_norm_Y = data%revert%inf_norm_Y
        data%penalty    = data%revert%penalty
        data%C_RES_l    = data%revert%C_RES_l
        data%C_RES_u    = data%revert%C_RES_u
     end if
     if ( m_a > 0 ) then
        nlp%Ax  = data%revert%Ax
        nlp%Y_a = data%revert%Y_a
        data%A_RES_l    = data%revert%A_RES_l
        data%A_RES_u    = data%revert%A_RES_u
     end if

     data%B%val      = data%revert%Bval
     data%merit      = data%revert%merit
     data%primal_vl  = data%revert%primal_vl
     data%dual_vl    = data%revert%dual_vl
     data%comp_vl    = data%revert%comp_vl
     data%X_RES_l    = data%revert%X_RES_l
     data%X_RES_u    = data%revert%X_RES_u
     data%min_TR     = data%revert%min_TR
     data%TRpred     = data%revert%TRpred
     data%mults_used = 'R'

     ! Exit if minimum trust-region is already used.

     if ( data%TRpred <= data%min_TR ) then

        inform%status = -35 ;  go to 813

     else

        data%inf_norm_step1 = data%revert%inf_norm_step1
        data%inf_norm_s_p   = data%revert%inf_norm_s_p
        data%TRpred         = data%revert%TRpred

        if ( data%revert%trial_step == 'ps' ) then
           dummy_real = data%inf_norm_s_p - data%TRpred
           if ( dummy_real / max(one,data%TRpred) <= point1 ) then
              data%TRpred = half * min( data%TRpred, data%inf_norm_s_p )
           else
              data%TRpred = point9 * data%TRpred
           end if
        else
           data%TRpred = half * min( data%TRpred, data%inf_norm_step1 )
        end if
        data%TRpred = max( data%TRpred, data%min_TR )

     end if

     go to 812

  end if

  ! Update the trial point and the residuals if the step has been accepted.
  ! Note: if we are here, then we have already saved data need for BFGS update.
  ! ***************************************************************************

  if ( data%step_accepted ) then

     nlp%X = data%X_trial ;  nlp%F = data%f_new ;  data%merit = data%merit_new

     if ( m > 0 ) then
        nlp%C = data%C_new
        data%norm_c = data%norm_c_new
        call get_residuals( m, nlp%C_l, nlp%C, nlp%C_u,  &
                            data%C_type, data%C_RES_l, data%C_RES_u )
        call eval_GJ( inform%status, nlp%X, userdata, nlp%G, nlp%J%val )
        if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_GJ'
        inform%num_J_eval = inform%num_J_eval + 1
     else
        call eval_GJ( inform%status, nlp%X, userdata, nlp%G )
        if ( inform%status /= GALAHAD_ok ) write( out, 1002 ) 'eval_GJ'
     end if

     inform%num_g_eval = inform%num_g_eval + 1

     if ( m_a > 0 ) then
        nlp%Ax = zero
        call mop_Ax( one, nlp%A, nlp%X, one, nlp%Ax,       &
                     out, error, transpose=.false. )
        call get_residuals( m_a, nlp%A_l, nlp%Ax, nlp%A_u, &
                            data%A_type, data%A_RES_l, data%A_RES_u )
     end if

     call get_residuals( n, nlp%X_l, nlp%X, nlp%X_u,  &
                         data%X_type, data%X_RES_l, data%X_RES_u )

  end if

  ! Compute quantities needed for optimality check.
  ! ***********************************************

  if ( m > 0 ) then

     if ( data%seqp_computed .or. data%siqp_computed ) then
        data%JtY_s = zero
        call mop_Ax( one, nlp%J, data%Y_s, one, data%JtY_s, &
                     out, error, transpose=.true. )
     end if

     data%JtY_p = zero
     call mop_Ax( one, nlp%J, data%QPpred%Y(:m), one, data%JtY_p, &
                  out, error, transpose=.true. )

     data%JtY = zero
     call mop_Ax( one, nlp%J, nlp%Y, one, data%JtY, &
                  out, error, transpose=.true. )

  end if

  if ( m_a > 0 ) then

     if ( data%seqp_computed .or. data%siqp_computed ) then
        data%AtYa_s = zero
        call mop_Ax( one, nlp%A, data%Ya_s, one, data%AtYa_s, &
                     out, error, transpose=.true.   )
     end if

     data%AtYa_p = zero
     call mop_Ax( one, nlp%A, data%QPpred%Y(m+1:m+m_a), one, data%AtYa_p, &
                  out, error, transpose=.true.   )

     data%AtYa = zero
     call mop_Ax( one, nlp%A, nlp%Y_a, one, data%AtYa, &
                  out, error, transpose=.true. )

  end if

  ! Calculate optimality measures for current, predictor and ACC mults.
  ! Take the best ones and update nlp%Y, nlp%Y_a, nlp%Z.
  ! ******************************************************************

  call cpu_time( inform%time%in )
  call get_best_opt( nlp, data )
  call cpu_time( inform%time%out )

  inform%time%opt_test_total = inform%time%opt_test_total &
                             + ( inform%time%out - inform%time%in )

  if ( print_level >= GALAHAD_details ) call print_optimal_info( nlp, data )
  if ( data%converged ) go to 812

  ! If successful step, get BFGS/LBFGS info at current and reference point.
  ! ***********************************************************************

  if ( data%success >= 1 ) then

     if ( B_type == 3 ) then  ! BFGS

        data%BFGS%gradLx_new = nlp%G
        data%BFGS%gradLx     = data%BFGS%G_ref

        if ( m > 0 ) then

           call mop_Ax( -one, nlp%J, nlp%Y, one, data%BFGS%gradLx_new, &
                         out, error, transpose=.true. )

           data%revert%Jval = nlp%J%val
           nlp%J%val        = data%BFGS%Jval_ref
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%BFGS%gradLx, &
                         out, error, transpose=.true. )
           nlp%J%val = data%revert%Jval

        end if

        data%BFGS%d = data%BFGS%gradLx_new - data%BFGS%gradLx
        data%BFGS%s = nlp%X - data%BFGS%X_ref

     elseif ( B_type == 2 ) then  ! L-BFGS

        data%L_BFGS%gradLx_new = nlp%G
        data%L_BFGS%gradLx     = data%L_BFGS%G_ref

        if ( m > 0 ) then

           call mop_Ax( -one, nlp%J, nlp%Y, one, data%L_BFGS%gradLx_new, &
                        out, error, transpose=.true. )

           data%revert%Jval = nlp%J%val
           nlp%J%val        = data%L_BFGS%Jval_ref
           call mop_Ax( -one, nlp%J, nlp%Y, one, data%L_BFGS%gradLx, &
                         out, error, transpose=.true. )
           nlp%J%val = data%revert%Jval

           data%L_BFGS%y = data%L_BFGS%gradLx_new - data%L_BFGS%gradLx
           data%L_BFGS%svec = nlp%X - data%L_BFGS%X_ref

        end if

     end if
  end if

  ! Go here if just reverted.

  812 continue

  ! Print main summary line.
  !*************************

  if ( print_level >= GALAHAD_TRACE ) then
     if ( mod( data%iterate, data%control%header_every ) == 0 ) then
        write( out, 1000 )  ! header
     end if
     ! write(*,*) data%iterate
!      write(*,*) data%penalty
!      write(*,*) data%merit
!      write(*,*) data%primal_vl
!      write(*,*) data%dual_vl
!      write(*,*) data%comp_vl
!      write(*,*) data%mults_used
!      write(*,*) data%inf_norm_Y
!      write(*,*) data%TRpred
     write( out, 1001 ) data%iterate, data%penalty, data%merit,     &
                        data%primal_vl, data%dual_vl, data%comp_vl, &
                        data%mults_used, data%inf_norm_Y, data%TRpred
  end if

  ! Check for optimality of problem NLP.

  if ( data%converged ) then
     inform%status = 0 ; go to 813
  endif

  ! Check if progress should be continued.

  if ( max(data%inf_norm_s_p, data%inf_norm_s_s) <= tenm12 ) then
     if ( data%primal_vl >= tenp2 * stop_p_abs .and. &
          data%dual_vl   <= stop_d_abs         .and. &
          data%comp_vl   <= stop_c_abs ) then
        inform%status = -32 ;  go to 813
     else
        inform%status = -33 ;  go to 813
     end if
  end if

  ! Increase iterate counter and start next iterate

  data%iterate = data%iterate + 1

 end do

 ! **********************************
 ! END: Main do loop.
 ! **********************************

 if ( data%iterate > max_iterate ) inform%status = GALAHAD_error_max_iterations

 813 continue

 ! Fill output variable inform.

 inform%iterate   = data%iterate
 inform%primal_vl = data%primal_vl
 inform%dual_vl   = data%dual_vl
 inform%comp_vl   = data%comp_vl
 inform%obj       = nlp%f

 814 continue

 ! Compute total times.

 inform%time%total_total      = inform%time%feas_total    &
                              !+ inform%time%pred_A_total  &
                              + inform%time%pred_C_total  &
                              !+ inform%time%steer_A_total &
                              + inform%time%steer_C_total &
                              + inform%time%cauchy_total  &
                              + inform%time%seqp_total    &
                              !+ inform%time%siqp_A_total  &
                              + inform%time%siqp_C_total  &
                              + inform%time%opt_test_total

 inform%time%total_preprocess = inform%time%feas_preprocess    &
                              !+ inform%time%pred_A_preprocess  &
                              + inform%time%pred_C_preprocess  &
                              !+ inform%time%steer_A_preprocess &
                              + inform%time%steer_C_preprocess &
                              !+ inform%time%siqp_A_preprocess  &
                              + inform%time%siqp_C_preprocess

 inform%time%total_depend      = inform%time%feas_depend   &
                               + inform%time%pred_C_depend &
                               + inform%time%steer_C_depend&
                               + inform%time%siqp_C_depend

 inform%time%total_analyse     = inform%time%feas_analyse    &
                               !+ inform%time%pred_A_analyse  &
                               + inform%time%pred_C_analyse &
                               !+ inform%time%steer_A_analyse &
                               + inform%time%steer_C_analyse &
                               !+ inform%time%siqp_A_analyse  &
                               + inform%time%siqp_C_analyse

 inform%time%total_factorize   = inform%time%feas_factorize    &
                               !+ inform%time%pred_A_factorize  &
                               + inform%time%pred_C_factorize &
                               !+ inform%time%steer_A_factorize &
                               + inform%time%steer_C_factorize &
                               + inform%time%seqp_factorize    &
                               !+ inform%time%siqp_A_factorize  &
                               + inform%time%siqp_C_factorize

 inform%time%total_solve   = inform%time%feas_solve    &
                           !+ inform%time%pred_A_solve  &
                           + inform%time%pred_C_solve &
                           !+ inform%time%steer_A_solve &
                           + inform%time%steer_C_solve &
                           + inform%time%seqp_solve    &
                           !+ inform%time%siqp_A_solve  &
                           + inform%time%siqp_C_solve

 call cpu_time( inform%time%exit_s2qp )
 inform%time%total = inform%time%total + &
                     inform%time%exit_s2qp - inform%time%enter_s2qp

 ! Print summary of timings, if requested.

 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, 4000 ) ! Print a pretty line
    dummy_real = max(tiny,inform%time%total)
    write( out, 4001 ) inform%time%feas_preprocess, inform%time%feas_depend,  &
                       inform%time%feas_analyse, inform%time%feas_factorize,  &
                       inform%time%feas_solve, inform%time%feas_total,        &
                       hundred*inform%time%feas_total / dummy_real,           &
! end feas
                       inform%time%pred_A_preprocess,                         &
                       inform%time%pred_A_analyse,                            &
                       inform%time%pred_A_factorize,                          &
                       inform%time%pred_A_solve, inform%time%pred_A_total,    &
                       hundred*inform%time%pred_A_total / dummy_real,         &
! end pred-A
                       inform%time%pred_B_preprocess,                         &
                       inform%time%pred_B_analyse,                            &
                       inform%time%pred_B_factorize,                          &
                       inform%time%pred_B_solve, inform%time%pred_B_total,    &
                       hundred*inform%time%pred_B_total / dummy_real,         &
! end pred-B
                       inform%time%pred_C_preprocess,                         &
                       inform%time%pred_C_depend, inform%time%pred_C_analyse, &
                       inform%time%pred_C_factorize,                          &
                       inform%time%pred_C_solve, inform%time%pred_C_total,    &
                       hundred*inform%time%pred_C_total / dummy_real,         &
! end pred-C
                       inform%time%steer_A_preprocess,                        &
                       inform%time%steer_A_analyse,                           &
                       inform%time%steer_A_factorize,                         &
                       inform%time%steer_A_solve, inform%time%steer_A_total,  &
                       hundred*inform%time%steer_A_total / dummy_real,        &
! end steer-A
                       inform%time%steer_B_preprocess,                        &
                       inform%time%steer_B_analyse,                           &
                       inform%time%steer_B_factorize,                         &
                       inform%time%steer_B_solve, inform%time%steer_B_total,  &
                       hundred*inform%time%steer_B_total / dummy_real,        &
! end steer-B
                       inform%time%steer_C_preprocess,                        &
                       inform%time%steer_C_depend,                            &
                       inform%time%steer_C_analyse,                           &
                       inform%time%steer_C_factorize,                         &
                       inform%time%steer_C_solve, inform%time%steer_C_total,  &
                       hundred*inform%time%steer_C_total / dummy_real,        &
! end steer-C
                       inform%time%cauchy_total,                              &
                       hundred*inform%time%cauchy_total / dummy_real,         &
! end cauchy
                       inform%time%seqp_factorize, inform%time%seqp_solve,    &
                       inform%time%seqp_total,                                &
                       hundred*inform%time%seqp_total / dummy_real,           &
! end seqp
                       inform%time%siqp_A_preprocess,                         &
                       inform%time%siqp_A_analyse,                            &
                       inform%time%siqp_A_factorize,                          &
                       inform%time%siqp_A_solve, inform%time%siqp_A_total,    &
                       hundred*inform%time%siqp_A_total / dummy_real,         &
! end siqp-A
                       inform%time%siqp_B_preprocess,                         &
                       inform%time%siqp_B_analyse,                            &
                       inform%time%siqp_B_factorize, inform%time%siqp_B_solve,&
                       inform%time%siqp_B_total,                              &
                       hundred*inform%time%siqp_B_total / dummy_real,         &
! end siqp-B
                       inform%time%siqp_C_preprocess,                         &
                       inform%time%siqp_C_depend,                             &
                       inform%time%siqp_C_analyse,                            &
                       inform%time%siqp_C_factorize,                          &
                       inform%time%siqp_C_solve, inform%time%siqp_C_total,    &
                       hundred*inform%time%siqp_C_total / dummy_real,         &
! end siqp-C
                       inform%time%opt_test_total,                            &
                       hundred*inform%time%opt_test_total / dummy_real,       &
! end optimal
                       inform%time%total_preprocess, inform%time%total_depend,&
                       inform%time%total_analyse, inform%time%total_factorize,&
                       inform%time%total_solve, inform%time%total_total,      &
                       hundred*inform%time%total_total / dummy_real,          &
! end total
                       inform%time%total
    write( out, 4000 ) ! Print a pretty line
 end if

 ! Print problem name, exit status, and (hopefully) the solution.

 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, "( /, ' PROBLEM NAME : ', A )" ) TRIM( nlp%pname )
    write( out, "(' EXIT STATUS  : ', I3, / )" ) inform%status
    if ( data%control%print_sol ) then
       if( data%control%fulsol ) then
          write( out, "( ' X', / ( 3ES24.16 ) )" ) nlp%X
          if ( m   > 0 ) write( out, "( ' Y  ', / ( 3ES24.16 ) )" ) nlp%Y
          if ( m_a > 0 ) write( out, "( ' Y_a', / ( 3ES24.16 ) )" ) nlp%Y_a
          write( out, "( ' Z', / ( 3ES24.16 ) )" ) nlp%Z
       else
          write( out, "( ' X', / ( 4ES24.16 ) )" ) nlp%X(1:min(4,n))
          if ( m   > 0 ) then
             write( out, "( ' Y', / ( 4ES24.16 ) )" ) nlp%Y(1:min(4,m))
          end if
          if ( m_a > 0 ) then
             write( out, "( ' Y_a', / ( 4ES24.16 ) )" ) nlp%Y_a(1:min(4,m_a))
          end if
          write( out, "( ' Z', / ( 4ES24.16 ) )" ) nlp%Z(1:min(4,n))
       end if
    end if
 end if

 ! Deallocate problems

 call dealloc_QPfeas( data%QPfeas, status, alloc_status, data%control )
 if ( status /= GALAHAD_ok ) go to 992

 call dealloc_QPpred( data%QPpred, status, alloc_status, data%control )
 if ( status /= GALAHAD_ok ) go to 992

 call dealloc_QPsiqp( data%QPsiqp, status, alloc_status, data%control )
 if ( status /= GALAHAD_ok ) go to 992

 call dealloc_QPseqp( data%QPseqp, status, alloc_status,  data%control )
 if ( status /= GALAHAD_ok ) go to 992

 ! Write the solution to file "sfilename".

 inquire( FILE = sfilename, EXIST = filexx )

 if ( filexx ) then
    open( sfiledevice, FILE = sfilename, FORM = 'FORMATTED', &
          STATUS = 'OLD', IOSTAT = iores )
 else
    open( sfiledevice, FILE = sfilename, FORM = 'FORMATTED', &
          STATUS = 'NEW', IOSTAT = iores )
 end if
 if ( iores /= 0 ) then
    write( out, "( ' IOSTAT = ', I6, ' when opening file ', A9 )" ) &
           iores, sfilename
    return
 end if

 write( sfiledevice, "( '*   S2QP solution for problem name: ', A10 )" ) &
        nlp%pname
 write( sfiledevice, "( /, '*   variables ', / )" )
 do i = 1, n
    write( sfiledevice, "( '    Solution  ', A10, ES12.5 )" ) &
           nlp%VNAMES( i ), nlp%X( i )
 end do

 write(sfiledevice, "(/, '*   Lagrange multipliers - general constraints', /)")
 do i = 1, m
    write( sfiledevice, "( ' M-gen  Solution  ', A10, ES12.5 )" ) &
           nlp%CNAMES( i ), nlp%Y( i )
 end do
 write(sfiledevice, "(/, '*   Lagrange multipliers - linear constraints ', /)")
 do i = 1, m_a
    write( sfiledevice, "( ' M-lin  Solution  ', A10, ES12.5 )" ) &
           nlp%ANAMES( i ), nlp%Y_a( i )
 end do
 write( sfiledevice, "( /, ' XL Solution  ', 10X, ES12.5 )" ) nlp%f

 close( UNIT = sfiledevice, IOSTAT = iores )
 if ( iores /= 0 ) GO TO 991

 ! Return statements.
 ! ------------------

 ! normal return

 return

 ! return because of allocation error.

 990 continue

 inform%status = GALAHAD_error_allocate
 write( error, 2000) inform%status, inform%alloc_status
 write( out, "( /, ' PROBLEM NAME : ', A8 )" ) nlp%pname
 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, "(' EXIT STATUS  : ', I4, / )" ) inform%status
 end if
 return

 ! return because of error closing file

 991 continue

 inform%status = GALAHAD_error_file
 write( error, "( ' S2QP: error closing file.  iores =  ', I0 )" ) iores
 write( out, "( /, ' PROBLEM NAME : ', A8 )" ) nlp%pname
 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, "(' EXIT STATUS  : ', I4, / )" ) inform%status
 end if
 return

 ! return because of deallocation error

 992 continue

 inform%status = GALAHAD_error_deallocate
 inform%alloc_status = alloc_status
 write( error, 2001) inform%status, inform%alloc_status
 write( out, "( /, ' PROBLEM NAME : ', A8 )" ) nlp%pname
 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, "(' EXIT STATUS  : ', I4, / )" ) inform%status
 end if
 return

 ! return because some problem function evaluation was undefined.

 993 continue

 inform%status = -80
 write( out, "( /, ' PROBLEM NAME : ', A8 )" ) nlp%pname
 if ( data%control%print_level >= GALAHAD_TRACE ) then
    write( out, "(' EXIT STATUS  : ', I4, / )" ) inform%status
 end if
 return

 ! Formatting statements.
 ! ----------------------

 1000 FORMAT( /,  &
      ' Iter  Penalty    Merit     Primal     Dual    ', &
      '  Comp    Y     |Y|     TRpred' )
 1001 FORMAT(1X, I4, 2X, ES8.2, 1X, ES9.2, 3(2X, ES8.2), &
             2X, A, 2x, ES8.2, 2X, ES8.2 )
 1002 format( 1x, 'stat /=0 on return from ', A )
 1004 FORMAT(1X, I4, 2X, ES8.2, 1X, ES9.2, 3(2X, ES8.2), &
             2X, A, 2X, ES8.2, 2X, ES8.2 )
 1005 format(1x, 'WARNING:s2qp:qpc returned value of ', i5,   &
                 ' from SIQP computation.  Setting acc step', &
                 ' to zero and proceeding.' )
 1006 format(1x, 'WARNING:s2qp:qpc returned that the descent constraint', &
                ' was lower active.  Setting ACC step to zero', &
                ' and proceeding on.' )
 1007 format(1x,'WARNING s2qp:feasibility stage:no further progress possible.')
 2000 format(1x, '** ERROR : allocation error in subroutine S2QP_solve.', &
             ' error= ', I0, ' status= ', I0, '.')
 2001 format(1x, '** ERROR : de-allocation error in subroutine S2QP_solve.', &
             ' error= ', I0, ' status= ', I0, '.')
 3078 format(1x,'Current X is linearly feasible, no feasibility stage needed.')
 4000 format(/, 1x, 79('='))
 4001 format( /, &
     t37, 'TIMINGS', /,/, &
     1x, '         |  preproc    depend   analyse   ',   &
         ' factor     solve      time   percent', /, &
     1x, 79('-'), /, &
     1x, 'Feas     | ', 7(F8.2, 2x), /, &
     1x, 'Pred(A)  | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'Pred(B)  | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'Pred(C)  | ', 7(F8.2, 2x), /, &
     1x, 'Steer(A) | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'Steer(B) | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'Steer(C) | ', 7(F8.2, 2x), /, &
     1x, 'Cauchy   | ', 5('    ----', 2x), 2(F8.2, 2x), /, &
     1x, 'SEQP     | ', 3('    ----', 2x), 4(F8.2, 2x), /, &
     1x, 'SIQP(A)  | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'SIQP(B)  | ', F8.2, 2x, '    ----', 5(2x, F8.2), /, &
     1x, 'SIQP(C)  | ', 7(F8.2, 2x), /, &
     1x, 'Optimal  | ', 5('    ----', 2x), 2(F8.2, 2x), /, &
     1x, 79('-'), /, &
     1x, 'Total    | ', 7(F8.2, 2x), /, /, &
     1x, 'Actual Total Time : ', F10.2 )
 9000 format(/, &
      1x, 85('-'), /, &
      1x, 27('-'), '    BEGIN : Cauchy Problem     ', 27('-'), /, &
      1x, 85('-') )
 9001 format(/, &
      1x, 85('-'), /, &
      1x, 27('-'), '     END : Cauchy Problem      ', 27('-'), /, &
      1x, 85('-') )

 END SUBROUTINE S2QP_solve

 !************ G A L A H A D  S2QP_terminate  S U B R O U T I N E *************

 SUBROUTINE S2QP_terminate( data, control, inform )

 !-----------------------------------------------------------------------------
 !
 ! Deallocate all private storage.
 !
 !-----------------------------------------------------------------------------

 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 TYPE ( S2QP_data_type ), INTENT( INOUT ) :: data
 TYPE ( S2QP_control_type ), INTENT( IN ) :: control
 TYPE ( S2QP_inform_type ), INTENT( INOUT ) :: inform

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 CHARACTER ( LEN = 80 ) :: array_name

 !-----------------------------------------------------------------------------

 CALL QPC_terminate( data%QPfeas_data, control%QPfeas_control, &
                     inform%QPfeas_inform )
 if ( inform%QPfeas_inform%status /= GALAHAD_ok ) then
    inform%status       = GALAHAD_error_deallocate
    inform%alloc_status = inform%QPfeas_inform%alloc_status
    inform%bad_alloc    = inform%QPfeas_inform%bad_alloc
    if ( control%deallocate_error_fatal ) return
 end if

 CALL QPC_terminate( data%QPsteer_data, control%QPsteer_control, &
                         inform%QPsteer_inform )
 if ( inform%QPsteer_inform%status /= GALAHAD_ok ) then
    inform%status       = GALAHAD_error_deallocate
    inform%alloc_status = inform%QPsteer_inform%alloc_status
    inform%bad_alloc    = inform%QPsteer_inform%bad_alloc
    if ( control%deallocate_error_fatal ) return
 end if

 CALL QPC_terminate( data%QPpred_data, control%QPpred_control, &
                     inform%QPpred_inform )
 if ( inform%QPpred_inform%status /= GALAHAD_ok ) then
    inform%status       = GALAHAD_error_deallocate
    inform%alloc_status = inform%QPpred_inform%alloc_status
    inform%bad_alloc    = inform%QPpred_inform%bad_alloc
    if ( control%deallocate_error_fatal ) return
 end if

 CALL QPC_terminate( data%QPsiqp_data, control%QPsiqp_control, &
                     inform%QPsiqp_inform )
 if ( inform%QPsiqp_inform%status /= GALAHAD_ok  ) then
    inform%status       = GALAHAD_error_deallocate
    inform%alloc_status = inform%QPsiqp_inform%alloc_status
    inform%bad_alloc    = inform%QPsiqp_inform%bad_alloc
    if ( control%deallocate_error_fatal ) return
 end if

 CALL EQP_terminate( data%QPseqp_data, control%QPseqp_control, &
                     inform%QPseqp_inform )
 if ( inform%QPseqp_inform%status /= GALAHAD_ok ) then
    inform%status       = GALAHAD_error_deallocate
    inform%alloc_status = inform%QPseqp_inform%alloc_status
    inform%bad_alloc    = inform%QPseqp_inform%bad_alloc
    if ( control%deallocate_error_fatal ) return
 end if

 call SILS_finalize(data%SILS_factors,data%control%SILS_control,inform%status)
 if ( inform%status /= GALAHAD_ok ) then
    inform%status = GALAHAD_error_deallocate
    if ( control%deallocate_error_fatal ) return
 end if

 array_name = 's2qp: data%IBREAK'
 CALL SPACE_dealloc_array( data%IBREAK, inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%BREAKP'
 CALL SPACE_dealloc_array( data%BREAKP, inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%J_norms'
 CALL SPACE_dealloc_array( data%J_norms, inform%status,                      &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%H_norms'
 CALL SPACE_dealloc_array( data%H_norms,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_p'
 CALL SPACE_dealloc_array( data%s_p,inform%status,                           &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_p_saved'
 CALL SPACE_dealloc_array( data%s_p_saved, inform%status,                    &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_c'
 CALL SPACE_dealloc_array( data%s_c,inform%status,                           &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_s'
 CALL SPACE_dealloc_array( data%s_s,inform%status,                           &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_f'
 CALL SPACE_dealloc_array( data%s_f,inform%status,                           &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%s_steer'
 CALL SPACE_dealloc_array( data%s_steer, inform%status,                      &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%HxSp'
 CALL SPACE_dealloc_array( data%HxSp, inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%HxSc'
 CALL SPACE_dealloc_array( data%HxSc, inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%HxSs'
 CALL SPACE_dealloc_array( data%HxSs, inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%HxSf'
 CALL SPACE_dealloc_array( data%HxSf, inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%BxSp'
 CALL SPACE_dealloc_array( data%BxSp, inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%X_type'
 CALL SPACE_dealloc_array( data%X_type,inform%status,                        &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%A_type'
 CALL SPACE_dealloc_array( data%A_type,inform%status,                        &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%C_type'
 CALL SPACE_dealloc_array( data%C_type,inform%status,                        &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%approx_type'
 CALL SPACE_dealloc_array( data%approx_type,inform%status,                   &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%X_RES_l'
 CALL SPACE_dealloc_array( data%X_RES_l,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%X_RES_u'
 CALL SPACE_dealloc_array( data%X_RES_u,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%A_RES_l'
 CALL SPACE_dealloc_array( data%A_RES_l,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%A_RES_u'
 CALL SPACE_dealloc_array( data%A_RES_u,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%C_RES_l'
 CALL SPACE_dealloc_array( data%C_RES_l,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%C_RES_u'
 CALL SPACE_dealloc_array( data%C_RES_u,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%C_new'
 CALL SPACE_dealloc_array( data%C_new,inform%status,                         &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 array_name = 's2qp: data%gplusHs'
 CALL SPACE_dealloc_array( data%gplusHs,inform%status,                       &
                           inform%alloc_status, array_name = array_name,     &
                           bad_alloc = inform%bad_alloc, out = control%error )
 if ( control%deallocate_error_fatal .and. inform%status /= GALAHAD_ok ) return

 return

 END SUBROUTINE S2QP_terminate

 !************* G A L A H A D  build_QPfeas  S U B R O U T I N E **************

 SUBROUTINE build_QPfeas( nlp, QPfeas, status, control )

 IMPLICIT NONE

 !-----------------------------------------------------------------------------
 !
 !   Allocates components of QPfeas of type QPT_problem_type.  Components are
 !   allocated for future calls to GALAHAD subroutine QPC to solve:
 !
 !        minimize     0.5 *  sum_i^n w_i^2 * (x_i - x0_i)^2
 !        subject to         Al <= Ax <= Au
 !                           Xl <=  x <= Xu,
 !
 !   where w_i are regularization weights and x0 is a given vector.  This
 !   attempts to find the "closest" feasible point x to x0.  For more details
 !   see the GALAHAD document for QPC.
 !
 !   Coordinate storage format is used for the relevant matricies since this is
 !   the storage format of the underlying linear algebra solver.
 !
 !   Note1: this subroutine also fills vectors with data that does NOT change
 !          from one feasibility subproblem to the next.
 !-----------------------------------------------------------------------------

 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_control_type ), intent( inout ) :: control
 type( NLPT_problem_type ), intent( in ) :: nlp
 type( QPT_problem_type ), intent( inout ) :: QPfeas

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: Ane, alloc_status

 !-----------------------------------------------------------------------------

 ! Dummy values to prevent print errors.

 QPfeas%q = zero;  QPfeas%rho_b = zero;  QPfeas%rho_g = zero

 ! Dummy value to make sure that correct gradient is printed.

 QPfeas%gradient_kind = 2

 ! Signify new problem structure.

 QPfeas%new_problem_structure = .true.

 ! Set problem dimensions

 QPfeas%m   = nlp%m_a  ;   QPfeas%n   = nlp%n
 QPfeas%A%m = QPfeas%m ;   QPfeas%A%n = QPfeas%n
 QPfeas%H%m = QPfeas%n ;   QPfeas%H%n = QPfeas%n

 Ane = nlp%A%ne ;  QPfeas%A%ne = Ane ;  QPfeas%H%ne = nlp%n

 ! Allocate problem vectors.

 CALL SPACE_resize_array( QPfeas%n, QPfeas%G, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%n, QPfeas%X_l, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%n, QPfeas%X, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%n, QPfeas%X_u, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array(QPfeas%n,QPfeas%X_status,status,alloc_status)
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%n, QPfeas%Z, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%m, QPfeas%C_l, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%m, QPfeas%C, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%m, QPfeas%C_u, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array(QPfeas%m,QPfeas%C_status,status,alloc_status)
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPfeas%m, QPfeas%Y, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( Ane, QPfeas%A%val, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( Ane, QPfeas%A%col, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( Ane, QPfeas%A%row, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( nlp%n, QPfeas%H%val, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SMT_put( QPfeas%A%type, 'COORDINATE', status )
 IF ( status /= 0 ) GO TO 991
 CALL SMT_put( QPfeas%H%type, 'DIAGONAL', status )
 IF ( status /= 0 ) GO TO 991

 ! fill the values that do not change from one call to the next.

 QPfeas%f = zero

 QPfeas%X_l = nlp%X_l ;  QPfeas%X_u = nlp%X_u
 QPfeas%C_l = nlp%A_l ;  QPfeas%C_u = nlp%A_u

 QPfeas%A%row = nlp%A%row
 QPfeas%A%col = nlp%A%col
 QPfeas%A%val = nlp%A%val

 ! successful return

 return

 ! abnormal returns

 990 continue
     write( control%error, 1001) status, alloc_status
     return

 991 continue
     write( control%error, 1002)
     return

 ! format statements

 1001 FORMAT(1X, '**ERROR:s2qp_solve:build_QPfeas allocation.', &
                ' error= ', I0, ' status= ', I0, '.')
 1002 FORMAT(1x, '**ERROR:s2qp_solve:build_QPfeas SMT_put.')

 END SUBROUTINE build_QPfeas

 !************* G A L A H A D  build_QPsteer  S U B R O U T I N E **************

 SUBROUTINE build_QPsteer( nlp, QPsteer, status, control, data )
 !-----------------------------------------------------------------------------
 !                                                                             |
 ! Allocates components of QPsteer of type QPT_problem_type.  Components are   |
 ! allocated for future calls to GALAHAD subroutine QPC to solve:              |
 !                                                                             |
 !      minimize     e^T (u+v)                                                 |
 !      subject to  Cl-c <= Js + Ir(ur-vr) + Ie(ue-ve) + Il ul - Iu vu <= Cu-c |
 !                  Al-Ax  <=  As  <=  Au-Ax                                   |
 !                  max( Xl-x, -delta )  <=  s  <=  min( Xu-x, delta )         |
 !                  u,v >= 0                                                   |
 !                                                                             |
 ! where u and v are elastic variables, s is the best local step for           |
 ! improving feasibility, and delta is a given trust-region radius.            |
 ! NOTE: delta should be less than or equal to the trust-region radius         |
 !       associated with the predictor step subproblem.  This will ensure that |
 !       they are comparable.  If no predictor trust-region radius is used,    |
 !       then no restriction on delta should be required, but one probably     |
 !       should be used; currently, we use a radius of delta = 1 in this case. |
 !       See subroutine fill_QPsteer for this definition.                      |
 !                                                                             |
 ! Coordinate storage format is used for the relevant matricies since this is  |
 ! the storage format of the underlying linear algebra solver.                 |
 !                                                                             |
 !-----------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s                                               |
 !------------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_control_type ), intent( inout ) :: control
 type( S2QP_data_type ), intent( inout ) :: data
 type( NLPT_problem_type ), intent( in ) :: nlp
 type( QPT_problem_type ), intent( inout ) :: QPsteer

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s                                               |
 !------------------------------------------------------------------------------

 integer :: len, alloc_status, m, m_a, n, i
 integer :: lenu, lenv, npupv, ncrb, nce, nclb, ncub
 integer :: Une, Vne, colu, colv, Ane, Jne, ind
 real( kind=wp ) :: infinity

 !------------------------------------------------------------------------------

 ! Dummy values to prevent print errors.

 QPsteer%q = zero ;  QPsteer%rho_b = zero ;  QPsteer%rho_g = zero

 QPsteer%gradient_kind = 2  ! has a linear term.
 QPsteer%Hessian_kind  = 0  ! linear program.

 ! Signify new problem structure.

 QPsteer%new_problem_structure = .true.

 ! For convenience

 m = nlp%m ;   m_a = nlp%m_a ;   n = nlp%n

 ncrb  = data%ncrb
 nce   = data%nce
 nclb  = data%nclb
 ncub  = data%ncub
 lenu  = ncrb + nce + nclb
 lenv  = ncrb + nce + ncub
 npupv = n + lenu + lenv

 infinity = control%QPsteer_control%infinity

 ! Set problem dimensions

 QPsteer%m   = m+m_a     ;   QPsteer%n   = npupv
 QPsteer%A%m = QPsteer%m ;   QPsteer%A%n = QPsteer%n
 QPsteer%H%m = QPsteer%n ;   QPsteer%H%n = QPsteer%n

 len = nlp%A%ne + nlp%J%ne + lenu + lenv

 QPsteer%A%ne = len
 QPsteer%H%ne = 0

 ! Allocate vector components.

 CALL SPACE_resize_array( QPsteer%n, QPsteer%G, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%n, QPsteer%X_l, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%n, QPsteer%X, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%n, QPsteer%X_u, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%n, QPsteer%X_status, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%n, QPsteer%Z, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%m, QPsteer%C_l, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%m, QPsteer%C, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%m, QPsteer%C_u, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%m, QPsteer%C_status, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( QPsteer%m, QPsteer%Y, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( len, QPsteer%A%row, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( len, QPsteer%A%col, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( len, QPsteer%A%val, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( 0, QPsteer%H%val, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( 0, QPsteer%H%row, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SPACE_resize_array( 0, QPsteer%H%col, status, alloc_status )
 IF ( status /= 0 ) GO TO 990
 CALL SMT_put( QPsteer%A%type, 'COORDINATE', status )
 IF ( status /= GALAHAD_ok ) GO TO 991
 CALL SMT_put( QPsteer%H%type, 'COORDINATE', status )
 IF ( status /= GALAHAD_ok ) GO TO 991

 QPsteer%Z = zero
 QPsteer%Y = zero
 QPsteer%X_status = 0
 QPsteer%C_status = 0

 ! fill in those components that do no change from one call to the next

 QPsteer%f = zero

 QPsteer%G(:n)     = zero
 QPsteer%G(n+1:)   = one

 QPsteer%X_l(n+1:) = zero
 QPsteer%X_u(n+1:) = infinity

 Jne = nlp%J%ne
 QPsteer%A%row(:Jne) = nlp%J%row
 QPsteer%A%col(:Jne) = nlp%J%col

 Une = Jne      ;   colu = n
 Vne = Une+lenu ;   colv = n+lenu
 do i = 1, ncrb
    Une = Une+1 ;  colu = colu+1
    Vne = Vne+1 ;  colv = colv+1
    ind = data%crb(i)
    QPsteer%A%row(Une) = ind
    QPsteer%A%col(Une) = colu
    QPsteer%A%val(Une) = one
    QPsteer%A%row(Vne) = ind
    QPsteer%A%col(Vne) = colv
    QPsteer%A%val(Vne) = -one
 end do
 do i = 1, nce
    Une = Une+1 ;  colu = colu+1
    Vne = Vne+1 ;  colv = colv+1
    ind = data%ce(i)
    QPsteer%A%row(Une) = ind
    QPsteer%A%col(Une) = colu
    QPsteer%A%val(Une) = one
    QPsteer%A%row(Vne) = ind
    QPsteer%A%col(Vne) = colv
    QPsteer%A%val(Vne) = -one
 end do
 do i = 1, nclb
    Une = Une + 1 ;  colu = colu+1
    ind = data%clb(i)
    QPsteer%A%row(Une) = ind
    QPsteer%A%col(Une) = colu
    QPsteer%A%val(Une) = one
 end do
 do i = 1, ncub
    Vne = Vne+1 ;  colv = colv+1
    ind = data%cub(i)
    QPsteer%A%row(Vne) = ind
    QPsteer%A%col(Vne) = colv
    QPsteer%A%val(Vne) = -one
 end do

 Ane = nlp%A%ne
 QPsteer%A%row(Vne+1:Vne+Ane) = m + nlp%A%row
 QPsteer%A%col(Vne+1:Vne+Ane) = nlp%A%col
 QPsteer%A%val(Vne+1:Vne+Ane) = nlp%A%val

 ! normal return

 return

 ! abnormal returns

 990 continue
     write( control%error, 1001) status, alloc_status
     return

 991 continue
     write( control%error, 1002)
     return

 ! format statements

 1001 format(1X, ' **ERROR:s2qp_solve:build_QPsteer allocation.',   &
                ' error= ', I0, ' status= ', I0, '.')
 1002 format(1X, ' **ERROR:s2qp_solve:build_QPsteer SMT_put.' )

 END SUBROUTINE build_QPsteer

 !************* G A L A H A D  build_QPpred  S U B R O U T I N E **************

 SUBROUTINE build_QPpred( nlp, QPpred, status, data )
 !-----------------------------------------------------------------------------
 ! Allocates components of variable QPpred of type QPT_problem_type and sets  |
 ! sparsity patterns.  Future calls are made to GALAHAD subroutine QPC for    |
 ! solving one of the two following quadratic programs:                       |
 !                                                                            |
 !   ( B_type = 2 )                                                           |
 !                                                                            |
 !   minimize     g^T s + half s^T B0 s + penalty*( e^Tu + e^Tv )             |
 !                                      - half wa^Twa + half wb^T wb          |
 !   subject to   Cl-c <= Js + Ir(ur-vr) + Ie(ue-ve) + Il ul - Iu vu <= Cu-c  |
 !                Al-Ax <= As <= Au-Ax                                        |
 !                Ahat^T s - wa = 0                                           |
 !                Bhat^T s - wb = 0                                           |
 !                max( Xl-x, -delta )  <=  s  <=  min( Xu-x, delta )          |
 !                u,v >= 0, w_a and w_b free                                  |
 !                                                                            |
 !   where g is the gradient, B0 is a DIAGONAL positive-definite matrix that  |
 !   (loosly) approximates the Hessian of the Lagrangian,  u=(ur,ue,ul) and   !
 !   v=(vr,ve,vu) are elastic variables, s the predictor step, and delta the  !
 !   trust-region radius.  In the previous, the postfixed r, e, l, represent  |
 !   constraints that are r=range bounded, e=equalities, l=lower bounded, and !
 !   u=upper bounded.  The quanty Ir represents the columns of the identity   !
 !   matrix (of order m) associated with range constraints; analogous         !
 !   notation is used for equality, lower, and upper bounded constraint.      !
 !   The quantities wa and wb are auxiliary variables (see Section 2.2 of     |
 !   "A second derivative SQP method: local convergence" for more details),   |
 !   the matrices Ahat and Bhat are matrices associated with                  |
 !   the L-BFGS update (limited memory BFGS) update, and delta is the         |
 !   trust-region radius;                                                     |
 !                                                                            |
 !                              --- or ---                                    |
 !                                                                            |
 !   ( B_type = 0,1,3 )                                                       |
 !                                                                            |
 !   minimize     g^T s + half s^T B s + penalty*( e^Tu + e^Tv )              |
 !   subject to   Cl-c <= Js + Ir(ur-vr) + Ie(ue-ve) + Il ul - Iu vu <= Cu-c  |
 !                Al-Ax <= As <= Au-Ax                                        |
 !                max( Xl-x, -delta )  <=  s  <=  min( Xu-x, delta )          |
 !                u,v >= 0                                                    |
 !                                                                            |
 !   where g is the gradient, B is a positive-definite symmetric              |
 !   approximation to the Hessian of the Lagrangian, u=(ur,ue,ul) and         !
 !   v=(vr,ve,vu) are elastic variables, s the predictor step, and delta the  !
 !   trust-region radius.  In the previous, the postfixed r, e, l, represent  |
 !   constraints that are r=range bounded, e=equalities, l=lower bounded, and !
 !   u=upper bounded.  The quanty Ir represents the columns of the identity   !
 !   matrix (of order m) associated with range constraints; analogous         !
 !   notation is used for equality, lower, and upper bounded constraint.      !
 !                                                                            |
 ! Coordinate storage format is used for the relevant matrices since this is  |
 ! the storage format of the underlying linear algebra solver.  The values    |
 ! and their meaning for the variable B_type are:                             |
 !                                                                            |
 ! B_type   0   identity                                                      |
 !          1   weighted identity                                             |
 !          2   L-BFGS                                                        |
 !          3   Full BFGS                                                     |
 !-----------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_data_type ), intent( inout ) :: data
 type( NLPT_problem_type ), intent( in ) :: nlp
 type( QPT_problem_type ), intent( inout ) :: QPpred

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: len, Ane, tally, i, j, error, m, n, m_a, L, alloc_status
 integer :: B_type, lenu, lenv, npupv, ncrb, nce, nclb, ncub
 integer :: Une, Vne, colu, colv, colI

 !-----------------------------------------------------------------------------

 ! To prevent printing errors (these are not used).

 QPpred%q = zero ;  QPpred%rho_b = zero ;  QPpred%rho_g = zero

 ! Set parameter that describe the "kind" of problem to be solved

 QPpred%gradient_kind = 2

 ! For convenience

 m_a = nlp%m_a ;  m = nlp%m ;  n = nlp%n

 ncrb = data%ncrb ;  nce  = data%nce
 nclb = data%nclb ;  ncub = data%ncub

 lenu  = ncrb + nce + nclb ;  lenv = ncrb + nce + ncub
 npupv = n + lenu + lenv

 error  = data%control%error
 B_type = data%control%B_type
 L      = data%control%L_BFGS_number

 ! Signify new problem structure.

 QPpred%new_problem_structure = .true.

 ! Set problem dimensions.

 if ( B_type == 2 )  then ! L-BFGS
    ! Note: these are maximum dimensions encountered, since they
    !       will actually be less until the full L limited memory
    !       vectors are built up.
    QPpred%m   =  m + m_a + 2*L
    QPpred%n   =  npupv + 2*L
 elseif ( 0 <= B_type .and. B_type <= 3 )  then
    QPpred%m   = m + m_a
    QPpred%n   = npupv
 else
    write(error,*) ' **ERROR:s2qp_solve:build_QPpred B_type=?.'
    status = -1 ; return
 end if

 QPpred%A%m = QPpred%m ;    QPpred%H%m = QPpred%n
 QPpred%A%n = QPpred%n ;    QPpred%H%n = QPpred%n

 ! Allocate some vector components.

 CALL SPACE_resize_array( QPpred%n, QPpred%G, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%n, QPpred%X_l, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%n, QPpred%X, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%n, QPpred%X_u, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%n, QPpred%Z, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%n, QPpred%X_status, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%m, QPpred%C_l, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%m, QPpred%C, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%m, QPpred%C_u, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%m, QPpred%Y, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 CALL SPACE_resize_array( QPpred%m, QPpred%C_status, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 QPpred%G   = zero
 QPpred%X_l = zero
 QPpred%X   = zero ;   QPpred%X_status = 0
 QPpred%X_u = zero
 QPpred%Z   = zero
 QPpred%C_l = zero
 QPpred%C   = zero ;   QPpred%C_status = 0
 QPpred%C_u = zero
 QPpred%Y   = zero

 ! fill associated values that do not change from one call to the next.

 QPpred%f = zero

 if ( m > 0 ) then
    QPpred%X_l( n+1 : npupv ) = zero
    QPpred%X_u( n+1 : npupv ) = data%control%QPpred_control%infinity
 end if

 if ( B_type == 2 ) then ! L-BFGS
    QPpred%G(   npupv+1 : ) =  zero
    QPpred%X_l( npupv+1 : ) = -data%control%QPpred_control%infinity
    QPpred%X_u( npupv+1 : ) =  data%control%QPpred_control%infinity
    QPpred%C_l( m+m_a+1 : ) =  zero
    QPpred%C_u( m+m_a+1 : ) =  zero
 end if

 ! Initialize dummy values for status vectors.

 QPpred%C_status = 0
 QPpred%X_status = 0

 ! Allocate constraint matrix A (coordinate storage)
 ! **************************************************

 ! vectors needed for L-BFGS

 if ( B_type == 2 ) then

    allocate(data%L_BFGS%A( 1:n, 1:L ), data%L_BFGS%B( 1:n, 1:L ), STAT=status)
    if ( status /= 0 ) go to 990
    allocate( data%L_BFGS%S( 1:n, 1:L ), STAT=status )
    if ( status /= 0 ) go to 990
    allocate( data%L_BFGS%BSinner( 1:L, 1:L ), STAT=status )
    if ( status /= 0 ) go to 990
    allocate( data%L_BFGS%y( 1:n ), STAT=status )
    if ( status /= 0 ) go to 990
    allocate( data%L_BFGS%svec( 1:n ), STAT=status )
    if ( status /= 0 ) go to 990

    data%L_BFGS%A       = zero    ;   data%L_BFGS%B    = zero
    data%L_BFGS%S       = zero    ;   data%L_BFGS%y    = zero
    data%L_BFGS%BSinner = zero    ;   data%L_BFGS%svec = zero

 end if

 ! vectors for holding constraint matrix.

 len = 0
 if ( m   > 0 )     len = nlp%J%ne + lenu + lenv
 if ( m_a > 0 )     len = len + nlp%A%ne
 if ( B_type == 2 ) len = len + 2*(n+1)*L  ! Again, max required

 CALL SPACE_resize_array( len, QPpred%A%val, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPpred%A%col, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPpred%A%row, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990

 QPpred%A%val = zero
 QPpred%A%row = 0
 QPpred%A%col = 0

 CALL SMT_put( QPpred%A%type, 'COORDINATE', status )
 IF ( status /= 0 ) GO TO 991

 QPpred%A%ne = len

 ! Fill sparsity pattern

 Ane = 0

 ! Do the [ J Irb Ieq Ilb -Irb -Ieq -Iub ] part.

 if ( m > 0 ) then

    QPpred%A%row( 1 : nlp%J%ne ) = nlp%J%row
    QPpred%A%col( 1 : nlp%J%ne ) = nlp%J%col
    Ane = nlp%J%ne

    colu = n+1       ;  Une = Ane+1
    colv = colu+lenu ;  Vne = Une+lenu

    do i = 1, ncrb
       QPpred%A%row(Une) = data%crb(i)
       QPpred%A%col(Une) = colu
       QPpred%A%val(Une) = one
       QPpred%A%row(Vne) = data%crb(i)
       QPpred%A%col(Vne) = colv
       QPpred%A%val(Vne) = -one
       colu = colu+1
       colv = colv+1
       Une  = Une+1
       Vne  = Vne+1
    end do
    do i = 1, nce
       QPpred%A%row(Une) = data%ce(i)
       QPpred%A%col(Une) = colu
       QPpred%A%val(Une) = one
       QPpred%A%row(Vne) = data%ce(i)
       QPpred%A%col(Vne) = colv
       QPpred%A%val(Vne) = -one
       colu = colu+1
       colv = colv+1
       Une  = Une+1
       Vne  = Vne+1
    end do
    do i = 1, nclb
       QPpred%A%row(Une) = data%clb(i)
       QPpred%A%col(Une) = colu
       QPpred%A%val(Une) = one
       colu = colu+1
       Une  = Une+1
    end do
    do i = 1, ncub
       QPpred%A%row(Vne) = data%cub(i)
       QPpred%A%col(Vne) = colv
       QPpred%A%val(Vne) = -one
       colv = colv+1
       Vne  = Vne+1
    end do

    Ane = Ane+lenu+lenv

 end if

 ! Next the A part.

 if ( m_a > 0 ) then
    QPpred%A%row( Ane + 1 : Ane + nlp%A%ne ) = m + nlp%A%row
    QPpred%A%col( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%col
    QPpred%A%val( Ane + 1 : Ane + nlp%A%ne ) = nlp%A%val
    Ane = Ane + nlp%A%ne
 end if

 ! Finally, the additional constraints from the limited memory vectors.

 if ( B_type == 2 ) then
    colI = npupv + 1
    do i = 1, 2*L
       QPpred%A%row(Ane+1:Ane+n) = m + m_a + i
       do j = 1, n
          QPpred%A%col(Ane+j) = j
       end do
       Ane = Ane + n
       QPpred%A%row(Ane+1) = m + m_a + i
       QPpred%A%col(Ane+1) = colI ;  colI = colI + 1
       QPpred%A%val(Ane+1) = -one
       Ane = Ane + 1
    end do
 end if

 ! Allocate storage for the Hessian of the predictor QP.
 !*******************************************************

 if ( B_type == 3 ) then   ! BFGS

    ! Define lower triangular part of positive-definite B and H.
    ! H = | B 0 0 |
    !     | 0 0 0 |
    !     | 0 0 0 |


    ! "B" stuff.

    len = (n*(n+1))/2

    data%B%m  = n
    data%B%n  = n
    data%B%ne = len

    call SMT_put( data%B%type, 'COORDINATE', status )
    if ( status /= 0 ) go to 991

    call SPACE_resize_array( len, data%B%val, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( len, data%B%row, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( len, data%B%col, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( len, data%Bval_saved, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( len, data%revert%Bval, status, alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990

    data%revert%Bval = zero
    data%B%val       = zero

    tally = 0
    do i = 1, n
       do j = 1, i
          tally = tally + 1
          data%B%row( tally ) = i
          data%B%col( tally ) = j
       end do
    end do

    ! Full "H" stuff.

    QPpred%H%m  = npupv
    QPpred%H%n  = npupv
    QPpred%H%ne = len

    call SMT_put( QPpred%H%type, 'COORDINATE', status )
    if ( status /= 0 ) go to 991

    CALL SPACE_resize_array( len, QPpred%H%val, status, alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( len, QPpred%H%row, status, alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990
    CALL SPACE_resize_array( len, QPpred%H%col, status, alloc_status )
    IF ( status /= GALAHAD_ok ) GO TO 990

    QPpred%H%val = zero
    QPpred%H%row = data%B%row
    QPpred%H%col = data%B%col

    ! vectors associated with the BFGS update.

    call SPACE_resize_array( n, data%BFGS%d, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%s, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%Bs, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%gradLx, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%gradLx_new, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%g_ref, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    call SPACE_resize_array( n, data%BFGS%X_ref, status, alloc_status )
    if ( status /= GALAHAD_ok ) GO TO 990
    if ( m > 0 ) then
       call SPACE_resize_array( nlp%J%ne, data%BFGS%Jval_ref, &
                                status, alloc_status )
       if ( status /= GALAHAD_ok ) GO TO 990
       data%BFGS%Jval_ref = zero
    end if

    data%BFGS%d          = zero
    data%BFGS%s          = zero
    data%BFGS%Bs         = zero
    data%BFGS%gradLx_new = zero
    data%BFGS%gradLx     = zero
    data%BFGS%g_ref      = zero
    data%BFGS%X_ref      = zero

 else  ! Identity, weighted-diagonal, or L-BFGS.

    ! Allocate vectors for diagonal Hessian (remember elastic variables),
    ! vectors associated with L-BFGS (if applicable), matrix B, and
    ! possibly for reverting if nonmonotone is being used.

    call SMT_put( QPpred%H%type, 'DIAGONAL', status )
    if ( status /= 0 ) go to 991
    call SMT_put( data%B%type, 'DIAGONAL', status ) ! B_0 part if L-BFGS
    if ( status /= 0 ) go to 991

    len = npupv

    if ( B_type == 2 ) then  ! L-BFGS

       len = len + 2*L  ! maximum length needed.

       call SPACE_resize_array( n, data%L_BFGS%gradLx, status, alloc_status )
       if ( status /= GALAHAD_ok ) GO TO 990
       call SPACE_resize_array(n, data%L_BFGS%gradLx_new, status, alloc_status)
       if ( status /= GALAHAD_ok ) GO TO 990
       call SPACE_resize_array( n, data%L_BFGS%g_ref, status, alloc_status )
       if ( status /= GALAHAD_ok ) GO TO 990
       call SPACE_resize_array( n, data%L_BFGS%X_ref, status, alloc_status )
       if ( status /= GALAHAD_ok ) GO TO 990
       if ( m > 0 ) then
          call SPACE_resize_array( nlp%J%ne, data%L_BFGS%Jval_ref, &
                                   status, alloc_status )
          if ( status /= GALAHAD_ok ) GO TO 990
          data%L_BFGS%Jval_ref = zero
       end if

       data%L_BFGS%gradLx     = zero
       data%L_BFGS%gradLx_new = zero
       data%L_BFGS%g_ref      = zero
       data%L_BFGS%X_ref      = zero

    end if

    call SPACE_resize_array( len, QPpred%H%val, status, alloc_status )
    if ( status /= GALAHAD_ok ) go to 990
    call SPACE_resize_array( n, data%B%val, status, alloc_status )
    if ( status /= GALAHAD_ok ) go to 990
    call SPACE_resize_array( n, data%Bval_saved, status, alloc_status )
    if ( status /= GALAHAD_ok ) go to 990

    if ( data%control%NM_steps > 0 ) then  ! non-monotone
       call SPACE_resize_array( n, data%revert%Bval, status, alloc_status )
       if ( status /= GALAHAD_ok ) go to 990
       data%revert%Bval = zero
    end if

    QPpred%H%val = zero
    data%B%val   = zero

    ! Set dimension - maximum in L-BFGS case.

    QPpred%H%m  = len ;   data%B%m  = n
    QPpred%H%n  = len ;   data%B%n  = n
    QPpred%H%ne = len ;   data%B%ne = n  ! this should not be needed.

    ! If identity used, define the identity in the x-variables.

    if ( B_type == 0 ) then
       QPpred%H%val( 1:n ) = one ;  data%B%val( 1:n ) = one
    end if

    ! Define zero values for diagonal entries for elastic variables.

    do i = n+1, npupv
       QPpred%H%val( i ) = zero
    end do

    ! Note: entries associated with I and -I along the block diagonal
    !       of "H" in the L-BFGS case must be filled each time in the
    !       subroutine fill_QPpred.

 end if

 ! normal return

 return

 ! abnormal returns

 990 continue
     write( error, 1001 ) status, alloc_status
     return

 991 continue
     write( error, 1002 )
     return

 ! format statements

 1001 FORMAT(1X, '**ERROR:s2qp_solve:build_QPpred allocation.', &
                ' status= ', I0, ' alloc_status= ', I0, '.'  )
 1002 FORMAT(1X, '**ERROR:s2qp_solve:build_QPpred SMT_put.')

 END SUBROUTINE build_QPpred

 !************* G A L A H A D  build_QPsiqp  S U B R O U T I N E **************

 SUBROUTINE build_QPsiqp( nlp, QPsiqp, status, control )

 IMPLICIT NONE

 !-----------------------------------------------------------------------------
 ! Allocates components of variable QPsiqp of type QPT_problem_type and sets  |
 ! sparsity patterns.  Future calls are made to GALAHAD subroutine QPC for    |
 ! solving the quadratic programs                                             |
 !                                                                            |
 !    minimize   (g + H s_c)^T s + half s^T H s + penalty * e^T ( u + v )     |
 !                              subject to                                    |
 ! [Cl - (c + J s_c)]_i  <=  [Js]_i  <=  [Cu - (c + J s_c)]_i          i in S |
 ! [Cl - (c + J s_c)]_i  <=  [Js + u - v]_i  <=  [Cu - (c + J s_c)]_i  i in V |
 !               Al - A(x+s_c)  <=  As  <=  Au - A(x+s_c)                     |
 !                   (g + H s_c + J^T w)^T s <= 0                             |
 !   max[ Xl-(x+s_c), -delta ]  <=  s  <=  min[ Xu-(x+s_c), delta ]           |
 !                                                                            |
 ! where s_c is the Cauchy step, u and v are elastic variables associated with|
 ! the violated constraints at the Cauchy point as defined by the index sets  |
 !                                                                            |
 !     S = { i : [Cl]_i <= [c + J s_c] _i <= [Cu]_i }     (satisfied)         |
 !     V = { i : i in [0:m] and i not in S }               (violated)         |
 !                                                                            |
 ! where m is the number of constraints, i.e., the length of c.  Delta is the |
 ! trust region constraint for the ACC subproblem, and w is defined by        |
 !                                                                            |
 !           | -1  if [c+Js_c]_i < Cl                                         |
 !     w_i = |  1  if [c+Js_c]_i > Cu                                         |
 !           |  0  otherwise                                                  |
 !                                                                            |
 ! Coordinate storage format is used for the relevant matrices since this is  |
 ! the storage format of the underlying linear algebra solver.                |
 !-----------------------------------------------------------------------------

 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_control_type ), intent( inout ) :: control
 type( NLPT_problem_type ), intent( in ) :: nlp
 type( QPT_problem_type ), intent( out ) :: QPsiqp

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: len, Ane, Jne, Hne, i, alloc_status, n, m, m_a, tally

 !-----------------------------------------------------------------------------

 ! To prevent printing errors (these are not used).

 QPsiqp%q = zero ;  QPsiqp%rho_b = zero ;  QPsiqp%rho_g = zero

 ! For convenience

 n   = nlp%n
 m   = nlp%m
 m_a = nlp%m_a

 ! Set dimensions (maximum possible)

 QPsiqp%m   = m + m_a + 1 ;  QPsiqp%n   = n + 2*m
 QPsiqp%A%m = m + m_a + 1 ;  QPsiqp%A%n = n + 2*m

 len = n
 if ( m > 0 ) then
    Jne = nlp%J%ne
    len = len + Jne + 2*m
 else
    Jne = 0 ! To prevent warnings during compilation.
 end if
 if ( m_a > 0 ) then
    Ane = nlp%A%ne
    len = len + Ane
 else
    Ane = 0 ! To prevent warnings during compilation.
 end if
 QPsiqp%A%ne = len

 QPsiqp%H%m  = n + 2*m
 QPsiqp%H%n  = n + 2*m
 Hne         = nlp%H%ne
 QPsiqp%H%ne = Hne

 ! Indicate new problem structure.

 QPsiqp%new_problem_structure = .true.

 ! Allocate problem vectors.

 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%G, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_l, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_u, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%X_status, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%n, QPsiqp%Z, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_l, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_u, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%m, QPsiqp%C_status, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPsiqp%m, QPsiqp%Y, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPsiqp%A%row, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPsiqp%A%col, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPsiqp%A%val, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPsiqp%H%row, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPsiqp%H%col, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPsiqp%H%val, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SMT_put( QPsiqp%A%type, 'COORDINATE', status  )
 IF ( status /= 0 ) GO TO 991
 CALL SMT_put( QPsiqp%H%type, 'COORDINATE', status  )
 IF ( status /= 0 ) GO TO 991

 QPsiqp%A%row = 0
 QPsiqp%A%col = 0
 QPsiqp%A%val = zero

 ! Decide whether to use warm starts. ! DPR: What to do with this?

 if ( QPsiqp%m > 0 ) QPsiqp%C_status  = 0
 QPsiqp%X_status = 0

 control%QPsiqp_control%QPA_control%cold_start = 0

 ! Constraint matrix A (coordinate storage).
 ! Note: set only part that does not change, which
 !       is the J, A, and descent constraint parts.
 !*************************************************

 ! Define sparsity pattern.

 tally = 0

 ! First the J in the [ J  I  -I ] part.

 if ( m > 0 ) then
    QPsiqp%A%row( tally + 1 : tally + Jne ) = nlp%J%row
    QPsiqp%A%col( tally + 1 : tally + Jne ) = nlp%J%col
    tally = Jne
 end if

 ! Now A.

 if ( m_a > 0 ) then
    QPsiqp%A%row( tally + 1 : tally + Ane ) = nlp%A%row + m
    QPsiqp%A%col( tally + 1 : tally + Ane ) = nlp%A%col
    QPsiqp%A%val( tally + 1 : tally + Ane ) = nlp%A%val
    tally = tally + Ane
 end if

 ! The descent constraint.

 do i = 1, n
    QPsiqp%A%row( tally + i ) = m + m_a + 1
    QPsiqp%A%col( tally + i ) = i
 end do

 ! Hessian matrix H (coordinate storage).
 ! **************************************

 QPsiqp%H%row = nlp%H%row ;  QPsiqp%H%col = nlp%H%col

 ! Fill in other problem vectors that do not change from
 ! one subproblem formulation to the next.

 QPsiqp%C_l( m+m_a+1 : ) = -control%QPsiqp_control%infinity
 QPsiqp%C_u( m+m_a+1 : ) = zero

 QPsiqp%X_l = zero
 QPsiqp%X_u = control%QPsiqp_control%infinity

 ! normal return

 return

 ! abnormal returns

 990 CONTINUE
    WRITE( control%error, 1001 ) status, alloc_status
    RETURN

 991 CONTINUE
    WRITE( control%error, 1002 )
    RETURN

 ! format statements

 1001 FORMAT(1X, '** ERROR s2qp:build_QPsiqp:allocation error.',   &
                ' statis= ', I0, ' alloc_status= ', I0, '.')
 1002 FORMAT(1X, '** ERROR s2qp:build_QPsiqp:error in SMT_put.')

 END SUBROUTINE build_QPsiqp

 !************* G A L A H A D  build_QPseqp  S U B R O U T I N E **************

 SUBROUTINE build_QPseqp( nlp, QPseqp, status, data )
 !-----------------------------------------------------------------------------
 ! Allocates components of variable QPseqp of type QPT_problem_type and sets  |
 ! sparsity patterns.  Future calls are made to GALAHAD subroutine EQP for    |
 ! solving the quadratic program                                              |
 !                                                                            |
 !    minimize      (g + H s_ac)^T s + half s^T H s                           |
 !    subject to    [Js]_i  =  0  i in WC(s_ac)                               |
 !                  [As]_i  =  0  i in WA(s_ac)                               |
 !                  [s]_i   =  0  i in FX(s_ac)                               |
 !                  || s || <= delta                                          |
 !                                                                            |
 ! where s_ac is the APPROXIMATE Cauchy step, delta the ACC trust-regin       |
 ! radius, and working set indexing sets WC, WA, and FX are defined as        |
 !                                                                            |
 !     WC = { i : [c + J s_ac]_i = {Cl,Cu} }      (active general constraints)|
 !     WA = { i : [A(x+s_ac)_i = {Al,Au} }        (active linear constraints) |
 !     FX = { i : [x+s_ac]_i = {Xl,Xu} }          (fixed variables)           |
 !                                                                            |
 ! where m is the number of constraints, i.e., the length of c.  Delta is the |
 ! trust region constraint for the ACC subproblem.                            |
 !                                                                            |
 ! Coordinate storage format is used for the relevant matrices since this is  |
 ! the storage format of the underlying linear algebra solver.                |
 !                                                                            |
 ! Note: special care must be used after solving the above problem since the  |
 !       resulting step may be infeasible for those constraints/variables not |
 !       in the sets WC, WA, or FX.                                           |
 !-----------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_data_type ), intent( inout ) :: data
 type( NLPT_problem_type ), intent( in ) :: nlp
 type( QPT_problem_type ), intent( out ) :: QPseqp

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: len, alloc_status, n, m, m_a
 integer :: Ane, Jne, Hne

 !------------------------------------------------------------------------------

 ! For convenience.

 n = nlp%n ;  m = nlp%m ;  m_a = nlp%m_a

 ! Declar new problem structer....I think this is irrelavant for this problem.

 QPseqp%new_problem_structure = .true.

 ! To avoid print errors.

 QPseqp%q             = zero
 QPseqp%rho_b         = zero
 QPseqp%rho_g         = zero
 QPseqp%gradient_kind = 2

 ! Set MAXIMUM dimensions.
 ! Note: Must set correct values for a given instance of
 !       each subproblem in subroutine fill_QPseqp each time.

 QPseqp%m    = m + m_a + n
 QPseqp%n    = n
 QPseqp%A%ne = nlp%J%ne + nlp%A%ne + n
 QPseqp%A%m  = m + m_a + n
 QPseqp%A%n  = n
 QPseqp%H%m  = n
 QPseqp%H%n  = n
 QPseqp%H%ne = nlp%H%ne

 Hne = nlp%H%ne
 len = n
 if ( m > 0 ) then
    Jne = nlp%J%ne
    len = len + Jne
 end if
 if ( m_a > 0 ) then
    Ane = nlp%A%ne
    len = len + Ane
 end if

 ! Allocate components independent of storage type.

 CALL SPACE_resize_array( QPseqp%n, QPseqp%G, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPseqp%n, QPseqp%X, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPseqp%n, QPseqp%X_status, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 ! This is the constant term in the constraints, NOT the constraints!
 CALL SPACE_resize_array( QPseqp%m, QPseqp%C, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPseqp%n, data%fr, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPseqp%n, data%fx, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( nlp%m_a, data%wA, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( nlp%m_a, data%wA_comp, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( nlp%m, data%wJ, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( nlp%m, data%wJ_comp, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( QPseqp%m, QPseqp%Y, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPseqp%A%row, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPseqp%A%col, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( len, QPseqp%A%val, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPseqp%H%row, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPseqp%H%col, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SPACE_resize_array( Hne, QPseqp%H%val, status, alloc_status )
 IF ( status /= GALAHAD_ok ) GO TO 990
 CALL SMT_put( QPseqp%A%type, 'COORDINATE', status  )
 IF ( status /= 0 ) GO TO 991
 CALL SMT_put( QPseqp%H%type, 'COORDINATE', status )
 IF ( status /= 0 ) GO TO 991

 ! Set constant in objective to zero.

 QPseqp%f = zero

 ! Set constant in constraint to zero.

 QPseqp%C = zero


 ! Dummy initialize constraint matrix A.
 ! Note1: we can not do more since sparsity may change from
 !        one instance of the seqp problem till the next.
 ! Note2: IMPORTANT : values AND sparsity must be done in
 !        subroutine fill_QPseqp each time.

 QPseqp%A%val = zero
 QPseqp%A%row = 0
 QPseqp%A%col = 0

 ! Hessian matrix H (coordinate storage).

 QPseqp%H%row = nlp%H%row
 QPseqp%H%col = nlp%H%col

 ! normal return

 return

 ! abnormal returns

 990 continue
     write( data%control%error, 1001 ) status, alloc_status
     return

 991 continue
     write( data%control%error, 1002 )
     return

 ! format statements

 1001 format(1X, ' ** ERROR s2qp:build_QPseqp:allocation error.',   &
                ' status = ', I0, ' alloc_status= ', I0, '.')
 1002 format(1X, ' ** ERROR s2qp:build_QPseqp:error in SMT_put.')

 END SUBROUTINE build_QPseqp

 !************** G A L A H A D  fill_QPfeas  S U B R O U T I N E ***************

 SUBROUTINE fill_QPfeas( nlp, QPfeas )

 IMPLICIT NONE

 !------------------------------------------------------------------------------
 ! See subroutine build_QPfeas for more information.
 !
 ! Note1: all data that does not change from one feasibility subproblem to the
 !        next is set in the subroutine build_QPfeas.
 ! Note2: currently, we set the weights w_i in the objective all equal to 1.
 !------------------------------------------------------------------------------

 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( QPT_problem_type ),  intent( inout ) :: QPfeas

 !------------------------------------------------------------------------------

 QPfeas%H%val = one         ! more generally, H = W^2
 QPfeas%G     = -nlp%X      ! more generally, G = -W^2 X0
 QPfeas%X     = nlp%X
 QPfeas%Z     = nlp%Z
 QPfeas%C     = nlp%Ax
 QPfeas%Y     = nlp%Y_a

 return

 END SUBROUTINE fill_QPfeas

 !************* G A L A H A D  fill_QPsteer  S U B R O U T I N E ***************

 SUBROUTINE fill_QPsteer( nlp, QPsteer, data )
 !------------------------------------------------------------------------------
 ! See subroutine build_QPsteer for more information.
 !------------------------------------------------------------------------------
 IMPLICIT NONE
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 !type( S2QP_control_type ), intent( inout ) :: control
 type( S2QP_data_type ), intent( in ) :: data
 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( QPT_problem_type ), intent( inout ) :: QPsteer

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: i, m, m_a, n, ind
 integer :: ncrb, nce, nclb, ncub, utally, vtally
 integer :: lenu, lenv
 real( kind = wp ) :: infinity
 real( kind = wp ),allocatable, dimension(:) :: u_in, v_in

 !------------------------------------------------------------------------------

 ! For convenience.

 m   = nlp%m
 m_a = nlp%m_a
 n   = nlp%n

 ncrb     = data%ncrb
 nce      = data%nce
 nclb     = data%nclb
 ncub     = data%ncub
 lenu     = ncrb+nce+nclb
 lenv     = ncrb+nce+ncub
 infinity = data%control%QPsteer_control%infinity

 ! Compute QPsteer%C to prevent printing errors

 allocate( u_in(lenu), v_in(lenv) )

 utally = 0
 vtally = 0

 QPsteer%C = zero

 do i = 1, ncrb
    utally = utally+1
    vtally = vtally+1
    ind = data%crb(i)
    u_in(utally) = max( nlp%C_l(ind) - nlp%C(ind), zero )
    v_in(vtally) = max( nlp%C(ind) - nlp%C_u(ind), zero )
    QPsteer%C(ind) = u_in(utally) - v_in(vtally)
 end do
 do i = 1, nce
    utally = utally+1
    vtally = vtally+1
    ind = data%ce(i)
    u_in(utally) = max( nlp%C_l(ind) - nlp%C(ind), zero )
    v_in(vtally) = max( nlp%C(ind) - nlp%C_u(ind), zero )
    QPsteer%C(ind) = u_in(utally) - v_in(vtally)
 end do
 do i = 1, nclb
    utally = utally+1
    ind = data%clb(i)
    u_in(utally) = max( nlp%C_l(ind) - nlp%C(ind), zero )
    QPsteer%C(ind) = u_in(utally)
 end do
 do i = 1, ncub
    vtally = vtally+1
    ind = data%cub(i)
    v_in(vtally) = max( nlp%C(ind) - nlp%C_u(ind), zero )
    QPsteer%C(ind) = -v_in(vtally)
 end do

 ! Bounds on constraints and variables.
 ! Note: elastic bounds set in build_QPsteer.

 if ( m > 0 ) then
    do i = 1, m
       QPsteer%C_l( i ) = max( nlp%C_l(i) - nlp%C(i), -infinity )
       QPsteer%C_u( i ) = min( nlp%C_u(i) - nlp%C(i),  infinity )
    end do
 end if
 if ( m_a > 0 ) then
    do i = 1, m_a
       QPsteer%C_l( m+i ) = max( nlp%A_l(i) - nlp%Ax(i), -infinity )
       QPsteer%C_u( m+i ) = min( nlp%A_u(i) - nlp%Ax(i),  infinity )
    end do
 end if
 do i = 1, n
    if ( data%control%use_TRpred ) then
       QPsteer%X_l( i ) = max( nlp%X_l(i)-nlp%X(i), -data%TRpred )
       QPsteer%X_u( i ) = min( nlp%X_u(i)-nlp%X(i),  data%TRpred )
    else
       QPsteer%X_l( i ) = max( nlp%X_l(i) - nlp%X(i), -four )
       QPsteer%X_u( i ) = min( nlp%X_u(i) - nlp%X(i),  four )
    end if
 end do

 ! Initial point.

 QPsteer%X = zero
 QPsteer%X( n+1 : n+lenu )           = u_in
 QPsteer%X( n+lenu+1 : n+lenu+lenv ) = v_in

 deallocate( u_in, v_in )

 ! **************************************************
 ! Load QPsteer%A with the portions from | J I -I |
 !                                       | A 0  0 |
 ! Note: everything but J is loaded in build_QPsteer.
 ! **************************************************

 QPsteer%A%val( 1:nlp%J%ne ) = nlp%J%val

 return

 END SUBROUTINE fill_QPsteer

 !************** G A L A H A D  fill_QPpred  S U B R O U T I N E ***************

 SUBROUTINE fill_QPpred( nlp, QPpred, status, data )
 !------------------------------------------------------------------------------
 !  See subroutine build_QPpred for more information.
 !
 !     status    0   okay
 !             -25   incorrect storage type used.
 !             -99   error returned from subroutine get_L_BFGS.
 !
 !  Note1: The values data%BFGS%gradLx and data%BFGS%gradLx_new must be defined
 !         from the previous iteration when successful, since currently a BFGS
 !         update is only performed following successful iterations.
 !  Note2: The call to subroutine get_L_BFGS should only be performed if
 !         data%L_BFGS%update_number >= 1.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_data_type ), intent( inout )    :: data
 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( QPT_problem_type ), intent( inout )  :: QPpred

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: tally, i, j, l, Ane, m, n, m_a, B_type, error, out
 integer :: A_ind, B_ind, iterate
 integer :: success, update_number, print_level, number_used
 integer :: ncrb, nce, nclb, ncub, lenu, lenv, npupv
 integer :: utally, vtally, ind
 real( kind = wp ) :: Bij, di, Bsi, stBs, damp_factor, infinity
 real( kind = wp ) :: B_lammin, B_lammax, dummy_real, Hii

 !------------------------------------------------------------------------------

 ! Bounds on the allowed entries in the matrix B.

 B_lammin = tenm5 ;  B_lammax = tenp5

 ! For convenience

 n = nlp%n ;  m = nlp%m ;  m_a = nlp%m_a

 ncrb = data%ncrb ;  nce  = data%nce
 nclb = data%nclb ;  ncub = data%ncub

 lenu  = ncrb + nce + nclb
 lenv  = ncrb + nce + ncub
 npupv = n + lenu + lenv

 out     = data%control%out
 error   = data%control%error
 L       = data%control%L_BFGS_number
 B_type  = data%control%B_type
 iterate = data%iterate
 success = data%success
 infinity = data%control%QPpred_control%infinity

 print_level = data%control%print_level

 ! Get quantities needed for L-BFGS update, and set correct dimensions.

 if ( B_type == 2 ) then

    update_number = data%L_BFGS%update_number

    if ( update_number == 0 ) then
       data%B%val(1:n)   = one  ! this is B0 in the LBFGS update.
       data%L_BFGS%theta = one  ! damping scalar.
       number_used = 0          ! number used in this update.
       status   = 0
       if ( print_level >= GALAHAD_debug ) then
          write(out,1024) ! header
          write(out,"(/,' L-BFGS - formed identity ( iterate - 1)')")
          write(out,1025) ! write footer
       end if
    else
       if ( success >= 1 ) then
          call get_L_BFGS( data%L_bfgs%S, data%L_bfgs%A, data%L_BFGS%B,        &
                           data%L_bfgs%BSinner, data%B, data%L_BFGS%svec,      &
                           data%L_bfgs%gradLx_new, data%L_bfgs%gradLx,         &
                           data%L_BFGS%y, update_number, L, n,                 &
                           data%L_BFGS%theta, data%L_BFGS%damp_factor, status, &
                           print_level, out, error )
          if ( status == 0 ) then
             number_used = min( update_number, L )
          elseif ( status == -97 ) then ! restart L-BFGS
             data%B%val(1:n)   = one
             data%L_BFGS%theta = one
             number_used   = 0
             update_number = 0
             data%L_BFGS%update_number = 0
          elseif ( status == -99 .or. status == -80 .or. status == -81 ) then
             status = -99  ! error
             return
          else ! skipped update : -96 or -98
             number_used = min( update_number-1, L )
          end if
       else  ! success = 0
          status = 0
          number_used = min( update_number-1, L )
          if ( update_number == 1 ) then
             if ( data%seqp_computed .or. data%siqp_computed ) then
                do i = 1, n
                   call mop_getval( nlp%H, i, i, Hii, .true., out, error, 0 )
                   data%B%val(i) = min( max(B_lammin, abs(Hii)), B_lammax )
                end do
             else
                data%B%val = min( max(B_lammin, NRM2(n,nlp%G,1)), B_lammax )
             end if
             data%L_BFGS%theta = one
             if ( print_level >= GALAHAD_debug ) then
                write(out,1024) ! header
                write(out,"(/,' L-BFGS - no update yet : scaling the identity')")
                write(out,1025) ! write footer
             end if
          else
             data%L_BFGS%theta = one
             if ( print_level >= GALAHAD_debug ) then
                write(out,1024) ! header
                write(out,"(/, ' No update : unsuccessful step')" )
                write(out,1025) ! footer
             end if
          end if
       end if
    end if

    if ( update_number == 0 .or. &
         ( update_number > 0 .and. success >= 1 .and. status == 0) ) then
       data%L_BFGS%update_number = update_number + 1
    end if

    data%L_BFGS%number_used = number_used

    ! Set correct dimensions.

    QPpred%m = m + m_a + 2*number_used
    QPpred%n = npupv + 2*number_used
    QPpred%A%m = QPpred%m
    QPpred%A%n = QPpred%n
    QPpred%H%m = QPpred%n
    QPpred%H%n = QPpred%n

 end if

 ! Fill vectors.

 QPpred%G( 1 : n )         = nlp%G
 QPpred%G( n + 1 : npupv ) = data%penalty

 if ( m > 0 ) then
    do i = 1, m
       QPpred%C_l( i )  = max( nlp%C_l(i) - nlp%C(i), -infinity )
       QPpred%C_u( i )  = min( nlp%C_u(i) - nlp%C(i),  infinity )
       QPpred%Y(   i )  = nlp%Y(i)
    end do
 end if
 if ( m_a > 0 ) then
    do i = 1, m_a
       QPpred%C_l( m + i )  = max( nlp%A_l(i) - nlp%Ax(i), -infinity )
       QPpred%C_u( m + i )  = min( nlp%A_u(i) - nlp%Ax(i),  infinity )
       QPpred%Y(   m + i )  = nlp%Y_a(i)
    end do
 end if
 if ( B_type == 2 ) then  ! L-BFGS
    QPpred%Y( m + m_a + 1 : m + m_a + 2*number_used ) = zero
    ! Cl and Cu for L-BFGS is set in build_QPpred.
 end if

 QPpred%X = zero
 QPpred%C = zero

 if ( m > 0 ) then
    ! Note: C --- Js+u-v = u-v  because s=0.
    utally = n+1
    vtally = utally + lenu
    do i = 1, ncrb
       ind = data%crb(i)
       QPpred%X(utally)   = max( zero, nlp%C_l(ind)-nlp%C(ind) )
       QPpred%X(vtally)   = max( zero, nlp%C(ind)-nlp%C_u(ind) )
       !QPpred%X_u(utally) = min( QPpred%X(utally) + hundred, infinity )
       !QPpred%X_u(vtally) = min( QPpred%X(vtally) + hundred, infinity )
       QPpred%C(ind)      = QPpred%X(utally) - QPpred%X(vtally)
       utally = utally + 1
       vtally = vtally + 1
    end do
    do i = 1, nce
       ind = data%ce(i)
       QPpred%X(utally)   = max( zero, nlp%C_l(ind)-nlp%C(ind) )
       QPpred%X(vtally)   = max( zero, nlp%C(ind)-nlp%C_u(ind) )
       !QPpred%X_u(utally) = min( QPpred%X(utally) + hundred, infinity )
       !QPpred%X_u(vtally) = min( QPpred%X(vtally) + hundred, infinity )
       QPpred%C(ind)      = QPpred%X(utally) - QPpred%X(vtally)
       utally = utally + 1
       vtally = vtally + 1
    end do
    do i = 1, nclb
       ind = data%clb(i)
       QPpred%X(utally)   = max( zero, nlp%C_l(ind)-nlp%C(ind) )
       !QPpred%X_u(utally) = min( QPpred%X(utally) + hundred, infinity )
       QPpred%C(ind)      = QPpred%X(utally)
       utally = utally + 1
    end do
    do i = 1, ncub
       ind = data%cub(i)
       QPpred%X(vtally)   = max( zero, nlp%C(ind)-nlp%C_u(ind) )
       !QPpred%X_u(vtally) = min( QPpred%X(vtally) + hundred, infinity )
       QPpred%C(ind)      = QPpred%X(vtally)
       vtally = vtally + 1
    end do
 end if
 if ( B_type == 2 ) then  ! L-BFGS
    QPpred%X( npupv + 1 : npupv + 2*number_used ) = zero
 end if

 do i = 1, n
    if ( data%control%use_TRpred ) then
       QPpred%X_l( i )  = max( nlp%X_l(i) - nlp%X(i), -data%TRpred )
       QPpred%X_u( i )  = min( nlp%X_u(i) - nlp%X(i),  data%TRpred )
    else
       !QPpred%X_l( i )  = max( nlp%X_l(i) - nlp%X(i), -infinity )
       !QPpred%X_u( i )  = min( nlp%X_u(i) - nlp%X(i), infinity )
       QPpred%X_l( i )  = max( nlp%X_l(i) - nlp%X(i), -max(data%TRpred,five) )
       QPpred%X_u( i )  = min( nlp%X_u(i) - nlp%X(i),  max(data%TRpred,five) )
    end if
 end do

 QPpred%Z( 1 : n )         = nlp%Z
 QPpred%Z( n + 1 : npupv ) = zero
 if ( B_type == 2 ) then  ! L-BFGS
    QPpred%Z( npupv + 1 : npupv + 2*number_used ) = zero
 end if

 ! Load QPpred%A with the portion from constraints c(x) and Ax,
 ! and maybe more if limited memeory BFGS is being used.
 !------------------------------------------------------------

 Ane = 0

 ! First the [ J Ir Ie Il -Ir -Ie -Iu ] row.

 if ( m > 0 ) then
    QPpred%A%val( 1 : nlp%J%ne ) = nlp%J%val
    Ane = nlp%J%ne + lenu + lenv  ! rest is set in build_QPpred
 end if

 ! Next, the A part.

 if ( m_a > 0 ) then
    Ane = Ane + nlp%A%ne ! values set in build_QPpred.
 end if

 ! Finally, the matrices for L-BFGS

 if ( B_type == 2 ) then
    A_ind = Ane
    B_ind = A_ind + n*number_used + number_used
    do i = 1, number_used
       QPpred%A%val( A_ind+1 : A_ind+n ) = data%L_bfgs%A(:,i)
       QPpred%A%val( B_ind+1 : B_ind+n ) = data%L_bfgs%B(:,i)
       A_ind = A_ind + n + 1
       B_ind = B_ind + n + 1
    end do
    Ane = Ane + 2*(n+1)*number_used
 end if

 QPpred%A%ne = Ane

 ! Load QPpred%H with the portion from "B".
 ! B_type :  0 = identity     1 = weighted diagonal
 !           2 = L-BFGS       3 = BFGS
 !-------------------------------------------------

 SELECT CASE ( B_type)

 CASE ( 0 )  ! "B" is Identity - diagonal storage.

    ! Relax....all done in build_QPpred

 CASE ( 1 )  ! "B" is weighted diagonal - diagonal storage set in build_QPpred.

    if ( iterate == 1 ) then

       data%B%val(   : n ) = max( B_lammin, min( B_lammax, NRM2(n,nlp%G,1) ) )
       QPpred%H%val( : n ) = data%B%val( : n )

    else

       if ( mod(iterate,2) == 0 ) then
          if ( data%control%use_seqp .or. data%control%use_siqp ) then
             dummy_real = min( max(B_lammin, abs(data%Ss_H_Ss)), B_lammax )
          else
             dummy_real = min( max(B_lammin, abs(data%Sp_B_Sp)), B_lammax )
          end if
          data%B%val(   : n) = dummy_real
          QPpred%H%val( : n) = dummy_real
       else
          if ( data%control%use_seqp .or. data%control%use_siqp ) then
             do i = 1, n
                call mop_getval( nlp%H, i, i, data%B%val(i), .true., out,      &
                                 error, 0 )
                data%B%val(i) = min( max(B_lammin, data%B%val(i) ), B_lammax )
             end do
             QPpred%H%val(1:n) = data%B%val(1:n)
          else
             dummy_real = min( max(B_lammin, NRM2(n,nlp%G,1) ), B_lammax )
             data%B%val(   : n) = dummy_real
             QPpred%H%val( : n) = dummy_real
          end if
       end if

    end if

 CASE ( 2 )  ! L-BFGS : "B" is actually B0 - diagonal storage set as above.

    QPpred%H%val(1:n) = data%B%val
    QPpred%H%val( npupv+1 : npupv+number_used ) = -one
    QPpred%H%val( npupv+number_used+1 : npupv+2*number_used ) = one

 CASE ( 3 )  ! BFGS - coordinate storage.

    if ( print_level >= GALAHAD_debug ) write(out, 1020) ! header

    ! If first B, then fill with identity and exit.

    if ( data%BFGS%update_number == 0 ) then

       data%B%val = zero
       tally = 0
       do i = 1, n
          tally = tally + i
          data%B%val(tally) = one
       end do
       data%QPpred%H%val( : data%B%ne ) = data%B%val

       data%BFGS%mod_type      = 0
       data%BFGS%update_number = 1
       data%BFGS%theta         = one

       if ( print_level >= GALAHAD_debug ) then
          write(out, "(/, ' BFGS - formed identity for first B.')" )
          write(out, 1021) ! footer
       end if

       status = 0
       return

    end if

    if ( data%success >= 1 ) then

       ! Compute needed data.

       data%BFGS%std  = dot_product( data%BFGS%s, data%BFGS%d )
       data%BFGS%Bs   = zero
       call mop_Ax( one, data%B, data%BFGS%s, one, data%BFGS%Bs, &
                    out, error, symmetric=.true. )
       stBs = dot_product( data%BFGS%BS, data%BFGS%s )

       if ( print_level >= GALAHAD_debug ) then
          write(out,1022) data%BFGS%std, stBs
          if ( print_level >= GALAHAD_crazy ) then
             write(out, 1015)
             write( out,  '(4(2x, ES15.8))')                                   &
                  ( data%BFGS%d(i), data%BFGS%BS(i),                           &
                    data%BFGS%gradLx_new(i), data%BFGS%gradLx(i), i = 1, n )
          end if
       end if

       ! If B has not yet been updated, scale it as in Dennis and Schnabel.

       if ( data%BFGS%update_number == 1 ) then

          dummy_real = min( max( abs(data%BFGS%std/stBs), B_lammin), B_lammax)

          data%B%val   = dummy_real * data%B%val
          data%BFGS%update_number = data%BFGS%update_number + 1

          data%BFGS%Bs   = zero
          call mop_Ax( one, data%B, data%BFGS%s, one, data%BFGS%Bs, &
                       out, error, symmetric=.true. )
          stBs = dot_product( data%BFGS%BS, data%BFGS%s )

          if ( print_level >= GALAHAD_debug ) then
             write(out,1023) dummy_real, data%BFGS%std, stBs
             if ( print_level >= GALAHAD_crazy ) then
                write(out, 1015)
                write( out,  '(4(2x, ES15.8))')                                &
                     ( data%BFGS%d(i), data%BFGS%Bs(i),                        &
                       data%BFGS%gradLx_new(i), data%BFGS%gradLx(i), i = 1, n )
             end if
          end if

       end if

       ! Skip the BFGS update if stBs is too small.

       if ( stBs <= teneps) then
          if ( print_level >= GALAHAD_debug ) then
             write( out, 1018 ) stBs
             write( out, 1021 ) ! footer
          end if
          data%QPpred%H%val( : data%B%ne ) = data%B%val
          data%BFGS%mod_type = 0
          data%BFGS%theta = one
          status = 0
          return
       end if

       ! Skip the update if Bs = y already.

       dummy_real = tenm8 * max( one, NRM2(n, data%BFGS%d, 1) )

       if ( NRM2( n, data%BFGS%Bs - data%BFGS%d, 1 ) <= dummy_real ) then
          if ( print_level >= GALAHAD_debug ) then
             write(out,1019)
             write(out,1021) ! footer
          end if
          data%QPpred%H%val( : data%B%ne ) = data%B%val
          data%BFGS%mod_type = 0
          data%BFGS%theta = one
          status = 0
          return
       end if

       ! Perform the BFGS update (currently only damping option available)

       damp_factor = data%BFGS%damp_factor

       if ( data%BFGS%std >= damp_factor * stBs ) then
          data%BFGS%theta = one
          data%BFGS%mod_type = 0
       else

          data%BFGS%mod_type = 1

          data%BFGS%theta = (one - damp_factor)*stBs / (stBs - data%BFGS%std)
          data%BFGS%d = data%BFGS%theta * data%BFGS%d
          data%BFGS%d = data%BFGS%d + (one-data%BFGS%theta) * data%BFGS%Bs

          data%BFGS%std  = dot_product( data%BFGS%d, data%BFGS%s )

          if ( print_level >= GALAHAD_debug ) then
             write( out, 1001 ) data%BFGS%theta, damp_factor
             write( out, 1016 ) data%BFGS%std, stBs
             if ( print_level >= GALAHAD_crazy ) then
                write(out, 1015)
                write( out,  '(4(2x, ES15.8))') &
                     ( data%BFGS%d(i), data%BFGS%Bs(i), data%BFGS%gradLx_new(i), data%BFGS%gradLx(i),  i = 1, n )
             end if
          end if

       end if

       tally = 0
       do i = 1, n
          di  = data%BFGS%d(i)
          Bsi = data%BFGS%Bs(i)
          do j = 1, i
             tally = tally + 1
             Bij = data%B%val(tally)
             Bij = Bij - Bsi*data%BFGS%Bs(j) / stBs
             Bij = Bij + di*data%BFGS%d(j) / data%BFGS%std
             data%B%val(tally) = Bij
          end do
       end do
       data%QPpred%H%val( : data%B%ne ) = data%B%val

       data%BFGS%update_number = data%BFGS%update_number + 1

    else ! Step not successful - just refill.

       data%BFGS%mod_type = 0
       data%BFGS%theta = one

       data%QPpred%H%val( : data%B%ne ) = data%B%val

       if ( print_level >= GALAHAD_debug ) then
          write(out,"(/, ' No update : unsuccessful step')" )
       end if
    end if

    if ( print_level >= GALAHAD_debug ) write(out,1021) ! footer

 CASE DEFAULT

    write( error, 1000  )
    status = GALAHAD_error_input_status
    return

 END SELECT

 ! successful return

 status = 0
 return

 ! format statements

 1000 format(1X, '**ERROR:s2qp_solve:fill_QPpred unrecognized storage type.')
 1001 format(/, 1x, 'Damping used with theta = ', ES12.5, &
                3x, 'and damp-factor = ', ES12.5 )
 1015 format( /, &
      10x, 'd', 15x, 'Bs', 11x, 'gradLxnew', 10x, 'gradLx', /          &
      6x, '---------', 7x, '---------', 8x, '---------', 8x, '----------' )
 1016 format(/, 1x, 'Damped data : std = ', ES14.7, '  stBs =  ', ES14.7 )
 1018 format(/, 1x, 'Skipping update :  stBs = ', ES16.9 )
 1019 format(/, 1x, 'Skipping update :  BS = y already.' )
 1020 format(/, &
      1x, 67('-'), /, &
      1x, 20('-'), '    BEGIN : BFGS Details    ', 19('-'), /, &
      1x, 67('-') )
 1021 format(/, &
      1x, 67('-'), /, &
      1x, 21('-'), '   END : BFGS Details   ', 22('-'), /, &
      1x, 67('-') )
 1022 format(/, 1x, 'Initial data :  std = ', ES14.7, '  stBs = ', ES14.7 )
 1023 format(/, 1x, 'First update : scale = ', ES8.1, &
                 '  std = ', ES9.2, '  stBs = ', ES8.2)
 1024 format(/, &
      1x, 67('-'), /, &
      1x, 20('-'), '   BEGIN : L-BFGS Details    ', 19('-'), /, &
      1x, 67('-') )
 1025 format(/, &
      1x, 67('-'), /, &
      1x, 21('-'), '  END : L-BFGS Details   ', 22('-'), /, &
      1x, 67('-') )

 END SUBROUTINE fill_QPpred

 !************** G A L A H A D  fill_QPsiqp  S U B R O U T I N E ***************

 SUBROUTINE fill_QPsiqp( nlp, QPsiqp, status, data )
 !------------------------------------------------------------------------------
 ! See subroutine build_QPsiqp for more information.
 !------------------------------------------------------------------------------
 IMPLICIT NONE
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_data_type ), intent( in ) :: data
 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( QPT_problem_type ), intent( inout ) :: QPsiqp

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: tally, i, m, m_a, n, num_vl_l, num_vl_u, col, cind, lenuv

 !------------------------------------------------------------------------------

 ! For convenience

 m   = nlp%m
 m_a = nlp%m_a
 n   = nlp%n

 num_vl_l = data%num_vl_l
 num_vl_u = data%num_vl_u

 ! Define X and initial constraint value C while filling the sparsity/values
 ! for A.  Finally, compute proper problem dimensions and fill the Hessian H.
 ! --------------------------------------------------------------------------

 col   = n
 lenuv = 0

 tally = 0
 if ( m > 0 ) then
    tally = tally + nlp%J%ne
    QPsiqp%A%val( : nlp%J%ne ) = nlp%J%val
 end if
 if ( m_a > 0 ) then
    tally = tally + nlp%A%ne
 end if
 QPsiqp%A%val( tally+1 : tally+n ) = data%descent_con
 tally = tally + n

 QPsiqp%X = zero
 QPsiqp%C = zero

 ! Constraints for which c+Jsc violates nlp%C_l.

 do i = 1, num_vl_l

    cind = data%vl_l(i)

    select case ( data%C_type(cind) )

    case ('RB')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = one
       QPsiqp%X(col)       = -data%cauchyRESl(cind)
       QPsiqp%C(cind)      = QPsiqp%X(col)

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = -one

       lenuv = lenuv + 2

    case ('LB')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = one
       QPsiqp%X(col)       = -data%cauchyRESl(cind)
       QPsiqp%C(cind)      = QPsiqp%X(col)

       lenuv = lenuv + 1

    case ('EQ')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = one
       QPsiqp%X(col)       = -data%cauchyRESl(cind)
       QPsiqp%C(cind)      = QPsiqp%X(col)

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = -one

       lenuv = lenuv + 2

    case default

       go to 990

    end select

 end do

 ! constraints for which c+Jsc violates nlp%C_u.

 do i = 1, num_vl_u

    cind = data%vl_u(i)

    select case ( data%C_type(cind) )

    case ('RB')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = one

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = -one
       QPsiqp%X(col)       = -data%cauchyRESu(cind)
       QPsiqp%C(cind)      =  data%cauchyRESu(cind)

       lenuv = lenuv + 2

    case ('UB')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = -one
       QPsiqp%X(col)       = -data%cauchyRESu(cind)
       QPsiqp%C(cind)      =  data%cauchyRESu(cind)

       lenuv = lenuv + 1

    case ('EQ')

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = one

       col = col + 1
       tally = tally + 1
       QPsiqp%A%row(tally) = cind
       QPsiqp%A%col(tally) = col
       QPsiqp%A%val(tally) = -one
       QPsiqp%X(col)       = -data%cauchyRESu(cind)
       QPsiqp%C(cind)      =  data%cauchyRESu(cind)

       lenuv = lenuv + 2

    case default

       go to 990

    end select

 end do

 ! Set dimensions not set in build_QPsiqp

 QPsiqp%n    = n + lenuv
 QPsiqp%A%n  = QPsiqp%n
 QPsiqp%A%ne = tally
 QPsiqp%H%m  = QPsiqp%n
 QPsiqp%H%n  = QPsiqp%n

 QPsiqp%H%val = nlp%H%val

 ! Fill vectors C and Y.
 ! ---------------------

 tally = 0

 ! General constraints C and Y

 if ( m > 0 ) then
    QPsiqp%C_l( : m ) = nlp%C_l( : m ) - data%CplusJSc
    QPsiqp%C_u( : m ) = nlp%C_u( : m ) - data%CplusJSc
    QPsiqp%Y  ( : m ) = nlp%Y
    tally             = m
 end if

 ! Linear constraints C and Y

 if ( m_a > 0 ) then
    QPsiqp%C_l( tally + 1 : tally + m_a )  = nlp%A_l - data%AXplusSc
    QPsiqp%C_u( tally + 1 : tally + m_a )  = nlp%A_u - data%AXplusSc
    QPsiqp%Y(   tally + 1 : tally + m_a )  = nlp%Y_a
    tally                                  = m + m_a
 end if

 ! Descent condition C and Y.

 QPsiqp%Y  ( tally + 1 )  = zero
 if ( data%primal_vl <= sqrt(data%control%stop_p_abs) .and. &
      data%dual_vl   <= sqrt(data%control%stop_d_abs) .and. &
      data%comp_vl   <= sqrt(data%control%stop_c_abs) ) then
    QPsiqp%C_u( tally + 1 )  = data%control%QPsiqp_control%infinity
 else
    QPsiqp%C_u( tally + 1 )  = zero + tenm5  ! DPR: change this?
 end if

 ! Fill the vectors G, Z, X_l, X_u, and status vectors.
 ! ----------------------------------------------------

 QPsiqp%G( : n )             = data%GplusHs
 QPsiqp%G( n + 1 : n + 2*m ) = data%penalty

 do i = 1, n
    QPsiqp%X_l(i) = max(nlp%X_l(i) - ( nlp%X(i)+data%s_c(i) ), -data%TRacc )
    QPsiqp%X_u(i) = min(nlp%X_u(i) - ( nlp%X(i)+data%s_c(i) ),  data%TRacc )
 end do

 QPsiqp%Z(      : n ) = nlp%Z
 QPsiqp%Z ( n+1 :   ) = zero

 ! Set status vectors.
 ! -------------------

 QPsiqp%C_status( 1 : m+m_a ) = data%QPpred%C_status( 1 : m+m_a )
 QPsiqp%X_status( 1 : n )     = data%QPpred%X_status( 1 : n )

 ! Different returns.
 ! ------------------

 ! normal return

 status = 0
 return

 ! abnormal return

 990 continue
     write(data%control%error,1000)
     status = -1
     return

 ! format statement

 1000 format(1x, '**ERROR:s2qp_solve:fill_QPsiqp logic flow descrepancy.')

 END SUBROUTINE fill_QPsiqp

 !*************** G A L A H A D  fill_QPseqp  S U B R O U T I N E **************

 SUBROUTINE fill_QPseqp( nlp, QPseqp, status, data )
 !------------------------------------------------------------------------------
 !  See subroutine build_QPseqp for more information
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 integer, intent( out ) :: status
 type( S2QP_data_type ), intent( inout )    :: data
 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( QPT_problem_type ), intent( inout )  :: QPseqp

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 real ( kind = wp ) :: tol
 integer :: n, m, m_a, row_j
 integer :: nfx, nwA, nwA_comp, nwJ, nwJ_comp, i, j, tally, out, nvl!, nfr
 integer, dimension( max(max(nlp%m, nlp%n), nlp%m_a ) ) :: vl

 !------------------------------------------------------------------------------

 ! For convenience

 n   = nlp%n
 m   = nlp%m
 m_a = nlp%m_a
 out = data%control%out
 tol = point1 * data%control%stop_p_abs

 ! Compute variables and constraints that are active and inactive.
 ! ***************************************************************

 nfx = 0

 ! Get variables that are fixed, free, and violated.

!write(*,*) 'x = ', nlp%X
!write(*,*) 'xl = ', nlp%X_l
!write(*,*) 'resxl = ', data%X_RES_l
!write(*,*) 'sp = ', data%s_p

 if ( data%seqp_try_pred ) then
    call get_active( nlp%X_l, nlp%X_u,                                      &
                     data%X_RES_l + data%s_p, data%X_RES_u - data%s_p,      &
                     data%X_type, nfx, data%fx, data%nfr, data%fr, nvl, vl, &
                     tol, out )
 else
    call get_active( nlp%X_l, nlp%X_u,                                      &
                     data%X_RES_l + data%s_c, data%X_RES_u - data%s_c,      &
                     data%X_type, nfx, data%fx, data%nfr, data%fr, nvl, vl, &
                     tol, out )
 end if

 !if ( nfx + nvl > 0 ) then
 !   write(*,*) 'nfx = ', nfx
 !   write(*,*) 'fx  = ', data%fx
 !   write(*,*) 'nvl = ', nvl
 !   write(*,*) 'vl  = ', vl
 !end if

 data%fx( nfx+1 : nfx+nvl ) = vl( : nvl )  ! count violated as fixed.
 nfx = nfx + nvl
 data%nfx = nfx

 ! Next the linear constraints.

 nwA      = 0
 nwA_comp = 0

 if ( m_a > 0 ) then
    if ( data%seqp_try_pred ) then
       call get_active( nlp%A_l, nlp%A_u,                                    &
                        data%A_RES_l + data%AxSp, data%A_RES_u - data%AxSp,  &
                        data%A_type, nwA, data%wA, nwA_comp, data%wA_comp,   &
                        nvl, vl, tol, out )
       data%wA( nwA+1 : nwA+nvl ) = vl( : nvl )  ! nvl should be zero.
       nwA = nwA + nvl
       !   do i = 1, m_a
       !        if ( data%QPpred%C_status( m + i ) /= 0 ) then
       !           nwA = nwA + 1 ;  data%wA( nwA ) = i
       !        else
       !           nwA_comp = nwA_comp + 1 ;  data%wA_comp( nwA_comp ) = i
       !        end if
       !   end do
    else
       call get_active( nlp%A_l, nlp%A_u,                                    &
                        data%A_RES_l + data%AxSc, data%A_RES_u - data%AxSc,  &
                        data%A_type, nwA, data%wA, nwA_comp, data%wA_comp,   &
                        nvl, vl, tol, out )
       data%wA( nwA+1 : nwA+nvl ) = vl( : nvl )  ! nvl should be zero.
       nwA = nwA + nvl
    end if
 end if

 data%nwA      = nwA
 data%nwA_comp = nwA_comp

 ! Finally, the general constraints.

 nwJ      = 0
 nwJ_comp = 0

 if ( m > 0 ) then
    if ( data%seqp_try_pred ) then
       call get_active( nlp%C_l, nlp%C_u,                                   &
                        data%C_RES_l + data%JxSp, data%C_RES_u - data%JxSp, &
                        data%C_type, nwJ, data%wJ, nwJ_comp, data%wJ_comp,  &
                        nvl, vl, tol, out )
       data%wJ( nwJ+1 : nwJ+nvl ) = vl( : nvl ) ! count violated as active.
       nwJ = nwJ + nvl
       ! do i = 1, m
       !           if ( data%QPpred%C_status( i ) /= 0 ) then
       !              nwJ = nwJ + 1 ;  data%wJ( nwJ ) = i
       !           else
       !              nwJ_comp = nwJ_comp + 1 ;  data%wJ_comp( nwJ_comp ) = i
       !           end if
       ! end do
    else
       call get_active( nlp%C_l, nlp%C_u,                                   &
                        data%C_RES_l + data%JxSc, data%C_RES_u - data%JxSc, &
                        data%C_type, nwJ, data%wJ, nwJ_comp, data%wJ_comp,  &
                        nvl, vl, tol, out )
       data%wJ( nwJ+1 : nwJ+nvl ) = vl( : nvl ) ! count violated as active.
       nwJ = nwJ + nvl
    end if
 end if

 data%nwJ = nwJ ;  data%nwJ_comp = nwJ_comp

 ! Now define the problem.
 ! ***********************

 ! The dimensions.

 QPseqp%n = n ;  QPseqp%m = nwJ + nwA + nfx

 ! The QP Hessian.

 QPseqp%H%val = nlp%H%val

 ! The gradient.

 if ( data%seqp_try_pred ) then
    QPseqp%G = nlp%G + data%HxSp
 else
    QPseqp%G = nlp%G + data%HxSc
 end if

 ! Starting point and multipliers

 QPseqp%X = zero ;  QPseqp%Y = zero ! DPR: maybe change later.

 ! ------------------------
 ! The constraint matrix A.
 ! ------------------------

 tally = 0

 ! First the predicted active nonlinear constraints.

 if ( m > 0 ) then
    J_1: do j = 1, nlp%J%ne
       row_j = nlp%J%row( j )
       I_1: do i = 1, nwJ
          if ( row_j == data%wJ( i ) ) then
             tally = tally + 1
             QPseqp%A%row( tally ) = i
             QPseqp%A%col( tally ) = nlp%J%col( j )
             QPseqp%A%val( tally ) = nlp%J%val( j )
             exit I_1
          end if
       end do I_1
    end do J_1
 end if

 ! Now the active linear constraints.

 if ( m_a > 0 ) then
    J_2: do j = 1, nlp%A%ne
       row_j = nlp%A%row( j )
       I_2: do i = 1, nwA
          if ( row_j == data%wA( i ) ) then
             tally = tally + 1
             QPseqp%A%row( tally ) = nwJ + i
             QPseqp%A%col( tally ) = nlp%A%col( j )
             QPseqp%A%val( tally ) = nlp%A%val( j )
             exit I_2
          end if
       end do I_2
    end do J_2
 end if

 ! Finally the fixed variables.

 do i = 1, nfx
    tally = tally + 1
    QPseqp%A%row( tally ) = nwJ + nwA + i
    QPseqp%A%col( tally ) = data%fx( i )
    QPseqp%A%val( tally ) = one
 end do

 ! The total

 QPseqp%A%ne = tally ;  QPseqp%A%m = nwJ + nwA + nfx ;  QPseqp%A%n = n

 ! Set the trust-region.

 data%control%QPseqp_control%radius = data%TRacc

 status = 0
 return

 END SUBROUTINE fill_QPseqp

! ******************************************************************************
!                      d e a l l o c _ Q P f e a s                             |
! ******************************************************************************

  SUBROUTINE dealloc_QPfeas( QPfeas, status, alloc_status, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Deallocates components of variable QPfeas of type QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------------
!   D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------------

    !type( S2QP_inform_type ), intent( inout )  :: inform
    integer, intent( out ) :: status, alloc_status
    type( S2QP_control_type ), intent( inout ) :: control
    type( QPT_problem_type ), intent( inout )  :: QPfeas

!-------------------------------------------------------------------------------
!   L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------------

    character( len = 80 ) :: array_name, bad_alloc
    integer :: error

!-------------------------------------------------------------------------------

    ! For convenience.

    error = control%error

    ! Deallocate components independent of storage type.

    array_name = 'S2QP : data%QPfeas%G'
    CALL SPACE_dealloc_array( QPfeas%G, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%X_l'
    CALL SPACE_dealloc_array( QPfeas%X_l, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND.  status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%X'
    CALL SPACE_dealloc_array( QPfeas%X, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%X_u'
    CALL SPACE_dealloc_array( QPfeas%X_u, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%X_status'
    CALL SPACE_dealloc_array( QPfeas%X_status, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%Z'
    CALL SPACE_dealloc_array( QPfeas%Z, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%C_l'
    CALL SPACE_dealloc_array( QPfeas%C_l, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%C'
    CALL SPACE_dealloc_array( QPfeas%C, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%C_u'
    CALL SPACE_dealloc_array( QPfeas%C_u, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%C_status'
    CALL SPACE_dealloc_array( QPfeas%C_status, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%Y'
    CALL SPACE_dealloc_array( QPfeas%Y, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%A%val'
    CALL SPACE_dealloc_array( QPfeas%A%val, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%A%row'
    CALL SPACE_dealloc_array( QPfeas%A%row, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%A%col'
    CALL SPACE_dealloc_array( QPfeas%A%col, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%A%ptr'
    CALL SPACE_dealloc_array( QPfeas%A%ptr, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    array_name = 'S2QP : data%QPfeas%A%type'
    CALL SPACE_dealloc_array( QPfeas%A%type, status, alloc_status, &
         array_name = array_name, bad_alloc = bad_alloc, out = error )
    IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

    ! Normal return

    return

  END SUBROUTINE dealloc_QPfeas

 !************ G A L A H A D  dealloc_QPpred  S U B R O U T I N E **************

 SUBROUTINE dealloc_QPpred( QPpred, status, alloc_status, control )
 !------------------------------------------------------------------------------
 ! Deallocates components of variable QPpred of type QPT_problem_type.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 integer, intent( out ) :: status, alloc_status
 type( S2QP_control_type ), intent( inout ) :: control
 type( QPT_problem_type ), intent( inout )  :: QPpred

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 character( len = 80 ) :: array_name, bad_alloc
 integer :: error

 !------------------------------------------------------------------------------

 ! For convenience.

 error = control%error

 ! Deallocate components independent of storage type.

 array_name = 'S2QP : data%QPpred%G'
 CALL SPACE_dealloc_array( QPpred%G, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%X_l'
 CALL SPACE_dealloc_array( QPpred%X_l, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%X'
 CALL SPACE_dealloc_array( QPpred%X, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%X_u'
 CALL SPACE_dealloc_array( QPpred%X_u, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%X_status'
 CALL SPACE_dealloc_array( QPpred%X_status, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%Z'
 CALL SPACE_dealloc_array( QPpred%Z, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%C_l'
 CALL SPACE_dealloc_array( QPpred%C_l, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%C'
 CALL SPACE_dealloc_array( QPpred%C, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%C_u'
 CALL SPACE_dealloc_array( QPpred%C_u, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%C_status'
 CALL SPACE_dealloc_array( QPpred%C_status, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%Y'
 CALL SPACE_dealloc_array( QPpred%Y, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%A%val'
 CALL SPACE_dealloc_array( QPpred%A%val, status, alloc_status, &
      array_name = array_name,  bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%A%row'
 CALL SPACE_dealloc_array( QPpred%A%row, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%A%col'
 CALL SPACE_dealloc_array( QPpred%A%col, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%A%type'
 CALL SPACE_dealloc_array( QPpred%A%type, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%A%ptr'
 CALL SPACE_dealloc_array( QPpred%A%ptr, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%H%val'
 CALL SPACE_dealloc_array( QPpred%H%val, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%H%row'
 CALL SPACE_dealloc_array( QPpred%H%row, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%H%col'
 CALL SPACE_dealloc_array( QPpred%H%col, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%H%ptr'
 CALL SPACE_dealloc_array( QPpred%H%ptr, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 array_name = 'S2QP : data%QPpred%H%type'
 CALL SPACE_dealloc_array( QPpred%H%type, status, alloc_status, &
      array_name = array_name, bad_alloc = bad_alloc, out = error )
 IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

 END SUBROUTINE dealloc_QPpred

! ******************************************************************************
!                      d e a l l o c _ Q P s i q p                             |
! ******************************************************************************

  SUBROUTINE dealloc_QPsiqp( QPsiqp, status, alloc_status, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Deallocates components of variable QPsiqp of type QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------------
! D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------------

  !type( S2QP_inform_type ), intent( inout )  :: inform
  integer, intent( out ) :: status, alloc_status
  type( S2QP_control_type ), intent( inout ) :: control
  type( QPT_problem_type ), intent( inout )     :: QPsiqp

!-------------------------------------------------------------------------------
! L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------------

  character( len = 80 ) :: array_name, bad_alloc
  integer :: error

!-------------------------------------------------------------------------------

  ! For convenience.

  error = control%error

  ! Deallocate components independent of storage type.

  array_name = 'S2QP : data%QPsiqp%G'
  CALL SPACE_dealloc_array( QPsiqp%G, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%X_l'
  CALL SPACE_dealloc_array( QPsiqp%X_l, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%X'
  CALL SPACE_dealloc_array( QPsiqp%X, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%X_u'
  CALL SPACE_dealloc_array( QPsiqp%X_u, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%X_status'
  CALL SPACE_dealloc_array( QPsiqp%X_status, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%Z'
  CALL SPACE_dealloc_array( QPsiqp%Z, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%C_l'
  CALL SPACE_dealloc_array( QPsiqp%C_l, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%C'
  CALL SPACE_dealloc_array( QPsiqp%C, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%C_u'
  CALL SPACE_dealloc_array( QPsiqp%C_u, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%C_status'
  CALL SPACE_dealloc_array( QPsiqp%C_status, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%Y'
  CALL SPACE_dealloc_array( QPsiqp%Y, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%A%val'
  CALL SPACE_dealloc_array( QPsiqp%A%val, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%A%row'
  CALL SPACE_dealloc_array( QPsiqp%A%row, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%A%col'
  CALL SPACE_dealloc_array( QPsiqp%A%col, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%A%type'
  CALL SPACE_dealloc_array( QPsiqp%A%type, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%A%ptr'
  CALL SPACE_dealloc_array( QPsiqp%A%ptr, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%H%val'
  CALL SPACE_dealloc_array( QPsiqp%H%val, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%H%row'
  CALL SPACE_dealloc_array( QPsiqp%H%row, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%H%col'
  CALL SPACE_dealloc_array( QPsiqp%H%col, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%H%ptr'
  CALL SPACE_dealloc_array( QPsiqp%H%ptr, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPsiqp%H%type'
  CALL SPACE_dealloc_array( QPsiqp%H%type, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  END SUBROUTINE dealloc_QPsiqp

! ******************************************************************************
!                      d e a l l o c _ Q P s e q p                             |
! ******************************************************************************

  SUBROUTINE dealloc_QPseqp( QPseqp, status, alloc_status, control )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!   Deallocates components of variable QPseqp of type QPT_problem_type.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------------
! D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------------

  !type( S2QP_inform_type ), intent( inout )  :: inform
  integer, intent( out ) :: status, alloc_status
  type( S2QP_control_type ), intent( inout ) :: control
  type( QPT_problem_type ), intent( inout )  :: QPseqp

!-------------------------------------------------------------------------------
! L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------------

  character( len = 80 ) :: array_name, bad_alloc
  integer :: error

!-------------------------------------------------------------------------------

  ! For convenience.

  error = control%error

  ! Deallocate components independent of storage type.

  array_name = 'S2QP : data%QPseqp%G'
  CALL SPACE_dealloc_array( QPseqp%G, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%X'
  CALL SPACE_dealloc_array( QPseqp%X, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%C'
  CALL SPACE_dealloc_array( QPseqp%C, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%Y'
  CALL SPACE_dealloc_array( QPseqp%Y, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  !array_name = 'S2QP : data%QPseqp%Z'
  !CALL SPACE_dealloc_array( QPseqp%Z, status, alloc_status, &
  !     array_name = array_name, bad_alloc = bad_alloc, out = error )
  !IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%A%val'
  CALL SPACE_dealloc_array( QPseqp%A%val, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%A%row'
  CALL SPACE_dealloc_array( QPseqp%A%row, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%A%col'
  CALL SPACE_dealloc_array( QPseqp%A%col, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%A%type'
  CALL SPACE_dealloc_array( QPseqp%A%type, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%A%ptr'
  CALL SPACE_dealloc_array( QPseqp%A%ptr, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%H%val'
  CALL SPACE_dealloc_array( QPseqp%H%val, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%H%row'
  CALL SPACE_dealloc_array( QPseqp%H%row, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%H%col'
  CALL SPACE_dealloc_array( QPseqp%H%col, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%H%ptr'
  CALL SPACE_dealloc_array( QPseqp%H%ptr, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  array_name = 'S2QP : data%QPseqp%H%type'
  CALL SPACE_dealloc_array( QPseqp%H%type, status, alloc_status, &
       array_name = array_name, bad_alloc = bad_alloc, out = error )
  IF ( control%deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

  END SUBROUTINE dealloc_QPseqp

! ******************************************************************************
!                            p r i n t _  S M T                                |
! ******************************************************************************

  SUBROUTINE print_SMT( A, name, error, out, status )

    IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!   Prints variable of type SMT_type that is stored in coordinate format.
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------------
! D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------------

  integer, intent( in ) :: error, out
  integer, intent( inout ) :: status
  character(len=*), intent( in ) :: name
  type( SMT_type ), intent( in )  :: A

!-------------------------------------------------------------------------------
! L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------------

  integer :: i

!------------------------------------------------------------------------------

  ! Default return value.

  status = -1

  ! Print the matrix.
  ! *****************

  if ( SMT_get( A%type ) == 'COORDINATE' ) then

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

  else

     status = -1
     write( error, 1000)
     return

  end if

  return

! format statements

1000 FORMAT(3X, ' ** ERROR s2qp:print_SMT:only prints coordinate storage.')
2000 FORMAT( /, &
  1X, '                    STATISTICS FOR MATRIX : ', A, /, &
  1X, '                    --------------------- ' )
3000 FORMAT(/, T16, 'm  = ', I10, 7X, 'type = ', 60A )
3001 FORMAT(   T16, 'n  = ', I10, 7X, 'id   = ', 60A )
3002 FORMAT(   T16, 'ne = ', I10 )
3003 FORMAT(   T16, 'n  = ', I10 )

  END SUBROUTINE print_SMT


! ******************************************************************************
!                        g e t _ b e s t _ o p t                               |
! ******************************************************************************

 SUBROUTINE get_best_opt( nlp, data )

 IMPLICIT NONE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Computes optimality measure for problem NLP.  Attempts to find the "best"
!  multipliers by considering the current multipliers, the predictor
!  multipliers, and the ACC multipliers (when computed).
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-------------------------------------------------------------------------------
! D u m m y   A r g u m e n t s
!-------------------------------------------------------------------------------

  type( NLPT_problem_type ), intent( inout ) :: nlp
  type( s2qp_data_type ), intent( inout ) :: data

!-------------------------------------------------------------------------------
! L o c a l   V a r i a b l e s
!-------------------------------------------------------------------------------

  integer :: out, error, m, n, m_a, best_loc
  logical :: optimal, feas_optimal
  real( kind = wp ) :: infinity

! ------------------------------------------------------------------------------

  ! For convenience

  out      = data%control%out
  error    = data%control%error
  m        = nlp%m
  n        = nlp%n
  m_a      = nlp%m_a
  infinity = data%control%infinity

 ! Default return values.

  data%primal_vl_cur   = infinity
  data%dual_vl_cur     = infinity
  data%comp_vl_cur     = infinity
  data%opt_measure_cur = infinity

  data%primal_vl_s   = infinity
  data%dual_vl_s     = infinity
  data%comp_vl_s     = infinity
  data%opt_measure_s = infinity

  data%primal_vl_p   = infinity
  data%dual_vl_p     = infinity
  data%comp_vl_p     = infinity
  data%opt_measure_p = infinity

  ! First check if we are primal feasible (independent of multipliers).
  ! *******************************************************************

  call get_opt( .true., nlp, data,                                          &
                C_type=data%C_type, A_type=data%A_type, X_type=data%X_type, &
                C_RES_l=data%C_RES_l, C_RES_u=data%C_RES_u,                 &
                A_RES_l=data%A_RES_l, A_RES_u=data%A_RES_u,                 &
                X_RES_l=data%X_RES_l, X_RES_u=data%X_RES_u,                 &
                primal_vl=data%primal_vl_cur, optimal=feas_optimal,         &
                i_dev=out, e_dev=error )

  data%primal_vl_p = data%primal_vl_cur
  data%primal_vl_s = data%primal_vl_cur

  ! Be optimistic, check ACC multipliers first.
  ! *******************************************

  if ( data%seqp_computed .or. data%siqp_computed ) then

     call get_opt( feas_optimal, nlp, data, nlp%G, data%JtY_s, data%AtYa_s, &
                   data%Z_s, data%Y_s, data%Ya_s,                           &
                   data%C_type, data%A_type, data%X_type,                   &
                   data%C_RES_l, data%C_RES_u,                              &
                   data%A_RES_l, data%A_RES_u,                              &
                   data%X_RES_l, data%X_RES_u,                              &
                   dual_vl=data%dual_vl_s, comp_vl=data%comp_vl_s,          &
                   optimal=optimal, i_dev=out, e_dev=error )

     data%opt_measure_s = half * max( data%dual_vl_s, data%comp_vl_s )

     if ( optimal ) then
        data%converged = .true.
        best_loc = 3
        go to 700
     end if

  end if

  ! Try the predictor multipliers (ACC multipliers must not have been optimal).
  ! ***************************************************************************

  call get_opt( feas_optimal, nlp, data, nlp%G, data%JtY_p, data%AtYa_p, &
                data%QPpred%Z( : n), data%QPpred%Y( : m),           &
                data%QPpred%Y(m+1 : m+m_a),                         &
                data%C_type, data%A_type, data%X_type,              &
                data%C_RES_l, data%C_RES_u,                         &
                data%A_RES_l, data%A_RES_u,                         &
                data%X_RES_l, data%X_RES_u,                         &
                dual_vl=data%dual_vl_p, comp_vl=data%comp_vl_p,     &
                optimal=optimal, i_dev=out, e_dev=error )

  data%opt_measure_p = point9 * max( data%dual_vl_p, data%comp_vl_p )

  if ( optimal ) then
     data%converged = .true.
     best_loc = 2
     go to 700
  end if

  ! Check current multipliers (ACC and predictor multipliers both not optimal).
  ! ***************************************************************************

  call get_opt( feas_optimal, nlp, data, nlp%G, data%JtY, data%AtYa, &
                nlp%Z, nlp%Y, nlp%Y_a,                               &
                data%C_type, data%A_type, data%X_type,               &
                data%C_RES_l, data%C_RES_u,                          &
                data%A_RES_l, data%A_RES_u,                          &
                data%X_RES_l, data%X_RES_u,                          &
                dual_vl=data%dual_vl_cur,  comp_vl=data%comp_vl_cur, &
                optimal=optimal, i_dev=out, e_dev = error )

  data%opt_measure_cur = max( data%dual_vl_cur, data%comp_vl_cur )

  if ( optimal ) then
     data%converged = .true.
     best_loc = 1
     go to 700
  end if

  ! Choose the best
  ! ***************

  if ( data%iterate == 1 ) then
     if ( data%seqp_computed .or. data%siqp_computed ) then
        best_loc = 3
     else
        best_loc = 2
     end if
  else
     best_loc = minloc( (/ data%opt_measure_cur,        &
                           data%opt_measure_p,          &
                           data%opt_measure_s /), dim=1 )
  end if

 700 continue

  if ( best_loc == 3 ) then  ! use ACC multipliers

     data%best_mults = 3
     data%mults_used = 'S'

     if ( m > 0 ) then
        nlp%Y = data%Y_s
     end if
     if ( m_a > 0 ) then
        nlp%Y_a = data%Ya_s
     end if
     nlp%Z = data%Z_s

     data%inf_norm_Y = data%inf_norm_Y_s

     data%primal_vl = data%primal_vl_s
     data%dual_vl   = data%dual_vl_s
     data%comp_vl   = data%comp_vl_s

     data%opt_measure = data%opt_measure_s
     data%JtY = data%JtY_s
     data%AtYa = data%AtYa_s

  elseif ( best_loc == 2 ) then  ! use predictor multipliers.

     data%best_mults = 2
     data%mults_used = 'P'

     if ( m > 0 ) then
        nlp%Y = data%QPpred%Y( : m )
     end if
     if ( m_a > 0 ) then
        nlp%Y_a = data%QPpred%Y( m + 1 : m + m_a )
     end if
     nlp%Z = data%QPpred%Z( : n)

     data%inf_norm_Y = data%inf_norm_Y_p

     data%primal_vl = data%primal_vl_p
     data%dual_vl   = data%dual_vl_p
     data%comp_vl   = data%comp_vl_p

     data%opt_measure = data%opt_measure_p
     data%JtY = data%JtY_p
     data%AtYa = data%AtYa_p

  else  ! keep the current mults.

     data%best_mults = 1
     data%mults_used = 'C'

     data%primal_vl = data%primal_vl_cur
     data%dual_vl   = data%dual_vl_cur
     data%comp_vl   = data%comp_vl_cur

     data%opt_measure = data%opt_measure_cur
     !data%JtY = data%JtY_cur
     !data%AtYa = data%AtYa_cur

  end if

  return

  end SUBROUTINE get_best_opt

 !**************** G A L A H A D  get_opt  S U B R O U T I N E *****************

 SUBROUTINE get_opt( test_opt, nlp, data,                             &
                     G, JtY, AtY_a, Z, Y, Y_a,                        &
                     C_type, A_type, X_type,                          &
                     C_RES_l, C_RES_u, A_RES_l, A_RES_u,              &
                     X_RES_l, X_RES_u, primal_vl, dual_vl, comp_vl,   &
                     optimal, i_dev, e_dev )
 !------------------------------------------------------------------------------
 !
 !  Computes optimality measures and verifies optimality for problem NLP using
 !  a combination of absolute and relative measures.
 !
 !  test_opt        scalar logical variable.  Set true if user wishes to check
 !                  optimality for problem NLP by checking a combination of
 !                  absolute and relative optimality measures.  Otherwise, set
 !                  it to false.  Even in this case the values for primal_vl,
 !                  dual_vl, and comp_vl will be computed if they are present.
 !  nlp             scalar variable of type nlpt_problem_type that holds the
 !                  problem NLP.  If input value is test_optimal=.true., then
 !                  the user must also supply the optional argument "optimal".
 !  data            scalar variable of type s2qp_data_type that holds the data
 !                  required for using the module s2qp.
 !  G               real vector of length nlp%n that holds the gradient of the
 !                  the objective function for problem NLP.
 !  JtY             real vector valued variable holding the Jacobian of the
 !                  general constraints (transposed) multiplied by the current
 !                  estimate Y of optimal Lagrange multiplier
 !                  vector for the nonlinear constraints for problem NLP.
 !  AtY             real vector valued variable holding the Jacobian of the
 !                  linear constraints (transposed) multiplied by the current
 !                  estimate Y_a of optimal Lagrange multiplier
 !                  vector for the linear constraints for problem NLP.
 !  Z               real vector of length nlp%n that holds the gradient of the
 !                  the objective function for problem NLP.
 !  Y               real vector of length nlp%m that holds an estimate of the
 !                  optimal Lagrange multiplier vector for the general
 !                  constraints of problem NLP.
 !  Y_a             real vector of length nlp%m_a that holds an estimate of the
 !                  optimal Lagrange multiplier vector for the linear
 !                  constraints of problem NLP.
 !  C_type, ...     vector array of type char of length 2 holding the
 !                  constraint type: LB=lower bound, UB=upper bound,
 !                                   EQ=equality, RB=range bound, FR=free.
 !  C_res_l, ...    vector of type real of length nlp%m.  If the ith constraint
 !                  is a lower bound constraint, i.e. C_type(i)="LB", or if the
 !                  ith constraint is an equality constraint, i.e.,
 !                  C_type(i)="EQ", or if the ith constraint is range bounded,
 !                  i.e. C_type(i)="RB", then C_res_l(i) holds the residual of
 !                  the ith general constraint, i.e.,
 !                  C_res_l(i) = nlp%C(i) - nlp%C_l(i).  Note: if C_res_l(i) is
 !                  nonnegative and a lower bound, then the ith constraint is
 !                  feasible.  If it is an equality constraint, then it is
 !                  feasible if the residual is zero.
 !  C_res_u, ...    vector of type real of length nlp%m.  If the ith constraint
 !                  is an upper bound constraint, i.e. C_type(i)="UB", or the
 !                  ith constraint is ranged bounded, i.e. C_type="RB", then
 !                  C_res_u(i) holds the residual of the ith general constraint,
 !                  i.e., C_res_u(i) = nlp%C_u(i) - nlp%C(i).  Note: if
 !                  C_res_u(i) is nonnegative and an upper bound, then the
 !                  ith constraint is feasible.
 !  primal_vl       (optional) real variable.  If present, then it will contain
 !                  the primal violation on exit.  User must also supply C_type,
 !                  A_type, X_type, C_res, A_res, and X_res.  Otherwise, we do
 !                  not compute the primal violation.
 !  dual_vl         (optional) real variable.  If present, then it will contain
 !                  the dual violation on exit.  User must also supply JtY,
 !                  AtY_a, G, and Z.  Otherwise, we do not compute the dual
 !                  violation.
 !  comp_vl         (optional) real variable.  If present, then it will contain
 !                  the complementarity violation on exit.  User must also
 !                  supply Y, Y_a, Z, C_res, A_res, X_res, C_type, A_type, and
 !                  X_type.  Otherwise, we do not
 !                  compute the complementarity violation.
 !  optimal         (optional) logical scalar variable that upon exit is set to
 !                  true if the optimality criterion for NLP are tested AND met
 !                  (only tests those from primal_vl, dual_vl, and comp_vl that
 !                  are present).  Set to false otherwise.
 !  i_dev           (optional) integer variable containing the device number for
 !                  the printing of information.
 !  e_dev           (optional) integer variable containing the device number for
 !                  the printing of error messages.
 !
 !  Note1: If test_opt is set true., we compute absolute and relative quantities
 !         for each of primal_vl, dual_vl and comp_vl that are present as
 !         argments; if at any point one of these criterion fails, we no longer
 !         check the remaining absolute/relative quantities for optimality.
 !         In all cases, we exit with (absolute) optimality values for those
 !         of primal_vl, dual_vl, and comp_vl that are present.  If test_opt is
 !         set false, we only compute (absolute) optimality values for those
 !         of primal_vl, dual_vl, and comp_vl that are present.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( inout ) :: nlp
 type( S2QP_data_type ), intent(in) :: data
 real( kind = wp ), dimension( : ), intent( in ), optional :: JtY, AtY_a
 real( kind = wp ), dimension( : ), intent( in ), optional :: G
 real( kind = wp ), dimension( : ), intent( in ), optional :: Y, Y_a
 real( kind = wp ), dimension( : ), intent( in ), optional :: Z
 real( kind = wp ), dimension( : ), intent( in ), optional :: C_RES_l, C_RES_u
 real( kind = wp ), dimension( : ), intent( in ), optional :: A_RES_l, A_RES_u
 real( kind = wp ), dimension( : ), intent( in ), optional :: X_RES_l, X_RES_u
 real( kind = wp ), intent( out ), optional :: primal_vl, dual_vl, comp_vl
 character( len = 2 ), dimension( : ), intent( in ), optional :: C_type, A_type
 character( len = 2 ), dimension( : ), intent( in ), optional :: X_type
 integer, intent( in ), optional  :: i_dev, e_dev
 logical, intent( out ), optional :: optimal
 logical, intent( in ) :: test_opt

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: i, l, out, error, m, m_a, n
 logical :: continue_testing
 real( kind = wp ) :: aZi, aYi, ires
 real( kind = wp ) :: stop_p_abs, stop_p_rel
 real( kind = wp ) :: stop_d_abs, stop_d_rel
 real( kind = wp ) :: stop_c_abs, stop_c_rel
 real( kind = wp ) :: comp_meas, comp_meas1, comp_meas2, dummy
 real( kind = wp ), dimension( nlp%n ) :: agradLag

 !------------------------------------------------------------------------------

 ! Check for output device.

 if ( present( i_dev ) ) then
    out = i_dev
 else
    out = 6
 end if
 if ( present( e_dev ) ) then
    error = e_dev
 else
    error = 6
 end if

 ! Check to make sure things make sense.

 if ( test_opt ) then
    if ( .not. present(optimal) ) then
      write(error,*) " **ERROR:s2qp_solve:get_opt argument OPTIMAL is required."
       return
    else
       optimal = .true. ! default return
    end if
 else
    if ( present(optimal) ) then
       optimal = .false. ! just to prevent errors in returns
    end if
 end if

 continue_testing = test_opt

 ! For convenience.

 m   = nlp%m
 m_a = nlp%m_a
 n   = nlp%n

 stop_p_abs = data%control%stop_p_abs
 stop_p_rel = data%control%stop_p_rel
 stop_d_abs = data%control%stop_d_abs
 stop_d_rel = data%control%stop_d_rel
 stop_c_abs = data%control%stop_c_abs
 stop_c_rel = data%control%stop_c_rel

 ! Primal violation.
 !------------------

 if ( present( primal_vl ) ) then

    primal_vl = zero

    ! First constraints nlp%C

    l = 0

    if ( continue_testing ) then
       do i = 1, m
          l = i
          select case( C_type(i) )
          case('LB')
             ires = C_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%C_l(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             ires = C_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%C_u(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             ires = C_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%C_l(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
             ires = C_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%C_u(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ires = abs( C_RES_l(i) )
             primal_vl =  min( primal_vl, -ires )
             dummy = stop_p_rel * max( abs(nlp%C_l(i)), one )
             if ( ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('FR')
             ! relax
          case default
             write( out, * ) '**ERROR:s2qp_solve:get_opt C_TYPE = ? '
          end select
       end do
    end if

    do i = l+1, m
       select case( C_type(i) )
       case('LB')
          primal_vl =  min( primal_vl, C_RES_l(i) )
       case('UB')
          primal_vl =  min( primal_vl, C_RES_u(i) )
       case('RB')
          primal_vl =  min( primal_vl, C_RES_l(i) )
          primal_vl =  min( primal_vl, C_RES_u(i) )
       case('EQ')
          primal_vl =  min( primal_vl, -abs(C_RES_l(i)) )
       case('FR')
          ! relax
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt: C_TYPE = ?'
       end select
    end do

    ! Now linear constraints

    l = 0

    if ( continue_testing ) then
       do i = 1, m_a
          l = i
          select case( A_type(i) )
          case('LB')
             ires = A_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%A_l(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             ires = A_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%A_u(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             ires = A_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%A_l(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
             ires = A_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( abs(nlp%A_u(i)), one )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ires = abs( A_RES_l(i) )
             primal_vl = min( primal_vl, -ires )
             dummy = stop_p_rel * max( abs(nlp%A_l(i)), one )
             if ( ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
             end if
          case('FR')
             ! relax
          case default
             write( out, * ) ' **ERROR:s2qp_solve:get_opt A_TYPE = ?'
          end select
       end do
    end if

    do i = l+1, m_a
       select case( A_type(i) )
       case('LB')
          primal_vl =  min( primal_vl, A_RES_l(i) )
       case('UB')
          primal_vl =  min( primal_vl, A_RES_u(i) )
       case('RB')
          primal_vl =  min( primal_vl, A_RES_l(i) )
          primal_vl =  min( primal_vl, A_RES_u(i) )
       case('EQ')
          primal_vl =  min( primal_vl, -abs(A_RES_l(i)) )
       case('FR')
          ! relax
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt A_TYPE = ?'
       end select
    end do

    ! Finally the bound constraints.

    l = 0

    if ( continue_testing ) then
       do i = 1, n
          l = i
          select case( X_type(i) )
          case('LB')
             ires = X_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( one, abs(nlp%X_l(i)) )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             ires = X_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( one, abs(nlp%X_u(i)) )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             ires = X_RES_l(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( one, abs(nlp%X_l(i)) )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
             ires = X_RES_u(i)
             primal_vl = min( primal_vl, ires )
             dummy = stop_p_rel * max( one, abs(nlp%X_u(i)) )
             if ( -ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ires = abs( X_RES_l(i) )
             primal_vl = min( primal_vl, -ires )
             dummy = stop_p_rel * max( one, abs(nlp%X_l(i)) )
             if ( ires > max( stop_p_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('FR')
             ! relax
          case default
             write( out, * ) ' **ERROR:s2qp_solve:get_opt X_TYPE = ?'
          end select
       end do
    end if

    do i = l+1, n
       select case( X_type(i) )
       case('LB')
          primal_vl =  min( primal_vl, X_RES_l(i) )
       case('UB')
          primal_vl =  min( primal_vl, X_RES_u(i) )
       case('RB')
          primal_vl =  min( primal_vl, X_RES_l(i) )
          primal_vl =  min( primal_vl, X_RES_u(i) )
       case('EQ')
          primal_vl =  min( primal_vl, -abs(X_RES_l(i)) )
       case('FR')
          ! relax
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt X_TYPE = ?'
       end select
    end do

    primal_vl = abs( primal_vl )

 end if

 ! Dual violation
 !---------------

 if ( present( dual_vl ) ) then

    agradLag = G
    if ( m > 0 ) then
       agradLag = agradLag - JtY
    end if
    if ( m_a > 0 ) then
       agradLag = agradLag - AtY_a
    end if
    agradLag = agradLag - Z
    agradLag = abs( agradLag )

    dual_vl = maxval( agradLag )

    if ( continue_testing ) then
       do i = 1, n
          dummy = stop_d_rel * max( abs(G(i)), one )
          if ( agradLag(i) >  max( stop_d_abs, dummy ) ) then
             optimal = .false.
             continue_testing = .false.
             exit
          end if
       end do
    end if

 end if

 ! Complementarity violation
 !--------------------------

 if ( present( comp_vl ) ) then

    comp_vl = zero

    l = 0

    ! First general constraints/multipliers.

    if ( continue_testing ) then
       do i = 1, m
          l = i
          aYi = abs(Y(i))
          select case( C_type(i) )
          case('LB')
             comp_meas = aYi * abs(C_RES_l(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aYi, one ) * max( abs(nlp%C_l(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             comp_meas = aYi * abs(C_RES_u(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aYi, one ) * max( abs(nlp%C_u(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             comp_meas1 = aYi * abs( C_RES_l(i) )
             comp_meas2 = aYi * abs( C_RES_u(i) )
             comp_meas = min( comp_meas1, comp_meas2 )
             comp_vl = max( comp_vl, comp_meas )
             if ( comp_meas1 < comp_meas2 ) then
                dummy =                                                        &
                  stop_c_rel * max( aYi, one ) * max( abs(nlp%C_l(i)), one )
             else
                dummy =                                                        &
                  stop_c_rel * max( aYi, one ) * max( abs(nlp%C_u(i)), one )
             end if
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ! relax
          case('FR')
             comp_vl = max( comp_vl, aYi )
             if ( aYi > stop_c_abs ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case default
             write( out, * ) ' **ERROR:s2qp_solve:get_opt C_TYPE = ?'
          end select
       end do
    end if

    do i = l+1, m
       select case( C_type(i) )
       case('LB')
          comp_meas = abs(Y(i)) * abs(C_RES_l(i))
          comp_vl = max( comp_vl, comp_meas )
       case('UB')
          comp_meas = abs(Y(i)) * abs(C_RES_u(i))
          comp_vl = max( comp_vl, comp_meas )
       case('RB')
          comp_meas = abs(Y(i)) * min( abs(C_RES_l(i)), abs(C_RES_u(i)) )
          comp_vl = max( comp_vl, comp_meas )
       case('EQ')
          ! relax
       case('FR')
          comp_vl = max( comp_vl, abs(Y(i)) )
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt C_TYPE = ?'
       end select
    end do

    ! Next, linear constraints/multipliers.

    l = 0

    if ( continue_testing ) then
       do i = 1, m_a
          l = i
          aYi = abs(Y_a(i))
          select case( A_type(i) )
          case('LB')
             comp_meas = aYi * abs(A_RES_l(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aYi, one ) * max( abs(nlp%A_l(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             comp_meas = aYi * abs(A_RES_u(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aYi, one ) * max( abs(nlp%A_u(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             comp_meas1 = aYi * abs(A_RES_l(i))
             comp_meas2 = aYi * abs(A_RES_u(i))
             comp_meas = min( comp_meas1, comp_meas2 )
             comp_vl = max( comp_vl, comp_meas )

             if ( comp_meas1 < comp_meas2 ) then
                dummy =                                                        &
                  stop_c_rel * max( aYi, one ) * max( abs(nlp%A_l(i)), one )
             else
                dummy =                                                        &
                  stop_c_rel * max( aYi, one ) * max( abs(nlp%A_u(i)), one )
             end if
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ! relax
          case('FR')
             comp_vl = max( comp_vl, aYi )
             if ( aYi > stop_c_abs ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case default
             write( out, * ) ' **ERROR:s2qp_solve:get_opt A_TYPE = ?'
          end select
       end do
    end if

    do i = l+1, m_a
       select case( A_type(i) )
       case('LB')
          comp_meas = abs(Y_a(i)) * abs(A_RES_l(i))
          comp_vl = max( comp_vl, comp_meas )
       case('UB')
          comp_meas = abs(Y_a(i)) * abs(A_RES_u(i))
          comp_vl = max( comp_vl, comp_meas )
       case('RB')
          comp_meas = abs(Y_a(i)) * min( abs(A_RES_l(i)), abs(A_RES_u(i)) )
          comp_vl = max( comp_vl, comp_meas )
       case('EQ')
          ! relax
       case('FR')
          comp_vl = max( comp_vl, abs(Y_a(i)) )
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt A_TYPE = ?'
       end select
    end do

    ! Finally, bound variables and reduced costs.

    l = 0

    if ( continue_testing ) then
       do i = 1, n
          l = i
          aZi = abs(Z(i))
          select case( X_type(i) )
          case('LB')
             comp_meas = aZi * abs(X_RES_l(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aZi, one ) * max( abs(nlp%X_l(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('UB')
             comp_meas = aZi * abs(X_RES_u(i))
             comp_vl = max( comp_vl, comp_meas )
             dummy = stop_c_rel * max( aZi, one ) * max( abs(nlp%X_u(i)), one )
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('RB')
             comp_meas1 = aZi * abs(X_RES_l(i))
             comp_meas2 = aZi * abs(X_RES_u(i))
             comp_meas = min( comp_meas1, comp_meas2 )
             comp_vl = max( comp_vl, comp_meas )
             if ( comp_meas1 < comp_meas2 ) then
                dummy =                                                        &
                  stop_c_rel * max( aZi, one ) * max( abs(nlp%X_l(i)), one )
             else
                dummy =                                                        &
                  stop_c_rel * max( aZi, one ) * max( abs(nlp%X_u(i)), one )
             end if
             if ( comp_meas > max( stop_c_abs, dummy ) ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case('EQ')
             ! relax
          case('FR')
             comp_vl = max( comp_vl, aZi )
             if ( aZi > stop_c_abs ) then
                optimal = .false.
                continue_testing = .false.
                exit
             end if
          case default
             write( out, * ) ' **ERROR:s2qp_solve:get_opt X_TYPE = ?'
          end select
       end do
    end if

    do i = l+1, n
       select case( X_type(i) )
       case('LB')
          comp_meas = abs(Z(i)) * abs(X_RES_l(i))
          comp_vl = max( comp_vl, comp_meas )
       case('UB')
          comp_meas = abs(Z(i)) * abs(X_RES_u(i))
          comp_vl = max( comp_vl, comp_meas )
       case('RB')
          comp_meas = abs(Z(i)) * min( abs(X_RES_l(i)), abs(X_RES_u(i)) )
          comp_vl = max( comp_vl, comp_meas )
       case('EQ')
          ! relax
       case('FR')
          comp_vl = max( comp_vl, abs(Z(i)) )
       case default
          write( out, * ) ' **ERROR:s2qp_solve:get_opt X_TYPE = ?'
       end select
    end do

 end if

 return

 END SUBROUTINE get_opt

 !************** G A L A H A D  cauchy_step  S U B R O U T I N E ***************

 SUBROUTINE cauchy_step( m, t_end, A_type, f, g_s, s_hs, s_norm, rho,       &
                         RES_l, RES_u, As, A_norms, lbreak, IBREAK, BREAKP, &
                         out, print_1line, print_debug, t_min, inform )
 !------------------------------------------------------------------------------
 ! Find the global minimizer of the function
 !
 !     P(x) = 1/2 x(T) H x + c(T) x + rho max( A_l - Ax, Ax - A_u, 0 )
 !
 ! along the arc x(t) = x + t s  for 0 <= t <= t_end <= infinity, and where x is
 ! a vector of n components, H is a symmetric matrix, and A is an m by n matrix.
 !------------------------------------------------------------------------------
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
 !    0     the minimizer given in t_min occurs between breakpoints after first.
 !    1     the minimizer given in t_min occurs at a breakpoint.
 !    2     the minimizer occurs at t_min = t_end.
 !    3     the minimizer occurs after last breakpoint, but before t_end.
 !   -1     the minimizer given in t_min occurs before first breakpoint, not 0.
 !   -2     the minimizer occurs at t = 0.
 !   -3     the function is unbounded from below. Ignore the value in t_min.
 !   -4     the value m is negative. Ignore the value in t_min.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 INTEGER, INTENT( IN ) :: m, out, lbreak
 INTEGER, INTENT( OUT ) :: inform
 LOGICAL, INTENT( IN ) :: print_1line, print_debug
 REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, s_norm, rho, t_end
 REAL ( KIND = wp ), INTENT( OUT ) :: t_min
 INTEGER, INTENT( INOUT ), DIMENSION( lbreak ) :: IBREAK
 REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) ::  RES_l, RES_u, A_norms, AS
 REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lbreak ) :: BREAKP
 CHARACTER ( Len = 2 ), INTENT( IN ), DIMENSION( m ) :: A_type

 !------------------------------------------------------------------------------
 ! L o c a l   v a r i a b l e s
 !------------------------------------------------------------------------------

 INTEGER :: i, j, nbreak, inheap, iter, ibreakp
!INTEGER :: nbreak_total
 REAL ( KIND = wp ) :: Asi, too_small, res, P_min, PatX, cosine, tiny_cosine
 REAL ( KIND = wp ) :: breakp_max, t_break, t_star, t_old, dt
 REAL ( KIND = wp ) :: val, exact_val, curv, infeas
 REAL ( KIND = wp ) :: slope, slope_infeas, slope_in, slope_old, exact_slope
 LOGICAL :: beyond_first_breakpoint, check_t_end, recover

 !------------------------------------------------------------------------------

 IF ( m < 0 ) THEN
    t_min  = zero
    P_min  = f
    inform = -4
    RETURN
 END IF

 ! Find distance to constraint boundaries, and the slope of penalty functon P.
 ! ===========================================================================

 tiny_cosine = epsmch ** 0.75_wp   ! What will be considered 0 for orthogonality

 too_small = sqrteps / max( one, t_end * rho )
 too_small = min( too_small, sqrteps )
 too_small = max( too_small, teneps )

 nbreak       = 0
 infeas       = zero
 slope_infeas = zero
 breakp_max   = zero

 do i = 1, m

    select case ( A_type(i) )

    case ( 'EQ' ) ! equality constraint

       res = RES_l(i)
       infeas = infeas + ABS( res )

       Asi = As(i)
       IF ( ABS( Asi ) <= too_small ) CYCLE

       IF ( res < zero ) THEN
          slope_infeas = slope_infeas - Asi
       ELSEIF ( res > zero ) THEN
          slope_infeas = slope_infeas + Asi
       ELSE
          IF ( Asi > zero ) THEN
             slope_infeas = slope_infeas + Asi
          ELSE
             slope_infeas = slope_infeas - Asi
          END IF
          CYCLE
       END IF

       ! Find if the step will change the status of the constraint.

       t_break = - res / Asi

       IF ( min(Asi,res) > zero .OR. max(Asi,res) < zero ) CYCLE

       cosine = ABS( Asi ) / ( s_norm * A_norms(i) )
       IF ( print_debug .AND. cosine < tiny_cosine ) THEN
         WRITE( out, "( 'rconst ', i5, 4ES12.4 )" ) i, res, Asi, t_break, cosine
       END IF

       ! Find the breakpoint

       IF ( print_debug ) THEN
         WRITE(out, "(' const EQ ', i5, 4ES18.10 )")i, res, Asi, t_break, cosine
       END IF

       IF ( t_break < t_end  ) THEN
          nbreak           = nbreak + 1
          IBREAK( nbreak ) = - i
          BREAKP( nbreak ) = t_break
          breakp_max       = MAX( breakp_max, t_break )
       END IF

    case ( 'LB' ) ! constraint with only lower bounds

       res = RES_l(i)
       IF ( res < zero ) infeas = infeas - res

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) CYCLE

       IF ( res < zero ) THEN
          slope_infeas = slope_infeas - Asi
       ELSEIF ( res == zero ) THEN
          IF ( Asi < zero ) THEN
             slope_infeas = slope_infeas - Asi
          END IF
          CYCLE
       END IF

       ! Find if the step will change the status of the constraint

       t_break = - res / Asi

       IF ( min(Asi,res) > zero .OR. max(Asi,res) < zero ) CYCLE

       cosine = ABS( Asi ) / ( s_norm * A_norms(i) )
       IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )i, res, Asi, t_break, cosine
       END IF

       ! Find the breakpoint

       IF ( print_debug ) THEN
          WRITE(out,"( ' const LB ', i5, 4ES18.10 )")i,res, Asi, t_break, cosine
       END IF

       IF ( t_break < t_end ) THEN
          nbreak           = nbreak + 1
          IBREAK( nbreak ) = i
          BREAKP( nbreak ) = t_break
          breakp_max       = MAX( breakp_max, t_break )
       END IF

    case ( 'UB' ) ! constraint with only upper bounds

       res = RES_u(i)
       IF ( res < zero ) infeas = infeas - res

       Asi = - As(i) ! for convenience

       IF ( ABS( Asi ) < too_small ) CYCLE

       IF ( res < zero ) THEN
          slope_infeas = slope_infeas - Asi
       ElSEIF ( res == 0 ) THEN
          IF ( Asi < zero ) THEN
             slope_infeas = slope_infeas - Asi
          END IF
          CYCLE
       END IF

       ! Find if the step will change the status of the constraint

       t_break = - res / Asi

       IF ( min(Asi,res) > zero .OR. max(Asi,res) < zero ) CYCLE

       cosine = ABS( Asi ) / ( s_norm * A_norms(i) )
       IF ( print_debug .AND. cosine < tiny_cosine ) THEN
          WRITE( out, "( 'rconst ', i5, 4ES12.4 )" )i, res, Asi, t_break, cosine
       END IF

       !  Find the breakpoint

       IF ( print_debug ) THEN
          WRITE(out,"( ' const UB ', i5, 4ES18.10 )")i,res, Asi, t_break, cosine
       END IF

       IF ( t_break < t_end ) THEN
          nbreak           = nbreak + 1
          IBREAK( nbreak ) = i
          BREAKP( nbreak ) = t_break
          breakp_max       = MAX( breakp_max, t_break )
       END IF

    case ( 'RB' ) ! constraint with BOTH upper and lower bounds (not equality)

       DO j = 1,2

          IF ( j == 1 ) THEN
             res = RES_l(i)
             Asi = As(i)
          ELSE
             res = RES_u(i)
             Asi = - As(i)  ! again, for convenience.
          END IF

          IF ( res < zero ) infeas = infeas - res

          IF ( ABS( Asi ) < too_small ) CYCLE

          IF ( res < zero ) THEN
             slope_infeas = slope_infeas - Asi
          ELSEIF ( res == zero ) THEN
             IF ( Asi < zero ) THEN
                slope_infeas = slope_infeas - Asi
             END IF
             CYCLE
          END IF

          ! Find if the step will change the status of the constraint

          t_break = - res / Asi

          IF ( min(Asi,res) > zero .OR. max(Asi,res) < zero ) CYCLE

          cosine = ABS( Asi ) / ( s_norm * A_norms(i) )
          IF ( print_debug .AND. cosine < tiny_cosine ) THEN
             WRITE(out,"( 'rconst ', i5, 4ES12.4 )")i,res, Asi, t_break, cosine
          END IF

          ! Find the breakpoint

          IF ( print_debug ) THEN
             WRITE(out,"( ' const RB ', i5, 4ES18.10)")i,res,Asi,t_break,cosine
          END IF

          IF ( t_break < t_end ) THEN
             nbreak = nbreak + 1
             IBREAK( nbreak ) =   i
             BREAKP( nbreak ) = t_break
             breakp_max = MAX( breakp_max, t_break )
          END IF

       END DO

    case ( 'FR' ) ! constraints which are actually free.

       ! Relax....do nothing.

    case default

       write(*,*) '**ERROR:s2qp_solve:cauchy_step A_type=?'

    end select

 END DO

!nbreak_total = nbreak

 ! Record the initial function value, slope, and curvature of P.

 val   = f + rho * infeas
 PatX  = val
 slope = g_s + rho * slope_infeas
 curv  = s_hs

 ! Give an intial gradient value coming IN (arbitrarily negative is fine).

 slope_in = - one

 IF ( print_debug ) THEN
    CALL cauchy_val_and_slope( m, A_type, f, g_s, s_hs, rho, RES_l, RES_u, &
                               As, too_small, zero, exact_val, exact_slope )
    write( out, 2010 ) '  val', val, exact_val
    write( out, 2010 ) 'slope', slope, exact_slope
 END IF

 ! Order breakpoints in increasing size using a heapsort.  Build the heap.

 CALL SORT_heapsort_build( nbreak, BREAKP, inheap, INDA = IBREAK )

 ! =====================================================================
 ! Start the main loop to find the global minimizer of the piecewise
 ! quadratic function for 0 <= t <= t_end. Consider the problem over
 ! successive taylor polynomial quadratic pieces.
 ! =====================================================================

 iter    = 0
 t_break = zero
 t_min   = zero
 P_min   = val
 inform  = -2
 recover = .FALSE.
 beyond_first_breakpoint = .FALSE.

 DO

    ! --------------------------------------------------------------------------
    ! The piecewise quadratic within the current interval is given by the Taylor
    ! function T(t) = val + slope * (t-tbreak) + 0.5 * curv * (t-tbreak)**2.
    ! Note: T(tbreak = P(tbreak), T'(tbreak) = P'(tbreak), T'' = P''.
    ! --------------------------------------------------------------------------

    ! Print details of the piecewise quadratic in the current interval

    IF ( print_1line .OR. print_debug ) THEN
       WRITE( out, 2000 )
       WRITE( out,"(4X, I7, ES12.4, 3ES12.4)" ) iter, t_break, val, slope, curv
    END IF

    ! If the gradient of the unvariate function increases.

    if ( curv >= zero ) then
       if ( slope >= zero ) then
          if ( .not. beyond_first_breakpoint ) then
             t_min   = zero
             inform  = -2
             if ( print_debug ) then
                write(out, "(' Global minimum occurs at t=0.')")
             end if
          else
             t_min   = t_break
             inform  = 1
             if ( print_debug ) then
                write(out, "(' Global minimum occurs at current break point.')")
             end if
          end if
          check_t_end = .false.
          exit
       end if
    else
       if ( slope_in < zero .and. slope > zero ) then
          if ( .not. beyond_first_breakpoint ) then
             t_min  = zero
             P_min  = val
             inform = -2
          else
             if ( val < P_min ) then
                t_min  = t_break
                P_min  = val
                inform = 1
             end if
          end if
          if ( print_debug ) then
             write( out, 2060 )
             write( out, "('(P_min,P) = ', 2ES18.12)" ) P_min, val
          end if
       end if
    end if

    !  Find the next breakpoint

    iter = iter + 1

    t_old = t_break

    if ( nbreak > 0 ) then
       t_break = BREAKP(1)
       CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, INDA=IBREAK )
    else
       if ( curv <= zero ) then

          check_t_end = .true.

          if ( print_debug ) then
             write(out,*) 'All break points passed and not positive definite.'
          end if

       else

          t_star = t_break - slope / curv

          if ( print_debug ) then
             write(out,*) 'All break points passed and positive definite:'
             write( out, 2050 ) t_old, t_end, t_star
          end if

          if ( t_star < t_end ) then

             t_min  = t_star

             ! Calculate the function value and slope of P at t_min.

             dt    = t_min - t_old
             val   = val + dt  * ( slope + half * dt * curv )
             slope = slope + curv * dt

             inform = 3
             check_t_end = .false.

             if ( print_debug ) then
                write( out, 2062 )
                write( out, 2000 )
             end if
             if ( print_1line ) then
                write(out,"(4X,I7, ES12.4, 3ES12.4)")iter,t_min,val, slope, curv
             end if

          else

             check_t_end = .true.
             if ( print_debug ) then
                write( out, 2101 ) t_star, t_end
             end if

          end if
       end if

       exit

    end if

    ! If curv > zero, then we must have slope < zero.  Compute line minimum.

    if ( curv > zero ) then

       t_star = t_old - slope / curv

       ! If minimum occurs before breakpoint, then it is the global minimizer.

       if ( t_star < t_break ) then

          if ( print_debug ) then
             write( out, 2050 ) t_old, t_break, t_star
             write( out, 2062 )
          end if

          t_min = t_star

          ! Calculate the function value and slope of P at t_min.

          dt    = t_min - t_old
          val   = val + dt  * ( slope + half * dt * curv )
          slope = slope + curv * dt

          if ( print_debug ) write( out, 2000 )
          if ( print_1line ) then
             write(out,"(4X,I7, ES12.4, 3ES12.4)") iter, t_min, val, slope, curv
          end if
          if ( print_debug ) then
             exact_val = cauchy_val( m, A_type, f, g_s, s_hs, rho, &
                                     RES_l, RES_u, As, too_small, t_min )
             write( out, 2010 ) '  val', val, exact_val
          end if

          if ( beyond_first_breakpoint ) then
             inform = 0
          else
             inform = -1
          end if

          check_t_end = .false.
          exit

       else  ! minimum is past or equal to next break point.

          if ( print_debug ) then
             write( out, 2050 ) t_old, t_break, t_star
             write( out, "(' Proceeding on ...' )" )
          end if

       end if

    else  ! curv <= zero. Take full step to next break point.

       if ( print_debug ) write( out, 2040 ) t_old, t_break

    end if

    ! Record the slope coming INTO new breakpoint breakpoint.

    dt       = t_break - t_old
    slope_in = slope + curv * dt

    ! Update the univariate function and slope values

    slope_old = slope
    slope     = slope_in

    DO

       ibreakp = IBREAK(nbreak)

       ! Update the slope

       IF ( ibreakp < 0 ) THEN  ! account for equality constraint.
          ibreakp = - ibreakp
          slope = slope + rho * ABS( As(ibreakp) )
       END IF

       IF (print_debug) WRITE( out, 2020 ) 'C', IBREAK(nbreak), BREAKP(nbreak)

       slope = slope + rho * ABS( As(ibreakp) )

       ! If the last breakpoint has been passed, exit

       nbreak = nbreak - 1
       IF( nbreak == 0 ) EXIT

       ! Determine if other terms become active at the breakpoint

       IF ( BREAKP(1) > t_break ) EXIT
       CALL SORT_heapsort_smallest(nbreak, BREAKP(:nbreak), inheap, INDA=IBREAK)

    END DO

    ! Compute the function value at the new point.

    val = val + dt * ( slope_old + half * dt * curv )

    IF ( print_debug ) THEN
       CALL cauchy_val_and_slope( m, A_type, f, g_s, s_hs, rho, RES_l, RES_u, &
                                As, too_small, t_break, exact_val, exact_slope )
       WRITE( out, 2010 ) '  val', val, exact_val
       WRITE( out, 2010 ) 'slope', slope, exact_slope
    END IF

    beyond_first_breakpoint = .TRUE.

 ! ================
 ! End of main loop
 ! ================

 END DO

 ! See if we need to also check t_end,

 if ( check_t_end ) then
    val = cauchy_val( m, A_type, f, g_s, s_hs, rho, &
                      RES_l, RES_u, As, too_small, t_end )
    if ( print_debug ) then
       write( out, 2070 ) P_min, val
    end if
    if ( val < P_min ) then
       t_min  = t_end
       P_min  = val
       inform = 2
       if ( print_debug ) then
          write( out, 2080 )
       end if
    else
       if ( print_debug ) then
          write( out, 2090 ) t_min, P_min
       end if
    end if
 end if

 ! Compute final value at the global minimizer.

 val = cauchy_val( m, A_type, f, g_s, s_hs, rho, &
                   RES_l, RES_u, As, too_small, t_min )

 P_min = val

 if ( print_debug ) then
    write( out, 2091 ) t_min, P_min, PatX
 end if

 ! Check to ensure that rounding has not caused objective to increase.

 recover = f + rho * infeas + epsmch ** 0.33_wp <= P_min

 if ( .not. recover ) return

 write(*,*) 'get_cauchy : RECOVERY entered (not yet implemented)'
 inform = -4
 return

!   IF ( print_detail ) WRITE( out,                                          &
!        "( ' *** predicted vs actual function values =', /, 2ES22.14,       &
!        &        /, ' .... being more careful ... ' )" )                      &
!        f + rho * infeas, val

!  !  =========================================================================
!  !  This part of the code is to cope with the possibility that rounding errors
!  !  have so dominated the search that a descent point has not been found. A
!  !  more cautious search will be performed.
!  !  =========================================================================

!   iter = 0 ; t_break = zero ; t_min = zero

!   !  Compute the initial function, slope and curvature

!   call cauchy_val_and_slope( m, A_type, f, g_s, s_hs, rho, RES_l, RES_u, &
!                              As, too_small, zero, val, slope )
!   curv = s_hs

!   !  Record the function value and gradient at (just on the other side of)
!   !  the initial point

!   fun = val ; gradient = slope ; val_old = val

!   nbreak = nbreak_total
!   cluster_start = 1
!   cluster_end = 0
!   cluster = '      0      0'

!   !  =========================================================================
!   !  Start the main recovery loop to find the first local minimizer of the
!   !  piecewise quadratic function. Consider the problem over successive pieces
!   !  =========================================================================

!   beyond_first_breakpoint = .FALSE.
!   inform = - 1
!   DO

!      !  ---------------------------------------------------------------
!      !  The piecewise quadratic function within the current interval is
!      !    val + slope * t + 0.5 * curv * t**2
!      !  ---------------------------------------------------------------

!      !  Print details of the piecewise quadratic in the next interval

!      iter = iter + 1
!      IF ( ( print_1line .AND. cluster_end == 0 ) .OR. print_detail )        &
!           WRITE( out, 2000 )
!      IF ( print_1line ) WRITE( out, "( 3X, I7, ES12.4, A14, 3ES12.4 )" )    &
!           iter, t_break, cluster, fun, gradient, curv

!      !  If the gradient of the unvariate function increases, exit

!      IF ( gradient > gzero ) THEN
!         if ( curv >= 0 ) then
!            IF ( inform == 0 ) t_min = t_break
!            EXIT
!         else
!            if ( gradient_in < -gzero ) then

!            end if
!         end if
!      END IF


!      !  If the gradient of the univariate function is small and its curvature
!      !  is positive, exit

!      IF ( ABS( gradient ) <= gzero ) THEN
!         IF ( curv > - hzero ) THEN
!            IF ( inform == 0 ) t_min = t_break
!            EXIT
!         END IF
!      END IF

!      !  Find the next breakpoint

!      t_old = t_break
!      IF ( nbreak > 0 ) THEN
!         t_break = BREAKP( nbreak )
!         cluster_end = cluster_end + 1
!         cluster_start = cluster_end
!      ELSE
!         t_break = one !biginf
!      END IF

!      !  If the gradient of the univariate function is nonzero and its
!      !  curvature is positive, compute the line minimum

!      IF ( curv > zero ) THEN
!         IF ( print_detail ) WRITE( out, "( ' slope, curv ', 2ES12.4 )" )     &
!              slope,  curv
!         t_star = - slope / curv

!         !  If the line minimum occurs before the breakpoint, the line minimum
!         !   gives the required minimizer. Exit

!         IF ( nbreak == 0 .OR. t_star < t_break ) THEN
!            t_min = t_star
!            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star

!            !  Calculate the function value for the piecewise quadratic

!            val = cauchy_val( m, A_type, f, g_s, s_hs, rho, &
!                              RES_l, RES_u, As, too_small, t_break )


!            !  If the function value has risen in the current interval, search
!            !   the interval for a better value, and exit

!            IF ( val_old < val ) THEN
! !!$              CALL QPA_linesearch_interval( dims, n, m,                   &
! !!$                                            f, g_s, s_hs, rho, rho_b,     &
! !!$                                            X, X_l, X_u, RES_l,           &
! !!$                                            RES_u, S, Asi,                &
! !!$                                            t_old, val_old, t_min, val,   &
! !!$                                            too_small, out, print_detail )
!            END IF

!            IF ( print_detail ) WRITE( out, 2000 )
!            IF ( print_1line ) WRITE( out, &
!                 "( 3X, I7, ES12.4, A14, 3ES12.4 )" ) &
!                 iter, t_min, '      -      -', fun, zero, curv
!            IF ( print_debug ) THEN
!               exact_val = cauchy_val( m, A_type, f, g_s, s_hs, rho, &
!                                       RES_l, RES_u, Asi, too_small, t_min )

!               WRITE( out, 2010 ) '  val', fun, exact_val
!            END IF
!            IF ( beyond_first_breakpoint ) inform = 0
!            EXIT
!         ELSE
!            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star
!         END IF
!      ELSE

!         IF ( print_detail ) WRITE( out, 2040 ) t_old, t_break

!         !  Exit if the function is unbounded from below

!         IF ( nbreak == 0 ) THEN
!            t_min = one !biginf
!            IF ( print_detail ) WRITE( out, 2000 )
!            IF ( print_1line ) WRITE( out, &
!                 "( 3X, I7, 5X, 'inf', 22X, '-inf')" ) iter
!            inform = - 2
!            EXIT
!         END IF

!      END IF

!      !  Update the univariate function and slope values

!      slope_old = slope

!      !  Record the new breakpoint and the amount by which other breakpoints
!      !  are allowed to vary from this one and still be considered to be
!      !  within the same cluster

!      pert_val = MAX( teneps, 0.001_wp * t_break )
!      pert_eps = epsmch

!      feasep = t_break + pert_val
!      t_pert = pert_val + pert_eps
!      t_break = feasep

!      IF ( feasep < breakp_max ) THEN
!         DO

!            !  Update the slope

!            IF ( ibreakp <= m ) THEN
!               IF ( print_detail ) WRITE( out, 2020 )                         &
!                    'C', IBREAK( nbreak ), BREAKP( nbreak )
!            ELSE
!               IF ( print_detail ) WRITE( out, 2020 )                         &
!                    'B', IBREAK( nbreak ) - m, BREAKP( nbreak )
!            END IF

!            !  If the last breakpoint has been passed, exit

!            nbreak = nbreak - 1
!            IF( nbreak == 0 ) EXIT

!            !  Determine if other terms become active at the breakpoint

!            IF ( BREAKP( nbreak ) >= feasep ) EXIT
!            cluster_end = cluster_end + 1
!         END DO

!      ELSE

!         !  Special case: all the remaining breakpoints are reached

!         nbreak = 0
!         cluster_end = nbreak_total
!      END IF

!      !  Compute the function value and gradient at (just on the other side of)
!      !  the breakpoint
!      call cauchy_val_and_slope( m, A_type, f, g_s, s_hs, rho, RES_l, RES_u, &
!                                 As, too_small, t_break, val, slope )

!      gradient = slope + t_break * curv

!      IF ( print_debug )                                                     &
!           WRITE( out, "( ' val_old, val ', 2ES22.14 )" ) val_old, val
!      IF ( val_old < val ) THEN
!         t_min = t_break
! !!$          CALL QPA_linesearch_interval( dims, n, m,                       &
! !!$                                        f, g_s, s_hs, rho, rho_b, X,      &
! !!$                                        X_l, X_u, RES_l, RES_u, S, Asi,   &
! !!$                                        t_old, val_old, t_min, val,       &
! !!$                                        too_small, out, print_detail )
!         EXIT
!      END IF
!      val_old = val

!      beyond_first_breakpoint = .TRUE.



!      !  =========================
!      !  End of main recovery loop
!      !  =========================

!   END DO

!  RETURN

  ! format statements

2000 FORMAT(4x, '**  iter breakpoint      val       slope        curv ')
2010 FORMAT(1x, A5, '(est,true) = ', 2ES22.14 )
2020 FORMAT(1x, 'breakpoint for ', A1, '-term ', I7, ' reached, step = ',ES12.4)
2040 FORMAT(1x, 'Interval = [', ES15.8, ',', ES15.8, ']' )
2050 FORMAT(4x, 'Interval = [', ES15.8, ',', ES15.8, &
                '], stationary point = ', ES15.8 )
2060 FORMAT(1x, 'Indefinite:local minimum found. Documenting and moving on.',/)
2062 FORMAT(4x, 'Unique minimizer must occur at stationary point.' )
2070 FORMAT(1x, 'Checking end point t = t_end : ', /, &
            1x, '   ( P(x+tmin*s), P(x+tend*s) ) = (', ES12.4, ',', ES12.4, ')')
2080 FORMAT(1x, '   The value t_end is the global minimizer.')
2090 FORMAT(1x, '   The value t_end is NOT the global minimizer.', /, &
            1x, '   Minimizer is ( t_min,P(x+tmin*s) )= (', ES14.7, ',', &
            ES16.9, ').')
2091 FORMAT(1x, 'GLOBAL minimizer of P is t_min = ', ES16.9, '.', / &
            1x, 'GLOBAL minimum P( x + tmin*s ) = ', ES16.9, '.', / &
            1x, 'The value at t = 0 was P( x )  = ', ES16.9, '.' )
2101 FORMAT(4x, 'Stationary point t = ', ES12.4, ' is past t_end.', /, &
            4x, 'Global minimizer must occur at t_end = ', ES12.4, '.' )

  END SUBROUTINE cauchy_step


 !************* G A L A H A D  cauchy_val  S U B R O U T I N E **************

 FUNCTION cauchy_val( m, A_type, f, g_s, s_hs, rho,  &
                      RES_l, RES_u, As, too_small, t )
 !------------------------------------------------------------------------------
 ! Compute the value of the quadratic penalty function given by
 !     P(x) = 1/2 x'Hx + c'x + rho * max( Al-Ax, Ax-Au, 0 )
 ! at the value x(t) := x + ts.  The value of x and s are assumed fixed.
 !
 ! m          number of linear constraints Ax.
 ! A_type     constraint types, i.e., RB, UB, LB, EQ, or FR.
 ! f          holds the value x'c + 1/2 x'Hx.
 ! g_s        holds the value s'(c+Hx).
 ! s_hs       holds the value s'Hs for
 ! rho        current value of the penalty paramter.
 ! RES_l      holds residual values Ax-Al for all lower bound constraints AND
 !            equality constraints.
 ! RES_u      holds residual values for Au-Ax for all upper bound constraints.
 ! As         holds the value of As.
 ! t          will evaluate the penalty function at x(t) := x + ts.
 ! too_small  value for deciding if As(i) is small enough to be considered zero.
 !
 ! Note: this subroutine is used by subroutine cauchy_step.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 REAL ( KIND = wp ) cauchy_val
 INTEGER, INTENT( IN ) :: m
 REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho, t, too_small
 REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: As, RES_l, RES_u
 CHARACTER ( Len = 2 ), INTENT( IN ), DIMENSION( m ) :: A_type

 !------------------------------------------------------------------------------
 ! L o c a l   v a r i a b l e s
 !------------------------------------------------------------------------------

 INTEGER :: i
 REAL ( KIND = wp ) :: Asi, infeas

 !------------------------------------------------------------------------------

 infeas = zero

 DO i = 1, m

    SELECT CASE( A_type(i) )

    CASE ('EQ') ! equality constraint,

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas + ABS( RES_l(i) )
       ELSE
          infeas = infeas + ABS( RES_l(i) + t*Asi )
       END IF

    CASE ('LB') ! constraint with lower bound only.

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_l(i), zero )
       ELSE
          infeas = infeas - MIN( RES_l(i) + t*Asi, zero )
       END IF

    CASE ('UB') ! constraint with upper bound only.

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_u(i), zero )
       ELSE
          infeas = infeas - MIN( RES_u(i) - t*Asi, zero )
       END IF

    CASE ('RB') ! constraint with both an upper and lower bound.  Not equality.

       Asi = As(i)

       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_l(i), zero )
          infeas = infeas - MIN( RES_u(i), zero )
       ELSE
          infeas = infeas - MIN( RES_l(i) + t*Asi, zero )
          infeas = infeas - MIN( RES_u(i) - t*Asi, zero )
       END IF

    CASE ('FR') ! constraint has no lower or upper bound, i.e., free.

       ! Relax

    CASE default

       WRITE(*,*) '**ERROR:s2qp_solve:cauchy_val: A_type=?'

    END SELECT

 END DO

 cauchy_val = f + t * ( g_s + half * t * s_hs ) + rho * infeas

 RETURN

 END FUNCTION cauchy_val

 !*********** G A L A H A D  cauchy_val_and_slope  S U B R O U T I N E *********

 SUBROUTINE cauchy_val_and_slope( m, A_type, f, g_s, s_hs, rho, RES_l, RES_u, &
                                  As, too_small, t, val, slope )
 !------------------------------------------------------------------------------
 ! Compute the value and slope of the quadratic penalty function given by
 !     1/2 x'Hx + c'x + rho * max( Al-Ax, Ax-Au, 0 )
 ! at the value x(t) := x + ts.  The value of x and s are assumed fixed.
 !
 ! ARGUMENTS:
 !
 ! m          number of linear constraints Ax.
 ! A_type     constraint types, i.e., RB, UB, LB, EQ, or FR.
 ! f          holds the value x'c + 1/2 x'Hx.
 ! g_s        holds the value s'(c+Hx).
 ! s_hs       holds the value s'Hs for
 ! rho        current value of the penalty paramter.
 ! RES_l      holds residual values Ax-Al for all lower bound constraints AND
 !            equality constraints.
 ! RES_u      holds residual values for Au-Ax for all upper bound constraints.
 ! As         holds the value of As.
 ! too_small  value for deciding if As(i) is small enough to be considered zero.
 ! t          will evaluate the penalty function at x(t) := x + ts.
 ! val        value of the penalty function at x + ts.
 ! slope      slope of the penalty function at x + ts.
 !
 ! Note: this subroutine is used by subroutine cauchy_step.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 INTEGER, INTENT( IN ) :: m
 CHARACTER ( Len = 2 ), DIMENSION( m ), INTENT( IN ) :: A_type
 REAL ( KIND = wp ), INTENT( IN ) :: f, g_s, s_hs, rho, t, too_small
 REAL ( KIND = wp ), INTENT( OUT ) :: val, slope
 REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: As, RES_l, RES_u

 !------------------------------------------------------------------------------
 ! L o c a l   v a r i a b l e s
 !------------------------------------------------------------------------------

 INTEGER :: i
 REAL ( KIND = wp ) :: Asi, infeas, dum_real

 !------------------------------------------------------------------------------

 infeas = zero
 slope  = zero

 DO i = 1, m

    SELECT CASE ( A_type(i) )

    CASE ('EQ') ! equalities

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas + ABS( RES_l(i) )
       ELSE
          infeas = infeas + ABS( RES_l(i) + t*Asi )
          dum_real = RES_l(i) + t*Asi
          IF ( dum_real < zero ) THEN
             slope = slope - Asi
          ELSEIF ( dum_real > zero ) THEN
             slope = slope + Asi
          ELSE
             slope = slope + abs( Asi )
          END IF
       END IF

    CASE ('LB') ! lower bounds

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_l(i), zero )
       ELSE
          infeas   = infeas - MIN( RES_l(i) + t*Asi, zero )
          dum_real =  RES_l(i) + t*Asi
          IF ( dum_real < zero ) THEN
             slope = slope - Asi
          ELSEIF ( dum_real == zero .AND. Asi < zero ) THEN
             slope = slope - Asi
          END IF
       END IF

    CASE ('UB') ! constraints with upper bounds

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_u(i), zero )
       ELSE
          infeas = infeas - MIN( RES_u(i) - t*Asi, zero )
          dum_real = RES_u(i) - t*Asi
          IF ( dum_real < zero ) THEN
             slope = slope + Asi
          ELSEIF ( dum_real == zero .AND. Asi > zero ) THEN
             slope = slope + Asi
          END IF
       END IF

    CASE ('RB') ! constraints bounded below and above.  Not equality.

       Asi = As(i)
       IF ( ABS( Asi ) < too_small ) THEN
          infeas = infeas - MIN( RES_l(i), zero )
          infeas = infeas - MIN( RES_u(i), zero )
       ELSE
          infeas = infeas - MIN( RES_l(i) + t*Asi, zero )
          infeas = infeas - MIN( RES_u(i) - t*Asi, zero )
          dum_real = RES_l(i) + t*Asi
          IF ( dum_real < zero ) THEN
             slope = slope - Asi
          ELSEIF ( dum_real == zero .AND. Asi < zero ) THEN
             slope = slope - Asi
          ELSE
             dum_real = RES_u(i) - t*Asi
             IF ( dum_real < zero ) THEN
                slope = slope + Asi
             ELSEIF ( dum_real == zero .AND. Asi > zero ) THEN
                slope = slope + Asi
             END IF
          END IF
       END IF

    CASE ('FR') ! Free constraint

       ! Relax

    CASE default

       write(*,*) '**Error:s2qp_solve:cauchy_val_and_slope A_type=?.'

    END SELECT

 END DO

 val = f + t * ( g_s + half * t * s_hs ) + rho * infeas
 slope = ( g_s + t * s_hs ) + rho * slope

 RETURN

 END SUBROUTINE cauchy_val_and_slope

 !**************** G A L A H A D  L1_viol  S U B R O U T I N E *****************

 SUBROUTINE L1_viol( lv, res_vl, res_vu, v_type, viol, error, tiny, &
                     sat, vl_l, vl_u, num_sat, num_vl_l, num_vl_u )
 !------------------------------------------------------------------------------
 !
 ! Computes the "ell one" violation of the vector v, but does so by computing
 ! with the residual vectors that may be computed from a previous call to the
 ! external subroutine get_residuals. Specifically,
 !
 !        viol = sum( abs( min( v - v_l, v_u - v, 0 ) ) )
 !             = sum( abs( min( res_vl, res_vu, 0 ) ) )
 !
 ! Input:
 !
 !    lv       number of components of res_vl and res_vu to use in computing the
 !             violation.  Thus we have the restrictions lv <= length(v_type)
 !             lv <= length(res_vl) and lv <= length(res_vu).  Generally, the
 !             value for lv should be length(vl_l) = length(vl_u).
 !    v_type   character array of length 2 that holds the type of constraint.
 !             Allowed values are: RB, EQ, LB, UB, or FR.
 !    res_vl   the lower residual vector as described in subroutine
 !             get_residuals; only need to be computed for RB, EQ, and LB.
 !    res_vu   the upper residual vector as described in subroutine
 !             get_residuals; only need to be computed for RB and UB.
 !    viol     the "ell one" violation on output.
 !    error    device for printing error messages.
 !    tiny     (optional) error allowed in determining violated components.
 !
 ! On Exit (if parameter "tiny" is present):
 !
 !    num_sat    number of feasible components of v.
 !    num_vl_l   number of components of v that violate the lower bound.
 !    num_vl_u   number of components of v that violate the upper bound.
 !    sat        indices of the feasible components of v.
 !    vl_l       indices of the components of v that violate the lower bound.
 !    vl_u       indices of the components of v that violate the upper bound.
 !
 ! Note1: Although the parameter "tiny" is used for determining which variables
 !        will be considered violated, the value in output parameter "viol" will
 !        always contain the exact value of the violation.  Thus, if there is
 !        only a single "tiny" violation, then viol will be nonzero, but all
 !        components will be considered satisfied.
 ! Note2: Suppose that the ith constraint is an equality, i.e. v_type(i)='EQ'.
 !        Then: if res_vl(i) < 0, the ith constraint will be considered as
 !        violating its lower bound, and if res_vl(i) > 0, the ith constraint
 !        will be considered as violating its upper bound.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 integer, intent( in ) :: lv, error
 real( kind = wp ), intent( out ) :: viol
 real( kind = wp ), intent( in ), dimension( : ) :: res_vl, res_vu
 character ( len = 2 ), intent( in ), dimension( : ) :: v_type
 real( kind = wp ), intent( in ), optional :: tiny
 integer, intent( inout ), dimension( : ), optional :: sat, vl_l, vl_u
 integer, intent( inout ), optional :: num_sat, num_vl_l, num_vl_u

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: i
 real( kind = wp ) :: ires

 !------------------------------------------------------------------------------

 ! Compute the violation

 viol = zero

 if ( present(tiny) ) then

    num_sat  = 0
    num_vl_l = 0
    num_vl_u = 0

    do i = 1, lv
       select case( v_type( i ) )
       case('LB')
          ires = res_vl(i)
          viol = viol + min( zero, ires )
          if ( ires < - tiny ) then
             num_vl_l = num_vl_l + 1
             vl_l( num_vl_l ) = i
          else
             num_sat = num_sat + 1
             sat( num_sat ) = i
          end if
       case('UB')
          ires = res_vu(i)
          viol = viol + min( zero, ires )
          if ( ires < -tiny ) then
             num_vl_u = num_vl_u + 1
             vl_u( num_vl_u ) = i
          else
             num_sat = num_sat + 1
             sat( num_sat ) = i
          end if
       case('RB')
          ires = res_vl(i)
          viol = viol + min( zero, ires )
          if ( ires < -tiny ) then
             num_vl_l = num_vl_l + 1
             vl_l( num_vl_l ) = i
          else
             ires = res_vu(i)
             viol = viol + min( zero, ires )
             if ( ires < -tiny ) then
                num_vl_u = num_vl_u + 1
                vl_u( num_vl_u ) = i
             else
                num_sat = num_sat + 1
                sat( num_sat ) = i
             end if
          end if
       case('EQ')
          ires = res_vl(i)
          viol = viol - abs(ires)
          if ( ires > tiny ) then
             num_vl_u = num_vl_u + 1
             vl_u( num_vl_u ) = i
          elseif ( ires < -tiny ) then
             num_vl_l = num_vl_l + 1
             vl_l( num_vl_l ) = i
          else
             num_sat = num_sat + 1
             sat( num_sat ) = i
          end if
       case('FR')
          num_sat = num_sat + 1
          sat( num_sat ) = i
       case default
          write( error, * ) ' **ERROR:s2qp_solve:l1_viol v_type = ?'
       end select
    end do

 else

    do i = 1, lv
       select case( v_type( i ) )
       case('LB')
          viol = viol + min( res_vl(i), zero )
       case('UB')
          viol = viol + min( res_vu(i), zero )
       case('RB')
          viol = viol + min( res_vl(i), zero )
          viol = viol + min( res_vu(i), zero )
       case('EQ')
          viol = viol - abs(res_vl(i))
       case('FR')
          ! relax
       case default
          write( error, * ) ' **ERROR:s2qp_solve:l1_viol v_type = ?'
       end select
    end do

 end if

 viol = abs(viol)

 return

 END SUBROUTINE L1_viol

!! ******************************************************************************
!!                             q p c _ r e c o v e r                            |
!! ******************************************************************************
!
!  SUBROUTINE qpc_recover( qp, qp_control, qp_inform, nlp, n_fail, run_level, &
!                          which_qp, error, B_type, status )
!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!!
!!  Attempts to recover from a failed call to QPC_solve.  First attempts to use
!!  the control option qpb_or_qpa.  If this fails then the only reasonable hope
!!  is to change the initial barrier parameter; this is controlled by the
!!  control parameter muzero.
!!
!! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!!-------------------------------------------------------------------------------
!! D u m m y   A r g u m e n t s
!!-------------------------------------------------------------------------------
!
!  type( QPT_problem_type ), intent( inout ) :: qp
!  type( QPC_control_type ), intent( inout ) :: qp_control
!  type( QPC_inform_type ), intent( inout ) :: qp_inform
!  type( NLPT_problem_type ), intent( inout ) :: nlp
!  integer, intent( inout ) :: n_fail
!  integer, intent( in ) :: run_level, error, B_type
!  character ( len=5 ), intent( in ) :: which_qp
!  integer, intent( out ) :: status
!
!!-------------------------------------------------------------------------------
!! L o c a l   V a r i a b l e s
!!-------------------------------------------------------------------------------
!
!  integer :: n, m, m_a, i
!
!  ! Define default return value.
!
!  status = 0
!
!  ! For convenience.
!
!  n   = nlp%n
!  m   = nlp%m
!  m_a = nlp%m_a
!
!  ! Possible quit trying, based on run_level.
!
!  n_fail = n_fail + 1
!
!  if ( run_level < n_fail ) then
!     status = -1
!     return
!  end if
!
!  ! Perform the necessary change in the control parameters.
!  ! *******************************************************
!
!  if ( qp_inform%status == GALAHAD_ok ) then
!     if ( qp_control%qpb_or_qpa ) then
!        qp_control%qpb_or_qpa = .false.
!        qp_control%no_qpb     = .false.
!        qp_control%no_qpa     = .false.
!        qp_inform%status      = GALAHAD_ok
!        n_fail = run_level                 ! last chance
!     else
!        if ( qp_control%no_qpa ) then
!           qp_control%no_qpa = .false.
!           qp_inform%status  =  GALAHAD_ok
!           n_fail = run_level              ! last chance
!        else
!           status = -1                     ! we failed ... sob.
!           return
!        end if
!     end if
!  else
!     if ( qp_control%qpb_or_qpa ) then
!        qp_control%QPB_control%muzero =                                        &
!          ten * max( qp_control%QPB_control%muzero, one )
!        qp_inform%status               = GALAHAD_ok
!        if ( qp_control%QPB_control%muzero >= tenp5 ) then
!           n_fail = run_level   ! last chance
!        end if
!     else ! this can happen at most once.
!        qp_control%qpb_or_qpa = .true.
!        qp_inform%status      = GALAHAD_ok
!     end if
!  end if
!
!  ! Update the QP data
!  ! ******************
!
!  if ( which_qp == ' pred' .or. which_qp == 'steer' ) then
!
!     ! Set step s, elastics u and v, and possibly L-BFGS variables to zero.
!
!     qp%X  = zero
!
!     ! Define the elastics and multipliers for c + Js + u - v.
!
!     if ( m > 0 ) then
!        do i = 1, m
!           qp%X( n+i )   = max( zero, nlp%C_l(i) - nlp%C(i) )
!           qp%X( n+m+i ) = max( zero, nlp%C(i) - nlp%C_u(i) )
!        end do
!        qp%Y( 1:m ) = zero ! maybe nlp%Y
!     end if
!
!     ! Define multipliers for constraint A(x+s).
!
!     if ( m_a > 0 ) then
!        qp%Y( m+1:m+m_a ) = zero ! maybe nlp%Y_a
!     end if
!
!     ! Define multipliers for steering direction if using L-BFGS.
!
!     if ( which_qp == 'pred' .and. B_type == 2 ) then
!        qp%Y( m+m_a + 1 : ) = zero ! something else?
!     end if
!
!     ! Define reduced costs.
!
!     qp%Z      = zero
!     qp%Z(1:n) = nlp%Z
!
!  else
!
!     write(error, *) ' **ERROR:s2qp : qpc_recover : invalid value for which_qp'
!     status = -1
!     return
!
!  end if
!
!  return
!
!  END SUBROUTINE qpc_recover

 !******** G A L A H A D  print_optimality_summary  S U B R O U T I N E ********

 SUBROUTINE print_optimality_summary( data )
 !------------------------------------------------------------------------------
 !  Prints details of optimality check.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: out

 !------------------------------------------------------------------------------

 ! For convenience.

 out = data%control%out

 ! Print information.

 write(out, 3000) data%primal_vl_cur, data%dual_vl_cur, &
                  data%comp_vl_cur, data%opt_measure_cur

 write(out, 3002) data%primal_vl_p, data%dual_vl_p, &
                  data%comp_vl_p, data%opt_measure_p

 if ( data%seqp_computed .or. data%siqp_computed ) then
    write(out, 3003) data%primal_vl_s, data%dual_vl_s, &
                     data%comp_vl_s, data%opt_measure_s
 end if

 write(out, 3004 ) data%primal_vl, data%dual_vl, &
                   data%comp_vl, data%opt_measure, data%mults_used

 return

 ! format statements

 3000 format(/,                     &
      T38, 'Optimality Results', /, &
      T38, '------------------', /, &
      1x,  'primal_cur = ', ES8.2,  T26, 'dual_cur = ', ES8.2,    &
      T48, 'comp_cur = ',  ES8.2,   T70, 'opt_measure_cur = ', ES8.2 )
 3002 format( &
      1x,  'primal_p   = ', ES8.2, T26, 'dual_p   = ', ES8.2,   &
      T48, 'comp_p   = ',   ES8.2, T70, 'opt_measure_p   = ', ES8.2 )
 3003 format( &
      1x,  'primal_s   = ', ES8.2, T26, 'dual_s   = ', ES8.2,   &
      T48, 'comp_s   = ',   ES8.2, T70, 'opt_measure_s   = ', ES8.2 )
 3004 format(                                                           &
      1x, 'primal     = ',  ES8.2, T26, 'dual     = ', ES8.2,           &
      T48, 'comp     = ',   ES8.2, T70, 'opt_measure     = ', ES8.2, /, &
      1x, 'mults_used = ', A )

 END SUBROUTINE print_optimality_summary

 !*********** G A L A H A D  print_pred_steer  S U B R O U T I N E *************

 SUBROUTINE print_pred_steer( data )
 !------------------------------------------------------------------------------
 ! Print details of predictor step and steering step.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: out

 !------------------------------------------------------------------------------

 ! For convenience.

 out = data%control%out

 ! Print information.

 write(out, 3000) data%steering_good,   &
                  data%computed_steering, data%inf_norm_s_p,       &
                  data%inf_norm_Y_p, data%decreaseB, data%norm_c,  &
                  data%norm_c_linearize_pred, data%dec_norm_c_pred

 if ( data%computed_steering ) then
    write(out, 3002) data%norm_c, data%norm_c_linearize_steer,     &
         data%dec_norm_c_steer, data%inf_norm_s_steer, &
         data%inf_norm_Y_steer
 end if

 write(out, 3001) data%penalty, data%TRpred

 return

 ! formating statements.

 3000 format(/, 1x, (79('*')), /,            &
     T28, 'Predictor - Steering Results', /, &
     T28, '----------------------------', /, &
     1x,  'sp_good = ', L1,     T28, 'ss-computed = ', L1, /,    &
     1x,  '|sp|    = ', ES11.5, T28, '|Yp|        = ', ES11.5,     &
                                T58, 'decB_sp = ', ES12.5 /, &
     1x,  '|c|     = ', ES11.5, T28, '|c+Jsp|     = ', ES11.5,     &
                                T58, 'decC_sp = ', ES12.5    )
 3001 format( &
     1x,  'pen_new = ', ES11.5, T28, 'TRpred      = ', ES11.5, /,  &
     1x, (79('*')) )
 3002 format( &
     1x,  '|c|     = ', ES11.5, T28, '|c+Jss|     = ', ES11.5,    &
                                T58, 'decC_ss = ', ES12.5,/,&
     1x,  '|ss|    = ', ES11.5, T28, '|Ys|        = ', ES11.5 )

 END SUBROUTINE print_pred_steer

 !*********** G A L A H A D  print_pred_steer  S U B R O U T I N E *************

 FUNCTION max_feas_step( nlp, data, X, s1, Axs1, Jxs1, s2, Axs2, Jxs2, &
                          X_type, A_type, C_type )
 !------------------------------------------------------------------------------
 ! Computes maximum feasible step.  Assumes that the base point s1 is feasible
 ! and then computes the maximum value of alpha such that x + s1 + alpha*s2
 ! is feasible.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 real( kind = wp ) :: max_feas_step
 real( kind = wp ), intent( in ), dimension(:) :: X
 real( kind = wp ), intent( in ), dimension(:) :: s1, Axs1, Jxs1
 real( kind = wp ), intent( in ), dimension(:) :: s2, Axs2, Jxs2
 character( len = 2 ), intent( in ), dimension(:) :: X_type, A_type, C_type
 type( S2QP_data_type ), intent( inout ) :: data
 type( NLPT_problem_type ), intent( in ) :: nlp

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: nspos, nsneg, nApos, nAneg, nJpos, nJneg
 integer :: i
 real( kind = wp ) :: alpha_x, alpha_A, alpha_C
 real( kind = wp ) :: dummy

 !------------------------------------------------------------------------------

 nspos = 0
 nsneg = 0
 do i = 1, data%nfr
    if ( s2( data%fr(i) ) > tenm9 ) then
       nspos = nspos + 1
       data%spos( nspos ) = data%fr(i)
    elseif ( s2( data%fr(i) ) < - tenm9 ) then
       nsneg = nsneg + 1
       data%sneg( nsneg ) = data%fr(i)
    end if
 end do
 data%nspos = nspos
 data%nsneg = nsneg

 nApos = 0
 nAneg = 0
 do i = 1, data%nwA_comp
    if ( Axs2( data%wA_comp(i) ) > tenm9 ) then
       nApos = nApos + 1
       data%Apos( nApos ) = data%wA_comp(i)
    elseif ( Axs2( data%wA_comp(i) ) < - tenm9 ) then
       nAneg = nAneg + 1
       data%Aneg( nAneg ) = data%wA_comp(i)
    end if
 end do
 data%nApos = nApos
 data%nAneg = nAneg

 nJpos = 0
 nJneg = 0
 do i = 1, data%nwJ_comp
    if ( Jxs2( data%wJ_comp(i) ) > tenm9 ) then
       nJpos = nJpos + 1
       data%Jpos(nJpos) = data%wJ_comp(i)
    elseif ( Jxs2( data%wJ_comp(i) ) < - tenm9 ) then
       nJneg = nJneg + 1
       data%Jneg(nJneg) = data%wJ_comp(i)
    end if
 end do
 data%nJpos = nJpos
 data%nJneg = nJneg

 ! Now compute maximal feasible distance along s2.

 alpha_x = one
 alpha_A = one
 alpha_C = one

 ! write(*,*) 'nfx = ', data%nfx
!  write(*,*) 'fx = ', data%fx
!  write(*,*) 'nfr = ', data%nfr
!  write(*,*) 'fr = ', data%fr
!  write(*,*) 'nspos = ', data%nspos
!  write(*,*) 'nsneg = ', data%nsneg
!  write(*,*) 'spos = ', data%spos
!  write(*,*) 'sneg = ', data%sneg
!  write(*,*) 'Xtype = ', X_type
!  write(*,*) 'X = ', nlp%X
!  write(*,*) 'sp = ', data%s_p
!  write(*,*) 'ss = ', data%S_s
!  write(*,*) 'nApos = ', data%nApos
!  write(*,*) 'nAneg = ', data%nAneg
!  write(*,*) 'Apos = ', data%Apos
!  write(*,*) 'Aneg = ', data%Aneg
!  write(*,*) 'Ax = ', nlp%Ax
!  write(*,*) 'Asp = ', data%AxSp
!  write(*,*) 'Ass = ', data%AxSs
!  write(*,*) 'C = ', nlp%C
!  write(*,*) 'Jsp = ', data%JxSp
!  write(*,*) 'Jss = ', data%JxSs

 do i = 1, nspos
    if ( X_type( data%spos(i) ) == 'RB' .or.                                   &
         X_type( data%spos(i) ) == 'UB' ) then
       dummy = nlp%X_u( data%spos(i) ) -                                       &
        ( X( data%spos(i) ) + s1( data%spos(i) ) )
       dummy = dummy / s2( data%spos(i) )
       alpha_x = min( alpha_x,  dummy )
    end if
 end do
 do i = 1, nsneg
    if ( X_type( data%sneg(i) ) == 'RB' .or.                                   &
         X_type( data%sneg(i) ) == 'LB' ) then
       dummy = nlp%X_l( data%sneg(i) ) -                                       &
         ( X( data%sneg(i) ) + s1( data%sneg(i) ) )
       dummy = dummy / s2( data%sneg(i) )
       alpha_x = min( alpha_x,  dummy )
    end if
 end do

 do i = 1, nApos
    if ( A_type( data%Apos(i) ) == 'RB' .or.                                   &
         A_type( data%Apos(i) ) == 'UB' ) then
       dummy = nlp%A_u( data%Apos(i) ) -                                       &
         ( nlp%Ax( data%Apos(i) ) + Axs1( data%Apos(i) ) )
       dummy = dummy / Axs2( data%Apos(i) )
       alpha_A = min( alpha_A,  dummy )
    end if
 end do
 do i = 1, nAneg
    if ( A_type( data%Aneg(i) ) == 'RB' .or.                                   &
         A_type( data%Aneg(i) ) == 'LB' ) then
       dummy = nlp%A_l( data%Aneg(i) ) -                                       &
         ( nlp%Ax( data%Aneg(i) ) + Axs1( data%Aneg(i) ) )
       dummy = dummy / Axs2( data%Aneg(i) )
       alpha_A = min( alpha_A,  dummy )
    end if
 end do

 do i = 1, nJpos
    if ( C_type( data%Jpos(i)) == 'RB' .or. C_type( data%Jpos(i)) == 'UB' ) then
       dummy = nlp%C_u( data%Jpos(i) ) -                                       &
         ( nlp%C( data%Jpos(i) ) + Jxs1( data%Jpos(i) ) )
       dummy = dummy / Jxs2( data%Jpos(i) )
       alpha_C = min( alpha_C,  dummy )
    end if
 end do
 do i = 1, nJneg
    if ( C_type( data%Jneg(i) ) == 'RB' .or.                                   &
         C_type( data%Jneg(i) ) == 'LB' ) then
       dummy = nlp%C_l( data%Jneg(i) ) -                                       &
         ( nlp%C( data%Jneg(i) ) + Jxs1( data%Jneg(i) ) )
       dummy = dummy / Jxs2( data%Jneg(i) )
       alpha_C = min( alpha_C,  dummy )
    end if
 end do

 max_feas_step = min( alpha_x, alpha_A, alpha_C )

 return

 END FUNCTION max_feas_step

 !************* G A L A H A D  get_residuals  S U B R O U T I N E **************

  SUBROUTINE get_residuals( n, vl, v, vu, v_type, res_vl, res_vu )
 !------------------------------------------------------------------------------
 ! Computes the residual vectors for a given input vector v of length n.
 ! Note: the residual vectors should be of length n, but components res_vl(i)
 !       are computed for equality, lower bound, and range bound variables only,
 !       while res_vu(i) is computed for upper and range bound variables only.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 character( len = 2 ), intent( in ), dimension(:) :: v_type
 integer, intent( in ) :: n
 real( kind = wp ), intent( in ), dimension(:) :: vl, v, vu
 real( kind = wp ), intent( inout ), dimension(:) :: res_vl, res_vu

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: i

 !------------------------------------------------------------------------------

 if ( n <= 0 ) then
    return
 end if

 ! make sure zeros for the "extra" components.

 res_vl = zero
 res_vu = zero

 ! compute the residuals.

 do i = 1, n
    if ( v_type( i ) == 'EQ' .or. v_type( i ) == 'LB' .or.                     &
         v_type( i ) == 'RB'  ) then
       res_vl( i ) = v( i ) - vl( i )
    end if
    if ( v_type( i ) == 'UB' .or. v_type( i ) == 'RB' ) then
       res_vu( i ) = vu( i ) - v( i )
    end if
 end do

 return

 END SUBROUTINE get_residuals

 !************* G A L A H A D  get_active  S U B R O U T I N E **************

 SUBROUTINE get_active( xl, xu, res_xl, res_xu, x_type, &
                        nfx, fx, nfr, fr, nvl, vl, tol, out )
 !------------------------------------------------------------------------------
 ! Computes those vectors that are active at their bounds.
 !
 ! ON EXIT:
 !
 !    nfx     number of variables fixed on there bound.
 !    fx      vector containing indices of fixed variables, i.e., fx(:nfx).
 !    nfr     number of variables free from their bounds.
 !    fr      vector containing indices of free variables, i.e., fr(:nfr)
 !    nvl     number of variables that violate their bounds.
 !    vl      vector containing indices of violated variables, i.e., vl(:nvl)
 !
 !    NOTE:  although it gives which variables are violated and active, this
 !           subroutine does not report the way they are violated or active.
 !           In other words, for range bounds we are not sure if it is violated
 !           below or above, or whether it may be fixed at the upper or lower
 !           bound. Maybe this should be addeded later.
 !    NOTE2: This subroutine assumes that res_xl holds the residuals for the
 !           equality constriants, i.e., res_xl(i) x(i)-xl(i) for all equality
 !           constraints i.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 character( len = 2 ), intent( in ), dimension(:) :: x_type
 integer, intent( in ) :: out
 real( kind = wp ), intent( in ) :: tol
 real( kind = wp ), intent( in ), dimension(:) :: xl, xu, res_xl, res_xu
 integer, intent( out ), dimension(:) :: fx, fr, vl
 integer, intent( out ) :: nfx, nfr, nvl

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: n, i
 real( kind = wp) :: r

 !------------------------------------------------------------------------------

 n = size( xl )

 nfx = 0 ; fx = 0
 nfr = 0 ; fr = 0
 nvl = 0 ; vl = 0

 do i = 1, n

    select case ( x_type(i) )
    case ('LB')
       r = res_xl(i) / (one + abs(xl(i)))
       if ( abs(r) < tol ) then
          nfx = nfx + 1 ;  fx( nfx ) = i
       elseif ( r >= tol ) then
          nfr = nfr + 1 ;  fr( nfr ) = i
       else
          nvl = nvl + 1 ;  vl( nvl ) = i
       end if
    case ('UB')
       r = res_xu(i) / (one + abs(xu(i)))
       if ( abs(r) < tol ) then
          nfx = nfx + 1 ;  fx( nfx ) = i
       elseif ( r >= tol ) then
          nfr = nfr + 1 ;  fr( nfr ) = i
       else
          nvl = nvl + 1 ;  vl( nvl ) = i
       end if
    case ('FR')
       nfr = nfr + 1 ;  fr( nfr ) = i
    case ('EQ')
       r = res_xl(i) / (one + abs(xl(i)))
       if ( abs(r) < tol ) then
          nfx = nfx + 1 ;  fx( nfx ) = i
       else
          nvl = nvl + 1 ;  vl( nvl ) = i
       end if
    case ('RB')
       r = res_xu(i) / (one + abs(xu(i)))
       if ( abs(r) < tol ) then
          nfx = nfx + 1 ;  fx( nfx ) = i
       elseif ( r <= -tol ) then
          nvl = nvl + 1 ;  vl( nvl ) = i
       else
          r = res_xl(i) / (one + abs(xl(i)))
          if ( abs(r) < tol ) then
             nfx = nfx + 1 ;  fx( nfx ) = i
          elseif ( r <= -tol ) then
             nvl = nvl + 1 ;  vl( nvl ) = i
          else
             nfr = nfr + 1 ;  fr( nfr ) = i
          end if
       end if
    case default
       write( out, * ) '**ERROR:s2qp_solve:get_active x_type=?.'
       return
    end select

 end do

 return

 END SUBROUTINE get_active

 !************** G A L A H A D  get_L_BFGS  S U B R O U T I N E ***************

 SUBROUTINE get_L_BFGS( S, A, B, BSinner, B0, svec, gradLx_new, gradLx, y, &
                        num_update, L, n, theta, damp_factor, status,      &
                        print_level, out, error )
 !-----------------------------------------------------------------------------
 ! Computes the matrics B0, A, and B associated with a limited memory BFGS
 ! update:  C = B0 - AA' + BB'.
 !
 ! S             a real rank 2 (n by L) intent inout array whose columns hold
 !               the vectors of displacesments used to compute the L-BFGS
 !               update, i.e., S = [ s1, s2, ..., sL ].
 ! A             a real rank 2 (n by L) intent inout array whose columns hold
 !               the vectors A = [ a1, a2, ..., aL] associated with the L-BFGS
 !               update B0 - AA' + BB'.
 ! B             a real rank 2 (n by L) intent inout array whose columns hold
 !               the vectors B = [ b1, b2, ..., bL] associated with the L-BFGS
 !               update B0 - AA' + BB'.
 ! BSinner       a real rank 2 (L by L) intent inout array that holds the inner
 !               products of the vectors B = [ b1, b2, ... , bL ] with the
 !               vectors S = [ s1, s2, ... sL ].
 ! B0            scalar intent inout argument of type SMT_type that holds the
 !               matrix B0 used in the L-BFGS on exit.  The storage type is
 !               assumed to be DIAGONAL.
 ! svec          the new step, i.e., x_new - x.
 ! gradLx_new    real intent in vector that holds the gradient of the Lagrangian
 !               at the new point.  Only used for possible printing.
 ! gradLx        real intent in vector that holds the gradient of the Lagrangian
 !               at the previous point.  Only used for possible printing.
 ! y             real intent inout vector holding gradLx_new - gradLx.
 ! num_update    integer intent in scalar that indicates the number of times
 !               that the limited memory lbfgs update has been called, i.e.,
 !               the number of times that this subroutine has been called to
 !               provide the L-BFGS update.
 ! L             integer intent in scalar that holds the maximum number of
 !               limited memory vectors that may be used.
 ! n             integer intent in scalar that holds the number of variables.
 ! theta         real intent out scalar that holds the value used to damp the
 !               vector y, as done originally by Powel.  A value of one
 !               indicates that no damping was required.
 ! damp_factor   real intent in scalar that holds the damp factor scalar used
 !               in determining whether y'svec is sufficiently positive.
 ! status        integer scalar that indicates the outcome of the call.
 !                 0  okay
 !               -99  num_update <= 0
 !               -98  skip update because Bs = y
 !               -97  should restart L_BFGS because stBs <= 0
 !               -96  skip update because stBs is small
 !               -80  allocation error
 !               -81  deallocation error
 ! print_level   integer intent in scalar that indicates the print leveel.
 ! out           integer intent in scalar that holds the device number for
 !               informational output.
 ! error         integer intent in scalar that holds the device number for
 !               error messages.
 !-----------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 integer, intent( in ) :: L, error, out, n, print_level, num_update
 integer, intent( out ) :: status
 real( kind = wp ), intent( out ) :: theta
 real( kind = wp ), intent( in ) :: damp_factor
 real( kind = wp ), intent( in ), dimension(:) :: svec, gradLx_new, gradLx
 real( kind = wp ), intent( inout ), dimension(:) :: y
 real( kind = wp ), intent( inout ), dimension(:,:) :: A, B, S, BSinner
 type( SMT_type ), intent( inout ) :: B0

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: i, j, col_replace, col_start, coli, colj, num_used
 real( kind = wp ) :: stBs, sty, Atsi, Btsi, sj, Bsi, delta
 real( kind = wp ), allocatable, dimension(:) :: Bs, Ats, Bts

 !-----------------------------------------------------------------------------

 if (print_level >= GALAHAD_debug ) write(out,1009) ! header

 ! Check for invalid value for num_update

 if ( num_update <= 0 ) then
    if ( print_level >= GALAHAD_debug ) then
       write(error, "(' **ERROR:S2QP_solve:get_L_BFGS num_update <= 0')")
       write(out,1010) ! footer
    end if
    theta  = one
    status = -99
    return
 end if

 ! Define the columns to be replaced and the starting column.

 if ( num_update <= L ) then
    col_replace = num_update
    col_start   = 1
 else
    col_replace = mod( (num_update-1), L ) + 1
    col_start   = mod( num_update, L ) + 1
 end if

 ! Compute Bs = (B_0 - AA' + BB')s for the current "B".

 allocate( Bs(1:n), stat=status )
 if ( status /= GALAHAD_ok ) go to 900
 allocate( Ats(1:num_update), stat=status )
 if ( status /= GALAHAD_ok ) go to 900
 allocate( Bts(1:num_update), stat=status )
 if ( status /= GALAHAD_ok ) go to 900

 num_used = min( L, num_update )

 do i = 1, num_used
    Atsi = zero
    Btsi = zero
    do j = 1, n
       sj = svec(j)
       Atsi = Atsi + A(j,i)*sj
       Btsi = Btsi + B(j,i)*sj
    end do
    Ats(i) = Atsi
    Bts(i) = Btsi
 end do

 do i = 1, n
    Bsi = B0%val(i) * svec(i)
    do j = 1, num_used
       Bsi = Bsi - A(i,j)*Ats(j)
       Bsi = Bsi + B(i,j)*Bts(j)
    end do
    Bs(i) = Bsi
 end do

 ! Skip the update if the current implict "B" satisfies Bs = y.

 if ( NRM2(n,Bs-y,1) <= tenm8 * max( one, NRM2(n,y,1) ) ) then
    if ( print_level >= GALAHAD_debug ) then
       write(out,1000)
       write(out,1010) ! footer
    end if
    theta  = one
    status = -98
    return
 end if

 ! If stBs is not positive, return with status = -97 for a restart

 stBs = dot_product( svec, Bs )
 if ( stBs <= zero  ) then
    if ( print_level >= GALAHAD_debug ) then
       write(out,1001)
       write(out,1010) ! footer
    end if
    theta  = one
    status = -97
    return
 end if

 ! Skip the update if stBs is small.

 if ( stBs <= tenm8 * NRM2(n,svec,1)  ) then
    if ( print_level >= GALAHAD_debug ) then
       write(out,1002) stBs
       write(out,1010) ! footer
    end if
    theta  = one
    status = -96
    return
 end if

 ! Modify y if required.

 sty = dot_product(svec,y)

 if ( print_level >= GALAHAD_debug ) then
    write(out, 1007) col_start, col_replace, num_used
    write(out, 1004) sty, stBs
    if ( print_level >= GALAHAD_crazy ) then
       write(out, 1005)
       write( out,  '(5(2x, ES15.8))') &
            ( svec(i), BS(i), gradLx_new(i), gradLx(i), y,  i = 1, n )
    end if
 end if

 if ( sty >= damp_factor*stBs ) then
    theta = one
 else
    theta = (one-damp_factor)*stBs / ( stBs - sty )
    y = theta*y + (one-theta)*Bs
    sty = dot_product(svec,y)
    if (print_level >= GALAHAD_debug) then
       write(out, 1003) theta, damp_factor
       write(out, 1006) sty, stBs
       if ( print_level >= GALAHAD_crazy ) then
          write(out, 1005)
          write( out,  '(5(2x, ES15.8))') &
               ( svec(i), BS(i), gradLx_new(i), gradLx(i), y,  i = 1, n )
       end if
    end if
 end if

 ! Replace specified column of S.

 S( :, col_replace ) = svec

 ! Compute b_j for the new column.

 B( :, col_replace ) = y / sqrt( sty )

 ! Compute all the required inner-products with the new s.

 if ( num_update < L ) then
    do i = 1, num_update - 1
       BSinner( i, col_replace ) = dot_product( B(:,i), svec )
    end do
 else
    do i = 1, L
       if ( i == col_replace ) then
          cycle
       end if
       BSinner( i, col_replace ) = dot_product( B(:,i), svec )
    end do
 end if

 ! Compute the initial L-BFGS matrix : B0.

 delta = dot_product(y,y) / sty
 B0%val = delta

 ! Compute the vectors that compose the matrix A.

 do  i = col_start, col_start + min(L,num_update) - 1
    coli = mod( i-1, L ) + 1
    do j = 1, n
       A(j,coli) = B0%val(j) * S(j,coli)
    end do
    do j = col_start, i-1
       colj = mod( j-1, L ) + 1
       A(:,coli) = A(:,coli) - dot_product(A(:,colj),S(:,coli)) * A(:,colj)
       A(:,coli) = A(:,coli) + BSinner(colj,coli) * B(:,colj) ;
    end do
    A(:,coli) = A(:,coli) / sqrt( dot_product(S(:,coli),A(:,coli)) )
 end do

 deallocate( Bs, stat=status )
 if ( status /= GALAHAD_ok ) go to 901
 deallocate( Ats, stat=status )
 if ( status /= GALAHAD_ok ) go to 901
 deallocate( Bts, stat=status )
 if ( status /= GALAHAD_ok ) go to 901

 if ( print_level >= GALAHAD_debug ) then
    write(out,1008) delta
    write(out,1010) ! footer
 end if

 ! normal return

 status = 0
 return

 ! abnormal returns

 900 continue
     theta  = one
     status = -80
     return

 901 continue
     status = -81
     return

 ! format statments

 1000 format(/,1x, 'Skipping L-BFGS update : y = Bs already.')
 1001 format(/,1x, 'Restarting L-BFGS : stBs <= 0.')
 1002 format(/,1x, 'Skipping L-BFGS update : stBs = ', ES15.8, ' is too small.')
 1003 format(/,1x, 'Damping used with theta = ', ES15.8, 3x, &
                   'and damp-factor = ', ES15.8 )
 1004 format(/,1x, 'Initial data :  sty = ', ES14.7, '  stBs = ', ES14.7 )
 1005 format( /, &
      10x, 's', 15x, 'Bs', 11x, 'gradLxnew', 10x, 'gradLx', 14x, 'y', /    &
      6x, '---------', 7x, '---------', 8x, '---------', 8x, '----------', &
      8x, '---------')
 1006 format(/, 1x, 'Damped data : sty = ', ES14.7, '  stBs =  ', ES14.7 )
 1007 format(/, ' start-column   = ', I2, /, ' replace-column = ', I2, /, &
                ' number-used    = ', I2 )
 1008 format(/, 1x, 'B0 matrix : scale = ', ES8.1 )
 1009 format(/, &
      1x, 67('-'), /, &
      1x, 20('-'), '   BEGIN : L-BFGS Details    ', 19('-'), /, &
      1x, 67('-') )
 1010 format(/, &
      1x, 67('-'), /, &
      1x, 21('-'), '  END : L-BFGS Details   ', 22('-'), /, &
      1x, 67('-') )

 END SUBROUTINE get_L_BFGS

 !********** G A L A H A D  print_predictor_info  S U B R O U T I N E **********

 SUBROUTINE print_predictor_info( data, nlp )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: m, m_a, n, num_print, out, error, lenu, lenv

 !-----------------------------------------------------------------------------

 ! For convenience

 m   = nlp%m   ;     out       = data%control%out
 n   = nlp%n   ;     error     = data%control%error
 m_a = nlp%m_a ;     num_print = data%control%print_number

 lenu  = data%ncrb + data%nce + data%nclb ! u-elastics in predictor/steering.
 lenv  = data%ncrb + data%nce + data%ncub ! v-elastics in predictor/steering.

 ! Write out the desired data.

 write( out, 3000 ) ! header
 write( out, * ) 'iterate          = ', data%iterate

 if ( m > 0 ) then
    write( out, * ) 'norm_c           = ', data%norm_c
    write( out, * ) 'norm_c_linearize = ', data%norm_c_linearize_pred
    write( out, * ) 'penalty          = ', data%penalty
    write( out, * ) 'dec_cons_viol    = ', data%dec_norm_c_pred
 end if

 write( out, * ) 'gts_p            = ', data%gtSp
 write( out, * ) 'sp_B_sp          = ', data%Sp_B_sp
 write( out, * ) 'decreaseB        = ', data%decreaseB
 write( out, * ) 'decreaseBsmooth  = ', data%decreaseB_smooth

 if ( m > 0 ) then
    call print_real_vec( 's_p', data%s_p, 5, num_print, out, error )
    call print_real_vec( 'Y_p', data%QPpred%Y(:m), 5, num_print, out, error )
 end if

 if ( lenu > 0 ) then
    !call print_real_vec( 'u_in', data%u_in, 5, num_print, out, error )
    call print_real_vec( 'u_out', data%u_out, 5, num_print, out, error )
 end if

 if ( lenv > 0 ) then
    !call print_real_vec( 'v_in', data%v_in, 5, num_print, out, error )
    call print_real_vec( 'v_out', data%v_out, 5, num_print, out, error )
 end if

 if ( m_a > 0 ) then
    call print_real_vec( 'Ya_p', data%QPpred%Y(m+1:m+m_a), 5, num_print, out, error )
 end if

 call print_real_vec( 's_p', data%s_p, 5, num_print, out, error )
 call print_real_vec( 'Z_p', data%QPpred%Z(:n), 5, num_print, out, error )
 call print_real_vec( 'X+S_p', nlp%X+data%s_p, 5, num_print, out, error )

 write( out, 3001 ) ! footer

 return

 ! format statement

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 23('-'), '    BEGIN : Predictor Details    ', 23('-'), /, &
      1x, 79('-') )
 3001 format(/, &
      1x, 79('-'), /, &
      1x, 23('-'), '     END : Predictor Details     ', 23('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_predictor_info

 !*********** G A L A H A D  print_steering_info  S U B R O U T I N E **********

 SUBROUTINE print_steering_info( data, nlp )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: m, num_print, out, error

 !-----------------------------------------------------------------------------

 ! For convenience

 m         = nlp%m
 out       = data%control%out
 error     = data%control%error
 num_print = data%control%print_number

 ! Write out the desired data.

 write( out, 3000 )  ! header

 write( out, * ) 'iterate          = ', data%iterate

 if ( m > 0 ) then
    write( out, * ) 'norm_c           = ', data%norm_c
    write( out, * ) 'norm_c_steer     = ', data%norm_c_linearize_steer
    write( out, * ) 'penalty          = ', data%penalty
    write( out, * ) 'dec_cons_viol    = ', data%dec_norm_c_steer
 end if
 if ( m > 0 ) then
    call print_real_vec( 's_steer', data%s_steer, 5, num_print, out, error )
    call print_real_vec( 'u_in', data%u_in, 5, num_print, out, error )
    call print_real_vec( 'v_in', data%v_in, 5, num_print, out, error )
    call print_real_vec( 'u_out', data%u_out, 5, num_print, out, error )
    call print_real_vec( 'v_out', data%v_out, 5, num_print, out, error )
 end if

 call print_real_vec( 'X + s_steer', nlp%X+data%s_steer, 5, num_print, out, error )

 write( out, 3001 )

 return

 ! format statement

 3000 format(/, &
      1x, 85('-'), /, &
      1x, 26('-'), '    BEGIN : Steering Details    ', 26('-'), /, &
      1x, 85('-') )
 3001 format(/, &
      1x, 85('-'), /, &
      1x, 26('-'), '     END : Steering Details     ', 26('-'), /, &
      1x, 85('-') )

 END SUBROUTINE print_steering_info

 !************** G A L A H A D  fill_QPpred  S U B R O U T I N E ***************

 SUBROUTINE print_predictor_problem( data, out )
 !-----------------------------------------------------------------------------
 ! Does what it says!
 !-----------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data
 integer, intent( in ) :: out

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------
 ! none
 !-----------------------------------------------------------------------------

 write(out,*) ' =============== BEGIN : QPpred ================='
 call QPT_write_problem( out, data%QPpred, 1 )
 write(out,*) ' ================ END : QPpred =================='

 !write( out, * ) 'QPpred%Xl     = ', data%QPpred%X_l
 !write( out, * ) 'QPpred%Xu     = ', data%QPpred%X_u
 !write( out, * ) 'QPpred%Cl     = ', data%QPpred%C_l
 !write( out, * ) 'QPpred%Cu     = ', data%QPpred%C_u
 !write( out, * ) 'Htype         = ', data%QPpred%H%type
 !write( out, * ) 'Hrow          = ', data%QPpred%H%row
 !write( out, * ) 'Hcol          = ', data%QPpred%H%col
 !write( out, * ) 'Hval          = ', data%QPpred%H%val
 !write( out, * ) 'pred%Arow     = ', data%QPpred%A%row
 !write( out, * ) 'pred%Acol     = ', data%QPpred%A%col
 !write( out, * ) 'pred%Aval     = ', data%QPpred%A%val
 !write( out, * ) 'pred%Ane      = ', data%QPpred%A%ne
 !write( out, * ) 'pred%G        = ', data%QPpred%G
 !write( out, * ) 'pred%n        = ', data%QPpred%n
 !write( out, * ) 'pred%m        = ', data%QPpred%m
 !write( out, * ) 'pred%X        = ', data%QPpred%X
 !write( out, * ) 'pred%Y        = ', data%QPpred%Y
 !write( out, * ) 'pred%Z        = ', data%QPpred%Z
 !write( out, * ) 'pred%infinity = ', data%control%QPpred_control%infinity

 return

 END SUBROUTINE print_predictor_problem

 !*********** G A L A H A D  print_cauchy_info  S U B R O U T I N E ************

 SUBROUTINE print_cauchy_info( data, nlp )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------

 integer :: m, m_a, out, error, num_print

 !------------------------------------------------------------------------------

 ! For convenience.

 m   = nlp%m   ;           out = data%control%out
 m_a = nlp%m_a ;         error = data%control%error
                     num_print = data%control%print_number

 ! Print the required data.

 write( out, 3000 ) ! header

 write(out, *) 'two_norm_s_p            = ', data%two_norm_s_p
 write(out, *) 'gtSp                    = ', data%gtSp
 write(out, *) 'Sp_H_Sp                 = ', data%Sp_H_Sp
 write(out, *) 'alpha_c                 = ', data%alpha_c
 write(out, *) 'Sc_H_Sc                 = ', data%Sc_H_Sc
 if ( m > 0 ) then
    write(out, *) 'c_norm                  = ', data%norm_c
    write(out, *) 'norm_c_linearize_cauchy = ', data%norm_c_linearize_cauchy
    write(out, *) 'dec_norm_c_cauchy       = ', data%dec_norm_c_cauchy
 end if
 write(out, *) 'decreaseH_cauchy        = ', data%decreaseH_cauchy
 call print_real_vec( 's_c', data%s_c, 5, num_print, out, error )
 call print_real_vec( 'X+S_c', nlp%X+data%s_c, 5, num_print, out, error )
 if ( m > 0 ) then
    call print_real_vec( 'JxS_c', data%JxSc, 5, num_print, out, error )
 end if
 if ( m_a > 0 ) then
    call print_real_vec( 'Ax', nlp%Ax, 5, num_print, out, error )
    call print_real_vec( 'AxSc', data%AxSc, 5, num_print, out, error )
    call print_real_vec( 'A(x+s_c)', nlp%Ax+data%AxSc, 5, num_print, out, error)
 end if

 write( out, 3001 ) ! footer

 return

 ! format statement

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 24('-'), '    BEGIN : Cauchy Details     ', 24('-'), /, &
      1x, 79('-') )
 3001 format(/, &
      1x, 79('-'), /, &
      1x, 24('-'), '     END : Cauchy Details      ', 24('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_cauchy_info

 !********** G A L A H A D  print_seqp_problem  S U B R O U T I N E ************

 SUBROUTINE print_seqp_problem( data, out )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data
 integer, intent( in ) :: out

 !------------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !------------------------------------------------------------------------------
 ! none
 !------------------------------------------------------------------------------

 write(out,3000) ! header
 call QPT_write_problem( out, data%QPseqp, 1 )
 write(out, "(' acc_radius = ', ES10.4, /)") data%control%QPseqp_control%radius
 write(out,3001) ! footer

 return

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '    BEGIN : SEQP problem     ', 25('-'), /, &
      1x, 79('-') )
 3001 format( &
      1x, 79('-'), /, &
      1x, 25('-'), '     END : SEQP problem      ', 25('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_seqp_problem

 !*********** G A L A H A D  print_seqp_info  S U B R O U T I N E **************

 SUBROUTINE print_seqp_info( data, nlp )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: m, m_a, out, error, num_print
 real ( kind = wp ) :: dummy_real

 !-----------------------------------------------------------------------------

 ! For convenience.

 m   = nlp%m    ;          out       = data%control%out
 m_a = nlp%m_a  ;          error     = data%control%error
                           num_print = data%control%print_number

 ! Print the information.

 write( out, 3000 ) ! header

 write( out, * ) 'iterate          = ', data%iterate
 write( out, * ) 'num_free         = ', data%nfr
 write( out, * ) 'num_fx           = ', data%nfx
 write( out, * ) 'num_wJ           = ', data%nwJ
 write( out, * ) 'num_wJ_comp      = ', data%nwJ_comp
 write( out, * ) 'num_wA           = ', data%nwA
 write( out, * ) 'num_wA_comp      = ', data%nwA_comp
 write( out, * ) 'alpha_feas       = ', data%alpha_feas
 write( out, * ) 'decreaseH_cauchy = ', data%decreaseH_cauchy
 write( out, * ) 'Ss_H_Ss          = ', data%Ss_H_Ss

 if ( data%seqp_try_pred ) then
    dummy_real = DOT_PRODUCT( nlp%G + data%HxSp, data%s_s )
    write( out, * ) '(g+HxSp)^T s_s   = ', dummy_real
    write( out, * ) 'QPseqp_obj       = ', dummy_real + half*data%Ss_H_Ss
    call print_real_vec( 'HxSp', data%HxSp, 5, num_print, out, error )
    call print_real_vec( 'X + S_p + alpha_feas * S_s',                &
                         nlp%X + data%s_p + data%alpha_feas*data%s_s, &
                         5, num_print, out, error )
    call print_real_vec( 'g+HxSp', nlp%G + data%HxSp, 5, num_print, out, error )
 else
    dummy_real = DOT_PRODUCT( nlp%G + data%HxSc, data%s_s )
    write( out, * ) '(g+HxS_c)^T s_s = ', dummy_real
    write( out, * ) 'QPseqp_obj     = ', dummy_real + half*data%Ss_H_Ss
    call print_real_vec( 'HxS_c', data%HxSc, 5, num_print, out, error )
    call print_real_vec( 'X + S_c + alpha_feas * S_s',                &
                         nlp%X + data%s_c + data%alpha_feas*data%s_s, &
                         5, num_print, out, error )
    call print_real_vec( 'g+HxSc', nlp%G + data%HxSc, 5, num_print, out, error )
 end if

 call print_int_vec( 'x-free', data%fr, 10, num_print, out, error )
 call print_int_vec( 'x-fixed', data%fx, 10, num_print, out, error )
 if ( m > 0 ) then
    call print_int_vec( 'J-in-working-set', data%wJ, 10, num_print, out, error )
    call print_int_vec( 'J-NOT-in-working-set', data%wJ_comp, 10, num_print, out, error )
 end if
 if ( m_a > 0 ) then
    call print_int_vec( 'A-in-working-set', data%wA, 6, num_print, out, error )
    call print_int_vec( 'A-NOT-in-working set', data%wA_comp, 10, num_print, out, error )
 end if

 call print_real_vec( 'S_s', data%s_s, 5, num_print, out, error )
 call print_real_vec( 'QPseqp%X', data%QPseqp%X, 5, num_print, out, error )
 call print_real_vec( 'HxSs', data%HxSs, 5, num_print, out, error )
 if ( m > 0 ) then
    call print_real_vec( 'Y_s', data%Y_s, 5, num_print, out, error )
    call print_real_vec( 'JxSs', data%JxSs, 5, num_print, out, error )
 end if
 if( m_a > 0 ) then
    call print_real_vec( 'Ya_s', data%Ya_s, 5, num_print, out, error )
    call print_real_vec( 'AxSs', data%AxSs, 5, num_print, out, error )
 end if
 call print_real_vec( 'Z_s', data%Z_s, 5, num_print, out, error )

 write( out, 3001 ) ! footer

 return

 ! format statement

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '    BEGIN : SEQP Details     ', 25('-'), /, &
      1x, 79('-'), / )
 3001 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '     END : SEQP Details      ', 25('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_seqp_info

 !*********** G A L A H A D  print_seqp_info  S U B R O U T I N E **************

 SUBROUTINE print_siqp_problem( data )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: out

 !-----------------------------------------------------------------------------

 ! For convenience.

 out = data%control%out

 ! Print the problem.

 write(out,3000) ! header
 call QPT_write_problem( out, data%QPsiqp, 1 )
 write(out,3001) ! footer

 return

 ! format statements

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '    BEGIN : SIQP problem     ', 25('-'), /, &
      1x, 79('-') )
 3001 format( &
      1x, 79('-'), /, &
      1x, 25('-'), '     END : SIQP problem      ', 25('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_siqp_problem

 !*********** G A L A H A D  print_siqp_info  S U B R O U T I N E **************

 SUBROUTINE print_siqp_info( data, nlp, descent_mult )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data
 real( kind = wp ), intent( in ) :: descent_mult

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: n, m, m_a, out, error, print_level, num_print, i

 !-----------------------------------------------------------------------------

 ! For convenience.

 m   = nlp%m   ;         out         = data%control%out
 m_a = nlp%m_a ;         error       = data%control%error
 n   = nlp%n   ;         print_level = data%control%print_level
                         num_print   = data%control%print_number

 ! Print the information.

 write(out,3000) ! header

 write( out, * ) 'iterate          = ', data%iterate
 write( out, * ) 'siqp-iterates    = ', data%iterates_acc
 write( out, * ) 'decreaseH_cauchy = ', data%decreaseH_cauchy
 write( out, * ) 'Ss_H_Ss          = ', data%Ss_H_Ss
 write( out, * ) 'descent_mult     = ', descent_mult
 write( out, * ) 'descent_status   =  ', adjustl(data%descent_constraint_status)
 write(out, 3003 ) dot_product(data%descent_con, data%S_s)
 write( out, * ) 'inf_norm_s_s     = ', data%inf_norm_s_s

 call print_real_vec( 's_s', data%s_s, 5, num_print, out, error )
 call print_real_vec( 'QPsiqp%X', data%QPsiqp%X, 5, num_print, out, error )
 call print_real_vec( 'X + s_s', nlp%X + data%s_s, 5, num_print, out, error )
 call print_real_vec( 'HxSs', data%HxSs, 5, num_print, out, error )
 call print_real_vec( 'g + HxSc', data%GplusHs, 5, num_print, out, error )
 call print_real_vec( 'descent_constraint', data%descent_con, 5, num_print, out, error )

 if ( m > 0 ) then
    call print_real_vec( 'Y_s', data%Y_s, 5, num_print, out, error )
    call print_real_vec( 'Y_s from QP', data%QPsiqp%Y(:m), 5, num_print, out, error )
    call print_real_vec( 'JxSs', data%JxSs, 5, num_print, out, error )
 end if
 if( m_a > 0 ) then
    call print_real_vec( 'Ya_s', data%Ya_s, 5, num_print, out, error )
    call print_real_vec( 'Ya_s from QP', data%QPsiqp%Y(m+1:m+m_a), 5, num_print, out, error )
    call print_real_vec( 'AxSs', data%AxSs, 5, num_print, out, error )
 end if
 call print_real_vec( 'Z_s', data%Z_s, 5, num_print, out, error )
 call print_real_vec( 'Z_s from QP', data%QPsiqp%Z(:n), 5, num_print, out, error )

 if ( print_level >= GALAHAD_crazy .and. m > 0 ) then
    write(out, 3002) data%num_sat, data%num_vl_l, data%num_vl_u
    write(out,'(3(3x, I7, 13x) )') (data%sat(i), data%vl_l(i), data%vl_u(i), i = 1, m)
 end if

 write(out,3001) ! footer

 return

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '    BEGIN : SIQP details     ', 25('-'), /, &
      1x, 79('-'), / )
 3001 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '     END : SIQP details      ', 25('-'), /, &
      1x, 79('-') )
 3002 format( /, &
      1x, 'num_sat = ', I7, 5x, 'num_vl_l = ', I7, 5x, 'num_vl_u = ', I7, /, &
      1x, '    sat                   vl_l                   vl_u', /,   &
      1x, '  -------                -------                -------'     )
 3003 format(1x, 'desc_con^T s_s   = ', ES16.9 )

 END SUBROUTINE print_siqp_info

 !*********** G A L A H A D  print_siqp_info  S U B R O U T I N E **************

 SUBROUTINE print_trial_info( data, nlp )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !------------------------------------------------------------------------------

 type( NLPT_problem_type ), intent( in ) :: nlp
 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: out, error, num_print

 !-----------------------------------------------------------------------------

 ! For convenience

 out       = data%control%out
 error     = data%control%error
 num_print = data%control%print_number

 ! Print information.

 write( out, 3000 ) ! header

 write( out, * ) 'trial_step              =  ', data%trial_step
 write( out, * ) 'norm_c                  = ', data%norm_c
 write( out, * ) 'norm_c_new              = ', data%norm_c_new
 write( out, * ) 'nlp%f                   = ', nlp%f
 write( out, * ) 'data%f_new              = ', data%f_new

 if ( data%trial_step == 'cs' ) then
    write( out, * ) 'norm_c_linearize_full   = ', data%norm_c_linearize_full
    write( out, * ) 'inf_norm_s_f            = ', data%inf_norm_s_f
 elseif ( data%trial_step == 'ps' ) then
    write( out, * ) 'norm_c_linearize_full   = ', data%norm_c_linearize_full
    write( out, * ) 'inf_norm_s_f            = ', data%inf_norm_s_f
 elseif ( data%trial_step == 'c ' ) then
    write( out, * ) 'norm_c_linearize_cauchy = ', data%norm_c_linearize_cauchy
    write( out, * ) 'inf_norm_s_c            = ', data%inf_norm_s_c
 else
    write( out, * ) 'norm_c_linearize_pred   = ', data%norm_c_linearize_pred
    write( out, * ) 'inf_norm_s_p            = ', data%inf_norm_s_p
 end if

 write( out, * ) 'delmod                  = ', data%delmod
 write( out, * ) 'M                       = ', data%merit
 write( out, * ) 'M_new                   = ', data%merit_new
 write( out, * ) 'M - M_new               = ', data%merit - data%merit_new

 write( out, * ) 'NM-active               = ', data%NM%active
 if ( data%NM%active ) then
    write( out, * ) 'delmod_ref              = ', data%NM%delmod_ref
    write( out, * ) 'M_ref                   = ', data%NM%merit_ref
    write( out, * ) 'M_ref - M_new           = ', data%NM%merit_ref - data%merit_new
 end if

 write( out, * ) 'ratio                   = ', data%ratio

 call print_real_vec( 's_f', data%s_f, 5, num_print, out, error )
 call print_real_vec( 'X_trial', data%X_trial, 5, num_print, out, error )

 if ( data%trial_step == 'cs' ) then
    call print_real_vec( 'X + s_f', nlp%X + data%s_f, 5, num_print, out, error )
 elseif ( data%trial_step == 'ps' ) then
    call print_real_vec( 'X + s_f', nlp%X + data%s_f, 5, num_print, out, error )
 elseif ( data%trial_step == 'c ' ) then
    call print_real_vec( 'X + s_c', nlp%X + data%s_c, 5, num_print, out, error )
 else
    call print_real_vec( 'X + s_p', nlp%X + data%s_p, 5, num_print, out, error )
 end if

 write( out, 3001 ) ! footer

 return

 ! format statement

 3000 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '    BEGIN : Trial step details     ', 25('-'), /, &
      1x, 79('-'), / )
 3001 format(/, &
      1x, 79('-'), /, &
      1x, 25('-'), '     END : Trial step details      ', 25('-'), /, &
      1x, 79('-') )

 END SUBROUTINE print_trial_info

 !********** G A L A H A D  print_step_summary  S U B R O U T I N E ************

 SUBROUTINE print_step_summary( data )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: out
 logical :: acc_computed

 !-----------------------------------------------------------------------------

 ! For convenience

 out = data%control%out

 ! Print information.

 acc_computed = data%seqp_computed .or. data%siqp_computed

 if ( acc_computed ) then
    write(out, 1005) data%iterate-1,                                   &
          data%control%B_type, data%alpha_c, data%trial_step,          &
          data%BFGS%mod_type, data%decreaseH_cauchy, data%delmod,      &
          data%use_prev_pred, data%seqp_computed, data%ratio,          &
          data%control%use_TRpred, data%alpha_feas, data%success_str,  &
          data%iterates_pred, data%siqp_computed,  data%step_accepted, &
          data%TRpred, data%descent_constraint_status, data%NM%revert, &
          data%inf_norm_s_p, data%iterates_acc, data%NM%active,        &
          data%inf_norm_Y_p, data%TRacc, data%control%NM_steps,        &
                             data%inf_norm_s_s, data%NM%num_fail,      &
                             data%inf_norm_Y_s
 else
    write(out, 1006) data%iterate-1,                  &
          data%control%B_type, acc_computed,           &
          data%BFGS%mod_type, data%delmod,             &
          data%TRpred, data%ratio,                     &
          data%iterates_pred, data%success_str,        &
          data%inf_norm_s_p, data%NM%active,           &
          data%inf_norm_Y_p, data%control%NM_steps,    &
          data%control%use_TRpred, data%step_accepted, &
          data%use_prev_pred, data%NM%num_fail,        &
          data%NM%revert
 end if

 return

 ! format statement

 1005 FORMAT( /, &
  1X, 79('*'),/, &
  1X, 23('*'),'BEGIN SUMMARY (S2QP) : ITERATE S(', I7,')', 23('*'), /, &
  1X, 79('*'),/, &
  1X, 'B-approx-type = ', 12x, I1, T34, 'alpha-cauchy  = ', ES13.6, T66, 'trial-step    = ', 11x,A2,  /, &
  1X, 'BFGS-mod-type = ', 12x, I1, T34, 'dec-cauchy    = ', ES13.6, T66, 'dec-model     = ', ES13.6,  /, &
  1x, 'use-prev-pred = ', 12x, L1, T34, 'sEqp-computed = ', 12x,L1, T66, 'ratio         = ', ES13.6,  /, &
  1x, 'use-TR-pred   = ', 12x, L1, T34, 'alpha-feas    = ', ES13.6, T66, 'step-sucess   = ', 6x,  A7, /, &
  1X, 'iters-pred    = ', 6x,  I7, T34, 'sIqp-computed = ', 12x,L1, T66, 'step-accepted = ', 12x, L1, /, &
  1X, 'radius-pred   = ', ES13.6,  T34, 'descent       = ', 9x, A4, T66, 'reverting     = ', 12x, L1, /, &
  1X, 'inf-norm-pred = ', ES13.6,  T34, 'iters-acc     = ', 6x, I7, T66, 'NM-active     = ', 12x, L1, /, &
  1x, 'inf-norm-Y_p  = ', ES13.6,  T34, 'acc-radius    = ', ES13.6, T66, 'NM-steps      = ', 12x, I1, /, &
                                   T34, 'inf-norm-acc  = ', ES13.6, T66, 'NM-#-fail     = ', 12x, I1, /, &
                                   T34, 'inf-norm-Y_s  = ', ES13.6,                                   /, &
  1X, 79('*'),/, &
  1X, 23('*'),'END SUMMARY (S2QP)', 23('*'), /, &
  1X, 79('*') )
 1006 FORMAT( /, &
  1X, 79('*'),/, &
  1X, 23('*'), 'BEGIN SUMMARY (S2QP) : ITERATE S(', I7,')', 23('*'), /, &
  1X, 79('*'),/, &
  20X, 'B-approx-type   = ', 12x, I1, T58, 'ACC-computed    = ', 12x, L1, /, &
  20X, 'BFGS-mod-type   = ', 12x, I1, T58, 'change-in-model = ', ES13.6,  /, &
  20X, 'predict-radius  = ', ES13.6,  T58, 'ratio           = ', ES13.6,  /, &
  20X, 'iterations-pred = ', 6x, I7,  T58, 'step-sucess     = ', 6x, A7,  /, &
  20X, 'inf-norm-pred   = ', ES13.6,  T58, 'non-mono-active = ', 12x, L1, /, &
  20x, 'inf-norm-Y_p    = ', ES13.6,  T58, 'non-mono-steps  = ', 12x, I1, /, &
  20x, 'use-TR-pred     = ', 12x, L1, T58, 'step-accepted   = ', 12x, L1, /, &
  20x, 'use-prev-pred   = ', 12x, L1, T58, 'non-mono-#-fail = ', 12x, I1, /, &
                                      T58, 'reverting       = ', 12x, L1, /, &
  1X, 79('*'),/, &
  1X, 23('*'), 'END SUMMARY (S2QP)', 23('*'), /, &
  1X, 79('*') )

 END SUBROUTINE print_step_summary

 !********* G A L A H A D  print_optimal_info  S U B R O U T I N E *************

 SUBROUTINE print_optimal_info( nlp, data )
 !------------------------------------------------------------------------------
 ! Does what it says!
 !------------------------------------------------------------------------------
 implicit none
 !-----------------------------------------------------------------------------
 ! D u m m y   A r g u m e n t s
 !-----------------------------------------------------------------------------

 type( S2QP_data_type ), intent( in ) :: data
 type( NLPT_problem_type ), intent( in ) :: nlp

 !-----------------------------------------------------------------------------
 ! L o c a l   V a r i a b l e s
 !-----------------------------------------------------------------------------

 integer :: m, m_a, n, out, error, status
 integer :: print_level, num_print

 !-----------------------------------------------------------------------------

 ! for convenience

 n = nlp%n ;  m = nlp%m ;  m_a = nlp%m_a

 out             = data%control%out
 error           = data%control%error
 num_print       = data%control%print_number
 print_level     = data%control%print_level

 ! Print information.

 write(out,1000) ! header

 if ( print_level >= GALAHAD_debug ) then
    call print_real_vec( 'X', nlp%X, 5, num_print, out, error )
    call print_real_vec( 'G', nlp%G, 5, num_print, out, error )

    if ( m > 0 ) then
       call print_real_vec( 'Y', nlp%Y, 5, num_print, out, error )
       call print_real_vec( 'JtY', data%JtY, 5, num_print, out, error )
    end if
    if ( m_a > 0 ) then
       call print_real_vec( 'Ya', nlp%Y_a, 5, num_print, out, error )
       call print_real_vec( 'AtYa', data%AtYa, 5, num_print, out, error )
    end if
    call print_real_vec( 'Z', nlp%Z, 5, num_print, out, error )

    if ( m > 0 ) then
       call print_real_vec( 'Y_p', data%QPpred%Y(:m), 5, num_print, out, error )
       call print_real_vec( 'JtY_p', data%JtY_p, 5, num_print, out, error )
    end if
    if ( m_a > 0 ) then
       call print_real_vec( 'Ya_p', data%QPpred%Y(m+1:m+m_a), 5, num_print, out, error )
       call print_real_vec( 'AtYa_p', data%AtYa_p, 5, num_print, out, error )
    end if
    call print_real_vec( 'Z_p', data%QPpred%Z(:n), 5, num_print, out, error )

    if ( data%seqp_computed .or. data%siqp_computed ) then
       if ( m > 0 ) then
          call print_real_vec( 'Y_s', data%Y_s, 5, num_print, out, error )
          call print_real_vec( 'JtY_s', data%JtY_s, 5, num_print, out, error )
       end if
       if ( m_a > 0 ) then
          call print_real_vec( 'Ya_s', data%Ya_s, 5, num_print, out, error )
          call print_real_vec( 'AtYa_s', data%AtYa_s, 5, num_print, out, error )
       end if
       call print_real_vec( 'Z_s', data%Z_s, 5, num_print, out, error )
    end if
 end if

 if ( print_level >= GALAHAD_crazy ) then
    if ( m > 0 ) then
       call print_SMT( nlp%J, 'nlp%J', error, out, status )
    end if
    if ( m_a > 0 ) then
       call print_SMT( nlp%A, 'nlp%A', error, out, status )
    end if
 end if

 if ( print_level >= GALAHAD_details ) call print_optimality_summary( data )

 write(out,1001)

 return

 ! format statements

 1000 format(/,                                                           &
      1x, (79('~')), /,                                                   &
      1x, (22('~')), '   BEGIN (S2QP) optimality data   ', (23('~')), /, &
      1x, (79('~')) )
 1001 format(/,                                                           &
      1x, (79('~')), /,                                                   &
      1x, (22('~')), '    END (S2QP) optimality data    ', (23('~')), /, &
      1x, (79('~')) )

 END SUBROUTINE print_optimal_info

! ******************************************************************************
!                          p r i n t _ r e a l _ v e c                         |
! ******************************************************************************

  SUBROUTINE print_real_vec( name, v, num_col, num_print, out, error )

  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  !
  ! Does what it says!
  !
  ! Restriction: num_col <= 6
  !
  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  !-----------------------------------------------------------------------------
  ! D u m m y   A r g u m e n t s
  !-----------------------------------------------------------------------------

  real( kind = wp ), intent( in ), dimension(:) :: v
  integer, intent( in ) :: num_col
  integer, intent( in ), optional :: out, error, num_print
  character(len=*), intent( in ) :: name

  !-----------------------------------------------------------------------------
  !   L o c a l   V a r i a b l e s
  !-----------------------------------------------------------------------------

  integer :: lv, num_full_row, size_last_row, num, e_dev, i_dev, i
  integer :: number_print, number_col

  !-----------------------------------------------------------------------------

  ! Check for error print device and output device.

  if ( present(error) ) then
     e_dev = error
  else
     e_dev = 6
  end if
  if ( present(out) ) then
     i_dev = out
  else
     i_dev = 6
  end if
  if ( present(num_print) ) then
     number_print = num_print
  else
     number_print = -1
  end if

  ! Make sure num_col makes sense.

  number_col = max( 1, num_col )
  if ( number_col > 6 ) then
     write( e_dev, 1000 )
     return
  end if

  ! Print the name of the vector.

  write( i_dev, "(1x, A, ' =')") name

  ! Return if number_print == 0

  if ( number_print == 0 ) return

  ! Print the appropriate components of the vector v.

  lv = size(v)

  if ( number_print < 0 .or. 2 * number_print >= lv ) then

     ! **************************
     ! print the entire vector v.
     ! **************************

     ! Number of full rows of width number_col.

     num_full_row = floor(real(lv)/real(number_col))

     ! Number of components to print in last row.

     size_last_row = mod(lv,number_col)

     ! Print all but the last "ragged" row.

     num = 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
           num = num + 5
        else
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4), v(num+5)
           num = num + 6
        end if
     end do

     ! Possibly print the last "ragged" row.

     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     else
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
     end if

  else

     ! *************************************************************
     ! print first and last number_print components of the vector v.
     ! *************************************************************

     ! Number of full rows of width number_col.

     num_full_row = floor(real(number_print)/real(number_col))

     ! Number of components to print in last "ragged" row.

     size_last_row = mod(number_print,number_col)

     ! Print all but the last "ragged" row of the first number_print components

     num = 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
           num = num + 5
        else
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4), v(num+5)
           num = num + 6
        end if
     end do

     ! Print the last "ragged" row of the first number_print components

     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     else
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
     end if

     ! print a line of dots.

     write( i_dev, 2000)

     ! Print all but the last "ragged" row of the last number_print components

     num = lv - number_print + 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
           num = num + 5
        else
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4), v(num+5)
           num = num + 6
        end if
     end do

     ! Possibly print the last "ragged" row.
     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     else
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
     end if

  end if

  return

  !format statements

1000 format(' ERROR:S2QP:print_real_vec: input arg NUM_COL > 6.')
1001 format(3x, I5, 1x, 1(ES13.6,3x) )
1002 format(3x, I5, 1x, 2(ES13.6,3x) )
1003 format(3x, I5, 1x, 3(ES13.6,3x) )
1004 format(3x, I5, 1x, 4(ES13.6,3x) )
1005 format(3x, I5, 1x, 5(ES13.6,3x) )
1006 format(3x, I5, 1x, 6(ES13.6,3x) )
2000 format(7x, (80('.')))

 END SUBROUTINE print_real_vec

! ******************************************************************************
!                          p r i n t _ i n t _ v e c                           |
! ******************************************************************************

  SUBROUTINE print_int_vec( name, v, num_col, num_print, out, error )

  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  !
  ! Does what it says!
  !
  ! Restriction: num_col <= 8
  !
  ! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  !-----------------------------------------------------------------------------
  ! D u m m y   A r g u m e n t s
  !-----------------------------------------------------------------------------

  integer, intent( in ), dimension(:) :: v
  integer, intent( in ) :: num_col
  integer, intent( in ), optional :: out, error, num_print
  character(len=*), intent( in ) :: name

  !-----------------------------------------------------------------------------
  !   L o c a l   V a r i a b l e s
  !-----------------------------------------------------------------------------

  integer :: lv, num_full_row, size_last_row, num, e_dev, i_dev, i
  integer :: number_print, number_col

  !-----------------------------------------------------------------------------

  ! Check for error print device and output device.

  if ( present(error) ) then
     e_dev = error
  else
     e_dev = 6
  end if
  if ( present(out) ) then
     i_dev = out
  else
     i_dev = 6
  end if
  if ( present(num_print) ) then
     number_print = num_print
  else
     number_print = -1
  end if

  ! Make sure num_col makes sense.

  number_col = max( 1, num_col )
  if ( number_col > 10 ) then
     write( e_dev, 1000 )
     return
  end if

  ! Print the name of the vector.

  write( i_dev, "(1x, (A), ' =')") name

  ! Return if number_print == 0

  if ( number_print == 0 ) return

  ! Print the appropriate components of the vector v.

  lv = size(v)

  if ( number_print < 0 .or. 2 * number_print >= lv ) then

     ! **********************************
     ! print the entire INTEGER vector v.
     ! **********************************

     ! Number of full rows of width number_col.

     num_full_row = floor(real(lv)/real(number_col))

     ! Number of components to print in last row.

     size_last_row = mod(lv,number_col)

     ! Print all but the last "ragged" row.

     num = 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1),          &
                                     v(num+2), v(num+3), v(num+4)
           num = num + 5
        elseif( number_col == 6 ) then
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), &
                                     v(num+3), v(num+4), v(num+5)
           num = num + 6
        elseif( number_col == 7 ) then
           write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6)
           num = num + 7
        elseif( number_col == 8 ) then
           write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6), v(num+7)
           num = num + 8
        elseif( number_col == 9 ) then
           write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8)
           num = num + 9
        else
           write( i_dev, 1010 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8), v(num+9)
           num = num + 10
        end if
     end do

     ! Possibly print the last "ragged" row.

     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     elseif ( size_last_row == 5 ) then
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
     elseif ( size_last_row == 6 ) then
        write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5)
     elseif ( size_last_row == 7 ) then
        write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6)
     elseif ( size_last_row == 8 ) then
        write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6), v(num+7)
     else
        write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                  v(num+4), v(num+5), v(num+6), v(num+7), &
                                  v(num+8)
     end if

  else

     ! *************************************************************
     ! print first and last number_print components of the vector v.
     ! *************************************************************

     ! Number of full rows of width number_col.

     num_full_row = floor(real(number_print)/real(number_col))

     ! Number of components to print in last "ragged" row.

     size_last_row = mod(number_print,number_col)

     ! Print all but the last "ragged" row of the first number_print components

     num = 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), &
                                     v(num+3), v(num+4)
           num = num + 5
        elseif( number_col == 6 ) then
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), &
                                     v(num+3), v(num+4), v(num+5)
           num = num + 6
        elseif( number_col == 7 ) then
           write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6)
           num = num + 7
        elseif( number_col == 8 ) then
           write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6), v(num+7)
           num = num + 8
        elseif( number_col == 9 ) then
           write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8)
           num = num + 9
        else
           write( i_dev, 1010 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8), v(num+9)
           num = num + 10
        end if
     end do

     ! Print the last "ragged" row of the first number_print components

     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     elseif ( size_last_row == 5 ) then
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4)
     elseif ( size_last_row == 6 ) then
        write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5)
     elseif ( size_last_row == 7 ) then
        write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6)
     elseif ( size_last_row == 8 ) then
        write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6), v(num+7)
     else
        write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                  v(num+4), v(num+5), v(num+6), v(num+7), &
                                  v(num+8)
     end if

     ! print a line of dots.

     write( i_dev, 2000)

     ! Print all but the last "ragged" row of the last number_print components

     num = lv - number_print + 1

     do i = 1, num_full_row
        if ( number_col == 1 ) then
           write( i_dev, 1001 ) num, v(num)
           num = num + 1
        elseif ( number_col == 2 ) then
           write( i_dev, 1002 ) num, v(num), v(num+1)
           num = num + 2
        elseif ( number_col == 3 ) then
           write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
           num = num + 3
        elseif ( number_col == 4 ) then
           write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
           num = num + 4
        elseif ( number_col == 5 ) then
           write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), v(num+4)
           num = num + 5
        elseif( number_col == 6 ) then
           write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), &
                                     v(num+3), v(num+4), v(num+5)
           num = num + 6
        elseif( number_col == 7 ) then
           write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6)
           num = num + 7
        elseif( number_col == 8 ) then
           write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                     v(num+4), v(num+5), v(num+6), v(num+7)
           num = num + 8
        elseif( number_col == 9 ) then
           write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8)
           num = num + 9
        else
           write( i_dev, 1010 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                     v(num+4), v(num+5), v(num+6), v(num+7), &
                                     v(num+8), v(num+9)
           num = num + 10
        end if
     end do

     ! Possibly print the last "ragged" row.

     if ( size_last_row == 0 ) then
        ! relax
     elseif( size_last_row == 1 ) then
        write( i_dev, 1001 ) num, v(num)
     elseif ( size_last_row == 2 ) then
        write( i_dev, 1002 ) num, v(num), v(num+1)
     elseif ( size_last_row == 3 ) then
        write( i_dev, 1003 ) num, v(num), v(num+1), v(num+2)
     elseif ( size_last_row == 4 ) then
        write( i_dev, 1004 ) num, v(num), v(num+1), v(num+2), v(num+3)
     elseif ( size_last_row == 5 ) then
        write( i_dev, 1005 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4)
     elseif ( size_last_row == 6 ) then
        write( i_dev, 1006 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5)
     elseif ( size_last_row == 7 ) then
        write( i_dev, 1007 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6)
     elseif ( size_last_row == 8 ) then
        write( i_dev, 1008 ) num, v(num), v(num+1), v(num+2), v(num+3), &
                                  v(num+4), v(num+5), v(num+6), v(num+7)
     else
        write( i_dev, 1009 ) num, v(num), v(num+1), v(num+2), v(num+3),   &
                                  v(num+4), v(num+5), v(num+6), v(num+7), &
                                  v(num+8)
     end if

  end if

  return

  !format statements

1000 format(' ERROR:S2QP:print_int_vec: input arg NUM_COL > 10.')
1001 format(3x, I5, ' <>', 1x, 1(I5,3x) )
1002 format(3x, I5, ' <>', 1x, 2(I5,3x) )
1003 format(3x, I5, ' <>', 1x, 3(I5,3x) )
1004 format(3x, I5, ' <>', 1x, 4(I5,3x) )
1005 format(3x, I5, ' <>', 1x, 5(I5,3x) )
1006 format(3x, I5, ' <>', 1x, 6(I5,3x) )
1007 format(3x, I5, ' <>', 1x, 7(I5,3x) )
1008 format(3x, I5, ' <>', 1x, 8(I5,3x) )
1009 format(3x, I5, ' <>', 1x, 9(I5,3x) )
1010 format(3x, I5, ' <>', 1x, 10(I5,3x) )
2000 format(7x, (80('.')))

END SUBROUTINE print_int_vec

END MODULE GALAHAD_S2QP_double


