! THIS VERSION: GALAHAD 2.5 - 20/07/2012 AT 08:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ F A S T R   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!                      Leyffer/Munson for Argonne Nationa Laboratory
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.0. May 25th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FASTR_double

!     ----------------------------------------------------------
!    |                                                          |
!    | FASTr, a Filter Active-Set Trust-region method for       |
!    |  nonlinear optimization                                  |
!    |                                                          |
!    | Aim: to find a (local) minimizer of the nonlinear        |
!    | programming problem                                      |
!    |                                                          |
!    |  minimize               f (x)                            |
!    |  subject to          a_i^T x   = b_i^e    i in E_l       !
!    |             c_i^l <= a_i^T x  <= b_i^u    i in I_l       |
!    |                       c_i (x)  = c_i^e    i in E_g       |
!    |             c_i^l <=  c_i (x) <= c_i^u    i in I_g       |
!    |  and          x^l <=       x  <= x^u                     |
!    |                                                          |
!     ----------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_CQP_double
     USE GALAHAD_EQP_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_STRING_double
     USE GALAHAD_FILTER_double
     USE GALAHAD_OPT_double
     USE GALAHAD_NORMS_double, ONLY: TWO_norm, INFINITY_norm
!PLPLOT USE PLPLOT

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: FASTR_initialize, FASTR_read_specfile, FASTR_solve,             &
               FASTR_terminate, NLPT_problem_type, NLPT_userdata_type,         &
               SMT_type, SMT_put

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER :: wsout = 0
!    INTEGER, PARAMETER :: wsout = 78
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
     REAL ( KIND = wp ), PARAMETER :: ten10 = ten ** 10
     REAL ( KIND = wp ), PARAMETER :: sixteenth = 0.0625_wp
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: mu_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: y_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: z_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: min_step_eqp = teneps
     LOGICAL :: add_all_to_filter = .FALSE.
     LOGICAL :: got_hessian = .TRUE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FASTR_control_type

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

!   the Lagrange multiplier estimate used. Possible choices are:
!    0 = autimatic
!    1 = first-order
!    2 = second-order
!    3 = least-squares

       INTEGER :: multipliers = 0

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

!  restoration convergence tolerance

       REAL ( KIND = wp ) :: stop_p_restoration = epsmch

!   initial values for the trust-region radiii for the RLP (activity 
!    determination) and EQP (step dtermination) phases

       REAL ( KIND = wp ) :: initial_radius_rlp = one
       REAL ( KIND = wp ) :: initial_radius_eqp = ten

!   on very successful iterations, the RLP trust-region radius will be 
!    increased by the factor %rlp_radius_increase, while if the iteration is 
!    unsucceful, the radius will be decreased by a factor %rlp_radius_reduce
!    but no more than %rlp_radius_reduce_max

       REAL ( KIND = wp ) :: rlp_radius_increase = ten
       REAL ( KIND = wp ) :: rlp_radius_reduce = tenth
       REAL ( KIND = wp ) :: rlp_radius_reduce_max = point01

!   on very successful iterations, the EQP trust-region radius will be 
!    increased by the factor %eqp_radius_increase, while if the iteration is 
!    unsucceful, the radius will be decreased by a factor %eqp_radius_reduce
!    but no more than %eqp_radius_reduce_max

       REAL ( KIND = wp ) :: eqp_radius_increase = two
       REAL ( KIND = wp ) :: eqp_radius_reduce = half
       REAL ( KIND = wp ) :: eqp_radius_reduce_max = sixteenth

!   the maximum infeasibility tolerated will be the larger of 
!    max_absolute_infeasibility and max_relative_infeasibility 
!    times the initial infeasibility

       REAL ( KIND = wp ) :: max_absolute_infeasibility = ten
       REAL ( KIND = wp ) :: max_relative_infeasibility = ten

!   a new point (o,v) will be acceptable to the filter (o_i,v_i) if
!      v < beta_filter v_i or o < o_i - gamma_filter beta_filter v_i
!    where 0 < gamma_filter < beta_filter < 1

       REAL ( KIND = wp ) :: beta_filter = one - point01
       REAL ( KIND = wp ) :: gamma_filter = point01

!   a potential filter point will be added to the filter whenever the linear 
!    decrease predicted by the RLP is smaller than delta_feas times
!    the square of the current violation

       REAL ( KIND = wp ) :: delta_feas = tenm5

!   a potential filter point whose linear decrease predicted by the RLP
!    is larger than the above will only be accepted if the actual decrease
!    f - f(x_new) is larger than eta_successful times that predicted by a
!    quadratic model of the decrease

       REAL ( KIND = wp ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = wp ) :: eta_very_successful = point9

!  zero Jacobian entry tolerance

       REAL ( KIND = wp ) :: jacobian_zero_tolerance = epsmch

!   if the EQP step is unsuccessful, should the RLP step be tried?

       LOGICAL :: try_rlp_step = .TRUE.

!   fulsol specifies whether the full solution or only highlights will be
!    printed

       LOGICAL :: fulsol = .TRUE.

!   choose between a primal or a dual formulation of the active-set
!    selection quadratic program

       LOGICAL :: primal_qp = .TRUE.

!   is an interior-point or a working set QP solver appropriate?

       LOGICAL :: interior_point_solver = .FALSE.

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!   should the linear constraints be trated explicitly?

       LOGICAL :: explicit_linear_constraints = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for EQP

       TYPE ( EQP_control_type ) :: EQP_control        

!  control parameters for CQP

       TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for FILTER

       TYPE ( FILTER_control_type ) :: FILTER_control

     END TYPE FASTR_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: FASTR_time_type

!  the total CPU time spent in the package

       REAL :: total = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

       REAL :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

       REAL :: factorize = 0.0

!  the CPU time spent computing the search direction

       REAL :: solve = 0.0

!  the CPU time spent in the restoration phase

       REAL :: restoration = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent analysing the required matrices prior to factorization

       REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

       REAL ( KIND = wp ) :: clock_solve = 0.0

!  the clock time spent spent in the restoration phase

       REAL ( KIND = wp ) :: clock_restoration = 0.0

     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FASTR_inform_type

!  return status. See TRU_solve for details

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

       INTEGER :: factorizations = 0

!  the number of factorizations that modified the original matrix

       INTEGER :: modifications = 0

!  the return status from the factorization

       INTEGER :: factorization_status = 0

!   the maximum number of entries in the factors

       INTEGER ( KIND = long ) :: max_entries_factors = 0

!  the total integer workspace required for the factorization

       INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

       INTEGER :: factorization_real = - 1

!  the value of the objective function at the best estimate of the solution 
!   determined by FASTR_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

       REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

       REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

       REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  same for the restoration phase

       REAL ( KIND = wp ) :: obj_restoration  = HUGE( one )
       REAL ( KIND = wp ) :: primal_infeasibility_rest  = HUGE( one )
       REAL ( KIND = wp ) :: dual_infeasibility_rest = HUGE( one )
       REAL ( KIND = wp ) :: complementary_slackness_rest = HUGE( one )

!  timings (see above)

       TYPE ( FASTR_time_type ) :: time

!  inform parameters for EQP

       TYPE ( EQP_inform_type ) :: EQP_inform

!  inform parameters for CQP

       TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for FILTER

       TYPE ( FILTER_inform_type ) :: FILTER_inform
       TYPE ( FILTER_inform_type ) :: FILTER_restoration_inform

     END TYPE FASTR_inform_type

!  - - - - - - - - - - -
!   filter derived type 
!  - - - - - - - - - - -

     TYPE, PUBLIC :: FASTR_filter_type
       REAL ( KIND = wp ) :: o, v
     END TYPE FASTR_filter_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: FASTR_data_type
       INTEGER :: branch, branch_restoration, eval_status, n_filter, max_filter
       INTEGER :: n, J_ne, H_ne, n_filter_restoration, max_filter_restoration
       INTEGER :: n_restoration, J_ne_restoration, H_ne_restoration, m_linear
       INTEGER :: out, start_print, stop_print, print_level
       INTEGER :: print_level_cqp, print_level_eqp, print_level_filter
       INTEGER :: print_level_sbls, print_level_gltr
       INTEGER :: n_mult_wrong_sign, n_pr_max, iter_restoration
!PLPLOT  INTEGER :: n_success
       REAL :: time_start, time_record, time_now
       REAL :: time_analyse, time_factorize
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
       REAL ( KIND = wp ) :: clock_analyse, clock_factorize
       REAL ( KIND = wp ) :: gtd, norm_dlp, old_norm_dlp, step, step_rlp
       REAL ( KIND = wp ) :: norm_dlp_restoration, old_norm_dlp_restoration
       REAL ( KIND = wp ) :: stop_p, stop_d, stop_c, mu, mu_new, mu_max, pr_max
       REAL ( KIND = wp ) :: stop_p_rest, stop_d_rest, stop_c_rest
       REAL ( KIND = wp ) :: f_best, v_best, pr_best, delta_l, delta_q
       REAL ( KIND = wp ) :: obj_eqp, primal_infeasibility_eqp, gtd_eqp
       REAL ( KIND = wp ) :: radius_eqp, norm_deqp, dtjc_eqp, step_eqp
!PLPLOT  REAL ( KIND = wp ) :: v_min, v_max, v_mine, v_maxe
!PLPLOT  REAL ( KIND = wp ) :: o_min, o_max, o_opt, o_minl, o_maxl
       LOGICAL :: set_printt, set_printi, set_printm, set_printw, set_printd
       LOGICAL :: printt, printi, printm, printw, printd, print_1st_header
       LOGICAL :: print_iteration_header, new_point, x_best_set, restoration
       LOGICAL :: first_filter_in_use, first_filter_in_use_restoration
       LOGICAL :: reverse_fc, reverse_gj, reverse_hl, reverse_hlprod
       LOGICAL :: successful, very_successful, new_gradient, take_eqp_step
       LOGICAL :: restoration_restoration, x_feas, any_linear, infeasible
       CHARACTER ( LEN = 1 ) :: bdry, d_type, it_type, rest
       CHARACTER ( LEN = 3 ) :: d_name
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: XFREE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX_trial_rlp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DS_trial_rlp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Hd
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ATDY
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_p
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       TYPE ( FASTR_control_type ) :: control
       TYPE ( QPT_problem_type ) :: CQP_prob, EQP_prob
       TYPE ( EQP_data_type ) :: EQP_data
       TYPE ( CQP_data_type ) :: CQP_data
       TYPE ( FILTER_data_type ) :: FILTER_data
       TYPE ( FILTER_data_type ) :: FILTER_restoration_data
!PLPLOT  TYPE ( FASTR_filter_type ), ALLOCATABLE, DIMENSION( : ) :: success

     END TYPE FASTR_data_type

   CONTAINS

!-*-*-*-*  G A L A H A D -  FASTR_initialize  S U B R O U T I N E -*-*-*-*

     SUBROUTINE FASTR_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for FASTR controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FASTR_data_type ), INTENT( INOUT ) :: data
     TYPE ( FASTR_control_type ), INTENT( OUT ) :: control
     TYPE ( FASTR_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  Set real control parameters

     control%stop_abs_p = epsmch ** 0.33
!    control%stop_rel_p = epsmch ** 0.33
     control%stop_abs_c = epsmch ** 0.33
!    control%stop_rel_c = epsmch ** 0.33
     control%stop_abs_d = epsmch ** 0.33
!    control%stop_rel_d = epsmch ** 0.33
     control%jacobian_zero_tolerance = epsmch ** 0.33

!  Initialize EQP data

     CALL EQP_initialize( data%EQP_data, control%EQP_control,                  &
                          inform%EQP_inform )
     control%EQP_control%prefix = '" - EQP:"                     '
     control%EQP_control%SBLS_control%prefix = '" -- SBLS:"                   '

!  Initialize QP data

     CALL CQP_initialize( data%CQP_data, control%CQP_control,                  &
                          inform%CQP_inform )
     control%CQP_control%prefix = '" - CQP:"                     '

!  Initialize FILTER data

     control%FILTER_control%prefix = '" - FILTER:"                  '

     RETURN

!  End of subroutine FASTR_initialize

     END SUBROUTINE FASTR_initialize

!-*-*-   F A S T R _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE FASTR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by FASTR_initialize could (roughly) 
!  have been set as:

! BEGIN FASTR SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  iterations-between-printing                    1
!  lagrange-multiplier-estimates-used             0
!  absolute-primal-accuracy                       6.0D-6
!  relative-primal-accuracy                       2.0D-16
!  absolute-dual-accuracy                         6.0D-6
!  relative-dual-accuracy                         2.0D-16
!  absolute-complementary-slackness-accuracy      6.0D-6
!  relative-complementary-slackness-accuracy      2.0D-16
!  restoration-accuracy-required                  6.0D-6
!  initial-rlp-radius                             1.0
!  initial-eqp-radius                             1.0
!  rlp-radius-increase-factor                     2.0
!  rlp-radius-decrease-factor                     0.5
!  rlp-radius-maximum-decrease-factor             0.0625
!  eqp-radius-increase-factor                     2.0
!  eqp-radius-decrease-factor                     0.5
!  eqp-radius-maximum-decrease-factor             0.0625
!  filter-constraint-improvement-factor           0.99
!  filter-function-improvement-factor             0.01
!  add-to-filter-tolerance                        1.0D-5
!  maximum-absolute-infeasibility                 10.0
!  maximum-relative-infeasibility                 10.0
!  successful-iteration-tolerance                 0.01
!  very-successful-iteration-tolerance            0.9
!  jacobian-zero-tolerance                        1.0D-8
!  try-rlp-step                                   YES
!  print-full-solution                            YES
!  explicit-linear-constraints                    NO
!  solve-primal-qp                                YES
!  interior-point-solver                          NO
!  space-critical                                 NO
!  deallocate-error-fatal                         NO
!  alive-filename                                 ALIVE.d
! END FASTR SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FASTR_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: alive_unit = out + 1
     INTEGER, PARAMETER :: print_level = alive_unit + 1
     INTEGER, PARAMETER :: maxit = print_level + 1
     INTEGER, PARAMETER :: start_print = maxit + 1
     INTEGER, PARAMETER :: stop_print  = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print  + 1
     INTEGER, PARAMETER :: multipliers = print_gap + 1
     INTEGER, PARAMETER :: stop_abs_p = multipliers + 1
     INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
     INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
     INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
     INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
     INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
     INTEGER, PARAMETER :: stop_p_restoration = stop_rel_c + 1
     INTEGER, PARAMETER :: initial_radius_rlp = stop_p_restoration + 1 
     INTEGER, PARAMETER :: initial_radius_eqp = initial_radius_rlp + 1
     INTEGER, PARAMETER :: rlp_radius_increase = initial_radius_eqp + 1
     INTEGER, PARAMETER :: rlp_radius_reduce = rlp_radius_increase + 1
     INTEGER, PARAMETER :: rlp_radius_reduce_max = rlp_radius_reduce + 1
     INTEGER, PARAMETER :: eqp_radius_increase = rlp_radius_reduce_max + 1
     INTEGER, PARAMETER :: eqp_radius_reduce = eqp_radius_increase + 1
     INTEGER, PARAMETER :: eqp_radius_reduce_max = eqp_radius_reduce + 1
     INTEGER, PARAMETER :: max_absolute_infeasibility                          &
                             = eqp_radius_reduce_max + 1
     INTEGER, PARAMETER :: max_relative_infeasibility                          &
                             = max_absolute_infeasibility + 1
     INTEGER, PARAMETER :: beta_filter = max_relative_infeasibility + 1
     INTEGER, PARAMETER :: gamma_filter = beta_filter + 1
     INTEGER, PARAMETER :: delta_feas = gamma_filter + 1
     INTEGER, PARAMETER :: eta_successful = delta_feas + 1
     INTEGER, PARAMETER :: eta_very_successful = eta_successful + 1
     INTEGER, PARAMETER :: jacobian_zero_tolerance = eta_very_successful + 1
     INTEGER, PARAMETER :: try_rlp_step = jacobian_zero_tolerance + 1
     INTEGER, PARAMETER :: fulsol = try_rlp_step + 1
     INTEGER, PARAMETER :: explicit_linear_constraints = fulsol  + 1
     INTEGER, PARAMETER :: interior_point_solver                               &
                             = explicit_linear_constraints + 1
     INTEGER, PARAMETER :: space_critical = interior_point_solver + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: primal_qp = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: alive_file = primal_qp + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 5 ), PARAMETER :: specname = 'FASTR'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( alive_unit )%keyword = 'alive-device'
     spec( print_level )%keyword = 'print-level' 
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print  )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( multipliers )%keyword = 'lagrange-multiplier-estimates-used'

!  Real key-words

     spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
     spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
     spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
     spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
     spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
     spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
     spec( stop_p_restoration )%keyword = 'restoration-accuracy-required'
     spec( initial_radius_rlp )%keyword = 'initial-rlp-radius'
     spec( initial_radius_eqp )%keyword = 'initial-eqp-radius'
     spec( rlp_radius_increase )%keyword = 'rlp-radius-increase-factor'
     spec( rlp_radius_reduce )%keyword = 'rlp-radius-decrease-factor'
     spec( rlp_radius_reduce_max )%keyword                                     &
       = 'rlp-radius-maximum-decrease-factor'
     spec( eqp_radius_increase )%keyword = 'eqp-radius-increase-factor'
     spec( eqp_radius_reduce )%keyword = 'eqp-radius-decrease-factor'
     spec( eqp_radius_reduce_max )%keyword                                     &
       = 'eqp-radius-maximum-decrease-factor'
     spec( max_absolute_infeasibility )%keyword                                &
       = 'maximum-absolute-infeasibility'
     spec( max_relative_infeasibility )%keyword                                &
       = 'maximum-relative-infeasibility'
     spec( beta_filter )%keyword = 'filter-constraint-improvement-factor'
     spec( gamma_filter )%keyword = 'filter-function-improvement-factor'
     spec( delta_feas )%keyword = 'add-to-filter-tolerance'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( jacobian_zero_tolerance )%keyword = 'jacobian-zero-tolerance'

!  Logical key-words

     spec( try_rlp_step )%keyword = 'try-rlp-step'
     spec( fulsol )%keyword = 'print-full-solution'
     spec( explicit_linear_constraints )%keyword = 'explicit-linear-constraints'
     spec( interior_point_solver )%keyword = 'interior-point-solver'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
     spec( primal_qp )%keyword = 'solve-primal-qp'

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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
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
     CALL SPECFILE_assign_value( spec( multipliers ),                          &
                                 control%multipliers,                          &
                                 control%error )

!  Set real values

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
     CALL SPECFILE_assign_value( spec( stop_p_restoration ),                   &
                                 control%stop_p_restoration,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_radius_rlp ),                   &
                                 control%initial_radius_rlp,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_radius_eqp ),                   &
                                 control%initial_radius_eqp,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rlp_radius_increase ),                  &
                                 control%rlp_radius_increase,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rlp_radius_reduce ),                    &
                                 control%rlp_radius_reduce,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rlp_radius_reduce_max ),                &
                                 control%rlp_radius_reduce_max,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eqp_radius_increase ),                  &
                                 control%eqp_radius_increase,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eqp_radius_reduce ),                    &
                                 control%eqp_radius_reduce,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eqp_radius_reduce_max ),                &
                                 control%eqp_radius_reduce_max,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_absolute_infeasibility ),           &
                                 control%max_absolute_infeasibility,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_relative_infeasibility ),           &
                                 control%max_relative_infeasibility,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( beta_filter ),                          &
                                 control%beta_filter,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_filter ),                         &
                                 control%gamma_filter,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( delta_feas ),                           &
                                 control%delta_feas,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_successful ),                       &
                                 control%eta_successful,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_very_successful ),                  &
                                 control%eta_very_successful,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( jacobian_zero_tolerance ),              &
                                 control%jacobian_zero_tolerance,              &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( try_rlp_step ),                         &
                                 control%try_rlp_step,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( fulsol ),                               &
                                 control%fulsol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( explicit_linear_constraints ),          &
                                 control%explicit_linear_constraints,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( interior_point_solver ),                &
                                 control%interior_point_solver,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( primal_qp ),                            &
                                 control%primal_qp,                            &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( alive_file ),                           &
                                 control%alive_file,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for CQP and EQP

     IF ( PRESENT( alt_specname ) ) THEN
       CALL EQP_read_specfile( control%EQP_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-EQP' )
       CALL CQP_read_specfile( control%CQP_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-CQP' )
       CALL FILTER_read_specfile( control%FILTER_control, device,              &
                               alt_specname = TRIM( alt_specname ) // '-FILTER')
     ELSE
       CALL EQP_read_specfile( control%EQP_control, device )
       CALL CQP_read_specfile( control%CQP_control, device )
       CALL FILTER_read_specfile( control%FILTER_control, device )
     END IF

     RETURN

     END SUBROUTINE FASTR_read_specfile

!-*-*-*-*  G A L A H A D -  F A S T R _ s o l v e  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE FASTR_solve( nlp, control, inform, data, userdata,             &
                             eval_FC, eval_GJ, eval_HL, eval_HLPROD )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  FASTR_solve, a method for finding a local minimizer of a function subject 
!  to general constraints and simple bounds on the sizes of the variables.

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_FASTR. 
!
!  ** NB. default real/complex means double precision real/complex in 
!  ** GALAHAD_FASTR_double
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
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains 
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
!    [7. The user should compute the product u = P(x)v of their preconditioner 
!        P(x) at the point x indicated in nlp%X with the vector v and then
!        re-enter the subroutine. The vectors v is given in data%V, the
!        resulting vector u = P(x)v should be set in data%U and 
!        data%eval_status should be set to 0. If the user is unable to evaluate
!        the product - for instance, if a component of the preconditioner is 
!        undefined at x - the user need not set data%U, but should then set 
!        data%eval_status to a non-zero value. *** IGNORE - NOT IMPLEMENTED ***]
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
!   be set to a nonzero value. If eval_FC is not present, FASTR_solve will
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
!   nonzero value. If eval_GJ is not present, FASTR_solve will return to the 
!   user with inform%status = 3 or 5 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated 
!   at x=X and y=Y must be returned in H_val in the same order as presented in 
!   nlp%H, and the status variable set to 0. If the evaluation is impossible 
!   at X, status should be set to a nonzero value. If eval_HL is not present, 
!   FASTR_solve will return to the user with inform%status = 4 or 5 each time 
!   an evaluation is required.
!
!  eval_HLPROD is an optional subroutine which if present must have
!   the arguments given below (see the interface blocks). The sum 
!   u + nabla_xx ( f(x) - sum_i=1^m y_i c_i(x) ) v of the product of the Hessian
!   nabla_xx f(x) + sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated 
!   at x=X and y=Y with the vector v=V and the vector u=U must be returned in U,
!   and the status variable set to 0. If the evaluation is impossible at X, 
!   status should be set to a nonzero value. If eval_HPROD is not present, 
!   FASTR_solve will return to the user with inform%status = 6 each time an 
!   evaluation is required.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the 
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result 
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC 
!   is not present, TRU_solve will return to the user with inform%status = 7
!   each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( FASTR_control_type ), INTENT( IN ) :: control
     TYPE ( FASTR_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( FASTR_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_GJ, eval_HL, eval_HLPROD

!----------------------------------
!   I n t e r f a c e   B l o c k s 
!----------------------------------

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

       SUBROUTINE eval_GJ( status, X, userdata, G, J_val )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
         REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
         REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ

       SUBROUTINE eval_HL( status, X, Y, userdata, H_val, no_f ) 
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) ::H_val
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
         LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL

       SUBROUTINE eval_HLPROD( status, X, Y, userdata, U, V, no_f, got_h )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
         LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
         LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HLPROD
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ii, ir, ic, j, l, n_mult_wrong_sign_ls
!    INTEGER :: m_l, m_le, m_li, m_ne, m_ni, m_working, n_working
     REAL ( KIND = wp ) :: alpha, max_y, dthd, delta_f, delta_m
     REAL ( KIND = wp ) :: obj_trial, primal_infeasibility_trial, c_required
     REAL ( KIND = wp ) :: dual_infeasibility_ls, dx, dc, y_t, o, v
     REAL ( KIND = wp ) :: complementary_slackness_ls
!    REAL ( KIND = wp ) :: delta_eqp, ratio_norm_dlp
     LOGICAL :: acceptable
     CHARACTER ( LEN = 8 ) :: tr_active
     CHARACTER ( LEN = 80 ) :: array_name

!    TYPE ( FASTR_control_type ) :: control_restoration
!    TYPE ( FASTR_inform_type ) :: inform_restoration

!PLPLOT  INTEGER, PARAMETER :: rect_n = 4
!PLPLOT  INTEGER, PARAMETER :: ffiledevice = 99
!PLPLOT  REAL ( KIND = plflt ), DIMENSION( rect_n ) :: rect_x, rect_y
!PLPLOT  LOGICAL :: ffileexits
!PLPLOT  CHARACTER ( LEN = 10 ) :: titer
!PLPLOT  CHARACTER ( LEN = 30 ), PARAMETER :: ffilename = 'FASTR_filter.data'
!PLPLOT  CHARACTER ( LEN = 80 ) :: tlabel

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 1

     SELECT CASE ( data%branch )
     CASE ( 1 )  ! initialization
       GO TO 10
     CASE ( 2 )  ! initial objective and constraint evaluation
       GO TO 20
     CASE ( 3 )  ! gradient and Jacobian evaluation
       GO TO 130
     CASE ( 4 )  ! Hessian evaluation
       GO TO 140
     CASE ( 5 )  ! Hessian-vector product
       GO TO 150
     CASE ( 6 )  ! objective and constraint evaluation
       GO TO 260
     CASE ( 7 )  ! various problem function evaluations
       GO TO 370
     CASE ( 8 )  ! objective and constraint evaluation
       GO TO 380
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE

     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
     data%control = control
     data%control%CQP_control%SBLS_control%preconditioner = 2

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     inform%iter = 0 ; inform%factorizations = 0 ; inform%modifications = 0 
     inform%f_eval = 0 ; inform%g_eval = 0

     inform%obj = HUGE( one )
     inform%primal_infeasibility = HUGE( one )
     inform%dual_infeasibility = HUGE( one )
     data%f_best = HUGE( one )
     data%v_best = HUGE( one )
     data%pr_best = HUGE( one )

     data%it_type = ' '
     data%bdry = ' '
     data%n_pr_max = 0
     data%rest = ' '

     data%step = zero
     data%norm_dlp = SQRT( HUGE( one ) )
     data%n_mult_wrong_sign = - 1
     data%mu_max = MAX( ten ** 5, data%control%initial_radius_rlp )
     data%mu_new = one
     data%successful = .TRUE.
     data%x_best_set = .FALSE.
     data%x_feas = .FALSE.
     data%new_point = .TRUE.
     data%new_gradient = .TRUE.
     data%restoration = .FALSE.
     IF ( data%control%multipliers < 0 .OR. data%control%multipliers > 3  )    &
        data%control%multipliers = 0

!PLPLOT  data%n_success = - 1

!  decide how much reverse communication is required

     data%reverse_fc = .NOT. PRESENT( eval_FC )
     data%reverse_gj = .NOT. PRESENT( eval_GJ )
     data%reverse_hl = .NOT. PRESENT( eval_HL )
     data%reverse_hlprod = .NOT. PRESENT( eval_HLPROD )

!  ===========================
!  Control the output printing
!  ===========================

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

     data%out = data%control%out

     data%print_level_cqp = data%control%CQP_control%print_level
     data%print_level_eqp = data%control%EQP_control%print_level
     data%print_level_sbls = data%control%EQP_control%SBLS_control%print_level
     data%print_level_gltr = data%control%EQP_control%GLTR_control%print_level

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

!  start setting control parameters

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
       data%control%CQP_control%print_level = data%print_level_cqp
       data%control%CQP_control%SBLS_control%print_level = data%print_level_sbls
       data%control%EQP_control%print_level = data%print_level_eqp
       data%control%EQP_control%SBLS_control%print_level = data%print_level_sbls
       data%control%EQP_control%GLTR_control%print_level = data%print_level_gltr
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
       data%control%CQP_control%print_level = 0
       data%control%CQP_control%SBLS_control%print_level = 0
       data%control%EQP_control%print_level = 0
       data%control%EQP_control%SBLS_control%print_level = 0
       data%control%EQP_control%GLTR_control%print_level = 0
     END IF
     
     data%print_iteration_header = data%print_level > 0
     data%print_1st_header = .TRUE.

     IF ( data%printd ) WRITE( data%out, "( A, ' x ', /, ( 5ES12.4 ) )" )      &
       prefix, nlp%X( : nlp%n )

!  ensure that the constraint bounds are consistent

     DO i = 1, nlp%m
       IF ( nlp%C_u( i ) < nlp%C_l( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds ; GO TO 990
       END IF
     END DO

!  ensure that the variable bounds are consistent

     DO i = 1, nlp%n
       IF ( nlp%X_u( i ) < nlp%X_l( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds ; GO TO 990
       END IF

!  ensure that the starting point satisfies its bounds

       nlp%X( i ) = MIN( MAX( nlp%X_l( i ), nlp%X( i ) ), nlp%X_u( i ) )
     END DO

!  are there linear constraints?

     data%m_linear = COUNT( nlp%LINEAR( : nlp%m ) )
     data%any_linear = data%m_linear > 0 .AND. nlp%m > 0

!  determine how many linear and equality constraints there are

!    m_le = 0 ; m_li = 0 ; m_ne  = 0 ; m_ni = 0

!    DO i = 1, nlp%m
!      IF ( nlp%LINEAR( i ) ) THEN
!        IF ( nlp%EQUATION( i ) ) THEN
!          m_le = m_le + 1
!        ELSE
!          m_li = m_li + 1
!        END IF
!      ELSE
!        IF ( nlp%EQUATION( i ) ) THEN
!          m_ne = m_ne + 1
!        ELSE
!          m_ni = m_ni + 1
!        END IF
!      END IF
!    END DO

!  starting addresses for constraints:

!    -----------------------------------------------------------------------
!    | linear equal | linear inequal | nonlinear equal | nonlinear inequal |
!    -----------------------------------------------------------------------
!     ^              ^              ^ ^                 ^                   ^
!     |              |            m_l |                 |                   | 
!    m_le           m_li             m_ne              m_ni                 m 

!    m_l = m_le + m_li
!    m_ni = nlp%m - m_ni + 1
!    m_ne = m_l + 1
!    m_li = m_le + 1
!    m_le = 1
!    write(6,"( 6I8 )" ) m_le, m_li, m_l, m_ne, m_ni, nlp%m 

!  set starting dual variables

     DO i = 1, nlp%n
       IF ( nlp%X( i ) - nlp%X_l( i ) < zero ) THEN
         nlp%Z( i ) = one
       ELSE IF ( nlp%X( i ) - nlp%X_u( i ) > zero ) THEN
         nlp%Z( i ) = - one
       ELSE
         nlp%Z( i ) = zero
       END IF
     END DO

!  record the number of variables for the true and restoration problems

     data%n = nlp%n
     data%n_restoration = data%n + nlp%m

!  determine how many nonzeros are required to store the matrix of gradients
!  of the objective function and constraints, when the matrix is stored in 
!  "co-ordinate" format; allow space for any restoration phase

!    CALL CDIMSJ( data%J_ne )
     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%J_ne = nlp%J%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%J_ne = nlp%J%ptr( nlp%m + 1 ) - 1 
     CASE ( 'DENSE' ) 
       data%J_ne = nlp%m * nlp%n
     END SELECT
     data%J_ne_restoration = data%J_ne + nlp%m

!  determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as a sparse matrix in "co-ordinate" 
!  format; allow space for any restoration phase
 
!    CALL CDIMSH( data%H_ne )
     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%H_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%H_ne = nlp%H%ptr( nlp%m + 1 ) - 1 
     CASE ( 'DENSE' ) 
       data%H_ne = ( nlp%n * ( nlp%n+ 1 ) ) / 2
     END SELECT
     data%H_ne_restoration = data%H_ne + nlp%m

!  allocate space to hold the problem data

!    array_name = 'fastr: nlp%g'
!    CALL SPACE_resize_array( nlp%n, nlp%g, inform%status,                     &
!           inform%alloc_status, array_name = array_name,                      &
!           deallocate_error_fatal = data%control%deallocate_error_fatal,      &
!           exact_size = data%control%space_critical,                          &
!           bad_alloc = inform%bad_alloc, out = data%control%error )
!    IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: nlp%gL'
     CALL SPACE_resize_array( data%n_restoration, nlp%gL, inform%status,       &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%G_p'
     CALL SPACE_resize_array( data%n_restoration, data%G_p, inform%status,     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

!    data%G_p =  nlp%gL

!    array_name = 'fastr: data%CQP_prob%H%row'
!    CALL SPACE_resize_array( data%H_ne, data%CQP_prob%H%row, inform%status,   &
!           inform%alloc_status, array_name = array_name,                      &
!           deallocate_error_fatal = data%control%deallocate_error_fatal,      &
!           exact_size = data%control%space_critical,                          &
!           bad_alloc = inform%bad_alloc, out = data%control%error )
!    IF ( inform%status /= 0 ) GO TO 910

!    array_name = 'fastr: data%CQP_prob%H%col'
!    CALL SPACE_resize_array( data%H_ne, data%CQP_prob%H%col, inform%status,   &
!           inform%alloc_status, array_name = array_name,                      &
!           deallocate_error_fatal = data%control%deallocate_error_fatal,      &
!           exact_size = data%control%space_critical,                          &
!           bad_alloc = inform%bad_alloc, out = data%control%error )
!    IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%H%val'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%H%val,         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%G'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%G,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%A%row'
     CALL SPACE_resize_array( data%J_ne_restoration, data%CQP_prob%A%row,      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%A%col'
     CALL SPACE_resize_array( data%J_ne_restoration, data%CQP_prob%A%col,      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%A%val'
     CALL SPACE_resize_array( data%J_ne_restoration, data%CQP_prob%A%val,      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%C_l'
     CALL SPACE_resize_array( nlp%m, data%CQP_prob%C_l, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%C_u'
     CALL SPACE_resize_array( nlp%m, data%CQP_prob%C_u, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%X_l'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%X_l,           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%X_u'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%X_u,           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%X'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%X,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%C'
     CALL SPACE_resize_array( nlp%m, data%CQP_prob%C, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: nlp%C_scale'
     CALL SPACE_resize_array( nlp%m, nlp%C_scale, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%Y'
     CALL SPACE_resize_array( nlp%m, data%CQP_prob%Y, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%CQP_prob%Z'
     CALL SPACE_resize_array( data%n_restoration, data%CQP_prob%Z,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%H%row'
     CALL SPACE_resize_array( data%H_ne_restoration, data%EQP_prob%H%row,      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%H%col'
     CALL SPACE_resize_array( data%H_ne_restoration, data%EQP_prob%H%col,      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%H%val'
     CALL SPACE_resize_array(                                                  &
            MAX( data%H_ne_restoration, data%n_restoration ),                  &
            data%EQP_prob%H%val,                                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%G'
     CALL SPACE_resize_array( data%n_restoration, data%EQP_prob%G,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%A%row'
     CALL SPACE_resize_array( data%J_ne_restoration + data%n_restoration,      &
            data%EQP_prob%A%row,                                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%A%col'
     CALL SPACE_resize_array( data%J_ne_restoration + data%n_restoration,      &
            data%EQP_prob%A%col,                                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%A%val'
     CALL SPACE_resize_array( data%J_ne_restoration + data%n_restoration,      &
            data%EQP_prob%A%val,                                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%X'
     CALL SPACE_resize_array( data%n_restoration, data%EQP_prob%X,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%C'
     CALL SPACE_resize_array( nlp%m + data%n_restoration, data%EQP_prob%C,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%EQP_prob%Y'
     CALL SPACE_resize_array( nlp%m + data%n_restoration, data%EQP_prob%Y,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: nlp%C_status'
     CALL SPACE_resize_array( nlp%m, nlp%C_status, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: nlp%X_status'
     CALL SPACE_resize_array( data%n_restoration, nlp%X_status,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%U'
     CALL SPACE_resize_array( nlp%n, data%U, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%V'
     CALL SPACE_resize_array( nlp%n, data%V, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%X_trial'
     CALL SPACE_resize_array( nlp%n, data%X_trial, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%X_best'
     CALL SPACE_resize_array( nlp%n, data%X_best, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%C_best'
     CALL SPACE_resize_array( nlp%m, data%C_best, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%DX_trial_rlp'
     CALL SPACE_resize_array( nlp%n, data%DX_trial_rlp, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%C_trial'
     CALL SPACE_resize_array( nlp%m, data%C_trial, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910
!    data%Y_p = data%C_trial

     array_name = 'fastr: data%X_p'
     CALL SPACE_resize_array( nlp%n, data%X_p, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%Y_p'
     CALL SPACE_resize_array( nlp%m, data%Y_p, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%Z_p'
     CALL SPACE_resize_array( nlp%n, data%Z_p, inform%status,                  &
            inform%alloc_status,  array_name = array_name,                     &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%C_p'
     CALL SPACE_resize_array( nlp%m, data%C_p, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%DY'
     CALL SPACE_resize_array( nlp%m, data%DY, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%DZ'
     CALL SPACE_resize_array( data%n_restoration, data%DZ, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%ATDY'
     CALL SPACE_resize_array( data%n_restoration, data%ATDY, inform%status,    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set up space for the filter
     
     CALL FILTER_initialize_filter( data%FILTER_data,                          &
                                    data%control%FILTER_control,               &
                                    inform%FILTER_inform )
     IF ( inform%FILTER_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_inform%status
       inform%alloc_status = inform%FILTER_inform%alloc_status
       GO TO 910
     END IF

     IF ( data%printi ) THEN
       WRITE( data%out, "( /, A, ' Solver: FASTr',                             &
!     &  /, A, ' Problem: ', 7X, A, //, A, ' There ', A, 1X, I0, ' variable',  &
      &  ', problem: ', A, //, A, 1X, I0, ' variable',                         &
      &         A, ' and ', I0, ' general constraint', A )", advance = 'no' )  &
!       prefix, prefix, nlp%pname, prefix, TRIM( STRING_are( nlp%n ) ), nlp%n, &
        prefix, nlp%pname, prefix, nlp%n,                                      &
        TRIM( STRING_pleural( nlp%n ) ), nlp%m, TRIM( STRING_pleural( nlp%m ) )
       IF ( data%any_linear ) THEN
         WRITE( data%out, "( ', of which ', I0, 1X, A, ' linear' )" )          &
          data%m_linear, TRIM( STRING_are( data%m_linear ) )
       ELSE
         WRITE( data%out, "( '' )" )
       END IF
     END IF

!  evaluate the objective and general constraint function values
   
     IF ( data%reverse_fc ) THEN
       data%branch = 2 ; inform%status = 2 ; RETURN
     ELSE  
       CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1

     inform%primal_infeasibility =                                             &
       OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                      &
                                 nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
     data%pr_max = MAX( data%control%max_absolute_infeasibility,               &
                        data%control%max_relative_infeasibility                &
                          * inform%primal_infeasibility ) 

!  record the primal infeasiblity stopping tolerance

     data%stop_p = MAX( control%stop_abs_p,                                    &
                        control%stop_rel_p * inform%primal_infeasibility )

!PLPLOT  data%v_min = MAX( - 18.0_plflt, LOG10( data%stop_p ) )
!PLPLOT  data%v_mine = ten ** data%v_min
!PLPLOT  data%v_max = LOG10( data%pr_max )
!PLPLOT  data%v_maxe = ten ** data%v_max
!PLPLOT  INQUIRE( FILE = ffilename, EXIST = ffileexits )
!PLPLOT  IF ( ffileexits ) THEN
!PLPLOT    OPEN( ffiledevice, FILE = ffilename, FORM = 'FORMATTED',            &
!PLPLOT         STATUS = 'OLD', IOSTAT = i )
!PLPLOT    READ( ffiledevice, * ) data%o_min, data%o_max, data%o_opt
!PLPLOT  ELSE
!PLPLOT    OPEN( ffiledevice, FILE = ffilename, FORM = 'FORMATTED',            &
!PLPLOT          STATUS = 'NEW', IOSTAT = i )
!PLPLOT    data%o_min = -20.0_plflt ; data%o_max = 20.0_plflt
!PLPLOT    data%o_opt = 0.0_plflt
!PLPLOT  END IF
!PLPLOT  IF ( tenm10 < data%o_opt - data%o_min ) THEN
!PLPLOT    data%o_minl = - LOG10( ten10 * ( data%o_opt - data%o_min ) )
!PLPLOT  ELSE
!PLPLOT    data%o_minl = - 1.0
!PLPLOT  END IF
!PLPLOT  IF ( tenm10 < data%o_max - data%o_opt ) THEN
!PLPLOT    data%o_maxl = LOG10( ten10 * ( data%o_max - data%o_opt ) )
!PLPLOT  ELSE
!PLPLOT    data%o_maxl = 1.0
!PLPLOT  END IF
!PLPLOT  CALL plparseopts( PL_PARSE_FULL ) ! parsing of command-line arguments
!PLPLOT  CALL plsdev( 'psc' )       !  set output type to colour postscript
!PLPLOT  CALL plsfnam( 'filter_plplot.ps' )  !  set output file name
! colours -
!   0 black 1 red 2 yellow 3 green 4 aquamarine 5 pink 6 wheat 7 grey 8 brown
!   9 blue 10 BlueViolet 11 cyan 12 turquoise 13 magenta 14 salmon 15 white
!PLPLOT  CALL plscolbg( 255, 255, 255 ) !  set background colour to white
!PLPLOT  CALL plscol0( 15, 0, 0, 0 )    !  reset internal colour 15 to black
!PLPLOT  CALL plinit( )                 !  initialize plplot
!PLPLOT  CALL plcol0( 15 )              !  set foreground to black
!PLPLOT  CALL plsesc( 64 )              !  set the ASCII value for character @
!PLPLOT  ALLOCATE( data%success( 0 : control%maxit ) )
!PLPLOT  data%n_success = 0
!PLPLOT  data%success( 0 )%o = inform%obj
!PLPLOT  data%success( 0 )%v = inform%primal_infeasibility

!  set up structural data for the linearized constraints in co-ordinate form

     data%CQP_prob%n = data%n ; data%CQP_prob%m = nlp%m
     data%CQP_prob%A%n = data%n ; data%CQP_prob%A%m = nlp%m

     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%CQP_prob%A%ne = nlp%J%ne
       IF ( data%CQP_prob%A%ne > 0 ) THEN
         data%CQP_prob%A%row( : data%CQP_prob%A%ne )                           &
           = nlp%J%row( : data%CQP_prob%A%ne )
         data%CQP_prob%A%col( : data%CQP_prob%A%ne )                           &
           = nlp%J%col( : data%CQP_prob%A%ne )
       END IF
     CASE ( 'SPARSE_BY_ROWS' )
       data%CQP_prob%A%ne = nlp%J%ptr( nlp%m + 1 ) - 1 
       IF ( data%CQP_prob%A%ne > 0 ) THEN
         DO i = 1, nlp%m
           data%CQP_prob%A%row( nlp%J%ptr( i ) : nlp%J%ptr( i + 1 ) - 1 ) = i
         END DO
         data%CQP_prob%A%col( : data%CQP_prob%A%ne )                           &
           = nlp%J%col( : data%CQP_prob%A%ne )
       END IF
     CASE ( 'DENSE' ) 
       data%CQP_prob%A%ne = 0
       DO i = 1, nlp%m
         DO j = 1, nlp%n
           data%CQP_prob%A%ne = data%CQP_prob%A%ne + 1
           data%CQP_prob%A%row( data%CQP_prob%A%ne ) = i
           data%CQP_prob%A%col( data%CQP_prob%A%ne ) = j
         END DO
       END DO
     END SELECT

!  allow room for any extra entries required by the restoration phase 

     DO i = 1, nlp%m
       data%CQP_prob%A%row( data%J_ne + i ) = i
       data%CQP_prob%A%col( data%J_ne + i ) = nlp%n + i
     END DO

     array_name = 'fastr: data%CQP_prob%A%type'
     CALL SPACE_dealloc_array( data%CQP_prob%A%type,                           &
        inform%status, inform%alloc_status,                                    &
        array_name = array_name, out = data%control%error )
     CALL SMT_put( data%CQP_prob%A%type, 'COORDINATE', inform%alloc_status )

!  set up the structural data for the RQP Hessian 

     IF ( data%control%primal_qp ) THEN
       data%CQP_prob%H%n = nlp%n ; data%CQP_prob%H%ne = nlp%n
       array_name = 'fastr: data%CQP_prob%H%type'
       CALL SPACE_dealloc_array( data%CQP_prob%H%type,                         &
          inform%status, inform%alloc_status,                                  &
          array_name = array_name, out = data%control%error )
       CALL SMT_put( data%CQP_prob%H%type, 'DIAGONAL', inform%alloc_status )
     ELSE
       IF ( data%printi )                                                      &
         WRITE( data%out, "( A, ' dual qp not yet implemented ' )" ) prefix
       inform%status = GALAHAD_not_yet_implemented ; GO TO 990
     END IF 

!  set up the structural data for the EQP Jacobian

     array_name = 'fastr: data%EQP_prob%A%type'
     CALL SPACE_dealloc_array( data%EQP_prob%A%type,                           &
        inform%status, inform%alloc_status,                                    &
        array_name = array_name, out = data%control%error )
     CALL SMT_put( data%EQP_prob%A%type, 'COORDINATE', inform%alloc_status )

!  set up the structural data for the EQP Hessian 

     data%EQP_prob%n = data%n
     data%EQP_prob%H%n = data%n
     data%EQP_prob%H%ne = data%H_ne
     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%EQP_prob%H%ne = nlp%H%ne
       data%EQP_prob%H%row( : data%EQP_prob%H%ne ) = nlp%H%row
       data%EQP_prob%H%col( : data%EQP_prob%H%ne ) = nlp%H%col
     CASE ( 'SPARSE_BY_ROWS' )
       data%EQP_prob%H%ne = nlp%H%ptr( nlp%m + 1 ) - 1 
       DO i = 1, nlp%n
         data%EQP_prob%H%row( nlp%H%ptr( i ) : nlp%H%ptr( i + 1 ) - 1 ) = i
       END DO
       data%EQP_prob%H%col( : data%EQP_prob%H%ne ) = nlp%H%col
     CASE ( 'DENSE' ) 
       data%EQP_prob%H%ne = 0
       DO i = 1, nlp%n
         DO j = 1, i
           data%EQP_prob%H%ne = data%EQP_prob%H%ne + 1
           data%EQP_prob%H%row( data%EQP_prob%H%ne ) = i
           data%EQP_prob%H%col( data%EQP_prob%H%ne ) = j
         END DO
       END DO
     END SELECT

!  allow room for any extra entries required by the restoration phase 

     DO i = 1, nlp%m
       data%EQP_prob%H%row( data%H_ne + i ) = nlp%n + i
       data%EQP_prob%H%col( data%H_ne + i ) = nlp%n + i
     END DO

!  initialize the regularization and trust-region radii

     data%mu = data%control%initial_radius_rlp
     data%radius_eqp = data%control%initial_radius_eqp

     IF ( data%printd ) THEN
       WRITE( data%out, 2020 ) prefix
       WRITE( data%out, "( A, I6, 3ES12.4, '      -     ', ES12.4)" ) ( prefix,&
         i, nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%Z( i ), i = 1, nlp%n )
       WRITE( data%out, "( / )" )
     END IF

     IF ( data%printi ) WRITE( data%out, "( /, A, ' Key: R=restoration, ',     &
    & 'r=reject filter, s=succ, v=v.succ, u=unsucc, i=infeas', /, A,           &
    & '      L=RLP step, E=EQP step, t=step truncated, b=TR boundary' )" )     &
      prefix, prefix

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      M A I N     I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 100 CONTINUE

!  record the "best most-feasible" point

       IF ( inform%primal_infeasibility <= data%stop_p ) THEN
         data%x_feas = .TRUE.
         IF ( inform%obj < data%f_best ) THEN
           data%f_best = inform%obj
           data%v_best = inform%primal_infeasibility
           data%X_best( : nlp%n ) = nlp%X( : nlp%n )
           data%C_best( : nlp%m ) = nlp%C( : nlp%m )
           data%x_best_set = .TRUE.
         END IF
       ELSE
         IF ( .NOT. data%x_feas .AND.                                          &
              inform%primal_infeasibility < data%v_best ) THEN
!          data%f_best = inform%obj
           data%v_best = inform%primal_infeasibility
           data%X_best( : nlp%n ) = nlp%X( : nlp%n )
           data%C_best( : nlp%m ) = nlp%C( : nlp%m )
           data%x_best_set = .TRUE.
         END IF
       END IF
!      WRITE( 6, * ) ' x best set ? ', data%x_best_set
!      WRITE( 6, * ) ' f_best ', data%f_best, ' v_best ', data%v_best

!  if required, give details of the current point

       IF ( data%printd ) THEN
         l = 2
         IF ( data%control%fulsol ) l = nlp%n 
         IF ( data%control%print_level >= 10 ) l = nlp%n

         WRITE( data%out, "( /, A, ' Variables: ', /, A, '                  ', &
        &     '              <------ Bounds ------> ', /, A,                   &
        &     '      # name          value      ',                             &
        &     ' Lower       Upper       Dual ' )" ) prefix, prefix, prefix
         DO j = 1, 2 
           IF ( j == 1 ) THEN 
             ir = 1 ; ic = MIN( l, nlp%n ) 
           ELSE 
             IF ( ic < nlp%n - l ) WRITE( data%out, 2010 ) prefix
             ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
           END IF 
           DO i = ir, ic 
             WRITE( data%out, 2000 ) prefix, i, nlp%VNAMES( i ), nlp%X( i ),   &
               nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
           END DO
         END DO

         IF ( nlp%m > 0 ) THEN
           l = 2
           IF ( data%control%fulsol ) l = nlp%m
           IF ( data%control%print_level >= 10 ) l = nlp%m

           WRITE( data%out, "( /, A, ' Constraints:', /, A, '               ', &
          &    '                 <------ Bounds ------> ', /, A,               &
          &    '      # name           value      ',                           &
          &    ' Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
           DO j = 1, 2 
             IF ( j == 1 ) THEN 
               ir = 1 ; ic = MIN( l, nlp%m ) 
             ELSE 
               IF ( ic < nlp%m - l ) WRITE( data%out, 2010 ) prefix
               ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
             END IF 
             DO i = ir, ic 
               WRITE( data%out, 2000 ) prefix, i, nlp%CNAMES( i ), nlp%C( i ), &
                 nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
             END DO
           END DO
         END IF
       END IF

!  obtain the gradient of the objective function and the Jacobian
!  of the constraints. The data is stored in a sparse format

       IF ( data%new_gradient ) THEN
         IF ( data%reverse_gj ) THEN
           data%branch = 3 ; inform%status = 3 ; RETURN
         ELSE  
           CALL eval_GJ( data%eval_status, nlp%X, userdata, nlp%G, nlp%J%val )
         END IF
       END IF

!  return from reverse communication to obtain the gradient, Jacobian and/or 
!  Hessian

  130  CONTINUE
       IF ( data%new_gradient ) THEN
         data%CQP_prob%G( : nlp%n ) = nlp%G( : nlp%n )
         inform%g_eval = inform%g_eval + 1
       END IF

!  compute the gradient of the Lagrangian

       nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) - nlp%Z( : nlp%n )
       DO l = 1, data%CQP_prob%A%ne
         i = data%CQP_prob%A%col( l )
         nlp%gL( i ) = nlp%gL( i ) -                                           &
           nlp%J%val( l ) * nlp%Y( data%CQP_prob%A%row( l ) )
       END DO

!      WRITE(6,*) ' gl, y ', maxval( nlp%gL ), maxval( nlp%Y )
!      WRITE( data%out, "(A, /, ( 4ES20.12 ) )" ) ' gl_after ',  nlp%gl

!  compute norms of the primal and dual feasibility and the complemntary
!  slackness

       inform%dual_infeasibility =                                             &
         OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) )
       IF ( inform%primal_infeasibility > zero )                               &
         data%pr_best = MIN( data%pr_best, inform%primal_infeasibility )
       inform%complementary_slackness =                                        &
         OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                 &
            nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),          &
            nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                       &
            nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) )

!  compute the stopping tolerances

       IF ( inform%iter == 0 ) THEN
         data%stop_d = MAX( control%stop_abs_d,                                &
                            control%stop_rel_d * inform%dual_infeasibility )
         data%stop_c = MAX( control%stop_abs_c,                                &
                            control%stop_rel_c * inform%complementary_slackness)
         IF ( data%printi ) WRITE( data%out,                                   &
             "(  /, A, '  Primal    convergence tolerance =', ES11.4,          &
            &    /, A, '  Dual      convergence tolerance =', ES11.4,          &
            &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )      &
                 prefix, data%stop_p, prefix, data%stop_d, prefix, data%stop_c
       END IF 

!  -------------------------------------
!  Print a summary of the last iteration
!  -------------------------------------

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out,                          &
           "( /, A, ' iter      obj fun  pr_feas du_feas cmp_slk actve ',      &
        &        '  step   #fil      mu     CPU')" ) prefix
         data%print_1st_header = .FALSE.
         data%print_iteration_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out,                                                    &
             "( A, I5, 2A1, ES12.4, 3ES8.1, A6, ES8.1, 2A1, I4, ES8.1, F8.1 )")&
             prefix, inform%iter, data%it_type, data%rest, inform%obj,         &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness,                                   &
             STRING_integer_6( data%EQP_prob%A%m ), data%step, data%d_type,    &
             data%bdry, inform%FILTER_inform%filter_size, data%mu,             &
             data%clock_now
         ELSE
           WRITE( data%out, "( A, I5, '  ', ES12.4, 3ES8.1, '     -     -    ',&
          &   I4, ES8.1, F8.1 )" ) prefix, inform%iter,                        &
             inform%obj, inform%primal_infeasibility,                          &
             inform%dual_infeasibility, inform%complementary_slackness,        &
             inform%FILTER_inform%filter_size, data%mu, data%clock_now
         END IF
       END IF

!  ---------------------
!  Check for termination
!  ---------------------

!      WRITE(6,*) inform%primal_infeasibility <= data%stop_p,                  &
!           data%n_mult_wrong_sign == 0,                                       &
!           inform%dual_infeasibility <= data%stop_d,                          &
!           inform%complementary_slackness <= data%stop_c
       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            data%n_mult_wrong_sign == 0 .AND.                                  &
            inform%dual_infeasibility <= data%stop_d .AND.                     &
            inform%complementary_slackness <= data%stop_c ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
                "( /, A, ' Termination criteria satisfied ' )" ) prefix
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

       IF ( data%printd ) WRITE( data%out,                                     &
          "( A, ' X = ', /, ( 6X, 6ES12.4 ) )" ) prefix, nlp%X( : nlp%n )

!  ------------------------
!  Start the next iteration
!  ------------------------

       inform%iter = inform%iter + 1
!      IF ( inform%iter >= 60 ) data%control%multipliers = 2

       IF ( inform%iter > data%control%maxit ) THEN
         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Iteration limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_max_iterations ; GO TO 905
       END IF

!  ----------------------------------------------------------------------------
!                     CHOOSE THE CURRENT ACTIVE SET
!  ----------------------------------------------------------------------------

       IF ( inform%primal_infeasibility == zero                                &
            .AND. data%mu_new == zero .AND. .NOT. data%successful ) THEN
         data%DX_trial_rlp( : nlp%n ) = tenth * data%DX_trial_rlp( : nlp%n )
       ELSE


!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Compute the solution to the regularized quadratic program (RQP)

!      min  mu g^T d + 1/2 d^T d
!      s.t. c_l - c <= A d <= c_u - c and x_l - x <= d <= x_u - x 

!   perhaps by solving the dual

!     max mu/2 || A^T ( y_l - y_u ) - (z_l - z_u) - g ||_2^2
!         + y_l c_l - y_u c_u + z_l x_l - z_u - x_u
!     s.t. (y_l,y_u,z_l,z_u) >= 0

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set up the vector problem data

         data%CQP_prob%new_problem_structure = .TRUE.     
         data%CQP_prob%n = data%n
         data%CQP_prob%m = nlp%m
         data%CQP_prob%A%n = data%n 
         data%CQP_prob%A%m = nlp%m
         data%CQP_prob%H%n = data%n
         data%CQP_prob%A%ne = data%J_ne
         data%CQP_prob%f = zero
         data%CQP_prob%X_l( : nlp%n ) = nlp%X_l( : nlp%n ) - nlp%X( : nlp%n )
         data%CQP_prob%X_u( : nlp%n ) = nlp%X_u( : nlp%n ) - nlp%X( : nlp%n )
         IF ( nlp%m > 0 ) THEN
           data%CQP_prob%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) - nlp%C( : nlp%m )
           data%CQP_prob%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) - nlp%C( : nlp%m )
         END IF
         data%CQP_prob%X( : nlp%n ) = zero
         data%CQP_prob%Z( : nlp%n ) = nlp%Z( : nlp%n )
         data%CQP_prob%Y( : nlp%m ) = zero
         IF ( data%J_ne > 0 )                                                  &
           data%CQP_prob%A%val( : data%J_ne ) = nlp%J%val( : data%J_ne )
         IF ( data%control%primal_qp ) THEN
           data%CQP_prob%G( : nlp%n ) = data%mu * nlp%G( : nlp%n )
           data%CQP_prob%H%val( : nlp%n ) = one
         ELSE
         END IF 

!  if required, print a description of the problem

         IF ( data%printt ) WRITE( data%out, "( /, A, ' * Find the working',   &
        &    ' set - entering CQP: n = ', I0, ', m = ', I0, ', mu = ',         &
        &    ES8.2 )" ) prefix, nlp%n, data%CQP_prob%m, data%mu

         IF ( data%out > 0 .AND. data%control%print_level >= 20 ) THEN
           WRITE( data%out, "( A, ' n, m = ', I0, 1X, I0 )" )                  &
             prefix, data%CQP_prob%n, data%CQP_prob%m
           WRITE( data%out, "( A, ' f = ', ES12.4 )" )                         &
             prefix, data%CQP_prob%f
           WRITE( data%out, "( A, ' G = ', /, ( 5ES12.4 ) )" ) prefix,         &
             data%CQP_prob%G( : data%CQP_prob%n )
           IF ( SMT_get( data%CQP_prob%H%type ) == 'DIAGONAL' ) THEN
             WRITE( data%out, "( A, ' H (diagonal) = ', /, ( 5ES12.4 ) )" )    &
               prefix, data%CQP_prob%H%val( : data%CQP_prob%n )
           ELSE IF ( SMT_get( data%CQP_prob%H%type ) == 'DENSE' ) THEN
             WRITE( data%out, "( A, ' H (dense) = ', /, ( 5ES12.4 ) )" )       &
              prefix, data%CQP_prob%H%val( : data%CQP_prob%n *                 &
                ( data%CQP_prob%n + 1 ) / 2 )
           ELSE IF ( SMT_get( data%CQP_prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
             WRITE( data%out, "( A, ' H (row-wise) = ' )" ) prefix
             DO i = 1, data%CQP_prob%m
               WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
                 ( i, data%CQP_prob%H%col( j ), data%CQP_prob%H%val( j ),      &
                   j = data%CQP_prob%H%ptr( i ),                               &
                       data%CQP_prob%H%ptr( i + 1 ) - 1 )
             END DO
           ELSE
             WRITE( data%out, "( A, ' H (co-ordinate) = ' )" ) prefix
             WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
             ( data%CQP_prob%H%row( i ), data%CQP_prob%H%col( i ),             &
               data%CQP_prob%H%val( i ), i = 1, data%CQP_prob%H%ne )
           END IF
           WRITE( data%out, "( A, ' X_l = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%X_l( : data%CQP_prob%n )
           WRITE( data%out, "( A, ' X_u = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%X_u( : data%CQP_prob%n )
           IF ( SMT_get( data%CQP_prob%A%type ) == 'DENSE' ) THEN
             WRITE( data%out, "( A, ' A (dense) = ', /, ( 5ES12.4 ) )" )       &
               prefix,                                                         &
               data%CQP_prob%A%val( : data%CQP_prob%n * data%CQP_prob%m )
           ELSE IF ( SMT_get( data%CQP_prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
             WRITE( data%out, "( A, ' A (row-wise) = ' )" ) prefix
             DO i = 1, data%CQP_prob%m
               WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
                 ( i, data%CQP_prob%A%col( j ), data%CQP_prob%A%val( j ),      &
                   j = data%CQP_prob%A%ptr( i ),                               &
                       data%CQP_prob%A%ptr( i + 1 ) - 1 )
             END DO
           ELSE
             WRITE( data%out, "( A, ' A (co-ordinate) = ' )" ) prefix
             WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
             ( data%CQP_prob%A%row( i ), data%CQP_prob%A%col( i ),             &
               data%CQP_prob%A%val( i ), i = 1, data%CQP_prob%A%ne )
           END IF
           WRITE( data%out, "( A, ' C_l = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%C_l( : data%CQP_prob%m )
           WRITE( data%out, "( A, ' C_u = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%C_u( : data%CQP_prob%m )
         END IF

!  set control parameters

!        data%control%CQP_control%dufeas =                                     &
!          MAXVAL( ABS( data%CQP_prob%G( : data%CQP_prob%n ) ) )

!  solve the RQP

         CALL CQP_solve( data%CQP_prob, data%CQP_data,                         &
                         data%control%CQP_control, inform%CQP_inform,          &
                         C_stat = nlp%C_status, B_stat = nlp%X_status )

!        write(6,"( ' x ', / ( 5ES14.6) )" )  data%CQP_prob%X(:data%CQP_prob%n )
!        write(6,"( ' y ', / ( 5ES14.6) )" )  data%CQP_prob%Y(:data%CQP_prob%m )
!        write(6,"( ' z ', / ( 5ES14.6) )" )  data%CQP_prob%Z(:data%CQP_prob%n )
!        write(6,"( ' c ', / ( 5ES14.6) )" )  data%CQP_prob%C(:data%CQP_prob%m )

!       write(6,*) nlp%C_status
!       write(6,*) nlp%CNAMES

!       write(6,*) ' c_status ', nlp%C_status
!       write(6,*) ' x_status ', nlp%X_status

!  check to see if feasibility restoration is required ...

         IF ( inform%CQP_inform%status == GALAHAD_error_primal_infeasible .OR. &
              inform%CQP_inform%status == GALAHAD_error_dual_infeasible .OR.   &
              inform%CQP_inform%status == GALAHAD_error_tiny_step .OR.         &
              inform%CQP_inform%status == GALAHAD_error_ill_conditioned ) THEN
           SELECT CASE ( inform%CQP_inform%status ) 
           CASE ( GALAHAD_error_primal_infeasible )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - primal infeasible' )" ) prefix
           CASE ( GALAHAD_error_dual_infeasible )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - dual infeasible' )" ) prefix
           CASE ( GALAHAD_error_tiny_step )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - no further progress possible' )" ) prefix
           CASE DEFAULT
             IF ( data%printi ) WRITE( data%out, "( A, ' ** CQP error exit,',  &
            & '  status = ', I0 )" ) prefix, inform%CQP_inform%status
           END SELECT
           data%restoration = .TRUE. ; GO TO 310
         ELSE IF ( inform%CQP_inform%status == GALAHAD_error_unbounded ) THEN
           IF ( data%printt ) WRITE( data%out, "( A, '    * Exit from CQP',    &
          & ' unbounded from below' )" ) prefix

!  ... or if an error occured 


         ELSE IF ( inform%CQP_inform%status /= GALAHAD_ok ) THEN
           IF ( data%printi ) WRITE( data%out, "( A, ' ** CQP error exit,',    &
          & '  status = ', I0 )" ) prefix, inform%CQP_inform%status
           inform%status = GALAHAD_error_qp_solve ; GO TO 990
         END IF

         data%old_norm_dlp = data%norm_dlp
         data%norm_dlp = MAXVAL( ABS( data%CQP_prob%X( : data%CQP_prob%n ) ) )
! write(6,*) ' *** old, new norm dlp ', data%old_norm_dlp, data%norm_dlp

!  if required, summarize the RLP iteration

         IF ( data%printt ) THEN
           IF ( data%control%CQP_control%out > 0 .AND.                         &
                data%control%CQP_control%print_level > 0 )                     &
             WRITE( data%out, "( '' )" )
           WRITE( data%out, "( A, '  - on exit from CQP: status = ', I0,       &
         &   ', time = ', F0.2, ', iterations = ', I0 )" ) prefix,             &
            inform%CQP_inform%status, inform%CQP_inform%time%total,            &
            inform%CQP_inform%iter
           WRITE( data%out, "( A, '   - RLP objective decrease =', ES12.4,     &
         &  ', mu = ', ES10.4,                                                 &
         &   /, A, '   - || d_rlp || = ', ES10.4, ', || y_1 || = ', ES10.4,    &
         &   /, A, '   - # active counstraints: ', I0, ' (', I0,               &
         &  ' independent)', ' from ', I0, /, A, '   - # active bounds: ', I0, &
         &  ' (', I0, ' independent)', ' from ', I0 )" )                       &
              prefix, - inform%CQP_inform%obj, data%mu, prefix, data%norm_dlp, &
              MAX( MAXVAL( ABS( data%CQP_prob%Y( : data%CQP_prob%m ) ) ),      &
                   MAXVAL( ABS( data%CQP_prob%Z( : data%CQP_prob%n ) ) ) ),    &
              prefix, COUNT( nlp%C_status( : data%CQP_prob%m ) /= 0 ),         &
              COUNT( ABS( nlp%C_status( : data%CQP_prob%m ) ) == 1 ),          &
              data%CQP_prob%m,                                                 &
              prefix, COUNT( nlp%X_status( : data%CQP_prob%n ) /= 0 ),         &
              COUNT( ABS( nlp%X_status( : data%CQP_prob%n ) ) == 1 ),          &
              data%CQP_prob%n
         END IF

!  if required, print the subproblem and its solution

         IF ( data%printd ) THEN
           WRITE( data%out, "( /, A, ' RLP subproblem ' )" ) prefix
           WRITE( data%out, "( '      i  stat     X_l           X      ',      &
          &                    '   X_u            G          Z' )" )      
           WRITE( data%out, "( ( 1X, 2I6, 5ES12.4 ) )" )                       &
           ( i, nlp%X_status( i ),                                             &
             data%CQP_prob%X_l( i ), data%CQP_prob%X( i ),                     &
             data%CQP_prob%X_u( i ), data%CQP_prob%G( i ),                     &
             data%CQP_prob%Z( i ), i = 1, data%CQP_prob%n )
           IF ( nlp%m > 0 ) THEN
             WRITE( data%out, "( '      i  stat     C_l           C      ',    &
            &                    '   C_u           Y' )" )
             WRITE( data%out, "( ( 1X, 2I6, 4ES12.4 ) )" )                     &
               ( i, nlp%C_status( i ), data%CQP_prob%C_l( i ),                 &
                 data%CQP_prob%C( i ), data%CQP_prob%C_u( i ),                 &
                 data%CQP_prob%Y( i ), i = 1, data%CQP_prob%m )
             WRITE( data%out, "( '  A row   col      val       row   col    ', &
            & '  val       row   col      val', /, 3( 1X, 2I6, ES12.4 ) )" )   &
               ( data%CQP_prob%A%row( i ), data%CQP_prob%A%col( i ),           &
                 data%CQP_prob%A%val( i), i = 1, data%CQP_prob%A%ne )
           END IF
         END IF

!  --- for debugging --- write the working sets ---

         IF ( wsout > 0 ) THEN
           WRITE( wsout, "( /, ' FASTr iteration ', I7, //,                    &
          &  '      i status           c_l              c',                    &
          &  '              c_u         y' )" ) inform%iter
           WRITE( wsout, "( ( 2I7, 4ES16.8 ) )" ) ( i, nlp%C_status( i ),      &
             data%CQP_prob%C_l( i ), data%CQP_prob%C( i ),                     &
             data%CQP_prob%C_u( i ), data%CQP_prob%Y( i ),                     &
               i = 1, data%CQP_prob%m )
           WRITE( wsout, "( /,                                                 &
          &  '      i status           x_l              x',                    &
          &  '             x_u          z' )" )
           WRITE( wsout, "( ( 2I7, 4ES16.8 ) )" ) ( i, nlp%X_status( i ),      &
             data%CQP_prob%X_l( i ), data%CQP_prob%X( i ),                     &
             data%CQP_prob%X_u( i ), data%CQP_prob%Z( i ),                     &
               i = 1, data%CQP_prob%n )
         END IF

!  store the trial point d_RLP generated by the RLP

         data%DX_trial_rlp( : nlp%n ) = data%CQP_prob%X( : nlp%n )
       END IF
       data%norm_deqp = zero

!      write(6,*) ' x ', nlp%X
!      write(6,*) ' d_rlp', data%DX_trial_rlp

!  ----------------------------------------------------------------------------
!                 FIRST-ORDER LAGRANGE MULTIPLIER UPDATES
!  ----------------------------------------------------------------------------

       IF ( data%control%multipliers == 0 .OR.                                 &
            data%control%multipliers == 1  ) THEN

!  compute the changes in multipliers and dual variables

         data%DY( : nlp%m ) = data%CQP_prob%Y( : nlp%m ) - nlp%Y( : nlp%m )
         data%DZ( : nlp%n ) = data%CQP_prob%Z( : nlp%n ) - nlp%Z( : nlp%n )
 
!  compute A^T dy

         data%ATDY( : nlp%n ) = data%DZ( : nlp%n )
         DO l = 1, data%CQP_prob%A%ne
           i = data%CQP_prob%A%col( l )
           data%ATDY( i ) = data%ATDY( i ) +                                   &
             data%CQP_prob%A%val( l ) * data%DY( data%CQP_prob%A%row( l ) )
         END DO

!  find the minimizer alpha of || g_l - alpha A^T dy ||_2^2

         alpha = DOT_PRODUCT( data%ATDY( : nlp%n ), data%ATDY( : nlp%n ) )
! write(6,*) 'alpha', alpha
         IF ( alpha /= zero ) THEN
           alpha =                                                             &
             DOT_PRODUCT( nlp%GL( : nlp%n ), data%ATDY( : nlp%n ) ) / alpha 

!  update the multipliers

           IF ( alpha /= zero ) THEN
             nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) + alpha * data%DY( : nlp%m )
             nlp%Z( : nlp%n ) = nlp%Z( : nlp%n ) + alpha * data%DZ( : nlp%n )
             nlp%GL( : nlp%n ) = nlp%GL( : nlp%n ) - alpha * data%ATDY( : nlp%n)

!  recompute the dual infeasibility and complementarity 

             inform%dual_infeasibility =                                       &
               OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) )
             inform%complementary_slackness =                                  &
               OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),           &
                  nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),    &
                  nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                 &
                  nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) )
             IF ( data%printm ) WRITE( data%out, "( A, ' * Dual residual ',    &
            &  'with first-order multiplier estimate = ', ES10.4, /, A,        &
            &  '   and complementary slackness = ', ES10.4  )" )               &
               prefix, inform%dual_infeasibility,                              &
               prefix, inform%complementary_slackness

!  check for termination with the updated multipliers

             IF ( inform%primal_infeasibility <= data%stop_p .AND.             &
                  data%n_mult_wrong_sign == 0 .AND.                            &
                  inform%dual_infeasibility <= data%stop_d .AND.               &
                  inform%complementary_slackness <= data%stop_c ) THEN
               IF ( data%printt ) WRITE( data%out,                             &
                      "( /, A, ' Termination criteria satisfied ' )" ) prefix
               inform%status = GALAHAD_ok ; GO TO 900
             END IF
           END IF
         END IF
       END IF

! write(6,*) 'y_1', nlp%Y

!  ----------------------------------------------------------------------------
!                     COMPUTE LEAST_SQUARES MULTIPLIER ESTIMATES
!  ----------------------------------------------------------------------------

!      IF ( data%successful ) THEN

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  find the least-squares multipliers by solving the equality-constrained 
!  quadratic program

!      min  g^T s + 1/2 s^T s
!      s.t. A_A s = 0 and s_A = 0

!  where _A denotes the active constraints

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  compute how many active constraints there are

!        m_working = COUNT( ABS( nlp%C_status( : data%CQP_prob%m ) ) == 1 )
!        n_working = COUNT( ABS( nlp%X_status( : data%CQP_prob%n ) ) == 1 )

!  set the objective details

         CALL SMT_put( data%EQP_prob%H%type, 'DIAGONAL', inform%alloc_status )
         data%EQP_prob%n = data%n
         data%EQP_prob%H%n = data%EQP_prob%n
         data%EQP_prob%H%ne = data%EQP_prob%H%n
!        data%EQP_prob%m = m_working + n_working
         data%EQP_prob%f = zero
         data%EQP_prob%G( : nlp%n ) = nlp%G( : nlp%n )
         data%EQP_prob%H%val( : nlp%n ) = one
         data%EQP_prob%X( : nlp%n ) = zero
!        array_name = 'fastr: data%EQP_prob%H%type'
!        CALL SPACE_dealloc_array( data%EQP_prob%H%type,                       &
!         inform%status, inform%alloc_status,                                  &
!         array_name = array_name, out = control%error )

!write(6,*) ' --- g ', data%EQP_prob%g
!write(6,*) ' --- f ', data%EQP_prob%f
!write(6,*) ' --- H ', data%EQP_prob%H%val

!       IF ( data%printi ) WRITE( data%out, "(                                 &
!       &   /, A, ' - # active counstraints: ', I0, ' from ', I0, ', bounds: ',&
!       &  I0, ' from ', I0 )" ) prefix, m_working, nlp%m, n_working, nlp%n

!  set the right-hand sides for the active constraints

         data%EQP_prob%m = 0
         DO i = 1, nlp%m 
           IF ( nlp%C_status( i ) == - 1 ) THEN
             data%EQP_prob%m = data%EQP_prob%m + 1
             data%EQP_prob%C( data%EQP_prob%m ) = zero
!            data%EQP_prob%C( data%EQP_prob%m ) = nlp%C( i ) - nlp%C_l( i )
             data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
             nlp%C_status( i ) = - data%EQP_prob%m
           ELSE IF ( nlp%C_status( i ) == 1 ) THEN 
             data%EQP_prob%m = data%EQP_prob%m + 1
             data%EQP_prob%C( data%EQP_prob%m ) = zero
!            data%EQP_prob%C( data%EQP_prob%m ) = nlp%C( i ) - nlp%C_u( i )
             data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
             nlp%C_status( i ) = data%EQP_prob%m
           ELSE
             nlp%C_status( i ) = 0
           END IF
         END DO
         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) == - 1 ) THEN
             data%EQP_prob%m = data%EQP_prob%m + 1
             data%EQP_prob%C( data%EQP_prob%m ) = zero
!            data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_l( i )
             data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
             nlp%X_status( i ) = - data%EQP_prob%m
           ELSE IF ( nlp%X_status( i ) == 1 ) THEN 
             data%EQP_prob%m = data%EQP_prob%m + 1
             data%EQP_prob%C( data%EQP_prob%m ) = zero
!            data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_u( i )
             data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
             nlp%X_status( i ) = data%EQP_prob%m
           ELSE
             nlp%X_status( i ) = 0
           END IF
         END DO
         data%EQP_prob%A%n = data%EQP_prob%n
         data%EQP_prob%A%m = data%EQP_prob%m

!         WRITE( 6, "( ' nlp%X_status ', /, ( 10I7 ) )" ) nlp%X_status
!         WRITE( 6, "( ' nlp%C_status ', /, ( 10I7 ) )" ) nlp%C_status

!  place the entries in the Jacobian of working constraints

         data%EQP_prob%A%ne = 0
         SELECT CASE ( SMT_get( nlp%J%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, data%J_ne
             i = nlp%J%row( l )
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    control%jacobian_zero_tolerance ) THEN
               data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
               data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
               data%EQP_prob%A%col( data%EQP_prob%A%ne ) = nlp%J%col( l )
               data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
             END IF
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           data%EQP_prob%A%ne = 0
           DO i = 1, nlp%m
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    control%jacobian_zero_tolerance ) THEN
               DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                 data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
                 data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
                 data%EQP_prob%A%col( data%EQP_prob%A%ne ) = nlp%J%col( l )
                 data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
               END DO
             END IF
           END DO
         CASE ( 'DENSE' ) 
           data%EQP_prob%A%ne = 0 ; l = 0
           DO i = 1, nlp%m
             ii = ABS( nlp%C_status( i ) )
             IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                         &
                    control%jacobian_zero_tolerance ) THEN
               DO j = 1, nlp%n
                 data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1 ; l = l + 1
                 data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
                 data%EQP_prob%A%col( data%EQP_prob%A%ne ) = j
                 data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
               END DO
             ELSE
               l = l + nlp%n
             END IF
           END DO
         END SELECT
         DO j = 1, nlp%n
           i = ABS( nlp%X_status( j ) )
           IF ( i > 0 ) THEN
             data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
             data%EQP_prob%A%row( data%EQP_prob%A%ne ) = i
             data%EQP_prob%A%col( data%EQP_prob%A%ne ) = j
             data%EQP_prob%A%val( data%EQP_prob%A%ne ) = one
           END IF
         END DO

!  get rid of tiny entries

     do i = 1, data%EQP_prob%A%ne 
       if(abs(data%EQP_prob%A%val( i )) <= 1.0D-14 ) then
!      write(6,*) ' setting a(',i,')= ', data%EQP_prob%A%val( i ), 'to zero'
        data%EQP_prob%A%val( i )=zero
       end if
     end do

! write(6,"(' nlp%n, nlp%m ', I0, 1X, I0 )" ) data%EQP_prob%n, data%EQP_prob%m
! write(44,"( ' n, nnz ', 2I8 )" )  data%EQP_prob%n, data%EQP_prob%A%ne
! write(44,"( ' A_row ', / ( 5I6) )" )                                         &
!   data%EQP_prob%A%row( : data%EQP_prob%A%ne )
! write(44,"( ' A_col ', / ( 5I6) )" )                                         &
!   data%EQP_prob%A%col( : data%EQP_prob%A%ne )
! write(44,"( ' A_val ', / ( 5ES14.6))" )                                      &
!    data%EQP_prob%A%val(:data%EQP_prob%A%ne )
! write(44,"( ' c ', / ( 5ES14.6) )" )  data%EQP_prob%C(:data%EQP_prob%m )

!  find the Lagrange multiplier estimates y (stored in data%EQP_prob%Y)

         IF ( data%printt ) WRITE( data%out,                                   &
           "( /, A, ' * Compute least-squares Lagrange multiplier estimates',  &
        &     /, A, '   - entering EQP: n = ',  I0, ', m_working = ', I0 )" )  &
            prefix, prefix, nlp%n, data%EQP_prob%m

         data%control%EQP_control%radius = - one
         CALL EQP_solve( data%EQP_prob, data%EQP_data,                         &
                         data%control%EQP_control, inform%EQP_inform )

!  check to see if an error occured

         IF ( inform%EQP_inform%status == GALAHAD_error_primal_infeasible ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' EQP infeasible ... skipping' )" ) prefix
           GO TO 140
         ELSE IF ( inform%EQP_inform%status /= GALAHAD_ok ) THEN
           IF ( data%printi ) WRITE( data%out, 2030 )                          &
             prefix, inform%EQP_inform%status
           inform%status = GALAHAD_error_qp_solve ; GO TO 990
         END IF

         IF ( data%EQP_prob%m > 0 ) THEN
           y_t = MAXVAL( ABS( data%EQP_prob%Y( : data%EQP_prob%m ) ) )
         ELSE
           y_t = zero
        END IF

        dual_infeasibility_ls                                                  &
          = MAXVAL( ABS( data%EQP_prob%X( : data%EQP_prob%n ) ) )

! write(6,*) ' d_eqp', data%EQP_prob%X
! write(6,*) ' d_eqp', data%EQP_prob%Y( : data%EQP_prob%m )
! write(6,*) ' dx dy ', MAXVAL( ABS( data%EQP_prob%X(:data%EQP_prob%n ) ) ),   &
!                       MAXVAL( ABS( data%EQP_prob%Y(:data%EQP_prob%m ) ) )

!  if required, summarize the EQP iteration

         IF ( data%printt ) THEN
           IF ( data%control%EQP_control%out > 0 .AND.                         &
                data%control%EQP_control%print_level > 0 )                     &
             WRITE( data%out, "( '' )" )
           WRITE( data%out, "( A, '   - on exit from EQP: status = ', I0,      &
          & ', cg iter = ', I0, ', time = ', F0.2, /, A, '   - || g - A^T',    &
          & ' y_ls || = ', ES10.4, ', || y_ls || = ', ES10.4 )" )              &
                prefix, inform%EQP_inform%status, inform%EQP_inform%cg_iter,   &
                inform%EQP_inform%time%total, prefix,                          &
                dual_infeasibility_ls, y_t
         END IF

         IF ( data%printd ) THEN
           WRITE( data%out, "( /, A, ' Least-squares-EQP subproblem' )" ) prefix
           WRITE( data%out, "( 2 ( '      i       X           G    ' ) )" )
           WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                  &
             ( i, data%EQP_prob%X( i ), data%EQP_prob%G( i ),                  &
               i = 1, data%EQP_prob%n )
           IF ( data%EQP_prob%m > 0 ) THEN
             WRITE( data%out, "( 2 ( '      i       C           Y   ' ) )" )
             WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                &
               ( i, data%EQP_prob%C( i ), data%EQP_prob%Y( i ),                &
                 i = 1, data%EQP_prob%m )
             WRITE( data%out, "( '  A row   col      val      row   col     ', &
            & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" ) &
               ( data%EQP_prob%A%row( i ), data%EQP_prob%A%col( i ),           &
                 data%EQP_prob%A%val( i ), i = 1, data%EQP_prob%A%ne )
           END IF
           WRITE( data%out, "( '  H row   col      val      row   col     ',   &
          & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )   &
             ( data%EQP_prob%H%row( i ), data%EQP_prob%H%col( i ),             &
               data%EQP_prob%H%val( i ), i = 1, data%EQP_prob%H%ne )
         END IF

!  spread the solution into the vectors (x_p,y_p,z_p,c_p)

         data%X_p( : nlp%n ) = data%EQP_prob%X( : nlp%n )
         DO i = 1, nlp%m 
           IF ( nlp%C_status( i ) == 0 ) THEN
             data%Y_p( i ) = zero
           ELSE
             data%Y_p( i ) = data%EQP_prob%Y( ABS( nlp%C_status( i ) ) )
           END IF
         END DO
         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) == 0 ) THEN
             data%Z_p( i ) = zero
           ELSE
             data%Z_p( i ) = data%EQP_prob%Y( ABS( nlp%X_status( i ) ) )
           END IF
         END DO
         data%C_p( : nlp%m ) = zero
         DO l = 1, data%CQP_prob%A%ne
           i = data%CQP_prob%A%row( l )
           data%C_p( i ) = data%C_p( i )                                       &
             + data%CQP_prob%A%val( l ) * data%X_p( data%CQP_prob%A%col( l ) )
         END DO

!write(6,*) ' ||y_ls|| ', MAXVAL( ABS( nlp%Y( : nlp%m )                        &
!         + ( one - data%mu ) * data%Y_p( : nlp%m ) ))
!write(6,*) 'y_ls', data%Y_p

!      END IF

!  recompute the complementarity and, if necessary, the number of out-of-kilter 
!  multipliers

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            dual_infeasibility_ls <= data%stop_d ) THEN
         complementary_slackness_ls =                                          &
           OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),               &
              nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), data%Z_p( : nlp%n ),     &
              nlp%m, nlp%C( : nlp%m ),                                         &
              nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ),  data%Y_p( : nlp%m ) )

         IF ( data%printm ) WRITE( data%out, "( A, ' * Dual residual ',        &
        &  'with least-squares multiplier estimate = ', ES10.4, /, A,          &
        &  '   and complementary slackness = ', ES10.4  )" )                   &
           prefix, dual_infeasibility_ls,                                      &
           prefix, complementary_slackness_ls

         IF ( complementary_slackness_ls <= data%stop_c ) THEN
           n_mult_wrong_sign_ls = 0
           DO i = 1, nlp%m 
             IF ( nlp%C_l( i ) /= nlp%C_u( i ) .AND.                           &
              ( ( nlp%C_status( i ) < 0 .AND. data%Y_p( i ) < - y_tiny ) .OR.  &
                ( nlp%C_status( i ) > 0 .AND. data%Y_p( i ) > y_tiny ) ) )     &
               n_mult_wrong_sign_ls = n_mult_wrong_sign_ls + 1
           END DO
           DO i = 1, nlp%n
             IF ( nlp%X_l( i ) /= nlp%X_u( i ) .AND.                           &
              ( ( nlp%X_status( i ) < 0 .AND. data%Z_p( i ) < - z_tiny ) .OR.  &
                ( nlp%X_status( i ) > 0 .AND. data%Z_p( i ) > z_tiny ) ) )     &
               n_mult_wrong_sign_ls = n_mult_wrong_sign_ls + 1
           END DO

!  check for termination with the least-squares multipliers

           IF ( n_mult_wrong_sign_ls == 0 ) THEN
             inform%dual_infeasibility = dual_infeasibility_ls
             inform%complementary_slackness = complementary_slackness_ls
             data%n_mult_wrong_sign = n_mult_wrong_sign_ls
             nlp%Y( : nlp%m ) = data%Y_p( : nlp%m )
             nlp%Z( : nlp%n ) = data%Z_p( : nlp%n )
             IF ( data%printt ) WRITE( data%out,                               &
                "( /, A, ' Termination criteria satisfied ' )" ) prefix
             inform%status = GALAHAD_ok ; GO TO 900
           END IF
         END IF
       END IF

!  if required or desirable, use the least-squares multiplier estimates

!write(6,*) dual_infeasibility_ls, inform%dual_infeasibility
       IF ( data%control%multipliers == 3 .OR.                                 &
            ( data%control%multipliers == 0 .AND.                              &
              dual_infeasibility_ls < inform%dual_infeasibility ) ) THEN
         nlp%Y( : nlp%m ) = data%Y_p( : nlp%m )
         nlp%Z( : nlp%n ) = data%Z_p( : nlp%n )
       END IF

!  compute the Hessian

       IF ( data%reverse_hl ) THEN
          data%branch = 4 ; inform%status = 4 ; RETURN
       ELSE  
          CALL eval_HL( data%eval_status, nlp%X, nlp%Y, userdata, nlp%H%val )
       END IF

!  return from reverse communication to obtain the Hessian of the Lagrangian

 140   CONTINUE
!write(6,*) ' X ', nlp%X( : nlp%n )
!write(6,*) ' Y ', nlp%Y( : nlp%m )
!write(6,*) ' H ', nlp%H%val( : data%H_ne )

!  ----------------------------------------------------------------------------
!                     COMPUTE THE SEARCH DIRECTION
!  ----------------------------------------------------------------------------

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  solve the equality-constrained quadratic program

!      min  g^T s + s^T H s
!      s.t. A_A s = c_b_A - c_A and s = x_b_A - x_A

!  where _A denotes the active constraints & bounds and c_b and x_b are
!  the relevant lower or upper bounds

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set the objective details

       CALL SMT_put( data%EQP_prob%H%type, 'COORDINATE', inform%alloc_status )
       data%EQP_prob%n = data%n
       data%EQP_prob%H%n = data%EQP_prob%n
       data%EQP_prob%H%ne = data%H_ne
       data%EQP_prob%f = nlp%f
       data%EQP_prob%G( : nlp%n ) = nlp%G( : nlp%n )
       data%EQP_prob%H%val( : data%H_ne ) = nlp%H%val( : data%H_ne )
       data%EQP_prob%X( : nlp%n ) = data%CQP_prob%X( : nlp%n )

!write(6,*) ' --- g ', data%EQP_prob%g
!write(6,*) ' --- f ', data%EQP_prob%f
!write(6,*) ' --- H ', data%EQP_prob%H%val

!     IF ( data%printi ) WRITE( data%out, "(                                   &
!     &   /, A, ' - # active counstraints: ', I0, ' from ', I0, ', bounds: ',  &
!     &  I0, ' from ', I0 )" ) prefix, m_working, nlp%m, n_working, nlp%n

!  set the right-hand sides for the active constraints

       data%EQP_prob%m = 0
       DO i = 1, nlp%m 
         IF ( nlp%C_status( i ) < 0 ) THEN
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%C( i ) - nlp%C_l( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
         ELSE IF ( nlp%C_status( i ) > 0 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%C( i ) - nlp%C_u( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
         END IF
       END DO
       DO i = 1, nlp%n
         IF ( nlp%X_status( i ) < 0 ) THEN
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_l( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
         ELSE IF ( nlp%X_status( i ) > 0 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_u( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
         END IF
       END DO

!  restrict the EQP step to be 100 times that of the RLP

       data%radius_eqp = hundred * data%norm_dlp

!  find the step s (stored in data%EQP_prob%X) and Lagrange multiplier 
!  y (stored in data%EQP_prob%Y)

       IF ( data%printt ) WRITE( data%out, "( /, A, ' * Find the step -',      &
      &  ' entering EQP: n = ', I0, ', m_working = ', I0 )" )                  &
           prefix, nlp%n, data%EQP_prob%m
       data%control%EQP_control%radius = data%radius_eqp
       CALL EQP_solve( data%EQP_prob, data%EQP_data, data%control%EQP_control, &
                       inform%EQP_inform )

!  check to see if an error occured

       IF ( inform%EQP_inform%status == GALAHAD_error_primal_infeasible ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
           "( /, A, ' EQP infeasible ... skipping' )" ) prefix
         data%take_eqp_step = .FALSE.
         data%step_rlp = one
         GO TO 200
       ELSE IF ( inform%EQP_inform%status /= GALAHAD_ok ) THEN
         IF ( data%printi ) WRITE( data%out, 2030 )                            &
           prefix, inform%EQP_inform%status
         inform%status = GALAHAD_error_qp_solve ; GO TO 990
       END IF

!  compute the norm of the step and the slope for the objective and violated 
!  constraints

       data%norm_deqp = MAXVAL( ABS( data%EQP_prob%X( : data%EQP_prob%n ) ) )
       data%gtd_eqp = DOT_PRODUCT( nlp%G( : nlp%n ), data%EQP_prob%X( : nlp%n ))
       data%dtjc_eqp = zero
       DO l = 1, data%CQP_prob%A%ne
         i = data%CQP_prob%A%row( l )
         IF ( nlp%C( i ) <= nlp%C_l( i ) ) THEN
           data%dtjc_eqp = data%dtjc_eqp +                                     &
             data%EQP_prob%X( data%CQP_prob%A%col( l ) ) *                     &
             ( nlp%C( i ) - nlp%C_l( i ) ) * data%CQP_prob%A%val( l ) 
         ELSE IF ( nlp%C( i ) >= nlp%C_u( i ) ) THEN
           data%dtjc_eqp = data%dtjc_eqp +                                     &
             data%EQP_prob%X( data%CQP_prob%A%col( l ) ) *                     &
             ( nlp%C( i ) - nlp%C_u( i ) ) * data%CQP_prob%A%val( l ) 
         END IF
       END DO
       IF ( data%printm ) WRITE( data%out, "( A, ' d^Tg, d^TJc = ', 2ES12.4 )")&
         prefix, data%gtd_eqp, data%dtjc_eqp

! write(6,*) ' d_eqp', data%EQP_prob%X
! write(6,*) ' dx dy ', MAXVAL( ABS( data%EQP_prob%X(:data%EQP_prob%n ) ) ),   &
!                       MAXVAL( ABS( data%EQP_prob%Y(:data%EQP_prob%m ) ) )

!  if required, summarize the EQP iteration

       IF ( data%printt ) THEN
         IF ( data%control%EQP_control%out > 0 .AND.                           &
              data%control%EQP_control%print_level > 0 )                       &
           WRITE( data%out, "( '' )" )

         IF ( ABS( inform%EQP_inform%GLTR_inform%mnormx - data%radius_eqp )    &
                <= teneps ) THEN
           tr_active = 'active  '
         ELSE
           tr_active = 'inactive'
         END IF

         IF ( data%EQP_prob%m > 0 ) THEN
           y_t = MAXVAL( ABS( data%EQP_prob%Y( : data%EQP_prob%m ) ) )
         ELSE
           y_t = zero
         END IF
         WRITE( data%out, "( A, '   - on exit from EQP: status = ', I0,        &
        & ', cg iter = ', I0, ', time = ', F0.2, /, A, '   - EQP objective',   &
        & ' decrease =', ES12.4, ', radius = ', ES10.4, ' ', A,                &
        & /, A, '   - || d_eqp || = ', ES10.4, ', || y_2 || = ', ES10.4 )" )   &
              prefix, inform%EQP_inform%status, inform%EQP_inform%cg_iter,     &
              inform%EQP_inform%time%total, prefix,                            &
              nlp%f - inform%EQP_inform%obj,                                   &
              data%radius_eqp, TRIM( tr_active ), prefix, data%norm_deqp, y_t
       END IF

       IF ( data%printd ) THEN
         WRITE( data%out, "( /, A, ' EQP subproblem ' )" ) prefix
         WRITE( data%out, "( 2 ( '      i       X           G    ' ) )" )
         WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                    &
           ( i, data%EQP_prob%X( i ), data%EQP_prob%G( i ),                    &
               i = 1, data%EQP_prob%n )
         IF ( data%EQP_prob%m > 0 ) THEN
           WRITE( data%out, "( 2 ( '      i       C           Y    ' ) )" )
           WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                  &
             ( i, data%EQP_prob%C( i ), data%EQP_prob%Y( i ),                  &
                 i = 1, data%EQP_prob%m )
           WRITE( data%out, "( '  A row   col      val      row   col     ',   &
          & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )   &
             ( data%EQP_prob%A%row( i ), data%EQP_prob%A%col( i ),             &
               data%EQP_prob%A%val( i ), i = 1, data%EQP_prob%A%ne )
         END IF
         WRITE( data%out, "( '  H row   col      val      row   col     ',     &
        & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )     &
           ( data%EQP_prob%H%row( i ), data%EQP_prob%H%col( i ),               &
             data%EQP_prob%H%val( i ), i = 1, data%EQP_prob%H%ne )
       END IF
!write(6,*) 'y_2', data%EQP_prob%Y(:data%EQP_prob%m)

       data%bdry = ' '
       IF ( ABS( inform%EQP_inform%GLTR_inform%mnormx -                        &
                 data%radius_eqp ) < teneps) data%bdry = 'b'

!  ----------------------------------------------------------------------------
!                       STEP ACCEPTANCE OR REJECTION
!  ----------------------------------------------------------------------------

!  check to find the range of values [mu_new,mu] for which the active set for 
!  the current RLP stays active

       data%mu_new = FASTR_mu_new( nlp%m, nlp%n, data%mu,                      &
                                   data%CQP_prob%X_l, data%CQP_prob%X_u,       &
                                   data%CQP_prob%X, data%X_p,                  &
                                   data%CQP_prob%C_l, data%CQP_prob%C_u,       &
                                   data%CQP_prob%C, data%C_p,                  &
                                   data%CQP_prob%Y, data%Y_p,                  &
                                   data%CQP_prob%Z, data%Z_p,                  &
                                   nlp%X_status, nlp%C_status,                 &
                                   data%out, data%printt,                      &
                                   data%print_level > 10, prefix )

!  record the "decrease" delta_l in the linear model with this value of mu

       data%gtd = DOT_PRODUCT( nlp%G( : nlp%n ), data%DX_trial_rlp( : nlp%n ) )
!write(6,*) ' gtd ', data%gtd, data%mu_new
       data%gtd = DOT_PRODUCT( nlp%G( : nlp%n ), data%DX_trial_rlp( : nlp%n )  &
                            + ( data%mu_new - data%mu ) * data%X_p( : nlp%n ) )
!write(6,*) ' gtd ', data%gtd
       data%delta_l = - data%gtd

!  record the required "decrease" delta_q in the "quadratic" model, that is
!  to say at the Cauchy point, the smallest value of the quadratic model on 
!  the line joining x to x + d_RLP. Only consider the case where there is 
!  decrease in the linear model (i,e g^T d_RLP < 0)

       IF ( data%gtd < zero ) THEN

!  compute the product d^T H(x,y) d, temporarily storing H(x,y) d in Hd

         data%U( : nlp%n ) = zero
         IF ( got_hessian ) THEN
           DO l = 1, data%H_ne
             i = data%EQP_prob%H%row( l ) ; j = data%EQP_prob%H%col( l )
             data%U( i ) = data%U( i ) +                                       &
               data%EQP_prob%H%val( l ) * data%DX_trial_rlp( j ) 
             IF ( i /= j ) data%U( j ) = data%U( j ) +                         &
               data%EQP_prob%H%val( l ) * data%DX_trial_rlp( i ) 
           END DO
         ELSE
           IF ( data%reverse_hlprod ) THEN
             data%V( : nlp%n ) = data%DX_trial_rlp( : nlp%n )
             data%branch = 5 ; inform%status = 6 ; RETURN
           ELSE  
             CALL eval_HLPROD( data%eval_status, nlp%X, nlp%Y, userdata,       &
                               data%U, data%DX_trial_rlp )
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian-vector product

  150  CONTINUE
       IF ( data%gtd < zero ) THEN
         dthd = DOT_PRODUCT( data%DX_trial_rlp( : nlp%n ), data%U( : nlp%n ) )

!  record delta_q at a full Cauchy step ...

         IF ( dthd <= zero ) THEN
           data%delta_q = - data%gtd - dthd
         ELSE
           IF ( - data%gtd >= dthd ) THEN
             data%delta_q = - data%gtd - dthd

!  ... or at a restricted Cauchy step 

           ELSE
             data%delta_q = half * data%gtd * ( data%gtd / dthd )
           END IF
         END IF
       ELSE
         data%delta_q = zero
       END IF

       data%successful = .FALSE.

!  compute the maximum step for the EQP step to stay feasible

       IF ( data%norm_deqp > zero ) THEN
         data%step_eqp = one
         DO i = 1, nlp%n
           dx = data%EQP_prob%X( i ) 
           IF ( dx > epsmch ) THEN
             data%step_eqp =                                                   &
               MIN( data%step_eqp, ( nlp%X_u( i ) - nlp%X( i ) ) / dx )
           ELSE IF ( dx < - epsmch ) THEN
             data%step_eqp =                                                   &
                MIN( data%step_eqp, ( nlp%X_l( i ) - nlp%X( i ) ) / dx )
           END IF
         END DO

         IF ( data%any_linear ) THEN
           data%C_trial( : nlp%m ) = zero
           DO l = 1, data%CQP_prob%A%ne
             i = data%CQP_prob%A%row( l )
             data%C_trial( i ) = data%C_trial( i ) + data%CQP_prob%A%val( l )  &
               * data%EQP_prob%X( data%CQP_prob%A%col( l ) )
           END DO

           DO i = 1, nlp%m
             IF ( nlp%LINEAR( i ) ) THEN
               dc = data%C_trial( i ) 
               IF ( dc > epsmch ) THEN
                 data%step_eqp =                                               &
                   MIN( data%step_eqp, ( nlp%C_u( i ) - nlp%C( i ) ) / dc )
               ELSE IF ( dc < - epsmch ) THEN
                 data%step_eqp =                                               &
                   MIN( data%step_eqp, ( nlp%C_l( i ) - nlp%C( i ) ) / dc )
               END IF
             END IF
           END DO
         END IF
       ELSE
         data%step_eqp = zero
       END IF

!  decide whether to avoid the EQP search direction

       IF ( data%step_eqp <= zero ) THEN
         IF ( data%printt )                                                    &
           WRITE( data%out, "(  A, '   - EQP stepsize zero' )" ) prefix
         data%take_eqp_step = .FALSE.
       ELSE
!        IF ( inform%iter > 20 ) data%take_eqp_step = .TRUE.
         data%take_eqp_step = .TRUE.
       END IF
       data%step_rlp = one

!  -----------------------------------------
!  loop over the potential search directions
!  -----------------------------------------

 200   CONTINUE

!  try the EQP step

         IF ( data%take_eqp_step ) THEN
           data%d_name = 'EQP' ; data%d_type = 'E'
           IF ( .NOT. data%control%try_rlp_step                                &
              .AND. data%step_eqp > ten ** ( - 10 )                            &
              .AND. inform%primal_infeasibility <= epsmch ** 0.75 ) GO TO 290
           IF ( .NOT. data%step_eqp == one ) data%bdry = 't'

!  update the solution

           data%X_trial( : nlp%n ) =                                           &
             nlp%X( : nlp%n ) + data%step_eqp * data%EQP_prob%X( : nlp%n )
           data%step = data%norm_deqp * data%step_eqp

!  try the RLP step

         ELSE
           data%d_name = 'RLP' ; data%d_type = 'L'
           data%X_trial( : nlp%n ) =                                           &
             nlp%X( : nlp%n ) + data%step_rlp * data%DX_trial_rlp( : nlp%n )
           data%step = data%norm_dlp * data%step_rlp
         END IF

!  evaluate the objective and general constraint function values
     
         IF ( data%reverse_fc ) THEN
           data%EQP_prob%X( : nlp%n ) = nlp%X( : nlp%n )     ! temporary copy
           data%EQP_prob%C( : nlp%m ) = nlp%C( : nlp%m )     ! temporary copy
           data%EQP_prob%f = nlp%f                           ! temporary copy
           nlp%X = data%X_trial
           data%branch = 6 ; inform%status = 2 ; RETURN
         ELSE  
           CALL eval_FC( data%eval_status, data%X_trial, userdata,             &
                         obj_trial, data%C_trial )
         END IF

!  return from reverse communication to obtain the objective value

 260     CONTINUE
         IF ( data%reverse_fc ) THEN
           obj_trial = nlp%f
           data%C_trial = nlp%C
           nlp%X( : nlp%n ) = data%EQP_prob%X( : nlp%n )      ! restore copy
           nlp%C( : nlp%m ) = data%EQP_prob%C( : nlp%m )      ! restore copy
           nlp%f = data%EQP_prob%f                            ! restore copy
         END IF
         inform%f_eval = inform%f_eval + 1

!  compute the norm of the violation

         primal_infeasibility_trial =                                          &
           OPT_primal_infeasibility( nlp%m, data%C_trial( : nlp%m ),           &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )

         IF ( data%take_eqp_step ) THEN
           data%obj_eqp = obj_trial
           data%primal_infeasibility_eqp = primal_infeasibility_trial
         END IF

         o = inform%obj - data%control%gamma_filter *inform%primal_infeasibility
         v = data%control%beta_filter * inform%primal_infeasibility

!  is the new point is too infeasible ?

         IF ( primal_infeasibility_trial > data%pr_max ) THEN
           IF ( data%take_eqp_step ) data%n_pr_max = data%n_pr_max + 1
           data%it_type = 'i'
           IF ( data%printt ) WRITE( data%out, "( /, A, ' * ', A3, ' step',    &
          &  ' infeasibility = ', ES10.4, ' too large ' )" ) prefix,           &
               data%d_name, primal_infeasibility_trial

!  is the new point is acceptable to the filter and (f,c) ?

         ELSE 
           CALL FILTER_acceptable( obj_trial, primal_infeasibility_trial,      &
                                   data%FILTER_data,                           &
                                   data%control%FILTER_control,                &
                                   acceptable, o = o, v = v )
           IF ( acceptable ) THEN
             delta_f = inform%obj - obj_trial
             delta_m = MIN( data%delta_q, data%delta_l )
             IF ( data%take_eqp_step ) data%n_pr_max = 0
             IF ( data%printt ) WRITE( data%out,                               &
               "(  /, A, ' * ', A3, ' point acceptable to filter (v,o) = (',   &
              &    ES10.4, ',', ES11.4, ')', /, A,                             &
              &    '                       vs current (v,o) = (',              &
              &    ES10.4, ',', ES11.4, ')' )" )                               &
                 prefix, data%d_name, primal_infeasibility_trial, obj_trial,   &
                 prefix, inform%primal_infeasibility, inform%obj
             IF ( data%printt ) THEN
               WRITE( data%out, "( A, '   - ared (', A, ES8.2, '), pred (', A, &
            &    ES8.2, '), successful (', L1, '), very_successful (', L1,')', &
            &   /, A, '   - delta_l (', A, ES8.2,                              &
            &      '), delta h^2 (', ES8.2, '), v-step (', L1, ')' )" )        &
               prefix, TRIM( STRING_sign( delta_f, .FALSE. ) ), ABS( delta_f ),&
               TRIM( STRING_sign( delta_m, .FALSE. ) ), ABS( delta_m ),        &
               delta_f >= data%control%eta_successful * delta_m,               &
               delta_f >= data%control%eta_very_successful * delta_m,          &
               prefix, TRIM( STRING_sign( data%delta_l, .FALSE. ) ),           &
               ABS( data%delta_l ),                                            &
               inform%primal_infeasibility ** 2, data%delta_l <                &
               data%control%delta_feas * inform%primal_infeasibility ** 2
             END IF

!  the new point is acceptable. Now check whether the predicted reduction of 
!  the objective function is sufficiently positive and has been realized for 
!  the true objective function

             IF ( delta_f >= data%control%eta_successful * delta_m             &
                 .OR. data%delta_l < data%control%delta_feas *                 &
                                inform%primal_infeasibility ** 2 ) THEN
               data%successful = .TRUE.
               data%very_successful =                                          &
                 delta_f >= data%control%eta_very_successful * delta_m
               IF ( data%very_successful ) THEN
                 data%it_type = 'v'
               ELSE
                 data%it_type = 's'
               END IF
               IF ( data%printt ) THEN
                 IF ( data%very_successful ) THEN
                   IF ( data%take_eqp_step ) THEN
                     WRITE( data%out, "( A, '   - ', A3, ' step is',           &
                    &  ' very successful, stepsize =', ES11.4 )" )             &
                       prefix, data%d_name, data%step_eqp
                   ELSE
                     WRITE( data%out, "( A, '   - ', A3, ' step is',           &
                    &  ' very successful' )" ) prefix, data%d_name
                   END IF
                 ELSE 
                   IF ( data%take_eqp_step ) THEN
                     WRITE( data%out, "( A, '   - ', A3, ' step is',           &
                    &  ' successful, stepsize =', ES11.4 )" )                  &
                      prefix, data%d_name, data%step_eqp
                   ELSE
                     WRITE( data%out, "( A, '   - ', A3, ' step is',           &
                    &  ' successful' )" ) prefix, data%d_name
                   END IF
                 END IF
               END IF
               GO TO 290

!  the new point is unsuccessful

             ELSE
               data%it_type = 'u'
               IF ( data%printt ) THEN
                 IF ( data%take_eqp_step ) THEN
                   WRITE( data%out, "( A, '   - ', A3, ' step is ',            &
                  &  'unsuccessful, stepsize =', ES11.4 )" )                   &
                     prefix, data%d_name, data%step_eqp
                 ELSE
                   WRITE( data%out, "( A, '   - ', A3, ' step is',             &
                  & ' unsuccessful' )" ) prefix, data%d_name
                 END IF
               END IF
             END IF

!  the new point is not acceptable to the filter

           ELSE
             data%it_type = 'r'
             IF ( data%take_eqp_step ) data%n_pr_max = 0
             IF ( data%printt ) WRITE( data%out,                               &
            "( /, A, ' * ', A3, ' point not acceptable to filter (v,o) = (',   &
              &  ES10.4, ',', ES11.4, ')', /, A, 26X, ' vs current (v,o) = (', &
              &  ES10.4, ',', ES11.4, ')' )" )                                 &
                 prefix, data%d_name, primal_infeasibility_trial, obj_trial,   &
                 prefix, inform%primal_infeasibility, inform%obj
           END IF  
         END IF  

!  try the RLP step after the EQP step

         IF ( data%take_eqp_step ) THEN
           data%take_eqp_step = .FALSE.
           GO TO 200
         ELSE
           IF ( data%gtd < zero .AND.                                          &
                inform%primal_infeasibility <= data%stop_p ) THEN
             data%step_rlp = half * data%step_rlp
             GO TO 200
           END IF
         END IF
         data%d_type = ' '

!  ------------------------------------------------
!  end of loop over the potential search directions
!  ------------------------------------------------

 290   CONTINUE

!PLPLOT  tlabel = REPEAT( ' ', 80 )
!PLPLOT  WRITE( titer, "( I0 )" ) inform%iter
!PLPLOT  IF ( data%successful ) THEN
!PLPLOT    tlabel = "Filter - successful iteration " // titer
!PLPLOT    data%n_success = data%n_success + 1
!PLPLOT    data%success( data%n_success )%o = obj_trial
!PLPLOT    data%success( data%n_success )%v = primal_infeasibility_trial
!PLPLOT  ELSE
!PLPLOT    tlabel = "Filter - unsuccessful iteration "  // titer
!PLPLOT  END IF
!PLPLOT  CALL pladv(0)              !  advance the (sub-)page
!PLPLOT  CALL plvsta()              !  select standard viewport and box limits
!!PLPLOT  CALL plwind( data%v_min, data%v_max, data%o_min, data%o_max )
!PLPLOT  CALL plwind( data%v_min, data%v_max, data%o_minl, data%o_maxl )
!PLPLOT  CALL plbox( "bcnst", 0.0_plflt, 0, "bcnstv", 0.0_plflt, 0) ! and box it
!  * plot the current filter in green
!PLPLOT  DO i = 1, data%FILTER_data%n_filter     !  loop over the current filter
!PLPLOT    rect_x( 1 ) = LOG10( MIN( MAX( data%FILTER_data%filter( i )%v,      &
!PLPLOT      data%v_mine ), data%v_maxe ) )                 ! set (v,0) entry 
!PLPLOT    rect_y( 1 ) = FASTR_oe( data%FILTER_data%filter( i )%o, data%o_opt, &
!PLPLOT                            data%o_minl, data%o_maxl ) 
!!PLPLOT    rect_y( 1 ) = MIN( MAX( data%FILTER_data%filter( i )%o,            &
!!PLPLOT                            data%o_min ), data%o_max 
!PLPLOT    rect_x( 2 ) = rect_x( 1 ) ; rect_y( 4 ) = rect_y( 1 ) ! & create the 
!PLPLOT    rect_y( 2 ) = data%o_maxl ; rect_x( 3 ) = data%v_max   ! rest of the
!PLPLOT    rect_y( 3 ) = data%o_maxl ; rect_x( 4 ) = data%v_max   ! filter
!PLPLOT    CALL plcol0( 3 )                    !  green existing filter entry
!PLPLOT    CALL plfill( rect_x(1 : rect_n ), rect_y( 1 : rect_n ) ) 
!PLPLOT  END DO
!  * plot past successful points in blue
!PLPLOT  CALL plcol0( 9 )              !  mark previous successes in blue
!PLPLOT  DO i = 0, data%n_success - 1  !  loop over previous succeses
!PLPLOT    rect_x( 1 ) = LOG10( MIN( MAX( data%success( i )%v, data%v_mine ),  &
!PLPLOT                       data%v_maxe ) )                 ! set (v,0) entry 
!PLPLOT    rect_y( 1 ) = FASTR_oe( data%success( i )%o, data%o_opt,            &
!PLPLOT                            data%o_minl, data%o_maxl )
!!PLPLOT    rect_y( 1 ) = MIN( MAX(data%success( i )%o, data%o_min ),data%o_max)
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  END DO
!  * plot unsuccesful eqp point in yellow
!PLPLOT  IF ( data%take_eqp_step ) THEN
!PLPLOT    rect_x( 1 ) = LOG10( MIN( MAX( data%primal_infeasibility_eqp,       &
!PLPLOT                                   data%v_mine ), data%v_maxe ) )
!PLPLOT    rect_y( 1 ) = FASTR_oe( data%obj_eqp, data%o_opt,                   &
!PLPLOT                            data%o_minl, data%o_maxl )
!!PLPLOT    rect_y( 1 ) = MIN( MAX( data%obj_eqp, data%o_min ), data%o_max )
!PLPLOT    CALL plcol0( 2 )            !  yellow eqp new point
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 22 )
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  END IF
!  * plot new point in red
!PLPLOT  rect_x( 1 ) = LOG10( MIN( MAX( primal_infeasibility_trial,            &
!PLPLOT                                 data%v_mine ), data%v_maxe ) )
!PLPLOT  rect_y( 1 ) = FASTR_oe( obj_trial, data%o_opt,                        &
!PLPLOT                          data%o_minl, data%o_maxl )
!!PLPLOT  rect_y( 1 ) = MIN( MAX( obj_trial, data%o_min ), data%o_max )
!PLPLOT  IF ( obj_trial >= data%o_min .AND. obj_trial <= data%o_max ) THEN
!PLPLOT    CALL plcol0( 1 )            !  red new point
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 22 )
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  ELSE
!PLPLOT    CALL plcol0( 1  )           !  red new point (off edge of plot)
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 22 )
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  END IF
!  * plot current point in salmon
!PLPLOT  CALL plcol0( 14 )            !  current point and it shadow
!PLPLOT  rect_x( 1 ) = LOG10( MIN( MAX( inform%primal_infeasibility,           &
!PLPLOT                data%v_mine ), data%v_maxe ) ) ! set current (v,o) entry
!PLPLOT  rect_y( 1 ) = FASTR_oe( inform%obj, data%o_opt,                       &
!PLPLOT                          data%o_minl, data%o_maxl )
!!PLPLOT  rect_y( 1 ) = MIN( MAX( inform%obj, data%o_min ), data%o_max )
!PLPLOT  CALL pljoin( rect_x( 1 ), rect_y( 1 ), rect_x( 1 ), data%o_maxl )
!!PLPLOT  CALL pljoin( rect_x( 1 ), rect_y( 1 ), rect_x( 1 ), data%o_max )
!PLPLOT  CALL pljoin( rect_x( 1 ), rect_y( 1 ), data%v_max, rect_y( 1 ) )
!PLPLOT  CALL plcol0( 14 )   !  print a black dot
!PLPLOT  CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )

!  .......................
!  the step was successful
!  .......................

       IF ( data%successful ) THEN

!    data%pr_max = MAX( half * (data%pr_max + inform%primal_infeasibility ),   &
!                       100.0_wp * data%stop_p )
!write(6,*) ' pr_max ', data%pr_max

!  the objective from the RLP has not decreased sufficiently. Add the current
!   point to the filter

         IF ( add_all_to_filter .OR.                                           &
              ( inform%primal_infeasibility > data%stop_p .AND.                &
                data%delta_l < data%control%delta_feas *                       &
                 inform%primal_infeasibility ** 2 ) ) THEN
!PLPLOT    tlabel = TRIM( tlabel ) // " - update filter"
           IF ( data%printt ) WRITE( data%out, "( A, ' - Current point (v,o)', &
            & ' = (', ES10.4, ', ', ES11.4, ') added to filter:', /, A,        &
            & '   * v-step (insufficient RLP reduction)' )" )                  &
             prefix, inform%primal_infeasibility, inform%obj, prefix
           o = inform%obj - data%control%gamma_filter                          &
                 * inform%primal_infeasibility
           v = data%control%beta_filter * inform%primal_infeasibility
           CALL FILTER_update_filter( o, v, data%FILTER_data,                  &
                                      data%control%FILTER_control,             &
                                      inform%FILTER_inform )
         END IF

!  move to the new point

         inform%obj = obj_trial
         nlp%X( : nlp%n ) = data%X_trial( : nlp%n )
         nlp%C( : nlp%m ) = data%C_trial( : nlp%m )
         inform%primal_infeasibility = primal_infeasibility_trial
         data%new_point = .TRUE.
         data%new_gradient = .TRUE.

!        IF ( data%mu_new < data%mu ) data%mu = MAX( data%mu_new, mu_tiny )
!        IF ( data%mu_new < data%mu ) data%mu = MAX( data%mu_new, mu_tiny,     &
!          data%control%rlp_radius_reduce * data%mu )

!  reset the radius ?

         IF ( data%very_successful ) THEN
           IF ( data%take_eqp_step ) THEN
             data%radius_eqp                                                   &
               = data%control%eqp_radius_increase * data%radius_eqp
           ELSE
             data%mu = MIN( MAX( data%control%initial_radius_rlp,              &
                                 data%control%rlp_radius_increase * data%mu ), &
                            data%mu_max )
           END IF
!        ELSE
!          data%radius_eqp = data%control%eqp_radius_increase * data%radius_eqp
         END IF

!  .........................
!  the step was unsuccessful
!  .........................

       ELSE

!  reduce the trust-region radius

!write(6,*) ' mu_new, tiny ', data%mu_new, mu_tiny
         IF ( data%mu_new >= mu_tiny ) THEN
           data%mu =                                                           &
             data%control%rlp_radius_reduce * MIN( data%mu_new, data%mu )

!  if the active step will not change with a reduction in mu, enter the
!  restoration phase

         ELSE
           IF ( inform%primal_infeasibility <= data%stop_p ) THEN
             IF ( data%gtd < 0 ) THEN
               data%mu = data%control%rlp_radius_reduce * data%mu
             ELSE
               data%restoration = .TRUE. ; GO TO 300
             END IF
           ELSE
             IF ( data%mu_new >= mu_tiny ) THEN
               data%mu = data%control%rlp_radius_reduce * data%mu
             ELSE
               data%restoration = .TRUE. ; GO TO 300
             END IF
           END IF

!write(6,*) ' new_point ', data%new_point
!write(6,*) inform%primal_infeasibility <= data%stop_p

!             IF ( data%new_point ) THEN
!            IF ( inform%primal_infeasibility <= data%stop_p ) THEN
!              data%mu = data%control%rlp_radius_reduce * data%mu
!            ELSE
!              data%new_point = .FALSE.
!              data%mu =                                                       &
!                data%control%rlp_radius_reduce * MIN( data%mu_new, data%mu )



!           ELSE
!           IF ( data%gtd < 0 ) THEN
!               data%mu = data%control%rlp_radius_reduce * data%mu
!           ELSE 
!             data%restoration = .TRUE. ; GO TO 300
!           END IF

!          IF ( data%new_point ) THEN
!            IF ( inform%primal_infeasibility <= data%stop_p ) THEN
!              data%mu = data%control%rlp_radius_reduce * data%mu
!            ELSE
!              data%new_point = .FALSE.
!              data%mu =                                                       &
!                data%control%rlp_radius_reduce * MIN( data%mu_new, data%mu )
!            END IF
!          ELSE 
!            data%restoration = .TRUE. ; GO TO 300
!          END IF

!          IF ( .NOT. data%new_point ) THEN
!!           write(6,*) data%norm_dlp, data%old_norm_dlp
!            ratio_norm_dlp = data%norm_dlp / data%old_norm_dlp
!            IF ( data%printt ) WRITE( data%out, "( A, '   - ratio new/old',   &
!           &   ' norm d_lp = ', ES10.4 )" ) prefix, ratio_norm_dlp
!  if the norm of the step has not significantly decreased, enter the
!  restoration phase
!            IF ( ratio_norm_dlp > 0.9_wp .AND.                                &
!                 primal_infeasibility > zero ) THEN
!              data%restoration = .TRUE. ; GO TO 300
!            END IF
!          END IF
!          data%mu = half * data%mu
!          data%mu = tenth * data%mu
         END IF

!  reduce the EQP trust-region radius

         IF ( data%n_pr_max <= 1 ) THEN
           data%radius_eqp = data%control%eqp_radius_reduce * data%radius_eqp
         ELSE
           data%radius_eqp =                                                   &
           ( data%control%eqp_radius_reduce ** data%n_pr_max ) * data%radius_eqp
         END IF         
         data%new_gradient = .FALSE.
       END IF

!  ----------------------------------------------------------------------------
!                 SECOND-ORDER LAGRANGE MULTIPLIER UPDATES
!  ----------------------------------------------------------------------------

       IF ( data%control%multipliers == 2 ) THEN

!  compute the changes in multipliers and dual variables

         DO i = 1, nlp%m 
           IF ( nlp%C_status( i ) == 0 ) THEN
             data%DY( i ) = - nlp%Y( i )
           ELSE
             data%DY( i ) =                                                    &
               data%EQP_prob%Y( ABS( nlp%C_status( i ) ) ) - nlp%Y( i )
           END IF
         END DO
         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) == 0 ) THEN
             data%DZ( i ) = - nlp%Z( i )
           ELSE
             data%DZ( i ) =                                                    &
               data%EQP_prob%Y( ABS( nlp%X_status( i ) ) ) - nlp%Z( i )
           END IF
         END DO

!  compute A^T dy

         data%ATDY( : nlp%n ) = data%DZ( : nlp%n )
         DO l = 1, data%CQP_prob%A%ne
           i = data%CQP_prob%A%col( l )
           data%ATDY( i ) = data%ATDY( i ) +                                   &
             data%CQP_prob%A%val( l ) * data%DY( data%CQP_prob%A%row( l ) )
         END DO

!  find the minimizer alpha of || g_l - alpha A^T dy ||_2^2

         alpha = DOT_PRODUCT( data%ATDY( : nlp%n ), data%ATDY( : nlp%n ) )
         IF ( alpha > zero ) alpha =                                           &
           DOT_PRODUCT( nlp%GL( : nlp%n ), data%ATDY( : nlp%n ) ) / alpha

!  update the multipliers

         IF ( alpha /= zero ) THEN
           nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) + alpha * data%DY( : nlp%m )
           nlp%Z( : nlp%n ) = nlp%Z( : nlp%n ) + alpha * data%DZ( : nlp%n )
           nlp%GL( : nlp%n ) = nlp%GL( : nlp%n ) - alpha * data%ATDY( : nlp%n )
           inform%dual_infeasibility =                                         &
             OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) )
           IF ( data%printm ) WRITE( data%out,                                 &
             "( A, '   second-order dual residual = ', ES10.4 )") prefix,      &
               inform%dual_infeasibility
         END IF
       END IF

!  update the Lagrange multipliers; also compute the number of multipliers 
!  with the wrong signs

       IF ( data%successful .OR. ( data%norm_deqp <= data%stop_p               &
         .AND. primal_infeasibility_trial <= data%stop_p ) ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
           "( /, A, ' * updating the Lagrange multipliers' )" ) prefix
         data%n_mult_wrong_sign = 0
         DO i = 1, nlp%m 
           IF ( nlp%C_status( i ) == 0 ) THEN
           ELSE
             IF ( nlp%C_l( i ) /= nlp%C_u( i ) .AND.                           &
                ( ( nlp%C_status( i ) < 0 .AND. nlp%Y( i ) < - y_tiny ) .OR.   &
                  ( nlp%C_status( i ) > 0 .AND. nlp%Y( i ) > y_tiny ) ) )      &
               data%n_mult_wrong_sign = data%n_mult_wrong_sign + 1
           END IF
         END DO
!        WRITE( 6, * ) ' max y ', MAXVAL( ABS( nlp%Y( : nlp%m ) ) )

         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) == 0 ) THEN
             nlp%Z( i ) = zero
           ELSE
             nlp%Z( i ) = data%EQP_prob%Y( ABS( nlp%X_status( i ) ) )
             IF ( nlp%X_l( i ) /= nlp%X_u( i ) .AND.                           &
                ( ( nlp%X_status( i ) < 0 .AND. nlp%Z( i ) < - z_tiny ) .OR.   &
                  ( nlp%X_status( i ) > 0 .AND. nlp%Z( i ) > z_tiny ) ) )      &
               data%n_mult_wrong_sign = data%n_mult_wrong_sign + 1
           END IF
         END DO
!        WRITE( 6, * ) ' max z ', MAXVAL( ABS( nlp%Z( : nlp%n ) ) )
         IF ( data%printt ) WRITE( data%out,                                   &
           "( A, '    ', I0, ' multiplier', A, ' ', A, ' the wrong sign')" )   &
           prefix, data%n_mult_wrong_sign,                                     &
           TRIM( STRING_pleural( data%n_mult_wrong_sign ) ),                   &
           TRIM( STRING_have( data%n_mult_wrong_sign ) )

         IF ( data%printd ) THEN
           WRITE( data%out, 2020 ) prefix
           WRITE( data%out, "( A, I6, 5ES12.4 )" )                             &
            ( prefix, i, nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ),               &
              data%EQP_prob%X( i ), nlp%Z( i ), i = 1, nlp%n )
           WRITE( data%out, "( / )" )
         END IF
       END IF

!  ----------------------------------------------------------------------------
!                       RESTORATION PHASE
!  ----------------------------------------------------------------------------

  300  CONTINUE

!PLPLOT  IF ( data%restoration )                                               &
!PLPLOT    tlabel = TRIM( tlabel ) // " - update filter and enter restoration"
!PLPLOT  CALL plcol0( 15 )         !  update the figure legend
!PLPLOT  CALL pllab( "log@d10@u violation", "objective value", tlabel )

  310  CONTINUE
!      IF ( .NOT. data%restoration ) CYCLE
       IF ( data%restoration ) THEN
         data%it_type = 'R'

         IF ( data%printt ) WRITE( data%out, "( A, ' - Current point (v,o) ',  &
          & '= (', ES10.4, ', ', ES11.4, ') added to filter', //, A,           &
          & '   * Subproblem infeasible, entering restoration phase' )" )      &
            prefix, inform%primal_infeasibility, inform%obj, prefix

!  add the current point to the filter

!        data%bdry = 'r'
!write(6,*) "(f,c)", inform%obj, inform%primal_infeasibility
         IF ( inform%primal_infeasibility > data%stop_p ) THEN
           o = inform%obj - data%control%gamma_filter                          &
                 * inform%primal_infeasibility
           v = data%control%beta_filter * inform%primal_infeasibility
           CALL FILTER_update_filter( o, v, data%FILTER_data,                  &
                                      data%control%FILTER_control,             &
                                      inform%FILTER_inform )
         END IF

!  use the best feasible point if there is one

         IF ( data%x_best_set ) THEN
           nlp%X( : nlp%n ) = data%X_best( : nlp%n )
           nlp%C( : nlp%m ) = data%C_best( : nlp%m )

!  use this point if it is feasible

           IF (  data%v_best == zero ) THEN
             nlp%f = data%f_best
             inform%f_eval = inform%f_eval - 1
             data%infeasible = .FALSE.
             GO TO 380
           END IF
         END IF

!  find a new point for which the infeasibility is less than c_required

         c_required = MIN( point1 *                                            &
                        MINVAL( data%FILTER_data%filter( : data%n_filter )%v ),&
                                half * data%pr_best,                           &
                                data%control%stop_p_restoration )

!  find scaling factors for the constraints

         nlp%C_scale( : nlp%m ) = one
         DO l = 1, data%CQP_prob%A%ne
           i = data%CQP_prob%A%row( l )
           nlp%C_scale( i )                                                    &
             = MAX( nlp%C_scale( i ), ABS( data%CQP_prob%A%val( l ) ) )
         END DO
!        nlp%C_scale( : nlp%m ) = ten ** ( - 3 )
!        nlp%C_scale( : nlp%m ) = ten
!        nlp%C_scale( : nlp%m ) = ten ** 2
!        nlp%C_scale( : nlp%m ) = ten ** 3
!        nlp%C_scale( : nlp%m ) = MAX( nlp%C_l( : nlp%m ) - nlp%C( : nlp%m ),  &
!                                      nlp%C( : nlp%m ) - nlp%C_u( : nlp%m ),  &
!                                      one )
         nlp%C_scale( : nlp%m ) = one
         IF ( data%printd ) THEN
           WRITE( data%out, "( A, ' C_scale = ', /, ( 1X, 5ES12.4 ) )" )       &
             prefix, nlp%C_scale( : nlp%m )
           WRITE( data%out, "( A, ' C = ', /, ( 1X, 5ES12.4 ) )" )             &
             prefix, MAX( nlp%C_l( : nlp%m ) - nlp%C( : nlp%m ),               &
                          nlp%C( : nlp%m ) - nlp%C_u( : nlp%m ), zero )
         END IF

!  prepare for the restoration

!        control_restoration = data%control
         data%branch_restoration = 1
         CALL CPU_TIME( data%time_record ) ; CALL CLOCK_time( data%clock_record)
         data%iter_restoration = inform%iter
       END IF

!  return from reverse communication to obtain problem function values

  370  CONTINUE
       IF ( data%restoration ) THEN
         CALL FASTR_restoration( nlp, c_required, control, inform, data,       &
                                 userdata, eval_FC, eval_GJ, eval_HL,          &
                                 eval_HLPROD )

!  use reverse communication to obtain problem function values

! write(6,*) ' exit status from restoration ', inform%status

         IF ( inform%status >= 12 .AND. inform%status <= 16 ) THEN
           data%branch = 7 ; RETURN
         END IF

         CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now ) 
         inform%time%restoration =                                             &
           inform%time%restoration + data%time_now - data%time_record
         inform%time%clock_restoration =                                       &
           inform%time%clock_restoration + data%clock_now - data%clock_record

         data%infeasible = inform%status == GALAHAD_error_primal_infeasible
         data%print_iteration_header = .TRUE.
!        CALL FASTR_restoration_old( nlp%n, nlp%m, nlp%X_l, nlp%X_u, nlp%X,    &
!!                                   nlp%C_l, nlp%C_u, nlp%C, nlp%Y,           &
!                                    nlp%C_l, nlp%C_u, nlp%C, nlp%EQUATION,    &
!                                    c_required, data%printt, data%printm,     &
!                                    data%LANCELOT_prob, data%LANCELOT_data,   &
!                                    data%LANCELOT_arrays, control, inform )

         IF ( data%printt ) WRITE( data%out,                                   &
        "( A, '   - on exit from restoration: status = ', I0, ', violation = ',&
       &   ES10.4, ', iter = ', I0 )" ) prefix, inform%status,                 &
             SQRT( two * inform%obj_restoration ),                             &
             inform%iter - data%iter_restoration

!  evaluate the objective and general constraint function values
   
         IF ( data%reverse_fc ) THEN
           data%branch = 9 ; inform%status = 2 ; RETURN
         ELSE  
           CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
         END IF
       END IF

!  return from reverse communication to obtain the objective value

 380   CONTINUE
       IF ( data%restoration ) THEN
         inform%obj = nlp%f
!        write(6,*) ' f after ', inform%obj
         inform%f_eval = inform%f_eval + 1

!  compute the norm of the violation

         inform%primal_infeasibility =                                         &
           OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                  &
                                     nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
! write(6,*)  ' infeasibility after restoration = ', inform%primal_infeasibility

!  exit if the problem is deemed to be locally infeasible

         IF ( data%infeasible ) THEN
           IF ( data%x_feas ) THEN
             inform%obj = data%f_best
             nlp%X( : nlp%n ) = data%X_best( : nlp%n )
             nlp%C( : nlp%m ) = data%C_best( : nlp%m )
           ELSE
             GO TO 905
           END IF
         END IF

         data%CQP_prob%A%ne = data%J_ne
         data%new_point = .TRUE.
         data%restoration = .FALSE.
         data%successful = .TRUE.
         data%new_gradient = .TRUE.
         data%rest = ' '

!  re-initialize the radii

         data%mu = data%control%initial_radius_rlp
         data%radius_eqp = data%control%initial_radius_eqp

!PLPLOT  tlabel = REPEAT( ' ', 80 )
!PLPLOT  WRITE( titer, "( I0 )" ) inform%iter
!PLPLOT  tlabel = "Filter - return from restoration iteration " // titer
!PLPLOT  data%n_success = data%n_success + 1
!PLPLOT  data%success( data%n_success )%o = inform%obj
!PLPLOT  data%success( data%n_success )%v = inform%primal_infeasibility
!PLPLOT  CALL pladv(0)              !  advance the (sub-)page
!PLPLOT  CALL plvsta()              !  select standard viewport and box limits
!PLPLOT  CALL plwind( data%v_min, data%v_max, data%o_minl, data%o_maxl )
!!PLPLOT  CALL plwind( data%v_min, data%v_max, data%o_min, data%o_max )
!PLPLOT  CALL plbox( "bcnst", 0.0_plflt, 0, "bcnstv", 0.0_plflt, 0) ! and box it
!  * plot the current filter in green
!PLPLOT  DO i = 1, data%FILTER_data%n_filter     !  loop over the current filter
!PLPLOT    rect_x( 1 ) = LOG10( MIN( MAX( data%FILTER_data%filter( i )%v,      &
!PLPLOT                      data%v_mine ), data%v_maxe ) )    ! set (v,0) entry
!PLPLOT    rect_y( 1 ) = FASTR_oe( data%FILTER_data%filter( i )%o, data%o_opt, &
!PLPLOT                            data%o_minl, data%o_maxl ) 
!!PLPLOT    rect_y( 1 ) = MIN( MAX( data%FILTER_data%filter( i )%o,            &
!!PLPLOT                            data%o_min ), data%o_max )
!PLPLOT    rect_x( 2 ) = rect_x( 1 ) ; rect_y( 4 ) = rect_y( 1 ) ! & create the 
!PLPLOT    rect_y( 2 ) = data%o_maxl ; rect_x( 3 ) = data%v_max   ! rest of the
!PLPLOT    rect_y( 3 ) = data%o_maxl ; rect_x( 4 ) = data%v_max   ! filter
!PLPLOT    CALL plcol0( 3 )                    !  green existing filter entry
!PLPLOT    CALL plfill( rect_x(1 : rect_n ), rect_y( 1 : rect_n ) ) 
!PLPLOT  END DO
!  * plot past successful points in blue
!PLPLOT  CALL plcol0( 9 )              !  mark previous successes in blue
!PLPLOT  DO i = 0, data%n_success - 1  !  loop over previous succeses
!PLPLOT    rect_x( 1 ) = LOG10( MIN( MAX( data%success( i )%v, data%v_mine ),  &
!PLPLOT                       data%v_maxe ) )                 ! set (v,0) entry 
!PLPLOT    rect_y( 1 ) = FASTR_oe( data%success( i )%o, data%o_opt,            &
!PLPLOT                            data%o_minl, data%o_maxl ) 
!!PLPLOT    rect_y( 1 ) = MIN( MAX(data%success( i )%o, data%o_min ),data%o_max)
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  END DO
!  * plot new point in red
!PLPLOT  rect_x( 1 ) = LOG10( MIN( MAX( inform%primal_infeasibility,           &
!PLPLOT                data%v_mine ), data%v_maxe ) ) ! set current (v,o) entry
!PLPLOT  rect_y( 1 ) = FASTR_oe( inform%obj, data%o_opt,                       &
!PLPLOT                          data%o_minl, data%o_maxl ) 
!!PLPLOT  rect_y( 1 ) = MIN( MAX( inform%obj, data%o_min ), data%o_max )
!PLPLOT  IF ( inform%obj >= data%o_min .AND. inform%obj <= data%o_max ) THEN
!PLPLOT    CALL plcol0( 1 )            !  red new point
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 22 )
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 17 )
!PLPLOT  ELSE
!PLPLOT    CALL plcol0( 1  )           !  red new point (off edge of plot)
!PLPLOT    CALL plpoin( rect_x( 1 : 1 ), rect_y( 1 : 1 ), 16 )
!PLPLOT  END IF
!PLPLOT  CALL plcol0( 15 )         !  update the figure legend
!PLPLOT  CALL pllab( "log@d10@u violation", "log@d10@u objective violation",   &
!PLPLOT               tlabel )
       END IF

!  set print agenda for the next iteration

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%CQP_control%print_level = data%print_level_cqp
         data%control%CQP_control%print_level = data%print_level_cqp
         data%control%EQP_control%print_level = data%print_level_eqp
         data%control%EQP_control%SBLS_control%print_level                     &
           = data%print_level_sbls
         data%control%EQP_control%GLTR_control%print_level                     &
           = data%print_level_gltr
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%CQP_control%print_level = 0
         data%control%CQP_control%SBLS_control%print_level = 0
         data%control%EQP_control%print_level = 0
         data%control%EQP_control%SBLS_control%print_level = 0
         data%control%EQP_control%GLTR_control%print_level = 0
       END IF

       IF ( data%print_level > 1 .OR.                                          &
           data%control%EQP_control%print_level > 0 .OR.                       &
           data%control%CQP_control%print_level > 0 )                          &
         data%print_iteration_header = .TRUE.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    M A I N    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     GO TO 100

!  -------------
!  Normal return
!  -------------

 900 CONTINUE
     inform%status = 0

!  print the solution

 905 CONTINUE

!    WRITE( 44, * ) ' nlp%X_status '
!    WRITE( 44, "( ( 10I7 ) )" ) nlp%X_status
!    WRITE( 44, * ) ' nlp%C_status '
!    WRITE( 44, "( ( 10I7 ) )" ) nlp%C_status

     l = 2
     IF ( data%control%fulsol ) l = nlp%n 
     IF ( data%control%print_level >= 10 ) l = nlp%n

     WRITE( data%out, "( /, A, ' Solution: ', /, A, '                        ',&
    &           '        <------ Bounds ------> ', /, A,                       &
    &           '      # name          value   ',                              &
    &           '    Lower       Upper       Dual ' )" ) prefix, prefix, prefix
     DO j = 1, 2 
       IF ( j == 1 ) THEN 
         ir = 1 ; ic = MIN( l, nlp%n ) 
       ELSE 
         IF ( ic < nlp%n - l ) WRITE( data%out, 2010 ) prefix
         ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
       END IF 
       DO i = ir, ic 
         WRITE( data%out, 2000 ) prefix, i, nlp%VNAMES( i ), nlp%X( i ),       &
           nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
       END DO
     END DO

     IF ( nlp%m > 0 ) THEN
       l = 2
       IF ( data%control%fulsol ) l = nlp%m
       IF ( data%control%print_level >= 10 ) l = nlp%m

       WRITE( data%out, "( /, A, ' Constraints:', /, A, '                   ', &
      &        '               <------ Bounds ------> ', /, A,                 &
      &        '      # name           value   ',                              &
      &        '    Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
       DO j = 1, 2 
         IF ( j == 1 ) THEN 
           ir = 1 ; ic = MIN( l, nlp%m ) 
         ELSE 
           IF ( ic < nlp%m - l ) WRITE( data%out, 2010 ) prefix
           ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
         END IF 
         DO i = ir, ic 
           WRITE( data%out, 2000 ) prefix, i, nlp%CNAMES( i ), nlp%C( i ),     &
             nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
         END DO
       END DO
     END IF

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start

     IF ( nlp%m > 0 ) THEN ; max_y = MAXVAL( ABS( nlp%Y( : nlp%m ) ) ) ; 
       ELSE ; max_y = zero ; END IF

     WRITE( data%out, "( /, A, ' Problem: ', 16X, A10,                         &
    &          '     Solver: ', 9X, '  FASTr', /, A,                           &
    &  ' n              =     ',bn, I12, '       m               = ',bn, I12,/,&
    & A, ' Objective      = ', ES16.8, '       Complementarity = ', ES12.4, /, &
    & A,' Violation      =     ',ES12.4, '       Dual infeas     = ', ES12.4,/,&
    & A,' Max multiplier =     ',ES12.4, '       Max dual var.   = ', ES12.4,/,&
    & A,' Iterations     =     ',bn, I12, '       Time            = ', F12.2)")&
      prefix, nlp%pname, prefix, nlp%n, nlp%m, prefix, inform%obj,             &
      inform%complementary_slackness, prefix, inform%primal_infeasibility,     &
      inform%dual_infeasibility, prefix, max_y, MAXVAL( ABS( nlp%Z( : nlp%n))),&
      prefix, inform%iter, inform%time%clock_total

     IF ( data%control%error > 0 .AND. data%control%print_level > 0 ) THEN
       CALL SYMBOLS_status( inform%status, data%control%error, prefix,         &
                            'FASTR_solve' )
     END IF
     GO TO 999

!  -------------
!  Error returns
!  -------------

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start
     GO TO 999

!  other errors

 990 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start
     IF ( data%control%error > 0 .AND. data%control%print_level > 0 ) THEN
       CALL SYMBOLS_status( inform%status, data%control%error, prefix,         &
                            'FASTR_solve' )
     END IF
     GO TO 999

!  all returns

 999 CONTINUE
!PLPLOT  IF ( data%n_success >= 0 ) THEN
!PLPLOT    data%o_min = MINVAL( data%success( 0 : data%n_success )%o )
!PLPLOT    data%o_max = MAXVAL( data%success( 0 : data%n_success )%o )
!PLPLOT    delta_f = point1 * ( data%o_max - data%o_min )
!PLPLOT    data%o_max = data%o_max + delta_f
!PLPLOT    data%o_min = data%o_min - delta_f
!PLPLOT    data%o_opt = inform%obj
!PLPLOT    REWIND( ffiledevice )
!PLPLOT    WRITE( ffiledevice, * ) data%o_min, data%o_max, data%o_opt
!PLPLOT    CLOSE( ffiledevice )
!PLPLOT  END IF
!PLPLOT  CALL plend( )  !  end plotting session
     RETURN

!  non-executable statements

 2000 FORMAT( A, I7, 1X, A10, 4ES12.4 ) 
 2010 FORMAT( A, 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2020 FORMAT( /, A, '     i      X_l          X          X_u',                 &
                    '         Dx           Z')
 2030 FORMAT( A, '   - Error return from EQP_solve, status = ', I0 )

!  End of subroutine FASTR_solve

     END SUBROUTINE FASTR_solve

!PLPLOT  FUNCTION FASTR_oe( o, o_opt, o_minl, o_maxl )
!PLPLOT  REAL ( KIND = wp ) :: FASTR_oe
!PLPLOT  REAL ( KIND = wp ), INTENT( IN ) :: o, o_opt, o_minl, o_maxl
!PLPLOT  IF ( o - o_opt > tenm10 ) THEN
!PLPLOT    FASTR_oe = LOG10( ten10 * ( o - o_opt ) )
!PLPLOT  ELSE IF ( o - o_opt < - tenm10 ) THEN
!PLPLOT    FASTR_oe = - LOG10( ten10 * ( o_opt - o ) )
!PLPLOT  ELSE
!PLPLOT    FASTR_oe = zero
!PLPLOT  END IF
!PLPLOT  FASTR_oe = MIN( MAX( FASTR_oe, o_minl ), o_maxl )
!PLPLOT  END FUNCTION FASTR_oe

!-*-*- G A L A H A D -  F A S T R _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE FASTR_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FASTR_data_type ), INTENT( INOUT ) :: data
     TYPE ( FASTR_control_type ), INTENT( IN ) :: control
     TYPE ( FASTR_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate arrays set for CQP

     CALL CQP_terminate( data%CQP_data, control%CQP_control,                   &
                         inform%CQP_inform )
     IF ( inform%CQP_inform%status /= 0 ) THEN
       inform%status = - 2
       inform%alloc_status = inform%CQP_inform%alloc_status
       inform%bad_alloc = inform%CQP_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate arrays set for EQP

     CALL EQP_terminate( data%EQP_data, control%EQP_control,                   &
                         inform%EQP_inform )
     IF ( inform%EQP_inform%status /= 0 ) THEN
       inform%status = inform%EQP_inform%status
       inform%alloc_status = inform%EQP_inform%alloc_status
       inform%bad_alloc = inform%EQP_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate arrays set for FILTER

     CALL FILTER_terminate( data%FILTER_data, control%FILTER_control,          &
                            inform%FILTER_inform )
     IF ( inform%FILTER_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_inform%status
       inform%alloc_status = inform%FILTER_inform%alloc_status
       inform%bad_alloc = inform%FILTER_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL FILTER_terminate( data%FILTER_restoration_data,                      &
                            control%FILTER_control,                            &
                            inform%FILTER_restoration_inform )
     IF ( inform%FILTER_restoration_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_restoration_inform%status
       inform%alloc_status = inform%FILTER_restoration_inform%alloc_status
       inform%bad_alloc = inform%FILTER_restoration_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate all remaining allocated arrays

     array_name = 'fastr: data%XFREE'
     CALL SPACE_dealloc_array( data%XFREE,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!     array_name = 'fastr: nlp%gL'
!     CALL SPACE_dealloc_array( nlp%gL,                                        &
!        inform%status, inform%alloc_status, array_name = array_name,          &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%G_p'
     CALL SPACE_dealloc_array( data%G_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!    array_name = 'fastr: data%CQP_prob%H%row'
!    CALL SPACE_dealloc_array( data%CQP_prob%H%row,                            &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
 
!    array_name = 'fastr: data%CQP_prob%H%col'
!    CALL SPACE_dealloc_array( data%CQP_prob%H%col,                            &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%H%val'
     CALL SPACE_dealloc_array( data%CQP_prob%H%val,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%G'
     CALL SPACE_dealloc_array( data%CQP_prob%G,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%A%row'
     CALL SPACE_dealloc_array( data%CQP_prob%A%row,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%A%col'
     CALL SPACE_dealloc_array( data%CQP_prob%A%col,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%A%val'
     CALL SPACE_dealloc_array( data%CQP_prob%A%val,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%C_l'
     CALL SPACE_dealloc_array( data%CQP_prob%C_l,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%C_u'
     CALL SPACE_dealloc_array( data%CQP_prob%C_u,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%X_l'
     CALL SPACE_dealloc_array( data%CQP_prob%X_l,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%X_u'
     CALL SPACE_dealloc_array( data%CQP_prob%X_u,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%X'
     CALL SPACE_dealloc_array( data%CQP_prob%X,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%C'
     CALL SPACE_dealloc_array( data%CQP_prob%C,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%Y'
     CALL SPACE_dealloc_array( data%CQP_prob%Y,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%CQP_prob%Z'
     CALL SPACE_dealloc_array( data%CQP_prob%Z,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%H%row'
     CALL SPACE_dealloc_array( data%EQP_prob%H%row,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%H%col'
     CALL SPACE_dealloc_array( data%EQP_prob%H%col,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%H%val'
     CALL SPACE_dealloc_array( data%EQP_prob%H%val,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%G'
     CALL SPACE_dealloc_array( data%EQP_prob%G,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%A%row'
     CALL SPACE_dealloc_array( data%EQP_prob%A%row,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%A%col'
     CALL SPACE_dealloc_array( data%EQP_prob%A%col,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%A%val'
     CALL SPACE_dealloc_array( data%EQP_prob%A%val,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%X'
     CALL SPACE_dealloc_array( data%EQP_prob%X,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%C'
     CALL SPACE_dealloc_array( data%EQP_prob%C,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%EQP_prob%Y'
     CALL SPACE_dealloc_array( data%EQP_prob%Y,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!    array_name = 'fastr: nlp%C_status'
!    CALL SPACE_dealloc_array( nlp%C_status,                                   &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!    array_name = 'fastr: nlp%X_status'
!    CALL SPACE_dealloc_array( nlp%X_status,                                   &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%X_trial'
     CALL SPACE_dealloc_array( data%X_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%C_trial'
     CALL SPACE_dealloc_array( data%C_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%S_trial'
     CALL SPACE_dealloc_array( data%S_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%X_best'
     CALL SPACE_dealloc_array( data%X_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%C_best'
     CALL SPACE_dealloc_array( data%C_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%X_p'
     CALL SPACE_dealloc_array( data%X_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%Y_p'
     CALL SPACE_dealloc_array( data%Y_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%Z_p'
     CALL SPACE_dealloc_array( data%Z_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%C_p'
     CALL SPACE_dealloc_array( data%C_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%S_p'
     CALL SPACE_dealloc_array( data%S_p,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%DS'
     CALL SPACE_dealloc_array( data%DS,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%DY'
     CALL SPACE_dealloc_array( data%DY,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%DZ'
     CALL SPACE_dealloc_array( data%DZ,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%DX_trial_rlp'
     CALL SPACE_dealloc_array( data%DX_trial_rlp,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%DS_trial_rlp'
     CALL SPACE_dealloc_array( data%DS_trial_rlp,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%ATDY'
     CALL SPACE_dealloc_array( data%ATDY,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fastr: data%GL'
     CALL SPACE_dealloc_array( data%GL,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine FASTR_terminate

     END SUBROUTINE FASTR_terminate

!-*-*-*-*-  G A L A H A D -  F A S T R _ m u _ n e w  F U N C T I O N   -*-*-*-

     FUNCTION FASTR_mu_new( m, n, mu, X_l, X_u, X, DX, C_l, C_u, C, DC,        &
                            Y, DY, Z, DZ, X_status, C_status,                  &
                            out, printt, printw, prefix )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  check to find the range of values [mu_new,mu] for which the active set for 
!  the current RLP stays active

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

     REAL ( KIND = wp ) :: FASTR_mu_new

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, n, out
     REAL ( KIND = wp ), INTENT( IN ) :: mu
     LOGICAL, INTENT( IN ) :: printt, printw
     CHARACTER ( LEN = * ) :: prefix
     INTEGER, INTENT( IN ), DIMENSION( n ) :: X_status
     INTEGER, INTENT( IN ), DIMENSION( m ) :: C_status
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X, DX, Z, DZ
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u, C, DC, Y, DY

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: mu_break
     CHARACTER ( LEN = 1 ) :: st

     FASTR_mu_new = zero
     IF ( mu > zero ) THEN
       DO i = 1, n 
         IF ( X_l( i ) == X_u( i ) ) THEN
         ELSE IF ( X_status( i ) < 0 ) THEN
           IF ( DZ( i ) /= zero ) THEN
             mu_break = mu - Z( i ) / DZ( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         ELSE IF ( X_status( i ) > 0 ) THEN
           IF ( DZ( i ) /= zero ) THEN
             mu_break = mu - Z( i ) / DZ( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         ELSE
           IF ( DX( i ) > zero ) THEN
             mu_break = mu - ( X_u( i ) - X( i ) ) / DX( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           ELSE IF ( DX( i ) < zero ) THEN
             mu_break = mu - ( X_l( i ) - X( i ) ) / DX( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         END IF
       END DO
       DO i = 1, m
         IF ( C_l( i ) == C_u( i ) ) THEN
         ELSE IF ( C_status( i ) < 0 ) THEN
           IF ( DY( i ) /= zero ) THEN
             mu_break = mu - Y( i ) / DY( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         ELSE IF ( C_status( i ) > 0 ) THEN
           IF ( DY( i ) /= zero ) THEN
             mu_break = mu - Y( i ) / DY( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         ELSE
           IF ( DC( i ) > zero ) THEN
             mu_break = mu - ( C_u( i ) - C( i ) ) / DC( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           ELSE IF ( DC( i ) < zero ) THEN
             mu_break = mu - ( C_l( i ) - C( i ) ) / DC( i )
             IF ( mu_break < mu ) FASTR_mu_new = MAX( FASTR_mu_new, mu_break )
           END IF
         END IF
       END DO

       IF ( printt ) WRITE( out, "( /, A, ' * Active set unchanged on the',    &
      &  ' interval [', ES8.2, ', ', ES8.2, ']' )" ) prefix, FASTR_mu_new, mu 

!  give details about the change in status of the active set ...

       IF ( printw ) THEN
         DO i = 1, n 
           IF ( X_l( i ) == X_u( i ) ) THEN
           ELSE IF ( X_status( i ) < 0 ) THEN
             IF ( DZ( i ) /= zero ) THEN
               mu_break = mu - Z( i ) / DZ( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2010 ) prefix, i
                 WRITE( out, 2000 ) prefix, i, 'L', X_l( i ),                  &
                    X( i ) - mu * DX( i ), STRING_sign( DX( i ), .TRUE. ),     &
                    ABS( DX( i ) ), X_u( i ), mu * ( Z( i ) - mu * DZ( i ) ),  &
                    STRING_sign( DZ( i ), .TRUE. ), mu * ABS( DZ( i ) )
               END IF
             END IF
           ELSE IF ( X_status( i ) > 0 ) THEN
             IF ( DZ( i ) /= zero ) THEN
               mu_break = mu - Z( i ) / DZ( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2010 ) prefix, i
                 WRITE( out, 2000 ) prefix, i, 'U', X_l( i ),                  &
                    X( i ) - mu * DX( i ), STRING_sign( DX( i ), .TRUE. ),     &
                    ABS( DX( i ) ), X_u( i ), mu * ( Z( i ) - mu * DZ( i ) ),  &
                    STRING_sign( DZ( i ), .TRUE. ), mu * ABS( DZ( i ) )
               END IF
             END IF
           ELSE
             IF ( DX( i ) > zero ) THEN
               mu_break = mu - ( X_u( i ) - X( i ) ) / DX( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2010 ) prefix, i
                 WRITE( out, 2000 ) prefix, i, ' ', X_l( i ),                  &
                    X( i ) - mu * DX( i ), STRING_sign( DX( i ), .TRUE. ),     &
                    ABS( DX( i ) ), X_u( i ), mu * ( Z( i ) - mu * DZ( i ) ),  &
                    STRING_sign( DZ( i ), .TRUE. ), mu * ABS( DZ( i ) )
               END IF
             ELSE IF ( DX( i ) < zero ) THEN
               mu_break = mu - ( X_l( i ) - X( i ) ) /DX( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2010 ) prefix, i
                 WRITE( out, 2000 ) prefix, i, ' ', X_l( i ),                  &
                    X( i ) - mu * DX( i ), STRING_sign( DX( i ), .TRUE. ),     &
                    ABS( DX( i ) ), X_u( i ), mu * ( Z( i ) - mu * DZ( i ) ),  &
                    STRING_sign( DZ( i ), .TRUE. ), mu * ABS( DZ( i ) )
               END IF
             END IF
           END IF
         END DO
         DO i = 1, m
           IF ( C_l( i ) == C_u( i ) ) THEN
           ELSE IF ( C_status( i ) < 0 ) THEN
             IF ( DY( i ) /= zero ) THEN
               mu_break = mu - Y( i ) / DY( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2020 ) prefix, i
                 WRITE( out, 2000 )  prefix, i, 'L', C_l( i ),                 &
                   C( i ) - mu * DC( i ), STRING_sign( DC( i ), .TRUE. ),      &
                   ABS( DC( i ) ), C_u( i ), mu * ( Y( i ) - mu * DY( i ) ),   &
                   STRING_sign( DY( i ), .TRUE. ), mu * ABS( DY( i ) )
               END IF
             END IF
           ELSE IF ( C_status( i ) > 0 ) THEN
             IF ( DY( i ) /= zero ) THEN
               mu_break = mu - Y( i ) / DY( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2020 ) prefix, i
                 WRITE( out, 2000 )  prefix, i, 'U', C_l( i ),                 &
                   C( i ) - mu * DC( i ), STRING_sign( DC( i ), .TRUE. ),      &
                   ABS( DC( i ) ), C_u( i ), mu * ( Y( i ) - mu * DY( i ) ),   &
                   STRING_sign( DY( i ), .TRUE. ), mu * ABS( DY( i ) )
               END IF
             END IF
           ELSE
             IF ( DC( i ) > zero ) THEN
               mu_break = mu - ( C_u( i ) - C( i ) ) / DC( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2020 ) prefix, i
                 WRITE( out, 2000 )  prefix, i, ' ', C_l( i ),                 &
                   C( i ) - mu * DC( i ), STRING_sign( DC( i ), .TRUE. ),      &
                   ABS( DC( i ) ), C_u( i ), mu * ( Y( i ) - mu * DY( i ) ),   &
                   STRING_sign( DY( i ), .TRUE. ), mu * ABS( DY( i ) )
               END IF
             ELSE IF ( DC( i ) < zero ) THEN
               mu_break = mu - ( C_l( i ) - C( i ) ) / DC( i )
               IF ( mu_break == FASTR_mu_new ) THEN
                 WRITE( out, 2020 ) prefix, i
                 WRITE( out, 2000 )  prefix, i, ' ', C_l( i ),                 &
                   C( i ) - mu * DC( i ), STRING_sign( DC( i ), .TRUE. ),      &
                   ABS( DC( i ) ), C_u( i ), mu * ( Y( i ) - mu * DY( i ) ),   &
                   STRING_sign( DY( i ), .TRUE. ), mu * ABS( DY( i ) )
               END IF
             END IF
           END IF
         END DO

!  .. and about the value of mu for which there would be an active set change ..

         DO i = 1, n 
           IF ( X_l( i ) == X_u( i ) ) THEN
             WRITE( out, "( A, '  variable ', I0, ' is free' )" ) prefix, i
           ELSE IF ( X_status( i ) < 0 .AND. DZ( i ) /= zero ) THEN
             WRITE( out, 2030 ) prefix, 'z', i, mu - Z( i ) / DZ( i ) 
           ELSE IF ( X_status( i ) > 0 .AND. DZ( i ) /= zero ) THEN
             WRITE( out, 2030 ) prefix, 'z', i, mu - Z( i ) / DZ( i )
           ELSE
             IF ( DX( i ) > zero ) THEN
               WRITE( out, 2030 ) prefix, 'x', i,                              &
                  mu - ( X_u( i ) - X( i ) ) / DX( i )
             ELSE IF ( DX( i ) < zero ) THEN
               WRITE( out, 2030 ) prefix, 'x', i,                              &
                  mu - ( X_l( i ) - X( i ) ) / DX( i )
             END IF
           END IF
         END DO
         DO i = 1, m
           IF ( C_l( i ) == C_u( i ) ) THEN
             WRITE( out, "( A, '   constraint ', I0, ' is an equality' )" )    &
               prefix, i
           ELSE IF ( C_status( i ) < 0 .AND. DY( i ) /= zero ) THEN
             WRITE( out, 2030 ) prefix, 'y', i, mu - Y( i ) / DY( i )
           ELSE IF ( C_status( i ) > 0 .AND. DY( i ) /= zero ) THEN
             WRITE( out, 2030 ) prefix, 'y', i, mu - Y( i ) / DY( i )
           ELSE
             IF ( DC( i ) > zero ) THEN
               WRITE( out, 2030 ) prefix, 'c', i,                              &
                  mu - ( C_u( i ) - C( i ) ) / DC( i )
             ELSE IF ( DC( i ) < zero ) THEN
               WRITE( out, 2030 ) prefix, 'c', i,                              &
                  mu - ( C_l( i ) - C( i ) ) / DC( i )
             END IF
           END IF
         END DO

!  .. and the parametric solution

         WRITE( out, "( /, A, ' variable', A, ':', /,A,'     i  <- x_l -> <-', &
        &  '         x           -> <- x_u -> <-         z           ->' )" )  &
           prefix, TRIM( STRING_pleural( n ) ), prefix
         DO i = 1, n 
           IF ( X_l( i ) == X_u( i ) ) THEN 
             st = 'F'
           ELSE IF ( X_status( i ) < 0 ) THEN 
             st = 'L'
           ELSE IF ( X_status( i ) > 0 ) THEN
              st = 'U'
           ELSE ; 
              st = ' '
           END IF
           WRITE( out, 2000 ) prefix, i, st, X_l( i ),                         &
                    X( i ) - mu * DX( i ), STRING_sign( DX( i ), .TRUE. ),     &
                    ABS( DX( i ) ), X_u( i ), mu * ( Z( i ) - mu * DZ( i ) ),  &
                    STRING_sign( DZ( i ), .TRUE. ), mu * ABS( DZ( i ) )
         END DO
         IF ( m > 0 ) WRITE( out, "( A, ' constraint', A, ': ', /, A,          &
        &  '     i  <- c_l -> <-         c           -> <- c_u ->',            &
        &  ' <-         y           ->' )" )                                   &
           prefix, TRIM( STRING_pleural( m ) ), prefix
         DO i = 1, m
           IF ( C_l( i ) == C_u( i ) ) THEN 
             st = 'E'
           ELSE IF ( C_status( i ) < 0 ) THEN
             st = 'L'
           ELSE IF ( C_status( i ) > 0 ) THEN
             st = 'U'
           ELSE ; st = ' ' ; END IF
           WRITE( out, 2000 )  prefix, i, st, C_l( i ),                        &
                   C( i ) - mu * DC( i ), STRING_sign( DC( i ), .TRUE. ),      &
                   ABS( DC( i ) ), C_u( i ), mu * ( Y( i ) - mu * DY( i ) ),   &
                   STRING_sign( DY( i ), .TRUE. ), mu * ABS( DY( i ) )
         END DO
       END IF
     END IF
     RETURN

!  non-executable statements
     
2000 FORMAT( A, I6, A1, 2ES10.2, ' ', A, ' mu *', ES9.2, 2ES10.2, ' ', A,      &
             ' mu *', ES9.2 )
2010 FORMAT( A, ' variable ', I0, ' will change status' )
2020 FORMAT( A, ' constraint ', I0, ' will change status' )
2030 FORMAT( A, '   ', A, '(', I0, ') would change status when mu = ', ES12.4 )

!  End of function FASTR_mu_new

     END FUNCTION FASTR_mu_new

!-*-  G A L A H A D -  F A S T R _ r e s t o r a t i o n  S U B R O U T I N E -*

     SUBROUTINE FASTR_restoration( nlp, stop_p, control, inform, data,         &
                                   userdata, eval_FC, eval_GJ, eval_HL,        &
                                   eval_HLPROD )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  FASTR_restoration, find a local minimizer of the infeasibility for a 
!  set of general constraints and simple bounds:
!
!    min 1/2||s||^2 such that c^l <= c(x) - c^s s <= c^u and x^l <= x <= x^u
!
!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  See the preamble of FASTR_solve for details. But note 
!
!  stop_p is a scalar real variable that specifies the value below which 
!   the primal infeasibility ||s|| is required to be
!
!  inform is a scalar variable of type TRU_inform_type. On initial entry, 
!   inform%status should be set to 1. On exit, the following components will
!   have been set:
!
!   status is a scalar variable of type default integer, that gives
!    the exit status from the package. Additional possible values are:
!
!    12. The user should compute the constraint function values c(x) at the 
!        point x indicated in nlp%X and then re-enter the subroutine. The 
!        required values should be set in nlp%C, and data%eval_status should be
!        set to 0. If the user is unable to evaluate c(x)  - for instance, if 
!        any of the functions is undefined at x - the user need not set
!        nlp%C, but should then set data%eval_status to a non-zero value.
!    13. The user should compute the Jacobian of the constraints nabla_x c(x) 
!        at the point x indicated in nlp%X  and then re-enter the subroutine. 
!        The nonzeros of the Jacobian should be set in nlp%J%val in the same 
!        order as in the storage scheme already established in nlp%J,, and 
!        data%eval_status should be set to 0. If the user is unable to evaluate
!        any of the components of the Jacobian - for instance if a component of
!        the Jacobian is undefined at x - the user need not set nlp%J%val, but 
!        should then set data%eval_status to a non-zero value.
!    14. The user should compute the Hessian of the objective-free Lagrangian 
!        function - sum_i=1^m y_i c_i(x) at the point x indicated in nlp%X
!        and y in nlp%Y and then re-enter the subroutine. The nonzeros of the
!        Hessian should be set in nlp%H%val in the same order as in the storage
!        scheme already established in nlp%H, and data%eval_status should be 
!        set to 0. If the user is unable to evaluate a component of the Hessian
!        - for instance, if a component of the Hessian is undefined at x - the 
!        user need not set nlp%H%val, but should then set data%eval_status to 
!        a non-zero value.
!    15. The user should compute both the Jacobian and Hessian as described
!        in 13 and 14 above, and then re-enter the subroutine with
!        data%eval_status set to 0. If the user is unable to evaluate 
!        any of this data nlp%J%val and nlp%H%val need not be set but 
!        then data%eval_status should be set to a non-zero value.
!    16. The user should compute the product 
!        ( - sum_i=1^m y_i c_i(x) ) v of the Hessian of the objective-free
!        Lagrangian function - sum_i=1^m y_i c_i(x) at the point x indicated in 
!        nlp%X with the vector v, and add the result to the vector u and then 
!        re-enter the subroutine. The vectors u and v are given in data%U and 
!        data%V respectively, the resulting vector u + nabla_xx f(x)v should be 
!        set in data%U and  data%eval_status should be set to 0. If the user is
!        unable to evaluate the product - for instance, if a component of the 
!        Hessian is undefined at x - the user need not alter data%U, but
!        should then set data%eval_status to a non-zero value.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = wp ), INTENT( IN ) :: stop_p
     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( FASTR_control_type ), INTENT( IN ) :: control
     TYPE ( FASTR_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( FASTR_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_GJ, eval_HL, eval_HLPROD

!----------------------------------
!   I n t e r f a c e   B l o c k s 
!----------------------------------

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

       SUBROUTINE eval_GJ( status, X, userdata, G, J_val )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
         REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
         REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ

       SUBROUTINE eval_HL( status, X, Y, userdata, H_val, no_f ) 
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) ::H_val
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
         LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL

       SUBROUTINE eval_HLPROD( status, X, Y, userdata, U, V, no_f, got_h )
         USE GALAHAD_NLPT_double
         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
         INTEGER, INTENT( OUT ) :: status
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
         REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
         TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
         LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
         LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HLPROD
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ii, ir, ic, j, l, n_mult_wrong_sign_ls
!    INTEGER :: m_working, n_working
     REAL ( KIND = wp ) :: alpha, max_y, dthd, delta_f, delta_m, dx
     REAL ( KIND = wp ) :: obj_trial, y_t, complementary_slackness_ls, o, v
     REAL ( KIND = wp ) :: primal_infeasibility_trial, dual_infeasibility_ls
!    REAL ( KIND = wp ) :: delta_eqp, ratio_norm_dlp
     LOGICAL :: acceptable
     CHARACTER ( LEN = 8 ) :: tr_active
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     SELECT CASE ( data%branch_restoration )
     CASE ( 1 )  ! initialization
       GO TO 10
!    CASE ( 2 )  ! initial constraint evaluation
!      GO TO 20
     CASE ( 3 )  ! Jacobian evaluation
       GO TO 130
     CASE ( 4 )  ! Hessian evaluation
       GO TO 140
     CASE ( 5 )  ! Hessian-vector product
       GO TO 150
     CASE ( 6 )  ! constraint evaluation
       GO TO 260
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE
!    CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
     data%control = control

!    inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
!    inform%iter = 0 ; inform%factorizations = 0 ; inform%modifications = 0 
!    inform%f_eval = 0 ; inform%g_eval = 0

     inform%obj_restoration = HUGE( one ) 
     inform%primal_infeasibility_rest = HUGE( one )
     inform%dual_infeasibility_rest = HUGE( one )

     data%restoration_restoration = .FALSE.
     data%new_point = .TRUE.
     data%it_type = ' '
     data%bdry = ' '
     data%d_type = ' '
     data%n_pr_max = 0
     data%rest = 'R'
     data%print_iteration_header = .TRUE.

     data%step = zero
     data%mu_new = one
     data%norm_dlp_restoration = SQRT( HUGE( one ) )
     data%n_mult_wrong_sign = - 1
     data%successful = .FALSE.
!    data%new_gradient = .FALSE.

!  allocate additional space to hold the slack variables s and their duals w

     array_name = 'fastr: data%S'
     CALL SPACE_resize_array( nlp%m, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%S_p'
     CALL SPACE_resize_array( nlp%m, data%S_p, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%DS'
     CALL SPACE_resize_array( nlp%m, data%DS, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%S_trial'
     CALL SPACE_resize_array( nlp%m, data%S_trial, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%DS_trial_rlp'
     CALL SPACE_resize_array( nlp%m, data%DS_trial_rlp, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'fastr: data%GL'
     CALL SPACE_resize_array( data%n_restoration, data%GL, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set up space for the filter


     CALL FILTER_initialize_filter( data%FILTER_restoration_data,              &
                                    data%control%FILTER_control,               &
                                    inform%FILTER_restoration_inform )     
     IF ( inform%FILTER_restoration_inform%status /= 0 ) THEN
       inform%status = inform%FILTER_restoration_inform%status
       inform%alloc_status = inform%FILTER_restoration_inform%alloc_status
       GO TO 910
     END IF

!    WRITE( data%out, "( /, A, ' Solver:         FASTr_restoration',           &
!   &               /, A, ' Problem: ', 7X, A10 )" ) prefix, prefix, nlp%pname

!  evaluate the general constraint function values
   
!    IF ( data%reverse_fc ) THEN
!      data%branch_restoration = 2 ; inform%status = 12 ; RETURN
!    ELSE  
!      CALL eval_FC( data%eval_status, nlp%X, userdata, C = nlp%C )
!    END IF

!  return from reverse communication to obtain the constraint values

! 20 CONTINUE
!    inform%f_eval = inform%f_eval + 1

!  initialize the slack variables

     DO i = 1, nlp%m
       IF ( nlp%C( i ) < nlp%C_l( i ) ) THEN
         data%S( i ) = ( nlp%C( i ) - nlp%C_l( i ) ) / nlp%C_scale( i )
       ELSE IF ( nlp%C( i ) > nlp%C_u( i ) ) THEN
         data%S( i ) = ( nlp%C( i ) - nlp%C_u( i ) ) / nlp%C_scale( i )
       ELSE
         data%S( i ) = zero
       END IF
     END DO

!  record the infeasibility

     inform%primal_infeasibility                                               &
       = INFINITY_norm( nlp%C_scale( : nlp%m ) * data%S( : nlp%m ) )
     inform%obj_restoration                                                    &
       = half * DOT_PRODUCT( data%S( : nlp%m ), data%S( : nlp%m ) )

     inform%primal_infeasibility_rest = zero

!  record the primal infeasiblity stopping tolerance

     data%stop_p_rest = MAX( control%stop_abs_p, control%stop_rel_p *          &
                             inform%primal_infeasibility )

!    IF ( nlp%m > 0 ) THEN
!    inform%primal_infeasibility_rest =                                        &
!        MAXVAL( MAX( nlp%C_l( : nlp%m ) - nlp%C( : nlp%m ),                   &
!                     nlp%C( : nlp%m ) - nlp%C_u( : nlp%m ), zero ) )
!    ELSE
!      inform%primal_infeasibility_rest = zero
!    END IF
!    data%pr_max = MAX( data%control%max_absolute_infeasibility,               &
!                       data%control%max_relative_infeasibility                &
!                         * inform%primal_infeasibility )

!  initialize the regularization and trust-region radii

     data%mu = data%control%initial_radius_rlp
     data%radius_eqp = data%control%initial_radius_eqp

     IF ( data%printd ) THEN
       WRITE( data%out, 2020 ) prefix
       WRITE( data%out, "( A, I6, 3ES12.4, '      -     ', ES12.4 )" )         &
         ( prefix, i, nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%Z( i ),      &
           i = 1, nlp%n )
       WRITE( data%out, "( / )" )
     END IF

!    IF ( data%printi ) WRITE( data%out, "( /, A, ' s=successful, v=v.succ,',  &
!   &  ' u=unsucc, i=infeas, r=reject filter, R=restoration' )" ) prefix

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      M A I N     I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 100 CONTINUE

!  set print agenda for the current iteration

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%CQP_control%print_level = data%print_level_cqp
         data%control%CQP_control%SBLS_control%print_level                     &
           = data%print_level_sbls
         data%control%EQP_control%print_level = data%print_level_eqp
         data%control%EQP_control%SBLS_control%print_level                     &
           = data%print_level_sbls
         data%control%EQP_control%GLTR_control%print_level                     &
           = data%print_level_gltr
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%CQP_control%print_level = 0
         data%control%CQP_control%SBLS_control%print_level = 0
         data%control%EQP_control%print_level = 0
         data%control%EQP_control%SBLS_control%print_level = 0
         data%control%EQP_control%GLTR_control%print_level = 0
       END IF
       IF ( data%print_level > 1 .OR.                                          &
           data%control%EQP_control%print_level > 0 .OR.                       &
           data%control%CQP_control%print_level > 0 )                          &
         data%print_iteration_header = .TRUE.

!  ----------------------------------------------------------------------------
!                       COMPUTE DERIVATIVE VALUES
!  ----------------------------------------------------------------------------

!  evaluate the Jacobian of the general constraint functions, stored in 
!  "co-ordinate" format

       IF ( data%new_gradient ) THEN
         IF ( data%reverse_gj ) THEN
           data%branch_restoration = 3 ; inform%status = 13 ; RETURN
         ELSE  
           CALL eval_GJ( data%eval_status, nlp%X, userdata, J_val = nlp%J%val )
         END IF
       END IF

!  return from reverse communication to obtain the Jacobian

  130  CONTINUE
       IF ( data%new_gradient ) THEN
         data%CQP_prob%G( : nlp%n ) = nlp%G( : nlp%n )
         inform%g_eval = inform%g_eval + 1
       END IF

!  compute the gradient of the Lagrangian  ( - A^T y - z )
!                                          (   s + c^s y )

       data%GL( : nlp%n ) = - nlp%Z( : nlp%n )
       data%GL( nlp%n + 1 : data%n_restoration )                               &
         = data%S( : nlp%m ) + nlp%C_scale( : nlp%m ) * nlp%Y( : nlp%m )
       DO l = 1, data%J_ne
         i = data%CQP_prob%A%col( l )
         data%GL( i ) = data%GL( i ) -                                         &
           nlp%J%val( l ) * nlp%Y( data%CQP_prob%A%row( l ) )
       END DO

!      WRITE(6,*) ' gl, y ', maxval( data%GL ), maxval( nlp%Y )
!      WRITE( data%out, "(A, /, ( 4ES20.12 ) )" ) ' gl_after ',  data%GL

!  compute norms of the primal and dual feasibility and the complemntary
!  slackness

       inform%dual_infeasibility_rest =                                        &
         OPT_dual_infeasibility( data%n_restoration,                           &
                                 data%GL( : data%n_restoration ) )
       inform%complementary_slackness_rest =                                   &
         OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                 &
            nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),          &
            nlp%m, nlp%C( : nlp%m ) - data%S( : nlp%m ),                       &
            nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) )

!  compute the stopping tolerances

       IF ( inform%iter == data%iter_restoration ) THEN
         data%stop_d_rest = MAX( control%stop_abs_d, control%stop_rel_d *      &
                                 inform%dual_infeasibility_rest )
         data%stop_c_rest = MAX( control%stop_abs_c, control%stop_rel_c *      &
                                 inform%complementary_slackness_rest )
         IF ( data%printt ) WRITE( data%out,                                   &
             "(  /, A, '  Primal    convergence tolerance =', ES11.4,          &
            &    /, A, '  Dual      convergence tolerance =', ES11.4,          &
            &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )      &
                 prefix, data%stop_p_rest, prefix, data%stop_d_rest, prefix,   &
                 data%stop_c_rest
       END IF 

!  -------------------------------------
!  Print a summary of the last iteration
!  -------------------------------------

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out,                          &
!          "( /, A, ' iter      infeas   pr_feas du_feas cmp_slk actve ',      &
           "( /, A, ' iter     obj rest  pr_feas du_feas cmp_slk actve ',      &
        &        '  step   #fil      mu     CPU')" ) prefix
         data%print_iteration_header = .FALSE.
         data%print_1st_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out,                                                    &
             "( A, I5, 2A1, ES12.4, 3ES8.1, A6, ES8.1, 2A1, I4, ES8.1, F8.1 )")&
             prefix, inform%iter, data%rest, data%it_type,                     &
!            inform%primal_infeasibility, inform%primal_infeasibility_rest,    &
!            inform%obj_restoration, inform%primal_infeasibility_rest,         &
             inform%obj_restoration, inform%primal_infeasibility,              &
             inform%dual_infeasibility_rest,                                   &
             inform%complementary_slackness_rest,                              &
             STRING_integer_6( data%EQP_prob%A%m ), data%step, data%d_type,    &
             data%bdry, data%FILTER_restoration_data%n_filter, data%mu,        &
             data%clock_now
         ELSE
           WRITE( data%out, "( A, I5, 'R ', ES12.4, 3ES8.1, '     -     -    ',&
          &   I4, ES8.1, F8.1 )" ) prefix, inform%iter,                        &
!            inform%primal_infeasibility, inform%primal_infeasibility_rest,    &
!            inform%obj_restoration, inform%primal_infeasibility_rest,         &
             inform%obj_restoration, inform%primal_infeasibility,              &
             inform%primal_infeasibility, inform%primal_infeasibility_rest,    &
             inform%dual_infeasibility_rest,                                   &
             inform%complementary_slackness_rest,                              &
             data%FILTER_restoration_data%n_filter, data%mu, data%clock_now
         END IF
       END IF

!  ---------------------
!  Check for termination
!  ---------------------

!      WRITE(6,*) inform%primal_infeasibility_rest <= data%stop_p_rest,        &
!           data%n_mult_wrong_sign == 0,                                       &
!           inform%dual_infeasibility_rest <= data%stop_d_rest,                &
!           inform%complementary_slackness_rest <= data%stop_c_rest

       IF ( inform%primal_infeasibility_rest <= data%stop_p_rest .AND.         &
            data%n_mult_wrong_sign == 0 .AND.                                  &
            inform%dual_infeasibility_rest <= data%stop_d_rest .AND.           &
            inform%complementary_slackness_rest <= data%stop_c_rest ) THEN
         IF ( inform%primal_infeasibility <= data%stop_p_rest ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
              "( /, A, ' Termination criteria satisfied ' )" ) prefix
           inform%status = GALAHAD_ok ; GO TO 900
         ELSE
           IF ( data%printt ) WRITE( data%out, "( /, A, ' Termination',        &
          &  ' criteria satisfied: problem locally infeasible ' )" ) prefix
           inform%status = GALAHAD_error_primal_infeasible ; GO TO 905
         END IF
       END IF

       IF ( inform%primal_infeasibility <= stop_p ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
            "( /, A, ' Required infeasibility decrease achieved ' )" ) prefix
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  ------------------------
!  Start the next iteration
!  ------------------------

       inform%iter = inform%iter + 1

       IF ( inform%iter > data%control%maxit ) THEN
         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Iteration limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_max_iterations ; GO TO 905
       END IF

!  ----------------------------------------------------------------------------
!                     CHOOSE THE CURRENT ACTIVE SET
!  ----------------------------------------------------------------------------

       IF ( inform%primal_infeasibility_rest == zero .AND.                     &
            data%mu_new == zero .AND. .NOT. data%successful ) THEN
         data%DX_trial_rlp( : nlp%n ) = tenth * data%DX_trial_rlp( : nlp%n )
         data%DS_trial_rlp( : nlp%m ) = tenth * data%DS_trial_rlp( : nlp%m )
       ELSE

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Compute the solution to the regularized quadratic program (RQP)

!      min  mu ( 0  s )^T ( dx )  + 1/2 dx^T dx + 1/2 ds^T ds
!                         ( ds )
!
!      s.t. c_l - c + c^s s <= A dx - c^s ds <= c_u - c + c^s s 
!      and  x_l - x <= dx <= x_u - x 

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set up the vector problem data

         data%CQP_prob%new_problem_structure = .TRUE.     
         data%CQP_prob%n = data%n_restoration
         data%CQP_prob%m = nlp%m
         data%CQP_prob%A%n = data%n_restoration
         data%CQP_prob%A%m = nlp%m
         data%CQP_prob%H%n = data%n_restoration
         data%CQP_prob%A%ne = data%J_ne_restoration
         data%CQP_prob%f = zero
         data%CQP_prob%X_l( : nlp%n ) = nlp%X_l( : nlp%n ) - nlp%X( : nlp%n )
         data%CQP_prob%X_l( nlp%n + 1 : data%n_restoration )                   &
           = - ten * data%control%CQP_control%infinity
         data%CQP_prob%X_u( : nlp%n ) = nlp%X_u( : nlp%n ) - nlp%X( : nlp%n )
         data%CQP_prob%X_u( nlp%n + 1 : data%n_restoration )                   &
           = ten * data%control%CQP_control%infinity
         IF ( nlp%m > 0 ) THEN
           data%CQP_prob%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) - nlp%C( : nlp%m )&
             + nlp%C_scale( : nlp%m ) * data%S( : nlp%m )
           data%CQP_prob%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) - nlp%C( : nlp%m )&
             + nlp%C_scale( : nlp%m ) * data%S( : nlp%m )
         END IF
         data%CQP_prob%X( : nlp%n ) = zero
         data%CQP_prob%X( nlp%n + 1 : data%n_restoration ) = zero
         data%CQP_prob%Z( : nlp%n ) = nlp%Z( : nlp%n )
         data%CQP_prob%Z( nlp%n + 1 : data%n_restoration ) = zero
         data%CQP_prob%Y( : nlp%m ) = zero
         data%CQP_prob%A%val( : data%J_ne ) = nlp%J%val( : data%J_ne )
         data%CQP_prob%A%val( data%J_ne + 1 : data%J_ne_restoration )          &
           = - nlp%C_scale( : nlp%m )
         IF ( data%control%primal_qp ) THEN
           data%CQP_prob%G( : nlp%n ) = zero
           data%CQP_prob%G( nlp%n + 1 : data%n_restoration ) =                 &
             data%mu * data%S( : nlp%m )
           data%CQP_prob%H%val( : data%n_restoration ) = one
         ELSE
         END IF 

!  if required, print a description of the problem

         IF ( data%printt ) WRITE( data%out, "( /, A, ' * Find the working',   &
        &    ' set - entering CQP: n = ', I0, ', m = ', I0, ', mu = ',         &
        &    ES8.2 )" ) prefix, nlp%n, data%CQP_prob%m, data%mu

         IF ( data%out > 0 .AND. data%control%print_level >= 20 ) THEN
           WRITE( data%out, "( A, ' n, m = ', I0, 1X, I0 )" )                  &
             prefix, data%CQP_prob%n, data%CQP_prob%m
           WRITE( data%out, "( A, ' f = ', ES12.4 )" )                         &
             prefix, data%CQP_prob%f
           WRITE( data%out, "( A, ' G = ', /, ( 5ES12.4 ) )" ) prefix,         &
             data%CQP_prob%G( : data%CQP_prob%n )
           IF ( SMT_get( data%CQP_prob%H%type ) == 'DIAGONAL' ) THEN
             WRITE( data%out, "( A, ' H (diagonal) = ', /, ( 5ES12.4 ) )" )    &
               prefix, data%CQP_prob%H%val( : data%CQP_prob%n )
           ELSE IF ( SMT_get( data%CQP_prob%H%type ) == 'DENSE' ) THEN
             WRITE( data%out, "( A, ' H (dense) = ', /, ( 5ES12.4 ) )" )       &
              prefix, data%CQP_prob%H%val( : data%CQP_prob%n *                 &
                ( data%CQP_prob%n + 1 ) / 2 )
           ELSE IF ( SMT_get( data%CQP_prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
             WRITE( data%out, "( A, ' H (row-wise) = ' )" ) prefix
             DO i = 1, data%CQP_prob%m
               WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
                 ( i, data%CQP_prob%H%col( j ), data%CQP_prob%H%val( j ),      &
                   j = data%CQP_prob%H%ptr( i ),                               &
                       data%CQP_prob%H%ptr( i + 1 ) - 1 )
             END DO
           ELSE
             WRITE( data%out, "( A, ' H (co-ordinate) = ' )" ) prefix
             WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
             ( data%CQP_prob%H%row( i ), data%CQP_prob%H%col( i ),             &
               data%CQP_prob%H%val( i ), i = 1, data%CQP_prob%H%ne )
           END IF
           WRITE( data%out, "( A, ' X_l = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%X_l( : data%CQP_prob%n )
           WRITE( data%out, "( A, ' X_u = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%X_u( : data%CQP_prob%n )
           IF ( SMT_get( data%CQP_prob%A%type ) == 'DENSE' ) THEN
             WRITE( data%out, "( A, ' A (dense) = ', /, ( 5ES12.4 ) )" )       &
               prefix,                                                         &
               data%CQP_prob%A%val( : data%CQP_prob%n * data%CQP_prob%m )
           ELSE IF ( SMT_get( data%CQP_prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
             WRITE( data%out, "( A, ' A (row-wise) = ' )" ) prefix
             DO i = 1, data%CQP_prob%m
               WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
                 ( i, data%CQP_prob%A%col( j ), data%CQP_prob%A%val( j ),      &
                   j = data%CQP_prob%A%ptr( i ),                               &
                       data%CQP_prob%A%ptr( i + 1 ) - 1 )
             END DO
           ELSE
             WRITE( data%out, "( A, ' A (co-ordinate) = ' )" ) prefix
             WRITE( data%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
             ( data%CQP_prob%A%row( i ), data%CQP_prob%A%col( i ),             &
               data%CQP_prob%A%val( i ), i = 1, data%CQP_prob%A%ne )
           END IF
           WRITE( data%out, "( A, ' C_l = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%C_l( : data%CQP_prob%m )
           WRITE( data%out, "( A, ' C_u = ', /, ( 5ES12.4 ) )" ) prefix,       &
             data%CQP_prob%C_u( : data%CQP_prob%m )
         END IF

!  set control parameters

!        data%control%CQP_control%dufeas = data%mu

!  solve the RQP

        CALL CQP_solve( data%CQP_prob, data%CQP_data,                          &
                        data%control%CQP_control, inform%CQP_inform,           &
                        C_stat = nlp%C_status, B_stat = nlp%X_status )

!        write(6,"( ' x ', / ( 5ES14.6) )" )  data%CQP_prob%X(:data%CQP_prob%n )
!        write(6,"( ' y ', / ( 5ES14.6) )" )  data%CQP_prob%Y(:data%CQP_prob%m )
!        write(6,"( ' z ', / ( 5ES14.6) )" )  data%CQP_prob%Z(:data%CQP_prob%n )
!        write(6,"( ' c ', / ( 5ES14.6) )" )  data%CQP_prob%C(:data%CQP_prob%m )

!       write(6,*) nlp%C_status
!       write(6,*) nlp%CNAMES

!       write(6,*) ' c_status ', nlp%C_status
!       write(6,*) ' x_status ', nlp%X_status

!  check to see if feasibility restoration is required ...

         IF ( inform%CQP_inform%status == GALAHAD_error_primal_infeasible .OR. &
              inform%CQP_inform%status == GALAHAD_error_dual_infeasible .OR.   &
              inform%CQP_inform%status == GALAHAD_error_tiny_step ) THEN
           SELECT CASE ( inform%CQP_inform%status ) 
           CASE ( GALAHAD_error_primal_infeasible )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - primal infeasible' )" ) prefix
           CASE ( GALAHAD_error_dual_infeasible )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - dual infeasible' )" ) prefix
           CASE ( GALAHAD_error_tiny_step )
             IF ( data%printi ) WRITE( data%out, "( A, '    * Exit from CQP',  &
            & ' - no further progress possible' )" ) prefix
           END SELECT
           data%restoration_restoration = .TRUE. ; GO TO 300
         ELSE IF ( inform%CQP_inform%status == GALAHAD_error_unbounded ) THEN
           IF ( data%printt ) WRITE( data%out, "( A, '    * Exit from CQP',    &
          & ' unbounded from below' )" ) prefix

!  ... or if an error occured 

         ELSE IF ( inform%CQP_inform%status /= GALAHAD_ok ) THEN
           IF ( data%printi ) WRITE( data%out, "( A, ' ** CQP error exit,',    &
          & '  status = ', I0 )" ) prefix, inform%CQP_inform%status
           inform%status = GALAHAD_error_qp_solve ; GO TO 990
         END IF

         data%old_norm_dlp_restoration = data%norm_dlp_restoration
         data%norm_dlp_restoration = MAXVAL( ABS( data%CQP_prob%X ) )
! write(6,*) ' *** old, new norm dlp ', data%old_norm_dlp_restoration,         &
!  data%norm_dlp_restoration

!  if required, summarize the RLP iteration

         IF ( data%printt ) THEN
           IF ( data%control%CQP_control%out > 0 .AND.                         &
                data%control%CQP_control%print_level > 0 )                     &
             WRITE( data%out, "( '' )" )
           WRITE( data%out, "( A, '  - on exit from CQP: status = ', I0,       &
         &   ', time = ', F0.2, ', iterations = ', I0 )" ) prefix,             &
            inform%CQP_inform%status, inform%CQP_inform%time%total,            &
            inform%CQP_inform%iter
           WRITE( data%out, "( A, '   - RLP objective decrease =', ES12.4,     &
         &  ', mu = ', ES10.4,                                                 &
         &   /, A, '   - || d_rlp || = ', ES10.4, ', || y_1 || = ', ES10.4,    &
         &   /, A, '   - # active counstraints: ', I0, ' (', I0,               &
         &  ' independent)', ' from ', I0, /, A, '   - # active bounds: ', I0, &
         &  ' (', I0, ' independent)', ' from ', I0 )" )                       &
              prefix, - inform%CQP_inform%obj, data%mu,                        &
              prefix, data%norm_dlp_restoration,                               &
              MAX( MAXVAL( ABS( data%CQP_prob%Y( : data%CQP_prob%m ) ) ),      &
                   MAXVAL( ABS( data%CQP_prob%Z( : data%CQP_prob%n ) ) ) ),    &
              prefix, COUNT( nlp%C_status( : data%CQP_prob%m ) /= 0 ),         &
              COUNT( ABS( nlp%C_status( : data%CQP_prob%m ) ) == 1 ),          &
              data%CQP_prob%m,                                                 &
              prefix, COUNT( nlp%X_status( : data%CQP_prob%n ) /= 0 ),         &
              COUNT( ABS( nlp%X_status( : data%CQP_prob%n ) ) == 1 ),          &
              data%CQP_prob%n
         END IF

!  if required, print the subproblem and its solution

         IF ( data%printd ) THEN
           WRITE( data%out, "( /, A, ' RLP subproblem ' )" ) prefix
           WRITE( data%out, "( '      i  stat     X_l           X      ',      &
          &                    '   X_u          G' )" )
           WRITE( data%out, "( ( 1X, 2I6, 4ES12.4 ) )" )                       &
           ( i, nlp%X_status( i ),                                             &
             data%CQP_prob%X_l( i ), data%CQP_prob%X( i ),                     &
             data%CQP_prob%X_u( i ), data%CQP_prob%G( i ),                     &
               i = 1, data%CQP_prob%n )
           IF ( nlp%m > 0 ) THEN
             WRITE( data%out, "( '      i  stat     C_l           C      ',    &
            &                    '   C_u' )" )
             WRITE( data%out, "( ( 1X, 2I6, 3ES12.4 ) )" )                     &
               ( i, nlp%C_status( i ), data%CQP_prob%C_l( i ),                 &
                 data%CQP_prob%C( i ), data%CQP_prob%C_u( i ),                 &
                   i = 1, data%CQP_prob%m )
             WRITE( data%out, "( '  A row   col      val      row   col     ', &
            & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" ) &
               ( data%CQP_prob%A%row( i ), data%CQP_prob%A%col( i ),           &
                 data%CQP_prob%A%val( i), i = 1, data%CQP_prob%A%ne )
           END IF
         END IF

!  --- for debugging --- write the working sets ---

         IF ( wsout > 0 ) THEN
           WRITE( wsout, "( /, ' FASTr iteration ', I7, //,                    &
          &  '      i status           c_l              c',                    &
          &  '              c_u         y' )" ) inform%iter
           WRITE( wsout, "( ( 2I7, 4ES16.8 ) )" ) ( i, nlp%C_status( i ),      &
             data%CQP_prob%C_l( i ), data%CQP_prob%C( i ),                     &
             data%CQP_prob%C_u( i ), data%CQP_prob%Y( i ),                     &
               i = 1, data%CQP_prob%m )
           WRITE( wsout, "( /,                                                 &
          &  '      i status           x_l              x',                    &
          &  '             x_u          z' )" )
           WRITE( wsout, "( ( 2I7, 4ES16.8 ) )" ) ( i, nlp%X_status( i ),      &
             data%CQP_prob%X_l( i ), data%CQP_prob%X( i ),                     &
             data%CQP_prob%X_u( i ), data%CQP_prob%Z( i ),                     &
               i = 1, data%CQP_prob%n )
         END IF

!  store the trial point d_RLP generated by the RLP

         data%DX_trial_rlp( : nlp%n ) = data%CQP_prob%X( : nlp%n )
         data%DS_trial_rlp( : nlp%m ) =                                        &
           data%CQP_prob%X( nlp%n + 1 : data%n_restoration )
       END IF

!      write(6,*) ' x ', nlp%X
!      write(6,*) ' d_rlp', data%DX_trial_rlp

       data%norm_deqp = zero

!  ----------------------------------------------------------------------------
!                 FIRST-ORDER LAGRANGE MULTIPLIER UPDATES
!  ----------------------------------------------------------------------------

       IF ( data%control%multipliers == 0 .OR.                                 &
            data%control%multipliers == 1  ) THEN
!      IF ( data%control%multipliers == 1 ) THEN

!  compute the changes in multipliers and dual variables

         data%DY( : nlp%m ) = data%CQP_prob%Y( : nlp%m ) - nlp%Y( : nlp%m )
         data%DZ( : nlp%n ) = data%CQP_prob%Z( : nlp%n ) - nlp%Z( : nlp%n )
 
!  compute ( A^T dy + dz )
!          ( - c^s dy    )

         data%ATDY( : nlp%n ) = data%DZ( : nlp%n )
         data%ATDY( nlp%n + 1 : data%n_restoration )                           &
           = - nlp%C_scale( : nlp%m ) * data%DY( : nlp%m )
         DO l = 1, data%J_ne
           i = data%CQP_prob%A%col( l )
           data%ATDY( i ) = data%ATDY( i )                                     &
             + data%CQP_prob%A%val( l ) * data%DY( data%CQP_prob%A%row( l ) )
         END DO

!  find the minimizer alpha of || g_l - alpha A^T dy ||_2^2

         alpha = DOT_PRODUCT( data%ATDY( : data%n_restoration ),               &
                              data%ATDY( : data%n_restoration ) )
         IF ( alpha /= zero ) THEN
           alpha = DOT_PRODUCT( data%GL( : data%n_restoration ),               &
                                data%ATDY( : data%n_restoration ) ) / alpha

!  update the multipliers and the gradient of the Lagrangian

           IF ( alpha /= zero ) THEN
             nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) + alpha * data%DY( : nlp%m )
             nlp%Z( : nlp%n ) = nlp%Z( : nlp%n ) + alpha * data%DZ( : nlp%n )
             data%GL( : data%n_restoration )                                   &
               = data%GL( : data%n_restoration )                               &
                 - alpha * data%ATDY( : data%n_restoration )

!  recompute the dual infeasibility and complementarity 

             inform%dual_infeasibility_rest =                                  &
               OPT_dual_infeasibility( data%n_restoration,                     &
                                       data%gL( : data%n_restoration ) )
             inform%complementary_slackness_rest =                             &
               OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),           &
                  nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),    &
                  nlp%m, nlp%C( : nlp%m ) - data%S( : nlp%m ),                 &
                  nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) )

             IF ( data%printm ) WRITE( data%out, "( A, ' * Dual residual ',    &
            &  'with first-order multiplier estimate = ', ES10.4, /, A,        &
            &  '   and complementary slackness = ', ES10.4  )" )               &
               prefix, inform%dual_infeasibility_rest,                         &
               prefix, inform%complementary_slackness_rest

!  check for termination with the updated multipliers

             IF ( inform%primal_infeasibility_rest <= data%stop_p_rest         &
                  .AND. data%n_mult_wrong_sign == 0 .AND.                      &
                  inform%dual_infeasibility_rest <= data%stop_d_rest .AND.     &
                  inform%complementary_slackness_rest <= data%stop_c_rest )    &
                 THEN
               IF ( inform%primal_infeasibility <= data%stop_p_rest ) THEN
                 IF ( data%printt ) WRITE( data%out,                           &
                    "( /, A, ' Termination criteria satisfied ' )" ) prefix
                 inform%status = GALAHAD_ok ; GO TO 900
               ELSE
                 IF ( data%printt ) WRITE( data%out, "( /, A, ' Termination ', &
                &  'criteria satisfied: problem locally infeasible ' )" ) prefix
                 inform%status = GALAHAD_error_primal_infeasible ; GO TO 905
               END IF
             END IF
           END IF
         END IF
       END IF

!  ----------------------------------------------------------------------------
!                     COMPUTE LEAST_SQUARES MULTIPLIER ESTIMATES
!  ----------------------------------------------------------------------------

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  find the least-squares multipliers by solving the equality-constrained 
!  quadratic program

!      min  ( 0  s )^T ( dx ) + 1/2 ( dx ds )^T ( I 0 ) ( dx )
!                      ( ds )                   ( 0 I ) ( ds )
!      s.t. A_A dx - c^s ds = 0 and dx_A = 0

!  where _A denotes the active constraints

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  compute how many active constraints there are

!      m_working = COUNT( ABS( nlp%C_status( : data%CQP_prob%m ) ) == 1 )
!      n_working = COUNT( ABS( nlp%X_status( : data%CQP_prob%n ) ) == 1 )

!  set the objective details

       CALL SMT_put( data%EQP_prob%H%type, 'DIAGONAL', inform%alloc_status )
       data%EQP_prob%n = data%n_restoration
       data%EQP_prob%H%n = data%EQP_prob%n
       data%EQP_prob%H%ne = data%EQP_prob%H%n
       data%EQP_prob%f = zero
       data%EQP_prob%G( : nlp%n ) = zero
       data%EQP_prob%G( nlp%n + 1 : data%n_restoration ) = data%S( : nlp%m )
       data%EQP_prob%H%val( : data%n_restoration ) = one
       data%EQP_prob%X( : data%n_restoration ) = zero
!      array_name = 'fastr: data%EQP_prob%H%type'
!      CALL SPACE_dealloc_array( data%EQP_prob%H%type,                         &
!       inform%status, inform%alloc_status,                                    &
!       array_name = array_name, data%out = control%error )

!write(6,*) ' --- g ', data%EQP_prob%g
!write(6,*) ' --- f ', data%EQP_prob%f
!write(6,*) ' --- H ', data%EQP_prob%H%val

!     IF ( data%printi ) WRITE( data%out, "(                                   &
!     &   /, A, ' - # active counstraints: ', I0, ' from ', I0, ', bounds: ',  &
!     &  I0, ' from ', I0 )" ) prefix, m_working, nlp%m, n_working, nlp%n

!  set the right-hand sides for the active constraints

       data%EQP_prob%m = 0
       DO i = 1, nlp%m 
         IF ( nlp%C_status( i ) == - 1 ) THEN
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = zero
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
           nlp%C_status( i ) = - data%EQP_prob%m
         ELSE IF ( nlp%C_status( i ) == 1 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = zero
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
           nlp%C_status( i ) = data%EQP_prob%m
         ELSE
           nlp%C_status( i ) = 0
         END IF
       END DO
       DO i = 1, nlp%n
         IF ( nlp%X_status( i ) == - 1 ) THEN
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = zero
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
           nlp%X_status( i ) = - data%EQP_prob%m
         ELSE IF ( nlp%X_status( i ) == 1 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = zero
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
           nlp%X_status( i ) = data%EQP_prob%m
         ELSE
           nlp%X_status( i ) = 0
         END IF
       END DO
       data%EQP_prob%A%n = data%EQP_prob%n
       data%EQP_prob%A%m = data%EQP_prob%m

!       WRITE( 6, "( ' nlp%X_status ', /, ( 10I7 ) )" ) nlp%X_status
!       WRITE( 6, "( ' nlp%C_status ', /, ( 10I7 ) )" ) nlp%C_status

!  place the entries in the Jacobian of working constraints

       data%EQP_prob%A%ne = 0
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, data%J_ne
           i = nlp%J%row( l )
           ii = ABS( nlp%C_status( i ) )
           IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                           &
                  control%jacobian_zero_tolerance ) THEN
             data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
             data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
             data%EQP_prob%A%col( data%EQP_prob%A%ne ) = nlp%J%col( l )
             data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         data%EQP_prob%A%ne = 0
         DO i = 1, nlp%m
           ii = ABS( nlp%C_status( i ) )
           IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                           &
                  control%jacobian_zero_tolerance ) THEN
             DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
               data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
               data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
               data%EQP_prob%A%col( data%EQP_prob%A%ne ) = nlp%J%col( l )
               data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
             END DO
           END IF
         END DO
       CASE ( 'DENSE' ) 
         data%EQP_prob%A%ne = 0 ; l = 0
         DO i = 1, nlp%m
           ii = ABS( nlp%C_status( i ) )
           IF ( ii > 0 .AND. ABS( nlp%J%val( l ) ) >                           &
                  control%jacobian_zero_tolerance ) THEN
             DO j = 1, nlp%n
               data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1 ; l = l + 1
               data%EQP_prob%A%row( data%EQP_prob%A%ne ) = ii
               data%EQP_prob%A%col( data%EQP_prob%A%ne ) = j
               data%EQP_prob%A%val( data%EQP_prob%A%ne ) = nlp%J%val( l )
             END DO
           ELSE
             l = l + nlp%n
           END IF
         END DO
       END SELECT
       DO j = 1, nlp%m
         i = ABS( nlp%C_status( j ) )
         IF ( i > 0 ) THEN
           data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
           data%EQP_prob%A%row( data%EQP_prob%A%ne ) = i
           data%EQP_prob%A%col( data%EQP_prob%A%ne ) = nlp%n + j
           data%EQP_prob%A%val( data%EQP_prob%A%ne ) = - nlp%C_scale( j )
         END IF
       END DO
       DO j = 1, nlp%n
         i = ABS( nlp%X_status( j ) )
         IF ( i > 0 ) THEN
           data%EQP_prob%A%ne = data%EQP_prob%A%ne + 1
           data%EQP_prob%A%row( data%EQP_prob%A%ne ) = i
           data%EQP_prob%A%col( data%EQP_prob%A%ne ) = j
           data%EQP_prob%A%val( data%EQP_prob%A%ne ) = one
         END IF
       END DO

! write(6,"(' nlp%n, nlp%m ', I0, 1X, I0 )" ) data%EQP_prob%n, data%EQP_prob%m
! write(44,"( ' n, nnz ', 2I8 )" )  data%EQP_prob%n, data%EQP_prob%A%ne
! write(44,"( ' A_row ', / ( 5I6) )" )                                         &
!   data%EQP_prob%A%row( : data%EQP_prob%A%ne )
! write(44,"( ' A_col ', / ( 5I6) )" )                                         &
!   data%EQP_prob%A%col( : data%EQP_prob%A%ne )
! write(44,"( ' A_val ', / ( 5ES14.6))" )                                      &
!    data%EQP_prob%A%val(:data%EQP_prob%A%ne )
! write(44,"( ' c ', / ( 5ES14.6) )" )  data%EQP_prob%C(:data%EQP_prob%m )

!  find the Lagrange multiplier estimates y (stored in data%EQP_prob%Y)

       IF ( data%printt ) WRITE( data%out, "( /, A, ' * Compute least-squares',&
      &   ' Lagrange multiplier estimates', /, A, '   - entering EQP: n = ',   &
      &   I0, ', m_working = ', I0 )" ) prefix, prefix, nlp%n, data%EQP_prob%m

       data%control%EQP_control%radius = - one
       CALL EQP_solve( data%EQP_prob, data%EQP_data, data%control%EQP_control, &
                       inform%EQP_inform )

!  check to see if an error occured

       IF ( inform%EQP_inform%status == GALAHAD_error_primal_infeasible ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
           "( /, A, ' EQP infeasible ... skipping' )" ) prefix
         GO TO 140
       ELSE IF ( inform%EQP_inform%status /= GALAHAD_ok ) THEN
         IF ( data%printi ) WRITE( data%out, 2030 )                            &
           prefix, inform%EQP_inform%status
         inform%status = GALAHAD_error_qp_solve ; GO TO 990
       END IF

       IF ( data%EQP_prob%m > 0 ) THEN
         y_t = MAXVAL( ABS( data%EQP_prob%Y( : data%EQP_prob%m ) ) )
       ELSE
         y_t = zero
       END IF

       dual_infeasibility_ls =                                                 &
         MAXVAL( ABS( data%EQP_prob%X( : data%EQP_prob%n ) ) )

! write(6,*) ' d_eqp', data%EQP_prob%X
! write(6,*) ' dx dy ', MAXVAL( ABS( data%EQP_prob%X(:data%EQP_prob%n ) ) ),   &
!                       MAXVAL( ABS( data%EQP_prob%Y(:data%EQP_prob%m ) ) )

!  if required, summarize the EQP iteration

       IF ( data%printt ) THEN
         IF ( data%control%EQP_control%out > 0 .AND.                           &
              data%control%EQP_control%print_level > 0 )                       &
           WRITE( data%out, "( '' )" )
         WRITE( data%out, "( A, '   - on exit from EQP: status = ', I0,        &
        & ', cg iter = ', I0, ', time = ', F0.2, /, A, '   - || g - A^T',      &
        & ' y_ls || = ', ES10.4, ', || y_ls || = ', ES10.4 )" )                &
              prefix, inform%EQP_inform%status, inform%EQP_inform%cg_iter,     &
              inform%EQP_inform%time%total, prefix,                            &
              dual_infeasibility_ls, y_t
       END IF

       IF ( data%printd ) THEN
         WRITE( data%out, "( /, A, ' Least-squares EQP subproblem' )" ) prefix
         WRITE( data%out, "( 2 ( '      i       X           G    ' ) )" )
         WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                    &
           ( i, data%EQP_prob%X( i ), data%EQP_prob%G( i ),                    &
               i = 1, data%EQP_prob%n )
         IF ( data%EQP_prob%m > 0 ) THEN
           WRITE( data%out, "( 2 ( '      i       C           Y   ' ) )" )
           WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                  &
             ( i, data%EQP_prob%C( i ), data%EQP_prob%Y( i ),                  &
                 i = 1, data%EQP_prob%m )
           WRITE( data%out, "( '  A row   col      val      row   col     ',   &
          & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )   &
             ( data%EQP_prob%A%row( i ), data%EQP_prob%A%col( i ),             &
               data%EQP_prob%A%val( i ), i = 1, data%EQP_prob%A%ne )
         END IF
         WRITE( data%out, "( '  H row   col      val      row   col     ',     &
        & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )     &
           ( data%EQP_prob%H%row( i ), data%EQP_prob%H%col( i ),               &
             data%EQP_prob%H%val( i ), i = 1, data%EQP_prob%H%ne )
       END IF

!  spread the solution into the vectors (x_p,y_p,z_p,s_p,c_p)

       data%X_p( : nlp%n ) = data%EQP_prob%X( : nlp%n )
       data%S_p( : nlp%m ) = data%EQP_prob%X( nlp%n + 1 : data%n_restoration )
       DO i = 1, nlp%m 
         IF ( nlp%C_status( i ) == 0 ) THEN
           data%Y_p( i ) = zero
         ELSE
           data%Y_p( i ) = data%EQP_prob%Y( ABS( nlp%C_status( i ) ) )
         END IF
       END DO
       DO i = 1, nlp%n
         IF ( nlp%X_status( i ) == 0 ) THEN
           data%Z_p( i ) = zero
         ELSE
           data%Z_p( i ) = data%EQP_prob%Y( ABS( nlp%X_status( i ) ) )
         END IF
       END DO
       data%C_p( : nlp%m ) = - nlp%C_scale( : nlp%m ) * data%S_p( : nlp%m )
       DO l = 1, data%J_ne
         i = data%CQP_prob%A%row( l )
         data%C_p( i ) = data%C_p( i )                                         &
           + data%CQP_prob%A%val( l ) * data%X_p( data%CQP_prob%A%col( l ) )
       END DO

!  recompute the complementarity and, if necessary, the number of out-of-kilter 
!  multipliers

       IF ( inform%primal_infeasibility_rest <= data%stop_p_rest .AND.         &
!           dual_infeasibility_ls <= data%stop_d_rest ) THEN
            dual_infeasibility_ls <= data%stop_d_rest * ten ** ( - 3 ) ) THEN
         complementary_slackness_ls =                                          &
           OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),               &
              nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), data%Z_p( : nlp%n ),     &
              nlp%m, nlp%C( : nlp%m ) - data%S( : nlp%m ),                     &
              nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ),  data%Y_p( : nlp%m ) )

         IF ( data%printm ) WRITE( data%out, "( A, ' * Dual residual ',        &
        &  'with least-squares multiplier estimate = ', ES10.4, /, A,          &
        &  '   and complementary slackness = ', ES10.4  )" )                   &
           prefix, dual_infeasibility_ls, prefix, complementary_slackness_ls

         IF ( complementary_slackness_ls <= data%stop_c_rest ) THEN
           n_mult_wrong_sign_ls = 0
           DO i = 1, nlp%m 
             IF ( nlp%C_l( i ) /= nlp%C_u( i ) .AND.                           &
              ( ( nlp%C_status( i ) < 0 .AND. data%Y_p( i ) < - y_tiny ) .OR.  &
                ( nlp%C_status( i ) > 0 .AND. data%Y_p( i ) > y_tiny ) ) )     &
               n_mult_wrong_sign_ls = n_mult_wrong_sign_ls + 1
           END DO
           DO i = 1, nlp%n
             IF ( nlp%X_l( i ) /= nlp%X_u( i ) .AND.                           &
              ( ( nlp%X_status( i ) < 0 .AND. data%Z_p( i ) < - z_tiny ) .OR.  &
                ( nlp%X_status( i ) > 0 .AND. data%Z_p( i ) > z_tiny ) ) )     &
               n_mult_wrong_sign_ls = n_mult_wrong_sign_ls + 1
           END DO

!  check for termination with the least-squares multipliers

           IF ( n_mult_wrong_sign_ls == 0 ) THEN
             inform%dual_infeasibility_rest = dual_infeasibility_ls
             inform%complementary_slackness_rest = complementary_slackness_ls
             data%n_mult_wrong_sign = n_mult_wrong_sign_ls
             nlp%Y( : nlp%m ) = data%Y_p( : nlp%m )
             nlp%Z( : nlp%n ) = data%Z_p( : nlp%n )
             IF ( inform%primal_infeasibility <= data%stop_p_rest ) THEN
               IF ( data%printt ) WRITE( data%out,                             &
                  "( /, A, ' Termination criteria satisfied ' )" ) prefix
               inform%status = GALAHAD_ok ; GO TO 900
             ELSE
               IF ( data%printt ) WRITE( data%out, "( /, A, ' Termination ',   &
              &  'criteria satisfied: problem locally infeasible ' )" ) prefix
               inform%status = GALAHAD_error_primal_infeasible ; GO TO 905
             END IF
           END IF
         END IF
       END IF

!write(6,*) 'y_ls', data%Y_p

!  use the least-squares multiplier estimates if desired

!  if required or desirable, use the least-squares multiplier estimates

!write(6,*) dual_infeasibility_ls, inform%dual_infeasibility
       IF ( data%control%multipliers == 3 .OR.                                 &
            ( data%control%multipliers == 0 .AND.                              &
              dual_infeasibility_ls < inform%dual_infeasibility_rest ) ) THEN
!      IF ( data%control%multipliers == 2 ) THEN
         nlp%Y( : nlp%m ) = data%Y_p( : nlp%m )
         nlp%Z( : nlp%n ) = data%Z_p( : nlp%n )
       END IF

!  recompute the Hessian

       IF ( data%reverse_hl ) THEN
          data%branch_restoration = 4 ; inform%status = 4 ; RETURN
       ELSE  
          CALL eval_HL( data%eval_status, nlp%X, nlp%Y, userdata, nlp%H%val,   &
                        no_f = .TRUE. ) 
       END IF

!  return from reverse communication to obtain the Hessian of the Lagrangian

 140   CONTINUE
!write(6,*) ' H ', nlp%H%val( : data%H_ne )

!  ----------------------------------------------------------------------------
!                     COMPUTE THE SEARCH DIRECTION
!  ----------------------------------------------------------------------------

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  solve the equality-constrained quadratic program

!      min  ( 0  s )^T ( dx ) + 1/2 ( dx ds )^T ( H 0 ) ( dx )
!                      ( ds )                   ( 0 I ) ( ds )
!      s.t. A_A dx - c^s_A ds = c_b_A - c_A + c^s s and dx = x_b_A - x_A

!  where _A denotes the active constraints & bounds and c_b and x_b are
!  the relevant lower or upper bounds

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set the objective details

       CALL SMT_put( data%EQP_prob%H%type, 'COORDINATE', inform%alloc_status )
       CALL SMT_put( data%EQP_prob%H%type, 'COORDINATE', inform%alloc_status )
       data%EQP_prob%n = data%n_restoration
       data%EQP_prob%H%n = data%EQP_prob%n
       data%EQP_prob%H%ne = data%H_ne_restoration
       data%EQP_prob%f = zero
       data%EQP_prob%G( : nlp%n ) = zero
       data%EQP_prob%G( nlp%n + 1 : data%n_restoration ) = data%S( : nlp%m )
       data%EQP_prob%H%val( : data%H_ne ) = nlp%H%val( : data%H_ne )
       data%EQP_prob%H%val( data%H_ne + 1 : data%H_ne_restoration ) = one
       data%EQP_prob%X( : data%n_restoration ) =                               &
         data%CQP_prob%X( : data%n_restoration )

!write(6,*) ' --- g ', data%EQP_prob%g
!write(6,*) ' --- f ', data%EQP_prob%f
!write(6,*) ' --- H ', data%EQP_prob%H%val

!     IF ( data%printi ) WRITE( data%out, "(                                   &
!     &   /, A, ' - # active counstraints: ', I0, ' from ', I0, ', bounds: ',  &
!     &  I0, ' from ', I0 )" ) prefix, m_working, nlp%m, n_working, nlp%n

!  set the right-hand sides for the active constraints

       data%EQP_prob%m = 0
!write(6,*) ' c,s 1', &
! nlp%C_l(1), nlp%C(1), nlp%C_scale(1) * data%S(1), nlp%C_u(1)
       DO i = 1, nlp%m 
         IF ( nlp%C_status( i ) < 0 ) THEN
!write(6,*) ' c-s l', i, &
! nlp%C_l(i), nlp%C(i) - nlp%C_scale(i) * data%S(i), nlp%C_u(i)
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m )                                  &
             = nlp%C( i ) - nlp%C_scale( i ) * data%S( i ) - nlp%C_l( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
         ELSE IF ( nlp%C_status( i ) > 0 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
!write(6,*) ' c-s u', i, &
! nlp%C_l(i), nlp%C(i) - nlp%C_scale(i) * data%S(i), nlp%C_u(i)
           data%EQP_prob%C( data%EQP_prob%m )                                  &
             = nlp%C( i ) - nlp%C_scale( i ) * data%S( i ) - nlp%C_u( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Y( i )
         END IF
       END DO
       DO i = 1, nlp%n
         IF ( nlp%X_status( i ) < 0 ) THEN
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_l( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
         ELSE IF ( nlp%X_status( i ) > 0 ) THEN 
           data%EQP_prob%m = data%EQP_prob%m + 1
           data%EQP_prob%C( data%EQP_prob%m ) = nlp%X( i ) - nlp%X_u( i )
           data%EQP_prob%Y( data%EQP_prob%m ) = data%CQP_prob%Z( i )
         END IF
       END DO
!write(6,*) ' infeas ', inform%primal_infeasibility_rest
!write(6,*) ' C ', data%EQP_prob%C( : data%EQP_prob%m )

!  restrict the EQP step to be 100 times that of the RLP

       data%radius_eqp = hundred * data%norm_dlp_restoration

!  find the step s (stored in data%EQP_prob%X) and Lagrange multiplier 
!  y (stored in data%EQP_prob%Y)

       IF ( data%printt ) WRITE( data%out, "( /, A, ' * Find the step -',      &
      &  ' entering EQP: n = ', I0, ', m_working = ', I0 )" )                  &
           prefix, nlp%n, data%EQP_prob%m

       data%control%EQP_control%radius = data%radius_eqp
       CALL EQP_solve( data%EQP_prob, data%EQP_data, data%control%EQP_control, &
                       inform%EQP_inform )

!  check to see if an error occured

       IF ( inform%EQP_inform%status == GALAHAD_error_primal_infeasible ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
           "( /, A, ' EQP infeasible ... skipping' )" ) prefix
         data%take_eqp_step = .FALSE.
         data%step_rlp = one
         GO TO 200
       ELSE IF ( inform%EQP_inform%status /= GALAHAD_ok ) THEN
         IF ( data%printi ) WRITE( data%out, 2030 )                            &
           prefix, inform%EQP_inform%status
         inform%status = GALAHAD_error_qp_solve ; GO TO 990
       END IF

!  compute the norm of the step and the slope for the objective and violated 
!  constraints

       data%norm_deqp = MAXVAL( ABS( data%EQP_prob%X( : data%EQP_prob%n ) ) )
       data%gtd_eqp = DOT_PRODUCT( data%S( : nlp%m ),                          &
                        data%EQP_prob%X( nlp%n + 1 : data%n_restoration ) )
       data%dtjc_eqp = zero
       DO l = 1, data%CQP_prob%A%ne
         i = data%CQP_prob%A%row( l )
         IF ( nlp%C( i ) <= nlp%C_l( i ) ) THEN
           data%dtjc_eqp = data%dtjc_eqp +                                     &
             data%EQP_prob%X( data%CQP_prob%A%col( l ) ) *                     &
             ( nlp%C( i ) - nlp%C_l( i ) ) * data%CQP_prob%A%val( l ) 
         ELSE IF ( nlp%C( i ) >= nlp%C_u( i ) ) THEN
           data%dtjc_eqp = data%dtjc_eqp +                                     &
             data%EQP_prob%X( data%CQP_prob%A%col( l ) ) *                     &
             ( nlp%C( i ) - nlp%C_u( i ) ) * data%CQP_prob%A%val( l ) 
         END IF
       END DO
       IF ( data%printm ) WRITE( data%out, "( A, ' d^Tg, d^TJc = ', 2ES12.4 )")&
         prefix, data%gtd_eqp, data%dtjc_eqp

! write(6,*) ' d_eqp', data%EQP_prob%X
! write(6,*) ' dx dy ', MAXVAL( ABS( data%EQP_prob%X(:data%EQP_prob%n ) ) ),   &
!                       MAXVAL( ABS( data%EQP_prob%Y(:data%EQP_prob%m ) ) )

!  if required, summarize the EQP iteration

       IF ( data%printt ) THEN
         IF ( data%control%EQP_control%out > 0 .AND.                           &
              data%control%EQP_control%print_level > 0 )                       &
           WRITE( data%out, "( '' )" )
         IF ( ABS( inform%EQP_inform%GLTR_inform%mnormx - data%radius_eqp )    &
                <= teneps ) THEN
           tr_active = 'active  '
         ELSE
           tr_active = 'inactive'
         END IF

         IF ( data%EQP_prob%m > 0 ) THEN
           y_t = MAXVAL( ABS( data%EQP_prob%Y( : data%EQP_prob%m ) ) )
         ELSE
           y_t = zero
         END IF
         WRITE( data%out, "( A, '   - on exit from EQP: status = ', I0,        &
        & ', cg iter = ', I0, ', time = ', F0.2, /, A, '   - EQP objective',   &
        & ' decrease =', ES12.4, ', radius = ', ES10.4, ' ', A,                &
        & /, A, '   - || d_eqp || = ', ES10.4, ', || y_2 || = ', ES10.4 )" )   &
              prefix, inform%EQP_inform%status, inform%EQP_inform%cg_iter,     &
              inform%EQP_inform%time%total, prefix, - inform%EQP_inform%obj,   &
              data%radius_eqp, TRIM( tr_active ), prefix, data%norm_deqp, y_t
       END IF

       IF ( data%printd ) THEN
         WRITE( data%out, "( /, A, ' EQP subproblem ' )" ) prefix
         WRITE( data%out, "( 2 ( '      i       X           G    ' ) )" )
         WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                    &
           ( i, data%EQP_prob%X( i ), data%EQP_prob%G( i ),                    &
               i = 1, data%EQP_prob%n )
         IF ( data%EQP_prob%m > 0 ) THEN
           WRITE( data%out, "( 2 ( '      i       C           Y    ' ) )" )
           WRITE( data%out, "( ( 2 ( 1X, I6, 2ES12.4 ) ) )" )                  &
             ( i, data%EQP_prob%C( i ), data%EQP_prob%Y( i ),                  &
                 i = 1, data%EQP_prob%m )
           WRITE( data%out, "( '  A row   col      val      row   col     ',   &
          & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )   &
             ( data%EQP_prob%A%row( i ), data%EQP_prob%A%col( i ),             &
               data%EQP_prob%A%val( i ), i = 1, data%EQP_prob%A%ne )
         END IF
         WRITE( data%out, "( '  H row   col      val      row   col     ',     &
        & ' val      row   col      val', /, ( 1X, 3( 2I6, ES12.4 ) ) )" )     &
           ( data%EQP_prob%H%row( i ), data%EQP_prob%H%col( i ),               &
             data%EQP_prob%H%val( i ), i = 1, data%EQP_prob%H%ne )
       END IF
!write(6,*) 'y_2', data%EQP_prob%Y(:data%EQP_prob%m)

       data%bdry = ' '
       IF ( ABS( inform%EQP_inform%GLTR_inform%mnormx - data%radius_eqp )      &
              < teneps) data%bdry = 'b'

!  ----------------------------------------------------------------------------
!                       STEP ACCEPTANCE OR REJECTION
!  ----------------------------------------------------------------------------

!  check to find the range of values [mu_new,mu] for which the active set for 
!  the current RLP stays active

       data%mu_new = FASTR_mu_new( nlp%m, nlp%n, data%mu,                      &
                              data%CQP_prob%X_l, data%CQP_prob%X_u,            &
                              data%CQP_prob%X, data%X_p,                       &
                              data%CQP_prob%C_l, data%CQP_prob%C_u,            &
                              data%CQP_prob%C, data%C_p,                       &
                              data%CQP_prob%Y, data%Y_p,                       &
                              data%CQP_prob%Z, data%Z_p,                       &
                              nlp%X_status, nlp%C_status,                      &
                              data%out, data%printt, data%print_level > 10,    &
                              prefix )

!  record the "decrease" delta_l in the linear model

!      data%gtd                                                                &
!        = DOT_PRODUCT( data%S( : nlp%m ), data%DS_trial_rlp( : nlp%m ) ) )
       data%gtd                                                                &
         = DOT_PRODUCT( data%S( : nlp%m ), data%DS_trial_rlp( : nlp%m ) +      &
                       ( data%mu_new - data%mu ) * data%S_p( : nlp%m ) )
       data%delta_l = - data%gtd

!write(6,*) ' deqp, gtdeqp ', data%norm_deqp, data%gtd_eqp

!  record the required "decrease" delta_q in the "quadratic" model, that is
!  to say at the Cauchy point, the smallest value of the quadratic model on 
!  the line joining x to x + d_RLP. Only consider the case where there is 
!  decrease in the linear model (i,e g^T d_RLP < 0)

!      WRITE(6,*) ' gtd ', data%gtd
       IF ( data%gtd < zero ) THEN

!  compute the product d^T H(x,y) d, temporarily storing H(x,y) d in Hd

         data%U( : nlp%n ) = zero
         IF ( got_hessian ) THEN
           DO l = 1, data%H_ne
             i = data%EQP_prob%H%row( l ) ; j = data%EQP_prob%H%col( l )
             data%U( i ) = data%U( i ) +                                       &
               data%EQP_prob%H%val( l ) * data%DX_trial_rlp( j ) 
             IF ( i /= j ) data%U( j ) = data%U( j ) +                         &
               data%EQP_prob%H%val( l ) * data%DX_trial_rlp( i ) 
           END DO
         ELSE
           IF ( data%reverse_hlprod ) THEN
             data%V( : nlp%n ) = data%DX_trial_rlp( : nlp%n )
             data%branch_restoration = 5 ; inform%status = 16 ; RETURN
           ELSE  
             CALL eval_HLPROD( data%eval_status, nlp%X, nlp%Y, userdata,       &
                               data%U, data%DX_trial_rlp, no_f = .TRUE. ) 

           END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian-vector product

  150  CONTINUE
       IF ( data%gtd < zero ) THEN
         dthd = DOT_PRODUCT( data%DX_trial_rlp( : nlp%n ),                     &
                             data%U( : nlp%n ) )                               &
                + DOT_PRODUCT( data%DS_trial_rlp( : nlp%m ),                   &
                               data%DS_trial_rlp( : nlp%m ) )

!        WRITE(6,*) ' dthd ', dthd

!  record delta_q at a full Cauchy step ...

         IF ( dthd <= zero ) THEN
           data%delta_q = - data%gtd - dthd
         ELSE
!          WRITE(6,*) ' Cauchy step ', - half * data%gtd / dthd
           IF ( - data%gtd >= dthd ) THEN
             data%delta_q = - data%gtd - dthd

!  ... or at a restricted Cauchy step 

           ELSE
             data%delta_q = half * data%gtd * ( data%gtd / dthd )
           END IF
         END IF
       ELSE
         data%delta_q = zero
       END IF

       data%successful = .FALSE.

!  compute the maximum step for the EQP step to stay feasible

       IF ( data%norm_deqp > zero ) THEN
         data%step_eqp = one
         DO i = 1, nlp%n
           dx = data%EQP_prob%X( i ) 
! write(6,"( I7, 4ES12.4, 1X, I0 )")                                           &
!  i, nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), dx, nlp%X_status( i )
           IF ( dx > epsmch ) THEN
             data%step_eqp =                                                   &
               MIN( data%step_eqp, ( nlp%X_u( i ) - nlp%X( i ) ) / dx )
           ELSE IF ( dx < - epsmch ) THEN
             data%step_eqp =                                                   &
               MIN( data%step_eqp, ( nlp%X_l( i ) - nlp%X( i ) ) / dx )
           END IF
         END DO
       ELSE
         data%step_eqp = zero
       END IF

!  decide whether to avoid the EQP search direction

       IF ( data%step_eqp <= zero ) THEN
         IF ( data%printt )                                                    &
           WRITE( data%out, "(  A, '   - EQP stepsize zero' )" ) prefix
         data%take_eqp_step = .FALSE.
       ELSE
         data%take_eqp_step = .TRUE.
       END IF

!  -----------------------------------------
!  loop over the potential search directions
!  -----------------------------------------

 200   CONTINUE

!  try the EQP step

         IF ( data%take_eqp_step ) THEN
           data%d_name = 'EQP' ; data%d_type = 'E'
           IF ( .NOT. data%control%try_rlp_step .AND.                          &
                data%step_eqp > ten ** ( - 10 ) .AND.                          &
                inform%primal_infeasibility_rest <= epsmch ** 0.75 ) GO TO 290
           IF ( data%step_eqp /= one ) data%bdry = 't'

!  update the solution

           data%X_trial( : nlp%n ) =                                           &
             nlp%X( : nlp%n ) + data%step_eqp * data%EQP_prob%X( : nlp%n )
           data%step = data%norm_deqp * data%step_eqp
        
!  try the RLP step

         ELSE
           data%d_name = 'RLP' ; data%d_type = 'L'
           data%step = data%norm_dlp_restoration
           data%X_trial( : nlp%n ) =                                           &
             nlp%X( : nlp%n ) + data%DX_trial_rlp( : nlp%n )
         END IF

!  evaluate the general constraint function values
     
         IF ( data%reverse_fc ) THEN
           data%EQP_prob%X( : nlp%n ) = nlp%X( : nlp%n )     ! temporary copy
           data%EQP_prob%C( : nlp%m ) = nlp%C( : nlp%m )     ! temporary copy
           nlp%X = data%X_trial
           data%branch_restoration = 6 ; inform%status = 12 ; RETURN
         ELSE  
           CALL eval_FC( data%eval_status, data%X_trial, userdata,             &
                         C = data%C_trial )
         END IF

!  return from reverse communication to obtain the constraint values

 260     CONTINUE
         IF ( data%reverse_fc ) THEN
           data%C_trial = nlp%C
           nlp%X( : nlp%n ) = data%EQP_prob%X( : nlp%n )      ! restore copy
           nlp%C( : nlp%m ) = data%EQP_prob%C( : nlp%m )      ! restore copy
         END IF
         inform%f_eval = inform%f_eval + 1

!  update the slack variables

         DO i = 1, nlp%m
           IF ( data%C_trial( i ) < nlp%C_l( i ) ) THEN
             data%S_trial( i )                                                 &
               = ( data%C_trial( i ) - nlp%C_l( i ) )  / nlp%C_scale( i )
           ELSE IF ( data%C_trial( i ) > nlp%C_u( i ) ) THEN
             data%S_trial( i )                                                 &
               = ( data%C_trial( i ) - nlp%C_u( i ) )  / nlp%C_scale( i )
           ELSE
             data%S_trial( i ) = zero
           END IF
         END DO 

!  record the infeasibility

         obj_trial = half * DOT_PRODUCT( data%S_trial( : nlp%m ),              &
                                         data%S_trial( : nlp%m ) )

!  compute the norm of the violation

         primal_infeasibility_trial = zero
         o = inform%obj_restoration                                            &
               - data%control%gamma_filter * inform%primal_infeasibility_rest
         v = data%control%beta_filter * inform%primal_infeasibility_rest

!  is the new point is too infeasible ?

         IF ( primal_infeasibility_trial > data%pr_max ) THEN
           IF ( data%take_eqp_step ) data%n_pr_max = data%n_pr_max + 1
           data%it_type = 'i'
           IF ( data%printt ) WRITE( data%out, "( /, A, ' * ', A3, ' step',    &
          &  ' infeasibility = ', ES10.4, ' too large ' )" ) prefix,           &
               data%d_name, primal_infeasibility_trial

!  is the new point is acceptable to the filter and (f,c) ?

         ELSE 
           CALL FILTER_acceptable( obj_trial, primal_infeasibility_trial,      &
                                      data%FILTER_restoration_data,            &
                                      data%control%FILTER_control,             &
                                      acceptable, o = o, v = v )
           IF ( acceptable ) THEN
             IF ( data%take_eqp_step ) data%n_pr_max = 0
             IF ( data%printt ) WRITE( data%out,                               &
               "(  /, A, ' * ', A3, ' point acceptable to filter (v,o) = (',   &
              &    ES10.4, ',', ES11.4, ')', /, A,                             &
              &    '                       vs current (v,o) = (',              &
              &    ES10.4, ',', ES11.4, ')' )" )                               &
                 prefix, data%d_name, primal_infeasibility_trial, obj_trial,   &
                 prefix, inform%primal_infeasibility_rest,                     &
                 inform%obj_restoration
             delta_f = inform%obj_restoration - obj_trial
             delta_m = MIN( data%delta_q, data%delta_l )
             IF ( data%printt ) THEN
               WRITE( data%out, "( A, '   - ared (', A, ES8.2, '), pred (', A, &
            &   ES8.2, '), successful (', L1, '), very_successful (', L1,')',  &
            &   /, A, '   - delta_l (', A, ES8.2,                              &
            &      '), delta h^2 (', ES8.2, '), v-step (', L1, ')' )" )        &
               prefix, TRIM( STRING_sign( delta_f, .FALSE. ) ), ABS( delta_f ),&
               TRIM( STRING_sign( delta_m, .FALSE. ) ), ABS( delta_m ),        &
               delta_f >= data%control%eta_successful * delta_m,               &
               delta_f >= data%control%eta_very_successful* delta_m,           &
               prefix, TRIM( STRING_sign( data%delta_l,  .FALSE. ) ),          &
               ABS( data%delta_l ),                                            &
               inform%primal_infeasibility_rest ** 2, data%delta_l <           &
               data%control%delta_feas * inform%primal_infeasibility_rest ** 2
  !            IF ( data%take_eqp_step )                                       &
  !              WRITE( data%out, "( ' - EQP stepsize =', ES11.4 )" ) data%step
             END IF

!  the new point is acceptable. Now check whether the predicted reduction of 
!  the objective function is sufficiently positive and has been realized for 
!  the true objective function

             IF ( delta_f >= data%control%eta_successful * delta_m &
                 .OR. data%delta_l < data%control%delta_feas *                 &
                                  inform%primal_infeasibility_rest ** 2 ) THEN
               data%successful = .TRUE.
               data%very_successful                                            &
                 = delta_f >= data%control%eta_very_successful * delta_m
               IF ( data%very_successful ) THEN
                 data%it_type = 'v'
               ELSE
                 data%it_type = 's'
               END IF
               IF ( data%printt ) THEN
                 IF ( data%very_successful ) THEN
                   IF ( data%take_eqp_step ) THEN
                     WRITE( data%out, "( A, '   - ', A3, ' step is very',      &
                    &    ' successful stepsize =', ES11.4 )" )                 &
                            prefix, data%d_name, data%step_eqp
                   ELSE
                     WRITE( data%out, "( A, '   - ', A3, ' step is very',      &
                    & ' successful' )" ) prefix, data%d_name
                   END IF
                 ELSE 
                   IF ( data%take_eqp_step ) THEN
                     WRITE( data%out, "( A, '   - ', A3, ' step is ',          &
                    &  ' successful, stepsize =', ES11.4 )" )                  &
                       prefix, data%d_name, data%step_eqp
                   ELSE
                     WRITE( data%out, "( A, '   - ', A3, ' step is',           &
                    &  ' successful' )" ) prefix, data%d_name
                   END IF
                 END IF
               END IF
               GO TO 290

!  the new point is unsuccessful

             ELSE
               data%it_type = 'u'
               IF ( data%printt ) THEN
                 IF ( data%take_eqp_step ) THEN
                   WRITE( data%out, "( A, '   - ', A3, ' step is ',            &
                  &  'unsuccessful, stepsize =', ES11.4 )" )                   &
                    prefix, data%d_name, data%step_eqp
                 ELSE
                   WRITE( data%out, "( A, '   - ', A3,                         &
                  &  ' step is unsuccessful')") prefix, data%d_name
                 END IF
               END IF
             END IF

!  the new point is not acceptable to the filter

           ELSE
             data%it_type = 'r'
             IF ( data%take_eqp_step ) data%n_pr_max = 0
             IF ( data%printt ) WRITE( data%out,                               &
            "( /, A, ' * ', A3, ' point not acceptable to filter (v,o) = (',   &
              &    ES10.4, ',', ES11.4, ')', /, A, 26X,                        &
              &  ' vs current (v,o) = (', ES10.4, ',', ES11.4, ')' )" )        &
                 prefix, data%d_name, primal_infeasibility_trial, obj_trial,   &
                 prefix, inform%primal_infeasibility_rest,                     &
                 inform%obj_restoration
           END IF  
         END IF  

!  try the RLP step after the EQP step

         IF ( data%take_eqp_step ) THEN
           data%take_eqp_step = .FALSE.
           GO TO 200
         END IF
         data%d_type = ' '

!  ------------------------------------------------
!  end of loop over the potential search directions
!  ------------------------------------------------

 290   CONTINUE

!  .......................
!  the step was successful
!  .......................

       IF ( data%successful ) THEN

!  the objective from the RLP has not decreased sufficiently. Add the current
!   point to the filter

         IF ( add_all_to_filter .OR.                                           &
              ( inform%primal_infeasibility_rest > data%stop_p .AND.           &
                data%delta_l < data%control%delta_feas *                       &
                 inform%primal_infeasibility_rest ** 2 ) ) THEN
           IF ( data%printt ) WRITE( data%out, "( A, ' - Current point added', &
          & ' to filter: insufficient RLP reduction' )" ) prefix
           o = inform%obj_restoration - data%control%gamma_filter              &
                 * inform%primal_infeasibility_rest
           v = data%control%beta_filter * inform%primal_infeasibility_rest
           CALL FILTER_update_filter( o, v, data%FILTER_restoration_data,      &
                                      data%control%FILTER_control,             &
                                      inform%FILTER_restoration_inform )
         END IF

!  move to the new point

         inform%obj_restoration = obj_trial
         nlp%X( : nlp%n ) = data%X_trial( : nlp%n )
         nlp%C( : nlp%m ) = data%C_trial( : nlp%m )
         data%S( : nlp%m ) = data%S_trial( : nlp%m )
         inform%primal_infeasibility                                           &
           = INFINITY_norm( nlp%C_scale( : nlp%m ) * data%S( : nlp%m ) )
         inform%primal_infeasibility_rest = primal_infeasibility_trial
         data%new_point = .TRUE.
         data%new_gradient = .TRUE.

!  reset the radius ?

         IF ( data%very_successful ) THEN
           data%mu = MIN( MAX( data%control%initial_radius_rlp,                &
                               data%control%rlp_radius_increase * data%mu ),   &
                          data%mu_max )
           data%radius_eqp = data%control%eqp_radius_increase * data%radius_eqp
         ELSE
!          data%radius_eqp = two * data%radius_eqp
         END IF

!  .........................
!  the step was unsuccessful
!  .........................

       ELSE

!  reduce the trust-region radius

         IF ( data%mu_new >= mu_tiny ) THEN
           data%mu =                                                           &
             data%control%rlp_radius_reduce * MIN( data%mu_new, data%mu )

!  if the active step will not change with a reduction in mu, enter the
!  restoration phase

         ELSE
           IF ( data%new_point ) THEN
             IF ( inform%primal_infeasibility_rest <= data%stop_p_rest ) THEN
               data%mu = data%control%rlp_radius_reduce * data%mu
             ELSE
               data%new_point = .FALSE.
               data%mu =                                                       &
                 data%control%rlp_radius_reduce * MIN( data%mu_new, data%mu )
             END IF
           ELSE 
             data%restoration_restoration = .TRUE. ; GO TO 300
           END IF
!          IF ( .NOT. data%new_point ) THEN
!!           write(6,*) data%norm_dlp_restoration, data%old_norm_dlp_restoration
!            ratio_norm_dlp =                                                  &
!              data%norm_dlp_restoration / data%old_norm_dlp_restoration
!            IF ( data%printt ) WRITE( data%out,                               &
!             "( A, '   - ratio new/old norm d_lp = ', ES10.4 )" )             &
!               prefix, ratio_norm_dlp
!  if the norm of the step has not significantly decreased, enter the
!  restoration phase
!            IF ( ratio_norm_dlp > 0.9_wp .AND.                                &
!                 inform%primal_infeasibility_rest > zero ) THEN
!              data%restoration_restoration = .TRUE. ; GO TO 300
!            END IF
!          END IF
!          data%mu = half * data%mu
!          data%mu = tenth * data%mu
         END IF

!  reduce the EQP trust-region radius

         IF ( data%n_pr_max <= 1 ) THEN
           data%radius_eqp = data%control%eqp_radius_reduce * data%radius_eqp
         ELSE
           data%radius_eqp =                                                   &
           ( data%control%eqp_radius_reduce ** data%n_pr_max ) * data%radius_eqp
         END IF
         data%new_gradient = .FALSE.
       END IF
!write(6,*) ' c_new violation ', &
! MAXVAL( MAX( nlp%C_l(:nlp%m) -  nlp%C(:nlp%m) +    &
!              nlp%C_scale(:nlp%m) * data%S(:nlp%m),  &
!              nlp%C(:nlp%m) - nlp%C_scale(:nlp%m) * &
!              data%S(:nlp%m) - nlp%C_u(:nlp%m), zero ) )
!write(6,*) ' c,s 1', &
! nlp%C_l(1), nlp%C(1), nlp%C_scale(1) * data%S(1), nlp%C_u(1)

!  ----------------------------------------------------------------------------
!                 SECOND-ORDER LAGRANGE MULTIPLIER UPDATES
!  ----------------------------------------------------------------------------

       IF ( data%control%multipliers == 2 ) THEN

!  compute the changes in multipliers and dual variables

         DO i = 1, nlp%m 
           IF ( nlp%C_status( i ) == 0 ) THEN
             data%DY( i ) = - nlp%Y( i )
           ELSE
             data%DY( i ) =                                                    &
               data%EQP_prob%Y( ABS( nlp%C_status( i ) ) ) - nlp%Y( i )
           END IF
         END DO
         DO i = 1, nlp%n
           IF ( nlp%X_status( i ) == 0 ) THEN
             data%DZ( i ) = - nlp%Z( i )
           ELSE
             data%DZ( i ) =                                                    &
               data%EQP_prob%Y( ABS( nlp%X_status( i ) ) ) - nlp%Z( i )
           END IF
         END DO

!        WRITE( 6, * ) ' max y ', MAXVAL( ABS( nlp%Y( : nlp%m ) ) )

!         DO i = 1, nlp%n
!           IF ( nlp%X_status( i ) == 0 ) THEN
!             data%DZ( i ) = - nlp%Z( i )
!           ELSE
!             data%EQP_prob%m = data%EQP_prob%m + 1
!             data%DZ( i ) = data%EQP_prob%Y( data%EQP_prob%m ) - nlp%Z( i )
!           END IF
!         END DO

!  compute A^T dy

         data%ATDY( : nlp%n ) = data%DZ( : nlp%n )
         data%ATDY( nlp%n + 1 : data%n_restoration )                           &
           = - nlp%C_scale( : nlp%m ) * data%DY( : nlp%m )
         DO l = 1, data%J_ne
           i = data%CQP_prob%A%col( l )
           data%ATDY( i ) = data%ATDY( i ) +                                   &
             data%CQP_prob%A%val( l ) * data%DY( data%CQP_prob%A%row( l ) )
         END DO

!  find the minimizer alpha of || g_l - alpha A^T dy ||_2^2

         alpha = DOT_PRODUCT( data%ATDY( : data%n_restoration ),               &
                              data%ATDY( : data%n_restoration ) )

         IF ( alpha /= zero )                                                  &
           alpha = DOT_PRODUCT( data%GL( : data%n_restoration ),               &
                                data%ATDY( : data%n_restoration ) ) / alpha
!  update the multipliers

         IF ( alpha /= zero ) THEN
           nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) + alpha * data%DY( : nlp%m )
           nlp%Z( : nlp%n ) = nlp%Z( : nlp%n ) + alpha * data%DZ( : nlp%n )
           data%GL( : data%n_restoration )                                     &
             = data%GL( : data%n_restoration )                                 &
               - alpha * data%ATDY( : data%n_restoration )

           inform%dual_infeasibility_rest =                                    &
             OPT_dual_infeasibility( data%n_restoration,                       &
                                     data%gL( : data%n_restoration ) )
           IF ( data%printm ) WRITE( data%out,                                 &
             "( A, '   second-order dual residual = ', ES10.4 )") prefix,      &
               inform%dual_infeasibility_rest

!  recompute the Hessian

         END IF
       END IF

!      IF ( data%control%multipliers /= 2 ) THEN
!write(6,*) data%successful 
!write(6,*) data%norm_deqp <= data%stop_p_rest
!write(6,*) primal_infeasibility_trial
!write(6,*) data%stop_p_rest
         IF ( data%successful .OR. ( data%norm_deqp <= data%stop_p_rest     &
              .AND. primal_infeasibility_trial <= data%stop_p_rest ) ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' * updating the Lagrange multipliers' )" ) prefix

!  update the Lagrange multipliers; also compute the number of multipliers 
!  with the wrong signs

           data%n_mult_wrong_sign = 0
           DO i = 1, nlp%m 
             IF ( nlp%C_status( i ) == 0 ) THEN
!              nlp%Y( i ) = zero
             ELSE
!              nlp%Y( i ) = data%EQP_prob%Y(  ABS( nlp%C_status( i ) ) )
               IF ( nlp%C_l( i ) /= nlp%C_u( i ) .AND.                         &
                  ( ( nlp%C_status( i ) < 0 .AND. nlp%Y( i ) < - y_tiny ) .OR. &
                    ( nlp%C_status( i ) > 0 .AND. nlp%Y( i ) > y_tiny ) ) )    &
                 data%n_mult_wrong_sign = data%n_mult_wrong_sign + 1
             END IF
           END DO
!          WRITE( 6, * ) ' max y ', MAXVAL( ABS( nlp%Y( : nlp%m ) ) )

           DO i = 1, nlp%n
             IF ( nlp%X_status( i ) == 0 ) THEN
!              nlp%Z( i ) = zero
             ELSE
!              nlp%Z( i ) = data%EQP_prob%Y( ABS( nlp%X_status( i ) ) )
               IF ( nlp%X_l( i ) /= nlp%X_u( i ) .AND.                         &
                  ( ( nlp%X_status( i ) < 0 .AND. nlp%Z( i ) < - z_tiny ) .OR. &
                    ( nlp%X_status( i ) > 0 .AND. nlp%Z( i ) > z_tiny ) ) )    &
                 data%n_mult_wrong_sign = data%n_mult_wrong_sign + 1
             END IF
           END DO
!          WRITE( 6, * ) ' max z ', MAXVAL( ABS( nlp%Z( : nlp%n ) ) )
           IF ( data%printt ) WRITE( data%out,                                 &
             "( A, '    ', I0, ' multiplier', A, ' ', A, ' the wrong sign')" ) &
             prefix, data%n_mult_wrong_sign,                                   &
             TRIM( STRING_pleural( data%n_mult_wrong_sign ) ),                 &
             TRIM( STRING_have( data%n_mult_wrong_sign ) )

           IF ( data%printd ) THEN
             WRITE( data%out, 2020 ) prefix
             WRITE( data%out, "( A, I6, 5ES12.4 )" )                           &
              ( prefix, i, nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ),             &
                data%EQP_prob%X( i ), nlp%Z( i ), i = 1, nlp%n )
             WRITE( data%out, "( / )" )
           END IF
         END IF
!      END IF

!  ----------------------------------------------------------------------------
!                       RESTORATION PHASE
!  ----------------------------------------------------------------------------

  300  CONTINUE
       IF ( data%restoration_restoration ) THEN
         write(6,*) ' should not be in need of restoration ... stopping'
         STOP
       END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    M A I N    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     GO TO 100

!  -------------
!  Normal return
!  -------------

 900 CONTINUE
     inform%status = 0

!  print the solution

 905 CONTINUE

!    WRITE( 44, * ) ' nlp%X_status '
!    WRITE( 44, "( ( 10I7 ) )" ) nlp%X_status
!    WRITE( 44, * ) ' nlp%C_status '
!    WRITE( 44, "( ( 10I7 ) )" ) nlp%C_status

     IF ( data%printd ) THEN
       l = 2
       IF ( data%control%fulsol ) l = nlp%n 
       IF ( data%control%print_level >= 10 ) l = nlp%n

       WRITE( data%out, "( /, A, ' Solution: ', /, A, '                   ',   &
      &         '             <------ Bounds ------> ', /, A,                  &
      &         '      # name          value   ',                              &
      &         '    Lower       Upper       Dual ' )" ) prefix, prefix, prefix
       DO j = 1, 2 
         IF ( j == 1 ) THEN 
           ir = 1 ; ic = MIN( l, nlp%n ) 
         ELSE 
           IF ( ic < nlp%n - l ) WRITE( data%out, 2010 ) prefix
           ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
         END IF 
         DO i = ir, ic 
           WRITE( data%out, 2000 ) prefix, i, nlp%VNAMES( i ), nlp%X( i ),     &
             nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
         END DO
       END DO

       IF ( nlp%m > 0 ) THEN
         l = 2
         IF ( data%control%fulsol ) l = nlp%m
         IF ( data%control%print_level >= 10 ) l = nlp%m

         WRITE( data%out, "( /, A, ' Constraints:', /, A, '                 ', &
        &      '               <------ Bounds ------> ', /, A,                 &
        &      '      # name           value   ',                              &
        &      '    Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
         DO j = 1, 2 
           IF ( j == 1 ) THEN 
             ir = 1 ; ic = MIN( l, nlp%m ) 
           ELSE 
             IF ( ic < nlp%m - l ) WRITE( data%out, 2010 ) prefix
             ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
           END IF 
           DO i = ir, ic 
             WRITE( data%out, 2000 ) prefix, i, nlp%CNAMES( i ), nlp%C( i ),   &
               nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
           END DO
         END DO
       END IF
     END IF

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start

     IF ( nlp%m > 0 ) THEN ; max_y = MAXVAL( ABS( nlp%Y( : nlp%m ) ) ) ; 
       ELSE ; max_y = zero ; END IF

     IF ( data%printt ) WRITE( data%out, "( /, A, ' Problem: ', 16X, A10,      &
    &          '     Solver: ', 9X, '  FASTr_restoration', /, A,               &
    &  ' n              =     ',bn, I12, '       m               = ',bn, I12,/,&
    & A, ' Infeasibility  = ', ES16.8, '       Complementarity = ', ES12.4, /, &
    & A,' Violation      =     ',ES12.4, '       Dual infeas     = ', ES12.4,/,&
    & A,' Max multiplier =     ',ES12.4, '       Max dual var.   = ', ES12.4,/,&
    & A,' Iterations     =     ',bn, I12, '       Time            = ', F12.2)")&
      prefix, nlp%pname, prefix, nlp%n, nlp%m, prefix,                         &
      inform%primal_infeasibility, inform%complementary_slackness_rest,        &
      prefix, inform%primal_infeasibility_rest,                                &
      inform%dual_infeasibility_rest, prefix, max_y,                           &
      MAXVAL( ABS( nlp%Z( : nlp%n ) ) ),                                       &
      prefix, inform%iter, inform%time%clock_total

     IF ( data%control%error > 0 .AND. data%control%print_level > 0 ) THEN
       CALL SYMBOLS_status( inform%status, data%control%error, prefix,         &
                            'FASTr_restoration' )
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start
     RETURN

!  other errors

 990 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = data%time_now - data%time_start
     inform%time%clock_total = data%clock_now - data%clock_start
!    WRITE( data%control%error, "( /, ' ** Message from -FASTR_solve-',        &
!   &  ' error exit (status = ', I0, ')', / )" ) inform%status
     RETURN

!  non-executable statements

 2000 FORMAT( A, I7, 1X, A10, 4ES12.4 ) 
 2010 FORMAT( A, 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2020 FORMAT( /, A, '     i      X_l         X         X_u',                   &
                    '          Dx          Z')
 2030 FORMAT( A, '   - Error return from EQP_solve, status = ', I0 )

!  End of subroutine FASTR_restoration

     END SUBROUTINE FASTR_restoration

!  End of module GALAHAD_FASTR

   END MODULE GALAHAD_FASTR_double




