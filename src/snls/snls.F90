! THIS VERSION: GALAHAD 5.5 - 2026-03-07 AT 15:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ S N L S   M O D U L E  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 5.5 January 14th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SNLS_precision

!     --------------------------------------------------------------------
!    |                                                                    |
!    |   SNLS, an algorithm for nonlinear least-squares over simplices    |
!    |                                                                    |
!    |   Aim: find a (local) minimizer x in R^n of the objective function |
!    |                                                                    |
!    |            1/2 ||r(x)||_W^2 = 1/2 sum_i=1^m_r w_i r_i^2(x)         |
!    |                                                                    |
!    |   subject to the non-overlapping simplex constraints               |
!    |                                                                    |
!    |       C(x) = { x:  e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m },     |
!    |                                                                    |
!    |   where the residual r(x) is a smooth vector-valued function,      |
!    |   the weights w_i are positive and the Ci are non-overlapping      |
!    |   index subsets of {1,...,n}                                       |
!    |                                                                    |
!     --------------------------------------------------------------------

     USE GALAHAD_KINDS_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_REVERSE_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_QPT_precision
     USE GALAHAD_SLLS_precision
     USE GALAHAD_SLLSB_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_MOP_precision, ONLY: mop_Ax
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_STRING, ONLY: STRING_integer_6

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SNLS_initialize, SNLS_read_specfile, SNLS_solve, SNLS_terminate,&
               SNLS_full_initialize, SNLS_full_terminate, SNLS_import,         &
               SNLS_import_without_jac, SNLS_information, SNLS_solve_with_jac, &
               SNLS_solve_with_jacprod, SNLS_solve_reverse_with_jac,           &
               SNLS_solve_reverse_with_jacprod, SNLS_reset_control,            &
               NLPT_problem_type, USERDATA_type, REVERSE_type, SMT_type, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SNLS_initialize
       MODULE PROCEDURE SNLS_initialize, SNLS_full_initialize
     END INTERFACE SNLS_initialize

     INTERFACE SNLS_terminate
       MODULE PROCEDURE SNLS_terminate, SNLS_full_terminate
     END INTERFACE SNLS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER ( KIND = ip_ ), PARAMETER  :: nskip_prec_max = 0
     INTEGER ( KIND = ip_ ), PARAMETER  :: history_max = 100
     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: four = 4.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: quarter = 0.25_rp_
     REAL ( KIND = rp_ ), PARAMETER :: eighth = 0.125_rp_
     REAL ( KIND = rp_ ), PARAMETER :: tenth = 0.1_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteenth = 0.0625_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: hundred = 100.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteen = 16.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: tenm3 = ten ** ( - 3 )
     REAL ( KIND = rp_ ), PARAMETER :: tenm6 = ten ** ( - 6 )
     REAL ( KIND = rp_ ), PARAMETER :: tenm8 = ten ** ( - 8 )
     REAL ( KIND = rp_ ), PARAMETER :: point9 = 0.9_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = rp_ ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: teneps = ten * epsmch

     REAL ( KIND = rp_ ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = rp_ ), PARAMETER :: gamma_2 = half
     REAL ( KIND = rp_ ), PARAMETER :: gamma_3 = two
     REAL ( KIND = rp_ ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = rp_ ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = rp_ ), PARAMETER :: mu_2 = point1
!    REAL ( KIND = rp_ ), PARAMETER :: theta = half
     REAL ( KIND = rp_ ), PARAMETER :: theta = point1
     REAL ( KIND = rp_ ), PARAMETER :: weight_zero = epsmch

!  weight update strategies

     INTEGER ( KIND = ip_ ), PARAMETER  :: weight_update_basic = 1
     INTEGER ( KIND = ip_ ), PARAMETER  :: weight_update_zero_reset = 2
     INTEGER ( KIND = ip_ ), PARAMETER  :: weight_update_imitate_tr = 3
     INTEGER ( KIND = ip_ ), PARAMETER  :: weight_update_increase = 4
     INTEGER ( KIND = ip_ ), PARAMETER  :: weight_update_gpt = 5

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SNLS_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!   any printing will start on this iteration

       INTEGER ( KIND = ip_ ) :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER ( KIND = ip_ ) :: stop_print = - 1

!   the number of iterations between printing

       INTEGER ( KIND = ip_ ) :: print_gap = 1

!   the maximum number of iterations performed

       INTEGER ( KIND = ip_ ) :: maxit = 100

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER ( KIND = ip_ ) :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d                       '

!   is the Jacobian matrix of first derivatives available (>= 2), is access
!    only via matrix-vector products (=1) or is it not available (<=0) ?

       INTEGER ( KIND = ip_ ) :: jacobian_available = 1

!   specify method to be used to solve the subproblem
!
!     1  use a projection method (SLLS)
!     2  use an interior-point method (SLLSB)
!     3  start with an interior-point method but later switch to projection

       INTEGER ( KIND = ip_ ) :: subproblem_solver = 1

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!     strategy with this history length used

       INTEGER ( KIND = ip_ ) :: non_monotone = 1

!   define the weight-update strategy:
!        1 (basic), 2 (reset to zero when very successful),
!        3 (imitate TR), 4 (increase lower bound), 5 (GPT)

       INTEGER ( KIND = ip_ ) :: weight_update_strategy = weight_update_basic

!   overall convergence tolerances. The iteration will terminate when
!      ||r||_2 <= MAX( %stop_r_absolute, %stop_r_relative * ||r_initial||_2
!     or when the norm of the projected gradient pg = P[x-J^T(x) r(x)]-x] 
!     of ||r||_2 satisfies ||pg||_2 <= 
!       MAX( %stop_pg_absolute, %stop_pg_relative * ||pg_initial||_2,
!     or if the step is less than %stop_s

#ifdef REAL_32
       REAL ( KIND = rp_ ) :: stop_r_absolute = tenm3
#else
       REAL ( KIND = rp_ ) :: stop_r_absolute = tenm6
#endif
       REAL ( KIND = rp_ ) :: stop_r_relative = - one
#ifdef REAL_32
       REAL ( KIND = rp_ ) :: stop_pg_absolute = tenm3
#else
       REAL ( KIND = rp_ ) :: stop_pg_absolute = tenm6
#endif
       REAL ( KIND = rp_ ) :: stop_pg_relative = - one
       REAL ( KIND = rp_ ) :: stop_s = epsmch

!     The iteration will switch from an interior-point to a projection solver
!     when subproblem_solver = 3 if ||pg||_2 satisfies ||pg||_2 <= 
!       MAX( %stop_pg_absolute, %stop_pg_switch * ||pg_initial||_2,

#ifdef REAL_32
       REAL ( KIND = rp_ ) :: stop_pg_switch = point01
#else
       REAL ( KIND = rp_ ) :: stop_pg_switch = tenm3
#endif

!   initial value for the regularization weight  (-ve => 1/||g_0||)

       REAL ( KIND = rp_ ) :: initial_weight = hundred

!   minimum permitted regularization weight

       REAL ( KIND = rp_ ) :: minimum_weight = tenm8

!   a potential iterate will only be accepted if the actual decrease
!    f - f(x_new) is larger than %eta_successful times that predicted
!    by a quadratic model of the decrease. The regularization weight will be
!    decreaed if this relative decrease is greater than %eta_very_successful
!    but smaller than %eta_too_successful

       REAL ( KIND = rp_ ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = rp_ ) :: eta_very_successful = point9
       REAL ( KIND = rp_ ) :: eta_too_successful = two

!   on very successful iterations, the regularization weight will be reduced
!    by the factor %weight_decrease but no more than %weight_decrease_min
!    while if the iteration is unsucceful, the weight will be increased by a
!    factor %weight_increase but no more than %weight_increase_max
!    (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
!    and Toint, 2011)

       REAL ( KIND = rp_ ) :: weight_decrease_min = point1
!      REAL ( KIND = rp_ ) :: weight_decrease = half
       REAL ( KIND = rp_ ) :: weight_decrease = point1
!      REAL ( KIND = rp_ ) :: weight_increase = two
       REAL ( KIND = rp_ ) :: weight_increase = ten
       REAL ( KIND = rp_ ) :: weight_increase_max = hundred

!   the value of the two-norm of the projected gradient required before
!   a switch is made from the Gauss-Newton to the Newton model when 
!   %newton_acceleration is .TRUE. (Not yet implemented)

       REAL ( KIND = rp_ ) :: switch_to_newton = point1

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: clock_time_limit = - one

!   should second derivatives be used to accelerate the convergence of the 
!   algorithm? (Not yet implemented)

       LOGICAL :: newton_acceleration = .FALSE.

!   allow the user to perform a "magic" step to improve the objective

       LOGICAL :: magic_step = .FALSE.

!   print values of the objective/projected gradient rather than ||r|| and 
!   the projected gradient

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

!  control parameters for SLLS

       TYPE ( SLLS_control_type ) :: SLLS_control

!  control parameters for SLLSB

       TYPE ( SLLSB_control_type ) :: SLLSB_control

     END TYPE SNLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived types with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SNLS_time_type

!  the total CPU time spent in the package

       REAL ( KIND = rp_ ) :: total = 0.0

!  the total CPU time spent in the slls package

       REAL ( KIND = rp_ ) :: slls = 0.0

!  the total CPU time spent in the sllsb package

       REAL ( KIND = rp_ ) :: sllsb = 0.0

!  the total clock time spent in the package

       REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the clock time spent in the slls package

       REAL ( KIND = rp_ ) :: clock_slls = 0.0

!  the clock time spent in the sllsb package

       REAL ( KIND = rp_ ) :: clock_sllsb = 0.0

     END TYPE SNLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived types with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SNLS_inform_type

!  return status. See SNLS_solve for details

       INTEGER ( KIND = ip_ ) :: status = 1  ! initial entry

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the name of the user-supplied evaluation routine for which an error ocurred

       CHARACTER ( LEN = 12 ) :: bad_eval = REPEAT( ' ', 12 )

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

!  the total number of inner iterations performed

       INTEGER ( KIND = ip_ ) :: inner_iter = 0

!  the total number of evaluations of the residual function r(x)

       INTEGER ( KIND = ip_ ) :: r_eval = 0

!  the total number of evaluations of the Jacobian J(x) of r(x)

       INTEGER ( KIND = ip_ ) :: jr_eval = 0

!  the value of the objective function 1/2||r(x)||^2_W at the best estimate of
!   the solution, x, determined by SNLS_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the norm of the residual ||r(x)||_W at the best estimate of the solution,
!   x, determined by SNLS_solve

       REAL ( KIND = rp_ ) :: norm_r = HUGE( one )

!  the norm of the gradient of ||r(x)||_W of the objective function
!   at the best estimate, x, of the solution determined by SNLS_solve

       REAL ( KIND = rp_ ) :: norm_g = HUGE( one )

!  the norm of the projected gradient ||P[x - alpha g(x)]-x||_W of the 
!   objective function into the feasible set at the best estimate, x, 
!   of the solution determined by SNLS_solve

       REAL ( KIND = rp_ ) :: norm_pg = HUGE( one )

!  the current regularization weight used

       REAL ( KIND = rp_ ) :: weight = one

!  timings (see above)

       TYPE ( SNLS_time_type ) :: time

!  inform parameters for SLLS

       TYPE ( SLLS_inform_type ) :: SLLS_inform

!  inform parameters for SLLSB

       TYPE ( SLLSB_inform_type ) :: SLLSB_inform

     END TYPE SNLS_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: SNLS_data_type
       INTEGER ( KIND = ip_ ) :: branch = 1
       INTEGER ( KIND = ip_ ) :: eval_status, out, start_print, stop_print
       INTEGER ( KIND = ip_ ) :: print_level
       INTEGER ( KIND = ip_ ) :: print_level_slls, print_level_sllsb
       INTEGER ( KIND = ip_ ) :: len_history, ibound, ipoint, icp, lbfgs_mem
       INTEGER ( KIND = ip_ ) :: nskip_lbfgs, nskip_prec, non_monotone_history
       INTEGER ( KIND = ip_ ) :: print_gap, max_diffs, latest_diff, total_diffs
       INTEGER ( KIND = ip_ ) :: total_facts, h_ne, s_ne, max_hist, nf
       INTEGER ( KIND = ip_ ) :: ref( 1 )
       REAL ( KIND = rp_ ) :: time_start, time_record, time_now
       REAL ( KIND = rp_ ) :: clock_start, clock_record, clock_now, delta
       REAL ( KIND = rp_ ) :: f_ref, f_trial, f_best, m_best, model, ratio, rp
       REAL ( KIND = rp_ ) :: weight, old_weight, weight_trial, etat, ometat
       REAL ( KIND = rp_ ) :: df, stg, hstbs, s_norm, weight_max, xtsx_current
       REAL ( KIND = rp_ ) :: dm, stop_r, stop_pg, stop_pg_switch
       REAL ( KIND = rp_ ) :: s_new_norm, norm_r_trial
       REAL ( KIND = rp_ ) :: a0, a1, a2, a3, a4, steplength, g_norm, phi, xtsx
       REAL ( KIND = rp_ ) :: inner_weight, s_norm_successful, final_weight
       REAL ( KIND = rp_ ) :: minimum_weight, obj_current, norm_r_current
       LOGICAL :: printi, printt, printd, printw, printm
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm, set_printw
       LOGICAL :: monotone, new_point, got_jr, got_h, poor_model, n_or_gn
       LOGICAL :: reverse_r, reverse_jr, reverse_jr_prod, reverse_jr_scol
       LOGICAL :: reverse_jr_sprod, reverse_internal
       LOGICAL :: successful, transpose, reduce, f_is_nan, g_is_nan
       LOGICAL :: jacobian_available, re_entry, multiple_simplices
       LOGICAL :: w_eq_identity, step_accepted, solve_projection
       CHARACTER ( LEN = 1 ) :: perturb = ' '
       CHARACTER ( LEN = 1 ) :: hard = ' '
       CHARACTER ( LEN = 1 ) :: accept = ' '
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_current
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R_current
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_current
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: D_hist
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: F_hist

       TYPE ( QPT_problem_type ) :: GN_model

!  copys of controls and informs

       TYPE ( SNLS_control_type ) :: control
       TYPE ( SNLS_inform_type ) :: inform

!  data for SLLS

       TYPE ( SLLS_data_type ) :: SLLS_data
       TYPE ( REVERSE_type ) :: reverse

!  data for SLLSB

       TYPE ( SLLSB_data_type ) :: SLLSB_data

     END TYPE SNLS_data_type

     TYPE, PUBLIC :: SNLS_full_data_type
       LOGICAL :: f_indexing = .TRUE.
       TYPE ( SNLS_data_type ) :: SNLS_data
       TYPE ( SNLS_control_type ) :: SNLS_control
       TYPE ( SNLS_inform_type ) :: SNLS_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( USERDATA_type ) :: userdata
       TYPE ( REVERSE_type ) :: reverse
     END TYPE SNLS_full_data_type

   CONTAINS

! G A L A H A D - N E W T O N _ S N L S _ I N I T I A L I Z E  S U B R O U TI NE

     SUBROUTINE SNLS_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SNLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SNLS_inform_type ), INTENT( OUT ) :: inform
     TYPE ( SNLS_data_type ), INTENT( INOUT ) :: data

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize SLLS components

     CALL SLLS_initialize( data%SLLS_data, control%SLLS_control,               &
                           inform%SLLS_inform )
     control%SLLS_control%prefix = '" - SLLS:"                    '

!  initalize SLLSB components

     CALL SLLSB_initialize( data%SLLSB_data, control%SLLSB_control,            &
                            inform%SLLSB_inform )
     control%SLLSB_control%prefix = '" - SLLSB:"                   '

!  initial private data. Set branch for initial entry

     data%branch = 1

     RETURN

!  End of subroutine SNLS_initialize

     END SUBROUTINE SNLS_initialize

!- G A L A H A D -  S N L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE SNLS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SNLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SNLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SNLS_inform_type ), INTENT( OUT ) :: inform

     CALL SNLS_initialize( data%snls_data, control, inform )

     RETURN

!  End of subroutine SNLS_full_initialize

     END SUBROUTINE SNLS_full_initialize

!-*-  S N L S _ N E W T O N _ R E A D _ S P E C F I L E  S U B R O U T I N E  -

     SUBROUTINE SNLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by SNLS_initialize could (roughly)
!  have been set as:

! BEGIN SNLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  jacobian-available                              2
!  subproblem-solver                               1
!  history-length-for-non-monotone-descent         0
!  weight-update-strategy                          0
!  absolute-residual-accuracy-required             1.0D-8
!  relative-residual-reduction-required            2.0D-16
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-8
!  minimum-step-allowed                            2.0D-16
!  relative-gradient-switch-tolerance              1.0D-3
!  initial-regularization-weight                   1.0D+0
!  minimum-regularization-weight                   1.0D-9
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  too-successful-iteration-tolerance              2.0
!  regularization-weight-minimum-decrease-factor   0.1
!  regularization-weight-decrease-factor           0.5
!  regularization-weight-increase-factor           2.0
!  regularization-weight-maximum-increase-factor   100.0
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  try-newton-acceleration                         no
!  choose-magic-step                               no
!  print-objective                                 no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END SNLS SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_control_type ), INTENT( INOUT ) :: control
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_print = start_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_gap = stop_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: maxit = print_gap + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_unit = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: jacobian_available = alive_unit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: subproblem_solver                    &
                                            = jacobian_available + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: non_monotone = subproblem_solver + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight_update_strategy               &
                                            = non_monotone + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_r_absolute                      &
                                            = weight_update_strategy + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_r_relative = stop_r_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_pg_absolute = stop_r_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_pg_relative                     &
                                            = stop_pg_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_s = stop_pg_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_pg_switch = stop_s + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_weight = stop_pg_switch + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: minimum_weight = initial_weight + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_successful                       &
                                            = minimum_weight + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_very_successful                  &
                                            = eta_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_too_successful                   &
                                            = eta_very_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight_decrease_min                  &
                                            = eta_too_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight_decrease                      &
                                            = weight_decrease_min + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight_increase = weight_decrease + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight_increase_max                  &
                                            = weight_increase + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit                       &
                                            = weight_increase_max + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: newton_acceleration                  &
                                            = clock_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: magic_step = newton_acceleration + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_obj = magic_step + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = print_obj + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_file                           &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = alive_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 6 ), PARAMETER :: specname = 'SNLS'
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
     spec( subproblem_solver )%keyword = 'subproblem-solver'
     spec( non_monotone )%keyword = 'history-length-for-non-monotone-descent'
     spec( weight_update_strategy )%keyword = 'weight-update-strategy'

!  Real key-words

     spec( stop_r_absolute )%keyword = 'absolute-residual-accuracy-required'
     spec( stop_r_relative )%keyword = 'relative-residual-reduction-required'
     spec( stop_pg_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_pg_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( stop_pg_switch )%keyword = 'relative-gradient-switch-tolerance'
     spec( initial_weight )%keyword = 'initial-regularization-weight'
     spec( minimum_weight )%keyword = 'minimum-regularization-weight'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( eta_too_successful )%keyword = 'too-successful-iteration-tolerance'
     spec( weight_decrease_min )%keyword =                                     &
       'regularization-weight-minimum-decrease-factor'
     spec( weight_decrease )%keyword = 'regularization-weight-decrease-factor'
     spec( weight_increase )%keyword = 'regularization-weight-increase-factor'
     spec( weight_increase_max )%keyword =                                     &
       'regularization-weight-maximum-increase-factor'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( newton_acceleration )%keyword = 'try-newton-acceleration'
     spec( magic_step )%keyword = 'choose-magic-step'
     spec( print_obj )%keyword = 'print-objective'
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
     CALL SPECFILE_assign_value( spec( subproblem_solver ),                    &
                                 control%subproblem_solver,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( non_monotone ),                         &
                                 control%non_monotone,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( weight_update_strategy ),               &
                                 control%weight_update_strategy,               &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( stop_r_absolute ),                      &
                                 control%stop_r_absolute,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_r_relative ),                      &
                                 control%stop_r_relative,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_pg_absolute ),                     &
                                 control%stop_pg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_pg_relative ),                     &
                                 control%stop_pg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_pg_switch),                        &
                                 control%stop_pg_switch,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_weight ),                       &
                                 control%initial_weight,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minimum_weight ),                       &
                                 control%minimum_weight,                       &
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
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( newton_acceleration ),                  &
                                 control%newton_acceleration,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( magic_step ),                           &
                                 control%magic_step,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_obj ),                            &
                                 control%print_obj,                            &
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
       CALL SLLS_read_specfile( control%SLLS_control, device,                  &
              alt_specname = TRIM( alt_specname ) // '-SLLS' )
       CALL SLLSB_read_specfile( control%SLLSB_control, device,                &
              alt_specname = TRIM( alt_specname ) // '-SLLSB' )
     ELSE
       CALL SLLS_read_specfile( control%SLLS_control, device )
       CALL SLLSB_read_specfile( control%SLLSB_control, device )
     END IF

     RETURN

!  End of subroutine SNLS_read_specfile

     END SUBROUTINE SNLS_read_specfile

!-*-*-*-*-  G A L A H A D -  S N L S _ s o l v e  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE SNLS_solve( nlp, control, inform, data, userdata, reverse,     &
                            eval_R, eval_Jr, eval_Jr_prod, eval_Jr_scol,       &
                            eval_Jr_sprod )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  SNLS_solve, a regularization method for finding a local unconstrained
!    minimizer of a nonlinear least-squares objective,
!
!      f(x) = 1/2 ||r(x)||_W^2
!
!    subject to the non-overlapping simplex constraints
!
!       e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m_c,
!
!    where W is a positive-definite matrix, ||r||_W^2 = r^T W r
!    and the cohorts Ci are non-overlapping index subsets of {1,...,n}

!  This variant implements a Gauss-Newton-type method

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_SNLS.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_SNLS_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m_r is a scalar variable of type default integer, that holds the number of
!   residuals
!
!  m_c is a scalar variable of type default integer, that holds the number of
!   cohorts
!
!  R is a rank-one allocatable array of dimension m and type default real,
!   that holds the residuals r(x). The i-th component of R, i = 1,  ... ,  m_r,
!   contains r_i(x).
!
!  Jr is scalar variable of type SMT_TYPE that holds the Jacobian matrix
!   J(x) = nabla r(x), i.e., J_i,j(x) = d r_i(x) / d x_j. The following
!   components are used here:
!
!   Jr%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme is
!    used, the first five components of Jr%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of Jr%type must
!    contain the string COORDINATE, and for the sparse row-wise storage scheme,
!    the first fourteen components of Jr%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into J%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Jacobian we wish
!    to store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%Jr%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   Jr%ne is a scalar variable of type default integer, that holds the number
!    of entries in the Jacobian J(x) in the sparse co-ordinate storage scheme.
!    It need not be set for any of the other two schemes.
!
!   Jr%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries in the Jacobian J(x) in any of the available
!    storage schemes.
!
!   Jr%row is a rank-one allocatable array of type default integer, that holds
!    the row indices in the Jacobian J(x) in the sparse co-ordinate storage
!    scheme. It need not be allocated for any of the other two schemes.
!
!   Jr%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the Jacobian J(x) in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme.
!    It need not be allocated when the dense scheme is used.
!
!   Jr%ptr is a rank-one allocatable array of dimension m+1 and type default
!    integer, that holds the starting position of each row of Jacobian J(x),
!    as well as Jr%ptr(m+1) = the total number of entries plus one, in the
!    sparse row-wise storage scheme. It need not be allocated when the other
!    schemes are used.
!
!  COHORT is a rank-one allocatable array of dimension n and type default
!   intege whose j-th component may be set to the number, between 1 and m_c
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If COHORT is unallocated, all variables will be 
!   assumed to belong to a single cohort.
!
!  X is a rank-one allocatable array of dimension n and type default real, that
!   holds the values x of the optimization variables. The j-th component of
!   X, j = 1, ... , n, contains x_j.
!
!  Y is a rank-one allocatable array of dimension m_c and type default real, 
!   that holds the values y of the Lagrange multipliers for the simplex 
!   constraints. The i-th component of Y, i = 1, ... , m_c, contains 
!   the multiplier for the constraint involving variables in the i-th cohort.
!
!  Z is a rank-one allocatable array of dimension n and type default real, that
!   holds the values z of the dual variables for the non-neagtivity constraints.
!   The j-th component of Z, j = 1, ... , n, contains z_j.
!
!  G is a rank-one allocatable array of dimension n and type default real, that
!   holds the values g(x) = Jr^T r(x) of the gradient of the objective function.
!   The j-th component of G, j = 1, ... , n, contains g_j(x).
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
! controls is a scalar variable of type SNLS_control_type. See
!  SNLS_initialize for details
!
! inform is a scalar variable of type SNLS_inform_type. On initial entry,
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
!    -3. The restriction nlp%n > 0 or requirement that nlp%H_type contains
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
!     2. The user should compute the residual function value r(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The value of
!        the i-th component of the residual should be set in nlp%R(i), for i =
!        1, ..., m_r and reverse%eval_status should be set to 0. If the user is
!        unable to evaluate a component of r(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%R, but should then set
!        reverse%eval_status to a non-zero value.
!
!     3. The user should compute the Jacobian of the residual function Jr(x) =
!        nabla_x r(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value l-th component of the Jacobian stored according
!        to the scheme input in the remainder of nlp%Jr should be set in
!        nlp%Jr%val(l), for l = 1, ..., nlp%Jr%ne and reverse%eval_status should
!        be set to 0. If the user is unable to evaluate a component of J(x) -
!        for instance if a component of the Jacobian is undefined at x - the
!        user need not set nlp%Jr%val, but should then set reverse%eval_status
!        to a non-zero value.
!
!     4. The product Jr(x) * v of the matrix Jr(x) at the point x indicated 
!        in nlp%X with a given vector v is required from the user. The vector 
!        v will be provided in reverse%v and the required product must be 
!        returned in reverse%p. SNLS_solve must then be re-entered with 
!        reverse%eval_status set to 0, and any remaining arguments unchanged. 
!        Should the user be unable to form the product, this should be flagged 
!        by setting reverse%eval_status to a nonzero value
!
!     5. The product Jr(x)^T * v of the transpose of the matrix Jr(x) at the 
!        point x indicated in nlp%X with a given vector v is required from
!        the user. The vector v will be provided in reverse%v and the required 
!        product must be returned in reverse%p. SNLS_solve must then be
!        re-entered with reverse%eval_status set to 0, and any remaining 
!        arguments unchanged. Should the user be unable to form the product, 
!        this should be flagged by setting reverse%eval_status to a nonzero 
!        value
!
!     6. The j-th column of Jr(x) at the point x indicated in nlp%X is 
!        required from the user, where reverse%index holds the value of j. 
!        The resulting NONZEROS and their corresponding row indices of the 
!        j-th column of Jr must be placed in reverse%p( 1 : reverse%lp ) and
!        reverse%ip( 1 : reverse%lp ) with reverse%lp set accordingly. 
!        SNLS_solve should then be re-entered with all other arguments 
!        unchanged. Once again reverse%eval_status should be set to zero 
!        unless the column cannot be formed, in which case a nonzero value 
!        should be returned.
!
!     7. The product J(x) * v of the matrix J(x) at the point x indicated in 
!        nlp%X with a given sparse vector v is required from the user. Only 
!        components reverse%iv( reverse%lvl : reverse%lvu ) of the vector v 
!        stored in reverse%v are nonzero. The required product should be 
!        returned in reverse%p. SNLS_solve must then be re-entered with all 
!        other arguments unchanged. Typically v will be very sparse (i.e., 
!        reverse%lvu - reverse%lvl will be small). reverse%eval_status should 
!        be set to zero unless the product cannot be formed, in which case 
!        a nonzero value should be returned.
!
!     8. Specified components of the product Jr(x)^T * v of the transpose of
!        the matrix Jr(x) at the point x indicated in nlp%X with a given vector
!        v stored in reverse%v are requiredfrom the user. Only components 
!        indexed by reverse%iv( reverse%lvl : reverse%lvu ) of the product 
!        should be computed, and these should be recorded in
!        reverse%p( reverse%iv( reverse%lvl : reverse%lvu ) )
!        and SNLS_solve then re-entered with all other arguments unchanged.
!        reverse%eval_status should be set to zero unless the product cannot
!        be formed, in which case a nonzero value should be returned.
!
!     9. The user has the opportunity to replace the estimate x in nlp%X
!        by a value x_better for which f(x_better) <= f(x). If the user
!        choses to do so, she should replace nlp%X by x_better and also
!        record r(x_better) in nlp%R.
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
!  inner_iter is a scalar variable of type default integer, that gives the
!   total number of inner iterations required.
!
!  r_eval is a scalar variable of type default integer, that gives the
!   total number of residual function evaluations performed.
!
!  jr_eval is a scalar variable of type default integer, that gives the
!   total number of residual Jacobian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function 1/2 ||r(x)||_2^2 at the best estimate
!   of the solution found.
!
!  norm_r is a scalar variable of type default real, that holds the value of
!   the norm of the residual function ||r(x)||_2 at the best estimate of the
!   solution found.
!
!  norm_g is a scalar variable of type default real, that holds the value of
!   the norm of the residual function gradient ||J^T(x)r(x)||_2/||r(x)||_2
!   at the best estimate of the solution found.
!
!  time is a scalar variable of type SNLS_time_type whose components are
!   used to hold elapsed CPU and clock times for the various parts of the
!   calculation.
!
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    slls is a scalar variable of type default real, that gives
!      the total CPU time spent in the slls package.
!
!    sllsb is a scalar variable of type default real, that gives
!      the total CPU time spent in the sllsb package.
!
!    clock_total is a scalar variable of type default real, that gives
!     the total clock time spent in the package.
!
!    clock_slls is a scalar variable of type default real, that gives
!      the clock time spent in the slls package.
!
!    clock_sllsb is a scalar variable of type default real, that gives
!      the clock time spent in the sllsb package.
!
!  data is an a scalar variable of type SNLS_data_type used for
!   internal data.
!
!  userdata is a scalar variable of type USERDATA_type which may be
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
!  reverse is an OPTIONAL structure of type REVERSE_type which is used
!   to pass intermediate data to and from SNLS_solve. This will only be 
!   necessary if reverse-communication is to be used to form matrix-vector 
!   products of the form Jr * v, find columns of Jr or compute preconditioning 
!   steps of the form P^{-1} * v. If reverse is present (and eval_Jr_prod,
!   eval_Jr_scol, eval_Jr_sprod or eval_prec is absent), reverse communication 
!   will be used and the user must monitor the value of inform%status (see 
!   above) to await instructions about required  matrix-vector products.

!  eval_R is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the residual
!   function r(x) evaluated at x=X must be returned in R, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_R is not present, SNLS_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_Jr is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Jacobian
!   nabla_x r(x) of the residual function evaluated at x=X must be returned in
!   Jr_val in the same order as presented in nlp%Jr,, and the status variable
!   set to 0. If the evaluation is impossible at X, status should be set to a
!   nonzero value. If eval_Jr is not present, SNLS_solve will return to the
!   user with inform%status = 3 each time an evaluation is required.
!
!  eval_Jr_prod is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The product p = Jr(x) v,
!   (when transpose=.FALSE.) or p = Jr^T(x) v (when transpose=.TRUE.)
!   of the Jacobian (or its transpose) evaluated at x=X with the vector v=V
!   and the vector p must be returned in P, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_Jr_prod is not present, SNLS_solve will return to the user
!   with inform%status = 5 each time an evaluation is required. The Jacobian
!   has already been evaluated or used at x=X if got_jr is .TRUE.
!
!  eval_Jr_scol is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The index-th column of Jr
!   evaluated at x=X should be returned in VAL as a spare vector. Specifically,
!   the NONZEROS in the index-th column of Jr must be placed in their
!   appropriate comnponents of VAL, while a list of row indices of the
!   nonzeros placed in ROW( 1 : nz ). The status variable should 
!   be set to 0 unless the column is unavailable in which case status should 
!   be set to a nonzero value. If eval_Jr_scol is not present, SNLS_solve will 
!   either return to the user each time an evaluation is required 
!   (see reverse above) or form the product directly from user-provided nlp%Jr.
!   The Jacobian has already been evaluated or used at x=X if got_jr is .TRUE.
!
!  eval_Jr_sprod is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The product J(x) * v
!   (if transpose is .FALSE.) or Jr(x)^T v (if transpose is .TRUE.) involving
!   the given Jacobian J(x) evaluated at x=X and the vector v stored in V must 
!   be returned in P. If transpose is .FALSE., only the components of V with
!   indices FREE(:n_free) should be used, the remaining components should be
!   treated as zero. If transpose is .TRUE., all of V should be used, but
!   only the components P(IFREE(:nfree) need be computed, the remainder will
!   be ignored. The status variable should be set to 0 unless the product
!   is impossible in which case status should be set to a nonzero value.
!   If eval_Jr_sprod is not present, SNLS_solve will either return to the user
!   each time an evaluation is required (see reverse above) or form the
!   product directly from user-provided %Jr. The Jacobian has already been 
!   evaluated or used at x=X if got_jr is .TRUE.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( SNLS_control_type ), INTENT( IN ) :: control
     TYPE ( SNLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( SNLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_R, eval_Jr, eval_Jr_prod, eval_Jr_scol, eval_Jr_sprod

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_R( status, X, userdata, R )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
       END SUBROUTINE eval_R
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_Jr( status, X, userdata, Jr_val )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: Jr_val
       END SUBROUTINE eval_Jr
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_Jr_prod( status, X, userdata, transpose, V, P, got_jr )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
       END SUBROUTINE eval_Jr_prod
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_Jr_scol( status, X, userdata, index, VAL, ROW, nz,      &
                                got_jr )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: index
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: VAL
       INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: ROW
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: nz
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
       END SUBROUTINE eval_Jr_scol
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_Jr_sprod( status, X, userdata, transpose, V, P,         &
                                 FREE, n_free, got_jr )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
       END SUBROUTINE eval_Jr_sprod
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ic, ir, j, jr_ne, l
     REAL ( KIND = rp_ ) :: ared, prered, rounding, obj, val
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_facts, char_sit
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
     END IF

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 30 )  ! initial residual evaluation
       GO TO 30
     CASE ( 110 ) ! initial Jacobian evaluation or Jacobian transpose vect prod
       GO TO 110
     CASE ( 180 ) ! Jacobian vector product
       GO TO 180
     CASE ( 220 ) ! Jacobian vector product or Jacobian column
       GO TO 220
     CASE ( 320 ) ! residual evaluation
       GO TO 320
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
     data%set_printw = control%out > 0 .AND. control%print_level >= 4
     IF ( data%set_printw )                                                    &
       WRITE( control%out, "( A, ' statement 10' )" ) prefix

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 .OR. nlp%m_r <= 0 ) THEN
       IF ( control%error > 0 ) WRITE( control%error,                          &
         "( A, ' error: input m_r, n = ', I0, ', ', I0, ' not permitted' )" )  &
         prefix, nlp%m_r , nlp%n
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  check that the Jacobian is available in some form (may change in future)

     IF ( control%jacobian_available <= 0 ) THEN
       IF ( control%error > 0 ) WRITE( control%error,                          &
         "( A, ' error: Jacobian must be available in some form' )" ) prefix
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  see if W = I

     data%w_eq_identity = .NOT. ALLOCATED( nlp%W )
     IF ( .NOT. data%w_eq_identity ) THEN
       IF ( COUNT( nlp%W( : nlp%m_r ) <= zero ) > 0 ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input entries of W must be strictly positive' )" )   &
           prefix
         inform%status = GALAHAD_error_restrictions
         GO TO 990
       END IF
     END IF

!  record controls and ensure that data is consistent

     data%control = control
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%successful = .TRUE.
     data%ratio = - one
     data%total_facts = 0
     data%nskip_prec = nskip_prec_max
     data%reduce = .FALSE.

     inform%iter = 0 ; inform%inner_iter = 0
     inform%r_eval = 0 ; inform%jr_eval = 0

!  decide how much reverse communication is required

     data%reverse_r = .NOT. PRESENT( eval_R )

!  check to see if the Jacobian is available explicitly or only via its
!  action on a vector, and whether reverse communication will be required

     data%jacobian_available                                                   &
       = ALLOCATED( nlp%Jr%type ) .AND. data%control%jacobian_available >= 2
     IF ( data%jacobian_available ) THEN
       data%reverse_jr = .NOT. PRESENT( eval_Jr )
     ELSE
       data%reverse_jr = .FALSE.

!  check to see if other operations with Jr are provided

       data%reverse_jr_prod = .NOT. PRESENT( eval_Jr_prod )
       data%reverse_jr_scol = .NOT. PRESENT( eval_Jr_scol )
       data%reverse_jr_sprod = .NOT. PRESENT( eval_Jr_sprod )
       IF ( data%reverse_jr_prod .OR. data%reverse_jr_scol .OR.                &
            data%reverse_jr_sprod ) THEN
         IF ( .NOT. PRESENT( reverse ) ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: reverse must be present if',                       &
            &      ' eval_Jr_prod etc is absent' )" ) prefix
           inform%status = GALAHAD_error_optional
           GO TO 990
         END IF
         data%reverse_internal = .FALSE.
       ELSE
         data%reverse_internal = .TRUE.
       END IF
     END IF

!  solve the subproblem by projection if requested or if the Jacobian is only
!  accessible by products

     data%solve_projection = data%control%subproblem_solver == 1               &
       .OR. .NOT. data%jacobian_available

!  record the problem dimensions

     IF ( data%jacobian_available ) THEN
        nlp%Jr%m = nlp%m_r ; nlp%Jr%n = nlp%n
     END IF

!  check that the Jacobian is specified in a permitted format

     IF ( data%jacobian_available ) THEN
       SELECT CASE ( SMT_get( nlp%Jr%type ) )
       CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',                    &
              'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' )
       CASE DEFAULT
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: input J%type ', A, ' not permitted' )" )             &
             prefix, SMT_get( nlp%Jr%type )
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
         WRITE( control%alive_unit, "( ' GALAHAD rampages onwards' )" )
         CLOSE( control%alive_unit )
       END IF
     END IF

!  allocate sufficient space to solve the problem

     array_name = 'snls: nlp%G'
     CALL SPACE_resize_array( nlp%n, nlp%G, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: nlp%R'
     CALL SPACE_resize_array( nlp%m_r, nlp%R, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: nlp%Y'
     CALL SPACE_resize_array( nlp%m_c, nlp%Y, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980
     nlp%Y( : nlp%m_c ) = zero

     array_name = 'snls: nlp%Z'
     CALL SPACE_resize_array( nlp%n, nlp%Z, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980
     nlp%Z( : nlp%n ) = zero

     array_name = 'snls: nlp%X_status'
     CALL SPACE_resize_array( nlp%n, nlp%X_status, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980
     nlp%X_status( : nlp%n ) = 0

     array_name = 'snls: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%R_current'
     CALL SPACE_resize_array( nlp%m_r, data%R_current, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%G_current'
     CALL SPACE_resize_array( nlp%n, data%G_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%S'
     CALL SPACE_resize_array( nlp%n, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%Y'
     CALL SPACE_resize_array( nlp%m_r, data%Y, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( .NOT. data%monotone ) THEN
       array_name = 'snls: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'snls: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  allocate space to hold the simplex-constrained regularized linear 
!  least-squares subproblem

     data%GN_model%n = nlp%n ; data%GN_model%o = nlp%m_r
     data%GN_model%m = nlp%m_c

     array_name = 'snls: data%GN_model%B'
     CALL SPACE_resize_array( nlp%m_r, data%GN_model%B,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%X'
     CALL SPACE_resize_array( nlp%n, data%GN_model%X,                          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%Y'
     CALL SPACE_resize_array( data%GN_model%m, data%GN_model%Y,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%Z'
     CALL SPACE_resize_array( nlp%n, data%GN_model%Z,                          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%R'
     CALL SPACE_resize_array( nlp%m_r, data%GN_model%R,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%G'
     CALL SPACE_resize_array( nlp%n, data%GN_model%G,                          &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%X_status'
     CALL SPACE_resize_array( nlp%n, data%GN_model%X_status,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'snls: data%GN_model%X_s'
     CALL SPACE_resize_array( nlp%n, data%GN_model%X_s,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
          bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= 0 ) GO TO 980                      
                                                                        
!  save the weights if they are present

     IF ( ALLOCATED( nlp%W ) ) THEN
       array_name = 'snls: data%GN_model%W'
       CALL SPACE_resize_array( nlp%m_r, data%GN_model%W,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
       data%GN_model%W( : nlp%m_r ) = nlp%W( : nlp%m_r )
     END IF

!  set Ao appropriately in the least-squares storage format

     IF ( data%jacobian_available ) THEN

!  make space for Jr%val

       SELECT CASE( SMT_get( nlp%Jr%type ) )
       CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
         nlp%Jr%ne = nlp%m_r * nlp%n
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%Jr%ne = nlp%Jr%ptr( nlp%m_r + 1 ) - 1
       CASE ( 'SPARSE_BY_COLUMNS' )
         nlp%Jr%ne = nlp%Jr%ptr( nlp%n + 1 ) - 1
!      CASE ( 'COORDINATE' )
!        nlp%Jr%ne = nlp%Jr%ne
       END SELECT

       array_name = 'snls: nlp%J%val'
       CALL SPACE_resize_array( nlp%Jr%ne, nlp%Jr%val,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  make space for data%GN_model%Ao

       SELECT CASE ( SMT_get( nlp%Jr%type ) )
       CASE ( 'coordinate', 'COORDINATE' )
         IF ( .NOT. ( ALLOCATED( nlp%Jr%row ) .AND.                            &
                      ALLOCATED( nlp%Jr%col ) ) ) THEN
           inform%status = GALAHAD_error_optional
           GO TO 990
         END IF
         CALL SMT_put( data%GN_model%Ao%type, 'COORDINATE',                    &
                       inform%alloc_status )
         data%GN_model%Ao%n = nlp%n ; data%GN_model%Ao%m = nlp%m_r
         data%GN_model%Ao%ne = nlp%Jr%ne

         array_name = 'snls: data%GN_model%Ao%row'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%row,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%col'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%col,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%val'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%val,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         data%GN_model%Ao%row( : data%GN_model%Ao%ne )                         &
           = nlp%Jr%row( : data%GN_model%Ao%ne )
         data%GN_model%Ao%col( : data%GN_model%Ao%ne )                         &
           = nlp%Jr%col( : data%GN_model%Ao%ne )

       CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
         IF ( .NOT. ( ALLOCATED( nlp%Jr%ptr ) .AND.                            &
                      ALLOCATED( nlp%Jr%col ) ) ) THEN
           inform%status = GALAHAD_error_optional
           GO TO 990
         END IF
         CALL SMT_put( data%GN_model%Ao%type, 'SPARSE_BY_ROWS',                &
                       inform%alloc_status )
         data%GN_model%Ao%n = nlp%n ; data%GN_model%Ao%m = nlp%m_r
         data%GN_model%Ao%ne = nlp%Jr%ptr( nlp%m_r + 1 ) - 1

         array_name = 'snls: data%GN_model%Ao%ptr'
         CALL SPACE_resize_array( nlp%m_r + 1, data%GN_model%Ao%ptr,           &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%col'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%col,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%val'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%val,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

           data%GN_model%Ao%ptr( : nlp%m_r + 1 ) = nlp%Jr%ptr( : nlp%m_r + 1 )
           data%GN_model%Ao%col( : data%GN_model%Ao%ne )                       &
             = nlp%Jr%col( : data%GN_model%Ao%ne )

       CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
         IF ( .NOT. ( ALLOCATED( nlp%Jr%ptr ) .AND.                            &
                      ALLOCATED( nlp%Jr%row ) ) ) THEN
           inform%status = GALAHAD_error_optional
           GO TO 990
         END IF
         CALL SMT_put( data%GN_model%Ao%type, 'SPARSE_BY_COLUMNS',             &
                       inform%alloc_status )
         data%GN_model%Ao%n = nlp%n ; data%GN_model%Ao%m = nlp%m_r
         data%GN_model%Ao%ne = nlp%Jr%ptr( nlp%n + 1 ) - 1
         array_name = 'snls: data%GN_model%Ao%ptr'
         CALL SPACE_resize_array( nlp%n + 1, data%GN_model%Ao%ptr,             &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%row'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%row,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'snls: data%GN_model%Ao%val'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%val,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         data%GN_model%Ao%ptr( : nlp%n + 1 ) = nlp%Jr%ptr( : nlp%n + 1 )
         data%GN_model%Ao%row( : data%GN_model%Ao%ne )                         &
           = nlp%Jr%row( : data%GN_model%Ao%ne )

       CASE ( 'dense_by_rows', 'DENSE_BY_ROWS' )
         CALL SMT_put( data%GN_model%Ao%type, 'DENSE_BY_ROWS',                 &
                       inform%alloc_status )
         data%GN_model%Ao%n = nlp%n ; data%GN_model%Ao%m = nlp%m_r
         data%GN_model%Ao%ne = nlp%m_r * nlp%n

         array_name = 'snls: data%GN_model%Ao%val'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%val,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

       CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
         CALL SMT_put( data%GN_model%Ao%type, 'DENSE_BY_COLUMNS',              &
                       inform%alloc_status )
         data%GN_model%Ao%n = nlp%n ; data%GN_model%Ao%m = nlp%m_r
         data%GN_model%Ao%ne = nlp%m_r * nlp%n

         array_name = 'snls: data%GN_model%Ao%val'
         CALL SPACE_resize_array( data%GN_model%Ao%ne, data%GN_model%Ao%val,   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

       CASE DEFAULT
         inform%status = GALAHAD_error_unknown_storage
         GO TO 990
       END SELECT

!  allocate space for components of reverse, if needed

     ELSE
       IF ( data%reverse_internal ) THEN
         array_name = 'snls: data%reverse%iv'
         CALL SPACE_resize_array( nlp%n, data%reverse%iv, inform%status,       &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: data%reverse%ip'
         CALL SPACE_resize_array( nlp%m_r, data%reverse%ip, inform%status,     &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: data%reverse%v'
         CALL SPACE_resize_array( MAX( nlp%m_r, nlp%n ), data%reverse%v,       &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: data%reverse%p'
         CALL SPACE_resize_array( MAX( nlp%m_r, nlp%n ), data%reverse%p,       &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980
       ELSE
         array_name = 'snls: reverse%iv'
         CALL SPACE_resize_array( nlp%n, reverse%iv, inform%status,            &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: reverse%ip'
         CALL SPACE_resize_array( nlp%m_r, reverse%ip, inform%status,          &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: reverse%v'
         CALL SPACE_resize_array( MAX( nlp%m_r, nlp%n ), reverse%v,            &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980

         array_name = 'snls: reverse%p'
         CALL SPACE_resize_array( MAX( nlp%m_r, nlp%n ), reverse%p,              &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 980
       END IF
     END IF

!  save the cohorts if they are available

     data%multiple_simplices = ALLOCATED( nlp%COHORT )
     IF ( data%multiple_simplices ) THEN
       array_name = 'snls: data%SLLS_data%S_ptr'
       CALL SPACE_resize_array( nlp%m_c + 1, data%SLLS_data%S_ptr,             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 980

       data%SLLS_data%S_ptr( 1 : nlp%m_c ) = 0
       DO j = 1, nlp%n
         i = nlp%COHORT( j )
         IF ( i > nlp%n .OR. i < 0 ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
           "( A, ' error: cohort[', I0, '] = ', I0, ' not in [0,', I0, ']' )" )&
               prefix, j, i, nlp%n
           inform%status = GALAHAD_error_restrictions ; GO TO 990
         ELSE IF ( i > 0 ) THEN
           data%SLLS_data%S_ptr( i ) = data%SLLS_data%S_ptr( i ) + 1
         END IF
       END DO
       IF ( MINVAL( data%SLLS_data%S_ptr( 1 : nlp%m_c ) ) == 0 ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 990
       END IF

!  if there are multiple simplices, make a continguous list of variables 
!  (the cohort) for each simplex. That is assign
!
!    cohort     0     1     2    ...    m   
!    S_ind:  | ... | ... | ... | ... | ... |
!                   ^     ^     ^     ^     ^
!    S_ptr:         |     |     |     |     |

!  allocate further workspace arrays

       array_name = 'snls: data%SLLS_data%S_ind'
       CALL SPACE_resize_array( nlp%n, data%SLLS_data%S_ind, inform%status,    &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 980

!  next compute how many variables each simplex uses (temporarily in S_ptr)

       data%SLLS_data%S_ptr( 1 : nlp%m_c ) = 0
       l = 0
       DO i = 1, nlp%n
         j = nlp%COHORT( i )
         IF ( j > 0 ) THEN
           data%SLLS_data%S_ptr( j ) = data%SLLS_data%S_ptr( j ) + 1
         ELSE
           l = l + 1
         END IF
       END DO
!      WRITE( 6, "( ' size cohorts', 5( ' ', I0 ) )" ) data%SLLS_data%S_ptr

!  record the maximum number

       data%SLLS_data%n_c = MAXVAL( data%SLLS_data%S_ptr( 1 : nlp%m_c ) )

!  now assign the starting address for each cohort in S_ptr

       l = l + 1
       DO i = 1, nlp%m_c
         j = data%SLLS_data%S_ptr( i )
         data%SLLS_data%S_ptr( i ) = l
         l = l + j
       END DO
 !     WRITE( 6, "( ' starts cohorts', 5( ' ', I0 ) )" ) data%SLLS_data%S_ptr

!  next, put the variables for each cohort in adjacent positions in S_val
!  and adjust S_ptr accordingly so that the last variable in cohort j is
!  in position S_ptr(j), with any remaining variables at the start of S_val

       l = 0
       DO i = 1, nlp%n
         j = nlp%COHORT( i )
         IF ( j > 0 ) THEN
           data%SLLS_data%S_ind( data%SLLS_data%S_ptr( j ) ) = i
           data%SLLS_data%S_ptr( j ) = data%SLLS_data%S_ptr( j ) + 1
         ELSE
           l = l + 1
           data%SLLS_data%S_ind( l ) = i
         END IF 
       END DO

!  finally, recover the starting address S_ptr for each cohort (and
!  set S_ptr( m + 1 ) to be one beyond n

       data%SLLS_data%S_ptr( nlp%m_c + 1 ) = nlp%n + 1
       DO i = nlp%m_c, 2, - 1
         data%SLLS_data%S_ptr( i ) = data%SLLS_data%S_ptr( i - 1 )
       END DO
       data%SLLS_data%S_ptr( 1 ) = l + 1
!      WRITE( 6, "( ' start cohorts', 5( ' ', I0 ) )" ) data%SLLS_data%S_ptr
!      DO i = 1, nlp%m_c
!       WRITE( 6, "( ' cohort ', I0, ':', 5( ' ', I0 ) )" ) &
!         i, ( data%SLLS_data%S_ind( j ), j = data%SLLS_data%S_ptr( i ), 
!         data%SLLS_data%S_ptr( i + 1 ) - 1 )
!      END DO
!      IF ( data%SLLS_data%S_ptr( 1 ) > 1 ) &
!        WRITE( 6, "( ' cohort 0:', 5( ' ', I0 ) )" ) &
!         ( data%SLLS_data%S_ind( j ), j = 1, data%SLLS_data%S_ptr( 1 ) - 1 )

!  allocate space for the sub-projections into each simplex

       array_name = 'snls: data%SLLS_data%X_c'
       CALL SPACE_resize_array( data%SLLS_data%n_c, data%SLLS_data%X_c,        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 980

       array_name = 'snls: data%SLLS_data%X_c_proj'
       CALL SPACE_resize_array( data%SLLS_data%n_c, data%SLLS_data%X_c_proj,   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 980

!  save the cohorts if they are available

       array_name = 'snls: data%GN_model%COHORT'
       CALL SPACE_resize_array( nlp%n, data%GN_model%COHORT,                   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980
       
       data%GN_model%COHORT( : nlp%n ) = MAX( nlp%COHORT( : nlp%n ), 0 )

!  check that input estimate of the solution is in the intersection of 
!  simplices, and if not project it so that it is

       CALL SLLS_project_onto_simplices( nlp%n, nlp%m_c, data%SLLS_data%n_c,   &
                                         data%SLLS_data%S_ptr,                 &
                                         data%SLLS_data%S_ind, nlp%X,          &
                                         data%X_current, data%SLLS_data%X_c,   &
                                         data%SLLS_data%X_c_proj, i )
     ELSE
       data%GN_model%m = 1
       CALL SLLS_project_onto_simplex( nlp%n, nlp%X, data%X_current, i )
     END IF

!  check that the projection succeeded

    IF ( i < 0 ) THEN
       inform%status = GALAHAD_error_sort
       GO TO 990
     ELSE IF ( i > 0 ) THEN
       nlp%X( : nlp%n ) = data%X_current( : nlp%n )
       IF ( data%printi ) WRITE( control%out,                                  &
       "( ' ', /, A, '   **  Warning: input point projected onto simplex' )" ) &
         prefix
     END IF

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
     data%print_level_slls = data%control%SLLS_control%print_level
     data%print_level_sllsb = data%control%SLLSB_control%print_level
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
     data%minimum_weight = data%control%minimum_weight
     data%got_jr = .FALSE.

! evaluate the residual function r(x) at the initial point

     IF ( data%reverse_r ) THEN
       data%branch = 30 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_R( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%R( : nlp%m_r ) )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_R'
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: reported eval_R evaluation' )" ) prefix
         inform%status = GALAHAD_error_evaluation ; GO TO 990
       END IF
     END IF

!  return from reverse communication with the residual function value r(x)

  30 CONTINUE
!    CALL CLOCK_time( data%clock_now )
!    write(6,*) ' 30 elapsed', data%clock_now - data%clock_start
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 30' )" ) prefix
     inform%r_eval = inform%r_eval + 1

     IF ( data%reverse_r ) THEN
       IF ( reverse%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_R'
         inform%status = GALAHAD_error_evaluation ; GO TO 990
       END IF
     END IF

     IF ( data%w_eq_identity ) THEN
       data%Y( : nlp%m_r ) = nlp%R( : nlp%m_r )
       inform%norm_r = TWO_NORM( nlp%R( : nlp%m_r ) )
       inform%obj = half * inform%norm_r ** 2
     ELSE
       data%Y( : nlp%m_r ) = nlp%W( : nlp%m_r ) * nlp%R( : nlp%m_r )
       val = DOT_PRODUCT( data%Y( : nlp%m_r ), nlp%R( : nlp%m_r ) )
       inform%norm_r = SQRT( val )
       inform%obj = half * val
     END IF

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%bad_eval = 'NaN'
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  compute the residual stopping tolerance

     data%stop_r = MAX( MAX( control%stop_r_absolute, zero ),                  &
                        MAX( control%stop_r_relative, zero ) * inform%norm_r,  &
                        epsmch )

!  stop in the unlikely event that the initial residual is already small

     IF ( inform%norm_r <= data%stop_r ) THEN
       inform%status = GALAHAD_ok ; GO TO 900
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

!  evaluate the Jacobian Jr(x) of r(x)

       IF ( .NOT. data%poor_model ) THEN
         IF ( data%jacobian_available ) THEN
           IF ( data%reverse_jr ) THEN
             data%branch = 110 ; inform%status = 3 ; RETURN
           ELSE
             CALL eval_Jr( data%eval_status, nlp%X( : nlp%n ), userdata,       &
                           nlp%Jr%val )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
           END IF

!  otherwise evaluate the product g = Jr^T(x) W r(x)

         ELSE
           IF ( data%reverse_jr_prod ) THEN
             reverse%V( : nlp%m_r ) = data%Y( : nlp%m_r )
             reverse%transpose = .TRUE.
             data%branch = 110 ; inform%status = 5 ; RETURN
           ELSE
             CALL eval_Jr_prod( data%eval_status, nlp%X( : nlp%n ), userdata,  &
                                .TRUE., data%Y( : nlp%m_r ),                   &
                                nlp%G( : nlp%n ), .FALSE. )
             IF ( data%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr_prod'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
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
         inform%jr_eval = inform%jr_eval + 1

         IF ( data%jacobian_available ) THEN
           IF ( data%reverse_jr ) THEN
             IF ( reverse%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
           END IF
         ELSE
           IF ( data%reverse_jr_prod ) THEN
             IF ( reverse%eval_status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr_prod'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
           END IF
         END IF

!  compute the product g = J^T(x) W r(x) from J(x) if necessary

         IF ( data%jacobian_available ) THEN
           CALL mop_Ax( one, nlp%Jr, data%Y( : nlp%m_r ), zero,                &
                        nlp%G( : nlp%n ), out = data%out,                      &
                        error = data%control%error, print_level = 0_ip_,       &
                        transpose = .TRUE. )
         ELSE
           IF ( data%reverse_jr_prod ) nlp%G( : nlp%n ) = reverse%P( : nlp%n )
         END IF

!  compute the gradient of ||g(x)||

         data%g_norm = TWO_NORM( nlp%G( : nlp%n ) )
         IF ( inform%norm_r > zero ) THEN
           inform%norm_g = data%g_norm / inform%norm_r
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
             nlp%R( : nlp%m_r ) = data%R_current( : nlp%m_r )

!  control printing for the NaN case

             IF ( inform%iter >= data%start_print .AND.                        &
                  inform%iter < data%stop_print .AND.                          &
                  MOD( inform%iter + 1 - data%start_print, data%print_gap )    &
                    == 0 ) THEN
               data%printi = data%set_printi ; data%printt = data%set_printt
               data%printm = data%set_printm ; data%printw = data%set_printw
               data%printd = data%set_printd
               data%print_level = data%control%print_level
               data%control%SLLS_control%print_level = data%print_level_slls
               data%control%SLLSB_control%print_level = data%print_level_sllsb
             ELSE
               data%printi = .FALSE. ; data%printt = .FALSE.
               data%printm = .FALSE. ; data%printw = .FALSE.
               data%printd = .FALSE.
               data%print_level = 0
               data%control%SLLS_control%print_level = 0
               data%control%SLLSB_control%print_level = 0
             END IF
             data%print_iteration_header = data%print_level > 1 .OR.           &
               ( data%control%SLLS_control%print_level > 0 .AND.               &
                 data%solve_projection ) .OR.                                  &
               ( data%control%SLLSB_control%print_level > 0 .AND. .NOT.        &
                 data%solve_projection )

!  print one-line summary

             IF ( data%printi ) THEN
                IF ( data%print_iteration_header .OR.                          &
                     data%print_1st_header ) THEN
                 WRITE( data%out, 2090 ) prefix
                 IF ( data%solve_projection ) THEN
                   IF ( data%control%print_obj ) THEN
                     WRITE( data%out, 2130 ) prefix
                   ELSE
                     WRITE( data%out, 2120 ) prefix
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
               IF ( data%solve_projection ) THEN
                 char_facts = ADJUSTR( STRING_integer_6( data%total_facts ) )
                 WRITE( data%out,  "( A, A6, 1X, 2A1, ES11.4, '    NaN    ',   &
                &  ES9.1,  2ES8.1, 1X, A6, F8.2 )" )                           &
                    prefix, char_iter, data%accept,  data%hard,                &
                    inform%norm_r, data%ratio, data%old_weight, data%s_norm,   &
                    char_facts, data%clock_now
               ELSE
                 char_sit =                                                    &
                    ADJUSTR( STRING_integer_6( inform%SLLS_inform%iter ) )
                 WRITE( data%out, "( A, A6, 1X, 2A1, ES11.4, '    NaN    ',    &
                &  ES9.1, 2ES8.1, 1X, A6, F8.2 )" ) prefix,                    &
                    char_iter, data%accept, data%perturb,                      &
                    inform%norm_r, data%ratio, data%old_weight, data%s_norm,   &
                    char_sit, data%clock_now
               END IF
             END IF
             inform%obj = data%obj_current
             inform%norm_r = data%norm_R_current

!  check to see if we are still "alive"

             IF ( data%control%alive_unit > 0 ) THEN
               INQUIRE( FILE = data%control%alive_file, EXIST = alive )
               IF ( .NOT. alive ) THEN
                 IF ( control%error > 0 ) WRITE( control%error,                &
                  "( A, ' error: alive file removed' )" ) prefix
                 inform%status = GALAHAD_error_alive ; GO TO 990
               END IF
             END IF

!  check to see if the iteration limit has been exceeded

             inform%iter = inform%iter + 1
             IF ( inform%iter > data%control%maxit .AND.                       &
                  data%step_accepted ) THEN
               IF ( control%error > 0 ) WRITE( control%error,                  &
                "( A, ' error: iteration limit exceeded' )" ) prefix
               inform%status = GALAHAD_error_max_iterations ; GO TO 990
             END IF

!  increase the regularization weight and try again

             inform%weight = data%control%weight_increase * data%old_weight
             GO TO 100
           ELSE
             IF ( data%printi ) WRITE( data%out,                               &
                "( A, ' initial gradient value is a NaN' )" ) prefix
             inform%bad_eval = 'eval_Jr_prod'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
         END IF

!  compute the norm of the projected gradient

         val = MIN( one, one / TWO_NORM( nlp%G( : nlp%n ) ) )
         data%S = nlp%X - val * nlp%G( : nlp%n )
         IF ( data%multiple_simplices ) THEN
           CALL SLLS_project_onto_simplices( nlp%n, nlp%m_c,                   &
                                             data%SLLS_data%n_c,               &
                                             data%SLLS_data%S_ptr,             &
                                             data%SLLS_data%S_ind, data%S,     &
                                             data%X_current,                   &
                                             data%SLLS_data%X_c,               &
                                             data%SLLS_data%X_c_proj, i )
         ELSE
           CALL SLLS_project_onto_simplex( nlp%n, data%S, data%X_current, i )
         END IF
         inform%norm_pg = MAXVAL( ABS( data%X_current - nlp%X ) )

!  reset the initial weight to ||g|| if no sensible value is given

         IF ( inform%iter == 0 ) THEN
           IF ( data%control%initial_weight <= zero )                          &
              inform%weight = one / inform%norm_g

!  compute the gradient stopping tolerance

           data%stop_pg = MAX( MAX( control%stop_pg_absolute, zero ),          &
             MAX( control%stop_pg_relative, zero ) * inform%norm_pg, epsmch )
           data%stop_pg_switch = MAX( MAX( control%stop_pg_absolute, zero ),   &
             MAX( control%stop_pg_switch, zero ) * inform%norm_pg, epsmch )

           IF ( data%printi ) THEN
             WRITE( data%out, "( A, '  Problem: ', A, ' (n = ', I0,            &
            &  ', m_r = ', I0, ', m_c = ', I0, ')', /, A,                      &
            &   '  SNLS stopping tolerances (r,P[-J''r]) =', 2ES9.2, / )" )    &
               prefix, TRIM( nlp%pname ), nlp%n, nlp%m_r, nlp%m_c,             &
               prefix, data%stop_r, data%stop_pg
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
         data%control%SLLS_control%print_level = data%print_level_slls
         data%control%SLLSB_control%print_level = data%print_level_sllsb
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%SLLS_control%print_level = 0
         data%control%SLLSB_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         ( data%control%SLLS_control%print_level > 0 .AND.                     &
           data%solve_projection ) .OR.                                        &
         ( data%control%SLLSB_control%print_level > 0 .AND. .NOT.              &
           data%solve_projection )

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2090 ) prefix
           IF ( data%solve_projection ) THEN
             IF ( data%control%print_obj ) THEN
               WRITE( data%out, 2130 ) prefix
             ELSE
               WRITE( data%out, 2120 ) prefix
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
         CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( data%control%print_obj ) THEN
           obj = inform%obj
         ELSE
           obj = inform%norm_r
         END IF
         IF ( inform%iter > 0 ) THEN
           IF ( data%solve_projection ) THEN
             char_sit = ADJUSTR( STRING_integer_6( inform%SLLS_inform%iter ) )
           ELSE
             char_sit = ADJUSTR( STRING_integer_6( inform%SLLSB_inform%iter ) )
           END IF
           WRITE( data%out, 2140 ) prefix, char_iter, data%accept,             &
              data%perturb, obj, inform%norm_pg, data%ratio, data%old_weight,  &
              data%s_norm, char_sit, data%clock_now
         ELSE
           WRITE( data%out, 2150 ) prefix,                                     &
                  char_iter, inform%norm_r, inform%norm_pg
         END IF
       END IF

!  ============================================================================
!  1. Test for convergence
!  ============================================================================

!  stop if the gradient is small enough

       IF ( inform%norm_r <= data%stop_r .OR.                                  &
            inform%norm_pg <= data%stop_pg ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  switch from an interior-point to a projection iteration

       IF ( control%subproblem_solver == 3 .AND. .NOT. data%solve_projection   &
            .AND. inform%norm_pg <= data%stop_pg_switch ) THEN
         data%solve_projection = .TRUE.
         IF ( data%printi ) THEN
           IF ( data%control%print_obj ) THEN
             WRITE( data%out, 2130 ) prefix
           ELSE
             WRITE( data%out, 2120 ) prefix
           END IF
         END IF
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: alive file removed' )" ) prefix
           inform%status = GALAHAD_error_alive ; GO TO 990
         END IF
       END IF

!  check to see if the iteration limit has been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit ) THEN
        IF ( control%error > 0 ) WRITE( control%error,                         &
          "( A, ' error: iteration limit exceeded' )" ) prefix
         inform%status = GALAHAD_error_max_iterations ; GO TO 990
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%r_eval, prefix,               &
           inform%jr_eval, prefix, inform%iter, prefix, inform%inner_iter,     &
           prefix, inform%obj, prefix, inform%norm_pg
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

!  store the values of the Jacobian, J_k, and the observations b = J_k x_k - r_k

       IF ( data%jacobian_available ) THEN
         data%GN_model%Ao%val = nlp%Jr%val
         data%GN_model%B = - nlp%R
         CALL mop_Ax( one, nlp%Jr, nlp%X( : nlp%n ), one,                      &
                      data%GN_model%B( : nlp%m_r ), out = data%out,            &
                      error = data%control%error, print_level = 0_ip_,         &
                      transpose = .FALSE. )
       ELSE
         IF ( data%reverse_jr_prod ) THEN
           reverse%v( : nlp%n ) = nlp%X( : nlp%n )
           reverse%transpose = .FALSE.
           inform%status = 4 ; data%branch = 180 ; RETURN
         ELSE
           CALL eval_Jr_prod( data%eval_status, nlp%X, userdata,               &
                              .FALSE., nlp%X, data%GN_model%B, data%got_jr )
           IF ( data%eval_status /= 0 ) THEN
             inform%bad_eval = 'eval_Jr_prod'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
           data%GN_model%B = data%GN_model%B - nlp%R
           data%got_jr = .TRUE.
         END IF
       END IF

!  return from reverse communication with the Jacobian-vector product

   180 CONTINUE
       IF ( .NOT. data%jacobian_available .AND. data%reverse_jr_prod ) THEN
         IF ( reverse%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_Jr_prod'
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF
         data%GN_model%B( : nlp%m_r ) = reverse%p( : nlp%m_r ) - nlp%R( : nlp%m_r )
       END IF

!  store the regularization weight 

       data%GN_model%regularization_weight = inform%weight

!  store initial guesses for x, y and z

       data%GN_model%X = nlp%X ; data%GN_model%Y = zero ; data%GN_model%Z = zero

!  store the shift x_s

       data%GN_model%X_s = nlp%X

!  ============================================================================
!  2. Calculate the search direction, s_k
!  ============================================================================

!  find s_k to (approximately) solve the sub-problem

!   s_k = arg min 1/2|| J_k s + r_k ||_W^2 + 1/2 weight ||s||^2 
!            s
!         s.t. x_k + s in C(x_k + s)

!  or (more usefully)

!   x_k^+ = arg min 1/2 || J_k x + r_k - J_k x_k ||_W^2 +
!                   1/2 weight || x - x_k ||^2
!              x
!           s.t. x in C(x)

   190 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 190' )" ) prefix
!      write(6,*) ' f ', inform%obj

!  2a. solution using projection with available Jr
!  -----------------------------------------------

       IF ( data%solve_projection ) THEN
         IF ( data%jacobian_available ) THEN
           inform%SLLS_inform%status = 1
           data%control%SLLS_control%stop_d                                    &
             = MIN( control%SLLS_control%stop_d, point1 * inform%norm_pg )
           CALL CPU_TIME( data%time_record )
           CALL CLOCK_time( data%clock_record )
           CALL SLLS_solve( data%GN_model, data%SLLS_data,                     &
                            data%control%SLLS_control, inform%SLLS_inform,     &
                            userdata )
           CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
           inform%time%slls                                                    &
             = inform%time%slls + data%time_now - data%time_record
           inform%time%clock_slls                                              &
             = inform%time%clock_slls + data%clock_now - data%clock_record
           inform%inner_iter = inform%inner_iter + inform%SLLS_inform%iter
           data%model = inform%SLLS_inform%ls_obj
           GO TO 300
         END IF

!  2b. solution using an interior-point method
!  -------------------------------------------

       ELSE
         inform%SLLSB_inform%status = 1
         CALL CPU_TIME( data%time_record )
         CALL CLOCK_time( data%clock_record )
         CALL SLLSB_solve( data%GN_model, data%SLLSB_data,                     &
                           data%control%SLLSB_control, inform%SLLSB_inform )
         CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         inform%time%sllsb                                                     &
           = inform%time%sllsb + data%time_now - data%time_record
         inform%time%clock_sllsb                                               &
           = inform%time%clock_sllsb + data%clock_now - data%clock_record
         inform%inner_iter = inform%inner_iter + inform%SLLSB_inform%iter
         data%model = inform%SLLSB_inform%ls_obj
         GO TO 300
       END IF

!  2c. solution using projection with suitable products with Jr
!  ------------------------------------------------------------

       inform%SLLS_inform%status = 1
       data%control%SLLS_control%stop_d                                        &
         = MIN( control%SLLS_control%stop_d, point1 * inform%norm_pg )
       CALL CPU_TIME( data%time_record ) ; CALL CLOCK_time( data%clock_record )
       IF ( data%reverse_internal ) GO TO 230

!  solve problem with a reverse commmunication loop

 210   CONTINUE
         CALL SLLS_solve( data%GN_model, data%SLLS_data,                       &
                          data%control%SLLS_control, inform%SLLS_inform,       &
                          userdata, reverse = reverse )

         SELECT CASE ( inform%SLLS_inform%status )

!  termination return

         CASE ( : 0 )
           data%model = inform%SLLS_inform%ls_obj
           GO TO 260

!  compute Jr * v or Jr^T * v

         CASE ( 2, 3 ) 
           IF ( data%reverse_jr_prod ) THEN
             inform%status = inform%SLLS_inform%status + 2
             data%branch = 220 ; RETURN
           ELSE
             CALL eval_Jr_prod( inform%status, nlp%X, userdata,                &
                                reverse%transpose, reverse%V, reverse%P,       &
                                data%got_jr )
             IF ( inform%status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr_prod'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
             data%got_jr = .TRUE.
           END IF

!  compute the index-th column of Jr

         CASE ( 4 )
           IF ( data%reverse_jr_scol ) THEN
             inform%status = inform%SLLS_inform%status + 2
             data%branch = 220 ; RETURN
           ELSE
             CALL eval_Jr_scol( inform%status, nlp%X, userdata, reverse%index, &
                                reverse%P, reverse%IP, reverse%lp, data%got_jr )
             IF ( inform%status /= 0 ) THEN
               inform%bad_eval = 'eval_Jr_scol'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
             data%got_jr = .TRUE.
           END IF

 !  compute Jr * sparse v or sparse( Jr^T * v )

         CASE ( 5, 6 )
           IF ( data%reverse_jr_sprod ) THEN
             inform%status = inform%SLLS_inform%status + 2
             data%branch = 220 ; RETURN
           ELSE
             CALL eval_Jr_sprod( inform%status, nlp%X, userdata,               &
                                 reverse%transpose, reverse%V, reverse%P,      &
                                 reverse%IV, reverse%lvu, data%got_jr )
             IF ( inform%status /= 0 ) THEN
               inform%bad_eval = 'evalJr_sprod'
               inform%status = GALAHAD_error_evaluation ; GO TO 990
             END IF
             data%got_jr = .TRUE.
           END IF

!  error returns

         CASE DEFAULT
           IF ( data%printt ) WRITE( data%out, "( /,                           &
           &  A, ' Error return from SLLS, status = ', I0 )" ) prefix,         &
             inform%SLLS_inform%status
           inform%status = inform%SLLS_inform%status
           GO TO 990
         END SELECT

!  check that the evaluation succeeded

         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF
         GO TO 210

!  return from reverse communication with the Jacobian-vector product

   220   CONTINUE
         IF ( data%printw ) WRITE( data%out, "( A, ' statement 220, SLLS_',    &
        &  'inform%status = ', I0 )" ) prefix, inform%SLLS_inform%status

         IF ( reverse%eval_status /= 0 ) THEN
           inform%bad_eval = 'evalJr_prods'
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF

!  end of the reverse commmunication loop

       GO TO 210

!  solve problem with an internal reverse commmunication loop with

 230   CONTINUE

!write(6,*) ' slls in  ', inform%SLLS_inform%status
         CALL SLLS_solve( data%GN_model, data%SLLS_data,                       &
                          data%control%SLLS_control, inform%SLLS_inform,       &
                          userdata, reverse = data%reverse )
!write(6,*) ' slls out ', inform%SLLS_inform%status

         SELECT CASE ( inform%SLLS_inform%status )

!  termination return

         CASE ( : 0 )
           data%model = inform%SLLS_inform%ls_obj
           GO TO 260

!  compute Jr * v or Jr^T * v

         CASE ( 2, 3 ) 
           CALL eval_Jr_prod( inform%status, nlp%X, userdata,                  &
                              data%reverse%transpose, data%reverse%V,          &
                              data%reverse%P, data%got_jr )
           IF ( inform%status /= 0 ) THEN
             inform%bad_eval = 'eval_Jr_prod'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
           data%got_jr = .TRUE.

!  compute the index-th column of Jr

         CASE ( 4 )
           CALL eval_Jr_scol( inform%status, nlp%X, userdata,                  &
                              data%reverse%index, data%reverse%P,              &
                              data%reverse%IP, data%reverse%lp, data%got_jr )
           IF ( inform%status /= 0 ) THEN
             inform%bad_eval = 'eval_Jr_scol'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
           data%got_jr = .TRUE.

 !  compute Jr * sparse v or sparse( Jr^T * v )

         CASE ( 5, 6 )
           CALL eval_Jr_sprod( inform%status, nlp%X, userdata,                 &
                               data%reverse%transpose, data%reverse%V,         &
                               data%reverse%P, data%reverse%IV,                &
                               data%reverse%lvu, data%got_jr )
           IF ( inform%status /= 0 ) THEN
             inform%bad_eval = 'evalJr_sprod'
             inform%status = GALAHAD_error_evaluation ; GO TO 990
           END IF
           data%got_jr = .TRUE.

!  error returns

         CASE DEFAULT
           IF ( data%printt ) WRITE( data%out, "( /,                           &
           &  A, ' Error return from SLLS, status = ', I0 )" ) prefix,         &
             inform%SLLS_inform%status
           inform%status = inform%SLLS_inform%status
           GO TO 990
         END SELECT

!  check that the evaluation succeeded

         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF
         GO TO 230

!  end of the reverse commmunication loop

       GO TO 230

!  end of the solution 2c iteration
!  ................................

   260 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 260' )" ) prefix
       CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       inform%time%slls = inform%time%slls + data%time_now - data%time_record
       inform%time%clock_slls                                                  &
         = inform%time%clock_slls + data%clock_now - data%clock_record

!      Record the total number of Lanczos iterations

       inform%inner_iter = inform%inner_iter + inform%SLLS_inform%iter
       IF ( data%printt ) WRITE( data%out,                                     &
          "( /, A, ' CG iterations required = ', I8 )" )                       &
            prefix, inform%SLLS_inform%iter

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

!  recover the step

       data%S( : nlp%n ) = data%GN_model%X( : nlp%n ) - nlp%X( : nlp%n )
!write(6,"( ' s = ', 2ES12.4 )" ) data%S( : nlp%n )
       data%s_norm = TWO_NORM( data%S( : nlp%n ) )
!      WRITE( 6, "( ' s_norm ', ES12.4 )" ) data%s_norm
!      WRITE( 6, "( ' ||s|| = ', ES12.4 )" ) MAXVAL( ABS( data%S( : nlp%n ) ) )

!  see if the correction will make any difference

       IF ( MAXVAL( ABS( data%S( : nlp%n ) ) / MAX( one, nlp%X( : nlp%n ) ) )  &
            <= data%control%stop_s ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 990
       END IF

!  record the change (decrease) in the model

       data%dm = inform%obj - data%model

!  compute the slope and curvature along the step

       data%stg = DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )
       data%hstbs = data%model - data%stg
!      write(6,*) ' stg = ', data%stg

!  record the current point

       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%R_current( : nlp%m_r ) = nlp%R( : nlp%m_r )
       data%obj_current = inform%obj
       data%norm_R_current = inform%norm_r

!  form the trial point

       nlp%X( : nlp%n ) = data%GN_model%X( : nlp%n )

!  evaluate the objective function at the trial point

       IF ( data%reverse_r ) THEN
         data%branch = 320 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_R( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%R( : nlp%m_r ) )
         IF ( data%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_R'
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF
       END IF

!  return from reverse communication with the objective value

   320 CONTINUE
       IF ( data%printw ) WRITE( data%out, "( A, ' statement 320' )" ) prefix
       inform%r_eval = inform%r_eval + 1

       IF ( data%reverse_r ) THEN
         IF ( reverse%eval_status /= 0 ) THEN
           inform%bad_eval = 'eval_R'
           inform%status = GALAHAD_error_evaluation ; GO TO 990
         END IF
       END IF

       IF ( data%w_eq_identity ) THEN
         data%Y( : nlp%m_r ) = nlp%R( : nlp%m_r )
         data%norm_r_trial = TWO_NORM( nlp%R( : nlp%m_r ) )
         data%f_trial = half * data%norm_r_trial ** 2
       ELSE
         data%Y( : nlp%m_r ) = nlp%W( : nlp%m_r ) * nlp%R( : nlp%m_r )
         val = DOT_PRODUCT( data%Y( : nlp%m_r ), nlp%R( : nlp%m_r ) )
         data%norm_r_trial = SQRT( val )
         data%f_trial = half * val
       END IF

!      IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
!        WRITE( data%out, "( /, A, ' name                  C' )" ) prefix
!        DO i = 1, nlp%m_r
!          WRITE( data%out, "(  A, 1X, I10, ES22.14 )" )  prefix, i, nlp%R( i )
!        END DO
!      END IF

!      write(6,*) ' ftrial before, ||s|| ', data%f_trial,TWO_NORM(nlp%X(:nlp%n))

!  deal with NaN trial objective values
!  ------------------------------------

!      data%f_is_nan = IEEE_IS_NAN( data%f_trial )
       data%f_is_nan = data%f_trial /= data%f_trial
       IF ( data%f_is_nan ) THEN
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%R( : nlp%m_r ) = data%R_current( : nlp%m_r )

!  control printing for the NaN case

         IF ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print .AND.                              &
              MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) &
             THEN
           data%printi = data%set_printi ; data%printt = data%set_printt
           data%printm = data%set_printm ; data%printw = data%set_printw
           data%printd = data%set_printd
           data%print_level = data%control%print_level
           data%control%SLLS_control%print_level = data%print_level_slls
           data%control%SLLSB_control%print_level = data%print_level_sllsb
         ELSE
           data%printi = .FALSE. ; data%printt = .FALSE.
           data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
           data%print_level = 0
           data%control%SLLS_control%print_level = 0
           data%control%SLLSB_control%print_level = 0
         END IF
         data%print_iteration_header = data%print_level > 1 .OR.               &
           ( data%control%SLLS_control%print_level > 0 .AND.                   &
             data%solve_projection ) .OR.                                      &
           ( data%control%SLLSB_control%print_level > 0 .AND. .NOT.            &
             data%solve_projection )

!  print one-line summary

         IF ( data%printi ) THEN
           IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
             WRITE( data%out, 2090 ) prefix
             IF ( data%solve_projection ) THEN
               IF ( data%control%print_obj ) THEN
                 WRITE( data%out, 2130 ) prefix
               ELSE
                 WRITE( data%out, 2120 ) prefix
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
           IF ( data%solve_projection ) THEN
             char_facts                                                        &
               = ADJUSTR( STRING_integer_6( inform%SLLS_inform%iter ) )
!              = ADJUSTR( STRING_integer_6( data%total_facts ) )
             WRITE( data%out,  "( A, A6, 1X, 2A1, '    NaN           -    ',   &
            &  '    - Inf ',  2ES8.1, 1X, A6, F8.2 )" )                        &
                prefix, char_iter, data%accept, data%hard,                     &
                inform%weight, data%s_norm, char_facts, data%clock_now
           ELSE
             char_sit = ADJUSTR( STRING_integer_6( inform%SLLSB_inform%iter ) )
             WRITE( data%out, "( A, A6, 1X, 2A1, '    NaN           -    ',    &
            &  '    - Inf ', 2ES8.1, 1X, A6, F8.2 )" ) prefix, char_iter,      &
                data%accept, data%perturb,                                     &
                inform%weight, data%s_norm, char_sit, data%clock_now
           END IF
         END IF

!  check to see if we are still "alive"

         IF ( data%control%alive_unit > 0 ) THEN
           INQUIRE( FILE = data%control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             IF ( control%error > 0 ) WRITE( control%error,                    &
               "( A, ' error: alive file removed' )" ) prefix
             inform%status = GALAHAD_error_alive ; GO TO 990
           END IF
         END IF

!  check to see if the iteration limit has been exceeded

         inform%iter = inform%iter + 1
         IF ( inform%iter > data%control%maxit .AND. data%step_accepted ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: iteration limit exceeded' )" ) prefix
           inform%status = GALAHAD_error_max_iterations ; GO TO 990
         END IF

!  increase the regularization weight and try again

         inform%weight = data%control%weight_increase * inform%weight
         GO TO 190
       END IF

!  compute the change in the objective function

       data%df = inform%obj - data%f_trial
!      if (data%printi) write(6,*) ' dm, df ', data%dm, data%df

!  compute the ratio of actual to predicted reduction over the current iteration

!      rounding = MAX( one, ABS( inform%obj ) ) * teneps
       rounding =                                                              &
         MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = rp_ ) * epsmch

       ared = data%df + rounding
       prered = data%dm + rounding
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
!        data%s_norm_successful = data%s_norm
         data%got_jr = .FALSE.

!  save the new estimated solution characteristics

         inform%norm_r = data%norm_r_trial
         inform%obj = data%f_trial
         nlp%Y( : nlp%m_c ) = data%GN_model%Y( : nlp%m_c )
         nlp%Z( : nlp%n ) = data%GN_model%Z( : nlp%n )
         nlp%X_status( : nlp%n ) = data%GN_model%X_status( : nlp%n )

!  stop if the residual is sufficiently small

         IF ( inform%norm_r <= data%stop_r ) THEN

!  print one-line summary

           IF ( data%printi ) THEN
             CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
             data%time_now = data%time_now - data%time_start
             data%clock_now = data%clock_now - data%clock_start
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2090 ) prefix
               IF ( data%solve_projection ) THEN
                 IF ( data%control%print_obj ) THEN
                   WRITE( data%out, 2130 ) prefix
                 ELSE
                   WRITE( data%out, 2120 ) prefix
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
             IF ( data%control%print_obj ) THEN
               obj = inform%obj
             ELSE
               obj = inform%norm_r
             END IF
             IF ( inform%iter > 0 ) THEN
               IF ( data%solve_projection ) THEN
                 char_sit                                                      &
                   = ADJUSTR( STRING_integer_6( inform%SLLS_inform%iter ) )
               ELSE
                 char_sit                                                      &
                   = ADJUSTR( STRING_integer_6( inform%SLLSB_inform%iter ) )
               END IF
               WRITE( data%out, 2140 ) prefix, char_iter, data%accept,         &
                  data%perturb, obj, inform%norm_pg, data%ratio,               &
                  data%old_weight, data%s_norm, char_sit, data%clock_now
             ELSE
               WRITE( data%out, 2150 ) prefix,                                 &
                      char_iter, inform%norm_r, inform%norm_pg
             END IF
           END IF
           inform%status = GALAHAD_ok ; GO TO 900
         END IF

!  the new point is not acceptable

       ELSE
         data%poor_model = .TRUE.
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         nlp%R( : nlp%m_r ) = data%R_current( : nlp%m_r )
         IF ( data%w_eq_identity ) THEN
           data%Y( : nlp%m_r ) = nlp%R( : nlp%m_r )
         ELSE
           data%Y( : nlp%m_r ) = nlp%W( : nlp%m_r ) * nlp%R( : nlp%m_r )
         END IF
         data%new_point = .FALSE.
       END IF

       IF ( data%ratio >= data%control%eta_successful ) THEN

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
       CASE DEFAULT
         IF ( data%ratio < data%control%eta_successful ) THEN
           inform%weight = data%control%weight_increase * inform%weight
           IF ( data%control%weight_update_strategy ==                         &
                weight_update_increase ) THEN
             IF ( data%s_norm <= ten ** ( - 4 ) .AND.                          &
                  inform%norm_pg < inform%norm_r ) THEN
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

!      write(6,*) ' weight update ', inform%weight

!  record the clock time

       CALL CPU_TIME( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start
       IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,     &
      &    F12.2,  ' seconds' )" ) prefix, data%clock_now
       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
              data%time_now > data%control%cpu_time_limit ) .OR.               &
            ( data%control%clock_time_limit >= zero .AND.                      &
              data%clock_now > data%control%clock_time_limit ) ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' error: time limit exceeded' )" ) prefix
         inform%status = GALAHAD_error_cpu_limit ; GO TO 990
       END IF

!write(6,*) ' f ', data%f_trial, data%ratio
!stop
     GO TO 100

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 910' )" ) prefix
     CALL CPU_TIME( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

!  print details of solution

     IF ( inform%norm_r > zero ) THEN
       inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) ) / inform%norm_r
     ELSE
       inform%norm_g = zero
     END IF
!    write(6,*) ' final weight = ', inform%weight

     IF ( data%printi ) THEN

!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%r_eval, inform%jr_eval, &
!         inform%iter, inform%inner_iter, inform%obj, inform%norm_g
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
    &   ')', /, A, '  SNLS stopping tolerances (r,P[-J''r]) =', 2ES9.2 )" )    &
          prefix, TRIM( nlp%pname ), nlp%n, nlp%m_r, prefix,                     &
          data%stop_r, data%stop_pg
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" )        &
         prefix, data%non_monotone_history
       IF ( data%control%magic_step )                                          &
         WRITE( data%out, "( A, '  Magic step used' )" ) prefix
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation and deallocation errors

 980 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 980' )" ) prefix
     CALL CPU_TIME( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

!  other errors

 990 CONTINUE
     IF ( data%printw ) WRITE( data%out, "( A, ' statement 990' )" ) prefix
     CALL CPU_TIME( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'SNLS_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # major  iterations     = ', I10,                        &
              /, A, ' # minor (cg) iterations = ', I10,                        &
             //, A, ' objective value         = ', ES22.14,                    &
              /, A, ' projected gradient norm = ', ES12.4 )
 2010 FORMAT( /, A, ' name                  X                   G ' )
 2020 FORMAT(  A, 1X, A10, 2ES22.14 )
 2030 FORMAT(  A, 1X, I10, 2ES22.14 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2090 FORMAT( A, '        (a=accept r=reject)' )
 2100 FORMAT( A, '    It        r        ||pg||    ',                          &
             ' ratio   weight   step  it sllsb    time' )
 2110 FORMAT( A, '    It        f        ||pg||    ',                          &
             ' ratio   weight   step  it sllsb    time' )
 2120 FORMAT( A, '    It         r       ||pg||    ',                          &
             ' ratio   weight   step   it slls    time' )
 2130 FORMAT( A, '    It         f       ||pg||    ',                          &
             ' ratio   weight   step   it slls    time' )
 2140 FORMAT( A, A6, 1X, 2A1, 2ES11.4, ES9.1, 2ES8.1, 3X, A6, F8.2 )
 2150 FORMAT( A, A6, 3X, 2ES11.4 )

 !  End of subroutine SNLS_solve

     END SUBROUTINE SNLS_solve

!! G A L A H A D - S N L S _ u p d a t e _ h i s t o r y  S U B R O U T I N E
!
!     SUBROUTINE SNLS_update_history( history, max_hist, F_hist, F_ref, f )
!
!!-----------------------------------------------
!!   D u m m y   A r g u m e n t s
!!-----------------------------------------------
!
!     INTEGER ( KIND = ip_ ), INTENT( IN ) :: history
!     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: max_hist
!     REAL ( KIND = rp_ ), INTENT( OUT ) :: F_ref
!     REAL ( KIND = rp_ ), INTENT( IN ) :: f
!     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( history + 1 ) :: F_hist
!
!!-----------------------------------------------
!!   L o c a l   V a r i a b l e s
!!-----------------------------------------------
!
!     INTEGER ( KIND = ip_ ) :: i
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
!     END SUBROUTINE SNLS_update_history

!-*-*-  G A L A H A D -  S N L S _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE SNLS_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SNLS_control_type ), INTENT( IN ) :: control
     TYPE ( SNLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

!  integer arrays

     array_name = 'snls: data%GN_model%Ao%row'
     CALL SPACE_dealloc_array( data%GN_model%Ao%row,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%Ao%col'
     CALL SPACE_dealloc_array( data%GN_model%Ao%col,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%Ao%ptr'
     CALL SPACE_dealloc_array( data%GN_model%Ao%ptr,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%COHORT'
     CALL SPACE_dealloc_array( data%GN_model%COHORT,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  real arrays

     array_name = 'snls: data%X_current'
     CALL SPACE_dealloc_array( data%X_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%R_current'
     CALL SPACE_dealloc_array( data%R_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%G_current'
     CALL SPACE_dealloc_array( data%G_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%D_hist'
     CALL SPACE_dealloc_array( data%D_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%F_hist'
     CALL SPACE_dealloc_array( data%F_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%B'
     CALL SPACE_dealloc_array( data%GN_model%B,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%X'
     CALL SPACE_dealloc_array( data%GN_model%X,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%Y'
     CALL SPACE_dealloc_array( data%GN_model%Y,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%Z'
     CALL SPACE_dealloc_array( data%GN_model%Z,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%R'
     CALL SPACE_dealloc_array( data%GN_model%R,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%G'
     CALL SPACE_dealloc_array( data%GN_model%G,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%X_status'
     CALL SPACE_dealloc_array( data%GN_model%X_status,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%X_s'
     CALL SPACE_dealloc_array( data%GN_model%X_s,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
                                                                        
     array_name = 'snls: data%GN_model%W'
     CALL SPACE_dealloc_array( data%GN_model%W,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%GN_model%Ao%val'
     CALL SPACE_dealloc_array( data%GN_model%Ao%val,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  characacter arrays

     array_name = 'snls: data%GN_model%Ao%type'
     CALL SPACE_dealloc_array( data%GN_model%Ao%type,                          &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     CALL REVERSE_terminate( data%reverse, inform%status, inform%alloc_status, &
                             bad_alloc = inform%bad_alloc,                     &
                             out = control%error, deallocate_error_fatal =     &
                             control%deallocate_error_fatal )

!  Deallocate all arrays allocated within SLLS

     CALL SLLS_terminate( data%SLLS_data, data%control%SLLS_control,           &
                          inform%SLLS_inform )
     inform%status = inform%SLLS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%SLLS_inform%alloc_status
       inform%bad_alloc = inform%SLLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arrays allocated within SLLSB

     CALL SLLSB_terminate( data%SLLSB_data, data%control%SLLSB_control,        &
                          inform%SLLSB_inform )
     inform%status = inform%SLLSB_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%SLLSB_inform%alloc_status
       inform%bad_alloc = inform%SLLSB_inform%bad_alloc
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

!  End of subroutine SNLS_terminate

     END SUBROUTINE SNLS_terminate

! -  G A L A H A D -  S N L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE SNLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SNLS_control_type ), INTENT( IN ) :: control
     TYPE ( SNLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL SNLS_terminate( data%snls_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'snls: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Y'
     CALL SPACE_dealloc_array( data%nlp%Y,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Z'
     CALL SPACE_dealloc_array( data%nlp%Z,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%R'
     CALL SPACE_dealloc_array( data%nlp%R,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%W'
     CALL SPACE_dealloc_array( data%nlp%W,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Jr%row'
     CALL SPACE_dealloc_array( data%nlp%Jr%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Jr%col'
     CALL SPACE_dealloc_array( data%nlp%Jr%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Jr%ptr'
     CALL SPACE_dealloc_array( data%nlp%Jr%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Jr%val'
     CALL SPACE_dealloc_array( data%nlp%Jr%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'snls: data%nlp%Jr%type'
     CALL SPACE_dealloc_array( data%nlp%Jr%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     CALL REVERSE_terminate( data%reverse, inform%status, inform%alloc_status, &
                             bad_alloc = inform%bad_alloc,                     &
                             out = control%error, deallocate_error_fatal =     &
                             control%deallocate_error_fatal )

     RETURN

!  End of subroutine SNLS_full_terminate

     END SUBROUTINE SNLS_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

! G A L A H A D -  S N L S _ i m p o r t _ w i t h o u t _ j a c  S U B R OUTINE

     SUBROUTINE SNLS_import_without_jac( control, data, status, n, m_r, m_c,   &
                                         COHORT )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SNLS_solve
!
!  data is a scalar variable of type SNLS_full_data_type used for internal data
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
!   -3. The restriction n > 0 has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables (columns of Jr)
!
!  m_r is a scalar variable of type default integer, that holds the number of
!   residuals (rows of Jr)
!
!  m_c is a scalar variable of type default integer, that holds the
!   number of cohorts
!
!  COHORT is an optional rank-one array of type default integer and length n
!   that must be set so that its j-th component is a number, between 1 and m, 
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If m or COHORT is absent, all variables will be 
!   assumed to belong to a single cohort
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m_r, m_c
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( n ), OPTIONAL, INTENT( IN ) :: COHORT

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%snls_control = control

     error = data%snls_control%error
     space_critical = data%snls_control%space_critical
     deallocate_error_fatal = data%snls_control%space_critical

!  if there are multiple cohorts, record them

     IF ( PRESENT( COHORT ) ) THEN
       data%nlp%m_c = m_c
       array_name = 'snls: data%nlp%COHORT'
       CALL SPACE_resize_array( n, data%nlp%COHORT,                            &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900
       
       IF ( data%f_indexing ) THEN
         data%nlp%COHORT( : n ) = MAX( COHORT( : n ), 0 )
       ELSE
         data%nlp%COHORT( : n ) = MAX( COHORT( : n ) + 1, 0 )
       END IF
     ELSE
       data%nlp%m_c = 1
     END IF

!  allocate vector space if required

     array_name = 'snls: data%nlp%X'
     CALL SPACE_resize_array( n, data%nlp%X,                                   &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

     array_name = 'snls: data%nlp%Y'
     CALL SPACE_resize_array( data%nlp%m_c, data%nlp%Y,                        &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

     array_name = 'snls: data%nlp%Z'
     CALL SPACE_resize_array( n, data%nlp%Z,                                   &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

     array_name = 'snls: data%nlp%R'
     CALL SPACE_resize_array( m_r, data%nlp%R,                                 &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

     array_name = 'snls: data%nlp%G'
     CALL SPACE_resize_array( n, data%nlp%G,                                   &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

     array_name = 'snls: data%nlp%X_status'
     CALL SPACE_resize_array( n, data%nlp%X_status,                            &
            data%snls_inform%status, data%snls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%snls_inform%bad_alloc, out = error )
     IF ( data%snls_inform%status /= 0 ) GO TO 900

!  put data into the required components of the qpt storage type

     data%nlp%n = n ; data%nlp%m_r = m_r

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%snls_inform%status
     RETURN

!  End of subroutine SNLS_import_without_jac

     END SUBROUTINE SNLS_import_without_jac

!-*-*-*-  G A L A H A D -  S N L S _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE SNLS_import( control, data, status, n, m_r, m_c, Jr_type,      &
                             Jr_ne, Jr_row, Jr_col, Jr_ptr, COHORT )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SNLS_solve
!
!  data is a scalar variable of type SNLS_full_data_type used for internal data
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
!   -3. The restriction n > 0, m_r >= 0 or requirement that Jr_type contains
!       its relevant string 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
!       'COORDINATE', 'SPARSE_BY_ROWS', or 'SPARSE_BY_COLUMNS'
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables (columns of Jr)
!
!  m_r is a scalar variable of type default integer, that holds the number of
!   residuals (rows of Jr)
!
!  m_c is a scalar variable of type default integer, that holds the number of
!   cohorts
!
!  Jr_type is a character string that specifies the design matrix storage
!   scheme used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   or 'absent', the latter if m = 0; lower or upper case variants are allowed
!
!  Jr_ne is a scalar variable of type default integer, that holds the number of
!   entries in Jr in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  Jr_row is a rank-one array of type default integer, that holds the row
!   indices Jr in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  Jr_col is a rank-one array of type default integer, that holds the column
!   indices of Jr in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense scheme is used, and
!   in this case can be of length 0
!
!  Jr_ptr is a rank-one array of dimension max(o+1,n+1) and type default
!   integer, that holds the starting position of each row of J, as well as the
!   total number of entries plus one, in the sparse row-wise storage scheme,
!   or the starting position of each column of Jr, as well as the total
!   number of entries plus one, in the sparse column-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!  COHORT is an optional rank-one array of type default integer and length n
!   that must be set so that its j-th component is a number, between 1 and m, 
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If m or COHORT is absent, all variables will be 
!   assumed to belong to a single cohort
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m_r, m_c, Jr_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Jr_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Jr_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Jr_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Jr_ptr
     INTEGER ( KIND = ip_ ), DIMENSION( n ), OPTIONAL, INTENT( IN ) :: COHORT

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug

!  assign space for vector data

     CALL SNLS_import_without_jac( control, data, status, n, m_r, m_c,         &
                                   COHORT = COHORT )
     IF ( status /= GALAHAD_ready_to_solve ) GO TO 900

     error = data%snls_control%error
     space_critical = data%snls_control%space_critical
     deallocate_error_fatal = data%snls_control%space_critical

!  set Jr appropriately in the nlp storage type

     SELECT CASE ( Jr_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( Jr_row ) .AND. PRESENT( Jr_col ) ) ) THEN
         data%snls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%nlp%Jr%type, 'COORDINATE',                           &
                     data%snls_inform%alloc_status )
       data%nlp%Jr%n = n ; data%nlp%Jr%m = m_r
       data%nlp%Jr%ne = Jr_ne

       array_name = 'snls: data%nlp%Jr%row'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%row,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%col'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%col,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%val'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%val,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%nlp%Jr%row( : data%nlp%Jr%ne ) = Jr_row( : data%nlp%Jr%ne )
         data%nlp%Jr%col( : data%nlp%Jr%ne ) = Jr_col( : data%nlp%Jr%ne )
       ELSE
         data%nlp%Jr%row( : data%nlp%Jr%ne ) = Jr_row( : data%nlp%Jr%ne ) + 1
         data%nlp%Jr%col( : data%nlp%Jr%ne ) = Jr_col( : data%nlp%Jr%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( Jr_ptr ) .AND. PRESENT( Jr_col ) ) ) THEN
         data%snls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%nlp%Jr%type, 'SPARSE_BY_ROWS',                       &
                     data%snls_inform%alloc_status )
       data%nlp%Jr%n = n ; data%nlp%Jr%m = m_r
       IF ( data%f_indexing ) THEN
         data%nlp%Jr%ne = Jr_ptr( m_r + 1 ) - 1
       ELSE
         data%nlp%Jr%ne = Jr_ptr( m_r + 1 )
       END IF

       array_name = 'snls: data%nlp%Jr%ptr'
       CALL SPACE_resize_array( m_r + 1, data%nlp%Jr%ptr,                      &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%col'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%col,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%val'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%val,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%nlp%Jr%ptr( : m_r + 1 ) = Jr_ptr( : m_r + 1 )
         data%nlp%Jr%col( : data%nlp%Jr%ne ) = Jr_col( : data%nlp%Jr%ne )
       ELSE
         data%nlp%Jr%ptr( : m_r + 1 ) = Jr_ptr( : m_r + 1 ) + 1
         data%nlp%Jr%col( : data%nlp%Jr%ne ) = Jr_col( : data%nlp%Jr%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( Jr_ptr ) .AND. PRESENT( Jr_row ) ) ) THEN
         data%snls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%nlp%Jr%type, 'SPARSE_BY_COLUMNS',                    &
                     data%snls_inform%alloc_status )
       data%nlp%Jr%n = n ; data%nlp%Jr%m = m_r
       IF ( data%f_indexing ) THEN
         data%nlp%Jr%ne = Jr_ptr( n + 1 ) - 1
       ELSE
         data%nlp%Jr%ne = Jr_ptr( n + 1 )
       END IF
       array_name = 'snls: data%nlp%Jr%ptr'
       CALL SPACE_resize_array( n + 1, data%nlp%Jr%ptr,                        &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%row'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%row,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       array_name = 'snls: data%nlp%Jr%val'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%val,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%nlp%Jr%ptr( : n + 1 ) = Jr_ptr( : n + 1 )
         data%nlp%Jr%row( : data%nlp%Jr%ne ) = Jr_row( : data%nlp%Jr%ne )
       ELSE
         data%nlp%Jr%ptr( : n + 1 ) = Jr_ptr( : n + 1 ) + 1
         data%nlp%Jr%row( : data%nlp%Jr%ne ) = Jr_row( : data%nlp%Jr%ne ) + 1
       END IF

     CASE ( 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%nlp%Jr%type, 'DENSE_BY_ROWS',                        &
                     data%snls_inform%alloc_status )
       data%nlp%Jr%n = n ; data%nlp%Jr%m = m_r
       data%nlp%Jr%ne = m_r * n

       array_name = 'snls: data%nlp%Jr%val'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%val,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%nlp%Jr%type, 'DENSE_BY_COLUMNS',                     &
                     data%snls_inform%alloc_status )
       data%nlp%Jr%n = n ; data%nlp%Jr%m = m_r
       data%nlp%Jr%ne = m_r * n

       array_name = 'snls: data%nlp%Jr%val'
       CALL SPACE_resize_array( data%nlp%Jr%ne, data%nlp%Jr%val,               &
              data%snls_inform%status, data%snls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( data%snls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%snls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ready_to_solve
     data%snls_inform%status = 1
     RETURN

!  error returns

 900 CONTINUE
     status = data%snls_inform%status
     RETURN

!  End of subroutine SNLS_import

     END SUBROUTINE SNLS_import

!-  G A L A H A D -  S N L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE SNLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SNLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_control_type ), INTENT( IN ) :: control
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%snls_control = control

!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine SNLS_reset_control

     END SUBROUTINE SNLS_reset_control

!-  G A L A H A D -  S N L S _ s o l v e _ w i t h _ j a c  S U B R O U T I N E

     SUBROUTINE SNLS_solve_with_jac( data, userdata, status, X, Y, Z, R, G,    &
                                     X_stat, eval_R, eval_Jr, W )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual and Jacobian operations are available via subroutine calls. 
!  See SNLS_solve for a description of the required arguments. The variable 
!  status is a proxy for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: X_stat
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W
     EXTERNAL :: eval_R, eval_Jr

!  local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  assign input X

     data%snls_inform%status = status
     IF ( data%snls_inform%status == 1 ) THEN
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  add space for, and assign, diagonal weights if required

       IF ( PRESENT( W ) ) THEN
         array_name = 'snls: data%nlp%W'
         CALL SPACE_resize_array( data%nlp%m_r, data%nlp%W,                    &
                status, data%snls_inform%alloc_status,                         &
                array_name = array_name,                                       &
                deallocate_error_fatal = data%snls_control%space_critical,     &
                exact_size = data%snls_control%space_critical,                 &
                bad_alloc = data%snls_inform%bad_alloc,                        &
                out = data%snls_control%error )
         IF ( status /= 0 ) GO TO 900
         data%nlp%W( : data%nlp%m_r ) = W( : data%nlp%m_r )
       END IF
     END IF

!  call the solver

     CALL SNLS_solve( data%nlp, data%snls_control, data%snls_inform,           &
                      data%snls_data, userdata,                                &
                      eval_R = eval_R, eval_Jr = eval_Jr )
     status = data%snls_inform%status

!  recover the optimal primal and dual variables, and Lagrange multipliers

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     Y( : data%nlp%m_c ) = data%nlp%Y( : data%nlp%m_c )
     Z( : data%nlp%n ) = data%nlp%Z( : data%nlp%n )

!  recover the residual value and gradient

     R( : data%nlp%m_r ) = data%nlp%R( : data%nlp%m_r )
     G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

!  recover the status of x

     X_stat( : data%nlp%n ) = data%nlp%X_status( : data%nlp%n )

     RETURN

!  error returns

 900 CONTINUE
     RETURN

!  end of subroutine SNLS_solve_with_jac

     END SUBROUTINE SNLS_solve_with_jac

! - G A L A H A D -  S N L S _ s o l v e _ with _ jacprod  S U B R O U T I N E 

     SUBROUTINE SNLS_solve_with_jacprod( data, userdata, status, X, Y, Z,      &
                                         R, G, X_stat, eval_R, eval_Jr_PROD,   &
                                         eval_Jr_SCOL, eval_Jr_SPROD, W )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual, and Jacobian-vector product operations are available via 
!  subroutine calls. See SNLS_solve for a description of the required 
!  arguments. The variable status is a proxy for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: X_stat
     EXTERNAL :: eval_R, eval_Jr_PROD, eval_Jr_SCOL, eval_Jr_SPROD

!  local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  assign input X

     data%snls_inform%status = status
     IF ( data%snls_inform%status == 1 ) THEN
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  add space for, and assign, diagonal weights if required

       IF ( PRESENT( W ) ) THEN
         array_name = 'snls: data%nlp%W'
         CALL SPACE_resize_array( data%nlp%m_r, data%nlp%W,                    &
                status, data%snls_inform%alloc_status,                         &
                array_name = array_name,                                       &
                deallocate_error_fatal = data%snls_control%space_critical,     &
                exact_size = data%snls_control%space_critical,                 &
                bad_alloc = data%snls_inform%bad_alloc,                        &
                out = data%snls_control%error )
         IF ( status /= 0 ) GO TO 900
         data%nlp%W( : data%nlp%m_r ) = W( : data%nlp%m_r )
       END IF
     END IF

!  call the solver

     CALL SNLS_solve( data%nlp, data%snls_control, data%snls_inform,           &
                      data%snls_data, userdata, eval_R = eval_R,               &
                      eval_Jr_prod = eval_Jr_PROD,                             &
                      eval_Jr_scol = eval_Jr_SCOL,                             &
                      eval_Jr_sprod = eval_Jr_SPROD )
     status = data%snls_inform%status

!  recover the optimal primal and dual variables, and Lagrange multipliers

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     Y( : data%nlp%m_c ) = data%nlp%Y( : data%nlp%m_c )
     Z( : data%nlp%n ) = data%nlp%Z( : data%nlp%n )

!  recover the residual value and gradient

     R( : data%nlp%m_r ) = data%nlp%R( : data%nlp%m_r )
     G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

!  recover the status of x

     X_stat( : data%nlp%n ) = data%nlp%X_status( : data%nlp%n )

     RETURN

!  error returns

 900 CONTINUE
     RETURN

!  end of subroutine SNLS_solve_with_jacprod

     END SUBROUTINE SNLS_solve_with_jacprod

!-  G A L A H A D -  S N L S _ s o l v e _ reverse _ with _ jac  S U B R OUTINE 

     SUBROUTINE SNLS_solve_reverse_with_jac( data, status, eval_status, X, Y,  &
                                             Z, R, G, X_stat, Jr_val, W )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual and Jacobians are available via reverse communications. 
!  See SNLS_solve for a description of the required arguments. The variable 
!  status is a proxy for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: eval_status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: R
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: Jr_val
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: X_stat
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W

!  local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  recover data from reverse communication

     data%snls_inform%status = status
     data%snls_data%eval_status = eval_status
     SELECT CASE ( data%snls_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
       IF ( PRESENT( W ) ) THEN
         array_name = 'snls: data%nlp%W'
         CALL SPACE_resize_array( data%nlp%m_r, data%nlp%W,                    &
                status, data%snls_inform%alloc_status,                         &
                array_name = array_name,                                       &
                deallocate_error_fatal = data%snls_control%space_critical,     &
                exact_size = data%snls_control%space_critical,                 &
                bad_alloc = data%snls_inform%bad_alloc,                        &
                out = data%snls_control%error )
         IF ( status /= 0 ) GO TO 900
         data%nlp%W( : data%nlp%m_r ) = W( : data%nlp%m_r )
       END IF
     CASE ( 2 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%R( : data%nlp%m_r ) = R( : data%nlp%m_r )
     CASE( 3 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%Jr%val( : data%nlp%Jr%ne ) = Jr_val( : data%nlp%Jr%ne )
     END SELECT

!  call the solver

     CALL SNLS_solve( data%nlp, data%snls_control, data%snls_inform,           &
                      data%snls_data, data%userdata, reverse = data%reverse )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%snls_inform%status )
     CASE( 0 )
       Y( : data%nlp%m_c ) = data%nlp%Y( : data%nlp%m_c )
       Z( : data%nlp%n ) = data%nlp%Z( : data%nlp%n )
       R( : data%nlp%m_r ) = data%nlp%R( : data%nlp%m_r )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
       X_stat( : data%nlp%n ) = data%nlp%X_status( : data%nlp%n )
     CASE( 4, 5, 6, 7 )
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%snls_inform%status
     END SELECT
     status = data%snls_inform%status

!  error returns

 900 CONTINUE
     RETURN

     RETURN

!  end of subroutine SNLS_solve_reverse_with_jac

     END SUBROUTINE SNLS_solve_reverse_with_jac

!-  G A L A H A D -  S N L S _ s o l v e _ reverse _ with _ jacprod  SUBROUTINE

     SUBROUTINE SNLS_solve_reverse_with_jacprod( data, status, eval_status,    &
                                                 X, Y, Z, R, G, X_stat,        &
                                                 V, IV, lvl, lvu, index,       &
                                                 P, IP, lp, W )

!  solve the nonlinear least-squares problem previously imported when access
!  to residual and Jacobian-vector product operations are available via 
!  reverse communications. See SNLS_solve for a description of the required 
!  arguments. The variable status is a proxy for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: eval_status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: R
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: X_stat
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: V
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: IV
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: lvl, lvu, index
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: P
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: IP
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: lp
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     error = data%snls_control%error
     space_critical = data%snls_control%space_critical
     deallocate_error_fatal = data%snls_control%deallocate_error_fatal

!  recover data from reverse communication

     data%snls_inform%status = status
     data%snls_data%eval_status = eval_status
     SELECT CASE ( data%snls_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
       IF ( PRESENT( W ) ) THEN
         array_name = 'snls: data%nlp%W'
         CALL SPACE_resize_array( data%nlp%m_r, data%nlp%W,                    &
              status, data%snls_inform%alloc_status, array_name = array_name,  &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
         IF ( status /= 0 ) GO TO 900
         data%nlp%W( : data%nlp%m_r ) = W( : data%nlp%m_r )
       END IF

       array_name = 'snls: data%reverse%iv'
       CALL SPACE_resize_array( data%nlp%n, data%reverse%iv,                   &
              status, data%snls_inform%alloc_status, array_name = array_name,  &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( status /= 0 ) GO TO 900

       array_name = 'snls: data%reverse%ip'
       CALL SPACE_resize_array( data%nlp%m_r, data%reverse%ip,                 &
              status, data%snls_inform%alloc_status, array_name = array_name,  &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( status /= 0 ) GO TO 900

       array_name = 'snls: data%reverse%v'
       CALL SPACE_resize_array( MAX( data%nlp%m_r, data%nlp%n ),               &
              data%reverse%v,                                                  &
              status, data%snls_inform%alloc_status, array_name = array_name,  &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( status /= 0 ) GO TO 900

       array_name = 'snls: data%reverse%p'
       CALL SPACE_resize_array( MAX( data%nlp%m_r, data%nlp%n ),               &
              data%reverse%p,                                                  &
              status, data%snls_inform%alloc_status, array_name = array_name,  &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%snls_inform%bad_alloc, out = error )
       IF ( status /= 0 ) GO TO 900
     CASE( 2 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%R( : data%nlp%m_r ) = R( : data%nlp%m_r )
     CASE( 3 )
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%snls_inform%status
     CASE( 4 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%reverse%P( : data%nlp%m_r ) = P( : data%nlp%m_r )
     CASE( 5 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%reverse%P( : data%nlp%n ) = P( : data%nlp%n )
     CASE( 6 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%reverse%lp = lp
         data%reverse%P( : lp ) = P( : lp )
         IF ( data%f_indexing ) THEN
           data%reverse%IP( : lp ) = IP( : lp )
         ELSE
           data%reverse%IP( : lp ) = IP( : lp ) + 1
         END IF
       END IF
     CASE( 7 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%reverse%P( : data%nlp%m_r ) = P( : data%nlp%m_r )
     CASE( 8 )
       data%reverse%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         IF ( .NOT. data%f_indexing ) THEN
           lvl = data%reverse%lvl ; lvu = data%reverse%lvu
           IV( lvl : lvu ) = data%reverse%IV( lvl : lvu )
         END IF
         data%reverse%P( IV( lvl : lvu ) ) = P( IV( lvl : lvu ) )
       END IF
     END SELECT

!  call the solver

     CALL SNLS_solve( data%nlp, data%snls_control, data%snls_inform,           &
                      data%snls_data, data%userdata, reverse = data%reverse )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%snls_inform%status )
     CASE( 0 )
       Y( : data%nlp%m_c ) = data%nlp%Y( : data%nlp%m_c )
       Z( : data%nlp%n ) = data%nlp%Z( : data%nlp%n )
       R( : data%nlp%m_r ) = data%nlp%R( : data%nlp%m_r )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
       X_stat( : data%nlp%n ) = data%nlp%X_status( : data%nlp%n )
     CASE( 2 )
     CASE( 3 )
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%snls_inform%status
     CASE( 4 )
       V( : data%nlp%n ) = data%reverse%V( : data%nlp%n )
     CASE( 5 )
       V( : data%nlp%m_r ) = data%reverse%V( : data%nlp%m_r )
     CASE( 6 )
       IF ( data%f_indexing ) THEN
         index = data%reverse%index
       ELSE
         index = data%reverse%index - 1
       END IF
     CASE( 7 )
       lvl = data%reverse%lvl ; lvu = data%reverse%lvu
       IV( lvl : lvu ) = data%reverse%IV( lvl : lvu )
       V( IV( lvl : lvu ) ) = data%reverse%V( IV( lvl : lvu ) )
       IF ( .NOT. data%f_indexing ) THEN
         IV( lvl : lvu ) = IV( lvl : lvu ) - 1
!        lvl = lvl - 1 ; lvu = lvu - 1
       END IF
     CASE( 8 )
       lvl = data%reverse%lvl ; lvu = data%reverse%lvu
       V( : data%nlp%m_r ) = data%reverse%V( : data%nlp%m_r )
       IF ( data%f_indexing ) THEN
         IV( lvl : lvu ) = data%reverse%IV( lvl : lvu )
       ELSE
         IV( lvl : lvu ) = data%reverse%IV( lvl : lvu ) - 1
!        lvl = lvl - 1 ; lvu = lvu - 1
       END IF
     END SELECT
     status = data%snls_inform%status

     RETURN

!  error returns

 900 CONTINUE
     RETURN

!  end of subroutine SNLS_solve_reverse_with_jacprod

     END SUBROUTINE SNLS_solve_reverse_with_jacprod

!-  G A L A H A D -  S N L S _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE SNLS_information( data, inform, status )

!  return solver information during or after solution by SNLS
!  See SNLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SNLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SNLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%snls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SNLS_information

     END SUBROUTINE SNLS_information

!  End of module GALAHAD_SNLS

   END MODULE GALAHAD_SNLS_precision
