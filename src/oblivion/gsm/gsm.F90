! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ G S M   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 4.1. January 29th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_GSM_precision

!     --------------------------------------------------------
!    |                                                        |
!    | GSM, a first-order (steepest-descent) subspace         |
!    |  trust-region algorithm for unconstrained optimization |
!    |                                                        |
!    |   Aim: find a (local) minimizer of the objective f(x)  |
!    |                                                        |
!     --------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SMT_precision
     USE GALAHAD_TRS_precision
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_STRING, ONLY: STRING_integer_6
     USE GALAHAD_BLAS_inter_precision, ONLY: SWAP
!    USE SPDSOL
!    USE HSL_MI13
!    USE IEEE_ARITHMETIC

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: GSM_initialize, GSM_read_specfile, GSM_solve, GSM_solve_full,   &
               GSM_terminate, NLPT_problem_type, GALAHAD_userdata_type,        &
               GSM_import, GSM_solve_direct, GSM_solve_reverse,                &
               GSM_full_initialize, GSM_full_terminate, GSM_reset_control,     &
               GSM_information

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE GSM_initialize
       MODULE PROCEDURE GSM_initialize, GSM_full_initialize
     END INTERFACE GSM_initialize

     INTERFACE GSM_terminate
       MODULE PROCEDURE GSM_terminate, GSM_full_terminate
     END INTERFACE GSM_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER ( KIND = ip_ ), PARAMETER  :: nskip_prec_max = 0
     INTEGER ( KIND = ip_ ), PARAMETER  :: history_max = 100
     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: tenth = 0.1_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteenth = 0.0625_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: hundred = 100.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteen = 16.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point9 = 0.9_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = rp_ ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: rho_quad = epsmch * ten ** 4

     REAL ( KIND = rp_ ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = rp_ ), PARAMETER :: gamma_2 = half
     REAL ( KIND = rp_ ), PARAMETER :: gamma_3 = two
     REAL ( KIND = rp_ ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = rp_ ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = rp_ ), PARAMETER :: mu_2 = point1
     REAL ( KIND = rp_ ), PARAMETER :: theta = half

     INTEGER ( KIND = ip_ ), PARAMETER  :: first_order_model = 1
     INTEGER ( KIND = ip_ ), PARAMETER  :: second_order_model = 2
     INTEGER ( KIND = ip_ ), PARAMETER  :: model = second_order_model

!    LOGICAL, PARAMETER  :: reset_to_I = .TRUE.
     LOGICAL, PARAMETER  :: reset_to_I = .FALSE.
     REAL ( KIND = rp_ ), PARAMETER :: h_diag_initial = 1.0_rp_
!    REAL ( KIND = rp_ ), PARAMETER :: h_diag_initial = 10.0_rp_

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: GSM_control_type

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

!   the maximum number of inner iterations performed per outer iteration

       INTEGER ( KIND = ip_ ) :: inner_maxit = 100

!   the maximum dimension of the search subspace

       INTEGER ( KIND = ip_ ) :: max_subspace = 2

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER ( KIND = ip_ ) :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!     strategy with this history length used

       INTEGER ( KIND = ip_ ) :: non_monotone = 1

!   try to pick a good initial trust-region radius using %advanced_start
!    iterates of a variant on the strategy of Sartenaer SISC 18(6)1990:1788-1803

       INTEGER ( KIND = ip_ ) :: advanced_start = 0

!   overall convergence tolerances. The iteration will terminate when the
!     norm of the gradient of the objective function is smaller than
!       MAX( %stop_g_absolute, %stop_g_relative * norm of the initial gradient )
!     or if the step is less than %stop_s

#ifdef REAL_32
       REAL ( KIND = rp_ ) :: stop_g_absolute = ten ** ( - 3 )
       REAL ( KIND = rp_ ) :: stop_g_relative = ten ** ( - 4 )
#else
       REAL ( KIND = rp_ ) :: stop_g_absolute = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_g_relative = ten ** ( - 8 )
#endif
       REAL ( KIND = rp_ ) :: stop_s = epsmch

!   initial value for the trust-region radius (-ve => ||g_0||)

       REAL ( KIND = rp_ ) :: initial_radius = hundred

!   maximum permitted trust-region radius

       REAL ( KIND = rp_ ) :: maximum_radius = ten ** 8

!   a potential iterate will only be accepted if the actual decrease
!    f - f(x_new) is larger than %eta_successful times that predicted
!    by a quadratic model of the decrease. The trust-region radius will be
!    increased if this relative decrease is greater than %eta_very_successful
!    but smaller than %eta_too_successful

       REAL ( KIND = rp_ ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = rp_ ) :: eta_very_successful = point9
       REAL ( KIND = rp_ ) :: eta_too_successful = two

!   on very successful iterations, the trust-region radius will be increased by
!    the factor %radius_increase, while if the iteration is unsucceful, the
!    radius will be decreased by a factor %radius_reduce but no more than
!    %radius_reduce_max

       REAL ( KIND = rp_ ) :: radius_increase = two
       REAL ( KIND = rp_ ) :: radius_reduce = half
       REAL ( KIND = rp_ ) :: radius_reduce_max = sixteenth

!   the smallest value the onjective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = rp_ ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the smallest value of |(y-Hs)'s| / ||s|| || y-Hs||| that is allowed
!   when updating the SR1 Hessian approximation

       REAL ( KIND = rp_ ) :: sr1_skip = ten ** ( - 8 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: clock_time_limit = - one

!   should the radius be renormalized to account for a change in preconditioner?

       LOGICAL :: renormalize_radius = .FALSE.

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

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: TRS_control

     END TYPE GSM_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: GSM_time_type

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

       REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

       REAL ( KIND = rp_ ) :: clock_preprocess = 0.0

!  the clock time spent analysing the required matrices prior to factorization

       REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

       REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

       REAL ( KIND = rp_ ) :: clock_solve = 0.0

     END TYPE GSM_time_type

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type for dense solve with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: GSM_dense_inform_type

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

     END TYPE GSM_dense_inform_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: GSM_inform_type

!  return status. See GSM_solve for details

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

!  the total number of CG iterations performed

       INTEGER ( KIND = ip_ ) :: cg_iter = 0

!  the total number of evaluations of the objection function

       INTEGER ( KIND = ip_ ) :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER ( KIND = ip_ ) :: g_eval = 0

!  the value of the objective function at the best estimate of the solution
!   determined by GSM_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the norm of the gradient of the objective function at the best estimate
!   of the solution determined by GSM_solve

       REAL ( KIND = rp_ ) :: norm_g = HUGE( one )

!  the current value of the trust-region radius

       REAL ( KIND = rp_ ) :: radius = zero

!  timings (see above)

       TYPE ( GSM_time_type ) :: time

!  dense data type

       TYPE( GSM_dense_inform_type ) :: dense_inform

!  inform parameters for TRS

       TYPE ( TRS_inform_type ) :: TRS_inform

     END TYPE GSM_inform_type

!  - - - - - - - - - -
!   data derived types
!  - - - - - - - - - -

     TYPE, PUBLIC :: GSM_dense_data_type
       REAL ( KIND = rp_ ) :: f, model, norm_g, stop_g, norm_s, radius, ratio
       LOGICAL :: successful
       CHARACTER ( LEN = 1 ) :: accept
       TYPE ( TRS_data_type ) :: trs_data
     END TYPE GSM_dense_data_type

     TYPE, PUBLIC :: GSM_data_type
       INTEGER ( KIND = ip_ ) :: branch = 1
       INTEGER ( KIND = ip_ ) :: eval_status, out, start_print, stop_print
       INTEGER ( KIND = ip_ ) :: advanced_start_iter, lbfgs_mem, max_hist
       INTEGER ( KIND = ip_ ) :: print_level
       INTEGER ( KIND = ip_ ) :: h_ne, len_history, ibound, ipoint, icp
       INTEGER ( KIND = ip_ ) :: nprec, nskip_lbfgs, nskip_prec, it_succ
       INTEGER ( KIND = ip_ ) :: print_gap, max_diffs, latest_diff, total_diffs
       INTEGER ( KIND = ip_ ) :: non_monotone_history, n_s
       INTEGER ( KIND = ip_ ) :: ref( 1 )
       REAL :: time_start, time_record, time_now
       REAL ( KIND = rp_ ) :: clock_start, clock_record, clock_now
       REAL ( KIND = rp_ ) :: f_ref, f_trial, f_best, m_best, model, ratio
       REAL ( KIND = rp_ ) :: old_radius, radius_trial, etat, ometat
       REAL ( KIND = rp_ ) :: dxtdg, dgtdg, df, stg, norm_s, radius_max
       REAL ( KIND = rp_ ) :: stop_g, s_new_norm, F_s, dense_radius
       LOGICAL :: printi, printt, printd, printm
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm
       LOGICAL :: monotone, f_is_nan, non_trivial_p
       LOGICAL :: successful
       LOGICAL :: reverse_f, reverse_g
       CHARACTER ( LEN = 1 ) :: negcur = ' '
       CHARACTER ( LEN = 1 ) :: bndry = ' '
       CHARACTER ( LEN = 1 ) :: perturb = ' '
       CHARACTER ( LEN = 1 ) :: hard = ' '
       CHARACTER ( LEN = 1 ) :: accept = ' '
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_current
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_best
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_s
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_s
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: D_hist
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: F_hist
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: Q
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_dense
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S_dense
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_dense
       TYPE ( SMT_type ) :: H_dense

!  dense data type

       TYPE( GSM_dense_data_type ) :: dense_data

!  copy of controls

       TYPE ( GSM_control_type ) :: control

     END TYPE GSM_data_type

     TYPE, PUBLIC :: GSM_full_data_type
       LOGICAL :: f_indexing = .TRUE.
       TYPE ( GSM_data_type ) :: gsm_data
       TYPE ( GSM_control_type ) :: gsm_control
       TYPE ( GSM_inform_type ) :: gsm_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( GALAHAD_userdata_type ) :: userdata
     END TYPE GSM_full_data_type

   CONTAINS

!-*-*-  G A L A H A D -  G S M _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE GSM_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRU controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_control_type ), INTENT( OUT ) :: control
     TYPE ( GSM_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize TRS components

     CALL TRS_initialize( data%dense_data%TRS_data, control%TRS_control,       &
                          inform%TRS_inform )
     control%TRS_control%prefix = '" - TRS:"                     '

!  initial private data. Set branch for initial entry

     data%branch = 1

     RETURN

!  End of subroutine GSM_initialize

     END SUBROUTINE GSM_initialize

!- G A L A H A D -  G S M _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE GSM_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRU controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_control_type ), INTENT( OUT ) :: control
     TYPE ( GSM_inform_type ), INTENT( OUT ) :: inform

     CALL GSM_initialize( data%gsm_data, data%gsm_control, data%gsm_inform )
     control = data%gsm_control
     inform = data%gsm_inform

     RETURN

!  End of subroutine GSM_full_initialize

     END SUBROUTINE GSM_full_initialize

!-*-*-*-*-   G S M _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE GSM_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by GSM_initialize could (roughly)
!  have been set as:

! BEGIN TRU SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  maximum-number-of-iterations                    100
!  maximum-number-of-inner-iterations              100
!  maximum-subspace-dimension                      2
!  advanced-start                                  5
!  history-length-for-non-monotone-descent         0
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-5
!  minimum-step-allowed                            2.0D-16
!  initial-trust-region-radius                     1.0D+0
!  maximum-trust-region-radius                     1.0D+19
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  too-successful-iteration-tolerance              2.0
!  trust-region-increase-factor                    2.0
!  trust-region-decrease-factor                    0.5
!  trust-region-maximum-decrease-factor            0.0625
!  minimum-objective-before-unbounded              -1.0D+32
!  sr1-update-skip-threshold                       1.0D-8
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  sub-problem-direct                              no
!  renormalize-radius                              no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
!  output-line-prefix                              ""
! END TRU SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_control_type ), INTENT( INOUT ) :: control
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
     INTEGER ( KIND = ip_ ), PARAMETER :: inner_maxit = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_subspace = inner_maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_unit = max_subspace + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: non_monotone = alive_unit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advanced_start = non_monotone + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_g_absolute = advanced_start + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_g_relative = stop_g_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_s = stop_g_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_radius = stop_s + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: maximum_radius = initial_radius + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_successful = maximum_radius + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_very_successful                  &
                                            = eta_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: eta_too_successful                   &
                                            = eta_very_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: radius_increase                      &
                                            = eta_too_successful + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: radius_reduce = radius_increase + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: radius_reduce_max = radius_reduce + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: obj_unbounded = radius_reduce_max + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: sr1_skip = obj_unbounded + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit = sr1_skip + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: renormalize_radius                   &
                                            = clock_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                       &
                                            = renormalize_radius + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_file                           &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = alive_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'GSM '
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
     spec( inner_maxit )%keyword = 'maximum-number-of-inner-iterations'
     spec( max_subspace )%keyword = 'maximum-subspace-dimension'
     spec( alive_unit )%keyword = 'alive-device'
     spec( non_monotone )%keyword = 'history-length-for-non-monotone-descent'
     spec( advanced_start )%keyword = 'advanced-start'

!  Real key-words

     spec( stop_g_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_g_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( initial_radius )%keyword = 'initial-trust-region-radius'
     spec( maximum_radius )%keyword = 'maximum-trust-region-radius'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( eta_too_successful )%keyword = 'too-successful-iteration-tolerance'
     spec( radius_increase )%keyword = 'trust-region-increase-factor'
     spec( radius_reduce )%keyword = 'trust-region-decrease-factor'
     spec( radius_reduce_max )%keyword = 'trust-region-maximum-decrease-factor'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( sr1_skip )%keyword = 'sr1-update-skip-threshold'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( renormalize_radius )%keyword = 'renormalize-radius'
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
     CALL SPECFILE_assign_value( spec( inner_maxit ),                          &
                                 control%inner_maxit,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_subspace ),                         &
                                 control%max_subspace,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( non_monotone ),                         &
                                 control%non_monotone,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( advanced_start ),                       &
                                 control%advanced_start,                       &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( stop_g_absolute ),                      &
                                 control%stop_g_absolute,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_g_relative ),                      &
                                 control%stop_g_relative,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_radius ),                       &
                                 control%initial_radius,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maximum_radius ),                       &
                                 control%maximum_radius,                       &
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
     CALL SPECFILE_assign_value( spec( radius_increase ),                      &
                                 control%radius_increase,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_reduce ),                        &
                                 control%radius_reduce,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_reduce_max ),                    &
                                 control%radius_reduce_max,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sr1_skip ),                             &
                                 control%sr1_skip,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( renormalize_radius ),                   &
                                 control%renormalize_radius,                   &
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


!  Read specfile data for SLS and IR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL TRS_read_specfile( control%TRS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-TRS' )
      ELSE
        CALL TRS_read_specfile( control%TRS_control, device )
      END IF

     RETURN

     END SUBROUTINE GSM_read_specfile

!-*-*-*-  G A L A H A D -  G S M _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE GSM_solve( nlp, control, inform, data, userdata,               &
                           eval_F, eval_G )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  GSM_solve, a first-order trust-region method for finding a local
!    unconstrained minimizer of a given function

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_GSM.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_GSM_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
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
! control is a scalar variable of type GSM_control_type. See GSM_initialize
!  for details
!
! inform is a scalar variable of type GSM_inform_type. On initial entry,
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
!    -7. The objective function appears to be unbounded from below
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
!     2. The user should compute the objective function value f(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The required
!        value should be set in nlp%f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x)
!        - for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
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
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type GSM_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
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
!  data is a scalar variable of type GSM_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be used
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
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, GSM_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   GSM_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GSM_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ic, ir, j, k, l
     REAL ( KIND = rp_ ) :: h_diag
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_sit, char_sit2
     CHARACTER ( LEN = 80 ) :: array_name
!    REAL ( KIND = rp_ ), DIMENSION( nlp%n ) :: V

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective evaluation
       GO TO 20
     CASE ( 30 )  ! initial gradient evaluation
       GO TO 30
     CASE ( 310 ) ! objective or gradient evaluation
       GO TO 310
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  allocate sufficient space for the problem

     array_name = 'gsm: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%Q'
     CALL SPACE_resize_array( nlp%n, control%max_subspace,                     &
            data%Q, inform%status,                                             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%X_s'
     CALL SPACE_resize_array( control%max_subspace, data%X_s,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%G_s'
     CALL SPACE_resize_array( control%max_subspace, data%G_s,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( control%advanced_start > 0 ) THEN
       array_name = 'gsm: data%X_best'
       CALL SPACE_resize_array( nlp%n, data%X_best, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     array_name = 'gsm: data%X_dense'
     CALL SPACE_resize_array( control%max_subspace, data%X_dense,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%S_dense'
     CALL SPACE_resize_array( control%max_subspace, data%S_dense,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%Y_dense'
     CALL SPACE_resize_array( control%max_subspace, data%Y_dense,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     data%H_dense%n = control%max_subspace
     data%H_dense%ne = control%max_subspace * ( control%max_subspace + 1 ) / 2
     array_name = 'gsm: data%H_dense%val'
     CALL SPACE_resize_array( data%H_dense%ne, data%H_dense%val,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980
     CALL SMT_put( data%H_dense%type, 'DENSE', inform%status )

!  ensure that the data is consistent

     data%control = control
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     inform%radius = data%control%initial_radius
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%advanced_start_iter = 0
!    data%lbfgs_mem = MAX( 1, data%control%lbfgs_vectors )
     data%negcur = ' '
     data%it_succ = 0
     data%dense_radius = - one
     data%control%trs_control%dense_factorization = 1

!  decide how much reverse communication is required

     data%reverse_f = .NOT. PRESENT( eval_F )
     data%reverse_g = .NOT. PRESENT( eval_G )

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
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
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

!  allocate further arrays

     IF ( .NOT. data%monotone ) THEN
       array_name = 'gsm: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'gsm: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

! evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, inform%obj )
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     IF ( data%reverse_f ) inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1
!write(6,"(' X ', 5ES12.4)" ) nlp%X( : nlp%n )

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj
!write(6,*) ' objective is NaN? ', data%f_is_nan

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  test to see if the objective appears to be unbounded from below

     IF ( inform%obj < control%obj_unbounded ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' objective value', ES12.4, ' is lower than unbounded limit',  &
         &   ES12.4 )" ) prefix, inform%obj, control%obj_unbounded
       inform%status = GALAHAD_error_unbounded ; GO TO 990
     END IF

     data%f_ref = inform%obj
     IF ( .NOT. data%monotone ) THEN
        data%F_hist = data%f_ref ; data%D_hist = zero ; data%max_hist = 1
     END IF

!  evaluate the gradient of the objective function

     IF ( data%reverse_g ) THEN
       data%branch = 30 ; inform%status = 3 ; RETURN
     ELSE
       CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%G( : nlp%n ) )
     END IF

!  return from reverse communication to obtain the gradient

  30 CONTINUE
     inform%g_eval = inform%g_eval + 1
     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
!    WRITE(6,"( ' g: ', ( 5ES12.4 ) )" ) nlp%G( : nlp%n )

!  reset the initial radius to ||g|| if no sensible value is given

     IF ( data%control%initial_radius <= zero )                                &
       inform%radius = inform%norm_g

!  compute the stopping tolerance

!    write(6,*) ' stop a, r, g ', control%stop_g_absolute,                     &
!                       control%stop_g_relative, inform%norm_g
     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * inform%norm_g )

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start

     IF ( data%printi ) WRITE( data%out, "( /, A, '  Problem: ', A,            &
    &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4, / )" )            &
       prefix, TRIM( nlp%pname ), nlp%n, data%stop_g

     data%n_s = 0 ; data%H_dense%ne = 0

!  ============================================================================
!  Start of main iteration
!  ============================================================================

 100 CONTINUE

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printd = data%set_printd
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1

!  print one-line summary

       IF ( data%printi ) THEN
          IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2100 ) prefix
         END IF
         data%print_1st_header = .FALSE.
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2130 ) prefix, char_iter, data%n_s, data%accept,   &
              data%bndry, data%negcur, data%perturb, inform%obj,               &
              inform%norm_g, data%norm_s, inform%dense_inform%iter,            &
              data%clock_now
         ELSE
           WRITE( data%out, 2140 ) prefix, char_iter, inform%obj,              &
               inform%norm_g, data%clock_now
         END IF
       END IF

!  ============================================================================
!  1. Test for convergence
!  ============================================================================

!  stop if the gradient is small enough

       IF ( inform%norm_g <= data%stop_g ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
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
       IF ( inform%iter > data%control%maxit ) THEN
!      IF ( inform%f_eval > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%f_eval, prefix,               &
           inform%g_eval, prefix, inform%iter, prefix, inform%cg_iter,         &
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
               WRITE( data%out, 2020 ) prefix, nlp%vnames( i ), nlp%X( i ),    &
                                       nlp%G( i )
             END DO
           ELSE
             DO i = ir, ic
               WRITE( data%out, 2030 ) prefix, i, nlp%X( i ), nlp%G( i )
             END DO
           END IF
         END DO
       END IF

!  ============================================================================
!  2. Calculate the new search subspace, Q, by appending the latest gradient,
!     if there is space, and orthogonalising with respect to the existing set
!  ============================================================================

   200 CONTINUE

!  compute the dimension of the space spanned by Q

!      write(6,*) data%n_s, control%max_subspace
       IF ( data%n_s < data%control%max_subspace ) THEN
         data%n_s = data%n_s + 1
       ELSE
         data%n_s = 1 ; data%H_dense%n = 0
       END IF

!  apply modified Gram-Schmidt to orthogonalize the new subspace component,
!  g(x), in Q with respect to the existing ones:
!    q_n_s = g
!    for j = 1 : n_s - 1
!      q_n_s <- q_n_s − ( q_j^T q_n_s ) q_j
!    endfor
!    q_n_s <- q_n_s / || q_n_s ||^2

       data%Q( : nlp%n, data%n_s ) = nlp%G( : nlp%n )
       DO j = 1, data%n_s - 1
         data%Q( : nlp%n, data%n_s ) = data%Q( : nlp%n, data%n_s )             &
           - DOT_PRODUCT( data%Q( : nlp%n, j ), data%Q( : nlp%n, data%n_s ) )  &
             * data%Q( : nlp%n, j )
       END DO
       data%Q( : nlp%n, data%n_s ) = data%Q( : nlp%n, data%n_s )               &
           / TWO_NORM( data%Q( : nlp%n, data%n_s ) )
       write(6,"( ' q_j^Tq_n ', ( 5ES12.4 ) )" ) &
         ( DOT_PRODUCT( data%Q( : nlp%n, j ), data%Q( : nlp%n, data%n_s ) ),   &
           j = 1, data%n_s )

!  ============================================================================
!  3. Find the minimizer of f(x + Q x_s) using a low-dimensional server
!  ============================================================================

!  initialize x_s to zero and F_s and G_s accordingly

       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%X_s = zero
       data%F_s = inform%obj
       DO j = 1, data%n_s
         data%G_s( j ) = DOT_PRODUCT( nlp%G( : nlp%n ), data%Q( : nlp%n, j ) )
       END DO

!  initialize dense H = I

       h_diag = h_diag_initial
       data%H_dense%ne = 0
       DO i = 1, data%n_s - 1
         data%H_dense%ne = data%H_dense%ne + i
         h_diag = MAX( h_diag, data%H_dense%val( data%H_dense%ne ) )
       END DO

       data%H_dense%n = data%n_s
       IF ( reset_to_I ) THEN
         data%H_dense%ne = 0
         DO i = 1, data%n_s
           DO j = 1, i - 1
             data%H_dense%ne = data%H_dense%ne + 1
             data%H_dense%val( data%H_dense%ne ) = 0.0_rp_
           END DO
           data%H_dense%ne = data%H_dense%ne + 1
           data%H_dense%val( data%H_dense%ne ) = h_diag
         END DO


!  initialize the new row of dense H to that of the identity matrix

       ELSE
         DO j = 1, data%n_s - 1
           data%H_dense%ne = data%H_dense%ne + 1
           data%H_dense%val( data%H_dense%ne ) = 0.0_rp_
         END DO
         data%H_dense%ne = data%H_dense%ne + 1
         data%H_dense%val( data%H_dense%ne ) = h_diag
       END IF
!write(6,"( ' ne = ', I0, ' H_initial =', /, ( 5ES12.4 ) )" )
!    data%H_dense%ne, data%H_dense%val( : data%H_dense%ne )

!  subspace problem iteration loop

       inform%status = 1
  300  CONTINUE

!  perform an iteration in the subspace

         CALL GSM_dense_solve( inform%status, data%n_s, data%X_s, data%F_s,    &
                               data%G_s, data%dense_radius, data%control,      &
                               data%dense_data, data%X_dense, data%S_dense,    &
                               data%Y_dense, data%H_dense,                     &
                               inform%dense_inform, inform%TRS_inform )

!  move the iterate back to the full space

         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         DO j = 1, data%n_s
            nlp%X( : nlp%n ) =  nlp%X( : nlp%n )                               &
               + data%Q( 1 : nlp%n, j ) * data%X_s( j )
         END DO

!  react to demands from the subspace solver

         SELECT CASE ( inform%status )

!  evaluate the objective function

         CASE ( 2 )
           IF ( data%reverse_f ) THEN
             data%branch = 310 ; inform%status = 2 ; RETURN
           ELSE
             CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%f )
           END IF

!  evaluate the gradient of the objective function

         CASE ( 3 )
           IF ( data%reverse_g ) THEN
              data%branch = 310 ; inform%status = 3 ; RETURN
           ELSE
             CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%G( : nlp%n ) )
           END IF

!  successful conclusion

         CASE ( 0 )
           GO TO 320

!  error exit

         CASE DEFAULT
           GO TO 320
         END SELECT

!  return from reverse communication with f or g

 310     CONTINUE
         SELECT CASE ( inform%status )
         CASE ( 2 )
           data%F_s = nlp%f
           inform%f_eval = inform%f_eval + 1
         CASE ( 3 )
           DO j = 1, data%n_s
             data%G_s( j )                                                     &
               = DOT_PRODUCT( nlp%G( : nlp%n ), data%Q( : nlp%n, j ) )
           END DO
           inform%g_eval = inform%g_eval + 1
         END SELECT
       GO TO 300

!  end of subproblem iteration loop

 320   CONTINUE
       inform%obj = nlp%f ; inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
!      inform%iter = inform%f_eval
       data%norm_s = TWO_NORM( data%X_s( : data%n_s ) )

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
     GO TO 100

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE
     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) /= 0 ) THEN
       char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
       WRITE( data%out, 2130 ) prefix, char_iter, data%n_s, data%accept,       &
          data%bndry, data%negcur, data%perturb, inform%obj,                   &
          inform%norm_g, data%norm_s, inform%dense_inform%iter, data%clock_now
     END IF

!  print details of solution

     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%f_eval, inform%g_eval,   &
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
       WRITE( data%out, "( /, A, '  Problem: ', A,                             &
      &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4 )" )             &
         prefix, TRIM( nlp%pname ), nlp%n, data%stop_g
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
         data%non_monotone_history
       WRITE( data%out, "( A, '  First-order model used' )" ) prefix
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( control%error > 0 ) THEN
       CALL SYMBOLS_status( inform%status, control%error, prefix, 'GSM_solve' )
       WRITE( control%error, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I0,                         &
              /, A, ' # gradient evaluations  = ', I0,                         &
              /, A, ' # major  iterations     = ', I0,                         &
              /, A, ' # minor (cg) iterations = ', I0,                         &
              /, A, ' objective value         = ', ES21.14,                    &
              /, A, ' gradient norm           = ', ES11.4 )
 2010 FORMAT( /, A, ' name             X         G ' )
 2020 FORMAT(  A, 1X, A10, 2ES12.4 )
 2030 FORMAT(  A, 1X, I10, 2ES12.4 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I0 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2100 FORMAT( A, '    It  n_s          f         grad    ',                    &
             '  step   it_s    time' )
 2130 FORMAT( A, A6, I4, 1X, 4A1, ES12.4, ES11.4, ES8.1, I6, F8.2 )
 2140 FORMAT( A, A6, 4X, 5X, ES12.4, ES11.4, 14X, F8.2 )

 !  End of subroutine GSM_solve

     END SUBROUTINE GSM_solve

!-*-*-  G A L A H A D -  G S M _ s o l v e _ f u l l  S U B R O U T I N E  -*-*-

     SUBROUTINE GSM_solve_full( nlp, control, inform, data, userdata,          &
                               eval_F, eval_G )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  GSM_solve_full, a first-order trust-region method for finding a local
!    unconstrained minimizer of a given function

!  This variant treats the problem as if it were dense

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_GSM.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_GSM_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
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
! control is a scalar variable of type GSM_control_type. See GSM_initialize
!  for details
!
! inform is a scalar variable of type GSM_inform_type. On initial entry,
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
!    -7. The objective function appears to be unbounded from below
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
!     2. The user should compute the objective function value f(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The required
!        value should be set in nlp%f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x)
!        - for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
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
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type GSM_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
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
!  data is a scalar variable of type GSM_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be used
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
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, GSM_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   GSM_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GSM_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ic, ir, j, k, l
     REAL ( KIND = rp_ ) :: ared, prered, rounding
!    REAL ( KIND = rp_ ) :: radmin
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_sit, char_sit2
     CHARACTER ( LEN = 80 ) :: array_name
!    REAL ( KIND = rp_ ), DIMENSION( nlp%n ) :: V

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective evaluation
       GO TO 20
     CASE ( 30 )  ! initial gradient evaluation
       GO TO 30
     CASE ( 300 ) ! objective or gradient evaluation
       GO TO 300
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  allocate sufficient space for the problem

     array_name = 'gsm: data%X_dense'
     CALL SPACE_resize_array( nlp%n, data%X_dense,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%S_dense'
     CALL SPACE_resize_array( nlp%n, data%S_dense,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%Y_dense'
     CALL SPACE_resize_array( nlp%n, data%Y_dense,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     data%H_dense%n = nlp%n
     data%H_dense%ne = nlp%n * ( nlp%n + 1 ) / 2
     array_name = 'gsm: data%H_dense%val'
     CALL SPACE_resize_array( data%H_dense%ne, data%H_dense%val,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980
     CALL SMT_put( data%H_dense%type, 'DENSE', inform%status )

!  ensure that the data is consistent

     data%control = control
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     inform%radius = data%control%initial_radius
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%advanced_start_iter = 0
!    data%lbfgs_mem = MAX( 1, data%control%lbfgs_vectors )
     data%negcur = ' '
     data%it_succ = 0
     data%dense_radius = - one
     data%control%trs_control%dense_factorization = 1

!  decide how much reverse communication is required

     data%reverse_f = .NOT. PRESENT( eval_F )
     data%reverse_g = .NOT. PRESENT( eval_G )

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
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
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

!  allocate further arrays

     IF ( .NOT. data%monotone ) THEN
       array_name = 'gsm: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'gsm: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

! evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%f_eval = inform%f_eval + 1

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj
!write(6,*) ' objective is NaN? ', data%f_is_nan

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  test to see if the objective appears to be unbounded from below

     IF ( inform%obj < control%obj_unbounded ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' objective value', ES12.4, ' is lower than unbounded limit',  &
         &   ES12.4 )" ) prefix, inform%obj, control%obj_unbounded
       inform%status = GALAHAD_error_unbounded ; GO TO 990
     END IF

     data%f_ref = inform%obj
     IF ( .NOT. data%monotone ) THEN
        data%F_hist = data%f_ref ; data%D_hist = zero ; data%max_hist = 1
     END IF

!  evaluate the gradient of the objective function

     IF ( data%reverse_g ) THEN
       data%branch = 30 ; inform%status = 3 ; RETURN
     ELSE
       CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%G( : nlp%n ) )
     END IF

!  return from reverse communication to obtain the gradient

  30 CONTINUE
     inform%g_eval = inform%g_eval + 1
     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
!    WRITE(6,"( ' g: ', ( 5ES12.4 ) )" ) nlp%G( : nlp%n )

!  reset the initial radius to ||g|| if no sensible value is given

     IF ( data%control%initial_radius <= zero )                                &
       inform%radius = inform%norm_g

!  compute the stopping tolerance

     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * inform%norm_g )

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start

     IF ( data%printi ) WRITE( data%out, "( /, A, '  Problem: ', A,            &
    &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4, / )" )            &
       prefix, TRIM( nlp%pname ), nlp%n, data%stop_g

     data%n_s = 0 ; data%H_dense%n = 0

!  initialize dense H = I

     data%H_dense%n = nlp%n
     data%H_dense%ne = 0
     DO i = 1, nlp%n
       DO j = 1, i - 1
         data%H_dense%ne = data%H_dense%ne + 1
         data%H_dense%val( data%H_dense%ne ) = 0.0_rp_
       END DO
       data%H_dense%ne = data%H_dense%ne + 1
       data%H_dense%val( data%H_dense%ne ) = 1.0_rp_
     END DO

!  subspace problem iteration loop

     inform%status = 1
 300 CONTINUE

!  perform an iteration in the subspace

       CALL GSM_dense_solve( inform%status, nlp%n, nlp%X, nlp%f,               &
                             nlp%G, data%dense_radius, data%control,           &
                             data%dense_data, data%X_dense, data%S_dense,      &
                             data%Y_dense, data%H_dense,                       &
                             inform%dense_inform, inform%TRS_inform )

!  react to demands from the subspace solver

       SELECT CASE ( inform%status )

!  evaluate the objective function

       CASE ( 2 )
         IF ( data%reverse_f ) THEN
           data%branch = 300 ; inform%status = 2 ; RETURN
         ELSE
           CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                        nlp%f )
         END IF

!  evaluate the gradient of the objective function

       CASE ( 3 )
         IF ( data%reverse_g ) THEN
            data%branch = 300 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                        nlp%G( : nlp%n ) )
         END IF

!  successful conclusion

       CASE ( 0 )
         GO TO 320

!  error exit

       CASE DEFAULT
         GO TO 320
       END SELECT
     GO TO 300

!  end of subproblem iteration loop

 320 CONTINUE
     inform%obj = nlp%f ; inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
!    inform%iter = inform%f_eval
     data%norm_s = TWO_NORM( nlp%X( : nlp%n ) )

!  record the clock time

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start
     IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,       &
    &    F12.2,  ' seconds' )" ) prefix, data%clock_now
     IF ( ( data%control%cpu_time_limit >= zero .AND.                          &
            data%time_now > data%control%cpu_time_limit ) .OR.                 &
          ( data%control%clock_time_limit >= zero .AND.                        &
            data%clock_now > data%control%clock_time_limit ) ) THEN
       inform%status = GALAHAD_error_cpu_limit ; GO TO 900
     END IF

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE
     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) /= 0 ) THEN
       char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
       WRITE( data%out, 2130 ) prefix, char_iter, data%n_s, data%accept,       &
          data%bndry, data%negcur, data%perturb, inform%obj,                   &
          inform%norm_g, data%norm_s, inform%dense_inform%iter, data%clock_now
     END IF

!  print details of solution

     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%f_eval, inform%g_eval,   &
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
       WRITE( data%out, "( /, A, '  Problem: ', A,                             &
      &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4 )" )             &
         prefix, TRIM( nlp%pname ), nlp%n, data%stop_g
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
         data%non_monotone_history
       WRITE( data%out, "( A, '  First-order model used' )" ) prefix
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( control%error > 0 ) THEN
       CALL SYMBOLS_status( inform%status, control%error, prefix, 'GSM_solve' )
       WRITE( control%error, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I0,                         &
              /, A, ' # gradient evaluations  = ', I0,                         &
              /, A, ' # major  iterations     = ', I0,                         &
              /, A, ' # minor (cg) iterations = ', I0,                         &
              /, A, ' objective value         = ', ES21.14,                    &
              /, A, ' gradient norm           = ', ES11.4 )
 2010 FORMAT( /, A, ' name             X         G ' )
 2020 FORMAT(  A, 1X, A10, 2ES12.4 )
 2030 FORMAT(  A, 1X, I10, 2ES12.4 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I0 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2100 FORMAT( A, '    It  n_s          f         grad    ',                    &
             '  step   it_s    time' )
 2130 FORMAT( A, A6, I4, 1X, 4A1, ES12.4, ES11.4, ES8.1, I6, F8.2 )
 2140 FORMAT( A, A6, 4X, 5X, ES12.4, ES11.4, 14X, F8.2 )

 !  End of subroutine GSM_solve_full

     END SUBROUTINE GSM_solve_full

!-*-  G A L A H A D -  G S M _ d e n s e _ s o l v e _ S U B R O U T I N E -*-

     SUBROUTINE GSM_dense_solve( status, n, X, f, G, radius, control, data,    &
                                 X_current, S, Y, H, inform, trs_inform )

!  find a critical point of f(x), involving n variables x

!  flow controlled by integer status:

!   1  initial input with f = f(x) and G = g(x) evaluated at initial X = x,
!      and a positive radius (a non-positive value will be reset to ||G||)
!   0  termination test satisfied at x = X with f(x) = f and g(x) = G
!   2  return to evaluate f = f(x) at new X = x
!   3  return to evaluate G = g(x) at new X = x
!  <0  error exit

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     REAL ( KIND = rp_ ), INTENT( INOUT ) :: f, radius
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, G
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X_current, S, Y
     TYPE ( GSM_dense_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_dense_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( TRS_inform_type ), INTENT( INOUT ) :: trs_inform
     TYPE ( SMT_type ), INTENT( INOUT) :: H

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, k, alloc_status
     REAL ( KIND = rp_ ) :: ared, prered, df, rounding, yts, yioveryts
     LOGICAL :: f_is_nan
     CHARACTER ( LEN = 80 ) :: bad_alloc, array_name
     TYPE ( TRS_control_type ) :: trs_control

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!   termination may be checked with status = 2, 3 returns

     SELECT CASE ( status )
     CASE ( 2 )  ! objective evaluation
       GO TO 20
     CASE ( 3 )  ! gradient evaluation
       GO TO 30
     CASE DEFAULT ! initial entry
     END SELECT

!  record initial values

     data%f = f
     X_current( : n ) = X( : n )
     data%norm_g = TWO_NORM( G( : n ) )
     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * data%norm_g )
     IF ( radius <= zero ) radius = data%norm_g
     inform%iter = 0

     IF ( control%out > 0 .AND. control%print_level > 1 )                      &
       WRITE( control%out, "( A, '        (a=accept, r=reject, stop_g =',      &
      &  ES11.4, ')', /, A, ' iter       f         ||g||       ratio    ',     &
      &  '    radius', /, A, I6, A1, 2ES12.4, '       -           -' )" )      &
        prefix, data%stop_g, prefix, prefix, inform%iter, 'a', f, data%norm_g

!  ============================================================================
!  MAIN LOOP
!  ============================================================================

  10 CONTINUE
       inform%iter = inform%iter + 1

!  check to see if the iteration limit has not been exceeded

       IF ( inform%iter > control%inner_maxit ) THEN
         status = GALAHAD_error_max_iterations ; RETURN
       END IF

!  ============================================================================
!  1. Calculate the search direction, s
!  ============================================================================

!  use a first-order (steepest descent) model

       IF ( model == first_order_model ) THEN

!  compute the step

         S( : n ) = - ( radius / data%norm_g ) * G( : n )
         data%norm_s = radius

!  record the decrease in the model obtained

         data%model = - radius * data%norm_g

!  use a second-order (symmetric rank-one Hessian approximation) model

       ELSE IF ( model == second_order_model ) THEN

!  compute the step

         CALL TRS_solve( n, radius, 0.0_rp_, G( : n ), H,                 &
                         S( : n ), data%TRS_data,                         &
                         control%trs_control, trs_inform )
         data%norm_s = trs_inform%x_norm

!  record the decrease in the model obtained

         data%model = trs_inform%obj
       END IF

!  ============================================================================
!  2. check for acceptance of the new point
!  ============================================================================

!  check that the correction will make a difference

       IF ( TWO_NORM( S( : n ) / MAX( one, TWO_NORM( X( : n ) ) ) )       &
              <= control%stop_s ) THEN
         status = GALAHAD_error_tiny_step ; RETURN
       END IF

!  record the current point

       X_current( : n )  = X( : n )

!  form the trial point

       X( : n ) = X_current( : n ) + S( : n )
       status = 2 ; RETURN

!  return from reverse communication with the objective value

  20   CONTINUE

!  if the trial objective value is a NaN, reduce the trust region radiius
!  and try again

       f_is_nan = f /= f
       IF ( f_is_nan ) THEN
         radius = control%radius_reduce * data%norm_s
         GO TO 10
       END IF

!  compute the change in objective and the slope

       df = data%f - f

!  compute the ratio of actual to predicted reduction over the current iteration

       rounding = MAX( one, ABS( data%f ) ) * REAL( n, KIND = rp_ ) * epsmch
       ared = df + rounding
       prered = - data%model + rounding
       IF ( ABS( ared ) < teneps .AND. ABS( f ) > teneps ) ared = prered
       data%ratio = ared / prered

!  see if the new point is acceptable

       data%successful = data%ratio >= control%eta_successful

!  evaluate the gradient of the objective function

       IF ( data%successful ) THEN
         Y( : n ) = G( : n )
         data%accept = 'a'
         data%f = f
         status = 3 ; RETURN
       ELSE
         data%accept = 'r'
         X( : n ) = X_current( : n )
       END IF

!  return from reverse communication with the gradient

 30   CONTINUE

!  ============================================================================
!  3. Update the trust-region radius and other book-keeping
!  ============================================================================

!  record information about a successful iteration

       IF ( data%successful ) THEN
         data%norm_g = TWO_NORM( G )
         IF ( control%out > 0 .AND. control%print_level > 1 )                  &
           WRITE( control%out, "( A, I6, A1, 4ES12.4 )" )                      &
             prefix, inform%iter, 'a', f, data%norm_g, data%ratio, radius

!  exit if the termination test is satisfied

         IF ( data%norm_g <= data%stop_g ) THEN
           status = GALAHAD_ok ; RETURN
         END IF

!  update the symmetric rank-one Hessian estimate,
!  H^+ = H + (y - H s) (y - Hs)^T / (y - Hs)^T s
!  if (y - Hs)^T s is not too small

!  compute y = g^+ - g

         Y( : n ) = G( : n ) - Y( : n )

!  overwrite y by y - H s

         k = 0
         DO i = 1, n
           DO j = 1, i - 1
             k = k + 1
             Y( i ) = Y( i ) - H%val( k ) * S( j )
             Y( j ) = Y( j ) - H%val( k ) * S( i )
           END DO
           k = k + 1
           Y( i ) = Y( i ) - H%val( k ) * S( i )
         END DO

!  perform the update if | (y - Hs)^T s | >= tol || y-Hs || || s ||

         yts = DOT_PRODUCT( Y( : n ), S( : n ) )
         IF ( ABS( yts ) > control%sr1_skip * data%norm_s                      &
                             * TWO_NORM( Y( : n ) ) ) THEN
           k = 0
           DO i = 1, n
             yioveryts = Y( i ) / yts
             DO j = 1, i
               k = k + 1
               H%val( k ) = H%val( k ) + yioveryts * Y( j )
             END DO
           END DO
         END IF

!  record information about an unsuccessful iteration

       ELSE
         IF ( control%out > 0 .AND. control%print_level > 1 )                  &
           WRITE( control%out, "( A, I6, A1, ES12.4, 6X, '-', 5X, 2ES12.4 )" ) &
             prefix, inform%iter, 'r', f, data%ratio, radius
       END IF

!  if the iteration has increased the objective, decrease the radius so that
!  had the objective been quadratic the next iteration would be very successful

       IF ( data%ratio < zero ) THEN
         radius = control%radius_reduce * data%norm_s

!  if the iteration was very unsuccesful, decrease the radius to chop off the
!  current step

       ELSE IF ( data%ratio < control%eta_successful ) THEN
         radius = control%radius_reduce * data%norm_s

!  if the iteration was very (but not too) successful, increase the radius

       ELSE
         IF ( data%ratio >= control%eta_very_successful .AND.                  &
              data%ratio <= control%eta_too_successful ) THEN
           IF ( control%radius_increase * data%norm_s > radius )               &
             radius = MIN( control%maximum_radius,                             &
                                control%radius_increase * data%norm_s )
         END IF
       END IF
     GO TO 10

!  end of subroutine GSM_dense_solve

     END SUBROUTINE GSM_dense_solve

!-*-*-  G A L A H A D -  G S M _ s o l v e _ o l d   S U B R O U T I N E  -*-*-

     SUBROUTINE GSM_solve_old( nlp, control, inform, data, userdata,           &
                               eval_F, eval_G )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  GSM_solve, a first-order trust-region method for finding a local
!    unconstrained minimizer of a given function

! *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED ***
! *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED ***
! *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED *** DEPRICATED ***

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_GSM.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_GSM_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
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
! control is a scalar variable of type GSM_control_type. See GSM_initialize
!  for details
!
! inform is a scalar variable of type GSM_inform_type. On initial entry,
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
!    -7. The objective function appears to be unbounded from below
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
!     2. The user should compute the objective function value f(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The required
!        value should be set in nlp%f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x)
!        - for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
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
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type GSM_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
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
!  data is a scalar variable of type GSM_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be used
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
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, GSM_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   GSM_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GSM_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, ic, ir, l
     REAL ( KIND = rp_ ) :: ared, prered, rounding
!    REAL ( KIND = rp_ ) :: radmin
     REAL ( KIND = rp_ ) :: tau, tau_1, tau_2, tau_min, tau_max
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_sit, char_sit2
     CHARACTER ( LEN = 80 ) :: array_name
!    REAL ( KIND = rp_ ), DIMENSION( nlp%n ) :: V

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective evaluation
       GO TO 20
     CASE ( 30 )  ! initial gradient evaluation
       GO TO 30
     CASE ( 320 ) ! objective evaluation
       GO TO 320
     CASE ( 340 ) ! gradient evaluation
       GO TO 340
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

if ( .false. ) then
     CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
     CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,                &
                  nlp%G( : nlp%n ) )
     data%dense_radius = - one
     inform%f_eval = 1 ; inform%g_eval = 1 ; inform%status = 1
     DO
       CALL GSM_dense_solve( inform%status, nlp%n, nlp%X, nlp%f, nlp%G,        &
                             data%dense_radius, control, data%dense_data,      &
                             data%X_dense, data%S_dense,                       &
                             data%Y_dense, data%H_dense,                       &
                             inform%dense_inform, inform%TRS_inform )
       SELECT CASE ( inform%status )
       CASE ( 2 )
         CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
         inform%f_eval = inform%f_eval + 1
       CASE ( 3 )
         CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%G( : nlp%n ) )
         inform%g_eval = inform%g_eval + 1
       CASE ( 0 )
         EXIT
       CASE DEFAULT
         EXIT
       END SELECT
     END DO
     inform%obj = nlp%f ; inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
     inform%iter = inform%f_eval
     RETURN
end if

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  allocate sufficient space for the problem

     array_name = 'gsm: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%S'
     CALL SPACE_resize_array( nlp%n, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%Q'
     CALL SPACE_resize_array( nlp%n, data%control%max_subspace,                &
            data%Q, inform%status,                                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%X_s'
     CALL SPACE_resize_array( data%control%max_subspace, data%X_s,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'gsm: data%G_s'
     CALL SPACE_resize_array( data%control%max_subspace, data%G_s,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( control%advanced_start > 0 ) THEN
       array_name = 'gsm: data%X_best'
       CALL SPACE_resize_array( nlp%n, data%X_best, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  ensure that the data is consistent

     data%control = control
     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     inform%radius = data%control%initial_radius
     data%etat = half * ( data%control%eta_very_successful +                   &
                          data%control%eta_successful )
     data%ometat = one - data%etat
     data%advanced_start_iter = 0
!    data%lbfgs_mem = MAX( 1, data%control%lbfgs_vectors )
     data%negcur = ' '
     data%it_succ = 0
     data%dense_radius = - one

!  decide how much reverse communication is required

     data%reverse_f = .NOT. PRESENT( eval_F )
     data%reverse_g = .NOT. PRESENT( eval_G )

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
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
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

!  allocate further arrays

     IF ( .NOT. data%monotone ) THEN
       array_name = 'gsm: data%F_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%F_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'gsm: data%D_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%D_hist,    &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

! evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, inform%obj )
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     IF ( data%reverse_f ) inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1
!write(6,"(' X ', 5ES12.4)" ) nlp%X( : nlp%n )

!  test to see if the initial objective value is undefined

!    data%f_is_nan = IEEE_IS_NAN( inform%obj )
     data%f_is_nan = inform%obj /= inform%obj
!write(6,*) ' objective is NaN? ', data%f_is_nan

     IF ( data%f_is_nan ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' initial objective value is a NaN' )" ) prefix
       inform%status = GALAHAD_error_evaluation ; GO TO 990
     END IF

!  test to see if the objective appears to be unbounded from below

     IF ( inform%obj < control%obj_unbounded ) THEN
       IF ( data%printi ) WRITE( data%out,                                     &
          "( A, ' objective value', ES12.4, ' is lower than unbounded limit',  &
         &   ES12.4 )" ) prefix, inform%obj, control%obj_unbounded
       inform%status = GALAHAD_error_unbounded ; GO TO 990
     END IF

     data%f_ref = inform%obj
     IF ( .NOT. data%monotone ) THEN
        data%F_hist = data%f_ref ; data%D_hist = zero ; data%max_hist = 1
     END IF

!  evaluate the gradient of the objective function

     IF ( data%reverse_g ) THEN
       data%branch = 30 ; inform%status = 3 ; RETURN
     ELSE
       CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%G( : nlp%n ) )
     END IF

!  return from reverse communication to obtain the gradient

  30 CONTINUE
     inform%g_eval = inform%g_eval + 1
     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
!    WRITE(6,"( ' g: ', ( 5ES12.4 ) )" ) nlp%G( : nlp%n )

!  reset the initial radius to ||g|| if no sensible value is given

     IF ( data%control%initial_radius <= zero )                                &
       inform%radius = inform%norm_g

!  compute the stopping tolerance

!    write(6,*) ' stop a, r, g ', control%stop_g_absolute,                     &
!                       control%stop_g_relative, inform%norm_g
     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * inform%norm_g )

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start

     IF ( data%printi ) WRITE( data%out, "( /, A, '  Problem: ', A,            &
    &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4, / )" )            &
       prefix, TRIM( nlp%pname ), nlp%n, data%stop_g

     data%n_s = 1

!  ============================================================================
!  Start of main iteration
!  ============================================================================

 100 CONTINUE

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printd = data%set_printd
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1

!  print one-line summary

       IF ( data%printi ) THEN
          IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2100 ) prefix
         END IF
         data%print_1st_header = .FALSE.
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2130 ) prefix, char_iter, data%accept,             &
              data%bndry, data%negcur, data%perturb, inform%obj,               &
              inform%norm_g, data%ratio, inform%radius, data%norm_s,           &
              data%clock_now
         ELSE
           WRITE( data%out, 2140 ) prefix, char_iter, inform%obj,              &
               inform%norm_g, inform%radius, data%clock_now
         END IF
       END IF

!  ============================================================================
!  1. Test for convergence
!  ============================================================================

!  stop if the gradient is small enough

       IF ( inform%norm_g <= data%stop_g ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
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
       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%f_eval, prefix,               &
           inform%g_eval, prefix, inform%iter, prefix, inform%cg_iter,         &
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
               WRITE( data%out, 2020 ) prefix, nlp%vnames( i ), nlp%X( i ),    &
                                       nlp%G( i )
             END DO
           ELSE
             DO i = ir, ic
               WRITE( data%out, 2030 ) prefix, i, nlp%X( i ), nlp%G( i )
             END DO
           END IF
         END DO
       END IF

!  ============================================================================
!  2. Calculate the search direction, s
!  ============================================================================

   200 CONTINUE
       data%S( : nlp%n )                                                       &
         = - ( inform%radius / inform%norm_g ) * nlp%G( : nlp%n )
       data%model = - inform%radius * inform%norm_g
       data%norm_s = inform%radius
       data%bndry = 'b'

!  ============================================================================
!  3. check for acceptance of the new point
!  ============================================================================

!  see if the correction will make any difference

       IF ( TWO_NORM( data%S( : nlp%n ) / MAX( one,                            &
            TWO_NORM( nlp%X( : nlp%n ) ) ) ) <= data%control%stop_s ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 900
       END IF

!  compute the slope and curvature along the step

       data%stg = DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )

!  prepare for advanced starting-point calculation if requested

       IF ( inform%iter == 1 .AND. data%control%advanced_start > 0 ) THEN
         data%radius_max = data%control%maximum_radius
!write(6,*) ' radius_max ', data%radius_max
         inform%radius = data%norm_s
         data%X_best( : nlp%n )  = nlp%X( : nlp%n )
         data%f_best = inform%obj
         data%m_best = data%model
       END IF

!  record the current point

       data%X_current( : nlp%n )  = nlp%X( : nlp%n )

!  form the trial point

 310   CONTINUE
       nlp%X( : nlp%n ) = data%X_current( : nlp%n ) + data%S( : nlp%n )
!write(6,"(' X ', 5ES12.4)" ) data%X_current( : nlp%n )
!write(6,"(' S ', 5ES12.4)" ) data%S( : nlp%n )
!write(6,"(' X ', 5ES12.4)" ) nlp%X( : nlp%n )

!  evaluate the objective function at the trial point

       IF ( data%reverse_f ) THEN
         data%branch = 320 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      data%f_trial )
       END IF

!  return from reverse communication to obtain the objective value

 320   CONTINUE
       IF ( data%reverse_f ) data%f_trial = nlp%f
       inform%f_eval = inform%f_eval + 1

!  check to ensure that the trial objective value is not a NaN

!      data%f_is_nan = IEEE_IS_NAN( data%f_trial )
       data%f_is_nan = data%f_trial /= data%f_trial
!write(6,*) ' objective is NaN? ', data%f_is_nan
       IF ( data%f_is_nan ) THEN
         data%accept = 'n'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )

!  control printing for the NaN case

         IF ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print .AND.                              &
              MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) &
             THEN
           data%printi = data%set_printi ; data%printt = data%set_printt
           data%printm = data%set_printm ; data%printd = data%set_printd
           data%print_level = data%control%print_level
         ELSE
           data%printi = .FALSE. ; data%printt = .FALSE.
           data%printm = .FALSE. ; data%printd = .FALSE.
           data%print_level = 0
         END IF
         data%print_iteration_header = data%print_level > 1

!  print one-line summary

         IF ( data%printi ) THEN
            IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
             WRITE( data%out, 2100 ) prefix
           END IF
           data%print_1st_header = .FALSE.
           char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
!            char_sit =
!            char_sit2 =                                                       &
             WRITE( data%out, "( A, A6, 1X, 4A1, '    NaN           -    ',    &
            &  '    - Inf ', '    -    ', ES9.1, 1X, 2A6, F8.2 )" ) prefix,    &
                char_iter, data%accept, data%bndry, data%negcur, data%perturb, &
                inform%radius, char_sit, char_sit2, data%clock_now
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
         IF ( inform%iter > data%control%maxit ) THEN
           inform%status = GALAHAD_error_max_iterations ; GO TO 900
         END IF

!  reduce the trust region radiius and try again

         inform%radius = data%control%radius_reduce * data%norm_s
         GO TO 200
       END IF

!  test to see if the objective appears to be unbounded from below

       IF ( data%f_trial < control%obj_unbounded ) THEN
         inform%obj = data%f_trial
         IF ( data%printi ) WRITE( data%out,                                   &
          "( A, ' objective value', ES12.4, ' is lower than unbounded limit',  &
         &   ES12.4 )" ) prefix, data%f_trial, control%obj_unbounded
         inform%status = GALAHAD_error_unbounded ; GO TO 990
       END IF

!  Advanced starting point
!  .......................

!  if an advanced starting point/radius is desired, proceed as per
!  Sartenaer (SISC 18(6) 1990:1788-1803)

       IF ( inform%iter == 1 .AND. data%control%advanced_start > 0 ) THEN
         data%advanced_start_iter = data%advanced_start_iter + 1

!  If the predicted radius is larger than its upper bound, exit

         IF ( inform%radius >= data%radius_max ) GO TO 330

!  perform another iteration

         IF ( data%advanced_start_iter <= data%control%advanced_start ) THEN

!  compute the change in objective and the slope

           data%df = inform%obj - data%f_trial

!  record any improvement in the objective value

           IF ( data%f_trial < data%f_best ) THEN
             data%X_best( : nlp%n )  = nlp%X( : nlp%n )
             data%f_best = data%f_trial
             data%m_best = data%model
           END IF

!  compute the ratio of actual to predicted reduction over the current iteration

           ared = data%df + MAX( one, ABS( inform%obj ) ) * teneps
           prered = - data%model + MAX( one, ABS( inform%obj ) ) * teneps
           IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )        &
             ared = prered
           data%ratio = ared / prered
           IF ( data%printm ) WRITE( data%out, "( /, A, ' actual, predicted',  &
          & ' reductions = ', 2ES12.4 )" ) prefix, ared, prered

!  compute radius adjustment factors

           tau_1 = - theta * data%stg / ( - ( one - theta ) *                  &
                    ( data%f_trial - inform%obj - data%stg ) )
           tau_2 = theta * data%stg / (  - ( one + theta ) *                   &
                    ( data%f_trial - inform%obj - data%stg ) )
           tau_min = MIN( tau_1, tau_2 )
           tau_max = MAX( tau_1, tau_2 )

!  very good agreement - increase step using Sartenaer's formula

           IF ( ABS( data%ratio - one ) <= mu_2 ) THEN
!write(6,*) ' very good agreement '
             IF ( tau_max < one ) THEN
               tau = gamma_3
             ELSE IF ( tau_max > gamma_4 ) THEN
               tau = gamma_4
             ELSE IF ( tau_1 >= one .AND. tau_1 <= gamma_4 .AND.               &
                       tau_2 < one ) THEN
               tau = tau_1
             ELSE IF ( tau_2 >= one .AND. tau_2 <= gamma_4 .AND.               &
                       tau_1 < one ) THEN
               tau = tau_2
             ELSE
               tau = tau_max
             END IF

!  poor agreement  - decrease step using Sartenaer's formula

           ELSE IF ( ABS( data%ratio - one ) > mu_1 ) THEN
!write(6,*) ' poor agreement '
             IF ( tau_min > one ) THEN
               tau = gamma_2
             ELSE IF ( tau_max < gamma_1 ) THEN
               tau = gamma_1
             ELSE IF ( tau_min < gamma_1 .AND. tau_max >= one ) THEN
               tau = gamma_1
             ELSE IF ( tau_1 >= gamma_1 .AND. tau_1 < one .AND.                &
                     ( tau_2 < gamma_1 .OR. tau_2 >= one ) ) THEN
               tau = tau_1
             ELSE IF ( tau_2 >= gamma_1 .AND. tau_2 < one .AND.                &
                     ( tau_1 < gamma_1 .OR. tau_1 >= one ) ) THEN
               tau = tau_2
             ELSE
               tau = tau_max
             END IF

!  acceptable agreement - refine step

           ELSE
!write(6,*) ' acceptable agreement '
             IF ( tau_max < gamma_2 ) THEN
               tau = gamma_2
             ELSE IF ( tau_max > gamma_3 ) THEN
               tau = gamma_3
             ELSE
               tau = tau_max
             END IF
           END IF

!  restrict any increasze so that the radius does not exceed its maximim value

           tau = MIN( tau, data%radius_max / inform%radius )
!write(6,*) ' tau ', tau

!  update the radius and step length

           data%old_radius = inform%radius
           inform%radius = inform%radius * tau
           data%norm_s = data%norm_s * tau

!  update the slope, curvature and model value

           data%stg = tau * data%stg
           data%model = data%stg

           IF ( data%printi ) THEN
              IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2100 ) prefix
             END IF
             data%print_1st_header = .FALSE.
             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
             data%time_now = data%time_now - data%time_start
             data%clock_now = data%clock_now - data%clock_start
             char_iter = STRING_integer_6( inform%iter +                       &
                                             data%advanced_start_iter )
             WRITE( data%out, 2130 ) prefix, char_iter, data%accept,           &
                data%bndry, data%negcur, data%perturb, inform%obj,             &
                inform%norm_g, data%ratio, inform%radius, data%norm_s,         &
                data%clock_now
           END IF

!  form the next trial step

           data%S( : nlp%n ) = tau * data%S( : nlp%n )
           GO TO 310
         END IF

!  record the best value found

 330     CONTINUE
         inform%iter = inform%iter + data%advanced_start_iter - 1
         IF ( data%f_best < inform%obj ) THEN
           data%S( : nlp%n ) = data%X_best( : nlp%n ) - nlp%X( : nlp%n )
           nlp%X( : nlp%n )  = data%X_best( : nlp%n )
           data%f_trial = data%f_best
           data%model = data%m_best
         END IF
       END IF

!  compute the change in objective and the slope

       data%df = inform%obj - data%f_trial

!  compute the ratio of actual to predicted reduction over the current iteration

       rounding = MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = rp_ )    &
                    * epsmch
       ared = data%df + rounding
       prered = - data%model + rounding
       IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )            &
         ared = prered
       data%ratio = ared / prered
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

       data%successful = data%ratio >= data%control%eta_successful
       IF ( data%successful ) THEN
         data%accept = 'a'
         inform%obj = data%f_trial

!  evaluate the gradient of the objective function

         IF ( data%reverse_g ) THEN
            data%branch = 340 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_G( data%eval_status, nlp%X( : nlp%n ),                    &
                        userdata, nlp%G( : nlp%n ) )
         END IF
       ELSE
         data%accept = 'r'
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
       END IF

!  return from reverse communication to obtain the gradient

 340   CONTINUE
       IF ( data%successful ) THEN
         inform%g_eval = inform%g_eval + 1
         inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

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
!          write( 6, "( ' f, fref ', 2ES12.4 ) " ) inform%obj, data%f_ref
!          write( 6, "( ' fhist ', ( 6ES12.4 ) ) " ) &
!            data%F_hist( data%non_monotone_history + 2 - data%max_hist :      &
!                         data%non_monotone_history + 1 )
         END IF
       END IF

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

!  ============================================================================
!  4. Update the trust-region radius and other book-keeping
!  ============================================================================

       data%old_radius = inform%radius

!  if the iteration has increased the objective, decrease the radius so that
!  had the objective been quadratic the next iteration would be very successful

       IF ( data%ratio < zero ) THEN
         inform%radius = data%control%radius_reduce * data%norm_s

!  if the iteration was very unsuccesful, decrease the radius to chop off the
!  current step

       ELSE IF ( data%ratio < data%control%eta_successful ) THEN
         inform%radius = data%control%radius_reduce * data%norm_s

!  if the iteration was very (but not too) successful, increase the radius

       ELSE
         IF ( data%ratio >= data%control%eta_very_successful .AND.             &
              data%ratio <= data%control%eta_too_successful ) THEN
           IF ( data%control%radius_increase * data%norm_s > inform%radius )   &
             inform%radius = MIN( data%control%maximum_radius,                 &
                                  data%control%radius_increase * data%norm_s )
         END IF
       END IF

!  only add subspace components if the iteration was successful

       IF ( .NOT. data%successful ) GO TO 100

!  add a new subspace component

       IF ( data%n_s < data%control%max_subspace ) THEN
         data%Q( : nlp%n, data%n_s ) = data%S( : nlp%n ) / data%norm_s
         data%n_s = data%n_s + 1
         GO TO 100
       END IF

!  apply modified Gram-Schmidt to orthogonalize the new subspace component
!  against the esisting ones:

! q_n_s = s_n_s
!  for j = 1 : n_s - 1
!   q_n_s <- q_n_s − ( q_j^T q_n_s ) q_j
! endfor
! q_n_s <- q_n_s / || q_n_s ||^2



!  there are sufficieny subspace components; orthogonalize the latest
!  against the existing ones

       data%Q( : nlp%n, data%n_s ) = data%Q( : nlp%n, 1 ) - data%S( : nlp%n )&
          / DOT_PRODUCT( data%Q( : nlp%n, 1 ), data%S( : nlp%n ) )
       data%Q( : nlp%n, data%n_s ) = data%Q( : nlp%n, data%n_s )             &
          / TWO_NORM( data%Q( : nlp%n, data%n_s ) )

!      WRITE(6,"( ' ||s_1||, ||s_2||, s_1^Ts_2 = ', 3ES12.4 )")                &
!        TWO_NORM( data%Q( : nlp%n, 1 ) ),                              &
!        TWO_NORM( data%Q( : nlp%n, 2 ) ),                              &
!        DOT_PRODUCT( data%Q( : nlp%n, 1 ), data%Q( : nlp%n, 2 ) )

!      CALL DGEQRF( M, N,    A, LDA, TAU, WORK, LWORK, INFO )
!      CALL DORGQR( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
!         TAU( min(m,n)), k = min(m,n)

!  compute a subspace minimizer. That is, from the current x = x_k,
!  consider points x = x_k + Q_k x_s, and the subspace function
!  F(x_s) = f(x_k + Q_k x_s) (with gradient G(x_s) = Q_k^T g(x_k + Q_k x_s).
!  Minimize F(x_s) starting from x_s = 0

       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%X_s = zero
       data%F_s = inform%obj
       DO i = 1, data%n_s
         data%G_s( i ) = DOT_PRODUCT( nlp%G( : nlp%n ), data%Q( : nlp%n, i ) )
       END DO
!      CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
!      CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
!                   nlp%G( : nlp%n ) )
!      inform%f_eval = 1 ; inform%g_eval = 1 ;
       inform%status = 1
       DO
         CALL GSM_dense_solve( inform%status, data%n_s, data%X_s, data%F_s,    &
                               data%G_s, data%dense_radius, control,           &
                               data%dense_data, data%X_dense, data%S_dense,    &
                               data%Y_dense, data%H_dense,                     &
                               inform%dense_inform, inform%TRS_inform )
         nlp%X( : nlp%n ) = data%X_current( : nlp%n )
         DO i = 1, data%n_s
            nlp%X( : nlp%n ) =  nlp%X( : nlp%n )                               &
               + data%Q( 1 : nlp%n, i ) * data%X_s( i )
         END DO
         SELECT CASE ( inform%status )
         CASE ( 2 )
           CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
           data%F_s = nlp%f
           inform%f_eval = inform%f_eval + 1
         CASE ( 3 )
           CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,          &
                        nlp%G( : nlp%n ) )
           DO i = 1, data%n_s
             data%G_s( i )                                                     &
               = DOT_PRODUCT( nlp%G( : nlp%n ), data%Q( : nlp%n, i ) )
           END DO
           inform%g_eval = inform%g_eval + 1
         CASE ( 0 )
           EXIT
         CASE DEFAULT
           EXIT
         END SELECT
       END DO
       inform%obj = nlp%f ; inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
       inform%iter = inform%f_eval
       data%n_s = 1
     GO TO 100

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE
     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) /= 0 ) THEN
       char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
       WRITE( data%out, 2130 ) prefix, char_iter, data%accept,                 &
          data%bndry, data%negcur, data%perturb, inform%obj,                   &
          inform%norm_g, data%ratio, inform%radius, data%norm_s,               &
          data%clock_now
     END IF

!  print details of solution

     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2040 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2000 ) inform%f_eval, inform%g_eval,   &
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
       WRITE( data%out, "( /, A, '  Problem: ', A,                             &
      &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4 )" )             &
         prefix, TRIM( nlp%pname ), nlp%n, data%stop_g
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
         data%non_monotone_history
       WRITE( data%out, "( A, '  First-order model used' )" ) prefix
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( control%error > 0 ) THEN
       CALL SYMBOLS_status( inform%status, control%error, prefix, 'GSM_solve' )
       WRITE( control%error, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I0,                         &
              /, A, ' # gradient evaluations  = ', I0,                         &
              /, A, ' # major  iterations     = ', I0,                         &
              /, A, ' # minor (cg) iterations = ', I0,                         &
              /, A, ' objective value         = ', ES21.14,                    &
              /, A, ' gradient norm           = ', ES11.4 )
 2010 FORMAT( /, A, ' name             X         G ' )
 2020 FORMAT(  A, 1X, A10, 2ES12.4 )
 2030 FORMAT(  A, 1X, I10, 2ES12.4 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I0 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2100 FORMAT( A, '    It           f         grad    ',                        &
             ' ratio   radius   step   h_error    time' )
 2130 FORMAT( A, A6, 1X, 4A1, ES12.4, ES11.4, ES9.1, 2ES8.1, 8X, F8.2 )
 2140 FORMAT( A, A6, 5X, ES12.4, ES11.4, 9X, ES8.1, 16X, F8.2 )

 !  End of subroutine GSM_solve_old

     END SUBROUTINE GSM_solve_old

!-*-*-  G A L A H A D -  G S M _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE GSM_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'gsm: data%X_best'
     CALL SPACE_dealloc_array( data%X_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%X_current'
     CALL SPACE_dealloc_array( data%X_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%D_hist'
     CALL SPACE_dealloc_array( data%D_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%F_hist'
     CALL SPACE_dealloc_array( data%F_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

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

!  End of subroutine GSM_terminate

     END SUBROUTINE GSM_terminate

!-  G A L A H A D -  G S M _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE GSM_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL GSM_terminate( data%gsm_data, data%gsm_control, data%gsm_inform )
     inform = data%gsm_inform

!  deallocate any internal problem arrays

     array_name = 'gsm: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%Z'
     CALL SPACE_dealloc_array( data%nlp%Z,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%H%row'
     CALL SPACE_dealloc_array( data%nlp%H%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%H%col'
     CALL SPACE_dealloc_array( data%nlp%H%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%H%ptr'
     CALL SPACE_dealloc_array( data%nlp%H%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%H%val'
     CALL SPACE_dealloc_array( data%nlp%H%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'gsm: data%nlp%H%type'
     CALL SPACE_dealloc_array( data%nlp%H%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine GSM_full_terminate

     END SUBROUTINE GSM_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  G S M _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE GSM_import( control, data, status, n )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to GSM_solve
!
!  data is a scalar variable of type GSM_full_data_type used for internal data
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
!   -3. The restriction n > 0 or requirement that H_type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'ABSENT' has been violated.
!  -79. An optional array required by storage type H_type is missing
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_control_type ), INTENT( INOUT ) :: control
     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%gsm_control = control

     error = data%gsm_control%error
     space_critical = data%gsm_control%space_critical
     deallocate_error_fatal = data%gsm_control%deallocate_error_fatal

!  allocate space if required

     array_name = 'gsm: data%nlp%X'
     CALL SPACE_resize_array( n, data%nlp%X,                                   &
            data%gsm_inform%status, data%gsm_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%gsm_inform%bad_alloc, out = error )
     IF ( data%gsm_inform%status /= 0 ) GO TO 900

     array_name = 'gsm: data%nlp%G'
     CALL SPACE_resize_array( n, data%nlp%G,                                   &
            data%gsm_inform%status, data%gsm_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%gsm_inform%bad_alloc, out = error )
     IF ( data%gsm_inform%status /= 0 ) GO TO 900

     data%nlp%n = n

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status =  data%gsm_inform%status
     RETURN

!  End of subroutine GSM_import

     END SUBROUTINE GSM_import

!-  G A L A H A D -  G S M _ r e s e t _ c o n t r o l   S U B R O U T I N E  -

     SUBROUTINE GSM_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See GSM_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_control_type ), INTENT( IN ) :: control
     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%gsm_control = control

!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine GSM_reset_control

     END SUBROUTINE GSM_reset_control

! - G A L A H A D -  G S M _ s o l v e _ d i r e c t   S U B R O U T I N E -

     SUBROUTINE GSM_solve_direct( data, userdata, status, X, G, eval_F, eval_G )

!  solve the unconstrained problem previously imported when accessto the
!  function and gradient are available via subroutine calls. See GSM_solve for
!  a description of the required arguments. The variable status is a proxy
!  for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_F, eval_G

     data%gsm_inform%status = status
     IF ( data%gsm_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     CALL GSM_solve( data%nlp, data%gsm_control, data%gsm_inform,              &
                     data%gsm_data, userdata, eval_F = eval_F,                 &
                     eval_G = eval_G )

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     IF ( data%gsm_inform%status == GALAHAD_ok )                               &
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     status = data%gsm_inform%status

     RETURN

!  end of subroutine GSM_solve_direct

     END SUBROUTINE GSM_solve_direct

!-  G A L A H A D -  G S M _ s o l v e _ r e v e r s e _ S U B R O U T I N E -

     SUBROUTINE GSM_solve_reverse( data, status, eval_status, X, f, G )

!  solve the unconstrained problem previously imported when access to the
!  function and gradient are available via reverse communication. See
!  GSM_solve for a description of the required arguments. The variable status
!  is a proxy for inform%status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: eval_status
     REAL ( KIND = rp_ ), INTENT( IN ) :: f
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: G

!  recover data from reverse communication

     data%gsm_inform%status = status
     data%gsm_data%eval_status = eval_status
     SELECT CASE ( data%gsm_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%gsm_data%eval_status = eval_status
       IF ( eval_status == 0 ) data%nlp%f = f
     CASE( 3 )
       data%gsm_data%eval_status = eval_status
       IF ( eval_status == 0 ) data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
     END SELECT

!  call the solver

     CALL GSM_solve( data%nlp, data%gsm_control, data%gsm_inform,              &
                     data%gsm_data, data%userdata )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%gsm_inform%status )
     CASE( 0 )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 5 )
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%gsm_inform%status
     END SELECT
     status = data%gsm_inform%status

     RETURN

!  end of subroutine GSM_solve_reverse

     END SUBROUTINE GSM_solve_reverse

!-  G A L A H A D -  G S M _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE GSM_information( data, inform, status )

!  return solver information during or after solution by TRU
!  See GSM_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( GSM_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GSM_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%gsm_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine GSM_information

     END SUBROUTINE GSM_information

!  End of module GALAHAD_GSM

   END MODULE GALAHAD_GSM_precision



!for j=1:n
!   v_j=x_j
!endfor
!for j = 1:n
!  q_j=v_j / || v_j ||^2
!  for  k=j+1:n
!     v_k = v_k− ( q_j^T v_k ) q_j
!  endfor
!endfor

