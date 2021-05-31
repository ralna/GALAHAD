! THIS VERSION: GALAHAD 2.1 - 17/10/2007 AT 12:00 GMT.

!-*-*-*-*-  G A L A H A D _ F U N N E L _ E Q U A L I T Y   M O D U L E -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.1. October 17th 2007

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FUNNEL_equality_double

!     ----------------------------------------------------------
!    |                                                          |
!    | FUNNEL_equality, a trust-funnel method for               |
!    |  equality-constrained nonlinear optimization             |
!    |                                                          |
!    | Aim: to find a (local) minimizer of the nonlinear        |
!    | programming problem                                      |
!    |                                                          |
!    |  minimize               f (x)                            |
!    |  subject to           c_i (x)  =  0       i in E         |
!    |                                                          |
!     ----------------------------------------------------------

     USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SPACE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_LLS_double
     USE GALAHAD_EQP_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_NORMS_double, ONLY: NORM => TWO_NORM
     USE GALAHAD_FUNNEL_double, ONLY: FUNNEL_control_type, FUNNEL_time_type,   &
                                      FUNNEL_inform_type, FUNNEL_data_type,    &
                                      FUNNEL_initialize, FUNNEL_read_specfile, &
                                      FUNNEL_terminate
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: FUNNEL_initialize, FUNNEL_read_specfile,                        &
               FUNNEL_equality_solve, FUNNEL_terminate,                        &
               NLPT_problem_type

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     INTEGER, PARAMETER :: wsout = 0
!    INTEGER, PARAMETER :: wsout = 78

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: mu_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: y_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: z_tiny = ten ** ( - 6 )

!    LOGICAL, PARAMETER :: print_debug = .TRUE.
     LOGICAL, PARAMETER :: print_debug = .FALSE.

!    LOGICAL, PARAMETER :: nop = .TRUE.
     LOGICAL, PARAMETER :: nop = .FALSE.

!  ================================
!  The FUNNEL_equality_data_type derived type
!  ================================

     TYPE, PUBLIC :: FUNNEL_equality_data_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_diag
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_n
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: T
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: N
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_plus
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_soc
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_plus
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_soc
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_mod
       TYPE ( QPT_problem_type ) :: prob
       TYPE ( LLS_data_type ) :: LLS_data
       TYPE ( EQP_data_type ) :: EQP_data
       TYPE ( SMT_type ) :: J, A_active, H, K
     END TYPE FUNNEL_equality_data_type

!  ===================================
!  The FUNNEL_equality_control_type derived type
!  ===================================

     TYPE, PUBLIC :: FUNNEL_equality_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap

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

       REAL ( KIND = wp ) :: initial_t_model_radius, initial_n_model_radius
       REAL ( KIND = wp ) :: eta_successful, eta_very_successful
       LOGICAL :: use_second_order_correction, fulsol, space_critical
       LOGICAL :: deallocate_error_fatal
       CHARACTER ( LEN = 30 ) :: alive_file
       TYPE ( LLS_control_type ) :: LLS_control_n
       TYPE ( LLS_control_type ) :: LLS_control_y
       TYPE ( LLS_control_type ) :: LLS_control_s
       TYPE ( EQP_control_type ) :: EQP_control
     END TYPE FUNNEL_equality_control_type

!  ================================
!  The FUNNEL_EQUALITY_time_type derived type
!  ================================

     TYPE, PUBLIC :: FUNNEL_equality_time_type
       REAL :: total, preprocess, analyse, factorize, solve
     END TYPE FUNNEL_equality_time_type

!  ==================================
!  The FUNNEL_EQUALITY_inform_type derived type
!  ==================================

     TYPE, PUBLIC :: FUNNEL_equality_inform_type
       INTEGER :: status, alloc_status, iter, cg_iter
       INTEGER :: f_eval, g_eval, factorizations_normal, modifications
       INTEGER :: factorization_status
       INTEGER :: factorization_integer, factorization_real
       REAL ( KIND = wp ) :: obj, primal_infeasibility
       REAL ( KIND = wp ) :: dual_infeasibility, complementary_slackness
       CHARACTER ( LEN = 80 ) :: bad_alloc
       TYPE ( LLS_inform_type ) :: LLS_inform_n
       TYPE ( LLS_inform_type ) :: LLS_inform_y
       TYPE ( LLS_inform_type ) :: LLS_inform_s
       TYPE ( EQP_inform_type ) :: EQP_inform
       TYPE ( FUNNEL_EQUALITY_time_type ) :: time
     END TYPE FUNNEL_equality_inform_type

   CONTAINS

!!-*  G A L A H A D -  F U N N E L _ I N I T I A L I Z E  S U B R O U T I N E  *-
!
!     SUBROUTINE FUNNEL_equality_initialize( data, control, inform )
!
!!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!!   Provide default values for FUNNEL controls
!
!!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!!-----------------------------------------------
!!   D u m m y   A r g u m e n t s
!!-----------------------------------------------
!
!     TYPE ( FUNNEL_data_type ), INTENT( OUT ) :: data
!     TYPE ( FUNNEL_control_type ), INTENT( OUT ) :: control
!     TYPE ( FUNNEL_inform_type ), INTENT( OUT ) :: inform
!
!     inform%status = GALAHAD_ok
!
!!  Intialize LLS data
!
!     CALL LLS_initialize( data%LLS_data, control%LLS_control_n,                &
!                          inform%LLS_inform_n )
!     control%LLS_control_n%prefix = '" - LLS:"                     '
!     control%LLS_control_n%SBLS_control%prefix =                               &
!       '" -- SBLS:"                   '
!     control%LLS_control_y = control%LLS_control_n
!     control%LLS_control_s = control%LLS_control_n
!
!!  Intialize EQP data
!
!     CALL EQP_initialize( data%EQP_data, control%EQP_control,                  &
!                          inform%EQP_inform )
!     control%EQP_control%prefix = '" - EQP:"                     '
!     control%EQP_control%SBLS_control%prefix = '" -- SBLS:"                   '
!
!!  Error and ordinary output unit numbers
!
!     control%error = 6
!     control%out = 6
!
!!  Removal of the file alive_file from unit alive_unit causes execution
!!  to cease
!
!     control%alive_unit = 60
!     control%alive_file = 'ALIVE.d'
!
!!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!!  summary for every iteration, = 2 gives a summary of the inner iteration
!!  for each iteration, >= 3 gives increasingly verbose (debugging) output
!
!     control%print_level = 0
!
!!  Maximum number of iterations
!
!     control%maxit = 1
!
!!   Any printing will start on this iteration
!
!     control%start_print = - 1
!
!!   Any printing will stop on this iteration
!
!     control%stop_print = - 1
!
!!   Printing will only occur every print_gap iterations
!
!     control%print_gap = 1
!
!!   overall convergence tolerances. The iteration will terminate when the norm
!!    of violation of the constraints (the "primal infeasibility") is smaller
!!    than stop_p, the norm of the gradient of the Lagrangian function (the
!!    "dual infeasibility") is smaller than stop_d, and the norm of the
!!    complementary slackness is smaller than stop_c
!
!     control%stop_abs_p = tenm5
!     control%stop_abs_d = tenm5
!     control%stop_abs_c = tenm5
!
!!  The iteration will stop at a minimizer of the infeasibility if the
!!  gradient of the infeasibility (J^T c) is smaller in norm than
!!  control%stop_i times the norm of c
!
!     control%stop_abs_i = tenm5
!
!!  Initial values for the trust-region radiii for the objective and contraint
!!  models
!
!     control%initial_t_model_radius = ten
!     control%initial_n_model_radius = ten
!
!!  A potential point whose linear decrease predicted by the RLP
!!  is larger than the above will only be accepted if the actual decrease
!!  f - f(x_new) is larger than control%eta_successful times that predicted
!!  by a quadratic model of the decrease.
!
!      control%eta_successful = ten ** ( - 8 )
!      control%eta_very_successful = point9
!
!!  Use a second-order correction if necessary
!
!     control%use_second_order_correction = .TRUE.
!!    control%use_second_order_correction = .FALSE.
!
!!  Print the full solution or only highlights
!
!     control%fulsol = .TRUE.
!
!!  If space_critical is true, every effort will be made to use as little
!!  space as possible. This may result in longer computation times
!
!     control%space_critical = .FALSE.
!
!!   If deallocate_error_fatal is true, any array/pointer deallocation error
!!     will terminate execution. Otherwise, computation will continue
!
!     control%deallocate_error_fatal  = .FALSE.
!
!     RETURN
!
!!  End of subroutine FUNNEL_equality_initialize
!
!     END SUBROUTINE FUNNEL_equality_initialize

!!-*-*-   F U N N E L _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-
!
!     SUBROUTINE FUNNEL_equality_read_specfile( control, device, alt_specname )
!
!!  Reads the content of a specification file, and performs the assignment of
!!  values associated with given keywords to the corresponding control parameters
!
!!  The default values as given by FUNNEL_equality_initialize could (roughly)
!!  have been set as:
!
!! BEGIN FUNNEL_EQUALITY SPECIFICATIONS (DEFAULT)
!!  error-printout-device                           6
!!  printout-device                                 6
!!  alive-device
!!  print-level                                     1
!!  maximum-number-of-iterations                    50
!!  start-print                                     22
!!  stop-print                                      20
!!  iterations-between-printing                     1
!!  primal-accuracy-required                        1.0D-5
!!  dual-accuracy-required                          1.0D-5
!!  complementarity-accuracy-required               1.0D-5
!!  relative-infeasiblity-tolerated                 1.0D-5
!!  initial-f-model-radius                          1.0D+1
!!  initial-c-model-radius                          1.0D+1
!!  successful-iteration-tolerance                  0.01
!!  very-successful-iteration-tolerance             0.9
!!  maximum-infeasibility                           10.0
!!  use-second-order-correction                     no
!!  print-full-solution                             no
!!  space-critical                                  no
!!  deallocate-error-fatal                          no
!!  alive-filename                                  ALIVE.d
!! END FUNNEL_EQUALITY SPECIFICATIONS
!
!!-----------------------------------------------
!!   D u m m y   A r g u m e n t s
!!-----------------------------------------------
!
!     TYPE ( FUNNEL_control_type ), INTENT( INOUT ) :: control
!     INTEGER, INTENT( IN ) :: device
!     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname
!
!!  Programming: Nick Gould and Ph. Toint, January 2002.
!
!!-----------------------------------------------
!!   L o c a l   V a r i a b l e s
!!-----------------------------------------------
!
!     INTEGER, PARAMETER :: lspec = 62
!     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'FUNNEL          '
!     CHARACTER( LEN = 16 ), PARAMETER :: specname_n = 'LLS-normal      '
!     CHARACTER( LEN = 16 ), PARAMETER :: specname_y = 'LLS-multiplier  '
!     CHARACTER( LEN = 16 ), PARAMETER :: specname_s = 'LLS-soc         '
!
!     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
!
!!  Define the keywords
!
!     spec%keyword = ''
!
!!  Integer key-words
!
!     spec(  1 )%keyword = 'error-printout-device'
!     spec(  2 )%keyword = 'printout-device'
!     spec(  3 )%keyword = 'alive-device'
!     spec(  4 )%keyword = 'print-level'
!     spec(  5 )%keyword = 'maximum-number-of-iterations'
!     spec(  6 )%keyword = 'start-print'
!     spec(  7 )%keyword = 'stop-print'
!     spec(  8 )%keyword = 'iterations-between-printing'
!
!!  Real key-words
!
!     spec( 17 )%keyword = 'primal-accuracy-required'
!     spec( 18 )%keyword = 'dual-accuracy-required'
!     spec( 19 )%keyword = 'complementarity-accuracy-required'
!     spec( 27 )%keyword = 'relative-infeasiblity-tolerated'
!     spec( 23 )%keyword = 'successful-iteration-tolerance'
!     spec( 24 )%keyword = 'very-successful-iteration-tolerance'
!     spec( 25 )%keyword = 'initial-f-model-radius'
!     spec( 26 )%keyword = 'initial-c-model-radius'
!
!!  Logical key-words
!
!     spec( 31 )%keyword = 'use-second-order-correction'
!     spec( 32 )%keyword = 'space-critical'
!     spec( 35 )%keyword = 'deallocate-error-fatal'
!     spec( 57 )%keyword = 'print-full-solution'
!
!!  Character key-words
!
!     spec( lspec )%keyword = 'alive-filename'
!
!!  Read the specfile
!
!     IF ( PRESENT( alt_specname ) ) THEN
!       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
!     ELSE
!       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
!     END IF
!
!!  Interpret the result
!
!!  Set integer values
!
!     CALL SPECFILE_assign_integer( spec( 1 ), control%error,                   &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 2 ), control%out,                     &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 3 ), control%out,                     &
!                                   control%alive_unit )
!     CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,             &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 5 ), control%maxit,                   &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,             &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,              &
!                                   control%error )
!     CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,               &
!                                   control%error )
!!  Set real values
!
!     CALL SPECFILE_assign_real( spec( 17 ), control%stop_abs_p,                &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 18 ), control%stop_abs_d,                &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 19 ), control%stop_abs_c,                &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 27 ), control%stop_abs_i,                &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 23 ), control%eta_successful,            &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 24 ), control%eta_very_successful,       &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 25 ), control%initial_t_model_radius,    &
!                                control%error )
!     CALL SPECFILE_assign_real( spec( 26 ), control%initial_n_model_radius,    &
!                                control%error )
!
!!  Set logical values
!
!     CALL SPECFILE_assign_logical( spec( 31 ),                                 &
!                                   control%use_second_order_correction,        &
!                                   control%error )
!     CALL SPECFILE_assign_logical( spec( 32 ), control%space_critical,         &
!                                   control%error )
!     CALL SPECFILE_assign_logical( spec( 35 ),                                 &
!                                   control%deallocate_error_fatal,             &
!                                   control%error )
!     CALL SPECFILE_assign_logical( spec( 57 ), control%fulsol,                 &
!                                   control%error )
!
!!  Set character values
!
!     CALL SPECFILE_assign_string( spec( lspec ), control%alive_file,           &
!                                  control%error )
!
!!  Set LLS and EQP control values
!
!     CALL LLS_read_specfile( control%LLS_control_n, device,                    &
!                             alt_specname = specname_n )
!     CALL LLS_read_specfile( control%LLS_control_y, device,                    &
!                             alt_specname = specname_y )
!     CALL LLS_read_specfile( control%LLS_control_s, device,                    &
!                             alt_specname = specname_s )
!     CALL EQP_read_specfile( control%EQP_control, device )
!
!     RETURN
!
!     END SUBROUTINE FUNNEL_equality_read_specfile

!-*-*-*-  G A L A H A D -  F U N N E L _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE FUNNEL_equality_solve( nlp, control, inform, data, userdata,   &
                                       eval_FC, eval_GJ, eval_HL )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  FUNNEL_equality_solve, a method for finding a local minimizer of a function
!  subject to general equality constraints on the variables.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

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

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ir, ic, j, l, out, eval_stat
     INTEGER :: start_print, stop_print, print_level
     INTEGER :: print_level_llsn, print_level_llsn_sbls, print_level_llsn_gltr
     INTEGER :: print_level_llsy, print_level_llsy_sbls, print_level_llsy_gltr
     INTEGER :: print_level_llss, print_level_llss_sbls, print_level_llss_gltr
     INTEGER :: print_level_eqp, print_level_eqp_sbls, print_level_eqp_gltr
     INTEGER :: dim_m, dim_n
!    INTEGER :: iterative_solver

     REAL ( KIND = wp ) :: f, f_plus, f_soc, max_y, val
     REAL ( KIND = wp ) :: norm_n, norm_t, norm_r, norm_s, q_n, q_t, q_y, q_s
     REAL ( KIND = wp ) :: theta, theta_max, theta_plus, theta_soc, ratio, pi
     REAL ( KIND = wp ) :: delta_c, delta_f, delta_ft, m_xps, m_xpn
     REAL ( KIND = wp ) :: radius, radius_c, radius_f, radius_within
     REAL ( KIND = wp ) :: kappa_n, kappa_ca, kappa_cr, kappa_b
     REAL ( KIND = wp ) :: kappa_tx1, kappa_tx2, kappa_delta
     REAL ( KIND = wp ) :: eta_1, eta_2, eta_3, gamma_1, gamma_3
!    REAL ( KIND = wp ) :: d1, d2, rat
     LOGICAL :: set_printt, set_printi, set_printw, set_printd, print_1st_header
     LOGICAL :: set_printm, printe, printi, printt, printm, printw, printd
     LOGICAL :: print_iteration_header
     LOGICAL :: try_soc, use_alt_y

     CHARACTER ( LEN = 1 ) :: it_type, n_end, t_end, s_end, suc
     CHARACTER ( LEN = 80 ) :: array_name
     REAL :: time_new, time_total

!  Parameters

     REAL ( KIND = wp ), PARAMETER :: delta = point01

!  Initialize

     CALL CPU_TIME( time_total ) ; inform%time%total = time_total

! compute the problem dimensions

     dim_n = nlp%n ; dim_m = nlp%m

     inform%iter = 0 ; inform%factorizations_normal = 0
     inform%modifications = 0 ; inform%f_eval = 0 ; inform%g_eval = 0

     f = HUGE( one ) ; inform%obj = f
     inform%primal_infeasibility = HUGE( one )
     inform%dual_infeasibility = HUGE( one )
     inform%complementary_slackness = zero
     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     it_type = ' '

!  ===========================
!  Control the output printing
!  ===========================

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

     out = control%out

     print_level_llsn = control%LLS_control_n%print_level
     print_level_llsn_sbls = control%LLS_control_n%SBLS_control%print_level
     print_level_llsn_gltr = control%LLS_control_n%GLTR_control%print_level
     print_level_llsy = control%LLS_control_y%print_level
     print_level_llsy_sbls = control%LLS_control_y%SBLS_control%print_level
     print_level_llsy_gltr = control%LLS_control_y%GLTR_control%print_level
     print_level_llss = control%LLS_control_s%print_level
     print_level_llss_sbls = control%LLS_control_s%SBLS_control%print_level
     print_level_llss_gltr = control%LLS_control_s%GLTR_control%print_level
     print_level_eqp = control%EQP_control%print_level
     print_level_eqp_sbls = control%EQP_control%SBLS_control%print_level
     print_level_eqp_gltr = control%EQP_control%GLTR_control%print_level

     printe = control%error > 0 .AND. control%print_level >= 0

!  Basic single line of output per iteration

     set_printi = out > 0 .AND. control%print_level >= 1

!  As per printi, but with additional timings for various operations

     set_printt = out > 0 .AND. control%print_level >= 2

!  As per printt, but with checking of residuals, etc

     set_printm = out > 0 .AND. control%print_level >= 3

!  As per printm but also with an indication of where in the code we are

     set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

     set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

     IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
       printi = set_printi ; printt = set_printt
       printm = set_printm ; printw = set_printw ; printd = set_printd
       print_level = control%print_level
     ELSE
       printi = .FALSE. ; printt = .FALSE.
       printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
       print_level = 0
     END IF
     print_1st_header = .TRUE.

     IF ( printd ) WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  nlp%X( : dim_n )

! set constants

     kappa_n = 100.0_wp
     kappa_ca = 1000.0_wp
     kappa_cr = two
     kappa_b = point9
     kappa_delta = point1
     eta_1 = point1
     eta_2 = point9
     eta_3 = half
     gamma_1 = half
     gamma_3 = two
     kappa_tx1 = point9
     kappa_tx2 = half

     try_soc = control%use_second_order_correction
     IF ( try_soc ) THEN
       gamma_1 = 0.1_wp
       eta_2 = 0.1_wp
       gamma_3 = 2000.0_wp
     END IF

     nlp%Z = zero

! ------------------------
!  Step 0: Initialization
! ------------------------

! check for faulty input dimensions

     IF ( dim_m < 0 .OR. dim_n < 0 .OR. dim_m > dim_n ) THEN
       IF ( printe ) WRITE( control%error,                                     &
          "( ' - the problem dimensions are faulty ' )" )
       inform%status = - 4
       RETURN
     END IF

! check that the problem only involves equality constraints

     DO i = 1, dim_n
       IF ( nlp%X_l( i ) > - infinity .OR. nlp%X_u( i ) < infinity ) THEN
         IF ( printe ) WRITE( control%error,                                   &
           "( ' - the problem contains simple-bound constraints  ' )" )
!        inform%status = - 20
!        RETURN
         EXIT
       END IF
     END DO

     DO i = 1, dim_m
       IF ( nlp%C_l( i ) /= nlp%C_u( i ) ) THEN
         IF ( printe ) WRITE( control%error,                                   &
           "( ' - the problem contains inequality constraints  ' )" )
         inform%status = - 21
         RETURN
       END IF
     END DO

!  Allocate space to hold the problem data

     array_name = 'funnel: data%G_n'
     CALL SPACE_resize_array( dim_n, data%G_n, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%R'
     CALL SPACE_resize_array( dim_n, data%R, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%PROD'
     CALL SPACE_resize_array( dim_n, data%PROD, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%G_l'
     CALL SPACE_resize_array( dim_n, data%G_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%X_plus'
     CALL SPACE_resize_array( dim_n, data%X_plus, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%X_soc'
     CALL SPACE_resize_array( dim_n, data%X_soc, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_plus'
     CALL SPACE_resize_array( dim_m, data%C_plus, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%Y'
     CALL SPACE_resize_array( dim_m, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_soc'
     CALL SPACE_resize_array( dim_m, data%C_soc, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%C_mod'
     CALL SPACE_resize_array( dim_m, data%C_mod, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%S'
     CALL SPACE_resize_array( dim_n, data%S, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%N'
     CALL SPACE_resize_array( dim_n, data%N, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'funnel: data%T'
     CALL SPACE_resize_array( dim_n, data%T, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

! evaluate the objective and constraint values at the initial point

     CALL eval_FC( eval_stat, nlp%X, userdata, inform%obj, nlp%C )
     inform%f_eval = inform%f_eval + 1

! compute the violation

     theta = half * DOT_PRODUCT( nlp%C, nlp%C )

! evaluate the initial funnel radius

     theta_max = max( kappa_ca, kappa_cr * theta )

! Record the initial trust-region radii

     radius_f = control%initial_t_model_radius
     radius_c = control%initial_n_model_radius

! record the problem, variable and constraint  names

!   iterative_solver = 0
!   IF ( iterative_solver == 1 ) THEN
!      IF ( printi ) WRITE( out, "( ' GMRES solver ' )" )
!    ELSE
       IF ( printi ) WRITE( out, "( ' GLTR solver ' )" )
!    END IF

!  evaluate the gradient of the objective function and the Jacobian
!  of the constraints

     CALL eval_GJ( eval_stat, nlp%X( : dim_n ), userdata,                      &
                   nlp%G( : dim_n ), nlp%J%val( : nlp%J%ne ) )
     inform%g_eval = inform%g_eval + 1

!  evaluate the gradient of the Lagrangian

     data%G_l = nlp%G - nlp%Z
     DO l = 1, nlp%J%ne
       i = nlp%J%col( l )
       data%G_l( i ) = data%G_l( i )                                           &
         - nlp%J%val( l ) * nlp%Y( nlp%J%row( l ) )
     END DO

!  evaluate the initial Hessian approximation

     CALL eval_HL( eval_stat, nlp%X, nlp%Y, userdata, nlp%H%val )

     inform%primal_infeasibility = NORM( nlp%C )
     inform%dual_infeasibility = NORM( data%G_l )

     IF ( printi ) THEN
       WRITE( out, "( /, ' Problem: ', A )" ) nlp%pname
       WRITE( out, 2040 )
       print_1st_header = .FALSE.
       WRITE( out, "( I6, 6X, ES14.6, ES12.5, '     -      ', 3ES9.2, I6 )" )  &
         0, inform%obj, inform%primal_infeasibility, radius_f, radius_c, theta_max
     END IF

     data%Y = nlp%Y

!  set the stopping tolerances

     data%stop_p = MAX( control%stop_abs_p, control%stop_rel_p *               &
                        inform%primal_infeasibility )
     data%stop_d = MAX( control%stop_abs_d, control%stop_rel_d *               &
                        inform%dual_infeasibility )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               S T A R T    O F    M A I N    I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     DO

       inform%iter = inform%iter + 1
       IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
         printi = set_printi ; printt = set_printt
         printm = set_printm ; printw = set_printw ; printd = set_printd
         print_level = control%print_level
         control%LLS_control_n%print_level = print_level_llsn
         control%LLS_control_n%SBLS_control%print_level = print_level_llsn_sbls
         control%LLS_control_n%GLTR_control%print_level = print_level_llsn_gltr
         control%LLS_control_y%print_level = print_level_llsy
         control%LLS_control_y%SBLS_control%print_level = print_level_llsy_sbls
         control%LLS_control_y%GLTR_control%print_level = print_level_llsy_gltr
         control%LLS_control_s%print_level = print_level_llss
         control%LLS_control_s%SBLS_control%print_level = print_level_llss_sbls
         control%LLS_control_s%GLTR_control%print_level = print_level_llss_gltr
         control%EQP_control%print_level = print_level_eqp
         control%EQP_control%SBLS_control%print_level = print_level_eqp_sbls
         control%EQP_control%GLTR_control%print_level = print_level_eqp_gltr
       ELSE
         printi = .FALSE. ; printt = .FALSE.
         printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
         print_level = 0
         control%LLS_control_n%print_level = 0
         control%LLS_control_n%SBLS_control%print_level = 0
         control%LLS_control_n%GLTR_control%print_level = 0
         control%LLS_control_y%print_level = 0
         control%LLS_control_y%SBLS_control%print_level = 0
         control%LLS_control_y%GLTR_control%print_level = 0
         control%LLS_control_s%print_level = 0
         control%LLS_control_s%SBLS_control%print_level = 0
         control%LLS_control_s%GLTR_control%print_level = 0
         control%EQP_control%print_level = 0
         control%EQP_control%SBLS_control%print_level = 0
         control%EQP_control%GLTR_control%print_level = 0
       END IF
       print_iteration_header = print_level > 1 .OR.                           &
         control%LLS_control_n%print_level > 0 .OR.                            &
         control%LLS_control_y%print_level > 0 .OR.                            &
         control%LLS_control_s%print_level > 0 .OR.                            &
         control%EQP_control%print_level > 0

       radius = MIN( radius_f, radius_c )

!  test for optimality

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%dual_infeasibility <= data%stop_d ) THEN
         inform%status = 0
         EXIT
       END IF

!  test that the iteration limit has not been reached

       IF ( inform%iter > control%maxit ) THEN
         inform%status = - 10
         WRITE( out, "( /, ' - the iteration limit has been reached ' )" )
         EXIT
       END IF

! ---------------------
!  Step 1: Normal step
! ---------------------

!  Compute the normal step as the minimizer of || c + J n || subject
!  to || n || <= min( radius_c, kappa_n * norm(c) ) for large
!  problems this should be done approximately

!      IF ( inform%primal_infeasibility <= control%stop_p .AND.                &
!           MOD( inform%iter, 10 ) == 0 ) THEN
!      IF ( inform%primal_infeasibility <= zero ) THEN
       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%dual_infeasibility >= ( ten ** 4 ) * data%stop_d ) THEN
         data%N = zero
         norm_n = zero
         n_end = '0'
         m_xpn = zero
       ELSE

!  Find the normal step

         control%LLS_control_n%radius                                          &
           = min( radius_c, kappa_n * inform%primal_infeasibility )
         IF ( printt ) WRITE( control%out,                                     &
           "( /, ' * entering LLS for normal step: radius = ', ES9.2 )" )      &
            control%LLS_control_n%radius

         IF ( printw ) WRITE( out, "( ' enter LLS ' )" )

         CALL LLS_solve_main( dim_n, dim_m, nlp%J, nlp%C, q_n, data%N,         &
                              data%LLS_data, control%LLS_control_n,            &
                              inform%LLS_inform_n )
         IF ( printw ) WRITE( out, "( ' exit LLS ' )" )

!  If required, summarize the LLS iteration

         IF ( printt ) THEN
           IF ( control%LLS_control_n%out > 0 .AND.                            &
                control%LLS_control_n%print_level > 0 )                        &
             WRITE( control%out, "( '' )" )
           WRITE( control%out, "( ' - on exit from LLS: status = ', I0,        &
          & ', cg iter = ', I0, ', time = ', F0.2, /,                          &
          & ' - LLS objective value =', ES12.4,                                &
          & ', || n || = ', ES10.4 )" )                                        &
                inform%LLS_inform_n%status, inform%LLS_inform_n%cg_iter,       &
                inform%LLS_inform_n%time%total, inform%LLS_inform_n%obj,       &
                inform%LLS_inform_n%norm_x
         END IF

!  Record the normal step, n

         norm_n = NORM( data%N )
         IF ( inform%LLS_inform_n%status == 0 ) THEN
           n_end = 'c'
           IF ( ABS( inform%LLS_inform_n%norm_x -                              &
                control%LLS_control_n%radius ) <= epsmch ** 0.75 ) n_end = 'b'
         ELSE
           n_end = ':'
         END IF

!  compute the model of f after the normal step

         data%PROD = zero
         DO l = 1, nlp%H%ne
           i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
           data%PROD( i ) = data%PROD( i )  + val * data%N( j )
           IF ( i /= j ) data%PROD( j ) = data%PROD( j )  + val * data%N( i )
         END DO

         m_xpn = DOT_PRODUCT( nlp%G, data%N ) +                                &
                   half * DOT_PRODUCT( data%N, data%PROD )
       END IF
! -------------------------
!  Step 2: Tangential step
! -------------------------

       IF ( norm_n <= kappa_b * radius ) THEN

!  Step 2.1: compute Lagrange multiplier estimates and dual residuals
!  by minimizing ||g + H n + J^T y ||

!  Form g_n = g + H n

         IF ( norm_n > zero ) THEN
           data%G_n = nlp%G + data%PROD
         ELSE
           data%G_n = nlp%G
         END IF

!  Swap the row and column indices of J -> J^T

         DO i = 1, nlp%J%ne
           j = nlp%J%row( i )
           nlp%J%row( i ) = nlp%J%col( i )
           nlp%J%col( i ) = j
         END DO

!  Find the first-order multiplier estimates

         control%LLS_control_y%radius = infinity
         IF ( printt ) WRITE( control%out,                                     &
           "( /, ' * entering LLS for multipliers: radius = ', ES9.2 )" )      &
            control%LLS_control_y%radius

         IF ( printw ) WRITE( out, "( ' enter LLS ' )" )
         CALL LLS_solve_main( dim_m, dim_n, nlp%J, data%G_n, q_y, nlp%Y,       &
                              data%LLS_data, control%LLS_control_y,            &
                              inform%LLS_inform_y )
         IF ( printw ) WRITE( out, "( ' exit LLS ' )" )

!  If required, summarize the LLS iteration

         IF ( printt ) THEN
           IF ( control%LLS_control_y%out > 0 .AND.                            &
                control%LLS_control_y%print_level > 0 )                        &
             WRITE( control%out, "( '' )" )
           WRITE( control%out, "( ' - on exit from LLS: status = ', I0,        &
          & ', cg iter = ', I0, ', time = ', F0.2, /,                          &
          & ' - LLS objective value =', ES12.4,                                &
          & ', || y || = ', ES10.4 )" )                                        &
                inform%LLS_inform_y%status, inform%LLS_inform_y%cg_iter,       &
                inform%LLS_inform_y%time%total, inform%LLS_inform_y%obj,       &
                inform%LLS_inform_y%norm_x
         END IF

!  Swap back the row and column indices of J^T -> J

         DO i = 1, nlp%J%ne
           j = nlp%J%row( i )
           nlp%J%row( i ) = nlp%J%col( i )
           nlp%J%col( i ) = j
         END DO

!  Record the first-order multiplier estimates, y

         nlp%Y = - nlp%Y

!  Compute the dual residuals, r

         data%R = data%G_n
         DO l = 1, nlp%J%ne
           i = nlp%J%col( l )
           data%R( i ) = data%R( i ) + nlp%J%val( l ) * nlp%Y( nlp%J%row( l ) )
         END DO

!        DO i = 1, maxit_refine
!          resid = J * data%R
!          dy = - JJT \ resid
!          nlp%Y = nlp%Y + dy
!          data%R = data%G_n + J.' * nlp%Y
!        END DO

!  compute the dual optimality measure

         norm_r = NORM( data%R )
         IF ( norm_r > zero ) THEN
           pi = ABS( DOT_PRODUCT( data%G_n, data%R ) ) / norm_r
         ELSE
           pi = zero
         END IF

!  Step 2.2: if the dual optimality measure is sufficiently large,
!  compute a suitable tangential step t to minimize a quadratic
!  model of the objective function so that J t = 0 and ||n+t|| <= radius

         IF ( pi > FORCING( 3, theta ) ) THEN
           radius_within = radius - norm_n

!          IF ( iterative_solver == 0 ) THEN

!  Use an EQP iteration - set up the input data

             data%C_plus( : dim_m ) = zero
             data%T( : dim_n ) = zero
             data%Y( : dim_m ) = zero

!  Debug if necessary

             IF ( printd ) THEN
               WRITE( out, "( /, ' EQP subproblem ' )" )
               WRITE( out, "( ( ' i, X, G ', I6, 2ES12.4 ) )" )                &
                 ( i, data%T( i ), data%G_n( i ),  i = 1, dim_n )
               IF ( nlp%m > 0 ) THEN
                 WRITE( out, "( ( ' i, C, Y ', I6, 2ES12.4 ) )" ) ( i,         &
                   data%C_plus( i ), data%Y( i ), i = 1, nlp%m )
                 WRITE( out, "( ' J: row, col, val ', /, 3( 2I6, ES12.4 ) )" ) &
                   ( nlp%J%row( i ), nlp%J%col( i ), nlp%J%val( i ),           &
                     i = 1, nlp%J%ne )
               END IF
               WRITE( out, "( ' H: row, col, val ', /, 3( 2I6, ES12.4 ) )" )   &
                 ( nlp%H%row( i ), nlp%H%col( i ), nlp%H%val( i ),             &
                   i = 1, nlp%H%ne )
             END IF

!  Call the EQP solver

             control%EQP_control%radius = radius_within
             IF ( printt ) WRITE( control%out,                                 &
               "( /, ' * entering EQP for tangential step: radius = ', ES9.2)")&
                control%EQP_control%radius

             IF ( printw ) WRITE( out, "( ' enter EQP ' )" )
             CALL EQP_solve_main( dim_n, dim_m, nlp%H, data%G_n, zero, nlp%J,  &
                                  q_t, data%T, data%Y, data%EQP_data,          &
                                  control%EQP_control, inform%EQP_inform )
             IF ( printw ) WRITE( out, "( ' exit EQP ' )" )

!  Record the tangential step, t

             norm_t = NORM( data%T )

!  If required, summarize the EQP iteration

             IF ( printt ) THEN
               IF ( control%EQP_control%out > 0 .AND.                          &
                    control%EQP_control%print_level > 0 )                      &
                 WRITE( control%out, "( '' )" )
               WRITE( control%out, "( ' - on exit from EQP: status = ', I0,    &
              & ', cg iter = ', I0, ', time = ', F0.2, /,                      &
              & ' - EQP objective decrease =', ES12.4,                         &
              & ', || t || = ', ES10.4 )" )                                    &
                    inform%EQP_inform%status, inform%EQP_inform%cg_iter,       &
                    inform%EQP_inform%time%total, - inform%EQP_inform%obj,     &
                    norm_t
             END IF

             IF ( inform%EQP_inform%status == 0 ) THEN
               t_end = 'c'
               IF ( ABS( norm_t -                                              &
                   control%EQP_control%radius ) <= epsmch ** 0.75 ) t_end = 'b'
               IF (  inform%EQP_inform%GLTR_inform%negative_curvature )        &
                 t_end = '-'
               IF ( inform%EQP_inform%GLTR_inform%status ==                    &
                    GALAHAD_error_max_iterations ) t_end = '>'
             ELSE
               t_end = ':'
             END IF
!          ELSE
!            tgmres_print_level = print_level - 1
!            [ tgmres_status, data%T, data%prob%Y ] = ...
!              tgmres( g, H, J, tgmres_print_level )
!!                       restart, tol, maxit, t0, y0 )
!            IF ( tgmres_status == 0 ) THEN
!              t_end = 'c'
!            ELSE IF ( tgmres_status == 1 ) THEN
!              t_end = 'b'
!            ELSE IF ( tgmres_status == 2 ) THEN
!              t_end = '-'
!            ELSE IF ( tgmres_status == 3 ) THEN
!              t_end = '>'
!            ELSE
!              t_end = ':'
!            END IF
!          END IF
           data%S = data%N + data%T
           norm_s = NORM( data%S )

           data%PROD = zero
           DO l = 1, nlp%H%ne
             i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
             data%PROD( i ) = data%PROD( i )  + val * data%S( j )
             IF ( i /= j ) data%PROD( j ) = data%PROD( j )  + val * data%S( i )
           END DO

           m_xps = DOT_PRODUCT( nlp%G, data%S ) +                              &
                     half * DOT_PRODUCT( data%S, data%PROD )
         ELSE
           t_end = ':'
           norm_t = zero
           data%S = data%N
           norm_s = norm_n
           data%Y = nlp%Y
           m_xps = m_xpn
         END IF
       ELSE
         t_end = ':'
         nlp%Y = zero
         data%Y = nlp%Y
         norm_t = zero
         data%S = data%N
         norm_s = norm_n
         m_xps = m_xpn
       END IF
       s_end = ' '

!  compute the trial point

       data%X_plus = nlp%X + data%S

! evaluate the objective and constraint values at the trial point

       CALL eval_FC( eval_stat, data%X_plus, userdata, f_plus, data%C_plus )
       inform%f_eval = inform%f_eval + 1

! compute the violation at the trial point

       theta_plus = half * DOT_PRODUCT( data%C_plus, data%C_plus )

!  compute the model of f at the trial point

       data%PROD = zero
       DO l = 1, nlp%H%ne
         i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
         data%PROD( i ) = data%PROD( i )  + val * data%S( j )
         IF ( i /= j ) data%PROD( j ) = data%PROD( j )  + val * data%S( i )
       END DO
       m_xps = DOT_PRODUCT( nlp%G, data%S ) +                                  &
                 half * DOT_PRODUCT( data%S, data%PROD )

!  compute the improvement in the model of f at the trial point

       delta_f = - m_xps

!  compute the improvement in the model of f at the trial point
!  over that after the normal step

       delta_ft = m_xpn - m_xps

!  compute the improvement in the model of m at the trial point

       data%C_mod = nlp%C
       DO l = 1, nlp%J%ne
         i = nlp%J%row( l )
         data%C_mod( i ) = data%C_mod( i )                                     &
            + nlp%J%val( l ) * data%S( nlp%J%col( l ) )
       END DO
       delta_c = theta - half * DOT_PRODUCT( data%C_mod, data%C_mod )

!  see if this is an f- or c-iteration

       IF ( ( nlp%m == 0 .OR. norm_t > zero ) .AND.                            &
            delta_f >= FORCING( 2, theta ) .AND.                               &
            delta_f >= kappa_delta * delta_ft  .AND.                           &
            theta_plus <= theta_max ) THEN

! ---------------------
!  Step 3: f-iteration
! ---------------------

         it_type = 'f'
         IF ( printm ) WRITE( out, "( '  f-iteration ' )" )

!  Step 3.1: check the trial point for acceptability

         ratio = ( inform%obj - f_plus ) / delta_f
         IF ( printm )  WRITE( out, "( '  ratio =', ES12.4 )" ) ratio
         IF ( ratio >= eta_1 ) THEN
           suc = 's'
           nlp%X = data%X_plus
           inform%obj = f_plus
           nlp%C = data%C_plus
           theta = theta_plus

!  Step 3.2: update radius_f

           IF ( ratio >= eta_2 ) THEN
             radius_f = min( max( radius_f, gamma_3 * norm_s ), ten ** 10 )
!            radius_f = min( gamma_3 * radius_f, ten ** 10 )
             suc = 'v'
           END IF

! Maybe update radius_c

           IF ( theta_plus < eta_3 * theta_max ) THEN
!            radius_c = min( gamma_3 * radius_c, ten ** 10 )
             radius_c = min( max( radius_c, gamma_3 * norm_n ), ten ** 10 )
           END IF
         ELSE

! hack!
! try a 2nd-order correction

!  Compute the 2nd-order correction as the minimizer of || c(x+s) + J n ||
!  subject to || n || <= radius_c for large problems this should be
!  done approximately

!          try_soc = t_end == 'c'
           IF ( try_soc ) THEN

!  Find the second-order correction

!            control%LLS_control_s%radius = radius_c
             control%LLS_control_s%radius = infinity
             IF ( printt ) WRITE( control%out,                                 &
               "( /, ' * entering LLS for 2nd-order correction: radius = ',    &
              &   ES9.2 )" ) control%LLS_control_s%radius

             IF ( printw ) WRITE( out, "( ' enter LLS ' )" )
             CALL LLS_solve_main( dim_n, dim_m, nlp%J, data%C_plus, q_s,       &
                                  data%X_soc, data%LLS_data,                   &
                                  control%LLS_control_s, inform%LLS_inform_s )
             IF ( printw ) WRITE( out, "( ' exit LLS ' )" )

!  If required, summarize the LLS iteration

             IF ( printt ) THEN
               IF ( control%LLS_control_s%out > 0 .AND.                        &
                    control%LLS_control_s%print_level > 0 )                    &
                 WRITE( control%out, "( '' )" )
               WRITE( control%out, "( ' - on exit from LLS: status = ', I0,    &
              & ', cg iter = ', I0, ', time = ', F0.2, /,                      &
              & ' - LLS objective value =', ES12.4,                            &
              & ', || d_lls || = ', ES10.4 )" )                                &
                    inform%LLS_inform_s%status, inform%LLS_inform_s%cg_iter,   &
                    inform%LLS_inform_s%time%total, inform%LLS_inform_s%obj,   &
                    inform%LLS_inform_s%norm_x
             END IF

!  Record the second-order correction

             data%X_soc = data%X_plus + data%X_soc

             IF ( inform%LLS_inform_s%status == 0 ) THEN
               s_end = 'c'
             ELSE
               s_end = ':'
             END IF

! evaluate the objective and constraint values at the corrected point

             CALL eval_FC( eval_stat, data%X_soc, userdata, f_soc, data%C_soc )
             inform%f_eval = inform%f_eval + 1
             theta_soc = half * DOT_PRODUCT( data%C_soc, data%C_soc )
             ratio = ( inform%obj - f_soc ) / delta_f

!            d1 = ABS( m_xps - f_soc )
!            d2 = ABS( m_xps - f_plus )
!            rat = d1 / d2

             IF ( ratio >= eta_1 .AND. theta_soc <= theta_max ) THEN
               suc = '2'
               nlp%X = data%X_soc
               inform%obj = f_soc
               nlp%C = data%C_soc
               theta = theta_soc
             ELSE
               radius_f = gamma_1 * radius_f
               suc = 'u'
             END IF
           ELSE
             radius_f = gamma_1 * radius_f
             suc = 'u'
           END IF
         END IF

       ELSE

! ---------------------
!  Step 4: c-iteration
! ---------------------

         it_type = 'c'
         IF ( printm ) THEN
           IF ( norm_t == zero )                                               &
             WRITE( out, "( '  c-iteration because norm(t) =  0.0 ' )" )
           IF ( delta_f < FORCING( 2, theta ) )                                &
             WRITE( out, "( '  c-iteration because delta_f =', ES10.2,         &
            &       ' < forcing = ', ES9.2 )" ) delta_f, FORCING( 2, theta )
           IF ( delta_f < kappa_delta * delta_ft )                             &
             WRITE( out, "( '  c-iteration because delta_f =', ES10.2,         &
            &       ' < frac * delta_ft = ', ES9.2 )" ) delta_f,               &
                kappa_delta * delta_ft
           IF ( theta_plus > theta_max )                                       &
             WRITE( out, "( '  c-iteration because theta_+ =', ES10.2,         &
            &       ' > theta_max = ', ES9.2 )" ) theta_plus, theta_max
         END IF

!  Step 4.1: check the trial point for acceptability

         IF ( delta_c < 0.0 .AND.  printi )                                    &
           WRITE( out, "( ' ** warning delta_c is negative **' )" )

         ratio = ( theta - theta_plus + epsmch ) / ( delta_c + epsmch )
         IF ( printm )  WRITE( out, "( '  ratio =', ES12.4 )" ) ratio
         IF ( ratio >= eta_1 ) THEN
           nlp%X = data%X_plus
           inform%obj = f_plus
           nlp%C = data%C_plus
           suc = 's'

!  Step 4.2: update radius_c

           IF ( ratio >= eta_2 ) THEN
!            radius_c = MIN( gamma_3 * radius_c, ten ** 10 )
             radius_c = MIN( MAX( radius_c, gamma_3 * norm_n ), ten ** 10 )
             suc = 'v'
           END IF

!  Step 4.3: update theta_max

           theta_max = MAX( kappa_tx1 * theta_max, kappa_tx2 * theta +         &
                            ( one - kappa_tx2 ) * theta_plus )
           theta = theta_plus
         ELSE
           radius_c = gamma_1 * radius_c
           suc = 'u'
         END IF

       END IF

! ---------------------
!  Step 5: book keeping
! ---------------------

!  evaluate the gradient of the objective function and the Jacobian
!  of the constraints

       IF ( ratio >= eta_1 ) THEN

!  evaluate the gradient of the objective function and the Jacobian
!  of the constraints

         CALL eval_GJ( eval_stat, nlp%X( : dim_n ), userdata,                  &
                       nlp%G( : dim_n ), nlp%J%val( : nlp%J%ne ) )
         inform%g_eval = inform%g_eval + 1
       END IF

!  evaluate the gradient of the Lagraingian function

       data%G_l = nlp%G - nlp%Z
       DO l = 1, nlp%J%ne
         i = nlp%J%col( l )
         data%G_l( i )                                                         &
           = data%G_l( i ) - nlp%J%val( l ) * nlp%Y( nlp%J%row( l ) )
       END DO

!  evaluate the Hessian of the Lagrangian function

       use_alt_y = .FALSE.
!      use_alt_y = NORM( data%Y ) /= zero
       IF ( use_alt_y ) THEN
         CALL eval_HL( eval_stat, nlp%X, data%Y, userdata, nlp%H%val )
       ELSE
         CALL eval_HL( eval_stat, nlp%X, nlp%Y, userdata, nlp%H%val )
       END IF

       inform%primal_infeasibility = NORM( nlp%C )
       inform%dual_infeasibility = NORM( data%G_l )

       IF ( printi ) THEN
         IF ( print_iteration_header .OR. print_1st_header ) WRITE( out, 2040 )
         print_1st_header = .FALSE.
         WRITE( out, "( I6, 2A1, 1X, 3A1, ES14.6, 2ES12.5, 3ES9.2, I6 )" )     &
            inform%iter, it_type, suc, n_end, t_end, s_end, inform%obj,        &
            inform%primal_infeasibility, inform%dual_infeasibility, radius_f,  &
            radius_c, theta_max
       END IF

!  evaluate the gradient of the infeasibilities

       IF ( inform%primal_infeasibility > data%stop_p ) THEN
         data%R = zero
         DO l = 1, nlp%J%ne
           i = nlp%J%col( l )
           data%R( i ) = data%R( i ) + nlp%J%val( l ) * nlp%C( nlp%J%row( l ) )
         END DO
         norm_r = NORM( data%R )
!          write(6,*) ' norm_r ', norm_r
         IF ( norm_r <= control%stop_abs_i * inform%primal_infeasibility ) THEN
           inform%status = - 6
           WRITE( out, "( /, ' - the constraints appear locally infeasible' )" )
           EXIT
         END IF
       END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    M A I N    I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     END DO

!  -------------
!  Normal return
!  -------------

!  Print the solution

     l = 2
     IF ( control%fulsol ) l = dim_n
     IF ( control%print_level >= 10 ) l = dim_n

     WRITE( out, 2000 )
     DO j = 1, 2
       IF ( j == 1 ) THEN
         ir = 1 ; ic = MIN( l, dim_n )
       ELSE
         IF ( ic < dim_n - l ) WRITE( out, 2030 )
         ir = MAX( ic + 1, dim_n - ic + 1 ) ; ic = dim_n
       END IF
       DO i = ir, ic
         WRITE( out, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ), nlp%X_l( i ),      &
           nlp%X_u( i ), nlp%Z( i )
       END DO
     END DO

     IF ( dim_m > 0 ) THEN
       l = 2
       IF ( control%fulsol ) l = dim_m
       IF ( control%print_level >= 10 ) l = dim_m

       WRITE( out, 2010 )
       DO j = 1, 2
         IF ( j == 1 ) THEN
           ir = 1 ; ic = MIN( l, dim_m )
         ELSE
           IF ( ic < dim_m - l ) WRITE( out, 2030 )
           ir = MAX( ic + 1, dim_m - ic + 1 ) ; ic = dim_m
         END IF
         DO i = ir, ic
           WRITE( out, 2020 ) i, nlp%CNAMES( i ), nlp%C( i ), nlp%C_l( i ),    &
             nlp%C_u( i ), nlp%Y( i )
         END DO
       END DO
     END IF

     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total

     IF ( dim_m > 0 ) THEN ; max_y = MAXVAL( ABS( nlp%Y ) ) ;
       ELSE ; max_y = zero ; END IF

     WRITE( out, "( /, ' Problem: ', 16X, A10,                                 &
    &          '     Solver: ', 9X, '  Funnel', /,                             &
    &  ' n              =     ',bn, I12, '       m               = ',bn, I12,/,&
    &  ' Objective      = ', ES16.8, '       Complementarity = ', ES12.4, /,   &
    &  ' Violation      =     ',ES12.4, '       Dual infeas     = ', ES12.4, /,&
    &  ' Max multiplier =     ',ES12.4, '       Max dual var.   = ', ES12.4, /,&
    &  ' Iterations     =     ',bn, I12, '       Function evals  = ',bn, I12,/,&
    &  ' Gradient evals =     ',bn, I12, '       Time            = ', F12.2 )")&
      nlp%pname, dim_n, dim_m, inform%obj, inform%complementary_slackness,     &
      inform%primal_infeasibility, inform%dual_infeasibility, max_y,           &
      MAXVAL( ABS( nlp%Z ) ), inform%iter, inform%f_eval, inform%g_eval,       &
      inform%time%total
     WRITE( out, "( '' )" )

     IF ( inform%status /= 0 ) THEN
       WRITE( control%error, "( ' ** Message from -FUNNEL_equality_solve- ',   &
      &  '    Error exit (status = ', I6, ')', / )" ) inform%status
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

 910 CONTINUE
     inform%status = - 1
     CALL CPU_TIME( time_new ); inform%time%total = time_new - inform%time%total
     RETURN

!  Non-executable statements

 2000 FORMAT( /,' Solution: ', /,'                        ',                   &
                '        <------ Bounds ------> ', /                           &
                '      # name          value   ',                              &
                '    Lower       Upper       Dual ' )
 2010 FORMAT( /,' Constraints: ', /, '                        ',               &
                '        <------ Bounds ------> ', /                           &
                '      # name           value   ',                             &
                '    Lower       Upper    Multiplier ' )
 2020 FORMAT( I7, 1X, A10, 4ES12.4 )
 2030 FORMAT( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2040 FORMAT( /, '  iter   nts       f          ||c||    ',                    &
                 ' ||g+JTy||  radius_f radius_c theta_mx' )

!  End of subroutine FUNNEL_equality_solve

     CONTAINS

! -*-*-*-*-*-*-*- F O R C I N G   I N T E R N A L   F U N C T I O N -*-*-*-*-*-

       FUNCTION FORCING( number, argument )

!  Evaluates the forcing function min( |argument|, argument^2 )

!  Dummy arguments

       REAL ( KIND = wp ) :: FORCING
       INTEGER, INTENT( IN ) :: number
       REAL ( KIND = wp ), INTENT( IN ) :: argument

!  Local variable

       SELECT CASE ( number )
       CASE ( 1 )
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       CASE ( 2 )
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       CASE DEFAULT
!        FORCING = point01 * MIN( point01, MIN( ABS( argument ), argument ** 2))
         FORCING = point01 * MIN( one, MIN( ABS( argument ), argument ** 2 ) )
       END SELECT

!  End of function FORCING

       END FUNCTION FORCING

     END SUBROUTINE FUNNEL_equality_solve

!!-*-  G A L A H A D -  F U N N E L _ t e r m i n a t e  S U B R O U T I N E -*-
!
!     SUBROUTINE FUNNEL_equality_terminate( data, control, inform )
!
!!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!!   Deallocate all private storage
!
!!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!!-----------------------------------------------
!!   D u m m y   A r g u m e n t s
!!-----------------------------------------------
!
!     TYPE ( FUNNEL_data_type ), INTENT( INOUT ) :: data
!     TYPE ( FUNNEL_control_type ), INTENT( IN ) :: control
!     TYPE ( FUNNEL_inform_type ), INTENT( INOUT ) :: inform
!
!!-----------------------------------------------
!!   L o c a l   V a r i a b l e s
!!-----------------------------------------------
!
!     CHARACTER ( LEN = 80 ) :: array_name
!
!!  Deallocate arrays set for EQP
!
!     CALL EQP_terminate( data%EQP_data, control%EQP_control,                   &
!                         inform%EQP_inform )
!     IF ( inform%EQP_inform%status /= 0 ) THEN
!       inform%status = inform%EQP_inform%status
!       inform%alloc_status = inform%EQP_inform%alloc_status
!       inform%bad_alloc = inform%EQP_inform%bad_alloc
!       IF ( control%deallocate_error_fatal ) RETURN
!     END IF
!
!!  Deallocate all remaining allocated arrays
!
!     array_name = 'funnel: data%Y'
!     CALL SPACE_dealloc_array( data%Y,                                         &
!        inform%status, inform%alloc_status, array_name = array_name,           &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
!
!     array_name = 'funnel: data%H_diag'
!     CALL SPACE_dealloc_array( data%H_diag,                                    &
!        inform%status, inform%alloc_status, array_name = array_name,           &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
!
!     RETURN
!
!!  End of subroutine FUNNEL_equality_terminate
!
!     END SUBROUTINE FUNNEL_equality_terminate

!  End of module GALAHAD_FUNNEL_equality

   END MODULE GALAHAD_funnel_equality_double




