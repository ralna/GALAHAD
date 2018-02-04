! THIS VERSION: GALAHAD 2.4 - 15/05/2010 AT 15:15 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ B A R C   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.2. February 7th 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BARC_double

!     -------------------------------------------------
!    |                                                 |
!    | BARC, an adaptive regularised cubic model       |
!    |  algorithm for otimization subject to bounds    |
!    |                                                 |
!    | Aim: find a (local) minimizer of the problem    |
!    |                                                 |
!    |      minimize   f(x)   where x_l <= x <= x_u    |
!    |                                                 |
!     -------------------------------------------------

     USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SBLS_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_GLRT_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: BARC_initialize, BARC_read_specfile, BARC_solve,                &
               BARC_terminate, NLPT_problem_type, NLPT_userdata_type,          &
               BARC_projection

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!--------------------------
!  Derived type definitions
!--------------------------

     TYPE, PUBLIC :: BARC_data_type
       INTEGER :: n_free
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: VAR_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: VAR_free
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_free_save
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: O_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_plus
       TYPE ( SMT_type ) :: A, C
       TYPE ( SBLS_data_type ) :: SBLS_data
       TYPE ( GLRT_data_type ) :: GLRT_data
     END TYPE BARC_data_type

     TYPE, PUBLIC :: BARC_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit, minor_maxit
       INTEGER :: start_print, stop_print, print_gap
       REAL ( KIND = wp ) :: stop_pg
       REAL ( KIND = wp ) :: initial_sigma, gamma_reduce, gamma_increase
       REAL ( KIND = wp ) :: delta_feas, eta_successful, eta_very_successful
       LOGICAL :: fulsol, space_critical, deallocate_error_fatal
       CHARACTER ( LEN = 30 ) :: alive_file
       TYPE ( SBLS_control_type ) :: SBLS_control
       TYPE ( GLRT_control_type ) :: GLRT_control
     END TYPE BARC_control_type

     TYPE, PUBLIC :: BARC_time_type
       REAL :: total, preprocess, analyse, factorize, solve
     END TYPE

     TYPE, PUBLIC :: BARC_inform_type
       INTEGER :: status, alloc_status, iter, cg_iter, f_eval, g_eval
       INTEGER :: factorization_status
       INTEGER :: factorization_integer, factorization_real
       REAL ( KIND = wp ) :: obj, norm_pg
       CHARACTER ( LEN = 80 ) :: bad_alloc
       TYPE ( BARC_time_type ) :: time
       TYPE ( SBLS_inform_type ) :: SBLS_inform
       TYPE ( GLRT_inform_type ) :: GLRT_inform
     END TYPE BARC_inform_type

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: third = one / three
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

!    REAL ( KIND = wp ), PARAMETER :: kap_ubs = ten ** ( - 4 )
     REAL ( KIND = wp ), PARAMETER :: kap_ubs = point1
     REAL ( KIND = wp ), PARAMETER :: kap_lbs = point9
     REAL ( KIND = wp ), PARAMETER :: kap_epp = point1
     REAL ( KIND = wp ), PARAMETER :: eps_active = teneps

   CONTAINS

!-*-  G A L A H A D -  B A R C _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE BARC_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for ACO controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BARC_data_type ), INTENT( INOUT ) :: data
     TYPE ( BARC_control_type ), INTENT( OUT ) :: control
     TYPE ( BARC_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: status, alloc_status

     inform%status = GALAHAD_ok

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%prefix = '" - SBLS:"                    '

!  Initalize GLRT components

      CALL GLRT_initialize( data%GLRT_data, control%GLRT_control,              &
                            inform%GLRT_inform )
      control%GLRT_control%prefix = '" - GLRT:"                    '

!  Error and ordinary output unit numbers

     control%error = 6
     control%out = 6
     control%SBLS_control%error = control%error
     control%SBLS_control%out = control%out
     control%GLRT_control%error = control%error
     control%GLRT_control%out = control%out

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

     control%alive_unit = 60
     control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

     control%print_level = 0

!  Maximum number of iterations

     control%maxit = 100

!  Maximum number of minor iterations

     control%minor_maxit = 2

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of the projected gradient of the objective function is smaller than
!  control%stop_pg

     control%stop_pg = tenm5

!  Initial value for the regularisation weight, sigma

     control%initial_sigma = ten

!  A potential iterate will only be accepted if the actual decrease
!  f - f(x_new) is larger than control%eta_successful times that predicted
!  by a quadratic model of the decrease. The regularisation weight will be
!  reduced if this relative decrease is greater thancontrol%eta_very_successful

      control%eta_successful = ten ** ( - 8 )
      control%eta_very_successful = point9

!  On very successful iterations, the regularisation weight will be reduced by
!  the factor control%gamma_reduce, while if the iteration is unsucceful, the
!  weight will be increased by the factor control%gamma_increase

      control%gamma_reduce = half
      control%gamma_increase = two

!  Print the full solution or only highlights

     control%fulsol = .TRUE.

!  If space_critical is true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

     control%space_critical = .FALSE.

!   If deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

     control%deallocate_error_fatal  = .FALSE.

     data%A%ne = 0
     CALL SPACE_resize_array( data%A%ne, data%A%row, status, alloc_status )
     CALL SPACE_resize_array( data%A%ne, data%A%col, status, alloc_status )
     CALL SPACE_resize_array( data%A%ne, data%A%val, status, alloc_status )
     CALL SPACE_dealloc_array( data%A%type, status, alloc_status )
     CALL SMT_put( data%A%type, 'COORDINATE', alloc_status )

     data%C%ne = 0
     CALL SPACE_resize_array( data%C%ne, data%C%row, status, alloc_status )
     CALL SPACE_resize_array( data%C%ne, data%C%col, status, alloc_status )
     CALL SPACE_resize_array( data%C%ne, data%C%val, status, alloc_status )
     CALL SPACE_dealloc_array( data%C%type, status, alloc_status )
     CALL SMT_put( data%C%type, 'COORDINATE', alloc_status )

     RETURN

!  End of subroutine BARC_initialize

     END SUBROUTINE BARC_initialize

!-*-*-*-*-   B A R C _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-

     SUBROUTINE BARC_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by BARC_initialize could (roughly)
!  have been set as:

! BEGIN BARC SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device
!  print-level                                     1
!  maximum-number-of-iterations                    100
!  maximum-number-of-minor-iterations              2
!  start-print                                     22
!  stop-print                                      20
!  iterations-between-printing                     1
!  gradient-accuracy-required                      1.0D-5
!  initial-regularisation-weight                   1.0D+1
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  regularisation-weight-decrease-factor           0.5
!  regularisation-weight-increase-factor           2.0
!  very-successful-iteration-tolerance             0.9
!  print-full-solution                             no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END BARC SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BARC_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: lspec = 62
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BARC'
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
     spec(  9 )%keyword = 'maximum-number-of-minor-iterations'

!  Real key-words

     spec( 18 )%keyword = 'gradient-accuracy-required'
     spec( 23 )%keyword = 'successful-iteration-tolerance'
     spec( 24 )%keyword = 'very-successful-iteration-tolerance'
     spec( 25 )%keyword = 'initial-regularisation-weight'
     spec( 26 )%keyword = 'regularisation-weight-decrease-factor'
     spec( 27 )%keyword = 'regularisation-weight-increase-factor'

!  Logical key-words

     spec( 32 )%keyword = 'space-critical'
     spec( 35 )%keyword = 'deallocate-error-fatal'
     spec( 57 )%keyword = 'print-full-solution'

!  Character key-words

     spec( lspec )%keyword = 'alive-filename'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( 1 ), control%error,                     &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 2 ), control%out,                       &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 3 ), control%out,                       &
                                  control%alive_unit )
     CALL SPECFILE_assign_value( spec( 4 ), control%print_level,               &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 5 ), control%maxit,                     &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 6 ), control%start_print,               &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 7 ), control%stop_print,                &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 8 ), control%print_gap,                 &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 9 ), control%minor_maxit,               &
                                  control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( 18 ), control%stop_pg,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 23 ), control%eta_successful,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 24 ), control%eta_very_successful,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 25 ), control%initial_sigma,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 26 ), control%gamma_reduce,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 27 ), control%gamma_increase,           &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( 32 ), control%space_critical,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 35 ),                                   &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 57 ), control%fulsol,                   &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( lspec ), control%alive_file,            &
                                 control%error )

!  Read the controls for the preconditioner and iterative solver

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SBLS_read_specfile( control%SBLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-SBLS' )
       CALL GLRT_read_specfile( control%GLRT_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-GLRT' )
     ELSE
       CALL SBLS_read_specfile( control%SBLS_control, device )
       CALL GLRT_read_specfile( control%GLRT_control, device )
     END IF

     RETURN

     END SUBROUTINE BARC_read_specfile

!-*-*-*-  G A L A H A D -  B A R C _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE BARC_solve( nlp, control, inform, data, userdata,              &
                            eval_F, eval_G, eval_H )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  BARC_solve, a method for finding a local unconstrained minimizer of a
!    function where the variables are constrained to lie between given bounds
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( BARC_control_type ), INTENT( INOUT ) :: control
     TYPE ( BARC_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BARC_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), INTENT( OUT ) :: f
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_H( status, X, userdata, Hval )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ir, ic, j, k, l, n, out, cg_iter, eval_stat, cauchy_iter
     INTEGER :: start_print, stop_print, print_level, print_level_glrt
     INTEGER :: minor, n_free, nh_free
     REAL ( KIND = wp ) :: f_plus, f_model, val, sigma, ratio, model_decrease
     REAL ( KIND = wp ) :: t, t_min, t_max, gts, norm_PTmg, eps, q_model
     REAL ( KIND = wp ) :: alpha, f_new, curv
     LOGICAL :: set_printt, set_printi, set_printw, set_printd, print_1st_header
     LOGICAL :: printe, printi, printt, printw, printd, printm, set_printm
     LOGICAL :: print_iteration_header, firsti

     CHARACTER ( LEN = 1 ) :: suc
     CHARACTER ( LEN = 80 ) :: array_name
     REAL :: time_new, time_total

!  Initialize

     CALL CPU_TIME( time_total ) ; inform%time%total = time_total

!  record the problem dimensions

     n = nlp%n

!  set initial values

     inform%iter = 0 ; inform%cg_iter = 0
     inform%f_eval = 0 ; inform%g_eval = 0

     inform%obj = HUGE( one ) ; inform%norm_pg = HUGE( one )
     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''

!  control the output printing

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
     print_level_glrt = control%GLRT_control%print_level
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
       printm = set_printm
       printw = set_printw ; printd = set_printd
       print_level = control%print_level
     ELSE
       printi = .FALSE. ; printt = .FALSE.
       printm = .FALSE.
       printw = .FALSE. ; printd = .FALSE.
       print_level = 0
     END IF
     print_1st_header = .TRUE.

     IF ( printd ) WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  nlp%X( : n )

     nlp%Z = zero

! ------------------------
!  Step 0: Initialization
! ------------------------

!  check for faulty input dimensions

     IF ( n <= 0 ) THEN
       IF ( printe ) WRITE( control%error,                                     &
          "( ' - the problem dimensions are faulty ' )" )
       inform%status = - 4
       RETURN
     END IF

!  check that the simple bounds and consistent

     DO i = 1, n
       IF ( nlp%X_l( i ) > nlp%X_u( i ) ) THEN
         IF ( printe ) WRITE( control%error,                                   &
           "( ' - the bounds on the variables are inconsistent ' )" )
         inform%status = - 5
         RETURN
       END IF
     END DO

!  Allocate space to hold the problem data

     array_name = 'aco: data%VAR_status'
     CALL SPACE_resize_array( n, data%VAR_status, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%VAR_free'
     CALL SPACE_resize_array( n, data%VAR_free, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%H_free'
     CALL SPACE_resize_array( nlp%H%ne, data%H_free, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%PROD'
     CALL SPACE_resize_array( n, data%PROD, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%R'
     CALL SPACE_resize_array( n, data%R, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%R_free'
     CALL SPACE_resize_array( n, data%R_free, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%R_free_save'
     CALL SPACE_resize_array( n, data%R_free_save, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%V'
     CALL SPACE_resize_array( n, data%V, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%V_free'
     CALL SPACE_resize_array( n, data%V_free, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%S'
     CALL SPACE_resize_array( n, data%S, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%S_free'
     CALL SPACE_resize_array( n, data%S_free, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%O_free'
     CALL SPACE_resize_array( n, data%O_free, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'aco: data%X_plus'
     CALL SPACE_resize_array( n, data%X_plus, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  Ensure that the initial point is feasible

     CALL BARC_projection( n, nlp%X, nlp%X_l, nlp%X_u, data%X_plus )
     nlp%X( : n ) = data%X_plus( : n )

! evaluate the objective function at the initial point

     CALL eval_F( eval_stat, nlp%X( : n ), userdata, inform%obj )
     inform%f_eval = inform%f_eval + 1

!  evaluate the gradient of the objective function

     CALL eval_G( eval_stat, nlp%X( : n ), userdata, nlp%G( : n ) )
     inform%g_eval = inform%g_eval + 1

!  compute the norm of the projected gradient

     CALL BARC_projection( n, nlp%X - nlp%G, nlp%X_l, nlp%X_u, data%X_plus )
     inform%norm_pg = MAXVAL( ABS( nlp%X - data%X_plus ) )

!  evaluate the initial Hessian matrix

     CALL eval_H( eval_stat, nlp%X( : n ), userdata, nlp%H%val( : nlp%H%ne ) )

     control%SBLS_control%new_a = 1
     control%SBLS_control%new_h = 2

! record the initial regularisation weight

     sigma = control%initial_sigma

     IF ( printi ) THEN
       n_free = 0
       DO i = 1, n
         IF ( data%X_plus( i ) > nlp%X_l( i ) + eps_active .AND.               &
              data%X_plus( i ) < nlp%X_u( i ) - eps_active )                   &
           n_free = n_free + 1
       END DO
       WRITE( out, "( /, ' Problem: ', A )" ) nlp%pname
       WRITE( out, 2040 )
       print_1st_header = .FALSE.
       WRITE( out, "( I6, 1X, ES14.6, 2ES12.5, '     -           - ', I6 )" ) &
         0, inform%obj, inform%norm_pg, sigma, n_free
     END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               S T A R T    O F    M A I N    I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     DO
       inform%iter = inform%iter + 1
       IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
         printi = set_printi ; printt = set_printt
         printm = set_printm
         printw = set_printw ; printd = set_printd
         print_level = control%print_level
         control%GLRT_control%print_level = print_level_glrt
       ELSE
         printi = .FALSE. ; printt = .FALSE.
         printm = .FALSE.
         printw = .FALSE. ; printd = .FALSE.
         print_level = 0
         control%GLRT_control%print_level = 0
       END IF
       print_iteration_header = print_level > 1 .OR.                           &
         control%GLRT_control%print_level > 0

!  test for optimality

       IF ( inform%norm_pg <= control%stop_pg ) THEN
         inform%status = 0
         EXIT
       END IF

!  test that the iteration limit has not been reached

       IF ( inform%iter > control%maxit ) THEN
         inform%status = - 10
         WRITE( out, "( /, ' - the iteration limit has been reached ' )" )
         EXIT
       END IF

! ----------------------------------
!  Step 1: Compute the Cauchy point
! ----------------------------------

!  Step 1.0: Initialisation

       cauchy_iter = 0
       t_min = zero
       t_max = infinity
       t = one
       IF ( printm ) WRITE( out, "( /, '   Cauchy-point calculation: ', /,     &
      &                                '      t        lin_below    model   ', &
      &                                '   lin_above   proj_grad      gTs ' )" )

       DO
         cauchy_iter = cauchy_iter + 1
         IF ( cauchy_iter > 100 ) STOP

!  Step 1.1: compute a point x(t) on the projected gradient path

         CALL BARC_projection( n, nlp%X - t * nlp%G, nlp%X_l, nlp%X_u,        &
                                data%X_plus )

!  Compute the correction

         data%S( : n ) = data%X_plus( : n ) - nlp%X( : n )

!  Evaluate the model at x(t)

         CALL BARC_model( n, data%S( : n ), inform%obj, nlp%G( : n ),  nlp%H, &
                           sigma, gts, q_model, f_model, data%PROD( : n ) )

!  Compute || P_T(-g)||

         norm_PTmg = zero
         DO i = 1, n
           IF ( data%X_plus( i ) > nlp%X_l( i ) + eps_active .AND.             &
                data%X_plus( i ) < nlp%X_u( i ) - eps_active )                 &
             norm_PTmg = norm_PTmg + nlp%G( i ) ** 2
         END DO
         norm_PTmg = SQRT( norm_PTmg )

!  Step 1.2: check the stopping conditions

         IF ( printm ) WRITE( out, "( 6ES12.4 )" ) t,                          &
            inform%obj + kap_lbs * gts, f_model, inform%obj + kap_ubs * gts,   &
           norm_PTmg, kap_epp * ABS( gts )

         IF ( f_model > inform%obj + kap_ubs * gts ) THEN
           t_max = t
         ELSE IF ( f_model < inform%obj + kap_lbs * gts .AND.                  &
                   norm_PTmg > kap_epp * ABS( gts ) ) THEN
           t_min = t
         ELSE
           EXIT
         END IF

!  Step 1.3: find a new value of the arc parameter

         IF ( t_max == infinity ) THEN
           t = two * t
         ELSE
           t = half * ( t_min + t_max )
         END IF
       END DO

!  Find how many variables are free

       n_free = 0
       DO i = 1, n
         IF ( data%X_plus( i ) > nlp%X_l( i ) + eps_active .AND.               &
              data%X_plus( i ) < nlp%X_u( i ) - eps_active )                   &
           n_free = n_free + 1
       END DO

       IF ( printt ) THEN
         WRITE( out,                                                           &
           "( /, '   Cauchy point: model =', ES16.8, ', free = ', I0 )" )      &
             f_model, n_free
       END IF

       cg_iter = 0
       minor = 0

       IF ( control%minor_maxit > 0 .AND. n_free > 0 ) THEN

! --------------------------
!  Step 2: Step calculation
! --------------------------

!  Determine which variables are free

         n_free = 0
         DO i = 1, n
           IF ( data%X_plus( i ) <= nlp%X_l( i ) + eps_active ) THEN
             data%VAR_status( i ) = - 1
             data%V( i ) = zero
           ELSE IF ( data%X_plus( i ) >= nlp%X_u( i ) - eps_active ) THEN
             data%VAR_status( i ) = 1
             data%V( i ) = zero
           ELSE
             data%VAR_status( i ) = 0
             n_free = n_free + 1
             data%VAR_free( n_free ) = i

!  Set free components to zero

             data%O_free( n_free ) = data%S( i )
           END IF
         END DO

!  Record free components of H

         nh_free = 0
         DO l = 1, nlp%H%ne
           IF ( data%VAR_status( nlp%H%row( l ) ) == 0 .AND.                   &
                data%VAR_status( nlp%H%col( l ) ) == 0 ) THEN
             nh_free = nh_free + 1
             data%H_free( nh_free ) = l
           END IF
         END DO

!  Minor iteration

  minit: DO
           minor = minor + 1

!  if necessary form and factorize the preconditioner

!          IF ( control%SBLS_control%preconditioner > 1 ) THEN
!            control%SBLS_control%factorization = 2
!            control%SBLS_control%preconditioner =                             &
!              MIN( control%SBLS_control%preconditioner, 4 )
!            control%GLRT_control%unitm = .FALSE.
!            CALL SBLS_form_and_factorize( n, 0, nlp%H, data%A, data%C,        &
!                    data%SBLS_data, control%SBLS_control, inform%SBLS_inform )
!!           control%SBLS_control%new_h = 1
!            control%SBLS_control%new_h = 2
!          ELSE
            control%GLRT_control%unitm = .TRUE.
!          END IF

!  set initial data

           control%GLRT_control%f_0 = q_model
           control%GLRT_control%impose_descent = .TRUE.
           inform%GLRT_inform%status = 1
           cg_iter = 0

!  form the product H_free s_free

           data%PROD = zero
           DO k = 1, nh_free
             l = data%H_free( k )
             i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
             data%PROD( i ) = data%PROD( i ) + val * data%S( j )
             IF ( i /= j ) data%PROD( j ) = data%PROD( j ) + val * data%S( i)
           END DO

!  form r_free = g_free + H_free s_free and eps = ||s_fixed||^2

           eps = zero
           n_free = 0
           DO i = 1, n
             IF ( data%VAR_status( i ) == 0 ) THEN
               n_free = n_free + 1
               data%R_free( n_free ) = nlp%G( i ) + data%PROD( i )
             ELSE
               eps = eps +  data%S( i ) ** 2
             END IF
           END DO

!  store r_fr for use later

           data%R_free_save( : n_free ) = data%R_free( : n_free )

!  -.-.-.-.-.-.-.-.-.-.-.-.-
!  use GLRT to find the step
!  -.-.-.-.-.-.-.-.-.-.-.-.-

           firsti = .TRUE.
           DO
             CALL GLRT_solve( n_free, three, sigma,                            &
                        data%S_free( : n_free ), data%R_free( : n_free ),      &
                        data%V_free( : n_free ), data%GLRT_data,               &
                        control%GLRT_control, inform%GLRT_inform,              &
                        eps = eps, O = data%O_free( : n_free  ) )
!            WRITE(6,"( ' case ', i3  )" ) inform%GLRT_inform%status

!  check for error returns

             SELECT CASE( inform%GLRT_inform%status )

!  successful return

             CASE ( GALAHAD_ok )
!              f_model = inform%GLRT_inform%obj_regularized
               inform%cg_iter = inform%cg_iter + cg_iter
               EXIT

!  warnings

             CASE ( GALAHAD_error_max_iterations )
               IF ( printt ) WRITE( out, "( /,                                 &
              &  ' Warning return from GLRT, status = ', I6 )" )               &
                  inform%GLRT_inform%status
!              f_model = inform%GLRT_inform%obj_regularized
               inform%cg_iter = inform%cg_iter + cg_iter
               EXIT

!  allocation errors

             CASE ( GALAHAD_error_allocate )
                inform%status = GALAHAD_error_allocate
                inform%alloc_status = inform%glrt_inform%alloc_status
                inform%bad_alloc = inform%glrt_inform%bad_alloc
                GO TO 910

!  deallocation errors

             CASE ( GALAHAD_error_deallocate )
                inform%status = GALAHAD_error_deallocate
                inform%alloc_status = inform%glrt_inform%alloc_status
                inform%bad_alloc = inform%glrt_inform%bad_alloc
                GO TO 910

!  error return

             CASE DEFAULT
               IF ( printt ) WRITE( out, "( /,                                 &
              &  ' Error return from GLRT, status = ', I6 )" )                 &
                   inform%GLRT_inform%status
!              f_model = inform%GLRT_inform%obj_regularized
               inform%cg_iter = inform%cg_iter + cg_iter
               EXIT

!  find the preconditioned gradient

             CASE ( 2 )
               IF ( printw ) WRITE( out,                                       &
                  "( ' ............... precondition  ............... ' )" )

!              CALL SBLS_solve( n, 0, data%A, data%C, data%SBLS_data,          &
!                 control%SBLS_control, inform%SBLS_inform, data%V_free )

!              IF ( inform%SBLS_inform%status < 0 ) THEN
!                inform%status = inform%SBLS_inform%status
!                GO TO 910
!              END IF

!  form the product of V_free with H

             CASE ( 3 )

               IF ( inform%GLRT_inform%status == 3 ) cg_iter = cg_iter + 1
               IF ( printw ) WRITE( out,                                       &
                 "( ' ............ matrix-vector product ..........' )" )


!  extract the free componets

               DO j = 1, n_free
                 i = data%VAR_free( j )
                 data%V( i ) = data%V_free( j )
               END DO

!  form the product with H_free

               data%PROD = zero
               DO k = 1, nh_free
                 l = data%H_free( k )
                 i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
                 data%PROD( i ) = data%PROD( i ) + val * data%V( j )
                 IF ( i /= j ) data%PROD( j ) = data%PROD( j ) + val * data%V(i)
               END DO

!  re-distribute the free componets

               DO j = 1, n_free
                 i = data%VAR_free( j )
                 data%V_free( j ) = data%PROD( i )
               END DO

               IF ( firsti ) THEN
                 curv = DOT_PRODUCT( data%V_free( : n_free ),                  &
                                     data%V( data%VAR_free( : n_free ) ) )
!                write(6,*) ' === curv ', curv
                 IF ( curv < zero ) THEN
                   data%S_free( : n_free ) = zero
                   EXIT minit
                 END IF
                 firsti = .FALSE.
               END IF

!  reform the initial residual

             CASE ( 4 )

               IF ( printw ) WRITE( out,                                       &
                 "( ' ................. restarting ................ ' )" )

               data%R_free( : n_free ) = data%R_free_save( : n_free )

!  find the preconditioned gradient

             CASE ( 5 )
               IF ( printw ) WRITE( out,                                       &
                  "( ' .......... product with preconditioner  .......... ' )" )

!              CALL SBLS_solve( n, 0, data%A, data%C, data%SBLS_data,          &
!                 control%SBLS_control, inform%SBLS_inform, data%V_free )

!              IF ( inform%SBLS_inform%status < 0 ) THEN
!                inform%status = inform%SBLS_inform%status
!                GO TO 910
!              END IF

             END SELECT

           END DO

!  -.-.-.-.-.-
!  end of GLRT
!  -.-.-.-.-.-

           IF ( DOT_PRODUCT( data%R_free_save( : n_free ),                     &
                             data%S_free( : n_free ) ) >= 0 ) THEN
             IF ( printt ) WRITE( out,                                         &
               "( '   Terminate inner iteration: uphill direction ' )" )
             EXIT
           END IF

!  find the maximum step to the boundary

           alpha = infinity
           DO j = 1, n_free
             i = data%VAR_free( j )
             IF ( data%S_free( j ) > zero ) THEN
               alpha = MIN( alpha,  ( nlp%X_u( i ) - nlp%X( i )                &
                                      - data%O_free( j ) ) / data%S_free( j ) )
             ELSE IF ( data%S_free( j ) < zero ) THEN
               alpha = MIN( alpha,  ( nlp%X_l( i ) - nlp%X( i )                &
                                      - data%O_free( j ) ) / data%S_free( j ) )
             END IF

!  record the free components of s

             data%S( i ) = data%O_free( j ) + data%S_free( j )
           END DO

!  compute the projection of x + s onto the feasible region

           CALL BARC_projection( n, nlp%X + data%S, nlp%X_l, nlp%X_u,         &
                                  data%X_plus )

!  compute the correction

           data%S( : n ) = data%X_plus( : n ) - nlp%X( : n )

!  evaluate the model at the correction

           CALL BARC_model( n, data%S( : n ), inform%obj, nlp%G( : n ),        &
                             nlp%H, sigma, gts, q_model, f_new,                &
                             data%PROD( : n ) )

!  The model did not decrease at the projected point. Backtrack to the
!  boundary and try again

           IF ( f_new >= f_model ) THEN

!  record the free components of s

             DO j = 1, n_free
               i = data%VAR_free( j )
               data%S( i ) = data%O_free( j ) + alpha * data%S_free( j )
             END DO

!  compute the projection of x + s onto the feasible region

             CALL BARC_projection( n, nlp%X + data%S, nlp%X_l, nlp%X_u,       &
                                    data%X_plus )

!  compute the correction

             data%S( : n ) = data%X_plus( : n ) - nlp%X( : n )

!  evaluate the model at the correction

             CALL BARC_model( n, data%S( : n ), inform%obj, nlp%G( : n ),      &
                               nlp%H, sigma, gts, q_model, f_new,              &
                               data%PROD( : n ) )
           END IF
           f_model = f_new

!  find how many variables are still free

           n_free = 0
           DO i = 1, n
             IF ( data%X_plus( i ) > nlp%X_l( i ) + eps_active .AND.           &
                  data%X_plus( i ) < nlp%X_u( i ) - eps_active )               &
               n_free = n_free + 1
           END DO

           IF ( printt ) THEN
             IF ( control%GLRT_control%print_level > 0 ) WRITE( out, "( '' )" )
             WRITE( out,                                                       &
               "( '   After GLRT:   model =', ES16.8, ', free = ', I0 )" )     &
                 f_model, n_free
           END IF

!  test for convergence

           IF ( minor > control%minor_maxit ) THEN
             IF ( printt ) WRITE( out,                                         &
               "( '   Terminate inner iteration: iteration limit reached' )" )
             EXIT
           END IF

!  determine which variables are still free

           l = n_free
           n_free = 0
           DO j = 1, l
             i = data%VAR_free( j )
             IF ( data%X_plus( i ) <= nlp%X_l( i ) + eps_active ) THEN
               data%VAR_status( i ) = - 1
               data%V( i ) = zero
             ELSE IF ( data%X_plus( i ) >= nlp%X_u( i ) - eps_active ) THEN
               data%VAR_status( i ) = 1
               data%V( i ) = zero
             ELSE
               n_free = n_free + 1
               data%O_free( n_free ) = data%S( i )
               data%S( i ) = zero
               data%VAR_free( n_free ) = i
             END IF
           END DO

!  check to see if there was any improvement

           IF ( n_free == l ) THEN
             data%S( : n ) = data%X_plus( : n ) - nlp%X( : n )
             IF ( printt ) WRITE( out,                                         &
               "( '   Terminate inner iteration: no change to active set' )" )
             EXIT
           END IF
           IF ( n_free == 0 ) THEN
             IF ( printt ) WRITE( out,                                         &
               "( '   Terminate inner iteration: no free variables ' )" )
             EXIT
           END IF

!  Update the list of free components of H

           j = nh_free
           nh_free = 0

           DO k = 1, j
             l = data%H_free( k )
             IF ( data%VAR_status( nlp%H%row( l ) ) == 0 .AND.                 &
                  data%VAR_status( nlp%H%col( l ) ) == 0 ) THEN
               nh_free = nh_free + 1
               data%H_free( nh_free ) = l
             END IF
           END DO

!  end of minor iteration

         END DO minit

       ELSE
         IF ( n_free == 0 ) THEN
           IF ( printt ) WRITE( out,                                           &
             "( '   No inner iteration: no free variables ' )" )
         END IF

!  check that the step will make a difference

       END IF

       IF ( MAXVAL( ABS( data%S ) ) <=                                         &
            epsmch * MIN( one, MAXVAL( ABS( nlp%X ) ) ) ) THEN
         IF ( printe ) WRITE( control%error,                                   &
           "( ' - the step is too small to make further progress ' )" )
         EXIT
       END IF

! -------------------------------------
!  Step 3: Acceptance of the new point
! -------------------------------------

!  compute the trial point

       data%X_plus( : n ) = nlp%X( : n ) + data%S( : n )

!  evaluate the objective function at the trial point

       CALL eval_F( eval_stat, data%X_plus( : n ), userdata, f_plus )
       inform%f_eval = inform%f_eval + 1

!  compute the change in the model

       CALL BARC_model( n, data%S( : n ), zero, nlp%G( : n ),  nlp%H,        &
                         sigma, gts, q_model, model_decrease,  data%PROD( : n ))

!  check the trial point for acceptability

        IF ( model_decrease /= zero ) THEN
         ratio = ( f_plus - inform%obj ) / model_decrease
       ELSE
         ratio = 1.0
       END IF

! -----------------------------------------------
!  Steps 3 and 4: Check the step for acceptibilty
! -----------------------------------------------

!  successful step

       IF ( model_decrease < zero .AND. ratio >= control%eta_successful ) THEN
         nlp%X = data%X_plus
         inform%obj = f_plus

!  evaluate the gradient of the objective function

         CALL eval_G( eval_stat, nlp%X( : n ), userdata, nlp%G( : n ) )
         inform%g_eval = inform%g_eval + 1

!  compute the norm of the projected gradient

         CALL BARC_projection( n, nlp%X - nlp%G, nlp%X_l, nlp%X_u, data%X_plus )
         inform%norm_pg = MAXVAL( ABS( nlp%X - data%X_plus ) )

!  evaluate the Hessian matrix

         CALL eval_H( eval_stat, nlp%X( : n ),                                 &
                      userdata, nlp%H%val( : nlp%H%ne ) )

!  very successful step

         IF ( ratio >= control%eta_very_successful ) THEN
           suc = 'v'
!          sigma = MAX( one, control%gamma_reduce * sigma )
!          sigma = control%gamma_reduce * sigma
!          sigma = MAX( epsmch, MIN( sigma, inform%norm_pg ) )
           sigma = MAX( SQRT( epsmch ),                                        &
                     MIN( control%gamma_reduce * sigma, inform%norm_pg ) )
         ELSE
           suc = 's'
         END IF

!  unsuccessful step

       ELSE
         suc = 'u'
!        sigma = MAX( one, control%gamma_increase * sigma )
         sigma = control%gamma_increase * sigma
       END IF

!  print details of the iteration

       IF ( printi ) THEN
         IF ( print_iteration_header .OR. print_1st_header ) WRITE( out, 2040 )
         print_1st_header = .FALSE.
         WRITE( out, "( I6, A1, ES14.6, 3ES12.5, I6, 1X, I6 )" )               &
           inform%iter, suc, inform%obj, inform%norm_pg, sigma, ratio,         &
           cg_iter, n_free
       END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    M A I N    I T E R A T I O N
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     END DO

     nlp%Z = nlp%G

!  -------------
!  Normal return
!  -------------

!  Print the solution

     l = 2
     IF ( control%fulsol ) l = n
     IF ( control%print_level >= 10 ) l = n

     WRITE( out, 2000 )
     DO j = 1, 2
       IF ( j == 1 ) THEN
         ir = 1 ; ic = MIN( l, n )
       ELSE
         IF ( ic < n - l ) WRITE( out, 2030 )
         ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
       END IF
       DO i = ir, ic
         WRITE( out, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ), nlp%X_l( i ),      &
           nlp%X_u( i ), nlp%Z( i )
       END DO
     END DO

     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total

     WRITE( out, "( /, ' Problem: ', 16X, A10,                                 &
    &          '     Solver: ', 9X, '  BARC', /,                               &
    &  ' n              =     ',bn, I12, /,                                    &
    &  ' Objective      = ', ES16.8, '       Norm proj. grad = ', ES12.4, /,   &
    &  ' Iterations     =     ',bn, I12, '       Function evals  = ',bn, I12,/,&
    &  ' Gradient evals =     ',bn, I12, '       Time            = ', F12.2 )")&
      nlp%pname, n, inform%obj, inform%norm_pg,                                &
      inform%iter, inform%f_eval, inform%g_eval, inform%time%total
     WRITE( out, "( '' )" )

     IF ( inform%status /= 0 ) THEN
       WRITE( control%error, "( ' ** Message from -BARC_solve- ',              &
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

!  Other errors

!990 CONTINUE
!    CALL CPU_TIME( time_new ); inform%time%total = time_new - inform%time%total
!    WRITE( control%error, "( /, ' ** Message from -BARC_solve-',              &
!   &  '    Error exit (status = ', I6, ')', / )" ) inform%status
!    RETURN

!  Non-executable statements

 2000 FORMAT( /,' Solution: ', /,'                        ',                   &
                '        <------ Bounds ------> ', /                           &
                '      # name          value   ',                              &
                '    Lower       Upper       Dual ' )
 2020 FORMAT( I7, 1X, A10, 4ES12.4 )
 2030 FORMAT( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2040 FORMAT( /, '  iter        f        ||P(g||      sigma     ',             &
                 '  ratio   cg iter   free' )

!  End of subroutine BARC_solve

     END SUBROUTINE BARC_solve

!-*-*-*-*-  G A L A H A D -  B A R C _ m o d e l   S U B R O U T I N E -*-*-*-

     SUBROUTINE BARC_model( n, X, f, C, H, sigma, l_model, q_model, c_model, W )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the value of the model
!    1/3 sigma ||x||^3 + 1/2 <x, H x> + <c, x> + f

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ) :: f, sigma
     REAL ( KIND = wp ), INTENT( OUT ) :: l_model, q_model, c_model
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C, X
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: W

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, l
     REAL ( KIND = wp ) :: val

!  compute the Hessian-vector product with X

     W = zero
     DO l = 1, H%ne
       i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
       W( i ) = W( i ) + val * X( j )
       IF ( i /= j ) W( j ) = W( j ) + val * X( i )
     END DO

!  evaluate the linear, quadratic and cubic models

     l_model = DOT_PRODUCT( C, X )
     q_model = f + l_model + half * DOT_PRODUCT( W, X )
     c_model = q_model + third * sigma * ( TWO_NORM( X ) ) ** 3

!  End of subroutine BARC_model

     END SUBROUTINE BARC_model

!-*-*-  G A L A H A D -  B A R C _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE BARC_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BARC_data_type ), INTENT( INOUT ) :: data
     TYPE ( BARC_control_type ), INTENT( IN ) :: control
     TYPE ( BARC_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'aco: data%VAR_free'
     CALL SPACE_dealloc_array( data%VAR_free,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%VAR_status'
     CALL SPACE_dealloc_array( data%VAR_status,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%H_free'
     CALL SPACE_dealloc_array( data%H_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%S_free'
     CALL SPACE_dealloc_array( data%S_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%O_free'
     CALL SPACE_dealloc_array( data%O_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%V_free'
     CALL SPACE_dealloc_array( data%V_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%R'
     CALL SPACE_dealloc_array( data%R,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%R_free'
     CALL SPACE_dealloc_array( data%R_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%R_free_save'
     CALL SPACE_dealloc_array( data%R_free_save,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%PROD'
     CALL SPACE_dealloc_array( data%PROD,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%X_plus'
     CALL SPACE_dealloc_array( data%X_plus,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%A%row'
     CALL SPACE_dealloc_array( data%A%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%A%col'
     CALL SPACE_dealloc_array( data%A%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%A%val'
     CALL SPACE_dealloc_array( data%A%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%A%type'
     CALL SPACE_dealloc_array( data%A%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%C%row'
     CALL SPACE_dealloc_array( data%C%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%C%col'
     CALL SPACE_dealloc_array( data%C%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%C%val'
     CALL SPACE_dealloc_array( data%C%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'aco: data%C%type'
     CALL SPACE_dealloc_array( data%C%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BARC_terminate

     END SUBROUTINE BARC_terminate

!-*-  G A L A H A D -  B A R C _ p r o j e c t i o n   S U B R O U T I N E -*-

     SUBROUTINE BARC_projection( n, C, X_l, X_u, X, iter, A, b )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the projection of c into the set
!    x_l <= x <= x_u and optionally a^T x = b

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     INTEGER, OPTIONAL, INTENT( OUT ) :: iter
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C, X_l, X_u
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: A
     REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: b

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: mu, y, dy, athinva, rhs, viol8, pr_feas, du_feas,step
     REAL ( KIND = wp ), DIMENSION( n ) :: DX, Z_l, Z_u, DZ_l, DZ_u, H

!  Trivial projection if the optional linear constraint is absent

     IF ( .NOT. ( PRESENT( A ) .AND. PRESENT( b ) ) ) THEN
       X = MAX( X_l, MIN( C, X_u ) )
       RETURN
     END IF

!  Compute a starting point

     DO i = 1, n
       IF ( ABS( X_l( i ) ) < ABS( X_l( i ) ) ) THEN
         X( i ) = X_l( i ) + MIN( one, X_u( i ) - X_l( i ) )
       ELSE
         X( i ) = X_u( i ) - MIN( one, X_u( i ) - X_l( i ) )
       END IF
       Z_l( i ) = MAX( one, - C( i ) )
       Z_u( i ) = MAX( one, C( i ) )
     END DO
     y = zero

     IF ( PRESENT( iter ) ) iter = 0
     DO
       viol8 = DOT_PRODUCT( A, X ) - b

       pr_feas = ABS( viol8 )
       du_feas = MAXVAL( ABS( X - C - y * A - Z_l + Z_u ) )
!      WRITE( 6, "(' primal, dual infeasibility ', 2ES12.4 )" ) pr_feas, du_feas

!  Check for optimality

       IF ( MAX( pr_feas, du_feas ) <= ten ** ( - 8 ) ) EXIT
       IF ( PRESENT( iter ) ) iter = iter + 1

!  Compute the target parameter

       mu = 0.05_wp * ( DOT_PRODUCT( X - X_l, Z_l ) +                          &
                        DOT_PRODUCT( X_u - X, Z_u ) ) / FLOAT( n )

!  Solve ( H   a ) ( dx ) = - ( x - c - mu ( (X-X_l)^-1 (X_u-X)^-1 ) e )
!        ( a^T 0 ) ( dy )     (           a^T x - b                    )
!  where H = I + (X-X_l)^-1 Z_l + (X_u-X)^-1 Z_u and e = (1,1,..,1)^T

       rhs = - viol8 ; athinva = zero
       DO i = 1, n
         DZ_l( i ) = X( i ) - X_l( i )
         DZ_u( i ) = X_u( i ) - X( i )
         DX( i ) = X( i ) - C( i ) - y * A( i )                                &
                  - mu / DZ_l( i ) + mu / DZ_u( i )
         H( i ) = one + Z_l( i ) / DZ_l( i ) + Z_u( i ) / DZ_u( i )
         athinva = athinva + A( i ) * ( A( i ) / H( i ) )
         rhs = rhs + A( i ) * ( DX( i ) / H( i ) )
       END DO

!  Record the primal correction

       dy = rhs / athinva
       DX = ( A * dy - DX ) / H

!  Compute the maximum primal stepsize

       step = one
       DO i = 1, n
         IF ( DX( i ) > zero ) THEN
           step = MIN( step, DZ_u( i ) / DX( i ) )
         ELSE IF ( DX( i ) < zero ) THEN
           step = MIN( step, - DZ_l( i ) / DX( i ) )
         END IF
       END DO

!  Recover the corrections to the dual variables

       DO i = 1, n
         DZ_l( i ) = ( mu - Z_l( i ) * ( DZ_l( i ) + DX( i ) ) ) / DZ_l( i )
         DZ_u( i ) = ( mu - Z_u( i ) * ( DZ_u( i ) - DX( i ) ) ) / DZ_u( i )
       END DO

!  Compute the maximum dual stepsize

       DO i = 1, n
         IF ( DZ_l( i ) < zero ) step = MIN( step, - Z_l( i ) / DZ_l( i ) )
         IF ( DZ_u( i ) < zero ) step = MIN( step, - Z_u( i ) / DZ_u( i ) )
       END DO

!  Stop just short of infeasibility

       step = MIN( one, 0.9999_wp * step )

!  Update the primal and dual variables

       X = X + step * DX
       y = y + step * dy
       Z_l = Z_l + step * DZ_l
       Z_u = Z_u + step * DZ_u

     END DO
!    WRITE( 6, "(' primal, dual infeasibility ', 2ES12.4, 1X, I0,             &
!   &   ' iterations' )" ) pr_feas, du_feas, iter
     WRITE( 6, "( ' fraction active ', ES12.4 )" ) &
     FLOAT( COUNT( X-X_l < Z_l ) + COUNT( X_U-X < Z_u ) ) / FLOAT( n )
     RETURN

     END SUBROUTINE BARC_projection

!  End of module GALAHAD_BARC

   END MODULE GALAHAD_BARC_double

