! THIS VERSION: GALAHAD 2.2 - 07/02/2008 AT 17:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ B A R C   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.2. February 7th 2008

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BARC_double

!     ---------------------------------------------------
!    |                                                   |
!    | Barc, an adaptive cubic-regularisation algorithm |
!    |  for optimization subject to convex constraints   |
!    |                                                   |
!    | Aim: find a (local) minimizer of the problem      |
!    |                                                   |
!    |         minimize   f(x) : x in convex X           |
!    |                                                   |
!     ---------------------------------------------------

     USE CUTEr_interface_double
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SBLS_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_GLRT_double
     USE GALAHAD_SPACE_double

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: BARC_initialize, BARC_read_specfile, BARC_solve,              &
               BARC_terminate, NLPT_problem_type, BARC_projection

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!--------------------------
!  Derived type definitions
!--------------------------

     TYPE, PUBLIC :: BARC_data_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: T
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: N
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_plus
       TYPE ( SMT_type ) :: H, A, C
       TYPE ( SBLS_data_type ) :: SBLS_data
       TYPE ( GLRT_data_type ) :: GLRT_data
     END TYPE BARC_data_type

     TYPE, PUBLIC :: BARC_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap
       REAL ( KIND = wp ) :: stop_g
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
       REAL ( KIND = wp ) :: obj, norm_g
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
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

   CONTAINS

!-*-  G A L A H A D -  B A R C _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE BARC_initialize( data, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BARC controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BARC_data_type ), INTENT( OUT ) :: data
     TYPE ( BARC_control_type ), INTENT( OUT ) :: control
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: status, alloc_status

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control )
      control%SBLS_control%prefix = '" - SBLS:"                     '

!  Initalize GLRT components

      CALL GLRT_initialize( data%GLRT_data, control%GLRT_control )
      control%GLRT_control%prefix = '" - GLRT:"                     '

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

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of the gradient of the Lagrangian function is smaller than control%stop_g

     control%stop_g = tenm5

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

!-*-*-*-*-   B A R C _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

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
!  maximum-number-of-iterations                    50
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
     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: lspec = 62
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'BARC          '
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

     CALL SPECFILE_assign_integer( spec( 1 ), control%error,                   &
                                   control%error )
     CALL SPECFILE_assign_integer( spec( 2 ), control%out,                     &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 3 ), control%out,                     &
                                   control%alive_unit )                         
     CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 5 ), control%maxit,                   &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,               &
                                   control%error )                           
!  Set real values

     CALL SPECFILE_assign_real( spec( 18 ), control%stop_g,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 23 ), control%eta_successful,            &
                                control%error )                    
     CALL SPECFILE_assign_real( spec( 24 ), control%eta_very_successful,       &
                                control%error )                    
     CALL SPECFILE_assign_real( spec( 25 ), control%initial_sigma,             &
                                control%error )                    
     CALL SPECFILE_assign_real( spec( 26 ), control%gamma_reduce,              &
                                control%error )                    
     CALL SPECFILE_assign_real( spec( 27 ), control%gamma_increase,            &
                                control%error )                    

!  Set logical values

     CALL SPECFILE_assign_logical( spec( 32 ), control%space_critical,         &
                                   control%error )
     CALL SPECFILE_assign_logical( spec( 35 ),                                 &
                                   control%deallocate_error_fatal,             &
                                   control%error )
     CALL SPECFILE_assign_logical( spec( 57 ), control%fulsol,                 &
                                   control%error )                           

!  Set character values

     CALL SPECFILE_assign_string( spec( lspec ), control%alive_file,           &
                                  control%error )                           

!  Read the controls for the preconditioner and iterative solver

     CALL SBLS_read_specfile( control%SBLS_control, device )
     CALL GLRT_read_specfile( control%GLRT_control, device )

     RETURN

     END SUBROUTINE BARC_read_specfile

!-*-*-*-  G A L A H A D -  B A R C _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE BARC_solve( nlp, control, inform, data )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  BARC_solve, a method for finding a local minimizer of a function subject 
!  to convex constraints.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( BARC_control_type ), INTENT( INOUT ) :: control
     TYPE ( BARC_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BARC_data_type ), INTENT( INOUT ) :: data

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ir, ic, j, l, h_len, n, out, cg_iter
     INTEGER :: start_print, stop_print, print_level, print_level_glrt
     REAL ( KIND = wp ) :: f_plus, f_model, val, sigma, ratio
     LOGICAL :: set_printt, set_printi, set_printw, set_printd, print_1st_header
     LOGICAL :: printe, printi, printt, printw, printd
!    LOGICAL :: set_printm, printe, printi, printt, printm, printw, printd
     LOGICAL :: print_iteration_header

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

     inform%obj = HUGE( one ) ; inform%norm_g = HUGE( one )
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

!    set_printm = out > 0 .AND. control%print_level >= 3 

!  As per printm but also with an indication of where in the code we are

     set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

     set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

     IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
       printi = set_printi ; printt = set_printt
!      printm = set_printm
       printw = set_printw ; printd = set_printd
       print_level = control%print_level
     ELSE
       printi = .FALSE. ; printt = .FALSE.
!      printm = .FALSE. 
       printw = .FALSE. ; printd = .FALSE.
       print_level = 0
     END IF
     print_1st_header = .TRUE.

     IF ( printd ) WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  nlp%X( : n )

     nlp%Z = zero
  
! ------------------------
!  Step 0: Initialization
! ------------------------
 
! check for faulty input dimensions 

     IF ( n <= 0 ) THEN
       IF ( printe ) WRITE( control%error,                                      &
          "( ' - the problem dimensions are faulty ' )" )
       inform%status = - 4
       RETURN
     END IF
  
! check that the problem does not include simple bounds
  
     DO i = 1, n
       IF ( nlp%X_l( i ) > - infinity .OR. nlp%X_u( i ) < infinity ) THEN
         IF ( printe ) WRITE( control%error,                                    &
           "( ' - the problem contains simple-bound constraints  ' )" )
!        inform%status = - 20
!        RETURN
         EXIT
       END IF
     END DO
      
!  Determine how many nonzeros are required to store the Hessian matrix,
!  when the matrix is stored as a sparse matrix in "co-ordinate" format
 
     CALL UDIMSH( h_len )

!  Allocate space to hold the problem data

     array_name = 'barc: nlp%G'
     CALL SPACE_resize_array( n, nlp%G, inform%status,                         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%PROD'
     CALL SPACE_resize_array( n, data%PROD, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%R'
     CALL SPACE_resize_array( n, data%R, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%VECTOR'
     CALL SPACE_resize_array( n, data%VECTOR, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%X_plus'
     CALL SPACE_resize_array( n, data%X_plus, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%S'
     CALL SPACE_resize_array( n, data%S, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%H%row'
     CALL SPACE_resize_array( h_len, data%H%row, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%H%col'
     CALL SPACE_resize_array( h_len, data%H%col, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'barc: data%H%val'
     CALL SPACE_resize_array( h_len, data%H%val, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

! evaluate the objective function at the initial point

     CALL UFN( n, nlp%X, inform%obj )
     inform%f_eval = inform%f_eval + 1

!  evaluate the gradient of the objective function

     CALL UGR( n, nlp%X, nlp%G )
     inform%g_eval = inform%g_eval + 1
     inform%norm_g = MAXVAL( ABS( nlp%G ) )
 
!  evaluate the initial Hessian matrix
    
     CALL USH( n, nlp%X, data%H%ne, h_len, data%H%val, data%H%row, data%H%col )
     CALL SMT_put( data%H%type, 'COORDINATE', i )
     control%SBLS_control%new_a = 1
     control%SBLS_control%new_h = 2

! record the initial regularisation weight

     sigma = control%initial_sigma
  
     IF ( printi ) THEN
       WRITE( out, "( /, ' Problem: ', A )" ) nlp%pname
       WRITE( out, 2040 )
       print_1st_header = .FALSE.
       WRITE( out, "( I6, 1X, ES14.6, 2ES12.5, '     -           -  ' )" )     &
         0, inform%obj, inform%norm_g, sigma
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
!        printm = set_printm 
         printw = set_printw ; printd = set_printd
         print_level = control%print_level
         control%GLRT_control%print_level = print_level_glrt
       ELSE
         printi = .FALSE. ; printt = .FALSE.
!        printm = .FALSE. 
         printw = .FALSE. ; printd = .FALSE.
         print_level = 0
         control%GLRT_control%print_level = 0
       END IF
       print_iteration_header = print_level > 1 .OR.                            &
         control%GLRT_control%print_level > 0

!  test for optimality

       IF ( inform%norm_g <= control%stop_g ) THEN
         inform%status = 0
         EXIT
       END IF

!  test that the iteration limit has not been reached

       IF ( inform%iter > control%maxit ) THEN
         inform%status = - 10
         WRITE( out, "( /, ' - the iteration limit has been reached ' )" )
         EXIT
       END IF

! -------------------------
!  Step 1: Compute the step
! -------------------------
 
!  if necessary form and factorize the preconditioner

       IF ( control%SBLS_control%preconditioner > 1 ) THEN
         control%SBLS_control%factorization = 2
         control%SBLS_control%preconditioner =                                  &
           MIN( control%SBLS_control%preconditioner, 4 )
         control%GLRT_control%unitm = .FALSE.
         CALL SBLS_form_and_factorize( n, 0, data%H, data%A, data%C,            &
                     data%SBLS_data, control%SBLS_control, inform%SBLS_inform )
!        control%SBLS_control%new_h = 1
         control%SBLS_control%new_h = 2
       ELSE
         control%GLRT_control%unitm = .TRUE.
       END IF

!  set initial data
     
       f_model = inform%obj
       data%R = nlp%G
       control%GLRT_control%f_0 = inform%obj
       inform%GLRT_inform%status = 1
       cg_iter = 0

!  use GLRT to find the step

       DO
         CALL GLRT_solve( n, three, sigma, data%S, data%R, data%VECTOR,         &
                     data%GLRT_data, control%GLRT_control, inform%GLRT_inform )
!        WRITE(6,"( ' case ', i3  )" ) inform%GLRT_inform%status

!  check for error returns

         SELECT CASE( inform%GLRT_inform%status )

!  successful return

         CASE ( 0 )
           f_model = inform%GLRT_inform%obj
           inform%cg_iter = inform%cg_iter + cg_iter
           EXIT

!  warnings

         CASE ( - 2 )
           IF ( printt ) WRITE( out, "( /,                                      &
          &  ' Warning return from GLRT, status = ', I6 )" )                    &
              inform%GLRT_inform%status
           f_model = inform%GLRT_inform%obj
           inform%cg_iter = inform%cg_iter + cg_iter
           EXIT
          
!  allocation errors

         CASE ( - 6 )
            inform%status = - 1
            inform%alloc_status = inform%glrt_inform%alloc_status
            inform%bad_alloc = inform%glrt_inform%bad_alloc
            GO TO 910

!  deallocation errors

         CASE ( - 7 )
            inform%status = - 2
            inform%alloc_status = inform%glrt_inform%alloc_status
            inform%bad_alloc = inform%glrt_inform%bad_alloc
            GO TO 910

!  error return

         CASE DEFAULT
           IF ( printt ) WRITE( out, "( /,                                      &
          &  ' Error return from GLRT, status = ', I6 )" )                      &
               inform%GLRT_inform%status
           f_model = inform%GLRT_inform%obj
           inform%cg_iter = inform%cg_iter + cg_iter
           EXIT

!  find the preconditioned gradient

         CASE ( 2 )
           IF ( printw ) WRITE( out,                                            &
              "( ' ............... precondition  ............... ' )" )

           CALL SBLS_solve( n, 0, data%A, data%C, data%SBLS_data,               &
              control%SBLS_control, inform%SBLS_inform, data%VECTOR )

           IF ( inform%SBLS_inform%status < 0 ) THEN
             inform%status = inform%SBLS_inform%status
             GO TO 910
           END IF

!  form the product of VECTOR with H

         CASE ( 3 )

           IF ( inform%GLRT_inform%status == 3 ) cg_iter = cg_iter + 1
           IF ( printw ) WRITE( out,                                            &
             "( ' ............ matrix-vector product ..........' )" )

           data%PROD = zero
           DO l = 1, data%H%ne
             i = data%H%row( l ) ; j = data%H%col( l ) ; val = data%H%val( l )
             data%PROD( i ) = data%PROD( i ) + val * data%VECTOR( j )
             IF ( i /= j ) data%PROD( j )                                       &
               = data%PROD( j ) + val * data%VECTOR( i )
           END DO
           data%VECTOR = data%PROD

!  reform the initial residual

         CASE ( 4 )
          
           IF ( printw ) WRITE( out,                                            &
             "( ' ................. restarting ................ ' )" )

           data%R = nlp%G

         END SELECT

       END DO

!  check that the step will make a difference

       IF ( MAXVAL( ABS( data%S ) ) <=                                          &
            epsmch * MIN( one, MAXVAL( ABS( nlp%X ) ) ) ) THEN
         IF ( printe ) WRITE( control%error,                                    &
           "( ' - the step is too small to make further progress ' )" )
         EXIT
       END IF
    
! ---------------------------------------
!  Step 2: Compute the new function value
! ---------------------------------------
 
!  compute the trial point
    
       data%X_plus = nlp%X + data%S ;
    
!  evaluate the objective function at the trial point

     CALL UFN( n, data%X_plus, f_plus )
     inform%f_eval = inform%f_eval + 1

!  check the trial point for acceptability

    IF ( ABS( inform%obj - f_model ) /= zero ) THEN
      ratio = ( inform%obj - f_plus ) / ( inform%obj - f_model )
    ELSE
      ratio = 1.0
    END IF

! -----------------------------------------------
!  Steps 3 and 4: Check the step for acceptibilty
! -----------------------------------------------
 
!  successful step

    IF ( ratio >= control%eta_successful ) THEN
      nlp%X = data%X_plus
      inform%obj = f_plus

!  evaluate the gradient of the objective function

       CALL UGR( n, nlp%X, nlp%G )
       inform%g_eval = inform%g_eval + 1
       inform%norm_g = MAXVAL( ABS( nlp%G ) )

!  evaluate the Hessian matrix
    
       CALL USH( n, nlp%X, data%H%ne, h_len, data%H%val, data%H%row, data%H%col )

!  very successful step

      IF ( ratio >= control%eta_very_successful ) THEN
        suc = 'v'
!       sigma = MAX( one, control%gamma_reduce * sigma )
!       sigma = control%gamma_reduce * sigma
!       sigma = MAX( epsmch, MIN( sigma, inform%norm_g ) )
        sigma = MAX( SQRT( epsmch ),                                            &
                  MIN( control%gamma_reduce * sigma, inform%norm_g ) )
      ELSE
        suc = 's'
      END IF

!  unsuccessful step

    ELSE
      suc = 'u'
!     sigma = MAX( one, control%gamma_increase * sigma )
      sigma = control%gamma_increase * sigma
    END IF
    
!  print details of the iteration

     IF ( printi ) THEN
       IF ( print_iteration_header .OR. print_1st_header ) WRITE( out, 2040 )
       print_1st_header = .FALSE.
       WRITE( out, "( I6, A1, ES14.6, 3ES12.5, I6 )" )                          &
         inform%iter, suc, inform%obj, inform%norm_g, sigma, ratio, cg_iter
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
         WRITE( out, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ), nlp%X_l( i ),       &
           nlp%X_u( i ), nlp%Z( i )
       END DO
     END DO

     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total

     WRITE( out, "( /, ' Problem: ', 16X, A10,                                  &
    &          '     Solver: ', 9X, '  BARC', /,                               &
    &  ' n              =     ',bn, I12, /,                                     &
    &  ' Objective      = ', ES16.8, '       Norm gradient   = ', ES12.4, /,    &
    &  ' Iterations     =     ',bn, I12, '       Function evals  = ',bn, I12,/, &
    &  ' Gradient evals =     ',bn, I12, '       Time            = ', F12.2 )") &
      nlp%pname, n, inform%obj, inform%norm_g,                                  &
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
 2040 FORMAT( /, '  iter        f          ||g||       sigma     ',            &
                 ' ratio    cg iter' )

!  End of subroutine BARC_solve

     END SUBROUTINE BARC_solve

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

     array_name = 'barc: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%PROD'
     CALL SPACE_dealloc_array( data%PROD,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%VECTOR'
     CALL SPACE_dealloc_array( data%VECTOR,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%R'
     CALL SPACE_dealloc_array( data%R,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%X_plus'
     CALL SPACE_dealloc_array( data%X_plus,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%H%row'
     CALL SPACE_dealloc_array( data%H%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%H%type'
     CALL SPACE_dealloc_array( data%H%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%A%row'
     CALL SPACE_dealloc_array( data%A%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%A%col'
     CALL SPACE_dealloc_array( data%A%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%A%val'
     CALL SPACE_dealloc_array( data%A%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%A%type'
     CALL SPACE_dealloc_array( data%A%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%C%row'
     CALL SPACE_dealloc_array( data%C%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%C%col'
     CALL SPACE_dealloc_array( data%C%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%C%val'
     CALL SPACE_dealloc_array( data%C%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'barc: data%C%type'
     CALL SPACE_dealloc_array( data%C%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BARC_terminate

     END SUBROUTINE BARC_terminate


     SUBROUTINE BARC_projection( n, C, X_l, X_u, X, A, b )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the projection of c into the set
!    x_l <= x <= x_u and optionally a^T x = b

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C, X_l, X_u
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: A
     REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: b

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: mu, y, dy, athinva, rhs, viol8, pr_feas, du_feas, step
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
       Z_l( i ) = one
       Z_u( i ) = one
     END DO
     y = zero

     DO
       viol8 = DOT_PRODUCT( A, X ) - b

       pr_feas = ABS( viol8 )
       du_feas = MAXVAL( ABS( X( i ) - C( i ) - y * A( i ) - Z_l + Z_u ) )
       WRITE( 6, "(' primal, dual infeasibility ', 2ES12.4 )" ) pr_feas, du_feas

!  Check for optimality

       IF ( MAX( pr_feas, du_feas ) <= ten ** ( - 8 ) ) RETURN

!  Solve ( H   a ) ( dx ) = - ( x - c - mu ( (X-X_l)^-1 (X_u-X)^-1 ) e )
!        ( a^T 0 ) ( dy )     (           a^T x - b                    )
!  where H = I + (X-X_l)^-1 Z_l + (X_u-X)^-1 Z_u and e = (1,1,..,1)^T

       rhs = - viol8 ; athinva = zero
       DO i = 1, n
         DZ_l( i ) = X( i ) - X_l( i )
         DZ_u( i ) = X_u( i ) - X( i )
         DX( i ) = X( i ) - C( i ) - y * A( i )                                 &
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
     RETURN

     END SUBROUTINE BARC_projection


!  End of module GALAHAD_BARC

   END MODULE GALAHAD_BARC_double

