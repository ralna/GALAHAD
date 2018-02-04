! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D -  T R T N   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Philippe Toint
!
!  History -
!   started August 3rd, 2004
!
!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_TRTN_double

!   -----------------------------------------------------------
!  |                                                           |
!  |  Mminimize f(x) subject to simple bounds x_l <= x <= x_u, |
!  |  using a suitable transformation x(y). The unconstrained  |
!  |  minimizer of the transformed f(x(y)) is sought using a   |
!  |  using a trust-region truncated-Newton method             |
!  |                                                           |
!   -----------------------------------------------------------

     USE CUTEr_interface_double
     USE GALAHAD_NORMS_double
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SMT_double
     USE GALAHAD_QPT_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SILS_double
     USE GALAHAD_GLTR_double
     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: TRTN_initialize, TRTN_read_specfile, TRTN_solve

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

     REAL ( KIND = wp ), PARAMETER :: res_large = one

!  ======================================
!  The TRTN_control_type derived type
!  ======================================

     TYPE, PUBLIC :: TRTN_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap, precon, model
       INTEGER :: semibandwidth, io_buffer
       INTEGER :: non_monotone, first_derivatives, second_derivatives
       INTEGER :: cg_maxit, itref_max, indmin, valmin
       INTEGER :: lanczos_itmax
       REAL ( KIND = wp ) :: stop_g, acccg, initial_radius
       REAL ( KIND = wp ) :: rho_successful, rho_very_successful, maximum_radius
       REAL ( KIND = wp ) :: radius_decrease_factor
       REAL ( KIND = wp ) :: radius_small_increase_factor
       REAL ( KIND = wp ) :: radius_increase_factor
       REAL ( KIND = wp ) :: infinity, pert_x
       REAL ( KIND = wp ) :: inner_fraction_opt, pivot_tol
       REAL ( KIND = wp ) :: inner_stop_relative, inner_stop_absolute
       LOGICAL :: fulsol, print_matrix
       CHARACTER ( LEN = 30 ) :: alive_file
       TYPE ( GLTR_control_type ) :: gltr_control        
     END TYPE TRTN_control_type

!  =====================================
!  The TRTN_time_type derived type
!  =====================================

     TYPE, PUBLIC :: TRTN_time_type
       REAL :: total, analyse, factorize, solve
     END TYPE

!  =====================================
!  The TRTN_inform_type derived type
!  =====================================

     TYPE, PUBLIC :: TRTN_inform_type
       INTEGER :: status, f_eval, g_eval, iter, nfacts, cg_iter
       INTEGER :: factorization_status
       INTEGER :: factorization_integer, factorization_real
       REAL ( KIND = wp ) :: obj, norm_g, radius
       CHARACTER ( LEN = 10 ) :: pname
       CHARACTER ( LEN = 24 ) :: bad_alloc
       TYPE ( GLTR_info_type ) :: gltr_inform
       TYPE ( TRTN_time_type ) :: time
     END TYPE TRTN_inform_type

!  ===================================
!  The TRTN_data_type derived type
!  ===================================

     TYPE, PUBLIC :: TRTN_data_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G, G_m, S, X_trial, VECTOR
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL, RES, RHS, BEST, P_pert
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, X_grad, X_hess, G_wrty
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: FREE
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: X_name
       TYPE ( GLTR_data_type ) :: gltr_data
       TYPE ( SMT_type ) :: H
       TYPE ( SMT_type ) :: P
       TYPE ( SILS_factors ) :: FACTORS
       TYPE ( SILS_control ) :: CNTL
       TYPE ( SILS_ainfo ) :: AINFO
       TYPE ( SILS_finfo ) :: FINFO
     END TYPE TRTN_data_type

!--------------------------------
!   I n t e r f a c e  B l o c k
!--------------------------------

!     INTERFACE TWO_NORM
!       FUNCTION SNRM2( n, X, incx )
!       REAL :: SNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION SNRM2
!
!       FUNCTION DNRM2( n, X, incx )
!       DOUBLE PRECISION :: DNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION DNRM2
!     END INTERFACE 

   CONTAINS

!-*-*-*-*  G A L A H A D -  TRTN_initialize  S U B R O U T I N E -*-*-*-*

     SUBROUTINE TRTN_initialize( data, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRTN controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TRTN_data_type ), INTENT( OUT ) :: data
     TYPE ( TRTN_control_type ), INTENT( OUT ) :: control

!  Initalize SILS components

     CALL SILS_initialize( FACTORS = data%FACTORS, control = data%CNTL )
     data%CNTL%pivoting = 4

!  Intialize GLTR data

     CALL GLTR_initialize( data%gltr_data, control%gltr_control )

!  Error and ordinary output unit numbers

     control%error = 6
     control%out = 6

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

     control%alive_unit = 60
     control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

     control%print_level = 0

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  Maximum number of iterations

     control%maxit = 1000

!   cg_maxit. The maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1

     control%cg_maxit = 200

!  itref_max. The maximum number of iterative refinements allowed

      control%itref_max = 1

!   Precon specifies the preconditioner to be used for the CG. 
!     Possible values are
!
!      0  automatic 
!      1  no preconditioner, i.e, the identity
!      2  band within full factorization
!      3  full factorization

     control%precon = 0

!  The number of vectors allowed in Lin and More's incomplete factorization

!    control%icfact = 5

!  The semi-bandwidth of the band factorization

     control%semibandwidth = 5

!   indmin. An initial guess as to the integer workspace required by SILS

      control%indmin = 1000

!   valmin. An initial guess as to the real workspace required by SILS

      control%valmin = 1000

!  Stop the iteration when the norm of the gradient is smaller than stop_g

     control%stop_g = ten ** ( - 5 )

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

     control%initial_radius = - one

!  The largest possible trust-region radius

     control%maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region 
!  (specialists only!)

     control%rho_successful = point01
!    control%rho_successful = point1
!    control%rho_very_successful = 0.75_wp
     control%rho_very_successful = point9
!    control%rho_very_successful = point99
     
     control%radius_decrease_factor = half
     control%radius_small_increase_factor = 1.1_wp
     control%radius_increase_factor = two

!  Any bound larger than infinity in absolute value is infinite

     control%infinity = infinity

!   inner_stop_relative and inner_stop_absolute. The search direction is
!    considered as an acceptable approximation to the minimizer of the
!    model if the gradient of the model in the preconditioning(inverse) 
!    norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )

!    control%inner_stop_relative = zero
     control%inner_stop_relative = point01
     control%inner_stop_absolute = SQRT( epsmch )

!   inner_fraction_opt. a search direction which gives at least 
!    inner_fraction_opt times the optimal model decrease will be found

     control%inner_fraction_opt = point1

!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SILS for details

     control%pivot_tol = epsmch ** 0.75

!  Control the maximu number of Lanczos iterations that GLTR will
!  take on the trust-region boundary

     control%lanczos_itmax = 5

!  Define the closest to a bound that a variable is allowed to start

     control%pert_x = point1
     
!!  Remove on release !!
     control%print_matrix = .FALSE.

!  Ensure that the private data arrays have the correct initial status

     NULLIFY( data%FREE, data%G, data%G_m, data%S, data%X_trial, data%VECTOR )
     NULLIFY( data%SOL, data%RES, data%RHS, data%BEST, data%P_pert )
     NULLIFY( data%Y, data%X_grad, data%X_hess, data%G_wrty, data%X_name )

     RETURN

!  End of subroutine TRTN_initialize

     END SUBROUTINE TRTN_initialize

!-*-*-   S U P E R B _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE TRTN_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by TRTN_initialize could (roughly) 
!  have been set as:

! BEGIN TRTN SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  print-full-solution                            YES
!  print-matrix                                   NO
!  preconditioner-used                            0
!  model-used                                     1
!  semi-bandwidth-for-band-preconditioner         5
!  unit-number-for-temporary-io                   75
!  history-length-for-non-monotone-descent        0
!  max-lanczos-iterations                         5
!  first-derivative-approximations                EXACT
!  second-derivative-approximations               EXACT
!  gradient-accuracy-required                     1.0D-5
!  inner-iteration-relative-accuracy-required     0.01
!  initial-trust-region-radius                    -1.0
!  maximum-radius                                 1.0D+20
!  rho-successful                                 0.01
!  rho-very-successful                            0.9
!  radius-decrease-factor                         0.5
!  radius-small-increase-factor                   1.1
!  radius-increase_factor                         2.0
!  infinity-value                                 1.0D+19
!  inner-iteration-stop-relative                  0.01
!  inner-iteration-stop-abslute                   1.0D-8
!  alive-filename                                 ALIVE.d
! END TRTN SPECIFICATIONS

!  Dummy arguments

     TYPE ( TRTN_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: lspec = 61
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'TRTN          '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

     spec(  1 )%keyword = 'error-printout-device'
     spec(  2 )%keyword = 'printout-device'
     spec(  3 )%keyword = 'alive-device'
     spec(  4 )%keyword = 'print-level' 
     spec(  5 )%keyword = 'maximum-number-of-iterations'
     spec(  6 )%keyword = 'start-print'
     spec(  7 )%keyword = 'stop-print'
     spec(  8 )%keyword = 'iterations-between-printing'
     spec(  9 )%keyword = 'preconditioner-used'
     spec( 11 )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( 12 )%keyword = 'max-lanczos-iterations'
     spec( 13 )%keyword = 'unit-number-for-temporary-io'
     spec( 14 )%keyword = 'history-length-for-non-monotone-descent'
     spec( 15 )%keyword = 'first-derivative-approximations'
     spec( 16 )%keyword = 'second-derivative-approximations'
     spec( 50 )%keyword = 'model-used'
     spec( 60 )%keyword = 'print-matrix'

!  Real key-words

     spec( 17 )%keyword = 'gradient-accuracy-required'
     spec( 20 )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( 21 )%keyword = 'initial-trust-region-radius'
     spec( 22 )%keyword = 'maximum-radius'
     spec( 23 )%keyword = 'rho-successful'
     spec( 24 )%keyword = 'rho-very-successful'
     spec( 25 )%keyword = 'radius-decrease-factor'
     spec( 26 )%keyword = 'radius-small-increase-factor'
     spec( 27 )%keyword = 'radius-increase-factor'
     spec( 35 )%keyword = 'infinity-value'
     spec( 58 )%keyword = 'inner-iteration-stop-relative'
     spec( 59 )%keyword = 'inner-iteration-stop-abslute'

!  Logical key-words

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
     CALL SPECFILE_assign_symbol( spec( 9 ), control%precon,                   &
                                  control%error )                           
     CALL SPECFILE_assign_integer( spec( 11 ), control%semibandwidth,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 12 ), control%lanczos_itmax,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 13 ), control%io_buffer,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 14 ), control%non_monotone,           &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 15 ), control%first_derivatives,       &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 16 ), control%second_derivatives,      &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 50 ), control%model,                   &
                                  control%error )                           

!  Set real values

     CALL SPECFILE_assign_real( spec( 17 ), control%stop_g,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 20 ), control%acccg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 21 ), control%initial_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 22 ), control%maximum_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 23 ), control%rho_successful,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 24 ), control%rho_very_successful,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 25 ),                                    &
                 control%radius_decrease_factor, control%error )
     CALL SPECFILE_assign_real( spec( 26 ),                                    &
                 control%radius_small_increase_factor, control%error )
     CALL SPECFILE_assign_real( spec( 27 ),                                    &
                 control%radius_increase_factor, control%error )
     CALL SPECFILE_assign_real( spec( 35 ), control%infinity,                  &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 58 ), control%inner_stop_relative,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 59 ), control%inner_stop_absolute,       &
                                control%error )                           

!  Set logical values

     CALL SPECFILE_assign_logical( spec( 57 ), control%fulsol,                 &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 60 ), control%print_matrix,           &
                                   control%error )                           

!  Set character values

     CALL SPECFILE_assign_string( spec( lspec ), control%alive_file,           &
                                  control%error )                           

     RETURN

     END SUBROUTINE TRTN_read_specfile

!-*-*-*-*  G A L A H A D -  TRTN_solver  S U B R O U T I N E -*-*-*-*

     SUBROUTINE TRTN_solve( n, X, X_l, X_u, control, inform, data )
     INTEGER, INTENT( IN ) ::n
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( INOUT )  :: X
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN )  :: X_l, X_u
     TYPE ( TRTN_control_type ), INTENT( INOUT ) :: control
     TYPE ( TRTN_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( TRTN_data_type ), INTENT( INOUT ) :: data

!  ----------------------------------------------------------
!
!  Find a minimizer of the objective function f(x) subject
!  to simple bounds x_l <= x <= x_u, using a trust-region 
!  truncated-Newton method to minimize the scaled function 
!  f(x(y)) for some suitable x(y)
!
!  Nick Gould
!  August, 2004

!  ----------------------------------------------------------

!  Local variables

     INTEGER :: out, error, nnzh, nnzp, print_level, cg_iter, nsemib, itref_max 
     INTEGER :: start_print, stop_print, print_gap, precon
     INTEGER :: i, ir, ic, j, l, n_free
     REAL :: dum, time, time_new, time_total
     REAL ( KIND = wp ) :: ratio, old_radius, initial_radius, step, teneps
     REAL ( KIND = wp ) :: pred, ared, f_trial, res_norm, model
     LOGICAL :: goth, analyse, big_res, xney
     LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
     LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe 
     CHARACTER ( LEN = 1 ) :: mo

!  Set initial values

     CALL CPU_TIME( time_total ) ; inform%time%total = time_total

     inform%iter = 0 ; inform%f_eval = 0 ; inform%g_eval = 0
     precon = control%precon
     IF ( precon <= 0 ) precon = 1

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

!  Enable printing controls

     out = control%out ; error = control%error 
     set_printe = error > 0 .AND. control%print_level >= 1

!  Basic single line of output per iteration

     set_printi = out > 0 .AND. control%print_level >= 1 

!  As per printi, but with additional timings for various operations

     set_printt = out > 0 .AND. control%print_level >= 2 

!  As per printm, but with checking of residuals, etc

     set_printm = out > 0 .AND. control%print_level >= 3 

!  As per printm but also with an indication of where in the code we are

     set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

     set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

     IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
       printe = set_printe ; printi = set_printi ; printt = set_printt
       printm = set_printm ; printw = set_printw ; printd = set_printd
       print_level = control%print_level
     ELSE
       printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
       printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
       print_level = 0
     END IF

!  Record the problem name

     ALLOCATE( data%X_name( n ) )
     CALL UNAMES( n, inform%pname, data%X_name )

!  See if the problem is unconstrained or bound constrained

     xney = .FALSE.
     DO i = 1, n
       IF ( X_l( i ) > - infinity .OR. X_u( i ) < infinity ) THEN
         xney = .TRUE. ; EXIT
       END IF
     END DO

!  If the problem is bound-constrained, find its transformed variables
    
     IF ( xney ) THEN
       ALLOCATE( data%FREE( n ) )
       data%FREE = X_l /= X_u
       n_free = COUNT( data%FREE )
       IF ( n_free /= n .AND. printi )                                         &
         WRITE( out, "( I10, ' out of ', I10, ' variables are free ' )" )      &
           n_free, n
       ALLOCATE( data%Y( n ), data%G_wrty( n ) )
       ALLOCATE( data%X_grad( n ), data%X_hess( n ) )
       CALL TRTN_inverse_transform( n, X, X_l, X_u, control%infinity,          &
                                    control%pert_x, data%Y )

!  Compute the derivatives of the transformation wrt the transformed variables

       CALL TRTN_transform( n, data%Y, X_l, X_u, control%infinity,             &
                            X_gradient = data%X_grad, X_hessian = data%X_hess )

     END IF

!  Evaluate the objective function value
   
     CALL UFN( n, X, inform%obj )
     inform%f_eval = inform%f_eval + 1

!  Allocate space to store the gradient and Hessian

     ALLOCATE( data%G( n ) )
     CALL UDIMSH( nnzh )
     ALLOCATE( data%H%row( nnzh ), data%H%col( nnzh ), data%H%val( nnzh ) )

!  Allocate space to store workspace

     ALLOCATE( data%G_m( n ), data%S( n ), data%X_trial( n ), data%BEST( n ) )
     ALLOCATE( data%VECTOR( n ), data%SOL( n ), data%RHS( n ), data%RES( n ) )
     ALLOCATE( data%P_pert( 0 ) )

!  Set control parameters

     analyse = .TRUE.
     itref_max = control%itref_max
     ratio = - one
     IF ( control%initial_radius > zero ) THEN
       initial_radius = control%initial_radius
     ELSE
       initial_radius = one
     END IF
     inform%radius = initial_radius
     old_radius = initial_radius
     inform%nfacts = 0
     cg_iter = 0 ; mo = ' '
     nsemib = MAX( 0, MIN( control%semibandwidth, n ) )
     teneps = ten * epsmch

     IF ( printi ) THEN
       WRITE( out, "( ' ' )" )
       SELECT CASE( precon )
         CASE( 1 ) ; WRITE( out, "( '  Identity Hessian ' )" )
         CASE( 2 ) ; WRITE( out, "( '  Band (semi-bandwidth ', I3,            &
                    &               ') Hessian ' )" ) nsemib
         CASE( 3 ) ; WRITE( out, "( '  Full Hessian ' )" )
       END SELECT
     END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      M A I N    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

main:DO

!  Intialize GLTR data

       control%gltr_control%stop_relative = control%inner_stop_relative
       control%gltr_control%stop_absolute = control%inner_stop_absolute
       control%gltr_control%fraction_opt = control%inner_fraction_opt
       control%gltr_control%unitm = .FALSE.
       control%gltr_control%steihaug_toint = .FALSE.
       control%gltr_control%out = control%out
       control%gltr_control%print_level = print_level - 1
       control%gltr_control%itmax = control%cg_maxit
       control%gltr_control%lanczos_itmax = control%lanczos_itmax

!  Evaluate the gradient

       CALL UGR( n, X, data%G )     
       inform%g_eval = inform%g_eval + 1

!  Compute the derivatives of the transformation wrt the transformed variables

       IF ( xney ) THEN
         CALL TRTN_transform( n, data%Y, X_l, X_u, control%infinity,           &
                              X_gradient = data%X_grad,                        &
                              X_hessian = data%X_hess )
         data%G_wrty = data%G * data%X_grad

!  Compute the norm of the gradient

         inform%norm_g = MAXVAL( ABS( data%G_wrty ) )
       ELSE
         inform%norm_g = MAXVAL( ABS( data%G ) )
       END IF


!  Print a summary of the last iteration

       IF ( printi ) THEN
         CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
         IF ( inform%iter > 0 ) THEN
           IF ( print_level > 1 ) WRITE( out, 2030 )
           WRITE( out,                                                        &
                 "( I6, ES12.4, 3ES8.1, ES9.1, I7, A1, F8.1 )" )              &
               inform%iter, inform%obj, inform%norm_g,                        &
               step, old_radius, ratio, cg_iter, mo,                          &
               inform%time%total
         ELSE
           WRITE( out, 2030 )
           WRITE( out, "( I6, ES12.4, ES8.1, '    -   ', ES8.1,               &
          &     '     -         - ', F8.1 )" ) inform%iter, inform%obj,       &
            inform%norm_g, inform%radius, inform%time%total
         END IF
       END IF

!  Test for convergence

       IF ( inform%norm_g <= control%stop_g ) THEN
          inform%status = 0
          EXIT
       END IF

!  Evaluate the Hessian

       IF ( precon > 1 ) THEN
         CALL USH( n, X, data%H%ne, nnzh, data%H%val, data%H%row, data%H%col )
         goth = .TRUE.         

!  Analyse the preconditioner

         IF ( analyse ) THEN

!  First, determine how much space is required

           SELECT CASE( precon )
           CASE ( 2 ) ! band
             nnzp = COUNT( data%H%row( : data%H%ne ) > 0 .AND.                &
                           data%H%col( : data%H%ne ) > 0 .AND.                &
                           ABS( data%H%row( : data%H%ne ) -                   &
                                data%H%col( : data%H%ne ) ) <= nsemib )
           CASE ( 3 ) ! full
             nnzp = nnzh
           END SELECT
           IF ( xney ) nnzp = nnzp + n

!  Allocate space for the preconditioner

           data%P%n = n ; data%P%ne = nnzp
           ALLOCATE( data%P%row( nnzp ), data%P%col( nnzp ), data%P%val( nnzp ))

!  Form the sparsity pattern of the preconditioner 

           SELECT CASE( precon )
           CASE ( 2 ) ! band
             nnzp = 0
             DO l = 1, data%H%ne
               IF ( data%H%row( l ) > 0 .AND. data%H%col( l ) > 0 .AND.       &
                    ABS( data%H%row( l ) - data%H%col( l ) ) <= nsemib ) THEN
                 nnzp = nnzp + 1
                 data%P%row( nnzp ) = data%H%row( l )
                 data%P%col( nnzp ) = data%H%col( l )
               END IF
             END DO
           CASE ( 3 ) ! full
             data%P%row( : data%H%ne ) = data%H%row
             data%P%col( : data%H%ne ) = data%H%col
             nnzp = data%H%ne
           END SELECT

           IF ( xney ) THEN
             DO i = 1, n
               nnzp = nnzp + 1
               data%P%row( nnzp ) = i
               data%P%col( nnzp ) = i
            END DO
           END IF

!  Analyse the sparsity pattern of the preconditioner

            CALL SILS_analyse( data%P, data%FACTORS, data%CNTL, data%AINFO )

!  Record the storage requested

            inform%factorization_integer = data%AINFO%nirnec 
            inform%factorization_real = data%AINFO%nrlnec

            data%CNTL%la  = MAX( 2 * inform%factorization_integer,            &
                                 control%indmin )
            data%CNTL%liw = MAX( 2 * inform%factorization_real,               &
                                 control%valmin )
!  Check for error returns

            inform%factorization_status = data%AINFO%flag
            IF ( data%AINFO%flag < 0 ) THEN
              IF ( printe ) WRITE( control%error, 2100 ) data%AINFO%flag 
              inform%status = 6 ; EXIT
            ELSE IF ( data%AINFO%flag > 0 ) THEN 
              IF ( printt ) WRITE( out, 2060 ) data%AINFO%flag, 'SILS_analyse'
            END IF
        
            IF ( printt ) WRITE( out,                                         &
              "( ' real/integer space required for factors ', 2I10 )" )       &
                data%AINFO%nrladu, data%AINFO%niradu

!  Analysis complete

             analyse = .FALSE.

         END IF

!  Factorize the preconditioner; first store its numerical values

         SELECT CASE( precon )
         CASE ( 2 ) ! band
           nnzp = 0
           DO l = 1, data%H%ne
             i = data%H%row( l ) ; j = data%H%col( l )
             IF ( i > 0 .AND. j > 0 .AND. ABS( i - j ) <= nsemib ) THEN
               nnzp = nnzp + 1
               IF ( xney ) THEN
                 IF ( data%FREE( i ) .AND. data%FREE( j ) ) THEN
                   data%P%val( nnzp ) =                                        &
                     data%H%val( l ) * data%X_grad( i ) *  data%X_grad( j )
                 ELSE
                   IF ( i == j ) THEN
                     data%P%val( nnzp ) = one
                   ELSE
                     data%P%val( nnzp ) = zero
                   END IF
                 END IF
               ELSE
                 data%P%val( nnzp ) = data%H%val( l )
               END IF
             END IF
           END DO
         CASE ( 3 ) ! full
           IF ( xney ) THEN
             nnzp = 0
             DO l = 1, data%H%ne
               i = data%H%row( l ) ; j = data%H%col( l )
               nnzp = nnzp + 1
               IF ( data%FREE( i ) .AND. data%FREE( j ) ) THEN
                 data%P%val( nnzp ) =                                          &
                   data%H%val( l ) * data%X_grad( i ) *  data%X_grad( j )
               ELSE
                 IF ( i == j ) THEN
                   data%P%val( nnzp ) = one
                 ELSE
                   data%P%val( nnzp ) = zero
                 END IF
               END IF
             END DO
           ELSE
             data%P%val( : data%H%ne ) = data%H%val
             nnzp = data%H%ne
           END IF
         END SELECT

         IF ( xney ) THEN
            DO i = 1, n
             nnzp = nnzp + 1
             data%P%val( nnzp ) = data%G( i ) * data%X_hess( i )
           END DO
         END IF

         IF ( control%print_matrix ) THEN
           WRITE( out, "( ' n, nnz = ', 2I6, ' values ' )" ) data%P%n, data%P%ne
           WRITE( out, "( 2 ( 2I6, ES24.16 ) )" ) ( data%P%row( i ),           &
             data%P%col( i ), data%P%val( i ), i = 1, data%P%ne ) 
          END IF

!  Obtain the factors

         CALL SILS_factorize( data%P, data%FACTORS, data%CNTL, data%FINFO )

!  Record the storage required

         inform%nfacts = inform%nfacts + 1 
         inform%factorization_integer = data%FINFO%nirbdu 
         inform%factorization_real = data%FINFO%nrlbdu

!  Test that the factorization succeeded

         inform%factorization_status = data%FINFO%flag

!  Check to see if diagonal perturbations have been made 

         IF ( data%FINFO%modstep == 0 ) THEN
           mo = ' '  
         ELSE 
           IF ( printt ) WRITE( out, "( ' diagonal perturbation made ' )" )
           mo = 'm' 
           IF ( SIZE( data%P_pert ) /= n )  THEN
             DEALLOCATE( data%P_pert )
             ALLOCATE( data%P_pert( n ) )
           END IF
           CALL SILS_enquire( data%FACTORS, PERTURBATION = data%P_pert )
           IF ( printd ) WRITE( out, "( ' perturbations ', /, ( 5ES12.4 ) )" ) &
             data%P_pert
         END IF

!  The factorization failed. If possible, increase the pivot tolerance

         IF ( data%FINFO%flag < 0 ) THEN
           IF ( printe ) WRITE( control%error, 2100 ) data%FINFO%flag,         &
                                                      'SILS_factorize'
           inform%status = 7 ; EXIT
         END IF

         inform%nfacts = inform%nfacts + 1 
       ELSE
         goth = .FALSE.
       END IF

!  Start the next iteration

       inform%iter = inform%iter + 1
       step = zero ; ratio = zero

       IF ( inform%iter > control%maxit ) THEN
         inform%status = 10 ; EXIT
       END IF

       IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
         printe = set_printe ; printi = set_printi ; printt = set_printt
         printm = set_printm ; printw = set_printw ; printd = set_printd
         print_level = control%print_level
       ELSE
         printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
         printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
         print_level = 0
       END IF

!  Iteratively minimize the quadratic model in the trust region
!    m(s) = <g, s> + 1/2 <s, Hs>
!  Note that m(s) does not include f(x): m(0) = 0.

!  ----------------------------------------------------------------------------
!                       SEARCH DIRECTION COMPUTATION
!  ----------------------------------------------------------------------------

!  Solve the model problem to find a search direction.
!  Use the GLTR algorithm to compute a suitable trial step

!  Set initial data

       cg_iter = 0
       control%gltr_control%boundary = .FALSE.
       control%gltr_control%unitm = precon == 1
       inform%gltr_inform%status = 1
       inform%gltr_inform%negative_curvature = .TRUE.

       IF ( xney ) THEN
         data%G_m = data%G_wrty
       ELSE
         data%G_m = data%G
       END IF

       IF ( printm ) WRITE( out,                                               &
        "(/, '   |------------------------------------------------------|',    &
      &   /, '   |        start to solve trust-region subproblem        |',    &
      &   / )" )

       CALL CPU_TIME( time )

! Inner-most (GLTR) loop

       DO
         CALL GLTR_solve( n, inform%radius, model, data%S, data%G_m,           &
                          data%VECTOR, data%gltr_data, control%gltr_control,   &
                          inform%gltr_inform )

!  Check for error returns

         SELECT CASE( inform%gltr_inform%status )

!  Successful return

         CASE ( GALAHAD_ok )
           EXIT

!  Warnings

         CASE ( GALAHAD_warning_on_boundary, GALAHAD_error_max_iterations )
           IF ( printt ) WRITE( out, "( /,                                   &
          &  ' Warning return from GLTR, status = ', I6 )" )                 &
             inform%gltr_inform%status
           EXIT
          
!  Error return

         CASE DEFAULT
           IF ( printt ) WRITE( out, "( /,                                     &
          &  ' Error return from GLTR, status = ', I6 )" )                     &
             inform%gltr_inform%status
           inform%status = inform%gltr_inform%status ; GO TO 810

!  Find the preconditioned gradient

         CASE ( 2, 6 )
           IF ( printw ) WRITE( out,                                           &
              "( '    ............... precondition  ............... ' )" )

!  Compute the search direction, taking care to get small residuals

           IF ( data%FINFO%modstep == 0 ) THEN
             CALL TRTN_iterative_refinement( data%P, data%FACTORS, data%CNTL,  &
                  data%SOL, data%VECTOR, data%RES, data%BEST, res_norm,        &
                  big_res, itref_max, print_level, out )
           ELSE
             CALL TRTN_iterative_refinement( data%P, data%FACTORS, data%CNTL,  &
                  data%SOL, data%VECTOR, data%RES, data%BEST, res_norm,        &
                  big_res, itref_max, print_level, out, P_pert = data%P_pert )
           END IF

!  Ensure that the residuals are small. If not, try to obtain a more
!  accurate factorization and try again

           IF ( big_res ) THEN
             inform%status = 8 ; GO TO 810
           END IF

!  Replace the solution in VECTOR

           data%VECTOR = data%SOL

!  Form the product of VECTOR with H

         CASE ( 3, 7 )

           IF ( printw ) WRITE( out,                                          &
                "( '    ............ matrix-vector product .......... ' )" )

           IF ( xney ) THEN
             data%RES =  data%VECTOR * data%X_grad        
             CALL UPROD( n, goth, X, data%RES, data%SOL )
             data%SOL                                                         &
               = data%SOL * data%X_grad + data%G * data%X_hess * data%VECTOR
           ELSE
             CALL UPROD( n, goth, X, data%VECTOR, data%SOL )
           END IF
           goth = .TRUE.         

!  Replace the product in VECTOR

           data%VECTOR = data%SOL

!  Reform the initial residual

         CASE ( 5 )
         
           IF ( printw ) WRITE( out,                                         &
                "( '    ................. restarting ................ ' )" )

           IF ( xney ) THEN
             data%G_m = data%G_wrty
           ELSE
             data%G_m = data%G
           END IF

         END SELECT
       END DO

       CALL CPU_TIME( dum ) ; dum = dum - time
       IF ( precon < 0 ) precon = - precon

       IF ( printm ) WRITE( out,                                             &
        "(/, '   |           trust-region subproblem solved             |',  &
      &   /, '   |------------------------------------------------------|',  &
      &     / )" )

       IF ( printw ) WRITE( out,                                             &
            "( ' ............... step computed ............... ' )" )

       IF ( printt ) WRITE( out, "( ' solve time = ', F10.2 ) " ) dum
       inform%time%solve = inform%time%solve + dum

       cg_iter = cg_iter + inform%gltr_inform%iter
       inform%cg_iter = inform%cg_iter + cg_iter

!  If the overall search direction is unlikely to make a significant
!  impact on the residual, exit

       IF ( inform%gltr_inform%mnormx <= teneps ) THEN
         inform%status = 3 ; EXIT
       END IF

!  ----------------------------------------------------------------------------
!                       STEP ACCEPTANCE TESTS
!  ----------------------------------------------------------------------------
 
!  Compute the step size

!      step = SQRT( DOT_PRODUCT( S, S ) )
       step = inform%gltr_inform%mnormx

!  Ensure that the step is not too large

       DO i = 1, n
         IF ( data%S( i ) > control%maximum_radius ) THEN
           data%S( i ) = inform%radius
         ELSE IF ( data%S( i ) < - control%maximum_radius ) THEN
           data%S( i ) = - inform%radius
         END IF
       END DO

!  Compute the trial step

       IF ( xney ) THEN
         CALL TRTN_transform( n, data%Y + data%S, X_l, X_u, control%infinity,   &
                              X_value = data%X_trial )

       ELSE
         data%X_trial = X + data%S
       END IF

!  Evaluate the objective function value
   
       CALL UFN( n, data%X_trial, f_trial )
       inform%f_eval = inform%f_eval + 1

!  Compute the actual and predicted reduction

       pred = - model
       model = model + inform%obj
       ared = inform%obj - f_trial
       ared = ared + MAX( one, ABS( inform%obj ) ) * teneps
       pred = pred + MAX( one, ABS( inform%obj ) ) * teneps

       IF ( pred > 0 ) THEN
         ratio = ared / pred
       ELSE
         ratio = zero
       END IF

       old_radius = inform%radius
       
!  ----------------------------------------------------------------------------
!                         SUCCESSFUL STEP 
!  ----------------------------------------------------------------------------
      
       IF ( ratio  >= control%rho_successful ) THEN

         IF ( printw ) WRITE( out,                                             &
              "( ' ............... successful step ............... ' )" )

!  Update step

         X = data%X_trial
         IF ( xney ) data%Y = data%Y + data%S
         WRITE(6,"( 'Y', ( 4ES22.14 ) )" ) data%Y
         inform%obj = f_trial

!  ----------------------------------------------------------------------------
!                      VERY SUCCESSFUL STEP 
!  ----------------------------------------------------------------------------

!  Possibly increase radius

         IF ( ratio >= control%rho_very_successful ) THEN

!          WRITE(6,*)  inform%radius
!          WRITE(6,*)  control%maximum_radius,                                 &
!                      MAX( inform%radius,                                     &
!                           step * control%radius_increase_factor ),           &
!                      inform%radius * control%radius_increase_factor

           inform%radius = MIN( control%maximum_radius,                        &
                                MAX( inform%radius,                            &
                                     step * control%radius_increase_factor ),  &
                                inform%radius * control%radius_increase_factor )
         END IF

!  ----------------------------------------------------------------------------
!                         UNSUCCESSFUL STEP 
!  ----------------------------------------------------------------------------

       ELSE

         IF ( printw ) WRITE( out,                                           &
              "( ' .............. unsuccessful step .............. ' )" )

!  Reduce radius

           inform%radius = control%radius_decrease_factor * inform%radius
       END IF

!  End of main iteration loop

     END DO main

 810 CONTINUE

!  Indicate why the iteration terminated

     IF ( printd ) THEN
       WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  X( : n )
     END IF
     IF ( printi ) THEN
       WRITE( out, 2080 ) inform%status
       SELECT CASE ( inform%status )
       CASE ( 0 )
         WRITE( out, "( '   stopping based on gradient of Lagrangian ' )" )
         WRITE( out, 2090 )                                                  &
           inform%norm_g, control%stop_g
       CASE ( 3 )
         WRITE( out, "( '   step too small' )" )
       CASE ( 4 )
         WRITE( out, "( '   step too small following linesearch' )" )
       CASE (  5 )
         WRITE( out, "( '   actual and predicted reductions both zero ' )" )
       CASE ( 6 )
         WRITE( out, "( '   error from SILS (analysis phase) ' )" )
       CASE ( 7 )
         WRITE( out, "( '   error from SILS (factorization phase) ' )" )
       CASE ( 8 )
         WRITE( out, "( '   iterative refinement failed ' )" )
       CASE ( 9 )
         WRITE( out, "( '   allocation failed ' )" )
       CASE ( 10 )
         WRITE( out, "( '   iteration limit exceeded ' )" )
       CASE ( 11 )
         WRITE( out, "( '   error from GLTR ' )" )
       CASE DEFAULT
         WRITE( out, "( '   ???? why stop inner ???? ' )" )
       END SELECT
       WRITE( out, "( ' ' )" )
     END IF

!  Print the solution

     IF ( out > 0 .AND. control%print_level >= 0 ) THEN
       l = 4
       IF ( control%fulsol ) l = n 
       IF ( control%print_level >= 10 ) l = n

       WRITE( out, "( '  Solution: ', /,'                        ',            &
      &               '        <------ bounds ------> ', /                     &
      &               '      # name          value   ',                        &
      &               '    lower       upper        dual ' )" )
       DO j = 1, 2 
         IF ( j == 1 ) THEN 
           ir = 1 ; ic = MIN( l, n ) 
         ELSE 
           IF ( ic < n - l )                                                   &
             WRITE( out, "( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )" )
           ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
         END IF 
         DO i = ir, ic 
           WRITE( out, "( I7, 1X, A10, 4ES12.4 )" )                            &
             i, data%X_name( i ), X( i ), X_l( i ), X_u( i ), data%G( i )
         END DO
       END DO
       WRITE( out, "( ' ' )" )
     END IF

     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
     RETURN

!  Other errors

 990 CONTINUE
     CALL CPU_TIME( time_new ); inform%time%total = time_new - inform%time%total
!    inform%obj = f
     WRITE( control%error, "( /, ' ** Message from -TRTN_solve-',              &
    &  '    Error exit (status = ', I6, ')', / )" ) inform%status
     RETURN

!  Non-executable statements

 2030 FORMAT( /, '  iter   objective norm_gr  step ',                          &
                 '  radius ared/pred   cg its    CPU' )
 2060 FORMAT( '   **  Warning ', I6, ' from ', A15 ) 
 2070 FORMAT( '   ==>  increasing pivot tolerance to ', ES12.4 )
 2080 FORMAT( /, '  End of inner iteration (status = ', I2, '): ' )
 2090 FORMAT( '   norm of gradient ', ES10.4, ' smaller than required ',       &
              ES10.4 )
 2100 FORMAT( '   **  Error ', I6, ' from ', A15 ) 

     END SUBROUTINE TRTN_solve

!-*- G A L A H A D  -  S U P E R B _ i t e r a t i v e _ r e f i n e m e n t -*-

     SUBROUTINE TRTN_iterative_refinement( P, FACTORS, CNTL, SOL, RHS, RES,    &
                                           BEST, res_norm, big_res, itref_max, &
                                           print_level, out, P_pert )
                                        
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute the solution to the preconditioned system 

!      P sol  = rhs

!  using iterative refinement

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SMT_type ), INTENT( IN ) :: P
     TYPE ( SILS_factors ), INTENT( IN ) :: FACTORS
     TYPE ( SILS_control ), INTENT( IN ) :: CNTL
     INTEGER, INTENT( IN ) :: itref_max, print_level, out
     REAL ( KIND = wp ), INTENT( OUT ) :: res_norm
     LOGICAL, INTENT( OUT ) :: big_res
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( P%n ) :: RHS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( P%n ) :: SOL, RES, BEST
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( * ) :: P_pert
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, it, j, l
     REAL ( KIND = wp ) ::  old_res, val, res_stop
     TYPE ( SILS_sinfo ) :: SINFO

     big_res = .FALSE.
     old_res = NRM2( P%n, RHS, 1 )
     res_stop = MAX( res_large, old_res )

!    WRITE( out, "( ' rhs ', /, ( 5ES12.4 ) )" ) RHS
     SOL = RHS
     CALL SILS_solve( P, FACTORS, SOL, CNTL, SINFO )
!    WRITE( out, "( ' sol ', /, ( 5ES12.4 ) )" ) SOL

!  Perform the iterative refinement

     DO it = 1, itref_max

!  Compute the residual

       RES = RHS
       DO l = 1, P%ne
         i = P%row( l ) ; j = P%col( l ) ; val = P%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO
       IF ( PRESENT( P_pert ) ) RES = RES - P_pert( : P%n ) * SOL
       res_norm = NRM2( P%n, RES, 1 )
!      WRITE( out, "( ' res ', /, ( 5ES12.4 ) )" ) RES
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm
       IF ( res_norm > res_stop ) THEN
         WRITE( out, "( /, ' ** => res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF

!  If the norm has increased, quit

       IF ( it > 1 ) THEN
         IF ( res_norm < old_res ) THEN
           old_res = res_norm
           BEST = SOL
         ELSE
           res_norm = old_res
           SOL = BEST
           GO TO 100
         END IF
       ELSE
         BEST = SOL
       END IF

!  Obtain a new correction

       CALL SILS_solve( P, FACTORS, RES, CNTL, SINFO )
       SOL = SOL + RES

!  End of iterative refinement

     END DO

!  Obtain final residuals if required

     IF ( it >= itref_max .OR. itref_max == 0 ) THEN
       RES = RHS
       DO l = 1, P%ne
         i = P%row( l ) ; j = P%col( l ) ; val = P%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO
       IF ( PRESENT( P_pert ) ) RES = RES - P_pert( : P%n ) * SOL
       res_norm = NRM2( P%n, RES, 1 )
       IF ( res_norm > res_stop ) THEN
         WRITE( out, "( /, ' ** => res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm
     END IF

 100 CONTINUE
     RETURN

!  Non-executable statements

 2000 FORMAT( '    residual = ', ES12.4 )

!  End of subroutine TRTN_iterative_refinement

     END SUBROUTINE TRTN_iterative_refinement

     SUBROUTINE TRTN_transform( n, Y, X_l, X_u, infinity,                      &
                                X_value, X_gradient, X_hessian )

!  Compute the nonlinear scaling function x(y) and its derivatives

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN )  :: infinity
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN )  :: Y, X_l, X_u
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( OUT ),                       &
                         OPTIONAL :: X_value, X_gradient, X_hessian

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: ey, l, u

     DO i = 1, n
       l = X_l( i ) ; u = X_u( i )
       IF ( l == u ) THEN
         IF ( PRESENT( X_value ) )                                             &
           X_value( i ) = Y( i )
         IF ( PRESENT( X_gradient ) )                                          &
           X_gradient( i ) = zero
         IF ( PRESENT( X_hessian ) )                                           &
           X_hessian( i ) = zero
       ELSE IF ( l > - infinity .AND. u < infinity ) THEN
         IF ( Y( i ) >= zero ) THEN
           ey = EXP( - Y( i ) )
           IF ( PRESENT( X_value ) )                                           &
             X_value( i ) = ( l * ey + u ) / ( one + ey )
           IF ( PRESENT( X_gradient ) )                                        &
             X_gradient( i ) = ( u - l ) * ey / ( one + ey ) ** 2
           IF ( PRESENT( X_hessian ) )                                         &
             X_hessian( i ) = ( u - l ) * ey * ( ey - one ) / ( one + ey ) ** 3
         ELSE
           ey = EXP( Y( i ) )
           IF ( PRESENT( X_value ) )                                           &
             X_value( i ) = ( l + u * ey ) / ( one + ey )      
           IF ( PRESENT( X_gradient ) )                                        &
             X_gradient( i ) = ( u - l ) * ey / ( one + ey ) ** 2
           IF ( PRESENT( X_hessian ) )                                         &
             X_hessian( i ) = ( u - l ) * ey * ( one - ey ) / ( one + ey ) ** 3
         END IF
       ELSE IF ( l > - infinity ) THEN
         ey = EXP( Y( i ) )
         IF ( PRESENT( X_value ) )                                             &
           X_value( i ) = l + ey
         IF ( PRESENT( X_gradient ) )                                          &
           X_gradient( i ) = ey
         IF ( PRESENT( X_hessian ) )                                           &
           X_hessian( i ) = ey
       ELSE IF ( u < infinity ) THEN
         ey = EXP( Y( i ) )
         IF ( PRESENT( X_value ) )                                             &
           X_value( i ) = u - ey
         IF ( PRESENT( X_gradient ) )                                          &
           X_gradient( i ) = - ey
         IF ( PRESENT( X_hessian ) )                                           &
           X_hessian( i ) = -ey
       ELSE
         IF ( PRESENT( X_value ) )                                             &
           X_value( i ) = Y( i )
         IF ( PRESENT( X_gradient ) )                                          &
           X_gradient( i ) = one
         IF ( PRESENT( X_hessian ) )                                           &
           X_hessian( i ) = zero
       END IF
     END DO

     RETURN

!    End of subroutine TRTN_transform

     END SUBROUTINE TRTN_transform

     SUBROUTINE TRTN_inverse_transform( n, X, X_l, X_u, infinity, pert_x, Y )

!  Compute the inverse of the nonlinear scaling function x(y)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN )  :: infinity, pert_x
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN )  :: X_l, X_u
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( INOUT )  :: X
     REAL ( KIND = wp ), DIMENSION( n ), INTENT( OUT ) :: Y

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: l, u

     DO i = 1, n
       l = X_l( i ) ; u = X_u( i )
       
!  Ensure that X is at least pert_x from its nearest bound.
!  Then evaluate y = x(inv)

       IF ( l == u ) THEN
         X( i ) = l
         Y( i ) = X( i )
       ELSE IF ( l > - infinity .AND. u < infinity ) THEN
         IF ( u - l <= two * pert_x ) THEN 
            X( i ) = half * ( u + l )
         ELSE IF ( X( i ) < l + pert_x ) THEN
            X( i ) = l + pert_x
         ELSE IF ( X( i ) > u - pert_x ) THEN
            X( i ) = u - pert_x
         END IF
         Y( i ) = LOG( ( X( i ) - l ) / ( u - X( i ) ) )
       ELSE IF ( l > - infinity ) THEN
         IF ( X( i ) < l + pert_x ) X( i ) = l + pert_x
         Y( i ) = LOG( X( i ) - l ) 
       ELSE IF ( u < infinity ) THEN
         IF ( X( i ) > u - pert_x ) X( i ) = u - pert_x
         Y( i ) = LOG( u - X( i ) )
       ELSE
         Y( i ) = X( i )
       END IF
     END DO

     RETURN

!    End of subroutine TRTN_inverse_transform

     END SUBROUTINE TRTN_inverse_transform

   END MODULE GALAHAD_TRTN_double
   

   





