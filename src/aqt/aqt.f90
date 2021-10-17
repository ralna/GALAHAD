! THIS VERSION: GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ A Q T  double  M O D U L E  *-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. October 8th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_AQT_double

!      -----------------------------------------------
!      |                                             |
!      | Approximately solve the quadratic program   |
!      |                                             |
!      |    minimize     1/2 <x,Hx> + <c,x> + f0     |
!      |    subject to   <x,Mx> <= radius^2          |
!      |                                             |
!      | using a subspace PCG/Lanczos method         |
!      |                                             |
!      -----------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_ROOTS_double, ONLY: ROOTS_quadratic, ROOTS_quartic
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_LAPACK_interface, ONLY : LAEV2


      IMPLICIT NONE

      PRIVATE
      PUBLIC :: AQT_initialize, AQT_read_specfile, AQT_solve,                  &
                AQT_terminate, AQT_full_initialize, AQT_full_terminate,        &
                AQT_solve_2d, AQT_import, AQT_information

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: roots_tol = ten ** ( - 12 )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )
      INTEGER, PARAMETER :: itref_max = 1
      LOGICAL :: roots_debug = .FALSE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: AQT_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the maximum number of iterations allowed (-ve = no bound)

        INTEGER :: itmax = - 1

!   the maximum number of iterations allowed once the boundary has been
!    encountered (-ve = no bound)

        INTEGER :: phase_2_itmax = - 1

!   the iteration stops successfully when the gradient in the M(inverse) norm
!    is smaller than max( stop_relative * initial M(inverse)
!                         gradient norm, stop_absolute )

        REAL ( KIND = wp ) :: stop_relative = epsmch
        REAL ( KIND = wp ) :: stop_absolute = zero

!   an estimate of the solution that gives at least %fraction_opt times
!    the optimal objective value will be found

        REAL ( KIND = wp ) :: fraction_opt = one

!   the iteration stops if the objective-function value is lower than f_min

        REAL ( KIND = wp ) :: f_min = - ( biginf / two )

!   the smallest value that the square of the M norm of the gradient of the
!    the objective may be before it is considered to be zero

        REAL ( KIND = wp ) :: theta_zero = ten * epsmch

!   the constant term, f0, in the objective function

        REAL ( KIND = wp ) :: f_0 = zero

!   is M the identity matrix ?

        LOGICAL :: unitm = .TRUE.

!   should the iteration stop when the Trust-region is first encountered ?

        LOGICAL :: steihaug_toint  = .FALSE.

!  is the solution thought to lie on the constraint boundary ?

        LOGICAL :: boundary  = .FALSE.

!  if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!  if %deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '
      END TYPE AQT_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: AQT_inform_type

!  return status. See AQT_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the total number of pass-2 iterations required if the solution lies on
!   the trust-region boundary

        INTEGER :: iter_pass2 = - 1

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = wp ) :: multiplier = zero

!  the M-norm of x

        REAL ( KIND = wp ) :: mnormx = zero

!  the latest pivot in the Cholesky factorization of the Lanczos tridiagonal

        REAL ( KIND = wp ) :: piv = biginf

!  the most negative cuurvature encountered

        REAL ( KIND = wp ) :: curv = biginf

!  the current Rayleigh quotient

        REAL ( KIND = wp ) :: rayleigh = biginf

!  an estimate of the leftmost generalized eigenvalue of the pencil (H,M)

        REAL ( KIND = wp ) :: leftmost = biginf

!  was negative curvature encountered ?

        LOGICAL :: negative_curvature = .FALSE.

!  did the hard case occur ?

        LOGICAL :: hard_case = .FALSE.
      END TYPE AQT_inform_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: AQT_data_type
        PRIVATE
        INTEGER :: branch = 100
        INTEGER :: iter, itm1, itmax, phase_2_itmax
        REAL ( KIND = wp ) :: alpha, theta, theta_old, pgnorm, radius2
        REAL ( KIND = wp ) :: stop, normp, xmx, xmp, pmp, gamma, h_xx, c_x
        LOGICAL :: printi, printd, interior
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Q
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_old
      END TYPE AQT_data_type

      TYPE, PUBLIC :: AQT_full_data_type
        LOGICAL :: f_indexing
        TYPE ( AQT_data_type ) :: AQT_data
        TYPE ( AQT_control_type ) :: AQT_control
        TYPE ( AQT_inform_type ) :: AQT_inform
      END TYPE AQT_full_data_type

    CONTAINS

!-*-*-*-*-*-  A Q T _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE AQT_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  .  Set initial values for the AQT control parameters  .
!
!  Argument:
!  =========
!
!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( AQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( AQT_control_type ), INTENT( OUT ) :: control
      TYPE ( AQT_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set initial control parameter values

      control%stop_relative = SQRT( epsmch )

!  Set branch for initial entry

      data%branch = 100

      RETURN

!  End of subroutine AQT_initialize

      END SUBROUTINE AQT_initialize

!- G A L A H A D -  A Q T _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE AQT_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for AQT controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AQT_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( AQT_control_type ), INTENT( OUT ) :: control
     TYPE ( AQT_inform_type ), INTENT( OUT ) :: inform

     CALL AQT_initialize( data%aqt_data, control, inform )

     RETURN

!  End of subroutine AQT_full_initialize

     END SUBROUTINE AQT_full_initialize

!-*-*-*-   A Q T _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE AQT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by AQT_initialize could (roughly)
!  have been set as:

!  BEGIN AQT SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-number-of-iterations                    -1
!   maximum-number-of-phase-2-iterations            -1
!   relative-accuracy-required                      1.0E-8
!   absolute-accuracy-required                      0.0
!   fraction-optimality-required                    1.0
!   small-f-stop                                    - 1.0D+100
!   zero-gradient-tolerance                         2.0E-15
!   constant-term-in-objective                      0.0
!   two-norm-trust-region                           T
!   stop-as-soon-as-boundary-encountered            F
!   solution-is-likely-on-boundary                  F
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END AQT SPECIFICATIONS

!  Dummy arguments

      TYPE ( AQT_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: itmax = print_level + 1
      INTEGER, PARAMETER :: phase_2_itmax = itmax + 1
      INTEGER, PARAMETER :: stop_relative = phase_2_itmax + 1
      INTEGER, PARAMETER :: stop_absolute = stop_relative + 1
      INTEGER, PARAMETER :: fraction_opt = stop_absolute + 1
      INTEGER, PARAMETER :: theta_zero = fraction_opt + 1
      INTEGER, PARAMETER :: f_0 = theta_zero + 1
      INTEGER, PARAMETER :: f_min = f_0 + 1
      INTEGER, PARAMETER :: unitm = f_min + 1
      INTEGER, PARAMETER :: steihaug_toint = unitm + 1
      INTEGER, PARAMETER :: boundary = steihaug_toint + 1
      INTEGER, PARAMETER :: space_critical = boundary + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'AQT'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( itmax )%keyword = 'maximum-number-of-iterations'
      spec( phase_2_itmax )%keyword = 'maximum-number-of-phase-2-iterations'

!  real key-words

      spec( stop_relative )%keyword = 'relative-accuracy-required'
      spec( stop_absolute )%keyword = 'absolute-accuracy-required'
      spec( fraction_opt )%keyword = 'fraction-optimality-required'
      spec( theta_zero )%keyword = 'zero-gradient-tolerance'
      spec( f_0 )%keyword = 'constant-term-in-objective'
      spec( f_min )%keyword = 'small-f-stop'

!  logical key-words

      spec( unitm )%keyword = 'two-norm-trust-region'
      spec( steihaug_toint )%keyword = 'stop-as-soon-as-boundary-encountered'
      spec( boundary )%keyword = 'solution-is-likely-on-boundary'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!  read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itmax ),                               &
                                  control%itmax,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( phase_2_itmax ),                       &
                                  control%phase_2_itmax,                       &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( stop_relative ),                       &
                                  control%stop_relative,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_absolute ),                       &
                                  control%stop_absolute,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( fraction_opt ),                        &
                                  control%fraction_opt,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( f_min ),                               &
                                  control%f_min,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( theta_zero ),                         &
                                  control%theta_zero,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( f_0 ),                                 &
                                  control%f_0,                                 &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( unitm ),                               &
                                  control%unitm,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( steihaug_toint ),                      &
                                  control%steihaug_toint,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( boundary ),                            &
                                  control%boundary,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

      RETURN

      END SUBROUTINE AQT_read_specfile

!-*-*-*-*-*-*-*-*-*-*  A Q T _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE AQT_solve( n, radius, f, X, C, VECTOR, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   n        number of unknowns
!   radius   trust-region radius
!   f        the value of the quadratic function at the current point.
!            Need not be set on entry. On exit it will contain the value at
!            the best point found
!   X        the vector of unknowns. Need not be set on entry.
!            On exit, the best value found so far
!   C        On entry this must contain the vector c
!   VECTOR   see inform%status = 2 and 3
!   data     private internal data
!   control  a structure containing control information. See AQT_initialize
!   inform   a structure containing information. The component
!             %status is the input/output status. This must be set to 1 on
!              initial entry or 4 on a re-entry when only radius has
!              been reduced since the last entry. Other values are
!               2 on exit => the inverse of M must be applied to
!                 VECTOR with the result returned in VECTOR and the subroutine
!                 re-entered. This will only happen if unitm is .FALSE.
!               3 on exit => the product H * VECTOR must be formed, with
!                 the result returned in VECTOR and the subroutine re-entered
!               4 The iteration is to be restarted with a smaller radius but
!                 with all other data unchanged. Set R to c for this entry.
!               5 The iteration will be restarted. Reset R to c and re-enter.
!                 This exit will only occur if control%steihaug_toint is
!                 .FALSE. and the solution lies on the trust-region boundary
!               0 the solution has been found
!              -1 an array allocation has failed
!              -2 an array deallocation has failed
!              -3 n and/or radius is not positive
!              -15 the matrix M appears to be indefinite
!              -18 the iteration limit has been exceeded
!              -30 the trust-region has been encountered in Steihaug-Toint mode
!              -31 the function value is smaller than control%f_min
!             the remaining components are described in the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      REAL ( KIND = wp ), INTENT( INOUT ) :: f
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, C, VECTOR
      TYPE ( AQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( AQT_control_type ), INTENT( IN ) :: control
      TYPE ( AQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: it, itp1, nroots, info, status
      REAL ( KIND = wp ) :: alpha, beta, other_root, xmx_trial, sqrt_theta
      REAL ( KIND = wp ) :: h_xx, h_xq, h_qq, c_x, c_q, x_x, x_q, lambda, f_tol
      REAL ( KIND = wp ) :: delta
      LOGICAL :: filexx
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Branch to different sections of the code depending on input status

      IF ( inform%status == 1 ) THEN
        data%branch = 10
      ELSE IF ( inform%status == 4 ) THEN
        data%branch = 600
      END IF

      SELECT CASE ( data%branch )
      CASE ( 10 )
        GO TO 10
      CASE ( 110 )
        GO TO 110
      CASE ( 120 )
        GO TO 120
      CASE ( 210 )
        GO TO 210
      CASE ( 220 )
        GO TO 220
      CASE ( 600 )
        GO TO 600
      CASE ( 900 )
        GO TO 900
      END SELECT

!  on initial entry, set constants

   10 CONTINUE

!  check for obvious errors

      IF ( n <= 0 ) GO TO 940
      IF ( radius <= zero ) GO TO 950

      data%iter = 0 ; data%itm1 = - 1
      data%itmax = control%itmax ; IF ( data%itmax < 0 ) data%itmax = n
      data%phase_2_itmax = control%phase_2_itmax
      IF ( data%phase_2_itmax < 0 ) data%phase_2_itmax = n
      data%printi = control%out > 0 .AND. control%print_level >= 1
      data%printd = control%out > 0 .AND. control%print_level >= 2
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter_pass2 = 0 ; inform%negative_curvature = .FALSE.
      inform%curv = HUGE( one ) ; inform%piv = HUGE( one )
      inform%rayleigh = HUGE( one )
      inform%mnormx = radius ; inform%multiplier = zero
      inform%hard_case = .FALSE.
      data%interior = .TRUE.
      data%radius2 = radius * radius
      X = zero ; f = control%f_0

!  =====================
!  Array (re)allocations
!  =====================

      array_name = 'aqt: P'
      CALL SPACE_resize_array( n, data%P,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'aqt: Q'
      CALL SPACE_resize_array( n, data%Q,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'aqt: R'
      CALL SPACE_resize_array( n, data%R,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'aqt: W'
      CALL SPACE_resize_array( n, data%W,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'aqt: W_old'
      CALL SPACE_resize_array( n, data%W_old,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      data%R( : n ) = C( : n )

!  =======================================================================
!  Start of the main phase-1 (preconditioned conjugate-gradient) iteration
!  =======================================================================

  100 CONTINUE
      inform%iter = data%iter

!  --------------------------------------------
!  obtain the preconditioned residual v = M^-1 r 
!  (r output in VECTOR, v returned in VECTOR)
!  --------------------------------------------

      VECTOR( : n ) = data%R( : n )
      IF ( .NOT. control%unitm ) THEN
        data%branch = 110 ; inform%status = 2 ; RETURN
      END IF

!  obtain the scaled norm of the residual, pgnorm = sqrt(r'v)

  110 CONTINUE
      data%theta = DOT_PRODUCT( data%R( : n ), VECTOR( : n ) )
      IF ( ABS( data%theta ) < control%theta_zero ) data%theta = zero
      IF ( data%theta < zero ) THEN
        IF ( MAXVAL( ABS( VECTOR( : n ) ) ) <                                  &
               epsmch * MAXVAL( ABS( data%R( : n ) ) ) ) THEN
          data%theta = zero
        ELSE
          GO TO 930
        END IF
      END IF
      sqrt_theta = SIGN( SQRT( ABS( data%theta ) ), data%theta )
      data%pgnorm = sqrt_theta

!  compute beta = r'v / (r'v)_old

      IF ( data%iter > 0 ) THEN
        beta = data%theta / data%theta_old

!  record the scalar gamma and vector w_old that will be needed by phase 2

        data%gamma = SQRT( beta ) / ABS( data%alpha )
        data%W_old( : n ) = data%W( : n )

!  compute the stopping tolerance

      ELSE
        data%gamma = sqrt_theta
        data%xmx = zero
        data%stop = MAX( control%stop_relative * data%pgnorm,                  &
                         control%stop_absolute )
        IF ( data%printi )                                                     &
          WRITE( control%out, "( /, A, ' stopping tolerance = ',               &
         &         ES10.4, ', radius = ', ES10.4 )" ) prefix, data%stop, radius
      END IF

!  record the vectors w and q that will be needed by phase 2

      data%W( : n ) = data%R( : n ) / sqrt_theta
      data%Q( : n ) = VECTOR( : n ) / sqrt_theta

      IF ( data%printi ) THEN
        IF ( MOD( data%iter, 25 ) == 0 .OR. data%printd ) THEN
          IF ( data%interior ) THEN
            WRITE( control%out, "( /, A, '   Iter        f        ',           &
           &       ' pgnorm    step    norm p   norm x     curv ' )" ) prefix
          ELSE
            WRITE( control%out, 2000 ) prefix
          END IF
        ELSE
          IF ( .NOT. data%interior .AND. data%iter /= 0 .AND. data%printd )    &
            WRITE( control%out, 2000 ) prefix
        END IF

        IF ( data%interior ) THEN
          IF ( data%iter /= 0 ) THEN
            WRITE( control%out, "( A, I7, ES16.8, 4ES9.2, ES10.2 )" )          &
                   prefix, data%iter, f, data%pgnorm, data%alpha,              &
                   data%normp, SQRT( data%xmx ), inform%rayleigh
          ELSE
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, 4X, '-',              &
           &      2( 8X, '-' ), 9X, '-' )" ) prefix, data%iter, f, data%pgnorm
          END IF
        ELSE
          IF ( data%iter /= 0 ) THEN
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, ES9.2 )")&
                   prefix, data%iter, f
          END IF
        END IF
      END IF

!  test for an interior approximate solution

      IF ( data%interior .AND. data%pgnorm <= data%stop ) THEN
        IF ( data%printi ) WRITE( control%out,                                 &
          "( A, ' pgnorm ', ES10.4, ' < ', ES10.4 )" )                         &
             prefix, data%pgnorm, data%stop
        inform%mnormx = SQRT( data%xmx )
        GO TO 900
      END IF

!  test to see that iteration limit has not been exceeded

      IF ( data%iter > 0 ) THEN
        IF ( data%iter >= data%itmax .AND. data%interior ) THEN
          inform%mnormx = SQRT( data%xmx )
          GO TO 910
        END IF

!  compute a new search direction p that maintains conjugacy

        data%xmp = beta * ( data%xmp + data%alpha * data%pmp )
        data%pmp = data%theta + data%pmp * beta * beta
        data%P( : n ) = - VECTOR + beta * data%P( : n )

!  special case for the first iteration

      ELSE
        data%P( : n ) = - VECTOR
        data%pmp = data%theta ; data%xmp = zero
      END IF

      data%theta_old = data%theta

!  compute the 2-norm of the search direction

      data%normp = TWO_NORM( data%P( : n ) )

!  test for convergence

      IF ( data%interior .AND. data%normp <= two * epsmch ) THEN
        IF ( data%printi ) WRITE( control%out, "( A, ' pnorm ', ES12.4,        &
       &   ' < ', ES12.4 )" ) prefix, data%normp, ten * epsmch
        inform%mnormx = SQRT( data%xmx )
        GO TO 900
      END IF

      data%itm1 = data%iter ; data%iter = data%iter + 1

!  ------------------------------------------
!  Obtain the product w = H p 
!  (p output in VECTOR, w returned in VECTOR)
!  ------------------------------------------

      VECTOR = data%P( : n ) ; inform%iter = data%iter
      data%branch = 120 ; inform%status = 3 ; RETURN

!  calculate the curvature p' w = p' H p 

  120 CONTINUE
      inform%curv = DOT_PRODUCT( data%P( : n ) , VECTOR( : n ) )
      inform%rayleigh = inform%curv / data%pmp

!  calculate the stepsize alpha = r M^-1 r / p' H p

      IF ( inform%curv > zero ) THEN
        data%alpha = data%theta / inform%curv
      ELSE IF ( inform%curv == zero ) THEN
        data%alpha = HUGE( one ) ** 0.25
      ELSE
        inform%negative_curvature = .TRUE.
      END IF

!  See if the new estimate of the solution is interior


!  the new point will be on the boundary if there is negative curvature

      IF ( inform%negative_curvature ) THEN
        data%interior = .FALSE.

!  if negative curvature has not been detected, find the square of the
!  norm of the new point, and compare this with the radius

      ELSE
        xmx_trial = data%xmx +                                                 &
           data%alpha * ( data%xmp + data%xmp + data%alpha * data%pmp )

!  the new point is interior

        IF ( xmx_trial <= data%radius2 ) THEN
          data%xmx = xmx_trial
          X = X + data%alpha * data%P( : n )
          f = f - half * data%alpha * data%alpha * inform%curv

!  the new point lies on the boundary

        ELSE
          data%interior = .FALSE.
        END IF
      END IF

!  if the new point is on the boundary, compute the required step

      IF ( .NOT. data%interior ) THEN
        data%itmax = min( data%itmax, data%iter + data%phase_2_itmax )

!  find the boundary point and the value of the objective there

        IF ( data%iter == 1 .OR. control%steihaug_toint ) THEN
          CALL ROOTS_quadratic( data%xmx - data%radius2, two * data%xmp,       &
                                data%pmp, roots_tol, nroots, other_root,       &
                                alpha, roots_debug )
          data%xmx = data%xmx + alpha * ( two * data%xmp + alpha * data%pmp )
          X( : n ) = X( : n ) + alpha * data%P( : n )
          f = f + alpha * ( half * alpha * inform%curv - data%theta )
          inform%mnormx = SQRT( data%xmx )

!  terminate if f is smaller than the permitted minimum

          IF ( f < control%f_min ) GO TO 970

!  if the Steihaug-Toint strategy is to be used, stop on the boundary

          IF ( control%steihaug_toint ) THEN
            data%alpha = alpha ; GO TO 920

!  if a more accurate solution is required, switch to phase 2

          ELSE
            IF ( data%printi ) THEN
               WRITE( control%out, "( /, A, ' Boundary encountered,',          &
              &   ' entering phase 2' )" ) prefix
               IF ( .NOT. data%printd ) WRITE( control%out, 2000 ) prefix
            END IF
            GO TO 200
          END IF

!  if this is not the first iteration, move to phase 2

        ELSE
          VECTOR = data%Q( : n ) ; inform%iter = data%iter
          data%branch = 220 ; inform%status = 3 ; RETURN
          GO TO 220
        END IF

!  if this is the first iteration, move to the boundary

      END IF

!  update the residual r <- r + alpha v

      data%R = data%R + data%alpha * VECTOR

!  =====================================================================
!  End of the main phase-1 (preconditioned conjugate-gradient) iteration
!  =====================================================================

      GO TO 100

!  if the solution lies on the boundary, record the current M-orthogonal 
!  Lanczos vector, q = v / pgnorm

!      IF ( .NOT.  data%interior ) THEN
!        data%Q( : n ) = data%VECTOR( : n ) / data%pgnorm

!  M-orthogonalize q with respect to the current estimate of the mininizer, x,
!  i.e., find d = q - omega x such that d' Mx = 0, which gives that
!  omega = x'Mq / ||x||_M^2 = x'r / ( pgnorm * radius^2 )

!       omega = DOT_PRODUCT( data%R( : n ), X( : n ) ) / ( data%pgnorm )
!       d( : n ) = data%Q( : n ) - omega *  X( : n )


!  =======================================================================
!  Start of the main phase-2 (preconditioned Lanczos) iteration
!  =======================================================================

  200 CONTINUE
      inform%iter = data%iter

!  --------------------------------------------
!  obtain the preconditioned residual u = M^-1 r 
!  (r output in VECTOR, u returned in VECTOR)
!  --------------------------------------------

      VECTOR( : n ) = data%R( : n )
      IF ( .NOT. control%unitm ) THEN
        data%branch = 210 ; inform%status = 2 ; RETURN
      END IF

!  obtain the scaled norm of the residual, gamma = sqrt(r'u)

  210 CONTINUE
      data%theta = DOT_PRODUCT( data%R( : n ), VECTOR )
      IF ( ABS( data%theta ) < control%theta_zero ) data%theta = zero
      IF ( data%theta < zero ) THEN
        IF ( MAXVAL( ABS( VECTOR ) ) < epsmch * MAXVAL( ABS( data%R ) ) ) THEN
          data%theta = zero
        ELSE
          GO TO 930
        END IF
      END IF
      data%gamma = SIGN( SQRT( ABS( data%theta ) ), data%theta )

!  save the old w

      data%W_old( : n ) = data%W( : n )

!  calculate the vectors w and q

      data%W( : n ) = data%R( : n ) / data%gamma
      data%Q( : n ) = VECTOR( : n ) / data%gamma

      IF ( data%printi ) THEN
        IF ( MOD( data%iter, 25 ) == 0 .OR. data%printd ) THEN
          IF ( data%interior ) THEN
            WRITE( control%out, "( /, A, '   Iter        f        ',           &
           &       ' pgnorm    step    norm p   norm x     curv ' )" ) prefix
          ELSE
            WRITE( control%out, 2000 ) prefix
          END IF
        ELSE
          IF ( .NOT. data%interior .AND. data%iter /= 0 .AND. data%printd )    &
            WRITE( control%out, 2000 ) prefix
        END IF

        IF ( data%interior ) THEN
          IF ( data%iter /= 0 ) THEN
            WRITE( control%out, "( A, I7, ES16.8, 4ES9.2, ES10.2 )" )          &
                   prefix, data%iter, f, data%pgnorm, data%alpha,              &
                   data%normp, SQRT( data%xmx ), inform%rayleigh
          ELSE
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, 4X, '-',              &
           &      2( 8X, '-' ), 9X, '-' )" ) prefix, data%iter, f, data%pgnorm
          END IF
        ELSE
          IF ( data%iter /= 0 ) THEN
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, ES9.2 )")&
                   prefix, data%iter, f
          END IF
        END IF
      END IF

!  test to see that iteration limit has not been exceeded

      IF ( data%iter > 0 ) THEN
        IF ( data%iter >= data%itmax ) THEN
          GO TO 910
        END IF
      END IF

!  ------------------------------------------
!  Obtain the product t = H q
!  (q output in VECTOR, t returned in VECTOR)
!  ------------------------------------------

      VECTOR = data%Q( : n ) ; inform%iter = data%iter
      data%branch = 220 ; inform%status = 3 ; RETURN

!  calculate the curvature delta = t' q = q' H q

  220 CONTINUE
      WRITE( 6, "( ' q^T M x = ', ES12.4 )" )                                  &
        DOT_PRODUCT( data%Q( : n ), X( : n ) )

      delta = DOT_PRODUCT( data%Q( : n ), VECTOR( : n ) )

!  compute the new point x in the subspace spanned by the normalized 
!  old one x / ||x||_M and q

      h_xx = data%h_xx / data%xmx
      h_xq = DOT_PRODUCT( X( : n ), VECTOR( : n ) ) / inform%mnormx
      h_qq = delta
      c_x = data%c_x  / inform%mnormx
      c_q =  DOT_PRODUCT( C( : n ), data%Q( : n ) )
      CALL AQT_solve_2d( h_xx, h_xq, h_qq, c_x, c_q, radius,                   &
                         x_x, x_q, lambda, status )
      WRITE( 6, "( ' AQT_solve_2d status = ', I0 )" ) status
      X( : n ) = ( x_x / inform%mnormx ) * X( : n ) + x_q * data%Q( : n )
      data%h_xx = x_x * x_x * h_xx + two * x_x * x_q * h_xq + x_q * x_q * h_qq 
      data%c_x = x_x * c_x + x_q * c_q
      f = half * data%h_xx + data%c_x
      data%xmx = ( x_x / inform%mnormx ) ** 2 + x_q ** 2
      inform%mnormx = SQRT( data%xmx )

!  compute r = t - delta * w - gamma * w_old

      data%R( : n ) = VECTOR( : n ) - delta * data%W( : n )                    &
                       - data%gamma * data%W_old( : n )

!  ==========================================================
!  End of the main phase-2 (preconditioned Lanczos) iteration
!  ==========================================================

      GO TO 200

!  ======================================================
!  Re-entry for solution with smaller trust-region radius
!  ======================================================

  600 CONTINUE

!  ===============
!  Exit conditions
!  ===============

!  successful returns

  900 CONTINUE
      inform%status = GALAHAD_ok
      RETURN

!  too many iterations

  910 CONTINUE
      IF ( data%printi )                                                       &
        WRITE( control%out, "( /, A, ' Iteration limit exceeded ' ) " ) prefix
      inform%status = GALAHAD_error_max_iterations ; inform%iter = data%iter
      RETURN

!  boundary encountered in Steihaug-Toint method

  920 CONTINUE

!  find the gradient at the appropriate point on the boundary

!     R = R + data%alpha * VECTOR

!  record terminal information

      IF ( data%printi ) WRITE( control%out, "( A, I7, ES16.8, 4X, '-', 4X,    &
     &     3ES9.2, ES10.2, //, A,                                              &
     &     ' Now leaving trust region (Steihaug-Toint)' )" )                   &
          prefix, data%iter, f, data%alpha, data%normp, SQRT( data%xmx ),      &
          inform%rayleigh, prefix
      inform%status = GALAHAD_warning_on_boundary ; inform%iter = data%iter
      RETURN

!  unsuccessful returns

  930 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The matrix M appears to be indefinite. Inner product = ',     &
     &      ES12.4  )" ) prefix, data%theta
      inform%status = GALAHAD_error_preconditioner ; inform%iter = data%iter
      RETURN

  940 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' n = ', I6, ' is not positive ' )" ) prefix, n
      inform%status = GALAHAD_error_restrictions ; inform%iter = data%iter
      RETURN

  950 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, radius
      inform%status = GALAHAD_error_restrictions ; inform%iter = data%iter
      RETURN

!  Allocation or deallocation error

  960 CONTINUE
      inform%iter = data%iter
      RETURN

!  objective-function value is too small

  970 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The objective function value is smaller than', ES12.4 )" )    &
        prefix, control%f_min
      inform%status = GALAHAD_error_f_min ; inform%iter = data%iter
      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, '  Iter        f         pgnorm    tr it info' )

!  End of subroutine AQT_solve

      END SUBROUTINE AQT_solve

!-*-*-*-*-*-  A Q T _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE AQT_terminate( data, control, inform )

!  ..............................................
!  .                                            .
!  .  Deallocate arrays at end of AQT_solve    .
!  .                                            .
!  ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine AQT_initialize
!   inform  see Subroutine AQT_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( AQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( AQT_control_type ), INTENT( IN ) :: control
      TYPE ( AQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'aqt: P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'aqt: Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'aqt: W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'aqt: W_old'
      CALL SPACE_dealloc_array( data%W_old,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine AQT_terminate

      END SUBROUTINE AQT_terminate

! -  G A L A H A D -  A Q T _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

      SUBROUTINE AQT_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( AQT_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( AQT_control_type ), INTENT( IN ) :: control
      TYPE ( AQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      CALL AQT_terminate( data%aqt_data, control, inform )

      RETURN

!  End of subroutine AQT_full_terminate

      END SUBROUTINE AQT_full_terminate

!-*-*-*- G A L A H A D -  A Q T _ S O L V E _ 2 D   S U B R O U T I N E -*-*-*-

     SUBROUTINE AQT_solve_2d( h_11, h_12, h_22, g_1, g_2, radius,              &
                              x_1, x_2, lambda, status )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the minimizer x = (x1,x2) of the quadratic function
!
!    q(x) = 1/2 x^T H x + g^t x
!         = 1/2 h_11 x_1^2 + h_11 x_1 x_2 + 1/2 h_22 x_2^2 + g_1 x_1 + g_2 x_2
!
!  within the trust region ||x||^2 = x_1^2 + x_2^2 <= radius^2
!
!  The required solution satisfies the equations ( H + lambda I ) x = - g, i.e.,
!
!    ( h_11 + lambda      h_12      ) ( x_1 ) = - ( g_1 )
!    (      h_12      h_22 + lambda ) ( x_2 )     ( g_2 )
!
!  input arguments:
!  
!   h_11, h_12, h_22, g_1, g_2, radius - real, as above 
!
!  output arguments:
!  
!  x_1, x_2, lambda - real, as above
!  status - integer, output status. Possible values are
!             0  boundary solution, usual case
!             1  interior solution
!             2  pure eigenvector solution
!             3  boundary solution, hard (degenerate) case
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), INTENT( IN ) :: h_11, h_12, h_22, g_1, g_2, radius
     REAL ( KIND = wp ), INTENT( OUT ) :: x_1, x_2, lambda

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: leftmost, nroots
     REAL ( KIND = wp ) :: gamma_1, gamma_2, lambda_1, lambda_2, c, s
     REAL ( KIND = wp ) :: a4, a3, a2, a1, a0, tol, y_1, y_2, c1, c2
     REAL ( KIND = wp ) :: root1, root2, root3, root4, lambda_min
     LOGICAL :: interior, debug

!  compute the eigen-decomposition H = Q^T Lambda Q, i.e., 

!     ( h_11  h_12 ) = ( c -s ) ( lambda_1   0    ) (  c s )
!     ( h_12  h_22 )   ( s  c ) (     0  lambda_2 ) ( -s c )

!    WRITE( 6, "( ' h_11, h_12, h_22 = ', 3ES12.4 )" ) h_11, h_12, h_22
     CALL LAEV2( h_11, h_12, h_22, lambda_1, lambda_2, c, s )
     lambda_min = MAX( - lambda_1, - lambda_2, zero )
!    WRITE( 6, "( ' lambda_1, lambda_2, lambda_min = ', 3ES12.4 )" )           &
!      lambda_1, lambda_2, lambda_min

!  record the leftmost negative eigenvalue

     IF ( lambda_1 >= zero .AND. lambda_2 >= zero ) THEN
       leftmost = 0
     ELSE IF ( lambda_1 < lambda_2 ) THEN
       leftmost = 1
     ELSE IF ( lambda_1 == lambda_2 ) THEN
       leftmost = 12
     ELSE 
       leftmost = 2
     END IF

!  deal with the special case for which g = 0

     IF ( g_1 == zero .AND. g_2 == zero ) THEN

!  if the eigenvalues are non-negative, the solution is zero

       IF ( leftmost == 0 ) THEN
         x_1 = zero ; x_2 = zero ; lambda = zero
         status = 1 ; RETURN

!  otherwise, the solution is radius times the eigenvector for the 
!  leftmost eigenvalue

       ELSE IF ( leftmost == 1 ) THEN
         x_1 = radius * c ; x_2 = radius * s ; lambda = - lambda_1
         status = 2 ; RETURN
       ELSE
         x_1 = - radius * s ; x_2 = radius * c ; lambda = - lambda_2
         status = 2 ; RETURN
       END IF
     END IF
  
!  g is nonzero. Compute the vector gamma = Q g, i.e, 

!   ( gamma_1 ) =  (  c s ) ( g_1 )
!   ( gamma_2 )    ( -s c ) ( g_2 )

     gamma_1 = c * g_1 + s * g_2
     gamma_2 = - s * g_1 + c * g_2

!  is there an interior solution x = - H^-1 g?

     IF ( leftmost == 0 ) THEN
       interior = .TRUE.

!  set y_i = - gamma_i / lambda_i, i = 1, 2, if that is possible

       IF ( lambda_1 > 0 ) THEN
         y_1 = - gamma_1 / lambda_1
       ELSE IF ( gamma_1 == 0 ) THEN
         y_1 = zero
       ELSE 
         interior = .FALSE.
       END IF
       IF ( lambda_2 > 0 ) THEN
         y_2 = - gamma_2 / lambda_2
       ELSE IF ( gamma_2 == 0 ) THEN
         y_2 = zero
       ELSE 
         interior = .FALSE.
       END IF

!  try x = Q^T y, i.e., 

!   ( x_1 ) =  ( c -s ) ( y_1 )
!   ( x_2 )    ( s  c ) ( y_2 )

       IF ( interior ) THEN
         x_1 = c * y_1 - s * y_2 ; x_2 = s * y_1 + c * y_2

!  does x lie within the trust region?

         IF ( x_1 ** 2 + x_2 **2 <= radius ** 2 ) THEN
           lambda = zero
           status = 1 ; RETURN
         END IF
       END IF
     END IF

!  if gamma_1 and gamma_2 are nonzero, solve the secular equation

!             gamma_1^2                 gamma_2^2       
!     ----------------------- + ----------------------- = radius^2
!     ( lambda + lambda_1 )^2   ( lambda + lambda_2 )^2

!  or equivalently the quartic equation

!     ( lambda + lambda_1 )^2 ( lambda + lambda_2 )^2
!       - ( gamma_1 / radius )^2 ( lambda + lambda_2 )^2 
!       - ( gamma_2 / radius )^2 ( lambda + lambda_2 )^2 = 0

     IF ( gamma_1 /= zero .AND. gamma_2 /= zero ) THEN
       c1 = ( gamma_1 / radius ) ** 2
       c2 = ( gamma_2 / radius ) ** 2

       a4 = one
       a3 = two * ( lambda_1 + lambda_2 )
       a2 = lambda_1 ** 2 + lambda_2 ** 2 + four * lambda_1 * lambda_2         &
            - c1 - c2
       a1 = two * lambda_1 * lambda_2 * ( lambda_1 + lambda_2 )                &
            - two * ( c1 * lambda_2 + c2 * lambda_1 )
       a0 = ( lambda_1 * lambda_2 ) ** 2                                       &
            - c1 * lambda_2 ** 2 - c2 * lambda_1 ** 2
       CALL ROOTS_quartic( a0, a1, a2, a3, a4, epsmch,                         &
                           nroots, root1, root2, root3, root4, roots_debug )

!  record the required root

       IF ( nroots == 4 ) THEN
         lambda = root4
       ELSE IF ( nroots == 2 ) THEN
         lambda = root2
       ELSE
         WRITE( 6, "( ' Should not be here !!' )" )
         WRITE( 6, "( I0, ' roots ' )" ) nroots
         GO TO 900
       END IF
       status = 0

!  if gamma_i is zero but gamma_j isn't, solve the secular equation

!             gamma_j^2      
!     ----------------------- = radius^2
!     ( lambda + lambda_j )^2

!  or equivalently the quartic equation

!     ( lambda + lambda_j )^2 - ( gamma_j radius )^2 = 0

     ELSE IF ( gamma_1 /= zero .OR. gamma_2 /= zero ) THEN
       IF ( gamma_1 /= zero ) THEN
         a2 = one ; a1 = two * lambda_1
         a0 = lambda_1 ** 2 - ( gamma_1 / radius ) ** 2
       ELSE
         a2 = one ; a1 = two * lambda_2
         a0 = lambda_2 ** 2 - ( gamma_2 / radius ) ** 2
       END IF
       CALL ROOTS_quadratic( a0, a1, a2, epsmch,                               &
                             nroots, root1, root2, roots_debug )

!  record the required root, and check for the hard case

       IF ( nroots == 2 ) THEN
         lambda = root2
         IF ( root2 < lambda_min ) THEN
           lambda = lambda_min
           status = 3
         ELSE
           lambda = root2
           status = 0
         END IF
       ELSE
         WRITE( 6, "( ' Should not be here !!' )" )
         WRITE( 6, "( I0, ' roots ' )" ) nroots
         GO TO 900
       END IF
     ELSE
       WRITE( 6, "( ' Should not be here !!' )" )
       GO TO 900
     END IF

!  recover x1 and x2

     IF ( status == 0 ) THEN ! not hard case

!  set y_i = - gamma_i / ( lambda + lambda_i ), i = 1, 2

       y_1 = - gamma_1 / ( lambda + lambda_1 )
       y_2 = - gamma_2 / ( lambda + lambda_2 )

     ELSE ! hard case

!  if lambda = - lambda_1, set y_2 = - gamma_2 / ( lambda + lambda_2 )
!  let y_1 be the positive root of the equation y_1^2 = radius^2 - y_2^2
!  (the negative root provides another global minimizer)

       IF ( lambda + lambda_1 == zero ) THEN
         y_2 = - gamma_2 / ( lambda + lambda_2 )
         y_1 = SQRT( radius ** 2 - y_2 ** 2 )

!  if lambda = - lambda_2, set y_1 = - gamma_1 / ( lambda + lambda_1 )
!  let y_2 be the positive root of the equation y_2^2 = radius^2 - y_1^2
!  (the negative root provides another global minimizer)

       ELSE
         y_1 = - gamma_1 / ( lambda + lambda_1 )
         y_2 = SQRT( radius ** 2 - y_1 ** 2 )
       END IF

     END IF

!  set x = Q^T y

     x_1 = c * y_1 - s * y_2 ; x_2 = s * y_1 + c * y_2

     RETURN

!  error returns

 900 CONTINUE
     x_1 = zero ; x_2 = zero ; lambda = zero
     status = - 1
     RETURN

!  end of SUBROUTINE AQT_solve_2d

     END SUBROUTINE AQT_solve_2d

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  A Q T _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE AQT_import( control, data, status, n )

!  import fixed problem data into internal storage prior to solution. 
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading 
!   comments to AQT_solve
!
!  data is a scalar variable of type AQT_full_data_type used for internal data
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
!   -3. The restriction n > 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' 'SCALED_IDENTITY', 'IDENTITY', 'ZERO', or 'NONE'
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AQT_control_type ), INTENT( INOUT ) :: control
     TYPE ( AQT_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: n
     INTEGER, INTENT( OUT ) :: status

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%aqt_inform%status
     RETURN

!  End of subroutine AQT_import

     END SUBROUTINE AQT_import

!-  G A L A H A D -  A Q T _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE AQT_information( data, inform, status )

!  return solver information during or after solution by AQT
!  See AQT_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AQT_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( AQT_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%aqt_inform
     
!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine AQT_information

     END SUBROUTINE AQT_information

!-*-*-*-*-*-  End of G A L A H A D _ A Q T  double  M O D U L E  *-*-*-*-*-*-

   END MODULE GALAHAD_AQT_double
