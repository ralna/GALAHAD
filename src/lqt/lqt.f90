! THIS VERSION: GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ L Q T  double  M O D U L E  *-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. October 8th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LQT_double

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
      PUBLIC :: LQT_initialize, LQT_read_specfile, LQT_solve,                  &
                LQT_terminate, LQT_full_initialize, LQT_full_terminate,        &
                LQT_solve_2d, LQT_import, LQT_information

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

      TYPE, PUBLIC :: LQT_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the maximum number of iterations allowed (-ve = no bound)

        INTEGER :: itmax = - 1

!   the minimum number of iterations allowed (-ve = 0)

        INTEGER :: itmin = 0

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
      END TYPE LQT_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LQT_inform_type

!  return status. See LQT_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = wp ) :: multiplier = zero

!  the M-norm of x

        REAL ( KIND = wp ) :: x_norm = zero

!  the most negative cuurvature encountered

        REAL ( KIND = wp ) :: curv = biginf

!  was negative curvature encountered ?

        LOGICAL :: negative_curvature = .FALSE.

!  is the approximate solution interior?

        LOGICAL :: interior = .FALSE.

!  did the hard case occur ?

        LOGICAL :: hard_case = .FALSE.
      END TYPE LQT_inform_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LQT_data_type
        INTEGER :: branch = 100
        INTEGER :: iter, itmax, itmin
        REAL ( KIND = wp ) :: delta, delta_old, eta, gamma, gamma_0_squared
        REAL ( KIND = wp ) :: gamma_old, gamma_older, kappa, lambda
        REAL ( KIND = wp ) :: mu, mu_old, mu_older, omega, tau, vartheta
        REAL ( KIND = wp ) :: vartheta_old, xi, x_norm, x_norm_squared
        LOGICAL :: printi, printd
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Q
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_old
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
      END TYPE LQT_data_type

      TYPE, PUBLIC :: LQT_full_data_type
        LOGICAL :: f_indexing
        TYPE ( LQT_data_type ) :: LQT_data
        TYPE ( LQT_control_type ) :: LQT_control
        TYPE ( LQT_inform_type ) :: LQT_inform
      END TYPE LQT_full_data_type

    CONTAINS

!-*-*-*-*-*-  L Q T _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LQT_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  .  Set initial values for the LQT control parameters  .
!
!  Argument:
!  =========
!
!   data     internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( LQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQT_control_type ), INTENT( OUT ) :: control
      TYPE ( LQT_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  set initial control parameter values

      control%stop_relative = SQRT( epsmch )

!  set branch for initial entry

      data%branch = 100

      RETURN

!  End of subroutine LQT_initialize

      END SUBROUTINE LQT_initialize

!- G A L A H A D -  L Q T _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE LQT_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LQT controls

!   Arguments:

!   data     internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LQT_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( LQT_control_type ), INTENT( OUT ) :: control
     TYPE ( LQT_inform_type ), INTENT( OUT ) :: inform

     CALL LQT_initialize( data%lqt_data, control, inform )

     RETURN

!  End of subroutine LQT_full_initialize

     END SUBROUTINE LQT_full_initialize

!-*-*-*-   L Q T _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LQT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LQT_initialize could (roughly)
!  have been set as:

!  BEGIN LQT SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-number-of-iterations                    -1
!   minimum-number-of-iterations                    -1
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
!  END LQT SPECIFICATIONS

!  Dummy arguments

      TYPE ( LQT_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: itmax = print_level + 1
      INTEGER, PARAMETER :: itmin = itmax + 1
      INTEGER, PARAMETER :: stop_relative = itmin + 1
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
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LQT'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( itmax )%keyword = 'maximum-number-of-iterations'
      spec( itmin )%keyword = 'minimum-number-of-iterations'

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
      CALL SPECFILE_assign_value( spec( itmin ),                               &
                                  control%itmin,                               &
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

      END SUBROUTINE LQT_read_specfile

!-*-*-*-*-*-*-*-*-*-*  L Q T _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE LQT_solve( n, radius, f, X, C, data, control, inform )

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
!   data     internal data, but see inform%status = 2 and 3
!   control  a structure containing control information. See LQT_initialize
!   inform   a structure containing information. The component
!             %status is the input/output status. This must be set to 1 on
!              initial entry or 4 on a re-entry when only radius has
!              been reduced since the last entry. Other values are
!               2 on exit => the inverse of M must be applied to
!                 data%R with the result returned in data%U and the subroutine
!                 re-entered. This will only happen if unitm is .FALSE.
!               3 on exit => the product H * data%Q must be formed, with
!                 the result returned in data%Y the subroutine re-entered
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
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, C
      TYPE ( LQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQT_control_type ), INTENT( IN ) :: control
      TYPE ( LQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: status
      REAL ( KIND = wp ) :: g_norm_squared, g_s, g_q, h_ss, h_sq, h_qq, theta
      REAL ( KIND = wp ) :: rtu
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Branch to different sections of the code depending on input status

      IF ( inform%status == 1 ) data%branch = 10

      SELECT CASE ( data%branch )
      CASE ( 10 )
        GO TO 10
      CASE ( 110 )
        GO TO 110
      CASE ( 120 )
        GO TO 120
      END SELECT

!  on initial entry, set constants

   10 CONTINUE

!  check for obvious errors

      IF ( n <= 0 ) GO TO 940
      IF ( radius <= zero ) GO TO 950

!  set termination parameters

      data%itmax = control%itmax ; IF ( data%itmax < 0 ) data%itmax = n
      data%itmin = control%itmin ; IF ( data%itmin < 0 ) data%itmin = 0

      data%printi = control%out > 0 .AND. control%print_level >= 1
      data%printd = control%out > 0 .AND. control%print_level >= 2

!  =====================
!  Array (re)allocations
!  =====================

      inform%alloc_status = 0 ; inform%bad_alloc = ''
      array_name = 'lqt: Q'
      CALL SPACE_resize_array( n, data%Q,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqt: R'
      CALL SPACE_resize_array( n, data%R,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqt: W'
      CALL SPACE_resize_array( n, data%W,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqt: W_old'
      CALL SPACE_resize_array( n, data%W_old,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqt: Y'
      CALL SPACE_resize_array( n, data%Y,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      IF ( .NOT. control%unitm ) THEN
        array_name = 'lqt: U'
        CALL SPACE_resize_array( n, data%U,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960
      END IF

!  set r_0 = c, x_0 = 0 and k = 0 (store k in inform%iter)

      data%R( : n ) = C( : n )
      X( : n ) = zero
      data%x_norm = zero
      data%lambda = zero
      f = control%f_0
      inform%iter = 0

!  main iteration loop

  100 CONTINUE

!  -----------------------------------------------
!  compute u_k = M^-1 r_k by reverse communication
!  -----------------------------------------------

!      data%U( : n ) = M^-1 * data%R( : n )
       IF ( .NOT. control%unitm ) THEN
         data%branch = 110 ; inform%status = 2 ; RETURN
       END IF
 110   CONTINUE

!  set gamma_k = sqrt( r_k' u_k )

       IF ( inform%iter > 1 ) data%gamma_older = data%gamma_old
       IF ( inform%iter > 0 ) data%gamma_old = data%gamma
       IF ( control%unitm ) THEN
         data%gamma = SQRT( DOT_PRODUCT( data%R( : n ), data%R( : n ) ) )
       ELSE
         rtu = DOT_PRODUCT( data%R( : n ), data%U( : n ) )
         IF ( rtu < zero ) GO TO 930
         data%gamma = SQRT( rtu )
       END IF

!  compute xi_k

!  set xi_k = mu_0^2 ( gamma_1^2 + delta_0^2 )

       IF ( inform%iter == 1 ) THEN
          data%xi = data%mu ** 2 * ( data%gamma ** 2 + data%delta ** 2 )

!  update xi_k = vartheta_1^2 xi_1
!     + 2 vartheta_1 mu_1 mu_0 gamma_1 ( delta_1 + delta_0 )
!     + mu_1^2 ( gamma_2^2 + delta_1^2 + gamma_1^2 )

       ELSE IF ( inform%iter == 2 ) THEN
         data%xi = data%vartheta ** 2 * data%xi                                &
          + two * data%vartheta * data%mu * data%mu_old * data%gamma_old       &
             * ( data%delta + data%delta_old ) + data%mu ** 2                  &
               * ( data%gamma ** 2 + data%delta ** 2 + data%gamma_old ** 2 )

!  update xi_k = vartheta_k-1^2 xi_k-1
!     + 2 vartheta_k-1mu_k-1 ( gamma_k^2 + delta_k-1^2 + gamma_k-1^2 )
!     + mu_k-1^2 ( vartheta_k-1 mu_k-2 gamma_k-1 ( delta_k-1 + delta_k-2 )
!                 + vartheta_k-2 vartheta_k-1 mu_k-3 gamma_k-2 gamma_k-1 )

       ELSE IF ( inform%iter > 2 ) THEN
         data%xi = data%vartheta ** 2 * data%xi                                &
           + two * data%vartheta * data%mu                                     &
              * ( data%gamma ** 2 + data%delta ** 2 + data%gamma_old ** 2 )    &
           + data%mu ** 2 * ( data%vartheta * data%mu_old * data%gamma_old     &
              * ( data%delta + data%delta_old )                                &
           + data%vartheta_old * data%vartheta * data%mu_older                 &
              * data%gamma_older * data%gamma )
       END IF

!   compute g_norm_squared = ||g_0 + lambda_0 M s_0||^2_M^-1

       IF ( inform%iter == 0 ) THEN

!   compute ||g_0 + lambda_0 M s_0||^2_M^-1 = gamma_0^2

         data%gamma_0_squared = data%gamma ** 2
         g_norm_squared = data%gamma_0_squared

!  record w_k-1

       ELSE
         data%W_old = data%W

!  compute ||c + H x_k + lambda_k M x_k||^2_M^-1 = gamma_0^2 + 2 eta_k + xi_k
!            + 2 lambda_k ( kappa_k + tau_k ) + lambda_k^2 ||x_k||^2_M

         g_norm_squared = data%gamma_0_squared + two * data%eta + data%xi      &
          + two * data%lambda * ( data%kappa + data%tau )                      &
          + data%lambda ** 2 * data%x_norm_squared
       END IF

!  print details of the latest iteration

       IF ( data%printi ) THEN
         IF ( MOD( inform%iter, 25 ) == 0 .OR. data%printd )                     &
             WRITE( control%out, "( /, A, '   Iter        f        ',          &
            &       ' pgnorm    step    norm p   norm x     curv ' )" ) prefix
         WRITE( control%out, "( A, I7, ES16.8, 2ES9.2 )" )                     &
                prefix, inform%iter, f, SQRT( g_norm_squared ), data%x_norm
       END IF

!  stop if the gradient is small so long as there have been sufficient
!  iterations

       IF ( g_norm_squared <= control%stop_relative ** 2 .AND.                 &
            inform%iter >= data%itmin ) GO TO 900

!  set w_k = r_k / gamma_k and q_k = u_k / gamma_k

       data%W( : n ) = data%R( : n ) / data%gamma
       IF ( control%unitm ) THEN
         data%Q( : n ) = data%W( : n )
       ELSE
         data%Q( : n ) = data%U( : n ) / data%gamma
       END IF

!  --------------------------------------------
!  compute y_k = H q_k by reverse communication
!  --------------------------------------------

!      data%Y = H * data%Q
       data%branch = 120 ; inform%status = 3 ; RETURN
 120   CONTINUE

!  set omega_k =  x_k' y_k and delta_k =  q_k' y_k

       IF ( inform%iter > 0 ) THEN
         data%delta_old = data%delta
         data%omega = DOT_PRODUCT( X( : n ), data%Y( : n ) )
       END IF
       data%delta = DOT_PRODUCT( data%Q( : n ), data%Y( : n ) )

!  find mu_0 = argminin_{mu : mu^2 <= radius^2} half delta_0 mu^2 + gamma_0 mu
!  with associated multiplier lambda_1

       IF ( inform%iter == 0 ) THEN
         h_qq = data%delta
         g_q = data%gamma
         CALL LQT_solve_1d( h_qq, g_q, radius, data%mu, data%lambda, status )

!  set vartheta_0 = theta_0 = 0 and x_1 = mu_0 q_0

         data%vartheta = zero
         X( : n ) = data%mu * data%Q( : n )
       ELSE
         data%vartheta_old = data%vartheta
         IF ( inform%iter > 1 ) data%mu_older = data%mu_old
         data%mu_old = data%mu

!  find (theta_,kmu_k) = argminin_{(theta,mu) : theta^2 + mu^2 <= Delta^2}
!     half (tau_k/||x_k||_M^2) theta^2 + omega_k/||x_k||_M  theta mu
!      + half delta_k mu^2 + kappa_k / ||x_k||_M theta
!  with associated multiplier lambda_k+1

         h_ss = data%tau / data%x_norm_squared
         h_sq = data%omega / data%x_norm
         h_qq = data%delta
         g_s = data%kappa / data%x_norm
         g_q = zero
         CALL LQT_solve_2d( h_ss, h_sq, h_qq, g_s, g_q, radius,                &
                            theta, data%mu, data%lambda, status )

!  set vartheta_k = theta_k / ||x_k||_M and x_k+1  = vartheta_k x_k + mu_k q_k

         data%vartheta = theta / data%x_norm
         X( : n ) = data%vartheta * X( : n ) + data%mu * data%Q( : n )
       END IF

!  compute ||x_k+1||_M^2, kappa_k+1, tau_k+1, eta_k+1 and r_k+1

       IF ( inform%iter == 0 ) THEN

!  set ||x_1||_M = mu_0, kappa_1 = mu_0 gamma_0, and tau_1 = mu_0^2 delta_0,

         data%x_norm = data%mu
         data%x_norm_squared = data%x_norm ** 2
         data%kappa = data%mu * data%gamma
         data%tau = data%mu ** 2 * data%delta
       ELSE

!  set ||x_k+1||_M^2 = theta_k^2 + mu_k^2, update kappa_k+1 =vartheta_k kappa_k,
!  and tau_k+1 = vartheta_k^2 tau_k + 2 vartheta_k mu_k omega_k + mu_k^2 delta_k

         data%x_norm_squared = theta ** 2 + data%mu ** 2
         data%x_norm = SQRT( data%x_norm_squared )
         data%kappa = data%vartheta * data%kappa
         data%tau = data%vartheta ** 2 * data%tau                              &
                 + two * data%vartheta * data%mu * data%omega                  &
                 + data%mu ** 2 * data%delta
       END IF

       f = control%f_0 + data%kappa + half * data%tau
       IF ( inform%iter == data%itmax ) GO TO 910

!  compute eta_k+1 and r_k+1

       IF ( inform%iter == 0 ) THEN

!  set eta_1 = mu_0 gamma_0 delta_0 and r_1 = y_0 - delta_0 w_0

         data%eta = data%mu * data%gamma * data%delta
         data%R( : n ) = data%Y( : n ) - data%delta * data%W( : n )
       ELSE

!  set r_k+1 = y_k - delta_k w_k - gamma_k w_k-1

         data%R( : n ) = data%Y( : n ) - data%delta * data%W( : n )            &
                           - data%gamma * data%W_old( : n )

!  update eta_2 = vartheta_1 eta_1 + mu_1 gamma_0 gamma_1

         IF ( inform%iter == 1 ) THEN
           data%eta = data%vartheta * data%eta                                 &
                        + data%mu * data%gamma * data%gamma_old

!   update eta_k+1 = vartheta_k eta_k

         ELSE
           data%eta = data%vartheta * data%eta
         END IF
       END IF

!  k  = k + 1

       inform%iter = inform%iter + 1

!  end of main iteration loop

    GO TO 100

!  record output statistics

    inform%x_norm = data%x_norm ; inform%multiplier = data%lambda
    inform%interior = ABS( inform%x_norm - radius ) <= ten ** ( -15 )

!  ===============
!  Exit conditions
!  ===============

!  successful returns

 900 CONTINUE
     inform%status = GALAHAD_ok
     RETURN

!  too many iterations

 910 CONTINUE
     IF ( data%printi )                                                        &
       WRITE( control%out, "( /, A, ' Iteration limit exceeded ' ) " ) prefix
     inform%status = GALAHAD_error_max_iterations ; inform%iter = data%iter
     RETURN

!  boundary encountered in Steihaug-Toint method

  920 CONTINUE

!  find the gradient at the appropriate point on the boundary

!     R = R + data%alpha * VECTOR

!  record terminal information

!     IF ( data%printi ) WRITE( control%out, "( A, I7, ES16.8, 4X, '-', 4X,    &
!    &     3ES9.2, ES10.2, //, A,                                              &
!    &     ' Now leaving trust region (Steihaug-Toint)' )" )                   &
!         prefix, inform%iter, f, data%alpha, data%normp, SQRT( data%xmx ),    &
!         inform%rayleigh, prefix
!     inform%status = GALAHAD_warning_on_boundary
!     RETURN

!  unsuccessful returns

  930 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The matrix M appears to be indefinite. Inner product = ',     &
     &      ES12.4  )" ) prefix, rtu
      inform%status = GALAHAD_error_preconditioner
      RETURN

  940 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' n = ', I6, ' is not positive ' )" ) prefix, n
      inform%status = GALAHAD_error_restrictions
      RETURN

  950 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, radius
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Allocation or deallocation error

  960 CONTINUE
      RETURN

!  objective-function value is too small

  970 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The objective function value is smaller than', ES12.4 )" )    &
        prefix, control%f_min
      inform%status = GALAHAD_error_f_min ; inform%iter = data%iter
      RETURN

!  End of subroutine LQT_solve

      END SUBROUTINE LQT_solve

!-*-*-*-*-*-  L Q T _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LQT_terminate( data, control, inform )

!  ..............................................
!  .                                            .
!  .  Deallocate arrays at end of LQT_solve    .
!  .                                            .
!  ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine LQT_initialize
!   inform  see Subroutine LQT_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LQT_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQT_control_type ), INTENT( IN ) :: control
      TYPE ( LQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'lqt: Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqt: R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqt: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqt: W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqt: W_old'
      CALL SPACE_dealloc_array( data%W_old,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqt: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine LQT_terminate

      END SUBROUTINE LQT_terminate

! -  G A L A H A D -  L Q T _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

      SUBROUTINE LQT_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LQT_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQT_control_type ), INTENT( IN ) :: control
      TYPE ( LQT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      CALL LQT_terminate( data%lqt_data, control, inform )

      RETURN

!  End of subroutine LQT_full_terminate

      END SUBROUTINE LQT_full_terminate

!-*-*-*- G A L A H A D -  L Q T _ S O L V E _ 2 D   S U B R O U T I N E -*-*-*-

     SUBROUTINE LQT_solve_2d( h_11, h_12, h_22, g_1, g_2, radius,              &
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

!  end of SUBROUTINE LQT_solve_2d

     END SUBROUTINE LQT_solve_2d

!-*-*-*- G A L A H A D -  L Q T _ S O L V E _ 1 D   S U B R O U T I N E -*-*-*-

     SUBROUTINE LQT_solve_1d( h_11, g_1, radius, x_1, lambda, status )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the minimizer x = (x1,x2) of the quadratic function
!
!    q(x) = 1/2 x^T H x + g^t x
!         = 1/2 h_11 x_1^2 + g_1 x_1
!
!  within the trust region ||x||^2 = x_1^2 <= radius^2
!
!  The required solution satisfies the equations ( H + lambda I ) x = - g, i.e.,
!
!    ( h_11 + lambda ) x_1 = - g_1
!
!  input arguments:
!
!   h_11, g_1, radius - real, as above
!
!  output arguments:
!
!  x_1, lambda - real, as above
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
     REAL ( KIND = wp ), INTENT( IN ) :: h_11, g_1, radius
     REAL ( KIND = wp ), INTENT( OUT ) :: x_1, lambda

!  if the curvature is positive, check for an interior solution. Otherwise
!  step to the trust-region boundary

     IF ( h_11 > zero ) THEN
       x_1 = - g_1 / h_11
       IF ( g_1 > zero ) THEN
         IF ( x_1 >= - radius ) THEN
           lambda = zero
           status = 1
         ELSE
           x_1 = - radius
           lambda = - h_11 + g_1 / radius
           status = 0
         END IF
       ELSE
         IF ( x_1 <= radius ) THEN
           lambda = zero
           status = 1
         ELSE
           x_1 = radius
           lambda = - h_11 - g_1 / radius
           status = 0
         END IF
       END IF

!  if the curvature is not positive, the solution lies on the boundary

     ELSE IF ( g_1 == zero ) THEN
       x_1 = radius
       lambda = - h_11
       status = 3
     ELSE IF ( g_1 > zero ) THEN
       x_1 = - radius
       lambda = - h_11 + g_1 / radius
       status = 0
     ELSE
       x_1 = radius
       lambda = - h_11 - g_1 / radius
       status = 0
     END IF

     RETURN

!  end of SUBROUTINE LQT_solve_1d

     END SUBROUTINE LQT_solve_1d

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  L Q T _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE LQT_import( control, data, status, n )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to LQT_solve
!
!  data is a scalar variable of type LQT_full_data_type used for internal data
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

     TYPE ( LQT_control_type ), INTENT( INOUT ) :: control
     TYPE ( LQT_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: n
     INTEGER, INTENT( OUT ) :: status

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%lqt_inform%status
     RETURN

!  End of subroutine LQT_import

     END SUBROUTINE LQT_import

!-  G A L A H A D -  L Q T _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE LQT_information( data, inform, status )

!  return solver information during or after solution by LQT
!  See LQT_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LQT_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( LQT_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%lqt_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine LQT_information

     END SUBROUTINE LQT_information

!-*-*-*-*-*-  End of G A L A H A D _ L Q T  double  M O D U L E  *-*-*-*-*-*-

   END MODULE GALAHAD_LQT_double
