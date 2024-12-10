! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ L Q R  double  M O D U L E  *-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. October 8th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LQR_precision

!      ----------------------------------------------------------------------
!      |                                                                    |
!      | Approximately solve the regularised quadratic minimization problem |
!      |                                                                    |
!      |    minimize     1/p weight ||x||_M^p + 1/2 <x,Hx> + <c,x> + f0     |
!      |                                                                    |
!      ! where M is symmetric, positive definite, ||x||_M^2 = <x,Mx>        |
!      | and p (>=2) and weight (>0) are constants using a Lanczos method   |
!      |                                                                    |
!      ----------------------------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_precision
!     USE GALAHAD_ROOTS_precision, ONLY: ROOTS_quadratic, ROOTS_quartic
      USE GALAHAD_ROOTS_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_LAPACK_inter_precision, ONLY: LAEV2

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LQR_initialize, LQR_read_specfile, LQR_solve,                  &
                LQR_terminate, LQR_full_initialize, LQR_full_terminate,        &
                LQR_solve_2d, LQR_import, LQR_information

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: four = 4.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = rp_ ), PARAMETER :: biginf = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: boundary_tol = epsmch ** 0.75
      LOGICAL :: roots_debug = .FALSE.
      LOGICAL :: find_roots = .FALSE.
      LOGICAL :: shift_roots = .TRUE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LQR_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   the maximum number of iterations allowed (-ve = no bound)

        INTEGER ( KIND = ip_ ) :: itmax = - 1

!   the minimum number of iterations allowed (-ve = 0)

        INTEGER ( KIND = ip_ ) :: itmin = 0

!   the maximum number of iterations allowed once the trust-region boundary
!   has been achieved (-ve = no bound)

        INTEGER ( KIND = ip_ ) :: itmax_beyond_boundary = - 1

!   the iteration stops successfully when the gradient in the M(inverse) norm
!    is smaller than max( stop_relative * initial M(inverse)
!                         gradient norm, stop_absolute )

        REAL ( KIND = rp_ ) :: stop_relative = epsmch
        REAL ( KIND = rp_ ) :: stop_absolute = zero

!  the iteration stops successfully when the current decrease in the
!   objective function is smaller than stop_f_relative * the overall decrease

        REAL ( KIND = rp_ ) :: stop_f_relative = epsmch

!   the constant term, f0, in the objective function

        REAL ( KIND = rp_ ) :: f_0 = zero

!   is M the identity matrix ?

        LOGICAL :: unitm = .TRUE.

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
      END TYPE LQR_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LQR_inform_type

!  return status. See LQR_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the iteration that the boundary is achieved (-1=never)

        INTEGER ( KIND = ip_ ) :: iter_boundary = - 1

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = rp_ ) :: multiplier = zero

!  the M-norm of x

        REAL ( KIND = rp_ ) :: x_norm = zero

!  the most negative cuurvature encountered

        REAL ( KIND = rp_ ) :: curv = biginf

!  was negative curvature encountered ?

        LOGICAL :: negative_curvature = .FALSE.

!  is the approximate solution interior?

        LOGICAL :: interior = .TRUE.

!  did the hard case occur ?

        LOGICAL :: hard_case = .FALSE.
      END TYPE LQR_inform_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LQR_data_type
        INTEGER ( KIND = ip_ ) :: branch = 100
        INTEGER ( KIND = ip_ ) :: iter, itmax, itmin
        REAL ( KIND = rp_ ) :: delta, delta_old, eta, gamma, gamma_0_squared
        REAL ( KIND = rp_ ) :: gamma_old, gamma_older, kappa, lambda
        REAL ( KIND = rp_ ) :: mu, mu_old, mu_older, omega, tau, vartheta
        REAL ( KIND = rp_ ) :: vartheta_old, xi, x_norm, x_norm_squared
        REAL ( KIND = rp_ ) :: stop_g_squared, f_current, f_last
        LOGICAL :: printi, printd
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Q
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: U
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_old
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y
      END TYPE LQR_data_type

      TYPE, PUBLIC :: LQR_full_data_type
        LOGICAL :: f_indexing
        TYPE ( LQR_data_type ) :: LQR_data
        TYPE ( LQR_control_type ) :: LQR_control
        TYPE ( LQR_inform_type ) :: LQR_inform
      END TYPE LQR_full_data_type

    CONTAINS

!-*-*-*-*-*-  L Q R _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LQR_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  .  Set initial values for the LQR control parameters  .
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

      TYPE ( LQR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQR_control_type ), INTENT( OUT ) :: control
      TYPE ( LQR_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  set initial control parameter values

      control%stop_relative = SQRT( epsmch )
      control%stop_f_relative = SQRT( epsmch )

!  set branch for initial entry

      data%branch = 100

      RETURN

!  End of subroutine LQR_initialize

      END SUBROUTINE LQR_initialize

!- G A L A H A D -  L Q R _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE LQR_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LQR controls

!   Arguments:

!   data     internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LQR_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( LQR_control_type ), INTENT( OUT ) :: control
     TYPE ( LQR_inform_type ), INTENT( OUT ) :: inform

     CALL LQR_initialize( data%lqr_data, control, inform )

     RETURN

!  End of subroutine LQR_full_initialize

     END SUBROUTINE LQR_full_initialize

!-*-*-*-   L Q R _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LQR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LQR_initialize could (roughly)
!  have been set as:

!  BEGIN LQR SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-number-of-iterations                    -1
!   minimum-number-of-iterations                    -1
!   maximum-number-of-iterations-beyond-TR          -1
!   relative-accuracy-required                      1.0E-8
!   absolute-accuracy-required                      0.0
!   relative-objective-decrease-required            1.0E-8
!   small-f-stop                                    -1.0D+100
!   constant-term-in-objective                      0.0
!   two-norm-trust-region                           T
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END LQR SPECIFICATIONS

!  Dummy arguments

      TYPE ( LQR_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: itmax = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: itmin = itmax + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: itmax_beyond_boundary  = itmin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_relative                       &
                                             = itmax_beyond_boundary + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_absolute = stop_relative + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_f_relative = stop_absolute + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: f_0 = stop_f_relative
      INTEGER ( KIND = ip_ ), PARAMETER :: unitm = f_0 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = unitm + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LQR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( itmax )%keyword = 'maximum-number-of-iterations'
      spec( itmin )%keyword = 'minimum-number-of-iterations'
      spec( itmax_beyond_boundary )%keyword                                    &
         = 'maximum-number-of-iterations-beyond-TR'

!  real key-words

      spec( stop_relative )%keyword = 'relative-accuracy-required'
      spec( stop_absolute )%keyword = 'absolute-accuracy-required'
      spec( stop_f_relative )%keyword = 'relative-objective-decrease-required'
      spec( f_0 )%keyword = 'constant-term-in-objective'

!  logical key-words

      spec( unitm )%keyword = 'two-norm-trust-region'
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
      CALL SPECFILE_assign_value( spec( itmax_beyond_boundary ),               &
                                  control%itmax_beyond_boundary,               &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( stop_relative ),                       &
                                  control%stop_relative,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_absolute ),                       &
                                  control%stop_absolute,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_f_relative ),                     &
                                  control%stop_f_relative,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( f_0 ),                                 &
                                  control%f_0,                                 &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( unitm ),                               &
                                  control%unitm,                               &
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

      END SUBROUTINE LQR_read_specfile

!-*-*-*-*-*-*-*-*-*-*  L Q R _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE LQR_solve( n, radius, f, X, C, data, control, inform )

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
!   control  a structure containing control information. See LQR_initialize
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
!              -17 insufficient objective improvement occurs
!              -18 the iteration limit has been exceeded
!             the remaining components are described in the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      REAL ( KIND = rp_ ), INTENT( IN ) :: radius
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: f
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, C
      TYPE ( LQR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQR_control_type ), INTENT( IN ) :: control
      TYPE ( LQR_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: status
      REAL ( KIND = rp_ ) :: g_norm_squared, g_s, g_q, h_ss, h_sq, h_qq, theta
      REAL ( KIND = rp_ ) :: rtu
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
      array_name = 'lqr: Q'
      CALL SPACE_resize_array( n, data%Q,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqr: R'
      CALL SPACE_resize_array( n, data%R,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqr: W'
      CALL SPACE_resize_array( n, data%W,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqr: W_old'
      CALL SPACE_resize_array( n, data%W_old,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'lqr: Y'
      CALL SPACE_resize_array( n, data%Y,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      IF ( .NOT. control%unitm ) THEN
        array_name = 'lqr: U'
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
      data%f_current = zero
      f = control%f_0

!  initiliaze output information

      inform%iter = 0
      inform%iter_boundary = - 1
      inform%interior = .TRUE.

!  =============================
!  main iteration loop - here:
!   tau_k   = x_k' H x_k
!   omega_k = q_k' H x_k
!   delta_k = q_k' H q_k
!   kappa_k = c' x_k
!   eta_k   = c' M^-1 H x_k
!   xi_k    = x_k' H M^-1 H x_k
!  =============================

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
         data%xi = data%xi * data%vartheta ** 2                                &
          + two * data%vartheta * data%mu * data%mu_old * data%gamma_old       &
             * ( data%delta + data%delta_old )                                 &
          + ( data%gamma ** 2 + data%delta ** 2 + data%gamma_old ** 2 )        &
               * data%mu ** 2
!  update xi_k = vartheta_k-1^2 xi_k-1
!     + 2 vartheta_k-1 mu_k-1 ( mu_k-2 gamma_k-1 ( delta_k-1 + delta_k-2 )
!                               + vartheta_k-2 mu_k-3 gamma_k-2 gamma_k-1 )
!     + mu_k-1^2 ( gamma_k^2 + delta_k-1^2 + gamma_k-1^2 )

       ELSE IF ( inform%iter > 2 ) THEN
         data%xi = data%xi * data%vartheta ** 2                                &
           + two * data%vartheta * data%mu                                     &
           * ( data%mu_old * data%gamma_old * ( data%delta + data%delta_old )  &
              + data%vartheta_old * data%mu_older                              &
                 * data%gamma_older * data%gamma_old ) &
           + ( data%gamma ** 2 + data%delta ** 2 + data%gamma_old ** 2 )    &
               * data%mu ** 2
       END IF

!   compute g_norm_squared = ||g_0 + lambda_0 M s_0||^2_M^-1

       IF ( inform%iter == 0 ) THEN

!   compute ||g_0 + lambda_0 M s_0||^2_M^-1 = gamma_0^2

         data%gamma_0_squared = data%gamma ** 2
         g_norm_squared = data%gamma_0_squared
         data%stop_g_squared = MAX( control%stop_relative * data%gamma,        &
                                    control%stop_absolute ) ** 2

!  record w_k-1

       ELSE
         data%W_old = data%W

!  compute ||c + H x_k + lambda_k M x_k||^2_M^-1 = gamma_0^2 + 2 eta_k + xi_k
!            + 2 lambda_k ( kappa_k + tau_k ) + lambda_k^2 ||x_k||^2_M

         g_norm_squared = data%gamma_0_squared + two * data%eta + data%xi      &
          + two * data%lambda * ( data%kappa + data%tau )                      &
          + data%x_norm_squared * data%lambda ** 2
       END IF

!  print details of the latest iteration

       IF ( data%printi ) THEN
         IF ( MOD( inform%iter, 25 ) == 0 .OR. data%printd )                   &
             WRITE( control%out, "( /, A, '   Iter        f        ',          &
            &       ' pgnorm   norm x   step x   step_q   lambda' )" ) prefix
         IF ( inform%iter > 0 ) THEN
           WRITE( control%out, "( A, I7, ES16.8, 5ES9.2 )" )                   &
!                 prefix, inform%iter, f, 0.0_rp_, data%x_norm, &
                  prefix, inform%iter, f, SQRT( g_norm_squared ), data%x_norm, &
                  data%vartheta, data%mu, data%lambda
         ELSE
           WRITE( control%out, "( A, I7, ES16.8, 2ES9.2, '    -        -    ', &
          &                    ES9.2 )" ) prefix, inform%iter, f,              &
              SQRT( g_norm_squared ), data%x_norm, data%lambda
         END IF
       END IF

!  stop if the gradient is small so long_ as there have been sufficient
!  iterations

       IF ( g_norm_squared <= data%stop_g_squared .AND.                        &
            inform%iter >= data%itmin ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

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

!      IF ( .TRUE. ) &
       IF ( .FALSE. ) &
 write(6,"( ' tau   = ', ES22.14, /, ' omega = ', ES22.14, /, &
 &          ' delta = ', ES22.14, /, ' kappa = ', ES22.14, /, &
 &          ' eta   = ', ES22.14, /, ' xi    = ', ES22.14, /, &
 &          ' m     = ', ES22.14, / )" ) &
           data%tau, data%omega, data%delta, data%kappa, data%eta, data%xi, &
           data%f_current

!  find mu_0 = argminin_{mu : mu^2 <= radius^2} half delta_0 mu^2 + gamma_0 mu
!  with associated multiplier lambda_1

       IF ( inform%iter == 0 ) THEN
         h_qq = data%delta
         g_q = data%gamma
         CALL LQR_solve_1d( h_qq, g_q, radius, data%mu, data%lambda, status )

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
         CALL LQR_solve_2d( h_ss, h_sq, h_qq, g_s, g_q, radius,                &
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
         data%tau = data%tau * data%vartheta ** 2                              &
                      + two * data%vartheta * data%mu * data%omega             &
                      + data%delta * data%mu ** 2
       END IF

!  record when the boundary is reached, and adjust the iteration limit
!  accordingly

       IF ( inform%interior ) THEN
         IF ( ABS( data%x_norm - radius ) <= boundary_tol ) THEN
           inform%iter_boundary = inform%iter
           inform%interior = .FALSE.
           IF ( control%itmax_beyond_boundary > 0 )                            &
             data%itmax = MIN( data%itmax,                                     &
                               inform%iter + control%itmax_beyond_boundary )
         END IF
       END IF

!  record the new function value

       data%f_last = data%f_current
       data%f_current = data%kappa + half * data%tau
       f = control%f_0 + data%f_current

!  check to see if the iteration limit has been achieved

       IF ( inform%iter == data%itmax ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  check to see if the objective improvement is insufficient

!      WRITE( 6, "( ' Df, total f = ', 2ES22.14 )" )                           &
!        data%f_last - data%f_current, - data%f_current
       IF ( data%f_last - data%f_current <=                                    &
            - control%stop_f_relative * data%f_current .AND.                   &
            inform%iter >= data%itmin ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 900
       END IF

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

!  ==========================
!  end of main iteration loop
!  ==========================

    GO TO 100

!  record output statistics

 900 CONTINUE
    inform%x_norm = data%x_norm ; inform%multiplier = data%lambda
    inform%interior = ABS( inform%x_norm - radius ) <= boundary_tol

!  ===============
!  Exit conditions
!  ===============

    SELECT CASE ( inform%status )

!  successful returns

    CASE ( GALAHAD_ok )

!  too many iterations

    CASE ( GALAHAD_error_max_iterations )
      IF ( data%printi ) WRITE( control%out,                                   &
         "( /, A, ' Iteration limit exceeded' )" ) prefix

!  f improvement too small

    CASE ( GALAHAD_error_tiny_step )
      IF ( data%printi ) WRITE( control%out,                                   &
        "( /, A, ' Insufficient objective improvement' )" ) prefix
   END SELECT
   RETURN

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
         "( A, ' n = ', I6, ' is not positive' )" ) prefix, n
      inform%status = GALAHAD_error_restrictions
      RETURN

  950 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The radius ', ES12.4 , ' is not positive' )" ) prefix, radius
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Allocation or deallocation error

  960 CONTINUE
      RETURN

!  End of subroutine LQR_solve

      END SUBROUTINE LQR_solve

!-*-*-*-*-*-  L Q R _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LQR_terminate( data, control, inform )

!  ..............................................
!  .                                            .
!  .  Deallocate arrays at end of LQR_solve    .
!  .                                            .
!  ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine LQR_initialize
!   inform  see Subroutine LQR_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LQR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQR_control_type ), INTENT( IN ) :: control
      TYPE ( LQR_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'lqr: Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqr: R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqr: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqr: W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqr: W_old'
      CALL SPACE_dealloc_array( data%W_old,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lqr: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine LQR_terminate

      END SUBROUTINE LQR_terminate

! -  G A L A H A D -  L Q R _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

      SUBROUTINE LQR_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LQR_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( LQR_control_type ), INTENT( IN ) :: control
      TYPE ( LQR_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

      CALL LQR_terminate( data%lqr_data, control, inform )

      RETURN

!  End of subroutine LQR_full_terminate

      END SUBROUTINE LQR_full_terminate

!-*-*-*- G A L A H A D -  L Q R _ S O L V E _ 2 D   S U B R O U T I N E -*-*-*-

     SUBROUTINE LQR_solve_2d( h_11, h_12, h_22, g_1, g_2, radius,              &
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

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( IN ) :: h_11, h_12, h_22, g_1, g_2, radius
     REAL ( KIND = rp_ ), INTENT( OUT ) :: x_1, x_2, lambda

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, leftmost, nroots
     REAL ( KIND = rp_ ) :: gamma_1, gamma_2, lambda_1, lambda_2, c, s
     REAL ( KIND = rp_ ) :: y_1, y_2, c1, c2, lambda_min
     REAL ( KIND = rp_ ) :: dlambda, phi, phi_prime, lambda_n, cn, mu
     LOGICAL :: interior
!    LOGICAL :: debug

     REAL ( KIND = rp_ ), DIMENSION( 0 : 4 ) :: A
     REAL ( KIND = rp_ ), DIMENSION( 4 ) :: ROOTS
!    TYPE ( ROOTS_data_type ) :: data
!    TYPE ( ROOTS_control_type ) :: control
!    TYPE ( ROOTS_inform_type ) :: inform

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

!                         gamma_1^2                 gamma_2^2
!   phi(lambda) =  ----------------------- + ----------------------- = radius^2
!                  ( lambda + lambda_1 )^2   ( lambda + lambda_2 )^2

!  or equivalently the quartic equation

!     ( lambda + lambda_1 )^2 ( lambda + lambda_2 )^2
!       - ( gamma_1 / radius )^2 ( lambda + lambda_2 )^2
!       - ( gamma_2 / radius )^2 ( lambda + lambda_2 )^2 = 0

     IF ( gamma_1 /= zero .AND. gamma_2 /= zero ) THEN
       c1 = ( gamma_1 / radius ) ** 2
       c2 = ( gamma_2 / radius ) ** 2

!  either estimate the larger root by replacing phi by the underestimator
!  gamma_min^2 / ( lambda + lambda_min )^2

       IF ( .NOT. find_roots ) THEN
         IF ( lambda_1 < lambda_2 ) THEN
           lambda = - lambda_1 + ABS( gamma_1 ) / radius
           dlambda =  lambda + lambda_1
         ELSE
           lambda = - lambda_2 + ABS( gamma_2 ) / radius
           dlambda =  lambda + lambda_2
         END IF

!  or find all the roots via an appropriate rootfinder. In practice, shift
!  the root-finding problem by lambda_min to avoid some cancellation when
!  forming coefficients

       ELSE
         IF ( shift_roots ) THEN
           IF ( lambda_1 < lambda_2 ) THEN
             mu = lambda_2 - lambda_1
             A( 4 ) = one
             A( 3 ) = two * mu
             A( 2 ) = mu ** 2 - c1 - c2
             A( 1 ) = - two * c1 * mu
             A( 0 ) = - c1 * mu ** 2
           ELSE
             mu = lambda_1 - lambda_2
             A( 4 ) = one
             A( 3 ) = two * mu
             A( 2 ) = mu ** 2 - c1 - c2
             A( 1 ) = - two * c2 * mu
             A( 0 ) = - c2 * mu ** 2
           END IF

!  un-shifted case

         ELSE
           A( 4 ) = one
           A( 3 ) = two * ( lambda_1 + lambda_2 )
           A( 2 ) = lambda_1 ** 2 + lambda_2 ** 2 + four * lambda_1 * lambda_2 &
                    - c1 - c2
           A( 1 ) = two * lambda_1 * lambda_2 * ( lambda_1 + lambda_2 )        &
                    - two * ( c1 * lambda_2 + c2 * lambda_1 )
           A( 0 ) = ( lambda_1 * lambda_2 ) ** 2                               &
                    - c1 * lambda_2 ** 2 - c2 * lambda_1 ** 2
         END IF

!        CALL ROOTS_solve( A, nroots, ROOTS, control, inform, data )
         CALL ROOTS_quartic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), A( 4 ), epsmch,   &
                            nroots, ROOTS( 1 ), ROOTS( 2 ), ROOTS( 3 ),        &
                            ROOTS( 4 ), .TRUE. )

!  record the required root

         IF ( nroots == 4 ) THEN
           IF ( lambda_1 < lambda_2 ) THEN
             lambda = - lambda_1 + ROOTS( 4 )
           ELSE
             lambda = - lambda_2 + ROOTS( 4 )
           END IF
         ELSE IF ( nroots == 2 ) THEN
           IF ( lambda_1 < lambda_2 ) THEN
             lambda = - lambda_1 + ROOTS( 2 )
           ELSE
             lambda = - lambda_2 + ROOTS( 2 )
           END IF
         ELSE
           WRITE( 6, "( ' Should not be here !! - quartic case' )" )
           WRITE( 6, "( I0, ' roots ' )" ) nroots
           GO TO 900
         END IF
       END IF

!  perform a few iterations of Newton's method, applied to
!  ( phi(lambda) )^-1/2 = radiius^-1/2, to improve the root

       DO i = 1, 10
         phi = c1 / ( lambda + lambda_1 ) ** 2 + c2 / ( lambda + lambda_2 ) ** 2
         phi_prime = - two * c1 / ( lambda + lambda_1 ) ** 3                   &
                     - two * c2 / ( lambda + lambda_2 ) ** 3
!        dlambda = - ( phi - one ) / phi_prime
         dlambda = ( phi ** ( - 0.5_rp_ ) - one ) /                            &
                   ( half * phi_prime * phi ** ( - 1.5_rp_ ) )
!        IF ( ABS( dlambda ) <= ten * epsmch * MAX( one, lambda ) ) THEN
         IF ( ABS( dlambda ) <= epsmch * MAX( one, lambda ) ) THEN
           EXIT
         END IF
         lambda = lambda + dlambda
       END DO
       status = 0

!  if gamma_i is zero but gamma_j isn't, solve the secular equation

!                         gamma_j^2
!   phi(lambda) =  ----------------------- = radius^2
!                  ( lambda + lambda_j )^2

!  or equivalently the quartic equation

!     ( lambda + lambda_j )^2 - ( gamma_j radius )^2 = 0

     ELSE IF ( gamma_1 /= zero .OR. gamma_2 /= zero ) THEN
       IF ( gamma_1 /= zero ) THEN
         cn = ( gamma_1 / radius ) ** 2 ; lambda_n = lambda_1
       ELSE
         cn = ( gamma_2 / radius ) ** 2 ; lambda_n = lambda_2
       END IF
       A( 2 ) = one ; A( 1 ) = two * lambda_n
       A( 0 ) = lambda_n ** 2 - cn
       CALL ROOTS_quadratic( A( 0 ), A( 1 ), A( 2 ), epsmch,                   &
                             nroots, ROOTS( 1 ), ROOTS( 2 ), roots_debug )

!  record the required root, and check for the hard case

       IF ( nroots == 2 ) THEN
         lambda = ROOTS( 2 )
         IF ( ROOTS( 2 ) < lambda_min ) THEN
           lambda = lambda_min
           status = 3
         ELSE
           lambda = ROOTS( 2 )

!  perform a few iterations of Newton's method, applied to
!  ( phi(lambda) )^-1/2 = radiius^-1/2, to improve the root

!          dlambda = zero
           DO i = 1, 5
             phi = cn / ( lambda + lambda_n ) ** 2
             phi_prime = - two * cn / ( lambda + lambda_n ) ** 3
!            dlambda = - ( phi - one ) / phi_prime
             dlambda = ( phi ** ( - 0.5_rp_ ) - one ) /                        &
                       ( half * phi_prime * phi ** ( - 1.5_rp_ ) )
!            IF ( ABS( dlambda ) <= ten * epsmch * MAX( one, lambda ) ) EXIT
             IF ( ABS( dlambda ) <= epsmch * MAX( one, lambda ) ) EXIT
             lambda = lambda + dlambda
           END DO
           status = 0
         END IF
       ELSE
         WRITE( 6, "( ' Should not be here !! - quadratic case' )" )
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

     y_1 = g_1 + ( h_11 + lambda ) * x_1 + h_12 * x_2
     y_2 = g_2 + h_12 * x_1 + ( h_22 + lambda ) * x_2

!write(6,*) ' m ', half * h_11 * x_1 ** 2 + half * h_22 * x_2 ** 2 &
!           + h_12 * x_1 * x_2 + g_1 * x_1 + g_2 * x_2
!write(6,*) ' res ', y_1, y_2
!write(6,*) ' lambda, c ', lambda, x_1 ** 2 + x_2 ** 2 - radius ** 2
!write(6,"( ' data ', 5ES12.4)") h_11, h_12, h_22, g_1, g_2

     RETURN

!  error returns

 900 CONTINUE
     WRITE( 6, "( ' h_11, h_12, h_22 = ', 3ES12.4 )" ) h_11, h_12, h_22
     WRITE( 6, "( ' g_1, g_2 = ', 2ES12.4 )" ) g_1, g_2
     x_1 = zero ; x_2 = zero ; lambda = zero
     status = - 1
     RETURN

!  end of SUBROUTINE LQR_solve_2d

     END SUBROUTINE LQR_solve_2d

!-*-*-*- G A L A H A D -  L Q R _ S O L V E _ 1 D   S U B R O U T I N E -*-*-*-

     SUBROUTINE LQR_solve_1d( h_11, g_1, radius, x_1, lambda, status )

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

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( IN ) :: h_11, g_1, radius
     REAL ( KIND = rp_ ), INTENT( OUT ) :: x_1, lambda

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

!  end of SUBROUTINE LQR_solve_1d

     END SUBROUTINE LQR_solve_1d

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  L Q R _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE LQR_import( control, data, status, n )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to LQR_solve
!
!  data is a scalar variable of type LQR_full_data_type used for internal data
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

     TYPE ( LQR_control_type ), INTENT( INOUT ) :: control
     TYPE ( LQR_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

     status = GALAHAD_ready_to_solve
     RETURN

!  End of subroutine LQR_import

     END SUBROUTINE LQR_import

!-  G A L A H A D -  L Q R _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE LQR_information( data, inform, status )

!  return solver information during or after solution by LQR
!  See LQR_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LQR_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( LQR_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%lqr_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine LQR_information

     END SUBROUTINE LQR_information

!-*-*-*-*-*-  End of G A L A H A D _ L Q R  double  M O D U L E  *-*-*-*-*-*-

   END MODULE GALAHAD_LQR_precision
