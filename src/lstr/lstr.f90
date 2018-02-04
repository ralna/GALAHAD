! THIS VERSION: GALAHAD 2.6 - 08/04/2015 AT 12:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ L S T R   M O D U L E  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.1, November 4th, 2007

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LSTR_double

!      -----------------------------------------------
!      |                                             |
!      | Solve the regularised least-squares problem |
!      |                                             |
!      |    minimize     || A x - b ||_2^2           |
!      |    subject to   ||x|| <= radius             |
!      |                                             |
!      | using a Lanczos bi-diagonalisation method   |
!      |                                             |
!      -----------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_ROOTS_double, ONLY : ROOTS_quadratic
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_BLAS_interface, ONLY : ROTG

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LSTR_initialize, LSTR_read_specfile, LSTR_solve,               &
                LSTR_terminate, LSTR_transform_bidiagonal,                     &
                LSTR_backsolve_bidiagonal

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
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )

      REAL ( KIND = wp ), PARAMETER :: roots_tol = ten ** ( - 12 )
      REAL ( KIND = wp ), PARAMETER :: error_tol = ten ** ( - 12 )
      REAL ( KIND = wp ), PARAMETER :: error_tol_relax = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: lambda_start = ten ** ( - 12 )
      LOGICAL :: roots_debug = .FALSE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSTR_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   any printing will start on this iteration

       INTEGER :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER :: stop_print = - 1

!   the number of iterations between printing

       INTEGER :: print_gap = 1

!   the minimum number of iterations allowed (-ve = no bound)

        INTEGER :: itmin = - 1

!   the maximum number of iterations allowed (-ve = no bound)

        INTEGER :: itmax = - 1

!   the maximum number of iterations allowed once the boundary has been
!    encountered (-ve = no bound)

        INTEGER :: itmax_on_boundary = - 1

!   the maximum number of Newton inner iterations per outer iteration allowed
!    (-ve = no bound)

        INTEGER :: bitmax = - 1

!   the number of extra work vectors of length n used

        INTEGER :: extra_vectors = 0

!   the iteration stops successfully when ||A^Tr|| is less than
!     max( stop_relative * ||A^Tr initial ||, stop_absolute )

        REAL ( KIND = wp ) :: stop_relative = epsmch
        REAL ( KIND = wp ) :: stop_absolute = zero

!   an estimate of the solution that gives at least %fraction_opt times
!    the optimal objective value will be found

        REAL ( KIND = wp ) :: fraction_opt = one

!   the maximum elapsed time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: time_limit = - one

!   should the iteration stop when the Trust-region is first encountered ?

        LOGICAL :: steihaug_toint  = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSTR_inform_type

!  return status. See LSTR_solve for details

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

!  the total number of inner iterations performed

        INTEGER :: biters = - 1

!  the smallest number of inner iterations performed during an outer iteration

        INTEGER :: biter_min = - 1

!  the smallest number of inner iterations performed during an outer iteration

        INTEGER :: biter_max = - 1

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = wp ) :: multiplier = zero

!  the Euclidean norm of x

        REAL ( KIND = wp ) :: x_norm = zero

!  the Euclidean norm of Ax-b

        REAL ( KIND = wp ) :: r_norm = biginf

!  the Euclidean norm of A^T (Ax-b) + multiplier * x

        REAL ( KIND = wp ) :: Atr_norm = biginf

!  the average number of inner iterations performed during an outer iteration

        REAL ( KIND = wp ) :: biter_mean = - one
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LSTR_data_type
        PRIVATE
        INTEGER :: iter, itmin, itmax, end_pass2, switch, bitmax, bstatus, biter
        INTEGER :: branch, itmax_on_boundary, extra_vectors
        INTEGER :: start_print, stop_print, print_gap
        REAL ( KIND = wp ) :: alpha_kp1, beta_kp1, beta_1
        REAL ( KIND = wp ) :: rho_k, rho_bar, phi_k, phi_bar, theta_kp1
        REAL ( KIND = wp ) :: s, c, s_w, c_w, zeta_bar, gamma, z_norm
        REAL ( KIND = wp ) :: eta_bar, lambda_km1, lambda_bar, ww, radius2
        REAL ( KIND = wp ) :: error_tol, stop, decrease_st
        REAL ( KIND = wp ) :: time_start, time_now
        REAL ( KIND = wp ) :: clock_start, clock_now
        LOGICAL :: set_printi, printi, set_printd, printd, interior, header
        LOGICAL :: prev_steihaug_toint, save_vectors, try_warm
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_diag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_offdiag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DECREASE
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: LAMBDA
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_diag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_offdiag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: F
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U_extra
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: V_extra
      END TYPE

    CONTAINS

!-*-*-*-*-*-  L S T R _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LSTR_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  .  Set initial values for the LSTR control parameters  .
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

      TYPE ( LSTR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSTR_control_type ), INTENT( OUT ) :: control
      TYPE ( LSTR_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set initial control parameter values

      control%stop_relative = SQRT( EPSILON( one ) )
      data%branch = 1
      data%prev_steihaug_toint = .TRUE.

      RETURN

!  End of subroutine LSTR_initialize

      END SUBROUTINE LSTR_initialize

!-*-*-*-*-   L S T R _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LSTR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by LSTR_initialize could (roughly)
!  have been set as:

!  BEGIN LSTR SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   start-print                                     -1
!   stop-print                                      -1
!   minimum-number-of-iterations                    -1
!   maximum-number-of-iterations                    -1
!   maximum-number-of-boundary-iterations           -1
!   maximum-number-of-inner-iterations              -1
!   number-extra-n-vectors-used                     0
!   relative-accuracy-required                      1.0E-8
!   absolute-accuracy-required                      0.0
!   fraction-optimality-required                    1.0
!   maximum-time-limit                              -1.0
!   stop-as-soon-as-boundary-encountered            T
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END LSTR SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LSTR_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: start_print = print_level + 1
      INTEGER, PARAMETER :: stop_print  = start_print + 1
      INTEGER, PARAMETER :: print_gap = stop_print + 1
      INTEGER, PARAMETER :: itmin = print_gap + 1
      INTEGER, PARAMETER :: itmax = itmin + 1
      INTEGER, PARAMETER :: itmax_on_boundary = itmax + 1
      INTEGER, PARAMETER :: bitmax = itmax_on_boundary + 1
      INTEGER, PARAMETER :: extra_vectors = bitmax + 1
      INTEGER, PARAMETER :: stop_relative =  extra_vectors + 1
      INTEGER, PARAMETER :: stop_absolute = stop_relative + 1
      INTEGER, PARAMETER :: fraction_opt = stop_absolute + 1
      INTEGER, PARAMETER :: time_limit = fraction_opt + 1
      INTEGER, PARAMETER :: steihaug_toint = time_limit + 1
      INTEGER, PARAMETER :: space_critical = steihaug_toint + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LSTR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print  )%keyword = 'stop-print'
      spec( print_gap )%keyword = 'iterations-between-printing'
      spec( itmin )%keyword = 'minimum-number-of-iterations'
      spec( itmax )%keyword = 'maximum-number-of-iterations'
      spec( itmax_on_boundary )%keyword                                        &
        = 'maximum-number-of-boundary-iterations'
      spec( bitmax )%keyword = 'maximum-number-of-inner-iterations'
      spec( extra_vectors )%keyword = 'number-extra-n-vectors-used'

!  real key-words

      spec( stop_relative )%keyword = 'relative-accuracy-required'
      spec( stop_absolute )%keyword = 'absolute-accuracy-required'
      spec( fraction_opt )%keyword = 'fraction-optimality-required'
      spec( time_limit )%keyword = 'maximum-time-limit'

!  logical key-words

      spec( steihaug_toint )%keyword = 'stop-as-soon-as-boundary-encountered'
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

!  set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( start_print ),                         &
                                  control%start_print,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_print  ),                         &
                                  control%stop_print ,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_gap ),                           &
                                  control%print_gap,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itmin ),                               &
                                  control%itmin,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itmax ),                               &
                                  control%itmax,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itmax_on_boundary ),                   &
                                  control%itmax_on_boundary,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( bitmax ),                              &
                                  control%bitmax,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( extra_vectors ),                       &
                                  control%extra_vectors,                       &
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
      CALL SPECFILE_assign_value( spec( time_limit ),                          &
                                  control%time_limit,                          &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( steihaug_toint ),                      &
                                  control%steihaug_toint,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ), control%prefix,              &
                                  control%error )

      RETURN

      END SUBROUTINE LSTR_read_specfile

!-*-*-*-*-*-*-*-*-*-*  L S T R _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE LSTR_solve( m, n, radius, X, U, V, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   m        number of equations (= number of rows of A)
!   n        number of unknowns (= number of columns of A)
!   radius   trust-region radius
!   X        the vector of unknowns. Need not be set on entry.
!            On exit, the best value found so far
!   U        see inform%status = 2-5. On initial entry this must contain b
!   V        see inform%status = 2-3
!   data     private internal data
!   control  a structure containing control information. See LSTR_initialize
!   inform   a structure containing information. The component
!             %status is the input/output status. This must be set to 1 on
!              initial entry or 5 on a re-entry when only sigma has
!              been reduced since the last entry. Other values are
!               2 on exit, the product A with V must be added to U with
!                   the result kept in U, and the subroutine re-entered.
!               3 on exit, the product A^T with U must be added to V with
!                   the result kept in V, and the subroutine re-entered.
!               4 The iteration will be restarted. Reset U to b and re-enter.
!                 This exit will only occur if control%steihaug_toint is
!                 .FALSE. and the solution lies on the trust-region boundary
!               5 The iteration is to be restarted. Set U to b.
!               0 the solution has been found
!              -1 an array allocation has failed
!              -2 an array deallocation has failed
!              -3 m and/or n and/or radius is not positive
!              -18 the iteration limit has been exceeded
!              -25 status is not > 0 on entry
!              -36 the trust-region has been encountered in Steihaug-Toint mode
!             the remaining components are described in the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  A description of LSQR and the notation used here is based on the papers
!
!    C. C. Paige and M. A. Saunders,
!    "LSQR: an algorithm for sparse linear equations & least squares problems",
!    ACM Transactions on Mathematical Software, Vol 8, No 1 (1982), pp 43-71,
!
!  and
!
!    C. C. Paige and M. A. Saunders,
!    "Algorithm 583. LSQR: sparse linear equations and least squares problems",
!    ACM Transactions on Mathematical Software, Vol 8, No 2 (1982), pp 195-209.
!
!  The bi-diagonalisation is due to Golub and Kahan (SIMUM 2, 1965, pp 205-224)
!  (normalisation ||u_k|| = 1 = ||v_k||, 2-norm throughout).
!
!    Bi-diagonalisation initialization -
!
!       beta_1 u_1 = b   and
!      alpha_1 v_1 = A^T u_1
!
!    Bi-diagonalisation iteration (k = 1,2, ... ) -
!
!       beta_k+1 u_k+1 = A v_k - alpha_k u_k  and
!      alpha_k+1 v_k+1 = A^T u_k+1 - beta_k+1 v_k
!
!  This leads to
!
!    U_k+1^T A V_k = B_k  and  U_k+1^T b = beta_1 e_1 = ||b|| e_1 ,
!
!  where U_k = ( u_1 u_2 ... u_k ), V_k = ( v_1 v_2 ... v_k ) and
!
!         ( alpha_1                              )
!         (  beta_2  alpha_2                     )
!   B_k = (             .        .               )
!         (                    beta_k   alpha_k  )
!         (                             beta_k+1 )
!
!  To solve min || A x - b ||^2 find an approximation in the expanding
!  subspace x = V_k y => solve min || B_k y - beta_1 e_1 || =>
!  solve || R_k y - f_k ||, where for some product of plane-rotations Q_k,
!
!   Q_k ( B_k : beta_1 e_1 ) = ( R_k     f_k    ),  f_k = ( f_k-1 )  and
!                              (     phibar_k+1 )         ( phi_k )
!
!         ( rho_1  theta_2                   )
!   R_k = (            .        .            )
!         (                 rho_k-1  theta_k )
!         (                           rho_k  )
!
!  Thus R_k y_k = f_k or x_k = V_k R_k^-1 f_k = D_k f_k, where
!
!    V_k R_k^-1 = D_k = ( d_1 d_2 .... d_k ).
!
!  Hence
!
!    x_k = D_k-1 f_k-1 + d_k phi_k = x_k-1 + phi_k d_k.
!
!  Fortunately the precise (upper-bi-diagonal) form of R_k =>
!
!    d_k = ( v_k - theta_k d_k-1 ) / rho_k
!
!  A small saving can be made by defining w_k = rho_k d_k =>
!
!    x_k = x_k-1 + ( phi_k / rho_k ) w_k and
!    w_k+1 = v_k+1 - ( theta_k+1 / rho_k ) w_k
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER, INTENT( IN ) :: m, n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, V
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: U
      TYPE ( LSTR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSTR_control_type ), INTENT( IN ) :: control
      TYPE ( LSTR_inform_type ), INTENT( INOUT ) :: inform

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, nroots
      REAL ( KIND = wp ) :: alpha_1, num, zeta, x_kp1_norm, rat, xx, txw
      REAL ( KIND = wp ) :: st_step, other_root, phi_bar_k, dec_tol, rbiters
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Branch to different sections of the code depending on input status

      IF ( inform%status < 1 ) GO TO 930
      IF ( inform%status == 1 ) data%branch = 1
      IF ( inform%status == 5 ) data%branch = 8

      SELECT CASE ( data%branch )
      CASE ( 1 )
        GO TO 100
      CASE ( 2 )
        GO TO 200
      CASE ( 3 )
        GO TO 300
      CASE ( 4 )
        GO TO 400
      CASE ( 5 )
        GO TO 500
      CASE ( 6 )
        GO TO 600
      CASE ( 7 )
        GO TO 700
      CASE ( 8 )
        GO TO 800
      END SELECT

!  On initial entry, set constants

  100 CONTINUE

!  Check for obvious errors

      IF ( n <= 0 .OR. m < 0 ) GO TO 940
      IF ( radius <= zero ) GO TO 950

!  record the initial time

      CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  Set iteration limits

      data%itmin = control%itmin
      IF ( control%itmax >= 0 ) THEN
        data%itmax = control%itmax
      ELSE
        data%itmax = MAX( m, n ) + 1
      END IF

      IF ( control%itmax_on_boundary > 0 ) THEN
        data%itmax_on_boundary = control%itmax_on_boundary
      ELSE
        data%itmax_on_boundary = MAX( m, n ) + 1
      END IF

!  =====================
!  Array (re)allocations
!  =====================

      array_name = 'gltr: W'
      CALL SPACE_resize_array( n, data%W,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      IF ( .NOT. control%steihaug_toint ) THEN

        IF ( control%bitmax >= 0 ) THEN
          data%bitmax = control%bitmax
        ELSE
          data%bitmax = 10
        END IF

!  Allocate space for the solution, y, in the subspace x = V_k y

        array_name = 'gltr: Y'
        CALL SPACE_resize_array( data%itmax, data%Y,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate space for the Lanczos bidiagonal, B_k

        array_name = 'gltr: B_diag'
        CALL SPACE_resize_array( data%itmax + 1, data%B_diag,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'gltr: B_offdiag'
        CALL SPACE_resize_array( data%itmax, data%B_offdiag,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate space for the Cholesky factor R_k of B_k^T B_k

        array_name = 'gltr: R_diag'
        CALL SPACE_resize_array( data%itmax, data%R_diag,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'gltr: R_offdiag'
        CALL SPACE_resize_array( data%itmax - 1, data%R_offdiag,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate worspace for the bi-diagonal trust-region solves

        array_name = 'gltr: F'
        CALL SPACE_resize_array( data%itmax, data%F,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'gltr: G'
        CALL SPACE_resize_array( 0, data%itmax, data%G,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'gltr: H'
        CALL SPACE_resize_array( data%itmax, data%H,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate workspace for the sequence of smallest function values & multiplier

        array_name = 'gltr: DECREASE'
        CALL SPACE_resize_array( data%itmax, data%DECREASE,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'gltr: LAMBDA'
        CALL SPACE_resize_array( data%itmax, data%LAMBDA,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  If required allocate extra space to store U and V in case of a second pass

        data%extra_vectors = control%extra_vectors
        IF ( data%extra_vectors > 0 ) THEN
          array_name = 'gltr: U_extra'
          CALL SPACE_resize_array( m, data%U_extra,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status == 0 ) THEN
            array_name = 'gltr: V_extra'
            CALL SPACE_resize_array( n, data%extra_vectors, data%V_extra,      &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) THEN
              array_name = 'lstr: U_extra'
              CALL SPACE_dealloc_array( data%U_extra,                          &
                inform%status, inform%alloc_status, array_name = array_name,   &
                bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= 0 ) GO TO 960
              data%extra_vectors = 0
            END IF
          ELSE
            data%extra_vectors = 0
          END IF
        END IF
      END IF

!  initialization

      inform%iter = 0 ; inform%iter_pass2 = 0
      X = zero
      inform%x_norm = zero ; data%z_norm = zero
      inform%multiplier = zero
      inform%biters = 0
      inform%biter_min = 0 ; inform%biter_max = 0 ; inform%biter_mean = zero

      data%error_tol = error_tol
      data%radius2 = radius * radius
      data%interior = .TRUE.
      data%try_warm = .FALSE.
      data%prev_steihaug_toint = control%steihaug_toint
      data%save_vectors = .NOT. control%steihaug_toint .AND.                   &
        data%extra_vectors > 0

!  print level shorthands

      IF ( control%start_print < 0 ) THEN
        data%start_print = 0
      ELSE
        data%start_print = control%start_print
      END IF
      IF ( control%stop_print < 0 ) THEN
        data%stop_print = data%itmax + 1
      ELSE
        data%stop_print = control%stop_print
      END IF
      IF ( control%print_gap < 2 ) THEN
         data%print_gap = 1
       ELSE
         data%print_gap = control%print_gap
       END IF

!  basic single line of output per iteration

      data%set_printi = control%out > 0 .AND. control%print_level >= 1

!  full debugging printing with significant arrays printed

      data%set_printd = control%out > 0 .AND. control%print_level >= 2

!  set print agenda for the first iteration

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printd = data%set_printd
     ELSE
       data%printi = .FALSE. ; data%printd = .FALSE.
     END IF

!  Recur the bi-diagonalisation vectors in u (= u_k) and v (= v_k)
!
!  Bi-diagonalisation initialization: beta_1 u_1 = b

      V = zero

!  Normalize u_1

      data%beta_1 = TWO_NORM( U )
      IF ( data%beta_1 > zero ) THEN
        U = U / data%beta_1

!  ** Return ** to obtain the product v <- v + A^T * u

        data%branch = 2 ; inform%status = 3
        RETURN
      END IF

!  Continue bi-diagonalisation initialization: alpha_1 v_1 = A^T u_1

  200 CONTINUE

!  Normalize v_1 and initialize w

      alpha_1 = TWO_NORM( V )
      IF ( .NOT. control%steihaug_toint ) data%B_diag( 1 ) = alpha_1
      IF ( alpha_1 > zero ) THEN
        V = V / alpha_1
        data%W( : n ) = V
      END IF
      data%ww = one

!  Initialize ||r_1|| and ||A^T r_1|| , where the residual r_1 = b - A x_1

      inform%r_norm  = data%beta_1
      inform%ATr_norm = alpha_1 * data%beta_1

!  Check for convergence

      IF ( ABS( inform%ATr_norm ) <= control%stop_absolute .AND.               &
           inform%iter >= data%itmin ) GO TO 900

!   Compute the stopping tolerance

      data%stop = MAX( control%stop_relative * inform%ATr_norm,                &
                       control%stop_absolute )

!  Initialize R_1 and f_1

      data%rho_bar = alpha_1
      data%phi_bar = data%beta_1

!  Print initial information

      IF ( data%printi ) THEN
        WRITE( control%out, 2000 ) prefix
        WRITE( control%out, "( A, I7, ES16.8, 2ES9.2 )" )                      &
              prefix, inform%iter, inform%r_norm, inform%ATr_norm, inform%x_norm
      END IF

!  ===========================
!  Start of the main iteration
!  ===========================

    data%alpha_kp1 = alpha_1

  290   CONTINUE

!  Check that the iteration bound has not been exceeded

        IF ( inform%iter + 1 > data%itmax ) THEN
          IF ( data%interior ) GO TO 920
          GO TO 490
        END IF
        inform%iter = inform%iter + 1
        data%header = data%printd .OR. inform%iter == data%start_print .OR.    &
                      MOD( inform%iter - data%start_print, 25 ) == 0

!  check that the time limit has not been exceeded

        CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
        data%time_now = data%time_now - data%time_start
        data%clock_now = data%clock_now - data%clock_start

        IF ( control%time_limit > zero .AND.                                   &
              data%clock_now > control%time_limit ) THEN
          IF ( data%printi )                                                   &
            WRITE( control%out, "( /, A, ' Time limit exceeded ' )" ) prefix
          inform%status = GALAHAD_error_time_limit ; RETURN
        END IF

!  set printing agenda for the next iteration

        IF ( inform%iter >= data%start_print .AND.                             &
             inform%iter < data%stop_print .AND.                               &
             MOD( inform%iter - data%start_print, data%print_gap ) == 0 ) THEN
          data%printi = data%set_printi ; data%printd = data%set_printd
        ELSE
          data%printi = .FALSE. ; data%printd = .FALSE.
        END IF

!  If the user has asked to save vectors, save v and possibly u

        IF ( data%save_vectors ) THEN
          IF ( inform%iter <= data%extra_vectors )                             &
            data%V_extra( : n , inform%iter ) = V
          IF ( inform%iter == data%extra_vectors + 1 )                         &
            data%U_extra( : m ) = U
        END IF

!  Bi-diagonalisation iteration:
!    beta_k+1 u_k+1 = A v_k - alpha_k u_k

!  ** Return ** to obtain the product u <- u + A * v

        U = - data%alpha_kp1 * U
        data%branch = 3 ; inform%status = 2
        RETURN

!  Normalize u

  300   CONTINUE
        data%beta_kp1 = TWO_NORM( U )
        IF ( .NOT. control%steihaug_toint )                                    &
          data%B_offdiag( inform%iter ) = data%beta_kp1

        IF ( data%beta_kp1 > zero ) THEN
          U = U / data%beta_kp1

!  Continue bi-diagonalisation iteration:
!    alpha_k+1 v_k+1 = A^T u_k+1 - beta_k+1 v_k

!  ** Return ** to obtain the product v <- v + A^T * u

          V = - data%beta_kp1 * V
          data%branch = 4 ; inform%status = 3
          RETURN
        END IF

!  Normalize v

  400   CONTINUE
        data%alpha_kp1 = TWO_NORM( V )
        IF ( .NOT. control%steihaug_toint )                                    &
          data%B_diag( inform%iter + 1 ) = data%alpha_kp1
        IF ( data%alpha_kp1 > zero ) THEN
          V = V / data%alpha_kp1
        END IF

!  The iterates are interior

        IF ( data%interior ) THEN

!  Apply a plane rotation Q to B_k to remove the new sub-diagonal
!  entry and thus create the new entries in R_k and f_k

!  Construct the plane rotation Q to eliminate beta_k+1 from ( rho_bar beta_k+1)

          data%rho_k = data%rho_bar
          CALL ROTG( data%rho_k, data%beta_kp1, data%c, data%s )

!  Apply the rotation Q to the components ( phi_bar 0 ) of f_k

          phi_bar_k = data%phi_bar
          data%phi_k = data%c * data%phi_bar
          data%phi_bar = data%s * data%phi_bar

          IF ( inform%iter > 1 ) THEN

!  Construct a plane rotation W to remove the super-diagonal entry theta_k from
!  R_k (and thus incrementally turn R_k into a lower-bidiagonal matrix L_k)

            data%lambda_km1 = data%lambda_bar
            CALL ROTG( data%lambda_km1, data%theta_kp1, data%c_w, data%s_w )
            zeta = data%zeta_bar * ( data%lambda_bar ) / data%lambda_km1
            data%z_norm = ROOT_SUM_SQUARES( data%z_norm, zeta )

!  Apply the rotation W to the vector ( 0 rho_k )

            data%gamma = data%s_w * data%rho_k
            data%lambda_bar = data%c_w * data%rho_k

!  Compute the norm of ||x_k|| - see Paige and Saunders ACM TOMS 8(1) 1982,
!  Sec 5.2 for a brief summary

            num = data%phi_k - data%gamma * zeta
            data%zeta_bar = num / data%lambda_bar
            x_kp1_norm = ROOT_SUM_SQUARES( data%z_norm, data%zeta_bar )
          ELSE
            data%lambda_bar = data%rho_k
            data%z_norm = zero
            data%zeta_bar = data%phi_k / data%lambda_bar
            x_kp1_norm = ABS( data%zeta_bar )
          END IF

!  ||x_k|| exceeds the trust-region radius. The solution must lie on the
!  trust-region boundary

          IF ( x_kp1_norm > radius ) THEN
            data%interior = .FALSE.
            data%header = .TRUE.
            data%switch = inform%iter
            data%itmax = MIN( data%itmax, inform%iter + data%itmax_on_boundary )
            IF ( data%printi ) WRITE( control%out,                             &
              "( /, A, '   Solution lies on trust-region boundary' )" ) prefix

!  Compute the Steihaug-Toint point: find the appropriate point on the boundary

            xx = inform%x_norm ** 2
!           txw = two * DOT_PRODUCT( X, data%W( : n ) )
!           data%ww = DOT_PRODUCT( data%W( : n ), data%W( : n ) )
            alpha_1 = data%phi_k / data%rho_k
            txw = ( x_kp1_norm ** 2 - xx ) / alpha_1 - data%ww * alpha_1
            CALL ROOTS_quadratic( xx - data%radius2, txw, data%ww,             &
                                  roots_tol, nroots, other_root, st_step,      &
                                  roots_debug )

!  make sure we find the correct root

            IF ( phi_bar_k * data%c * data%rho_k < zero ) st_step = other_root

            xx = xx + st_step * ( txw + st_step * data%ww )
            X = X + st_step * data%W( : n )
            inform%x_norm = radius

!  Compute ||r|| and the decrease in ||r||^2 at the Steihaug-Toint point

            inform%r_norm = ROOT_SUM_SQUARES( data%rho_k * st_step * data%s,   &
              data%rho_k * st_step * data%c - phi_bar_k )
!           data%decrease_st = SQRT( data%beta_1 ** 2 - inform%r_norm ** 2 )
            data%decrease_st = data%beta_1 - inform%r_norm
            IF ( data%printi ) THEN
              WRITE( control%out, 2000 ) prefix
              WRITE( control%out, "( A, I7, ES16.8, '    -    ', ES9.2 )" )    &
                prefix, inform%iter, inform%r_norm, inform%x_norm
            END IF
            IF ( control%steihaug_toint ) GO TO 910

!  The iterate remains within the trust-region

          ELSE
            inform%x_norm = x_kp1_norm

!  Apply the rotation Q to the forthcoming column ( 0 alpha_k+1 ) in B_k+1

            data%theta_kp1 = data%s * data%alpha_kp1
            data%rho_bar = - data%c * data%alpha_kp1

!  Update the approximate solution x and the correction vector w

            X = X + ( data%phi_k / data%rho_k ) * data%W( : n )
            rat =  data%theta_kp1 / data%rho_k
            data%W( : n ) = V - rat * data%W( : n )

!  Update the square of the norm of w

            data%ww = one + data%ww * rat ** 2

!  Compute the norms ||r_k|| and ||A^T r_k|| - see Paige and Saunders
!  ACM TOMS 8(1) 1982, Sec 5.1

            inform%r_norm  = ABS( data%phi_bar )
            inform%ATr_norm = ABS( data%alpha_kp1 * data%c * data%phi_bar )
          END IF
        END IF

!  The iterates are on the boundary

        IF ( .NOT. data%interior ) THEN

!  Find the solution on the boundary in the subspace x = V_k y

          IF ( data%try_warm )                                                 &
            data%LAMBDA( inform%iter ) = data%LAMBDA( inform%iter - 1 )
          CALL LSTR_solve_bidiagonal( inform%iter,                             &
            data%B_diag( : inform%iter ), data%B_offdiag( : inform%iter ),     &
            data%beta_1, radius, data%LAMBDA( inform%iter ),                   &
            data%Y( : inform%iter ), data%H( : inform%iter ),                  &
            data%R_diag( : inform%iter ), data%R_offdiag( : inform%iter - 1 ), &
            data%F( : inform%iter ), data%G( : inform%iter ), data%error_tol,  &
            data%try_warm, data%printd, control%out, prefix, data%bitmax,      &
            data%biter, data%bstatus )

!  Record statistics about the number of inner iterations performed

          IF ( INFORM%biters > 0 ) THEN
            rbiters = INFORM%biters
            inform%biters = inform%biters + 1
            inform%biter_min = MIN( inform%biter_min, data%biter )
            inform%biter_max = MAX( inform%biter_max, data%biter )
            inform%biter_mean = ( rbiters / ( rbiters + one ) ) *              &
              ( inform%biter_mean + data%biter / rbiters )
          ELSE
            inform%biters = 1
            inform%biter_min = data%biter
            inform%biter_max = data%biter
            inform%biter_mean = data%biter
          END IF

!  Compute the norms ||r_k|| and ||A^T r_k||

          data%try_warm = .TRUE.
          inform%r_norm = SQRT( DOT_PRODUCT( data%G( : inform%iter ),          &
                                             data%G( : inform%iter ) )         &
                          - data%LAMBDA( inform%iter ) * (radius ** 2 ) )
          inform%ATr_norm = ABS( data%alpha_kp1 * data%B_offdiag( inform%iter )&
                                 * data%Y( inform%iter ) )
!         inform%ATr_norm = ABS( data%B_diag( inform%iter + 1 ) *              &
!                 data%B_offdiag( inform%iter ) * data%Y( inform%iter ) )
          data%DECREASE( inform%iter ) = data%beta_1 - inform%r_norm
        END IF

!  Print progress

        IF ( data%printi ) THEN
          IF ( data%header ) THEN
            IF ( data%interior ) THEN
              WRITE( control%out, 2000 ) prefix
            ELSE
              WRITE( control%out, 2010 ) prefix
            END IF
          END IF
          IF ( data%interior ) THEN
            WRITE( control%out, "( A, I7, ES16.8, 2ES9.2 )" ) prefix,          &
               inform%iter, inform%r_norm, inform%ATr_norm, inform%x_norm
          ELSE
            WRITE( control%out, "( A, I7, ES16.8, 3ES9.2, I7 )" ) prefix,      &
               inform%iter, inform%r_norm, inform%ATr_norm, radius,            &
               data%LAMBDA( inform%iter ), data%biter
          END IF
        END IF

!  Check for termination

        IF ( inform%ATr_norm < data%stop .AND. inform%iter >= data%itmin ) THEN

!  Convergence inside the trust-region occurred

          IF ( data%interior ) GO TO 900
          GO TO 490
        END IF

!  =========================
!  End of the main iteration
!  =========================

      GO TO 290

!  Termination on the trust-region boundary has occured. Determine at which
!  iteration a fraction, fraction_opt, of the optimal solution was found

  490 CONTINUE
      IF ( control%fraction_opt < one ) THEN
        dec_tol = data%DECREASE( inform%iter ) * control%fraction_opt

!  See if the required decrease occurred at or before the Steihaug-Toint point

        IF ( data%decrease_st >= dec_tol ) GO TO 900

!  Examine subsequent iterates

        DO i = data%switch, inform%iter
          data%end_pass2 = i
          IF ( data%DECREASE( data%end_pass2 ) >= dec_tol ) EXIT
        END DO
      ELSE
        data%end_pass2 = inform%iter
      END IF

!  Recover the solution on the boundary in the subspace x = V_pass2 y

      CALL LSTR_solve_bidiagonal( data%end_pass2,                              &
        data%B_diag( : data%end_pass2 ),                                       &
        data%B_offdiag( : data%end_pass2 ),                                    &
        data%beta_1, radius, data%LAMBDA( data%end_pass2 ),                    &
        data%Y( : data%end_pass2 ), data%H( : data%end_pass2 ),                &
        data%R_diag( : data%end_pass2 ),                                       &
        data%R_offdiag( : data%end_pass2 - 1 ),                                &
        data%F( : data%end_pass2 ), data%G( : data%end_pass2 ),                &
        data%error_tol, .TRUE., data%printd, control%out, prefix,              &
        data%bitmax, data%biter, data%bstatus )

!  Compute the norms ||r_pass2|| and ||A^T r_pass2|| and the Lagrange multiplier

      inform%multiplier = data%Y( data%end_pass2 )
      inform%r_norm = SQRT( DOT_PRODUCT( data%G( : data%end_pass2 ),           &
                                         data%G( : data%end_pass2 ) )          &
                        - data%LAMBDA( data%end_pass2 ) * (radius ** 2 ) )
      inform%ATr_norm = ABS( data%B_diag( data%end_pass2 + 1 ) *               &
              data%B_offdiag( data%end_pass2 ) * data%Y( data%end_pass2 ) )

!  Restart to compute x_pass2

      IF ( data%extra_vectors == 0 ) THEN
        data%branch = 5 ; inform%status = 4
        RETURN
      END IF

!  =================================================
!  Second pass to recover the solution x_k = V_k y_k
!  from the vectors V_k = ( v_1 : ... : v_k )
!  =================================================

  500 CONTINUE
      IF ( data%extra_vectors == 0 ) THEN
        X = zero
        inform%iter_pass2 = 0

!  Reentry with no saved vectors: Normalize u_1 and initialize v_1

        U = U / data%beta_1
        V = zero
      ELSE

!  Reentry with  data%extra_vectors saved vectors: loop over saved v_i to
!  update x_k = sum_i v_i y_i

        inform%iter_pass2 = MIN( data%end_pass2, data%extra_vectors )
        X = MATMUL( data%V_extra( : n, : inform%iter_pass2 ),                  &
                    data%Y( : inform%iter_pass2 ) )

!  Check to see if the solution has been recovered

        IF ( inform%iter_pass2 >= data%end_pass2 ) GO TO 900

!  Initialize u_k and - beta_k v_k-1

        U = data%U_extra( : m )
        V = - data%B_offdiag( inform%iter_pass2 ) *                            &
                data%V_extra( : n, inform%iter_pass2 )
      END IF

!  ===============================
!  Iterate over the columns of V_k
!  ===============================

  590   CONTINUE

!  Bi-diagonalisation iteration:
!    alpha_k v_k = A^T u_k - beta_k v_k-1

!  ** Return ** to obtain the product v <- v + A^T * u

        data%branch = 6 ; inform%status = 3
        RETURN

!  Normalize v_k

  600   CONTINUE
        inform%iter_pass2 = inform%iter_pass2 + 1
        V = V / data%B_diag( inform%iter_pass2 )

!  Update x_k

        X = X + data%Y( inform%iter_pass2 ) * V

!  Check to see if the solution has been recovered

        IF ( inform%iter_pass2 >= data%end_pass2 ) GO TO 900

!  Bi-diagonalisation iteration:
!    beta_k+1 u_k+1 = A v_k - alpha_k u_k

!  ** Return ** to obtain the product u <- u + A * v

        U = - data%B_diag( inform%iter_pass2 ) * U
        data%branch = 7 ; inform%status = 2
        RETURN

!  Normalize u_k+1 and scale v_k

  700   CONTINUE
        U = U / data%B_offdiag( inform%iter_pass2 )
        V = - data%B_offdiag( inform%iter_pass2 ) * V

!  ======================
!  End of the second pass
!  ======================

      GO TO 590

!  ======================================================
!  Re-entry for solution with smaller trust-region radius
!  ======================================================

  800 CONTINUE
      IF ( control%steihaug_toint ) GO TO 100
      IF ( data%prev_steihaug_toint ) GO TO 100
      inform%iter = 0

!  Find the solution to the Lanczos TR subproblem with this radius

      CALL LSTR_solve_bidiagonal( data%end_pass2,                              &
        data%B_diag( : data%end_pass2 ),                                       &
        data%B_offdiag( : data%end_pass2 ),                                    &
        data%beta_1, radius, data%LAMBDA( data%end_pass2 ),                    &
        data%Y( : data%end_pass2 ), data%H( : data%end_pass2 ),                &
        data%R_diag( : data%end_pass2 ),                                       &
        data%R_offdiag( : data%end_pass2 - 1 ),                                &
        data%F( : data%end_pass2 ), data%G( : data%end_pass2 ),                &
        data%error_tol, .TRUE., data%printd, control%out, prefix,              &
        data%bitmax, data%biter, data%bstatus )

!  Compute the norms ||r_pass2|| and ||A^T r_pass2|| and the Lagrange multiplier

      inform%multiplier = data%Y( data%end_pass2 )
      inform%r_norm = SQRT( DOT_PRODUCT( data%G( : data%end_pass2 ),           &
                                         data%G( : data%end_pass2 ) )          &
                        - data%LAMBDA( data%end_pass2 ) * (radius ** 2 ) )
      inform%ATr_norm = ABS( data%B_diag( data%end_pass2 + 1 ) *               &
              data%B_offdiag( data%end_pass2 ) * data%Y( data%end_pass2 ) )

      GO TO 500

!  Successful returns

  900 CONTINUE
      inform%status = 0
      RETURN

!  Boundary encountered in Steihaug-Toint method

  910 CONTINUE
      IF ( data%printi )                                                       &
        WRITE( control%out, "( /, A, ' Steihaug-Toint point found ' ) " ) prefix
      inform%status = GALAHAD_warning_on_boundary
      RETURN

!  Too many iterations

  920 CONTINUE
      IF ( data%printi )                                                       &
        WRITE( control%out, "( /, A, ' Iteration limit exceeded ' ) " ) prefix
      inform%status = GALAHAD_error_max_iterations
      RETURN

!  Inappropriate entry status

  930 CONTINUE
      inform%status = GALAHAD_error_input_status
      RETURN

!  Inappropriate dimension

  940 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' n = ', I0, ' or m = ', I0, ' is too small ' )" ) prefix, n, m
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Non-positive radius

  950 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, radius
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Allocation or deallocation errors

  960 CONTINUE
      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, '   Iter     ||Ax-b||    ||A^Tr||   ||x|| ' )
 2010 FORMAT( /, A, '   Iter     ||Ax-b||    ||A^Tr||   ||x||   lambda  Newton')

      CONTAINS

! -*-*- R O O T _ S U M _ S Q U A R E S    I N T E R N A L   F U N C T I O N -*-

        FUNCTION ROOT_SUM_SQUARES( a, b )

!  Evaluates the square root of the sum of squares of a and b

!  Dummy arguments

        REAL ( KIND = wp ) :: ROOT_SUM_SQUARES
        REAL ( KIND = wp ), INTENT( IN ) :: a, b

!  Local variable

        REAL ( KIND = wp ) :: s

!  Take precautions to try to prevent overflow and underflow

        s = ABS( a ) + ABS( b )
        IF ( s == zero ) THEN
           ROOT_SUM_SQUARES = zero
        ELSE
           ROOT_SUM_SQUARES = s * SQRT( ( a / s ) ** 2 + ( b / s ) ** 2 )
        END IF

!  End of function ROOT_SUM_SQUARES

        RETURN
        END FUNCTION ROOT_SUM_SQUARES

!  End of subroutine LSTR_solve

      END SUBROUTINE LSTR_solve

!-*-*-*-*-*-  L S T R _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE LSTR_terminate( data, control, inform )

!  ..............................................
!  .                                            .
!  .  Deallocate arrays at end of LSTR_solve    .
!  .                                            .
!  ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine LSTR_initialize
!   inform    see Subroutine LSTR_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LSTR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LSTR_control_type ), INTENT( IN ) :: control
      TYPE ( LSTR_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'lstr: W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: V'
      CALL SPACE_dealloc_array( data%H,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: F'
      CALL SPACE_dealloc_array( data%F,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: LAMBDA'
      CALL SPACE_dealloc_array( data%LAMBDA,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: B_diag'
      CALL SPACE_dealloc_array( data%B_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: B_offdiag'
      CALL SPACE_dealloc_array( data%B_offdiag,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: DECREASE'
      CALL SPACE_dealloc_array( data%DECREASE,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: R_diag'
      CALL SPACE_dealloc_array( data%R_offdiag,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: R_offdiag'
      CALL SPACE_dealloc_array( data%R_offdiag,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: U_extra'
      CALL SPACE_dealloc_array( data%U_extra,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lstr: V_extra'
      CALL SPACE_dealloc_array( data%V_extra,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine LSTR_terminate

      END SUBROUTINE LSTR_terminate

!-*-*-*-  L S T R _ S O L V E _ B I D I A G O N A L   S U B R O U T I N E  *-*-*-

      SUBROUTINE LSTR_solve_bidiagonal( n, B_diag, B_offdiag, beta, radius,    &
                                        lambda, Y, H, R_diag, R_offdiag, F, G, &
                                        error_tol, try_warm, debug, out,       &
                                        prefix, itmax, iter,  status )

! ---------------------------------------------------------------------------
!
!  Solve min || B y - beta e_1 || subject to || y || = radius,
!  where
!           (  diag_1                              )
!           ( offdiag_1  diag_2                    )
!     B   = (              .       .               )  is (n+1) by n
!           (                   offdiag_n  diag_n  )
!           (                            offdiag_n )
!
!  The optimality conditions are that y(lambda) solves the weighted problem
!
!     min || B y( lambda ) - beta e_1 ||^2 + lambda || y(lambda) ||^2
!
!  or equivalently
!
!     min || (        B       ) y( lambda ) - beta e_1 ||^2            (*)
!         || ( srqt(lambda) I )                        ||
!
!  and where
!
!              || y(lambda) || = radius                               (**)
!
!  So pick lambda to define y(lambda) via (*) and adjust it to enforce
!  the scalar equation (*) using Newton's method. This requires that we
!  can solve (*), which we do by reducing
!
!      (        B         beta e_1 ) -> ( R  f )
!      ( srqt(lambda) I      0     )    ( 0  g )
!
!  to n by n upper bi-diagonal form by pre-multiplying by plane rotations.
!  In this case R y(lambda) = f
!
! ---------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, itmax, out
      INTEGER, INTENT( OUT ) :: status, iter
      LOGICAL, INTENT( IN ) :: debug, try_warm
      REAL ( KIND = wp ), INTENT( IN ) :: radius, beta, error_tol
      REAL ( KIND = wp ), INTENT( INOUT ) :: lambda
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: B_diag, B_offdiag
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n - 1 ) :: R_offdiag
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: R_diag, Y, H, F
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 0 : n ) :: G
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: it1
      REAL ( KIND = wp ) :: y_norm, h_norm, error, error_old, delta_lambda

      IF ( debug ) WRITE( out, "( /, A, '   Bi-diagonal subproblem - ',        &
     &     /, A, '   Iter    ||y|| - radius        lambda       ||y||' )" )    &
         prefix, prefix

!  Choose starting lambda ... a warm start

      IF ( try_warm .AND. lambda >= zero ) THEN
        it1 = 2

!  Transform the bi-diagonal subproblem to upper-traingular form R(lambda)
!  for the current lambda

        CALL LSTR_transform_bidiagonal( n, B_diag, B_offdiag, beta,            &
                                        SQRT( lambda ), R_diag, R_offdiag, F, G)

!  Compute the solution y(lambda) to the bi-diagonal subproblem

        CALL LSTR_backsolve_bidiagonal( n, R_diag, R_offdiag, F, Y, .FALSE. )
        y_norm = TWO_NORM( Y )
        error = y_norm - radius
        IF ( debug ) WRITE( out, "( A, I7, ES22.14, 2ES12.4 )" )               &
          prefix, 1, error, lambda, y_norm

!  Test for convergence

        IF ( ABS( error ) < error_tol ) THEN
          iter = 1
          status = 0
          RETURN
        END IF
        error_old = error

!  Is this a suitable starting point?

        IF ( error > zero ) THEN

!  Compute the solution R^T(lambda) v = y

          CALL LSTR_backsolve_bidiagonal( n, R_diag, R_offdiag, Y, H, .TRUE. )
          h_norm = TWO_NORM( H )

!  Compute the Newton update

          lambda = lambda                                                      &
             + ( ( y_norm - radius ) / radius ) *  ( y_norm / h_norm ) ** 2

!  The warm start failed - revert to a cautious starting guess

        ELSE
          lambda = lambda_start
        END IF

!  Choose starting lambda ... a cold start

      ELSE
        it1 = 1
        lambda = lambda_start
        error_old = one
      END IF

!  Newton iteration

      DO iter = it1, itmax

!  Transform the bi-diagonal subproblem to upper-traingular form R(lambda)
!  for the current lambda

        CALL LSTR_transform_bidiagonal( n, B_diag, B_offdiag, beta,            &
                                        SQRT( lambda ), R_diag, R_offdiag, F, G)

!  Compute the solution y(lambda) to the bi-diagonal subproblem

        CALL LSTR_backsolve_bidiagonal( n, R_diag, R_offdiag, F, Y, .FALSE. )
        y_norm = TWO_NORM( Y )
        error = y_norm - radius
        IF ( debug ) WRITE( out, "( A, I7, ES22.14, 2ES12.4 )" )               &
          prefix, iter, error, lambda, y_norm

!  Test for convergence

        IF ( ABS( error ) < error_tol ) THEN
          status = 0
          RETURN
        END IF

!  Try to trap stalling when the error is already small

        IF ( ABS( error ) < error_tol_relax .AND. ( error < zero .OR.          &
          error >= error_old ) ) THEN
          status = 0
          RETURN
        END IF
        error_old = error

!  Compute the solution R^T(lambda) v = y

        CALL LSTR_backsolve_bidiagonal( n, R_diag, R_offdiag, Y, H, .TRUE. )
        h_norm = TWO_NORM( H )

!  Compute the Newton update

        delta_lambda =                                                         &
          ( ( y_norm - radius ) / radius ) *  ( y_norm / h_norm ) ** 2

!  Ensure the update is non-negligible

        IF ( lambda > zero ) THEN
          IF ( ABS( delta_lambda / lambda ) < hundred * epsmch  ) THEN
            status = 0
            RETURN
          END IF
        END IF

        lambda = lambda + delta_lambda
      END DO

      status = 1
      RETURN

!  End of subroutine LSTR_solve_bidiagonal

      END SUBROUTINE LSTR_solve_bidiagonal

!-*  L S T R _ T R A N S F O R M _ B I D I A G O N A L   S U B R O U T I N E  *-

      SUBROUTINE LSTR_transform_bidiagonal( n, B_diag, B_offdiag, beta,        &
                                            omega, R_diag, R_offdiag, F, G )

! ---------------------------------------------------------------------------
!
!  Let
!
!         (   B_diag_1                               )
!         ( B_offdiag_1  B_diag_2                    )
!     B = (                  .        .              ).
!         (                   B_offdiag_n  B_diag_n  )
!         (                              B_offdiag_n )
!
!  Reduce
!
!      (    B      beta e_1 ) -> ( R  f )
!      ( omega I      0     )    ( 0  g )
!
!  to n by n upper bi-diagonal form, where
!
!         ( R_diag_1  R_offdiag_2                   )
!     R = (            .        .                   )
!         (                 R_diag_n-1  R_offdiag_n )
!         (                               R_diag_n  )
!
!
!  by pre-multiplying by plane rotations
!
! ---------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: beta, omega
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: B_diag, B_offdiag
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n - 1 ) :: R_offdiag
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: R_diag, F
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 0 : n ) :: G

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: k
      REAL ( KIND = wp ) :: a, b, c, s
      REAL ( KIND = wp ) :: phi_bar, phi_tilde, r_diag_bar, r_diag_tilde

!  Initialization

      r_diag_bar = B_diag( 1 )
      phi_bar = beta

!  Apply the 2n plane rotations

      DO k = 1, n

!  Columns k and (n+1) contain entries

!       ( r_diag_bar  phi_bar )
!       (      b        0     )
!       (    omega      0     )

!  in rows k, (k+1) and (n+1), where b = B_offdiag_k. Proceed as in
!  Paige and Saunders, ACM TOMS, Vol 8, No 2 (1982), pp 195-209, Section 2

        b = B_offdiag( k )

!  Construct a plane rotation to eliminate omega from ( r_diag_bar omega )
!  to create ( r_diag_tilde 0 )

        a = omega
        CALL ROTG( r_diag_bar, a, c, s )
        r_diag_tilde = r_diag_bar

!  Apply the rotation to ( phi_bar 0 ) to create ( phi_tilde  g )

        phi_tilde = c * phi_bar
        G( k ) = s * phi_bar

!  Construct a plane rotation to eliminate b from ( r_diag_tilde b )
!  to create ( r_diag 0 )

        CALL ROTG( r_diag_tilde, b, c, s )
        R_diag( k ) = r_diag_tilde

!  Apply the rotation to ( phi_tilde 0 ) to create ( f_k phi_bar )

        F( k ) = c * phi_tilde
        IF ( k < n ) THEN
          phi_bar = s * phi_tilde

!  Apply the rotation Q to the forthcoming column ( 0 a ) where a = B_diag_k+1
!  in B to create ( r_offdiag r_diag_bar )

          a = B_diag( k + 1 )
          R_offdiag( k )  = s * a
          r_diag_bar = - c * a
        ELSE
          G( 0 ) = s * phi_tilde
        END IF

      END DO

      RETURN

!  End of subroutine LSTR_transform_bidiagonal

      END SUBROUTINE LSTR_transform_bidiagonal

!-*  L S T R _ B A C K S O L V E _ B I D I A G O N A L   S U B R O U T I N E  *-

      SUBROUTINE LSTR_backsolve_bidiagonal( n, R_diag, R_offdiag, F, Y, trans )

! -----------------------------------------------------------------------------
!
!  Solve the upper bi-diagonal system R y = f or the transpose system R^T y = f
!
! -----------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n - 1 ) :: R_offdiag
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: R_diag, F
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: Y
      LOGICAL, INTENT( IN ) :: trans

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i

!  Forward solve

      IF ( trans ) THEN
        Y( 1 ) = F( 1 ) / R_diag( 1 )
        DO i = 2, n
          Y( i ) = ( F( i ) - R_offdiag( i - 1 ) * Y( i - 1 ) ) / R_diag( i )
        END DO

!  Back solve

      ELSE
        Y( n ) = F( n ) / R_diag( n )
        DO i = n - 1, 1, - 1
          Y( i ) = ( F( i ) - R_offdiag( i ) * Y( i + 1 ) ) / R_diag( i )
        END DO
      END IF
      RETURN

!  End of subroutine LSTR_backsolve_bidiagonal

      END SUBROUTINE LSTR_backsolve_bidiagonal

!-*-*-*-*-*-  End of G A L A H A D _ L S T R  double  M O D U L E  *-*-*-*-*-*-

   END MODULE GALAHAD_LSTR_double

