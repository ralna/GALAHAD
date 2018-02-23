! THIS VERSION: GALAHAD 3.0 - 19/02/2018 AT 15:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ G L R T  M O D U L E  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   created from GALAHAD package GLTR, October 27th, 2007

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_GLRT_double

!      ---------------------------------------------------------------------
!      |                                                                   |
!      | Solve the regularised quadratic minimization problem              |
!      |                                                                   |
!      |    minimize     1/p sigma ( ||x + o||_M^2 + eps )^(p/2)           |
!      |                   + 1/2 <x, H x> + <c, x> + f0                    |
!      |                                                                   |
!      ! where M is symmetric, positive definite and p (>=2), eps (>=0)    |
!      | and sigma (>0) are constants using a generalized Lanczos method   |
!      |                                                                   |
!      ---------------------------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_RAND_double
      USE GALAHAD_ROOTS_double, ONLY : ROOTS_quadratic
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_LAPACK_interface, ONLY : PTTRF
      USE GALAHAD_GLTR_double, ONLY :                                          &
        GLRT_leftmost_eigenvalue => GLTR_leftmost_eigenvalue,                  &
        GLRT_leftmost_eigenvector => GLTR_leftmost_eigenvector,                &
        GLRT_tridiagonal_solve => GLTR_tridiagonal_solve

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: GLRT_initialize, GLRT_read_specfile, GLRT_solve, GLRT_terminate

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
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: roots_tol = ten ** ( - 12 )
!     REAL ( KIND = wp ), PARAMETER :: roots_tol = ten ** ( - 16 )
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

      TYPE, PUBLIC :: GLRT_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the maximum number of iterations allowed (-ve = no bound)

        INTEGER :: itmax = - 1

!   the stopping rule used (see below): 0=1.0, 1=norm step, 2=norm step/sigma

        INTEGER :: stopping_rule = 1

!   frequency for solving the reduced tri-diagonal problem

        INTEGER :: freq = 1

!   the number of extra work vectors of length n used

        INTEGER :: extra_vectors = 0

!   the iteration stops successfully when the gradient in the M(inverse) norm
!    is smaller than
!      max( stop_relative * min( 1, stop_rule ),
!           stop_absolute ) * norm initial gradient,
!           stop_norm * ||x+o||_M^stop_power )

        REAL ( KIND = wp ) :: stop_relative = epsmch
        REAL ( KIND = wp ) :: stop_absolute = zero
        REAL ( KIND = wp ) :: stop_norm = zero
        REAL ( KIND = wp ) :: stop_power = two

!   an estimate of the solution that gives at least %fraction_opt times
!    the optimal objective value will be found

        REAL ( KIND = wp ) :: fraction_opt = one

!   the smallest value that the square of the M norm of the gradient of the
!    the objective may be before it is considered to be zero

        REAL ( KIND = wp ) :: rminvr_zero = ten * epsmch

!   the constant term, f0, in the objective function

        REAL ( KIND = wp ) :: f_0 = zero

!   is M the identity matrix ?

        LOGICAL :: unitm = .TRUE.

!   is descent required i.e., should <c, x> < 0 ?

        LOGICAL :: impose_descent = .TRUE.

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

      TYPE, PUBLIC :: GLRT_inform_type

!  return status. See GLRT_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the total number of pass-2 iterations required

        INTEGER :: iter_pass2 = - 1

!  the value of the quadratic function

        REAL ( KIND = wp ) :: obj = biginf

!  the value of the regularized quadratic function

        REAL ( KIND = wp ) :: obj_regularized = HUGE( one )

!  the multiplier, sigma ||x||^(p-2)

        REAL ( KIND = wp ) :: multiplier = zero

!  the value of the norm ||x+o||_M

        REAL ( KIND = wp ) :: xpo_norm = zero

!  an estimate of the leftmost generalized eigenvalue of the pencil (H,M)

        REAL ( KIND = wp ) :: leftmost = biginf

!  was negative curvature encountered ?

        LOGICAL :: negative_curvature = .FALSE.

!  did the hard case occur ?

        LOGICAL :: hard_case = .FALSE.
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: GLRT_data_type
        PRIVATE
        INTEGER :: branch = 100
        TYPE ( RAND_seed ) :: seed
        INTEGER :: iter, itm1, itmax, dim_sub, switch, titmax, tinfo, titer
        INTEGER :: freq, extra_vectors, iter_descent
        REAL ( KIND = wp ) :: alpha, beta, rminvr, rminvr_old, curv, mult
        REAL ( KIND = wp ) :: o_mnorm_2_p_eps, tau, xmx, xmp, pmp, onorm2
        REAL ( KIND = wp ) :: stop, piv, x_last, norm_x, hard_case_step
        REAL ( KIND = wp ) :: diag, offdiag, rtol,  pgnorm, pgnorm_zero
        LOGICAL :: printi, printd, use_old, try_warm, solve_tcm, one_pass
        LOGICAL :: save_vectors
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: OFFD
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ALPHAS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RMINVRS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MIN_f
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MIN_f_regularized
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: LAMBDA
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D_fact
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: OFFD_fact
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MinvC
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: O_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_extra
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P_extra
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: V_extra
        CHARACTER ( LEN = 1 ) :: descent
      END TYPE

    CONTAINS

!-*-*-*-*-*-  G L R T _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE GLRT_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  .  Set initial values for the GLRT control parameters  .
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

      TYPE ( GLRT_data_type ), INTENT( INOUT ) :: data
      TYPE ( GLRT_control_type ), INTENT( OUT ) :: control
      TYPE ( GLRT_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  Set initial control parameter values

      control%stop_relative = SQRT( epsmch )

!  Set branch for initial entry

      data%branch = 100

      RETURN

!  End of subroutine GLRT_initialize

      END SUBROUTINE GLRT_initialize

!-*-*-*-*-   G L R T _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE GLRT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by GLRT_initialize could (roughly)
!  have been set as:

!  BEGIN GLRT SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-number-of-iterations                    -1
!   tri-diagonal-solve-frequency                    1
!   number-extra-n-vectors-used                     0
!   stopping-rule                                   1
!   relative-accuracy-required                      1.0E-8
!   absolute-accuracy-required                      0.0
!   fraction-optimality-required                    1.0
!   constant-term-in-objective                      0.0
!   zero-gradient-tolerance                         2.0E-15
!   two-norm-regularisation                         T
!   impose-initial-descent                          F
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END GLRT SPECIFICATIONS

!  Dummy arguments

      TYPE ( GLRT_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), INTENT( IN ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 34
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'GLRT'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec( 18 )%keyword = 'stopping-rule'
      spec(  5 )%keyword = 'number-extra-n-vectors-used'
      spec( 19 )%keyword = 'tri-diagonal-solve-frequency'

!  Real key-words

      spec(  6 )%keyword = 'relative-accuracy-required'
      spec(  7 )%keyword = 'absolute-accuracy-required'
      spec( 20 )%keyword = 'norm-accuracy-required'
      spec( 21 )%keyword = 'norm-power-used'
      spec(  8 )%keyword = 'fraction-optimality-required'
      spec(  9 )%keyword = 'constant-term-in-objective'
      spec( 17 )%keyword = 'zero-gradient-tolerance'

!  Logical key-words

      spec( 10 )%keyword = 'two-norm-regularisation'
      spec( 11 )%keyword = 'impose-initial-descent'
      spec( 12 )%keyword = ''
      spec( 13 )%keyword = ''
      spec( 14 )%keyword = 'space-critical'
      spec( 15 )%keyword = 'deallocate-error-fatal'

!  Character key-words

!     spec( 16 )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( 1 ), control%error,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 2 ), control%out,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 3 ), control%print_level,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 4 ), control%itmax,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 18 ), control%stopping_rule,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 5 ), control%extra_vectors,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 19 ), control%freq,                    &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( 6 ), control%stop_relative,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 7 ), control%stop_absolute,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 20 ), control%stop_norm,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 21 ), control%stop_power,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 8 ), control%fraction_opt,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 9 ), control%f_0,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 17 ), control%rminvr_zero,             &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( 10 ), control%unitm,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 11 ), control%impose_descent,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 14 ), control%space_critical,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 15 ),                                  &
                                  control%deallocate_error_fatal,              &
                                  control%error )
!  Set charcter values

      CALL SPECFILE_assign_value( spec( 16 ), control%prefix,                  &
                                  control%error )

      RETURN

      END SUBROUTINE GLRT_read_specfile

!-*-*-*-*-*-*-*-*-*-*  G L R T _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE GLRT_solve( n, p, sigma, X, R, VECTOR, data, control,         &
                             inform, eps, O )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   n        number of unknowns
!   p        regularisation order
!   sigma    regularisation parameter
!   X        the vector of unknowns. Need not be set on entry.
!            On exit, the best value found so far
!   R        the residual vector H x + c. On entry this must contain c
!   VECTOR   see inform%status = 2, 3 and 5
!   data     private internal data
!   control  a structure containing control information. See GLRT_initialize
!   inform   a structure containing information. The component
!             %status is the input/output status. This must be set to 1 on
!              initial entry or 4 on a re-entry when only sigma has
!              been reduced since the last entry. Other values are
!               2 on exit => the inverse of M must be applied to
!                 VECTOR with the result returned in VECTOR and the subroutine
!                 re-entered. This will only happen if unitm is .FALSE.
!               3 on exit => the product H * VECTOR must be formed, with
!                 the result returned in VECTOR and the subroutine re-entered
!               4 The iteration will be restarted. Reset R to c and re-enter.
!               5 on exit => the matrix M must be applied to VECTOR with the
!                 result returned in VECTOR and the subroutine re-entered. This
!                 will only happen if unitm is .FALSE. and O is present.
!               0 the solution has been found
!              -1 an array allocation has failed
!              -2 an array deallocation has failed
!              -3 n <= 0 and/or sigma < 0 and/or eps < 0 and/or p < 2
!              -7 the objective is unbounded from below
!              -15 the matrix M appears to be indefinite
!              -18 the iteration limit has been exceeded
!              -25 status is not > 0 on entry
!             the remaining components are described in the preamble
!   eps      optional shift parameter
!   O        optional offset vector o
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: p, sigma
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, R, VECTOR
      TYPE ( GLRT_data_type ), INTENT( INOUT ) :: data
      TYPE ( GLRT_control_type ), INTENT( IN ) :: control
      TYPE ( GLRT_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: eps
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: O

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: dim_sub, it, itp1
      REAL ( KIND = wp ) :: f_tol, f_0, u_norm
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Branch to different sections of the code depending on input status

      IF ( inform%status < 1 ) GO TO 930
      IF ( inform%status == 1 ) THEN
        data%branch = 100
      ELSE IF ( inform%status == 6 ) THEN
        data%branch = 400
      END IF

      SELECT CASE ( data%branch )
      CASE ( 100 )
        GO TO 100
      CASE ( 140 )
        GO TO 140
      CASE ( 150 )
        GO TO 150
      CASE ( 200 )
        GO TO 200
      CASE ( 300 )
        GO TO 300
      CASE ( 400 )
        GO TO 400
      CASE ( 500 )
        GO TO 500
      CASE ( 550 )
        GO TO 550
      CASE ( 600 )
        GO TO 600
      CASE ( 700 )
        GO TO 700
      CASE ( 900 )
        GO TO 900
      END SELECT

!  On initial entry, set constants

  100 CONTINUE

!  Check for obvious errors

      IF ( n <= 0 ) GO TO 940
      IF ( sigma < zero ) GO TO 950
      IF ( PRESENT( eps ) ) THEN
        IF ( eps < zero ) GO TO 970
      END IF
      IF ( p < two ) GO TO 980
      data%one_pass = sigma == zero .OR. ( p == two .AND. .NOT. PRESENT( O ) )
!     data%one_pass = .FALSE.

      data%iter = 0 ; data%iter_descent = 0 ; data%descent = ' '
      data%itmax = control%itmax ; IF ( data%itmax < 0 ) data%itmax = n
      data%printi = control%out > 0 .AND. control%print_level >= 1
      data%printd = control%out > 0 .AND. control%print_level >= 2
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter_pass2 = 0 ; inform%negative_curvature = .FALSE.
      inform%multiplier = zero ; data%tau = one
      inform%hard_case = .FALSE.
      data%switch = 0
      data%use_old = .FALSE. ; data%try_warm = .FALSE.
      data%rtol = roots_tol
      data%freq = MAX( 1, control%freq )
      IF ( PRESENT( O ) ) data%onorm2 = zero

!  If necessary, find M^-1 c

      IF ( PRESENT( O ) ) THEN
        array_name = 'glrt: MinvC'
        CALL SPACE_resize_array( n, data%MinvC,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960
        VECTOR = R
        IF ( .NOT. control%unitm ) THEN
          data%branch = 140 ; inform%status = 2 ; inform%iter = data%iter
          RETURN
        END IF
      END IF

!  If necessary, find M * o

  140 CONTINUE
      IF ( PRESENT( O ) ) THEN
        data%MinvC( : n ) = VECTOR
        VECTOR = O
        IF ( .NOT. control%unitm ) THEN
          data%branch = 150 ; inform%status = 5 ; inform%iter = data%iter
          RETURN
        END IF
      END IF

!  If o is present, modify the initial gradient

  150 CONTINUE
      IF ( PRESENT( O ) ) THEN
        IF ( PRESENT( eps ) ) THEN
          data%o_mnorm_2_p_eps = DOT_PRODUCT( O, VECTOR ) + eps
        ELSE
          data%o_mnorm_2_p_eps = DOT_PRODUCT( O, VECTOR )
        END IF
        data%mult = sigma * ( data%o_mnorm_2_p_eps ** ( p / two  - one ) )
        data%rminvr = DOT_PRODUCT(  data%MinvC( : n ), R ) +                   &
                        two * data%mult * DOT_PRODUCT(  O, R ) +               &
                        data%mult * data%mult * DOT_PRODUCT(  O, VECTOR )
        R = R + data%mult * VECTOR
      ELSE
        IF ( PRESENT( eps ) ) THEN
          data%o_mnorm_2_p_eps = eps
        ELSE
          data%o_mnorm_2_p_eps = zero
        END IF
      END IF

!  Start from the origin

      X = zero ; data%norm_x = zero ; data%x_last = zero
      inform%obj = control%f_0

!  =====================
!  Array (re)allocations
!  =====================

!  Allocate P and U

      array_name = 'glrt: P'
      CALL SPACE_resize_array( n, data%P,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      array_name = 'glrt: U'
      CALL SPACE_resize_array( n, data%U,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 960

      IF ( .NOT. data%one_pass ) THEN
        data%titmax = 100

!  Allocate space for the Lanczos tridiagonal

        array_name = 'glrt: D'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%D,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'glrt: OFFD'
        CALL SPACE_resize_array( data%itmax + 1, data%OFFD,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate space for the factors of the Lanczos tridiagonal

        array_name = 'glrt: D_fact'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%D_fact,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'glrt: OFFD_fact'
        CALL SPACE_resize_array( data%itmax + 1, data%OFFD_fact,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate space for the RHS and solution for the Lanczos subproblem

        array_name = 'glrt: C_sub'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%C_sub,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960
        data%C_sub( 0 ) = one ; data%C_sub( 1 : data%itmax + 1 ) = zero

        array_name = 'glrt: X_sub'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%X_sub,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        IF ( PRESENT( O ) ) THEN
          array_name = 'glrt: O_sub'
          CALL SPACE_resize_array( 0, data%itmax + 1, data%O_sub,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 960
        END IF

!  Allocate space for the history of Lagrange multipliers

        array_name = 'glrt: LAMBDA'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%LAMBDA,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960
        data%LAMBDA( 0 ) = zero

!  Allocate space for workspace associated with the Lanczos subproblem

        array_name = 'glrt: U_sub'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%U_sub,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'glrt: W'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%W,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate space to store the alphas and rminvr for more efficient processing
!  in the second pass

        array_name = 'glrt: ALPHAS'
        CALL SPACE_resize_array( data%itmax + 1, data%ALPHAS,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'glrt: RMINVRS'
        CALL SPACE_resize_array( data%itmax + 1, data%RMINVRS,                 &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  Allocate workspace for the sequence of smallest function values

        array_name = 'glrt: MIN_f'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%MIN_f,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

        array_name = 'glrt: MIN_f_regularized'
        CALL SPACE_resize_array( 0, data%itmax + 1, data%MIN_f_regularized,    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 960

!  If required allocate extra space to store R, P and V in case of a second pass

        data%extra_vectors = control%extra_vectors - 2
        IF ( data%extra_vectors > 0 ) THEN
          array_name = 'glrt: R_extra'
          CALL SPACE_resize_array( n, data%R_extra,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status == 0 ) THEN
            array_name = 'glrt: P_extra'
            CALL SPACE_resize_array( n, data%P_extra,                          &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status == 0 ) THEN
              array_name = 'glrt: V_extra'
              CALL SPACE_resize_array( n, data%extra_vectors, data%V_extra,    &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= 0 ) THEN
                array_name = 'glrt: R_extra'
                CALL SPACE_dealloc_array( data%R_extra,                        &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  bad_alloc = inform%bad_alloc, out = control%error )
                IF ( inform%status /= 0 ) GO TO 960
                array_name = 'glrt: P_extra'
                CALL SPACE_dealloc_array( data%P_extra,                        &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  bad_alloc = inform%bad_alloc, out = control%error )
                IF ( inform%status /= 0 ) GO TO 960
                data%extra_vectors = 0
              END IF
            ELSE
              array_name = 'glrt: R_extra'
              CALL SPACE_dealloc_array( data%R_extra,                          &
                inform%status, inform%alloc_status, array_name = array_name,   &
                bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= 0 ) GO TO 960
              data%extra_vectors = 0
            END IF
          ELSE
            data%extra_vectors = 0
          END IF
        END IF

        inform%obj_regularized = control%f_0 +                                 &
          ( sigma / p ) * ( data%o_mnorm_2_p_eps ** ( p / two ) )

!  Special case for the case p = 2

      ELSE

!  Allocate space for the preconditioned search direction

        IF ( sigma > zero ) THEN
          array_name = 'glrt: W'
          CALL SPACE_resize_array( n, data%W,                                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 960
        END IF
        data%extra_vectors = 0
        IF ( PRESENT( eps ) ) THEN
          inform%obj_regularized = inform%obj + half * sigma * eps
        ELSE
          inform%obj_regularized = inform%obj
        END IF
        data%xmx = zero ; inform%xpo_norm = zero
      END IF
      data%save_vectors = data%extra_vectors > 0

!  ===========================
!  Start of the main iteration
!  ===========================

  190 CONTINUE
      inform%iter = data%iter

!  ----------------------------------
!  Obtain the preconditioned residual
!  ----------------------------------

      IF ( .NOT. ( PRESENT( O ) .AND. data%iter == 0 ) ) THEN
        VECTOR = R
        IF ( .NOT. control%unitm ) THEN
          data%branch = 200 ; inform%status = 2
          RETURN
        END IF
      END IF

!  Obtain the scaled norm of the residual

  200 CONTINUE
      IF ( .NOT. ( PRESENT( O ) .AND. data%iter == 0 ) ) THEN
        data%rminvr = DOT_PRODUCT( R, VECTOR )
      ELSE
        VECTOR = data%MinvC( : n ) + data%mult * O
      END IF
      IF ( ABS( data%rminvr ) < control%rminvr_zero ) data%rminvr = zero
      IF ( data%rminvr < zero ) THEN
        IF ( MAXVAL( ABS( VECTOR ) ) < epsmch * MAXVAL( ABS( R ) ) ) THEN
          data%rminvr = zero
        ELSE
          GO TO 1000
        END IF
      END IF
      data%pgnorm = SIGN( SQRT( ABS( data%rminvr ) ), data%rminvr )

!  general case

      IF ( .NOT. data%one_pass ) THEN

!  If the user has asked to save vectors, save VECTOR, R and P

        IF ( data%save_vectors ) THEN
          IF ( inform%iter < data%extra_vectors )                              &
            data%V_extra( : n , inform%iter + 1 ) = VECTOR
          IF ( inform%iter == data%extra_vectors ) THEN
            data%R_extra( : n ) = R
            data%P_extra( : n ) = data%P( : n )
          END IF
        END IF

!  Compute the diagonal and off-diagonal entries of the Lanczos tridiagonal

        IF ( data%iter > 0 ) THEN
          data%beta = data%rminvr / data%rminvr_old
          data%diag = data%beta / data%alpha
          data%offdiag = SQRT( data%beta ) / ABS( data%alpha )

!  Compute the stopping tolerance

        ELSE
          data%diag = zero
          data%stop = MAX( control%stop_relative * data%pgnorm,                &
                           control%stop_absolute )
          data%pgnorm_zero = data%pgnorm
          IF ( data%printi )                                                   &
            WRITE( control%out, "( /, A, ' stopping tolerance = ',             &
           &         ES12.4, ', sigma = ', ES12.4 )" ) prefix, data%stop, sigma
          data%C_sub( 0 ) = data%pgnorm
        END IF

!  Print details of the latest iteration

!if ( data%itm1 >= 1 ) then
!  if ( data%LAMBDA( data%itm1 - 1 ) > &
!       data%LAMBDA( data%itm1 ) + sigma * roots_tol * inform%xpo_norm ) then
!    write(6,"(3ES24.16)" ) data%LAMBDA( data%itm1 - 1 ), &
!      data%LAMBDA( data%itm1), &
!      data%LAMBDA( data%itm1 - 1 ) - data%LAMBDA( data%itm1)
!!stop
!end if
!end if
        IF ( data%printi ) THEN
          IF ( MOD( data%iter, 25 ) == 0 .OR. data%printd )                    &
            WRITE( control%out, 2000 ) prefix
          IF ( data%iter /= 0 ) THEN
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, ES15.8, 2I5, A )" )   &
              prefix, data%iter, data%MIN_f_regularized( data%itm1 )           &
              + control%f_0, ABS( data%x_last * data%offdiag ),                &
              data%LAMBDA( data%itm1 ), data%titer, data%tinfo, data%descent
!write(6,"(2ES12.4)" ) data%x_last, data%offdiag
          ELSE
            WRITE( control%out, "( A, I7, ES16.8, ES9.2, 8X, '-', 6X,          &
             &      2( 4X, '-' ) )" )                                          &
               prefix, data%iter, inform%obj_regularized, data%pgnorm
          END IF
        END IF

!  Obtain the search direction P

        IF ( data%iter > 0 ) THEN
          data%pmp = data%rminvr + data%pmp * data%beta * data%beta
          data%P( : n ) = - VECTOR + data%beta * data%P( : n )

!  Continue accumulating the Lanczos tridiagonal

          data%D( data%iter ) = data%diag
          data%OFFD( data%iter ) = data%offdiag

!  Optionally refine the stopping tolereance

          IF ( control%stopping_rule == 1 ) THEN
            data%stop = MAX( control%stop_relative * MIN( one, data%norm_x )   &
                          * data%pgnorm_zero , control%stop_absolute )
          ELSE IF ( control%stopping_rule == 2 ) THEN
            data%stop = MAX( control%stop_relative *                           &
                          MIN( one, data%norm_x / MAX( one, sigma ) ) *        &
                          data%pgnorm_zero , control%stop_absolute )
          END IF

!  Check for convergence

          IF ( data%iter >= data%itmax .OR.                                    &
               ABS( data%offdiag * data%x_last ) <= data%stop ) THEN

!  Convergence has occured. Determine at which iteration a fraction,
!  fraction_opt, of the optimal solution was found

            f_0 = ( sigma / p ) * ( data%o_mnorm_2_p_eps ** ( p / two ) )
            IF ( control%fraction_opt < one ) THEN
              f_tol = ( f_0 - data%MIN_f_regularized( data%itm1 ) )           &
                          * control%fraction_opt
              IF ( control%impose_descent ) THEN
                it = data%iter_descent
              ELSE
                it = data%iter
              END IF
              DO dim_sub = 1, it
                 IF ( f_0 - data%MIN_f_regularized( dim_sub - 1 ) >= f_tol )   &
                   EXIT
              END DO
            ELSE
              IF ( control%impose_descent ) THEN
                dim_sub = data%iter_descent
              ELSE
                dim_sub = data%iter
              END IF
            END IF
            data%dim_sub = dim_sub

!           IF ( data%printi ) WRITE( control%out, 2020 )                      &
!           WRITE( control%out, 2020 ) data%MIN_f_regularized( dim_sub - 1 ),  &
!             dim_sub, iter

!  Restore the solution to the Lanczos subproblem for this iteration

            IF ( dim_sub < data%iter ) THEN
              data%use_old = .FALSE.
              IF ( dim_sub > 1 + data%switch ) THEN
                data%LAMBDA( dim_sub - 1 ) = data%LAMBDA( dim_sub - 2 )
              ELSE
                data%LAMBDA( dim_sub - 1 ) = zero
                data%try_warm = .FALSE.
              END IF

              IF ( PRESENT( O ) ) THEN
                data%onorm2 = DOT_PRODUCT( data%O_sub( : dim_sub - 1 ),        &
                                           data%O_sub( : dim_sub - 1 ) )
                CALL GLRT_trts( dim_sub, data%D( : dim_sub - 1 ),              &
                           data%OFFD( : dim_sub - 1 ),                         &
                           data%D_fact( : dim_sub - 1 ),                       &
                           data%OFFD_fact( : dim_sub - 1 ),                    &
                           data%C_sub( : dim_sub - 1 ), p, sigma, data%rtol,   &
                           data%titmax, data%try_warm, data%use_old,           &
                           inform%leftmost, data%LAMBDA( dim_sub - 1 ),        &
                           data%MIN_f( dim_sub - 1 ),                          &
                           data%MIN_f_regularized( dim_sub - 1 ),              &
                           data%X_sub( : dim_sub - 1 ),                        &
                           inform%xpo_norm, data%tinfo, data%titer,            &
                           data%U_sub( : dim_sub - 1 ),                        &
                           data%W( : dim_sub - 1 ),                            &
                           data%seed, control%print_level - 1, control%out,    &
                           prefix, inform%hard_case, data%hard_case_step,      &
                           eps = eps, O = data%O_sub( : dim_sub - 1 ), &
                           onorm2 = data%onorm2 )
              ELSE
                CALL GLRT_trts( dim_sub, data%D( : dim_sub - 1 ),              &
                           data%OFFD( : dim_sub - 1 ),                         &
                           data%D_fact( : dim_sub - 1 ),                       &
                           data%OFFD_fact( : dim_sub - 1 ),                    &
                           data%C_sub( : dim_sub - 1 ), p, sigma, data%rtol,   &
                           data%titmax, data%try_warm, data%use_old,           &
                           inform%leftmost, data%LAMBDA( dim_sub - 1 ),        &
                           data%MIN_f( dim_sub - 1 ),                          &
                           data%MIN_f_regularized( dim_sub - 1 ),              &
                           data%X_sub( : dim_sub - 1 ),                        &
                           inform%xpo_norm, data%tinfo, data%titer,            &
                           data%U_sub( : dim_sub - 1 ),                        &
                           data%W( : dim_sub - 1 ),                            &
                           data%seed, control%print_level - 1, control%out,    &
                           prefix, inform%hard_case, data%hard_case_step,      &
                           eps = eps )
              END IF
            END IF
!           IF ( data%printi ) WRITE( control%out, 2020 )                      &
!                data%MIN_f_regularized( dim_sub - 1 ), dim_sub, data%iter

!  Record the optimal objective function value and prepare to recover the
!  approximate solution

            inform%obj = data%MIN_f( dim_sub - 1 ) + control%f_0
            inform%obj_regularized                                             &
              = data%MIN_f_regularized( dim_sub - 1 ) + control%f_0
            inform%multiplier = data%LAMBDA( dim_sub - 1 )
            data%tau = one
            inform%iter = data%iter ; data%iter = 0
            IF ( data%save_vectors ) GO TO 390
            data%branch = 500 ; inform%status = 4
            RETURN
          END IF
        ELSE

!  Special case for the first iteration

          data%P( : n ) = - VECTOR
          data%pmp = data%rminvr
          data%D( 0 ) = data%diag
        END IF

        IF ( PRESENT( O ) ) THEN
          data%O_sub( data%iter ) =                                            &
            data%tau * DOT_PRODUCT( R, O ) / SQRT( data%rminvr )
          data%onorm2 = data%onorm2 + data%O_sub( data%iter ) ** 2
          data%C_sub( data%iter ) = data%tau *                                 &
            DOT_PRODUCT( R, data%MinvC( : n ) ) / SQRT( data%rminvr )
        END IF

        data%RMINVRS( data%iter + 1 ) = data%rminvr
        data%rminvr_old = data%rminvr

        data%itm1 = data%iter ; data%iter = data%iter + 1
        data%solve_tcm = data%freq == 1 .OR.                                   &
          MOD( data%iter, data%freq ) == 1 .OR. data%iter == data%itmax
!         .OR.  ( control%desent .AND. data%iter == 0 )

!  p = 2 case

      ELSE
        IF ( data%iter > 0 ) THEN
          data%beta = data%rminvr / data%rminvr_old
        ELSE

!  Compute the stopping tolerance

          data%stop = MAX( control%stop_relative * data%pgnorm,                &
                           control%stop_absolute )
          data%pgnorm_zero = data%pgnorm
          IF ( data%printi )                                                   &
            WRITE( control%out, "( /, A, ' stopping tolerance = ',             &
           &         ES12.4, ', sigma = ', ES12.4 )" ) prefix, data%stop, sigma
        END IF

!  Print details of the latest iteration

        IF ( data%printi ) THEN
          IF ( MOD( data%iter, 25 ) == 0 .OR. data%printd )                    &
            WRITE( control%out, 2010 ) prefix
          WRITE( control%out, "( A, I7, ES16.8, ES9.2 )" )                     &
            prefix, data%iter, inform%obj_regularized, data%pgnorm
        END IF

        IF ( data%iter > 0 ) THEN

!  Obtain the search direction P and its scaled version W = M * P

          IF ( sigma > zero ) data%W( : n ) = - R + data%beta * data%W( : n )
          data%xmp = data%beta * ( data%xmp + data%alpha * data%pmp )
          data%pmp = data%rminvr + data%pmp * data%beta * data%beta
          data%P( : n ) = - VECTOR + data%beta * data%P( : n )

!  Optionally refine the stopping tolereance

          IF ( control%stopping_rule == 1 ) THEN
            data%stop = MAX( control%stop_relative * MIN( one, data%norm_x )   &
                           * data%pgnorm_zero , control%stop_absolute )
          ELSE IF ( control%stopping_rule == 2 ) THEN
            data%stop = MAX( control%stop_relative *                           &
                             MIN( one, data%norm_x / MAX( one, sigma ) ) *     &
                             data%pgnorm_zero , control%stop_absolute )
          END IF

!  Check for convergence

          IF ( data%iter >= data%itmax .OR. data%rminvr <= data%stop ) THEN
            inform%multiplier = sigma
            GO TO 900
          END IF
        ELSE

!  Special case for the first iteration

          IF ( sigma > zero ) data%W( : n ) = - R
          data%P( : n ) = - VECTOR
          data%pmp = data%rminvr ; data%xmp = zero
        END IF
        data%rminvr_old = data%rminvr

        data%itm1 = data%iter ; data%iter = data%iter + 1
      END IF

!  ------------------------------
!  Obtain the product of H with p
!  ------------------------------

      VECTOR = data%P( : n )
      data%branch = 300  ; inform%status = 3 ; inform%iter = data%iter
      RETURN

!  Obtain the curvature

  300 CONTINUE
      data%curv = DOT_PRODUCT( VECTOR, data%P( : n ) )
      IF ( data%curv <= zero ) inform%negative_curvature = .TRUE.
!write(6,*) ' ========= curv ', data%curv
!  General case

      IF ( .NOT. data%one_pass ) THEN

!  Obtain the stepsize and the new diagonal of the Lanczos tridiagonal

        IF ( data%curv /= zero ) THEN
          data%alpha = data%rminvr / data%curv
          data%diag = data%diag + one / data%alpha
        ELSE
          data%alpha = HUGE( one ) ** 0.25
        END IF

!  Check that the Lanczos tridiagonal is still positive definite

        IF ( .NOT. inform%negative_curvature ) THEN
          IF ( data%iter > 1 ) THEN
            data%piv = data%diag - ( data%offdiag / data%piv ) * data%offdiag
          ELSE
            data%piv = data%diag
          END IF
          inform%negative_curvature = data%piv <= zero
        END IF

!  Complete the new diagonal of the Lanczos tridiagonal matrix

         data%D( data%itm1 ) = data%diag ; data%ALPHAS( data%iter ) = data%alpha

!  Solve the subproblem

        IF ( data%solve_tcm ) THEN
          IF ( PRESENT( O ) ) THEN
            CALL GLRT_trts( data%iter, data%D( : data%itm1 ),                  &
                 data%OFFD( : data%itm1 ), data%D_fact( : data%itm1 ),         &
                 data%OFFD_fact( : data%itm1 ), data%C_sub( : data%itm1 ),     &
                 p, sigma, data%rtol, data%titmax, data%try_warm,              &
                 data%use_old, inform%leftmost, data%LAMBDA( data%itm1 ),      &
                 data%MIN_f( data%itm1 ), data%MIN_f_regularized( data%itm1 ), &
                 data%X_sub( : data%itm1 ), inform%xpo_norm, data%tinfo,       &
                 data%titer, data%U_sub( : data%itm1),                         &
                 data%W( : data%itm1 ), data%seed, control%print_level - 1,    &
                 control%out, prefix, inform%hard_case, data%hard_case_step,   &
                 eps = eps, O = data%O_sub( : data%itm1 ),                     &
                 onorm2 = data%onorm2 )
            IF ( control%impose_descent .AND. DOT_PRODUCT(                     &
              data%X_sub( : data%itm1 ), data%C_sub( : data%itm1 )             &
                + data%mult *  data%O_sub( : data%itm1 ) ) < zero ) THEN
              data%descent = 'd'
              data%iter_descent = data%iter
            ELSE
              data%descent = ' '
            END IF
          ELSE
            CALL GLRT_trts( data%iter, data%D( : data%itm1 ),                  &
               data%OFFD( : data%itm1 ), data%D_fact( : data%itm1 ),           &
               data%OFFD_fact( : data%itm1 ), data%C_sub( : data%itm1 ),       &
               p, sigma, data%rtol, data%titmax, data%try_warm,                &
               data%use_old, inform%leftmost, data%LAMBDA( data%itm1 ),        &
               data%MIN_f( data%itm1 ), data%MIN_f_regularized( data%itm1 ),   &
               data%X_sub( : data%itm1 ), inform%xpo_norm, data%tinfo,         &
               data%titer, data%U_sub( : data%itm1),                           &
               data%W( : data%itm1 ), data%seed, control%print_level - 1,      &
               control%out, prefix, inform%hard_case, data%hard_case_step,     &
               eps = eps )
            IF ( control%impose_descent .AND. data%X_sub( 0 ) < zero ) THEN
              data%descent = 'd'
              data%iter_descent = data%iter
            ELSE
              data%descent = ' '
            END IF
          END IF
          inform%obj = data%MIN_f( data%itm1 ) + control%f_0
          inform%obj_regularized                                               &
            = data%MIN_f_regularized( data%itm1 ) + control%f_0
          data%x_last = data%X_sub( data%itm1 )
          data%norm_x = TWO_NORM( data%X_sub( : data%itm1 ) )
          data%use_old = inform%leftmost < zero
        ELSE
          data%MIN_f( data%itm1 ) = data%MIN_f( data%itm1 - 1 )
          data%MIN_f_regularized( data%itm1 )                                  &
            = data%MIN_f_regularized( data%itm1 - 1 )
          data%tinfo = 0 ; data%titer = 0
        END IF
        data%try_warm = .TRUE.
        data%LAMBDA( data%iter ) = data%LAMBDA( data%itm1 )

!  Update the residual

        R = R + data%alpha * VECTOR
        data%tau = - SIGN( one, data%alpha ) * data%tau

!  p = 2 case

      ELSE

!  Include the curvature from the regularisation

        IF ( sigma > zero ) data%curv = data%curv                              &
               + sigma * DOT_PRODUCT( data%W( : n ), data%P( : n ) )

!  Exit if the curvature is not positive

        IF ( data%curv <= zero ) THEN
          inform%negative_curvature = .TRUE.
          VECTOR = data%P( : n )
          GO TO 990
        END IF

!  Obtain the stepsize

        data%alpha = data%rminvr / data%curv

!  Update the solution and function value

        data%xmx = data%xmx + data%alpha *                                     &
                     ( two * data%xmp + data%alpha * data%pmp )
        inform%xpo_norm = SQRT( data%xmx )
        X = X + data%alpha * data%P( : n )
        inform%obj = inform%obj - half * data%alpha * data%alpha * data%curv
        IF ( PRESENT( eps ) ) THEN
          inform%obj_regularized = inform%obj + half * sigma * eps
        ELSE
          inform%obj_regularized = inform%obj
        END IF

!  Update the residual

        R = R + data%alpha * ( VECTOR + sigma * data%W( : n ) )
        data%tau = - SIGN( one, data%alpha ) * data%tau

      END IF

!  =========================
!  End of the main iteration
!  =========================

      GO TO 190

!  ===================================
!  Use saved vectors to start 2nd pass
!  ===================================

  390 CONTINUE
      inform%iter_pass2 = MIN( data%dim_sub, data%extra_vectors )
      DO it = 0, inform%iter_pass2 - 1
        itp1 = it + 1
        data%X_sub( it ) = data%tau *                                          &
          ( data%X_sub( it ) / SQRT( data%RMINVRS( itp1 ) ) )
        IF ( inform%hard_case )                                                &
          data%U_sub( it ) = data%tau *                                        &
          ( data%U_sub( it ) / SQRT( data%RMINVRS( itp1 ) ) )
        data%tau = - SIGN( one, data%ALPHAS( itp1 ) ) * data%tau
      END DO

!  Update the solution estimate using the saved vectors

      X = MATMUL( data%V_extra( : n, : inform%iter_pass2 ),                    &
                  data%X_sub( 0 : inform%iter_pass2 - 1 ) )
      IF ( inform%hard_case )                                                  &
        data%U = MATMUL( data%V_extra( : n, : inform%iter_pass2 ),             &
                         data%U_sub( 0 : inform%iter_pass2 - 1 ) )

      IF ( inform%iter_pass2 == data%dim_sub ) GO TO 800
      R = data%R_extra( : n )
      data%P( : n ) = data%P_extra( : n )
      data%iter = inform%iter_pass2
      data%rminvr_old = data%RMINVRS( data%iter )
      GO TO 590

!  =======================================
!  Re-entry for solution with larger sigma
!  =======================================

  400 CONTINUE
      X = zero ; inform%obj = control%f_0 ; data%U = zero
      inform%obj_regularized = control%f_0 +                                   &
        ( sigma / p ) * ( data%o_mnorm_2_p_eps ** ( p / two ) )
      inform%iter = 0 ; inform%iter_pass2 = 0
      data%use_old = .FALSE. ; data%try_warm = .TRUE.

!  Find the solution to the Lanczos subproblem with this sigma

      IF ( PRESENT( O ) ) THEN
        data%onorm2 = DOT_PRODUCT( data%O_sub( : dim_sub - 1 ),                &
                                   data%O_sub( : dim_sub - 1 ) )
        CALL GLRT_trts( data%dim_sub, data%D( : data%dim_sub - 1 ),            &
                   data%OFFD( : data%dim_sub - 1 ),                            &
                   data%D_fact( : data%dim_sub - 1 ),                          &
                   data%OFFD_fact( : data%dim_sub - 1 ),                       &
                   data%C_sub( : data%dim_sub - 1 ), p, sigma, data%rtol,      &
                   data%titmax, data%try_warm, data%use_old,                   &
                   inform%leftmost, data%LAMBDA( data%dim_sub - 1 ),           &
                   data%MIN_f( data%dim_sub - 1 ),                             &
                   data%MIN_f_regularized( data%dim_sub - 1 ),                 &
                   data%X_sub( : data%dim_sub - 1 ),                           &
                   inform%xpo_norm, data%tinfo, data%titer,                    &
                   data%U_sub( : data%dim_sub - 1 ),                           &
                   data%W( : data%dim_sub - 1 ),                               &
                   data%seed, control%print_level - 1, control%out, prefix,    &
                   inform%hard_case, data%hard_case_step,                      &
                   eps = eps, O = data%O_sub( data%dim_sub - 1 ),              &
                   onorm2 = data%onorm2 )
      ELSE
        CALL GLRT_trts( data%dim_sub, data%D( : data%dim_sub - 1 ),            &
                   data%OFFD( : data%dim_sub - 1 ),                            &
                   data%D_fact( : data%dim_sub - 1 ),                          &
                   data%OFFD_fact( : data%dim_sub - 1 ),                       &
                   data%C_sub( : data%dim_sub - 1 ), p, sigma, data%rtol,      &
                   data%titmax, data%try_warm, data%use_old,                   &
                   inform%leftmost, data%LAMBDA( data%dim_sub - 1 ),           &
                   data%MIN_f( data%dim_sub - 1 ),                             &
                   data%MIN_f_regularized( data%dim_sub - 1 ),                 &
                   data%X_sub( : data%dim_sub - 1 ),                           &
                   inform%xpo_norm, data%tinfo, data%titer,                    &
                   data%U_sub( : data%dim_sub - 1 ),                           &
                   data%W( : data%dim_sub - 1 ),                               &
                   data%seed, control%print_level - 1, control%out, prefix,    &
                   inform%hard_case, data%hard_case_step,                      &
                   eps = eps )
      END IF
!     IF ( data%printi ) WRITE( control%out, 2020 )                            &
!        WRITE( control%out, 2020 )                                            &
!          data%MIN_f_reularized( data%dim_sub - 1 ), data%dim_sub, data%iter

!  Record the optimal objective function value and prepare to recover the
!  approximate minimizer

      inform%multiplier = data%LAMBDA( data%dim_sub - 1 )
      inform%obj = data%MIN_f( data%dim_sub - 1 ) + control%f_0
      inform%obj_regularized                                                   &
        = data%MIN_f_regularized( data%dim_sub - 1 ) + control%f_0
      data%iter = 0 ; data%tau = one

!  ----------------------------------
!  Special part of the code to obtain
!  the approximate minimizer
!  ----------------------------------

  500 CONTINUE

!  If necessary, find M * o

      IF ( PRESENT( O ) ) THEN
        VECTOR = O
        IF ( .NOT. control%unitm ) THEN
          data%branch = 550 ; inform%status = 5
          RETURN
        END IF
      END IF

!  If o is present, modify the initial gradient

  550 CONTINUE
      IF ( PRESENT( O ) ) R = R + data%mult * VECTOR

!  ----------------------------------
!  Obtain the preconditioned residual
!  ----------------------------------

  590 CONTINUE
      VECTOR = R
      IF ( .NOT. control%unitm ) THEN
        data%branch = 600 ; inform%status = 2 ; inform%iter_pass2 = data%iter
        RETURN
      END IF

!  Obtain the scaled norm of the residual

  600 CONTINUE
      itp1 = data%iter + 1

!  Update the solution estimate

      data%rminvr = data%RMINVRS( itp1 )
      IF ( data%iter /= 0 ) THEN
        X = X + data%tau                                                       &
              * ( data%X_sub( data%iter ) / SQRT( data%rminvr ) ) * VECTOR
        IF ( inform%hard_case ) data%U = data%U + data%tau                     &
              * ( data%U_sub( data%iter ) / SQRT( data%rminvr ) ) * VECTOR
      ELSE
        X = data%tau                                                           &
              * ( data%X_sub( data%iter ) / SQRT( data%rminvr ) ) * VECTOR
        IF ( inform%hard_case ) data%U = data%tau                              &
              * ( data%U_sub( data%iter ) / SQRT( data%rminvr ) ) * VECTOR
      END IF

!  If the approximate minimizer is complete, exit

      IF ( itp1 == data%dim_sub ) THEN
        inform%iter_pass2 = itp1
        IF ( data%iter >= data%itmax ) GO TO 910
        GO TO 800
      END IF

      IF ( data%iter > 0 ) THEN
        data%beta = data%rminvr / data%rminvr_old
        data%P( : n ) = - VECTOR + data%beta * data%P( : n )
      ELSE

!  Special case for the first iteration

        data%P( : n ) = - VECTOR
      END IF
      data%rminvr_old = data%rminvr ; data%iter = itp1

!  ------------------------------
!  Obtain the product of H with p
!  ------------------------------

      VECTOR = data%P( : n )
      data%branch = 700 ; inform%status = 3 ; inform%iter_pass2 = data%iter
      RETURN

!  Obtain the curvature

  700 CONTINUE

!  Retreive the stepsize

      data%alpha = data%ALPHAS( data%iter )

!  Update the residual

      R = R + data%alpha * VECTOR
      data%tau = - SIGN( one, data%alpha ) * data%tau
      GO TO 590

!  if the hard case has occured, ensure that the recovered eigenvector has
!  unit norm

  800 CONTINUE
      IF ( inform%hard_case ) THEN
        IF ( .NOT. control%unitm ) THEN
          VECTOR = data%U( : n )
          data%branch = 900 ; inform%status = 3 ; RETURN
        END IF
      END IF

!  in the hard case, compute the complete solution

  900 CONTINUE
      IF ( inform%hard_case ) THEN
        IF ( control%unitm ) THEN
          u_norm = TWO_NORM( data%U( : n ) )
        ELSE
          u_norm = SQRT( - DOT_PRODUCT( VECTOR, data%U( : n ) )                &
                             / inform%multiplier )
        END IF
        X = X + ( data%hard_case_step / u_norm ) * data%U( : n )
      END IF

!  ===============
!  Exit conditions
!  ===============

!  Successful returns

      inform%status = GALAHAD_ok
      RETURN

!  Too many iterations

  910 CONTINUE
      IF ( data%printi )                                                       &
        WRITE( control%out, "( /, A, ' Iteration limit exceeded ' ) " ) prefix
      inform%status = GALAHAD_error_max_iterations ; inform%iter = data%iter
      RETURN

!  Inappropriate entry status

  930 CONTINUE
      inform%status = GALAHAD_error_input_status
      RETURN

!  Inappropriate dimension

  940 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' n = ', I6, ' is not positive ' )" ) prefix, n
      inform%status = GALAHAD_error_restrictions ; inform%iter = data%iter
      RETURN

!  Negative regularisation weight

  950 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' sigma ', ES12.4 , ' is negative ' )" ) prefix, sigma
      inform%status = GALAHAD_error_restrictions ; inform%iter = data%iter
      RETURN

!  Allocation error

  960 CONTINUE
      inform%iter = data%iter
      RETURN

!  Negative shift parameter

  970 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' eps ', ES12.4 , ' is negative ' )" ) prefix, eps
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Regularisation order too small

  980 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' p ', ES12.4 , ' is smaller than 2.0 ' )" ) prefix, p
      inform%status = GALAHAD_error_restrictions
      RETURN

!  Unbounded problem

  990 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The problem is unbounded from below ' )" ) prefix
      inform%status = GALAHAD_error_unbounded ; inform%iter = data%iter
      RETURN

!  Inappropriate preconditioner

 1000 CONTINUE
      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error,                                                  &
         "( A, ' The matrix M appears to be indefinite. Inner product = ',     &
     &      ES12.4  )" ) prefix, data%rminvr
      inform%status = GALAHAD_error_preconditioner ; inform%iter = data%iter
      RETURN

!  Non-executable statements

 2000 FORMAT( /, A, '   Iter    objective     pgnorm      lambda   ',          &
                    '    it info' )
 2010 FORMAT( /, A, '   Iter       f          pgnorm ' )
!2020 FORMAT( /, ' MIN_f, it_exit, it_total ', ES22.14, 2I6 )

!  End of subroutine GLRT_solve

      END SUBROUTINE GLRT_solve

!-*-*-*-*-*-*-  G L R T _ T E R M I N A T E  S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE GLRT_terminate( data, control, inform )

!  ..............................................
!  .                                            .
!  .  Deallocate arrays at end of GLRT_solve    .
!  .                                            .
!  ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine GLRT_initialize
!   info    see Subroutine GLRT_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( GLRT_data_type ), INTENT( INOUT ) :: data
      TYPE ( GLRT_control_type ), INTENT( IN ) :: control
      TYPE ( GLRT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'glrt: P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: OFFD'
      CALL SPACE_dealloc_array( data%OFFD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: ALPHAS'
      CALL SPACE_dealloc_array( data%ALPHAS,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: RMINVRS'
      CALL SPACE_dealloc_array( data%RMINVRS,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: MIN_f'
      CALL SPACE_dealloc_array( data%MIN_f,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: MIN_f_regularized'
      CALL SPACE_dealloc_array( data%MIN_f_regularized,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: LAMBDA'
      CALL SPACE_dealloc_array( data%LAMBDA,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: D_fact'
      CALL SPACE_dealloc_array( data%D_fact,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: OFFD_fact'
      CALL SPACE_dealloc_array( data%OFFD_fact,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: C_sub'
      CALL SPACE_dealloc_array( data%C_sub,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: O_sub'
      CALL SPACE_dealloc_array( data%O_sub,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: X_sub'
      CALL SPACE_dealloc_array( data%X_sub,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: R_extra'
      CALL SPACE_dealloc_array( data%R_extra,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: P_extra'
      CALL SPACE_dealloc_array( data%P_extra,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'glrt: V_extra'
      CALL SPACE_dealloc_array( data%V_extra,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine GLRT_terminate

      END SUBROUTINE GLRT_terminate

!-*-*-*-*-*-*-*-*-*-*-  G L R T _ t r t s  S U B R O U T I N E -*-*-*-*-*-*-*-*

      SUBROUTINE GLRT_trts( n, D, OFFD, D_fact, OFFD_fact, C, p, sigma,        &
                            rtol, itmax, try_warm, use_old, old_leftmost,      &
                            lambda, f, f_regularized, X, xponorm, inform,      &
                            iter, U, W, seed, print_level, out, prefix,        &
                            hard_case, hard_case_step, eps, O, onorm2 )

! ---------------------------------------------------------------------

!  Given an n by n symmetric tridiagonal matrix T, n-vectors c and o, and
!  nonnegative numbers sigma and eps, this subroutine determines a vector
!  x which approximately minimizes the regularised quadratic function

!     f(x) = sigma/p sqrt(||x + o||_2^2+eps)^p + 1/2 <x, T x> + <c, x>

!  This subroutine computes an approximation x and a positive Lagrange
!  multiplier lambda such that

!     ( T + lambda I ) x = - c - lambda o

!  and

!      | sqrt(||x+o||_2^2+eps) - lambda/sigma | <= rtol * ||x||

! ----------------------- dummy arguments -----------------------------

!    n is an integer (in) variable.
!      On entry n is the order of T

!    D is a real (in) array of dimension n.
!      On entry D must contain the diagonal of T

!    OFFD is a real (in) array of dimension n-1.
!      On entry D must contain the subdiagonal of T

!    D_fact is a real (inout) array of dimension n.
!      On entry D_fact need not be specified.
!      On exit D_fact contains the D part of the LDL(transpose)
!      factorization of T + lambda I

!    OFFD_fact is a real (inout) array of dimension n-1.
!      On entry OFFD_fact need not be specified.
!      On exit OFFD_fact contains the subdiagonal part of the L factor
!      from the LDL(transpose) factorization of T + lambda I

!    C is an real (in) array of dimension n.
!      On entry C specifies the linear term in the quadratic

!    p is a real (in) variable.
!      On entry p is the order of regularisation

!    sigma is a real (in) variable.
!      On entry sigma is the regularisation weight.

!    rtol is a real (in) variable.
!      On entry rtol is the relative accuracy desired in the
!         solution. Convergence occurs if

!      | sqrt(||x+o||_2^2+eps) - lambda/sigma | <= rtol * ||x + o||.

!    itmax is an integer (in) variable.
!      On entry itmax specifies the maximum number of iterations

!    try_warm is a logical (in) variable.
!      On entry try_warm is .TRUE. if the input value lambda is to be
!       tried before any other estimate

!    use_old is a logical (in) variable.
!      On entry use_old is .TRUE. if the leftmost eigenvalue of the
!       leading n-1 by n-1 block is given

!    old_leftmost is a real (inout) variable.
!      On entry old_leftmost gives the leftmost eigenvalue of the
!       leading n-1 by n-1 block. Only required if use_old is .TRUE.
!      On exit gives the leftmost eigenvalue of T if T is indefinite

!    lambda is a real (in) variable.
!      On entry lambda is an initial estimate of the Lagrange
!         multiplier ||x|| sigma

!    f is a real (out) variable.
!      On exit f is set to 1/2 <x, T x> + <c, x> at the output x

!    f_regularized is a real (out) variable.
!      On exit f is set to f(x) at the output x

!    X is a real (out) array of dimension n.
!      On exit x is set to the final estimate of the solution.

!    xponorm is a real (out) variable.
!      On exit xponorm is set to ||x + o|| at the output x

!    inform is an integer (out) variable.
!      On entry inform need not be specified.
!         inform = 0  The function value f(x) has the relative
!                   accuracy specified by rtol.
!         inform = 1  The Newton search direction is too small to make
!                   further progress
!         inform = 2  Failure to converge after itmax iterations.
!                   On exit x is the best available approximation.

!    iter is an integer variable.
!      On entry iter need not be specified.
!      On exit iter gives the total number of iterations required.

!    iter is an integer (out) variable.
!      On exit iter gives the total number of iterations required

!    U is a real work (out) array of dimension n that may hold an
!      eigenvector estimate

!    W is a real work (out) array of dimension n.

!    print_level is an integer (in) variable.
!      On entry print_level should be positive if debug printing is required

!    out is an integer (in) variable.
!      On entry the unit for output if required

!    prefix is a character (in) variable of unspecified length.
!      On entry prefix contains a prefix which will preceed each output line

!    hard_case is a logical (out) variable.
!      On exit, hard_case is .TRUE. if the hard case has arisen
!      and is .FALSE. otherwise

!    hard_case_step is a real (out) variable.
!      On exit, it will give the scalar alpha for which x + alpha u is the
!      required solution in the jard case. It may be ignored otherwise.

!    eps is an OPTIONAL real variable.
!      If PRESENT on entry eps is the regularisation shift, eps.
!      On exit eps is unchanged

!    O is an OPTIONAL real array of dimension n.
!      If PRESENT on entry O specifies the regularisation offset, o.
!      On exit S is unchanged
!
!    onorm2 is an OPTIONAL real variable.
!      If PRESENT on entry onorm2 is the square of the two-norm of O, ||o||.
!      On exit onorm2 is unchanged

! --------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, itmax, out, print_level
      INTEGER, INTENT( OUT ) :: inform, iter
      LOGICAL, INTENT( IN ) :: use_old, try_warm
      REAL ( KIND = wp ), INTENT( IN ) :: p, sigma, rtol
      REAL ( KIND = wp ), INTENT( INOUT ) :: lambda, old_leftmost
      REAL ( KIND = wp ), INTENT( OUT ) :: f, f_regularized, xponorm
      REAL ( KIND = wp ), INTENT( OUT ) :: hard_case_step
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n - 1 ) :: OFFD
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C, D
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n - 1 ) :: OFFD_fact
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: D_fact, X, U, W
      TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      LOGICAL, INTENT( OUT ) :: hard_case
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: eps, onorm2
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: O

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: indef, i, it, nroots
      REAL ( KIND = wp ) :: delta, lambda_pert, utxpo, rxnorm2, tol
      REAL ( KIND = wp ) :: leftmost, delta_lambda, pert_l
      REAL ( KIND = wp ) :: error, root1, root2, real_f, gamma
      REAL ( KIND = wp ) :: v2oy2, dl_phi_c, omega, omega_prime
      REAL ( KIND = wp ) :: pi, pi_prime
!     REAL ( KIND = wp ) :: zeta, zeta_prime, theta, theta_prime
!     REAL ( KIND = wp ) :: dl_theta, dl_phi, dl_zeta, phi, phi_prime
      LOGICAL :: debug

!  Initialization

      iter = 1
      hard_case = .FALSE. ; hard_case_step = zero
      debug = print_level > 0
!     pert_l = epsmch ** 0.5 ; tol = epsmch ** 0.66
      pert_l = epsmch ** 0.75 ; tol = epsmch ** 0.66

!  =======================
!  First, try a warm start
!  =======================

      IF ( try_warm ) THEN

!  Compute T + lambda*I

        OFFD_fact = OFFD
        D_fact = D + lambda

!  Find the Cholesky factors of T

        CALL PTTRF( n, D_fact, OFFD_fact, indef )

!  If T is positive definite, solve  T x = - c - lambda * o

        IF ( indef == 0 ) THEN
          IF ( PRESENT( O ) ) THEN
            CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda, D_fact, OFFD_fact,&
                                         - C - lambda * O, X, W, U, itref_max, &
                                         rxnorm2, out, debug, prefix )
          ELSE
            CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda, D_fact, OFFD_fact,&
                                         - C, X, W, U, itref_max, rxnorm2, out,&
                                         debug, prefix )
          END IF

!  If the (p-2)nd power of the solution is larger than lambda/sigma,
!  it provides a good initial estimate of the solution to the problem

          delta = lambda / sigma
          IF ( PRESENT( O ) ) THEN
            xponorm = TWO_NORM( X + O )
          ELSE
            xponorm = TWO_NORM( X )
          END IF
          IF ( PRESENT( eps ) ) THEN
            omega = SQRT( xponorm * xponorm + eps )
          ELSE
            omega = xponorm
          END IF
          error = omega ** ( p - two ) - delta
          IF ( print_level > 0 ) THEN
            WRITE( out, "( A, 12X, 'lambda', 17X, 'error', 19X,'||x||')") prefix
            WRITE( out, "( A, 3ES24.16 )" ) prefix, lambda, error, xponorm
          END IF

          IF ( ABS( error ) <= rtol * xponorm ) THEN
            CALL GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized,   &
                              eps = eps, onorm2 = onorm2 )
            inform = 0
            RETURN
          END IF
          IF ( error > zero ) GO TO 20
        END IF
      END IF

!  ====================================
!  Compute the leftmost eigenvalue of T
!  ====================================

      leftmost = GLRT_leftmost_eigenvalue( n, D, OFFD, tol, use_old,           &
                   old_leftmost, it, print_level > 0, out, prefix )

      IF ( print_level > 0 ) WRITE( out, "( A, ' iteration ', I6,              &
     &     ' leftmost eigenvalue = ', ES22.14 )") prefix, it, leftmost

      old_leftmost = leftmost ;  tol = point1 * tol

!  Pick an initial estimate of lambda

      IF ( leftmost <= zero ) THEN
        lambda_pert = - leftmost * ( one + epsmch ) + pert_l
      ELSE
!       lambda_pert = - leftmost * ( one - epsmch ) + pert_l
        lambda_pert = zero
      END IF

!  Compute T + lambda*I

      DO
        OFFD_fact = OFFD
        D_fact = D + lambda_pert

!  Find the Cholesky factors of T

        CALL PTTRF( n, D_fact, OFFD_fact, indef )

!  Make sure that T really is numerically positive definite

        IF ( indef == 0 ) EXIT

!  If H is still numerically indefinite it must be perturbed a bit more

        pert_l = pert_l + pert_l
        lambda_pert = lambda_pert + pert_l
      END DO

!  Solve T x = - c - lambda * o

      IF ( PRESENT( O ) ) THEN
        CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda_pert, D_fact,        &
                                     OFFD_fact, - C - lambda_pert * O, X, W, &
                                     U, itref_max, rxnorm2, out, debug, prefix )
      ELSE
        CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda_pert, D_fact,        &
                                     OFFD_fact, - C, X, W,                   &
                                     U, itref_max, rxnorm2, out, debug, prefix )
      END IF

      delta = lambda_pert / sigma
      IF ( PRESENT( O ) ) THEN
        xponorm = TWO_NORM( X + O )
      ELSE
        xponorm = TWO_NORM( X )
      END IF
      IF ( PRESENT( eps ) ) THEN
        omega = SQRT( xponorm * xponorm + eps )
      ELSE
        omega = xponorm
      END IF
      error = omega ** ( p - two ) - delta

      IF ( print_level > 0 ) THEN
        WRITE( out, "( A, 12X, 'lambda', 17X, 'error', 19X, 'x' )" ) prefix
        WRITE( out, "( A, 3ES24.16 )" ) prefix, lambda_pert, error, xponorm
      END IF

!  =========
!  Hard case
!  =========

!  If the (p-2)nd power of the norm of X is smaller than lambda/sigma,
!  we are in the hard case

      IF ( error < zero ) THEN
        hard_case = .TRUE.
        IF ( debug ) WRITE( out, "( A, ' hard case ' )" ) prefix
        lambda = - leftmost

!  Compute a leftmost eigenvector

        CALL GLRT_leftmost_eigenvector( n, leftmost, D, OFFD, D_fact,          &
                                        OFFD_fact, U, it, seed )
        IF ( print_level > 0 ) WRITE( out, "( A, ' iteration ', I6,            &
       &  ' hard case: leftmost eigenvector found ' )" ) prefix, it

!  Compute the step alpha so that
!    ( ||x + o + alpha u||^2 + eps )^(p-2)/2 = lambda/sigma
!  and gives the smaller value of q

        delta = ( ABS( lambda ) / sigma ) ** ( two / ( p - two ) ) - omega ** 2
        IF ( PRESENT( O ) ) THEN
          utxpo = DOT_PRODUCT( U, X + O )
        ELSE
          utxpo = DOT_PRODUCT( U, X )
        END IF
        hard_case_step = - utxpo + SQRT( utxpo ** 2 + delta )

!  Record the optimal values

        X = X + hard_case_step * U
        CALL GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized,       &
                          eps = eps, onorm2 = onorm2 )
        inform = 0
        RETURN

!  The Lagrange multiplier will be positive and lambda is in L

      ELSE
        lambda = lambda_pert
      END IF

!  It is now simply a matter of applying Newton's method starting from lambda

!  =====================
!  Main Newton iteration
!  =====================

   20 CONTINUE

      DO iter = 2, itmax

!  Compute a correction to lambda

        IF ( PRESENT( O ) ) THEN
          W = X + O
        ELSE
          W = X
        END IF
        DO i = 1, n - 1
          W( i + 1 ) = W( i + 1 ) - OFFD_fact( i ) * W( i )
        END DO

!  Compute omega = sqrt( || x(lambda) + o ||^2 + eps ) and its derivative

!        IF ( PRESENT( eps ) ) THEN
!          omega = SQRT( xponorm * xponorm + eps )
!        ELSE
!          omega = xponorm
!        END IF
        omega_prime = - DOT_PRODUCT( W, W / D_fact ) / omega

!  When p <= 3, a suitable correction for lambda is the positive root of
!    sigma/lambda = pi + ( lambda - lambda_current ) pi' ;
!  here the right-hand side is the linearization of pi = 1/||x(lambda)||^(p-2)

        IF ( p <= three ) THEN

          pi = omega ** ( two - p )
          pi_prime = ( two - p ) * ( omega ** ( one - p ) ) * omega_prime

          v2oy2 = pi_prime / pi

          CALL ROOTS_quadratic( - sigma / pi, one - lambda * v2oy2, v2oy2,     &
                                roots_tol, nroots, root1, root2, roots_debug )
          IF ( nroots == 2 ) THEN
            dl_phi_c = root2 - lambda
          ELSE
            dl_phi_c = root1 - lambda
          END IF

!  When p > 3, a suitable correction for lambda is the positive root of
!  the linearization of 1/||x(lambda)|| - (sigma/lambda)^1/(p-2) = 0

        ELSE
          IF ( lambda > zero ) THEN
            gamma = one / ( p - two )
            pi = omega ** ( - one ) - ( sigma / lambda ) ** gamma
            pi_prime = - ( omega ** ( - two ) ) * omega_prime -                &
             ( sigma ** gamma ) * ( - gamma ) * ( lambda ** ( - one - gamma ) )

!  special case when lambda = 0: use the linearization of
!  ||x(lambda)||^(p-2) - lambda/sigma = 0

          ELSE
            gamma = p - two
            pi = omega ** gamma
            pi_prime =                                                         &
              gamma * omega ** ( gamma - one ) * omega_prime - one / sigma
          END IF
          dl_phi_c = - pi / pi_prime
        END IF

!  Alternatives are Newton corrections to the functions given in the ARC paper

!       IF ( print_level > 1 ) THEN

!  theta correction

!         theta = xponorm ** ( p - two ) - lambda / sigma
!         theta_prime = ( p - two ) * ( omega ** ( p - three ) ) *             &
!               omega_prime - one / sigma
!         dl_theta = - theta / theta_prime

!  phi correction

!         IF ( lambda /= zero ) THEN
!           phi = pi - sigma / lambda
!           phi_prime = pi_prime + sigma / ( lambda ** 2 )
!           dl_phi = - phi / phi_prime
!         ELSE
!           dl_phi = zero
!         END IF

!  zeta correction

!         zeta = lambda * pi - sigma
!         zeta_prime = pi + lambda * pi_prime
!         dl_zeta = - zeta / zeta_prime

!         WRITE( out, 2010 )                                                   &
!           prefix, dl_phi_c, prefix, dl_theta, dl_phi, dl_zeta
!       END IF

!  Compute the Newton-like update

        delta_lambda = dl_phi_c

!  Check that the correction is significant

        IF ( ABS( delta_lambda ) < epsmch * ABS( lambda ) ) THEN
          CALL GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized,     &
                            eps = eps, onorm2 = onorm2 )
          inform = 1
          RETURN
        END IF

!  Compute the new estimate of lambda

        lambda = lambda + delta_lambda

!  Find the Cholesky factorization of T + lambda*I

        OFFD_fact = OFFD ; D_fact = D + lambda
        CALL PTTRF( n, D_fact, OFFD_fact, indef )

!  Solve the equation (T + lambda*I) x = - c - lambda * o

        IF ( PRESENT( O ) ) THEN
          CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda, D_fact, OFFD_fact,  &
                                       - C - lambda * O, X, W, U, itref_max,   &
                                       rxnorm2, out, debug, prefix )
        ELSE
          CALL GLRT_tridiagonal_solve( n, D, OFFD, lambda, D_fact, OFFD_fact,  &
                                       - C, X, W, U, itref_max, rxnorm2, out,  &
                                       debug, prefix )
        END IF

        delta = lambda / sigma
        IF ( PRESENT( O ) ) THEN
          xponorm = TWO_NORM( X + O )
        ELSE
          xponorm = TWO_NORM( X )
        END IF

       IF ( PRESENT( eps ) ) THEN
          omega = SQRT( xponorm * xponorm + eps )
        ELSE
          omega = xponorm
        END IF
        error = omega ** ( p - two ) - delta
        IF ( print_level > 1 )                                                 &
          WRITE( out, "( A, 12X, 'lambda', 17X, 'error', 19X, 'x' )" ) prefix
        IF ( print_level > 0 ) WRITE( out, "( A, 3ES24.16 )" )                 &
          prefix, lambda, error, xponorm

!  Test for convergence

        IF ( error <= rtol * xponorm ) THEN
          CALL GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized,     &
                            eps = eps, onorm2 = onorm2 )
          IF ( print_level > 1 ) THEN
            real_f = DOT_PRODUCT( C, X ) + half * DOT_PRODUCT( X, D * X )      &
                       + DOT_PRODUCT( X( : n - 1 ) * OFFD, X( 2 : ) )          &
                       + ( sigma / p ) * omega ** p
            WRITE( out, "( A, ' real, recurred f = ', 2ES22.14 )" ) prefix,    &
                                                                    real_f, f
          END IF
          inform = 0
          RETURN
        END IF

      END DO

!  Test for termination

      inform = 3
      CALL GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized,         &
                        eps = eps, onorm2 = onorm2 )
      RETURN

!  Non-executable statement

!2010 FORMAT( A, ' correction ', ES16.8, ' alternatives',/,                    &
!             A, ' theta      ', ES16.8, ' phi ', ES16.8, ' zeta ', ES16.8 )

!  End of subroutine GLRT_trts

      CONTAINS

!-*-*-  G L R T _ t r t s  _ f   I N T E R N A L  S U B R O U T I N E   -*-*-

        SUBROUTINE GLRT_trts_f( p, lambda, rxnorm2, xponorm, f, f_regularized, &
                                eps, onorm2 )

!----------------------------------------------------------------------

!  input variables
!   onorm2 = ||o||^2,
!   xponorm = ||x+o||,
!   rxnorm2 = (c+lamba o)' ( T + lamba I )^-1  (c+lambda o) and
!   lambda = sigma sqrt(||x+o||_2^2+eps)^p-2
!  and output variables
!   f =  1/2 ||x||^2 + <c, x> and
!   f_regularized = f + r, where r = (sigma/p) sqrt(||x+o||_2^2+eps)^p

!  note that
!   f = - 1/2 (c+lamba o)' ( T + lamba I )^-1  (c+lambda o)
!       - lambda/2 ||x+o||^2 + lambda/2 ||o||^2
!  and
!   r = (lambda/p) (||x+o||^2 + eps)

!----------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

        REAL ( KIND = wp ), INTENT( IN ) :: p, lambda, rxnorm2, xponorm
        REAL ( KIND = wp ), INTENT( OUT ) :: f, f_regularized
        REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: eps, onorm2

        IF ( PRESENT( onorm2 ) ) THEN
          f = - half * rxnorm2 - half * lambda * ( xponorm ** 2 - onorm2 )
        ELSE
          f = - half * rxnorm2 - half * lambda * xponorm ** 2
        END IF
        IF (  PRESENT( eps ) ) THEN
          f_regularized = f + ( lambda / p ) * ( xponorm ** 2 + eps )
        ELSE
          f_regularized = f + ( lambda / p ) * xponorm ** 2
        END IF

!       IF ( PRESENT( onorm2 ) ) THEN
!         IF ( PRESENT( eps ) ) THEN
!           f_regularized = - half * rxnorm2 +                                 &
!             ( lambda / p ) * ( ( one - half * p ) * xponorm ** 2 + eps       &
!               + half * p * onorm2 )
!         ELSE
!           f_regularized = - half * rxnorm2 +                                 &
!             ( lambda / p ) * ( ( one - half * p ) * xponorm ** 2             &
!               + half * p * onorm2 )
!         END IF
!       ELSE
!         IF ( PRESENT( eps ) ) THEN
!           f_regularized = - half * rxnorm2 +                                 &
!                ( lambda / p ) * ( ( one - half * p ) * xponorm ** 2 + eps )
!         ELSE
!           f_regularized = - half * rxnorm2 +                                 &
!                ( lambda / p ) * ( one - half * p ) * xponorm ** 2
!         END IF
!       END IF

        RETURN

!  End of function GLRT_trts_f

        END SUBROUTINE GLRT_trts_f

      END SUBROUTINE GLRT_trts

!-*-*-*-*-*-*-*-  End of G A L A H A D _ G L R T  M O D U L E  *-*-*-*-*-*-*-*-

   END MODULE GALAHAD_GLRT_double
