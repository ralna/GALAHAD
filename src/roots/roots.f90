! THIS VERSION: GALAHAD 3.2 - 09/06/2019 AT 14:10 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ R O O T S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   started life (quadratic roots) in GALAHAD_LSQP ~ 2000
!   released with GALAHAD Version 2.0. April 27th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_ROOTS_double

!     --------------------------------------------------------------------
!     |                                                                  |
!     |  Find (all the) real roots of polynomials with real coefficients |
!     |                                                                  |
!     --------------------------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SORT_double, ONLY : SORT_quicksort
      USE GALAHAD_LAPACK_interface, ONLY : HSEQR
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: ROOTS_solve, ROOTS_quadratic, ROOTS_cubic, ROOTS_quartic,      &
                ROOTS_polynomial, ROOTS_smallest_root_in_interval,             &
                ROOTS_polynomial_value,                                        &
                ROOTS_initialize, ROOTS_terminate, ROOTS_read_specfile

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: wcp = KIND( ( 1.0D+0, 1.0D+0 ) )

!------------------------------------
!   G e n e r i c   I n t e r f a c e
!------------------------------------

!     INTERFACE ROOTS_solve
!       MODULE PROCEDURE ROOTS_quadratic, ROOTS_cubic, ROOTS_quartic
!     END INTERFACE ROOTS_solve

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: six = 6.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: quarter = 0.25_wp
      REAL ( KIND = wp ), PARAMETER :: threequarters = 0.75_wp
      REAL ( KIND = wp ), PARAMETER :: onesixth = one / six
      REAL ( KIND = wp ), PARAMETER :: onethird = one / three
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: twothirds = two / three
      REAL ( KIND = wp ), PARAMETER :: fourthirds = four / three
!     REAL ( KIND = wp ), PARAMETER :: pi = four * ATAN( 1.0_wp )
      REAL ( KIND = wp ), PARAMETER :: pi = 3.1415926535897931_wp
!     REAL ( KIND = wp ), PARAMETER :: magic = twothirds * pi
      REAL ( KIND = wp ), PARAMETER :: magic = 2.0943951023931953_wp  !! 2 pi/3
      REAL ( KIND = wp ), PARAMETER :: base = RADIX( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: smallest = TINY( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

      INTEGER, PARAMETER :: out = 6

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ROOTS_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the required accuracy of the roots

        REAL ( KIND = wp ) :: tol = 10.0_wp * EPSILON( one )

!   any coefficient smaller in absolute value than zero_coef will be regarded
!     to be zero

        REAL ( KIND = wp ) :: zero_coef = 10.0_wp * EPSILON( one )

!   any value of the polynomial smaller in absolute value than zero_f
!     will be regarded as giving a root

        REAL ( KIND = wp ) :: zero_f = 10.0_wp * EPSILON( one )

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

      TYPE, PUBLIC :: ROOTS_inform_type

!  return status. See ROOTS_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ROOTS_data_type
        PRIVATE
        INTEGER :: n_max = - 1
        INTEGER :: degree_max = - 1
        INTEGER :: w_max = - 1
        INTEGER :: ig_max = - 1
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IG
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: P, A_mat, RHS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S, W, WORK
        COMPLEX ( KIND = wcp ), ALLOCATABLE, DIMENSION( : ) :: CROOTS
      END TYPE

   CONTAINS

!-*-*-*-*-*-   R O O T S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-

      SUBROUTINE ROOTS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for ROOTS. This routine should be called before
!  ROOTS_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ROOTS_control_type ), INTENT( INOUT ) :: control
      TYPE ( ROOTS_inform_type ), INTENT( OUT ) :: inform

      data%n_max = - 1
      data%degree_max = - 1
      data%w_max = - 1
      data%ig_max = - 1

      control%tol = 10.0_wp * EPSILON( one )
      control%zero_coef = 10.0_wp * EPSILON( one )
      control%zero_f = 10.0_wp * EPSILON( one )

      inform%status = GALAHAD_ok

      RETURN

!  End of ROOTS_initialize

      END SUBROUTINE ROOTS_initialize

!-*-*-*-   R O O T S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE ROOTS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by ROOTS_initialize could (roughly)
!  have been set as:

! BEGIN ROOTS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  root-tolerance                                    2.2D-16
!  zero-coefficient-tolerance                        2.2D-16
!  zero-polynomial-tolerance                         2.2D-16
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END ROOTS SPECIFICATIONS

!  Dummy arguments

      TYPE ( ROOTS_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: tol = print_level + 1
      INTEGER, PARAMETER :: zero_coef = tol + 1
      INTEGER, PARAMETER :: zero_f = zero_coef + 1
      INTEGER, PARAMETER :: space_critical = zero_f + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'ROOTS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

!  Real key-words

      spec( tol )%keyword = 'root-tolerance'
      spec( zero_coef )%keyword = 'zero-coefficient-tolerance'
      spec( zero_f )%keyword = 'zero-polynomial-tolerance'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = '""'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

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

!  Set real value

      CALL SPECFILE_assign_value( spec( tol ),                                 &
                                  control%tol,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( zero_coef ),                           &
                                  control%zero_coef,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( zero_f ),                              &
                                  control%zero_f,                              &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set character value

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

      RETURN

      END SUBROUTINE ROOTS_read_specfile

!-*-*-*-*-*-*-*-   R O O T S _ s o l v e   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE ROOTS_solve( A, nroots, ROOTS, control, inform, data )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==

!  Find all the real roots of a real polynomial sum_i>=0 a(i) x^i

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==

!  Dummy arguments

      INTEGER, INTENT( OUT ) :: nroots
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 0 : ) :: A
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: ROOTS
      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ROOTS_control_type ), INTENT( IN ) :: control
      TYPE ( ROOTS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: degree, i
      CHARACTER ( LEN = 80 ) :: array_name
      LOGICAL :: debug

!  Check input details for errors and consistency

      degree = UBOUND( A, 1 )
      IF ( degree < 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; RETURN
      ELSE IF ( UBOUND( ROOTS, 1 ) < degree ) THEN
        inform%status = GALAHAD_error_restrictions ; RETURN
      END IF
      inform%status = GALAHAD_ok
      debug = control%out > 0 .AND. control%print_level > 0

!  The data appears to be correct

      SELECT CASE( degree )

!  polynomials of degree 0

      CASE ( 0 )
        nroots = 0

!  polynomials of degree 1

      CASE ( 1 )
        IF ( A( 1 ) == zero ) THEN
          IF ( A( 1 ) == zero ) THEN
            nroots = 1
            ROOTS( 1 ) = zero
          ELSE
            nroots = 0
          END IF
        ELSE
          nroots = 1
          ROOTS( 1 ) = - A( 0 ) / A( 1 )
        END IF

!  polynomials of degree 2

      CASE ( 2 )
        CALL ROOTS_quadratic( A( 0 ), A( 1 ), A( 2 ), control%tol,             &
          nroots, ROOTS( 1 ), ROOTS( 2 ), debug )

!  polynomials of degree 3

      CASE ( 3 )
        CALL ROOTS_cubic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), control%tol,         &
          nroots, ROOTS( 1 ), ROOTS( 2 ), ROOTS( 3 ), debug )

!  polynomials of degree 4

!     CASE ( 4 )
!       CALL ROOTS_quartic( A( 0 ), A( 1 ), A( 2 ), A( 3 ), A( 4 ),            &
!         control%tol, nroots, ROOTS( 1 ), ROOTS( 2 ), ROOTS( 3 ), ROOTS( 4 ), &
!         debug )

!  polynomials of degree > 4

!     CASE ( 5 : )
      CASE ( 4 : )

!  allocate space for the complex roots

        IF ( degree > data%degree_max ) THEN
          data%degree_max = degree
          array_name = 'roots: data%CROOTS'
          CALL SPACE_resize_array( degree, data%CROOTS,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN
        END IF

!  find all the roots

        CALL ROOTS_polynomial( A, degree, data%CROOTS, control, inform, data )

!  extract the real roots

        nroots = 0
        DO i = 1, degree
          IF ( AIMAG( data%CROOTS( i ) ) == zero ) THEN
            nroots = nroots + 1
            ROOTS( nroots) = REAL( data%CROOTS( i ) )
          END IF
        END DO

!  order the real roots

        IF ( nroots > 0 ) CALL SORT_quicksort( nroots, ROOTS( : nroots ),      &
                                               inform%status )
      END SELECT

      RETURN

!  End of subroutine ROOTS_solve

      END SUBROUTINE ROOTS_solve

!-*-*-*-*-*-   R O O T S _ q u a d r a t i c  S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE ROOTS_quadratic( a0, a1, a2, tol, nroots, root1, root2, debug )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the number and values of real roots of the quadratic equation
!
!                   a2 * x**2 + a1 * x + a0 = 0
!
!  where a0, a1 and a2 are real
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( OUT ) :: nroots
      REAL ( KIND = wp ), INTENT( IN ) :: a2, a1, a0, tol
      REAL ( KIND = wp ), INTENT( OUT ) :: root1, root2
      LOGICAL, INTENT( IN ) :: debug

!  Local variables

      REAL ( KIND = wp ) :: rhs, d, p, pprime

      rhs = tol * a1 * a1
      IF ( ABS( a0 * a2 ) > rhs ) THEN  !  really is quadratic
        root2 = a1 * a1 - four * a2 * a0
        IF ( ABS( root2 ) <= ( epsmch * a1 ) ** 2 ) THEN ! numerical double root
          nroots = 2 ; root1 = -  half * a1 / a2 ; root2 = root1
        ELSE IF ( root2 < zero ) THEN    ! complex not real roots
          nroots = 0 ; root1 = zero ; root2 = zero
        ELSE                             ! distint real roots
          d = - half * ( a1 + SIGN( SQRT( root2 ), a1 ) )
          nroots = 2 ; root1 = d / a2 ; root2 = a0 / d
          IF ( root1 > root2 ) THEN
            d = root1 ; root1 = root2 ; root2 = d
          END IF
        END IF
      ELSE IF ( a2 == zero ) THEN
        IF ( a1 == zero ) THEN
          IF ( a0 == zero ) THEN         ! the function is zero
            nroots = 1 ; root1 = zero ; root2 = zero
          ELSE                           ! the function is constant
            nroots = 0 ; root1 = zero ; root2 = zero
          END IF
        ELSE                             ! the function is linear
          nroots = 1 ; root1 = - a0 / a1 ; root2 = zero
        END IF
      ELSE                               ! very ill-conditioned quadratic
        nroots = 2
        IF ( - a1 / a2 > zero ) THEN
          root1 = zero ; root2 = - a1 / a2
        ELSE
          root1 = - a1 / a2 ; root2 = zero
        END IF
      END IF

!  perfom a Newton iteration to ensure that the roots are accurate

      IF ( nroots >= 1 ) THEN
        p = ( a2 * root1 + a1 ) * root1 + a0
        pprime = two * a2 * root1 + a1
        IF ( pprime /= zero ) THEN
          IF ( debug ) WRITE( out, 2000 ) 1, root1, p, - p / pprime
          root1 = root1 - p / pprime
          p = ( a2 * root1 + a1 ) * root1 + a0
        END IF
        IF ( debug ) WRITE( out, 2010 ) 1, root1, p
        IF ( nroots == 2 ) THEN
          p = ( a2 * root2 + a1 ) * root2 + a0
          pprime = two * a2 * root2 + a1
          IF ( pprime /= zero ) THEN
            IF ( debug ) WRITE( out, 2000 ) 2, root2, p, - p / pprime
            root2 = root2 - p / pprime
            p = ( a2 * root2 + a1 ) * root2 + a0
          END IF
          IF ( debug ) WRITE( out, 2010 ) 2, root2, p
        END IF
      END IF

      RETURN

!  Non-executable statements

 2000 FORMAT( ' root ', I1, ': value = ', ES12.4, ' quadratic = ', ES12.4,     &
              ' delta = ', ES12.4 )
 2010 FORMAT( ' root ', I1, ': value = ', ES12.4, ' quadratic = ', ES12.4 )


!  End of subroutine ROOTS_quadratic

      END SUBROUTINE ROOTS_quadratic

!-*-*-*-*-*-*-*-   R O O T S _ c u b i c  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE ROOTS_cubic( a0, a1, a2, a3, tol, nroots, root1, root2,       &
                              root3, debug )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the number and values of real roots of the cubicc equation
!
!                a3 * x**3 + a2 * x**2 + a1 * x + a0 = 0
!
!  where a0, a1, a2 and a3 are real
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( OUT ) :: nroots
      REAL ( KIND = wp ), INTENT( IN ) :: a3, a2, a1, a0, tol
      REAL ( KIND = wp ), INTENT( OUT ) :: root1, root2, root3
      LOGICAL, INTENT( IN ) :: debug

!  Local variables

      INTEGER :: info, nroots_q
      REAL ( KIND = wp ) :: a, b, c, d, e, f, p, q, s, t, w, x, y, z
      REAL ( KIND = wp ) :: c0, c1, c2, b0, b1, pprime, u1, u2
      REAL ( KIND = wp ) :: H( 3, 3 ), ER( 3 ), EI( 3 ), ZZ( 1, 3 ), WORK( 33 )

!  define method used:
!    1 = Nonweiler, 2 = Littlewood, 3 = Viete, other = companion matrix

      INTEGER, PARAMETER :: method = 1

!  Check to see if the quartic is actually a cubic

      IF ( a3 == zero ) THEN
        CALL ROOTS_quadratic( a0, a1, a2, tol, nroots, root1, root2, debug )
        root3 = infinity
        RETURN
      END IF

!  Deflate the polnomial if the trailing coefficient is zero

      IF ( a0 == zero ) THEN
        root1 = zero
        CALL ROOTS_quadratic( a1, a2, a3, tol, nroots, root2, root3, debug )
        nroots = nroots + 1
        RETURN
      END IF

!  1. Use Nonweiler's method (CACM 11:4, 1968, pp269)

      IF ( method == 1 ) THEN
        c0 = a0 / a3
        c1 = a1 / a3
        c2 = a2 / a3

        s = c2 / three
        t = s * c2
        b = 0.5_wp * ( s * ( twothirds * t - c1 ) + c0 )
        t = ( t - c1 ) / three
        c = t * t * t ; d = b * b - c

! 1 real + 2 equal real or 2 complex roots

        IF ( d >= zero ) THEN
          d = ( SQRT( d ) + ABS( b ) ) ** onethird
          IF ( d /= zero ) then
            IF ( b > zero ) then
              b = - d
            ELSE
              b = d
            END IF
            c = t / b
          END IF
          d = SQRT( threequarters ) * ( b - c )
          b = b + c ; c = - 0.5 * b - s
          root1 = b - s
          IF ( d == zero ) THEN
            nroots = 3 ; root2 = c ; root3 = c
          ELSE
            nroots = 1
          END IF

! 3 real roots

        ELSE
          IF ( b == zero ) THEN
            d = twothirds * ATAN( one )
          ELSE
            d = ATAN( SQRT( - d ) / ABS( b ) ) / three
          END IF
          IF ( b < zero ) THEN
            b = two * SQRT( t )
          ELSE
            b = - two * SQRT( t )
          END IF
          c = COS( d ) * b
          t = - SQRT( threequarters ) * SIN( d ) * b - half * c
          d = - t - c - s ; c = c - s ; t = t - s
          IF ( ABS( c ) > ABS( t ) ) then
            root3 = c
          ELSE
            root3 = t
            t = c
          END IF
          IF ( ABS( d ) > ABS( t ) ) THEN
            root2 = d
          ELSE
            root2 = t
            t = d
          END IF
          root1 = t ; nroots = 3
        END IF

!  2. Use Littlewood's method

      ELSE IF ( method == 2 ) THEN
        c2 = a2 / ( three * a3 ) ; c1 = a1 / ( three * a3 ) ; c0 = a0 / a3
        x = c1 - c2 * c2
        y = c0 - c2* ( x + x + c1 )
        z = y ** 2 + four * x ** 3

!  there are three real roots

        IF ( z < zero ) THEN
          a = - two * SQRT( - x )
          b = y / ( a * x )
          y = ATAN2( SQRT( one - b ), SQRT( one + b ) ) * twothirds
          IF ( c2 < zero ) y = y + magic

!  calculate root which does not involve cancellation

          nroots = 1 ; root1 = a * COS( y ) - c2

!  there may be only one real root

        ELSE
          a = SQRT( z ) ; b = half * ( ABS( y ) + a ) ; c = b ** onethird
          IF ( c <= zero ) THEN
            nroots = 3 ; root1 = - c2 ; root2 = - c2 ; root3 = - c2
            GO TO 900
          ELSE
            nroots = 1
            c = c - ( c ** 3 - b ) / ( three * c * c )
            e = c * c + ABS( x )
            f = one / ( ( x / c ) ** 2 + e )
            IF ( x >= zero ) THEN
              x = e / c ; z = y * f
            ELSE
              x = a * f ; z = SIGN( one, y ) * e / c
            END IF
            IF ( z * c2 >= zero ) THEN
              root1 = - z - c2
            ELSE
              root2 = half * z - c2
              root3 = half * SQRT( three ) * ABS( x )
              root1 = - c0 / ( root2 * root2 + root3 * root3 )
              GO TO 900
            END IF
          END IF
        END IF

!  deflate cubic

        b0 = - c0 / root1
        IF ( ABS( root1 ** 3 ) <= ABS( c0 ) ) THEN
          b1 = root1 + three * c2
        ELSE
          b1 = ( b0 - three * c1 ) / root1
        END IF
        CALL ROOTS_quadratic( b0, b1, one, epsmch, nroots_q,                   &
                              root2, root3, debug )
        nroots = nroots + nroots_q


!  3. Use Viete's method

      ELSE IF ( method == 3 ) THEN
        w = a2 / ( three * a3 )
        p = ( a1 / ( three * a3 ) - w ** 2 ) ** 3
        q = - half * ( two * w ** 3 - ( a1 * w - a0 ) / a3 )
        d = p + q ** 2

!  three real roots

        IF ( d < zero ) THEN
          s = ACOS( MIN( one, MAX( - one, q / SQRT( - p ) ) ) )
          p = two * ( - p ) ** onesixth
          nroots = 3
          root1 = p * COS( onethird * ( s + two * pi ) ) - w
          root2 = p * COS( onethird * ( s + four * pi ) ) - w
          root3 = p * COS( onethird * ( s + six * pi ) ) - w

!  one real root

        ELSE
          d = SQRT( d ) ; u1 = q + d ; u2 = q - d
          nroots = 1
          root1 = SIGN( ABS( u1 ) ** onethird, u1 ) +                          &
                  SIGN( ABS( u2 ) ** onethird, u2 ) - w
        END IF

!  4. Compute the roots as the eigenvalues of the relevant compainion matrix

      ELSE
        H( 1, 1 ) = zero ; H( 2, 1 ) = one ; H( 3, 1 ) = zero
        H( 1, 2 ) = zero ; H( 2, 2 ) = zero ; H( 3, 2 ) = one
        H( 1, 3 ) = - a0 / a3 ; H( 2, 3 ) = - a1 / a3 ; H( 3, 3 ) = - a2 / a3
        CALL HSEQR( 'E', 'N', 3, 1, 3, H, 3, ER, EI, ZZ, 1, WORK, 33, info )
        IF ( info /= 0 ) THEN
          IF ( debug ) WRITE( out,                                             &
         &   "( ' ** error return ', I0, ' from HSEQR in ROOTS_cubic' )" ) info
          nroots = 0
          RETURN
        END IF

!  count and record the roots

        nroots = COUNT( ABS( EI ) <= epsmch )
        IF ( nroots == 1 ) THEN
          IF (  ABS( EI( 1 ) ) <= epsmch ) THEN
            root1 = ER( 1 )
          ELSE IF (  ABS( EI( 2 ) ) <= epsmch ) THEN
            root1 = ER( 2 )
          ELSE
            root1 = ER( 3 )
          END IF
        ELSE
          root1 = ER( 1 ) ;  root2 = ER( 2 ) ;  root3 = ER( 3 )
        END IF
      END IF

!  reorder the roots

  900 CONTINUE
      IF ( nroots == 3 ) THEN
        IF ( root1 > root2 ) THEN
          a = root2 ; root2 = root1 ; root1 = a
        END IF
        IF ( root2 > root3 ) THEN
          a = root3
          IF ( root1 > root3 ) THEN
            a = root1 ; root1 = root3
          END IF
          root3 = root2 ; root2 = a
        END IF
        IF ( debug ) WRITE( out, "( ' 3 real roots ' )" )
      ELSE IF ( nroots == 2 ) THEN
        IF ( debug ) WRITE( out, "( ' 2 real roots ' )" )
      ELSE
        IF ( debug ) WRITE( out, "( ' 1 real root ' )" )
      END IF

!  perfom a Newton iteration to ensure that the roots are accurate

      p = ( ( a3 * root1 + a2 ) * root1 + a1 ) * root1 + a0
      pprime = ( three * a3 * root1 + two * a2 ) * root1 + a1
      IF ( pprime /= zero ) THEN
        IF ( debug ) WRITE( out, 2000 ) 1, root1, p, - p / pprime
        root1 = root1 - p / pprime
        p = ( ( a3 * root1 + a2 ) * root1 + a1 ) * root1 + a0
      END IF
      IF ( debug ) WRITE( out, 2010 ) 1, root1, p

      IF ( nroots == 3 ) THEN
        p = ( ( a3 * root2 + a2 ) * root2 + a1 ) * root2 + a0
        pprime = ( three * a3 * root2 + two * a2 ) * root2 + a1
        IF ( pprime /= zero ) THEN
          IF ( debug ) WRITE( out, 2000 ) 2, root2, p, - p / pprime
          root2 = root2 - p / pprime
          p = ( ( a3 * root2 + a2 ) * root2 + a1 ) * root2 + a0
        END IF
        IF ( debug ) WRITE( out, 2010 ) 2, root2, p

        p = ( ( a3 * root3 + a2 ) * root3 + a1 ) * root3 + a0
        pprime = ( three * a3 * root3 + two * a2 ) * root3 + a1
        IF ( pprime /= zero ) THEN
          IF ( debug ) WRITE( out, 2000 ) 3, root3, p, - p / pprime
          root3 = root3 - p / pprime
          p = ( ( a3 * root3 + a2 ) * root3 + a1 ) * root3 + a0
        END IF
        IF ( debug ) WRITE( out, 2010 ) 3, root3, p
      END IF

      RETURN

!  Non-executable statements

 2000 FORMAT( ' root ', I1, ': value = ', ES12.4, ' cubic = ', ES12.4,         &
              ' delta = ', ES12.4 )
 2010 FORMAT( ' root ', I1, ': value = ', ES12.4, ' cubic = ', ES12.4 )


!  End of subroutine ROOTS_cubic

      END SUBROUTINE ROOTS_cubic

!-*-*-*-*-*-*-   R O O T S _ q u a r t i c   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE ROOTS_quartic( a0, a1, a2, a3, a4, tol, nroots, root1, root2, &
                                root3, root4, debug )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the number and values of real roots of the quartic equation
!
!        a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0 = 0
!
!  where a0, a1, a2, a3 and a4 are real
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( OUT ) :: nroots
      REAL ( KIND = wp ), INTENT( IN ) :: a4, a3, a2, a1, a0, tol
      REAL ( KIND = wp ), INTENT( OUT ) :: root1, root2, root3, root4
      LOGICAL, INTENT( IN ) :: debug

!  Local variables

      INTEGER :: type_roots, nrootsc
      REAL ( KIND = wp ) :: a, alpha, b, beta, c, d, delta, gamma, r
      REAL ( KIND = wp ) :: x1, xm, xmd, xn, xnd
      REAL ( KIND = wp ) :: d3, d2, d1, d0, b4, b3, b2, b1
      REAL ( KIND = wp ) :: rootc1, rootc2, rootc3, p, pprime

!  Check to see if the quartic is actually a cubic

      IF ( a4 == zero ) THEN
        CALL ROOTS_cubic( a0, a1, a2, a3, tol, nroots, root1, root2, root3,    &
                          debug )
        root4 = infinity
        RETURN
      END IF

!  Use Ferrari's algorithm

!  Initialize

      nroots = 0
      b1 = a3 / a4
      b2 = a2 / a4
      b3 = a1 / a4
      b4 = a0 / a4
      d3 = one
      d2 =  - b2
      d1 = b1 * b3 - four * b4
      d0 = b4 * ( four * b2 - b1 * b1 ) - b3 * b3

!  Compute the roots of the auxiliary cubic

      CALL ROOTS_cubic( d0, d1, d2, d3, tol, nrootsc, rootc1, rootc2, rootc3, &
                        debug )
      IF ( nrootsc > 1 ) rootc1 = rootc3
      x1 = b1 * b1 * quarter - b2 + rootc1
      IF ( x1 < zero ) THEN
        xmd = SQRT( - x1 )
        xnd = quarter * ( two * b3 - b1 * rootc1 ) / xmd
        alpha = half * b1 * b1 - rootc1 - b2
        beta = four * xnd - b1 * xmd
        r = SQRT( alpha * alpha + beta * beta )
        gamma = SQRT( half * ( alpha + r ) )
        IF ( gamma == zero ) THEN
          delta = SQRT( - alpha )
        ELSE
          delta = beta * half / gamma
        END IF
        root1 = half * ( - half * b1 + gamma )
        root2 = half * ( xmd + delta )
        root3 = half * ( - half * b1 - gamma )
        root4 = half * ( xmd - delta )
        GO TO 900
      END IF
      IF ( x1 /= zero ) THEN
        xm = SQRT( x1 )
        xn = quarter * ( b1 * rootc1 - two * b3 ) / xm
      ELSE
        xm = zero
        xn = SQRT( quarter * rootc1 * rootc1 - b4 )
      END IF
      alpha = half * b1 * b1 - rootc1 - b2
      beta = four * xn - b1 * xm
      gamma = alpha + beta
      delta = alpha - beta
      a = - half * b1

!  Compute how many real roots there are

      type_roots = 1
      IF ( gamma >= zero ) THEN
        nroots = nroots + 2
        type_roots = 0
        gamma = SQRT( gamma )
      ELSE
        gamma = SQRT( - gamma )
      END IF
      IF ( delta >= zero ) THEN
        nroots = nroots + 2
        delta = SQRT( delta )
      ELSE
        delta = SQRT( - delta )
      END IF
      type_roots = nroots + type_roots

!  Two real roots

      IF ( type_roots == 3 ) THEN
        root1 = half * ( a - xm - delta )
        root2 = half * ( a - xm + delta )
        root3 = half * ( a + xm )
        root4 = half * gamma
        GO TO 900
      ELSE IF ( type_roots /= 4 ) THEN
        IF ( type_roots == 2 ) THEN
          root1 = half * ( a + xm - gamma )
          root2 = half * ( a + xm + gamma )
        ELSE

!  No real roots

          root1 = half * ( a + xm )
          root2 = half * gamma
        END IF
        root3 = half * ( a - xm ) * half
        root4 = half * delta
        GO TO 900
      END IF

!  Four real roots

      b = half * ( a + xm + gamma )
      d = half * ( a - xm + delta )
      c = half * ( a - xm - delta )
      a = half * ( a + xm - gamma )

!  Sort the roots

      root1 = MIN( a, b, c, d )
      root4 = MAX( a, b, c, d )

      IF ( a == root1 ) THEN
        root2 = MIN( b, c, d )
      ELSE IF ( b == root1 ) THEN
        root2 = MIN( a, c, d )
      ELSE IF ( c == root1 ) THEN
        root2 = MIN( a, b, d )
      ELSE
        root2 = MIN( a, b, c )
      END IF

      IF ( a == root4 ) THEN
        root3 = MAX( b, c, d )
      ELSE IF ( b == root4 ) THEN
        root3 = MAX( a, c, d )
      ELSE IF ( c == root4 ) THEN
        root3 = MAX( a, b, d )
      ELSE
        root3 = MAX( a, b, c )
      END IF

  900 CONTINUE

!  Perfom a Newton iteration to ensure that the roots are accurate

      IF ( debug ) THEN
        IF ( nroots == 0 ) THEN
          WRITE( out, "( ' no real roots ' )" )
        ELSE IF ( nroots == 2 ) THEN
          WRITE( out, "( ' 2 real roots ' )" )
        ELSE IF ( nroots == 4 ) THEN
          WRITE( out, "( ' 4 real roots ' )" )
        END IF
      END IF
      IF ( nroots == 0 ) RETURN

      p = ( ( ( a4 * root1 + a3 ) * root1 + a2 ) * root1 + a1 ) * root1 + a0
      pprime = ( ( four * a4 * root1 + three * a3 ) * root1 + two * a2 )       &
                 * root1 + a1
      IF ( pprime /= zero ) THEN
        IF ( debug ) WRITE( out, 2000 ) 1, root1, p, - p / pprime
        root1 = root1 - p / pprime
        p = ( ( ( a4 * root1 + a3 ) * root1 + a2 ) * root1 + a1 ) * root1 + a0
      END IF
      IF ( debug ) WRITE( out, 2010 ) 1, root1, p

      p = ( ( ( a4 * root2 + a3 ) * root2 + a2 ) * root2 + a1 ) * root2 + a0
      pprime = ( ( four * a4 * root2 + three * a3 ) * root2 + two * a2 )       &
                 * root2 + a1
      IF ( pprime /= zero ) THEN
        IF ( debug ) WRITE( out, 2000 ) 2, root2, p, - p / pprime
        root2 = root2 - p / pprime
        p = ( ( ( a4 * root2 + a3 ) * root2 + a2 ) * root2 + a1 ) * root2 + a0
      END IF
      IF ( debug ) WRITE( out, 2010 ) 2, root2, p

      IF ( nroots == 4 ) THEN
        p = ( ( ( a4 * root3 + a3 ) * root3 + a2 ) * root3 + a1 ) * root3 + a0
        pprime = ( ( four * a4 * root3 + three * a3 ) * root3 + two * a2 )     &
                   * root3 + a1
        IF ( pprime /= zero ) THEN
          IF ( debug ) WRITE( out, 2000 ) 3, root3, p, - p / pprime
          root3 = root3 - p / pprime
          p = ( ( ( a4 * root3 + a3 ) * root3 + a2 ) * root3 + a1 ) * root3 + a0
        END IF
        IF ( debug ) WRITE( out, 2010 ) 3, root3, p

        p = ( ( ( a4 * root4 + a3 ) * root4 + a2 ) * root4 + a1 ) * root4 + a0
        pprime = ( ( four * a4 * root4 + three * a3 ) * root4 + two * a2 )     &
                   * root4 + a1
        IF ( pprime /= zero ) THEN
          IF ( debug ) WRITE( out, 2000 ) 4, root4, p, - p / pprime
          root4 = root4 - p / pprime
          p = ( ( ( a4 * root4 + a3 ) * root4 + a2 ) * root4 + a1 ) * root4 + a0
        END IF
        IF ( debug ) WRITE( out, 2010 ) 4, root4, p
      END IF

      RETURN

!  Non-executable statements

 2000 FORMAT( ' root ', I1, ': value = ', ES12.4, ' quartic = ', ES12.4,       &
              ' delta = ', ES12.4 )
 2010 FORMAT( ' root ', I1, ': value = ', ES12.4, ' quartic = ', ES12.4 )

!  End of subroutine ROOTS_quartic

      END SUBROUTINE ROOTS_quartic

!-*-*-*-*-   R O O T S _ p o l y n o m i a l   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE ROOTS_polynomial( A, n, ROOT, control, inform, data, E )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

! Find the roots of a real polynomial, with error bounds.

!  This is a fortran 90 version of PA17 from HSL

! A(i) must be set to hold the coefficient of x**(i), i=0,1,..., n. It is
!  changed during the execution, but is restored before return
! n must be set to the degree of the polynomial.
! ROOT is used to return the roots. The dummy value HUGE(1.0) is returned for
!   each infinite root corresponding to a zero leading coefficient
! control is a structure of type ROOTS_control_type holding control parameters
! inform is a structure of type ROOTS_inform_type holding information parameters
! data is a structure of type ROOTS_data_type holding private data
! E is an optional argument that if present must be set by the user to
!   error bounds for the coefficients, or to zero if these are accurate
!   to machine precision. On return, the first n locations will have been set
!   to approximate bounds on the moduli of the errors in the roots

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 0 : n ) :: A
      COMPLEX ( KIND = wcp ), INTENT( OUT ), DIMENSION( n ) :: ROOT
      REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL, DIMENSION( 0 : n ) :: E
      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ROOTS_control_type ), INTENT( IN ) :: control
      TYPE ( ROOTS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: m
      CHARACTER ( LEN = 80 ) :: array_name

!  allocate appropriate workspace

      IF ( n > data%w_max ) THEN
        data%w_max = n
        array_name = 'roots: data%W'
        CALL SPACE_resize_array( - n, n, data%W,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  Calculate the roots

      CALL ROOTS_polynomial_main( A, n, ROOT, data%W( 0 : n ) )

!  Calculate the error bounds. Check for zero leading coefficients

      IF ( PRESENT( E ) ) THEN
        DO m = n, 1, - 1
          IF ( ABS( A( m ) ) > zero ) EXIT
        END DO

!  Error bounds are required

        IF ( m >= 1 ) THEN
          IF ( n > data%ig_max ) THEN
            data%ig_max = n
            array_name = 'roots: data%IG'
            CALL SPACE_resize_array( n, data%IG,                               &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN
          END IF
          CALL ROOTS_polynomial_error_bound( A, m, ROOT, E, data%W, data%IG,   &
                                             data%W( 1 : m ) )
        END IF

!  Set dummy error bounds corresponding to infinite roots

        E( m + 1 : n ) = infinity
      END IF
      inform%status = GALAHAD_ok
      RETURN

      CONTAINS

        SUBROUTINE ROOTS_polynomial_main( A, N, ROOT, D )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

! Find the roots of a real polynomial
!
! n must be set to the degree of the polynomial
! A(i) must be set to hold the coefficient of x**(i),
!   i=0,1,..., n. It is changed during the execution,
!   but is restored before return.
! ROOT(i), i=1,2,..., n, holds the roots on return
! D is used as a workspace array

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

        INTEGER, INTENT( IN ) :: n
        REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 0 : n ) :: A
        REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 0 : n ) :: D
        COMPLEX ( KIND = wcp ), INTENT( OUT ), DIMENSION( n ) :: ROOT

!  Local variables

! When (n-nn) roots are found, D(0), ..., D(nn) will hold the coefficients of
!  the deflated polynomial (backwards), ROOT(nn + 1),..,ROOT(n) will hold the
!  roots found, A(0),...,A(nn - 1) will hold the coefficients of the derivative
!  of the deflated polynomial, and ROOT(1),...,ROOT(nn),A(nn),...,A(n) hold the
!  coefficients of the original polynomial (forwards)
! div2 .TRUE. if the search with step-lengths dz,dz/2,...is in use and .FALSE.
!  if the search with steps dz,2*dz,...is in use
! dz Tentative step
! fc Temporary variable used in the test for deflation
! fd Temporary variable used in the test for deflation
! fw f(w)
! fz f(z)
! f1z f'(z)
! f1z0 f'(z0)
! f2 Approx. to abs(f''(z))
! g Temporary variable used in the test for deflation
! iter Index for main iteration loop
! pp Temporary variable used in the complex deflation
! qq Temporary variable used in the complex deflation
! r Abs(DZ)
! rr Temporary variable used in the complex deflation
! afz0 abs(f(Z0))
! rz Real(Z)
! ss Temporary variable used in the complex deflation
! stage1 is .TRUE. during stage 1 of the iteration and .FALSE during stage 2
! tt Temporary variable used in the complex deflation
! u Scale factor for deflated polynomial. Also temporary variable
! u1 Largest absolute value of a coefficient of the deflated polynomial
! u2 Smallest absolute value of a nonzero coefficient of the deflated
!  polynomial
! w Current point of line search
! z Current point
! zt Last tentative point
! z0 Last point

        INTEGER :: i, iter, k, kk, loop, nn
        REAL ( KIND = wp ) :: afw, afz, afz0, afzmin, arz, blog, f2, fc, fd, g
        REAL ( KIND = wp ) :: pp, qq, r, r0, rr, rz, ss, tt, u, u1, u2
        COMPLEX ( KIND = wcp ) :: dz, f1z, f1z0, fw, fz, w, z, z0, zt
        LOGICAL :: div2, stage1

!  Compute the logarithm of the arithmetic base

        blog = LOG( base )

! Test for zeros at infinity

        DO nn = n, 1, - 1
          IF (ABS( A( nn ) ) > zero ) EXIT
        END DO

! nn is the order of the deflated polynomial

        ROOT( nn + 1 : n ) = infinity
        IF ( nn < 1 ) RETURN

! Store original polynomial backwards in D and forwards in ROOT

        DO I = 1, nn
          D( i - 1 ) = A( nn + 1 - i )
          ROOT( i ) = A( i - 1 )
        END DO
        D( nn ) = A( 0 )

! Main loop

        DO loop = 1, n
          IF ( nn == 1 ) THEN
            A( 0 ) = REAL( ROOT( 1 ) )
            ROOT( 1 ) = - D( 1 ) / D( 0 )
            RETURN
          END IF

! Scale the coefficients

          u1 = zero
          u2 = infinity
          DO k = 0, nn
            u = ABS( D( k ) )
            u1 = MAX( u, u1 )
            IF ( u > zero ) u2 = MIN( u, u2 )
          END DO
          i = - INT( ( LOG( u1 ) + LOG( u2 ) ) / ( two * blog ) )
          u = base ** i
          DO k = 0, nn - 1
            D( k ) = D( k ) * u
            A( k ) = D( k ) * ( nn - k )
          END DO
          D( nn ) = D( nn ) * u

! Set initial iterates and quantity used in the convergence test

        z0 = zero
        afz0 = ABS( D( nn ) )
        afzmin = afz0 * nn * 16.0_wp * epsmch

! Test for roots at the origin

          IF ( afzmin <= smallest ) THEN
            A( nn - 1 ) = REAL( ROOT( nn ) )
            ROOT( nn ) = zero
            nn = nn - 1
            IF ( nn == 0 ) RETURN
            CYCLE
          END IF

          zt = z0
          r0 = infinity
          DO k = 0, nn - 1
            u = ABS( D( K ) )
            IF ( u == zero ) EXIT
            u = LOG( afz0 / u ) / ( nn - k )
            r0 = MIN( u, r0 )
          END DO
          r0 = EXP( r0 ) / 2
          f1z0 = D( nn - 1 )
          IF ( ABS( f1z0 ) == zero ) THEN
            z = r0
          ELSE
            z = - D( nn ) / f1z0
            z = r0 * ( z / ABS( z ) )
          END IF
          dz = z

!  evaluate the polynomial
!         fz = ROOTS_polynomial_val( z, nn, D )
          fz = D( 0 )
          DO k = 1, nn
            fz = fz * z + D( k )
          END DO

          afz = ABS( fz )

          DO iter = 1, 1000
            IF ( iter == 1 .OR. afz < afz0 ) THEN

!   First iteration or iteration following a successful one.
!   Calculate the tentative step DZ and whether in stage 1

!  evaluate the polynomial
!             f1z = ROOTS_polynomial_val( z, nn - 1, A )
              f1z = A( 0 )
              DO k = 1, nn - 1
                f1z = f1z * z + A( k )
              END DO

              u = ABS( f1z )
              IF ( u /= zero ) THEN
                dz = - fz / f1z
                f2 = ABS( f1z0 - f1z ) / ABS( z0 - z )
                stage1 = afz * f2 / u > u * half .OR. z /= zt
                r = ABS( dz )
                IF ( r > r0 * three ) dz = dz * ( 1.8_wp, 2.4_wp ) * r0 / r
              ELSE
                dz = dz * ( 1.8_wp, 2.4_wp )
                stage1 = .TRUE.
              END IF
              f1z0 = f1z
            END IF

!  Find the next point in the iteration

            z0 = z
            afz0 = afz
            z = z0 + dz
            w = z

!  evaluate the polynomial
!           fz = ROOTS_polynomial_val( z, nn, D )
            fz = D( 0 )
            DO k = 1, nn
              fz = fz * z + D( k )
            END DO

            afz = ABS( fz )
            zt = z
            IF ( stage1 ) THEN

!  Beginning of stage 1 search

              div2 = afz >= afz0
              DO k = 1, nn
                IF ( div2 ) THEN
                  IF ( k <= 2 ) THEN
                    w = ( w + z0 ) * half
                  ELSE IF ( k == 3 ) THEN
                    w = z0 + dz * ( 0.15_wp, 0.2_wp )
                  ELSE
                    EXIT
                  END IF
                ELSE
                  w = w + dz
                END IF

!  evaluate the polynomial
!               fw = ROOTS_polynomial_val( w, nn, D )
                fw = D( 0 )
                DO kk = 1, nn
                  fw = fw * w + D( kk )
                END DO

                afw = ABS( fw )
                IF ( afw >= afz ) EXIT
                afz = afw
                fz = fw
                z = w
              END DO
            END IF
            r0 = ABS( z0 - z )

! Convergence test

            IF ( afz >= afz0 ) z = z0
            IF ( r0 < epsmch * ABS( z ) ) EXIT
            IF ( afz >= afz0 ) THEN
              afz = afz0
              IF ( afz <= afzmin ) EXIT

!  Step unsuccessful - halve the tentative step and change its direction

              dz = dz * ( - 0.3_wp, - 0.4_wp )
            END IF
          END DO  ! end of loop iter

! Deflate, store root, restore coefficient of original polynomial and reduce nn

          A( nn - 1 ) = REAL( ROOT( nn ) )
          rz = REAL( z )
          arz = ABS( rz )
          g = zero
          fc = D( 0 )
          IF ( AIMAG( z ) /= zero ) THEN
            DO k = 1, nn
              fd = fc * rz + D( k )
              g = arz * ( g + ABS( fc ) ) + ABS( fd )
              FC = FD
            END DO
            IF ( ABS( FC ) > two * epsmch * g + MIN( afz, afz0 ) ) GO TO 10
          END IF
          ROOT( NN ) = rz
          DO K = 1, nn - 1
            D( K ) = D( K - 1 ) * rz + D( K )
          END DO
          nn = nn - 1
          IF ( nn == 0 ) RETURN
          CYCLE

! Deflation with a pair of complex conjugate roots

  10      CONTINUE
          ROOT( nn ) = z
          rr = zero
          ss = zero
          pp = - rz - rz
          qq = rz ** 2 + AIMAG( z ) ** 2
          nn = nn - 1
          DO k = 0, nn - 1
            tt = D( k ) - pp * rr - qq * ss
            D( k ) = tt
            ss = rr
            rr = tt
          END DO
          A( nn - 1 ) = REAL( ROOT( nn ) )
          ROOT( nn ) = CONJG( z )
          nn = nn - 1
          IF ( nn == 0 ) RETURN
        END DO  ! end of main loop

        RETURN

!  End of ROOTS_polynomial_main

        END SUBROUTINE ROOTS_polynomial_main

        SUBROUTINE ROOTS_polynomial_error_bound( A, n, ROOT, E, W, IG, CR )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Find error bounds on computed roots

! A( i ) must be set to hold the coefficient of x**( i ), i=0,1,...,n
! n must be set to the degree of the polynomial
! ROOT( i ), i=1,2,...,n, must be set to the computed roots
! E must be set by the user to error bounds for the coefficients, or to
!   zero if these are accurate to machine precision. On return, the
!   first N locations will have been set to approximate bounds on the
!   moduli of the errors in the roots
! W is a workspace array used to hold the coefficients
!   of the polynomial formed from the calculated roots and then
!   the coefficients of the error polynomial
! IG is a workspace array used to link together the roots
!   in a group. The first is IG1, IG( k ) follows k and the last is i
!   Other roots have IG( k )=0
! CR is a workspace array used to hold distances from root i to the rest

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

        INTEGER, INTENT( IN ) :: n
        INTEGER, INTENT( OUT ), DIMENSION( n ) :: IG
        REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 0 : n ) :: A
        REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 0 : n ) :: E
        REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: CR
        REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 0 : n ) :: W
        COMPLEX ( KIND = wcp ), INTENT( IN ), DIMENSION( n ) :: ROOT

!  Local variables

! btm   Denominator of the expression (3.9) of the report R 7986
! dmax  Maximum distance to a root in the group
! dist  Distance from root I to root K
! dmin  Minimum distance to a root not in the group
! i     Index of root under test
! ig1   First root of the group
! l     Nearset root not in the group
! loop  DO index
! loopb Index of bisection loop
! m     Number of roots in the group
! oldr  Old value of rad
! prod  The product in inequality (3.8) of the report R 7986
! r     Current root
! rad   Radius for Rouche circle.
! rl    Lower bound for RAD in bisection loop
! rm    Upper bound for RAD in bisection loop
! rp    Temporary variable used to form the polynomial
! rs    Temporary variable used to form the polynomial
! r     Real(r)
! s     Abs(r)+RAD
! tb    top/btm
! top   Numerator of the expression (3.9) of the report R 7986 polynomial

        INTEGER :: i, ig1, ii, j, k, l, loop, loopb, loopr, m
        REAL ( KIND = wp ) :: btm, dist, dmax, dmin, fact, oldr, prod, rad
        REAL ( KIND = wp ) :: rl, rm, rp, rr, rs, s, tb, top
        COMPLEX ( KIND = wcp ) :: r

! Multiply out the polynomial formed from the calculated roots

        W( 1 : n ) = zero
        W( 0 ) = A( n )
        i = 1
        DO k = 1, n
          IF ( i > n ) EXIT
          r = ROOT( n + 1 - i )
          rr = REAL( r )
          IF ( AIMAG( r ) == zero ) THEN
            DO j = i, 1, - 1
              W( j ) = W( j ) - rr * W( j - 1 )
            END DO
          ELSE
            rs = rr + rr
            rp = rr ** 2 + AIMAG( r ) ** 2
            DO j = i, 0, - 1
              W( j + 1 ) = W( j + 1 ) - rs * W( j )
              IF ( j /= 0 ) W( j + 1 ) = W( j + 1 ) + rp * W( j - 1 )
            END DO
            i = i + 1
          END IF
          i = i + 1
        END DO

! Find coefficients of error polynomial

        DO i = 0, n
          ii = n - i
          W( i ) = ABS( A( ii ) - W( i ) ) +                                   &
            MAX( E( ii ), ABS( A( ii ) ) * epsmch )
        END DO

! Initialize array IG, which records the grouping

        IG = 0

! Main error-bounding loop. It finds a bound for root i

        DO i = 1, n
          IG( i ) = - 1
          ig1 = i
          dmax = zero

!        Find distances to other roots

          R = ROOT( i )
          DO k = 1, n
            CR( k ) = ABS( ROOT( k ) - r )
          END DO

          DO m = 1, n
            fact = 1.05_wp ** ( REAL( m ) / REAL( n ) )
            dmin = infinity
            rad = dmax
            DO loopr = 1, 10

!           Test Rouche condition with radius rad

              top = W( 0 )
              prod = one
              btm = ABS( A( n ) )
              s = ABS( r ) + rad
              IF ( s > one ) top = W( n )
              DO k = 1, n
                IF ( s > one ) THEN
                  top = top / s + W( n - k )
                  btm = btm / s
                ELSE
                  top = s * top + W( k )
                END IF
                dist = CR( K )
                IF ( IG( k ) == 0 ) THEN
                  btm = btm * ( dist - rad )
                  IF ( dmin > dist ) THEN ; dmin = dist ; l = k ; END IF
                ELSE
                  prod = prod * ( rad - dist )
                END IF
              END DO
              IF ( btm == zero .OR. rad >= dmin ) EXIT
              tb = ABS( top / btm )
              IF ( prod >= tb ) GO TO 20

!           Find a new trial radius

              oldr = rad
              rl = rad
              rad = dmax + 1.1_wp * tb
              IF ( m > 1 ) THEN
                rm = dmax + 1.1_wp * tb ** ( one / REAL( m ) )

!           Bisection loop

                DO loopb = 1, 1000
                  rad = ( rl + rm ) / two
                  prod = one
                  k = ig1
                  DO loop = 1, n
                    prod = prod * ( rad - CR( k ) )
                    k = IG( k )
                    IF ( k <= 0 ) EXIT
                  END DO
                  IF ( prod < 1.10_wp ** m * tb ) THEN
                    IF ( rad >= dmin ) GO TO 10
                    IF ( prod > 1.05_wp * tb ) EXIT
                    rl = rad
                  ELSE
                    rm = rad
                  END IF
                END DO
              END IF
              IF ( rad >= dmin ) EXIT
              IF ( ( ABS( r ) + rad ) * ( ( dmin - oldr ) / ( dmin - rad ) )   &
                     <= fact * s ) GO TO 20
            END DO

!  Add root to group, unless all roots already in group

   10       CONTINUE  ! end of loop on loopr
            IF ( m == n ) THEN
              rad = infinity
              EXIT
            END IF
            IG( l ) = ig1
            ig1 = l
            dmax = dmin
          END DO  ! end of loop on m

!  Store error bound and reset IG

  20      CONTINUE
          E( i ) = rad
          DO loop = 1, n
            k = ig1
            ig1 = IG( ig1 )
            IG( k ) = 0
            IF ( k == i ) EXIT
          END DO
        END DO  ! end of main loop

        RETURN

!  End of ROOTS_polynomial_error_bound

        END SUBROUTINE ROOTS_polynomial_error_bound

!       FUNCTION ROOTS_polynomial_val( z, n, A )
!       COMPLEX ( KIND = wcp ) :: ROOTS_polynomial_val

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Calculate the polynomial value p(z)

! z  must be set to the point of evaluation
! n must be set to the degree of the polynomial
! A(i) contains the coefficient of x ** (n-i), i = 0, 1, ..., n, in p(x)

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

!       INTEGER, INTENT( IN ) :: n
!       COMPLEX ( KIND = wcp ), INTENT( IN ) :: Z
!       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 0 : n ) :: A

!  Local variables

!       INTEGER :: k
!       COMPLEX ( KIND = wcp ) :: fz

!       fz = A( 0 )
!       DO k = 1, n
!         fz = fz * z + A( k )
!       END DO
!       ROOTS_polynomial_val = fz

!       RETURN

!  End of ROOTS_polynomial_val

!       END FUNCTION ROOTS_polynomial_val

!  End of ROOTS_polynomial

      END SUBROUTINE ROOTS_polynomial

!-*-*-*-*-*-*-   R O O T _ I N _ I N T E R V A L   F U N C T I O N   -*-*-*-*-

      FUNCTION ROOTS_smallest_root_in_interval( C, a, b, data, control, inform )
      REAL ( KIND = wp ) :: ROOTS_smallest_root_in_interval

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Calculate the smallest real root of a polynomial p_n(x) in the
!  interval (a,b]. Return the value b if there is no such root.

!  n must be set to the polynomial degree
!  C(i) contains the coefficient of x ** i, i = 0, 1, ..., n, in p_n(x)
!  a and b are the interval bounds (a,b]
!  data, control, inform - see preamble

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( IN ) :: a, b
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 0 : ) :: C
      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ROOTS_control_type ), INTENT( IN ) :: control
      TYPE ( ROOTS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, im1, im2, ip1, j, n, nm1, nm2, nonzero, roots_in_int
      INTEGER :: ii, it, i_zero, lwork, nroots, n_roots_a, n_roots_b, nn, np1
      INTEGER :: var_low, var_up, var_z
      REAL ( KIND = wp ) :: smallest_root_in_interval, z, p_z, dp_z, teeny
      REAL ( KIND = wp ) :: low, up, f_low, f_up, df_low, df_up
      REAL ( KIND = wp ) :: c_max, root1, root2, root3, q_0, q_1
      LOGICAL :: debug
      LOGICAL, PARAMETER :: get_roots_ab = .FALSE.
      CHARACTER ( LEN = 80 ) :: array_name

      INTEGER, PARAMETER :: itmax = 200
!     REAL ( KIND = wp ) :: growth = ten ** 8

      inform%status = 0
      ROOTS_smallest_root_in_interval = zero
!     debug = .TRUE.
      debug = control%out > 0 .AND. control%print_level > 0

!  compute the degree of the polynomial

      c_max = MAXVAL( ABS( C ) )
      teeny =  control%zero_coef * c_max
      DO i = 0, UBOUND( C, 1 )
!       IF ( ABS( C( i ) ) >= control%zero_coef ) nn = i
        IF ( ABS( C( i ) ) >= teeny ) nn = i
!       IF ( ABS( C( i ) ) /= zero ) nn = i
!       if ( debug ) write(6,*) ABS( C( i ) ), teeny, n
      END DO
      IF ( debug ) write( control%out, "( ' order ', I0 )" ) nn

!  count (and subsequently remove) leading zeros

        DO i_zero = 0, nn
          IF ( C( i_zero ) /= zero ) EXIT
        END DO
        n = nn - i_zero

!  special cases

      SELECT CASE( n )

!  polynomials of degree 0

      CASE ( : 0 )
        smallest_root_in_interval = b
        IF ( debug ) write( control%out, 2000 ) b

!  linear polynomials

      CASE ( 1 )
        root1 = - C( i_zero ) / C( i_zero + 1 )
        IF ( root1 > a .AND. root1 <= b ) THEN
          smallest_root_in_interval = root1
        ELSE
          IF ( debug ) write( control%out, 2000 ) b
          smallest_root_in_interval = b
        END IF

!  quadratic polynomials

      CASE ( 2 )
        CALL ROOTS_quadratic( C( i_zero ), C(  i_zero + 1 ), C(  i_zero + 2 ), &
                              control%tol, nroots, root1, root2, debug )
        IF ( nroots == 2 ) THEN
          IF ( root1 > a .AND. root1 <= b ) THEN
            smallest_root_in_interval = root1
          ELSE IF ( root2 > a .AND. root2 <= b ) THEN
            smallest_root_in_interval = root2
          ELSE
            IF ( debug ) write( control%out, 2000 ) b
            smallest_root_in_interval = b
          END IF
        ELSE
          IF ( debug ) write( control%out, 2000 ) b
          smallest_root_in_interval = b
        END IF

!  cubic polynomials

      CASE ( 3 )
        CALL ROOTS_cubic( C(  i_zero ), C(  i_zero + 1 ), C(  i_zero + 2 ),    &
                          C(  i_zero + 3 ), control%tol,                       &
                          nroots, root1, root2, root3, debug )
        IF ( nroots == 3 ) THEN
          IF ( root1 > a .AND. root1 <= b ) THEN
            smallest_root_in_interval = root1
          ELSE IF ( root2 > a .AND. root2 <= b ) THEN
            smallest_root_in_interval = root2
          ELSE IF ( root3 > a .AND. root3 <= b ) THEN
            smallest_root_in_interval = root3
          ELSE
            IF ( debug ) write( control%out, 2000 ) b
            smallest_root_in_interval = b
          END IF
       ELSE
         IF ( root1 > a .AND. root1 <= b ) THEN
           smallest_root_in_interval = root1
         ELSE
           IF ( debug ) write( control%out, 2000 ) b
           smallest_root_in_interval = b
         END IF
       END IF

!  polynomials of degree four or more

      CASE DEFAULT
        nm1 = n - 1 ; nm2 = n - 2 ; np1 = n + 1

!  allocate appropriate workspace

        IF ( n > data%n_max ) THEN
          data%n_max = n

          array_name = 'roots: data%A_mat'
          CALL SPACE_resize_array( np1, np1, data%A_mat,                       &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'roots: data%RHS'
          CALL SPACE_resize_array( np1, 1, data%RHS,                           &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'roots: data%WORK'
          lwork = 2 * MAX( 1, np1, + MAX( 1, np1 ) )
          CALL SPACE_resize_array( lwork, data%WORK,                           &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'roots: data%S'
          CALL SPACE_resize_array( 0, n, data%S,                               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'roots: data%P'
          CALL SPACE_resize_array( 0, n, 0, n, data%P,                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'roots: data%CROOTS'
          CALL SPACE_resize_array( n, data%CROOTS,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

        END IF

!  given p_n(x), set p_{n-1}(x) = p_n'(x), and for i = n, ... , 2, compute
!
!     p_{i-2}(x) = - remainder( p_{i}(x), p_{i-1}(x))
!
!  using the Euclidean Algorithm. This proceeds as follows: if
!
!     p_{i}(x) = p_{0,i} + p_{1,i} x + .... + p_{i-1,i} x^{i-1} + p_{i,i} x^i
!
!  we seek a linear factor q(x) = q_0 + q_1 x so that
!
!     p_{i-2}(x) = p_{i}(x)  - q( x ) * p_{i-1}(x).
!
!  Thus we recover q_1 and q_0 from
!
!     p_{i-1,i-1} q_1 = p_{i,i}
!     p_{i-1,i-1} q_0 = p_{i-1,i} - p_{i-2,i-1} q_1
!
!  and then find
!
!     p_{0,i-2} = p_{0,i-1} q_0 - p_{0,i}
!     p_{j,i-2} = p_{j,i-1} q_0 + p_{j-1,i-1} q_1 - p_{j,i} for j = 1,..., i-2
!
!  The coefficients of p_i(x) are in column i of the array P

!  set p_n(x) as a scaled version of the input polynomial, deflated all
!  leading and trailing zero coefficients

!       data%P( 0 : n, n ) = C( i_zero : nn )
!       data%P( 0 : n, n ) = C( i_zero : nn ) / c_max
        data%P( 0 : n, n ) = C( i_zero : nn ) / C( i_zero )

!  set p_{n-1}(x) = p_n'(x)

        DO j = 1, n
          data%P( j - 1, nm1 ) = data%P( j, n ) * REAL( j, KIND = wp )
        END DO

! ------------------------------ not used at present --------------------------
! |                                                                           |
! v                                                                           v

!  see if there is a root at a

        IF ( get_roots_ab ) THEN
          n_roots_a = 0
          ii = n

!  evaluate p_n(a)

          p_z = data%P( n, n )
          DO j = nm1, 0, - 1
            p_z = p_z * a + data%P( j, n )
          END DO

!  calculate the multiplicity of the the root at a

          IF ( ABS( p_z ) <= control%zero_f ) THEN
            n_roots_a = n_roots_a + 1
            DO i = nm1, 1, - 1

!  exit if the (n-i-1)-st derivative is not zero

!  if the (n-i-1)-st derivative is zero, compute the (n-i)-th derivative value

              p_z = data%P( i, i )
              DO j = i - 1, 0, - 1
                p_z = p_z * a + data%P( j, i )
              END DO

              IF ( ABS( p_z ) > control%zero_f ) EXIT
              n_roots_a = n_roots_a + 1

!  compute the coefficients of the (n-i+1)-st derivative

              DO j = 1, i
                data%P( j - 1, i - 1 ) = data%P( j, i ) * REAL( j, KIND = wp )
              END DO
              ii = i
            END DO
          END IF

!  see if there is a root at b

          n_roots_b = 0

!  evaluate p_n(a)

          p_z = data%P( n, n )
          DO j = nm1, 0, - 1
            p_z = p_z * b + data%P( j, n )
          END DO

!  calculate the multiplicity of the the root at b

          IF ( ABS( p_z ) <= control%zero_f ) THEN
            n_roots_b = n_roots_b + 1
            DO i = nm1, 1, - 1

!  exit if the (n-i-1)-st derivative is not zero

!  if the (n-i-1)-st derivative is zero, compute the (n-i)-th derivative value

              p_z = data%P( i, i )
              DO j = i - 1, 0, - 1
                p_z = p_z * b + data%P( j, i )
              END DO

              IF ( ABS( p_z ) > control%zero_f ) EXIT
              n_roots_b = n_roots_b + 1

!  compute the coefficients of the (n-i+1)-st derivative

              IF ( i < ii ) THEN
                DO j = 1, i
                  data%P( j - 1, i - 1 ) = data%P( j, i ) * REAL( j, KIND = wp )
                END DO
              END IF
            END DO
          END IF
        END IF

! ^                                                                           ^
! |                                                                           |
! ------------------------------ not used at present --------------------------

! write(6, * ) 'p_', n, data%P( 0 : n, n )
! DO j = nm1, 0, - 1
!   write(6, * ) 'p_', j, data%P( 0 : j, j )
! END DO

!  now apply the Euclidean Algorithm

        DO i = n, 2, - 1
          im1 = i - 1
          im2 = i - 2

!  if the algorithm breaks down for numerical reasons, resort to our
!  general-purpose polynomial root finder ... with luck this should
!  be a rare event!

          IF ( data%P( im1, im1 ) == zero ) THEN
            IF ( debug ) write( control%out,                                   &
                "( ' call ROOTS_solve as a precaution against growth ' )" )

!  compute all the (complex) roots

            data%S( 0 : n ) = C( i_zero : nn )
            CALL ROOTS_polynomial( data%S( 0 : n ), n, data%CROOTS, control,   &
                                   inform, data )

!  extract the smallest real root in the desired interval

            smallest_root_in_interval = b
            DO j = 1, n
              IF ( AIMAG( data%CROOTS( j ) ) == zero .AND.                     &
                   REAL( data%CROOTS( j ) ) > a )                              &
                smallest_root_in_interval = MIN(                               &
                  REAL( data%CROOTS( j ), KIND = wp ),                         &
                  smallest_root_in_interval )
            END DO
            GO TO 990
          END IF

!  compute the coefficients of the linear factor q(x)

! ------------------------------ not used at present --------------------------
! |                                                                           |
! v                                                                           v

!  if growth is indicated, compute the coefficients by solving the
!  relevant linear system accurately

!         IF ( ABS( data%P( i, i ) ) > ABS( data%P( im1, im1 ) ) * growth ) THEN
          IF ( .FALSE. ) THEN
            IF ( debug ) write( control%out,                                   &
                "( ' precaution against growth of', ES12.4 )" )                &
               data%P( i, i ) / data%P( im1, im1 )
            ip1 = i + 1
            data%A_mat( : i, 1 ) = data%P( im1 : 0 : - 1, im1 )
            data%A_mat( ip1, 1 ) = zero
            data%A_mat( 1, 2 ) = zero
            data%A_mat( 2 : ip1, 2 ) = data%P( im1 : 0 : - 1, im1 )
            data%A_mat( : ip1, 3 : ip1 ) = zero
            DO j = 3, ip1
              data%A_mat( j, j ) = - one
            END DO
            data%RHS( : ip1, 1 ) =  data%P( i : 0 : - 1, i )

            CALL DGELS( 'N',  ip1, ip1, 1, data%A_mat( : np1, : ip1 ), np1,    &
                         data%RHS, np1, data%WORK, lwork, j )

            data%P( im2 : 0 : - 1, im2 ) =  data%RHS( 3 : ip1, 1 )
          ELSE
! ^                                                                           ^
! |                                                                           |
! ------------------------------ not used at present --------------------------

            q_1 = data%P( i, i ) / data%P( im1, im1 )
            q_0 = ( data%P( im1, i ) - data%P( im2 , im1 ) * q_1 ) /           &
                    data%P( im1, im1 )

!  compute the coefficients of the remainder p_{i-2}(x)

            data%P( 0, im2 ) = data%P( 0, im1 ) * q_0 - data%P( 0, i )
            DO j = 1, i - 2
              data%P( j, im2 ) = data%P( j, im1 ) * q_0 +                      &
                                 data%P( j - 1, im1 ) * q_1 - data%P( j, i )
            END DO
          END IF
        END DO

        IF ( debug ) THEN
          DO j = n, 2, - 1
            q_1 = data%P( j, j ) / data%P( j-1, j-1 )
            q_0 = ( data%P( j-1, j ) - data%P( j-2 , j-1 ) * q_1 ) /           &
                    data%P( j-1, j-1 )
            WRITE( control%out, "( ' q', 2ES24.16 )" ) q_1, q_0
          END DO

          DO j = n, 0, - 1
            WRITE( control%out,                                                &
              "( ' p_', I0, 2ES24.16, :, ( /, 4X, 3ES24.16 ) )" )              &
                 j, data%P( 0 : j, j )
          END DO
        END IF

!  let the variation, var(S_n(x)), be the number of sign changes in
!  the sequence
!
!     S_n(x) = { p_0(x), p_1(x), ... , p_n(x) }
!
!  after any zeros are removed. Then Sturm's theorem gives that
!
!     # real roots in the interval (a,b) = var(S_n(a) - var(S_n(b))

!  compute S_n(a)

        z = a ; low = a
        nonzero = - 1
        DO i = 0, nm2
          p_z = data%P( i, i )
          DO j = i - 1, 0, - 1
            p_z = p_z * z + data%P( j, i )
          END DO
          IF ( p_z /= zero ) THEN
            nonzero = nonzero + 1
            data%S( nonzero ) = p_z
          END IF
        END DO

!  special case: p_n'(a)

        df_low = data%P( nm1, nm1 )
        DO j = nm1 - 1, 0, - 1
          df_low = df_low * z + data%P( j, nm1 )
        END DO
        IF ( df_low /= zero ) THEN
          nonzero = nonzero + 1
          data%S( nonzero ) = df_low
        END IF

!  special case: p_n(a)

        f_low = data%P( n, n )
        DO j = nm1, 0, - 1
          f_low = f_low * z + data%P( j, n )
        END DO
        IF ( f_low /= zero ) THEN
          nonzero = nonzero + 1
          data%S( nonzero ) = f_low
        END IF

!  compute var(S_n(a)

        var_low = 0
        DO i = 0, nonzero - 1
          IF ( data%S( i ) * data%S( i + 1 ) < zero ) var_low = var_low + 1
        END DO

!  ------------------------------------------------------------
!  PHASE-1 loop to find an interval with only the required root
!  ------------------------------------------------------------

        IF ( debug ) write( control%out,*) ' phase 1'
        z = b

        DO it = 1, itmax

!  compute S_n(z)

          nonzero = - 1
          DO i = 0, nm2
            p_z = data%P( i, i )
            DO j = i - 1, 0, - 1
              p_z = p_z * z + data%P( j, i )
            END DO
            IF ( p_z /= zero ) THEN
              nonzero = nonzero + 1
              data%S( nonzero ) = p_z
            END IF
          END DO

!  special case: p_n'(z)

          dp_z = data%P( nm1, nm1 )
          DO j = nm2, 0, - 1
            dp_z = dp_z * z + data%P( j, nm1 )
          END DO
          IF ( dp_z /= zero ) THEN
            nonzero = nonzero + 1
            data%S( nonzero ) = dp_z
          END IF

!  special case: p_n(z)

          p_z = data%P( n, n )
          DO j = nm1, 0, - 1
            p_z = p_z * z + data%P( j, n )
          END DO
          IF ( p_z /= zero ) THEN
            nonzero = nonzero + 1
            data%S( nonzero ) = p_z
          END IF

!  compute var(S_n(z)

          var_z = 0
          DO i = 0, nonzero - 1
            IF ( data%S( i ) * data%S( i + 1 ) < zero ) var_z = var_z + 1
          END DO

!  compute the number of real roots in the interval (low,up)
!  using Sturm's theorem

!  check if the smallest root lies outside the initial interval

          IF ( low == a .AND. z == b ) THEN
            up = z
            f_up = p_z
            df_up = dp_z
            roots_in_int = var_low - var_z
            IF ( roots_in_int == 0 ) THEN
!write(6,*) up, f_up
              smallest_root_in_interval = b
              IF ( debug ) write( control%out, 2000 ) b
              GO TO 990

!  if the only roots are at the interval ends, exit

!           ELSE IF ( roots_in_int == n_roots_a + n_roots_b ) THEN
!             smallest_root_in_interval = b
!             IF ( debug ) write( control%out, 2000 ) b
!             RETURN

!  if there is only one root in [a,b], move to phase 2

            ELSE IF ( roots_in_int == 1 ) THEN
              EXIT
            END IF
            var_up = var_z
          ELSE
            roots_in_int = var_low - var_z

!  is the smallest root in [low,z] ...

            IF ( roots_in_int > 0 ) THEN
              up = z
              f_up = p_z
              df_up = dp_z

! (if it alone lies in [low,z], move to phase 2)

              IF ( roots_in_int == 1 ) EXIT
              var_up = var_z

!  ... or is it in [z,up] ?

            ELSE
              low = z
              f_low = p_z
              df_low = dp_z
              var_low = var_z
            END IF
          END IF

          roots_in_int = var_low - var_up
          IF ( debug ) write( control%out,                                     &
            "(' # roots in [', ES12.4, ', ', ES12.4, '] = ', I0 )" )           &
              low, up, roots_in_int

          IF ( up - low <= control%tol ) EXIT

!  bisect to produce two new intervals to test

          z = half * ( low + up )
!          IF ( it == itmax ) THEN
!            WRITE( control%out, * ) C
!  write(6,*) ' iteration bound stopping in ROOTS_smallest_root_in_interval'
!            STOP
!          END IF
        END DO

        IF ( debug ) THEN
          write( control%out,                                                  &
             "(' required root in [', ES24.16,', ', ES24.16,']')") low, up
          write( control%out, "(' f and df at ', ES12.4, ' = ', 2ES12.4 )")    &
            low, f_low, df_low
          write( control%out, "(' f and df at ', ES12.4, ' = ', 2ES12.4 )" )   &
            up, f_up, df_up
        END IF

!  -----------------------------------------------------------
!  we have located the desired interval - start PHASE 2 -
!  apply a safeguarded Newton method to find the required root
!  -----------------------------------------------------------

        IF ( debug ) write( control%out,*) ' phase 2'
        DO it = 1, itmax
          IF ( ABS( f_low ) < control%zero_f ) THEN
            smallest_root_in_interval = low
            EXIT
          ELSE IF ( ABS( f_up ) < control%zero_f ) THEN
            z = MAX( up * ( one - control%tol ), half * ( low  + up ) )
          ELSE
!           smallest_root_in_interval = up
!           EXIT

!  compute the Newton step from the currrent best point

            IF ( ABS( f_low ) <  ABS( f_up ) ) THEN
              z = low - f_low / df_low
            ELSE
              z = up - f_up / df_up
            END IF
!write(6,*) ' Newton ', z, low, up

!  if the Newton step is outside the interval, revert to bisection

            IF ( z <= low .OR. z >= up ) z = half * ( low  + up )
!           IF ( z <= low .OR. z >= up ) write(6,*) ' bisect '

!!  the root of the linear interpolant between the interval values
!!         z = - ( low * f_up - up * f_low ) / ( f_low - f_up )


!  compute the value and derivative at the new point

          END IF
          dp_z = data%P( nm1, nm1 )
          DO j = nm2, 0, - 1
            dp_z = dp_z * z + data%P( j, nm1 )
          END DO

          p_z = data%P( n, n )
          DO j = nm1, 0, - 1
            p_z = p_z * z + data%P( j, n )
          END DO

          IF ( debug ) write( control%out, "( ' low, up ', 2ES24.16 )" )       &
            low, up
          IF ( debug ) write( control%out, "( ' z, p_z, dp_z ', 3ES24.16 )" )  &
            z, p_z, dp_z
          smallest_root_in_interval = z

!  refine the interval

          IF ( p_z * f_low > zero ) THEN
            low = z
            f_low = p_z
            df_low = dp_z
          ELSE
            up = z
            f_up = p_z
            df_up = dp_z
          END IF
!          IF ( it == itmax ) THEN
!            WRITE( control%out, * ) C
!  write(6,*) ' iteration bound 2 stopping in ROOTS_smallest_root_in_interval'
!            STOP
!          END IF

!  check that the step makes progress

          IF ( ABS( up - low ) <= control%tol ) THEN
            smallest_root_in_interval = low
            EXIT
          END IF
        END DO
      END SELECT

  990 CONTINUE
!write(6,*) ' 990 out', smallest_root_in_interval
      ROOTS_smallest_root_in_interval = smallest_root_in_interval
      RETURN

!  non-executable statements

 2000 FORMAT( ' required root above ', ES12.4 )

!  End of ROOTS_smallest_root_in_interval

      END FUNCTION ROOTS_smallest_root_in_interval

!-*-*-*-   R O O T _ P O L Y N O M I A L _ V A L U E   F U N C T I O N  -*-*-*-

      FUNCTION ROOTS_polynomial_value( x, A )
      REAL ( KIND = wp ) :: ROOTS_polynomial_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Calculate the polynomial value p(x)

! z  must be set to the point of evaluation
! A(i) contains the coefficient of x ** i, i = 0, 1, ..., n, in p(x)

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( IN ) :: x
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 0 : ) :: A

!  Local variables

      INTEGER :: k, n
      REAL ( KIND = wp ) :: px

      n = UBOUND( A, 1 )
      px = A( n )
      DO k = n - 1, 0, - 1
        px = px * x + A( k )
      END DO
      ROOTS_polynomial_value = px

      RETURN

!  End of ROOTS_polynomial_value

      END FUNCTION ROOTS_polynomial_value

!-*-*-*-*-*-   R O O T S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE ROOTS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine ROOTS_initialize
!   control see Subroutine ROOTS_initialize
!   inform  see Subroutine ROOTS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( ROOTS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ROOTS_control_type ), INTENT( IN ) :: control
      TYPE ( ROOTS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      data%n_max = - 1 ; data%degree_max = - 1
      data%w_max = - 1 ; data%ig_max = - 1

      array_name = 'roots: data%CROOTS'
      CALL SPACE_dealloc_array( data%CROOTS,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%WORK'
      CALL SPACE_dealloc_array( data%WORK,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%A_mat'
      CALL SPACE_dealloc_array( data%A_mat,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'roots: data%IG'
      CALL SPACE_dealloc_array( data%IG,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine ROOTS_terminate

      END SUBROUTINE ROOTS_terminate

!  End of module ROOTS

   END MODULE GALAHAD_ROOTS_double
