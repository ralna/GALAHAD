! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 11:00 GMT.

!-*-*-*-*-*-*-*- G A L A H A D _ S C A L I N G   M O D U L E  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History - released, pre GALAHAD 1.0, September 2nd 1999
!  Updated for GALAHAD 2.0 July 14th, 2006
!  Name changed from oldscale to scaling GALAHAD 3.3 May 20th, 2021

   MODULE GALAHAD_SCALING_double

!     ------------------------------------------
!     | Scale data for the quadratic program   |
!     |                                        |
!     |    minimize     1/2 x(T) H x + g(T) x  |
!     |    subject to   c_l <= A x <= c_u      |
!     |                 x_l <=  x  <= x_u      |
!     |                                        |
!     | Nick Gould                             |
!     | September 1999                         |
!     ------------------------------------------

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SCALING_get_factors_from_K, SCALING_get_factors_from_A,        &
                SCALING_normalize_rows_of_A, SCALING_apply_factors,            &
                SCALING_initialize

      INTERFACE SCALING_apply_factors
        MODULE PROCEDURE SCALING_apply_factors, SCALING_apply_factorsd
      END INTERFACE SCALING_apply_factors

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

      TYPE, PUBLIC :: SCALING_control_type
        INTEGER :: print_level, out, out_error
      END TYPE

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp

   CONTAINS

!-*-*-*-*-*-*-*-*-*-*  S C A L I N G _ i n i t i a l i z e   *-*-*-*-*-*-*-*-*-

      SUBROUTINE SCALING_initialize( CONTROL )

!  Initialize control parameters for scaling routines

      TYPE( SCALING_control_type ), INTENT( OUT ) :: CONTROL
      CONTROL%print_level = 0
      CONTROL%out = 6
      CONTROL%out_error = 6
      RETURN

!  End of SCALING_initialize

      END SUBROUTINE SCALING_initialize

!-*-*-*-*-*-*   S C A L I N G _ g e t _ f a c t o r s _ f r o m _ K  *-*-*-*-*-

      SUBROUTINE SCALING_get_factors_from_K( n, m, H_val, H_col, H_ptr,        &
                                             A_val, A_col, A_ptr, SH, SA,      &
                                             CONTROL, ifail )

!  Compute column scaling factors for the symmetric matrix

!            ( H   A(transpose) )
!            ( A        0       )

!  This routine is based on John Reid's HSL code MC30.

!  arguments:
!  ---------

!  H_ and A_ See QPB
!  SH    is an array that need not be be on entry. On return, it holds the
!        scaling factor for the H rows
!  SA    is an array that need not be be on entry. On return, it holds the
!        scaling factor for the A rows
!  IFAIL need not be set by the user. On return it has one of the
!        following values:
!          >= 0 successful entry
!          -1 N+M < 1
!          -2 NE < 1

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: ifail
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: SH
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: SA
      TYPE( SCALING_control_type ) :: CONTROL

!  Automatic arrays

      REAL ( KIND = wp ), DIMENSION( n + m ) :: MM, R, P, MP

!  Constant parameters

!  maxit is the maximal permitted number of iterations.
!  rmin  is used in a convergence test on (residual norm)**2

      INTEGER, PARAMETER :: maxit = 10
      REAL ( KIND = wp ), PARAMETER :: rmin = 0.1_wp
!     INTEGER :: maxit = 100
!     REAL ( KIND = wp ), PARAMETER :: rmin = 0.01_wp

!  local variables
 
!  AK    Scalar of cg iteration.
!  BK    Scalar of cg iteration.
!  I     Row index.
!  ITER  Iteration index.
!  J     Column index.
!  K     Entry number.
!  PP    Scalar p'(M+E)p of cg iteration.
!  RM    Threshold for RR.
!  RR    Scalar r'(inv M)r of cg iteration.
!  RRL   Previous value of RR.
!  U     abs(A(K)).
!  LOGE2 log(2)

      INTEGER :: i, ii, j, l, npm, ne, iter
      REAL ( KIND = wp ) :: ak, bk, pp, rm, rr, u, rrl, loge2 , smax, smin

!  Intrinsics

      INTRINSIC LOG, ABS, MAX, MIN, ANINT

!  Nick Gould, Rutherford Appleton Laboratory.
!  September 1999

      loge2 = LOG( two )
      npm  = n + m ; ne = ( A_ptr( m + 1 ) - 1 ) + ( H_ptr( n + 1 ) - 1 )

!  Check npm and ne

      ifail = 0
      IF ( npm < 1 ) THEN
        ifail = - 1 ; GO TO 600
      ELSE IF ( ne <= 0 ) THEN
        ifail = - 2 ; GO TO 600
      END IF

!  Initialise for accumulation of sums and products

      SH = zero ; SA = zero ; MM = zero ; R = zero

!  Count non-zeros in the rows, and compute rhs vectors.
!  Contributions from H

      DO i = 1, n
        DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
          u = ABS( H_val( l ) )
          IF ( u /= zero ) THEN
            j = H_col( l )
            IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
              u = LOG( u ) / loge2
              MM( i ) = MM( i ) + one ; MM( j ) = MM( j ) + one
              R( i ) = R( i ) - u
              IF ( i /= j ) R( j ) = R( j ) - u
            END IF
          END IF
        END DO
      END DO

!  Contributions from A

      DO i = 1, m
        ii = n + i
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          u = ABS( A_val( l ) )
          IF ( u /= zero ) THEN
            j = A_col( l )
            IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
              u = LOG( u ) / loge2
              MM( ii ) = MM( ii ) + one ; R( ii ) = R( ii ) - u
              MM( j ) = MM( j ) + one ; R( j ) = R( j ) - u
            END IF
          END IF
        END DO
      END DO

!  Find the initial vectors

      WHERE ( MM == zero ) MM = one
      P = R / MM ; MP = R
      rr = SUM( R ** 2 / MM )

!  Compute the stopping tolerance

      rm = rmin * ne

!     Iteration loop

      IF ( rr > rm ) THEN
        DO iter = 1, maxit

!  Sweep through matrix to add Ep to Mp
!  Contributions from H.

          DO i = 1, n
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              IF ( H_val( l ) /= zero ) THEN
                j = H_col( l )
                IF ( i /= j ) THEN
                  IF ( MIN( i, j ) >= 1 .AND.  MAX( i, j ) <= n ) THEN
                    MP( i ) = MP( i ) + P( j ) ; MP( j ) = MP( j ) + P( i )
                  END IF
                END IF
              END IF
            END DO
          END DO

!  Contributions from A

          DO i = 1, m
            ii = n + i
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( A_val( l ) /= zero ) THEN
                j = A_col( l )
                IF ( MIN( i, j ) >= 1 .AND. i <= m .AND. j <= n ) THEN
                  MP( ii ) = MP( ii ) + P( j ) ; MP( j ) = MP( j ) + P( ii )
                END IF
              END IF
            END DO
          END DO
          PP = DOT_PRODUCT( P, MP ) ; ak = rr / pp

!  Update solution and residual

          SH = SH + ak * P( : n ) ; SA = SA + ak * P( n + 1 : npm )
          R = R - ak * MP
          rrl = rr ; rr  = SUM( R ** 2 / MM )
          IF ( rr <= rm ) EXIT

!  Update vector P

          bk = rr / rrl
          P = R / MM + bk * P ; MP = P * MM
        END DO
      END IF

!  Obtain the scaling factors
!  Factors for the H rows

      SH( 1 ) = two ** ANINT( SH( 1 ) )
      smax = SH( 1 ) ; smin = smax
      DO i = 2, n
        SH( i ) = two ** ANINT( SH( i ) )
        smax = MAX( smax, SH( i ) ) ; smin = MIN( smin, SH( i ) )
      END DO
      IF ( smax /= smin ) ifail = ifail + 1   
      IF ( CONTROL%print_level > 0 .AND. CONTROL%out > 0 )                    &
           WRITE( CONTROL%out, 2010 ) smin, smax

!  Factors for the A rows

      IF ( m > 0 ) THEN
        SA( 1 ) = two ** ANINT( SA( 1 ) )
        smax = SA( 1 ) ; smin = SA( 1 )
        DO i = 2, m
          SA( i ) = two ** ANINT( SA( i ) )
          smax = MAX( smax, SA( i ) ) ; smin = MIN( smin, SA( i ) )
        END DO
        IF ( smax /= smin ) ifail = ifail + 2
        IF ( CONTROL%print_level > 0 .AND. CONTROL%out > 0 )           &
             WRITE( CONTROL%out, 2020 ) smin, smax
      END IF
      RETURN

!  Error returns

  600 CONTINUE
      IF ( CONTROL%out_error > 0 ) WRITE ( CONTROL%out_error, 2000 ) ifail
      RETURN

!  Non-executable statements

 2000 FORMAT( ' **** Error return from SCALING_get_factors **** IFAIL =', I4 )
 2010 FORMAT( ' MIN, MAX column scaling = ', 2ES12.4 )
 2020 FORMAT( ' MIN, MAX   row  scaling = ', 2ES12.4 )

!  End of SCALING_get_factors

      END SUBROUTINE SCALING_get_factors_from_K

!-*-*-*-*-*   S C A L I N G _ g e t _ f a c t o r s _ f r o m _ A  *-*-*-*-*

      SUBROUTINE SCALING_get_factors_from_A( n, m, A_val, A_col, A_ptr,        &
                                             C, R, control, ifail )

!   Compute row and column scalings using the algorithm of Curtis and Reid
!   (J.I.M.A. 10 (1972) 118-124) by approximately minimizing the function

!        sum (nonzero A) ( log_2(|a_ij|) + r_i + c_j)^2

!   The required scalings are then 2^nint(r) and 2^nint(c) respectively

!   Use Reid's special purpose method for matrices with property "A". 
!   Comments refer to equation numbers in Curtis and Reid's paper

!  arguments:
!  ---------

!  A_ See QPB
!  C     is an array that need not be be on entry. On return, it holds the
!        scaling factor for the columns of A
!  R     is an array that need not be be on entry. On return, it holds the
!        scaling factor for the rows of A
!  IFAIL need not be set by the user. On return it has one of the
!        following values:
!          >= 0 successful entry
!          -1 M < 1
!          -2 N < 1

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m , n
      INTEGER, INTENT( OUT ) :: ifail 
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( M  ) :: R 
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( N  ) :: C 
      TYPE( SCALING_control_type ) :: CONTROL

!  Constant parameters

!  itmax is the maximal permitted number of iterations.
!  rmin  is used in a convergence test on (residual norm)**2

      INTEGER, PARAMETER :: itmax = 100 
      REAL ( KIND = wp ), PARAMETER :: rmin = 0.1_wp

!  Automatic arrays

      REAL ( KIND = wp ), DIMENSION( m ) :: ROW_count, M_inv_sig
      REAL ( KIND = wp ), DIMENSION( n ) :: COL_count, COL_rhs, SHIFT_c

!  local variables

     INTEGER :: i, iter, j, k, ne
     REAL ( KIND = wp ) :: e, e1, em, loge2, q, q1, qm
     REAL ( KIND = wp ) :: s, s1, smax, smin, stop_tol, u, v

!  Check m and n

      ifail = 0
      IF ( m <= 0 ) THEN
        ifail = - 1 ; GO TO 600
      ELSE IF ( n <= 0 ) THEN
        ifail = - 2 ; GO TO 600
      END IF

!  Set the stopping tolerance

      loge2 = LOG( two )
      ne = A_ptr( m + 1 ) - 1
      stop_tol = ne * rmin

!  Initialise for accumulation of sums and products

      R = ZERO 
      C = ZERO 
      ROW_count = ZERO 
      COL_count = ZERO 
      COL_rhs = ZERO 

!  Count non-zeros in the rows, and compute r.h.s. vectors; use R to store
!  the row r.h.s. (sigma in Curtis+Reid)

      DO i = 1, m
        DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
          u = ABS( A_val( k ) )
          IF ( u /= zero ) THEN
            j = A_col( k )
            u = LOG( u ) / loge2
            ROW_count( i ) = ROW_count( i ) + one 
            COL_count( j ) = COL_count( j ) + one 
            R( i ) = R( i ) + u 
            COL_rhs( j ) = COL_rhs( j ) + u 
           END IF 
         END DO 
       END DO 

!  Account for structural singularity

      WHERE ( ROW_count == zero ) ROW_count = one 
      WHERE ( COL_count == zero ) COL_count = one 

!  Form M^-1 sigma and N^-1 tau (in C+R's notation)

      M_inv_sig = R / ROW_count
      COL_rhs = COL_rhs / COL_count

!  Compute initial residual vector

      R = M_inv_sig

!  Compute initial residual vector

      DO i = 1, m
        DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
        IF ( A_val( k ) /= zero )                                            &
           R( i ) = R( i ) - COL_rhs( A_col( k ) ) / ROW_count( i )  ! (4.3)
        END DO 
      END DO 

!  Set initial values

      e = zero 
      q = one 
      s = DOT_PRODUCT( ROW_count, R ** 2 ) 
      IF ( s > stop_tol ) THEN
        SHIFT_c = zero 

!  Iteration loop

        DO iter = 1, itmax 

!  Update column residual vector

          DO i = 1, m
            DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( A_val( k ) /= zero ) THEN
                j = A_col( k )
                C( j ) = C( j ) + R( i ) 
              END IF
            END DO 
          END DO 

!  Rescale column residual

          s1 = s ; s = zero 
          DO j = 1, n 
            v = - C( j ) / q 
            C( j ) = v / COL_count( j )    ! (4.4a)
            s = s + v * C( j )       ! (4.5a)
          END DO 

!  Rescale row residual vector

          e1 = e 
          e = q * s / s1                ! (4.6)
          q = one - e                   ! (4.7)
          IF ( s <= stop_tol ) e = zero 
          R = R * e * ROW_count 

!  Test for termination

          IF ( s <= stop_tol ) GO TO 100 
          em = e * e1 

!  Update row residual

          DO i = 1, m
            DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( A_val( k ) /= zero ) R( i ) = R( i ) + C( A_col( k ) ) 
            END DO 
          END DO 

!  Again, rescale row residual

          s1 = s ; s = zero 
          DO i = 1, m 
             v = - R( i ) / q 
             R( i ) = v / ROW_count( i )  ! (4.4b)
             s = s + v * R( i )     ! (4.5b)
          END DO 
          e1 = e ; e = q * s / s1     ! (4.6)
          q1 = q ; q = one - e        ! (4.7)

!  Special fixup for last iteration

          IF ( s <= stop_tol ) q = one 

!  Rescale column residual vector

          qm = q * q1 
          SHIFT_c = ( em * SHIFT_c + C ) / qm 
          COL_rhs = COL_rhs + SHIFT_c 

!  Test for termination

          IF ( s <= stop_tol ) EXIT  

!  Update column scaling factors

          C = e * C * COL_count 
        END DO 
      END IF
      R = R * ROW_count 

!  Sweep through matrix to prepare to get row scaling powers

  100 CONTINUE 

      DO i = 1, m
        DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
          IF ( A_val( k ) /= zero ) R( i ) = R( i ) + COL_rhs( A_col( k ) )
        END DO 
      END DO 

!  Final conversion to output values

      R = R / ROW_count - M_inv_sig 
      C = - COL_rhs 

!  Obtain the scaling factors
!  Factors for the H rows

      C( 1 ) = two ** ANINT( C( 1 ) )
!     C( 1 ) = exp( C( 1 ) )
      smax = C( 1 ) ; smin = smax
      DO i = 2, n
        C( i ) = two ** ANINT( C( i ) )
!       C( i ) = exp( C( i ) )
        smax = MAX( smax, C( i ) ) ; smin = MIN( smin, C( i ) )
      END DO
      IF ( CONTROL%print_level > 0 .AND. CONTROL%out > 0 )                    &
           WRITE( CONTROL%out, 2010 ) smin, smax

!  Factors for the A rows

      IF ( m > 0 ) THEN
        R( 1 ) = two ** ANINT( R( 1 ) )
!       R( 1 ) = exp( R( 1 ) )
        smax = R( 1 ) ; smin = R( 1 )
        DO i = 2, m
          R( i ) = two ** ANINT( R( i ) )
!         R( i ) = exp( R( i ) )
          smax = MAX( smax, R( i ) ) ; smin = MIN( smin, R( i ) )
        END DO
        IF ( CONTROL%print_level > 0 .AND. CONTROL%out > 0 )                   &
             WRITE( CONTROL%out, 2020 ) smin, smax
      END IF
      RETURN  

!  Error returns

  600 CONTINUE 
      IF ( CONTROL%out_error > 0 ) WRITE ( CONTROL%out_error, 2000 ) ifail
      RETURN

!  Non-executable statements

 2000 FORMAT( ' **** Error return from SCALING_get_factors **** IFAIL =', I4 )
 2010 FORMAT( ' MIN, MAX column scaling = ', 2ES12.4 )
 2020 FORMAT( ' MIN, MAX   row  scaling = ', 2ES12.4 )

!  End of subroutine SCALING_get_factors_from_A

      END SUBROUTINE SCALING_get_factors_from_A

!-*-*-*-*-*   S C A L I N G _ g e t _ f a c t o r s _ f r o m _ A  *-*-*-*-*

      SUBROUTINE SCALING_normalize_rows_of_A( n, m, A_val, A_col, A_ptr, C, R )

!   Renormalize the rows of R * A * C so that each has a one-norm close
!   to one

!  arguments:
!  ---------

!  A_ See QPB
!  R     is an array that must be set on entry to the current row scaling 
!        factors R. On exit, R may have been altered to reflect the
!        rescaling
!  C     is an array that must be set on entry to the current column scaling 
!        factors C. It is unaltered on exit

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m , n
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m  ) :: R 
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n  ) :: C 

!  local variables

      INTEGER :: i, k
      REAL ( KIND = wp ) :: ri, scale, loge2

      loge2 = LOG( two )
      DO i = 1, m
        ri = R( i )
        scale = zero
        DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
          scale = scale + ABS( Ri * C( A_col( k ) ) * A_val( k ) )
        END DO
        IF ( scale /= zero ) R( i ) = Ri / two ** ANINT( LOG( scale ) / loge2 )
      END DO

      RETURN

!  End of SCALING_normalize_rows_of_A

      END SUBROUTINE SCALING_normalize_rows_of_A

!-*-*-*-*-*-*-*-*-*   S C A L I N G _ a p p l y _ f a c t o r s  *-*-*-*-*-*-*

      SUBROUTINE SCALING_apply_factors( n, m, H_val, H_col, H_ptr, A_val,      &
                                        A_col, A_ptr, G, X, X_l, X_u, C_l,     &
                                        C_u, Y, Z, biginf, SH, SA, scale,      &
                                        C, DG, DX_l, DX_u, DC_l, DC_u )

!  -------------------------------------------------------------------
!  Scale or unscale the problem
!
!      min 1/2 x^T H x + x^T g
!
!      s.t. c_l <= A x <= c_u, x_l <= x < x_u
!
!   (optionally the parametric problem
!
!      min  1/2 x(T) H x + g(T) x + theta dg(T) x
!
!      s.t.  c_l + theta dc_l <= A x <= c_u + theta dc_u
!      and   x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  and its Lagrange multipliers/dual variables so that the resulting problem is
!
!      min 1/2 y^T ( Sh H Sh ) v + y^T ( Sh g )
!
!      s.t.  ( Sa c_l )  <= ( Sa A Sh ) v <=  ( Sa c_u ),   
!      and  ( Sh^-1 x_l) <=       v       <= ( Sh^-1 x_u ).
!
!   (optionally the parametric problem
!
!      min 1/2 y^T ( Sh H Sh ) v + y^T ( Sh g ) + theta y^T ( Sh dg )
!
!      s.t.  ( Sa c_l ) + theta ( Sa dc_l ) <= ( Sa A Sh ) v 
!                                           <=  ( Sa c_u ) + theta ( Sa dc_u )
!      and  ( Sh^-1 x_l ) + theta ( Sh^-1 dx_l ) <= v
!                                       <= ( Sh^-1 x_u ) + theta ( Sh^-1 x_u ).)
!
!  If SCALE is .TRUE., Sh and Sa are as input in SH and SA. 
!  Otherwise, Sh and Sa are the reciprocals of SH and SA.
!
!  The data H, x, g, A, c_l, c_u, x_l and x_u and the multipliers for
!  the general constraints and dual variables for the bounds is input as 
!           H, X, G, A, C_l, C_u, X_l, X_u, Y and Z 
!  (and optionally C = Ax, DG, DC_l, DC_u, DX_l and DX_u ).
!
!  The resulting scaled variants, 
!  ( Sh H Sh ), ( Sh^-1 x ), ( Sh g ), ( Sa A Sh ), ( Sa c bounds ), 
!  ( Sh^-1 x bounds), ( Sa^-1 multipliers) and ( Sh^-1 duals ) are output as 
!           H, X, G, A, ( C_l, C_u ), ( X_l, X_u ), Y and Z
!  (optionally (Sa c ), ( Sh g ), ( Sa c bounds ) and ( Sh^-1 x bounds) are 
!     output as C, DG, DC_l, DC_u, DX_l and DX_u ).
!
!  Nick Gould, Rutherford Appleton Laboratory.
!  September 1999
!  -------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: biginf
      LOGICAL, INTENT( IN ) :: scale
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_l, X_u, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y, C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SH
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: SA
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                             DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                             DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: C, DC_l, DC_u

!  local variables

      INTEGER :: i, l

! ================
!  Scale the data
! ================

      IF ( scale ) THEN

!  Scale H

        DO i = 1, n
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            H_val( l ) = H_val( l ) * SH( i ) * SH( H_col( l ) )
          END DO
        END DO

!  Scale A

        DO i = 1, m
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            A_val( l ) = A_val( l ) * SA( i ) * SH( A_col( l ) )
          END DO
        END DO

!  Scale X and G

        X = X / SH ; G = G * SH
        IF ( PRESENT( DG ) ) DG = DG * SH
        IF ( PRESENT( C ) ) C = C * SA

!  Scale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) / SH( i )
            X_u( i ) = X_u( i ) / SH( i )
          ELSE 
            IF ( X_l( i ) > - biginf ) THEN
              X_l( i )  = X_l( i ) / SH( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) / SH( i )
            END IF
            IF  ( X_u( i ) < biginf ) THEN 
              X_u( i ) = X_u( i ) / SH( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) / SH( i )
            END IF
          END IF
        END DO

        Z = Z * SH

!  Scale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) * SA( i )
            C_u( i ) = C_u( i ) * SA( i )
          ELSE 
            IF ( C_l( i ) > - biginf ) THEN
              C_l( i )  = C_l( i ) * SA( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) * SA( i )
            END IF
            IF  ( C_u( i ) < biginf ) THEN 
              C_u( i ) = C_u( i ) * SA( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) * SA( i )
            END IF
          END IF
!         WRITE(6,*) i, MAXVAL( ABS( A_val( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) ), C_l( i ), C_u( i ), &
!       SUM( A_val(A_ptr( i ): A_ptr( i + 1 ) - 1) * X_l( A_col(A_ptr( i ): A_ptr( i + 1 ) - 1) ) )
        END DO

        Y = Y / SA
        
! ==================
!  Unscale the data
! ==================

      ELSE

!  Unscale H

        DO i = 1, n
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            H_val( l ) = H_val( l ) / ( SH( i ) * SH( H_col( l ) ) )
          END DO
        END DO

!  Unscale A

        DO i = 1, m
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            A_val( l ) = A_val( l ) / ( SA( i ) * SH( A_col( l ) ) )
          END DO
        END DO

!  Unscale X and G

        X = X * SH ; G = G / SH
        IF ( PRESENT( DG ) ) DG = DG / SH
        IF ( PRESENT( C ) ) C = C / SA

!  Unscale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) * SH( i )
            X_u( i ) = X_u( i ) * SH( i )
          ELSE 
            IF ( X_l( i ) > - biginf ) THEN
              X_l( i )  = X_l( i ) * SH( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) * SH( i )
            END IF
            IF  ( X_u( i ) < biginf ) THEN 
              X_u( i ) = X_u( i ) * SH( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) * SH( i )
            END IF
          END IF
        END DO

        Z = Z / SH

!  Unscale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) / SA( i )
            C_u( i ) = C_u( i ) / SA( i )
          ELSE 
            IF ( C_l( i ) > - biginf ) THEN
              C_l( i )  = C_l( i ) / SA( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) / SA( i )
            END IF
            IF  ( C_u( i ) < biginf ) THEN 
              C_u( i ) = C_u( i ) / SA( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) / SA( i )
            END IF
          END IF
        END DO

        Y = Y * SA

      END IF
      RETURN

!  End of SCALING_apply_factors

      END SUBROUTINE SCALING_apply_factors

!-*-*-*-*-*-*-*-*-*   S C A L I N G _ a p p l y _ f a c t o r s d *-*-*-*-*-*-*

      SUBROUTINE SCALING_apply_factorsd( n, m, H_val, H_col, H_ptr, A_val,     &
                                         A_col, A_ptr, G, X, X_l, X_u, C_l,    &
                                         C_u, Y_l, Y_u, Z_l, Z_u, biginf,      &
                                         SH, SA, scale, C, DG, DX_l, DX_u,     &
                                         DC_l, DC_u )

!  -------------------------------------------------------------------
!  Scale or unscale the problem
!
!      min 1/2 x^T H x + x^T g
!
!      s.t. c_l <= A x <= c_u, x_l <= x < x_u
!
!   (optionally the parametric problem
!
!      min  1/2 x(T) H x + g(T) x + theta dg(T) x
!
!      s.t.  c_l + theta dc_l <= A x <= c_u + theta dc_u
!      and   x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  and its Lagrange multipliers/dual variables so that the resulting problem is
!
!      min 1/2 y^T ( Sh H Sh ) v + y^T ( Sh g )
!
!      s.t.  ( Sa c_l )  <= ( Sa A Sh ) v <=  ( Sa c_u ),   
!      and  ( Sh^-1 x_l) <=       v       <= ( Sh^-1 x_u ).
!
!   (optionally the parametric problem
!
!      min 1/2 y^T ( Sh H Sh ) v + y^T ( Sh g ) + theta y^T ( Sh dg )
!
!      s.t.  ( Sa c_l ) + theta ( Sa dc_l ) <= ( Sa A Sh ) v 
!                                           <=  ( Sa c_u ) + theta ( Sa dc_u )
!      and  ( Sh^-1 x_l ) + theta ( Sh^-1 dx_l ) <= v
!                                       <= ( Sh^-1 x_u ) + theta ( Sh^-1 x_u ).)
!
!  If SCALE is .TRUE., Sh and Sa are as input in SH and SA. 
!  Otherwise, Sh and Sa are the reciprocals of SH and SA.
!
!  The data H, x, g, A, c_l, c_u, x_l and x_u and the multipliers for
!  the general constraints and dual variables for the bounds is input as 
!           H, X, G, A, C_l, C_u, X_l, X_u, Y_l, Y_u, Z_l and Z_u
!  (and optionally C = Ax, DG, DC_l, DC_u, DX_l and DX_u ).
!
!  The resulting scaled variants, 
!  ( Sh H Sh ), ( Sh^-1 x ), ( Sh g ), ( Sa A Sh ), ( Sa c bounds ), 
!  ( Sh^-1 x bounds), ( Sa^-1 multipliers) and ( Sh^-1 duals ) are output as 
!           H, X, G, A, ( C_l, C_u ), ( X_l, X_u ), Y_l, Y_u, Z_l and Z_u
!  (optionally (Sa c ), ( Sh g ), ( Sa c bounds ) and ( Sh^-1 x bounds) are 
!     output as C, DG, DC_l, DC_u, DX_l and DX_u ).
!
!  Nick Gould, Rutherford Appleton Laboratory.
!  September 1999
!  -------------------------------------------------------------------

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: biginf
      LOGICAL, INTENT( IN ) :: scale
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_l, X_u, Z_l, Z_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u, Y_l, Y_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SH
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: SA
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                             DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                             DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
      REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: C, DC_l, DC_u

!  local variables

      INTEGER :: i, l

! ================
!  Scale the data
! ================

      IF ( scale ) THEN

!  Scale H

        DO i = 1, n
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            H_val( l ) = H_val( l ) * SH( i ) * SH( H_col( l ) )
          END DO
        END DO

!  Scale A

        DO i = 1, m
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            A_val( l ) = A_val( l ) * SA( i ) * SH( A_col( l ) )
          END DO
        END DO

!  Scale X and G

        X = X / SH ; G = G * SH
        IF ( PRESENT( DG ) ) DG = DG * SH
        IF ( PRESENT( C ) ) C = C * SA

!  Scale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) / SH( i )
            X_u( i ) = X_u( i ) / SH( i )
          ELSE 
            IF ( X_l( i ) > - biginf ) THEN
              X_l( i )  = X_l( i ) / SH( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) / SH( i )
            END IF
            IF  ( X_u( i ) < biginf ) THEN 
              X_u( i ) = X_u( i ) / SH( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) / SH( i )
            END IF
          END IF
        END DO

        Z_l = Z_l / SH ; Z_u = Z_u / SH

!  Scale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) * SA( i )
            C_u( i ) = C_u( i ) * SA( i )
          ELSE 
            IF ( C_l( i ) > - biginf ) THEN
              C_l( i )  = C_l( i ) * SA( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) * SA( i )
            END IF
            IF  ( C_u( i ) < biginf ) THEN 
              C_u( i ) = C_u( i ) * SA( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) * SA( i )
            END IF
          END IF
!         WRITE(6,*) i, MAXVAL( ABS( A_val( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) ), C_l( i ), C_u( i ), &
!       SUM( A_val(A_ptr( i ): A_ptr( i + 1 ) - 1) * X_l( A_col(A_ptr( i ): A_ptr( i + 1 ) - 1) ) )
        END DO

        Y_l = Y_l / SA ; Y_u = Y_u / SA
        
! ==================
!  Unscale the data
! ==================

      ELSE

!  Unscale H

        DO i = 1, n
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            H_val( l ) = H_val( l ) / ( SH( i ) * SH( H_col( l ) ) )
          END DO
        END DO

!  Unscale A

        DO i = 1, m
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            A_val( l ) = A_val( l ) / ( SA( i ) * SH( A_col( l ) ) )
          END DO
        END DO

!  Unscale X and G

        X = X * SH ; G = G / SH
        IF ( PRESENT( DG ) ) DG = DG / SH
        IF ( PRESENT( C ) ) C = C / SA

!  Unscale the bounds on X and the associated dual variables

        DO i = 1, n
          IF ( X_l( i ) == X_u( i ) ) THEN
            X_l( i ) = X_l( i ) * SH( i )
            X_u( i ) = X_u( i ) * SH( i )
          ELSE 
            IF ( X_l( i ) > - biginf ) THEN
              X_l( i )  = X_l( i ) * SH( i )
              IF ( PRESENT( DX_l ) ) DX_l( i )  = DX_l( i ) * SH( i )
            END IF
            IF  ( X_u( i ) < biginf ) THEN 
              X_u( i ) = X_u( i ) * SH( i )
              IF ( PRESENT( DX_u ) ) DX_u( i )  = DX_u( i ) * SH( i )
            END IF
          END IF
        END DO

        Z_l = Z_l * SH ; Z_u = Z_u * SH

!  Unscale the bounds on Ax and the associated Lagrange multipliers

        DO i = 1, m
          IF ( C_l( i ) == C_u( i ) ) THEN
            C_l( i ) = C_l( i ) / SA( i )
            C_u( i ) = C_u( i ) / SA( i )
          ELSE 
            IF ( C_l( i ) > - biginf ) THEN
              C_l( i )  = C_l( i ) / SA( i )
              IF ( PRESENT( DC_l ) ) DC_l( i ) = DC_l( i ) / SA( i )
            END IF
            IF  ( C_u( i ) < biginf ) THEN 
              C_u( i ) = C_u( i ) / SA( i )
              IF ( PRESENT( DC_u ) ) DC_u( i ) = DC_u( i ) / SA( i )
            END IF
          END IF
        END DO

        Y_l = Y_l * SA ; Y_u = Y_u * SA

      END IF
      RETURN

!  End of SCALING_apply_factorsd

      END SUBROUTINE SCALING_apply_factorsd

!  End of module GALAHAD_SCALING_double

   END MODULE GALAHAD_SCALING_double




