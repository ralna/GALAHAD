   PROGRAM GALAHAD_UGO_EXAMPLE  !  GALAHAD 2.8 - 03/06/2016 AT 08:35 GMT
   USE GALAHAD_UGO_double                       ! double precision version
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( UGO_control_type ) :: control
   TYPE ( UGO_inform_type ) :: inform
   TYPE ( UGO_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FGH
   INTEGER :: prob
   REAL ( KIND = wp ) :: x_l, x_u, x, f, g, h, f_min, x_min
!  REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ), PARAMETER :: p = 0.0_wp
   REAL ( KIND = wp ), PARAMETER :: pi = 4.0_wp * ATAN( 1.0_wp )
   x_l = - 1.0_wp; x_u = 2.0_wp                 ! bounds on x
   DO
     WRITE( 6, "( ' Input test problem number [1-23]:' )" )
     READ( 5, * ) prob
     IF ( prob >= 1 .AND.  prob <= 23 ) EXIT
     WRITE( 6, "( ' problem input outside range [1,23], please try again ' )" )
   END DO
   WRITE( 6, "( ' problem ', I0, ' will be attempted' )" )

!  assign problem bounds and record known solutions

   SELECT CASE ( prob )

!  problem 01

   CASE ( 01 )
     x_l = - 1.5_wp ; x_u = 11.0_wp
     f_min = - 29763.233_wp ; x_min = 10.0_wp

!  problem 02

   CASE ( 02 )
     x_l = 2.7_wp ; x_u = 7.5_wp
     f_min = - 1.899599_wp ; x_min = 5.145735_wp

!  problem 03

   CASE ( 03 )
     x_l = - 10.0_wp ; x_u = 10.0_wp
     f_min = - 12.03124_wp ; x_min = -6.7745761_wp

!  problem 04

   CASE ( 04 )
     x_l = 1.9_wp ; x_u = 3.9_wp
     f_min = - 3.85045_wp ; x_min = 2.868034_wp

!  problem 05

   CASE ( 05 )
     x_l = 0.0_wp ; x_u = 1.2_wp
     f_min = -1.48907_wp ; x_min = 0.96609_wp

!  problem 06

   CASE ( 06 )
     x_l = - 10.0_wp ; x_u = 10.0_wp
     f_min = -0.824239_wp ; x_min = 0.67956_wp

!  problem 07

   CASE ( 07 )
     x_l = 2.7_wp ; x_u = 7.5_wp
     f_min = - 1.6013_wp ; x_min = 5.19978_wp

!  problem 08

   CASE ( 08 )
     x_l = - 10.0_wp ; x_u = 10.0_wp
     f_min = - 14.508_wp ; x_min = -7.083506_wp

!  problem 09

   CASE ( 09 )
     x_l = 3.1_wp ; x_u = 20.4_wp
     f_min = - 1.90596_wp ; x_min = 17.039_wp

!  problem 10

   CASE ( 10 )
     x_l = 0.0_wp ; x_u = 10.0_wp
     f_min = - 7.916727_wp ; x_min = 7.9787_wp

!  problem 11

   CASE ( 11 )
     x_l = - 0.5_wp * pi ; x_u = 2.0_wp * pi
     f_min = - 1.5_wp ; x_min = 2.09439_wp

!  problem 12

   CASE ( 12 )
     x_l = 0.0_wp ; x_u = 2.0_wp * pi
     f_min = - 1.0_wp ; x_min = pi

!  problem 13

   CASE ( 13 )
     x_l = 0.001_wp ; x_u = 0.99_wp
     f_min = - 1.5874_wp ; x_min = 1.0_wp / SQRT( 2.0_wp )

!  problem 14

   CASE ( 14 )
     x_l = 0.0_wp ; x_u = 4.0_wp
     f_min = - 0.788685_wp ; x_min = 0.224885_wp

!  problem 15

   CASE ( 15 )
     x_l = - 5.0_wp ; x_u = 5.0_wp
     f_min = - 0.03553_wp ; x_min = 2.41422_wp

!  problem 16

   CASE ( 16 )
     x_l = - 3.0_wp ; x_u = 3.0_wp
     f_min = 7.515924_wp ; x_min = 1.5907_wp

!  problem 17

   CASE ( 17 )
     x_l = - 4.0_wp ; x_u = 4.0_wp
     f_min = 7.0_wp ; x_min = 3.0_wp

!  problem 18

   CASE ( 18 )
     x_l = 0.0_wp ; x_u = 6.0_wp
     f_min = 0.0_wp ; x_min = 2.0_wp

!  problem 19

   CASE ( 19 )
     x_l = 0.0_wp ; x_u = 6.5_wp
     f_min = - 7.81567_wp ; x_min = 5.87287_wp

!  problem 20

   CASE ( 20 )
     x_l = - 10.0_wp ; x_u = 10.0_wp
     f_min = - 0.0634905_wp ; x_min = 1.195137_wp

!  problem 21

   CASE ( 21 )
     x_l = 0.0_wp ; x_u = 10.0_wp
     f_min = - 9.50835_wp  ; x_min = 4.79507_wp

!  problem 22

   CASE ( 22 )
     x_l = 0.0_wp ; x_u = 20.0_wp
     f_min = EXP( - 13.5_wp * pi ) - 1.0_wp ; x_min = 4.5_wp * pi

!  problem 23 (from the Chebfun manual)

   CASE ( 23 )
     x_l = 0.0_wp ; x_u = 15.0_wp
     f_min = -1.990085468159411_wp ; x_min = 4.852581429906174_wp
   END SELECT

   ALLOCATE( userdata%integer( 1 ) )             ! Allocate space for parameter
   userdata%integer( 1 ) = prob                  ! Record problem # prob
   CALL UGO_initialize( data, control, inform )
   control%print_level = 1
!  control%maxit = 100
   control%lipschitz_estimate_used = 3
!  control%stop_length = 5.0_wp * 10.0_wp ** ( - 5 )

   inform%status = 1                            ! set for initial entry
   CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata,      &
                   eval_FGH = FGH )   ! Solve problem
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' Problem ', I0, ' UGO: ', I0, ' evaluations', /,            &
    &     ' Optimal solution =', ES14.6, /,                                    &
    &     ' Optimal objective value =', ES14.6, ' and gradient =', ES14.6, /,  &
    &     ' Alleged solution =', ES14.6, /,                                    &
    &     ' Alleged objective value =', ES14.6 )" )                            &
              prob, inform%iter, x, f, g, x_min, f_min
   ELSE                                         ! Error returns
     WRITE( 6, "( ' UGO_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL UGO_terminate( data, control, inform )  ! delete internal workspace
   END PROGRAM GALAHAD_UGO_EXAMPLE

   SUBROUTINE FGH( status, x, userdata, f, g, h )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( IN ) :: x
   REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

   INTEGER :: k
   REAL ( KIND = wp ) :: a, b, c, ca, c2, c3, c4, c5, c6, e, rk, rkp1, s, sa, s2
   REAL ( KIND = wp ) :: da, db, bm1
   REAL ( KIND = wp ), PARAMETER :: pi = 4.0_wp * ATAN( 1.0_wp )

!  branch to the desired test problem

   SELECT CASE ( userdata%integer( 1 ) )

!  problem 01

   CASE ( 01 )
     c2 = - 79.0_wp / 20.0_wp
     c3 = 71.0_wp / 10.0_wp
     c4 = 39.0_wp / 80.0_wp
     c5 = - 52.0_wp / 25.0_wp
     c6 = 1.0_wp / 6.0_wp
     f = c6 * x ** 6 + c5 * x ** 5 + c4 * x ** 4 + c3 * x ** 3                 &
         + c2 * x ** 2 - x + 0.1_wp
     g = 6.0_wp * c6 * x ** 5 + 5.0_wp * c5 * x ** 4  + 4.0_wp * c4 * x ** 3   &
         + 3.0_wp * c3 * x ** 2 + 2.0_wp * c2 * x - 1.0_wp
     h = 30.0_wp * c6 * x ** 4 + 20.0_wp * c5 * x ** 3                         &
         + 12.0_wp * c4 * x ** 2 + 6.0_wp * c3 * x + 2.0_wp * c2

!  problem 02

   CASE ( 02 )
     a = 10.0_wp / 3.0_wp
     s = SIN( x )
     sa = SIN( a * x )
     f = s + sa
     g = COS( x ) + a * COS( a * x )
     h = - s - a * a * sa

!  problem 03

   CASE ( 03 )
     f = 0.0_wp ; g = 0.0_wp ; h = 0.0_wp
     DO k = 1, 5
       rk = REAL( k, KIND = wp )
       rkp1 =  rk + 1.0_wp
       s = SIN( rkp1 * x + rk )
       f = f - rk * s
       g = g - rk * rkp1 * COS( rkp1 * x + rk )
       h = h + rk * rkp1 * rkp1 * s
     END DO

!  problem 04

   CASE ( 04 )
     e = EXP( - x )
     f = - ( 16.0_wp * x ** 2 - 24.0_wp * x + 5.0_wp ) * e
     g = ( 16.0_wp * x ** 2 - 56.0_wp * x + 29.0_wp ) * e
     h = - ( 16.0_wp * x ** 2 - 88.0_wp * x + 85.0_wp ) * e

!  problem 05

   CASE ( 05 )
     a = 18.0_wp
     s = SIN( a * x )
     c = COS( a * x )
     f = - ( 1.4_wp - 3.0_wp * x ) * s
     g = 3.0_wp * s - a * ( 1.4_wp - 3.0_wp * x ) * c
     h = 6.0_wp * a * c + a * a * ( 1.4_wp - 3.0_wp * x ) * s

!  problem 06

   CASE ( 06 )
     s = SIN( x )
     c = COS( x )
     e = EXP( - x ** 2 )
     f = - ( x + s ) * e
     g = - ( 1.0_wp + c ) * e + 2.0_wp * x * ( x + s ) * e
     h = s * e + 4.0_wp * x * ( 1.0_wp + c ) * e                               &
         + ( 2.0_wp  + 4.0_wp * x * x ) * ( x + s ) * e

!  problem 07

   CASE ( 07 )
     a = 10.0_wp / 3.0_wp
     s = SIN( x )
     sa = SIN( a * x )
     f = s + sa + LOG( x ) - 0.84_wp * x + 3.0_wp
     g = COS( x ) + a * COS( a * x ) + 1.0_wp / x - 0.84_wp
     h = - s - a * a * sa - 1.0_wp / x ** 2

!  problem 08

   CASE ( 08 )
     f = 0.0_wp ; g = 0.0_wp ; h = 0.0_wp
     DO k = 1, 5
       rk = REAL( k, KIND = wp )
       rkp1 =  rk + 1.0_wp
       f = f - rk * COS( rkp1 * x + rk )
       g = g + rk * rkp1 * SIN( rkp1 * x + rk )
       h = h + rk * rkp1 * rkp1 * COS( rkp1 * x + rk )
     END DO

!  problem 09

   CASE ( 09 )
     a = 2.0_wp / 3.0_wp
     s = SIN( x )
     sa = SIN( a * x )
     f = s + sa
     g = COS( x ) + a * COS( a * x )
     h = - s - a * a * sa

!  problem 10

   CASE ( 10 )
     s = SIN( x )
     c = COS( x )
     f = - x * s
     g = - s - x * c
     h = - 2.0_wp * c + x * s

!  problem 11

   CASE ( 11 )
     a = 2.0_wp
     c = COS( x )
     ca = COS( a * x )
     f = 2.0_wp * c + ca
     g = - 2.0_wp * SIN( x ) - a * SIN( a * x )
     h = - 2.0_wp * c - a * 8 * ca

!  problem 12

   CASE ( 12 )
     s = SIN( x )
     c = COS( x )
     f = s ** 3 + c ** 3
     g = 3.0_wp * c * s ** 2 - 3.0_wp * s * c ** 2
     h = - 3.0_wp * s ** 3 - 3.0_wp * c ** 3                                   &
         + 6.0_wp * c ** 2 * s + 6.0_wp * c * s ** 2

!  problem 13

   CASE ( 13 )
     a =  2.0_wp / 3.0_wp
     b = 1.0_wp / 3.0_wp
     bm1 = b - 1.0_wp
     c = 1.0_wp - x ** 2
     f = - x ** a - c ** b
     g = - a * x ** ( a - 1.0_wp ) + 2.0_wp * b * x * c ** bm1
     h = - a * ( a - 1.0_wp ) * x ** ( a - 2.0_wp ) + 2.0_wp * b * c ** bm1    &
         - 4.0_wp * b * bm1 * x * x * c ** ( b - 2.0_wp )

!  problem 14

   CASE ( 14 )
     a = 2.0_wp * pi
     e = EXP( - x )
     sa = SIN( a * x )
     ca = COS( a * x )
     f = - e * sa
     g = e * sa - a * e * ca
     h = - e * sa + 2.0_wp * a * e * ca + a * a * e * sa

!  problem 15

   CASE ( 15 )
     a = x ** 2 - 5.0_wp * x + 6.0_wp
     b = x ** 2 + 1.0_wp
     da = 2.0_wp * x - 5.0_wp
     db = 2.0_wp * x
     f = a / b
     g = da / b - a * db / b ** 2
     h = 2.0_wp / b - 2.0_wp * da * db / b ** 2                                &
         - a * 2.0_wp / b ** 2 + 2.0_wp * a * db * db / b ** 3

!  problem 16

   CASE ( 16 )
     a = ( x - 3.0_wp )
     e = EXP( 0.5_wp * x ** 2 )
     f = 2.0_wp * a ** 2 + e
     g = 4.0_wp * a + x * e
     h = 4.0_wp + ( x ** 2 + 1.0_wp ) * e

!  problem 17

   CASE ( 17 )
     c2 = 27.0_wp
     c4 = - 15.0_wp
     c6 = 1.0_wp
     f = c6 * x ** 6 + c4 * x ** 4 + c2 * x ** 2 + 250.0_wp
     g = 6.0_wp * c6 * x ** 5 + 4.0_wp * c4 * x ** 3 + 2.0_wp * c2 * x
     h = 30.0_wp * c6 * x ** 4 + 12.0_wp * c4 * x ** 2 + 2.0_wp * c2

!  problem 18

   CASE ( 18 )
     a = x - 2.0_wp
     IF ( x <= 3.0_wp ) THEN
       f = a ** 2
       g = 2.0_wp * a
       h = 2.0_wp
     ELSE
       f = 2.0_wp * LOG( a ) + 1.0_wp
       g = 1.0_wp / a
       h = - 1.0_wp / a ** 2
     END IF

!  problem 19

   CASE ( 19 )
     a = 3.0_wp
     sa = SIN( a * x )
     f = - x + sa - 1.0_wp
     g = - 1.0_wp + a * COS( a * x )
     h = - a * a * sa

!  problem 20

   CASE ( 20 )
     e = EXP( - x ** 2 )
     s = SIN( x )
     c = COS( x )
     f = - ( x - s ) * e
     g = - ( 1.0_wp - c ) * e + 2.0_wp * x * ( x - s ) * e
     h = - s * e + 4.0_wp * x * ( 1.0_wp - c ) * e                             &
         - 4.0_wp * x * x * ( x - s ) * e

!  problem 21

   CASE ( 21 )
     a = 2.0_wp
     s = SIN( x )
     c = COS( x )
     sa = SIN( a * x )
     ca = COS( a * x )
     f = x * ( s + ca )
     g = ( s + ca ) + x * ( c - a * sa )
     h = 2.0_wp * ( c - a * sa ) - x * ( s + a * a * ca )

!  problem 22

   CASE ( 22 )
     a = - 3.0_wp
     e = EXP( a * x )
     s = SIN( x )
     c = COS( x )
     f = e - s ** 3
     g = a * e - 3.0_wp * c * s ** 2
     h = a * a * e  + 3.0_wp * s ** 3 - 6.0_wp * s * c ** 2

!  problem 23

   CASE ( 23 )
     s = SIN( x )
     s2 = SIN( x ** 2 )
     f = s + s2
     g = COS( x ) + 2.0_wp * x * s2
     h = - s + 2.0_wp * s2 - 4.0_wp * x * x * s2

!  out of range

   CASE DEFAULT
     status = 1
     RETURN
   END SELECT

   status = 0
   RETURN
   END SUBROUTINE FGH















