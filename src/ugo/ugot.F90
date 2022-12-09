#include "galahad_modules.h"
   PROGRAM GALAHAD_UGO_TEST_PROGRAM  !  GALAHAD 4.1 - 2022-12-09 AT 07:45 GMT
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_UGO_precision
   IMPLICIT NONE
   TYPE ( UGO_control_type ) :: control
   TYPE ( UGO_inform_type ) :: inform
   TYPE ( UGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FGH
   INTEGER ( KIND = ip_ ) :: prob
   REAL ( KIND = rp_ ) :: x_l, x_u, x, f, g, h, f_min, x_min
   REAL ( KIND = rp_ ), PARAMETER :: pi = 4.0_rp_ * ATAN( 1.0_rp_ )
   REAL ( KIND = rp_ ), PARAMETER :: accurate_x = ( 10.0_rp_ ) ** ( - 5 )
   REAL ( KIND = rp_ ), PARAMETER :: accurate_f = ( 10.0_rp_ ) ** ( - 5 )
   x_l = - 1.0_rp_; x_u = 2.0_rp_                 ! bounds on x

   WRITE( 6, "( ' tests for real precision of type ', I0 )" ) rp_
   ALLOCATE( userdata%integer( 1 ) )           ! Allocate space for parameter
   CALL UGO_initialize( data, control, inform )

!  loop over the test examples

   DO prob = 1, 22

!  assign problem bounds and record known solutions

     SELECT CASE ( prob )

!    problem 01

     CASE ( 01 )
       x_l = - 1.5_rp_ ; x_u = 11.0_rp_
       f_min = - 29763.233_rp_ ; x_min = 10.0_rp_

!    problem 02

     CASE ( 02 )
       x_l = 2.7_rp_ ; x_u = 7.5_rp_
       f_min = - 1.899599_rp_ ; x_min = 5.145735_rp_

!    problem 03

     CASE ( 03 )
       x_l = - 10.0_rp_ ; x_u = 10.0_rp_
       f_min = - 12.03124_rp_ ; x_min = -6.7745761_rp_

!    problem 04

     CASE ( 04 )
       x_l = 1.9_rp_ ; x_u = 3.9_rp_
       f_min = - 3.85045_rp_ ; x_min = 2.868034_rp_

!    problem 05

     CASE ( 05 )
       x_l = 0.0_rp_ ; x_u = 1.2_rp_
       f_min = -1.48907_rp_ ; x_min = 0.96609_rp_

!    problem 06

     CASE ( 06 )
       x_l = - 10.0_rp_ ; x_u = 10.0_rp_
       f_min = -0.824239_rp_ ; x_min = 0.67956_rp_

!    problem 07

     CASE ( 07 )
       x_l = 2.7_rp_ ; x_u = 7.5_rp_
       f_min = - 1.6013_rp_ ; x_min = 5.19978_rp_

!    problem 08

     CASE ( 08 )
       x_l = - 10.0_rp_ ; x_u = 10.0_rp_
       f_min = - 14.508_rp_ ; x_min = -7.083506_rp_

!    problem 09

     CASE ( 09 )
       x_l = 3.1_rp_ ; x_u = 20.4_rp_
       f_min = - 1.90596_rp_ ; x_min = 17.039_rp_

!    problem 10

     CASE ( 10 )
       x_l = 0.0_rp_ ; x_u = 10.0_rp_
       f_min = - 7.916727_rp_ ; x_min = 7.9787_rp_

!    problem 11

     CASE ( 11 )
       x_l = - 0.5_rp_ * pi ; x_u = 2.0_rp_ * pi
       f_min = - 1.5_rp_ ; x_min = 2.09439_rp_

!    problem 12

     CASE ( 12 )
       x_l = 0.0_rp_ ; x_u = 2.0_rp_ * pi
       f_min = - 1.0_rp_ ; x_min = pi

!    problem 13

     CASE ( 13 )
       x_l = 0.001_rp_ ; x_u = 0.99_rp_
       f_min = - 1.5874_rp_ ; x_min = 1.0_rp_ / SQRT( 2.0_rp_ )

!    problem 14

     CASE ( 14 )
       x_l = 0.0_rp_ ; x_u = 4.0_rp_
       f_min = - 0.788685_rp_ ; x_min = 0.224885_rp_

!    problem 15

     CASE ( 15 )
       x_l = - 5.0_rp_ ; x_u = 5.0_rp_
       f_min = - 0.03553_rp_ ; x_min = 2.41422_rp_

!    problem 16

     CASE ( 16 )
       x_l = - 3.0_rp_ ; x_u = 3.0_rp_
       f_min = 7.515924_rp_ ; x_min = 1.5907_rp_

!    problem 17

     CASE ( 17 )
       x_l = - 4.0_rp_ ; x_u = 4.0_rp_
       f_min = 7.0_rp_ ; x_min = 3.0_rp_

!    problem 18

     CASE ( 18 )
       x_l = 0.0_rp_ ; x_u = 6.0_rp_
       f_min = 0.0_rp_ ; x_min = 2.0_rp_

!    problem 19

     CASE ( 19 )
       x_l = 0.0_rp_ ; x_u = 6.5_rp_
       f_min = - 7.81567_rp_ ; x_min = 5.87287_rp_

!    problem 20

     CASE ( 20 )
       x_l = - 10.0_rp_ ; x_u = 10.0_rp_
       f_min = - 0.0634905_rp_ ; x_min = 1.195137_rp_

!    problem 21

     CASE ( 21 )
       x_l = 0.0_rp_ ; x_u = 10.0_rp_
       f_min = - 9.50835_rp_  ; x_min = 4.79507_rp_

!    problem 22

     CASE ( 22 )
       x_l = 0.0_rp_ ; x_u = 20.0_rp_
       f_min = EXP( - 13.5_rp_ * pi ) - 1.0_rp_ ; x_min = 4.5_rp_ * pi

     END SELECT

     userdata%integer( 1 ) = prob                ! Record problem # prob
     IF ( prob == 6 .OR. prob == 11 ) THEN
       control%print_level = 0
       control%maxit = 1000
     ELSE
       control%print_level = 0
       control%maxit = 100
     END IF
     IF ( prob == 11 ) THEN
       control%reliability_parameter = 10.0_rp_
     ELSE
       control%reliability_parameter = -1.0_rp_
     END IF
     control%lipschitz_estimate_used = 3

     WRITE( 6, "( ' second derivatives provided' )" )
     control%second_derivative_available = .TRUE.
     inform%status = 1                            ! set for initial entry
     CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata,    &
                     eval_FGH = FGH )   ! Solve problem
     IF ( inform%status == 0 ) THEN               ! Successful return
       IF ( ABS( x - x_min ) / MAX( 1.0_rp_, ABS( x_min ) ) <= accurate_x .OR. &
            ABS( f - f_min ) / MAX( 1.0_rp_, ABS( f_min ) ) <= accurate_f ) THEN
         WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0,        &
        &             ' with global minimizer' ) " ) prob, inform%status
       ELSE
         WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0,        &
        &             ' with non-global minimizer' ) " ) prob, inform%status
       END IF
     ELSE                                         ! Error returns
       WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0 )" )      &
         prob, inform%status
     END IF

     WRITE( 6, "( ' no second derivatives provided' )" )
     control%second_derivative_available = .FALSE.
     inform%status = 1                            ! set for initial entry
     CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata,    &
                     eval_FGH = FGH )   ! Solve problem
     IF ( inform%status == 0 ) THEN               ! Successful return
       IF ( ABS( x - x_min ) / MAX( 1.0_rp_, ABS( x_min ) ) <= accurate_x .OR. &
            ABS( f - f_min ) / MAX( 1.0_rp_, ABS( f_min ) ) <= accurate_f ) THEN
         WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0,        &
        &             ' with global minimizer' )" ) prob, inform%status
       ELSE
         WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0,        &
        &             ' with non-global minimizer' )" ) prob, inform%status
       END IF
     ELSE                                         ! Error returns
       WRITE( 6, "( ' problem ', I2, ' UGO_solve exit status = ', I0 )" )      &
         prob, inform%status
     END IF

   END DO
   CALL UGO_terminate( data, control, inform )  ! delete internal workspace
   STOP

   END PROGRAM GALAHAD_UGO_TEST_PROGRAM

     SUBROUTINE FGH( status, x, userdata, f, g, h )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( IN ) :: x
     REAL ( KIND = rp_ ), INTENT( OUT ) :: f, g, h
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

     INTEGER ( KIND = ip_ ) :: k
     REAL ( KIND = rp_ ) :: a, b, c, ca, c2, c3, c4, c5, c6, e, rk, rkp1, s, sa
     REAL ( KIND = rp_ ) :: da, db, bm1
     REAL ( KIND = rp_ ), PARAMETER :: pi = 4.0_rp_ * ATAN( 1.0_rp_ )

!    branch to the desired test problem

     SELECT CASE ( userdata%integer( 1 ) )

!    problem 01

     CASE ( 01 )
       c2 = - 79.0_rp_ / 20.0_rp_
       c3 = 71.0_rp_ / 10.0_rp_
       c4 = 39.0_rp_ / 80.0_rp_
       c5 = - 52.0_rp_ / 25.0_rp_
       c6 = 1.0_rp_ / 6.0_rp_
       f = c6 * x ** 6 + c5 * x ** 5 + c4 * x ** 4 + c3 * x ** 3               &
           + c2 * x ** 2 - x + 0.1_rp_
       g = 6.0_rp_ * c6 * x ** 5 + 5.0_rp_ * c5 * x ** 4  + 4.0_rp_ * c4       &
           * x ** 3 + 3.0_rp_ * c3 * x ** 2 + 2.0_rp_ * c2 * x - 1.0_rp_
       h = 30.0_rp_ * c6 * x ** 4 + 20.0_rp_ * c5 * x ** 3                     &
           + 12.0_rp_ * c4 * x ** 2 + 6.0_rp_ * c3 * x + 2.0_rp_ * c2

!    problem 02

     CASE ( 02 )
       a = 10.0_rp_ / 3.0_rp_
       s = SIN( x )
       sa = SIN( a * x )
       f = s + sa
       g = COS( x ) + a * COS( a * x )
       h = - s - a * a * sa

!    problem 03

     CASE ( 03 )
       f = 0.0_rp_ ; g = 0.0_rp_ ; h = 0.0_rp_
       DO k = 1, 5
         rk = REAL( k, KIND = rp_ )
         rkp1 =  rk + 1.0_rp_
         s = SIN( rkp1 * x + rk )
         f = f - rk * s
         g = g - rk * rkp1 * COS( rkp1 * x + rk )
         h = h + rk * rkp1 * rkp1 * s
       END DO

!    problem 04

     CASE ( 04 )
       e = EXP( - x )
       f = - ( 16.0_rp_ * x ** 2 - 24.0_rp_ * x + 5.0_rp_ ) * e
       g = ( 16.0_rp_ * x ** 2 - 56.0_rp_ * x + 29.0_rp_ ) * e
       h = - ( 16.0_rp_ * x ** 2 - 88.0_rp_ * x + 85.0_rp_ ) * e

!    problem 05

     CASE ( 05 )
       a = 18.0_rp_
       s = SIN( a * x )
       c = COS( a * x )
       f = - ( 1.4_rp_ - 3.0_rp_ * x ) * s
       g = 3.0_rp_ * s - a * ( 1.4_rp_ - 3.0_rp_ * x ) * c
       h = 6.0_rp_ * a * c + a * a * ( 1.4_rp_ - 3.0_rp_ * x ) * s

!    problem 06

     CASE ( 06 )
       s = SIN( x )
       c = COS( x )
       e = EXP( - x ** 2 )
       f = - ( x + s ) * e
       g = - ( 1.0_rp_ + c ) * e + 2.0_rp_ * x * ( x + s ) * e
       h = s * e + 4.0_rp_ * x * ( 1.0_rp_ + c ) * e                           &
           + ( 2.0_rp_  + 4.0_rp_ * x * x ) * ( x + s ) * e

!    problem 07

     CASE ( 07 )
       a = 10.0_rp_ / 3.0_rp_
       s = SIN( x )
       sa = SIN( a * x )
       f = s + sa + LOG( x ) - 0.84_rp_ * x + 3.0_rp_
       g = COS( x ) + a * COS( a * x ) + 1.0_rp_ / x - 0.84_rp_
       h = - s - a * a * sa - 1.0_rp_ / x ** 2

!    problem 08

     CASE ( 08 )
       f = 0.0_rp_ ; g = 0.0_rp_ ; h = 0.0_rp_
       DO k = 1, 5
         rk = REAL( k, KIND = rp_ )
         rkp1 =  rk + 1.0_rp_
         f = f - rk * COS( rkp1 * x + rk )
         g = g + rk * rkp1 * SIN( rkp1 * x + rk )
         h = h + rk * rkp1 * rkp1 * COS( rkp1 * x + rk )
       END DO

!    problem 09

     CASE ( 09 )
       a = 2.0_rp_ / 3.0_rp_
       s = SIN( x )
       sa = SIN( a * x )
       f = s + sa
       g = COS( x ) + a * COS( a * x )
       h = - s - a * a * sa

!    problem 10

     CASE ( 10 )
       s = SIN( x )
       c = COS( x )
       f = - x * s
       g = - s - x * c
       h = - 2.0_rp_ * c + x * s

!    problem 11

     CASE ( 11 )
       a = 2.0_rp_
       c = COS( x )
       ca = COS( a * x )
       f = 2.0_rp_ * c + ca
       g = - 2.0_rp_ * SIN( x ) - a * SIN( a * x )
       h = - 2.0_rp_ * c - a * 8 * ca

!    problem 12

     CASE ( 12 )
       s = SIN( x )
       c = COS( x )
       f = s ** 3 + c ** 3
       g = 3.0_rp_ * c * s ** 2 - 3.0_rp_ * s * c ** 2
       h = - 3.0_rp_ * s ** 3 - 3.0_rp_ * c ** 3                               &
           + 6.0_rp_ * c ** 2 * s + 6.0_rp_ * c * s ** 2

!    problem 13

     CASE ( 13 )
       a =  2.0_rp_ / 3.0_rp_
       b = 1.0_rp_ / 3.0_rp_
       bm1 = b - 1.0_rp_
       c = 1.0_rp_ - x ** 2
       f = - x ** a - c ** b
       g = - a * x ** ( a - 1.0_rp_ ) + 2.0_rp_ * b * x * c ** bm1
       h = - a * ( a - 1.0_rp_ ) * x ** ( a - 2.0_rp_ ) + 2.0_rp_ * b          &
           * c ** bm1 - 4.0_rp_ * b * bm1 * x * x * c ** ( b - 2.0_rp_ )

!    problem 14

     CASE ( 14 )
       a = 2.0_rp_ * pi
       e = EXP( - x )
       sa = SIN( a * x )
       ca = COS( a * x )
       f = - e * sa
       g = e * sa - a * e * ca
       h = - e * sa + 2.0_rp_ * a * e * ca + a * a * e * sa

!    problem 15

     CASE ( 15 )
       a = x ** 2 - 5.0_rp_ * x + 6.0_rp_
       b = x ** 2 + 1.0_rp_
       da = 2.0_rp_ * x - 5.0_rp_
       db = 2.0_rp_ * x
       f = a / b
       g = da / b - a * db / b ** 2
       h = 2.0_rp_ / b - 2.0_rp_ * da * db / b ** 2                            &
           - a * 2.0_rp_ / b ** 2 + 2.0_rp_ * a * db * db / b ** 3

!    problem 16

     CASE ( 16 )
       a = ( x - 3.0_rp_ )
       e = EXP( 0.5_rp_ * x ** 2 )
       f = 2.0_rp_ * a ** 2 + e
       g = 4.0_rp_ * a + x * e
       h = 4.0_rp_ + ( x ** 2 + 1.0_rp_ ) * e

!    problem 17

     CASE ( 17 )
       c2 = 27.0_rp_
       c4 = - 15.0_rp_
       c6 = 1.0_rp_
       f = c6 * x ** 6 + c4 * x ** 4 + c2 * x ** 2 + 250.0_rp_
       g = 6.0_rp_ * c6 * x ** 5 + 4.0_rp_ * c4 * x ** 3 + 2.0_rp_ * c2 * x
       h = 30.0_rp_ * c6 * x ** 4 + 12.0_rp_ * c4 * x ** 2 + 2.0_rp_ * c2

!    problem 18

     CASE ( 18 )
       a = x - 2.0_rp_
       IF ( x <= 3.0_rp_ ) THEN
         f = a ** 2
         g = 2.0_rp_ * a
         h = 2.0_rp_
       ELSE
         f = 2.0_rp_ * LOG( a ) + 1.0_rp_
         g = 1.0_rp_ / a
         h = - 1.0_rp_ / a ** 2
       END IF

!    problem 19

     CASE ( 19 )
       a = 3.0_rp_
       sa = SIN( a * x )
       f = - x + sa - 1.0_rp_
       g = - 1.0_rp_ + a * COS( a * x )
       h = - a * a * sa

!    problem 20

     CASE ( 20 )
       e = EXP( - x ** 2 )
       s = SIN( x )
       c = COS( x )
       f = - ( x - s ) * e
       g = - ( 1.0_rp_ - c ) * e + 2.0_rp_ * x * ( x - s ) * e
       h = - s * e + 4.0_rp_ * x * ( 1.0_rp_ - c ) * e                         &
           - 4.0_rp_ * x * x * ( x - s ) * e

!    problem 21

     CASE ( 21 )
       a = 2.0_rp_
       s = SIN( x )
       c = COS( x )
       sa = SIN( a * x )
       ca = COS( a * x )
       f = x * ( s + ca )
       g = ( s + ca ) + x * ( c - a * sa )
       h = 2.0_rp_ * ( c - a * sa ) - x * ( s + a * a * ca )

!    problem 22

     CASE ( 22 )
       a = - 3.0_rp_
       e = EXP( a * x )
       s = SIN( x )
       c = COS( x )
       f = e - s ** 3
       g = a * e - 3.0_rp_ * c * s ** 2
       h = a * a * e  + 3.0_rp_ * s ** 3 - 6.0_rp_ * s * c ** 2

!    out of range

     CASE DEFAULT
       status = 1
       RETURN
     END SELECT

     status = 0
     RETURN
     END SUBROUTINE FGH
