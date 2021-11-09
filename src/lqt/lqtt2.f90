! THIS VERSION: GALAHAD 3.3 - 11/10/2021 AT 08:30 GMT.
   PROGRAM GALAHAD_LQT_2d_test_deck
   USE GALAHAD_LQT_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER :: status, test
   REAL ( KIND = wp ) :: h_11, h_12, h_22, g_1, g_2, radius
   REAL ( KIND = wp ) :: x_1, x_2, lambda, r1, r2

   DO test = 0, 3
     SELECT CASE ( test )
     CASE ( 0 )
       h_11 = 1.0_wp ; h_12 = 2.0_wp ; h_22 = 2.0_wp
       g_1 = 1.0_wp ; g_2 = 2.0_wp ; radius = 1.0_wp
     CASE ( 1 )
       h_11 = 1.0_wp ; h_12 = 2.0_wp ; h_22 = 6.0_wp
       g_1 = 1.0_wp ; g_2 = 2.0_wp ; radius = 10.0_wp
     CASE ( 2 )
       h_11 = 1.0_wp ; h_12 = 2.0_wp ; h_22 = 2.0_wp
       g_1 = 0.0_wp ; g_2 = 0.0_wp ; radius = 1.0_wp
     CASE ( 3 )
       h_11 = 1.0_wp ; h_12 = 0.0_wp ; h_22 = - 2.0_wp
       g_1 = 1.0_wp ; g_2 = 0.0_wp ; radius = 1.0_wp
     END SELECT
     CALL LQT_solve_2d( h_11, h_12, h_22, g_1, g_2, radius,                    &
                        x_1, x_2, lambda, status )
     WRITE( 6, "( ' test ', I0, ' status = ', I0 )" ) test, status
     WRITE( 6, "( ' x_1, x_2, lambda = ', 3ES12.4 )" ) x_1, x_2, lambda
     WRITE( 6, "( ' r_1, r_2, ||x||, radius = ', 4ES12.4 )" )                  &
       h_11 * x_1 + h_12 * x_2 + lambda * x_1 + g_1,                           &
       h_12 * x_1 + h_22 * x_2 + lambda * x_2 + g_2,                           &
       SQRT( x_1 ** 2 + x_2 ** 2 ), radius

   END DO
   STOP
   END PROGRAM GALAHAD_LQT_2d_test_deck
