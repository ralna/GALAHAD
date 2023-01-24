! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LQT_2d_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LQT_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ) :: status, test
   REAL ( KIND = rp_ ) :: h_11, h_12, h_22, g_1, g_2, radius
   REAL ( KIND = rp_ ) :: x_1, x_2, lambda

   DO test = 0, 3
     SELECT CASE ( test )
     CASE ( 0 )
       h_11 = 1.0_rp_ ; h_12 = 2.0_rp_ ; h_22 = 2.0_rp_
       g_1 = 1.0_rp_ ; g_2 = 2.0_rp_ ; radius = 1.0_rp_
     CASE ( 1 )
       h_11 = 1.0_rp_ ; h_12 = 2.0_rp_ ; h_22 = 6.0_rp_
       g_1 = 1.0_rp_ ; g_2 = 2.0_rp_ ; radius = 10.0_rp_
     CASE ( 2 )
       h_11 = 1.0_rp_ ; h_12 = 2.0_rp_ ; h_22 = 2.0_rp_
       g_1 = 0.0_rp_ ; g_2 = 0.0_rp_ ; radius = 1.0_rp_
     CASE ( 3 )
       h_11 = 1.0_rp_ ; h_12 = 0.0_rp_ ; h_22 = - 2.0_rp_
       g_1 = 1.0_rp_ ; g_2 = 0.0_rp_ ; radius = 1.0_rp_
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
   WRITE( 6, "( /, ' tests completed' )" )
   END PROGRAM GALAHAD_LQT_2d_test_deck
