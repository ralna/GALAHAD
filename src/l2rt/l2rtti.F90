! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_L2RT_test_interface
   USE GALAHAD_KINDS_precision
   USE GALAHAD_L2RT_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 50, m = 2 * n  ! problem dimensions
   INTEGER ( KIND = ip_ ) :: i, status
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X, V
   REAL ( KIND = rp_ ), DIMENSION( m ) :: U
   REAL ( KIND = rp_ ) :: power, weight, shift

   TYPE ( L2RT_full_data_type ) :: data
   TYPE ( L2RT_control_type ) :: control
   TYPE ( L2RT_inform_type ) :: inform

   CALL L2RT_initialize( data, control, inform )
   power = 3.0_rp_ ; weight = 1.0_rp_ ; shift = 1.0_rp_ ; status = 1
   U = 1.0_rp_
   control%print_level = 0
   CALL L2RT_import_control( control, data, status )
   DO
     CALL L2RT_solve_problem( data, status, m, n, power, weight, shift,        &
                              X, U, V )
     SELECT CASE( status )  ! Branch as a result of status
     CASE( 2 )                     !  Form u <- u + A * v
       U( : n ) = U( : n ) + V
       DO i = 1, n
         U( n + i ) = U( n + i ) + i * V( i )
       END DO
     CASE( 3 )                     !  Form v <- v + A^T * u
       V = V + U( : n )            !  A^T = ( I : diag(1:n) )
       DO i = 1, n
         V( i ) = V( i ) + i * U( n + i )
       END DO
     CASE ( 4 )                    ! Restart
        U = 1.0_rp_
     CASE DEFAULT
        EXIT
     END SELECT
   END DO
   CALL L2RT_information( data, inform, status )
   WRITE( 6, "( ' L2RT_solve exit status = ', I0, ', f = ', F0.2  )" )         &
     status, inform%obj
!  WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!            inform%iter + inform%iter_pass2, f, inform%multiplier
   CALL L2RT_terminate( data, control, inform ) ! delete internal workspace

   END PROGRAM GALAHAD_L2RT_test_interface
