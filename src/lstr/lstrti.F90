! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LSTR_test_interface
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LSTR_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 50, m = 2 * n   ! problem dimensions
!  INTEGER ( KIND = ip_ ), PARAMETER :: n = 1, m = 2 * n   ! problem dimensions
   INTEGER ( KIND = ip_ ) :: i, new_radius, problem, status
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X, V, X_ref
   REAL ( KIND = rp_ ), DIMENSION( m ) :: U
   REAL ( KIND = rp_ ) :: radius, lambda_x
   INTEGER ( KIND = ip_ ) :: failures = 0

   TYPE ( LSTR_full_data_type ) :: data
   TYPE ( LSTR_control_type ) :: control
   TYPE ( LSTR_inform_type ) :: inform

   CALL LSTR_initialize( data, control, inform )
   DO new_radius = 0, 1 ! resolve with a smaller radius ?
     control%print_level = 0
     CALL LSTR_import_control( control, data, status )
     IF ( new_radius == 0 ) THEN
       radius = 1.0_rp_ ; status = 1
     ELSE
       radius = 0.1_rp_ ; status = 5
     END IF
     U = 1.0_rp_
     DO
       CALL LSTR_solve_problem( data, status, m, n, radius, X, U, V )
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
     CALL LSTR_information( data, inform, status )
     WRITE( 6, "( I1, ' LSTR_solve exit status = ', I0, ', f = ', F0.2  )" )   &
       new_radius, status, inform%r_norm

!  the solution satisfies ( A^T A + lambda I ) x = A^T b, whose first
!  component gives lambda = 2 / x_1 - 2

     lambda_x = 2.0_rp_ / X( 1 ) - 2.0_rp_
     IF ( inform%status /= 0 .OR. ABS( inform%multiplier - lambda_x ) >        &
          EPSILON( 1.0_rp_ ) ** 0.25 * MAX( 1.0_rp_, lambda_x ) ) THEN
       WRITE( 6, "( ' multiplier = ', ES12.4, ' but x implies ', ES12.4 )" )   &
         inform%multiplier, lambda_x
       failures = failures + 1
     END IF
     IF ( ABS( inform%x_norm - radius ) > EPSILON( 1.0_rp_ ) ** 0.25 ) THEN
       WRITE( 6, "( ' x_norm = ', ES12.4, ' but radius = ', ES12.4 )" )        &
         inform%x_norm, radius
       failures = failures + 1
     END IF
     IF ( new_radius == 0 ) X_ref = X
!    WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )") &
!              inform%iter + inform%iter_pass2, f, inform%multiplier
   END DO
   CALL LSTR_terminate( data, control, inform ) ! delete internal workspace

!  with radius 100 the solution is interior; a restart with radius 1 must
!  then give the solution found above with radius 1

   CALL LSTR_initialize( data, control, inform )
   DO new_radius = 0, 1
     control%print_level = 0
     CALL LSTR_import_control( control, data, status )
     IF ( new_radius == 0 ) THEN
       radius = 100.0_rp_ ; status = 1
     ELSE
       radius = 1.0_rp_ ; status = 5
     END IF
     U = 1.0_rp_
     DO
       CALL LSTR_solve_problem( data, status, m, n, radius, X, U, V )
       SELECT CASE( status )  ! Branch as a result of status
       CASE( 2 )                     !  Form u <- u + A * v
         U( : n ) = U( : n ) + V
         DO i = 1, n
           U( n + i ) = U( n + i ) + i * V( i )
         END DO
       CASE( 3 )                     !  Form v <- v + A^T * u
         V = V + U( : n )
         DO i = 1, n
           V( i ) = V( i ) + i * U( n + i )
         END DO
       CASE ( 4 )                    ! Restart
          U = 1.0_rp_
       CASE DEFAULT
          EXIT
       END SELECT
     END DO
   END DO
   CALL LSTR_information( data, inform, status )
   IF ( inform%status /= 0 .OR. MAXVAL( ABS( X - X_ref ) ) >                   &
        EPSILON( 1.0_rp_ ) ** 0.25 ) THEN
     WRITE( 6, "( ' restart after an interior solution failed, status = ',     &
    &             I0 )" ) inform%status
     failures = failures + 1
   ELSE
     WRITE( 6, "( ' restart after an interior solution: ok' )" )
   END IF
   CALL LSTR_terminate( data, control, inform ) ! delete internal workspace
   IF ( failures > 0 ) STOP 1

   END PROGRAM GALAHAD_LSTR_test_interface
