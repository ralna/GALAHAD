! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LSRT_test_interface
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LSRT_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 50, m = 2 * n  ! problem dimensions
   INTEGER ( KIND = ip_ ) :: i, status, i_power
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X, V
   REAL ( KIND = rp_ ), DIMENSION( m ) :: U
   REAL ( KIND = rp_ ) :: power, weight, lambda_x

   TYPE ( LSRT_full_data_type ) :: data
   TYPE ( LSRT_control_type ) :: control
   TYPE ( LSRT_inform_type ) :: inform

   CALL LSRT_initialize( data, control, inform )
   control%print_level = 0
   CALL LSRT_import_control( control, data, status )
   power = 3.0_rp_ ; weight = 1.0_rp_ ; status = 1
   U = 1.0_rp_
   DO
     CALL LSRT_solve_problem( data, status, m, n, power, weight, X, U, V )
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
   CALL LSRT_information( data, inform, status )
   WRITE( 6, "( ' LSRT_solve exit status = ', I0, ', f = ', F0.2  )" )         &
     status, inform%obj
!  WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!            inform%iter + inform%iter_pass2, f, inform%multiplier
   CALL LSRT_terminate( data, control, inform ) ! delete internal workspace

!  check the multiplier, for p = 3 and for p = 2 (when it is sigma): the
!  solution satisfies ( A^T A + lambda I ) x = A^T b, whose first component
!  gives lambda = 2 / x_1 - 2

   DO i_power = 2, 3
     CALL LSRT_initialize( data, control, inform )
     control%print_level = 0
     CALL LSRT_import_control( control, data, status )
     power = REAL( i_power, KIND = rp_ ) ; weight = 1.0_rp_ ; status = 1
     U = 1.0_rp_
     DO
       CALL LSRT_solve_problem( data, status, m, n, power, weight, X, U, V )
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
     CALL LSRT_information( data, inform, status )
     CALL LSRT_terminate( data, control, inform ) ! delete internal workspace
     lambda_x = 2.0_rp_ / X( 1 ) - 2.0_rp_
     IF ( inform%status /= 0 .OR. ABS( inform%multiplier - lambda_x ) >        &
          EPSILON( 1.0_rp_ ) ** 0.25 * MAX( 1.0_rp_, lambda_x ) ) THEN
       WRITE( 6, "( ' multiplier = ', ES12.4, ' but x implies ', ES12.4 )" )   &
         inform%multiplier, lambda_x
       STOP 1
     END IF
   END DO

!  a single pass (p = 2) stopped by the iteration limit returns the
!  current point

   CALL LSRT_initialize( data, control, inform )
   control%print_level = 0 ; control%itmax = 1
   CALL LSRT_import_control( control, data, status )
   power = 2.0_rp_ ; weight = 1.0_rp_ ; status = 1
   U = 1.0_rp_
   DO
       CALL LSRT_solve_problem( data, status, m, n, power, weight, X, U, V )
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
   CALL LSRT_information( data, inform, status )
   CALL LSRT_terminate( data, control, inform ) ! delete internal workspace
   IF ( inform%status /= 0 .OR. inform%iter /= 1 ) THEN
     WRITE( 6, "( ' iteration limit: status = ', I0, ', iter = ', I0 )" )     &
       inform%status, inform%iter
     STOP 1
   END IF

   END PROGRAM GALAHAD_LSRT_test_interface
