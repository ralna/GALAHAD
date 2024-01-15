! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_GLRT_test_interface
   USE GALAHAD_KINDS_precision
   USE GALAHAD_GLRT_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 100             ! problem dimension
   INTEGER ( KIND = ip_ ) :: i, status, unit_m, new_weight
   REAL ( KIND = rp_ ) :: weight
   REAL ( KIND = rp_ ) :: power = 3.0_rp_
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   TYPE ( GLRT_full_data_type ) :: data
   TYPE ( GLRT_control_type ) :: control
   TYPE ( GLRT_inform_type ) :: inform

   CALL GLRT_initialize( data, control, inform )
   DO unit_m = 0, 1 ! use a unit M ?
     control%unitm = unit_m == 1
     CALL GLRT_import_control( control, data, status )
     DO new_weight = 0, 1 ! resolve with a larger weight ?
       IF ( new_weight == 0 ) THEN
         weight = 1.0_rp_ ; status = 1
       ELSE
         weight = 10.0_rp_ ; status = 4
       END IF
       R = 1.0_rp_
       DO ! Iteration loop to find the minimizer
         CALL GLRT_solve_problem( data, status, n, power, weight, X, R, VECTOR )
         SELECT CASE( status ) ! Branch as a result of status
         CASE( 2 ) !  form the preconditioned gradient
           VECTOR = VECTOR / 2.0_rp_
         CASE ( 3 ) !  form the Hessian-vector product
           H_vector( 1 ) =  2.0_rp_ * VECTOR( 1 ) + VECTOR( 2 )
           DO i = 2, n - 1
             H_vector( i ) = VECTOR( i - 1 ) + 2.0_rp_ * VECTOR( i ) +         &
                             VECTOR( i + 1 )
           END DO
           H_vector( n ) = VECTOR( n - 1 ) + 2.0_rp_ * VECTOR( n )
           VECTOR = H_vector
         CASE ( 4 ) ! restart
           R = 1.0_rp_
         CASE ( 0 )  ! successful return
           EXIT
         CASE DEFAULT ! error returns
           EXIT
         END SELECT
       END DO ! end of iteration loop
       CALL GLRT_information( data, inform, status )
       WRITE( 6, "( ' MR = ', 2I1, ', GLRT_solve exit status = ', I0,          &
      &       ', f = ', F0.2 )" ) unit_m, new_weight, inform%status, inform%obj
!      WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6,           &
!     &  2ES12.4 )") inform%iter + inform%iter_pass2, f, inform%multiplier
     END DO
   END DO
   CALL GLRT_terminate( data, control, inform ) !  delete internal workspace

   STOP
   END PROGRAM GALAHAD_GLRT_test_interface
