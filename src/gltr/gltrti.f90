! THIS VERSION: GALAHAD 3.3 - 16/12/2021 AT 16:00 GMT.
   PROGRAM GALAHAD_GLTR_test_interface
   USE GALAHAD_GLTR_DOUBLE                   ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 100             ! problem dimension
   INTEGER :: i, status, unit_m, new_radius
   REAL ( KIND = wp ) :: radius
   REAL ( KIND = wp ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   TYPE ( GLTR_full_data_type ) :: data
   TYPE ( GLTR_control_type ) :: control        
   TYPE ( GLTR_inform_type ) :: inform

   CALL GLTR_initialize( data, control, inform )
   DO unit_m = 0, 1 ! use a unit M ?
     control%unitm = unit_m == 1
     CALL GLTR_import_control( control, data, status )
     radius = 1.0_wp
     DO new_radius = 0, 1 ! resolve with a smaller radius ?
       IF ( new_radius == 0 ) THEN
         radius = 1.0_wp ; status = 1
       ELSE
         radius = 0.1_wp ; status = 4
       END IF
       R = 1.0_wp
       DO ! Iteration loop to find the minimizer
         CALL GLTR_solve_problem( data, status, n, radius, X, R, VECTOR )
         SELECT CASE( status ) ! Branch as a result of status
         CASE( 2 ) !  form the preconditioned gradient
           VECTOR = VECTOR / 2.0_wp
         CASE ( 3 ) !  form the Hessian-vector product
           H_vector( 1 ) =  2.0_wp * VECTOR( 1 ) + VECTOR( 2 )
           DO i = 2, n - 1
             H_vector( i ) = VECTOR( i - 1 ) + 2.0_wp * VECTOR( i ) +          &
                             VECTOR( i + 1 )
           END DO
           H_vector( n ) = VECTOR( n - 1 ) + 2.0_wp * VECTOR( n )
           VECTOR = H_vector 
         CASE ( 5 ) ! restart
           R = 1.0_wp
         CASE ( 0 )  ! successful return
           EXIT
         CASE DEFAULT ! error returns
           EXIT
         END SELECT
       END DO ! end of iteration loop
       CALL GLTR_information( data, inform, status )
       WRITE( 6, "( ' MR = ', 2I1, ', GLTR_solve exit status = ', I0,          &
      &       ', f = ', F0.2 )" ) unit_m, new_radius, inform%status, inform%obj
!      WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6,           &
!     &  2ES12.4 )") inform%iter + inform%iter_pass2, f, inform%multiplier
     END DO
   END DO
   CALL GLTR_terminate( data, control, inform ) !  delete internal workspace

   STOP
   END PROGRAM GALAHAD_GLTR_test_interface
