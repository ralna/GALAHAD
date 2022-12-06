! THIS VERSION: GALAHAD 3.3 - 20/12/2021 AT 08:50 GMT.
   PROGRAM GALAHAD_LSTR_test_interface
   USE GALAHAD_LSTR_DOUBLE                    ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )  ! set precision
   INTEGER, PARAMETER :: n = 50, m = 2 * n    ! problem dimensions
!  INTEGER, PARAMETER :: n = 1, m = 2 * n    ! problem dimensions
   INTEGER :: i, new_radius, problem, status
   REAL ( KIND = wp ), DIMENSION( n ) :: X, V
   REAL ( KIND = wp ), DIMENSION( m ) :: U
   REAL ( KIND = wp ) :: radius

   TYPE ( LSTR_full_data_type ) :: data
   TYPE ( LSTR_control_type ) :: control
   TYPE ( LSTR_inform_type ) :: inform

   CALL LSTR_initialize( data, control, inform )
   DO new_radius = 0, 1 ! resolve with a smaller radius ?
     IF ( new_radius == 0 ) THEN
       radius = 1.0_wp ; status = 1
     ELSE
       radius = 0.1_wp ; status = 5
     END IF
     control%print_level = 0
     CALL LSTR_import_control( control, data, status )
     U = 1.0_wp
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
          U = 1.0_wp
       CASE DEFAULT      
          EXIT
       END SELECT
     END DO
     CALL LSTR_information( data, inform, status )
     WRITE( 6, "( I1, ' LSTR_solve exit status = ', I0, ', f = ', F0.2  )" ) &
       new_radius, status, inform%r_norm
!    WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!              inform%iter + inform%iter_pass2, f, inform%multiplier
   END DO
   CALL LSTR_terminate( data, control, inform ) ! delete internal workspace

   END PROGRAM GALAHAD_LSTR_test_interface
