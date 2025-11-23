! THIS VERSION: GALAHAD 5.4 - 2025-11-22 AT 09:10 GMT.
   PROGRAM GALAHAD_NREK_EXAMPLE
! double precision version
   USE GALAHAD_NREK_double, ONLY: SMT_type, SMT_put, NREK_control_type,        &
          NREK_inform_type, NREK_data_type, NREK_initialize, NREK_solve,       &
          NREK_terminate
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 10000, m = 4, h_ne = 2 * n - 1
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C
   REAL ( KIND = wp ) :: power, weight
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   TYPE ( NREK_control_type ) :: control
   TYPE ( NREK_inform_type ) :: inform
   TYPE ( NREK_data_type ) :: data
   INTEGER :: stat, i
   H%n = n ; H%ne = h_ne
   CALL SMT_put( H%type, 'COORDINATE', stat ) ! Specify co-ordinate for H
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   DO i = 1, n
    H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_wp
   END DO
   DO i = 1, n - 1
    H%row( n + i ) = i + 1 ; H%col( n + i ) = i ; H%val( n + i ) = 1.0_wp
   END DO
   C = 1.0_wp ! c is a vector of ones
   power = 3.0_wp ! power (cubic regularization)
   weight = 10.0_wp ! initial weight
   CALL NREK_initialize( data, control, inform )
   control%linear_solver = 'pbtr '
   DO i = 1, m ! loop over a sequence of decreasing radii
     control%new_weight = i > 1
     inform%time%clock_total = 0.0 ! reset time
     CALL NREK_solve( n, H, C, power, weight, X, data, control, inform )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' weight, ||x||, f, multiplier =',  2ES11.4, 2ES12.4 )" )  &
         weight, TWO_NORM( X ), inform%obj_regularized, inform%multiplier
       IF ( i < m ) weight = inform%next_weight ! pick the next (larger) weight
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
       STOP
     END IF
     WRITE( 6, "( ' total time NREK = ', F0.2 )" ) inform%time%clock_total
   END DO
   CALL NREK_terminate( data, control, inform )
   DEALLOCATE( H%type, H%row, H%col, H%val )
   END PROGRAM GALAHAD_NREK_EXAMPLE
