! THIS VERSION: GALAHAD 5.4 - 2025-11-15 AT 10:00 GMT.
   PROGRAM GALAHAD_TREK_EXAMPLE
! double precision version
   USE GALAHAD_TREK_double, ONLY: SMT_type, SMT_put, TREK_control_type,        &
          TREK_inform_type, TREK_data_type, TREK_initialize, TREK_solve,       &
          TREK_terminate
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 10000, m = 4, h_ne = 2 * n - 1
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C
   REAL ( KIND = wp ) :: radius
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   TYPE ( TREK_data_type ) :: data
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
   radius = 10.0_wp ! initial radius
   CALL TREK_initialize( data, control, inform )
   control%linear_solver = 'pbtr '
   DO i = 1, m ! loop over a sequence of decreasing radii
     control%new_radius = i > 1
     inform%time%clock_total = 0.0 ! reset time
     CALL TREK_solve( n, H, C, radius, X, data, control, inform )
    IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' radius, ||x||, f, multiplier =',  2ES11.4, 2ES12.4 )" )  &
         radius, TWO_NORM( X ), inform%obj, inform%multiplier
       IF ( i < m ) radius = inform%next_radius ! pick the next (smaller) radius
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
       STOP
     END IF
     WRITE( 6, "( ' total time TREK = ', F0.2 )" ) inform%time%clock_total
   END DO
   CALL TREK_terminate( data, control, inform )
   DEALLOCATE( H%type, H%row, H%col, H%val )
   END PROGRAM GALAHAD_TREK_EXAMPLE
