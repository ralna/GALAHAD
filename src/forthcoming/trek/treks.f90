! THIS VERSION: GALAHAD 5.3 - 2025-07-01 AT 12:50 GMT.
   PROGRAM GALAHAD_TREK_EXAMPLE
   USE GALAHAD_TREK_double         ! double precision version
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 4, m = 4, h_ne = 7
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C
   REAL ( KIND = wp ), DIMENSION( m ) :: radius
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   TYPE ( TREK_data_type ) :: data
   INTEGER :: stat, i
   H%n = n ; H%ne = h_ne
   CALL SMT_put( H%type, 'COORDINATE', stat ) ! storage for H
   ALLOCATE( H%row( h_ne ), H%col( h_ne ), H%val( h_ne ) )
   H%row = (/ 1, 2, 2, 3, 3, 4, 4 /)
   H%col = (/ 1, 1, 2, 2, 3, 3, 4 /) ; 
   H%val = (/ 2.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /)
   C = (/ - 3.0_wp, - 2.0_wp, - 6.0_wp, - 9.0_wp /)
   RADIUS( 1 ) = 10.0_wp
   CALL TREK_initialize( data, control, inform )
   control%solver = 'ma57 '
   control%exact_shift = .TRUE.
   DO i = 1, m
     CALL TREK_solve( n, H, C, RADIUS( i ), X, data, control, inform,          &
                      resolve = i > 1 )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' radius, ||x||, f, multiplier =',  2ES11.4, 2ES12.4 )" )  &
         RADIUS( i ), TWO_NORM( X ), inform%obj, inform%multiplier
       IF ( i < m ) RADIUS( i + 1 ) = inform%next_radius
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
       STOP
     END IF
     WRITE( 6, "( ' total time TREK = ', F0.2 )" ) inform%time%clock_total
   END DO
   CALL TREK_terminate( data, control, inform )
   DEALLOCATE( H%type, H%row, H%col, H%val )
   END PROGRAM GALAHAD_TREK_EXAMPLE
