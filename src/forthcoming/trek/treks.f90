! THIS VERSION: GALAHAD 5.3 - 2025-05-20 AT 07:50 GMT.
   PROGRAM GALAHAD_TREK_EXAMPLE
   USE GALAHAD_TREK_double         ! double precision version
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 4, m = 4, h_ne = 7
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C
   REAL ( KIND = wp ), DIMENSION( m ) :: RADIUS
   REAL ( KIND = wp ), DIMENSION( n, m ) :: X
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   TYPE ( TREK_data_type ) :: data
   INTEGER :: stat, i
   H%n = n ; H%ne = h_ne
   CALL SMT_put( H%type, 'COORDINATE', stat ) ! storage for A
   ALLOCATE( H%row( h_ne ), H%col( h_ne ), H%val( h_ne ) )
   H%row = (/ 1, 2, 2, 3, 3, 4, 4 /)
   H%col = (/ 1, 1, 2, 2, 3, 3, 4 /) ; 
   H%val = (/ 2.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /)
   C = (/ - 3.0_wp, - 2.0_wp, - 6.0_wp, - 9.0_wp /)
   RADIUS( 1 ) = 10.0_wp
   CALL TREK_initialize( data, control, inform )
   control%print_level = 2
   control%solver = 'ma57 '
   control%exact_shift = .TRUE.
   CALL TREK_solve_all( n, m, H, C, RADIUS, X, data, control, inform )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )           &
       inform%n_vec, inform%error
     DO i = 1, m
       WRITE( 6, "( ' radius ', I2, ' = ',  ES11.4, ' ||x|| = ', ES11.4 )" )   &
         i, RADIUS( i ), TWO_NORM( X( 1 : n, i ) )
     END DO
   ELSE
     WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
   END IF
   CALL TREK_terminate( data, control, inform )
   DEALLOCATE( H%type, H%row, H%col, H%val )
   END PROGRAM GALAHAD_TREK_EXAMPLE
