! THIS VERSION: GALAHAD 5.3 - 2025-05-20 AT 07:50 GMT.
   PROGRAM GALAHAD_TREK_TEST
   USE GALAHAD_TREK_double   ! double precision version
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
!  INTEGER, PARAMETER :: n = 10, m = 10, h_ne = n
!  INTEGER, PARAMETER :: n = 100, m = 10, h_ne = n
   INTEGER, PARAMETER :: n = 100000, m = 10, h_ne = n
!  INTEGER, PARAMETER :: n = 1000000, m = 10, h_ne = n
!  INTEGER, PARAMETER :: n = 2000000, m = 4, h_ne = n
!  INTEGER, PARAMETER :: n = 2000000, m = 10, h_ne = n
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C, RADIUS
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   TYPE ( TREK_data_type ) :: data
   INTEGER :: i, pass, stat
   REAL :: cpu_start, cpu_end

   WRITE( 6, "( /, ' Exhaustive tests of trek' )" )

!  set up data

   H%n = n ; H%ne = h_ne
   CALL SMT_put( H%type, 'COORDINATE', stat ) ! storage for A
   ALLOCATE( H%row( h_ne ), H%col( h_ne ), H%val( h_ne ) )
   H%row = (/ ( i, i = 1, n ) /)
   H%col = (/ ( i, i = 1, n ) /)
   C = - 1.0_wp
   RADIUS( 1 ) = 1.0_wp
 
   DO pass = 1, 2
     IF ( pass == 1 ) THEN
       WRITE( 6, "( /, ' testing solve (convex) ', / )" )
       H%val = (/ ( REAL( i, wp ), i = 1, n ) /)
!      H%val = (/ ( REAL( i, wp ) / REAL( n, wp ), i = 1, n ) /)
     ELSE
       WRITE( 6, "( /, ' testing solve (non-convex) ', / )" )
!      H%val = (/ ( - REAL( i, wp ), i = 1, n ) /)
       H%val = (/ ( - REAL( i, wp ) / REAL( n, wp ), i = 1, n ) /)
     END IF
     CALL TREK_initialize( data, control, inform )
!    control%print_level = 1
!    control%trs_control%print_level = 1
     control%sls_control%print_level = 2
     control%sls_control%print_level_solver = 2
     control%sls_control%ordering = 7
     control%solver = 'ma57 '
!    control%solver = 'pbtr '
!    control%maxit = 10
     control%maxit = 100
     control%exact_shift = .TRUE.
!    control%reorthogonalize = .FALSE.
     CALL CPU_time( cpu_start )
     CALL TREK_solve( n, H, C, RADIUS( 1 ), X, data, control, inform )
     CALL CPU_time( cpu_end )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' radius, ||x||, shift, obj = ', 3ES11.4, ES12.4 )" )      &
         RADIUS( 1 ), TWO_NORM( X( 1 : n ) ), inform%multiplier, inform%obj
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' elapsed cpu time = ', F0.2 )" ) cpu_end - cpu_start

     WRITE( 6, "( /, ' testing resolve ', / )" )

     DO i = 2, m
       RADIUS( i ) = inform%next_radius   
       CALL CPU_time( cpu_start )
       CALL TREK_solve( n, H, C, RADIUS( i ), X, data, control, inform,        &
                        resolve = .TRUE. )
       CALL CPU_time( cpu_end )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )       &
           inform%n_vec, inform%error
         WRITE( 6, "( ' radius, ||x||, shift, obj = ', 3ES11.4, ES12.4 )" )    &
           RADIUS( i ), TWO_NORM( X( 1 : n ) ), inform%multiplier, inform%obj
       ELSE
         WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
       END IF
       WRITE( 6, "( ' elapsed cpu time = ', F0.2 )" ) cpu_end - cpu_start
     END DO
     CALL TREK_terminate( data, control, inform )

     WRITE( 6, "( /, ' testing solve ', / )" )
     CALL TREK_initialize( data, control, inform )
!    control%print_level = 1
!    control%trs_control%print_level = 1
     control%sls_control%print_level = 2
!    control%sls_control%print_level_solver = 2
     control%sls_control%ordering = 7
     control%solver = 'ma57 '
!    control%solver = 'pbtr '
!    control%maxit = 10
     control%maxit = 100
     control%exact_shift = .TRUE.
!    control%reorthogonalize = .FALSE.
     CALL CPU_time( cpu_start )
     CALL TREK_solve( n, H, C, RADIUS( 2 ), X, data, control, inform )
     CALL CPU_time( cpu_end )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' radius, ||x||, shift, obj = ', 3ES11.4, ES12.4 )" )      &
         RADIUS( 2 ), TWO_NORM( X( 1 : n ) ), inform%multiplier, inform%obj
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' elapsed cpu time = ', F0.2 )" ) cpu_end - cpu_start

     WRITE( 6, "( /, ' testing new_values ', / )" )
     H%val = H%val + 0.1_wp
     CALL CPU_time( cpu_start )
     CALL TREK_solve( n, H, C, RADIUS( 3 ), X, data, control, inform,          &
                      new_values = .TRUE. )
     CALL CPU_time( cpu_end )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( 1X, I0, ' vectors required, error = ', ES11.4 )" )         &
         inform%n_vec, inform%error
       WRITE( 6, "( ' radius, ||x||, shift, obj = ', 3ES11.4, ES12.4 )" )      &
         RADIUS( 2 ), TWO_NORM( X( 1 : n ) ), inform%multiplier, inform%obj
     ELSE
       WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' elapsed cpu time = ', F0.2 )" ) cpu_end - cpu_start
     CALL TREK_terminate( data, control, inform )
   END DO

   DEALLOCATE( H%type, H%row, H%col, H%val )
   WRITE( 6, "( /, ' tests completed' )" )
   END PROGRAM GALAHAD_TREK_TEST
