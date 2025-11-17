! THIS VERSION: GALAHAD 5.4 - 2025-10-19 AT 14:00 GMT.
   PROGRAM GALAHAD_TREK_2ND_EXAMPLE
! double precision version
   USE GALAHAD_TREK_double, ONLY: SMT_type, SMT_put, TREK_control_type,        &
          TREK_inform_type, TREK_data_type, TREK_initialize, TREK_solve,       &
          TREK_terminate
   USE GALAHAD_NORMS_double, ONLY: TWO_NORM
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 1000000
!  INTEGER, PARAMETER :: n = 10
   INTEGER, PARAMETER :: h_ne = n
   TYPE ( SMT_type ) :: H, S
   REAL ( KIND = wp ), DIMENSION( n ) :: C
   REAL ( KIND = wp ) :: radius
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   TYPE ( TREK_data_type ) :: data
   INTEGER :: stat, i
   WRITE( 6, "( ' problem dimension ', I0 )" ) n
   H%n = n ; H%ne = h_ne
   CALL SMT_put( H%type, 'COORDINATE', stat ) ! storage for H
   ALLOCATE( H%row( h_ne ), H%col( h_ne ), H%val( h_ne ) )
   DO i = 1, n
    H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - REAL( n + 1 - i, wp )
!   H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = REAL( i, wp )
   END DO
!H%val = H%val + 5.0_wp
   S%n = n ; S%ne = n
   CALL SMT_put( S%type, 'COORDINATE', stat ) ! storage for S
!  CALL SMT_put( S%type, 'DIAGONAL', stat ) ! storage for S
   ALLOCATE( S%row( n ), S%col( n ), S%val( n ) )
   DO i = 1, n
    S%row( i ) = i ; S%col( i ) = i ; S%val( i ) = 1.0_wp
   END DO
   C = 1.0_wp
!  C = 0.0_wp
!  radius = 10.0_wp
   radius = 1.0_wp
!  radius = 0.01_wp
   CALL TREK_initialize( data, control, inform )
   control%print_level = 1
!  control%trs_control%print_level = 1
!  control%sls_control%print_level = 2
!  control%sls_control%print_level_solver = 2
!  control%linear_solver_for_H = 'ma57 '
!  control%linear_solver = 'ma97 '
!  control%linear_solver_for_S = 'ma97 '
   control%linear_solver = 'pbtr '
   control%linear_solver_for_S = 'pbtr '
!  control% s_version_52 = .FALSE.
!  control%it_max = 2
   control%stop_check_all_orders = .TRUE.
   CALL TREK_solve( n, H, C, radius, X, data, control, inform, S = S )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( 1X, I0, ' vectors required, error =', ES11.4 )" )            &
       inform%n_vec, inform%error
     WRITE( 6, "( ' radius, ||x||, f, multiplier =',  2ES11.4, 2ES12.4 )" )    &
       radius, TWO_NORM( X ), inform%obj, inform%multiplier
   ELSE
     WRITE( 6, "( ' error exit, status = ', I0 )" ) inform%status
   END IF
   WRITE( 6, "( ' total time TREK = ', F0.2 )" ) inform%time%clock_total
   CALL TREK_terminate( data, control, inform )
   DEALLOCATE( H%type, H%row, H%col, H%val, S%type, S%row, S%col, S%val )
   END PROGRAM GALAHAD_TREK_2ND_EXAMPLE
