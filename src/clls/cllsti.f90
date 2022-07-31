! THIS VERSION: GALAHAD 4.1 - 2022-07-20 AT 10:25 GMT.
   PROGRAM GALAHAD_CCQP_interface_test
   USE GALAHAD_CCQP_double                      ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( CCQP_control_type ) :: control
   TYPE ( CCQP_inform_type ) :: inform
   TYPE ( CCQP_full_data_type ) :: data
   INTEGER :: n, m, L_ne, H_ne
   INTEGER :: data_storage_type, status
   REAL ( KIND = wp ) :: f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G, W, X_0
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: L_row, L_col, L_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: L_val, L_dense, H_zero
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; L_ne = 4 ; H_ne = 3
   f = 1.0_wp
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), C_u( m ), C_stat( m ) )
   G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   ALLOCATE( L_val( L_ne ), L_row( L_ne ), L_col( L_ne ), L_ptr( m + 1 ) )
   L_val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   L_row = (/ 1, 1, 2, 2 /)
   L_col = (/ 1, 2, 2, 3 /)
   L_ptr = (/ 1, 3, 5 /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   H_row = (/ 1, 2, 3 /)
   H_col = (/ 1, 2, 3 /)
   H_ptr = (/ 1, 2, 3, 4 /)
   ALLOCATE( L_dense( m * n ), H_dense( n * ( n + 1 ) / 2 ) )
   L_dense = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
   H_dense = (/ 1.0_wp, 0.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
   ALLOCATE( H_diag( n ), H_zero( 0 ) )
   H_diag = (/ 1.0_wp, 1.0_wp, 1.0_wp /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 7
     CALL CCQP_initialize( data, control, inform )
     X = 0.0_wp ; Y = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'coordinate', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_val, G, f, L_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_rows', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_val, G, f, L_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_dense, G, f, L_dense, C_l, C_u,     &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_diag, G, f, L_val, C_l, C_u,        &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr,          &
                        'sparse_by_rows', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_diag, G, f, L_val, C_l, C_u,        &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'identity', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_zero, G, f, L_val, C_l, C_u,        &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 7 ) ! zero
       st = ' Z'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'zero', H_ne, H_row, H_col, H_ptr,                     &
                        'sparse_by_rows', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_qp( data, status, H_zero, G, f, L_val, C_l, C_u,        &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     END SELECT
     CALL CCQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': CCQP_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL CCQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

!  shifted least-distance example

   ALLOCATE( W( n ), X_0( n ) )
   W = 1.0_wp    ! weights
   X_0 = 0.0_wp  ! shifts

   DO data_storage_type = 1, 1
     CALL CCQP_initialize( data, control, inform )
!    control%print_level = 1
     X = 0.0_wp ; Y = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' W'
       CALL CCQP_import( control, data, status, n, m,                          &
                        'shifted_least_distance', H_ne, H_row, H_col, H_ptr,   &
                        'coordinate', L_ne, L_row, L_col, L_ptr )
       CALL CCQP_solve_sldqp( data, status, W, X_0, G, f, L_val, C_l, C_u,     &
                             X_l, X_u, X, C, Y, Z, X_stat, C_stat )

     END SELECT
     CALL CCQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': CCQP_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL CCQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, Z, W, X_0, x_l, X_u, C_l, C_u, X_stat, C_stat )
   DEALLOCATE( L_val, L_row, L_col, L_ptr, L_dense )

   END PROGRAM GALAHAD_CCQP_interface_test
