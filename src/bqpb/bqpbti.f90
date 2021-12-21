! THIS VERSION: GALAHAD 3.3 - 21/12/2021 AT 13:30 GMT.
   PROGRAM GALAHAD_BQPB_interface_test
   USE GALAHAD_BQPB_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( BQPB_control_type ) :: control
   TYPE ( BQPB_inform_type ) :: inform
   TYPE ( BQPB_full_data_type ) :: data
   INTEGER :: n, m, A_ne, H_ne
   INTEGER :: data_storage_type, status
   REAL ( KIND = wp ) :: f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G, W, X_0
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_zero
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  H_ne = 3
   f = 1.0_wp
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   G = (/ 2.0_wp, 0.0_wp, 0.0_wp /)         ! objective gradient
   X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   H_row = (/ 1, 2, 3 /)
   H_col = (/ 1, 2, 3 /)
   H_ptr = (/ 1, 2, 3, 4 /)
   ALLOCATE( H_dense( n * ( n + 1 ) / 2 ), H_diag( n ), H_zero( 0 ) )
   H_dense = (/ 1.0_wp, 0.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
   H_diag = (/ 1.0_wp, 1.0_wp, 1.0_wp /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 7
     CALL BQPB_initialize( data, control, inform )
     X = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL BQPB_import( control, data, status, n,                             &
                         'coordinate', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_val, G, f,                          &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL BQPB_import( control, data, status, n,                             &
                         'sparse_by_rows', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_val, G, f,                          &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL BQPB_import( control, data, status, n,                             &
                         'dense', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_dense, G, f,                        &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL BQPB_import( control, data, status, n,                             &
                         'diagonal', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_diag, G, f,                         &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL BQPB_import( control, data, status, n,                             &
                         'scaled_identity', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_diag, G, f,                         &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL BQPB_import( control, data, status, n,                             &
                         'identity', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_zero, G, f,                         &
                          X_l, X_u, X, Z, X_stat )
     CASE ( 7 ) ! zero
       st = ' Z'
       CALL BQPB_import( control, data, status, n,                             &
                         'zero', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_qp( data, status, H_zero, G, f,                         &
                           X_l, X_u, X, Z, X_stat )
     END SELECT
     CALL BQPB_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': BQPB_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL BQPB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

!  shifted least-distance example

   ALLOCATE( W( n ), X_0( n ) )
   W = 1.0_wp    ! weights
   X_0 = 0.0_wp  ! shifts

   DO data_storage_type = 1, 1
     CALL BQPB_initialize( data, control, inform )
!    control%print_level = 1
     X = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' W'
       CALL BQPB_import( control, data, status, n,                             &
                         'shifted_least_distance', H_ne, H_row, H_col, H_ptr )
       CALL BQPB_solve_sldqp( data, status, W, X_0, G, f,                      &
                              X_l, X_u, X, Z, X_stat )

     END SELECT
     CALL BQPB_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': BQPB_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL BQPB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, G, Z, W, X_0, x_l, X_u, X_stat )

   END PROGRAM GALAHAD_BQPB_interface_test
