! THIS VERSION: GALAHAD 3.3 - 26/07/2021 AT 13:50 GMT.
   PROGRAM GALAHAD_CQP_interface_test
   USE GALAHAD_CQP_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( CQP_control_type ) :: control
   TYPE ( CQP_inform_type ) :: inform
   TYPE ( CQP_full_data_type ) :: data
   INTEGER :: n, m, A_ne, H_ne
   INTEGER :: data_storage_type, status
   REAL ( KIND = wp ) :: f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G, W, X_0
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4 ; H_ne = 3
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), W( n ), X_0( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), c_u( m ) )
   ALLOCATE( C_stat( m ), X_stat( n ) )
   G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   W = 1.0_wp                               ! weights
   X_0 = 0.0_wp                             ! shifts
   ALLOCATE( A_val( A_ne ), A_row( A_ne ), A_col( A_ne ), A_ptr( m + 1 ) )
   A_val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   A_row = (/ 1, 1, 2, 2 /)
   A_col = (/ 1, 2, 2, 3 /)
   A_ptr = (/ 1, 3, 5 /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   H_row = (/ 1, 2, 3 /)
   H_col = (/ 1, 2, 3 /)
   H_ptr = (/ 1, 2, 3, 4 /)
   ALLOCATE( A_dense( m * n ), H_dense( n * ( n + 1 ) / 2 ) )
   A_dense = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
   H_dense = (/ 1.0_wp, 0.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
   ALLOCATE( H_diag( n ), H_zero( 0 ) )
   H_diag = (/ 1.0_wp, 1.0_wp, 1.0_wp /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 7
     CALL CQP_initialize( data, control, inform )
     control%print_level = 1
     X = 0.0_wp ; Y = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL CQP_import( control, data, status, n, m,                           &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL CQP_import( control, data, status, n, m,                           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL CQP_import( control, data, status, n, m,                           &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_dense, G, f, A_dense, C_l, C_u,      &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL CQP_import( control, data, status, n, m,                           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL CQP_import( control, data, status, n, m,                           &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr,          &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL CQP_import( control, data, status, n, m,                           &
                        'identity', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_zero, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 7 ) ! zero
       st = ' Z'
       CALL CQP_import( control, data, status, n, m,                           &
                        'zero', H_ne, H_row, H_col, H_ptr,                     &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL CQP_solve_qp( data, status, H_zero, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     END SELECT
     CALL CQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': CQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL CQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, Z, W, X_stat, C_stat )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

   END PROGRAM GALAHAD_CQP_interface_test
