! THIS VERSION: GALAHAD 4.0 - 2022-01-10 AT 11:15 GMT.
   PROGRAM GALAHAD_EQP_interface_test
   USE GALAHAD_EQP_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( EQP_control_type ) :: control
   TYPE ( EQP_inform_type ) :: inform
   TYPE ( EQP_full_data_type ) :: data
   INTEGER :: n, m, A_ne, H_ne
   INTEGER :: data_storage_type, status
   REAL ( KIND = wp ) :: f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, G, W, X_0
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4 ; H_ne = 3
   f = 1.0_wp
   ALLOCATE( X( n ), G( n ), C( m ), Y( m ) )
   G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
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
     CALL EQP_initialize( data, control, inform )
     X = 0.0_wp ; Y = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL EQP_import( control, data, status, n, m,                           &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_qp( data, status, H_val, G, f, A_val, C, X, Y )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL EQP_import( control, data, status, n, m,                           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_qp( data, status, H_val, G, f, A_val, C, X, Y )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL EQP_import( control, data, status, n, m,                           &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_qp( data, status, H_dense, G, f, A_dense, C, X, Y )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL EQP_import( control, data, status, n, m,                           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL EQP_import( control, data, status, n, m,                           &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr,          &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_qp( data, status, H_diag, G, f, A_val, C, X, Y )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL EQP_import( control, data, status, n, m,                           &
                        'identity', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_qp( data, status, H_zero, G, f, A_val, C, X, Y )
     END SELECT
     CALL EQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
        WRITE( 6, "( A2, ':', I6, ' cg iterations. Optimal objective',         &
      &    ' value = ', F5.2, ' status = ', I0 )" ) st, inform%cg_iter,        &
                                                    inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': EQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL EQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

!  shifted least-distance example

   ALLOCATE( W( n ), X_0( n ) )
   W = 1.0_wp    ! weights
   X_0 = 0.0_wp  ! shifts

   DO data_storage_type = 1, 1
     CALL EQP_initialize( data, control, inform )
!    control%print_level = 1
     X = 0.0_wp ; Y = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' W'
       CALL EQP_import( control, data, status, n, m,                           &
                        'shifted_least_distance', H_ne, H_row, H_col, H_ptr,   &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL EQP_solve_sldqp( data, status, W, X_0, G, f, A_val, C, X, Y )
     END SELECT
     CALL EQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
      WRITE( 6, "( A2, ':', I6, ' cg iterations. Optimal objective value = ', &
     &    F5.2, ' status = ', I0 )" ) st, inform%cg_iter, inform%obj,          &
                                      inform%status
     ELSE
       WRITE( 6, "( A2, ': EQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL EQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, W, X_0 )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )

   END PROGRAM GALAHAD_EQP_interface_test
