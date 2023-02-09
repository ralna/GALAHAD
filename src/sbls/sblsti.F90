! THIS VERSION: GALAHAD 3.3 - 25/11/2021 AT 10:30 GMT.
   PROGRAM GALAHAD_SBLS_interface_test
   USE GALAHAD_SBLS_double                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SBLS_full_data_type ) :: data
   TYPE ( SBLS_control_type ) :: control
   TYPE ( SBLS_inform_type ) :: inform
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_row, C_col, C_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_val
   INTEGER :: n, m, h_ne, a_ne, c_ne
   INTEGER :: data_storage_type, status
   CHARACTER ( LEN = 15 ), DIMENSION( 7 ) :: scheme =                          &
     (/ 'coordinate     ', 'sparse-by-rows ', 'dense          ',               &
        'diagonal       ', 'scaled-identity', 'identity       ',               &
        'zero           ' /)

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats', //,                      &
  &                ' scheme          precon factor status residual' )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 3 ; c_ne = 3
   ALLOCATE( H_ptr( n + 1 ), A_ptr( m + 1 ), C_ptr( m + 1 ), SOL( n + m ) )
   DO data_storage_type = 1, 7
     CALL SBLS_initialize( data, control, inform )
!    control%print_level = 1
     control%preconditioner = 2 ; control%factorization = 2
     control%get_norm_residual = .TRUE.

!  set up data for the appropriate storage type 

     IF ( data_storage_type == 1 ) THEN   ! sparse co-ordinate storage
       ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_col( h_ne ) )
       H_val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 1.0_wp /)
       H_row = (/ 1, 2, 3, 3 /) ; H_col = (/ 1, 2, 3, 1 /)
       ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ) )
       A_val = (/ 2.0_wp, 1.0_wp, 1.0_wp /)
       A_row = (/ 1, 1, 2 /) ; A_col = (/ 1, 2, 3 /)
       ALLOCATE( C_val( c_ne ), C_row( c_ne ), C_col( c_ne ) )
       C_val = (/ 4.0_wp, 1.0_wp, 2.0_wp /)
       C_row = (/ 1, 2, 2 /) ; C_col = (/ 1, 1, 2 /)
       CALL SBLS_import( control, data, status, n, m,                          &
                         'coordinate', h_ne, H_row, H_col, H_ptr,              &
                         'coordinate', a_ne, A_row, A_col, A_ptr,              &
                         'coordinate', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 2 ) THEN ! sparse row-wise storage
       ALLOCATE( H_val( h_ne ), H_row( 0 ), H_col( h_ne ) )
       H_val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 1.0_wp /)
       H_col = (/ 1, 2, 3, 1 /) ; H_ptr = (/ 1, 2, 3, 5 /)
       ALLOCATE( A_val( a_ne ), A_row( 0 ), A_col( a_ne ) )
       A_val = (/ 2.0_wp, 1.0_wp, 1.0_wp /)
       A_col = (/ 1, 2, 3 /) ; A_ptr = (/ 1, 3, 4 /)
       ALLOCATE( C_val( c_ne ), C_row( 0 ), C_col( c_ne ) )
       C_val = (/ 4.0_wp, 1.0_wp, 2.0_wp /)
       C_col = (/ 1, 1, 2 /)  ; C_ptr = (/ 1, 2, 4 /)
       CALL SBLS_import( control, data, status, n, m,                          &
                         'sparse_by_rows', h_ne, H_row, H_col, H_ptr,          &
                         'sparse_by_rows', a_ne, A_row, A_col, A_ptr,          &
                         'sparse_by_rows', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 3 ) THEN ! dense storage
       ALLOCATE( H_val( n * ( n + 1 ) / 2 ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 1.0_wp, 0.0_wp, 3.0_wp /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       ALLOCATE( C_val( m * ( m + 1 ) / 2 ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 4.0_wp, 1.0_wp, 2.0_wp /)
       CALL SBLS_import( control, data, status, n, m,                          &
                         'dense', h_ne, H_row, H_col, H_ptr,                   &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'dense', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 4 ) THEN ! diagonal storage
       ALLOCATE( H_val( n ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 1.0_wp, 1.0_wp, 2.0_wp /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       ALLOCATE( C_val( m ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 4.0_wp, 2.0_wp /)
       CALL SBLS_import( control, data, status, n, m,                          &
                         'diagonal', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'diagonal', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 5 ) THEN ! scaled identity storage
       ALLOCATE( H_val( 1 ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 2.0_wp /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       ALLOCATE( C_val( 1 ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 2.0_wp /)
       CALL SBLS_import( control, data, status, n, m,                          &
                         'scaled_identity', h_ne, H_row, H_col, H_ptr,         &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'scaled_identity', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 6 ) THEN ! identity storage
       ALLOCATE( H_val( 0 ), H_row( 0 ), H_col( 0 ) )
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       ALLOCATE( C_val( 0 ), C_row( 0 ), C_col( 0 ) )
       CALL SBLS_import( control, data, status, n, m,                          &
                         'identity', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'identity', c_ne, C_row, C_col, C_ptr )

     ELSE IF ( data_storage_type == 7 ) THEN ! zero storage
       ALLOCATE( H_val( 0 ), H_row( 0 ), H_col( 0 ) )
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       ALLOCATE( C_val( 0 ), C_row( 0 ), C_col( 0 ) )
       CALL SBLS_import( control, data, status, n, m,                          &
                         'identity', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'zero', c_ne, C_row, C_col, C_ptr )

     END IF
     IF ( status < 0 ) THEN
       CALL SBLS_information( data, inform, status )
       WRITE( 6, "( 1X, A15, 3I7 )" ) scheme( data_storage_type ),             &
         control%preconditioner, control%factorization, inform%status
       CYCLE
     END IF

!  factorize the block matrix

     CALL SBLS_factorize_matrix( data, status, H_val, A_val, C_val )
     IF ( status < 0 ) THEN
       CALL SBLS_information( data, inform, status )
       WRITE( 6, "( 1X, A15, 3I7 )" ) scheme( data_storage_type ),             &
         control%preconditioner, control%factorization, inform%status
       CYCLE
     END IF

!  solve the block linear system

     SOL( : n ) = (/ 3.0_wp, 2.0_wp, 4.0_wp /)
     SOL( n + 1 : ) = (/ 2.0_wp, 0.0_wp /)
     CALL SBLS_solve_system( data, status, SOL )
     CALL SBLS_information( data, inform, status )
     IF ( status == 0 ) THEN
       WRITE( 6, "( 1X, A15, 3I7, ES9.1 )" ) scheme( data_storage_type ),     &
         control%preconditioner, control%factorization,                        &
         inform%status, inform%norm_residual
     ELSE
       WRITE( 6, "( 1X, A15, 3I7 )" ) scheme( data_storage_type ),             &
         control%preconditioner, control%factorization, inform%status
     END IF
     CALL SBLS_terminate( data, control, inform )
     DEALLOCATE( H_val, H_row, H_col, A_val, A_row, A_col, C_val, C_row, C_col )
   END DO
   DEALLOCATE( SOL, H_ptr, A_ptr, C_ptr )
   STOP
   END PROGRAM GALAHAD_SBLS_interface_test


