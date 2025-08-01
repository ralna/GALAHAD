! THIS VERSION: GALAHAD 5.1 - 2024-09-10 AT 14:15 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SSLS_interface_test
   USE GALAHAD_KINDS_precision, C_ptr_rename => C_ptr
   USE GALAHAD_SSLS_precision
   IMPLICIT NONE
   TYPE ( SSLS_full_data_type ) :: data
   TYPE ( SSLS_control_type ) :: control
   TYPE ( SSLS_inform_type ) :: inform
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SOL
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_row, C_col, C_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_val
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, c_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   CHARACTER ( LEN = 15 ), DIMENSION( 7 ) :: scheme =                          &
     (/ 'coordinate     ', 'sparse-by-rows ', 'dense          ',               &
        'diagonal       ', 'scaled-identity', 'identity       ',               &
        'zero           ' /)

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats', //,                      &
  &                ' scheme          status  ok' )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 3 ; c_ne = 3
   ALLOCATE( H_ptr( n + 1 ), A_ptr( m + 1 ), C_ptr( m + 1 ), SOL( n + m ) )
   DO data_storage_type = 1, 7
     CALL SSLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
!    control%print_level = 1

!  set up data for the appropriate storage type

     IF ( data_storage_type == 1 ) THEN   ! sparse co-ordinate storage
       ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_col( h_ne ) )
       H_val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
       H_row = (/ 1, 2, 3, 3 /) ; H_col = (/ 1, 2, 3, 1 /)
       ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A_row = (/ 1, 1, 2 /) ; A_col = (/ 1, 2, 3 /)
       ALLOCATE( C_val( c_ne ), C_row( c_ne ), C_col( c_ne ) )
       C_val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
       C_row = (/ 1, 2, 2 /) ; C_col = (/ 1, 1, 2 /)
       CALL SSLS_import( control, data, status, n, m,                          &
                         'coordinate', h_ne, H_row, H_col, H_ptr,              &
                         'coordinate', a_ne, A_row, A_col, A_ptr,              &
                         'coordinate', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 2 ) THEN ! sparse row-wise storage
       ALLOCATE( H_val( h_ne ), H_row( 0 ), H_col( h_ne ) )
       H_val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
       H_col = (/ 1, 2, 3, 1 /) ; H_ptr = (/ 1, 2, 3, 5 /)
       ALLOCATE( A_val( a_ne ), A_row( 0 ), A_col( a_ne ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A_col = (/ 1, 2, 3 /) ; A_ptr = (/ 1, 3, 4 /)
       ALLOCATE( C_val( c_ne ), C_row( 0 ), C_col( c_ne ) )
       C_val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
       C_col = (/ 1, 1, 2 /)  ; C_ptr = (/ 1, 2, 4 /)
       CALL SSLS_import( control, data, status, n, m,                          &
                         'sparse_by_rows', h_ne, H_row, H_col, H_ptr,          &
                         'sparse_by_rows', a_ne, A_row, A_col, A_ptr,          &
                         'sparse_by_rows', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 3 ) THEN ! dense storage
       ALLOCATE( H_val( n * ( n + 1 ) / 2 ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 1.0_rp_, 0.0_rp_, 3.0_rp_ /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       ALLOCATE( C_val( m * ( m + 1 ) / 2 ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
       CALL SSLS_import( control, data, status, n, m,                          &
                         'dense', h_ne, H_row, H_col, H_ptr,                   &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'dense', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 4 ) THEN ! diagonal storage
       ALLOCATE( H_val( n ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_ /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       ALLOCATE( C_val( m ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 4.0_rp_, 2.0_rp_ /)
       CALL SSLS_import( control, data, status, n, m,                          &
                         'diagonal', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'diagonal', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 5 ) THEN ! scaled identity storage
       ALLOCATE( H_val( 1 ), H_row( 0 ), H_col( 0 ) )
       H_val = (/ 2.0_rp_ /)
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       ALLOCATE( C_val( 1 ), C_row( 0 ), C_col( 0 ) )
       C_val = (/ 2.0_rp_ /)
       CALL SSLS_import( control, data, status, n, m,                          &
                         'scaled_identity', h_ne, H_row, H_col, H_ptr,         &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'scaled_identity', c_ne, C_row, C_col, C_ptr )
     ELSE IF ( data_storage_type == 6 ) THEN ! identity storage
       ALLOCATE( H_val( 0 ), H_row( 0 ), H_col( 0 ) )
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       ALLOCATE( C_val( 0 ), C_row( 0 ), C_col( 0 ) )
       CALL SSLS_import( control, data, status, n, m,                          &
                         'identity', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'identity', c_ne, C_row, C_col, C_ptr )

     ELSE IF ( data_storage_type == 7 ) THEN ! zero storage
       ALLOCATE( H_val( 0 ), H_row( 0 ), H_col( 0 ) )
       ALLOCATE( A_val( n * m ), A_row( 0 ), A_col( 0 ) )
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       ALLOCATE( C_val( 0 ), C_row( 0 ), C_col( 0 ) )
       CALL SSLS_import( control, data, status, n, m,                          &
                         'identity', h_ne, H_row, H_col, H_ptr,                &
                         'dense', a_ne, A_row, A_col, A_ptr,                   &
                         'zero', c_ne, C_row, C_col, C_ptr )

     END IF
     IF ( status < 0 ) THEN
       CALL SSLS_information( data, inform, status )
       WRITE( 6, "( 1X, A15, I7 )" ) scheme( data_storage_type ), inform%status
       GO TO 90
     END IF

!  factorize the block matrix

     CALL SSLS_factorize_matrix( data, status, H_val, A_val, C_val )
     IF ( status < 0 ) THEN
       CALL SSLS_information( data, inform, status )
       WRITE( 6, "( 1X, A15, I7 )" ) scheme( data_storage_type ), inform%status
       GO TO 90
     END IF

!  solve the block linear system; pick the rhs so that the solution is vec(1.0)

     IF ( data_storage_type == 4 ) THEN ! diagonal storage
       SOL( : n ) = (/ 3.0_rp_, 2.0_rp_, 3.0_rp_ /)
       SOL( n + 1 : ) = (/ - 1.0_rp_, - 1.0_rp_ /)
     ELSE IF ( data_storage_type == 5 ) THEN ! scaled identity storage
       SOL( : n ) = (/ 4.0_rp_, 3.0_rp_, 3.0_rp_ /)
       SOL( n + 1 : ) = (/ 1.0_rp_, - 1.0_rp_ /)
     ELSE IF ( data_storage_type == 6 ) THEN ! identity storage
       SOL( : n ) = (/ 3.0_rp_, 2.0_rp_, 2.0_rp_ /)
       SOL( n + 1 : ) = (/ 2.0_rp_, 0.0_rp_ /)
     ELSE IF ( data_storage_type == 7 ) THEN ! zero storage
       SOL( : n ) = (/ 3.0_rp_, 2.0_rp_, 2.0_rp_ /)
       SOL( n + 1 : ) = (/ 3.0_rp_, 1.0_rp_ /)
     ELSE
       SOL( : n ) = (/ 4.0_rp_, 3.0_rp_, 5.0_rp_ /)
       SOL( n + 1 : ) = (/ - 2.0_rp_, - 2.0_rp_ /)
     END IF
     CALL SSLS_solve_system( data, status, SOL )
     CALL SSLS_information( data, inform, status )
     SOL = SOL - 1.0_rp_
     IF ( MAXVAL( ABS( SOL ) ) <= 10.0_rp_ * EPSILON( 1.0_rp_ ) ) THEN
       WRITE( 6, "( 1X, A15, I7, A4 )" ) scheme( data_storage_type ),          &
         inform%status, ' yes'
     ELSE
       WRITE( 6, "( 1X, A15, I7, A4 )" ) scheme( data_storage_type ),          &
         inform%status, '  no'
     END IF
 90  CONTINUE
     CALL SSLS_terminate( data, control, inform )
     DEALLOCATE( H_val, H_row, H_col, A_val, A_row, A_col, C_val, C_row, C_col )
   END DO
   DEALLOCATE( SOL, H_ptr, A_ptr, C_ptr )
   STOP

  CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( SSLS_control_type ) :: control
#include "galahad_sls_defaults_sls.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_SSLS_interface_test


