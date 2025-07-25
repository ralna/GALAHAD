! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 15:45 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QPA_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPA_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPA_control_type ) :: control
   TYPE ( QPA_inform_type ) :: inform
   TYPE ( QPA_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, H_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f, rho_g, rho_b
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4 ; H_ne = 3
   f = 1.0_rp_
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), C_u( m ), C_stat( m ) )
   G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)         ! objective gradient
   C_l = (/ 1.0_rp_, 2.0_rp_ /)               ! constraint lower bound
   C_u = (/ 2.0_rp_, 2.0_rp_ /)               ! constraint upper bound
   X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)     ! variable upper bound
   ALLOCATE( A_val( A_ne ), A_row( A_ne ), A_col( A_ne ), A_ptr( m + 1 ) )
   A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A_row = (/ 1, 1, 2, 2 /)
   A_col = (/ 1, 2, 2, 3 /)
   A_ptr = (/ 1, 3, 5 /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   H_row = (/ 1, 2, 3 /)
   H_col = (/ 1, 2, 3 /)
   H_ptr = (/ 1, 2, 3, 4 /)
   ALLOCATE( A_dense( m * n ), H_dense( n * ( n + 1 ) / 2 ) )
   A_dense = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
   H_dense = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
   ALLOCATE( H_diag( n ), H_zero( 0 ) )
   H_diag = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 7
     CALL QPA_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL QPA_import( control, data, status, n, m,                           &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL QPA_import( control, data, status, n, m,                           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL QPA_import( control, data, status, n, m,                           &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_dense, G, f, A_dense, C_l, C_u,      &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL QPA_import( control, data, status, n, m,                           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL QPA_import( control, data, status, n, m,                           &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr,          &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL QPA_import( control, data, status, n, m,                           &
                        'identity', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_zero, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 7 ) ! zero
       st = ' Z'
       CALL QPA_import( control, data, status, n, m,                           &
                        'zero', H_ne, H_row, H_col, H_ptr,                     &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL QPA_solve_qp( data, status, H_zero, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     END SELECT
     CALL QPA_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': QPA_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL QPA_terminate( data, control, inform )  ! delete internal workspace
   END DO

!  ==========================
!  basic test of l_1 variants
!  ==========================

   WRITE( 6, "( /, ' basic tests of l_1 qp storage formats', / )" )

   CALL QPA_initialize( data, control, inform )
   CALL WHICH_sls( control )
   X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
   rho_g = 0.1_rp_ ; rho_b = 0.1_rp_ ! penalty parameters
   CALL QPA_import( control, data, status, n, m,                               &
                    'coordinate', H_ne, H_row, H_col, H_ptr,                   &
                    'coordinate', A_ne, A_row, A_col, A_ptr )
   CALL QPA_solve_l1qp( data, status, H_val, G, f, rho_g, rho_b, A_val,        &
                        C_l, C_u, X_l, X_u, X, C, Y, Z, X_stat, C_stat )
   CALL QPA_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A6, I6, ' iterations. Optimal objective value = ', F5.2,     &
   &    ' status = ', I0 )" ) 'l1qp', inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A2, ': QPA_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
   rho_g = 0.01_rp_ ! penalty parameters
   CALL QPA_solve_bcl1qp( data, status, H_val, G, f, rho_g, A_val, C_l, C_u,   &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
   CALL QPA_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A6, I6, ' iterations. Optimal objective value = ', F5.2,     &
   &    ' status = ', I0 )" ) 'bcl1qp', inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A2, ': QPA_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   CALL QPA_terminate( data, control, inform )  ! delete internal workspace

   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )
   DEALLOCATE( X, C, G, Y, Z, x_l, X_u, C_l, C_u, X_stat, C_stat )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( QPA_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%symmetric_linear_solver = symmetric_linear_solver
!control%symmetric_linear_solver = 'ma97 '
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_QPA_interface_test
