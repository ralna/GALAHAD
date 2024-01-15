! THIS VERSION: GALAHAD 4.1 - 2023-02-11 AT 17:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_EQP_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_EQP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( EQP_control_type ) :: control
   TYPE ( EQP_inform_type ) :: inform
   TYPE ( EQP_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, H_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, G, W, X_0
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4 ; H_ne = 3
   f = 1.0_rp_
   ALLOCATE( X( n ), G( n ), C( m ), Y( m ) )
   G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)         ! objective gradient
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
   C = (/ 3.0_rp_, 0.0_rp_ /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 6
     CALL EQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y = 0.0_rp_ ! start from zero
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
       CALL EQP_solve_qp( data, status, H_diag, G, f, A_val, C, X, Y )
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
   W = 1.0_rp_    ! weights
   X_0 = 0.0_rp_  ! shifts

   DO data_storage_type = 1, 1
     CALL EQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y = 0.0_rp_ ! start from zero
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
      WRITE( 6, "( A2, ':', I6, ' cg iterations. Optimal objective value = ',  &
     &    F5.2, ' status = ', I0 )" ) st, inform%cg_iter, inform%obj,          &
                                      inform%status
     ELSE
       WRITE( 6, "( A2, ': EQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL EQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, W, X_0 )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( EQP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_EQP_interface_test
