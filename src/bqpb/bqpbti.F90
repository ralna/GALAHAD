! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 12:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BQPB_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BQPB_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( BQPB_control_type ) :: control
   TYPE ( BQPB_inform_type ) :: inform
   TYPE ( BQPB_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, H_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G, W, X_0
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_zero
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  H_ne = 3
   f = 1.0_rp_
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   G = (/ 2.0_rp_, 0.0_rp_, 0.0_rp_ /)         ! objective gradient
   X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)     ! variable upper bound
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   H_row = (/ 1, 2, 3 /)
   H_col = (/ 1, 2, 3 /)
   H_ptr = (/ 1, 2, 3, 4 /)
   ALLOCATE( H_dense( n * ( n + 1 ) / 2 ), H_diag( n ), H_zero( 0 ) )
   H_dense = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
   H_diag = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 7
     CALL BQPB_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
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
       WRITE( 6, "( A2, ': BQPB_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL BQPB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

!  shifted least-distance example

   ALLOCATE( W( n ), X_0( n ) )
   W = 1.0_rp_    ! weights
   X_0 = 0.0_rp_  ! shifts

   DO data_storage_type = 1, 1
     CALL BQPB_initialize( data, control, inform )
     CALL WHICH_sls( control )
!    control%print_level = 1
     X = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
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
       WRITE( 6, "( A2, ': BQPB_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL BQPB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, G, Z, W, X_0, X_l, X_u, X_stat )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( BQPB_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_BQPB_interface_test
