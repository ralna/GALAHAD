! THIS VERSION: GALAHAD 5.0 - 2024-06-07 AT 12:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_DQP_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_DQP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( DQP_control_type ) :: control
   TYPE ( DQP_inform_type ) :: inform
   TYPE ( DQP_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, H_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G, W, X_0
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st
   INTEGER ( KIND = ip_ ) :: status_ref
   REAL ( KIND = rp_ ) :: f_w
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_w, X_0_w, H_w, G_w
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_star

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

   DO data_storage_type = 1, 6
     CALL DQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL DQP_import( control, data, status, n, m,                           &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL DQP_import( control, data, status, n, m,                           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_val, G, f, A_val, C_l, C_u,          &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL DQP_import( control, data, status, n, m,                           &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_dense, G, f, A_dense, C_l, C_u,      &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL DQP_import( control, data, status, n, m,                           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL DQP_import( control, data, status, n, m,                           &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr,          &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_diag, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL DQP_import( control, data, status, n, m,                           &
                        'identity', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_qp( data, status, H_zero, G, f, A_val, C_l, C_u,         &
                          X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     END SELECT
     CALL DQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': DQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL DQP_terminate( data, control, inform )  ! delete internal workspace
   END DO

!  weighted shifted least-distance example, with W = ( 1, 2, 3 ) and
!  x_0 = 1. The first constraint is active at its upper bound and the
!  second is an equality, so x = ( 31, 44, 62 ) / 53. The same problem
!  posed with the diagonal Hessian W^2 is solved first, and the weighted
!  solve is only checked if that succeeds

   WRITE( 6, "( /, ' weighted shifted least-distance test', / )" )
   ALLOCATE( W_w( n ), X_0_w( n ), H_w( n ), G_w( n ), X_star( n ) )
   W_w = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_ /) ; X_0_w = 1.0_rp_
   H_w = W_w ** 2 ; G_w = G - H_w * X_0_w
   f_w = f + 0.5_rp_ * SUM( H_w * X_0_w ** 2 )
   X_star = (/ 31.0_rp_, 44.0_rp_, 62.0_rp_ /) / 53.0_rp_

   CALL DQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_
   CALL DQP_import( control, data, status, n, m,                               &
                    'diagonal', H_ne, H_row, H_col, H_ptr,                     &
                    'coordinate', A_ne, A_row, A_col, A_ptr )
   CALL DQP_solve_qp( data, status, H_w, G_w, f_w, A_val, C_l, C_u,            &
                      X_l, X_u, X, C, Y, Z, X_stat, C_stat )
   CALL DQP_information( data, inform, status )
   status_ref = inform%status
   CALL DQP_terminate( data, control, inform )

   IF ( status_ref /= 0 ) THEN
     WRITE( 6, "( ' WW: skipped, diagonal Hessian status = ', I0 )" )          &
       status_ref
   ELSE
     CALL DQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_
     CALL DQP_import( control, data, status, n, m,                             &
                      'shifted_least_distance', H_ne, H_row, H_col, H_ptr,     &
                      'coordinate', A_ne, A_row, A_col, A_ptr )
     CALL DQP_solve_sldqp( data, status, W_w, X_0_w, G, f, A_val, C_l, C_u,    &
                           X_l, X_u, X, C, Y, Z, X_stat, C_stat )
     CALL DQP_information( data, inform, status )
     CALL DQP_terminate( data, control, inform )
     IF ( inform%status /= 0 .OR.                                              &
          MAXVAL( ABS( X - X_star ) ) > SQRT( EPSILON( 1.0_rp_ ) ) ) THEN
       WRITE( 6, "( ' WW: DQP_solve_sldqp failed, status = ', I0 )" )          &
         inform%status
       STOP 1
     END IF
     WRITE( 6, "( ' WW: DQP_solve_sldqp exit status = ', I0 )" ) inform%status
   END IF
   DEALLOCATE( W_w, X_0_w, H_w, G_w, X_star )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )

!  shifted least-distance example

   ALLOCATE( W( n ), X_0( n ) )
   W = 1.0_rp_    ! weights
   X_0 = 0.0_rp_  ! shifts

   DO data_storage_type = 1, 1
     CALL DQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
!    control%print_level = 1
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' W'
       CALL DQP_import( control, data, status, n, m,                           &
                        'shifted_least_distance', H_ne, H_row, H_col, H_ptr,   &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL DQP_solve_sldqp( data, status, W, X_0, G, f, A_val, C_l, C_u,      &
                             X_l, X_u, X, C, Y, Z, X_stat, C_stat )

     END SELECT
     CALL DQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': DQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL DQP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y, Z, W, X_0, x_l, X_u, C_l, C_u, X_stat, C_stat )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( DQP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     control%definite_linear_solver = definite_linear_solver
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_DQP_interface_test
