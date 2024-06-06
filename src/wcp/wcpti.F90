! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 13:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_WCP_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_WCP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( WCP_control_type ) :: control
   TYPE ( WCP_inform_type ) :: inform
   TYPE ( WCP_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, A_dense_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u, Z_l, Z_u, G
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, C_l, C_u, Y_l, Y_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 3 ;  m = 2 ; A_ne = 4 ; a_dense_ne = m * n ;
   ALLOCATE( X( n ), Z_l( n ), Z_u( n ), X_l( n ), X_u( n ), G( n ) )
   ALLOCATE( C( m ), Y_l( m ), Y_u( m ), C_l( m ), C_u( m ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
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
   ALLOCATE( A_dense( m * n ) )
   A_dense = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

   DO data_storage_type = 1, 3
     CALL WCP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ; Y_l = 0.0_rp_ ; Y_u = 0.0_rp_
     Z_l = 0.0_rp_ ; Z_u = 0.0_rp_
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL WCP_import( control, data, status, n, m,                           &
                        'coordinate', A_ne, A_row, A_col, A_ptr )
       CALL WCP_find_wcp( data, status, G, A_val, C_l, C_u,                    &
                          X_l, X_u, X, C, Y_l, Y_u, Z_l, Z_u, X_stat, C_stat )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL WCP_import( control, data, status, n, m,                           &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
       CALL WCP_find_wcp( data, status, G, A_val, C_l, C_u,                    &
                          X_l, X_u, X, C, Y_l, Y_u, Z_l, Z_u, X_stat, C_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL WCP_import( control, data, status, n, m,                           &
                        'dense', A_dense_ne, A_row, A_col, A_ptr )
       CALL WCP_find_wcp( data, status, G, A_dense, C_l, C_u,                  &
                          X_l, X_u, X, C, Y_l, Y_u, Z_l, Z_u, X_stat, C_stat )

     END SELECT
     CALL WCP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': WCP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL WCP_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, G, Y_l, Y_u, Z_l, Z_u, X_l, X_u, C_l, C_u, X_stat, C_stat )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( WCP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_WCP_interface_test
