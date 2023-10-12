! THIS VERSION: GALAHAD 4.1 - 2023-10-04 AT 09:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_CLLS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_CLLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( CLLS_control_type ) :: control
   TYPE ( CLLS_inform_type ) :: inform
   TYPE ( CLLS_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, o, m, Ao_ne, A_ne, Ao_dense_ne, A_dense_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: regularization_weight
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B, R, W
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_col, Ao_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   CHARACTER ( len = 2 ) :: st
   CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = REPEAT( ' ', 30 )
!  symmetric_linear_solver = 'ssids'
!  symmetric_linear_solver = 'ma97 '
   symmetric_linear_solver = 'sytr '

! set up problem data

   n = 3 ; o = 4 ; m = 2 ; ao_ne = 7 ; a_ne = 4
   ao_dense_ne = o * n ; a_dense_ne = m * n
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), X_stat( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), C_u( m ), C_stat( m ) )
   ALLOCATE( B( o ), R( o ), W( o ) )
   B = (/ 2.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)  ! observations
   C_l = (/ 1.0_rp_, 2.0_rp_ /)                  ! constraint lower bound
   C_u = (/ 2.0_rp_, 2.0_rp_ /)                  ! constraint upper bound
   X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)        ! variable upper bound
   X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_       ! start from zero
   W = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_ /)  ! weights of one and two
   regularization_weight = 0.0_rp_               ! no regularization

! vector problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of least-squares storage formats', / )" )

   DO data_storage_type = 1, 5
!  DO data_storage_type = 1, 1
     CALL CLLS_initialize( data, control, inform )
!    control%print_level = 101 ; control%out = 6
     control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%use_sls = .TRUE.
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'CO'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_col( ao_ne ),            &
                 Ao_ptr( 0 ) )
       ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ),                  &
                 A_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_row = (/ 1, 1, 2, 2, 3, 3, 4 /)
       Ao_col = (/ 1, 2, 2, 3, 1, 3, 2 /)
       A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! A
       A_row = (/ 1, 1, 2, 2 /)
       A_col = (/ 1, 2, 2, 3 /)
       CALL CLLS_import( control, data, status, n, o, m,                       &
                         'coordinate', Ao_ne, Ao_row, Ao_col, Ao_ptr,          &
                         'coordinate', A_ne, A_row, A_col, A_ptr )
     CASE ( 2 ) ! sparse by rows
       st = 'SR'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( 0 ), Ao_col( ao_ne ),                &
                 Ao_ptr( o + 1 ) )
       ALLOCATE( A_val( a_ne ), A_row( 0 ), A_col( a_ne ),                     &
                 A_ptr( m + 1 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_col = (/ 1, 2, 2, 3, 1, 3, 2 /)
       Ao_ptr = (/ 1, 3, 5, 7, 8 /)
       A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! A
       A_col = (/ 1, 2, 2, 3 /)
       A_ptr = (/ 1, 3, 5 /)
       CALL CLLS_import( control, data, status, n, o, m,                       &
                         'sparse_by_rows', Ao_ne, Ao_row, Ao_col, Ao_ptr,      &
                         'sparse_by_rows', A_ne, A_row, A_col, A_ptr )
     CASE ( 3 ) ! sparse by columns
       st = 'SC'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_col( 0 ),                &
                 Ao_ptr( n + 1 ) )
       ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( 0 ),                     &
                 A_ptr( n + 1 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_row = (/ 1, 3, 1, 2, 4, 2, 3 /)
       Ao_ptr = (/ 1, 3, 6, 8 /)
       A_val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! A
       A_row = (/ 1, 1, 2, 2 /)
       A_ptr = (/ 1, 2, 4, 5 /)
       CALL CLLS_import( control, data, status, n, o, m,                       &
                         'sparse_by_columns', Ao_ne, Ao_row, Ao_col, Ao_ptr,   &
                         'sparse_by_columns', A_ne, A_row, A_col, A_ptr )
     CASE ( 4 ) ! dense by rows
       st = 'DR'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       ALLOCATE( A_val( a_dense_ne ), A_row( 0 ), A_col( 0 ), A_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,       &
                   1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_ /) ! Ao
       A_val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /) ! A
       CALL CLLS_import( control, data, status, n, o, m,                       &
                         'dense', Ao_dense_ne, Ao_row, Ao_col, Ao_ptr,         &
                         'dense', A_dense_ne, A_row, A_col, A_ptr )
     CASE ( 5 ) ! dense by columns
       st = 'DC'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       ALLOCATE( A_val( a_dense_ne ), A_row( 0 ), A_col( 0 ), A_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,       &
                   0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_ /) ! Ao
       A_val = (/ 2.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_ /) ! A
       CALL CLLS_import( control, data, status, n, o, m, 'dense_by_columns',   &
                         Ao_dense_ne, Ao_row, Ao_col, Ao_ptr,                  &
                         'dense_by_columns', A_dense_ne, A_row, A_col, A_ptr )
     END SELECT
     IF ( status == 0 ) THEN
       CALL CLLS_solve_clls( data, status, Ao_val, B, A_val, C_l, C_u,         &
                             X_l, X_u, X, R, C, Y, Z, X_stat, C_stat,          &
                             regularization_weight = regularization_weight,    &
                             W = W )
     END IF
     DEALLOCATE( Ao_val, Ao_row, Ao_col, Ao_ptr )
     DEALLOCATE( A_val, A_row, A_col, A_ptr )
     CALL CLLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': CLLS_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL CLLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, C, B, R, Y, Z, W, X_l, X_u, C_l, C_u, X_stat, C_stat )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_CLLS_interface_test
