! THIS VERSION: GALAHAD 4.1 - 2023-12-23 AT 14:25 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BLLSB_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BLLSB_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( BLLSB_control_type ) :: control
   TYPE ( BLLSB_inform_type ) :: inform
   TYPE ( BLLSB_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, o, Ao_ne, Ao_dense_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: regularization_weight
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B, R, W
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_col, Ao_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat
   CHARACTER ( len = 2 ) :: st
   CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = REPEAT( ' ', 30 )
!  symmetric_linear_solver = 'ssids'
!  symmetric_linear_solver = 'ma97 '
   symmetric_linear_solver = 'sytr '

! set up problem data

   n = 3 ; o = 4 ; ao_ne = 7
   ao_dense_ne = o * n
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), X_stat( n ) )
   ALLOCATE( B( o ), R( o ), W( o ) )
   B = (/ 2.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)  ! observations
   X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)        ! variable upper bound
   X = 0.0_rp_ ; Z = 0.0_rp_                     ! start from zero
   W = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_ /)  ! weights of one and two
   regularization_weight = 0.0_rp_               ! no regularization

! vector problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of least-squares storage formats', / )" )

   DO data_storage_type = 1, 5
!  DO data_storage_type = 1, 1
     CALL BLLSB_initialize( data, control, inform )
!    control%print_level = 101 ; control%out = 6
     control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%use_sls = .TRUE.
     X = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'CO'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_col( ao_ne ),            &
                 Ao_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_row = (/ 1, 1, 2, 2, 3, 3, 4 /)
       Ao_col = (/ 1, 2, 2, 3, 1, 3, 2 /)
       CALL BLLSB_import( control, data, status, n, o,                         &
                         'coordinate', Ao_ne, Ao_row, Ao_col, Ao_ptr )
     CASE ( 2 ) ! sparse by rows
       st = 'SR'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( 0 ), Ao_col( ao_ne ),                &
                 Ao_ptr( o + 1 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_col = (/ 1, 2, 2, 3, 1, 3, 2 /)
       Ao_ptr = (/ 1, 3, 5, 7, 8 /)
       CALL BLLSB_import( control, data, status, n, o,                         &
                         'sparse_by_rows', Ao_ne, Ao_row, Ao_col, Ao_ptr )
     CASE ( 3 ) ! sparse by columns
       st = 'SC'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_col( 0 ),                &
                 Ao_ptr( n + 1 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /) ! Ao
       Ao_row = (/ 1, 3, 1, 2, 4, 2, 3 /)
       Ao_ptr = (/ 1, 3, 6, 8 /)
       CALL BLLSB_import( control, data, status, n, o,                         &
                         'sparse_by_columns', Ao_ne, Ao_row, Ao_col, Ao_ptr )
     CASE ( 4 ) ! dense by rows
       st = 'DR'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,       &
                   1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_ /) ! Ao
       CALL BLLSB_import( control, data, status, n, o,                         &
                         'dense', Ao_dense_ne, Ao_row, Ao_col, Ao_ptr )
     CASE ( 5 ) ! dense by columns
       st = 'DC'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       Ao_val = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,       &
                   0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_ /) ! Ao
       CALL BLLSB_import( control, data, status, n, o, 'dense_by_columns',     &
                         Ao_dense_ne, Ao_row, Ao_col, Ao_ptr )
     END SELECT
     IF ( status == 0 ) THEN
       CALL BLLSB_solve_blls( data, status, Ao_val, B,                         &
                              X_l, X_u, X, R, Z, X_stat,                       &
                              regularization_weight = regularization_weight,   &
                              W = W )
     END IF
     DEALLOCATE( Ao_val, Ao_row, Ao_col, Ao_ptr )
     CALL BLLSB_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': BLLSB_solve exit status = ', I0 )") st, inform%status
     END IF
     CALL BLLSB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, B, R, Z, W, X_l, X_u, X_stat )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_BLLSB_interface_test
