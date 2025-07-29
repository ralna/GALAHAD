! THIS VERSION: GALAHAD 5.3 - 2025-07-29 AT 14:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SSLS_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SSLS_precision
   IMPLICIT NONE
   TYPE ( SMT_type ) :: H, A, C
   TYPE ( SSLS_data_type ) :: data
   TYPE ( SSLS_control_type ) :: control
   TYPE ( SSLS_inform_type ) :: inform
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SOL, R
!  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SOL1
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, c_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, l, tests, status, solvers
   INTEGER ( KIND = ip_ ) :: smt_stat
   INTEGER ( KIND = ip_ ) :: scratch_out = 56
   REAL ( KIND = rp_ ) :: norm_residual
!  LOGICAL :: all_generic_tests = .FALSE.
   LOGICAL :: all_generic_tests = .TRUE.

   IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
   CALL SMT_put( C%type, 'COORDINATE', smt_stat ) ; C%ne = 0
   ALLOCATE( C%val( C%ne ), C%row( C%ne ), C%col( C%ne ) )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( H%ptr( n + 1 ), A%ptr( m + 1 ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ' )" )
   WRITE( 6, "( /, ' test   status' )" )

   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
!  control%print_level = 1
   control%print_level = 0
   control%sls_control%warning = - 1
   control%sls_control%out = - 1

!  tests for status = - 3 and - 15

   DO l = 1, 1
     IF ( l == 1 ) THEN
       status = 3
     ELSE
       status = 15
     END IF
     ALLOCATE( H%val( h_ne ), H%col( h_ne ) )
     ALLOCATE( A%val( a_ne ), A%col( a_ne ) )
     IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
     CALL SMT_put( H%type, 'SPARSE_BY_ROWS', smt_stat )
     H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
     H%col = (/ 1, 2, 3, 1 /)
     H%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
     CALL SMT_put( A%type, 'SPARSE_BY_ROWS', smt_stat )
     A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     A%col = (/ 1, 2, 2, 3 /)
     A%ptr = (/ 1, 3, 5 /)

     IF ( status == 3 ) THEN
       n = 0 ; m = - 1
     ELSE
       n = 3 ; m = 2
       A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A%col = (/ 1, 2, 1, 2 /)
     END IF
     CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
     IF ( inform%status == 0 )                                                 &
       CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
     WRITE( 6, "( I5, I9 )" ) status, inform%status
     DEALLOCATE( H%val, H%col )
     DEALLOCATE( A%val, A%col )
   END DO

   CALL SSLS_terminate( data, control, inform )
   DEALLOCATE( H%ptr, A%ptr )
   DEALLOCATE( C%val, C%row, C%col )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ' )" )
   WRITE( 6, "( /, 8X, 'storage   status residual' )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 3 ; c_ne = 3
   ALLOCATE( H%ptr( n + 1 ), A%ptr( m + 1 ), C%ptr( m + 1 ) )
   ALLOCATE( SOL( n + m ), R( n + m ) )

   DO data_storage_type = - 5, 0
     CALL SSLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
!    control%print_level = 1
     IF ( data_storage_type == 0 ) THEN   ! sparse co-ordinate storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'COORDINATE', smt_stat )  ; H%ne = h_ne
       ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
       H%row = (/ 1, 2, 3, 3 /)
       H%col = (/ 1, 2, 3, 1 /)
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'COORDINATE', smt_stat )  ; A%ne = a_ne
       ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
       A%row = (/ 1, 1, 2 /)
       A%col = (/ 1, 2, 3 /)
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'COORDINATE', smt_stat ) ; C%ne = c_ne
       ALLOCATE( C%val( c_ne ), C%row( c_ne ), C%col( c_ne ) )
       C%row = (/ 1, 2, 2 /)
       C%col = (/ 1, 1, 2 /)
     ELSE IF ( data_storage_type == - 1 ) THEN ! sparse row-wise storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'SPARSE_BY_ROWS', smt_stat )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', smt_stat )
       ALLOCATE( H%val( h_ne ), H%col( h_ne ) )
       ALLOCATE( A%val( a_ne ), A%col( a_ne ) )
       H%col = (/ 1, 2, 3, 1 /)
       H%ptr = (/ 1, 2, 3, 5 /)
       A%col = (/ 1, 2, 3 /)
       A%ptr = (/ 1, 3, 4 /)
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'SPARSE_BY_ROWS', smt_stat )
       ALLOCATE( C%val( c_ne ), C%col( c_ne ) )
       C%col = (/ 1, 1, 2 /)
       C%ptr = (/ 1, 2, 4 /)
     ELSE IF ( data_storage_type == - 2 ) THEN ! dense storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
       ALLOCATE( H%val( n * ( n + 1 ) / 2 ) )
       ALLOCATE( A%val( n * m ) )
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'DENSE', smt_stat )
       ALLOCATE( C%val( m * ( m + 1 ) / 2 ) )
     ELSE IF ( data_storage_type == - 3 ) THEN ! diagonal storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DIAGONAL', smt_stat )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
       ALLOCATE( H%val( n ) )
       ALLOCATE( A%val( n * m ) )
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'DIAGONAL', smt_stat )
       ALLOCATE( C%val( m ) )
     ELSE IF ( data_storage_type == - 4 ) THEN ! scaled identity storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'SCALED_IDENTITY', smt_stat )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
       ALLOCATE( H%val( 1 ) )
       ALLOCATE( A%val( n * m ) )
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'SCALED_IDENTITY', smt_stat )
       ALLOCATE( C%val( 1 ) )
     ELSE IF ( data_storage_type == - 5 ) THEN ! identity storage
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'IDENTITY', smt_stat )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
       ALLOCATE( A%val( n * m ) )
       IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
       CALL SMT_put( C%type, 'IDENTITY', smt_stat )
     END IF
     IF ( data_storage_type == 0 ) THEN     ! sparse co-ordinate storage
       H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
       A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
       C%val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
     ELSE IF ( data_storage_type == - 1 ) THEN  !  sparse row-wise storage
       H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
       A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
       C%val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
     ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
       H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 1.0_rp_, 0.0_rp_, 3.0_rp_ /)
       A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       C%val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
     ELSE IF ( data_storage_type == - 3 ) THEN    !  dense storage
       H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_ /)
       A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       C%val = (/ 4.0_rp_, 2.0_rp_ /)
     ELSE IF ( data_storage_type == - 4 ) THEN    !  scaled identity
       H%val = (/ 2.0_rp_ /)
       A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
       C%val = (/ 2.0_rp_ /)
     ELSE IF ( data_storage_type == - 5 ) THEN    !  identity
       A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
     END IF
     CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
     CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
     IF ( inform%status < 0 ) THEN
       WRITE( 6, "( A15, I9 )" ) SMT_get( H%type ), inform%status
       CYCLE
     END IF
     SOL( : n ) = (/ 3.0_rp_, 2.0_rp_, 4.0_rp_ /)
     SOL( n + 1 : ) = (/ 2.0_rp_, 0.0_rp_ /)
     R = SOL
     CALL SSLS_solve( n, m, SOL, data, control, inform )
     IF ( inform%status == 0 ) THEN
       CALL RESIDUAL( n, m, H, A, C, SOL, R )
       norm_residual = MAXVAL( ABS( R ) )
       WRITE( 6, "( A15, I9, A9 )" ) SMT_get( H%type ),                        &
         inform%status, type_residual( norm_residual )
     ELSE
       WRITE( 6, "( A15, I9 )" ) SMT_get( H%type ), inform%status
     END IF
     CALL SSLS_terminate( data, control, inform )
     IF ( data_storage_type == 0 ) THEN
       DEALLOCATE( H%val, H%row, H%col )
       DEALLOCATE( A%val, A%row, A%col )
     ELSE IF ( data_storage_type == - 1 ) THEN
       DEALLOCATE( H%val, H%col )
       DEALLOCATE( A%val, A%col )
     ELSE IF ( data_storage_type == - 2 ) THEN
       DEALLOCATE( H%val )
       DEALLOCATE( A%val )
     ELSE IF ( data_storage_type == - 3 ) THEN
       DEALLOCATE( H%val )
       DEALLOCATE( A%val )
     ELSE IF ( data_storage_type == - 4 ) THEN
       DEALLOCATE( H%val )
       DEALLOCATE( A%val )
     ELSE IF ( data_storage_type == - 5 ) THEN
       DEALLOCATE( A%val )
     END IF
     IF ( data_storage_type == 0 ) THEN
       DEALLOCATE( C%val, C%row, C%col )
     ELSE IF ( data_storage_type == - 1 ) THEN
       DEALLOCATE( C%val, C%col )
     ELSE IF ( data_storage_type == - 2 .OR.                                   &
               data_storage_type == - 3 .OR.                                   &
               data_storage_type == - 4 ) THEN
       DEALLOCATE( C%val )
     END IF
   END DO
   DEALLOCATE( SOL, R )
   DEALLOCATE( H%ptr, A%ptr, C%ptr )

!  ==============================================
!  basic test of various symmetric linear solvers
!  ==============================================

   WRITE( 6, "( /, ' basic tests of symmetric linear solvers ' )" )
   WRITE( 6, "( /, 4X, 'solver   status residual' )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 3 ; c_ne = 3
   ALLOCATE( H%ptr( n + 1 ), A%ptr( m + 1 ), C%ptr( m + 1 ) )
   ALLOCATE( SOL( n + m ), R( n + m ) )
   DO solvers = 1, 14
     CALL SSLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = - 1
     control%sls_control%error = - 1
!    control%print_level = 1
     SELECT CASE( solvers )
     CASE ( 1 )
       control%symmetric_linear_solver = 'sils'
     CASE ( 2 )
       control%symmetric_linear_solver = 'ma57'
     CASE ( 3 )
       control%symmetric_linear_solver = 'ma77'
     CASE ( 4 )
       control%symmetric_linear_solver = 'ma86'
     CASE ( 5 )
       control%symmetric_linear_solver = 'ssids'
     CASE ( 6 )
       control%symmetric_linear_solver = 'ma97'
     CASE ( 7 )
       control%symmetric_linear_solver = 'pardiso'
     CASE ( 8 )
       control%symmetric_linear_solver = 'mkl_pardiso'
     CASE ( 9 )
       control%symmetric_linear_solver = 'wsmp'
     CASE ( 10 )
       control%symmetric_linear_solver = 'pastix'
     CASE ( 11 )
       control%symmetric_linear_solver = 'sytr'
     CASE ( 12 )
       control%symmetric_linear_solver = 'sytr'
     CASE ( 13 )
       CYCLE
     CASE ( 14 )
       control%symmetric_linear_solver = 'ssids'
     END SELECT
     IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
     CALL SMT_put( H%type, 'COORDINATE', smt_stat )  ; H%ne = h_ne
     ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
     H%row = (/ 1, 2, 3, 3 /)
     H%col = (/ 1, 2, 3, 1 /)
     IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
     CALL SMT_put( A%type, 'COORDINATE', smt_stat )  ; A%ne = a_ne
     ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
     A%row = (/ 1, 1, 2 /)
     A%col = (/ 1, 2, 3 /)
     IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
     CALL SMT_put( C%type, 'COORDINATE', smt_stat ) ; C%ne = c_ne
     ALLOCATE( C%val( c_ne ), C%row( c_ne ), C%col( c_ne ) )
     C%row = (/ 1, 2, 2 /)
     C%col = (/ 1, 1, 2 /)

!  test with new and existing data

     H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
     C%val = (/ 4.0_rp_, 1.0_rp_, 2.0_rp_ /)
     CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
     CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
     IF ( inform%status < 0 ) THEN
       WRITE( 6, "( A10, I9 )" )                                               &
         ADJUSTR( control%symmetric_linear_solver( 1 : 10 ) ),                 &
         inform%status
     ELSE
       SOL( : n ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
       SOL( n + 1 : ) = (/ 2.0_rp_, 1.0_rp_ /)
       R = SOL
       CALL SSLS_solve( n, m, SOL, data, control, inform )
       IF ( inform%status == 0 ) THEN
         CALL RESIDUAL( n, m, H, A, C, SOL, R )
         norm_residual = MAXVAL( ABS( R ) )
         WRITE( 6, "( A10, I9, A9 )" )                                         &
           ADJUSTR( control%symmetric_linear_solver( 1 : 10 ) ),               &
           inform%status, type_residual( norm_residual )
       ELSE
         WRITE( 6, "( A10, I9 )" )                                             &
           ADJUSTR( control%symmetric_linear_solver( 1 : 10 ) ),               &
           inform%status
       END IF
     END IF

     CALL SSLS_terminate( data, control, inform )
     DEALLOCATE( H%val, H%row, H%col )
     DEALLOCATE( A%val, A%row, A%col )
     DEALLOCATE( C%val, C%row, C%col )
   END DO
   DEALLOCATE( SOL, R )
   DEALLOCATE( H%ptr, A%ptr, C%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ' )" )

   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2
   ALLOCATE( H%ptr( n + 1 ), A%ptr( m + 1 ) )
   ALLOCATE( SOL( n + m ), R( n + m ) )

   IF ( ALLOCATED( C%type ) ) DEALLOCATE( C%type )
   CALL SMT_put( C%type, 'COORDINATE', smt_stat ) ; C%ne = 0
   ALLOCATE( C%val( C%ne ), C%row( C%ne ), C%col( C%ne ) )

   ALLOCATE( H%val( h_ne ), H%row( 0 ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( 0 ), A%col( a_ne ) )
   IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
   CALL SMT_put( H%type, 'SPARSE_BY_ROWS', smt_stat )
   H%col = (/ 1, 2 /)
   H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
   CALL SMT_put( A%type, 'SPARSE_BY_ROWS', smt_stat )
   A%col = (/ 1, 2 /)
   A%ptr = (/ 1, 3 /)
   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )

!  test with new and existing data

   WRITE( 6, "( /, ' test   status residual' )" )
   tests = 0
   DO i = 0, tests

     H%val = (/ 1.0_rp_, 1.0_rp_ /)
     A%val = (/ 1.0_rp_, 1.0_rp_ /)

!    control%print_level = 4
     CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
     CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
     IF ( inform%status == 0 ) THEN
       SOL( : n ) = (/ 0.0_rp_, 0.0_rp_ /)
       SOL( n + 1 : ) = (/ 1.0_rp_ /)
       R = SOL
     END IF
     CALL SSLS_solve( n, m, SOL, data, control, inform )
!    write(6,"('x=', 2ES12.4)") X
     IF ( inform%status == 0 ) THEN
       CALL RESIDUAL( n, m, H, A, C, SOL, R )
       norm_residual = MAXVAL( ABS( R ) )
       WRITE( 6, "( I5, I9, A9 )" )                                            &
         i, inform%status, type_residual( norm_residual )
     ELSE
       WRITE( 6, "( I5, I9 )" ) i, inform%status
     END IF
   END DO
   CALL SSLS_terminate( data, control, inform )

   DEALLOCATE( H%val, H%row, H%col, H%ptr, H%type )
   DEALLOCATE( A%val, A%row, A%col, A%ptr, A%type )
   DEALLOCATE( SOL, R )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ' )" )
   WRITE( 6, "( /, ' test   status residual' )" )

   n = 14 ; m = 8 ; h_ne = 28 ; a_ne = 27
   ALLOCATE( SOL( n + m ) , R( n + m ) )
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
   CALL SMT_put( A%type, 'COORDINATE', smt_stat )
   H%ne = h_ne ; A%ne = a_ne
   SOL( : n ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,       &
                   2.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_,       &
                   0.0_rp_, 2.0_rp_ /)
   SOL( n + 1 : ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
                       1.0_rp_, 2.0_rp_ /)
   R = SOL
   H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,            &
              4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,            &
              7.0_rp_, 7.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_,        &
              20.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_,      &
              20.0_rp_, 20.0_rp_, 20.0_rp_, 20.0_rp_ /)
   H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                   &
              1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                        &
              1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,            &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                  &
              6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                  &
              8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)

   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%print_level = 101
   control%out = scratch_out ;  control%error = scratch_out
!  control%print_level = 1 ; control%out = 6 ; control%error = 6
!  control%symmetric_linear_solver = 'ma57'
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status == 0 )                                                   &
     CALL SSLS_solve( n, m, SOL, data, control, inform )
   CLOSE( UNIT = scratch_out )
   IF ( inform%status == 0 ) THEN
     CALL RESIDUAL( n, m, H, A, C, SOL, R )
     norm_residual = MAXVAL( ABS( R ) )
     WRITE( 6, "( I5, I9, A9 )" )                                              &
       1, inform%status, type_residual( norm_residual )
   ELSE
     WRITE( 6, "( I5, I9 )" ) 1, inform%status
   END IF
   CALL SSLS_terminate( data, control, inform )
   DEALLOCATE( H%val, H%row, H%col, H%type )
   DEALLOCATE( A%val, A%row, A%col, A%type )
   DEALLOCATE( SOL, R )

!  Second problem

   IF ( .NOT. all_generic_tests ) GO TO 30
   n = 14 ; m = 8 ; h_ne = 14 ; a_ne = 27
   ALLOCATE( SOL( n + m ) , R( n + m ) )
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
   CALL SMT_put( A%type, 'COORDINATE', smt_stat )
   H%ne = h_ne ; A%ne = a_ne
   SOL( : n ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,       &
                   2.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_,       &
                   0.0_rp_, 2.0_rp_ /)
   SOL( n + 1 : ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_,                     &
                       0.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_ /)
   R = SOL
   H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,            &
              4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,            &
              7.0_rp_, 7.0_rp_ /)
   H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,            &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                  &
              6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                  &
              8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)
   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
!  control%print_level = 1 ; control%out = 6 ; control%error = 6
!  control%sls_control%ordering = 7
   CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status == 0 )                                                   &
     CALL SSLS_solve( n, m, SOL, data, control, inform )
   IF ( inform%status == 0 ) THEN
     CALL RESIDUAL( n, m, H, A, C, SOL, R )
     norm_residual = MAXVAL( ABS( R ) )
     WRITE( 6, "( I5, I9, A9 )" )                                              &
       2, inform%status, type_residual( norm_residual )
   ELSE
     WRITE( 6, "( I5, I9 )" ) 2, inform%status
   END IF
   CALL SSLS_terminate( data, control, inform )
   DEALLOCATE( H%val, H%row, H%col, H%type )
   DEALLOCATE( A%val, A%row, A%col, A%type )
   DEALLOCATE( SOL, R )

!  Third problem

30 CONTINUE
!  IF ( .NOT. all_generic_tests ) GO TO 40
!  WRITE( 25, "( /, ' third problem ', / )" )
   n = 14 ; m = 8 ; h_ne = 14 ; a_ne = 26
   ALLOCATE( SOL( n + m ) , R( n + m ) )
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
   CALL SMT_put( A%type, 'COORDINATE', smt_stat )
   H%ne = h_ne ; A%ne = a_ne
   SOL( : n ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,       &
                   2.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_,       &
                   0.0_rp_, 2.0_rp_ /)
   SOL( n + 1 : ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,            &
                       0.0_rp_, 1.0_rp_, 2.0_rp_ /)
   R = SOL
   A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5,                     &
              6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 6,                     &
              8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)
   H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_, 6.0_rp_, 7.0_rp_,   &
              1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_, 6.0_rp_, 7.0_rp_ /)
   H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
!  SOL = 0.0_rp_
!  DO l = 1, a_ne
!    i = A%row( l )
!    j = A%col( l )
!    val = A%val( l )
!    SOL( n + i ) = SOL( n + i ) + val
!    SOL( j ) = SOL( j ) + val
!  END DO
!  DO l = 1, h_ne
!    i = H%row( l )
!    j = H%col( l )
!    val = H%val( l )
!    SOL( i ) = SOL( i ) + val
!    IF ( i /= j ) SOL( j ) = SOL( j ) + val
!  END DO
   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status == 0 )                                                   &
     CALL SSLS_solve( n, m, SOL, data, control, inform )
!  ALLOCATE( SOL1( n + m ) )
!  SOL1 = SOL
!  WRITE( 6, "( ' solution ', /, ( 3ES24.16 ) )" ) SOL
!     WRITE( 25,"( ' solution ', /, ( 5ES24.16 ) )" ) SOL
   IF ( inform%status == 0 ) THEN
     CALL RESIDUAL( n, m, H, A, C, SOL, R )
     norm_residual = MAXVAL( ABS( R ) )
     WRITE( 6, "( I5, I9, A9 )" )                                              &
       3, inform%status, type_residual( norm_residual )
   ELSE
     WRITE( 6, "( I5, I9 )" ) 3, inform%status
   END IF
   CALL SSLS_terminate( data, control, inform )
   DEALLOCATE( H%val, H%row, H%col, H%type )
   DEALLOCATE( A%val, A%row, A%col, A%type )
   DEALLOCATE( SOL, R )

!  Forth problem

!40 CONTINUE
   IF ( .NOT. all_generic_tests ) GO TO 50
!  WRITE( 25, "( /, ' forth problem ', / )" )
   n = 14 ; m = 8 ; h_ne = 14 ; a_ne = 26
   ALLOCATE( SOL( n + m ) , R( n + m ) )
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
   CALL SMT_put( A%type, 'COORDINATE', smt_stat )
   H%ne = h_ne ; A%ne = a_ne
   SOL( : n ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,       &
                   2.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_,       &
                   0.0_rp_, 2.0_rp_ /)
   SOL( n + 1 : ) = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,            &
                       0.0_rp_, 1.0_rp_, 2.0_rp_ /)
   R = SOL
   A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5,                     &
              6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 6,                     &
              8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)
   H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_, 6.0_rp_, 7.0_rp_,   &
              1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_, 6.0_rp_, 7.0_rp_ /)
   H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   CALL SSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
!  control%print_level = 2 ; control%out = 6 ; control%error = 6
   CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status == 0 )                                                   &
     CALL SSLS_solve( n, m, SOL, data, control, inform )
!  WRITE( 6, "( ' solution ', /, ( 3ES24.16 ) )" ) SOL
!  WRITE(6,*) ' diff ', MAXVAL( ABS( SOL - SOL1 ) )
!     WRITE( 25,"( ' solution ', /, ( 5ES24.16 ) )" ) SOL
   IF ( inform%status == 0 ) THEN
     CALL RESIDUAL( n, m, H, A, C, SOL, R )
     norm_residual = MAXVAL( ABS( R ) )
     WRITE( 6, "( I5, I9, A9 )" )                                              &
        4, inform%status, type_residual( norm_residual )
   ELSE
     WRITE( 6, "( I5, I9 )" ) 4, inform%status
   END IF
   CALL SSLS_terminate( data, control, inform )
   DEALLOCATE( H%val, H%row, H%col, H%type )
   DEALLOCATE( A%val, A%row, A%col, A%type )
   DEALLOCATE( SOL, R )
!  DEALLOCATE( SOL1 )

50 CONTINUE
   DEALLOCATE( C%val, C%row, C%col, C%type )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     CHARACTER ( len = 8 ) FUNCTION type_residual( residual )
     REAL ( KIND = rp_ ) :: residual
     REAL, PARAMETER :: ten = 10.0_rp_
#ifdef REAL_32
     REAL, PARAMETER :: tiny = ten ** ( - 6 )
     REAL, PARAMETER :: small = ten ** ( - 3 )
     REAL, PARAMETER :: medium = ten ** ( - 2 )
#else
     REAL, PARAMETER :: tiny = ten ** ( - 12 )
     REAL, PARAMETER :: small = ten ** ( - 8 )
     REAL, PARAMETER :: medium = ten ** ( - 4 )
#endif
     REAL, PARAMETER :: large = 1.0_rp_
     IF ( ABS( residual ) < tiny ) THEN
       type_residual = '    tiny'
     ELSE IF ( ABS( residual ) < small ) THEN
       type_residual = '   small'
     ELSE IF ( ABS( residual ) < medium ) THEN
       type_residual = '  medium'
     ELSE IF ( ABS( residual ) < large ) THEN
       type_residual = '   large'
     ELSE
       type_residual = '    huge'
     END IF
     RETURN
     END FUNCTION type_residual

     SUBROUTINE WHICH_sls( control )
     TYPE ( SSLS_control_type ) :: control
#include "galahad_sls_defaults_sls.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     END SUBROUTINE WHICH_sls

     SUBROUTINE RESIDUAL( n, m, H, A, C, SOL, R )
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
     TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n + m ) :: SOL
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + m ) :: R
     INTEGER ( KIND = ip_ ) :: i, ii, j, l, np1, npm
     np1 = n + 1 ; npm = n + m

!  compute the residual r <- r - K sol

     SELECT CASE ( SMT_get( H%type ) )
     CASE ( 'IDENTITY' )
       R( : n ) = R( : n ) - SOL( : n )
     CASE ( 'SCALED_IDENTITY' )
       R( : n ) = R( : n ) - H%val( 1 ) * SOL( : n )
     CASE ( 'DIAGONAL' )
       R( : n ) = R( : n ) - H%val( : n ) * SOL( : n )
     CASE ( 'DENSE' )
       l = 0
       DO i = 1, n
         DO j = 1, i - 1
           l = l + 1
           R( i ) = R( i ) - H%val( l ) * SOL( j )
           R( j ) = R( j ) - H%val( l ) * SOL( i )
         END DO
         l = l + 1
         R( i ) = R( i ) - H%val( l ) * SOL( i )
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO i = 1, n
         DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
           j = H%col( l )
           IF ( i /= j ) THEN
             R( i ) = R( i ) - H%val( l ) * SOL( j )
             R( j ) = R( j ) - H%val( l ) * SOL( i )
           ELSE
             R( i ) = R( i ) - H%val( l ) * SOL( i )
           END IF
         END DO
       END DO
     CASE ( 'COORDINATE' )
       DO l = 1, H%ne
         i = H%row( l ) ; j = H%col( l )
         IF ( i /= j ) THEN
           R( i ) = R( i ) - H%val( l ) * SOL( j )
           R( j ) = R( j ) - H%val( l ) * SOL( i )
         ELSE
           R( i ) = R( i ) - H%val( l ) * SOL( i )
         END IF
       END DO
     END SELECT

     SELECT CASE ( SMT_get( A%type ) )
     CASE ( 'DENSE' )
       l = 0
       DO ii = 1, m
         i = n + ii
         DO j = 1, n
           l = l + 1
           R( i ) = R( i ) - A%val( l ) * SOL( j )
           R( j ) = R( j ) - A%val( l ) * SOL( i )
         END DO
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO ii = 1, m
         i = n + ii
         DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
           j = A%col( l )
           R( i ) = R( i ) - A%val( l ) * SOL( j )
           R( j ) = R( j ) - A%val( l ) * SOL( i )
         END DO
       END DO
     CASE ( 'COORDINATE' )
       DO l = 1, A%ne
         i = n + A%row( l ) ; j = A%col( l )
         R( i ) = R( i ) - A%val( l ) * SOL( j )
         R( j ) = R( j ) - A%val( l ) * SOL( i )
       END DO
     END SELECT

     SELECT CASE ( SMT_get( C%type ) )
     CASE ( 'IDENTITY' )
       R( np1 : npm ) = R( np1 : npm ) + SOL( np1 : npm ) 
     CASE ( 'SCALED_IDENTITY' )
       R( np1 : npm ) = R( np1 : npm ) + C%val( 1 ) * SOL( np1 : npm ) 
     CASE ( 'DIAGONAL' )
       R( np1 : npm ) = R( np1 : npm ) + C%val( 1 : m ) * SOL( np1 : npm ) 
     CASE ( 'DENSE' )
       l = 0
       DO i = n + 1, n + m
         DO j = n + 1, i - 1
           l = l + 1
           R( i ) = R( i ) + C%val( l ) * SOL( j )
           R( j ) = R( j ) + C%val( l ) * SOL( i )
         END DO
         l = l + 1
         R( i ) = R( i ) + C%val( l ) * SOL( i )
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO ii = 1, m
         i = n + ii
         DO l = C%ptr( ii ), C%ptr( ii + 1 ) - 1
           j = n + C%col( l )
           IF ( i /= j ) THEN
             R( i ) = R( i ) + C%val( l ) * SOL( j )
             R( j ) = R( j ) + C%val( l ) * SOL( i )
           ELSE
             R( i ) = R( i ) + C%val( l ) * SOL( i )
           END IF
         END DO
       END DO
     CASE ( 'COORDINATE' )
       DO l = 1, C%ne
         i = n + C%row( l ) ; j = n + C%col( l )
         IF ( i /= j ) THEN
           R( i ) = R( i ) + C%val( l ) * SOL( j )
           R( j ) = R( j ) + C%val( l ) * SOL( i )
         ELSE
           R( i ) = R( i ) + C%val( l ) * SOL( i )
         END IF
       END DO
     END SELECT
     END SUBROUTINE RESIDUAL
   END PROGRAM GALAHAD_SSLS_EXAMPLE


