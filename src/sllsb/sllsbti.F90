! THIS VERSION: GALAHAD 5.5 - 2026-01-19 AT 10:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SLLSB_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SLLSB_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( SLLSB_control_type ) :: control
   TYPE ( SLLSB_inform_type ) :: inform
   TYPE ( SLLSB_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, o, m, Ao_ne, Ao_dense_ne, A_dense_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: sigma
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, W, X_s
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B, R, Y
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_col, Ao_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat, COHORT
   CHARACTER ( len = 2 ) :: st
   CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = REPEAT( ' ', 30 )
!  symmetric_linear_solver = 'ssids'
!  symmetric_linear_solver = 'ma97 '
   symmetric_linear_solver = 'sytr '

! set up problem data

   n = 3 ; o = 4 ; m = 1 ; ao_ne = 7
   ao_dense_ne = o * n ; a_dense_ne = m * n
   ALLOCATE( X( n ), Z( n ), X_s( n ), X_stat( n ) )
   ALLOCATE( Y( m ), COHORT( n ) )
   ALLOCATE( B( o ), R( o ), W( o ) )
   B = [ 2.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ ]  ! observations
   X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_     ! start from zero
   W = [ 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_ ]  ! weights of one and two
   X_s = [ 0.5_rp_, 0.5_rp_, 0.5_rp_ ]         ! shifts of a half
   COHORT = [ 1, 1, 0 ]                        ! cohort uses variables 1 & 2 
   sigma = 1.0_rp_                             ! regularize by one

! vector problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of least-squares storage formats', / )" )

   DO data_storage_type = 1, 5
!  DO data_storage_type = 1, 1
     CALL SLLSB_initialize( data, control, inform )
!    control%print_level = 1 ; control%out = 6
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
       Ao_val = [ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                 &
                  1.0_rp_, 1.0_rp_ ] ! Ao
       Ao_row = [ 1, 1, 2, 2, 3, 3, 4 ]
       Ao_col = [ 1, 2, 2, 3, 1, 3, 2 ]
       CALL SLLSB_import( control, data, status, n, o, m,                      &
                          'coordinate', Ao_ne, Ao_row, Ao_col, Ao_ptr,         &
                          COHORT = COHORT )
     CASE ( 2 ) ! sparse by rows
       st = 'SR'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( 0 ), Ao_col( ao_ne ),                &
                 Ao_ptr( o + 1 ) )
       Ao_val = [ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                 &
                  1.0_rp_, 1.0_rp_ ] ! Ao
       Ao_col = [ 1, 2, 2, 3, 1, 3, 2 ]
       Ao_ptr = [ 1, 3, 5, 7, 8 ]
       CALL SLLSB_import( control, data, status, n, o, m,                      &
                          'sparse_by_rows', Ao_ne, Ao_row, Ao_col, Ao_ptr,     &
                          COHORT = COHORT )
     CASE ( 3 ) ! sparse by columns
       st = 'SC'
       ALLOCATE( Ao_val( ao_ne ), Ao_row( ao_ne ), Ao_col( 0 ),                &
                 Ao_ptr( n + 1 ) )
       Ao_val = [ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                 &
                  1.0_rp_, 1.0_rp_ ] ! Ao
       Ao_row = [ 1, 3, 1, 2, 4, 2, 3 ]
       Ao_ptr = [ 1, 3, 6, 8 ]
       CALL SLLSB_import( control, data, status, n, o, m,                      &
                          'sparse_by_columns', Ao_ne, Ao_row, Ao_col, Ao_ptr,  &
                          COHORT = COHORT )
     CASE ( 4 ) ! dense by rows
       st = 'DR'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       Ao_val = [ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,        &
                   1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_ ] ! Ao
       CALL SLLSB_import( control, data, status, n, o, m,                      &
                          'dense', Ao_dense_ne, Ao_row, Ao_col, Ao_ptr,        &
                          COHORT = COHORT )
     CASE ( 5 ) ! dense by columns
       st = 'DC'
       ALLOCATE( Ao_val( ao_dense_ne ), Ao_row( 0 ), Ao_col( 0 ), Ao_ptr( 0 ) )
       Ao_val = [ 1.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_,        &
                  0.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_ ] ! Ao
       CALL SLLSB_import( control, data, status, n, o, m, 'dense_by_columns',  &
                          Ao_dense_ne, Ao_row, Ao_col, Ao_ptr,                 &
                          COHORT = COHORT )
     END SELECT
     IF ( status == 0 ) THEN
       CALL SLLSB_solve_given_a( data, status, Ao_val, B, sigma,              &
                                 X, Y, Z, R, X_stat, W = W, X_s = X_s )
     END IF
     DEALLOCATE( Ao_val, Ao_row, Ao_col, Ao_ptr )
     CALL SLLSB_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': SLLSB_solve exit status = ', I0 )" ) st, inform%status
     END IF
     CALL SLLSB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( X, B, R, Y, Z, W, X_s, X_stat, COHORT )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_SLLSB_interface_test
