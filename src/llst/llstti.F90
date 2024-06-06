! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 13:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LLST_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LLST_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, zero = 0.0_rp_
!  INTEGER ( KIND = ip_ ), PARAMETER :: m = 2, n = 2 * m + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 100, n = 2 * m + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: nea = 3 * m, nead = m * n
   INTEGER ( KIND = ip_ ), PARAMETER :: nes = n, nesd = n * ( n + 1 ) / 2
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), DIMENSION( m ) :: B
   REAL ( KIND = rp_ ) :: radius
   INTEGER ( KIND = ip_ ), DIMENSION( nea ) :: A_row
   INTEGER ( KIND = ip_ ), DIMENSION( nea ) :: A_col
   INTEGER ( KIND = ip_ ), DIMENSION( m + 1 ) :: A_ptr
   REAL ( KIND = rp_ ), DIMENSION( nea ) :: A_val
   REAL ( KIND = rp_ ), DIMENSION( nead ) :: A_dense_val
   INTEGER ( KIND = ip_ ), DIMENSION( nes ) :: S_row
   INTEGER ( KIND = ip_ ), DIMENSION( nes ) :: S_col
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: S_ptr
   REAL ( KIND = rp_ ), DIMENSION( nes ) :: S_val
   REAL ( KIND = rp_ ), DIMENSION( nesd ) :: S_dense_val
   CHARACTER ( LEN = 14 ) :: A_type, S_type
   TYPE ( LLST_full_data_type ) :: data
   TYPE ( LLST_control_type ) :: control
   TYPE ( LLST_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: A_ne, S_ne, i, l, data_storage_type, use_s, status
   CHARACTER ( len = 1 ) :: st

! A = ( I : Diag(1:n) : e )
   l = 1
   DO i = 1, m
     A_ptr( i ) = l
     A_row( l ) = i ; A_col( l ) = i ; A_val( l ) = one
     l = l + 1
     A_row( l ) = i ; A_col( l ) = m + i ;  A_val( l ) = REAL( i, rp_ )
     l = l + 1
     A_row( l ) = i ; A_col( l ) = n ;  A_val( l ) = one
     l = l + 1
   END DO
   A_ptr( m + 1 ) = l
   A_dense_val = zero
   l = 0
   DO i = 1, m
     A_dense_val( l + i ) = one
     A_dense_val( l + m + i ) = REAL( i, rp_ )
     A_dense_val( l + n ) = one
     l = l + n
   END DO

! S = diag(1:n)**2
   DO i = 1, n
     S_row( i ) = i ; S_col( i ) = i ; S_ptr( i ) = i
     S_val( i ) = REAL( i * i, rp_ )
   END DO
   S_ptr( n + 1 ) = n + 1
   S_dense_val = zero
   l = 0
   DO i = 1, n
     S_dense_val( l + i ) = REAL( i * i, rp_ )
     l = l + i
   END DO

! b is a vector of ones
   B = one

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

!  DO data_storage_type = 3, 3
   DO data_storage_type = 1, 4
     CALL LLST_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%print_level = 1
!    control%sls_control%print_level_solver = 3
!    control%sls_control%print_level = 3
     control%sbls_control%symmetric_linear_solver = "sytr  "
     control%sbls_control%definite_linear_solver = "potr  "
     control%definite_linear_solver = "potr  "
     radius = one
     IF ( data_storage_type == 1 ) THEN
       st = 'C' ; A_type = 'COORDINATE    ' ; A_ne = nea
     ELSE IF ( data_storage_type == 2 ) THEN
       st = 'R' ; A_type = 'SPARSE_BY_ROWS' ; A_ne = nea
     ELSE IF ( data_storage_type == 3 ) THEN
       st = 'D' ; A_type = 'DENSE         ' ; A_ne = nead
     ELSE IF ( data_storage_type == 4 ) THEN
       st = 'I' ; A_type = 'COORDINATE    ' ; A_ne = nea
     END IF
     CALL LLST_import( control, data, status, m, n,                            &
                       TRIM( A_type ), A_ne, A_row, A_col, A_ptr )
     DO use_s = 0, 1
       IF ( use_s == 0 ) THEN
         IF ( data_storage_type == 3 ) THEN
           CALL LLST_solve_problem( data, status, radius, A_dense_val, B, X )
         ELSE
           CALL LLST_solve_problem( data, status, radius, A_val, B, X )
         END IF
       ELSE
         IF ( data_storage_type == 1 ) THEN
           S_type = 'COORDINATE    ' ; S_ne = nes
         ELSE IF ( data_storage_type == 2 ) THEN
           S_type = 'SPARSE_BY_ROWS' ; S_ne = nes
         ELSE IF ( data_storage_type == 3 ) THEN
           S_type = 'DENSE         ' ; S_ne = nesd
         ELSE IF ( data_storage_type == 4 ) THEN
           S_type = 'DIAGONAL      ' ; S_ne = nes
         END IF
         CALL LLST_import_scaling( data, status,                               &
                                   TRIM( S_type ), S_ne, S_row, S_col, S_ptr )
         IF ( data_storage_type == 3 ) THEN
           CALL LLST_solve_problem( data, status, radius, A_dense_val, B, X,   &
                                    S_val = S_dense_val )
         ELSE
           CALL LLST_solve_problem( data, status, radius, A_val, B, X,         &
                                    S_val = S_val )
         END IF
       END IF
       CALL LLST_information( data, inform, status )
       WRITE( 6, "( ' storage type ', A1, I0, ' LLST_solve exit status = ',    &
      &  I0, ' ||r|| = ', F5.2 )" ) st, use_s, inform%status, inform%r_norm
     END DO
     CALL LLST_terminate( data, control, inform ) ! delete workspace
   END DO

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( LLST_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%definite_linear_solver = definite_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_LLST_interface_test
