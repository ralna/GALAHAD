! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 13:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LLSR_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LLSR_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, zero = 0.0_rp_
!  INTEGER ( KIND = ip_ ), PARAMETER :: m = 2, n = 2 * m + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 100, n = 2 * m + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: nea = 3 * m, nead = m * n
   INTEGER ( KIND = ip_ ), PARAMETER :: nes = n, nesd = n * ( n + 1 ) / 2
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), DIMENSION( m ) :: B
   REAL ( KIND = rp_ ) :: power = 3.0_rp_
   REAL ( KIND = rp_ ) :: weight = 1.0_rp_
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
   TYPE ( LLSR_full_data_type ) :: data
   TYPE ( LLSR_control_type ) :: control
   TYPE ( LLSR_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: A_ne, S_ne, i, l, data_storage_type, use_s, status
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), PARAMETER :: m2 = 4, n2 = 2, ne2 = 4
   INTEGER ( KIND = ip_ ), DIMENSION( ne2 ) :: A2_row = (/ 1, 2, 3, 4 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne2 ) :: A2_col = (/ 1, 2, 1, 2 /)
   REAL ( KIND = rp_ ), DIMENSION( ne2 ) ::                                    &
     A2_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( m2 ) ::                                     &
     B2 = (/ 0.9_rp_, 0.8_rp_, 0.9_rp_, 2.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n2 ) :: X_ref, X_s
   INTEGER ( KIND = ip_ ) :: status_ref, status_s, failures
   REAL ( KIND = rp_ ) :: tol

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
     CALL LLSR_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%print_level = 1
!    control%sls_control%print_level_solver = 3
!    control%sls_control%print_level = 3
     control%sbls_control%symmetric_linear_solver = "sytr  "
     control%sbls_control%definite_linear_solver = "potr  "
     control%definite_linear_solver = "potr  "
     IF ( data_storage_type == 1 ) THEN
       st = 'C' ; A_type = 'COORDINATE    ' ; A_ne = nea
     ELSE IF ( data_storage_type == 2 ) THEN
       st = 'R' ; A_type = 'SPARSE_BY_ROWS' ; A_ne = nea
     ELSE IF ( data_storage_type == 3 ) THEN
       st = 'D' ; A_type = 'DENSE         ' ; A_ne = nead
     ELSE IF ( data_storage_type == 4 ) THEN
       st = 'I' ; A_type = 'COORDINATE    ' ; A_ne = nea
     END IF
     CALL LLSR_import( control, data, status, m, n,                            &
                       TRIM( A_type ), A_ne, A_row, A_col, A_ptr )
     DO use_s = 0, 1
       IF ( use_s == 0 ) THEN
         IF ( data_storage_type == 3 ) THEN
           CALL LLSR_solve_problem( data, status, power, weight, A_dense_val,  &
                                    B, X )
         ELSE
           CALL LLSR_solve_problem( data, status, power, weight, A_val, B, X )
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
         CALL LLSR_import_scaling( data, status,                               &
                                   TRIM( S_type ), S_ne, S_row, S_col, S_ptr )
         IF ( data_storage_type == 3 ) THEN
           CALL LLSR_solve_problem( data, status, power, weight, A_dense_val,  &
                                    B, X, S_val = S_dense_val )
         ELSE
           CALL LLSR_solve_problem( data, status, power, weight, A_val, B, X,  &
                                    S_val = S_val )
         END IF
       END IF
       CALL LLSR_information( data, inform, status )
       WRITE( 6, "( ' storage type ', A1, I0, ' LLSR_solve exit status = ',    &
      &  I0, ' ||r|| = ', F5.2 )" ) st, use_s, inform%status, inform%r_norm
     END DO
     CALL LLSR_terminate( data, control, inform ) ! delete workspace
   END DO


!  ================================================================
!  identity-based and non-positive scalings S for a small problem:
!  minimize 1/2 || A x - b ||^2 + 1/3 || x ||_S^3
!  with A = ( I ; diag( 1, 2 ) ) and b = ( 0.9, 0.8, 0.9, 2 ). Each solve
!  is compared with one using explicit storage, skipped if that fails
!  ================================================================

   WRITE( 6, "( /, ' ==== tests of identity-based scalings ===== ', / )" )
   tol = SQRT( EPSILON( one ) ) ; failures = 0

!  S = 2 I given as a scaled identity, and as a diagonal

   CALL SOLVE_SMALL( 1_ip_, (/ 2.0_rp_, 2.0_rp_ /), status_ref, X_ref )
   CALL SOLVE_SMALL( 2_ip_, (/ 2.0_rp_ /), status_s, X_s )
   CALL CHECK( 'scaled identity S' )

!  S = I given as an identity without S_val, and no S

   CALL SOLVE_SMALL( 0_ip_, (/ one /), status_ref, X_ref )
   CALL SOLVE_SMALL( 3_ip_, (/ one /), status_s, X_s )
   CALL CHECK( 'identity S' )

!  a diagonal S with a negative entry is not positive definite (status -15)

   CALL SOLVE_SMALL( 1_ip_, (/ one, - one /), status_s, X_s )
   IF ( status_s /= - 15 ) THEN
     WRITE( 6, "( ' indefinite diagonal S: status = ', I0 )" ) status_s
     failures = failures + 1
   ELSE
     WRITE( 6, "( ' indefinite diagonal S: ok' )" )
   END IF
   IF ( failures > 0 ) STOP 1

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

     SUBROUTINE SOLVE_SMALL( s_kind, S_d, status, X2 )

!  solve the small problem with no S (s_kind = 0), a diagonal S = diag( S_d )
!  (1), a scaled identity S = S_d( 1 ) I (2) or an identity S given without
!  S_val (3), returning the exit status and solution

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: s_kind
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: S_d
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( n2 ), INTENT( OUT ) :: X2
     TYPE ( LLSR_full_data_type ) :: data
     TYPE ( LLSR_control_type ) :: control
     TYPE ( LLSR_inform_type ) :: inform
     INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: no_index

     X2 = zero
     CALL LLSR_initialize( data, control, inform )
     CALL WHICH_sls( control )
     CALL LLSR_import( control, data, status, m2, n2, 'COORDINATE',            &
                      ne2, A2_row, A2_col, no_index )
     SELECT CASE ( s_kind )
     CASE ( 0 )
       CALL LLSR_solve_problem( data, status, 3.0_rp_, 1.0_rp_, A2_val,        &
                                B2, X2 )
     CASE ( 1 )
       CALL LLSR_import_scaling( data, status, 'DIAGONAL', n2,                 &
                                no_index, no_index, no_index )
       CALL LLSR_solve_problem( data, status, 3.0_rp_, 1.0_rp_, A2_val,        &
                                B2, X2, S_val = S_d )
     CASE ( 2 )
       CALL LLSR_import_scaling( data, status, 'SCALED_IDENTITY', 1_ip_,       &
                                no_index, no_index, no_index )
       CALL LLSR_solve_problem( data, status, 3.0_rp_, 1.0_rp_, A2_val,        &
                                B2, X2, S_val = S_d )
     CASE ( 3 )
       CALL LLSR_import_scaling( data, status, 'IDENTITY', 0_ip_,              &
                                no_index, no_index, no_index )
       CALL LLSR_solve_problem( data, status, 3.0_rp_, 1.0_rp_, A2_val,        &
                                B2, X2 )
     END SELECT
     CALL LLSR_information( data, inform, status )
     status = inform%status
     CALL LLSR_terminate( data, control, inform )
     END SUBROUTINE SOLVE_SMALL

     SUBROUTINE CHECK( name )
     CHARACTER ( LEN = * ), INTENT( IN ) :: name
     IF ( status_ref /= 0 ) THEN
       WRITE( 6, "( 1X, A, ': skipped, reference status = ', I0 )" )           &
         name, status_ref
     ELSE IF ( status_s /= 0 .OR. MAXVAL( ABS( X_s - X_ref ) ) > tol ) THEN
       WRITE( 6, "( 1X, A, ': failed, status = ', I0 )" ) name, status_s
       failures = failures + 1
     ELSE
       WRITE( 6, "( 1X, A, ': ok' )" ) name
     END IF
     END SUBROUTINE CHECK
     SUBROUTINE WHICH_sls( control )
     TYPE ( LLSR_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%definite_linear_solver = definite_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_LLSR_interface_test
