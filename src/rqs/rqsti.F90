! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 13:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_RQS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_RQS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, two = 2.0_rp_
   INTEGER ( KIND = ip_ ) :: status, storage_type, a_is, m_is
   LOGICAL :: use_a, use_m
   INTEGER ( KIND = ip_ ) :: h_is, h_n, status_ref, failures
   REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
   REAL ( KIND = rp_ ) :: tol, sigma
   REAL ( KIND = rp_ ), DIMENSION( n ) :: G, H_s, X_ref
   REAL ( KIND = rp_ ), DIMENSION( m ) :: Y, Y_ref
   CHARACTER ( LEN = 15 ) :: h_type
   TYPE ( RQS_full_data_type ) :: data
   TYPE ( RQS_control_type ) :: control
   TYPE ( RQS_inform_type ) :: inform
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( len = 2 ) :: ma
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 1
   INTEGER ( KIND = ip_ ), PARAMETER :: h_ne = 4, m_ne = 3, a_ne = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: h_dense_ne = n * ( n + 1 ) / 2
   INTEGER ( KIND = ip_ ), PARAMETER :: m_dense_ne = h_dense_ne
   REAL ( KIND = rp_ ) :: f = 0.96_rp_, power = 3_rp_, weight = 1.0_rp_
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: empty
   REAL ( KIND = rp_ ), DIMENSION( n ) :: C = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   INTEGER ( KIND = ip_ ), DIMENSION( h_ne ) :: H_row = (/ 1, 2, 3, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( h_ne ) :: H_col = (/ 1, 2, 3, 1 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: H_ptr = (/ 1, 2, 3, 5 /)
   REAL ( KIND = rp_ ), DIMENSION( h_ne ) ::                                   &
     H_val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
   INTEGER ( KIND = ip_ ), DIMENSION( m_ne ) :: M_row = (/ 1, 2, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( m_ne ) :: M_col = (/ 1, 2, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: M_ptr = (/ 1, 2, 3, 4 /)
   REAL ( KIND = rp_ ), DIMENSION( m_ne ) ::                                   &
     M_val = (/ 1.0_rp_, 2.0_rp_, 1.0_rp_ /)
   INTEGER ( KIND = ip_ ), DIMENSION( a_ne ) :: A_row = (/ 1, 1, 1 /)
   INTEGER ( KIND = ip_ ), DIMENSION( a_ne ) :: A_col = (/ 1, 2, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( m + 1 ) :: A_ptr = (/ 1, 4 /)
   REAL ( KIND = rp_ ), DIMENSION( a_ne ) ::                                   &
     A_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( h_dense_ne ) ::                             &
     H_dense_val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( m_dense_ne ) ::                             &
     M_dense_val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     H_diag_val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     M_diag_val = (/ 1.0_rp_, 2.0_rp_, 1.0_rp_ /)

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

    DO a_is = 0, 1
     use_a = a_is == 1  ! add a linear constraint?
     DO m_is = 0, 1
       use_m = m_is == 1 ! include a scaling matrix?
       IF ( use_a .AND. use_m ) THEN
         ma = 'MA'
       ELSE IF ( use_a ) THEN
         ma = 'A '
       ELSE IF ( use_m ) THEN
         ma = 'M '
       ELSE
         ma = '  '
       END IF
       DO storage_type = 1, 4 ! loop over a variety of storage types
         CALL RQS_initialize( data, control, inform )
         CALL WHICH_sls( control )
         SELECT CASE ( storage_type )
         CASE ( 1 ) ! sparse co-ordinate storage
           st = 'C'
           CALL RQS_import( control, data, status, n, 'COORDINATE',            &
                            H_ne, H_row, H_col, empty )
           IF ( use_m ) CALL RQS_import_M( data, status, 'COORDINATE',         &
                                           M_ne, M_row, M_col, empty )
           IF ( use_a ) CALL RQS_import_A( data, status, m, 'COORDINATE',      &
                                           A_ne, A_row, A_col, empty )
           IF ( use_a .AND. use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, M_val = M_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, M_val = M_val )
           ELSE
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X )
           END IF
         CASE ( 2 ) ! sparse row-wise storage
           st = 'R'
           CALL RQS_import( control, data, status, n, 'SPARSE_BY_ROWS',        &
                            H_ne, empty, H_col, H_ptr )
           IF ( use_m ) CALL RQS_import_M( data, status, 'SPARSE_BY_ROWS',     &
                                           M_ne, empty, M_col, M_ptr )
           IF ( use_a ) CALL RQS_import_A( data, status, m, 'SPARSE_BY_ROWS',  &
                                           A_ne, empty, A_col, A_ptr )
           IF ( use_a .AND. use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, M_val = M_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X, M_val = M_val )
           ELSE
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_val, X )
           END IF
         CASE ( 3 ) ! dense storage
           st = 'D'
           CALL RQS_import( control, data, status, n, 'DENSE',                 &
                            H_ne, empty, empty, empty )
           IF ( use_m ) CALL RQS_import_M( data, status, 'DENSE',              &
                                           M_ne, empty, empty, empty )
           IF ( use_a ) CALL RQS_import_A( data, status, m, 'DENSE',           &
                                           A_ne, empty, empty, empty )
           IF ( use_a .AND. use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_dense_val, X,                           &
                                     M_val = M_dense_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_dense_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_dense_val, X, M_val = M_dense_val )
           ELSE
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_dense_val, X )
           END IF
         CASE ( 4 ) ! diagonal H, M, dense A
           st = 'I'
           CALL RQS_import( control, data, status, n, 'DIAGONAL',              &
                            H_ne, empty, empty, empty )
           IF ( use_m ) CALL RQS_import_M( data, status, 'DIAGONAL',           &
                                           M_ne, empty, empty, empty )
           IF ( use_a ) CALL RQS_import_A( data, status, m, 'DIAGONAL',        &
                                           A_ne, empty, empty, empty )
           IF ( use_a .AND. use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_diag_val, X,                            &
                                     M_val = M_diag_val, A_val = A_val )
           ELSE IF ( use_a ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_diag_val, X, A_val = A_val )
           ELSE IF ( use_m ) THEN
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_diag_val, X, M_val = M_diag_val )
           ELSE
             CALL RQS_solve_problem( data, status, power, weight, f, C,        &
                                     H_diag_val, X )
           END IF
         END SELECT
         CALL RQS_information( data, inform, status )
         WRITE( 6, "( ' format ', A1, A2, ':',                                 &
        &  ' RQS_solve_problem exit status = ', I4, ', f = ', f0.2 )" )        &
          st, ma, status, inform%obj
       END DO
     END DO
     CALL RQS_terminate( data, control, inform ) !  delete internal workspace
   END DO

!  =============================================================
!  tests of scaled-identity, identity and zero H and identity M
!  =============================================================

!  minimize g^T x + 1/2 x^T H x + sigma/p ||x||_M^p, optionally with
!  x_1 + x_2 + x_3 = 0. Each solve is compared with the same problem
!  using explicit storage (a diagonal H, and no M), skipped if that fails

   WRITE( 6, "( /, ' ==== tests of identity storage formats ===== ', / )" )
   failures = 0 ; tol = EPSILON( one ) ** 0.33_rp_
   sigma = one
   DO a_is = 0, 1
     use_a = a_is == 1
     IF ( use_a ) THEN
       G = (/ 4.0_rp_, 1.0_rp_, - 2.0_rp_ /)
     ELSE
       G = (/ 3.0_rp_, 0.0_rp_, 4.0_rp_ /)
     END IF
     DO h_is = 1, 4
       SELECT CASE ( h_is )
       CASE ( 1 )
         h_type = 'SCALED_IDENTITY' ; H_s = two ; h_n = 1
       CASE ( 2 )
         h_type = 'IDENTITY' ; H_s = one ; h_n = 0
       CASE ( 3 )
         h_type = 'ZERO' ; H_s = zero ; h_n = 0
       CASE ( 4 )
         h_type = 'NONE' ; H_s = zero ; h_n = 0
       END SELECT
       CALL SOLVE( 'DIAGONAL', H_s, .FALSE., use_a, G, sigma,                  &
                   X_ref, Y_ref, status_ref )
       CALL SOLVE( TRIM( h_type ), H_s( : h_n ), .FALSE., use_a, G, sigma,     &
                   X, Y, status )
       CALL CHECK( TRIM( h_type ), use_a )
     END DO
   END DO

!  the example H, with and without the constraint, and M = I

   G = (/ 0.0_rp_, 5.0_rp_ / 3.0_rp_, 4.0_rp_ / 3.0_rp_ /)
   DO a_is = 0, 1
     use_a = a_is == 1
     CALL SOLVE( 'COORDINATE', H_val, .FALSE., use_a, G, sigma,                &
                 X_ref, Y_ref, status_ref )
     CALL SOLVE( 'COORDINATE', H_val, .TRUE., use_a, G, sigma,                 &
                 X, Y, status )
     CALL CHECK( 'M = IDENTITY', use_a )
   END DO

   IF ( failures > 0 ) THEN
     WRITE( 6, "( /, 1X, I0, ' identity storage test(s) failed' )" ) failures
     STOP 1
   END IF

   WRITE( 6, "( /, ' tests completed' )" )

   STOP

   CONTAINS

     SUBROUTINE SOLVE( h_type, H_v, m_identity, use_a, G, sigma, X, Y, status )

!  solve the problem with the given H storage, M = I if m_identity, and
!  the constraint if use_a, returning the solve status

     CHARACTER ( LEN = * ), INTENT( IN ) :: h_type
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_v
     LOGICAL, INTENT( IN ) :: m_identity, use_a
     REAL ( KIND = rp_ ), DIMENSION( n ), INTENT( IN ) :: G
     REAL ( KIND = rp_ ), INTENT( IN ) :: sigma
     REAL ( KIND = rp_ ), DIMENSION( n ), INTENT( OUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( m ), INTENT( OUT ) :: Y
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( RQS_full_data_type ) :: data
     TYPE ( RQS_control_type ) :: control
     TYPE ( RQS_inform_type ) :: inform
     INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: no_index
     REAL ( KIND = rp_ ), DIMENSION( 0 ) :: no_val

     X = zero ; Y = zero
     CALL RQS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     IF ( h_type == 'COORDINATE' ) THEN
       CALL RQS_import( control, data, status, n, h_type,                      &
                        h_ne, H_row, H_col, no_index )
     ELSE
       CALL RQS_import( control, data, status, n, h_type,                      &
                        SIZE( H_v, KIND = ip_ ), no_index, no_index, no_index )
     END IF
     IF ( status == 0 .AND. m_identity )                                       &
       CALL RQS_import_M( data, status, 'IDENTITY', 0_ip_,                     &
                          no_index, no_index, no_index )
     IF ( status == 0 .AND. use_a )                                            &
       CALL RQS_import_A( data, status, m, 'COORDINATE',                       &
                          a_ne, A_row, A_col, no_index )
     IF ( status == 0 ) THEN
       IF ( m_identity .AND. use_a ) THEN
         CALL RQS_solve_problem( data, status, power, sigma, zero,             &
                                 G, H_v, X, M_val = no_val,                    &
                                 A_val = A_val, Y = Y )
       ELSE IF ( use_a ) THEN
         CALL RQS_solve_problem( data, status, power, sigma, zero,             &
                                 G, H_v, X, A_val = A_val, Y = Y )
       ELSE IF ( m_identity ) THEN
         CALL RQS_solve_problem( data, status, power, sigma, zero,             &
                                 G, H_v, X, M_val = no_val )
       ELSE
         CALL RQS_solve_problem( data, status, power, sigma, zero,             &
                                 G, H_v, X )
       END IF
     END IF
     CALL RQS_terminate( data, control, inform )
     END SUBROUTINE SOLVE

     SUBROUTINE CHECK( name, use_a )

!  compare the solve with the reference one, if that succeeded

     CHARACTER ( LEN = * ), INTENT( IN ) :: name
     LOGICAL, INTENT( IN ) :: use_a
     CHARACTER ( LEN = 4 ) :: with

     with = '    ' ; IF ( use_a ) with = ', A '
     IF ( status_ref /= 0 ) THEN
       WRITE( 6, "( 1X, A, A, ': skipped, reference status = ', I0 )" )        &
         name, TRIM( with ), status_ref
     ELSE IF ( status /= 0 .OR. MAXVAL( ABS( X - X_ref ) ) > tol .OR.          &
               MAXVAL( ABS( Y - Y_ref ) ) > tol ) THEN
       failures = failures + 1
       WRITE( 6, "( 1X, A, A, ': failed, status = ', I0 )" )                   &
         name, TRIM( with ), status
     ELSE
       WRITE( 6, "( 1X, A, A, ': ok' )" ) name, TRIM( with )
     END IF
     END SUBROUTINE CHECK

     SUBROUTINE WHICH_sls( control )
     TYPE ( RQS_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_RQS_interface_test
