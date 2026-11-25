! THIS VERSION: GALAHAD 5.4 - 2025-11-17 AT 13:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_TREK_interface_test
   USE GALAHAD_KINDS_precision, ONLY: rp_, ip_
   USE GALAHAD_TREK_precision, ONLY: TREK_full_data_type, TREK_control_type,   &
         TREK_inform_type, TREK_initialize, TREK_import, TREK_S_import,        &
         TREK_solve_problem, TREK_information, TREK_terminate
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ) :: status, storage_type, r_is, s_is
   LOGICAL :: use_s
   TYPE ( TREK_full_data_type ) :: data
   TYPE ( TREK_control_type ) :: control
   TYPE ( TREK_inform_type ) :: inform
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( len = 2 ) :: sr
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 1
   INTEGER ( KIND = ip_ ), PARAMETER :: h_ne = 4, s_ne = 3, a_ne = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: h_dense_ne = n * ( n + 1 ) / 2
   INTEGER ( KIND = ip_ ), PARAMETER :: s_dense_ne = h_dense_ne
   REAL ( KIND = rp_ ) :: radius
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null_
   REAL ( KIND = rp_ ), DIMENSION( n ) :: C = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]
   INTEGER ( KIND = ip_ ), DIMENSION( h_ne ) :: H_row = [ 1, 2, 3, 3 ]
   INTEGER ( KIND = ip_ ), DIMENSION( h_ne ) :: H_col = [ 1, 2, 3, 1 ]
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: H_ptr = [ 1, 2, 3, 5 ]
   REAL ( KIND = rp_ ), DIMENSION( h_ne ) ::                                   &
     H_val = [ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ ]
   INTEGER ( KIND = ip_ ), DIMENSION( s_ne ) :: S_row = [ 1, 2, 3 ]
   INTEGER ( KIND = ip_ ), DIMENSION( s_ne ) :: S_col = [ 1, 2, 3 ]
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: S_ptr = [ 1, 2, 3, 4 ]
   REAL ( KIND = rp_ ), DIMENSION( s_ne ) ::                                   &
     S_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_ ]
   REAL ( KIND = rp_ ), DIMENSION( h_dense_ne ) ::                             &
     H_dense_val = [ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_ ]
   REAL ( KIND = rp_ ), DIMENSION( s_dense_ne ) ::                             &
     S_dense_val = [ 1.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ ]
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     H_diag_val = [ 1.0_rp_, 0.0_rp_, 2.0_rp_ ]
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     S_diag_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_ ]
   INTEGER ( KIND = ip_ ) :: status_ref, status_s, failures, iter
   REAL ( KIND = rp_ ) :: tol, x_norm
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X_ref, X_s
   REAL ( KIND = rp_ ), DIMENSION( 0 ) :: null_r
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     C_t = [ 3.0_rp_, 0.0_rp_, 4.0_rp_ ]
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     C_b = [ 0.0_rp_, 5.0_rp_, 4.0_rp_ ] / 3.0_rp_

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

   DO s_is = 0, 1
     use_s = s_is == 1 ! include a scaling matrix?
     DO storage_type = 1, 4 ! loop over a variety of storage types
       CALL TREK_initialize( data, control, inform )
       CALL WHICH_sls( control )
       SELECT CASE ( storage_type ) ! import the data
       CASE ( 1 ) ! sparse co-ordinate storage
         st = 'C'
         CALL TREK_import( control, data, status, n, 'COORDINATE',             &
                           H_ne, H_row, H_col, null_ )
         IF ( use_s ) CALL TREK_S_import( data, status, 'COORDINATE',          &
                                          S_ne, S_row, S_col, null_ )
       CASE ( 2 ) ! sparse row-wise storage
         st = 'R'
         CALL TREK_import( control, data, status, n, 'SPARSE_BY_ROWS',         &
                           H_ne, null_, H_col, H_ptr )
         IF ( use_s ) CALL TREK_S_import( data, status, 'SPARSE_BY_ROWS',      &
                                          S_ne, null_, S_col, S_ptr )
       CASE ( 3 ) ! dense storage
         st = 'D'
         CALL TREK_import( control, data, status, n, 'DENSE',                  &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL TREK_S_import( data, status, 'DENSE',               &
                                          S_ne, null_, null_, null_ )
       CASE ( 4 ) ! diagonal H, S
         st = 'G'
         CALL TREK_import( control, data, status, n, 'DIAGONAL',               &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL TREK_S_import( data, status, 'DIAGONAL',            &
                                          S_ne, null_, null_, null_ )
       END SELECT
       DO r_is = 1, 2
         control%new_radius = r_is == 2
         IF ( control%new_radius ) THEN   ! use a smaller radius
           radius = inform%next_radius
           IF ( use_s ) THEN
             sr = 'S+'
           ELSE
             sr = '+ '
           END IF
         ELSE   ! use the original radius
           radius = 1.0_rp_
           IF ( use_s ) THEN
             sr = 'S '
           ELSE
             sr = '  '
           END IF
         END IF
         SELECT CASE ( storage_type ) ! solve the problem
         CASE ( 1 ) ! sparse co-ordinate storage
           IF ( use_s ) THEN
             CALL TREK_solve_problem( data, status, H_val, C, radius, X,       &
                                      S_val = S_val )
           ELSE
             CALL TREK_solve_problem( data, status, H_val, C, radius, X )
           END IF
         CASE ( 2 ) ! sparse row-wise storage
           IF ( use_s ) THEN
             CALL TREK_solve_problem( data, status, H_val, C, radius, X,       &
                                      S_val = S_val )
           ELSE
             CALL TREK_solve_problem( data, status, H_val, C, radius, X )
           END IF
         CASE ( 3 ) ! dense storage
           IF ( use_s ) THEN
             CALL TREK_solve_problem( data, status, H_dense_val, C, radius,    &
                                      X, S_val = S_dense_val )
           ELSE
             CALL TREK_solve_problem( data, status, H_dense_val, C, radius, X )
           END IF
         CASE ( 4 ) ! diagonal H, S
           st = 'G'
           IF ( use_s ) THEN
             CALL TREK_solve_problem( data, status, H_diag_val, C, radius,     &
                                      X, S_val = S_diag_val )
           ELSE
             CALL TREK_solve_problem( data, status, H_diag_val, C, radius, X )
           END IF
         END SELECT
         CALL TREK_information( data, inform, status )
         WRITE( 6, "( ' format ', A1, A2, ':',                                 &
        &  ' TREK_solve_problem exit status = ', I4, ', f = ', f0.2 )" )       &
          st, sr, status, inform%obj
       END DO
     END DO
     CALL TREK_terminate( data, control, inform ) !  delete internal workspace
   END DO

!  ================================================================
!  storage schemes without indices, for the problem
!  minimize c^T x + 1/2 x^T H x subject to ||x||_S <= radius
!  with H as above or a multiple of I, and S absent or a multiple of I. Each
!  solve is compared with one using explicit storage, skipped if that fails
!  ================================================================

   WRITE( 6, "( /, ' ==== tests of storage schemes without indices ===== ',  &
  &  / )" )
   tol = SQRT( EPSILON( 1.0_rp_ ) ) ; failures = 0

!  H = 2 I, I and 0 given as a scaled identity, identity, zero and none, and
!  as a diagonal

   CALL SOLVE_SMALL( 'DIAGONAL', [ 2.0_rp_, 2.0_rp_, 2.0_rp_ ], '', null_r,    &
                     C_t, 1.0_rp_, status_ref, X_ref )
   CALL SOLVE_SMALL( 'SCALED_IDENTITY', [ 2.0_rp_ ], '', null_r,               &
                     C_t, 1.0_rp_, status_s, X_s )
   CALL CHECK( 'scaled identity H' )
   CALL SOLVE_SMALL( 'DIAGONAL', [ 1.0_rp_, 1.0_rp_, 1.0_rp_ ], '', null_r,    &
                     C_t, 1.0_rp_, status_ref, X_ref )
   CALL SOLVE_SMALL( 'IDENTITY', null_r, '', null_r, C_t, 1.0_rp_,             &
                     status_s, X_s )
   CALL CHECK( 'identity H' )
   CALL SOLVE_SMALL( 'DIAGONAL', [ 0.0_rp_, 0.0_rp_, 0.0_rp_ ], '', null_r,    &
                     C_t, 1.0_rp_, status_ref, X_ref )
   CALL SOLVE_SMALL( 'ZERO', null_r, '', null_r, C_t, 1.0_rp_, status_s, X_s )
   CALL CHECK( 'zero H' )
   CALL SOLVE_SMALL( 'NONE', null_r, '', null_r, C_t, 1.0_rp_, status_s, X_s )
   CALL CHECK( 'none H' )

!  S = 4 I given as a scaled identity, and as a diagonal

   CALL SOLVE_SMALL( 'COORDINATE', H_val, 'DIAGONAL',                          &
                     [ 4.0_rp_, 4.0_rp_, 4.0_rp_ ], C_b / 2.0_rp_, 1.0_rp_,    &
                     status_ref, X_ref )
   CALL SOLVE_SMALL( 'COORDINATE', H_val, 'SCALED_IDENTITY', [ 4.0_rp_ ],      &
                     C_b / 2.0_rp_, 1.0_rp_, status_s, X_s )
   CALL CHECK( 'scaled identity S' )

!  S = I given as an identity without S_val, and no S

   CALL SOLVE_SMALL( 'COORDINATE', H_val, '', null_r, C_b, 1.0_rp_,            &
                     status_ref, X_ref )
   CALL SOLVE_SMALL( 'COORDINATE', H_val, 'IDENTITY', null_r, C_b, 1.0_rp_,    &
                     status_s, X_s )
   CALL CHECK( 'identity S' )

!  an interior solution, for the positive definite H + 3 I and radius 2,
!  reports ||x|| and no iterations

   CALL SOLVE_SMALL( 'COORDINATE', H_val + [ 3.0_rp_, 3.0_rp_, 3.0_rp_,        &
                     0.0_rp_ ], '', null_r, C_b, 2.0_rp_, status_s, X_s )
   IF ( status_s /= 0 .OR. iter /= 0 .OR.                                      &
        ABS( x_norm - SQRT( DOT_PRODUCT( X_s, X_s ) ) ) > tol ) THEN
     WRITE( 6, "( ' interior: failed, status = ', I0, ', iter = ', I0,         &
    &  ', ||x|| = ', ES12.4 )" ) status_s, iter, x_norm
     failures = failures + 1
   ELSE
     WRITE( 6, "( ' interior: ok' )" )
   END IF
   IF ( failures > 0 ) STOP 1

   STOP

   CONTAINS

     SUBROUTINE SOLVE_SMALL( h_type, H_v, s_type, S_v, C_v, par, status, X_v )

!  solve the problem with H of type h_type and values H_v (using the indices
!  above for 'COORDINATE'), and with no S (s_type = ''), or S of type s_type
!  and values S_v (none for 'IDENTITY'), for the radius par, returning the
!  exit status and solution, and setting x_norm and iter

     CHARACTER ( LEN = * ), INTENT( IN ) :: h_type, s_type
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_v, S_v, C_v
     REAL ( KIND = rp_ ), INTENT( IN ) :: par
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( n ), INTENT( OUT ) :: X_v
     TYPE ( TREK_full_data_type ) :: data2
     TYPE ( TREK_control_type ) :: control2
     TYPE ( TREK_inform_type ) :: inform2

     X_v = 0.0_rp_ ; x_norm = - 1.0_rp_ ; iter = - 1
     CALL TREK_initialize( data2, control2, inform2 )
     CALL WHICH_sls( control2 )
     IF ( h_type == 'COORDINATE' ) THEN
       CALL TREK_import( control2, data2, status, n, h_type,                   &
                         h_ne, H_row, H_col, null_ )
     ELSE
       CALL TREK_import( control2, data2, status, n, h_type,                   &
                         SIZE( H_v, KIND = ip_ ), null_, null_, null_ )
     END IF
     IF ( status == 0 .AND. s_type /= '' ) THEN
       CALL TREK_S_import( data2, status, s_type,                              &
                           SIZE( S_v, KIND = ip_ ), null_, null_, null_ )
     END IF
     IF ( status == 0 ) THEN
       IF ( s_type == '' .OR. s_type == 'IDENTITY' ) THEN
         CALL TREK_solve_problem( data2, status, H_v, C_v, par, X_v )
       ELSE
         CALL TREK_solve_problem( data2, status, H_v, C_v, par, X_v,           &
                                  S_val = S_v )
       END IF
       CALL TREK_information( data2, inform2, status )
       status = inform2%status
       x_norm = inform2%x_norm ; iter = inform2%iter
     END IF
     CALL TREK_terminate( data2, control2, inform2 )
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
     TYPE ( TREK_control_type ), INTENT( INOUT ) :: control
#include "galahad_sls_defaults_ls.h"
     control%linear_solver = definite_linear_solver
     control%linear_solver_for_S = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_TREK_interface_test
