! THIS VERSION: GALAHAD 5.4 - 2025-11-14 AT 13:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_TREK_interface_test
   USE GALAHAD_KINDS_precision, ONLY: rp_, ip_
   USE GALAHAD_TREK_precision, ONLY: TREK_full_data_type, TREK_control_type,   &
         TREK_inform_type, TREK_initialize, TREK_import, TREK_import_S,        &
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
         IF ( use_s ) CALL TREK_import_S( data, status, 'COORDINATE',          &
                                          S_ne, S_row, S_col, null_ )
       CASE ( 2 ) ! sparse row-wise storage
         st = 'R'
         CALL TREK_import( control, data, status, n, 'SPARSE_BY_ROWS',         &
                           H_ne, null_, H_col, H_ptr )
         IF ( use_s ) CALL TREK_import_S( data, status, 'SPARSE_BY_ROWS',      &
                                          S_ne, null_, S_col, S_ptr )
       CASE ( 3 ) ! dense storage
         st = 'D'
         CALL TREK_import( control, data, status, n, 'DENSE',                  &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL TREK_import_S( data, status, 'DENSE',               &
                                          S_ne, null_, null_, null_ )
       CASE ( 4 ) ! diagonal H, S
         st = 'G'
         CALL TREK_import( control, data, status, n, 'DIAGONAL',               &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL TREK_import_S( data, status, 'DIAGONAL',            &
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

   STOP

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( TREK_control_type ), INTENT( INOUT ) :: control
#include "galahad_sls_defaults_dls.h"
     control%linear_solver = definite_linear_solver
     control%linear_solver_for_S = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_TREK_interface_test
