! THIS VERSION: GALAHAD 5.4 - 2025-11-22 AT 09:20 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_NREK_interface_test
   USE GALAHAD_KINDS_precision, ONLY: rp_, ip_
   USE GALAHAD_NREK_precision, ONLY: NREK_full_data_type, NREK_control_type,   &
         NREK_inform_type, NREK_initialize, NREK_import, NREK_S_import,        &
         NREK_solve_problem, NREK_information, NREK_terminate
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ) :: status, storage_type, w_is, s_is
   LOGICAL :: use_s
   TYPE ( NREK_full_data_type ) :: data
   TYPE ( NREK_control_type ) :: control
   TYPE ( NREK_inform_type ) :: inform
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( len = 2 ) :: sw
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 1
   INTEGER ( KIND = ip_ ), PARAMETER :: h_ne = 4, s_ne = 3, a_ne = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: h_dense_ne = n * ( n + 1 ) / 2
   INTEGER ( KIND = ip_ ), PARAMETER :: s_dense_ne = h_dense_ne
   REAL ( KIND = rp_ ) :: power = 3.0_rp_
   REAL ( KIND = rp_ ) :: weight
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
       CALL NREK_initialize( data, control, inform )
       CALL WHICH_sls( control )
       SELECT CASE ( storage_type ) ! import the data
       CASE ( 1 ) ! sparse co-ordinate storage
         st = 'C'
         CALL NREK_import( control, data, status, n, 'COORDINATE',             &
                           H_ne, H_row, H_col, null_ )
         IF ( use_s ) CALL NREK_S_import( data, status, 'COORDINATE',          &
                                          S_ne, S_row, S_col, null_ )
       CASE ( 2 ) ! sparse row-wise storage
         st = 'R'
         CALL NREK_import( control, data, status, n, 'SPARSE_BY_ROWS',         &
                           H_ne, null_, H_col, H_ptr )
         IF ( use_s ) CALL NREK_S_import( data, status, 'SPARSE_BY_ROWS',      &
                                          S_ne, null_, S_col, S_ptr )
       CASE ( 3 ) ! dense storage
         st = 'D'
         CALL NREK_import( control, data, status, n, 'DENSE',                  &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL NREK_S_import( data, status, 'DENSE',               &
                                          S_ne, null_, null_, null_ )
       CASE ( 4 ) ! diagonal H, S
         st = 'G'
         CALL NREK_import( control, data, status, n, 'DIAGONAL',               &
                           H_ne, null_, null_, null_ )
         IF ( use_s ) CALL NREK_S_import( data, status, 'DIAGONAL',            &
                                          S_ne, null_, null_, null_ )
       END SELECT
       DO w_is = 1, 2
         control%new_weight = w_is == 2
         IF ( control%new_weight ) THEN   ! use a larger weight
           weight = inform%next_weight
           IF ( use_s ) THEN
             sw = 'S+'
           ELSE
             sw = '+ '
           END IF
         ELSE   ! use the original weight
           weight = 1.0_rp_
           IF ( use_s ) THEN
             sw = 'S '
           ELSE
             sw = '  '
           END IF
         END IF
         SELECT CASE ( storage_type ) ! solve the problem
         CASE ( 1 ) ! sparse co-ordinate storage
           IF ( use_s ) THEN
             CALL NREK_solve_problem( data, status, H_val, C, power, weight,   &
                                      X, S_val = S_val )
           ELSE
             CALL NREK_solve_problem( data, status, H_val, C, power, weight, X )
           END IF
         CASE ( 2 ) ! sparse row-wise storage
           IF ( use_s ) THEN
             CALL NREK_solve_problem( data, status, H_val, C, power, weight,   &
                                      X, S_val = S_val )
           ELSE
             CALL NREK_solve_problem( data, status, H_val, C, power, weight, X )
           END IF
         CASE ( 3 ) ! dense storage
           IF ( use_s ) THEN
             CALL NREK_solve_problem( data, status, H_dense_val, C, power,     &
                                      weight, X, S_val = S_dense_val )
           ELSE
             CALL NREK_solve_problem( data, status, H_dense_val, C, power,     &
                                      weight, X )
           END IF
         CASE ( 4 ) ! diagonal H, S
           st = 'G'
           IF ( use_s ) THEN
             CALL NREK_solve_problem( data, status, H_diag_val, C, power,      &
                                      weight, X, S_val = S_diag_val )
           ELSE
             CALL NREK_solve_problem( data, status, H_diag_val, C, power,      &
                                      weight, X )
           END IF
         END SELECT
         CALL NREK_information( data, inform, status )
         WRITE( 6, "( ' format ', A1, A2, ':',                                 &
        &  ' NREK_solve_problem exit status = ', I4, ', f = ', f0.2 )" )       &
          st, sw, status, inform%obj
       END DO
     END DO
     CALL NREK_terminate( data, control, inform ) !  delete internal workspace
   END DO

   STOP

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( NREK_control_type ), INTENT( INOUT ) :: control
#include "galahad_sls_defaults_dls.h"
     control%linear_solver = definite_linear_solver
     control%linear_solver_for_S = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_NREK_interface_test
