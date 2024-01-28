! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_PRESOLVE_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_PRESOLVE_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( PRESOLVE_control_type ) :: control
   TYPE ( PRESOLVE_inform_type ) :: inform
   TYPE ( PRESOLVE_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, H_ne, A_ne_dense, H_ne_dense
   INTEGER ( KIND = ip_ ) :: n_trans, m_trans, H_ne_trans, A_ne_trans
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   REAL ( KIND = rp_ ) :: f, f_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense, H_zero
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_l_trans, X_u_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_l_trans, C_u_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_l_trans, Y_u_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z_l_trans, Z_u_trans
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_col_trans
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val_trans
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_col_trans
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_trans, C_trans
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_trans, Z_trans

   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 6; m = 5; h_ne = 1; a_ne = 8
   f = 1.0_rp_
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ) )
   ALLOCATE( C( m ), Y( m ), C_l( m ), C_u( m ) )
   G = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   C_l = (/  0.0_rp_, 0.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ /)
   C_u = (/  1.0_rp_, 1.0_rp_, 3.0_rp_, 3.0_rp_, 3.0_rp_ /)
   X_l = (/ -3.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)
   X_u = (/  3.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_val = (/ 1.0_rp_ /)
   H_row = (/ 1 /)
   H_col = (/ 1 /)
   H_ptr = (/ 1, 2, 2, 2, 2, 2, 2 /)
   ALLOCATE( A_val( A_ne ), A_row( A_ne ), A_col( A_ne ), A_ptr( m + 1 ) )
   A_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                              &
              1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   A_row = (/ 3, 3, 3, 4, 4, 5, 5, 5 /)
   A_col = (/ 3, 4, 5, 3, 6, 4, 5, 6 /)
   A_ptr = (/ 1, 1, 1, 4, 6, 9 /)
   A_ne_dense = m * n ; H_ne_dense = n * ( n + 1 ) / 2
   ALLOCATE( A_dense( A_ne_dense ), H_dense( H_ne_dense ) )
   H_dense = (/ 1.0_rp_,                                                       &
                0.0_rp_, 0.0_rp_,                                              &
                0.0_rp_, 0.0_rp_, 0.0_rp_,                                     &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,                            &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,                   &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_  /)
   A_dense = (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,          &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,          &
                0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_,          &
                0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_,          &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_  /)
   ALLOCATE( H_diag( n ), H_zero( 0 ) )
   H_diag = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of qp storage formats', / )" )

!  input problem data and perform presolve to deduce transformed dimensions

   DO data_storage_type = 1, 7
!    WRITE( 6, "( /, ' storage type = ', I0 )" ) data_storage_type
     CALL PRESOLVE_initialize( data, control, inform )
     X = 0.0_rp_ ; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
!    WRITE( 6, * ) ' before  ', n, m, H_ne, A_ne
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
!      control%print_level = 3
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'coordinate', H_ne, H_row, H_col, H_ptr, H_val,        &
                        G, f,                                                  &
                        'coordinate', A_ne, A_row, A_col, A_ptr, A_val,        &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )

     CASE ( 2 ) ! sparse by rows
       st = ' R'
!     control%print_level = 3
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr, H_val,    &
                        G, f,                                                  &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr, A_val,    &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     CASE ( 3 ) ! dense
       st = ' D'
!      control%print_level = 6
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'dense', H_ne_dense, H_row, H_col, H_ptr, H_dense,     &
                        G, f,                                                  &
                        'dense', A_ne_dense, A_row, A_col, A_ptr, A_dense,     &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'diagonal', H_ne, H_row, H_col, H_ptr, H_diag,         &
                        G, f,                                                  &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr, A_val,    &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     CASE ( 5 ) ! scaled identity
       st = ' S'
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'scaled_identity', H_ne, H_row, H_col, H_ptr, H_diag,  &
                        G, f,                                                  &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr, A_val,    &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     CASE ( 6 ) ! identity
       st = ' I'
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'identity', H_ne, H_row, H_col, H_ptr, H_zero,         &
                        G, f,                                                  &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr, A_val,    &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     CASE ( 7 ) ! zero
       st = ' Z'
!      control%print_level = 3
       CALL PRESOLVE_import_problem( control, data, status, n, m,              &
                        'zero', H_ne, H_row, H_col, H_ptr, H_zero,             &
                        G, f,                                                  &
                        'sparse_by_rows', A_ne, A_row, A_col, A_ptr, A_val,    &
                        C_l, C_u, X_l, X_u,                                    &
                        n_trans, m_trans, H_ne_trans, A_ne_trans )
     END SELECT
!    WRITE( 6, * ) ' import status', status
!    IF ( status /= 0 ) &
!      WRITE( 6, "( A, /, A, /, A )" ) inform%message( 1 ),                    &
!        inform%message( 2 ),inform%message( 3 )
!    WRITE( 6, * ) ' after  ', n_trans, m_trans, H_ne_trans, A_ne_trans

     ALLOCATE( G_trans( n_trans ), X_l_trans( n_trans ), X_u_trans( n_trans ) )
     ALLOCATE( C_l_trans( m_trans ), C_u_trans( m_trans ) )
     ALLOCATE( Y_l_trans( m_trans ), Y_u_trans( m_trans ) )
     ALLOCATE( Z_l_trans( n_trans ), Z_u_trans( n_trans ) )
     ALLOCATE( A_val_trans( A_ne_trans ), A_col_trans( A_ne_trans ) )
     ALLOCATE( A_ptr_trans( m_trans + 1 ) )
     ALLOCATE( H_val_trans( H_ne_trans ), H_col_trans( H_ne_trans ) )
     ALLOCATE( H_ptr_trans( n_trans + 1 ) )

!  recover transformed problem

     CALL PRESOLVE_transform_problem( data, status,                            &
                                      H_col_trans, H_ptr_trans, H_val_trans,   &
                                      G_trans, f_trans,                        &
                                      A_col_trans, A_ptr_trans, A_val_trans,   &
                                      C_l_trans, C_u_trans,                    &
                                      X_l_trans, X_u_trans,                    &
                                      Y_l_trans, Y_u_trans,                    &
                                      Z_l_trans, Z_u_trans )
!    WRITE( 6, * ) ' transform status', status

!  solve transformed problem using a suitable QP routine - solution is null

     ALLOCATE( X_trans( n_trans ), C_trans( m_trans ) )
     ALLOCATE( Y_trans( m_trans ), Z_trans( n_trans ) )

!    CALL QP_solver( trans_proble, trans_solution, ... ) giving

     X_trans( : n_trans ) = 1.0_rp_
     C_trans( : m_trans ) = 1.0_rp_
     Y_trans( : m_trans ) = 1.0_rp_
     Z_trans( : n_trans ) = 1.0_rp_

     DEALLOCATE( G_trans, X_l_trans, X_u_trans, C_l_trans, C_u_trans )
     DEALLOCATE( Y_l_trans, Y_u_trans, Z_l_trans, Z_u_trans )
     DEALLOCATE( A_val_trans, A_col_trans, A_ptr_trans )
     DEALLOCATE( H_val_trans, H_col_trans, H_ptr_trans )

!  recover solution to original problem

     CALL PRESOLVE_restore_solution( data, status, X_trans, C_trans, Y_trans,  &
                                     Z_trans, X, C, Y, Z )

     DEALLOCATE( X_trans, C_trans, Y_trans, Z_trans )

     CALL PRESOLVE_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' transformations, n, m = ', 2I2,             &
      &  ', status = ', I0 )" ) st, inform%nbr_transforms, n_trans, m_trans,   &
        inform%status
     ELSE
       WRITE( 6, "( A2, ': PRESOLVE_solve exit status = ', I0 ) " )            &
         st, inform%status
     END IF
     CALL PRESOLVE_terminate( data, control, inform )! delete internal workspace
   END DO
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, H_zero )
   DEALLOCATE( X, C, G, Y, Z, X_l, X_u, C_l, C_u )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )

   END PROGRAM GALAHAD_PRESOLVE_interface_test
