! THIS VERSION: GALAHAD 5,2 - 2025-05-02 AT 09:45 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_AGD_interface_test
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_AGD_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( AGD_control_type ) :: control
   TYPE ( AGD_inform_type ) :: inform
   TYPE ( AGD_full_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER ( KIND = ip_ ) :: n, ne
   INTEGER ( KIND = ip_ ) :: i, s, data_storage_type, status, eval_status
   LOGICAL :: alive
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   REAL ( KIND = rp_ ) :: dum, f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, G, U, V
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   CHARACTER ( len = 1 ) :: st

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

! start problem data

   n = 3 ; ne = 5 ! dimensions
   ALLOCATE( X( n ), G( n ) )
   ALLOCATE( H_row( ne ), H_col( ne ), H_ptr( n + 1 ) )
   ALLOCATE( H_val( ne ), H_dense( n * ( n + 1 ) / 2 ), H_diag( n ) )
   H_row = (/ 1, 2, 3, 3, 3 /) ! Hessian H
   H_col = (/ 1, 2, 1, 2, 3 /) ! NB lower triangle
   H_ptr = (/ 1, 2, 3, 6 /)    ! row pointers
! problem data complete
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   DO data_storage_type = 1, 5
     CALL AGD_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_rp_  ! start from 1.5
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'C'
       CALL AGD_import( control, data, status, n,                              &
                        'coordinate', ne, H_row, H_col, H_ptr )
       CALL AGD_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD )
     CASE ( 2 ) ! sparse by rows
       st = 'R'
       CALL AGD_import( control, data, status, n,                              &
                        'sparse_by_rows', ne, H_row, H_col, H_ptr )
       CALL AGD_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD )
     CASE ( 3 ) ! dense
       st = 'D'
       CALL AGD_import( control, data, status, n,                              &
                        'dense', ne, H_row, H_col, H_ptr )
       CALL AGD_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD )
     CASE ( 4 ) ! diagonal
       st = 'I'
       CALL AGD_import( control, data, status, n,                              &
                        'diagonal', ne, H_row, H_col, H_ptr )
       CALL AGD_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN_diag, GRAD_diag )
     CASE ( 5 ) ! access by products
       st = 'P'
       CALL AGD_import( control, data, status, n,                              &
                        'absent', ne, H_row, H_col, H_ptr )
       CALL AGD_solve_without_mat( data, userdata, status, X, G,               &
                                   FUN, GRAD )
     END SELECT
     CALL AGD_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A1, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A1, ': AGD_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL AGD_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "( /, ' tests reverse-communication options ', / )" )

   f = 0.0_rp_
   ALLOCATE( U( n ), V( n ) ) ! reverse-communication input/output
   DO data_storage_type = 1, 5
     CALL AGD_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_rp_  ! start from 1.5
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'C'
       CALL AGD_import( control, data, status, n,                              &
                        'coordinate', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL AGD_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 2 ) ! sparse by rows
       st = 'R'
       CALL AGD_import( control, data, status, n,                              &
                        'sparse_by_rows', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL AGD_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 3 ) ! dense
       st = 'D'
       CALL AGD_import( control, data, status, n,                              &
                        'dense', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL AGD_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 4 ) ! diagonal
       st = 'I'
       CALL AGD_import( control, data, status, n,                              &
                        'diagonal', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL AGD_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN_diag( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD_diag( eval_status, X, userdata, G )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 5 ) ! access by products
       st = 'P'
       CALL AGD_import( control, data, status, n,                              &
                        'absent', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL AGD_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, f, G )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     END SELECT
     CALL AGD_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A1, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A1, ': AGD_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL AGD_terminate( data, control, inform )  ! delete internal workspace
   END DO

   DEALLOCATE( X, G )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, userdata%real )

CONTAINS

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), INTENT( OUT ) :: f
   REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + COS( X( 1 ) )
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   G( 1 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) - SIN( X( 1 ) )
   G( 2 ) = 2.0_rp_ * ( X( 2 ) + X( 3 ) )
   G( 3 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +               &
            2.0_rp_ * ( X( 2 ) + X( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE GRAD

   END PROGRAM GALAHAD_AGD_interface_test
