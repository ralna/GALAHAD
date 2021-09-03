! THIS VERSION: GALAHAD 3.3 - 10/07/2021 AT 09:45 GMT.
   PROGRAM GALAHAD_DGO_interface_test
   USE GALAHAD_DGO_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( DGO_control_type ) :: control
   TYPE ( DGO_inform_type ) :: inform
   TYPE ( DGO_full_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
!  EXTERNAL :: FUN, GRAD, HESS, HESSPROD, PREC
   INTEGER :: n, ne, nnz_v, nnz_u
   INTEGER :: i, s, status, data_storage_type, eval_status
   LOGICAL :: alive
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ) :: dum, f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, G, X_l, X_u, U, V
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_nz_v, INDEX_nz_u
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   CHARACTER ( len = 1 ) :: st

! problem data complete   

!  =====================================
!  basic test of various storage formats
!  =====================================

! start problem data

   n = 3 ; ne = 5 ! dimensions
   ALLOCATE( X( n ), X_l( n ), X_u( n ), G( n ) )
   X_l = -10.0_wp ; X_u = 0.5_wp ! search in [-10,1/2]
   ALLOCATE( H_row( ne ), H_col( ne ), H_ptr( n + 1 ) )
   ALLOCATE( H_val( ne ), H_dense( n * ( n + 1 ) / 2 ), H_diag( n ) )
   H_row = (/ 1, 2, 3, 3, 3 /) ! Hessian H
   H_col = (/ 1, 2, 1, 2, 3 /) ! NB lower triangle
   H_ptr = (/ 1, 2, 3, 6 /)    ! row pointers
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p
! problem data complete   

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

!  DO data_storage_type = 1, 0
!  DO data_storage_type = 1, 1
   DO data_storage_type = 1, 5
     CALL DGO_initialize( data, control, inform )
!    control%print_level = 1
     control%max_evals = 20000
     X = 0.0_wp  ! start from 1.0
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'C'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'coordinate', ne, H_row, H_col, H_ptr )
       CALL DGO_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD, HESS, HESSPROD, PREC )
     CASE ( 2 ) ! sparse by rows  
       st = 'R'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'sparse_by_rows', ne, H_row, H_col, H_ptr )
       CALL DGO_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD, HESS, HESSPROD, PREC )
     CASE ( 3 ) ! dense
       st = 'D'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'dense', ne, H_row, H_col, H_ptr )
       CALL DGO_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN, GRAD, HESS_dense, HESSPROD, PREC )
     CASE ( 4 ) ! diagonal
       st = 'I'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'diagonal', ne, H_row, H_col, H_ptr )
       CALL DGO_solve_with_mat( data, userdata, status, X, G,                  &
                                FUN_diag, GRAD_diag, HESS_diag, HESSPROD_diag, &
                                PREC )
     CASE ( 5 ) ! access by products
       st = 'P'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'absent', ne, H_row, H_col, H_ptr )
       CALL DGO_solve_without_mat( data, userdata, status, X, G,               &
                                   FUN, GRAD, HESSPROD, SHESSPROD, PREC )
     END SELECT
     CALL DGO_information( data, inform, status )

     IF ( inform%status == GALAHAD_ok .OR.                                     &
          inform%status == GALAHAD_error_max_iterations .OR.                   &
          inform%status == GALAHAD_error_max_evaluations ) THEN  ! Success
       WRITE( 6, "( A1, ':', I6, ' evaluations.',                              &
     &  ' Best objective value found = ', F8.2, ' status = ', I0 )" )          &
         st, inform%f_eval, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A1, ': DGO_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL DGO_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "( /, ' tests reverse-communication options ', / )" )

   ALLOCATE( U( n ), V( n ) ) ! reverse-communication input/output
   ALLOCATE( INDEX_nz_v( n ), INDEX_nz_u( n ) )
!  DO data_storage_type = 1, 1
!  DO data_storage_type = 5, 5
   DO data_storage_type = 1, 5
     CALL DGO_initialize( data, control, inform )
     control%max_evals = 20000
     X = 0.0_wp  ! start from 1.0
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = 'C'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'coordinate', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL DGO_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G, H_val, U, V )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, userdata, H_val ) 
         CASE ( 6 ) ! evaluate the product with P
           CALL PREC( eval_status, X, userdata, U, V )
         CASE ( 23 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 25 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE ( 235 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 2 ) ! sparse by rows  
       st = 'R'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'sparse_by_rows', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL DGO_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G, H_val, U, V )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, userdata, H_val ) 
         CASE ( 6 ) ! evaluate the product with P
           CALL PREC( eval_status, X, userdata, U, V )
         CASE ( 23 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 25 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE ( 235 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 3 ) ! dense
       st = 'D'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'dense', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL DGO_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G, H_dense, U, V )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 4 ) ! evaluate H
           CALL HESS_dense( eval_status, X, userdata, H_dense ) 
         CASE ( 6 ) ! evaluate the product with P
           CALL PREC( eval_status, X, userdata, U, V )
         CASE ( 23 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 25 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE ( 235 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 4 ) ! diagonal
       st = 'I'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'diagonal', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL DGO_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, f, G, H_diag, U, V )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN_diag( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD_diag( eval_status, X, userdata, G )
         CASE ( 4 ) ! evaluate H
           CALL HESS_diag( eval_status, X, userdata, H_diag ) 
         CASE ( 6 ) ! evaluate the product with P
           CALL PREC( eval_status, X, userdata, U, V )
         CASE ( 23 ) ! evaluate f and g
           CALL FUN_diag( eval_status, X, userdata, f )
           CALL GRAD_diag( eval_status, X, userdata, G )
         CASE ( 25 ) ! evaluate f and g
           CALL FUN_diag( eval_status, X, userdata, f )
           CALL HESSPROD_diag( eval_status, X, userdata, U, V )
         CASE ( 235 ) ! evaluate f and g
           CALL FUN_diag( eval_status, X, userdata, f )
           CALL GRAD_diag( eval_status, X, userdata, G )
           CALL HESSPROD_diag( eval_status, X, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 5 ) ! access by products
       st = 'P'
       CALL DGO_import( control, data, status, n, X_l, X_u,                    &
                        'absent', ne, H_row, H_col, H_ptr )
       DO ! reverse-communication loop
         CALL DGO_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, f, G, U, V,                    &
                                             INDEX_nz_v, nnz_v,                &
                                             INDEX_nz_u, nnz_u )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL FUN( eval_status, X, userdata, f )
         CASE ( 3 ) ! evaluate g
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 5 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE ( 6 ) ! evaluate the product with P
           CALL PREC( eval_status, X, userdata, U, V )
         CASE ( 7 ) ! evaluate sparse Hessian-vector product
           CALL SHESSPROD( eval_status, X, userdata, nnz_v, INDEX_nz_v, V,     &
                           nnz_u, INDEX_nz_u, U )
         CASE ( 23 ) ! evaluate f and g
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
         CASE ( 25 ) ! evaluate f and h product
           CALL FUN( eval_status, X, userdata, f )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE ( 235 ) ! evaluate f and h product
           CALL FUN( eval_status, X, userdata, f )
           CALL GRAD( eval_status, X, userdata, G )
           CALL HESSPROD( eval_status, X, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     END SELECT
     CALL DGO_information( data, inform, status )
     IF ( inform%status == GALAHAD_ok .OR.                                     &
          inform%status == GALAHAD_error_max_iterations .OR.                   &
          inform%status == GALAHAD_error_max_evaluations ) THEN  ! Success
       WRITE( 6, "( A1, ':', I6, ' evaluations.',                              &
     &  ' Best objective value found = ', F8.2, ' status = ', I0 )" )          &
         st, inform%f_eval, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A1, ': DGO_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL DGO_terminate( data, control, inform )  ! delete internal workspace
   END DO

   DEALLOCATE( X, X_l, x_u, G )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, userdata%real )

CONTAINS

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + mag * COS( freq * X( 1 ) ) +                 &
         X( 1 ) + X( 2 ) + X( 3 ) 
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   G( 1 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) )                  &
             - mag * freq * SIN( freq * X( 1 ) ) + 1.0_wp
   G( 2 ) = 2.0_wp * ( X( 2 ) + X( 3 ) ) + 1.0_wp
   G( 3 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +                &
            2.0_wp * ( X( 2 ) + X( 3 ) ) + 1.0_wp
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE HESS( status, X, userdata, H_Val ) ! Hessian of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   H_val( 1 ) = 2.0_wp - mag * freq * freq * COS( freq * X( 1 ) )
   H_val( 2 ) = 2.0_wp
   H_val( 3 ) = 2.0_wp
   H_val( 4 ) = 2.0_wp
   H_val( 5 ) = 4.0_wp
   status = 0
   RETURN
   END SUBROUTINE HESS

   SUBROUTINE HESS_dense( status, X, userdata, H_val ) ! Dense Hessian
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   H_val( 1 ) = 2.0_wp - mag * freq * freq * COS( freq * X( 1 ) )
   H_val( 2 ) = 0.0_wp
   H_val( 3 ) = 2.0_wp
   H_val( 4 ) = 2.0_wp
   H_val( 5 ) = 2.0_wp
   H_val( 6 ) = 4.0_wp
   status = 0
   RETURN
   END SUBROUTINE HESS_dense

   SUBROUTINE HESSPROD( status, X, userdata, U, V, got_h ) ! Hessian-vector prod
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   U( 1 ) = U( 1 )                                                             &
            + ( 2.0_wp - mag * freq * freq * COS( freq * X( 1 ) ) ) * V( 1 )   &
            + 2.0_wp * V( 3 )
   U( 2 ) = U( 2 ) + 2.0_wp * ( V( 2 ) + V( 3 ) )
   U( 3 ) = U( 3 ) + 2.0_wp * ( V( 1 ) + V( 2 ) + 2.0_wp * V( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE HESSPROD

   SUBROUTINE PREC( status, X, userdata, U, V ) ! apply preconditioner
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   U( 1 ) = 0.5_wp * V( 1 )
   U( 2 ) = 0.5_wp * V( 2 )
   U( 3 ) = 0.25_wp * V( 3 )
   U( 1 ) = V( 1 )
   U( 2 ) = V( 2 )
   U( 3 ) = V( 3 )
   status = 0
   RETURN
   END SUBROUTINE PREC

   SUBROUTINE SHESSPROD( status, X, userdata, nnz_v, INDEX_nz_v, V,            &
                         nnz_u, INDEX_nz_u, U, got_h ) ! sparse Hess-vect prod
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: nnz_v
   INTEGER, INTENT( OUT ) :: nnz_u
   INTEGER, INTENT( OUT ) :: status
   INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
   INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   INTEGER :: i, j
   REAL ( KIND = wp ), DIMENSION( 3 ) :: P
   LOGICAL, DIMENSION( 3 ) :: USED
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   P = 0.0_wp
   USED = .FALSE.
   DO i = 1, nnz_v 
     j = INDEX_nz_v( i )
     SELECT CASE( j )
     CASE( 1 )
       P( 1 ) = P( 1 ) + 2.0_wp * V( 1 )                                       &
         - mag * freq * freq * COS( freq * X( 1 ) ) * V( 1 )
       USED( 1 ) = .TRUE.
       P( 3 ) = P( 3 ) + 2.0_wp * V( 1 )
       USED( 3 ) = .TRUE.
     CASE( 2 )
       P( 2 ) = P( 2 ) + 2.0_wp * V( 2 )
       USED( 2 ) = .TRUE.
       P( 3 ) = P( 3 ) + 2.0_wp * V( 2 )
       USED( 3 ) = .TRUE.
     CASE( 3 )
       P( 1 ) = P( 1 ) + 2.0_wp * V( 3 )
       USED( 1 ) = .TRUE.
       P( 2 ) = P( 2 ) + 2.0_wp * V( 3 )
       USED( 2 ) = .TRUE.
       P( 3 ) = P( 3 ) + 4.0_wp * V( 3 )
       USED( 3 ) = .TRUE.
     END SELECT
   END DO
   nnz_u = 0
   DO j = 1, 3
     IF ( USED( j ) ) THEN
       U( j ) = P( j )
       nnz_u = nnz_u + 1
       INDEX_nz_u( nnz_u ) = j
     END IF
   END DO
   status = 0
   RETURN
   END SUBROUTINE SHESSPROD

   SUBROUTINE FUN_diag( status, X, userdata, f )    ! Objective function
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   f = ( X( 3 ) + userdata%real( 1 ) ) ** 2 + X( 2 ) ** 2                      &
           + mag * COS( freq * X( 1 ) ) + X( 1 ) + X( 2 ) + X( 3 ) 
   status = 0
   RETURN
   END SUBROUTINE FUN_diag

   SUBROUTINE GRAD_diag( status, X, userdata, G )   ! gradient of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   G( 1 ) = - mag * freq * SIN( freq * X( 1 ) ) + 1.0_wp
   G( 2 ) = 2.0_wp * X( 2 ) + 1.0_wp
   G( 3 ) = 2.0_wp * ( X( 3 ) + userdata%real( 1 ) ) + 1.0_wp
   status = 0
   RETURN
   END SUBROUTINE GRAD_diag

   SUBROUTINE HESS_diag( status, X, userdata, H_val ) ! Hessian of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   H_val( 1 ) = - mag * freq * freq * COS( freq * X( 1 ) )
   H_val( 2 ) = 2.0_wp
   H_val( 3 ) = 2.0_wp
   status = 0
   RETURN
   END SUBROUTINE HESS_diag

   SUBROUTINE HESSPROD_diag( status, X, userdata, U, V, got_h ) ! Hess-vect prod
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   U( 1 ) = U( 1 ) - mag * freq * freq * COS( freq * X( 1 ) ) * V( 1 )
   U( 2 ) = U( 2 ) + 2.0_wp * V( 2 )
   U( 3 ) = U( 3 ) + 2.0_wp * V( 3 )
   status = 0
   RETURN
   END SUBROUTINE HESSPROD_diag

   SUBROUTINE SHESSPROD_diag( status, X, userdata, nnz_v, INDEX_nz_v, V,       &
                              nnz_u, INDEX_nz_u, U, got_h ) ! sprse Hes-vec prod
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: nnz_v
   INTEGER, INTENT( OUT ) :: nnz_u
   INTEGER, INTENT( OUT ) :: status
   INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
   INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   INTEGER :: i, j
   REAL ( KIND = wp ), DIMENSION( 3 ) :: P
   LOGICAL, DIMENSION( 3 ) :: USED
   REAL, PARAMETER :: freq = 10.0_wp
   REAL, PARAMETER :: mag = 1000.0_wp
   P = 0.0_wp
   USED = .FALSE.
   DO i = 1, nnz_v 
     j = INDEX_nz_v( i )
     SELECT CASE( j )
     CASE( 1 )
       P( 1 ) = P( 1 ) - mag * freq * freq * COS( freq * X( 1 ) ) * V( 1 )
       USED( 1 ) = .TRUE.
     CASE( 2 )
       P( 2 ) = P( 2 ) + 2.0_wp * V( 2 )
       USED( 2 ) = .TRUE.
     CASE( 3 )
       P( 3 ) = P( 3 ) + 2.0_wp * V( 3 )
       USED( 3 ) = .TRUE.
     END SELECT
   END DO
   nnz_u = 0
   DO j = 1, 3
     IF ( USED( j ) ) THEN
       U( j ) = P( j )
       nnz_u = nnz_u + 1
       INDEX_nz_u( nnz_u ) = j
     END IF
   END DO
   status = 0
   RETURN
   END SUBROUTINE SHESSPROD_diag

   END PROGRAM GALAHAD_DGO_interface_test
