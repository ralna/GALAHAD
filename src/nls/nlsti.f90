! THIS VERSION: GALAHAD 3.3 - 26/07/2021 AT 13:50 GMT.
   PROGRAM GALAHAD_NLS_interface_test
   USE GALAHAD_NLS_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( NLS_control_type ) :: control
   TYPE ( NLS_inform_type ) :: inform
   TYPE ( NLS_full_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: n, m, J_ne, H_ne, P_ne
   INTEGER :: i, s, data_storage_type, model, status, eval_status
   LOGICAL :: alive, transpose
   REAL ( KIND = wp ), PARAMETER :: p = 1.0_wp
   REAL ( KIND = wp ) :: dum, f
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, G, C, Y, W, U, V
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: J_row, J_col, J_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: J_val, J_dense
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: P_row, P_col, P_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P_val, P_dense
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 2 ;  m = 3 ; J_ne = 5 ; H_ne = 2 ; P_ne = 2
   ALLOCATE( X( n ), G( n ), C( m ), Y( m ), W( m ), H_diag( n ) )
   W = 1.0_wp
   ALLOCATE( J_val( J_ne ), J_row( J_ne ), J_col( J_ne ), J_ptr( m + 1 ) )
   J_row = (/ 1, 2, 2, 3, 3 /)
   J_col = (/ 1, 1, 2, 1, 2 /)
   J_ptr = (/ 1, 2, 4, 6 /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_row = (/ 1, 2 /) 
   H_col = (/ 1, 2 /)
   H_ptr = (/ 1, 2, 3 /)
   ALLOCATE( P_val( P_ne ), P_row( P_ne ), P_ptr( m + 1 ) )
   P_row = (/ 1, 2 /)
   P_ptr = (/ 1, 2, 3, 3 /)
   ALLOCATE( J_dense( m * n ), H_dense( n * ( n + 1 ) / 2 ), P_dense( m * n ) )

   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

! problem data complete   

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats', / )" )

   DO data_storage_type = 1, 5
     CALL NLS_initialize( data, control, inform )
     control%jacobian_available = 2 ; control%hessian_available = 2
!    control%print_level = 1
     control%model = 6
     X = 1.5_wp  ! start from 1.5
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL NLS_import( control, data, status, n, m,                           &
                        'coordinate', J_ne, J_row, J_col, J_ptr,               &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS, RHESSPRODS )
     CASE ( 2 ) ! sparse by rows  
       st = ' R'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS, RHESSPRODS )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL NLS_import( control, data, status, n, m,                           &
                        'dense', J_ne, J_row, J_col, J_ptr,                    &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense_by_columns', P_ne, P_row, P_col, P_ptr,         &
                         W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC_dense, HESS_dense, RHESSPRODS_dense  )
     CASE ( 4 ) ! diagonal
       st = ' I'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS_diag, RHESSPRODS )
     CASE ( 5 ) ! access by products
       st = ' P'
       control%jacobian_available = 1 ; control%hessian_available = 1
!       control%print_level = 5 ; control%maxit = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
!                       'absent', P_ne, P_row, P_col, P_ptr,                   &
                         W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD, RHESSPRODS )
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

!  ===============================================================
!  basic test of various storage formats via reverse communication
!  ===============================================================

   WRITE( 6, "( /, ' tests reverse-communication options', / )" )

   ALLOCATE( U( MAX( m, n ) ), V( MAX( m, n ) ) ) ! reverse-communication i/o
   DO data_storage_type = 1, 5
     CALL NLS_initialize( data, control, inform )
!    control%print_level = 1
     control%model = 6
     control%jacobian_available = 2 ; control%hessian_available = 2
     X = 1.5_wp  ! start from 1.5
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL NLS_import( control, data, status, n, m,                           &
                        'coordinate', J_ne, J_row, J_col, J_ptr,               &
                        'coordinate', H_ne, H_row, H_col, H_ptr,               &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 2 ) ! sparse by rows  
       st = ' R'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 3 ) ! dense
       st = ' D'
       CALL NLS_import( control, data, status, n, m,                           &
                        'dense', J_ne, J_row, J_col, J_ptr,                    &
                        'dense', H_ne, H_row, H_col, H_ptr,                    &
                        'dense_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_dense, Y, H_dense,        &
                                          V, P_dense )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC_dense( eval_status, X, userdata, J_dense )
         CASE ( 4 ) ! evaluate H
           CALL HESS_dense( eval_status, X, Y, userdata, H_dense ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS_dense( eval_status, X, V, userdata, P_dense )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 4 ) ! diagonal
       st = ' I'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'diagonal', H_ne, H_row, H_col, H_ptr,                 &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_diag, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS_diag( eval_status, X, Y, userdata, H_diag ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 5 ) ! access by products
       st = ' P'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,        &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V,         &
                                             Y, P_val )
                                             
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

!  =========================
!  basic test of models used
!  =========================

   WRITE( 6, "( /, ' basic tests of models used, direct access', / )" )

   DO model = 3, 8
     CALL NLS_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_wp  ! start from 1.5
     control%model = model
     SELECT CASE ( model )
     CASE ( 3 ) ! Gauss-Newton model
       st = ' 3'
!      control%print_level = 5
!      control%maxit = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G, RES, JAC )
     CASE ( 4 ) ! Newton model
       st = ' 4'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G, RES,          &
                                JAC, HESS )
     CASE ( 5 ) ! Gauss-Newton to Newton model
       st = ' 5'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G, RES,          &
                                JAC, HESS )
     CASE ( 6 ) ! Tensor-Newton model using Gaus-Newton solve
       st = ' 6'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS, RHESSPRODS )
     CASE ( 7 ) ! Tensor-Newton model using Newton solve
       st = ' 7'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS, RHESSPRODS )
     CASE ( 8 ) ! Tensor-Newton model using Gaus-Newton to Newton solve
       st = ' 8'
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       CALL NLS_solve_with_mat( data, userdata, status, X, C, G,               &
                                RES, JAC, HESS, RHESSPRODS )
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "( /, ' basic tests of models used, direct access by products',/)")

   DO model = 3, 8
!  DO model = 1, 5
     CALL NLS_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_wp  ! start from 1.5
     control%model = model
     SELECT CASE ( model )
     CASE ( 3 ) ! Gauss-Newton model
       st = 'P3'
       control%jacobian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD )
     CASE ( 4 ) ! Newton model
       st = 'P4'
       control%jacobian_available = 1
       control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD )
     CASE ( 5 ) ! Gauss-Newton to Newton model
       st = 'P5'
       control%jacobian_available = 1
       control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD )
     CASE ( 6 ) ! Tensor-Newton model using Gaus-Newton solve
       st = 'P6'
       control%jacobian_available = 1
       control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD, RHESSPRODS )
     CASE ( 7 ) ! Tensor-Newton model using Newton solve
       st = 'P7'
       control%jacobian_available = 1
       control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD, RHESSPRODS )
     CASE ( 8 ) ! Tensor-Newton model using Gaus-Newton to Newton solve
       st = 'P8'
       control%jacobian_available = 1
       control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                        W = W )
       CALL NLS_solve_without_mat( data, userdata, status, X, C, G,            &
                                   RES, JACPROD, HESSPROD, RHESSPRODS )
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "( /, ' basic tests of models used, reverse access', / )" )

   DO model = 3, 8
     CALL NLS_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_wp  ! start from 1.5
     control%model = model
     SELECT CASE ( model )
     CASE ( 3 )  ! Gauss-Newton model
       st = ' 3'
       control%jacobian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
!                       'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
!                       'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate J
           CALL JAC( eval_status, X, userdata, J_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 4 )  ! Newton model
       st = ' 4'
       control%jacobian_available = 2 ; control%hessian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 5 )   ! Gauss-Newton to Newton model
       st = ' 5'
       control%jacobian_available = 2 ; control%hessian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 6 )    ! Tensor-Newton model using Gaus-Newton solve
       st = ' 6'
       control%jacobian_available = 2 ; control%hessian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 7 )    ! Tensor-Newton model using Newton solve
       st = ' 7'
       control%jacobian_available = 2 ; control%hessian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 8 )    ! Tensor-Newton model using Gaus-Newton to Newton solve
       st = ' 8'
       control%jacobian_available = 2 ; control%hessian_available = 2
       CALL NLS_import( control, data, status, n, m,                           &
                        'sparse_by_rows', J_ne, J_row, J_col, J_ptr,           &
                        'sparse_by_rows', H_ne, H_row, H_col, H_ptr,           &
                        'sparse_by_columns', P_ne, P_row, P_col, P_ptr,        &
                        W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_with_mat( data, status, eval_status,           &
                                          X, C, G, J_val, Y, H_val, V, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 3 ) ! evaluate g
           CALL JAC( eval_status, X, userdata, J_val )
         CASE ( 4 ) ! evaluate H
           CALL HESS( eval_status, X, Y, userdata, H_val ) 
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "(/, ' basic tests of models used, reverse access by products',/)")

   DO model = 3, 8
     CALL NLS_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_wp  ! start from 1.5
     control%model = model
     SELECT CASE ( model )
     CASE ( 3 )  ! Gauss-Newton model
       st = 'P3'
       control%jacobian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 4 )  ! Newton model
       st = 'P4'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V, Y )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 5 )  ! Gauss-Newton to Newton model
       st = 'P5'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V, Y )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 6 )  ! tensor-Newton model with Gauss-Newton solves
       st = 'P6'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V,         &
                                             Y, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 7 )  ! tensor-Newton model with Newton solves
       st = 'P7'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V,         &
                                             Y, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     CASE ( 8 )  ! tensor-Newton model with Gauss-Newton to Newton solves
       st = 'P8'
       control%jacobian_available = 1 ; control%hessian_available = 1
       CALL NLS_import( control, data, status, n, m,                           &
                        'absent', J_ne, J_row, J_col, J_ptr,                   &
                        'absent', H_ne, H_row, H_col, H_ptr,                   &
                        'absent', P_ne, P_row, P_col, P_ptr,                   &
                         W = W )
       DO ! reverse-communication loop
         CALL NLS_solve_reverse_without_mat( data, status, eval_status,        &
                                             X, C, G, transpose, U, V,         &
                                             Y, P_val )
         SELECT CASE ( status )
         CASE ( 0 ) ! successful termination
           EXIT
         CASE ( : - 1 ) ! error exit
           EXIT
         CASE ( 2 ) ! evaluate f
           CALL RES( eval_status, X, userdata, C )
         CASE ( 5 ) ! evaluate u + J v or u + J' v
           CALL JACPROD( eval_status, X, userdata, transpose, U, V )
         CASE ( 6 ) ! evaluate Hessian-vector product
           CALL HESSPROD( eval_status, X, Y, userdata, U, V )
         CASE ( 7 ) ! evaluate the product with P
           CALL RHESSPRODS( eval_status, X, V, userdata, P_val )
         CASE DEFAULT
           WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )  &
             status
           EXIT
         END SELECT
       END DO
     END SELECT
     CALL NLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': NLS_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

   DEALLOCATE( X, C, G, Y, W, U, V )
   DEALLOCATE( J_val, J_row, J_col, J_ptr, J_dense )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, userdata%real )
   DEALLOCATE( P_val, P_row, P_ptr, P_dense )
!  DEALLOCATE( P_val, P_row, P_col, P_ptr, P_dense )

CONTAINS

     SUBROUTINE RES( status, X, userdata, C )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     C( 1 ) = X( 1 ) ** 2 + userdata%real( 1 )
     C( 2 ) = X( 1 ) + X( 2 ) ** 2
     C( 3 ) = X( 1 ) - X( 2 )
     status = 0
     END SUBROUTINE RES

     SUBROUTINE JAC( status, X, userdata, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     J_val( 1 ) = 2.0_wp * X( 1 )
     J_val( 2 ) = 1.0_wp
     J_val( 3 ) = 2.0_wp * X( 2 )
     J_val( 4 ) = 1.0_wp
     J_val( 5 ) = - 1.0_wp
     status = 0
     END SUBROUTINE JAC

     SUBROUTINE HESS( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_wp * Y( 1 )
     H_val( 2 ) = 2.0_wp * Y( 2 )
     status = 0
     END SUBROUTINE HESS

     SUBROUTINE JACPROD( status, X, userdata, transpose, U, V, got_j )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j
!    write(6,"( 'X in = ', 2ES12.4 )" ) X( 1 ), X( 2 )
     IF ( transpose ) THEN
       U( 1 ) = U( 1 ) + 2.0_wp * X( 1 ) * V( 1 ) + V( 2 ) + V( 3 )
       U( 2 ) = U( 2 ) + 2.0_wp * X( 2 ) * V( 2 ) - V( 3 )
     ELSE
!      write(6,"( 'U in = ', 3ES12.4 )" ) U( 1 ), U( 2 ), U( 3 )
!      write(6,"( 'V in = ', 2ES12.4 )" ) V( 1 ), V( 2 )
       U( 1 ) = U( 1 ) + 2.0_wp * X( 1 ) * V( 1 )
       U( 2 ) = U( 2 ) + V( 1 )  + 2.0_wp * X( 2 ) * V( 2 )
       U( 3 ) = U( 3 ) + V( 1 ) - V( 2 )
!      write(6,"( 'U in = ', 3ES12.4 )" ) U( 1 ), U( 2 ), U( 3 )
     END IF
     status = 0
     END SUBROUTINE JACPROD

     SUBROUTINE HESSPROD( status, X, Y, userdata, U, V, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     U( 1 ) = U( 1 ) + 2.0_wp * Y( 1 ) * V( 1 )
     U( 2 ) = U( 2 ) + 2.0_wp * Y( 2 ) * V( 2 )
     status = 0
     END SUBROUTINE HESSPROD

     SUBROUTINE RHESSPRODS( status, X, V, userdata, P_val, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     P_val( 1 ) = 2.0_wp * V( 1 )
     P_val( 2 ) = 2.0_wp * V( 2 )
     status = 0
     END SUBROUTINE RHESSPRODS

     SUBROUTINE SCALE( status, X, userdata, U, V )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
!     U( 1 ) = 0.5_wp * V( 1 )
!     U( 2 ) = 0.5_wp * V( 2 )
     U( 1 ) = V( 1 )
     U( 2 ) = V( 2 )
     status = 0
     END SUBROUTINE SCALE

     SUBROUTINE JAC_dense( status, X, userdata, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     J_val( 1 ) = 2.0_wp * X( 1 )
     J_val( 2 ) = 0.0_wp
     J_val( 3 ) = 1.0_wp
     J_val( 4 ) = 2.0_wp * X( 2 )
     J_val( 5 ) = 1.0_wp
     J_val( 6 ) = - 1.0_wp
     status = 0
     END SUBROUTINE JAC_dense

     SUBROUTINE HESS_dense( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_wp * Y( 1 )
     H_val( 2 ) = 0.0_wp
     H_val( 3 ) = 2.0_wp * Y( 2 )
     status = 0
     END SUBROUTINE HESS_dense

     SUBROUTINE HESS_diag( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_wp * Y( 1 )
     H_val( 2 ) = 2.0_wp * Y( 2 )
     status = 0
     END SUBROUTINE HESS_diag

     SUBROUTINE RHESSPRODS_dense( status, X, V, userdata, P_val, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     P_val( 1 ) = 2.0_wp * V( 1 )
     P_val( 2 ) = 0.0_wp
     P_val( 3 ) = 0.0_wp
     P_val( 4 ) = 2.0_wp * V( 2 )
     P_val( 5 ) = 0.0_wp
     P_val( 6 ) = 0.0_wp
     status = 0
     END SUBROUTINE RHESSPRODS_dense

   END PROGRAM GALAHAD_NLS_interface_test
