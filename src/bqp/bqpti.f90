! THIS VERSION: GALAHAD 4.0 - 2022-02-22 AT 11:30 GMT.
   PROGRAM GALAHAD_BQP_interface_test
   USE GALAHAD_BQP_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( BQP_control_type ) :: control
   TYPE ( BQP_inform_type ) :: inform
   TYPE ( BQP_full_data_type ) :: data
   INTEGER :: n, H_ne
   INTEGER :: i, j, l, data_storage_type, status
   REAL ( KIND = wp ) :: f
   INTEGER, DIMENSION( 0 ) :: null
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER :: nz_v_start, nz_v_end, nz_prod_end
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_prod, MASK
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD
   CHARACTER ( len = 2 ) :: st

! set up problem data
!  H = tridiag(2,1), g = 2 e_1

   n = 10 ; H_ne = 2 * n - 1
   f = 1.0_wp
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ), X_stat( n ) )
   G( 1 ) = 2.0_wp ; G( 2 : n ) = 0.0_wp    ! objective gradient
   X_l( 1 ) = - 1.0_wp ; X_l( 2 : n ) = - infinity ! variable lower bound
   X_u( 1 ) = 1.0_wp ; X_u( 2 ) = infinity ; X_u( 3 : n ) = 2.0_wp ! upper
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   l = 1 ; H_ptr( 1 ) = 1
   H_row( l ) = 1 ;  H_col( l ) = 1 ; H_val( l ) = 2.0_wp
   DO i = 2, n
     l = l + 1 ; H_ptr( i ) = l
     H_row( l ) = i ; H_col( l ) = i - 1 ; H_val( l ) = 1.0_wp
     l = l + 1
     H_row( l ) = i ; H_col( l ) = i ; H_val( l ) = 2.0_wp
   END DO
   H_ptr( n + 1 ) = l + 1
   l = 0
   ALLOCATE( H_dense( n * ( n + 1 ) / 2 ), H_diag( n ) )
   DO i = 1, n
     H_diag( i ) = 2.0_wp
     DO j = 1, i
       l = l + 1
       IF ( j < i - 1 ) THEN
         H_dense( l ) = 0.0_wp
       ELSE IF ( j == i - 1 ) THEN
         H_dense( l ) = 1.0_wp
       ELSE
         H_dense( l ) = 2.0_wp
       END IF
     END DO
   END DO

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of Hessian storage formats', / )" )

   DO data_storage_type = 1, 4
     CALL BQP_initialize( data, control, inform )
!    control%print_level = 1
     X = 0.0_wp ; Z = 0.0_wp ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL BQP_import( control, data, status, n,                              &
                        'coordinate', H_ne, H_row, H_col, null )
       CALL BQP_solve_given_h( data, status, H_val, G, f,                      &
                               X_l, X_u, X, Z, X_stat )
       ! write(6,"( ' x = ', 5ES12.4, /, 5X, 5ES12.4 )" ) X
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL BQP_import( control, data, status, n,                              &
                        'sparse_by_rows', H_ne, null, H_col, H_ptr )
       CALL BQP_solve_given_h( data, status, H_val, G, f,                      &
                               X_l, X_u, X, Z, X_stat )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL BQP_import( control, data, status, n,                              &
                        'dense', H_ne, H_row, null, null )
       CALL BQP_solve_given_h( data, status, H_dense, G, f,                    &
                               X_l, X_u, X, Z, X_stat )
     CASE ( 4 ) ! diagonal
       st = ' L'
       CALL BQP_import( control, data, status, n,                              &
                        'diagonal', n, null, null, null )
       CALL BQP_solve_given_h( data, status, H_diag, G, f,                     &
                               X_l, X_u, X, Z, X_stat )
     END SELECT
     CALL BQP_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': BQP_solve exit status = ', I0 ) " ) st, inform%status
     END IF
     CALL BQP_terminate( data, control, inform )  ! delete internal workspace
   END DO

   WRITE( 6, "( /, ' test of reverse-communication interface', / )" )

   ALLOCATE( NZ_v( n ), NZ_prod( n ), V( n ), PROD( n ), MASK( n ) )
   CALL BQP_initialize( data, control, inform )
!  control%print_level = 1
!  control%maxit = 2
!  control%exact_arcsearch = .FALSE.
   X = 0.0_wp ; Z = 0.0_wp ! start from zero

   MASK = 0
   st = ' I'
   CALL BQP_import_without_h( control, data, status, n )
   status = 1
   DO
     CALL BQP_solve_reverse_h_prod( data, status, G, f, X_l, X_u, X, Z,        &
                                    X_stat, V, PROD, NZ_v, nz_v_start,         &
                                    nz_v_end, NZ_prod, nz_prod_end )
     SELECT CASE( status )
     CASE ( : 0 )
       EXIT
     CASE ( 2 )
       PROD( 1 ) = 2.0_wp * V( 1 ) + V( 2 )
       DO i = 2, n - 1
         PROD( i ) = 2.0_wp * V( i ) + V( i - 1 ) + V( i + 1 )
       END DO
       PROD( n ) = 2.0_wp * V( n ) + V( n - 1 )
     CASE ( 3 )
       PROD( : n ) = 0.0_wp
       DO l = nz_v_start, nz_v_end
         i = NZ_v( l )
         IF ( i > 1 ) PROD( i - 1 ) = PROD( i - 1 ) + V( i )
         PROD( i ) = PROD( i ) + 2.0_wp * V( i )
         IF ( i < n ) PROD( i + 1 ) = PROD( i + 1 ) + V( i )
       END DO
     CASE ( 4 )
       nz_prod_end = 0
       DO l = nz_v_start, nz_v_end
         i = NZ_v( l )
         IF ( i > 1 ) THEN
           IF ( MASK( i - 1 ) == 0 ) THEN
             MASK( i - 1 ) = 1
             nz_prod_end = nz_prod_end + 1
             NZ_prod( nz_prod_end ) = i - 1
             PROD( i - 1 ) = V( i )
           ELSE
             PROD( i - 1 ) = PROD( i - 1 ) + V( i )
           END IF
         END IF
         IF ( MASK( i ) == 0 ) THEN
           MASK( i ) = 1
           nz_prod_end = nz_prod_end + 1
           NZ_prod( nz_prod_end ) = i
           PROD( i ) = 2.0_wp * V( i )
         ELSE
           PROD( i ) = PROD( i ) + 2.0_wp * V( i )
         END IF
         IF ( i < n ) THEN
           IF ( MASK( i + 1 ) == 0 ) THEN
             MASK( i + 1 ) = 1
             nz_prod_end = nz_prod_end + 1
             NZ_prod( nz_prod_end ) = i + 1
             PROD( i + 1 ) = PROD( i + 1 ) + V( i )
           ELSE
             PROD( i + 1 ) = PROD( i + 1 ) + V( i )
           END IF
         END IF
       END DO
       MASK( NZ_prod( : nz_prod_end ) ) = 0
     END SELECT
   END DO
!write(6,"( ' x ', 5ES12.4, /, 3X, 5ES12.4 )" ) x
   CALL BQP_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',      &
   &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A2, ': BQP_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   CALL BQP_terminate( data, control, inform )  ! delete internal workspace

   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag )
   DEALLOCATE( X, G, Z, X_l, X_u, X_stat, NZ_v, NZ_prod, V, PROD, MASK )

   END PROGRAM GALAHAD_BQP_interface_test
