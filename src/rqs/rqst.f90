! THIS VERSION: GALAHAD 2.3 - 03/11/2008 AT 11:45 GMT.
   PROGRAM GALAHAD_RQS_test_deck
   USE GALAHAD_RQS_DOUBLE                            ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp, two = 2.0_wp
   REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
   INTEGER :: i, smt_stat, n, nn, pass, h_ne, m_ne, a_ne, data_storage_type
   INTEGER :: ia, im, ifa
   REAL ( KIND = wp ) :: f, sigma, p
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, C
   TYPE ( SMT_type ) :: H, M, A
   TYPE ( RQS_data_type ) :: data
   TYPE ( RQS_control_type ) :: control
   TYPE ( RQS_inform_type ) :: inform

   CHARACTER ( len = 1 ) :: st, afa
   CHARACTER ( len = 2 ) :: ma
   INTEGER, PARAMETER :: n_errors = 6
   INTEGER, DIMENSION( n_errors ) :: errors = (/                               &
       GALAHAD_error_restrictions,                                             &
       GALAHAD_error_restrictions,                                             &
       GALAHAD_error_restrictions,                                             &
       GALAHAD_error_preconditioner,                                           &
       GALAHAD_error_ill_conditioned,                                          &
       GALAHAD_error_max_iterations                                            &
         /)

! Initialize output unit

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
!  OPEN( UNIT = 23 )

!  =============
!  Error entries
!  =============

   n = 5
   f = 1.0_wp
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )    ! Specify co-ordinate for H
   H%ne = 2 * n - 1
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   DO i = 1, n
    H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_wp
   END DO
   DO i = 1, n - 1
    H%row( n + i ) = i + 1 ; H%col( n + i ) = i ; H%val( n + i ) = 1.0_wp
   END DO
   CALL SMT_put( M%type, 'DIAGONAL', smt_stat )        ! Specify diagonal for M
   ALLOCATE( M%val( n ) ) ; M%val = 2.0_wp
   WRITE( 6, "( /, ' ==== error exits ===== ', / )" )

! Initialize control parameters

   DO i = 1, n_errors
     pass = errors( i )
     nn = n
     sigma = one
     p = 3.0_wp
     CALL RQS_initialize( data, control, inform )
     control%definite_linear_solver = 'ma57'
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%error = 6 ; control%out = 6 ; control%print_level = 1
     IF ( pass == GALAHAD_error_restrictions ) THEN
       IF ( i == 1 ) THEN
         nn = 0
       ELSE IF ( i == 2 ) THEN
         sigma = - one
       ELSE
         p = 1.99_wp
       END IF
     ELSE IF ( pass == GALAHAD_error_preconditioner ) THEN
       M%val( 1 ) = - one
     ELSE IF ( pass == GALAHAD_error_ill_conditioned ) THEN
       M%val( 1 ) = 2.0_wp
       sigma = 0.00001_wp
       control%max_factorizations = 10
     ELSE IF ( pass == GALAHAD_error_max_iterations ) THEN
       control%max_factorizations = 1
     END IF

     ALLOCATE( X( nn ), C( nn ) )
     C = 1.0_wp

!    IF ( pass == GALAHAD_error_ill_conditioned ) THEN
!      control%error = 6 ; control%out = 6 ; control%print_level = 10
!      control%print_level = 3
!    END IF

!  Iteration to find the minimizer

     CALL RQS_solve( nn, p, sigma, f, C, H, X, data, control, inform, M = M )

     WRITE( 6, "( ' pass  ', I3, ': RQS_solve exit status = ', I6 )" )         &
            pass, inform%status
     CALL RQS_terminate( data, control, inform ) !  delete internal workspace
     DEALLOCATE( X, C )
     CALL RQS_terminate( data, control, inform ) !  delete internal workspace
   END DO
   DEALLOCATE( H%row, H%col, H%val, M%val )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

   n = 3 ; A%m = 1 ; h_ne = 4 ; m_ne = 3 ; a_ne = 3
   ALLOCATE( H%ptr( n + 1 ), M%ptr( n + 1 ), A%ptr( A%m + 1 ) )
   ALLOCATE( C( n ), X( n ) )

   f = 0.96_wp
   C = (/ 0.0_wp, 2.0_wp, 0.0_wp /)

   DO data_storage_type = -3, 0
     CALL RQS_initialize( data, control, inform )
     control%definite_linear_solver = 'ma57'
     control%error = 23 ; control%out = 23 ; control%print_level = 10
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'COORDINATE', smt_stat )
       H%row = (/ 1, 2, 3, 3 /)
       H%col = (/ 1, 2, 3, 1 /) ; H%ne = h_ne
       ALLOCATE( M%val( m_ne ), M%row( m_ne ), M%col( m_ne ) )
       IF ( ALLOCATED( M%type ) ) DEALLOCATE( M%type )
       CALL SMT_put( M%type, 'COORDINATE', smt_stat )
       M%row = (/ 1, 2, 3 /)
       M%col = (/ 1, 2, 3 /) ; M%ne = m_ne
       ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'COORDINATE', smt_stat )
       A%row = (/ 1, 1, 1 /)
       A%col = (/ 1, 2, 3 /) ; A%ne = n
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( H%val( h_ne ), H%row( 0 ), H%col( h_ne ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'SPARSE_BY_ROWS', smt_stat )
       H%col = (/ 1, 2, 3, 1 /)
       H%ptr = (/ 1, 2, 3, 5 /)
       ALLOCATE( M%val( m_ne ), M%row( 0 ), M%col( m_ne ) )
       IF ( ALLOCATED( M%type ) ) DEALLOCATE( M%type )
       CALL SMT_put( M%type, 'SPARSE_BY_ROWS', smt_stat )
       M%col = (/ 1, 2, 3 /)
       M%ptr = (/ 1, 2, 3, 4 /)
       ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', smt_stat )
       A%col = (/ 1, 2, 3 /)
       A%ptr = (/ 1, 4 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( H%val( n * ( n + 1 ) / 2 ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DENSE', smt_stat )
       ALLOCATE( M%val( n * ( n + 1 ) / 2 ), M%row( 0 ), M%col( 0 ) )
       IF ( ALLOCATED( M%type ) ) DEALLOCATE( M%type )
       CALL SMT_put( M%type, 'DENSE', smt_stat )
       ALLOCATE( A%val( n ), A%row( 0 ), A%col( 0 ) )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
       A%m = 1
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'I'
       ALLOCATE( H%val( n ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DIAGONAL', smt_stat )
       ALLOCATE( M%val( n ), M%row( 0 ), M%col( 0 ) )
       IF ( ALLOCATED( M%type ) ) DEALLOCATE( M%type )
       CALL SMT_put( M%type, 'DIAGONAL', smt_stat )
       ALLOCATE( A%val( n ), A%row( 0 ), A%col( 0 ) )
       IF ( ALLOCATED( A%type ) ) DEALLOCATE( A%type )
       CALL SMT_put( A%type, 'DENSE', smt_stat )
     END IF

!  test with new and existing data

     IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
       H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
       M%val = (/ 1.0_wp, 2.0_wp, 1.0_wp /)
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
     ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
       H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
       M%val = (/ 1.0_wp, 2.0_wp, 1.0_wp /)
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
     ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
       H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 4.0_wp, 0.0_wp, 3.0_wp /)
       M%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
     ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
       H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp /)
       M%val = (/ 1.0_wp, 2.0_wp, 1.0_wp /)
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
     END IF
     DO ia = 0, 1
       DO im = 0, 1
         DO ifa = 0, 1
           control%dense_factorization = ifa
           IF ( ifa == 0 ) THEN
             afa = 'S'
           ELSE
             afa = 'D'
           END IF
           DO i = 2, 0, -1
             control%new_h = i
             IF ( ia == 0 .AND. im == 0 ) THEN
               ma = '  '
               CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform )
             ELSE IF ( ia == 0 .AND. im == 1 ) THEN
               ma = 'M '
               CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, &
                               M = M )
             ELSE IF ( ia == 1 .AND. im == 0 ) THEN
               ma = 'A '
               CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, &
                               A = A )
             ELSE
               ma = 'MA'
               CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, &
                               M = M, A = A )
             END IF
             WRITE( 6, "( ' format ', A1, A1, I1, A2, ':',                     &
        &     ' RQS_solve exit status = ', I4 )" ) st, afa, i, ma, inform%status
!            WRITE( 6, "( ' format ', A1, A1, I1, A2, ':',                     &
!       &     ' RQS_solve exit status = ', I4, ES12.4 )" )                     &
!                         st, afa, i, ma, inform%status, inform%obj
!            WRITE( 6,"( (5ES12.4) )" ) X( : n )
           END DO
         END DO
       END DO
     END DO
     CALL RQS_terminate( data, control, inform ) !  delete internal workspace
     DEALLOCATE( H%val, H%row, H%col )
     DEALLOCATE( M%val, M%row, M%col )
     DEALLOCATE( A%val, A%row, A%col )
!    STOP
   END DO
   DEALLOCATE( H%ptr, M%ptr, A%ptr, C, X )

!  ==============
!  Normal entries
!  ==============

   n = 3
   ALLOCATE( X( n ), C( n ) )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )    ! Specify co-ordinate for H
   H%ne = 4
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   H%row = (/ 1, 2, 3, 3 /)
   H%col = (/ 1, 2, 3, 1 /)
   H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
   CALL SMT_put( M%type, 'DIAGONAL', smt_stat )        ! Specify diagonal for M
   ALLOCATE( M%val( n ) ) ; M%val = 1.0_wp

   WRITE( 6, "( /, ' ==== normal exits ===== ', / )" )

   DO pass = 1, 13
     C = (/ 5.0_wp, 0.0_wp, 4.0_wp /)
     CALL RQS_initialize( data, control, inform )
!    control%definite_linear_solver = 'ma57'
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!     IF ( pass == 13 ) THEN
!       control%error = 6 ; control%out = 6 ; control%print_level = 10 ; END IF
     sigma = one
     p = 3.0_wp
     IF ( pass == 2 ) sigma = sigma / two
     IF ( pass == 3 ) sigma = 10.0_wp
     IF ( pass == 4 ) sigma = 10.0_wp
     IF ( pass == 5 .OR. pass == 9 .OR. pass == 12 )                           &
       C = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
     IF ( pass == 6 .OR. pass == 10 .OR. pass == 13 )                          &
       C = (/ 0.0_wp, 2.0_wp, 0.0001_wp /)
     IF ( pass == 7 ) C = (/ 0.0_wp, 0.0_wp, 0.0_wp /)
     IF ( pass >= 8 ) p = 2.5_wp
     IF ( pass >= 11 ) p = 3.5_wp
     IF ( pass == 6 .OR. pass == 13 ) control%stop_normal = epsmch ** 0.666

     CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, M = M )

     WRITE( 6, "( ' pass  ', I3, ': RQS_solve exit status = ', I6 )" )         &
            pass, inform%status
!    WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!              inform%iter + info%iter_pass2, f, info%multiplier
     CALL RQS_terminate( data, control, inform ) !  delete internal workspace
   END DO

   DEALLOCATE( X, C, H%row, H%col, H%val, M%val )

   CLOSE( unit = 23 )

   STOP
   END PROGRAM GALAHAD_RQS_test_deck
