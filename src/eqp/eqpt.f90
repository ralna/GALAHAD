! THIS VERSION: GALAHAD 2.4 - 01/09/2011 AT 15:30 GMT.
   PROGRAM GALAHAD_EQP_EXAMPLE
   USE GALAHAD_EQP_double                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infty = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( EQP_data_type ) :: data
   TYPE ( EQP_control_type ) :: control
   TYPE ( EQP_inform_type ) :: info
   INTEGER :: n, m, h_ne, a_ne, prec, preconditioner, factorization, smt_stat
   INTEGER :: data_storage_type, i, tests, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)

   CALL EQP_initialize( data, control, info )
   control%print_level = 0
!  control%print_level = 1
   control%sbls_control%sls_control%warning = - 1
   control%sbls_control%sls_control%out = - 1
   control%sbls_control%uls_control%warning = - 1

!  tests for status = - 1 ... - 10

   DO i = 1, 5
!  DO i = 1, 3
     ALLOCATE( p%H%val( h_ne ), p%H%col( h_ne ) )
     ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
     CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
     p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
     p%H%col = (/ 1, 2, 3, 1 /)
     p%H%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp

     IF ( i == 2 ) THEN
       p%n = 0 ; p%m = - 1
       status = 3
     ELSE IF ( i == 1 ) THEN
       p%n = n ; p%m = m
       p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       p%A%col = (/ 1, 2, 1, 2 /)
       p%C = (/ 1.0_wp, 1.00000001_wp /)
       control%SBLS_control%preconditioner = - 1 ; factorization = 2
       status = 25
!      control%print_level = 2
       p%H%val(4)=0.0_wp
     ELSE IF ( i == 5 ) THEN
       p%n = n ; p%m = m
       p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       p%A%col = (/ 1, 2, 1, 2 /)
       p%C = (/ 1.0_wp, 1.00000000001_wp /)
       control%SBLS_control%preconditioner = - 1 ; factorization = 2
       status = 16
!      control%print_level = 2
       p%H%val(4)=0.0_wp
     ELSE IF ( i == 3 ) THEN
       p%n = n ; p%m = m
       p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       p%A%col = (/ 1, 2, 1, 2 /)
       p%C = (/ 1.0_wp, 2.0_wp /)
       control%sbls_control%preconditioner = - 1
       status = 5
     ELSE IF ( i == 4 ) THEN
       p%n = n ; p%m = m
       p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       p%A%col = (/ 1, 2, 1, 2 /)
       p%C = (/ 1.0_wp, 2.0_wp /)
       control%sbls_control%preconditioner = 2
       status = 5
     ELSE
     END IF

     IF ( i == 1 ) THEN
       CALL EQP_resolve( p, data, control, info )
     ELSE
       CALL EQP_solve( p, data, control, info )
     END IF
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &   F6.1, ' status = ', I6 )" ) status, info%cg_iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': EQP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%H%val, p%H%col )
     DEALLOCATE( p%A%val, p%A%col )
   END DO

   CALL EQP_terminate( data, control, info )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 1 ; h_ne = 4 ; a_ne = 2
   ALLOCATE( p%C( m ) )
   ALLOCATE( p%G( n ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )

   p%n = n ; p%m = m ; p%f = 0.96_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C = (/ 2.0_wp /)

   DO prec = 1, 8
     SELECT CASE( prec)
     CASE( 1 : 4 ) ; preconditioner = prec
     CASE( 5 ) ; preconditioner = 11
     CASE( 6 ) ; preconditioner = 12
     CASE( 7 ) ; preconditioner = -1
     CASE( 8 ) ; preconditioner = -2
     END SELECT
     DO factorization = 1, 2
       DO data_storage_type = -3, 1
!      DO data_storage_type = -3, -3
         CALL EQP_initialize( data, control, info )
!        control%print_level = 1
         control%sbls_control%preconditioner = preconditioner
         control%SBLS_control%factorization = factorization
         p%new_problem_structure = .TRUE.
         IF ( data_storage_type == 0 ) THEN   ! sparse co-ordinate storage
           st = 'C'
           ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
           ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
           IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
           CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
           p%H%row = (/ 1, 2, 3, 3 /)
           p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
           IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
           p%A%row = (/ 1, 1 /)
           p%A%col = (/ 1, 2 /) ; p%A%ne = a_ne
         ELSE IF ( data_storage_type == - 1 ) THEN ! sparse row-wise storage
           st = 'R'
           ALLOCATE( p%H%val( h_ne ), p%H%col( h_ne ) )
           ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ) )
           IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
           CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
           p%H%col = (/ 1, 2, 3, 1 /)
           p%H%ptr = (/ 1, 2, 3, 5 /)
           IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
           p%A%col = (/ 1, 2 /)
           p%A%ptr = (/ 1, 3 /)
         ELSE IF ( data_storage_type == - 2 ) THEN ! dense storage
           st = 'D'
           ALLOCATE( p%H%val(n*(n+1)/2) )
           ALLOCATE( p%A%val(n*m) )
           IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
           CALL SMT_put( p%H%type, 'DENSE', smt_stat )
           IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'DENSE', smt_stat )
         ELSE IF ( data_storage_type == - 3 ) THEN ! diagonal/dense storage
           st = 'I'
           ALLOCATE( p%H%val(n) )
           ALLOCATE( p%A%val(n*m) )
           IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
           CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
           IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'DENSE', smt_stat )
         ELSE IF ( data_storage_type == 1 ) THEN ! weighted/dense storage
           st = 'W'
           ALLOCATE( p%WEIGHT(n) )
           ALLOCATE( p%X0(n) )
           ALLOCATE( p%A%val(n*m) )
           IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
           IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'DENSE', smt_stat )
         END IF

!  test with new and existing data

         DO i = 0, 2
           control%new_a = 2 - i
           control%new_h = 2 - i
           p%Hessian_kind = - 1
           IF ( data_storage_type == 0 ) THEN     ! sparse co-ordinate storage
             p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
             p%A%val = (/ 2.0_wp, 1.0_wp /)
           ELSE IF ( data_storage_type == - 1 ) THEN  !  sparse row-wise storage
             p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
             p%A%val = (/ 2.0_wp, 1.0_wp /)
           ELSE IF ( data_storage_type == - 2 ) THEN    ! dense storage
             p%H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 4.0_wp, 0.0_wp, 3.0_wp /)
             p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp /)
           ELSE IF ( data_storage_type == - 3 ) THEN    ! diagonal/dense storage
             p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp /)
             p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp /)
           ELSE IF ( data_storage_type == 1 ) THEN      ! weight/dense storage
             p%WEIGHT = (/ 1.0_wp, 1.0_wp, 2.0_wp /)
             p%X0 = (/ 1.0_wp, 0.0_wp, -1.0_wp /)
             p%Hessian_kind = 2
             p%target_kind = - 1
             p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp /)
           END IF
           p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
           CALL EQP_solve( p, data, control, info )
           IF ( info%status == 0 ) THEN
             WRITE( 6, "( A1, I1, A1, I2, A1, I1, ':', I6, ' iterations. ',    &
           &      'Optimal objective value = ', F6.1, ' status = ', I6 )")     &
               st, i, 'P', preconditioner, 'F', factorization,                 &
               info%cg_iter, info%obj, info%status
           ELSE
             WRITE( 6, "( A1, I1, A1, I2, A1, I1, ' ',                         &
            &   'EQP_solve exit status = ', I6 ) " )                           &
               st, i, 'P', preconditioner, 'F', factorization, info%status
           END IF
         END DO
         IF ( data_storage_type == 0 ) THEN   ! sparse co-ordinate storage
           DEALLOCATE( p%H%val, p%H%row, p%H%col )
           DEALLOCATE( p%A%val, p%A%row, p%A%col )
         ELSE IF ( data_storage_type == - 1 ) THEN ! sparse row-wise storage
           DEALLOCATE( p%H%val, p%H%col )
           DEALLOCATE( p%A%val, p%A%col )
         ELSE IF ( data_storage_type == - 2 ) THEN ! dense storage
           DEALLOCATE( p%H%val )
           DEALLOCATE( p%A%val )
         ELSE IF ( data_storage_type == - 3 ) THEN ! diagonal/dense storage
           DEALLOCATE( p%H%val )
           DEALLOCATE( p%A%val )
         ELSE IF ( data_storage_type == 1 ) THEN ! weight/dense storage
           DEALLOCATE( p%WEIGHT )
           DEALLOCATE( p%X0 )
           DEALLOCATE( p%A%val )
         END IF
         CALL EQP_terminate( data, control, info )
       END DO
     END DO
   END DO
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )

   p%n = n ; p%m = m ; p%f = 0.05_wp
   p%G = (/ 0.0_wp, 0.0_wp /)
   p%C = (/ 1.0_wp /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Hessian_kind = - 1
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL EQP_initialize( data, control, info )

!  test with new and existing data

   tests = 22
   DO i = 0, tests
     IF ( i == 0 ) THEN
       control%sbls_control%preconditioner = 0
     ELSE IF ( i == 1 ) THEN
       control%sbls_control%preconditioner = 1
     ELSE IF ( i == 2 ) THEN
       control%sbls_control%preconditioner = 2
     ELSE IF ( i == 3 ) THEN
       control%sbls_control%preconditioner = 3
     ELSE IF ( i == 4 ) THEN
       control%sbls_control%preconditioner = 4
     ELSE IF ( i == 5 ) THEN
       control%SBLS_control%factorization = - 1
     ELSE IF ( i == 6 ) THEN
       control%SBLS_control%factorization = 1
     ELSE IF ( i == 7 ) THEN
       control%max_col = 0
     ELSE IF ( i == 8 ) THEN
       control%SBLS_control%factorization = 2
       control%sbls_control%preconditioner = 0
     ELSE IF ( i == 9 ) THEN
!      control%print_level = 2
       control%sbls_control%preconditioner = 11
     ELSE IF ( i == 10 ) THEN
       control%sbls_control%preconditioner = 12
     ELSE IF ( i == 11 ) THEN
       control%sbls_control%preconditioner = - 1
     ELSE IF ( i == 12 ) THEN
       control%sbls_control%preconditioner = - 2
     ELSE IF ( i == 13 ) THEN

     ELSE IF ( i == 14 ) THEN
       control%max_col = 5
!      control%primal = .TRUE.
     ELSE IF ( i == 15 ) THEN
       control%max_col = 75
     ELSE IF ( i == 16 ) THEN
!      control%feasol = .FALSE.
     ELSE IF ( i == 16 ) THEN

     ELSE IF ( i == 17 ) THEN

     ELSE IF ( i == 18 ) THEN

     ELSE IF ( i == 19 ) THEN

     ELSE IF ( i == 20 ) THEN

     ELSE IF ( i == 21 ) THEN

     ELSE IF ( i == 22 ) THEN

     END IF

     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp

!    control%print_level = 4
     CALL EQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%cg_iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL EQP_terminate( data, control, info )

!  case when there are no bounded variables

   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL EQP_initialize( data, control, info )
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
     CALL EQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%cg_iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL EQP_terminate( data, control, info )

!  case when there are no free variables

   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL EQP_initialize( data, control, info )
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 1
     CALL EQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%cg_iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL EQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 8 ; h_ne = 28 ; a_ne = 27
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /)
   p%C = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 2.0_wp /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp,                                                &
                20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, &
                20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp, 20.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)

   CALL EQP_initialize( data, control, info )
!  control%print_level = 1
   control%itref_max = 3
   control%out = scratch_out
   control%error = scratch_out
!  control%print_level = 1
!  control%out = 6
!  control%error = 6
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL EQP_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',   &
     &       F6.1, ' status = ', I6 )" ) 1, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL EQP_resolve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I1, 'R:', I6, ' iterations. Optimal objective value = ',   &
     &       F6.1, ' status = ', I6 )" ) 1, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I1, 'R: EQP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CLOSE( UNIT = scratch_out )
   CALL EQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )

!  Second problem

   n = 14 ; m = 8 ; h_ne = 14 ; a_ne = 27
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /)
   p%C = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 2.0_wp /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)

   CALL EQP_initialize( data, control, info )
!  control%print_level = 1
   control%sbls_control%preconditioner = - 2
   control%SBLS_control%factorization = 2
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL EQP_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 2, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL EQP_resolve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I1, 'R:', I6, ' iterations. Optimal objective value = ',   &
     &       F6.1, ' status = ', I6 )" ) 2, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I1, 'R: EQP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL EQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )

!  Third problem

   n = 14 ; m = 8 ; h_ne = 14 ; a_ne = 27
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /)
   p%C = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 2.0_wp /)
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 10, 11, 12, 13, 14 /)
   p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 7.0_wp,        &
                1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL EQP_initialize( data, control, info )
!  control%print_level = 1 ; control%out = 6 ; control%error = 6
   control%sbls_control%preconditioner = 2
   control%SBLS_control%factorization = 1
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL EQP_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 3, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': EQP_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL EQP_resolve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I1, 'R:', I6, ' iterations. Optimal objective value = ',   &
     &       F6.1, ' status = ', I6 )" ) 3, info%cg_iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I1, 'R: EQP_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL EQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%G, p%X, p%Y, p%Z, p%C )

   GO TO 10000
10000 CONTINUE
   DEALLOCATE( p%A%type, p%H%type )

   END PROGRAM GALAHAD_EQP_EXAMPLE


