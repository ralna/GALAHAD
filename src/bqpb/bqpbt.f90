! THIS VERSION: GALAHAD 3.3 - 03/06/2021 AT 09:45 GMT.
   PROGRAM GALAHAD_BQPB_EXAMPLE
   USE GALAHAD_BQPB_double                            ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infty = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BQPB_data_type ) :: data
   TYPE ( BQPB_control_type ) :: control        
   TYPE ( BQPB_inform_type ) :: info
   INTEGER :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat

   n = 3 ; h_ne = 4  
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( X_stat( n ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for status = - 1 ... - 24

   DO status = 1, 24

     IF ( status == - GALAHAD_error_allocate ) CYCLE
     IF ( status == - GALAHAD_error_deallocate ) CYCLE
!    IF ( status == - GALAHAD_error_restrictions ) CYCLE
!    IF ( status == - GALAHAD_error_bad_bounds ) CYCLE
     IF ( status == - GALAHAD_error_primal_infeasible ) CYCLE
     IF ( status == - GALAHAD_error_dual_infeasible ) CYCLE
     IF ( status == - GALAHAD_error_unbounded ) CYCLE
     IF ( status == - GALAHAD_error_no_center ) CYCLE
     IF ( status == - GALAHAD_error_analysis ) CYCLE
     IF ( status == - GALAHAD_error_factorization ) CYCLE
     IF ( status == - GALAHAD_error_solve ) CYCLE
     IF ( status == - GALAHAD_error_uls_analysis ) CYCLE
     IF ( status == - GALAHAD_error_uls_factorization ) CYCLE
     IF ( status == - GALAHAD_error_uls_solve ) CYCLE
     IF ( status == - GALAHAD_error_preconditioner ) CYCLE
     IF ( status == - GALAHAD_error_ill_conditioned ) CYCLE
     IF ( status == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( status == - GALAHAD_error_max_iterations ) CYCLE
!    IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL BQPB_initialize( data, control, info )
     control%infinity = infty

     p%new_problem_structure = .TRUE.
     p%n = n ; p%f = 1.0_wp
     p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
     p%X_l = (/ - 1.0_wp, - infty, - infty /)
     p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

     ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
     IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
     CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat ) 
     p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 3.0_wp /)
     p%H%col = (/ 1, 1, 2, 3 /)
     p%H%ptr = (/ 1, 2, 4, 5 /)
     p%X = 0.0_wp ; p%Z = 0.0_wp

     IF ( status == - GALAHAD_error_restrictions ) THEN
       p%n = 0
     ELSE IF ( status == - GALAHAD_error_bad_bounds ) THEN 
       p%X_u( 1 ) = - 2.0_wp
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
!      control%print_level = 1
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1
     CALL BQPB_solve( p, data, control, info, X_stat )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',   &
     &      F6.1, ' status = ', I6 )" ) status, info%iter,                    &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': BQPB_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     CALL BQPB_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP

   END DO
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  special test for status = - 20

   status = - GALAHAD_error_inertia
   n = 1 ; h_ne = 1
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( X_stat( n ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%H%ne = h_ne
   p%f = 0.0_wp
   p%G = (/ 0.0_wp /)
   p%X_l = (/ 0.0_wp /)
   p%X_u = (/ infty /)
   p%H%val = (/ - 1.0_wp /)
   p%H%row = (/ 1 /)
   p%H%col = (/ 1 /)
   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
!  control%print_level = 1
   p%X = 1.0_wp ; p%Z = 0.0_wp
   CALL BQPB_solve( p, data, control, info, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &          F6.1, ' status = ', I6 )" ) status, info%iter,                 &
                  info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': BQPB_solve exit status = ', I6 )") status, info%status
   END IF
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; h_ne = 4  
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( X_stat( n ) )

   p%n = n ; p%f = 0.96_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%X_l = (/ - 1.0_wp, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

   DO data_storage_type = -3, 0
     CALL BQPB_initialize( data, control, info )
     control%infinity = infty
!    control%out = 6 ; control%print_level = 11
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
       p%H%row = (/ 1, 1, 2, 3 /)
       p%H%col = (/ 1, 2, 2, 3 /) ; p%H%ne = h_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
       p%H%col = (/ 1, 1, 2, 3 /)
       p%H%ptr = (/ 1, 2, 4, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(0))
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'I'
       ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 3.0_wp /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 3.0_wp /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 3.0_wp /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp /)
       END IF
       p%X = 0.0_wp ; p%Z = 0.0_wp
       CALL BQPB_solve( p, data, control, info, X_stat )

      IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ',&
     &            F6.1, ' status = ', I6 )" ) st, i, info%iter,               &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': BQPB_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
     END DO
     CALL BQPB_terminate( data, control, info )
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; h_ne = 3
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( X_stat( n ) )

   p%n = n ; p%f = 0.05_wp
   p%G = (/ 0.0_wp, 0.0_wp /)
   p%X_l = (/ 0.0_wp, 0.0_wp /)
   p%X_u = (/ 2.0_wp, 3.0_wp /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 1, 2 /)
   p%H%ptr = (/ 1, 2, 4 /)
   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
!  control%out = 6 ; control%print_level = 1
   
!  test with new and existing data

   tests = 2
   DO i = 1, tests
     IF ( i == 1 ) THEN
       X_stat = 1
     ELSE IF ( i == 2 ) THEN
       X_stat = 0
     END IF

     p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp /)
     p%X = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 4
     CALL BQPB_solve( p, data, control, info, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BQPB_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Z = 0.0_wp
     CALL BQPB_solve( p, data, control, info, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BQPB_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_wp, 0.5_wp /)
   p%X_u = (/ 0.5_wp, 0.5_wp /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   CALL BQPB_initialize( data, control, info )
!  control%print_level = 1
   control%infinity = infty
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 1
     CALL BQPB_solve( p, data, control, info, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; h_ne = 21
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( X_stat( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   p%n = n ; p%H%ne = h_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp, 4.0_wp,        &
                4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp, 7.0_wp, 7.0_wp,        &
                8.0_wp, 9.0_wp, 10.0_wp, 11.0_wp, 12.0_wp, 13.0_wp, 14.0_wp /)
!  p%H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 3.0_wp, 0.0_wp, 4.0_wp,        &
!               0.0_wp, 5.0_wp, 0.0_wp, 6.0_wp, 0.0_wp, 7.0_wp, 0.0_wp,        &
!               8.0_wp, 9.0_wp, 10.0_wp, 11.0_wp, 12.0_wp, 13.0_wp, 14.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14 /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
                8, 9, 10, 11, 12, 13, 14 /)

   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
   control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
   p%X = 0.0_wp ; p%Z = 0.0_wp
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL BQPB_solve( p, data, control, info, X_stat )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  Second problem

   n = 14 ; h_ne = 14
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( X_stat( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   p%n = n ; p%H%ne = h_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Z = 0.0_wp
   CALL BQPB_solve( p, data, control, info, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  Third problem

   n = 14 ; h_ne = 14
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( X_stat( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   p%n = n ; p%H%ne = h_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Z = 0.0_wp
   X_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   CALL BQPB_solve( p, data, control, info, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

!  Fourth and Fifth problems

   n = 14 ; ; h_ne = 14
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( X_stat( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   p%n = n ; p%H%ne = h_ne
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%X_l = (/ - infty, - infty, - infty, - infty, - infty, - infty, - infty,   &
              - infty, - infty, - infty, - infty, - infty, - infty, - infty  /)
   p%X_u = - p%X_l
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL BQPB_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Z = 0.0_wp
   X_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   CALL BQPB_solve( p, data, control, info, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_wp ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_wp ; p%Z = 0.0_wp
   X_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   CALL BQPB_solve( p, data, control, info, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BQPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF

   CALL BQPB_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u )
   DEALLOCATE( p%X, p%Z, X_stat )
   DEALLOCATE( p%H%ptr )

   END PROGRAM GALAHAD_BQPB_EXAMPLE
