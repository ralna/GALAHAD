! THIS VERSION: GALAHAD 2.4 - 08/04/2010 AT 08:00 GMT.
   PROGRAM GALAHAD_QPC_EXAMPLE
   USE GALAHAD_QPC_double                            ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infty = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( QPC_data_type ) :: data
   TYPE ( QPC_control_type ) :: control        
   TYPE ( QPC_inform_type ) :: info
   INTEGER :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

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
!    IF ( status == - GALAHAD_error_primal_infeasible ) CYCLE
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
!    IF ( status == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( status == - GALAHAD_error_max_iterations ) CYCLE
!    IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
!    IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL QPC_initialize( data, control, info )
     control%infinity = infty
     control%restore_problem = 1

     p%new_problem_structure = .TRUE.
     p%n = n ; p%m = m ; p%f = 1.0_wp
     p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
     p%C_l = (/ 1.0_wp, 2.0_wp /)
     p%C_u = (/ 4.0_wp, infty /)
     p%X_l = (/ - 1.0_wp, - infty, - infty /)
     p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

     ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
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

     IF ( status == - GALAHAD_error_restrictions ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == - GALAHAD_error_bad_bounds ) THEN 
       p%X_u( 1 ) = - 2.0_wp
     ELSE IF ( status == - GALAHAD_error_primal_infeasible ) THEN
!      control%print_level = 1
       p%X_l = (/ - 1.0_wp, 8.0_wp, - infty /)
       p%X_u = (/ 1.0_wp, infty, 2.0_wp /)
     ELSE IF ( status == - GALAHAD_error_tiny_step ) THEN
!      control%print_level = 1
       control%QPB_control%initial_radius = EPSILON( 1.0_wp ) ** 2
       control%no_qpa = .TRUE.
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%QPA_control%maxit = 1 ; control%QPB_control%maxit = 1
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN 
       p%H%col( 1 ) = 2
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1
     CALL QPC_solve( p, C_stat, B_stat, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &      F6.1, ' status = ', I6 )" ) status, info%QPA_inform%iter,          &
              info%QPB_inform%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': QPC_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL QPC_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP

   END DO
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  special test for status = - 7

   status = - GALAHAD_error_unbounded
   n = 1 ; m = 0 ; h_ne = 1 ; a_ne = 0
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 0.0_wp
   p%G = (/ 0.0_wp /)
   p%X_l = (/ 0.0_wp /)
   p%X_u = (/ infty /)
   p%H%val = (/ - 1.0_wp /)
   p%H%row = (/ 1 /)
   p%H%col = (/ 1 /)
   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 1
!  control%print_level = 1
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &          F6.1, ' status = ', I6 )" ) status, info%QPA_inform%iter,      &
                  info%QPB_inform%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': QPC_solve exit status = ', I6 )") status, info%status
   END IF
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 4.0_wp, infty /)
   p%X_l = (/ - 1.0_wp, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, 2.0_wp /)

   DO data_storage_type = -3, 0
     CALL QPC_initialize( data, control, info )
     control%infinity = infty
     control%restore_problem = 2
!    control%out = 6 ; control%print_level = 11
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
       p%H%row = (/ 1, 2, 3, 3 /)
       p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%row = (/ 1, 1, 2, 2 /)
       p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
       p%H%col = (/ 1, 2, 3, 1 /)
       p%H%ptr = (/ 1, 2, 3, 5 /)
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 2, 3 /)
       p%A%ptr = (/ 1, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(n*(n+1)/2))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'I'
       ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
         p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
         p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 4.0_wp, 0.0_wp, 3.0_wp /)
         p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp /)
         p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /)
       END IF
       p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
       CALL QPC_solve( p, C_stat, B_stat, data, control, info )

      IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', 2I6,' iterations. Optimal objective value = ',&
     &            F6.1, ' status = ', I6 )" ) st, i, info%QPA_inform%iter,     &
                  info%QPB_inform%iter, info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': QPC_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
     END DO
     CALL QPC_terminate( data, control, info )
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.05_wp
   p%G = (/ 0.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp /)
   p%C_u = (/ 1.0_wp /)
   p%X_l = (/ 0.0_wp, 0.0_wp /)
   p%X_u = (/ 2.0_wp, 3.0_wp /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
   
!  test with new and existing data

   tests = 25
   DO i = 0, tests
     IF ( i == 0 ) THEN
       control%QPB_control%precon = 0
     ELSE IF ( i == 1 ) THEN
       control%QPB_control%precon = 1
     ELSE IF ( i == 2 ) THEN
       control%QPB_control%precon = 2
     ELSE IF ( i == 3 ) THEN
       control%QPB_control%precon = 3
     ELSE IF ( i == 4 ) THEN
       control%QPB_control%precon = 5
     ELSE IF ( i == 5 ) THEN     
       control%QPB_control%factor = - 1
     ELSE IF ( i == 6 ) THEN     
       control%QPB_control%factor = 1
     ELSE IF ( i == 7 ) THEN     
       control%QPB_control%max_col = 0
     ELSE IF ( i == 8 ) THEN     
       control%QPB_control%factor = 2
       control%QPB_control%precon = 0
     ELSE IF ( i == 9 ) THEN
!      control%print_level = 2
       control%QPB_control%precon = 1
     ELSE IF ( i == 10 ) THEN
       control%QPB_control%precon = 2
     ELSE IF ( i == 11 ) THEN
       control%QPB_control%precon = 3
     ELSE IF ( i == 12 ) THEN
       control%QPB_control%precon = 5
     ELSE IF ( i == 13 ) THEN
       control%QPB_control%center = .FALSE.
     ELSE IF ( i == 14 ) THEN
       control%QPB_control%primal = .TRUE.       
     ELSE IF ( i == 15 ) THEN
       control%QPB_control%feasol = .FALSE.


     ELSE IF ( i == 16 ) THEN
       control%QPA_control%cold_start = 0
       B_stat = 0 ; C_stat = 0 ; B_stat( 1 ) = - 1
     ELSE IF ( i == 17 ) THEN
       control%QPA_control%cold_start = 0
       B_stat = 0 ; C_stat = 0 ; C_stat( 1 ) = - 1
     ELSE IF ( i == 18 ) THEN
       control%QPA_control%cold_start = 2
     ELSE IF ( i == 19 ) THEN
       control%QPA_control%cold_start = 3
     ELSE IF ( i == 20 ) THEN
       control%QPA_control%cold_start = 4
     ELSE IF ( i == 21 ) THEN
       control%QPA_control%cold_start = 1
       control%QPA_control%deletion_strategy = 1
     ELSE IF ( i == 22 ) THEN
       control%QPA_control%deletion_strategy = 2
     ELSE IF ( i == 23 ) THEN
       control%QPA_control%solve_within_bounds = .FALSE.
!      control%feasol = .FALSE.
     ELSE IF ( i == 24 ) THEN
       control%no_qpa = .TRUE.
     ELSE IF ( i == 25 ) THEN
       control%no_qpb = .TRUE.
     END IF

     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 4
     CALL QPC_solve( p, C_stat, B_stat, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) i, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL QPC_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
     CALL QPC_solve( p, C_stat, B_stat, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) i, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL QPC_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_wp, 0.5_wp /)
   p%X_u = (/ 0.5_wp, 0.5_wp /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL QPC_initialize( data, control, info )
!  control%print_level = 1
   control%infinity = infty
   control%restore_problem = 2
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!  control%print_level = 1
!  control%QPB_control%print_level = 1
!  control%QPB_control%extrapolate = 1
!  control%EQP_control%print_level = 1
!  control%EQP_control%SBLS_control%print_level = 2

     CALL QPC_solve( p, C_stat, B_stat, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) i, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 1
   control%print_level = 101
   control%QPA_control%itref_max = 3 ; control%QPB_control%itref_max = 3
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
!  control%print_level = 1
!  control%QPB_control%print_level = 1
!  control%QPB_control%SBLS_control%print_level = 1
!  control%QPB_control%LSQP_control%SBLS_control%print_level = 1
!  control%QPB_control%SBLS_control%unsymmetric_linear_solver = 'ma48'
!  control%QPB_control%LSQP_control%SBLS_control%unsymmetric_linear_solver = 'ma48'
!  control%QPB_control%SBLS_control%symmetric_linear_solver = 'ma57'
!  control%QPB_control%LSQP_control%SBLS_control%symmetric_linear_solver = 'ma86'
!  control%QPB_control%remove_dependencies = .TRUE.
!  control%QPB_control%LSQP_control%remove_dependencies = .TRUE.
!  control%QPB_control%LSQP_control%FDC_control%print_level = 2
!  control%generate_sif_file = .TRUE.


   control%QPB_control%extrapolate = 1
!  control%EQP_control%print_level = 1
!  control%EQP_control%SBLS_control%print_level = 2
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 1, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Second problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
!  control%out = 6
!  control%error = 6
!  control%print_level = 1
!  control%QPB_control%print_level = 1
!  control%EQP_control%print_level = 1
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 2, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Third problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 1.0_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp,            &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              4.0_wp, 2.0_wp, 6.0_wp, - infty, - infty,                        &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              4.0_wp, infty, 10.0_wp, 2.0_wp, infty,                           &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty,       &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty,             &
              1.0_wp, infty, infty, 3.0_wp, 4.0_wp, 0.0_wp, infty /)
   p%H%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp,                &
                4.0_wp, 4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp,                &
                7.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%QPA_control%cold_start = 0
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
!  C_stat( 10 ) = 1 ; C_stat( 11 ) = 1
!  C_stat( 12 ) = 1 ; C_stat( 13 ) = 1
!  C_stat( 14 ) = 1 ; C_stat( 15 ) = 1
!  C_stat( 16 ) = 1 ; C_stat( 17 ) = 1
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 3, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Fourth and Fifth problems

   n = 14 ; m = 10 ; h_ne = 14 ; a_ne = 32
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
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
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13 /)
   p%C_l = 0.0_wp
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = p%C_l

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%QPA_control%cold_start = 0
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 4, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_wp ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 5, info%QPA_inform%iter,     &
                      info%QPB_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Sixth problem

   n = 11 ; m = 5 ; h_ne = 21 ; a_ne = 33
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 0.0_wp
   p%G = (/ 0.5_wp, -0.5_wp, -1.0_wp, -1.0_wp, -1.0_wp,  -1.0_wp,              &
           -1.0_wp, -1.0_wp, -1.0_wp, -1.0_wp, -0.5_wp /) ! objective gradient
   p%H%val = (/ 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp,                        &
                0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp,                        &
                1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp,                        &
                0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp /) ! H values
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,                            &
                7, 7, 8, 8, 9, 9, 10, 10, 11 /) ! H columns
   p%H%ptr = (/ 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 /) ! pointers to H col
   p%A%val  = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                       &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                       &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                       &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,               &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,               &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! A values
   p%A%col = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,                             &
                3, 4, 5, 6, 7, 8, 9, 10, 11,                                   &
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 11, 9 /) ! A columns
   p%A%ptr = (/ 1, 12, 21, 31, 33, 34 /) ! pointers to A columns
   p%X_l = (/ 0.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                  &
            1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! variable lower bound
   p%X_u  = (/ infty, infty, infty, infty, infty, infty,                       &
             infty, infty, infty, infty, infty /) ! upper bound
   p%C_l = (/  -infty, 9.0_wp, - infty, 2.0_wp, 1.0_wp /)  ! lower bound
   p%C_u = (/  1.0D+1, infty, 1.0D+1, infty, infty /)   ! upper bound
   p%C = (/ 1.0D+1, 9.0_wp, 1.0D+1, 2.0_wp, 1.0_wp /) ! optimal constraint value
   p%X = (/ 0.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                    &
            1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! optimal variables
   p%Y = (/  -1.0_wp, 3.5_wp, -2.0_wp, 0.1_wp, 0.1_wp /) ! optimal Lag mults
   p%Z = (/ 2.0_wp, 4.0_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp,                    &
            0.5_wp, 0.5_wp, 0.4_wp, 0.4_wp, 0.4_wp /) ! optimal dual variables

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%convex = .TRUE.
!  control%print_level = 1
!  control%EQP_control%print_level = 1
   control%QPA_control%cold_start = 0
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   B_stat = 0 ; C_stat = 0
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 6, info%QPA_inform%iter,     &
                      info%CQP_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 6, info%status
   END IF

   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Seventh problem

   n = 11 ; m = 4 ; h_ne = 21 ; a_ne = 24
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne 
   p%f = 0.0_wp
   p%G   = (/ 0.5_wp, -0.5_wp, -1.0_wp, -1.0_wp, -1.0_wp,  0.0_wp,             &
             -1.0_wp, -1.0_wp, -1.0_wp, -1.0_wp, -0.5_wp /) ! objective gradient
   p%H%val = (/ 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp,                        &
                0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp,                        &
                1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp,                        &
                0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp, 0.5_wp, 1.0_wp /) ! H values
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,                            &
                7, 7, 8, 8, 9, 9, 10, 10, 11 /) ! H columns
   p%H%ptr = (/ 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 /) ! pointers to H col
   p%A%val  = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,               &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,               &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,               &
                 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! A values
   p%A%col = (/ 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11,                          &
                1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11  /) ! A columns
   p%A%ptr = (/ 1, 7, 13, 19, 25 /) ! pointers to A columns
   p%X_l  = (/ -infty, -infty, -infty, -infty,- infty, -infty,                 &
               -infty, -infty, -infty, -infty, -infty /) ! lower bound
   p%X_u  = (/ infty, infty, infty, infty, infty, infty,                       &
               infty, infty, infty, infty, infty /)      ! upper bound
   p%C_l = (/  -infty, 6.0_wp, 5.0_wp, -infty /)  ! constraint lower bound
   p%C_u = (/  5.0_wp, infty, infty, 6.0_wp /)    ! constraint upper bound
   p%C = (/ 5.0_wp, 6.0_wp, 5.0_wp, 6.0_wp /)   ! optimal constraint value
   p%X = (/ 0.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                    &
            1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! optimal variables
   p%Y = (/  -1.0_wp, 2.0_wp, 2.0_wp, -1.0_wp /) ! optimal Lag multipliers
   p%Z = (/ 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                    &
            0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /) ! optimal dual variables

   CALL QPC_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%convex = .TRUE.
!  control%print_level = 1
!  control%EQP_control%print_level = 1
   control%QPA_control%cold_start = 0
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   B_stat = 0 ; C_stat = 0
   CALL QPC_solve( p, C_stat, B_stat, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', 2I6, ' iterations. Optimal objective value = ',   &
     &                F6.1, ' status = ', I6 )" ) 7, info%QPA_inform%iter,     &
                      info%CQP_inform%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': QPC_solve exit status = ', I6 ) " ) 7, info%status
   END IF

   CALL QPC_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )
   END PROGRAM GALAHAD_QPC_EXAMPLE
