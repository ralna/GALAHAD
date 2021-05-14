! THIS VERSION: GALAHAD 2.4 - 17/01/2011 AT 15:30 GMT.
   PROGRAM GALAHAD_SCALE_TEST
   USE GALAHAD_SCALE_double         ! double precision version
   USE GALAHAD_SMT_double
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infty = 10.0_wp ** 20
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SCALE_trans_type ) :: trans
   TYPE ( SCALE_data_type ) :: data
   TYPE ( SCALE_control_type ) :: control        
   TYPE ( SCALE_inform_type ) :: info
   INTEGER :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st
   INTEGER :: scale

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4 
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for status = - 1 ... - 24

   DO status = 1, 24

     IF ( status == - GALAHAD_error_allocate ) CYCLE
     IF ( status == - GALAHAD_error_deallocate ) CYCLE
!    IF ( status == - GALAHAD_error_restrictions ) CYCLE
     IF ( status == - GALAHAD_error_bad_bounds ) CYCLE
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
     IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL SCALE_initialize( data, control, info )
     control%infinity = infty

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
       p%n = - 1 ; p%m = - 1
       scale = 1
       control%print_level = - 1
     ELSE IF ( status == - GALAHAD_error_bad_bounds ) THEN 
       p%X_u( 1 ) = - 2.0_wp
       scale = 1
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
       scale = 2
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN 
       scale = 1
       p%H%col( 1 ) = 2
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1
     CALL SCALE_get( p, scale, trans, data, control, info ) 
!    CALL SCALE_apply( p, trans, data, control, info ) 
!    CALL SCALE_recover( p, trans, data, control, info ) 
     WRITE( 6, "( I2, ': SCALE exit status = ', I6 )") status, info%status
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL SCALE_terminate( data, control, info, trans )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP

   END DO
   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
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

   p%n = n ; p%m = m ; p%f = 0.96_wp
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C = (/ 1.0_wp, 2.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 4.0_wp, infty /)
   p%X_l = (/ - 1.0_wp, - infty, - infty /)
   p%X_u = (/ 1.0_wp, infty, 2.0_wp /)
   p%X = one ; p%Y = one ; p%Z = one

   DO data_storage_type = -3, 0
     CALL SCALE_initialize( data, control, info )
     control%infinity = infty
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

!  test with every available scaling

     DO scale = 1, 7
       IF ( scale == 3 ) THEN
!         control%out = 6
!         control%print_level = 2
          control%print_level = 0
       ELSE
          control%print_level = 0
       END IF
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
       CALL SCALE_get( p, scale, trans, data, control, info ) 
       WRITE( 6, "( A1, I1,': SCALE exit status = ', I6 ) " )                  &
         st, scale, info%status
       CALL SCALE_apply( p, trans, data, control, info ) 
       CALL SCALE_recover( p, trans, data, control, info ) 
     END DO
     CALL SCALE_terminate( data, control, info, trans )
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
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

   p%n = n ; p%m = m ; p%f = 0.05_wp
   p%G = (/ 0.0_wp, 0.0_wp /)
   p%C = (/ 1.0_wp, 2.0_wp /)
   p%C_l = (/ 1.0_wp /)
   p%C_u = (/ 1.0_wp /)
   p%X_l = (/ 0.0_wp, 0.0_wp /)
   p%X_u = (/ 2.0_wp, 3.0_wp /)
   p%X = one ; p%Y = one ; p%Z = one

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
   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
!  control%out = 6 ; control%print_level = 1
   
!  test with new and existing data

   tests = 8
   DO i = 0, tests
     IF ( i == 0 ) THEN
       scale = 1
       control%shift_x = 0
     ELSE IF ( i == 1 ) THEN
       control%shift_x = 1
       control%scale_x = 0
     ELSE IF ( i == 2 ) THEN
       control%scale_x = 1
       control%shift_c = 0
     ELSE IF ( i == 3 ) THEN
       control%shift_c = 1
       control%scale_c = 0
     ELSE IF ( i == 4 ) THEN
       control%scale_c = 1
       control%shift_f = 0
!      control%precon = 5
     ELSE IF ( i == 5 ) THEN     
       control%shift_f = 1
       control%scale_f = 0
     ELSE IF ( i == 6 ) THEN     
       control%scale_f = 0
       scale = 2 
     END IF
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 4
     CALL SCALE_get( p, scale, trans, data, control, info ) 
     WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) i, info%status
     CALL SCALE_apply( p, trans, data, control, info ) 
     CALL SCALE_recover( p, trans, data, control, info ) 
   END DO
   CALL SCALE_terminate( data, control, info, trans )

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
   CALL SCALE_initialize( data, control, info )
   scale = 1
   control%infinity = infty
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
     CALL SCALE_get( p, scale, trans, data, control, info ) 
     WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) i, info%status
     CALL SCALE_apply( p, trans, data, control, info ) 
     CALL SCALE_recover( p, trans, data, control, info ) 
   END DO
   CALL SCALE_terminate( data, control, info, trans )

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
   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 1
     CALL SCALE_get( p, scale, trans, data, control, info ) 
     WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) i, info%status
     CALL SCALE_apply( p, trans, data, control, info ) 
     CALL SCALE_recover( p, trans, data, control, info ) 
   END DO
   CALL SCALE_terminate( data, control, info, trans )

!  case when there are no constraints

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
   p%A%ptr = (/ 1, 1 /)
   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   DO i = tests + 3, tests + 3
     p%H%val = (/ 1.0_wp, 1.0_wp /)
     p%A%val = (/ 1.0_wp, 1.0_wp /)
     p%m = 0
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
!    control%print_level = 1
     CALL SCALE_get( p, scale, trans, data, control, info ) 
     WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) i, info%status
     CALL SCALE_apply( p, trans, data, control, info ) 
     CALL SCALE_recover( p, trans, data, control, info ) 
   END DO
   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; h_ne = 21 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
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
                7.0_wp, 7.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp,                &
                5.0_wp, 6.0_wp, 7.0_wp /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14  /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
                8, 9, 10, 11, 12, 13, 14  /)
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
   p%C = p%C_l
   p%X = one ; p%Y = one ; p%Z = one

   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL SCALE_get( p, scale, trans, data, control, info ) 
   WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) 1, info%status
   CALL SCALE_apply( p, trans, data, control, info ) 
   CALL SCALE_recover( p, trans, data, control, info ) 
   CLOSE( UNIT = scratch_out )
   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Second problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
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
   p%C = p%C_l
   p%X = one ; p%Y = one ; p%Z = one

   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL SCALE_get( p, scale, trans, data, control, info ) 
   WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) 2, info%status
   CALL SCALE_apply( p, trans, data, control, info ) 
   CALL SCALE_recover( p, trans, data, control, info ) 
   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Third problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
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
   p%C = p%C_l
   p%X = one ; p%Y = one ; p%Z = one

   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL SCALE_get( p, scale, trans, data, control, info ) 
   WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) 2, info%status
   CALL SCALE_apply( p, trans, data, control, info ) 
   CALL SCALE_recover( p, trans, data, control, info ) 
   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Fourth and Fifth problems

   n = 14 ; m = 10 ; h_ne = 14 ; a_ne = 32
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
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
   p%C = p%C_l
   p%X = one ; p%Y = one ; p%Z = one

   CALL SCALE_initialize( data, control, info )
   control%infinity = infty
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL SCALE_get( p, scale, trans, data, control, info ) 
   WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) 2, info%status
   CALL SCALE_apply( p, trans, data, control, info ) 
   CALL SCALE_recover( p, trans, data, control, info ) 

!  control%out = 6 ; control%print_level = 1
!  control%ESCALE_control%print_level = 2

   p%X_l( 1 ) = 1.0_wp ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL SCALE_get( p, scale, trans, data, control, info ) 
   WRITE( 6, "( I2, ': SCALE exit status = ', I6 ) " ) 2, info%status
   CALL SCALE_apply( p, trans, data, control, info ) 
   CALL SCALE_recover( p, trans, data, control, info ) 

   CALL SCALE_terminate( data, control, info, trans )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%H%ptr, p%A%ptr )

   END PROGRAM GALAHAD_SCALE_TEST
