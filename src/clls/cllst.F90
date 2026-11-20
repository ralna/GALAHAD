! THIS VERSION: GALAHAD 5.6 - 2026-11-20 AT 14:42 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_CLLS_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_CLLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( CLLS_data_type ) :: data
   TYPE ( CLLS_control_type ) :: control
   TYPE ( CLLS_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, m, o, ao_ne, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   INTEGER ( KIND = ip_ ) :: prob, itype, ired, errors
   REAL ( KIND = rp_ ) :: err, tol
   REAL ( KIND = rp_ ) :: x_sol( 3 ), y_sol( 2 ), z_sol( 3 )
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = REPEAT( ' ', 30 )
!  symmetric_linear_solver = 'slblt'
!  symmetric_linear_solver = 'ma97 '
   symmetric_linear_solver = 'sytr '

!go to 111
!go to 999
   n = 3 ; o = 4 ; m = 2 ; ao_ne = 7 ; a_ne = 4
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )

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
     IF ( status == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( status == - GALAHAD_error_max_iterations ) CYCLE
!    IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE
     CALL CLLS_initialize( data, control, info )
     control%infinity = infty
     control%restore_problem = 1
! control%print_level = 1
     control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%use_sls = .TRUE.

!control%print_level = 3
!control%maxit = 2
!control%print_level = 3
!control%preconditioner = 3

     p%new_problem_structure = .TRUE.
     p%n = n ; p%o = o ; p%m = m
     p%B = (/ 2.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
     p%C_u = (/ 4.0_rp_, infty /)
     p%X_l = (/ - 1.0_rp_, - infty, - infty /)
     p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

     ALLOCATE( p%Ao%val( ao_ne ), p%Ao%col( ao_ne ) )
     ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
     CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
     p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /)
     p%Ao%col = (/ 1, 2, 2, 3, 1, 3, 2 /)
     p%Ao%ptr = (/ 1, 3, 5, 6, 7 /)
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

     IF ( status == - GALAHAD_error_restrictions ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == - GALAHAD_error_bad_bounds ) THEN
       p%X_u( 1 ) = - 2.0_rp_
     ELSE IF ( status == - GALAHAD_error_primal_infeasible ) THEN
!      control%print_level = 1
       p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
       p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE IF ( status == - GALAHAD_error_tiny_step ) THEN
!      control%print_level = 1
       control%stop_abs_c = 0.0_rp_
       control%stop_rel_c = 0.0_rp_
!      p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
!      p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
       p%X( 2 ) = 100000000.0_rp_
!      control%print_level = 1
!      control%maxit = 1
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN
       p%Ao%col( 1 ) = 2
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1

     CALL CLLS_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F7.2, ' status = ', I6 )" ) status, info%iter,                     &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': CLLS_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%Ao%val, p%Ao%col )
     DEALLOCATE( p%A%val, p%A%col )
     CALL CLLS_terminate( data, control, info )

   END DO
   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

!999 continue
   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; o = 3 ; m = 2 ; ao_ne = 4 ; a_ne = 4
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )

   p%n = n ; p%o = o ; p%m = m
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   DO data_storage_type = - 3, 0
!  DO data_storage_type = - 1, - 1
     CALL CLLS_initialize( data, control, info )
     control%infinity = infty
     control%restore_problem = 2
!    control%out = 6 ; control%print_level = 1
!    control%print_level = 4
!    control%SLS_control%print_level = 3
!    control%SLS_control%print_level_solver = 2
!    control%preconditioner = 3
!    control%itref_max = 2
     control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%FDC_control%use_sls = .TRUE.
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
       p%Ao%row = (/ 1, 2, 3, 3 /)
       p%Ao%col = (/ 1, 2, 3, 1 /) ; p%Ao%ne = ao_ne
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%row = (/ 1, 1, 2, 2 /)
       p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( 0 ), p%Ao%col( ao_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
       p%Ao%col = (/ 1, 2, 3, 1 /)
       p%Ao%ptr = (/ 1, 2, 3, 5 /)
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 2, 3 /)
       p%A%ptr = (/ 1, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), STAT = i )
       ALLOCATE( p%A%val( n * m ), p%A%row( 0 ), p%A%col( 0 ), STAT = i )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! dense by column storage
       st = 'I'
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), STAT = i )
       ALLOCATE( p%A%val( n * m ), p%A%row( 0 ), p%A%col( 0 ), STAT = i )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'DENSE_BY_COLUMNS', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE_BY_COLUMNS', smt_stat )
     END IF

!  test with new and existing data - no regularization

     p%new_problem_structure = .TRUE.
     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      4.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  dense by column storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 4.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      0.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!      control%print_level = 101
!      control%print_level = 101
!      control%min_diagonal = 0.000000000000001_rp_
       CALL CLLS_solve( p, data, control, info )
       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1, 'N', I1,':', I6,' iterations. Optimal objective',    &
        &   ' value = ', F7.2, ' status = ', I6 )" ) st, i, info%iter,         &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, 'N', I1,': CLLS_solve exit status = ', I6 ) " )      &
           st, i, info%status
       END IF
     END DO

!  test with new and existing data - regularization

     p%regularization_weight = 0.01_rp_
!    p%regularization_weight = 1.0_rp_
     p%new_problem_structure = .TRUE.
     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      4.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  dense by column storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 4.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      0.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL CLLS_solve( p, data, control, info )
       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1, 'R', I1,':', I6,' iterations. Optimal objective',    &
        &   ' value = ', F7.2, ' status = ', I6 )" ) st, i, info%iter,         &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, 'R', I1,': CLLS_solve exit status = ', I6 ) " )      &
           st, i, info%status
       END IF
     END DO

!  test with new and existing data - regularization, scaling and shifts

     ALLOCATE( p%X_s( n ), p%W( o ) )
     p%W = 1.0001_rp_ ; p%X_s = 0.0001_rp_
     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      4.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  dense by column storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 4.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      0.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL CLLS_solve( p, data, control, info )
       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1, 'S', I1,':', I6,' iterations. Optimal objective',    &
        &   ' value = ', F7.2, ' status = ', I6 )" ) st, i, info%iter,         &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, 'S', I1,': CLLS_solve exit status = ', I6 ) " )      &
           st, i, info%status
       END IF
!      STOP
     END DO
!1 continue
     CALL CLLS_terminate( data, control, info )
     IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row )
     IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col )
     IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val )
     IF ( ALLOCATED( p%A%row ) ) DEALLOCATE( p%A%row )
     IF ( ALLOCATED( p%A%col ) ) DEALLOCATE( p%A%col )
     IF ( ALLOCATED( p%A%val ) ) DEALLOCATE( p%A%val )
     DEALLOCATE( p%X_s, p%W )
     p%regularization_weight = 0.0_rp_
!    STOP
   END DO
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%R, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )
!stop
!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; o = 2 ; m = 1 ; ao_ne = 2 ; a_ne = 2
   ALLOCATE( p%B( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )

   p%n = n ; p%o = o ; p%m = m
   p%B = (/ 0.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_ /)
   p%C_u = (/ 1.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( 0 ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL CLLS_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
!  control%print_level = 1
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.

!  test with new and existing data

   tests = 1
   DO i = 0, tests
     IF ( i == 1 ) THEN
       control%feasol = .FALSE.
     END IF

     p%Ao%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL CLLS_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL CLLS_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL CLLS_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   DO i = tests + 1, tests + 1
     p%Ao%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL CLLS_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL CLLS_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL CLLS_initialize( data, control, info )
   control%CRO_control%error = 0
!  control%print_level = 4
   control%infinity = infty
   control%restore_problem = 2
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   DO i = tests + 2, tests + 2
     p%Ao%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL CLLS_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u, STAT = i )
   DEALLOCATE( p%X, STAT = i )
   DEALLOCATE( p%Y, STAT = i )
   DEALLOCATE( p%Z, STAT = i )
   DEALLOCATE( p%C, STAT = i )
   DEALLOCATE( p%R, STAT = i )
   DEALLOCATE( p%C_status, p%X_status, STAT = i )
   DEALLOCATE( p%Ao%ptr, p%A%ptr, STAT = i )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, STAT = i )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, STAT = i )
!stop

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; o = 14 ; m = 17 ; ao_ne = 21 ; a_ne = 46
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%m = m ; p%Ao%ne = ao_ne ; p%A%ne = a_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_,          &
                5.0_rp_, 6.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                &
                8, 9, 10, 11, 12, 13, 14  /)
   p%Ao%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                     &
                8, 9, 10, 11, 12, 13, 14  /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL CLLS_initialize( data, control, info )
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%infinity = infty
   control%restore_problem = 1
   control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
!  control%out = 6 ; control%print_level = 101
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL CLLS_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%R, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )

!  Second problem

   n = 14 ; o = 14 ; m = 17 ; ao_ne = 14 ; a_ne = 46
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%m = m ; p%Ao%ne = ao_ne ; p%A%ne = a_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%Ao%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL CLLS_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CLLS_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%R, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )

!  Third problem

   n = 14 ; o = 14 ; m = 17 ; ao_ne = 14 ; a_ne = 46
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%m = m ; p%Ao%ne = ao_ne ; p%A%ne = a_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%Ao%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL CLLS_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CLLS_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%Ao%type )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%R, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )

!  Fourth and Fifth problems

   n = 14 ; o = 14 ; m = 10 ; ao_ne = 14 ; a_ne = 32
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%m = m ; p%Ao%ne = ao_ne ; p%A%ne = a_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - infty, - infty, - infty, - infty, - infty, - infty, - infty,   &
              - infty, - infty, - infty, - infty, - infty, - infty, - infty  /)
   p%X_u = - p%X_l
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                 4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,         &
                 7.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%Ao%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13 /)
   p%C_l = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = p%C_l

   CALL CLLS_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CLLS_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CLLS_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': CLLS_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL CLLS_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%Ao%type )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%R, p%C_status, p%X_status )
   DEALLOCATE( p%Ao%ptr, p%A%ptr )

!  =================================================================
!  pounce tests with active constraints and bounds, some of which are
!  degenerate (active with zero multipliers), which indicators of
!  type 1 or 3 may then flag with status +/-2
!  =================================================================

!  problem 1: minimize 1/2||x - e||^2 s.t. x_1 + x_2 <= 2 and x_3 <= 0,
!             with solution x = (1,1,0), y = (0,-1) and z = 0
!  problem 2: minimize 1/2||x - e||^2 s.t. x_1 + x_2 <= 2, x_1 <= 1 and
!             x_3 <= 0, with solution x = (1,1,0), y = 0 and z = (0,0,-1)
!  problem 3: minimize 1/2||x - e||^2 s.t. x_1 + x_2 <= 1 and x_3 <= 0,
!             with solution x = (1/2,1/2,0), y = -1/2 and z = (0,0,-1)
!  problem 4: minimize 1/2||x - e||^2 s.t. x_1 <= 1 and x_3 <= 0, with
!             solution x = (1,1,0) and z = (0,0,-1)

!  each is solved with indicators of types 1, 2 and 3, and of type 1 with
!  a larger tolerance (case 4), using both the reduced and full systems

   WRITE( 6, "( /, ' pounce tests with active constraints and bounds ', /)")
   tol = EPSILON( 1.0_rp_ ) ** 0.75
   errors = 0
   n = 3 ; o = 3
   DO prob = 1, 4
     IF ( prob == 1 ) THEN
       m = 2 ; a_ne = 3
       x_sol = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_ /)
       y_sol = (/ 0.0_rp_, - 1.0_rp_ /) ; z_sol = 0.0_rp_
     ELSE IF ( prob == 2 ) THEN
       m = 1 ; a_ne = 2
       x_sol = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_ /)
       y_sol = 0.0_rp_ ; z_sol = (/ 0.0_rp_, 0.0_rp_, - 1.0_rp_ /)
     ELSE IF ( prob == 3 ) THEN
       m = 1 ; a_ne = 2
       x_sol = (/ 0.5_rp_, 0.5_rp_, 0.0_rp_ /)
       y_sol = - 0.5_rp_ ; z_sol = (/ 0.0_rp_, 0.0_rp_, - 1.0_rp_ /)
     ELSE
       m = 0 ; a_ne = 0
       x_sol = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_ /)
       z_sol = (/ 0.0_rp_, 0.0_rp_, - 1.0_rp_ /)
     END IF
     DO itype = 1, 4

!  for problems 2 to 4, the type 3 indicator predicts that x_3 <= 0 is
!  inactive, since x_3 moves away from its bound in the final iteration;
!  the pounce is then rightly rejected, and the solution is only as
!  accurate as the final interior-point iterate

       IF ( prob >= 2 .AND. itype == 3 ) CYCLE
#ifdef REAL_32

!  in single precision, the default stopping tolerance for complementary
!  slackness (relative to its large initial value) allows the iteration to
!  stop far from the solution of problems 3 and 4, where the type 1
!  indicator then fails to predict the active set

       IF ( prob >= 3 .AND. ( itype == 1 .OR. itype == 4 ) ) CYCLE
#endif
       DO ired = 0, 1
         ALLOCATE( p%B( o ), p%R( o ), p%X_l( n ), p%X_u( n ), p%X( n ),       &
                   p%Z( n ), p%X_status( n ), p%C( m ), p%C_l( m ),            &
                   p%C_u( m ), p%Y( m ), p%C_status( m ) )
         ALLOCATE( p%Ao%row( o ), p%Ao%col( o ), p%Ao%val( o ) )
         ALLOCATE( p%A%row( a_ne ), p%A%col( a_ne ), p%A%val( a_ne ) )
         CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
         CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
         p%new_problem_structure = .TRUE.
         p%n = n ; p%o = o ; p%m = m
         p%Ao%ne = o ; p%Ao%row = (/ 1, 2, 3 /) ; p%Ao%col = (/ 1, 2, 3 /)
         p%Ao%val = 1.0_rp_
         p%B = 1.0_rp_
         p%A%ne = a_ne ; p%A%val = 1.0_rp_
         p%C_l = - infty ; p%X_l = - infty
         IF ( prob == 1 ) THEN
           p%A%row = (/ 1, 1, 2 /) ; p%A%col = (/ 1, 2, 3 /)
           p%C_u = (/ 2.0_rp_, 0.0_rp_ /)
           p%X_u = infty
         ELSE IF ( prob == 2 ) THEN
           p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /)
           p%C_u = 2.0_rp_
           p%X_u = (/ 1.0_rp_, infty, 0.0_rp_ /)
         ELSE IF ( prob == 3 ) THEN
           p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /)
           p%C_u = 1.0_rp_
           p%X_u = (/ infty, infty, 0.0_rp_ /)
         ELSE
           p%X_u = (/ 1.0_rp_, infty, 0.0_rp_ /)
         END IF
         p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
         CALL CLLS_initialize( data, control, info )
         control%symmetric_linear_solver = symmetric_linear_solver
         control%FDC_control%symmetric_linear_solver                           &
           = symmetric_linear_solver
         control%infinity = infty
         IF ( itype <= 3 ) THEN
           control%indicator_type = itype
         ELSE
           control%indicator_type = 1
           control%indicator_tol_p = 0.1_rp_
         END IF
         control%reduced_pounce_system = ired == 1
!        control%out = 6 ; control%print_level = 1
         CALL CLLS_solve( p, data, control, info )
         IF ( info%status == 0 ) THEN
           err = MAX( MAXVAL( ABS( p%X - x_sol ) ),                            &
                      MAXVAL( ABS( p%Y - y_sol( : m ) ) ),                     &
                      MAXVAL( ABS( p%Z - z_sol ) ) )
           IF ( err <= tol ) THEN
             WRITE( 6, "( ' P', I1, ' indicator ', I1, ' reduced ', L1,        &
            &             ': solution correct' )" ) prob, itype, ired == 1
           ELSE
             errors = errors + 1
             WRITE( 6, "( ' P', I1, ' indicator ', I1, ' reduced ', L1,        &
            &             ': solution error = ', ES8.1, ' FAILED' )" )         &
               prob, itype, ired == 1, err
           END IF
         ELSE
           errors = errors + 1
           WRITE( 6, "( ' P', I1, ' indicator ', I1, ' reduced ', L1,          &
          &             ': CLLS_solve exit status = ', I0, ' FAILED' )" )      &
             prob, itype, ired == 1, info%status
         END IF
         CALL CLLS_terminate( data, control, info )
         DEALLOCATE( p%B, p%R, p%X_l, p%X_u, p%X, p%Z, p%X_status, p%C,        &
                     p%C_l, p%C_u, p%Y, p%C_status )
         DEALLOCATE( p%Ao%row, p%Ao%col, p%Ao%val, p%Ao%type )
         DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )
       END DO
     END DO
   END DO
   IF ( errors > 0 ) THEN
     WRITE( 6, "( /, 1X, I0, ' pounce tests failed' )" ) errors
     STOP 1
   END IF

   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_CLLS_EXAMPLE
