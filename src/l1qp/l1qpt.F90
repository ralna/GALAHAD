! THIS VERSION: GALAHAD 5.6 - 2026-11-26 AT 11:16 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_L1QP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_L1QP_precision
   USE GALAHAD_SYMBOLS
   USE GALAHAD_LMS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( L1QP_data_type ) :: data
   TYPE ( L1QP_control_type ) :: control
   TYPE ( L1QP_inform_type ) :: info
   TYPE ( LMS_control_type ) :: LMS_control
   TYPE ( LMS_inform_type ) :: LMS_inform
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   REAL ( KIND = rp_ ) :: delta, err, f, f_sol
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   INTEGER ( KIND = ip_ ) :: lp_trial, lp_errors
   INTEGER ( KIND = i8_ ) :: lp_seed
   REAL ( KIND = rp_ ) :: lp_f, lp_f_sol
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S, Y

!go to 77
!go to 111
   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )

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

     CALL L1QP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 1
! control%print_level = 1

!control%print_level = 3
!control%CQP_control%maxit = 2
!control%CQP_control%SBLS_control%print_level = 3
!control%CQP_control%SBLS_control%preconditioner = 3

     p%new_problem_structure = .TRUE.
     p%n = n ; p%m = m ; p%f = 1.0_rp_
     p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
     p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
     p%C_u = (/ 4.0_rp_, infty /)
     p%X_l = (/ - 1.0_rp_, - infty, - infty /)
     p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

     ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
     CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
     p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_ /)
     p%H%col = (/ 1, 1, 2, 3 /)
     p%H%ptr = (/ 1, 2, 4, 5 /)
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
       control%CQP_control%maxit = 0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = -1.0
       p%X( 2 ) = 100000000.0_rp_
!      control%print_level = 1
!      control%CQP_control%maxit = 1
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN
       p%H%col( 1 ) = 2
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1

     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%CQP_inform%iter,          &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': L1QP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL L1QP_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP
   END DO
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
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
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 0.0_rp_
   p%G = (/ - 1.0_rp_ /)
!  p%X_l = (/ - infty /)
   p%X_l = (/ 0.0_rp_ /)
   p%X_u = (/ infty /)
   p%H%val = (/ 0.0_rp_ /)
   p%H%row = (/ 1 /)
   p%H%col = (/ 1 /)
   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = 0.1_rp_ * infty
   control%restore_problem = 1
!  control%print_level = 1
!  control%print_level = 4
!  control%stop_d = EPSILON( 1.0_rp_ )
!  control%CQP_control%SBLS_control%print_level = 4

!  control%CQP_control%maxit = 3
!  control%puiseux = .TRUE.
!  control%puiseux = .FALSE.
!  control%series_order = 2
!  control%every_order = .FALSE.

   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &          F6.1, ' status = ', I6 )" ) status, info%CQP_inform%iter,      &
                  info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': L1QP_solve exit status = ', I6 )") status, info%status
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  stop
!111 continue
!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

!  DO data_storage_type = -3, -3
!  DO data_storage_type = -7, -7
   DO data_storage_type = -7, 0
     CALL L1QP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 2
!    control%out = 6 ; control%print_level = 1
!    control%CQP_control%SBLS_control%print_level = 4
!    control%CQP_control%SBLS_control%SLS_control%print_level = 3
!    control%CQP_control%SBLS_control%SLS_control%print_level_solver = 2
!    control%CQP_control%SBLS_control%preconditioner = 3
!    control%CQP_control%SBLS_control%factorization = 2
!    control%CQP_control%SBLS_control%itref_max = 2
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
       p%H%row = (/ 1, 2, 2, 3 /)
       p%H%col = (/ 1, 1, 2, 3 /) ; p%H%ne = h_ne
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
       p%H%col = (/ 1, 1, 2, 3 /)
       p%H%ptr = (/ 1, 2, 4, 5 /)
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 2, 3 /)
       p%A%ptr = (/ 1, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(0))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'G'
       ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 4 ) THEN      ! scaled identity H, dense A
       st = 'S'
       ALLOCATE( p%H%val(1), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SCALED_IDENTITY', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 5 ) THEN      ! identity H, dense A
       st = 'I'
       ALLOCATE( p%H%val(0), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'IDENTITY', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 6 ) THEN      ! no H, dense A
       st = 'N'
!      ALLOCATE( p%H%val(0), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'NONE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 7 ) THEN      ! no H, dense A
       st = 'B'
!      ALLOCATE( p%H%val(0), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'LBFGS', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       CALL LMS_initialize( p%H_lm, LMS_control, LMS_inform )
       LMS_control%memory_length = 2
       LMS_control%method = 1
       CALL LMS_setup( n, p%H_lm, LMS_control, LMS_inform )
       ALLOCATE( S( p%n ), Y( p%n ) )
       DO i = 1, p%n + 2
         S = 1.0_rp_
         S( 1 ) = REAL( MOD( i, p%n ) + 1, KIND = rp_ )
         Y = S
         delta = 1.0_rp_ / S( 1 )
!         S = 0.0_rp_
!         S( MOD( i - 1, p%n ) + 1 ) = 1.0_rp_
!         Y = S
!         delta = REAL( MOD( i, 3 ) + 1, KIND = rp_ )
         CALL LMS_form( S, Y, delta, p%H_lm, LMS_control, LMS_inform )
       END DO
       DEALLOCATE( S, Y )
     END IF

!  test with new and existing data

     DO i = 1, 2
!    DO i = 1, 1
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 4 ) THEN    ! scaled I/dense storage
         p%H%val( 1 ) = 2.0_rp_
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 5 ) THEN    ! identity/dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 6 ) THEN    ! none/dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 7 ) THEN    ! LM/dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL L1QP_solve( p, data, control, info, C_stat, X_stat )

       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) st, i, info%CQP_inform%iter,     &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': L1QP_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
!      STOP
     END DO
     CALL L1QP_terminate( data, control, info )
     IF ( ALLOCATED( p%H%row ) ) DEALLOCATE( p%H%row )
     IF ( ALLOCATED( p%H%col ) ) DEALLOCATE( p%H%col )
     IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( data_storage_type == - 7 )                                           &
       CALL LMS_terminate( p%H_lm, LMS_control, LMS_inform )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )
!stop
!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.05_rp_
   p%G = (/ 0.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_ /)
   p%C_u = (/ 1.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

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
   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!    control%out = 6 ; control%print_level = 1
!    control%CQP_control%print_level = 5
!    control%CQP_control%FDC_control%print_level = 1
!    control%CQP_control%SBLS_control%print_level = 1
!    control%DQP_control%print_level = 1
!    control%DQP_control%SBLS_control%print_level = 1

!  test with new and existing data

   tests = 17
   DO i = 0, tests
     IF ( i == 0 ) THEN
       cycle
!      control%precon = 0
     ELSE IF ( i == 1 ) THEN
       cycle
!      control%precon = 1
     ELSE IF ( i == 2 ) THEN
       cycle
!      control%precon = 2
     ELSE IF ( i == 3 ) THEN
       cycle
!      control%precon = 3
     ELSE IF ( i == 4 ) THEN
       cycle
!      control%precon = 5
     ELSE IF ( i == 5 ) THEN
       control%CQP_control%SBLS_control%factorization = - 1
     ELSE IF ( i == 6 ) THEN
       control%CQP_control%SBLS_control%factorization = 1
     ELSE IF ( i == 7 ) THEN
       control%CQP_control%SBLS_control%max_col = 0
     ELSE IF ( i == 8 ) THEN
       control%CQP_control%SBLS_control%factorization = 2
!      control%precon = 0
     ELSE IF ( i == 9 ) THEN
       cycle
!      control%print_level = 2
!      control%precon = 1
     ELSE IF ( i == 10 ) THEN
       cycle
!      control%precon = 2
     ELSE IF ( i == 11 ) THEN
       cycle
!      control%precon = 3
     ELSE IF ( i == 12 ) THEN
       cycle
!      control%precon = 5
     ELSE IF ( i == 13 ) THEN
       cycle
     ELSE IF ( i == 14 ) THEN
       cycle
     ELSE IF ( i == 15 ) THEN
       control%CQP_control%feasol = .FALSE.
     ELSE IF ( i == 16 ) THEN
       X_stat = 0 ; C_stat = 0 ; X_stat( 1 ) = - 1
     ELSE IF ( i == 17 ) THEN
       X_stat = 0 ; C_stat = 0 ; C_stat( 1 ) = - 1
     END IF

     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%CQP_inform%iter,     &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL L1QP_terminate( data, control, info )

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
   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%CQP_inform%iter,     &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL L1QP_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2 /)
   p%H%ptr = (/ 1, 2, 3 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%CRO_control%error = 0
!  control%print_level = 4
   control%infinity = infty
   control%restore_problem = 2
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%CQP_inform%iter,     &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
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
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
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
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_, &
                4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_, 7.0_rp_, 7.0_rp_, &
                8.0_rp_, 9.0_rp_, 10.0_rp_, 11.0_rp_, 12.0_rp_, 13.0_rp_,      &
                14.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14  /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
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

   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1
   control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%CQP_inform%iter,     &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Second problem

   n = 14 ; m = 17 ; h_ne = 21 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
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
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_, &
                4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_, 7.0_rp_, 7.0_rp_, &
                8.0_rp_, 9.0_rp_, 10.0_rp_, 11.0_rp_, 12.0_rp_, 13.0_rp_,      &
                14.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14  /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
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

   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%CQP_inform%iter,     &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Third problem

   n = 14 ; m = 17 ; h_ne = 21 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
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
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_, &
                4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_, 7.0_rp_, 7.0_rp_, &
                8.0_rp_, 9.0_rp_, 10.0_rp_, 11.0_rp_, 12.0_rp_, 13.0_rp_,      &
                14.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14  /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
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

   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
!  C_stat( 10 ) = 1 ; C_stat( 11 ) = 1
!  C_stat( 12 ) = 1 ; C_stat( 13 ) = 1
!  C_stat( 14 ) = 1 ; C_stat( 15 ) = 1
!  C_stat( 16 ) = 1 ; C_stat( 17 ) = 1
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%CQP_inform%iter,     &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  Fourth and Fifth problems

!77 continue
   n = 14 ; m = 10 ; h_ne = 21 ; a_ne = 32
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - infty, - infty, - infty, - infty, - infty, - infty, - infty,   &
              - infty, - infty, - infty, - infty, - infty, - infty, - infty  /)
   p%X_u = - p%X_l
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_, &
                4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_, 7.0_rp_, 7.0_rp_, &
                8.0_rp_, 9.0_rp_, 10.0_rp_, 11.0_rp_, 12.0_rp_, 13.0_rp_,      &
                14.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                8, 9, 10, 11, 12, 13, 14  /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
                8, 9, 10, 11, 12, 13, 14  /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 5,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 12 /)
   p%C_l = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = p%C_l

   CALL L1QP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
!  control%out = 6 ; control%print_level = 1
!  control%CQP_control%print_level = 5
!  control%CQP_control%FDC_control%print_level = 1
!  control%CQP_control%SBLS_control%print_level = 1
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%CQP_inform%iter,     &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%CQP_inform%iter,     &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': L1QP_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type, p%H%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  ===============================================================
!  check that CQP's pounce gives the exact solution of the problem
!  minimize 1/2||x - e||^2 s.t. x_1 + x_2 <= 1 and x_3 <= 0, whose
!  solution is x = (1/2,1/2,0), y = -1/2 and z = (0,0,-1)
!  ===============================================================

   WRITE( 6, "( /, ' pounce test ', / )" )
   n = 3 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%G = - 1.0_rp_
   p%X_l = - infty ; p%X_u = (/ infty, infty, 0.0_rp_ /)
   p%C_l = - infty ; p%C_u = 1.0_rp_
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat ) ; p%H%ne = n
   ALLOCATE( p%H%row( n ), p%H%col( n ), p%H%val( n ) )
   p%H%row = (/ 1, 2, 3 /) ; p%H%col = (/ 1, 2, 3 /) ; p%H%val = 1.0_rp_
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 2
   ALLOCATE( p%A%row( 2 ), p%A%col( 2 ), p%A%val( 2 ) )
   p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /) ; p%A%val = 1.0_rp_
   CALL L1QP_initialize( data, control, info )
   control%infinity = infty

!  use LAPACK solvers, as CQP cannot use some of the external ones

   control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
   control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
   control%CQP_control%SBLS_pounce_control%symmetric_linear_solver = 'sytr '
   control%CQP_control%SBLS_pounce_control%definite_linear_solver = 'potr '
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == GALAHAD_ok ) THEN
     err = MAX( MAXVAL( ABS( p%X - (/ 0.5_rp_, 0.5_rp_, 0.0_rp_ /) ) ),        &
                ABS( p%Y( 1 ) + 0.5_rp_ ),                                     &
                MAXVAL( ABS( p%Z - (/ 0.0_rp_, 0.0_rp_, - 1.0_rp_ /) ) ) )
     IF ( err > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' pounce test: solution error = ', ES8.1, ' FAILED' )" ) err
       STOP 1
     END IF
     WRITE( 6, "( ' pounce test: solution correct' )" )
   ELSE
     WRITE( 6, "( ' pounce test: L1QP_solve exit status = ', I0, ' FAILED' )" )&
       info%status
     STOP 1
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )


!  ==================================================================
!  check that an LP solved with refinement (for which CQP is given a
!  diagonal Hessian, and does not pounce) runs to completion: minimize
!  - x_1 - x_2 - x_3 s.t. x_1 + x_2 <= 1, x >= 0 and x_3 <= 0
!  ==================================================================

   WRITE( 6, "( /, ' LP refinement test ', / )" )
   n = 3 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%G = - 1.0_rp_
   p%X_l = 0.0_rp_ ; p%X_u = (/ infty, infty, 0.0_rp_ /)
   p%C_l = - infty ; p%C_u = 1.0_rp_
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL SMT_put( p%H%type, 'ZERO', smt_stat )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 2
   ALLOCATE( p%A%row( 2 ), p%A%col( 2 ), p%A%val( 2 ) )
   p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /) ; p%A%val = 1.0_rp_
   CALL L1QP_initialize( data, control, info )
   control%infinity = infty ; control%refine = .TRUE.
   control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
   control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
   control%DQP_control%symmetric_linear_solver = 'sytr '
   control%DQP_control%definite_linear_solver = 'potr '
   CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == GALAHAD_ok ) THEN
     WRITE( 6, "( ' LP refinement test: solved' )" )
   ELSE
     WRITE( 6, "( ' LP refinement test: L1QP_solve exit status = ', I0,       &
    &             ' FAILED' )" ) info%status
     STOP 1
   END IF
   CALL L1QP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val )

!  ===================================================================
!  tests of problems with a zero or singular Hessian: minimize
!  - x_1 - x_2 / 10 - x_3 (+ x_1^2 / 2) s.t. x_1 + x_2 <= 1, x >= 0
!  and x_3 <= 0 (x_3 <= 1). CQP pounces on these problems, so their
!  solutions should be exact
!  ===================================================================

   WRITE( 6, "( /, ' zero and singular Hessian tests ', / )" )
   n = 3 ; m = 1
   DO i = 1, 1
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
     ALLOCATE( X_stat( n ), C_stat( m ) )
     p%new_problem_structure = .TRUE.
     p%n = n ; p%m = m ; p%f = 0.0_rp_
     p%G = (/ - 1.0_rp_, - 0.1_rp_, - 1.0_rp_ /)
     p%X_l = 0.0_rp_ ; p%C_l = - infty ; p%C_u = 1.0_rp_
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 2
     ALLOCATE( p%A%row( 2 ), p%A%col( 2 ), p%A%val( 2 ) )
     p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /) ; p%A%val = 1.0_rp_
     IF ( ALLOCATED( p%H%row ) ) DEALLOCATE( p%H%row )
     IF ( ALLOCATED( p%H%col ) ) DEALLOCATE( p%H%col )
     IF ( ALLOCATED( p%H%ptr ) ) DEALLOCATE( p%H%ptr )
     IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val )
     CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
     IF ( i == 1 ) THEN

!  an LP, with a Hessian with no entries; the solution is x = (1,0,0)

       p%H%ne = 0
       ALLOCATE( p%H%row( 0 ), p%H%col( 0 ), p%H%val( 0 ) )
       p%X_u = (/ infty, infty, 0.0_rp_ /)
       f_sol = - 1.0_rp_
     ELSE

!  a QP with positive semi-definite Hessian diag(1,0,0); the solution is
!  x = (9/10,1/10,1)

       p%H%ne = 1
       ALLOCATE( p%H%row( 1 ), p%H%col( 1 ), p%H%val( 1 ) )
       p%H%row = 1 ; p%H%col = 1 ; p%H%val = 1.0_rp_
       p%X_u = (/ infty, infty, 1.0_rp_ /)
       f_sol = - 1.505_rp_
     END IF
     CALL L1QP_initialize( data, control, info )
     control%infinity = infty
     control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
     control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
     control%DQP_control%symmetric_linear_solver = 'sytr '
     control%DQP_control%definite_linear_solver = 'potr '

!  refine the solution of the LP; the user's Hessian must be unchanged

     control%refine = .TRUE.
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
     f = DOT_PRODUCT( p%G, p%X ) + 0.5_rp_ * SUM( p%H%val( : p%H%ne ) )     &
           * p%X( 1 ) ** 2
     IF ( info%status /= GALAHAD_ok ) THEN
       WRITE( 6, "( ' case ', I0, ': L1QP_solve exit status = ', I0,          &
      &             ' FAILED' )" ) i, info%status
       STOP 1
     ELSE IF ( ABS( f - f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' case ', I0, ': objective error = ', ES8.1,              &
      &             ' FAILED' )" ) i, ABS( f - f_sol )
       STOP 1
     ELSE IF ( SMT_get( p%H%type ) /= 'COORDINATE' .OR. p%H%ne /= 0 ) THEN
       WRITE( 6, "( ' case ', I0, ': Hessian not restored FAILED' )" ) i
       STOP 1
     ELSE
       WRITE( 6, "( ' case ', I0, ': solution correct' )" ) i
     END IF
     CALL L1QP_terminate( data, control, info )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 X_stat, C_stat )
     DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )
   END DO


!  ======================================================================
!  random LPs with known solutions, solved with refinement, many of which
!  are degenerate, so that their perturbed solutions have constraints or
!  bounds on their bounds but not in the active set
!  ======================================================================

   WRITE( 6, "( /, ' random LP tests ', / )" )
   lp_errors = 0 ; lp_seed = 12345
   DO lp_trial = 1, 6
     n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
     CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
     ALLOCATE( X_stat( n ), C_stat( m ) )
     CALL L1QP_initialize( data, control, info )
     control%refine = .TRUE.
     control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
     control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
     control%DLP_control%symmetric_linear_solver = 'sytr '
     control%DLP_control%definite_linear_solver = 'potr '
     control%infinity = infty
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
     lp_f = DOT_PRODUCT( p%G( : n ), p%X( : n ) )
     IF ( info%status /= GALAHAD_ok ) THEN
       lp_errors = lp_errors + 1
       WRITE( 6, "( ' LP ', I0, ': L1QP_solve exit status = ', I0,         &
      &             ' FAILED' )" ) lp_trial, info%status
     ELSE IF ( ABS( lp_f - lp_f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 *          &
               MAX( 1.0_rp_, ABS( lp_f_sol ) ) ) THEN
       lp_errors = lp_errors + 1
       WRITE( 6, "( ' LP ', I0, ': objective error = ', ES8.1, ' FAILED' )" )  &
         lp_trial, ABS( lp_f - lp_f_sol )
     ELSE
       WRITE( 6, "( ' LP ', I0, ': solution correct' )" ) lp_trial
     END IF
     CALL L1QP_terminate( data, control, info )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 X_stat, C_stat )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   END DO
   IF ( lp_errors > 0 ) THEN
     WRITE( 6, "( /, 1X, I0, ' random LP tests failed' )" ) lp_errors
     STOP 1
   END IF

!  ======================================================================
!  check that the bounds of constraints bounded on both sides are restored
!  after solving an l_1 penalty problem, and that the refinement uses them.
!  For minimize 2 x_1 + x_2 + 2 x_3 subject to 1 <= x_1 + x_2 <= 2,
!  1 <= x_2 + x_3 <= 3 and 0 <= x <= 10, the solution is ( 0, 1, 0 ) for
!  any penalty parameter rho > 2
!  ======================================================================

   WRITE( 6, "( /, ' penalty range constraint tests ', / )" )
   n = 3 ; m = 2
   DO i = 0, 1
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ), X_stat( n ),         &
               C_stat( m ) )
     p%new_problem_structure = .TRUE.
     p%n = n ; p%m = m ; p%f = 0.0_rp_
     p%G = (/ 2.0_rp_, 1.0_rp_, 2.0_rp_ /)
     p%X_l = 0.0_rp_ ; p%X_u = 10.0_rp_
     p%C_l = (/ 1.0_rp_, 1.0_rp_ /) ; p%C_u = (/ 2.0_rp_, 3.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 4
     ALLOCATE( p%A%row( 4 ), p%A%col( 4 ), p%A%val( 4 ) )
     p%A%row = (/ 1, 1, 2, 2 /) ; p%A%col = (/ 1, 2, 2, 3 /)
     p%A%val = 1.0_rp_
     CALL SMT_put( p%H%type, 'ZERO', smt_stat )
     CALL L1QP_initialize( data, control, info )
     control%infinity = infty
     control%rho = 10.0_rp_
     control%refine = i == 1
     control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
     control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
     control%DLP_control%symmetric_linear_solver = 'sytr '
     control%DLP_control%definite_linear_solver = 'potr '
     CALL L1QP_solve( p, data, control, info, C_stat, X_stat )
     IF ( info%status /= GALAHAD_ok ) THEN
       WRITE( 6, "( ' refine = ', L1, ': L1QP_solve exit status = ', I0,      &
      &             ' FAILED' )" ) control%refine, info%status
       STOP 1
     ELSE IF ( ANY( p%C_l( : m ) /= (/ 1.0_rp_, 1.0_rp_ /) ) .OR.             &
               ANY( p%C_u( : m ) /= (/ 2.0_rp_, 3.0_rp_ /) ) ) THEN
       WRITE( 6, "( ' refine = ', L1, ': constraint bounds not restored,',    &
      &             ' c_l = ', 2ES9.1, ' c_u = ', 2ES9.1, ' FAILED' )" )       &
         control%refine, p%C_l( : m ), p%C_u( : m )
       STOP 1
     ELSE IF ( MAXVAL( ABS( p%X( : n ) - (/ 0.0_rp_, 1.0_rp_, 0.0_rp_ /) ) )  &
               > SQRT( EPSILON( 1.0_rp_ ) ) ) THEN
       WRITE( 6, "( ' refine = ', L1, ': wrong solution x = ', 3ES12.4,       &
      &             ' FAILED' )" ) control%refine, p%X( : n )
       STOP 1
     ELSE
       WRITE( 6, "( ' refine = ', L1, ': solution correct' )" ) control%refine
     END IF
     CALL L1QP_terminate( data, control, info )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 X_stat, C_stat )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
     IF ( ALLOCATED( p%A%ptr ) ) DEALLOCATE( p%A%ptr )
   END DO

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

!  generate a random LP, minimize g^T x s.t. c_l <= A x <= c_u and
!  x_l <= x <= x_u, with a known solution x_s, multipliers y_s and dual
!  variables z_s, and optimal value f_s = g^T x_s; about 30% of the variables
!  are free, 40% on a bound and 30% strictly between them, and about 15% of
!  the constraints are equations, 50% active inequalities and 35% inactive

     SUBROUTINE LP_planted( p, n, m, seed, f_s )
     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: p
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
     INTEGER ( KIND = i8_ ), INTENT( INOUT ) :: seed
     REAL ( KIND = rp_ ), INTENT( OUT ) :: f_s
     INTEGER ( KIND = ip_ ) :: i, j, k, l, nnzr, s
     REAL ( KIND = rp_ ) :: r
     INTEGER ( KIND = ip_ ), DIMENSION( MIN( n, 5 ) ) :: cols
     REAL ( KIND = rp_ ), DIMENSION( n ) :: x_s, z_s
     REAL ( KIND = rp_ ), DIMENSION( m ) :: y_s, c_s
     nnzr = MIN( n, 5_ip_ )
     p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%new_problem_structure = .TRUE.
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
     ALLOCATE( p%A%row( m * nnzr ), p%A%col( m * nnzr ), p%A%val( m * nnzr ) )
     l = 0
     DO i = 1, m
       k = 0
       DO WHILE ( k < nnzr )
         j = MIN( n, 1 + INT( uniform( seed ) * n, ip_ ) )
         IF ( ANY( cols( : k ) == j ) ) CYCLE
         k = k + 1 ; cols( k ) = j
         l = l + 1 ; p%A%row( l ) = i ; p%A%col( l ) = j
         p%A%val( l ) = 2.0_rp_ * uniform( seed ) - 1.0_rp_
       END DO
     END DO
     CALL SMT_put( p%A%type, 'COORDINATE', s ) ; p%A%ne = l
     CALL SMT_put( p%H%type, 'ZERO', s )
     DO j = 1, n
       x_s( j ) = 2.0_rp_ * uniform( seed ) - 1.0_rp_
     END DO
     DO j = 1, n
       r = uniform( seed ) ; z_s( j ) = 0.0_rp_
       IF ( r < 0.3_rp_ ) THEN
         p%X_l( j ) = - infty ; p%X_u( j ) = infty
       ELSE IF ( r < 0.5_rp_ ) THEN
         p%X_l( j ) = x_s( j ) ; p%X_u( j ) = x_s( j ) + 1.0_rp_
         z_s( j ) = 0.1_rp_ + uniform( seed )
       ELSE IF ( r < 0.7_rp_ ) THEN
         p%X_u( j ) = x_s( j ) ; p%X_l( j ) = x_s( j ) - 1.0_rp_
         z_s( j ) = - 0.1_rp_ - uniform( seed )
       ELSE
         p%X_l( j ) = x_s( j ) - 0.1_rp_ - uniform( seed )
         p%X_u( j ) = x_s( j ) + 0.1_rp_ + uniform( seed )
       END IF
     END DO
     c_s = 0.0_rp_
     DO l = 1, p%A%ne
       c_s( p%A%row( l ) ) = c_s( p%A%row( l ) )                              &
         + p%A%val( l ) * x_s( p%A%col( l ) )
     END DO
     DO i = 1, m
       r = uniform( seed ) ; y_s( i ) = 0.0_rp_
       IF ( r < 0.15_rp_ ) THEN
         p%C_l( i ) = c_s( i ) ; p%C_u( i ) = c_s( i )
         y_s( i ) = 2.0_rp_ * uniform( seed ) - 1.0_rp_
       ELSE IF ( r < 0.4_rp_ ) THEN
         p%C_l( i ) = c_s( i ) ; p%C_u( i ) = infty
         y_s( i ) = 0.1_rp_ + uniform( seed )
       ELSE IF ( r < 0.65_rp_ ) THEN
         p%C_u( i ) = c_s( i ) ; p%C_l( i ) = - infty
         y_s( i ) = - 0.1_rp_ - uniform( seed )
       ELSE
         p%C_l( i ) = c_s( i ) - 0.1_rp_ - uniform( seed )
         p%C_u( i ) = c_s( i ) + 0.1_rp_ + uniform( seed )
       END IF
     END DO

!  g = A^T y_s + z_s

     p%G = z_s
     DO l = 1, p%A%ne
       p%G( p%A%col( l ) ) = p%G( p%A%col( l ) )                              &
         + p%A%val( l ) * y_s( p%A%row( l ) )
     END DO
     f_s = DOT_PRODUCT( p%G, x_s )
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     END SUBROUTINE LP_planted

!  a uniform pseudo-random number in [0,1)

     REAL ( KIND = rp_ ) FUNCTION uniform( seed )
     INTEGER ( KIND = i8_ ), INTENT( INOUT ) :: seed
     seed = MOD( 1103515245_i8_ * seed + 12345_i8_, 2147483648_i8_ )
     uniform = REAL( seed, KIND = rp_ ) / 2147483648.0_rp_
     END FUNCTION uniform
     SUBROUTINE WHICH_sls( control )
     TYPE ( L1QP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%CQP_control%FDC_control%use_sls = use_sls
     control%CQP_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%CQP_control%SBLS_control%symmetric_linear_solver                  &
        = symmetric_linear_solver
     control%CQP_control%SBLS_control%definite_linear_solver                   &
        = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_L1QP_EXAMPLE
