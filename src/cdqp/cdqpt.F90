! THIS VERSION: GALAHAD 5.6 - 2026-11-22 AT 11:14 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_CDQP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_CDQP_precision
   USE GALAHAD_SYMBOLS
   USE GALAHAD_LMS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( CDQP_data_type ) :: data
   TYPE ( CDQP_control_type ) :: control
   TYPE ( CDQP_inform_type ) :: inform
   TYPE ( LMS_control_type ) :: LMS_control
   TYPE ( LMS_inform_type ) :: LMS_inform
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   REAL ( KIND = rp_ ) :: delta, err, f, f_sol
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S, Y

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

     CALL CDQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 1

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
     p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
     p%H%col = (/ 1, 2, 3, 1 /)
     p%H%ptr = (/ 1, 2, 3, 5 /)
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
       control%CQP_control%stop_abs_c = 0.0_rp_
       control%CQP_control%stop_rel_c = 0.0_rp_
!      p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
!      p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%CQP_control%maxit = 0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
       p%X( 2 ) = 100000000.0_rp_
!      control%print_level = 1
!      control%maxit = 1
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN
       p%H%col( 1 ) = 2
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1

     CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) status, inform%CQP_inform%iter,  &
                      inform%DQP_inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "(I2, ': CDQP_solve exit status = ', I6 )" )                  &
          status, inform%status
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL CDQP_terminate( data, control, inform )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP
   END DO
   CALL CDQP_terminate( data, control, inform )
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
   p%f = 0.0_rp_
   p%G = (/ - 1.0_rp_ /)
!  p%X_l = (/ - infty /)
   p%X_l = (/ 0.0_rp_ /)
   p%X_u = (/ infty /)
   p%H%val = (/ 0.0_rp_ /)
   p%H%row = (/ 1 /)
   p%H%col = (/ 1 /)
   CALL CDQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%infinity = 0.1_rp_ * infty
   control%restore_problem = 1
!  control%print_level = 1
!  control%print_level = 4
!  control%stop_d = EPSILON( 1.0_rp_ )
!  control%SBLS_control%print_level = 4

!   control%maxit = 3
!  control%puiseux = .TRUE.
!  control%puiseux = .FALSE.
!  control%series_order = 2
!  control%every_order = .FALSE.

   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) status, inform%CQP_inform%iter,  &
                      inform%DQP_inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "(I2, ': CDQP_solve exit status = ', I6 )" )                  &
         status, inform%status
   END IF
   CALL CDQP_terminate( data, control, inform )
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

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

!  DO data_storage_type = -3, -3
!  DO data_storage_type = -7, -7
!  DO data_storage_type = -7, 0
   DO data_storage_type = -6, 0
     CALL CDQP_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 2
!    control%out = 6 ; control%print_level = 1
!    control%SBLS_control%print_level = 4
!    control%SBLS_control%SLS_control%print_level = 3
!    control%SBLS_control%SLS_control%print_level_solver = 2
! control%SBLS_control%preconditioner = 3
!  control%SBLS_control%factorization = 2
! control%SBLS_control%itref_max = 2
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
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(n*(n+1)/2))
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
       CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )

       IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A1, I1, ':(', 2I6, ') iterations.',                        &
     &      ' Optimal objective value = ', F6.1, ' status = ', I6 )" ) st, i,  &
          inform%CQP_inform%iter, inform%DQP_inform%iter, inform%obj,          &
          inform%status
       ELSE
         WRITE( 6, "( A1, I1,': CDQP_solve exit status = ', I6 ) " )           &
           st, i, inform%status
       END IF
!      STOP
     END DO
     CALL CDQP_terminate( data, control, inform )
     IF ( ALLOCATED( p%H%row ) ) DEALLOCATE( p%H%row )
     IF ( ALLOCATED( p%H%col ) ) DEALLOCATE( p%H%col )
     IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( data_storage_type == - 7 )                                           &
       CALL LMS_terminate( p%H_lm, LMS_control, LMS_inform )
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
   CALL CDQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
!  control%sbls_control%print_level = 1

!  test with new and existing data

   tests = 2
   DO i = 0, tests
     IF ( i == 1 ) THEN
       B_stat = 0 ; C_stat = 0 ; B_stat( 1 ) = - 1
     ELSE IF ( i == 2 ) THEN
       B_stat = 0 ; C_stat = 0 ; C_stat( 1 ) = - 1
     END IF

     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) i, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) i, inform%status
     END IF
   END DO
   CALL CDQP_terminate( data, control, inform )

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
   CALL CDQP_initialize( data, control, inform )
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
     CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) i, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) i, inform%status
     END IF
   END DO
   CALL CDQP_terminate( data, control, inform )

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
   CALL CDQP_initialize( data, control, inform )
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
     CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) i, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
!      write(6,*) inform%obj
     ELSE
       WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) i, inform%status
     END IF
   END DO
   CALL CDQP_terminate( data, control, inform )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )

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
   ALLOCATE( B_stat( n ), C_stat( m ) )
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

   CALL CDQP_initialize( data, control, inform )
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
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   CLOSE( UNIT = scratch_out )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) 1, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) 1, inform%status
   END IF
   CALL CDQP_terminate( data, control, inform )
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
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,          &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
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

   CALL CDQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) 2, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) 2, inform%status
   END IF
   CALL CDQP_terminate( data, control, inform )
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
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,          &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
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

   CALL CDQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
!  C_stat( 10 ) = 1 ; C_stat( 11 ) = 1
!  C_stat( 12 ) = 1 ; C_stat( 13 ) = 1
!  C_stat( 14 ) = 1 ; C_stat( 15 ) = 1
!  C_stat( 16 ) = 1 ; C_stat( 17 ) = 1
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) 3, inform%CQP_inform%iter,       &
                      inform%DQP_inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) 3, inform%status
   END IF
   CALL CDQP_terminate( data, control, inform )
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
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - infty, - infty, - infty, - infty, - infty, - infty, - infty,   &
              - infty, - infty, - infty, - infty, - infty, - infty, - infty  /)
   p%X_u = - p%X_l
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,          &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
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

   CALL CDQP_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &                F6.1, ' status = ', I6 )" ) 4, inform%CQP_inform%iter,   &
                      inform%DQP_inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) 4, inform%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':(', 2I6, ') iterations. Optimal objective value = ', &
     &                F6.1, ' status = ', I6 )" ) 5, inform%CQP_inform%iter,   &
                      inform%DQP_inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( I2, ': CDQP_solve exit status = ', I6 ) " ) 5, inform%status
   END IF

   CALL CDQP_terminate( data, control, inform )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )

!  ===============================================================
!  check that CQP's pounce gives the exact solution of the problem
!  minimize 1/2||x - e||^2 s.t. x_1 + x_2 <= 1 and x_3 <= 0, whose
!  solution is x = (1/2,1/2,0), y = -1/2 and z = (0,0,-1)
!  ===============================================================

   WRITE( 6, "( /, ' pounce test ', / )" )
   n = 3 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
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
   CALL CDQP_initialize( data, control, inform )
   control%infinity = infty

!  use LAPACK solvers, as CQP cannot use some of the external ones

   control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
   control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
   control%CQP_control%SBLS_pounce_control%symmetric_linear_solver = 'sytr '
   control%CQP_control%SBLS_pounce_control%definite_linear_solver = 'potr '
   control%DQP_control%symmetric_linear_solver = 'sytr '
   control%DQP_control%definite_linear_solver = 'potr '
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == GALAHAD_ok ) THEN
     err = MAX( MAXVAL( ABS( p%X - (/ 0.5_rp_, 0.5_rp_, 0.0_rp_ /) ) ),        &
                ABS( p%Y( 1 ) + 0.5_rp_ ),                                     &
                MAXVAL( ABS( p%Z - (/ 0.0_rp_, 0.0_rp_, - 1.0_rp_ /) ) ) )
     IF ( err > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' pounce test: solution error = ', ES8.1, ' FAILED' )" ) err
       STOP 1
     END IF
     WRITE( 6, "( ' pounce test: solution correct' )" )
   ELSE
     WRITE( 6, "( ' pounce test: CDQP_solve exit status = ', I0, ' FAILED' )" )&
       inform%status
     STOP 1
   END IF
   CALL CDQP_terminate( data, control, inform )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               B_stat, C_stat )
   DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )

!  ===================================================================
!  tests of problems with a zero or singular Hessian: minimize
!  - x_1 - x_2 / 10 - x_3 (+ x_1^2 / 2) s.t. x_1 + x_2 <= 1, x >= 0
!  and x_3 <= 0 (x_3 <= 1). CQP pounces on these problems, so their
!  solutions should be exact
!  ===================================================================

   WRITE( 6, "( /, ' zero and singular Hessian tests ', / )" )
   n = 3 ; m = 1
   DO i = 1, 2
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
     ALLOCATE( B_stat( n ), C_stat( m ) )
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
     CALL CDQP_initialize( data, control, inform )
     control%infinity = infty
     control%CQP_control%SBLS_control%symmetric_linear_solver = 'sytr '
     control%CQP_control%SBLS_control%definite_linear_solver = 'potr '
     control%DQP_control%symmetric_linear_solver = 'sytr '
     control%DQP_control%definite_linear_solver = 'potr '
     CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
     f = DOT_PRODUCT( p%G, p%X ) + 0.5_rp_ * SUM( p%H%val( : p%H%ne ) )     &
           * p%X( 1 ) ** 2
     IF ( inform%status /= GALAHAD_ok ) THEN
       WRITE( 6, "( ' case ', I0, ': CDQP_solve exit status = ', I0,          &
      &             ' FAILED' )" ) i, inform%status
       STOP 1
     ELSE IF ( ABS( f - f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' case ', I0, ': objective error = ', ES8.1,              &
      &             ' FAILED' )" ) i, ABS( f - f_sol )
       STOP 1
     ELSE
       WRITE( 6, "( ' case ', I0, ': solution correct' )" ) i
     END IF
     CALL CDQP_terminate( data, control, inform )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 B_stat, C_stat )
     DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )
   END DO


!  ===================================================================
!  check that the problem minimize 1/2 x^T H x + g^T x s.t. x_1 + x_2 <= 1
!  and x_3 <= 0, with H = ( 2 1 0 ; 1 2 0 ; 0 0 1 ) and g = - (5/2,5/2,1),
!  whose solution is x = (1/2,1/2,0), is solved using the default linear
!  solvers (the other tests use those set by WHICH_sls). H is not diagonal,
!  so it has to be factorized
!  ===================================================================

   WRITE( 6, "( /, ' default solver test ', / )" )
   n = 3 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_
   p%G = (/ - 2.5_rp_, - 2.5_rp_, - 1.0_rp_ /)
   p%X_l = - infty ; p%X_u = (/ infty, infty, 0.0_rp_ /)
   p%C_l = - infty ; p%C_u = 1.0_rp_
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   IF ( ALLOCATED( p%H%row ) ) DEALLOCATE( p%H%row )
   IF ( ALLOCATED( p%H%col ) ) DEALLOCATE( p%H%col )
   IF ( ALLOCATED( p%H%ptr ) ) DEALLOCATE( p%H%ptr )
   IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat ) ; p%H%ne = 4
   ALLOCATE( p%H%row( 4 ), p%H%col( 4 ), p%H%val( 4 ) )
   p%H%row = (/ 1, 2, 2, 3 /) ; p%H%col = (/ 1, 1, 2, 3 /)
   p%H%val = (/ 2.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_ /)
   IF ( ALLOCATED( p%A%row ) ) DEALLOCATE( p%A%row )
   IF ( ALLOCATED( p%A%col ) ) DEALLOCATE( p%A%col )
   IF ( ALLOCATED( p%A%ptr ) ) DEALLOCATE( p%A%ptr )
   IF ( ALLOCATED( p%A%val ) ) DEALLOCATE( p%A%val )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 2
   ALLOCATE( p%A%row( 2 ), p%A%col( 2 ), p%A%val( 2 ) )
   p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /) ; p%A%val = 1.0_rp_
   CALL CDQP_initialize( data, control, inform )
   control%infinity = infty
   CALL CDQP_solve( p, data, control, inform, C_stat, B_stat )
   IF ( inform%status == GALAHAD_ok ) THEN
     err = MAXVAL( ABS( p%X - (/ 0.5_rp_, 0.5_rp_, 0.0_rp_ /) ) )
     IF ( err > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' default solver test: solution error = ', ES8.1,         &
      &             ' FAILED' )" ) err
       STOP 1
     END IF
     WRITE( 6, "( ' default solver test: solution correct' )" )
   ELSE
     WRITE( 6, "( ' default solver test: CDQP_solve exit status = ', I0,      &
    &             ' FAILED' )" ) inform%status
     STOP 1
   END IF
   CALL CDQP_terminate( data, control, inform )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               B_stat, C_stat )
   DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( CDQP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver                               &
       = symmetric_linear_solver
     control%CQP_control%FDC_control%use_sls = use_sls
     control%CQP_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%CQP_control%SBLS_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%CQP_control%SBLS_control%definite_linear_solver                   &
        = definite_linear_solver
     control%DQP_control%FDC_control%use_sls = use_sls
     control%DQP_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%DQP_control%symmetric_linear_solver = symmetric_linear_solver
     control%DQP_control%definite_linear_solver = definite_linear_solver
     control%CRO_control%SBLS_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%CRO_control%SBLS_control%definite_linear_solver                   &
        = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_CDQP_EXAMPLE
