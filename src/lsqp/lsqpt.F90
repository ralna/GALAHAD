! THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 09:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LSQP_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LSQP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LSQP_data_type ) :: data
   TYPE ( LSQP_control_type ) :: control
   TYPE ( LSQP_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: i, n, m, a_ne, Hessian_kind, data_storage_type
   INTEGER ( KIND = ip_ ) :: status, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: scratch_out = 56
   REAL ( KIND = rp_ ) :: stop_c
   CHARACTER ( len = 1 ) :: st

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )

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
!    IF ( status == - GALAHAD_error_ill_conditioned ) CYCLE
     IF ( status == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( status == - GALAHAD_error_max_iterations ) CYCLE
     IF ( status == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL LSQP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 1

     p%new_problem_structure = .TRUE.
     p%gradient_kind = 0
     p%Hessian_kind = 0
     p%n = n ; p%m = m ; p%f = 1.0_rp_
     p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
     p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
     p%C_u = (/ 4.0_rp_, infty /)
     p%X_l = (/ - 1.0_rp_, - infty, - infty /)
     p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
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
     ELSE IF ( status == - GALAHAD_error_ill_conditioned ) THEN
!      control%print_level = 1
       control%stop_c = EPSILON( 1.0_rp_ ) ** 4
       p%Hessian_kind = 1
       ALLOCATE( p%X0( n ) )
       p%X0 = (/  -2.0_rp_, 1.0_rp_, 3.0_rp_ /)
       p%X_l = (/ - 1.0_rp_, - infty, - infty /)
       p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 1
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
!      control%print_level = 1
     ELSE
     END IF

     CALL LSQP_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': LSQP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL LSQP_terminate( data, control, info )
   END DO
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%ptr )
   DEALLOCATE( p%X0 )

!  special test for status = - 8

   status = 8
   n = 1 ; m = 0 ; a_ne = 0
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   p%Hessian_kind = 0
   p%gradient_kind = 0
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%X_l = (/ 0.0_rp_ /)
   p%X_u = (/ infty /)
   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1
!  control%print_level = 1
#ifdef REAL_32
   control%stop_c = 10.0_rp_ ** ( - 37 )
#else
   control%stop_c = 10.0_rp_ ** ( - 80 )
#endif
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!  control%maxit = 1000
   CALL LSQP_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
      write(6,*) info%obj
     ELSE
       WRITE( 6, "(I2, ': LSQP_solve exit status = ', I6 )") status, info%status
   END IF
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )
   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )

   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
   p%gradient_kind = 0
   ALLOCATE( p%A%ptr( m + 1 ) )
   DO data_storage_type = 0, 2
     CALL LSQP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 2
     stop_c = control%stop_c
     control%print_level = 0
     DO Hessian_kind = 0, 2
       p%new_problem_structure = .TRUE.
       IF ( data_storage_type == 0 ) THEN
         st = 'C'
         ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
         IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
         CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
         p%A%row = (/ 1, 1, 2, 2 /)
         p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
       ELSE IF ( data_storage_type == 1 ) THEN
         st = 'R'
         ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
         IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
         CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
         p%A%col = (/ 1, 2, 2, 3 /)
         p%A%ptr = (/ 1, 3, 5 /)
       ELSE
         st = 'D'
         ALLOCATE( p%A%val( n*m ), p%A%row( 0 ), p%A%col( n*m ) )
         IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
         CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       END IF
       IF ( Hessian_kind == 0 ) THEN
         control%stop_c = 10.0_rp_ ** ( - 12 )
       ELSE
         control%stop_c = stop_c
       END IF
       p%Hessian_kind = Hessian_kind
       IF ( p%Hessian_kind == 2 ) THEN
         ALLOCATE( p%WEIGHT( n ) )
         p%WEIGHT = (/ 0.1_rp_, 1.0_rp_, 2.0_rp_ /)
       END IF
       IF ( p%Hessian_kind /= 0 ) THEN
         ALLOCATE( p%X0( n ) )
         p%X0 = (/  -2.0_rp_, 1.0_rp_, 3.0_rp_ /)
       END IF
       DO i = 1, 2
         IF ( data_storage_type == 0 ) THEN
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE IF ( data_storage_type == 1 ) THEN
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
         END IF
         p%X = (/  -2.0_rp_, 1.0_rp_,  3.0_rp_ /) ; p%Y = 0.0_rp_
         p%Z = 0.0_rp_
         CALL LSQP_solve( p, data, control, info )
         IF ( info%status == 0 ) THEN
           WRITE( 6, "( A1, 2I1, ':', I6, ' iterations. Value = ', F6.1 )" ) &
             st, Hessian_kind + 1, i, info%iter, info%obj
         ELSE
           WRITE( 6, "( ' LSQP_solve exit status = ', I6 ) " ) info%status
         END IF
       END DO
       IF ( p%Hessian_kind == 2 ) DEALLOCATE( p%WEIGHT )
       IF ( p%Hessian_kind /= 0 ) DEALLOCATE( p%X0 )
       DEALLOCATE( p%A%val, p%A%row, p%A%col )
     END DO
     CALL LSQP_terminate( data, control, info )
   END DO
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; a_ne = 2
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%X0( n ) )

   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_ /)
   p%C_u = (/ 1.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)
   p%X0 = (/  -2.0_rp_, 1.0_rp_ /)

   p%new_problem_structure = .TRUE.
   p%gradient_kind = 0
   p%Hessian_kind = 1
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2

!  test with new and existing data

   tests = 8
   DO i = 0, tests
     IF ( i == 1 ) THEN
       control%factor = - 1
     ELSE IF ( i == 2 ) THEN
       control%factor = 1
     ELSE IF ( i == 3 ) THEN
       control%max_col = 0
     ELSE IF ( i == 4 ) THEN
       control%factor = 2
     ELSE IF ( i == 5 ) THEN
       control%just_feasible = .TRUE.
     ELSE IF ( i == 6 ) THEN
       control%getdua = .TRUE.
     ELSE IF ( i == 7 ) THEN
       control%muzero = 1.0_rp_
       control%feasol = .FALSE.
     ELSE IF ( i == 8 ) THEN
       p%gradient_kind = 1
     END IF

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL LSQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LSQP_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 4
   DO i = tests + 1, tests + 1

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL LSQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LSQP_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1
   DO i = tests + 2, tests + 2

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL LSQP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%X0 )
   DEALLOCATE( p%A%ptr )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%X_l( n ), p%X_u( n ), p%X0( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%gradient_kind = 0
   p%Hessian_kind = 1
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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
   p%x0 =  (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
              0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)
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

   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 1
!  control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL LSQP_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
    IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 1, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%X0 )
   DEALLOCATE( p%A%ptr )

!  second example

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%X_l( n ), p%X_u( n ), p%X0( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   p%gradient_kind = 1
   p%Hessian_kind = 1
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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
   p%x0 =  (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
              0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)
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

   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%treat_zero_bounds_as_general = .TRUE.
   control%restore_problem = 2
!  control%print_level = 101
!  control%print_level = 1
   control%out = scratch_out
   control%error = scratch_out
   control%stop_c = 10.0_rp_ ** ( - 10 )
   control%potential_unbounded = - 20.0_rp_
!  control%out = 6
!  control%error = 6
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL LSQP_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 2, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%X0 )
   DEALLOCATE( p%A%ptr )

!  Third example

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%X_l( n ), p%X_u( n ), p%G( n ), p%X0( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   p%gradient_kind = 2
   p%Hessian_kind = 1
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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
   p%x0 =  (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
              0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)
   p%G =   (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
              0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_ /)
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

   CALL LSQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%treat_zero_bounds_as_general = .TRUE.
   control%restore_problem = 2
!  control%print_level = 101
!  control%print_level = 1
   control%out = scratch_out
   control%error = scratch_out
   control%stop_c = 10.0_rp_ ** ( - 10 )
   control%potential_unbounded = - 20.0_rp_
!  control%out = 6
!  control%error = 6
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL LSQP_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 3, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LSQP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL LSQP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%X0, p%G )
   DEALLOCATE( p%A%ptr )
   DEALLOCATE( p%A%type )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( LSQP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_LSQP_test_deck



