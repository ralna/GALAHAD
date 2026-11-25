! THIS VERSION: GALAHAD 5.6 - 2026-11-25 AT 08:17 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_DQP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_DQP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( DQP_data_type ) :: data
   TYPE ( DQP_control_type ) :: control
   TYPE ( DQP_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   INTEGER ( KIND = ip_ ) :: j, l, pl
   INTEGER ( KIND = ip_ ), DIMENSION( 0 : 1 ) :: iter_pl, status_pl
   REAL ( KIND = rp_ ), DIMENSION( 20, 0 : 1 ) :: X_pl
   REAL ( KIND = rp_ ) :: err
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat

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
!    IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
!    IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL DQP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 1
! control%print_level = 1

!control%print_level = 3
!control%maxit = 2
!control%SBLS_control%print_level = 3
!control%SBLS_control%preconditioner = 3

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
     p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%H%col = (/ 1, 2, 3, 2 /)
     p%H%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

!    control%out = 6 ; control%print_level = 1

     IF ( status == - GALAHAD_error_restrictions ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == - GALAHAD_error_bad_bounds ) THEN
       p%X_u( 1 ) = - 2.0_rp_
     ELSE IF ( status == - GALAHAD_error_primal_infeasible ) THEN
!      control%subspace_direct = .TRUE.
       control%print_level = 4
       control%gltr_control%print_level = 1
       control%maxit = 5
       p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
       p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE IF ( status == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
!      control%print_level = 1
     ELSE IF ( status == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0
       p%X( 2 ) = 100000000.0_rp_
!      control%print_level = 1
!      control%maxit = 1
     ELSE IF ( status == - GALAHAD_error_inertia ) THEN
       p%H%val( 4 ) = 4.0_rp_
     ELSE IF ( status == - GALAHAD_error_upper_entry ) THEN
       p%H%col( 1 ) = 2
     ELSE
     END IF

     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter,                     &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': DQP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL DQP_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP
   END DO

!  special case

   GO TO 10

   status = 5

   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1

   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
   p%H%col = (/ 1, 2, 3, 2 /)
   p%H%ptr = (/ 1, 2, 3, 5 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

   control%print_level = 101
   control%gltr_control%print_level = 1
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   control%exact_arc_search = .FALSE.
   IF ( info%status == 0 ) THEN
     WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',      &
   &      F6.1, ' status = ', I6 )" ) status, info%iter,                       &
            info%obj, info%status
   ELSE
     WRITE( 6, "(I2, ': DQP_solve exit status = ', I6 )") status, info%status
   END IF

   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
10 CONTINUE

   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  stop

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

   DO data_storage_type = -3, 0
     CALL DQP_initialize( data, control, info )
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
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 1.0_rp_, 0.0_rp_, 3.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL DQP_solve( p, data, control, info, C_stat, X_stat )

      IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) st, i, info%iter,                &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': DQP_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
!      STOP
     END DO
     CALL DQP_terminate( data, control, info )
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )

!  ===========================================
!  basic test of various weighted combinations
!  ===========================================

   WRITE( 6, "( /, ' basic tests of weighted combinations ', / )" )

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%X0( n ), p%WEIGHT( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )

   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
   control%dual_starting_point = 3
!  DO i = 1, 1
   DO i = 1, 4
!  DO i = 3, 3
     p%new_problem_structure = .TRUE.
     IF ( i == 1 ) THEN
       p%Hessian_kind = 1
       p%target_kind = 0
       p%gradient_kind = 2
       p%G = (/ - 1.0_rp_, -1.0_rp_, -1.0_rp_ /)
       p%f = 1.0_rp_
     ELSE IF ( i == 2 ) THEN
       p%Hessian_kind = 2
       p%target_kind = 0
       p%gradient_kind = 2
       p%WEIGHT = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       p%G = (/ - 1.0_rp_, -1.0_rp_, -1.0_rp_ /)
       p%f = 1.0_rp_
     ELSE IF ( i == 3 ) THEN
       p%Hessian_kind = 1
       p%target_kind = 1
       p%gradient_kind = 0
       p%f = -0.5_rp_
     ELSE IF ( i == 4 ) THEN
       p%Hessian_kind = 1
       p%target_kind = 2
       p%X0 = (/ 2.0_rp_, 2.0_rp_, 2.0_rp_ /)
       p%gradient_kind = 1
       p%f = - 5.0_rp_
     END IF
!    control%maxit = 1
!    control%print_level = 101
!    control%gltr_control%print_level = 1
     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
!write(6,*) ' X ', p%X

    IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6,' iterations. Optimal objective value = ',     &
   &            F6.1, ' status = ', I6 )" ) i, info%iter,                      &
                info%obj, info%status
     ELSE
       WRITE( 6, "( I2,': DQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
!    STOP
   END DO
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u, p%X0, p%WEIGHT )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

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
   p%X_l = (/ 0.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   p%new_problem_structure = .TRUE.
   p%Hessian_kind = - 1
   p%target_kind = - 1
   p%gradient_kind = - 1
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   p%H%row = (/ 1, 2, 3, 3 /)
   p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

!  GO TO 20

!   n = 2 ; m = 1 ; h_ne = 2 ; a_ne = 2
!   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
!   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
!   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
!   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
!   ALLOCATE( X_stat( n ), C_stat( m ) )

!   p%n = n ; p%m = m ; p%f = 0.05_rp_
!   p%G = (/ 0.0_rp_, 0.0_rp_ /)
!   p%C_l = (/ 1.0_rp_ /)
!   p%C_u = (/ 1.0_rp_ /)
!   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
!   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

!   p%new_problem_structure = .TRUE.
!   ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
!   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
!   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
!   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
!   p%H%col = (/ 1, 2 /)
!   p%H%ptr = (/ 1, 2, 3 /)
!   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
!   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
!   p%A%col = (/ 1, 2 /)
!   p%A%ptr = (/ 1, 3 /)

   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1

!  test with new and existing data

   tests = 11
   DO i = 0, tests
     IF ( i == 0 ) THEN
     ELSE IF ( i == 1 ) THEN
       control%dual_starting_point = 1
     ELSE IF ( i == 2 ) THEN
       control%dual_starting_point = 2
     ELSE IF ( i == 3 ) THEN
       control%dual_starting_point = 3
     ELSE IF ( i == 4 ) THEN
       control%dual_starting_point = 4
     ELSE IF ( i == 5 ) THEN
       control%dual_starting_point = 0
       control%exact_arc_search = .FALSE.
!      control%print_level = 1
!      control%GLTR_control%print_level = 1
     ELSE IF ( i == 6 ) THEN
!      control%print_level = 0
       control%exact_arc_search = .TRUE.
       control%subspace_direct = .TRUE.
     ELSE IF ( i == 7 ) THEN
       control%max_sc = 0
     ELSE IF ( i == 8 ) THEN
       control%max_sc = 100
       control%exact_arc_search = .FALSE.
       control%subspace_arc_search = .FALSE.
       control%subspace_direct = .FALSE.
     ELSE IF ( i == 9 ) THEN
!      control%print_level = 1
       control%exact_arc_search = .TRUE.
       control%subspace_direct = .TRUE.
       control%subspace_arc_search = .TRUE.
     ELSE IF ( i == 10 ) THEN
       control%max_sc = 0
     ELSE IF ( i == 11 ) THEN
       control%max_sc = 100
       control%subspace_arc_search = .TRUE.
     END IF
     p%H%val = (/ 1.0_rp_, 1.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
!write(6,*) ' X '
!do j = 1, p%n
!write(6,"( I2, 3ES12.4, I4 )" ) j, p%X_l( j ), p%X( j ), p%X_u( j ), X_stat( j )
!end do
!write(6,*) ' C '
!do j = 1, p%m
!write(6,"( I2, 3ES12.4, I4 )" ) j, p%C_l( j ), p%C( j ), p%C_u( j ), C_stat( j )
!end do
   END DO
   CALL DQP_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty, - infty /)
   p%X_u = (/ infty, infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2, 3, 1 /)
   p%H%ptr = (/ 1, 2, 3, 5 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL DQP_terminate( data, control, info )

!  case when there are no free variables

!20 CONTINUE
   p%X_l = (/ 0.5_rp_, 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_, 0.5_rp_ /)
   p%C_l = (/ 1.0_rp_, 1.0_rp_ /)
   p%C_u = (/ 2.0_rp_, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
   p%H%col = (/ 1, 2, 3, 1 /)
   p%H%ptr = (/ 1, 2, 3, 5 /)
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
!  control%CRO_control%error = 0
!  control%print_level = 1
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 101
!  control%gltr_control%print_level = 1
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
!  stop

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
   p%H%val = (/ 10.0_rp_, 1.0_rp_, 20.0_rp_, 2.0_rp_, 30.0_rp_, 3.0_rp_,       &
                40.0_rp_, 4.0_rp_, 50.0_rp_, 5.0_rp_, 60.0_rp_, 6.0_rp_,       &
                70.0_rp_, 7.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_,         &
                5.0_rp_, 6.0_rp_, 7.0_rp_ /)

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

   CALL DQP_initialize( data, control, info )
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
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )

!  Second problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
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

   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )

!  Third problem

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
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

   CALL DQP_initialize( data, control, info )
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
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )

!  Fourth and Fifth problems

   n = 14 ; m = 10 ; h_ne = 14 ; a_ne = 32
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

   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DQP_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )

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
   ALLOCATE( X_stat( n ), C_stat( m ) )
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
   CALL DQP_initialize( data, control, info )
   control%infinity = infty
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == GALAHAD_ok ) THEN
     err = MAXVAL( ABS( p%X - (/ 0.5_rp_, 0.5_rp_, 0.0_rp_ /) ) )
     IF ( err > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
       WRITE( 6, "( ' default solver test: solution error = ', ES8.1,         &
      &             ' FAILED' )" ) err
       STOP 1
     END IF
     WRITE( 6, "( ' default solver test: solution correct' )" )
   ELSE
     WRITE( 6, "( ' default solver test: DQP_solve exit status = ', I0,      &
    &             ' FAILED' )" ) info%status
     STOP 1
   END IF
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )

!  ===================================================================
!  check that the active set is recorded when the starting point is
!  optimal, so that no arc search is performed. The problem minimize
!  1/2 ||x||^2 s.t. x_1 + x_2 >= 2 and x >= 0 has the solution x = (1,1)
!  with multipliers y = 1 and z = 0, which are given as the starting dual
!  variables. The constraint is active, and the optimal matrix of active
!  constraints is formed (as it is by DLP)
!  ===================================================================

   WRITE( 6, "( /, ' optimal start test ', / )" )
   n = 2 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%G = 0.0_rp_
   p%X_l = 0.0_rp_ ; p%X_u = infty
   p%C_l = 2.0_rp_ ; p%C_u = infty
   p%X = 1.0_rp_ ; p%Y = 1.0_rp_ ; p%Z = 0.0_rp_
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat ) ; p%H%ne = 2
   ALLOCATE( p%H%row( 2 ), p%H%col( 2 ), p%H%val( 2 ) )
   p%H%row = (/ 1, 2 /) ; p%H%col = (/ 1, 2 /) ; p%H%val = 1.0_rp_
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 2
   ALLOCATE( p%A%row( 2 ), p%A%col( 2 ), p%A%val( 2 ) )
   p%A%row = (/ 1, 1 /) ; p%A%col = (/ 1, 2 /) ; p%A%val = 1.0_rp_
   CALL DQP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%factor_optimal_matrix = .TRUE.
   CALL DQP_solve( p, data, control, info, C_stat, X_stat )

!  the linear solver chosen for the tests may not be available

   IF ( info%status == GALAHAD_error_unknown_solver .OR.                      &
        info%status == GALAHAD_unavailable_option ) THEN
     WRITE( 6, "( ' optimal start test: skipped, linear solver',              &
    &             ' unavailable (status ', I0, ')' )" ) info%status
   ELSE IF ( info%status /= GALAHAD_ok ) THEN
     WRITE( 6, "( ' optimal start test: DQP_solve exit status = ', I0,       &
    &             ' FAILED' )" ) info%status
     STOP 1
   ELSE IF ( info%iter /= 0 ) THEN
     WRITE( 6, "( ' optimal start test: ', I0, ' iterations taken,',         &
    &             ' none expected FAILED' )" ) info%iter
     STOP 1
   ELSE IF ( C_stat( 1 ) /= - 1 .OR. ANY( X_stat /= 0 ) ) THEN
     WRITE( 6, "( ' optimal start test: C_stat = ', I0, ', X_stat = ',       &
    &             2( 1X, I0 ), ' FAILED' )" ) C_stat, X_stat
     STOP 1
   ELSE IF ( MAXVAL( ABS( p%X - 1.0_rp_ ) ) > EPSILON( 1.0_rp_ ) ** 0.75 ) THEN
     WRITE( 6, "( ' optimal start test: solution error = ', ES8.1,           &
    &             ' FAILED' )" ) MAXVAL( ABS( p%X - 1.0_rp_ ) )
     STOP 1
   ELSE
     WRITE( 6, "( ' optimal start test: active set correct' )" )
   END IF
   CALL DQP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )

!  ===================================================================
!  check that the result does not depend on the print level. The
!  stopping tolerance is relative to the initial dual gradient (with
!  stop_abs_p = 0), and was previously only computed correctly when the
!  first iteration was printed
!  ===================================================================

   WRITE( 6, "( /, ' print level test ', / )" )
   n = 20 ; m = 10
   DO pl = 0, 1
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
     ALLOCATE( X_stat( n ), C_stat( m ) )
     p%new_problem_structure = .TRUE.
     p%n = n ; p%m = m ; p%f = 0.0_rp_
     DO i = 1, n
       p%G( i ) = - 10.0_rp_ * SIN( REAL( 3 * i, rp_ ) )
     END DO
     p%X_l = - 1.0_rp_ ; p%X_u = 1.0_rp_
     p%C_l = - 1.0_rp_ ; p%C_u = 1.0_rp_
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL SMT_put( p%H%type, 'IDENTITY', smt_stat )
     CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 3 * m
     ALLOCATE( p%A%row( 3 * m ), p%A%col( 3 * m ), p%A%val( 3 * m ) )
     l = 0
     DO i = 1, m
       DO j = 0, 2
         l = l + 1 ; p%A%row( l ) = i
         p%A%col( l ) = MOD( 7 * i + 5 * j, n ) + 1
         p%A%val( l ) = COS( REAL( l, rp_ ) )
       END DO
     END DO
     CALL DQP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%stop_abs_p = 0.0_rp_ ; control%stop_rel_p = 0.1_rp_
     control%print_level = pl
     control%out = scratch_out ; control%error = scratch_out
     OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
     CALL DQP_solve( p, data, control, info, C_stat, X_stat )
     CLOSE( UNIT = scratch_out )
     status_pl( pl ) = info%status
     IF ( info%status /= GALAHAD_ok .AND.                                     &
          info%status /= GALAHAD_error_unknown_solver .AND.                   &
          info%status /= GALAHAD_unavailable_option ) THEN
       WRITE( 6, "( ' print level test: DQP_solve exit status = ', I0,       &
      &             ' FAILED' )" ) info%status
       STOP 1
     END IF
     iter_pl( pl ) = info%iter ; X_pl( : , pl ) = p%X
     CALL DQP_terminate( data, control, info )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 X_stat, C_stat )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   END DO

!  the linear solver chosen for the tests may not be available

   IF ( ANY( status_pl /= GALAHAD_ok ) ) THEN
     WRITE( 6, "( ' print level test: skipped, linear solver',                &
    &             ' unavailable (status ', I0, ')' )" ) status_pl( 0 )
   ELSE IF ( iter_pl( 0 ) /= iter_pl( 1 ) .OR.                                &
        MAXVAL( ABS( X_pl( : , 0 ) - X_pl( : , 1 ) ) ) >                      &
          SQRT( EPSILON( 1.0_rp_ ) ) ) THEN
     WRITE( 6, "( ' print level test: ', I0, ' and ', I0, ' iterations',      &
    &             ' with print levels 0 and 1 FAILED' )" ) iter_pl
     STOP 1
   ELSE
     WRITE( 6, "( ' print level test: results independent of print',        &
    &             ' level' )" )
   END IF

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( DQP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     control%definite_linear_solver = definite_linear_solver
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_DQP_EXAMPLE
