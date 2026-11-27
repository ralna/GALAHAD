! THIS VERSION: GALAHAD 5.6 - 2026-11-26 AT 09:08 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_DLP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_DLP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( DLP_data_type ) :: data
   TYPE ( DLP_control_type ) :: control
   TYPE ( DLP_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, m, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   INTEGER ( KIND = ip_ ) :: lp_trial, lp_errors
   INTEGER ( KIND = i8_ ) :: lp_seed
   REAL ( KIND = rp_ ) :: lp_f, lp_f_sol

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
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
     IF ( status == - GALAHAD_error_inertia ) CYCLE
     IF ( status == - GALAHAD_error_file ) CYCLE
     IF ( status == - GALAHAD_error_io ) CYCLE
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL DLP_initialize( data, control, info )
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

     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
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
     ELSE
     END IF

     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter,                     &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': DLP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL DLP_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP
   END DO

!  special case

   GO TO 10

   status = 5

   CALL DLP_initialize( data, control, info )
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

   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

   control%print_level = 101
   control%gltr_control%print_level = 1
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   control%exact_arc_search = .FALSE.
   IF ( info%status == 0 ) THEN
     WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',      &
   &      F6.1, ' status = ', I6 )" ) status, info%iter,                       &
            info%obj, info%status
   ELSE
     WRITE( 6, "(I2, ': DLP_solve exit status = ', I6 )") status, info%status
   END IF

   DEALLOCATE( p%A%val, p%A%row, p%A%col )
10 CONTINUE

   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  stop

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   DO data_storage_type = -3, 0
     CALL DLP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 2
!    control%out = 6 ; control%print_level = 1
!    control%SBLS_control%print_level = 4
!    control%SBLS_control%SLS_control%print_level = 3
!    control%SBLS_control%SLS_control%print_level_solver = 2
!    control%SBLS_control%preconditioner = 3
!    control%SBLS_control%factorization = 2
!    control%SBLS_control%itref_max = 2
     p%new_problem_structure = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%row = (/ 1, 1, 2, 2 /)
       p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 2, 3 /)
       p%A%ptr = (/ 1, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal H, dense A
       st = 'I'
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL DLP_solve( p, data, control, info, C_stat, X_stat )

      IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) st, i, info%iter,                &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': DLP_solve exit status = ', I6 ) " )           &
           st, i, info%status
       END IF
!      STOP
     END DO
     CALL DLP_terminate( data, control, info )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
!    STOP
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

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

   CALL DLP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
   control%dual_starting_point = 3
   DO i = 1, 4
!  DO i = 3, 3
     p%new_problem_structure = .TRUE.
     IF ( i == 1 ) THEN
       p%target_kind = 0
       p%gradient_kind = 2
       p%G = (/ - 1.0_rp_, -1.0_rp_, -1.0_rp_ /)
       p%f = 1.0_rp_
     ELSE IF ( i == 2 ) THEN
       p%target_kind = 0
       p%gradient_kind = 2
       p%WEIGHT = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       p%G = (/ - 1.0_rp_, -1.0_rp_, -1.0_rp_ /)
       p%f = 1.0_rp_
     ELSE IF ( i == 3 ) THEN
       p%target_kind = 1
       p%gradient_kind = 0
       p%f = -0.5_rp_
     ELSE IF ( i == 4 ) THEN
       p%target_kind = 2
       p%X0 = (/ 2.0_rp_, 2.0_rp_, 2.0_rp_ /)
       p%gradient_kind = 1
       p%f = - 5.0_rp_
     END IF
!    control%maxit = 1
!    control%print_level = 101
!    control%gltr_control%print_level = 1
     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
!write(6,*) ' X ', p%X

    IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6,' iterations. Optimal objective value = ',     &
   &            F6.1, ' status = ', I6 )" ) i, info%iter,                      &
                info%obj, info%status
     ELSE
       WRITE( 6, "( I2,': DLP_solve exit status = ', I6 ) " ) i, info%status
     END IF
!    STOP
   END DO
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u, p%X0, p%WEIGHT )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ 0.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   p%new_problem_structure = .TRUE.
   p%target_kind = - 1
   p%gradient_kind = - 1
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

!  GO TO 20

!   n = 2 ; m = 1 ; a_ne = 2
!   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
!   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
!   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
!   ALLOCATE( p%A%ptr( m + 1 ) )
!   ALLOCATE( X_stat( n ), C_stat( m ) )

!   p%n = n ; p%m = m ; p%f = 0.05_rp_
!   p%G = (/ 0.0_rp_, 0.0_rp_ /)
!   p%C_l = (/ 1.0_rp_ /)
!   p%C_u = (/ 1.0_rp_ /)
!   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
!   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

!   p%new_problem_structure = .TRUE.
!   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
!   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
!   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
!   p%A%col = (/ 1, 2 /)
!   p%A%ptr = (/ 1, 3 /)

   CALL DLP_initialize( data, control, info )
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
     ELSE IF ( i == 10 ) THEN
       control%max_sc = 0
     ELSE IF ( i == 11 ) THEN
       control%max_sc = 100
       control%subspace_arc_search = .TRUE.
     END IF
     p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) i, info%status
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
   CALL DLP_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty, - infty /)
   p%X_u = (/ infty, infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   CALL DLP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL DLP_terminate( data, control, info )

!  case when there are no free variables

!20 CONTINUE
   p%X_l = (/ 0.5_rp_, 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_, 0.5_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 2.0_rp_, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)
   CALL DLP_initialize( data, control, info )
   CALL WHICH_sls( control )
!  control%CRO_control%error = 0
!  control%print_level = 4
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 101
!  control%gltr_control%print_level = 1
   DO i = tests + 2, tests + 2
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )
!  stop

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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

   CALL DLP_initialize( data, control, info )
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
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  Second problem

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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

   CALL DLP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  Third problem

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
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

   CALL DLP_initialize( data, control, info )
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
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  Fourth and Fifth problems

   n = 14 ; m = 10 ; a_ne = 32
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - infty, - infty, - infty, - infty, - infty, - infty, - infty,   &
              - infty, - infty, - infty, - infty, - infty, - infty, - infty  /)
   p%X_u = - p%X_l
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

   CALL DLP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   X_stat = 0 ; C_stat = 0
   X_stat( 2 ) = - 1 ; X_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': DLP_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  ======================================================================
!  random LPs with known solutions, many of which are degenerate, so that
!  their perturbed solutions have constraints or bounds on their bounds
!  but not in the active set. In single precision, LP 7 is only solved if
!  the perturbed problems are solved more accurately than DQP's default
!  ======================================================================

   WRITE( 6, "( /, ' random LP tests ', / )" )
   lp_errors = 0 ; lp_seed = 12345
   DO lp_trial = 1, 12
     n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
     CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
     ALLOCATE( X_stat( n ), C_stat( m ) )
     CALL DLP_initialize( data, control, info )
     control%infinity = infty
     control%symmetric_linear_solver = 'sytr '
     control%definite_linear_solver = 'potr '
     CALL DLP_solve( p, data, control, info, C_stat, X_stat )
     lp_f = DOT_PRODUCT( p%G( : n ), p%X( : n ) )
     IF ( info%status /= GALAHAD_ok ) THEN
       lp_errors = lp_errors + 1
       WRITE( 6, "( ' LP ', I0, ': DLP_solve exit status = ', I0,         &
      &             ' FAILED' )" ) lp_trial, info%status
     ELSE IF ( ABS( lp_f - lp_f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 *          &
               MAX( 1.0_rp_, ABS( lp_f_sol ) ) ) THEN
       lp_errors = lp_errors + 1
       WRITE( 6, "( ' LP ', I0, ': objective error = ', ES8.1, ' FAILED' )" )  &
         lp_trial, ABS( lp_f - lp_f_sol )
     ELSE
       WRITE( 6, "( ' LP ', I0, ': solution correct' )" ) lp_trial
     END IF
     CALL DLP_terminate( data, control, info )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,          &
                 X_stat, C_stat )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   END DO
   IF ( lp_errors > 0 ) THEN
     WRITE( 6, "( /, 1X, I0, ' random LP tests failed' )" ) lp_errors
     STOP 1
   END IF

!  ======================================================================
!  a larger random LP (the 62nd in the sequence above, with n = 210), for
!  which the multipliers computed by the pounce have a stationarity
!  residual of about 10^5 EPSILON in double precision, and the pounce only
!  succeeds if this is accepted
!  ======================================================================

   WRITE( 6, "( /, ' larger random LP test ', / )" )
   lp_seed = 12345
   DO lp_trial = 1, 61
     n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
     CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   END DO
   lp_trial = 62
   n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
   CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   CALL DLP_initialize( data, control, info )
   control%infinity = infty
   control%symmetric_linear_solver = 'sytr '
   control%definite_linear_solver = 'potr '
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   lp_f = DOT_PRODUCT( p%G( : n ), p%X( : n ) )
   IF ( info%status /= GALAHAD_ok ) THEN
     WRITE( 6, "( ' LP ', I0, ': DLP_solve exit status = ', I0,               &
    &             ' FAILED' )" ) lp_trial, info%status
     STOP 1
   ELSE IF ( ABS( lp_f - lp_f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 *            &
             MAX( 1.0_rp_, ABS( lp_f_sol ) ) ) THEN
     WRITE( 6, "( ' LP ', I0, ': objective error = ', ES8.1, ' FAILED' )" )    &
       lp_trial, ABS( lp_f - lp_f_sol )
     STOP 1
   ELSE
     WRITE( 6, "( ' LP ', I0, ': solution correct' )" ) lp_trial
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )

!  ======================================================================
!  check that the extrapolation accounts for the free variables. For
!  minimize - x_1 + x_2 subject to x_1 <= 100 and x_2 >= 0, with x_1 free,
!  the perturbed problem with perturbation 0.1 has the solution ( 10, 0 ).
!  Along the extrapolation only the free variable x_1 changes, and it
!  reaches the constraint for the perturbation 0.01, so the extrapolated
!  point is not optimal (the solution is ( 100, 0 ))
!  ======================================================================

   WRITE( 6, "( /, ' free variable extrapolation test ', / )" )
   n = 2 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ), X_stat( n ),           &
             C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%G = (/ - 1.0_rp_, 1.0_rp_ /)
   p%X_l = (/ - infty, 0.0_rp_ /) ; p%X_u = infty
   p%C_l = - infty ; p%C_u = 100.0_rp_
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 1
   ALLOCATE( p%A%row( 1 ), p%A%col( 1 ), p%A%val( 1 ) )
   p%A%row = 1 ; p%A%col = 1 ; p%A%val = 1.0_rp_
   CALL SMT_put( p%H%type, 'ZERO', smt_stat )
   CALL DLP_initialize( data, control, info )
   control%infinity = infty
   control%symmetric_linear_solver = 'sytr '
   control%definite_linear_solver = 'potr '
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status /= GALAHAD_ok ) THEN
     WRITE( 6, "( ' free variable extrapolation test: DLP_solve exit',        &
    &             ' status = ', I0, ' FAILED' )" ) info%status
     STOP 1
   ELSE IF ( MAXVAL( ABS( p%X - (/ 100.0_rp_, 0.0_rp_ /) ) ) >                &
             SQRT( EPSILON( 1.0_rp_ ) ) * 100.0_rp_ ) THEN
     WRITE( 6, "( ' free variable extrapolation test: wrong solution',        &
    &             ' x = ', 2ES12.4, ' FAILED' )" ) p%X
     STOP 1
   ELSE
     WRITE( 6, "( ' free variable extrapolation test: solution correct' )" )
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )

!  ======================================================================
!  check that DLP reports failure if the perturbation reaches its lower
!  limit before the solution is found. For minimize - x_1 - x_2 - x_3
!  subject to x_1 + x_2 + x_3 <= 1000 and 0 <= x <= ( 100, 50, 1 ), the
!  perturbed problem with perturbation 0.1 has the solution ( 10, 10, 1 ),
!  and the extrapolation along its active set gives ( 50, 50, 1 ) for the
!  perturbation 0.02, at which x_2 reaches its bound. As the lower limit
!  on the perturbation is 0.05, DLP should stop there, report failure and
!  return the extrapolated point (the solution is ( 100, 50, 1 ))
!  ======================================================================

   WRITE( 6, "( /, ' perturbation limit test ', / )" )
   n = 3 ; m = 1
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),          &
             p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ), X_stat( n ),           &
             C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%f = 0.0_rp_ ; p%G = - 1.0_rp_
   p%X_l = 0.0_rp_ ; p%X_u = (/ 100.0_rp_, 50.0_rp_, 1.0_rp_ /)
   p%C_l = - infty ; p%C_u = 1000.0_rp_
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = n
   ALLOCATE( p%A%row( n ), p%A%col( n ), p%A%val( n ) )
   p%A%row = 1 ; p%A%col = (/ 1, 2, 3 /) ; p%A%val = 1.0_rp_
   CALL SMT_put( p%H%type, 'ZERO', smt_stat )
   CALL DLP_initialize( data, control, info )
   control%infinity = infty
   control%symmetric_linear_solver = 'sytr '
   control%definite_linear_solver = 'potr '
   control%final_perturbation = 0.05_rp_
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   IF ( info%status /= GALAHAD_error_tiny_step ) THEN
     WRITE( 6, "( ' perturbation limit test: DLP_solve exit status = ', I0,   &
    &             ' FAILED' )" ) info%status
     STOP 1
   ELSE IF ( MAXVAL( ABS( p%X - (/ 50.0_rp_, 50.0_rp_, 1.0_rp_ /) ) ) >       &
             SQRT( EPSILON( 1.0_rp_ ) ) * 50.0_rp_ ) THEN
     WRITE( 6, "( ' perturbation limit test: wrong extrapolated point',       &
    &             ' x = ', 3ES12.4, ' FAILED' )" ) p%X
     STOP 1
   ELSE
     WRITE( 6, "( ' perturbation limit test: failure reported' )" )
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )

!  ======================================================================
!  check that DLP does not accept an inaccurate extrapolated solution. The
!  75th random LP in the sequence above (n = 260) needs perturbations below
!  their default lower limit, at which the perturbed problems cannot be
!  solved accurately. With the lower limit 10^-10 and DQP's primal
!  accuracy EPSILON^0.33 (rather than DLP's default EPSILON^0.5, with which
!  DQP reports failure first), the extrapolation from the perturbation
!  10^-10 was accepted in double precision, although the extrapolated
!  solution is wrong. DLP must either find the solution or report failure
!  ======================================================================

   WRITE( 6, "( /, ' extrapolation acceptance test ', / )" )
   lp_seed = 12345
   DO lp_trial = 1, 74
     n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
     CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )
   END DO
   lp_trial = 75
   n = 10 * ( 1 + ( lp_trial + 2 ) / 3 ) ; m = n / 2
   CALL LP_planted( p, n, m, lp_seed, lp_f_sol )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   CALL DLP_initialize( data, control, info )
   control%infinity = infty
   control%symmetric_linear_solver = 'sytr '
   control%definite_linear_solver = 'potr '
   control%final_perturbation = 10.0_rp_ ** ( - 10 )
   control%stop_abs_p = EPSILON( 1.0_rp_ ) ** 0.33
   control%stop_rel_p = EPSILON( 1.0_rp_ ) ** 0.33
   CALL DLP_solve( p, data, control, info, C_stat, X_stat )
   lp_f = DOT_PRODUCT( p%G( : n ), p%X( : n ) )
   IF ( info%status /= GALAHAD_ok ) THEN
     WRITE( 6, "( ' LP ', I0, ': failure reported, status = ', I0 )" )         &
       lp_trial, info%status
   ELSE IF ( ABS( lp_f - lp_f_sol ) > EPSILON( 1.0_rp_ ) ** 0.75 *            &
             MAX( 1.0_rp_, ABS( lp_f_sol ) ) ) THEN
     WRITE( 6, "( ' LP ', I0, ': wrong solution accepted, objective',         &
    &             ' error = ', ES8.1, ' FAILED' )" )                           &
       lp_trial, ABS( lp_f - lp_f_sol )
     STOP 1
   ELSE
     WRITE( 6, "( ' LP ', I0, ': solution correct' )" ) lp_trial
   END IF
   CALL DLP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z,            &
               X_stat, C_stat )
   DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type, p%H%type )

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
     TYPE ( DLP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     control%definite_linear_solver = definite_linear_solver
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_DLP_EXAMPLE
