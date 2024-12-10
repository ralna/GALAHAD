! THIS VERSION: GALAHAD 4.1 - 2023-02-11 AT 08:10 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LPB_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LPB_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LPB_data_type ) :: data
   TYPE ( LPB_control_type ) :: control
   TYPE ( LPB_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, m, a_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status
   CHARACTER ( len = 1 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat

!go to 111
   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
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
     IF ( status == - GALAHAD_error_upper_entry ) CYCLE
     IF ( status == - GALAHAD_error_sort ) CYCLE

     CALL LPB_initialize( data, control, info )
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
     ELSE
     END IF

!    control%out = 6 ; control%print_level = 1

     CALL LPB_solve( p, data, control, info, C_stat, B_stat )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter,                     &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': LPB_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     CALL LPB_terminate( data, control, info )
!    IF ( status == - GALAHAD_error_max_iterations ) STOP
!    IF ( status == - GALAHAD_error_primal_infeasible ) STOP
!    IF ( status == - GALAHAD_error_cpu_limit ) STOP
   END DO
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  special test for status = - 7

   status = - GALAHAD_error_unbounded
   n = 1 ; m = 0 ; a_ne = 0
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 0.0_rp_
   p%G = (/ - 1.0_rp_ /)
!  p%X_l = (/ - infty /)
   p%X_l = (/ 0.0_rp_ /)
   p%X_u = (/ infty /)
   CALL LPB_initialize( data, control, info )
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
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &          F6.1, ' status = ', I6 )" ) status, info%iter,                 &
                  info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': LPB_solve exit status = ', I6 )") status, info%status
   END IF
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%A%ptr )

!  stop
!111 continue
!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   DO data_storage_type = -2, 0
     CALL LPB_initialize( data, control, info )
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
     END IF

!  test with new and existing data

     DO i = 1, 2
!    DO i = 1, 1
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!      control%print_level = 101
!      control%sbls_control%print_level = 101
!      control%sbls_control%factorization = 2
!      control%sbls_control%min_diagonal = 0.000000000000001_rp_
       CALL LPB_solve( p, data, control, info, C_stat, B_stat )

       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) st, i, info%iter,                &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': LPB_solve exit status = ', I6 ) " )            &
           st, i, info%status
       END IF
!      STOP
     END DO
     CALL LPB_terminate( data, control, info )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%A%ptr )
!stop
!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; a_ne = 2
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.05_rp_
   p%G = (/ 0.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_ /)
   p%C_u = (/ 1.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 1
!  control%sbls_control%print_level = 1

!  test with new and existing data

   tests = 17
   DO i = 0, tests
     IF ( i == 0 ) THEN
!      control%precon = 0
     ELSE IF ( i == 1 ) THEN
!      control%precon = 1
     ELSE IF ( i == 2 ) THEN
!      control%precon = 2
     ELSE IF ( i == 3 ) THEN
!      control%precon = 3
     ELSE IF ( i == 4 ) THEN
!      control%precon = 5
     ELSE IF ( i == 5 ) THEN
       control%SBLS_control%factorization = - 1
     ELSE IF ( i == 6 ) THEN
       control%SBLS_control%factorization = 1
     ELSE IF ( i == 7 ) THEN
       control%SBLS_control%max_col = 0
     ELSE IF ( i == 8 ) THEN
       control%SBLS_control%factorization = 2
!      control%precon = 0
     ELSE IF ( i == 9 ) THEN
!      control%print_level = 2
!      control%precon = 1
     ELSE IF ( i == 10 ) THEN
!      control%precon = 2
     ELSE IF ( i == 11 ) THEN
!      control%precon = 3
     ELSE IF ( i == 12 ) THEN
!      control%precon = 5
     ELSE IF ( i == 13 ) THEN
     ELSE IF ( i == 14 ) THEN
     ELSE IF ( i == 15 ) THEN
       control%feasol = .FALSE.
     ELSE IF ( i == 16 ) THEN
       B_stat = 0 ; C_stat = 0 ; B_stat( 1 ) = - 1
     ELSE IF ( i == 17 ) THEN
       B_stat = 0 ; C_stat = 0 ; C_stat( 1 ) = - 1
     END IF

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL LPB_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL LPB_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%CRO_control%error = 0
!  control%print_level = 4
   control%infinity = infty
   control%restore_problem = 2
   DO i = tests + 2, tests + 2
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL LPB_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%ptr, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

!  First and second problems

   n = 14 ; m = 10 ; a_ne = 34
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = - 1.0_rp_
   p%X_u = 2.0_rp_
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5,             &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6, 7,             &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13, 14 /)
   p%C_l = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = p%C_l

   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 1, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type, p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z )
   DEALLOCATE( B_stat, C_stat )
!stop
   DEALLOCATE( p%C )

!  Third problem

   n = 14 ; m = 10 ; a_ne = 34
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = - 1.0_rp_
   p%X_u = 2.0_rp_
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5,             &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6, 7,             &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13, 14 /)
   p%C_l = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = p%C_l + 1.0_rp_

   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 3, info%status
   END IF

   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type, p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z )
   DEALLOCATE( B_stat, C_stat )
!stop
   DEALLOCATE( p%C )

!  Fourth problem

   n = 14 ; m = 10 ; a_ne = 34
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = - 1.0_rp_
   p%X_u = 2.0_rp_
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5,             &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6, 7,             &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13, 14 /)
   p%C_l = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_l( p%A%row( i ) ) = p%C_l( p%A%row( i ) ) + p%A%val( i )
   END DO
   p%C_u = infty

   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 4, info%status
   END IF

   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type, p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z )
   DEALLOCATE( B_stat, C_stat )
!stop
   DEALLOCATE( p%C )

!  Fifth problem

   n = 14 ; m = 10 ; a_ne = 34
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = - 1.0_rp_
   p%X_u = 2.0_rp_
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5,             &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10  /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6, 7,             &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13, 14 /)
   p%C_l = - infty
   p%C_u = 0.0_rp_
   DO i = 1, p%A%ne
     p%C_u( p%A%row( i ) ) = p%C_u( p%A%row( i ) ) + p%A%val( i )
   END DO

   CALL LPB_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL LPB_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': LPB_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL LPB_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type, p%A%ptr )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z )
   DEALLOCATE( B_stat, C_stat )
!stop
   DEALLOCATE( p%C )

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( LPB_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_LPB_EXAMPLE
