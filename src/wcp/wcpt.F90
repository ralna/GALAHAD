! THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 10:10 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_WCP_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPT_precision
   USE GALAHAD_WCP_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( WCP_data_type ) :: data
   TYPE ( WCP_control_type ) :: control
   TYPE ( WCP_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: i, n, m, a_ne, gradient_kind, data_storage_type
   INTEGER ( KIND = ip_ ) :: status, smt_stat
   INTEGER ( KIND = ip_ ) :: scratch_out = 56
   CHARACTER ( len = 1 ) :: st
!  CHARACTER( 10 ) :: probname = "TESTDECK  "
!  CHARACTER( 16 ) :: filename = "TESTDECK.SIF    "

   p%gradient_kind = 1
   p%f = 0.0_rp_

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for status = - 1 ... - 3

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )

   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1

   DO status = 1, 3
     IF ( status == 2 ) CYCLE
     IF ( status == 3 ) CYCLE
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_
     p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_

     IF ( status == 1 ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == 3 ) THEN
       DEALLOCATE( p%A%val )
       ALLOCATE( p%A%val( 1 ) )
     END IF

     CALL WCP_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': WCP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( status == 1 ) THEN
       p%n = n ; p%m = m
     END IF
   END DO

   CALL WCP_terminate( data, control, info )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C, p%A%ptr )

!  special test for status = - 4

   status = 4
   n = 1 ; m = 0 ; a_ne = 0
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%X_l = (/ 0.0_rp_ /)
   p%X_u = (/ infty /)
   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1
   control%maxit = 1
!  control%print_level = 1
#ifdef REAL_32
   control%stop_c = 10.0_rp_ ** ( - 37 )
#else
   control%stop_c = 10.0_rp_ ** ( - 80 )
#endif
   p%X = 0.0_rp_
   p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
   CALL WCP_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': WCP_solve exit status = ', I6 )") status, info%status
   END IF
   CALL WCP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C, p%A%ptr )

!  tests for status = - 5 ... - 8

   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%X0( n ) )

   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
   p%X0 = (/  -2.0_rp_, 1.0_rp_, 3.0_rp_ /)

   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1

   DO status = 5, 8
     IF ( status == 7 ) CYCLE
     IF ( status == 8 ) CYCLE
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_
     p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_

     IF ( status == 5 ) THEN
       p%X_u( 1 ) = - 2.0_rp_
     ELSE IF ( status == 6 ) THEN
       p%new_problem_structure = .TRUE.
       p%n = n ; p%m = m
       p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
       p%C_u = (/ 4.0_rp_, infty /)
       p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
       p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
     ELSE
     END IF

     CALL WCP_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': WCP_solve exit status = ', I6 )") status, info%status
     END IF
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( status == 8 ) THEN
       control%stop_c = EPSILON( 1.0_rp_ ) ** 0.33
     END IF
   END DO

   CALL WCP_terminate( data, control, info )
   DEALLOCATE( p%X0, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C, p%A%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )
   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )

   p%new_problem_structure = .TRUE.
   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
   ALLOCATE( p%A%ptr( m + 1 ) )
   DO data_storage_type = 0, 2
     CALL WCP_initialize( data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
     control%restore_problem = 2
     control%print_level = 0
     control%mu_target = 1.0_rp_
!    control%print_level = 3
!    control%FDC_control%print_level = 3
     DO gradient_kind = 0, 2
       p%gradient_kind = gradient_kind
       p%new_problem_structure = .TRUE.
       IF ( p%gradient_kind == 2 ) THEN
         ALLOCATE( p%G( n ) )
         p%G = (/ 0.1_rp_, 1.0_rp_, 2.0_rp_ /)
       END IF
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
       control%stop_c = 10.0_rp_ ** ( - 12 )
       DO i = 1, 2
         IF ( data_storage_type == 0 ) THEN
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE IF ( data_storage_type == 1 ) THEN
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
         END IF
         p%X = (/  -2.0_rp_, 1.0_rp_,  3.0_rp_ /)
         p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
         CALL WCP_solve( p, data, control, info )
         IF ( info%status == 0 ) THEN
           WRITE( 6, "( A1, 2I1, ':', I6, ' iterations. Value = ', F6.1 )" ) &
             st, gradient_kind + 1, i, info%iter, info%obj
         ELSE
           WRITE( 6, "( ' WCP_solve exit status = ', I6 ) " ) info%status
         END IF
       END DO
       IF ( p%gradient_kind == 2 ) DEALLOCATE( p%G )
       DEALLOCATE( p%A%val, p%A%row, p%A%col )
     END DO
     CALL WCP_terminate( data, control, info )
   END DO
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C, p%A%ptr )

!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; m = 1 ; a_ne = 2
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%X0( n ) )

   p%n = n ; p%m = m
   p%C_l = (/ 1.0_rp_ /)
   p%C_u = (/ 1.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)
   p%X0 = (/  -2.0_rp_, 1.0_rp_ /)

   p%gradient_kind = 1
   p%new_problem_structure = .TRUE.
   ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
   control%mu_target = 1.0_rp_
!  control%print_level = 3

!  test with new and existing data

   DO i = 0, 7
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
     ELSE IF ( i == 7 ) THEN
       control%mu_target = 10.0_rp_
     END IF

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_
     p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
!    control%print_level = 4
     CALL WCP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': WCP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL WCP_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
!  control%print_level = 4
   control%mu_target = 1.0_rp_

   DO i = 8, 8

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_
     p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
     CALL WCP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': WCP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL WCP_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%col = (/ 1, 2 /)
   p%A%ptr = (/ 1, 3 /)
   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 1
!  control%print_level = 1
   DO i = 9, 9

     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_
     p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
!    control%print_level = 1
     CALL WCP_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) i, info%iter, info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': WCP_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL WCP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X0, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C, p%A%ptr )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%X0( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - 100.0_rp_, - 100.0_rp_,             &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - 100.0_rp_, - 100.0_rp_,             &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, 100.0_rp_, 10.0_rp_, 2.0_rp_, 100.0_rp_,                &
              4.0_rp_, 100.0_rp_, 10.0_rp_, 2.0_rp_, 100.0_rp_,                &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -100.0_rp_, -100.0_rp_,      &
              -100.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -100.0_rp_,      &
              -100.0_rp_, -100.0_rp_ /)
   p%X_u = (/ 1.0_rp_, 100.0_rp_, 100.0_rp_, 3.0_rp_, 4.0_rp_, 0.0_rp_,        &
              100.0_rp_, 1.0_rp_, 100.0_rp_, 100.0_rp_, 3.0_rp_, 4.0_rp_,      &
              0.0_rp_, 100.0_rp_ /)
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

   p%gradient_kind = 1

   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%restore_problem = 2
   control%out = scratch_out
   control%error = scratch_out
   control%print_level = 101
!  control%out = 6
!  control%error = 6
!  control%print_level = 1
   control%mu_target = 1.0_rp_
   p%X = 0.0_rp_
   p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
   p%X0 = p%X
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL WCP_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
    IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 1, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': WCP_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL WCP_terminate( data, control, info )

!  CALL QPT_A_from_C_to_S( p, i )
!  write(6,*) ' exitcode ', i
!  CALL QPT_write_to_sif( p, probname, filename, 6, .FALSE., .FALSE.,          &
!                         0.5 * infty, no_H = .TRUE. )
   DEALLOCATE( p%A%val, p%A%ptr, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C )

!  second example

   n = 14 ; m = 17 ; a_ne = 46
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y_l( m ), p%Z_l( n ), p%Y_u( m ), p%Z_u( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - 100.0_rp_, - 100.0_rp_,             &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - 100.0_rp_, - 100.0_rp_,             &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty, 10.0_rp_, 2.0_rp_, 100.0_rp_,                    &
              4.0_rp_, infty, 10.0_rp_, 2.0_rp_, 100.0_rp_,                    &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -100.0_rp_, -100.0_rp_,      &
              -100.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -100.0_rp_,      &
              -100.0_rp_, -100.0_rp_ /)
   p%X_u = (/ 1.0_rp_, 100.0_rp_, 100.0_rp_, 3.0_rp_, 4.0_rp_, 0.0_rp_,        &
              100.0_rp_, 1.0_rp_, 100.0_rp_, 100.0_rp_, 3.0_rp_, 4.0_rp_,      &
              0.0_rp_, 100.0_rp_ /)
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

   CALL WCP_initialize( data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%treat_zero_bounds_as_general = .TRUE.
   control%restore_problem = 2
   control%out = scratch_out
   control%error = scratch_out
   control%print_level = 101
!  control%out = 6
!  control%error = 6
!  control%print_level = 1
   control%stop_c = 10.0_rp_ ** ( - 10 )
   control%mu_target = 1.0_rp_
   p%X = 0.0_rp_
   p%Y_l = 0.0_rp_ ; p%Z_l = 0.0_rp_ ; p%Y_u = 0.0_rp_ ; p%Z_u = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL WCP_solve( p, data, control, info )
!  CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &       F6.1, ' status = ', I6 )" ) 2, info%iter, info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': WCP_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL WCP_terminate( data, control, info )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y_l, p%Z_l, p%Y_u, p%Z_u, p%C )
   DEALLOCATE( p%A%type )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( WCP_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%FDC_control%use_sls = use_sls
     control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_WCP_test_program



