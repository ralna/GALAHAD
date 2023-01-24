! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LPA_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LPA_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LPA_data_type ) :: data
   TYPE ( LPA_control_type ) :: control
   TYPE ( LPA_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: n, m, a_ne, tests, smt_stat, pass, passes
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, warm, dual, status
   CHARACTER ( len = 1 ) :: st, du, wa
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat

!GO TO 10
   n = 3 ; m = 2 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%ptr( m + 1 ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for status = - 1 ... - 24

   DO status = 1, 100
     SELECT CASE ( - status )
     CASE ( GALAHAD_error_restrictions )
       passes = 3
     CASE ( GALAHAD_error_bad_bounds )
       passes = 2
     CASE ( GALAHAD_error_primal_infeasible )
       passes = 9
     CASE ( GALAHAD_error_unbounded )
       passes = 3
     CASE ( GALAHAD_error_max_iterations )
       passes = 1
     CASE ( GALAHAD_error_time_limit )
       passes = 1
!    CASE ( GALAHAD_error_integer_ws )
!    CASE ( GALAHAD_unavailable_option )
     CASE DEFAULT
       CYCLE
     END SELECT

     CALL LPA_initialize( data, control, inform )
     control%error = 0 ; control%out = 0 ; control%print_level = - 1
     control%infinity = 0.1_rp_ * infty

     p%n = n ; p%m = m ; p%f = 1.0_rp_
     p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
     p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
     p%C_u = (/ 4.0_rp_, infty /)
     p%X_l = (/ - 1.0_rp_, - infty, - infty /)
     p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ) )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

     DO pass = 1, passes
       SELECT CASE ( - status )
       CASE ( GALAHAD_error_restrictions )
         SELECT CASE ( pass )
         CASE( 1 )
           p%n = 0
         CASE( 2 )
           p%n = n ; p%m = - 1
         CASE( 3 )
           p%m = - 1
           DEALLOCATE( p%A%type )
           CALL SMT_put( p%A%type, 'NONSENSE', smt_stat )
         END SELECT
       CASE ( GALAHAD_error_bad_bounds )
         SELECT CASE ( pass )
         CASE( 1 )
           p%X_u( 1 ) = - 2.0_rp_
         CASE( 2 )
           p%X_u( 1 ) = 1.0_rp_ ; p%C_u( 1 ) = - 2.0_rp_
         END SELECT
       CASE ( GALAHAD_error_primal_infeasible )
         SELECT CASE ( pass )
         CASE( 1 )
           p%X_l = (/ - 1.0_rp_, 8.0_rp_, - infty /)
           p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
         CASE( 2 )
           p%X_l = (/ 0.0_rp_, 0.0_rp_, - infty /)
           p%X_u = (/ 0.0_rp_, 0.0_rp_, infty /)
           p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
           p%C_u = (/ 1.0_rp_, infty /)
         CASE( 3 )
           p%C_l = (/ - infty, 2.0_rp_ /)
           p%C_u = (/ - 1.0_rp_, infty /)
         CASE( 4 )
           p%C_l = (/ - 2.0_rp_, 2.0_rp_ /)
           p%C_u = (/ - 1.0_rp_, infty /)
         CASE( 5 )
           p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
           p%C_u = (/ infty, infty /)
         CASE( 6 )
           control%dual = .TRUE.
           p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
           p%C_u = (/ 1.0_rp_, infty /)
         CASE( 7 )
           p%C_l = (/ - infty, 2.0_rp_ /)
           p%C_u = (/ - 1.0_rp_, infty /)
         CASE( 8 )
           p%C_l = (/ - 2.0_rp_, 2.0_rp_ /)
           p%C_u = (/ - 1.0_rp_, infty /)
         CASE( 9 )
           p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
           p%C_u = (/ infty, infty /)
         END SELECT
       CASE ( GALAHAD_error_unbounded )
         SELECT CASE ( pass )
         CASE( 1 )
           p%X_l = (/ - infty, - infty, - infty /)
           p%X_u = (/ infty, infty, infty /)
         CASE( 2 )
           DEALLOCATE( p%A%val, p%A%col, p%A%ptr )
           CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
           p%m = 0
           ALLOCATE( p%A%val( 0 ), p%A%col( 0 ), p%A%ptr( 1 ) )
           p%A%ptr = (/ 1 /)
           p%X_l( 1 ) = - infty ; p%G( 1 ) = 1.0_rp_
         CASE( 3 )
           p%X_u( 1 ) = infty ; p%G( 1 ) = - 1.0_rp_
         END SELECT
       CASE ( GALAHAD_error_max_iterations )
         control%maxit = 0
       CASE ( GALAHAD_error_time_limit )
         control%cpu_time_limit = 0.0
         p%X( 2 ) = 100000000.0_rp_
       END SELECT
!      control%out = 6 ; control%print_level = 2
       CALL LPA_solve( p, data, control, inform, C_stat, X_stat )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',  &
     &        F7.2, ' status = ', I4 )" ) status, inform%iter,                 &
              inform%obj, inform%status
       ELSE
         WRITE( 6,"(I2,': LPA_solve exit status = ', I4)") status, inform%status
       END IF
       IF ( status == GALAHAD_unavailable_option ) THEN
         WRITE( 6, "( ' necessary HSL solver LA04 not availble, stopping' )" )
         STOP
       END IF
     END DO
     CALL LPA_terminate( data, control, inform )
     DEALLOCATE( p%A%val, p%A%col )
   END DO
   CALL LPA_terminate( data, control, inform )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat, p%A%ptr, p%A%type )
!  stop

!  =====================================
!  basic test of various storage formats
!  =====================================

!10 CONTINUE
   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 14 ; m = 7 ; a_ne = 14
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 0.0_rp_
   p%G = (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, - 0.5_rp_,   &
            - 0.57143_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,        &
            0.0_rp_ /)
   p%X_l = (/ - infty, - infty, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.5_rp_,   &
              0.5_rp_, - infty, - infty, - infty, - infty, 0.1_rp_, 0.1_rp_  /)
   p%X_u = (/ infty, infty, infty, infty, infty, infty, 2.0_rp_,               &
              2.0_rp_, 3.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 0.1_rp_, 0.1_rp_  /)
   p%C_l = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, - infty, - infty, - infty, - infty /)
   p%C_u = (/ 1.0_rp_, infty, 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

!  DO data_storage_type = - 2, - 2
!  DO data_storage_type = - 1, - 1
!  DO data_storage_type = 0, 0
   DO data_storage_type = - 2, 0
     CALL LPA_initialize( data, control, inform )
     control%infinity = 0.1_rp_ * infty
     p%new_problem_structure = .TRUE.
!    control%print_level = 2
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,      &
                    1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,      &
                    1.0_rp_, 1.0_rp_ /)
       p%A%row = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 /)
       p%A%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%A%val( a_ne ), p%A%ptr( m + 1 ), p%A%col( a_ne ) )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,      &
                    1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,      &
                    1.0_rp_, 1.0_rp_ /)
       p%A%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
       p%A%ptr = (/ 1, 3, 5, 7, 9, 11, 13, 15 /)
     ELSE IF ( data_storage_type == - 2 ) THEN    ! dense storage
       st = 'D'
       ALLOCATE( p%A%val( m * n ) )
       p%A%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,      &
                    0.0_rp_, 1.0_rp_ /)
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     END IF

!  test with solves via the primal and dual

     DO dual = 0, 0
!    DO dual = 0, 1
       IF ( dual == 0 ) THEN
         control%dual = .FALSE.
         du = 'P'
       ELSE
         control%dual = .TRUE.
         du = 'D'
       END IF

!  test with cold and warm starts

!      DO warm = 0, 0
       DO warm = 0, 1
         IF ( warm == 0 ) THEN
           control%warm_start = .FALSE.
           wa = 'C'
         ELSE
           control%warm_start = .TRUE.
           wa = 'W'
         END IF
!        control%print_level = 1
         CALL LPA_solve( p, data, control, inform, C_stat, X_stat )
         IF ( inform%status == 0 ) THEN
           WRITE( 6, "( 3A1, ':', I6,' iterations. Optimal objective value',   &
       &     ' = ', F7.2, ' status = ', I4 )" ) st, du, wa, inform%iter,       &
                    inform%obj, inform%status
         ELSE
           WRITE( 6, "( 3A1, ': LPA_solve exit status = ', I4 ) " )            &
             st, du, wa, inform%status
           X_stat = - 1 ; C_stat = - 1 ! set on failure to allow next warm start
         END IF
!write(6,"( ' xstat ', 15I3 )" ) x_stat
!write(6,"( ' cstat ', 15I3 )" ) c_stat
       END DO
!      STOP
     END DO
     CALL LPA_terminate( data, control, inform )
     IF ( data_storage_type == 0 ) THEN
       DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
     ELSE IF ( data_storage_type == - 1 ) THEN
       st = 'R'
       DEALLOCATE( p%A%val, p%A%ptr, p%A%col, p%A%type )
     ELSE IF ( data_storage_type == - 2 ) THEN
       DEALLOCATE( p%A%val, p%A%type )
     END IF
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
!stop
!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 14 ; m = 7 ; a_ne = 14
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_stat( n ), C_stat( m ) )
   p%n = n ; p%m = m ; p%A%ne = a_ne
   p%f = 0.0_rp_
   p%G = (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, - 0.5_rp_,   &
            - 0.57143_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,        &
            0.0_rp_ /)
   p%X_l = (/ - infty, - infty, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.5_rp_,   &
              0.5_rp_, - infty, - infty, - infty, - infty, 0.1_rp_, 0.1_rp_  /)
   p%X_u = (/ infty, infty, infty, infty, infty, infty, 2.0_rp_,               &
              2.0_rp_, 3.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 0.1_rp_, 0.1_rp_  /)
   p%C_l = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, - infty, - infty, - infty, - infty /)
   p%C_u = (/ 1.0_rp_, infty, 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)

   ALLOCATE( p%A%val( a_ne ), p%A%ptr( m + 1 ), p%A%col( a_ne ) )
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   p%A%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%A%ptr = (/ 1, 3, 5, 7, 9, 11, 13, 15 /)

!  test with new and existing data

   tests = 4
   DO i = 0, tests
     CALL LPA_initialize( data, control, inform )
     control%infinity = 0.1_rp_ * infty

     IF ( i == 1 ) THEN
       control%min_real_factor_size = 0
     ELSE IF ( i == 2 ) THEN
       control%min_integer_factor_size = 0
     ELSE IF ( i == 3 ) THEN
       control%scale = .TRUE.
     ELSE IF ( i == 4 ) THEN
       control%steepest_edge = .FALSE.
     END IF

!    control%print_level = 2
     CALL LPA_solve( p, data, control, inform, C_stat, X_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I3, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F7.2, ' status = ', I4 )" ) i, inform%iter,              &
                      inform%obj, inform%status
     ELSE
       WRITE( 6, "( I3, ': LPA_solve exit status = ', I4 ) " ) i, inform%status
     END IF
   END DO
   CALL LPA_terminate( data, control, inform )
   DEALLOCATE( p%A%val, p%A%ptr, p%A%col, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_stat, C_stat )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_LPA_EXAMPLE
