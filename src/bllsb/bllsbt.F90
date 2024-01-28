! THIS VERSION: GALAHAD 4.3 - 2023-12-28 AT 15:50 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BLLSB_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BLLSB_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BLLSB_data_type ) :: data
   TYPE ( BLLSB_control_type ) :: control
   TYPE ( BLLSB_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, o, ao_ne, tests, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, status, scratch_out = 56
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = REPEAT( ' ', 30 )
!  symmetric_linear_solver = 'ssids'
!  symmetric_linear_solver = 'ma97 '
   symmetric_linear_solver = 'sytr '

!go to 111
   n = 3 ; o = 4 ; ao_ne = 7
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ) )
   ALLOCATE( p%X_status( n ) )

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
     CALL BLLSB_initialize( data, control, info )
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
     p%n = n ; p%o = o
     p%B = (/ 2.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /)
     p%X_l = (/ - 1.0_rp_, - infty, - infty /)
     p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

     ALLOCATE( p%Ao%val( ao_ne ), p%Ao%col( ao_ne ) )
     IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
     CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
     p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                &
                   1.0_rp_, 1.0_rp_ /)
     p%Ao%col = (/ 1, 2, 2, 3, 1, 3, 2 /)
     p%Ao%ptr = (/ 1, 3, 5, 6, 7 /)
     p%X = 0.0_rp_ ; p%Z = 0.0_rp_

     IF ( status == - GALAHAD_error_restrictions ) THEN
       p%n = 0 ; p%o = - 1
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

     CALL BLLSB_solve( p, data, control, info )
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &      F6.1, ' status = ', I6 )" ) status, info%iter,                     &
              info%obj, info%status
     ELSE
       WRITE( 6, "(I2, ': BLLSB_solve exit status = ', I6)") status, info%status
     END IF
     DEALLOCATE( p%Ao%val, p%Ao%col )
     CALL BLLSB_terminate( data, control, info )

   END DO
   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%X_status, p%Ao%ptr )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; o = 3 ; ao_ne = 4
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ) )
   ALLOCATE( p%X_status( n ) )

   p%n = n ; p%o = o
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   DO data_storage_type = - 3, 0
     CALL BLLSB_initialize( data, control, info )
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
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
       p%Ao%row = (/ 1, 2, 3, 3 /)
       p%Ao%col = (/ 1, 2, 3, 1 /) ; p%Ao%ne = ao_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( 0 ), p%Ao%col( ao_ne ) )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
       p%Ao%col = (/ 1, 2, 3, 1 /)
       p%Ao%ptr = (/ 1, 2, 3, 5 /)
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), STAT = i )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! dense by column storage
       st = 'I'
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), STAT = i )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'DENSE_BY_COLUMNS', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      4.0_rp_, 0.0_rp_, 3.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 ) THEN    !  dense by column storage
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 4.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,   &
                      0.0_rp_, 0.0_rp_, 3.0_rp_ /)
       END IF
       p%X = 0.0_rp_ ;  p%Z = 0.0_rp_
!      control%print_level = 101
!      control%print_level = 101
!      control%min_diagonal = 0.000000000000001_rp_
       CALL BLLSB_solve( p, data, control, info )

       IF ( info%status == 0 ) THEN
         WRITE( 6, "( A1,I1,':', I6,' iterations. Optimal objective value = ', &
     &            F6.1, ' status = ', I6 )" ) st, i, info%iter,                &
                  info%obj, info%status
       ELSE
         WRITE( 6, "( A1, I1,': BLLSB_solve exit status = ', I6 ) " )          &
           st, i, info%status
       END IF
!      STOP
     END DO
     CALL BLLSB_terminate( data, control, info )
     IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row )
     IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col )
     IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val )
!    STOP
   END DO
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%X_status, p%Ao%ptr )

!stop
!  =============================
!  basic test of various options
!  =============================

   WRITE( 6, "( /, ' basic tests of options ', / )" )

   n = 2 ; o = 2 ; ao_ne = 2
   ALLOCATE( p%B( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ) )
   ALLOCATE( p%X_status( n ) )

   p%n = n ; p%o = o
   p%B = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
   p%X_u = (/ 2.0_rp_, 3.0_rp_ /)

   p%new_problem_structure = .TRUE.
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( 0 ), p%Ao%col( ao_ne ) )
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   CALL BLLSB_initialize( data, control, info )
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
     p%X = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL BLLSB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BLLSB_terminate( data, control, info )

!  case when there are no bounded variables

   p%X_l = (/ - infty, - infty /)
   p%X_u = (/ infty, infty /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   CALL BLLSB_initialize( data, control, info )
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
     p%X = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL BLLSB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
     ELSE
       WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BLLSB_terminate( data, control, info )

!  case when there are no free variables

   p%X_l = (/ 0.5_rp_, 0.5_rp_ /)
   p%X_u = (/ 0.5_rp_, 0.5_rp_ /)
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
   p%Ao%col = (/ 1, 2 /)
   p%Ao%ptr = (/ 1, 2, 3 /)
   CALL BLLSB_initialize( data, control, info )
   control%CRO_control%error = 0
!  control%print_level = 4
   control%infinity = infty
   control%restore_problem = 2
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   DO i = tests + 2, tests + 2
     p%Ao%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     CALL BLLSB_solve( p, data, control, info )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) i, info%iter,                &
                      info%obj, info%status
!      write(6,*) info%obj
     ELSE
       WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) i, info%status
     END IF
   END DO
   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, STAT = i )
   DEALLOCATE( p%X, STAT = i )
   DEALLOCATE( p%Z, STAT = i )
   DEALLOCATE( p%X_status, STAT = i )
   DEALLOCATE( p%Ao%ptr, STAT = i )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, STAT = i )
!stop

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   n = 14 ; o = 14 ; ao_ne = 21
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%Ao%ne = ao_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
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

   CALL BLLSB_initialize( data, control, info )
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
   p%X = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL BLLSB_solve( p, data, control, info )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 1, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) 1, info%status
   END IF
   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%ptr )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%R, p%X_status )

!  Second problem

   n = 14 ; o = 14 ; ao_ne = 14
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%Ao%ne = ao_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%Ao%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL BLLSB_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL BLLSB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 2, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) 2, info%status
   END IF
   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%ptr )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%X_status )

!  Third problem

   n = 14 ; o = 14 ; ao_ne = 14
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%Ao%ne = ao_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%Ao%col = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL BLLSB_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL BLLSB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 3, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) 3, info%status
   END IF
   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%ptr, p%Ao%type )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%X_status )

!  Fourth and Fifth problems

   n = 14 ; o = 14 ; ao_ne = 14
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( p%Ao%ptr( o + 1 ) )
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   ALLOCATE( p%X_status( n ) )
   p%new_problem_structure = .TRUE.
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   p%n = n ; p%o = o ; p%Ao%ne = ao_ne
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

   CALL BLLSB_initialize( data, control, info )
   control%infinity = infty
   control%restore_problem = 0
   control%treat_zero_bounds_as_general = .TRUE.
   control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%symmetric_linear_solver = symmetric_linear_solver
   control%FDC_control%use_sls = .TRUE.
   p%X = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL BLLSB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 4, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) 4, info%status
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL BLLSB_solve( p, data, control, info )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &                F6.1, ' status = ', I6 )" ) 5, info%iter,                &
                      info%obj, info%status
   ELSE
     WRITE( 6, "( I2, ': BLLSB_solve exit status = ', I6 ) " ) 5, info%status
   END IF

   CALL BLLSB_terminate( data, control, info )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%ptr )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%X, p%Z, p%X_status )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_BLLSB_EXAMPLE
