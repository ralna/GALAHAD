! THIS VERSION: GALAHAD 5.6 - 2026-11-20 AT 14:10 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QP_TESTS
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   INTEGER ( KIND = ip_ ), PARAMETER :: cco_storage = 1 ! coordinate
   INTEGER ( KIND = ip_ ), PARAMETER :: csr_storage = 2 ! sparse row
   INTEGER ( KIND = ip_ ), PARAMETER :: csc_storage = 3 ! sparse column
   INTEGER ( KIND = ip_ ), PARAMETER :: di_storage = 4 ! diagonal
   INTEGER ( KIND = ip_ ), PARAMETER :: dr_storage = 5 ! dense row
   INTEGER ( KIND = ip_ ), PARAMETER :: dc_storage = 6 ! dense column
   INTEGER ( KIND = ip_ ), PARAMETER :: qpa_solver = 1
   INTEGER ( KIND = ip_ ), PARAMETER :: qpb_solver = 2
   INTEGER ( KIND = ip_ ), PARAMETER :: qpc_solver = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: cqp_solver = 4
   INTEGER ( KIND = ip_ ), PARAMETER :: dqp_solver = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: cdqp_solver = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: bpmpd_solver = 7
   INTEGER ( KIND = ip_ ), PARAMETER :: bqpd_solver = 8
   INTEGER ( KIND = ip_ ), PARAMETER :: highs_solver = 9
   INTEGER ( KIND = ip_ ), PARAMETER :: naglib_solver = 10
   INTEGER ( KIND = ip_ ), PARAMETER :: osqp_solver = 11
   INTEGER ( KIND = ip_ ), PARAMETER :: qpalm_solver = 12
   INTEGER ( KIND = ip_ ), PARAMETER :: qpoases_solver = 13
   TYPE ( QPT_problem_type ) :: p
   TYPE ( QP_data_type ) :: data
   TYPE ( QP_control_type ) :: control
   TYPE ( QP_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, smt_stat, solver_number
   INTEGER ( KIND = ip_ ) :: test, tests
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, sts, status, scratch_out = 56
   CHARACTER ( LEN = 3 ) :: st
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
   CHARACTER ( LEN = 20 ) :: solver_default = "cqp" // REPEAT( ' ', 17 )
!  CHARACTER ( LEN = 20 ) :: solver_default = "e04nqf" // REPEAT( ' ', 14 )
   CHARACTER ( LEN = 20 ) :: solver
   CHARACTER ( LEN = 20 ), DIMENSION( 13 ) :: solver_list = [                  &
                             "qpa" // REPEAT( ' ', 17 ),                       &
                             "qpb" // REPEAT( ' ', 17 ),                       &
                             "qpc" // REPEAT( ' ', 17 ),                       &
                             "cqp" // REPEAT( ' ', 17 ),                       &
                             "dqp" // REPEAT( ' ', 17 ),                       &
                             "cdqp" // REPEAT( ' ', 16 ),                      &
                             "bpmpd" // REPEAT( ' ', 15 ),                     &
                             "bqpd" // REPEAT( ' ', 16 ),                      &
                             "e04nqf" // REPEAT( ' ', 14 ),                    &
                             "highs" // REPEAT( ' ', 15 ),                     &
                             "osqp" // REPEAT( ' ', 16 ),                      &
                             "qalmp" // REPEAT( ' ', 15 ),                     &
                             "qpoases" // REPEAT( ' ', 13 ) ]
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

   DO test = 1, 4
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

     solver = solver_default
     SELECT CASE( test )
     CASE ( 1 )
       p%n = 0
     CASE ( 2 )
       p%m = - 1
     CASE ( 3 )
       CALL SMT_put( p%A%type, 'WRONG', smt_stat )
     CASE ( 4 )
       CALL SMT_put( p%H%type, 'WRONG', smt_stat )
     END SELECT

     CALL QP_initialize( solver, data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty

!    control%out = 6 ; control%print_level = 1
     CALL QP_solve( p, data, control, info, C_stat, B_stat )

     IF ( info%status == GALAHAD_error_scale ) THEN
       sts = info%SCALE_inform%status
     ELSE IF ( info%status ==  GALAHAD_error_presolve ) THEN
       sts = info%PRESOLVE_inform%status
     ELSE IF ( info%status ==  GALAHAD_error_qpa ) THEN
       sts = info%QPA_inform%status
     ELSE IF ( info%status == GALAHAD_error_qpb ) THEN
       sts = info%QPB_inform%status
     ELSE IF ( info%status == GALAHAD_error_qpc ) THEN
       sts = info%QPC_inform%status
     ELSE IF ( info%status == GALAHAD_error_cqp ) THEN
       sts = info%CQP_inform%status
     ELSE
       sts = 0
     END IF

     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &      F6.1, ' status = ', I0, ' solver is ', A  )" )                     &
        status, info%obj, info%status, TRIM( solver )
     ELSE IF ( sts == 0 ) THEN
       WRITE( 6, "(I2, ': QP_solve exit status = ', I0, ' solver is ', A  )")  &
         status, info%status, TRIM( solver )
     ELSE
       WRITE( 6, "(I2, ': QP_solve+solver exit status = ', I0, 1X, I0,         &
      & ' solver is ', A )" ) status, info%status, sts, TRIM( solver )
     END IF
     DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%type )
     DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
     CALL QP_terminate( data, control, info )
   END DO
   CALL QP_terminate( data, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
   DEALLOCATE( p%H%ptr, p%A%ptr )
! stop

!  ================================================
!  basic test of various solver and storage formats
!  ================================================

   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )

   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   DO solver_number = 1, 13
     solver = solver_list( solver_number )
     DO data_storage_type = 1, 6
!      solver = solver_default
       CALL QP_initialize( solver, data, control, info )
       CALL WHICH_sls( control )
       control%infinity = infty
!      control%out = 6 ; control%print_level = 11
!      control%out = 6 ; control%print_level = 1
       control%QPA_control%print_level = 1
!      control%QPC_control%print_level = 1
!      control%QPC_control%QPB_control%print_level = 1
!      control%QPC_control%QPB_control%SBLS_control%print_level = 1
       p%new_problem_structure = .TRUE.
       IF ( data_storage_type == cco_storage ) THEN ! sparse co-ordinate storage
         st = 'CCO'
         ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
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
       ELSE IF ( data_storage_type == csr_storage ) THEN ! sparse row storage
         st = 'CSR'
         ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
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
       ELSE IF ( data_storage_type == csc_storage ) THEN ! sparse column storage
         st = 'CSC'
           ! to do
         CYCLE
       ELSE IF ( data_storage_type == dr_storage ) THEN ! dense (row) storage
         st = 'DRO'
         ALLOCATE( p%H%ptr( 0 ), p%A%ptr( 0 ) )
         ALLOCATE( p%H%val( n * ( n + 1 ) / 2 ) )
         ALLOCATE( p%H%row( 0 ) , p%H%col( n * ( n + 1 ) / 2 ) )
         ALLOCATE( p%A%val( n * m ), p%A%row(0), p%A%col( n * m ) )
         IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
         CALL SMT_put( p%H%type, 'DENSE', smt_stat )
         IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
         CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       ELSE IF ( data_storage_type == dc_storage ) THEN ! dense (column) storage
         st = 'DCO'
           ! to do
         CYCLE
       ELSE IF ( data_storage_type == di_storage ) THEN ! diagonal H, dense A
         st = 'DIA'
         ALLOCATE( p%H%ptr( 0 ), p%A%ptr( 0 ) )
         ALLOCATE( p%H%val( n ), p%H%row( 0 ), p%H%col( 0 ) )
         ALLOCATE( p%A%val( n * m ), p%A%row( 0 ), p%A%col( n * m ) )
         IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
         CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
         IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
         CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       END IF

!    test with new and existing data

       DO i = 1, 2
         IF ( data_storage_type == cco_storage ) THEN
           p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE IF ( data_storage_type == csr_storage ) THEN
           p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE IF ( data_storage_type == csc_storage ) THEN
           ! to do
         ELSE IF ( data_storage_type == dr_storage ) THEN
           p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_ /)
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
         ELSE IF ( data_storage_type == dc_storage ) THEN
           ! to do
         ELSE IF ( data_storage_type == di_storage ) THEN
           p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_ /)
           p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
         END IF
         p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
         CALL QP_solve( p, data, control, info, C_stat, B_stat )

        IF ( info%status == 0 ) THEN
           WRITE( 6, "( A3, I1,': optimal objective value = ',                 &
       &            F6.1, ' status = ', I0, ' solver is ', A  )" )             &
             st, i, info%obj, info%status, TRIM( solver )
         ELSE
           WRITE( 6, "( A3, I1,': QP_solve exit status = ', I0,                &
          &  ' solver is ', A  ) " ) st, i, info%status, TRIM( solver )
         END IF
       END DO
       CALL QP_terminate( data, control, info )
       DEALLOCATE( p%H%ptr, p%A%ptr )
       DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%type )
       DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
!      STOP
     END DO
   END DO
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )

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

!  test with new and existing data

   tests = 18
   DO i = 0, tests

     p%n = n ; p%m = m ; p%f = 0.05_rp_
     p%G = (/ 0.0_rp_, 0.0_rp_ /)
     p%C_l = (/ 1.0_rp_ /)
     p%C_u = (/ 1.0_rp_ /)
     p%X_l = (/ 0.0_rp_, 0.0_rp_ /)
     p%X_u = (/ 2.0_rp_, 3.0_rp_ /)
     p%H%col = (/ 1, 2 /)
     p%H%ptr = (/ 1, 2, 3 /)
     p%A%col = (/ 1, 2 /)
     p%A%ptr = (/ 1, 3 /)

     IF ( i == 0 ) THEN
       solver = "qpa" // REPEAT( ' ', 17 )
     ELSE IF ( i == 1 ) THEN
       solver = "qpb" // REPEAT( ' ', 17 )
     ELSE IF ( i == 2 ) THEN
       solver = "qpc" // REPEAT( ' ', 17 )
     ELSE IF ( i == 3 ) THEN
       solver = "cqp" // REPEAT( ' ', 17 )
     ELSE IF ( i == 4 ) THEN
       solver = solver_default
       control%presolve = .TRUE.
       control%scale = 1
     ELSE IF ( i == 5 ) THEN
 !  control%print_level = 101
 !  control%QPC_control%print_level = 1
 !  control%QPC_control%QPB_control%print_level = 1
 !  control%SCALE_control%print_level = 2
       control%scale = 2
     ELSE IF ( i == 6 ) THEN
       control%scale = 3
     ELSE IF ( i == 7 ) THEN
       control%scale = 4
     ELSE IF ( i == 8 ) THEN
       control%scale = 5
     ELSE IF ( i == 9 ) THEN
       control%scale = 6
     ELSE IF ( i == 10 ) THEN
       control%scale = 7
     ELSE IF ( i == 11 ) THEN
       control%scale = - 1
     ELSE IF ( i == 12 ) THEN
       control%scale = - 2
     ELSE IF ( i == 13 ) THEN
       control%scale = - 3
     ELSE IF ( i == 14 ) THEN
       control%scale = - 4
     ELSE IF ( i == 15 ) THEN
       control%scale = - 5
     ELSE IF ( i == 16 ) THEN
       control%scale = - 6
     ELSE IF ( i == 17 ) THEN
       control%scale = - 7
     ELSE IF ( i == 18 ) THEN
       control%scale = 0
     END IF

     CALL QP_initialize( solver, data, control, info )
     CALL WHICH_sls( control )
     control%infinity = infty
!  control%out = 6 ; control%print_level = 1

     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 4
     CALL QP_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X

     IF ( info%status == GALAHAD_error_scale ) THEN
       sts = info%SCALE_inform%status
     ELSE IF ( info%status ==  GALAHAD_error_presolve ) THEN
       sts = info%PRESOLVE_inform%status
     ELSE IF ( info%status ==  GALAHAD_error_qpa ) THEN
       sts = info%QPA_inform%status
     ELSE IF ( info%status == GALAHAD_error_qpb ) THEN
       sts = info%QPB_inform%status
     ELSE IF ( info%status == GALAHAD_error_qpc ) THEN
       sts = info%QPC_inform%status
     ELSE IF ( info%status == GALAHAD_error_cqp ) THEN
       sts = info%CQP_inform%status
     ELSE
       sts = 0
     END IF

     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &      F6.1, ' status = ', I0, ' solver is ', A  )" )                     &
        i, info%obj, info%status, TRIM( solver )
     ELSE IF ( sts == 0 ) THEN
       WRITE( 6, "(I2, ': QP_solve exit status = ', I0, ' solver is ', A  )")  &
         i, info%status, TRIM( solver )
     ELSE
       WRITE( 6, "(I2, ': QP_solve+solver exit status = ', I0, 1X, I0,         &
      &  ' solver is ', A )" ) i, info%status, sts, TRIM( solver )
     END IF
   END DO
   CALL QP_terminate( data, control, info )

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
   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
!  control%out = 6 ; control%print_level = 11
!  control%EQP_control%print_level = 21
!  control%print_level = 4
   DO i = tests + 1, tests + 1
     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
     CALL QP_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &                F6.1, ' status = ', I0, ' solver is ', A  )" )           &
         i, info%obj, info%status, TRIM( solver )
     ELSE
       WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A )" ) &
         i, info%status, TRIM( solver )
     END IF
   END DO
   CALL QP_terminate( data, control, info )

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
   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
!   control%print_level = 101
!   control%QPC_control%print_level = 1
!   control%QPC_control%QPB_control%print_level = 1
   control%infinity = infty
   DO i = tests + 2, tests + 2
     p%H%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%A%val = (/ 1.0_rp_, 1.0_rp_ /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
!    control%print_level = 1
     control%CDQP_control%print_level = 1
     CALL QP_solve( p, data, control, info, C_stat, B_stat )
!    write(6,"('x=', 2ES12.4)") p%X
     IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &                F6.1, ' status = ', I0, ' solver is ', A  )" )           &
         i, info%obj, info%status, TRIM( solver )
!      write(6,*) info%obj

     ELSE
       WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A ) ") &
         i, info%status, TRIM( solver )
       write(6,*) ' qpc_status ', info%QPC_inform%status
     END IF
   END DO
   CALL QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
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
                7.0_rp_, 7.0_rp_, 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_,          &
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

   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   control%print_level = 101
   control%out = scratch_out
   control%error = scratch_out
!  control%out = 6
!  control%error = 6
!  control%print_level = 1
!  control%qpc_control%print_level = 1
!  control%qpc_control%qpb_control%print_level = 1
!  control%qpc_control%qpb_control%lsqp_control%print_level = 1
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
   CALL QP_solve( p, data, control, info, C_stat, B_stat )
   CLOSE( UNIT = scratch_out )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &                F6.1, ' status = ', I0, ' solver is ', A  )" )           &
        1, info%obj, info%status, TRIM( solver )
   ELSE
     WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A  ) " ) &
        1, info%status, TRIM( solver )
   END IF
   CALL QP_terminate( data, control, info )
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

   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL QP_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &                F6.1, ' status = ', I0, ' solver is ', A  )" )           &
         2, info%obj, info%status, TRIM( solver )
   ELSE
     WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A )" )   &
         2, info%status, TRIM( solver )
   END IF
   CALL QP_terminate( data, control, info )
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

   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
!  C_stat( 10 ) = 1 ; C_stat( 11 ) = 1
!  C_stat( 12 ) = 1 ; C_stat( 13 ) = 1
!  C_stat( 14 ) = 1 ; C_stat( 15 ) = 1
!  C_stat( 16 ) = 1 ; C_stat( 17 ) = 1
   CALL QP_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &                F6.1, ' status = ', I0, ' solver is ', A  )" )           &
         3, info%obj, info%status, TRIM( solver )
   ELSE
     WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A )" )   &
         3, info%status, TRIM( solver )
   END IF
   CALL QP_terminate( data, control, info )
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

   CALL QP_initialize( solver, data, control, info )
   CALL WHICH_sls( control )
   control%infinity = infty
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL QP_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &              F6.1, ' status = ', I0, ' solver is ', A )" )              &
         4, info%obj, info%status, TRIM( solver )
   ELSE
     WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A )" )   &
       4, info%status, TRIM( solver )
   END IF

!  control%out = 6 ; control%print_level = 1
!  control%EQP_control%print_level = 2

   p%X_l( 1 ) = 1.0_rp_ ; p%X_u( 1 ) =  p%X_l( 1 )
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   B_stat = 0 ; C_stat = 0
   B_stat( 2 ) = - 1 ; B_stat( 9 ) = - 1
   C_stat( 8 ) = - 1 ; C_stat( 9 ) = - 1
   CALL QP_solve( p, data, control, info, C_stat, B_stat )
   IF ( info%status == 0 ) THEN
       WRITE( 6, "( I2, ': optimal objective value = ',                        &
     &              F6.1, ' status = ', I0, ' solver is ', A )" )              &
         5, info%obj, info%status, TRIM( solver )
   ELSE
     WRITE( 6, "( I2, ': QP_solve exit status = ', I0, ' solver is ', A  ) " ) &
         5, info%status, TRIM( solver )
   END IF

   CALL QP_terminate( data, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%ptr, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr, p%A%type )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, B_stat, C_stat )
!  DEALLOCATE( p%Y_l, p%Y_u, p%Z_l, p%Z_u, p%X_status, p%C_status )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( QP_control_type ) :: control
#include "galahad_sls_defaults.h"
!symmetric_linear_solver = 'ma86 '
!definite_linear_solver = 'ma86 '
     control%QPA_control%symmetric_linear_solver = symmetric_linear_solver
     control%QPB_control%FDC_control%use_sls = use_sls
     control%QPB_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%QPB_control%SBLS_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%QPB_control%SBLS_control%definite_linear_solver                   &
       = definite_linear_solver
     control%QPB_control%LSQP_control%FDC_control%use_sls = use_sls
     control%QPB_control%LSQP_control%FDC_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%QPB_control%LSQP_control%SBLS_control%symmetric_linear_solver     &
       = symmetric_linear_solver
     control%QPB_control%LSQP_control%SBLS_control%definite_linear_solver      &
       = definite_linear_solver
     control%QPC_control%FDC_control%use_sls = use_sls
     control%QPC_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%QPC_control%QPA_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%QPC_control%QPB_control%FDC_control%use_sls = use_sls
     control%QPC_control%QPB_control%FDC_control%symmetric_linear_solver       &
       = symmetric_linear_solver
     control%QPC_control%QPB_control%SBLS_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%QPC_control%QPB_control%SBLS_control%definite_linear_solver       &
       = definite_linear_solver
     control%QPC_control%QPB_control%LSQP_control%FDC_control%use_sls = use_sls
     control%QPC_control%QPB_control%LSQP_control&
       &%FDC_control%symmetric_linear_solver = symmetric_linear_solver
     control%QPC_control%QPB_control%LSQP_control&
       &%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%QPC_control%QPB_control%LSQP_control&
       &%SBLS_control%definite_linear_solver = definite_linear_solver
     control%QPC_control%CQP_control%FDC_control%use_sls = use_sls
     control%QPC_control%CQP_control%FDC_control%symmetric_linear_solver       &
       = symmetric_linear_solver
     control%QPC_control%CQP_control%SBLS_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%QPC_control%CQP_control%SBLS_control%definite_linear_solver       &
       = definite_linear_solver
     control%QPC_control%EQP_control%FDC_control%use_sls = use_sls
     control%QPC_control%EQP_control%FDC_control%symmetric_linear_solver       &
       = symmetric_linear_solver
     control%QPC_control%EQP_control%SBLS_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%QPC_control%EQP_control%SBLS_control%definite_linear_solver       &
       = definite_linear_solver
     control%QPC_control%CRO_control%SBLS_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%QPC_control%CRO_control%SBLS_control%definite_linear_solver       &
        = definite_linear_solver
     control%CQP_control%FDC_control%use_sls = use_sls
     control%CQP_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%CQP_control%SBLS_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%CQP_control%SBLS_control%definite_linear_solver                   &
       = definite_linear_solver
     control%DQP_control%FDC_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     control%DQP_control%symmetric_linear_solver = symmetric_linear_solver
     control%DQP_control%definite_linear_solver = definite_linear_solver
     control%CDQP_control%FDC_control%use_sls = use_sls
     control%CDQP_control%FDC_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%CDQP_control%CQP_control%FDC_control%use_sls = use_sls
     control%CDQP_control%CQP_control%FDC_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%CDQP_control%CQP_control%SBLS_control%symmetric_linear_solver     &
       = symmetric_linear_solver
     control%CDQP_control%CQP_control%SBLS_control%definite_linear_solver      &
        = definite_linear_solver
     control%CDQP_control%DQP_control%FDC_control%use_sls = use_sls
     control%CDQP_control%DQP_control%FDC_control%symmetric_linear_solver      &
       = symmetric_linear_solver
     control%CDQP_control%DQP_control%symmetric_linear_solver                  &
       = symmetric_linear_solver
     control%CDQP_control%DQP_control%definite_linear_solver                   &
       = definite_linear_solver
     control%CDQP_control%CRO_control%SBLS_control%symmetric_linear_solver     &
       = symmetric_linear_solver
     control%CDQP_control%CRO_control%SBLS_control%definite_linear_solver      &
        = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_QP_TESTS
