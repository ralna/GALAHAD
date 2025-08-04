! THIS VERSION: GALAHAD 5.3 - 2024-06-15 AT 11:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_EXPO_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_EXPO_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( EXPO_control_type ) :: control
   TYPE ( EXPO_inform_type ) :: inform
   TYPE ( EXPO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FC, GJ, HL, GJ_dense, HL_dense
   INTEGER :: i, s, data_storage_type
   CHARACTER ( LEN = 2 ) :: st
   INTEGER, PARAMETER :: n = 2, m = 5, j_ne = 10, h_ne = 2
   INTEGER, PARAMETER :: j_ne_dense = 10, h_ne_dense = 3
   REAL ( KIND = rp_ ), PARAMETER :: p = 9.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
! start problem data
   nlp%pname = 'HS23'                           ! name
   nlp%n = n ; nlp%m = m ; nlp%J%ne = j_ne ; nlp%H%ne = h_ne
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   ALLOCATE( nlp%C( m ), nlp%C_l( m ), nlp%C_u( m ) )
   nlp%X_l = - 50.0_rp_ ; nlp%X_u = 50.0_rp_
   nlp%C_l = 0.0_rp_ ; nlp%C_u = infinity
!  sparse row-wise storage format for the Jacobian
   ALLOCATE( nlp%J%val( j_ne ), nlp%J%col( j_ne ), nlp%H%ptr( m + 1 ) )
   nlp%J%row = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 /)
   nlp%J%col = (/ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 /)
   nlp%J%ptr = (/ 1, 3, 5, 7, 9, 11 /)
!  sparse co-ordinate storage format for the Hessian (lower triangle part)
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 2 /)              ! Hessian H
   nlp%H%col = (/ 1, 2 /)              ! NB 
   nlp%H%ptr = (/ 1, 2, 3 /)
! problem data complete
   ALLOCATE( userdata%real( 1 ) )                ! allocate space for parameter
   userdata%real( 1 ) = p                        ! record parameter, p
!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for s = - 1 ... - 40

   DO s = 1, 24

     IF ( s == - GALAHAD_error_allocate ) CYCLE
     IF ( s == - GALAHAD_error_deallocate ) CYCLE
!    IF ( s == - GALAHAD_error_restrictions ) CYCLE
     IF ( s == - GALAHAD_error_bad_bounds ) CYCLE
     IF ( s == - GALAHAD_error_primal_infeasible ) CYCLE
     IF ( s == - GALAHAD_error_dual_infeasible ) CYCLE
     IF ( s == - GALAHAD_error_unbounded ) CYCLE
     IF ( s == - GALAHAD_error_no_center ) CYCLE
     IF ( s == - GALAHAD_error_analysis ) CYCLE
     IF ( s == - GALAHAD_error_factorization ) CYCLE
     IF ( s == - GALAHAD_error_solve ) CYCLE
     IF ( s == - GALAHAD_error_uls_analysis ) CYCLE
     IF ( s == - GALAHAD_error_uls_factorization ) CYCLE
     IF ( s == - GALAHAD_error_uls_solve ) CYCLE
     IF ( s == - GALAHAD_error_preconditioner ) CYCLE
     IF ( s == - GALAHAD_error_ill_conditioned ) CYCLE
     IF ( s == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( s == - GALAHAD_error_max_iterations ) CYCLE
!    IF ( s == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( s == - GALAHAD_error_inertia ) CYCLE
     IF ( s == - GALAHAD_error_file ) CYCLE
     IF ( s == - GALAHAD_error_io ) CYCLE
     IF ( s == - GALAHAD_error_upper_entry ) CYCLE
     IF ( s == - GALAHAD_error_sort ) CYCLE
     CALL EXPO_initialize( data, control, inform )  ! Initialize controls
     control%subproblem_direct = .TRUE.
     control%max_it = 20
     control%max_eval = 100
!    control%print_level = 1
!    control%tru_control%print_level = 1
#ifdef REAL_32
     control%stop_abs_p = 0.001_rp_
     control%stop_abs_d = 0.001_rp_
     control%stop_abs_c = 0.001_rp_
     control%tru_control%error = 0
#else
     control%stop_abs_p = 0.00001_rp_
     control%stop_abs_d = 0.00001_rp_
     control%stop_abs_c = 0.00001_rp_
#endif
!    control%print_level = 1
     nlp%n = n
     nlp%X( 1 ) = 3.0_rp_ ; nlp%X( 2 ) = 1.0_rp_
     IF ( s == - GALAHAD_error_restrictions ) THEN
       nlp%n = 0
     ELSE IF ( s == - GALAHAD_error_unbounded ) THEN
     ELSE IF ( s == - GALAHAD_error_max_iterations ) THEN
       control%max_it = 0
     ELSE IF ( s == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0_rp_
     END IF
     inform%status = 1                             
     CALL SMT_put( nlp%J%type, 'COORDINATE', i )
     CALL SMT_put( nlp%H%type, 'COORDINATE', i )
     CALL EXPO_solve( nlp, control, inform, data, userdata, eval_FC = FC,      &
                      eval_GJ = GJ, eval_HL = HL )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) s, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': EXPO_solve exit status = ', I6 ) " ) s, inform%status
     END IF
     CALL EXPO_terminate( data, control, inform )  ! delete internal workspace
     DEALLOCATE( nlp%J%type, nlp%H%type )
   END DO

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats', / )" )

   DO data_storage_type = 1, 3

 ! initialize control parameters

     CALL EXPO_initialize( data, control, inform )
     control%subproblem_direct = .TRUE.
     control%max_it = 20
     control%max_eval = 100
!    control%print_level = 1
!    control%tru_control%print_level = 1
!    control%ssls_control%print_level = 1
#ifdef REAL_32
     control%stop_abs_p = 0.001_rp_
     control%stop_abs_d = 0.001_rp_
     control%stop_abs_c = 0.001_rp_
     control%tru_control%error = 0
#else
     control%stop_abs_p = 0.00001_rp_
     control%stop_abs_d = 0.00001_rp_
     control%stop_abs_c = 0.00001_rp_
#endif
control%ssls_control%sls_control%generate_matrix_file = .TRUE.

!  solve the problem

     inform%status = 1                             
     nlp%X( 1 ) = 3.0_rp_ ; nlp%X( 2 ) = 1.0_rp_
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL SMT_put( nlp%J%type, 'COORDINATE', s )
       CALL SMT_put( nlp%H%type, 'COORDINATE', s )
       CALL EXPO_solve( nlp, control, inform, data, userdata, eval_FC = FC,    &
                        eval_GJ = GJ, eval_HL = HL )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL SMT_put( nlp%J%type, 'SPARSE_BY_ROWS', s )
       CALL SMT_put( nlp%H%type, 'SPARSE_BY_ROWS', s )
       CALL EXPO_solve( nlp, control, inform, data, userdata, eval_FC = FC,    &
                        eval_GJ = GJ, eval_HL = HL )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL SMT_put( nlp%J%type, 'DENSE', s )
       CALL SMT_put( nlp%H%type, 'DENSE', s )
       DEALLOCATE( nlp%J%val, nlp%H%val )
       ALLOCATE( nlp%J%val( j_ne_dense ), nlp%H%val( h_ne_dense ) )
       nlp%J%ne = j_ne_dense ; nlp%H%ne = h_ne_dense
       CALL EXPO_solve( nlp, control, inform, data, userdata, eval_FC = FC,    &
                        eval_GJ = GJ_dense, eval_HL = HL_dense )
     END SELECT

     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0, ' solver: ', A )" ) st, inform%iter,         &
        inform%obj, inform%status, TRIM( inform%ssls_inform%sls_inform%solver )
     ELSE
       WRITE( 6, "( A2, ': EXPO_solve exit status = ', I0 )" ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL EXPO_terminate( data, control, inform )  ! delete internal workspace
     DEALLOCATE( nlp%J%type, nlp%H%type )
   END DO

   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%col, nlp%J%ptr )
   DEALLOCATE( nlp%C, nlp%X_l, nlp%X_u, nlp%C_l, nlp%C_u, nlp%Gl )
   END PROGRAM GALAHAD_EXPO_test_program

   SUBROUTINE WHICH_sls( control )
   USE GALAHAD_EXPO_precision, ONLY: EXPO_control_type
   TYPE ( EXPO_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
!symmetric_linear_solver = 'ssids'
!definite_linear_solver = 'ssids'
   control%SSLS_control%symmetric_linear_solver = symmetric_linear_solver
   control%TRU_control%TRS_control%definite_linear_solver                      &
     = definite_linear_solver
   control%TRU_control%TRS_control%symmetric_linear_solver                     &
     = symmetric_linear_solver
   END SUBROUTINE WHICH_sls

   SUBROUTINE FC( status, X, userdata, F, C )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( kind = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( kind = rp_ ), OPTIONAL, INTENT( OUT ) :: F
   REAL ( kind = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( kind = rp_ ) :: p
   p = userdata%real( 1 )
   f = X( 1 ) ** 2 + X( 2 ) ** 2
   C( 1 ) = X( 1 ) + X( 2 ) - 1.0_rp_
   C( 2 ) = X( 1 ) ** 2 + X( 2 ) ** 2 - 1.0_rp_
   C( 3 ) = p * X( 1 ) ** 2 + X( 2 ) ** 2 - p
   C( 4 ) = X( 1 ) ** 2 - X( 2 )
   C( 5 ) = X( 2 ) ** 2 - X( 1 )
   status = 0
   END SUBROUTINE FC

   SUBROUTINE GJ( status, X, userdata, G, J_val )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
   REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( kind = rp_ ) :: p
   p = userdata%real( 1 )
   G( 1 ) = 2.0_rp_ * X( 1 )
   G( 2 ) = 2.0_rp_ * X( 2 )
   J_val( 1 ) = 1.0_rp_
   J_val( 2 ) = 1.0_rp_
   J_val( 3 ) = 2.0_rp_ * X( 1 )
   J_val( 4 ) = 2.0_rp_ * X( 2 )
   J_val( 5 ) = 2.0_rp_ * p * X( 1 )
   J_val( 6 ) = 2.0_rp_ * X( 2 )
   J_val( 7 ) = 2.0_rp_ * X( 1 )
   J_val( 8 ) = - 1.0_rp_
   J_val( 9 ) = - 1.0_rp_
   J_val( 10 ) = 2.0_rp_ * X( 2 )
   status = 0
   END SUBROUTINE GJ

   SUBROUTINE HL( status, X, Y, userdata, H_val )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( kind = rp_ ) :: p
   p = userdata%real( 1 )
   H_val( 1 ) = 2.0_rp_ - 2.0_rp_ * ( Y( 2 ) + p * Y( 3 ) + Y( 4 ) )
   H_val( 2 ) = 2.0_rp_ - 2.0_rp_ * ( Y( 2 ) + Y( 3 ) + Y( 5 ) )
   status = 0
   END SUBROUTINE HL

   SUBROUTINE GJ_dense( status, X, userdata, G, J_val )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
   REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( kind = rp_ ) :: p
   p = userdata%real( 1 )
   G( 1 ) = 2.0_rp_ * X( 1 )
   G( 2 ) = 2.0_rp_ * X( 2 )
   J_val( 1 ) = 1.0_rp_
   J_val( 2 ) = 1.0_rp_
   J_val( 3 ) = 2.0_rp_ * X( 1 )
   J_val( 4 ) = 2.0_rp_ * X( 2 )
   J_val( 5 ) = 2.0_rp_ * p * X( 1 )
   J_val( 6 ) = 2.0_rp_ * X( 2 )
   J_val( 7 ) = 2.0_rp_ * X( 1 )
   J_val( 8 ) = - 1.0_rp_
   J_val( 9 ) = - 1.0_rp_
   J_val( 10 ) = 2.0_rp_ * X( 2 )
   status = 0
   END SUBROUTINE GJ_dense

   SUBROUTINE HL_dense( status, X, Y, userdata, H_val )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( kind = rp_ ) :: p
   p = userdata%real( 1 )
   H_val( 1 ) = 2.0_rp_ - 2.0_rp_ * ( Y( 2 ) + p * Y( 3 ) + Y( 4 ) )
   H_val( 2 ) = 0.0_rp_
   H_val( 3 ) = 2.0_rp_ - 2.0_rp_ * ( Y( 2 ) + Y( 3 ) + Y( 5 ) )
   status = 0
   END SUBROUTINE HL_dense
